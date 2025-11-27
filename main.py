"""

Typical workflow:
  python main.py collect --topic nofaults --duration 600(already done in bioreactor.csv file)
  python main.py collect --topic single_fault --duration 600(already done in bioreactor.csv file)
  python main.py train
  python main.py monitor --topic variable_setpoints --duration x --threshold x
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BROKER = "engf0001.cs.ucl.ac.uk"
BASE_TOPIC = "bioreactor_sim"
TOPICS: Dict[str, str] = {
    "nofaults": f"{BASE_TOPIC}/nofaults/telemetry/summary",
    "single_fault": f"{BASE_TOPIC}/single_fault/telemetry/summary",
    "three_faults": f"{BASE_TOPIC}/three_faults/telemetry/summary",
    "three_fautls": f"{BASE_TOPIC}/three_faults/telemetry/summary",
    "variable_setpoints": f"{BASE_TOPIC}/variable_setpoints/telemetry/summary",
}

FEATURES = [
    "temperature_mean",
    "pH_mean",
    "rpm_mean",
    "heater_pwm",
    "motor_pwm",
    "acid_dose_l",
    "base_dose_l",
    "heater_energy_Wh",
    "photoevents",
    "temp_setpoint",
    "pH_setpoint",
    "rpm_setpoint",
]
LABEL_COL = "fault_label"
COLUMNS: List[str] = [
    "timestamp",
    *FEATURES,
    LABEL_COL,
    "source_topic",
]

DEFAULT_CSV = Path("bioreactor.csv")
DEFAULT_MODEL = Path("lstm_bioreactor_model.keras")
DEFAULT_SCALER = Path("scaler.pkl")
DEFAULT_WINDOW_SIZE = 60


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def ensure_dataset(csv_path: Path) -> None:
    """Create the CSV with the expected header if it does not exist yet."""
    if not csv_path.exists():
        logging.info("Creating dataset at %s", csv_path)
        pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)


def decode_payload(payload: bytes, source_topic: str) -> Dict[str, float]:
    """Parse the MQTT payload and return a dict matching COLUMNS."""
    data = json.loads(payload.decode())
    return {
        "timestamp": data.get("timestamp") or data.get("time")
        or pd.Timestamp.utcnow().isoformat(),
        "temperature_mean": data["temperature_C"]["mean"],
        "pH_mean": data["pH"]["mean"],
        "rpm_mean": data["rpm"]["mean"],
        "heater_pwm": data["actuators_avg"]["heater_pwm"],
        "motor_pwm": data["actuators_avg"]["motor_pwm"],
        "acid_dose_l": data["dosing_l"]["acid"],
        "base_dose_l": data["dosing_l"]["base"],
        "heater_energy_Wh": data["heater_energy_Wh"],
        "photoevents": data["photoevents"],
        "temp_setpoint": data["setpoints"]["temperature_C"],
        "pH_setpoint": data["setpoints"]["pH"],
        "rpm_setpoint": data["setpoints"]["rpm"],
        LABEL_COL: 1 if data["faults"]["last_active"] else 0,
        "source_topic": source_topic,
    }


def collect_stream(
    topic_key: str,
    duration: int,
    csv_path: Path,
    broker: str,
) -> None:
    """Listen to a topic for N seconds and append rows to the dataset."""
    if topic_key not in TOPICS:
        raise ValueError(f"Unknown topic '{topic_key}'. Options: {list(TOPICS)}")

    topic = TOPICS[topic_key]
    ensure_dataset(csv_path)

    client = mqtt.Client()
    stop_event = threading.Event()
    rows = []

    def flush_rows():
        nonlocal rows
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=False, index=False)
            logging.info("Appended %d rows to %s", len(rows), csv_path)
            rows = []

    def on_connect(_client, _userdata, _flags, rc):
        logging.info("Connected to broker %s (rc=%s)", broker, rc)
        _client.subscribe(topic)
        logging.info("Subscribed to %s", topic)

    def on_message(_client, _userdata, msg):
        nonlocal rows
        try:
            row = decode_payload(msg.payload, topic_key)
            rows.append(row)
            if len(rows) >= 50:
                flush_rows()
        except Exception as exc:
            logging.error("Failed to parse payload: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message

    def shutdown(_signum=None, _frame=None):
        stop_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.info("Collecting %s data for %s seconds", topic_key, duration)
    client.connect(broker, 1883, keepalive=60)
    client.loop_start()

    stop_event.wait(timeout=duration)
    shutdown()
    client.loop_stop()
    flush_rows()
    logging.info("Collection finished.")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset {csv_path} not found. Collect data first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty. Collect data first.")

    if not set(COLUMNS).issubset(df.columns):
        if len(df.columns) == len(COLUMNS):
            logging.warning(
                "Dataset %s is missing headers; assuming legacy format and applying defaults",
                csv_path,
            )
            df.columns = COLUMNS
        else:
            raise ValueError(
                f"Dataset columns do not match expected schema: {df.columns.tolist()}"
            )
    return df


def create_sequences(data: np.ndarray, labels: np.ndarray, window: int):
    xs, ys = [], []
    for idx in range(len(data) - window):
        xs.append(data[idx : idx + window])
        ys.append(labels[idx + window])
    return np.array(xs), np.array(ys)


def train_lstm(
    csv_path: Path,
    model_path: Path,
    scaler_path: Path,
    window_size: int,
    epochs: int,
    batch_size: int,
) -> None:
    """Train the LSTM anomaly detector."""
    df = load_dataset(csv_path)
    df = df.dropna(subset=FEATURES + [LABEL_COL])
    if len(df) <= window_size:
        raise ValueError("Not enough samples to build sequences. Collect more data.")

    X = df[FEATURES].values
    y = df[LABEL_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y, window_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq,
    )

    logging.info("Training samples: %s, Testing samples: %s", X_train.shape, X_test.shape)

    model = Sequential(
        [
            LSTM(64, input_shape=(window_size, len(FEATURES)), return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )
    logging.info("Training finished after %d epochs", len(history.history["loss"]))

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info("Test accuracy: %.2f%% (loss=%.4f)", acc * 100, loss)

    y_pred_probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    logging.info(
        "Confusion summary (threshold=0.5): TP=%d, TN=%d, FP=%d, FN=%d", tp, tn, fp, fn
    )
    logging.info("Precision=%.3f Recall=%.3f F1=%.3f", prec, rec, f1)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logging.info("Saved model -> %s and scaler -> %s", model_path, scaler_path)


def monitor_stream(
    topic_key: str,
    broker: str,
    model_path: Path,
    scaler_path: Path,
    window_size: int,
    threshold: float,
    duration: int | None,
) -> None:
    """Stream MQTT data and raise an alarm when model probability >= threshold."""
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler missing. Train before monitoring.")

    model = load_model(model_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    buffer: Deque[List[float]] = deque(maxlen=window_size)
    y_true: List[int] = []
    y_pred: List[int] = []
    topic = TOPICS[topic_key]

    client = mqtt.Client()
    stop_event = threading.Event()
    start_time = time.time()

    def on_connect(_client, _userdata, _flags, rc):
        logging.info("Connected to broker %s (rc=%s)", broker, rc)
        _client.subscribe(topic)
        logging.info("Subscribed to %s", topic)

    def score_sequence(sequence: np.ndarray) -> float:
        scaled = scaler.transform(sequence).reshape(1, window_size, len(FEATURES))
        prob = float(model.predict(scaled, verbose=0)[0][0])
        return prob

    def on_message(_client, _userdata, msg):
        try:
            row = decode_payload(msg.payload, topic_key)
            buffer.append([row[feat] for feat in FEATURES])
            if len(buffer) == window_size:
                seq = np.array(buffer)
                prob = score_sequence(seq)
                label = row.get(LABEL_COL)
                if label in (0, 1):
                    y_true.append(int(label))
                    y_pred.append(int(prob >= threshold))
                label_text = f"label={label}" if label in (0, 1) else "label=?"
                if prob >= threshold:
                    logging.warning(
                        "ALERT %.2f%% probability of fault on %s (%s)",
                        prob * 100,
                        topic_key,
                        label_text,
                    )
                else:
                    logging.info(
                        "OK %.2f%% probability (topic=%s, %s)",
                        prob * 100,
                        topic_key,
                        label_text,
                    )
            if duration and (time.time() - start_time) >= duration:
                stop_event.set()
                client.disconnect()
        except Exception as exc:
            logging.error("Monitoring error: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message

    def shutdown(_signum=None, _frame=None):
        logging.info("Stopping monitor...")
        stop_event.set()
        client.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.info(
        "Monitoring %s with threshold %.2f (window=%d)",
        topic_key,
        threshold,
        window_size,
    )
    client.connect(broker, 1883, keepalive=60)
    client.loop_start()

    try:
        if duration:
            stop_event.wait(timeout=duration)
            shutdown()
        else:
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        shutdown()
    finally:
        client.loop_stop()

    if y_true:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logging.info(
            "Monitoring summary (threshold=%.2f, samples=%d): TP=%d, TN=%d, FP=%d, FN=%d",
            threshold,
            len(y_true),
            tp,
            tn,
            fp,
            fn,
        )
        logging.info("Monitoring Precision=%.3f Recall=%.3f F1=%.3f", precision, recall, f1)
    else:
        logging.info("No labelled windows observed; skipping metrics.")


def sweep_thresholds(
    csv_path: Path,
    model_path: Path,
    scaler_path: Path,
    window_size: int,
    start: float,
    end: float,
    step: float,
) -> None:
    """Evaluate precision/recall/F1 over a range of thresholds."""
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Trained model or scaler not found. Run 'train' first.")

    df = load_dataset(csv_path)
    df = df.dropna(subset=FEATURES + [LABEL_COL])
    if len(df) <= window_size:
        raise ValueError("Not enough samples to create sequences; collect more data.")

    model = load_model(model_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    X = df[FEATURES].values
    y = df[LABEL_COL].values
    X_scaled = scaler.transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, window_size)

    probs = model.predict(X_seq, verbose=0).ravel()
    logging.info(
        "Sweeping thresholds from %.2f to %.2f (step=%.2f) on %d sequences",
        start,
        end,
        step,
        len(y_seq),
    )
    logging.info("thr\tprecision\trecall\tF1\tTP\tTN\tFP\tFN")

    thr = start
    best_f1 = -1.0
    best_thr = start
    while thr <= end + 1e-9:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_seq, preds, labels=[0, 1]).ravel()
        precision = precision_score(y_seq, preds, zero_division=0)
        recall = recall_score(y_seq, preds, zero_division=0)
        f1 = f1_score(y_seq, preds, zero_division=0)
        logging.info(
            "%.2f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d",
            thr,
            precision,
            recall,
            f1,
            tp,
            tn,
            fp,
            fn,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
        thr += step

    logging.info("Best F1 = %.3f at threshold %.2f", best_f1, best_thr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bioreactor anomaly detector")
    parser.add_argument("--broker", default=BROKER, help="MQTT broker hostname")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Capture MQTT telemetry into the CSV dataset"
    )
    collect_parser.add_argument(
        "--topic",
        required=True,
        choices=sorted(TOPICS.keys()),
        help="Which stream to collect",
    )
    collect_parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="How long to listen (seconds)",
    )
    collect_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Where to store telemetry",
    )

    train_parser = subparsers.add_parser("train", help="Train the LSTM model")
    train_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER)
    train_parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=64)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor a live stream and raise alarms"
    )
    monitor_parser.add_argument(
        "--topic",
        required=True,
        choices=sorted(TOPICS.keys()),
        help="Stream to monitor",
    )
    monitor_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    monitor_parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER)
    monitor_parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    monitor_parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Probability threshold for raising alarms",
    )
    monitor_parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="How long to monitor (seconds). 0 = run until interrupted",
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="Evaluate precision/recall/F1 across thresholds"
    )
    sweep_parser.add_argument("--csv", type=Path, default=Path("three_faults_data.csv"))
    sweep_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    sweep_parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER)
    sweep_parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    sweep_parser.add_argument(
        "--start",
        type=float,
        default=0.05,
        help="Starting threshold value",
    )
    sweep_parser.add_argument(
        "--end",
        type=float,
        default=0.95,
        help="End threshold value (inclusive)",
    )
    sweep_parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step between thresholds",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        collect_stream(args.topic, args.duration, args.csv, args.broker)
    elif args.command == "train":
        train_lstm(
            csv_path=args.csv,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.command == "monitor":
        monitor_stream(
            topic_key=args.topic,
            broker=args.broker,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            window_size=args.window_size,
            threshold=args.threshold,
            duration=args.duration or None,
        )
    elif args.command == "sweep":
        sweep_thresholds(
            csv_path=args.csv,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            window_size=args.window_size,
            start=args.start,
            end=args.end,
            step=args.step,
        )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
