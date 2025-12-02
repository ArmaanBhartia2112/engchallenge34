"""
Simple windowed statistical detector for bioreactor streams.

Workflow:
  1) Fit a baseline on fault-free data:
       python statistical.py fit --csv <your_fault_free_csv>
  2) Monitor a live topic using pooled z-score with hysteresis:
       python statistical.py monitor --topic three_faults --threshold-on 3.0 --threshold-off 2.0
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
from typing import Dict, List

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


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
    "temperature_mean",  # therm_voltage_bias
    "pH_mean",           # ph_offset_bias
    "heater_pwm",        # heater_power_loss
    "heater_energy_Wh",
]
LABEL_COL = "fault_label"
COLUMNS: List[str] = [
    "timestamp",
    *FEATURES,
    LABEL_COL,
    "source_topic",
]

BASELINE_FILE = Path("stats_baseline.json")
DEFAULT_CSV = Path("bioreactor.csv")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset {csv_path} not found.")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(COLUMNS).issubset(df.columns):
        legacy_cols = [
            "timestamp",
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
            "fault_label",
            "source_topic",
        ]
        if len(df.columns) == len(COLUMNS):
            logging.warning("Missing headers detected; applying defaults.")
            df.columns = COLUMNS
        elif len(df.columns) == len(legacy_cols):
            logging.warning("Detected legacy schema; renaming and subsetting to current features.")
            df.columns = legacy_cols
            df = df[["timestamp", *FEATURES, LABEL_COL, "source_topic"]]
        else:
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    return df


def ensure_dataset(csv_path: Path) -> None:
    """Create a CSV with the expected header if it does not exist."""
    if not csv_path.exists():
        logging.info("Creating dataset at %s", csv_path)
        pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)


def decode_payload(payload: bytes) -> Dict[str, float]:
    data = json.loads(payload.decode())
    return {
        "timestamp": data.get("timestamp")
        or data.get("time")
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
            row = decode_payload(msg.payload)
            row["source_topic"] = topic_key
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


def compute_baseline(csv_path: Path, baseline_path: Path, window_size: int) -> None:
    """
    Compute mean/std per feature and pooled z-score statistics from fault-free windows.
    """
    df = load_dataset(csv_path)
    normal_df = df[df[LABEL_COL] == 0]
    if normal_df.empty:
        raise ValueError("No fault-free samples found to build baseline.")

    stats = {}
    for feat in FEATURES:
        mean = float(normal_df[feat].mean())
        std = float(normal_df[feat].std(ddof=0))
        stats[feat] = {"mean": mean, "std": std if std > 1e-6 else 1e-6}

    baseline_path.write_text(json.dumps({"features": stats, "window_size": window_size}, indent=2))
    logging.info("Saved baseline to %s", baseline_path)


def load_baseline(baseline_path: Path) -> Dict[str, Dict[str, float]]:
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file {baseline_path} not found. Run 'fit'.")
    return json.loads(baseline_path.read_text())


def pooled_z(sample: Dict[str, float], baseline: Dict[str, Dict[str, float]]) -> float:
    z_squares = []
    for feat in FEATURES:
        ref = baseline["features"][feat]
        z = (sample[feat] - ref["mean"]) / ref["std"]
        z_squares.append(z * z)
    return float(np.sqrt(sum(z_squares) / len(z_squares)))


def monitor_stream(
    topic_key: str,
    broker: str,
    baseline_path: Path,
    threshold_on: float,
    threshold_off: float,
    window_size: int,
    duration: int | None,
) -> None:
    baseline = load_baseline(baseline_path)
    topic = TOPICS[topic_key]
    client = mqtt.Client()
    stop_event = threading.Event()
    start_time = time.time()
    buffer = deque(maxlen=window_size)
    alarm = False

    y_true: List[int] = []
    y_pred: List[int] = []

    def on_connect(_client, _userdata, _flags, rc):
        logging.info("Connected to %s (rc=%s)", broker, rc)
        _client.subscribe(topic)
        logging.info("Subscribed to %s", topic)

    def on_message(_client, _userdata, msg):
        nonlocal alarm
        try:
            row = decode_payload(msg.payload)
            buffer.append(row)
            if len(buffer) == window_size:
                # Average z over window
                z_vals = [pooled_z(sample, baseline) for sample in buffer]
                score = float(np.mean(z_vals))
                label = row.get(LABEL_COL)

                if not alarm and score >= threshold_on:
                    alarm = True
                    logging.warning(
                        "ALERT pooled z=%.2f (>= %.2f) topic=%s label=%s",
                        score,
                        threshold_on,
                        topic_key,
                        label,
                    )
                elif alarm and score <= threshold_off:
                    alarm = False
                    logging.info(
                        "Alarm cleared pooled z=%.2f (<= %.2f) topic=%s label=%s",
                        score,
                        threshold_off,
                        topic_key,
                        label,
                    )
                else:
                    state = "ALARM" if alarm else "OK"
                    logging.info("%s pooled z=%.2f topic=%s label=%s", state, score, topic_key, label)

                if label in (0, 1):
                    y_true.append(int(label))
                    y_pred.append(1 if alarm else 0)

            if duration and (time.time() - start_time) >= duration:
                stop_event.set()
                client.disconnect()
        except Exception as exc:
            logging.error("Monitoring error: %s", exc)

    def shutdown(_signum=None, _frame=None):
        logging.info("Stopping monitor…")
        stop_event.set()
        client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.info(
        "Monitoring %s (window=%d, threshold_on=%.2f, threshold_off=%.2f, duration=%s)",
        topic_key,
        window_size,
        threshold_on,
        threshold_off,
        duration or "infinite",
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
        logging.info("Monitoring summary: TP=%d TN=%d FP=%d FN=%d", tp, tn, fp, fn)
        logging.info("Precision=%.3f Recall=%.3f F1=%.3f", precision, recall, f1)
    else:
        logging.info("No labelled samples observed; skipping metrics.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Windowed statistical detector")
    parser.add_argument("--broker", default=BROKER, help="MQTT broker hostname")
    sub = parser.add_subparsers(dest="command", required=True)

    fit_parser = sub.add_parser("fit", help="Compute baseline stats from fault-free CSV")
    fit_parser.add_argument("--csv", type=Path, required=True, help="Path to fault-free CSV")
    fit_parser.add_argument("--baseline-path", type=Path, default=BASELINE_FILE)
    fit_parser.add_argument("--window-size", type=int, default=10)

    collect_parser = sub.add_parser(
        "collect", help="Capture MQTT telemetry into a CSV (default bioreactor.csv)"
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

    monitor_parser = sub.add_parser("monitor", help="Run pooled z-score detection")
    monitor_parser.add_argument(
        "--topic", required=True, choices=sorted(TOPICS.keys()), help="MQTT stream"
    )
    monitor_parser.add_argument("--baseline-path", type=Path, default=BASELINE_FILE)
    monitor_parser.add_argument("--threshold-on", type=float, default=3.0)
    monitor_parser.add_argument("--threshold-off", type=float, default=2.0)
    monitor_parser.add_argument("--window-size", type=int, default=30)
    monitor_parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="How long to listen (seconds). 0 = run until interrupted",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        compute_baseline(args.csv, args.baseline_path, args.window_size)
    elif args.command == "collect":
        collect_stream(args.topic, args.duration, args.csv, args.broker)
    elif args.command == "monitor":
        monitor_stream(
            topic_key=args.topic,
            broker=args.broker,
            baseline_path=args.baseline_path,
            threshold_on=args.threshold_on,
            threshold_off=args.threshold_off,
            window_size=args.window_size,
            duration=args.duration or None,
        )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
