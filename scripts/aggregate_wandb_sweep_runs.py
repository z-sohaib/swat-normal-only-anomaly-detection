from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate local SWaT W&B sweep metrics.")
    parser.add_argument("--runs-root", default="runs/wandb")
    parser.add_argument("--output", default="runs/wandb_sweep_summary.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in Path(args.runs_root).glob("*/*/metrics.json"):
        rows.append(_row(metrics_path))
    rows.sort(key=lambda row: _float(row.get("test_attack_f1")), reverse=True)
    _write_csv(Path(args.output), rows)
    print(f"Wrote {args.output} with {len(rows)} rows")
    return 0


def _row(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    attack = metrics.get("per_class", {}).get("attack", {})
    normal = metrics.get("per_class", {}).get("normal", {})
    operational = metrics.get("operational_metrics", {})
    latency = metrics.get("latency", {})
    config_path = metrics_path.with_name("config.toml")
    return {
        "experiment": metrics.get("experiment", metrics_path.parent.name),
        "model_name": metrics.get("model_name", ""),
        "loss": metrics.get("loss", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "score_method": metrics.get("anomaly_score_method", ""),
        "threshold_percentile": metrics.get("anomaly_threshold_percentile", ""),
        "min_consecutive_windows": metrics.get("anomaly_min_consecutive_windows", ""),
        "merge_gap_windows": metrics.get("anomaly_merge_gap_windows", ""),
        "test_accuracy": metrics.get("accuracy", ""),
        "test_macro_f1": metrics.get("macro_f1", ""),
        "test_weighted_f1": metrics.get("weighted_f1", ""),
        "test_attack_precision": attack.get("precision", ""),
        "test_attack_recall": attack.get("recall", ""),
        "test_attack_f1": attack.get("f1", ""),
        "test_attack_fpr": attack.get("false_positive_rate", ""),
        "test_normal_precision": normal.get("precision", ""),
        "test_normal_recall": normal.get("recall", ""),
        "test_normal_f1": normal.get("f1", ""),
        "official_event_recall": operational.get("event_recall", ""),
        "false_alarm_segments": operational.get("false_alarm_segment_count", ""),
        "false_alarm_windows": operational.get("false_alarm_windows", ""),
        "false_alarm_segments_per_hour": operational.get("false_alarm_segments_per_hour", ""),
        "mean_detection_delay_seconds": operational.get("mean_detection_delay_seconds", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "metrics_path": str(metrics_path),
        "config_path": str(config_path if config_path.exists() else metrics_path.with_name("resolved_config.toml")),
    }


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return -1.0
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
