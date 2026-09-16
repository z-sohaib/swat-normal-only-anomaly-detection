from __future__ import annotations

import argparse
import csv
import json
import statistics
import tomllib
from pathlib import Path
from typing import Any


GROUPS = {
    "final_wandb_lstm": [
        "runs/final_wandb_lstm_seed42",
        "runs/final_wandb_lstm_seed7",
        "runs/final_wandb_lstm_seed123",
    ],
    "final_wandb_attention": [
        "runs/final_wandb_attention_seed42",
        "runs/final_wandb_attention_seed7",
        "runs/final_wandb_attention_seed123",
    ],
}


METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "attack_fpr",
    "event_recall",
    "false_alarm_segments",
    "false_alarm_windows",
    "false_alarm_segments_per_hour",
    "mean_detection_delay_seconds",
    "seconds_per_sample",
    "samples_per_second",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize final SWaT W&B multi-seed reruns.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--detail-output", default="runs/final_wandb_multiseed_detail.csv")
    parser.add_argument("--summary-output", default="runs/final_wandb_multiseed_summary.csv")
    args = parser.parse_args()

    root = Path(args.project_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for group, run_dirs in GROUPS.items():
        for run_dir in run_dirs:
            path = root / run_dir
            if not (path / "metrics.json").exists():
                missing.append(run_dir)
                continue
            rows.append(_row(path, group))

    _write_csv(root / args.detail_output, rows)
    _write_csv(root / args.summary_output, _summary_rows(rows))
    print(f"Wrote {args.detail_output}")
    print(f"Wrote {args.summary_output}")
    if missing:
        print("Missing runs:")
        for item in missing:
            print(f"- {item}")
    return 0


def _row(run_dir: Path, group: str) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.toml"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    attack = metrics.get("per_class", {}).get("attack", {})
    normal = metrics.get("per_class", {}).get("normal", {})
    operational = metrics.get("operational_metrics", {})
    latency = metrics.get("latency", {})
    return {
        "group": group,
        "experiment": metrics.get("experiment", run_dir.name),
        "seed": config.get("experiment", {}).get("seed", ""),
        "model_name": metrics.get("model_name", ""),
        "loss": metrics.get("loss", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "score_method": metrics.get("anomaly_score_method", ""),
        "threshold_percentile": metrics.get("anomaly_threshold_percentile", ""),
        "min_consecutive_windows": metrics.get("anomaly_min_consecutive_windows", ""),
        "merge_gap_windows": metrics.get("anomaly_merge_gap_windows", ""),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "attack_precision": attack.get("precision", ""),
        "attack_recall": attack.get("recall", ""),
        "attack_f1": attack.get("f1", ""),
        "attack_fpr": attack.get("false_positive_rate", ""),
        "normal_recall": normal.get("recall", ""),
        "event_recall": operational.get("event_recall", ""),
        "false_alarm_segments": operational.get("false_alarm_segment_count", ""),
        "false_alarm_windows": operational.get("false_alarm_windows", ""),
        "false_alarm_segments_per_hour": operational.get("false_alarm_segments_per_hour", ""),
        "mean_detection_delay_seconds": operational.get("mean_detection_delay_seconds", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "metrics_path": str(metrics_path),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        payload: dict[str, Any] = {"group": group, "seed_count": len(group_rows)}
        for metric in METRICS:
            values = [_to_float(row.get(metric)) for row in group_rows if row.get(metric) != ""]
            payload[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            payload[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            payload[f"{metric}_min"] = min(values) if values else ""
            payload[f"{metric}_max"] = max(values) if values else ""
        output.append(payload)
    return output


def _to_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
