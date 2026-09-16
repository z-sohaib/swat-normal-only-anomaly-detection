from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate SWaT run metrics.")
    parser.add_argument("--pattern", default="runs/supervised_*/metrics.json")
    parser.add_argument("--output", default="runs/supervised_comparison.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.pattern)):
        metrics_path = Path(path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_row(metrics, metrics_path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model_name",
        "data_protocol",
        "window_size",
        "window_stride",
        "window_label_rule",
        "loss",
        "selection_metric",
        "threshold_strategy",
        "attack_threshold",
        "best_selected_score",
        "validation_test_gap",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "fpr_attack",
        "precision_normal",
        "recall_normal",
        "f1_normal",
        "fpr_normal",
        "parameter_count",
        "seconds_per_sample",
        "samples_per_second",
        "device",
        "metrics_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {output}")
    return 0


def _row(metrics: dict[str, Any], path: Path) -> dict[str, Any]:
    normal = metrics.get("per_class", {}).get("normal", {})
    attack = metrics.get("per_class", {}).get("attack", {})
    latency = metrics.get("latency", {})
    return {
        "experiment": metrics.get("experiment", path.parent.name),
        "model_name": metrics.get("model_name", ""),
        "data_protocol": metrics.get("data_protocol", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "window_label_rule": metrics.get("window_label_rule", ""),
        "loss": metrics.get("loss", ""),
        "selection_metric": metrics.get("selection_metric", ""),
        "threshold_strategy": metrics.get("threshold_strategy", ""),
        "attack_threshold": metrics.get("attack_threshold", ""),
        "best_selected_score": metrics.get("best_selected_score", ""),
        "validation_test_gap": metrics.get("validation_test_gap", ""),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "precision_attack": attack.get("precision", ""),
        "recall_attack": attack.get("recall", ""),
        "f1_attack": attack.get("f1", ""),
        "fpr_attack": attack.get("false_positive_rate", ""),
        "precision_normal": normal.get("precision", ""),
        "recall_normal": normal.get("recall", ""),
        "f1_normal": normal.get("f1", ""),
        "fpr_normal": normal.get("false_positive_rate", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "device": metrics.get("device", ""),
        "metrics_path": str(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
