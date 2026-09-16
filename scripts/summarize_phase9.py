from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


FIXED_RUNS = [
    (
        "final_lstm",
        "Final stable-feature LSTM forecaster",
        "runs/anomaly_final_event_sensitive_lstm_stable_top5_robust_z",
    ),
    (
        "proposed_attention",
        "Proposed multi-scale CNN-BiLSTM-attention forecaster",
        "runs/anomaly_final_proposed_attention_stable_max_robust_z",
    ),
    (
        "transformer_baseline",
        "Transformer forecasting baseline",
        "runs/anomaly_forecast_transformer_w100",
    ),
    (
        "cnn_transformer",
        "CNN-transformer forecasting candidate",
        "runs/anomaly_cnn_transformer_forecast_stable_features",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SWaT Phase 9 transformer experiments.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--cnn-transformer-sweep",
        default="runs/phase9_cnn_transformer_score_sweep.csv",
    )
    parser.add_argument(
        "--alarm-budget-selected",
        default="runs/phase9_alarm_budget_selected.csv",
    )
    parser.add_argument(
        "--output",
        default="runs/anomaly_transformer_gap_comparison.csv",
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    rows: list[dict[str, Any]] = []
    for run_id, description, run_dir_text in FIXED_RUNS:
        rows.append(
            {
                "comparison_type": "fixed_config",
                "run_id": run_id,
                "description": description,
                **_metrics_row(root / run_dir_text),
            }
        )

    cnn_sweep = pd.read_csv(root / args.cnn_transformer_sweep)
    rows.extend(
        _best_sweep_rows(
            cnn_sweep,
            run_id="cnn_transformer",
            description="CNN-transformer best threshold/post-processing sweep row",
        )
    )

    alarm_budget = pd.read_csv(root / args.alarm_budget_selected)
    rows.extend(_best_alarm_budget_rows(alarm_budget))

    _write_csv(root / args.output, rows)
    print(f"Wrote {args.output}")
    return 0


def _metrics_row(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    attack = metrics.get("per_class", {}).get("attack", {})
    operational = metrics.get("operational_metrics", {})
    latency = metrics.get("latency", {})
    event_precision = operational.get("event_precision")
    event_recall = operational.get("event_recall")
    if event_precision in {"", None}:
        detected_events = _float_or_none(operational.get("detected_event_count"))
        false_alarm_segments = _float_or_none(operational.get("false_alarm_segment_count"))
        if detected_events is not None and false_alarm_segments is not None:
            denominator = detected_events + false_alarm_segments
            event_precision = detected_events / denominator if denominator else 0.0
    event_f1 = operational.get("event_f1")
    if event_f1 in {"", None}:
        event_precision_float = _float_or_none(event_precision)
        event_recall_float = _float_or_none(event_recall)
        if event_precision_float is not None and event_recall_float is not None:
            denominator = event_precision_float + event_recall_float
            event_f1 = 2 * event_precision_float * event_recall_float / denominator if denominator else 0.0
    return {
        "experiment": metrics.get("experiment", run_dir.name),
        "model_name": metrics.get("model_name", ""),
        "score_method": metrics.get("anomaly_score_method", "global_mse"),
        "percentile": metrics.get("anomaly_threshold_percentile", ""),
        "min_consecutive_windows": metrics.get("anomaly_min_consecutive_windows", ""),
        "merge_gap_windows": metrics.get("anomaly_merge_gap_windows", ""),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "attack_precision": attack.get("precision", ""),
        "attack_recall": attack.get("recall", ""),
        "attack_f1": attack.get("f1", ""),
        "attack_fpr": attack.get("false_positive_rate", ""),
        "official_event_recall": event_recall if event_recall is not None else "",
        "official_event_precision": event_precision if event_precision is not None else "",
        "official_event_f1": event_f1 if event_f1 is not None else "",
        "false_alarm_segments": operational.get("false_alarm_segment_count", ""),
        "false_alarm_windows": operational.get("false_alarm_windows", ""),
        "false_alarm_segments_per_hour": operational.get("false_alarm_segments_per_hour", ""),
        "mean_detection_delay_seconds": operational.get("mean_detection_delay_seconds", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "metrics_path": str(metrics_path),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_sweep_rows(df: pd.DataFrame, run_id: str, description: str) -> list[dict[str, Any]]:
    metric_to_label = {
        "attack_f1": "best_cnn_transformer_attack_f1",
        "macro_f1": "best_cnn_transformer_macro_f1",
        "accuracy": "best_cnn_transformer_accuracy",
        "event_recall": "best_cnn_transformer_event_recall",
    }
    rows = []
    for metric, label in metric_to_label.items():
        best = df.sort_values(metric, ascending=False).iloc[0]
        rows.append(_csv_series_row(best, comparison_type=label, run_id=run_id, description=description))
    return rows


def _best_alarm_budget_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for experiment, group in df.groupby("experiment", sort=True):
        best = group.sort_values("attack_f1", ascending=False).iloc[0]
        rows.append(
            _csv_series_row(
                best,
                comparison_type="best_alarm_budget_attack_f1",
                run_id=str(experiment),
                description="Best Phase 9 alarm-budget-selected row by strict attack F1",
            )
        )
    return rows


def _csv_series_row(
    row: pd.Series,
    comparison_type: str,
    run_id: str,
    description: str,
) -> dict[str, Any]:
    return {
        "comparison_type": comparison_type,
        "run_id": run_id,
        "description": description,
        "experiment": row.get("experiment", ""),
        "model_name": row.get("model_name", ""),
        "score_method": row.get("score_method", ""),
        "percentile": row.get("percentile", ""),
        "min_consecutive_windows": row.get("min_consecutive_windows", ""),
        "merge_gap_windows": row.get("merge_gap_windows", ""),
        "accuracy": row.get("accuracy", ""),
        "macro_f1": row.get("macro_f1", ""),
        "weighted_f1": row.get("weighted_f1", ""),
        "attack_precision": row.get("attack_precision", ""),
        "attack_recall": row.get("attack_recall", ""),
        "attack_f1": row.get("attack_f1", ""),
        "attack_fpr": row.get("attack_fpr", ""),
        "official_event_recall": row.get("event_recall", ""),
        "official_event_precision": row.get("event_precision", ""),
        "official_event_f1": row.get("event_f1", ""),
        "false_alarm_segments": row.get("false_alarm_segment_count", ""),
        "false_alarm_windows": row.get("false_alarm_windows", ""),
        "false_alarm_segments_per_hour": row.get("false_alarm_segments_per_hour", ""),
        "mean_detection_delay_seconds": row.get("mean_detection_delay_seconds", ""),
        "parameter_count": row.get("parameter_count", ""),
        "seconds_per_sample": row.get("seconds_per_sample", ""),
        "samples_per_second": row.get("samples_per_second", ""),
        "metrics_path": row.get("metrics_path", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison_type",
        "run_id",
        "description",
        "experiment",
        "model_name",
        "score_method",
        "percentile",
        "min_consecutive_windows",
        "merge_gap_windows",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "attack_fpr",
        "official_event_recall",
        "official_event_precision",
        "official_event_f1",
        "false_alarm_segments",
        "false_alarm_windows",
        "false_alarm_segments_per_hour",
        "mean_detection_delay_seconds",
        "parameter_count",
        "seconds_per_sample",
        "samples_per_second",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
