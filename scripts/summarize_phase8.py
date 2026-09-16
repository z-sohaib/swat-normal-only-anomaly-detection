from __future__ import annotations

import argparse
import csv
import json
import statistics
import tomllib
from pathlib import Path
from typing import Any


MULTISEED_GROUPS = {
    "stable_feature_lstm_forecaster": [
        "runs/anomaly_final_event_sensitive_lstm_stable_top5_robust_z",
        "runs/phase8_lstm_seed7",
        "runs/phase8_lstm_seed123",
    ],
    "proposed_attention_forecaster": [
        "runs/anomaly_final_proposed_attention_stable_max_robust_z",
        "runs/phase8_attention_seed7",
        "runs/phase8_attention_seed123",
    ],
    "transformer_forecaster": [
        "runs/anomaly_forecast_transformer_w100",
        "runs/phase8_transformer_seed7",
        "runs/phase8_transformer_seed123",
    ],
}


ABLATION_RUNS = [
    (
        "reconstruction_baseline",
        "LSTM reconstruction baseline",
        "runs/anomaly_lstm_autoencoder_w100",
    ),
    (
        "forecasting_raw_features",
        "LSTM forecasting instead of reconstruction",
        "runs/anomaly_forecast_lstm_w100",
    ),
    (
        "forecasting_stable_features",
        "LSTM forecasting with stable-feature filtering",
        "runs/anomaly_forecast_lstm_w100_stable_features",
    ),
    (
        "final_lstm_robust_postprocessing",
        "Stable-feature LSTM with robust scoring and post-processing",
        "runs/anomaly_final_event_sensitive_lstm_stable_top5_robust_z",
    ),
    (
        "final_attention_robust_postprocessing",
        "Proposed attention forecaster with robust scoring",
        "runs/anomaly_final_proposed_attention_stable_max_robust_z",
    ),
    (
        "transformer_forecaster",
        "Transformer forecasting baseline",
        "runs/anomaly_forecast_transformer_w100",
    ),
]


ABLATION_DELTAS = [
    (
        "forecasting_vs_reconstruction",
        "Effect of forecasting objective",
        "reconstruction_baseline",
        "forecasting_raw_features",
    ),
    (
        "stable_features_vs_raw_features",
        "Effect of stable-feature filtering",
        "forecasting_raw_features",
        "forecasting_stable_features",
    ),
    (
        "robust_scoring_postprocessing_vs_default",
        "Effect of robust per-sensor scoring and temporal post-processing",
        "forecasting_stable_features",
        "final_lstm_robust_postprocessing",
    ),
    (
        "attention_vs_lstm_final",
        "Effect of proposed attention architecture at final operating point",
        "final_lstm_robust_postprocessing",
        "final_attention_robust_postprocessing",
    ),
    (
        "transformer_vs_lstm_raw_forecast",
        "Effect of transformer sequence model under raw-feature forecast setup",
        "forecasting_raw_features",
        "transformer_forecaster",
    ),
]


METRIC_FIELDS = [
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
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SWaT Phase 8 stability and ablations.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--multiseed-detail-output", default="runs/anomaly_multiseed_detail.csv")
    parser.add_argument("--multiseed-summary-output", default="runs/anomaly_multiseed_summary.csv")
    parser.add_argument("--ablation-output", default="runs/anomaly_ablation_summary.csv")
    parser.add_argument("--ablation-delta-output", default="runs/anomaly_ablation_deltas.csv")
    args = parser.parse_args()

    root = Path(args.project_root)
    multiseed_rows = []
    for group, run_dirs in MULTISEED_GROUPS.items():
        for run_dir_text in run_dirs:
            multiseed_rows.append(_row(root / run_dir_text, group=group))

    summary_rows = _summary_rows(multiseed_rows)
    ablation_rows = [
        {"ablation_id": ablation_id, "description": description, **_row(root / run_dir)}
        for ablation_id, description, run_dir in ABLATION_RUNS
    ]
    ablation_by_id = {row["ablation_id"]: row for row in ablation_rows}
    delta_rows = [
        _delta_row(delta_id, description, ablation_by_id[base], ablation_by_id[variant])
        for delta_id, description, base, variant in ABLATION_DELTAS
    ]

    _write_csv(root / args.multiseed_detail_output, multiseed_rows)
    _write_csv(root / args.multiseed_summary_output, summary_rows)
    _write_csv(root / args.ablation_output, ablation_rows)
    _write_csv(root / args.ablation_delta_output, delta_rows)
    print(f"Wrote {args.multiseed_detail_output}")
    print(f"Wrote {args.multiseed_summary_output}")
    print(f"Wrote {args.ablation_output}")
    print(f"Wrote {args.ablation_delta_output}")
    return 0


def _row(run_dir: Path, group: str = "") -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.toml"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    attack = metrics.get("per_class", {}).get("attack", {})
    operational = metrics.get("operational_metrics", {})
    latency = metrics.get("latency", {})
    return {
        "group": group,
        "experiment": metrics.get("experiment", run_dir.name),
        "seed": config.get("experiment", {}).get("seed", ""),
        "model_name": metrics.get("model_name", ""),
        "loss": metrics.get("loss", ""),
        "score_method": metrics.get("anomaly_score_method", "global_mse"),
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
        "event_recall": operational.get("event_recall", ""),
        "false_alarm_segments": operational.get("false_alarm_segment_count", ""),
        "false_alarm_windows": operational.get("false_alarm_windows", ""),
        "false_alarm_segments_per_hour": operational.get("false_alarm_segments_per_hour", ""),
        "mean_detection_delay_seconds": operational.get("mean_detection_delay_seconds", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "best_val_reconstruction_loss": metrics.get("best_selected_score", ""),
        "metrics_path": str(metrics_path),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({str(row["group"]) for row in rows})
    summaries = []
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        payload: dict[str, Any] = {"group": group, "seed_count": len(group_rows)}
        for field in METRIC_FIELDS:
            values = [_float(row.get(field)) for row in group_rows if row.get(field) != ""]
            payload[f"{field}_mean"] = statistics.fmean(values) if values else ""
            payload[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            payload[f"{field}_min"] = min(values) if values else ""
            payload[f"{field}_max"] = max(values) if values else ""
        summaries.append(payload)
    return summaries


def _delta_row(
    delta_id: str,
    description: str,
    base: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "delta_id": delta_id,
        "description": description,
        "base": base["ablation_id"],
        "variant": variant["ablation_id"],
    }
    for field in METRIC_FIELDS:
        payload[f"{field}_base"] = base.get(field, "")
        payload[f"{field}_variant"] = variant.get(field, "")
        payload[f"{field}_delta"] = _float(variant.get(field)) - _float(base.get(field))
    return payload


def _float(value: Any) -> float:
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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
