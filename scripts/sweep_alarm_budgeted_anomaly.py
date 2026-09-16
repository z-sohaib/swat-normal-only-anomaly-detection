from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_anomaly_score_methods import (  # noqa: E402
    _anomaly_input_target,
    _score_context,
    _score_from_feature_errors,
)
from swat_ids.config import load_config  # noqa: E402
from swat_ids.data import load_swat  # noqa: E402
from swat_ids.metrics import classification_metrics  # noqa: E402
from swat_ids.models import build_anomaly_model  # noqa: E402
from swat_ids.operational import load_official_attack_list  # noqa: E402
from swat_ids.postprocessing import postprocess_anomaly_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select SWaT anomaly operating points under validation alarm budgets."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "global_mse",
            "max_robust_z",
            "top3_robust_z",
            "top5_robust_z",
            "top10_robust_z",
        ],
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[90, 95, 97.5, 99, 99.5, 99.7, 99.9, 99.95, 99.99, 100],
    )
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--merge-gaps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--sweep-output", default="runs/anomaly_alarm_budget_sweep.csv")
    parser.add_argument("--selected-output", default="runs/anomaly_alarm_budget_selected.csv")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    sweep_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for run_text in args.run:
        run_dir = Path(run_text)
        candidates = _sweep_run(
            run_dir=run_dir,
            methods=args.methods,
            percentiles=args.percentiles,
            min_consecutive_values=args.min_consecutive,
            merge_gap_values=args.merge_gaps,
            batch_size=args.batch_size,
        )
        sweep_rows.extend(candidates)
        selected_rows.extend(_select_budgeted_rows(candidates, args.budgets))

    _write_rows(Path(args.sweep_output), sweep_rows)
    _write_rows(Path(args.selected_output), selected_rows)
    print(f"Wrote {args.sweep_output}")
    print(f"Wrote {args.selected_output}")
    return 0


def _sweep_run(
    run_dir: Path,
    methods: list[str],
    percentiles: list[float],
    min_consecutive_values: list[int],
    merge_gap_values: list[int],
    batch_size: int | None,
) -> list[dict[str, Any]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.toml")
    bundle = load_swat(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_anomaly_model(config.model, input_features=bundle.input_features).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))

    effective_batch_size = batch_size or config.training.batch_size
    y_val, val_feature_errors = _feature_errors(
        model,
        _loader(bundle.x_val, bundle.y_val, effective_batch_size),
        device,
        anomaly_loss=config.training.loss,
    )
    y_test, test_feature_errors = _feature_errors(
        model,
        _loader(bundle.x_test, bundle.y_test, effective_batch_size),
        device,
        anomaly_loss=config.training.loss,
    )
    if np.any(y_val != 0):
        raise ValueError("Alarm-budget calibration expects normal-only validation labels.")

    validation_timeline = _synthetic_timeline(
        first_timestamp=bundle.metadata["splits"]["validation"]["first_timestamp"],
        window_count=len(y_val),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )
    y_true_scores, _, test_starts, test_ends, origin = _read_test_score_timeline(run_dir / "window_scores.csv")
    if len(y_true_scores) != len(y_test) or not np.array_equal(y_true_scores, y_test):
        raise ValueError(f"Window-score labels do not match loaded test labels for {run_dir}.")

    attack_list_path = config.data.attack_list_path or Path("data/raw/List_of_attacks_Final.xlsx")
    event_masks = _event_masks(
        attack_list=load_official_attack_list(attack_list_path),
        window_starts=test_starts,
        window_ends=test_ends,
        origin=origin,
    )
    score_context = _score_context(val_feature_errors)
    rows: list[dict[str, Any]] = []

    for method in methods:
        val_scores = _score_from_feature_errors(val_feature_errors, method, score_context)
        test_scores = _score_from_feature_errors(test_feature_errors, method, score_context)
        for percentile in percentiles:
            threshold = float(np.percentile(val_scores, percentile))
            val_raw = (val_scores > threshold).astype(np.int64)
            test_raw = (test_scores > threshold).astype(np.int64)
            for min_consecutive in min_consecutive_values:
                for merge_gap in merge_gap_values:
                    val_pred = postprocess_anomaly_predictions(
                        val_raw,
                        min_consecutive_windows=min_consecutive,
                        merge_gap_windows=merge_gap,
                    )
                    test_pred = postprocess_anomaly_predictions(
                        test_raw,
                        min_consecutive_windows=min_consecutive,
                        merge_gap_windows=merge_gap,
                    )
                    val_alarm = _validation_alarm_metrics(val_pred, validation_timeline)
                    point = classification_metrics(y_test, test_pred, ["normal", "attack"])
                    events = _official_event_metrics(
                        y_pred=test_pred,
                        event_masks=event_masks,
                        window_ends=test_ends,
                        total_hours=_duration_hours(test_starts, test_ends),
                    )
                    rows.append(
                        _row(
                            run_dir=run_dir,
                            metrics=metrics,
                            method=method,
                            percentile=percentile,
                            threshold=threshold,
                            min_consecutive=min_consecutive,
                            merge_gap=merge_gap,
                            validation=val_alarm,
                            point=point,
                            events=events,
                        )
                    )
    return rows


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _feature_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    anomaly_loss: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    errors = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        reconstruction = model(model_input)
        batch_errors = torch.mean((reconstruction - target) ** 2, dim=1)
        errors.append(batch_errors.cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(errors)


def _synthetic_timeline(
    first_timestamp: str,
    window_count: int,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    origin = _parse_timestamp(first_timestamp)
    starts = np.asarray([idx * stride for idx in range(window_count)], dtype=np.float64)
    ends = starts + window_size - 1
    _ = origin
    return starts, ends


def _parse_timestamp(value: str) -> datetime:
    for fmt in ("%d/%m/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value)


def _read_test_score_timeline(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime]:
    y_true = []
    scores = []
    starts = []
    ends = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            scores.append(float(row["anomaly_score"]))
            starts.append(datetime.fromisoformat(row["start_time"]))
            ends.append(datetime.fromisoformat(row["end_time"]))

    if not starts:
        raise ValueError(f"No window scores found in {path}")
    origin = starts[0]
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray([(value - origin).total_seconds() for value in starts], dtype=np.float64),
        np.asarray([(value - origin).total_seconds() for value in ends], dtype=np.float64),
        origin,
    )


def _event_masks(
    attack_list: dict[str, Any],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    origin: datetime,
) -> list[dict[str, Any]]:
    masks = []
    timeline = {"start": float(window_starts[0]), "end": float(window_ends[-1])}
    for event in attack_list["events"]:
        start = (event["start"] - origin).total_seconds()
        end = (event["end"] - origin).total_seconds()
        if start > timeline["end"] or end < timeline["start"]:
            continue
        clipped_start = max(start, timeline["start"])
        clipped_end = min(end, timeline["end"])
        mask = (window_starts <= clipped_end) & (clipped_start <= window_ends)
        masks.append({"start_seconds": clipped_start, "end_seconds": clipped_end, "mask": mask})
    return masks


def _validation_alarm_metrics(
    y_pred: np.ndarray,
    timeline: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    segments = _segments(y_pred)
    starts, ends = timeline
    hours = _duration_hours(starts, ends)
    alarm_windows = int((y_pred == 1).sum())
    return {
        "validation_alarm_segment_count": len(segments),
        "validation_alarm_windows": alarm_windows,
        "validation_alarm_segments_per_hour": len(segments) / hours if hours > 0 else 0.0,
        "validation_alarm_window_rate": alarm_windows / len(y_pred) if len(y_pred) else 0.0,
    }


def _official_event_metrics(
    y_pred: np.ndarray,
    event_masks: list[dict[str, Any]],
    window_ends: np.ndarray,
    total_hours: float,
) -> dict[str, Any]:
    predicted = y_pred.astype(np.int64)
    official_any = np.zeros_like(predicted, dtype=bool)
    detected = 0
    missed = 0
    delays = []
    for event in event_masks:
        mask = event["mask"]
        official_any |= mask
        hit_indices = np.flatnonzero((predicted == 1) & mask)
        if len(hit_indices) == 0:
            missed += 1
            continue
        detected += 1
        first_hit = int(hit_indices[0])
        delays.append(max(float(window_ends[first_hit] - event["start_seconds"]), 0.0))

    pred_segments = _segments(predicted)
    true_positive_segments = 0
    false_alarm_segments = 0
    for start, end in pred_segments:
        if bool(official_any[start : end + 1].any()):
            true_positive_segments += 1
        else:
            false_alarm_segments += 1

    precision = true_positive_segments / len(pred_segments) if pred_segments else 0.0
    recall = detected / len(event_masks) if event_masks else 0.0
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_windows = int(((predicted == 1) & (~official_any)).sum())
    return {
        "true_event_count": len(event_masks),
        "predicted_event_count": len(pred_segments),
        "event_true_positive_segments": true_positive_segments,
        "detected_event_count": detected,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": event_f1,
        "missed_event_count": missed,
        "false_alarm_segment_count": false_alarm_segments,
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_segments_per_hour": false_alarm_segments / total_hours if total_hours > 0 else 0.0,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
        "p90_detection_delay_seconds": float(np.percentile(delays, 90)) if delays else None,
    }


def _segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for idx, value in enumerate(values):
        if value == 1 and start is None:
            start = idx
        elif value != 1 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _duration_hours(window_starts: np.ndarray, window_ends: np.ndarray) -> float:
    return max(float(window_ends[-1] - window_starts[0]) / 3600.0, 0.0)


def _select_budgeted_rows(rows: list[dict[str, Any]], budgets: list[float]) -> list[dict[str, Any]]:
    selected = []
    methods = sorted({str(row["score_method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["score_method"] == method]
        for budget in budgets:
            feasible = [
                row
                for row in method_rows
                if _float(row["validation_alarm_segments_per_hour"]) <= budget
            ]
            source = feasible or method_rows
            chosen = max(
                source,
                key=lambda row: (
                    _float(row["validation_alarm_windows"]),
                    _float(row["validation_alarm_segment_count"]),
                    -_float(row["percentile"]),
                    -_float(row["min_consecutive_windows"]),
                    -_float(row["merge_gap_windows"]),
                ),
            )
            payload = dict(chosen)
            payload["target_alarm_budget_per_hour"] = budget
            payload["budget_feasible_on_validation"] = bool(feasible)
            payload["selection_scope"] = "per_model_per_score_method"
            selected.append(payload)
    return selected


def _row(
    run_dir: Path,
    metrics: dict[str, Any],
    method: str,
    percentile: float,
    threshold: float,
    min_consecutive: int,
    merge_gap: int,
    validation: dict[str, Any],
    point: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    return {
        "experiment": metrics.get("experiment", run_dir.name),
        "model_name": metrics.get("model_name", ""),
        "loss": metrics.get("loss", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "score_method": method,
        "percentile": percentile,
        "threshold": threshold,
        "min_consecutive_windows": min_consecutive,
        "merge_gap_windows": merge_gap,
        **validation,
        "accuracy": point["accuracy"],
        "macro_f1": point["macro_f1"],
        "weighted_f1": point["weighted_f1"],
        "attack_precision": attack["precision"],
        "attack_recall": attack["recall"],
        "attack_f1": attack["f1"],
        "attack_fpr": attack["false_positive_rate"],
        "normal_precision": normal["precision"],
        "normal_recall": normal["recall"],
        "normal_f1": normal["f1"],
        **events,
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": metrics.get("latency", {}).get("seconds_per_sample", ""),
        "samples_per_second": metrics.get("latency", {}).get("samples_per_second", ""),
        "metrics_path": str(run_dir / "metrics.json"),
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
