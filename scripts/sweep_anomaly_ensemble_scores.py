from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swat_ids.config import load_config
from swat_ids.metrics import classification_metrics
from swat_ids.operational import load_official_attack_list
from swat_ids.postprocessing import postprocess_anomaly_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep ensembles of cached SWaT anomaly scores with causal online normalization."
    )
    parser.add_argument("--run", action="append", required=True, help="Run directory with validation/window scores.")
    parser.add_argument("--methods", nargs="+", default=["delta", "robust_z", "ratio"])
    parser.add_argument("--adaptive-windows", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[95, 97.5, 99, 99.5, 99.9])
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--merge-gaps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--combine-methods", nargs="+", default=["weighted_mean", "mean", "max"])
    parser.add_argument("--output", default="runs/anomaly_ensemble_online_sweep.csv")
    args = parser.parse_args()

    runs = [_load_run(Path(run_text)) for run_text in args.run]
    _validate_alignment(runs)

    rows = []
    for method in args.methods:
        for adaptive_window in args.adaptive_windows:
            transformed = [
                _transform_run_scores(run, method=method, adaptive_window=adaptive_window)
                for run in runs
            ]
            for combine_method in args.combine_methods:
                weights = args.weights if combine_method == "weighted_mean" and len(runs) == 2 else [None]
                for weight_first in weights:
                    validation_score, test_score = _combine_scores(
                        transformed,
                        combine_method=combine_method,
                        weight_first=weight_first,
                    )
                    for percentile in args.percentiles:
                        threshold = float(np.percentile(validation_score, percentile))
                        raw_pred = (test_score > threshold).astype(np.int64)
                        for min_consecutive in args.min_consecutive:
                            for merge_gap in args.merge_gaps:
                                pred = postprocess_anomaly_predictions(
                                    raw_pred,
                                    min_consecutive_windows=min_consecutive,
                                    merge_gap_windows=merge_gap,
                                )
                                point = classification_metrics(runs[0]["y_true"], pred, ["normal", "attack"])
                                events = _fast_official_event_metrics(
                                    pred,
                                    runs[0]["event_masks"],
                                    runs[0]["window_ends"],
                                    runs[0]["total_hours"],
                                )
                                rows.append(
                                    _row(
                                        runs=runs,
                                        method=method,
                                        adaptive_window=adaptive_window,
                                        combine_method=combine_method,
                                        weight_first=weight_first,
                                        percentile=percentile,
                                        threshold=threshold,
                                        point=point,
                                        events=events,
                                        min_consecutive=min_consecutive,
                                        merge_gap=merge_gap,
                                    )
                                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, rows)
    print(f"Wrote {output}")
    return 0


def _load_run(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.toml")
    _, validation_scores = _read_scores(run_dir / "validation_scores.csv")
    y_true, test_scores, window_starts, window_ends, origin = _read_test_scores(run_dir / "window_scores.csv")
    attack_list_path = config.data.attack_list_path or Path("data/raw/List_of_attacks_Final.xlsx")
    event_masks = _event_masks(
        load_official_attack_list(attack_list_path),
        window_starts=window_starts,
        window_ends=window_ends,
        origin=origin,
    )
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "y_true": y_true,
        "window_starts": window_starts,
        "window_ends": window_ends,
        "origin": origin,
        "event_masks": event_masks,
        "total_hours": _duration_hours(window_starts, window_ends),
    }


def _validate_alignment(runs: list[dict[str, Any]]) -> None:
    if len(runs) < 2:
        raise ValueError("At least two runs are required for ensemble scoring.")
    first = runs[0]
    for run in runs[1:]:
        if len(run["validation_scores"]) != len(first["validation_scores"]):
            raise ValueError("Validation score lengths do not match across runs.")
        if len(run["test_scores"]) != len(first["test_scores"]):
            raise ValueError("Test score lengths do not match across runs.")
        if not np.array_equal(run["y_true"], first["y_true"]):
            raise ValueError("Test labels do not match across runs.")
        if not np.allclose(run["window_starts"], first["window_starts"]):
            raise ValueError("Window starts do not match across runs.")
        if not np.allclose(run["window_ends"], first["window_ends"]):
            raise ValueError("Window ends do not match across runs.")


def _transform_run_scores(run: dict[str, Any], method: str, adaptive_window: int) -> dict[str, np.ndarray]:
    validation = _online_scores(
        run["validation_scores"],
        seed_scores=run["validation_scores"][:adaptive_window],
        method=method,
        adaptive_window=adaptive_window,
    )
    test = _online_scores(
        run["test_scores"],
        seed_scores=run["validation_scores"],
        method=method,
        adaptive_window=adaptive_window,
    )
    median = float(np.median(validation))
    q25 = float(np.percentile(validation, 25))
    q75 = float(np.percentile(validation, 75))
    iqr = max(q75 - q25, 1e-8)
    return {
        "validation": np.maximum((validation - median) / iqr, 0.0),
        "test": np.maximum((test - median) / iqr, 0.0),
    }


def _combine_scores(
    transformed: list[dict[str, np.ndarray]],
    combine_method: str,
    weight_first: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    validation_stack = np.vstack([item["validation"] for item in transformed])
    test_stack = np.vstack([item["test"] for item in transformed])
    if combine_method == "mean":
        return validation_stack.mean(axis=0), test_stack.mean(axis=0)
    if combine_method == "max":
        return validation_stack.max(axis=0), test_stack.max(axis=0)
    if combine_method == "weighted_mean":
        if len(transformed) != 2:
            raise ValueError("weighted_mean is only supported for two-run ensembles.")
        if weight_first is None:
            raise ValueError("weight_first is required for weighted_mean.")
        weight_second = 1.0 - weight_first
        validation = validation_stack[0] * weight_first + validation_stack[1] * weight_second
        test = test_stack[0] * weight_first + test_stack[1] * weight_second
        return validation, test
    raise ValueError(f"Unsupported combine method: {combine_method}")


def _online_scores(
    scores: np.ndarray,
    seed_scores: np.ndarray,
    method: str,
    adaptive_window: int,
) -> np.ndarray:
    history = deque((float(value) for value in seed_scores[-adaptive_window:]), maxlen=adaptive_window)
    transformed = []
    for score in scores:
        context = np.asarray(history, dtype=np.float64)
        median = float(np.median(context))
        q25 = float(np.percentile(context, 25))
        q75 = float(np.percentile(context, 75))
        iqr = max(q75 - q25, 1e-8)
        if method == "robust_z":
            value = max((float(score) - median) / iqr, 0.0)
        elif method == "ratio":
            value = float(score) / max(median, 1e-8)
        elif method == "delta":
            value = max(float(score) - median, 0.0)
        else:
            raise ValueError(f"Unsupported online score method: {method}")
        transformed.append(value)
        history.append(float(score))
    return np.asarray(transformed, dtype=np.float64)


def _read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    scores = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            scores.append(float(row["anomaly_score"]))
    return np.asarray(y_true, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _read_test_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime]:
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
        masks.append({"start_seconds": clipped_start, "mask": mask})
    return masks


def _fast_official_event_metrics(
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

    false_alarm_segments = 0
    for start, end in _segments(predicted):
        if not bool(official_any[start : end + 1].any()):
            false_alarm_segments += 1

    false_alarm_windows = int(((predicted == 1) & (~official_any)).sum())
    true_event_count = len(event_masks)
    return {
        "true_event_count": true_event_count,
        "detected_event_count": detected,
        "event_recall": detected / true_event_count if true_event_count else 0.0,
        "missed_event_count": missed,
        "false_alarm_segment_count": false_alarm_segments,
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_segments_per_hour": false_alarm_segments / total_hours if total_hours > 0 else 0.0,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
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


def _row(
    runs: list[dict[str, Any]],
    method: str,
    adaptive_window: int,
    combine_method: str,
    weight_first: float | None,
    percentile: float,
    threshold: float,
    point: dict[str, Any],
    events: dict[str, Any],
    min_consecutive: int,
    merge_gap: int,
) -> dict[str, Any]:
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    experiments = [run["metrics"].get("experiment", run["run_dir"].name) for run in runs]
    model_names = [run["metrics"].get("model_name", "") for run in runs]
    losses = [run["metrics"].get("loss", "") for run in runs]
    return {
        "experiment": "ensemble__" + "__".join(experiments),
        "ensemble_members": " + ".join(experiments),
        "model_names": " + ".join(model_names),
        "losses": " + ".join(losses),
        "window_size": runs[0]["metrics"].get("window_size", ""),
        "window_stride": runs[0]["metrics"].get("window_stride", ""),
        "online_method": method,
        "adaptive_window": adaptive_window,
        "combine_method": combine_method,
        "weight_first": "" if weight_first is None else weight_first,
        "percentile": percentile,
        "threshold": threshold,
        "min_consecutive_windows": min_consecutive,
        "merge_gap_windows": merge_gap,
        "accuracy": point["accuracy"],
        "macro_f1": point["macro_f1"],
        "weighted_f1": point["weighted_f1"],
        "attack_precision": attack["precision"],
        "attack_recall": attack["recall"],
        "attack_f1": attack["f1"],
        "attack_fpr": attack["false_positive_rate"],
        "normal_recall": normal["recall"],
        "true_event_count": events["true_event_count"],
        "detected_event_count": events["detected_event_count"],
        "event_recall": events["event_recall"],
        "missed_event_count": events["missed_event_count"],
        "false_alarm_segment_count": events["false_alarm_segment_count"],
        "false_alarm_windows": events["false_alarm_windows"],
        "false_alarm_segments_per_hour": events["false_alarm_segments_per_hour"],
        "mean_detection_delay_seconds": events["mean_detection_delay_seconds"],
        "metrics_path": " + ".join(str(run["run_dir"] / "metrics.json") for run in runs),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
