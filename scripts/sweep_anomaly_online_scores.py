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
    parser = argparse.ArgumentParser(description="Sweep causal online normalization of SWaT anomaly scores.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--methods", nargs="+", default=["robust_z", "ratio", "delta"])
    parser.add_argument("--adaptive-windows", nargs="+", type=int, default=[10, 25, 50, 100, 200])
    parser.add_argument("--percentiles", nargs="+", type=float, default=[95, 97.5, 99, 99.5, 99.9])
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--merge-gaps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument("--output", default="runs/anomaly_online_score_sweep.csv")
    args = parser.parse_args()

    rows = []
    for run_text in args.run:
        rows.extend(
            _sweep_run(
                run_dir=Path(run_text),
                methods=args.methods,
                adaptive_windows=args.adaptive_windows,
                percentiles=args.percentiles,
                min_consecutive_values=args.min_consecutive,
                merge_gap_values=args.merge_gaps,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, rows)
    print(f"Wrote {output}")
    return 0


def _sweep_run(
    run_dir: Path,
    methods: list[str],
    adaptive_windows: list[int],
    percentiles: list[float],
    min_consecutive_values: list[int],
    merge_gap_values: list[int],
) -> list[dict[str, Any]]:
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
    total_hours = _duration_hours(window_starts, window_ends)

    rows = []
    for method in methods:
        for adaptive_window in adaptive_windows:
            transformed_validation = _online_scores(
                validation_scores,
                seed_scores=validation_scores[:adaptive_window],
                method=method,
                adaptive_window=adaptive_window,
            )
            transformed_test = _online_scores(
                test_scores,
                seed_scores=validation_scores,
                method=method,
                adaptive_window=adaptive_window,
            )
            for percentile in percentiles:
                threshold = float(np.percentile(transformed_validation, percentile))
                raw_pred = (transformed_test > threshold).astype(np.int64)
                for min_consecutive in min_consecutive_values:
                    for merge_gap in merge_gap_values:
                        pred = postprocess_anomaly_predictions(
                            raw_pred,
                            min_consecutive_windows=min_consecutive,
                            merge_gap_windows=merge_gap,
                        )
                        point = classification_metrics(y_true, pred, ["normal", "attack"])
                        events = _fast_official_event_metrics(pred, event_masks, window_ends, total_hours)
                        rows.append(
                            _row(
                                run_dir=run_dir,
                                metrics=metrics,
                                method=method,
                                adaptive_window=adaptive_window,
                                percentile=percentile,
                                threshold=threshold,
                                point=point,
                                events=events,
                                min_consecutive=min_consecutive,
                                merge_gap=merge_gap,
                            )
                        )
    return rows


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
    run_dir: Path,
    metrics: dict[str, Any],
    method: str,
    adaptive_window: int,
    percentile: float,
    threshold: float,
    point: dict[str, Any],
    events: dict[str, Any],
    min_consecutive: int,
    merge_gap: int,
) -> dict[str, Any]:
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    return {
        "experiment": metrics.get("experiment", run_dir.name),
        "model_name": metrics.get("model_name", ""),
        "loss": metrics.get("loss", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "online_method": method,
        "adaptive_window": adaptive_window,
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
        "metrics_path": str(run_dir / "metrics.json"),
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
