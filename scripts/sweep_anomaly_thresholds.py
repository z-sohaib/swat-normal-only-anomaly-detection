from __future__ import annotations

import argparse
import csv
import json
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
        description="Sweep SWaT anomaly thresholds and temporal post-processing settings."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory containing metrics.json, config.toml, and window_scores.csv.",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[99.5, 99.7, 99.9, 99.95, 99.99],
    )
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--merge-gaps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument("--output", default="runs/anomaly_threshold_sweep.csv")
    args = parser.parse_args()

    rows = []
    for run_dir_text in args.run:
        run_dir = Path(run_dir_text)
        rows.extend(
            _sweep_run(
                run_dir=run_dir,
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
    percentiles: list[float],
    min_consecutive_values: list[int],
    merge_gap_values: list[int],
) -> list[dict[str, Any]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.toml")
    validation_path = run_dir / "validation_scores.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing {validation_path}. Run scripts/export_anomaly_validation_scores.py first."
        )
    _, validation_errors = _read_scores(validation_path)
    y_true, scores, window_starts, window_ends, origin = _read_test_scores(run_dir / "window_scores.csv")
    attack_list_path = config.data.attack_list_path or Path("data/raw/List_of_attacks_Final.xlsx")
    attack_list = load_official_attack_list(attack_list_path)
    event_masks = _event_masks(
        attack_list=attack_list,
        window_starts=window_starts,
        window_ends=window_ends,
        origin=origin,
    )

    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(validation_errors, percentile))
        raw_pred = (scores > threshold).astype(np.int64)
        for min_consecutive in min_consecutive_values:
            for merge_gap in merge_gap_values:
                pred = postprocess_anomaly_predictions(
                    raw_pred,
                    min_consecutive_windows=min_consecutive,
                    merge_gap_windows=merge_gap,
                )
                point = classification_metrics(y_true, pred, ["normal", "attack"])
                events = _fast_official_event_metrics(
                    y_pred=pred,
                    event_masks=event_masks,
                    window_ends=window_ends,
                    total_hours=_duration_hours(window_starts, window_ends),
                )
                rows.append(
                    _row(
                        run_dir=run_dir,
                        metrics=metrics,
                        percentile=percentile,
                        threshold=threshold,
                        min_consecutive=min_consecutive,
                        merge_gap=merge_gap,
                        point=point,
                        events=events,
                    )
                )
    return rows


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
            starts.append(_parse_iso(row["start_time"]))
            ends.append(_parse_iso(row["end_time"]))

    if not starts:
        raise ValueError(f"No window scores found in {path}")
    origin = starts[0]
    start_seconds = np.asarray([(value - origin).total_seconds() for value in starts], dtype=np.float64)
    end_seconds = np.asarray([(value - origin).total_seconds() for value in ends], dtype=np.float64)
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        start_seconds,
        end_seconds,
        origin,
    )


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _event_masks(
    attack_list: dict[str, Any],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    origin: datetime,
) -> list[dict[str, Any]]:
    if len(window_starts) == 0:
        return []
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
        masks.append(
            {
                "attack_id": event["attack_id"],
                "start_seconds": clipped_start,
                "end_seconds": clipped_end,
                "mask": mask,
            }
        )
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

    pred_segments = _segments(predicted)
    false_alarm_segments = 0
    for start, end in pred_segments:
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
    if len(window_starts) == 0:
        return 0.0
    return max(float(window_ends[-1] - window_starts[0]) / 3600.0, 0.0)


def _row(
    run_dir: Path,
    metrics: dict[str, Any],
    percentile: float,
    threshold: float,
    min_consecutive: int,
    merge_gap: int,
    point: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    return {
        "experiment": metrics.get("experiment", run_dir.name),
        "model_name": metrics.get("model_name", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
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
