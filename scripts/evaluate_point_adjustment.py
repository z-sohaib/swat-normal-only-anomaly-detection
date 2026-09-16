from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_ids.metrics import classification_metrics  # noqa: E402
from swat_ids.operational import load_official_attack_list  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit SWaT strict, point-adjusted, event, and alarm-budgeted metrics."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory containing metrics.json and window_scores.csv. Can be repeated.",
    )
    parser.add_argument(
        "--attack-list",
        default="data/raw/List_of_attacks_Final.xlsx",
        help="Official SWaT attack list workbook.",
    )
    parser.add_argument(
        "--alarm-budget-selected",
        default="runs/anomaly_alarm_budget_selected.csv",
        help="Optional Phase 6 selected alarm-budget table.",
    )
    parser.add_argument(
        "--output",
        default="runs/anomaly_metric_bias_audit.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    attack_list = load_official_attack_list(Path(args.attack_list))
    alarm_rows = _read_alarm_rows(Path(args.alarm_budget_selected))
    rows = []
    for run_text in args.run:
        rows.append(
            _audit_run(
                Path(run_text),
                attack_list=attack_list,
                alarm_rows=alarm_rows,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(f"Wrote {output}")
    return 0


def _audit_run(
    run_dir: Path,
    attack_list: dict[str, Any],
    alarm_rows: list[dict[str, str]],
) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    y_true, y_pred, starts, ends, origin = _read_window_scores(run_dir / "window_scores.csv")

    strict = classification_metrics(y_true, y_pred, ["normal", "attack"])
    adjusted_pred = _point_adjust(y_true, y_pred)
    adjusted = classification_metrics(y_true, adjusted_pred, ["normal", "attack"])
    strict_events = _official_event_metrics(
        attack_list=attack_list,
        y_pred=y_pred,
        starts=starts,
        ends=ends,
        origin=origin,
    )
    adjusted_events = _official_event_metrics(
        attack_list=attack_list,
        y_pred=adjusted_pred,
        starts=starts,
        ends=ends,
        origin=origin,
    )
    experiment = str(metrics.get("experiment", run_dir.name))
    best_alarm_event = _best_alarm_row(alarm_rows, experiment, sort_key="event_f1")
    best_alarm_attack = _best_alarm_row(alarm_rows, experiment, sort_key="attack_f1")

    payload = {
        "experiment": experiment,
        "model_name": metrics.get("model_name", ""),
        "run_dir": str(run_dir),
        "score_method": metrics.get("anomaly_score_method", ""),
        "threshold_percentile": metrics.get("anomaly_threshold_percentile", ""),
        "min_consecutive_windows": metrics.get("anomaly_min_consecutive_windows", ""),
        "merge_gap_windows": metrics.get("anomaly_merge_gap_windows", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "seconds_per_sample": metrics.get("latency", {}).get("seconds_per_sample", ""),
        "strict_accuracy": strict["accuracy"],
        "strict_macro_f1": strict["macro_f1"],
        "strict_weighted_f1": strict["weighted_f1"],
        "strict_attack_precision": strict["per_class"]["attack"]["precision"],
        "strict_attack_recall": strict["per_class"]["attack"]["recall"],
        "strict_attack_f1": strict["per_class"]["attack"]["f1"],
        "strict_attack_fpr": strict["per_class"]["attack"]["false_positive_rate"],
        "point_adjusted_accuracy": adjusted["accuracy"],
        "point_adjusted_macro_f1": adjusted["macro_f1"],
        "point_adjusted_weighted_f1": adjusted["weighted_f1"],
        "point_adjusted_attack_precision": adjusted["per_class"]["attack"]["precision"],
        "point_adjusted_attack_recall": adjusted["per_class"]["attack"]["recall"],
        "point_adjusted_attack_f1": adjusted["per_class"]["attack"]["f1"],
        "point_adjusted_attack_fpr": adjusted["per_class"]["attack"]["false_positive_rate"],
        "strict_to_adjusted_attack_f1_gain": (
            adjusted["per_class"]["attack"]["f1"] - strict["per_class"]["attack"]["f1"]
        ),
        "strict_to_adjusted_macro_f1_gain": adjusted["macro_f1"] - strict["macro_f1"],
        "official_event_recall": strict_events["event_recall"],
        "official_event_precision": strict_events["event_precision"],
        "official_event_f1": strict_events["event_f1"],
        "official_false_alarm_segments": strict_events["false_alarm_segment_count"],
        "official_false_alarm_windows": strict_events["false_alarm_windows"],
        "official_false_alarm_segments_per_hour": strict_events["false_alarm_segments_per_hour"],
        "official_mean_delay_seconds": strict_events["mean_detection_delay_seconds"],
        "point_adjusted_event_recall": adjusted_events["event_recall"],
        "point_adjusted_event_precision": adjusted_events["event_precision"],
        "point_adjusted_event_f1": adjusted_events["event_f1"],
        "best_alarm_budget_by_event_f1": _field(best_alarm_event, "target_alarm_budget_per_hour"),
        "best_alarm_event_f1": _field(best_alarm_event, "event_f1"),
        "best_alarm_event_recall": _field(best_alarm_event, "event_recall"),
        "best_alarm_event_precision": _field(best_alarm_event, "event_precision"),
        "best_alarm_false_alarm_segments": _field(best_alarm_event, "false_alarm_segment_count"),
        "best_alarm_false_alarm_windows": _field(best_alarm_event, "false_alarm_windows"),
        "best_alarm_attack_f1": _field(best_alarm_attack, "attack_f1"),
        "best_alarm_attack_f1_budget": _field(best_alarm_attack, "target_alarm_budget_per_hour"),
    }
    (run_dir / "metric_bias_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def _read_window_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime]:
    y_true = []
    y_pred = []
    starts = []
    ends = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
            starts.append(datetime.fromisoformat(row["start_time"]))
            ends.append(datetime.fromisoformat(row["end_time"]))
    if not starts:
        raise ValueError(f"No rows found in {path}")
    origin = starts[0]
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray([(value - origin).total_seconds() for value in starts], dtype=np.float64),
        np.asarray([(value - origin).total_seconds() for value in ends], dtype=np.float64),
        origin,
    )


def _point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    adjusted = y_pred.astype(np.int64).copy()
    for start, end in _segments(y_true):
        if bool((adjusted[start : end + 1] == 1).any()):
            adjusted[start : end + 1] = 1
    return adjusted


def _official_event_metrics(
    attack_list: dict[str, Any],
    y_pred: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    origin: datetime,
) -> dict[str, Any]:
    event_masks = _event_masks(attack_list, starts, ends, origin)
    official_any = np.zeros_like(y_pred, dtype=bool)
    detected = 0
    delays = []
    for event in event_masks:
        mask = event["mask"]
        official_any |= mask
        hit_indices = np.flatnonzero((y_pred == 1) & mask)
        if len(hit_indices) == 0:
            continue
        detected += 1
        first_hit = int(hit_indices[0])
        delays.append(max(float(ends[first_hit] - event["start_seconds"]), 0.0))

    pred_segments = _segments(y_pred)
    true_positive_segments = 0
    false_alarm_segments = 0
    for start, end in pred_segments:
        if bool(official_any[start : end + 1].any()):
            true_positive_segments += 1
        else:
            false_alarm_segments += 1

    event_precision = true_positive_segments / len(pred_segments) if pred_segments else 0.0
    event_recall = detected / len(event_masks) if event_masks else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    false_alarm_windows = int(((y_pred == 1) & (~official_any)).sum())
    total_hours = _duration_hours(starts, ends)
    return {
        "true_event_count": len(event_masks),
        "predicted_event_count": len(pred_segments),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "false_alarm_segment_count": false_alarm_segments,
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_segments_per_hour": false_alarm_segments / total_hours if total_hours else 0.0,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
    }


def _event_masks(
    attack_list: dict[str, Any],
    starts: np.ndarray,
    ends: np.ndarray,
    origin: datetime,
) -> list[dict[str, Any]]:
    masks = []
    timeline_start = float(starts[0])
    timeline_end = float(ends[-1])
    for event in attack_list["events"]:
        start = (event["start"] - origin).total_seconds()
        end = (event["end"] - origin).total_seconds()
        if start > timeline_end or end < timeline_start:
            continue
        clipped_start = max(start, timeline_start)
        clipped_end = min(end, timeline_end)
        masks.append(
            {
                "start_seconds": clipped_start,
                "end_seconds": clipped_end,
                "mask": (starts <= clipped_end) & (clipped_start <= ends),
            }
        )
    return masks


def _segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for idx, value in enumerate(values):
        if int(value) == 1 and start is None:
            start = idx
        elif int(value) != 1 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _duration_hours(starts: np.ndarray, ends: np.ndarray) -> float:
    return max(float(ends[-1] - starts[0]) / 3600.0, 0.0)


def _read_alarm_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_alarm_row(
    rows: list[dict[str, str]],
    experiment: str,
    sort_key: str,
) -> dict[str, str] | None:
    matching = [row for row in rows if row.get("experiment") == experiment]
    if not matching:
        return None
    return max(matching, key=lambda row: _float(row.get(sort_key)))


def _field(row: dict[str, str] | None, name: str) -> str:
    return "" if row is None else row.get(name, "")


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
