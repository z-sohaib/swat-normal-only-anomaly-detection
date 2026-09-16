from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, time, timedelta
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profile_attack_list import read_xlsx_first_sheet
from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SWaT trained runs with event-level and operational metrics."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification as RUN_DIR=CONFIG_PATH. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default="runs/operational_comparison.csv",
        help="CSV path for the operational comparison table.",
    )
    parser.add_argument(
        "--attack-list",
        default=None,
        help="Optional official List_of_attacks_Final.xlsx file for event metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    official_attack_list = (
        _load_official_attack_list(Path(args.attack_list)) if args.attack_list else None
    )
    rows = []
    for spec in args.run:
        run_dir, config_path = _parse_run_spec(spec)
        rows.append(
            _evaluate_run(
                run_dir,
                config_path,
                batch_size=args.batch_size,
                official_attack_list=official_attack_list,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(f"Wrote {output}")
    return 0


def _parse_run_spec(spec: str) -> tuple[Path, Path]:
    if "=" not in spec:
        raise ValueError("Each --run value must use RUN_DIR=CONFIG_PATH format.")
    run_dir, config_path = spec.split("=", 1)
    return Path(run_dir), Path(config_path)


def _evaluate_run(
    run_dir: Path,
    config_path: Path,
    batch_size: int,
    official_attack_list: dict[str, Any] | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    bundle = load_swat(config.data, seed=config.experiment.seed)

    model_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, bundle.input_features, bundle.num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    loader = _loader(bundle.x_test, bundle.y_test, batch_size=batch_size)
    y_true, probabilities = _predict_probs(model, loader, device)
    threshold = stored_metrics.get("attack_threshold")
    if threshold is None:
        y_pred = probabilities.argmax(axis=1)
    else:
        y_pred = (probabilities[:, 1] >= float(threshold)).astype(np.int64)

    point_metrics = classification_metrics(y_true, y_pred, bundle.class_names)
    split_meta = bundle.metadata["splits"]["test"]
    timeline = _window_timeline(
        first_timestamp=split_meta["first_timestamp"],
        window_count=len(y_true),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )
    label_event_metrics = _event_metrics(y_true, y_pred, timeline, stride=config.data.window_stride)
    official_event_metrics = None
    if official_attack_list is not None:
        official_event_metrics = _official_event_metrics(
            official_attack_list=official_attack_list,
            y_pred=y_pred,
            timeline=timeline,
            stride=config.data.window_stride,
        )

    payload = {
        "experiment": stored_metrics.get("experiment", run_dir.name),
        "model_name": stored_metrics.get("model_name", config.model.name),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "window_size": config.data.window_size,
        "window_stride": config.data.window_stride,
        "threshold_strategy": stored_metrics.get("threshold_strategy", "argmax"),
        "attack_threshold": threshold,
        "point_metrics": point_metrics,
        "operational_metrics": official_event_metrics or label_event_metrics,
        "label_derived_operational_metrics": label_event_metrics,
        "official_attack_list": _official_attack_list_summary(official_attack_list),
    }
    (run_dir / "operational_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {run_dir / 'operational_metrics.json'}")
    return _row(payload)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_probs = []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        y_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_probs)


def _window_timeline(
    first_timestamp: str | None,
    window_count: int,
    window_size: int,
    stride: int,
) -> list[dict[str, str | int | None]]:
    start = _parse_timestamp(first_timestamp) if first_timestamp else None
    timeline = []
    for idx in range(window_count):
        if start is None:
            window_start = None
            window_end = None
        else:
            window_start_dt = start + timedelta(seconds=idx * stride)
            window_end_dt = window_start_dt + timedelta(seconds=window_size - 1)
            window_start = window_start_dt.isoformat(sep=" ")
            window_end = window_end_dt.isoformat(sep=" ")
        timeline.append({"index": idx, "start": window_start, "end": window_end})
    return timeline


def _parse_timestamp(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=True, errors="raise")
    return parsed.to_pydatetime()


def _event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timeline: list[dict[str, str | int | None]],
    stride: int,
) -> dict[str, Any]:
    true_events = _segments(y_true)
    pred_events = _segments(y_pred)

    detected = []
    missed = []
    delays = []
    for event in true_events:
        overlapping = [segment for segment in pred_events if _overlaps(event, segment)]
        if overlapping:
            first_hit = min(max(segment["start"], event["start"]) for segment in overlapping)
            delay_windows = max(0, first_hit - event["start"])
            delays.append(delay_windows * stride)
            detected.append(_segment_payload(event, timeline, first_hit=first_hit))
        else:
            missed.append(_segment_payload(event, timeline))

    false_alarm_segments = [
        segment for segment in pred_events if not any(_overlaps(segment, event) for event in true_events)
    ]
    false_alarm_windows = int(((y_pred == 1) & (y_true == 0)).sum())
    normal_windows = int((y_true == 0).sum())
    total_hours = _duration_hours(timeline)

    return {
        "event_source": "label_derived_contiguous_windows",
        "true_event_count": len(true_events),
        "predicted_event_count": len(pred_events),
        "detected_event_count": len(detected),
        "missed_event_count": len(missed),
        "event_recall": len(detected) / len(true_events) if true_events else 0.0,
        "false_alarm_segment_count": len(false_alarm_segments),
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_window_rate": false_alarm_windows / normal_windows if normal_windows else 0.0,
        "false_alarm_segments_per_hour": (
            len(false_alarm_segments) / total_hours if total_hours > 0 else 0.0
        ),
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
        "detected_events": detected,
        "missed_events": missed,
        "false_alarm_segments": [_segment_payload(segment, timeline) for segment in false_alarm_segments],
    }


def _official_event_metrics(
    official_attack_list: dict[str, Any],
    y_pred: np.ndarray,
    timeline: list[dict[str, str | int | None]],
    stride: int,
) -> dict[str, Any]:
    pred_events = _segments(y_pred)
    pred_intervals = [_segment_interval(segment, timeline) for segment in pred_events]
    window_intervals = [_timeline_interval(row) for row in timeline]
    official_events = _events_overlapping_timeline(official_attack_list["events"], timeline)

    detected = []
    missed = []
    delays = []
    for event in official_events:
        window_hits = [
            (idx, interval)
            for idx, (pred, interval) in enumerate(zip(y_pred, window_intervals))
            if pred == 1 and interval is not None and _time_overlaps(event, interval)
        ]
        if window_hits:
            first_idx, first_interval = min(window_hits, key=lambda item: item[1]["end"])
            first_hit = max(first_interval["end"], event["start"])
            delay = max((first_hit - event["start"]).total_seconds(), 0.0)
            delays.append(delay)
            detected.append(_official_event_payload(event, first_hit=first_hit, pred_window_index=first_idx))
        else:
            missed.append(_official_event_payload(event))

    false_alarm_segments = []
    for segment, interval in zip(pred_events, pred_intervals):
        if interval is None:
            continue
        if not any(_time_overlaps(interval, event) for event in official_events):
            false_alarm_segments.append(segment)

    false_alarm_windows = 0
    for pred, interval in zip(y_pred, window_intervals):
        if pred != 1 or interval is None:
            continue
        if not any(_time_overlaps(interval, event) for event in official_events):
            false_alarm_windows += 1

    total_hours = _duration_hours(timeline)
    return {
        "event_source": "official_list_of_attacks",
        "true_event_count": len(official_events),
        "official_event_count_total": len(official_attack_list["events"]),
        "official_event_count_in_test_timeline": len(official_events),
        "skipped_official_rows": official_attack_list["skipped_rows"],
        "date_corrections": official_attack_list["date_corrections"],
        "predicted_event_count": len(pred_events),
        "detected_event_count": len(detected),
        "missed_event_count": len(missed),
        "event_recall": len(detected) / len(official_events) if official_events else 0.0,
        "false_alarm_segment_count": len(false_alarm_segments),
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_segments_per_hour": (
            len(false_alarm_segments) / total_hours if total_hours > 0 else 0.0
        ),
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "median_detection_delay_seconds": float(np.median(delays)) if delays else None,
        "detected_events": detected,
        "missed_events": missed,
        "false_alarm_segments": [_segment_payload(segment, timeline) for segment in false_alarm_segments],
    }


def _segments(labels: np.ndarray) -> list[dict[str, int]]:
    segments = []
    start = None
    for idx, value in enumerate(labels):
        if value == 1 and start is None:
            start = idx
        elif value != 1 and start is not None:
            segments.append({"start": start, "end": idx - 1, "length": idx - start})
            start = None
    if start is not None:
        segments.append({"start": start, "end": len(labels) - 1, "length": len(labels) - start})
    return segments


def _overlaps(left: dict[str, int], right: dict[str, int]) -> bool:
    return left["start"] <= right["end"] and right["start"] <= left["end"]


def _time_overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start"] <= right["end"] and right["start"] <= left["end"]


def _segment_payload(
    segment: dict[str, int],
    timeline: list[dict[str, str | int | None]],
    first_hit: int | None = None,
) -> dict[str, Any]:
    payload = {
        "start_index": segment["start"],
        "end_index": segment["end"],
        "length_windows": segment["length"],
        "start_time": timeline[segment["start"]]["start"],
        "end_time": timeline[segment["end"]]["end"],
    }
    if first_hit is not None:
        payload["first_hit_index"] = first_hit
        payload["first_hit_time"] = timeline[first_hit]["start"]
    return payload


def _segment_interval(
    segment: dict[str, int],
    timeline: list[dict[str, str | int | None]],
) -> dict[str, datetime] | None:
    start_end = _timeline_interval(timeline[segment["start"]])
    final_end = _timeline_interval(timeline[segment["end"]])
    if start_end is None or final_end is None:
        return None
    return {"start": start_end["start"], "end": final_end["end"]}


def _timeline_interval(row: dict[str, str | int | None]) -> dict[str, datetime] | None:
    if row["start"] is None or row["end"] is None:
        return None
    return {
        "start": pd.to_datetime(row["start"]).to_pydatetime(),
        "end": pd.to_datetime(row["end"]).to_pydatetime(),
    }


def _official_event_payload(
    event: dict[str, Any],
    first_hit: datetime | None = None,
    pred_segment: dict[str, int] | None = None,
    pred_window_index: int | None = None,
) -> dict[str, Any]:
    payload = {
        "attack_id": event["attack_id"],
        "attack_point": event["attack_point"],
        "start_time": event["start"].isoformat(sep=" "),
        "end_time": event["end"].isoformat(sep=" "),
        "duration_seconds": event["duration_seconds"],
    }
    if "original_start" in event:
        payload["original_start_time"] = event["original_start"].isoformat(sep=" ")
        payload["original_end_time"] = event["original_end"].isoformat(sep=" ")
    if first_hit is not None:
        payload["first_hit_time"] = first_hit.isoformat(sep=" ")
        payload["delay_seconds"] = max((first_hit - event["start"]).total_seconds(), 0.0)
    if pred_segment is not None:
        payload["predicted_segment_start_index"] = pred_segment["start"]
        payload["predicted_segment_end_index"] = pred_segment["end"]
    if pred_window_index is not None:
        payload["first_hit_window_index"] = pred_window_index
    return payload


def _duration_hours(timeline: list[dict[str, str | int | None]]) -> float:
    if not timeline or timeline[0]["start"] is None or timeline[-1]["end"] is None:
        return 0.0
    start = pd.to_datetime(timeline[0]["start"])
    end = pd.to_datetime(timeline[-1]["end"])
    return max((end - start).total_seconds() / 3600.0, 0.0)


def _events_overlapping_timeline(
    events: list[dict[str, Any]],
    timeline: list[dict[str, str | int | None]],
) -> list[dict[str, Any]]:
    if not timeline:
        return []
    start = pd.to_datetime(timeline[0]["start"]).to_pydatetime()
    end = pd.to_datetime(timeline[-1]["end"]).to_pydatetime()
    test_interval = {"start": start, "end": end}
    clipped = []
    for event in events:
        if not _time_overlaps(event, test_interval):
            continue
        clipped_event = dict(event)
        clipped_event["original_start"] = event["start"]
        clipped_event["original_end"] = event["end"]
        clipped_event["start"] = max(event["start"], start)
        clipped_event["end"] = min(event["end"], end)
        clipped_event["duration_seconds"] = (
            clipped_event["end"] - clipped_event["start"]
        ).total_seconds()
        clipped.append(clipped_event)
    return clipped


def _load_official_attack_list(path: Path) -> dict[str, Any]:
    rows = read_xlsx_first_sheet(path)
    if not rows:
        raise ValueError(f"Attack-list workbook is empty: {path}")
    header = [cell.strip() for cell in rows[0]]
    index = {name: idx for idx, name in enumerate(header)}
    required = ["Attack #", "Start Time", "End Time", "Attack Point"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Attack-list workbook missing columns: {missing}")

    events = []
    skipped_rows = []
    date_corrections = []
    for row in rows[1:]:
        attack_id_text = _cell(row, index["Attack #"])
        if not attack_id_text.isdigit():
            continue
        attack_id = int(attack_id_text)
        start_text = _cell(row, index["Start Time"])
        end_text = _cell(row, index["End Time"])
        if not start_text or not end_text:
            skipped_rows.append(
                {
                    "attack_id": attack_id,
                    "reason": "missing start or end time",
                    "row": row,
                }
            )
            continue
        start, start_corrected = _excel_datetime(start_text)
        end, end_corrected = _excel_end_datetime(end_text, start)
        if end < start:
            end = end + timedelta(days=1)
        if start_corrected or end_corrected:
            date_corrections.append(
                {
                    "attack_id": attack_id,
                    "start_corrected": start_corrected,
                    "end_corrected": end_corrected,
                }
            )
        events.append(
            {
                "attack_id": attack_id,
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
                "attack_point": _cell(row, index["Attack Point"]),
            }
        )

    return {
        "path": str(path),
        "events": events,
        "event_count": len(events),
        "skipped_rows": skipped_rows,
        "date_corrections": date_corrections,
    }


def _official_attack_list_summary(attack_list: dict[str, Any] | None) -> dict[str, Any] | None:
    if attack_list is None:
        return None
    return {
        "path": attack_list["path"],
        "event_count": attack_list["event_count"],
        "skipped_rows": attack_list["skipped_rows"],
        "date_corrections": attack_list["date_corrections"],
        "events": [
            {
                "attack_id": event["attack_id"],
                "attack_point": event["attack_point"],
                "start_time": event["start"].isoformat(sep=" "),
                "end_time": event["end"].isoformat(sep=" "),
                "duration_seconds": event["duration_seconds"],
            }
            for event in attack_list["events"]
        ],
    }


def _cell(row: list[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) else ""


def _excel_datetime(value: str) -> tuple[datetime, bool]:
    serial = float(value)
    corrected = False
    if serial < 42366:
        serial += 365.0
        corrected = True
    return datetime(1899, 12, 30) + timedelta(days=serial), corrected


def _excel_end_datetime(value: str, start: datetime) -> tuple[datetime, bool]:
    serial = float(value)
    if 0.0 <= serial < 1.0:
        seconds = round(serial * 24 * 60 * 60)
        return datetime.combine(start.date(), time()) + timedelta(seconds=seconds), False
    return _excel_datetime(value)


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    point = payload["point_metrics"]
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    operational = payload["operational_metrics"]
    return {
        "experiment": payload["experiment"],
        "model_name": payload["model_name"],
        "window_size": payload["window_size"],
        "window_stride": payload["window_stride"],
        "threshold_strategy": payload["threshold_strategy"],
        "attack_threshold": payload["attack_threshold"],
        "accuracy": point["accuracy"],
        "macro_f1": point["macro_f1"],
        "attack_precision": attack["precision"],
        "attack_recall": attack["recall"],
        "attack_f1": attack["f1"],
        "attack_fpr": attack["false_positive_rate"],
        "normal_recall": normal["recall"],
        "true_event_count": operational["true_event_count"],
        "detected_event_count": operational["detected_event_count"],
        "event_recall": operational["event_recall"],
        "missed_event_count": operational["missed_event_count"],
        "false_alarm_segment_count": operational["false_alarm_segment_count"],
        "false_alarm_windows": operational["false_alarm_windows"],
        "false_alarm_segments_per_hour": operational["false_alarm_segments_per_hour"],
        "mean_detection_delay_seconds": operational["mean_detection_delay_seconds"],
        "run_dir": payload["run_dir"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
