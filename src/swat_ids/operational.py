from __future__ import annotations

from datetime import datetime, time, timedelta
import re
from pathlib import Path
from typing import Any
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def window_timeline(
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


def label_event_metrics(
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
            delays.append(max(0, first_hit - event["start"]) * stride)
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


def official_event_metrics(
    attack_list: dict[str, Any],
    y_pred: np.ndarray,
    timeline: list[dict[str, str | int | None]],
    stride: int,
) -> dict[str, Any]:
    pred_events = _segments(y_pred)
    pred_intervals = [_segment_interval(segment, timeline) for segment in pred_events]
    window_intervals = [_timeline_interval(row) for row in timeline]
    official_events = _events_overlapping_timeline(attack_list["events"], timeline)

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
        if interval is not None and not any(_time_overlaps(interval, event) for event in official_events):
            false_alarm_segments.append(segment)

    false_alarm_windows = 0
    for pred, interval in zip(y_pred, window_intervals):
        if pred == 1 and interval is not None:
            false_alarm_windows += int(not any(_time_overlaps(interval, event) for event in official_events))

    total_hours = _duration_hours(timeline)
    return {
        "event_source": "official_list_of_attacks",
        "true_event_count": len(official_events),
        "official_event_count_total": len(attack_list["events"]),
        "official_event_count_in_test_timeline": len(official_events),
        "skipped_official_rows": attack_list["skipped_rows"],
        "date_corrections": attack_list["date_corrections"],
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


def load_official_attack_list(path: Path) -> dict[str, Any]:
    rows = _read_xlsx_first_sheet(path)
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
                {"attack_id": attack_id, "reason": "missing start or end time", "row": row}
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


def _parse_timestamp(value: str) -> datetime:
    parsed = pd.to_datetime(value, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, dayfirst=True, errors="raise")
    return parsed.to_pydatetime()


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
    first = _timeline_interval(timeline[segment["start"]])
    last = _timeline_interval(timeline[segment["end"]])
    if first is None or last is None:
        return None
    return {"start": first["start"], "end": last["end"]}


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
    if not timeline or timeline[0]["start"] is None or timeline[-1]["end"] is None:
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


def _read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        sheet_path = _first_sheet_path(archive)
        shared_strings = _load_shared_strings(archive)
        with archive.open(sheet_path) as sheet_xml:
            return [values for _, values in _iter_rows(sheet_xml, shared_strings)]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("Workbook does not contain worksheets.")
    rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
    target = rel_map[rel_id].lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as shared_xml:
        for _, element in ET.iterparse(shared_xml, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}si":
                strings.append("".join(text.text or "" for text in element.iter(f"{{{MAIN_NS}}}t")))
                element.clear()
    return strings


def _iter_rows(sheet_xml, shared_strings: list[str]):
    for _, row in ET.iterparse(sheet_xml, events=("end",)):
        if row.tag != f"{{{MAIN_NS}}}row":
            continue
        row_index = int(row.attrib.get("r", "0") or 0)
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            column_index = _column_index(cell.attrib.get("r", ""))
            while len(values) < column_index:
                values.append("")
            values[column_index - 1] = _cell_value(cell, shared_strings)
        while values and values[-1] == "":
            values.pop()
        yield row_index, values
        row.clear()


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        return "" if inline is None else "".join(text.text or "" for text in inline.iter(f"{{{MAIN_NS}}}t")).strip()
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        index = int(value.text)
        return shared_strings[index].strip() if index < len(shared_strings) else ""
    if cell_type == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text.strip()


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return max(index, 1)


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
