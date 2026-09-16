from __future__ import annotations

import numpy as np


def postprocess_anomaly_predictions(
    predictions: np.ndarray,
    min_consecutive_windows: int = 1,
    merge_gap_windows: int = 0,
) -> np.ndarray:
    if min_consecutive_windows < 1:
        raise ValueError("min_consecutive_windows must be at least 1.")
    if merge_gap_windows < 0:
        raise ValueError("merge_gap_windows must be non-negative.")

    values = predictions.astype(np.int64).copy()
    if values.size == 0:
        return values

    segments = _segments(values)
    if merge_gap_windows > 0:
        segments = _merge_close_segments(segments, merge_gap_windows)

    filtered = np.zeros_like(values)
    for start, end in segments:
        if end - start + 1 >= min_consecutive_windows:
            filtered[start : end + 1] = 1
    return filtered


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


def _merge_close_segments(
    segments: list[tuple[int, int]],
    merge_gap_windows: int,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= merge_gap_windows:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
