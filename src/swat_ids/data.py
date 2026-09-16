from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from swat_ids.bundle import DatasetBundle
from swat_ids.config import DataConfig
from swat_ids.utils import require_file


CLASS_NAMES = ["normal", "attack"]


@dataclass(frozen=True)
class PreparedFrame:
    features: pd.DataFrame
    labels: np.ndarray
    timestamps: pd.Series | None
    source: str
    missing_values_before_fill: int
    non_numeric_values_before_fill: int


def load_swat(config: DataConfig, seed: int) -> DatasetBundle:
    """Load SWaT according to the configured Phase 2 protocol.

    Supported protocols:

    - ``standard_normal_attack``: train/validation windows are created from the
      normal file and test windows from the attack-period file. This matches the
      common SWaT normal-only anomaly-detection data split, but it is not a
      valid supervised classifier protocol when training labels contain one
      class only. Training code should use it only for future normal-only
      reconstruction/forecasting models.
    - ``supervised_chronological``: create train/validation/test ranges from a
      labelled continuous frame before windowing. This is useful for supervised
      classifier coding and future labelled experiments.
    """

    if config.protocol == "standard_normal_attack":
        return _load_standard_normal_attack(config)
    if config.protocol == "supervised_chronological":
        return _load_supervised_chronological(config)
    raise ValueError(
        "Unsupported SWaT protocol: "
        f"{config.protocol!r}. Expected 'standard_normal_attack' or "
        "'supervised_chronological'."
    )


def _load_standard_normal_attack(config: DataConfig) -> DatasetBundle:
    normal_path = require_file(config.normal_path, "SWaT normal CSV")
    attack_path = require_file(config.attack_path, "SWaT attack CSV")

    normal = _read_prepared_frame(normal_path, config, default_label=0)
    attack = _read_prepared_frame(attack_path, config, default_label=1)

    normal_train, normal_val = _chronological_train_val_split(normal, config.validation_size)
    scaler = _fit_scaler(normal_train.features, config)

    x_train, y_train = _transform_and_window(normal_train, scaler, config)
    x_val, y_val = _transform_and_window(normal_val, scaler, config)
    x_test, y_test = _transform_and_window(attack, scaler, config)

    metadata = _metadata(
        config=config,
        source_files=[normal_path, attack_path],
        train=normal_train,
        validation=normal_val,
        test=attack,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        warning=(
            "standard_normal_attack creates normal-only training labels. This is "
            "appropriate for a future normal-only anomaly-detection track, not "
            "for supervised cross-entropy classifier training."
        ),
    )
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
        metadata=metadata,
    )


def _load_supervised_chronological(config: DataConfig) -> DatasetBundle:
    if config.merged_path is not None:
        source_path = require_file(config.merged_path, "SWaT merged labelled CSV")
        frame = _read_prepared_frame(source_path, config, default_label=0)
        source_files = [source_path]
    else:
        normal_path = require_file(config.normal_path, "SWaT normal CSV")
        attack_path = require_file(config.attack_path, "SWaT attack CSV")
        normal = _read_prepared_frame(normal_path, config, default_label=0)
        attack = _read_prepared_frame(attack_path, config, default_label=1)
        frame = _concat_frames([normal, attack], source="normal_plus_attack")
        source_files = [normal_path, attack_path]

    train, validation, test = _chronological_train_val_test_split(
        frame,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )
    scaler = _fit_scaler(train.features, config)

    x_train, y_train = _transform_and_window(train, scaler, config)
    x_val, y_val = _transform_and_window(validation, scaler, config)
    x_test, y_test = _transform_and_window(test, scaler, config)

    metadata = _metadata(
        config=config,
        source_files=source_files,
        train=train,
        validation=validation,
        test=test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
        metadata=metadata,
    )


def _read_prepared_frame(path: Path, config: DataConfig, default_label: int) -> PreparedFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    label_col = config.label_column if config.label_column in frame.columns else None
    if label_col is None:
        labels = np.full(len(frame), default_label, dtype=np.int64)
    else:
        labels = _normalise_labels(frame[label_col])
        frame = frame.drop(columns=[label_col])

    timestamps: pd.Series | None = None
    timestamp_col = _resolve_timestamp_column(frame, config.timestamp_column)
    if timestamp_col is not None:
        timestamps = frame[timestamp_col].astype(str)
        if config.drop_timestamp:
            frame = frame.drop(columns=[timestamp_col])

    drop_columns = [column for column in config.drop_columns if column in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    missing_values_before_fill = int(numeric_frame.isna().sum().sum())
    non_numeric_values_before_fill = int(
        numeric_frame.isna().sum().sum() - frame.isna().sum().sum()
    )
    numeric_frame = numeric_frame.ffill().bfill().fillna(0.0)
    return PreparedFrame(
        features=numeric_frame,
        labels=labels,
        timestamps=timestamps,
        source=path.name,
        missing_values_before_fill=missing_values_before_fill,
        non_numeric_values_before_fill=max(non_numeric_values_before_fill, 0),
    )


def _resolve_timestamp_column(frame: pd.DataFrame, configured: str) -> str | None:
    if configured in frame.columns:
        return configured
    for candidate in frame.columns:
        if candidate.lower().strip() in {"timestamp", "time", "date"}:
            return candidate
    return None


def _normalise_labels(series: pd.Series) -> np.ndarray:
    text = series.astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)
    return text.isin({"attack", "a", "1", "true", "malicious", "abnormal"}).astype(np.int64).to_numpy()


def _concat_frames(frames: list[PreparedFrame], source: str) -> PreparedFrame:
    features = pd.concat([frame.features for frame in frames], axis=0, ignore_index=True)
    labels = np.concatenate([frame.labels for frame in frames])
    timestamps = None
    if all(frame.timestamps is not None for frame in frames):
        timestamps = pd.concat([frame.timestamps for frame in frames if frame.timestamps is not None], ignore_index=True)
    return PreparedFrame(
        features=features,
        labels=labels,
        timestamps=timestamps,
        source=source,
        missing_values_before_fill=sum(frame.missing_values_before_fill for frame in frames),
        non_numeric_values_before_fill=sum(frame.non_numeric_values_before_fill for frame in frames),
    )


def _chronological_train_val_split(frame: PreparedFrame, validation_size: float) -> tuple[PreparedFrame, PreparedFrame]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    split = int(len(frame.labels) * (1.0 - validation_size))
    return _slice_frame(frame, 0, split, "train"), _slice_frame(frame, split, len(frame.labels), "validation")


def _chronological_train_val_test_split(
    frame: PreparedFrame,
    validation_size: float,
    test_size: float,
) -> tuple[PreparedFrame, PreparedFrame, PreparedFrame]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be lower than 1.")

    total = len(frame.labels)
    train_end = int(total * (1.0 - validation_size - test_size))
    val_end = int(total * (1.0 - test_size))
    return (
        _slice_frame(frame, 0, train_end, "train"),
        _slice_frame(frame, train_end, val_end, "validation"),
        _slice_frame(frame, val_end, total, "test"),
    )


def _slice_frame(frame: PreparedFrame, start: int, end: int, source_suffix: str) -> PreparedFrame:
    timestamps = None
    if frame.timestamps is not None:
        timestamps = frame.timestamps.iloc[start:end].reset_index(drop=True)
    return PreparedFrame(
        features=frame.features.iloc[start:end].reset_index(drop=True),
        labels=frame.labels[start:end],
        timestamps=timestamps,
        source=f"{frame.source}:{source_suffix}",
        missing_values_before_fill=0,
        non_numeric_values_before_fill=0,
    )


def _fit_scaler(features: pd.DataFrame, config: DataConfig) -> StandardScaler | None:
    if config.scale_method == "none":
        return None
    if config.scale_method != "standard":
        raise ValueError("Unsupported scale_method. Expected 'standard' or 'none'.")
    scaler = StandardScaler()
    scaler.fit(features.to_numpy(dtype=np.float32))
    return scaler


def _transform_and_window(
    frame: PreparedFrame,
    scaler: StandardScaler | None,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.features.to_numpy(dtype=np.float32)
    if scaler is not None:
        values = scaler.transform(values).astype(np.float32)
    return _make_windows(
        values=values,
        labels=frame.labels,
        window_size=config.window_size,
        stride=config.window_stride,
        label_rule=config.window_label_rule,
    )


def _make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    label_rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("window_stride must be positive.")

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        window_labels.append(_window_label(labels[start:end], label_rule))
    if not windows:
        raise ValueError("No SWaT windows were created. Check window_size and dataset length.")
    return np.stack(windows).astype(np.float32), np.asarray(window_labels, dtype=np.int64)


def _window_label(labels: np.ndarray, label_rule: str) -> int:
    if label_rule == "any_attack":
        return int(labels.max())
    if label_rule == "last_point":
        return int(labels[-1])
    if label_rule == "majority":
        counts = np.bincount(labels.astype(np.int64), minlength=2)
        return int(np.argmax(counts))
    raise ValueError(
        f"Unsupported window_label_rule: {label_rule!r}. "
        "Expected 'any_attack', 'last_point', or 'majority'."
    )


def _metadata(
    config: DataConfig,
    source_files: list[Path],
    train: PreparedFrame,
    validation: PreparedFrame,
    test: PreparedFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    warning: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "protocol": config.protocol,
        "source_files": [str(path) for path in source_files],
        "window_size": config.window_size,
        "window_stride": config.window_stride,
        "window_label_rule": config.window_label_rule,
        "scale_method": config.scale_method,
        "configured_drop_columns": list(config.drop_columns),
        "feature_count": int(train.features.shape[1]),
        "feature_columns": list(train.features.columns),
        "splits": {
            "train": _split_metadata(train, y_train),
            "validation": _split_metadata(validation, y_val),
            "test": _split_metadata(test, y_test),
        },
    }
    if warning is not None:
        metadata["warning"] = warning
    return metadata


def _split_metadata(frame: PreparedFrame, window_labels: np.ndarray) -> dict[str, Any]:
    point_counts = np.bincount(frame.labels.astype(np.int64), minlength=2)
    window_counts = np.bincount(window_labels.astype(np.int64), minlength=2)
    return {
        "source": frame.source,
        "point_rows": int(len(frame.labels)),
        "point_label_counts": {
            "normal": int(point_counts[0]),
            "attack": int(point_counts[1]),
        },
        "window_count": int(len(window_labels)),
        "window_label_counts": {
            "normal": int(window_counts[0]),
            "attack": int(window_counts[1]),
        },
        "first_timestamp": _timestamp_at(frame.timestamps, 0),
        "last_timestamp": _timestamp_at(frame.timestamps, -1),
        "timestamp_monotonic": _timestamps_monotonic(frame.timestamps),
        "estimated_sampling_seconds": _estimated_sampling_seconds(frame.timestamps),
        "missing_values_before_fill": int(frame.missing_values_before_fill),
        "non_numeric_values_before_fill": int(frame.non_numeric_values_before_fill),
    }


def _timestamp_at(timestamps: pd.Series | None, index: int) -> str | None:
    if timestamps is None or timestamps.empty:
        return None
    return str(timestamps.iloc[index])


def _timestamps_monotonic(timestamps: pd.Series | None) -> bool | None:
    parsed = _parse_timestamps(timestamps)
    if parsed is None:
        return None
    return bool(parsed.is_monotonic_increasing)


def _estimated_sampling_seconds(timestamps: pd.Series | None) -> float | None:
    parsed = _parse_timestamps(timestamps)
    if parsed is None or len(parsed) < 2:
        return None
    deltas = parsed.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    return float(deltas.mode().iloc[0])


def _parse_timestamps(timestamps: pd.Series | None) -> pd.Series | None:
    if timestamps is None or timestamps.empty:
        return None
    parsed = pd.to_datetime(timestamps, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(timestamps, dayfirst=True, errors="coerce")
        if parsed.isna().any():
            return None
    return parsed
