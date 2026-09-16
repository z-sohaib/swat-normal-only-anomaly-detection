from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    task: str
    validation_size: float
    apply_smote: bool
    protocol: str = "standard_normal_attack"
    train_path: Path | None = None
    test_path: Path | None = None
    normal_path: Path | None = None
    attack_path: Path | None = None
    attack_list_path: Path | None = None
    merged_path: Path | None = None
    test_size: float = 0.20
    window_size: int = 100
    window_stride: int = 50
    label_column: str = "Normal/Attack"
    timestamp_column: str = "Timestamp"
    window_label_rule: str = "any_attack"
    drop_timestamp: bool = True
    drop_columns: tuple[str, ...] = ()
    scale_method: str = "standard"
    processed_output_dir: Path = Path("data/processed")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    conv_channels: int
    kernel_sizes: tuple[int, ...]
    lstm_hidden: int
    lstm_layers: int
    attention_heads: int
    dense_hidden: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    loss: str
    early_stopping_patience: int
    selection_metric: str = "macro_f1"
    threshold_strategy: str = "argmax"
    threshold_min_precision: float = 0.95
    threshold_min_recall: float = 0.40
    threshold_grid_step: float = 0.01
    focal_gamma: float = 2.0
    anomaly_threshold_percentile: float = 99.5
    anomaly_min_consecutive_windows: int = 1
    anomaly_merge_gap_windows: int = 0
    anomaly_score_method: str = "global_mse"
    anomaly_calibration_source: str = "normal_validation"


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    experiment = ExperimentConfig(
        name=raw["experiment"]["name"],
        seed=int(raw["experiment"]["seed"]),
        output_dir=Path(raw["experiment"]["output_dir"]),
    )
    data_raw = raw["data"]
    data = DataConfig(
        dataset=data_raw["dataset"],
        task=data_raw["task"],
        validation_size=float(data_raw["validation_size"]),
        apply_smote=bool(data_raw.get("apply_smote", False)),
        protocol=data_raw.get("protocol", "standard_normal_attack"),
        train_path=_optional_path(data_raw.get("train_path")),
        test_path=_optional_path(data_raw.get("test_path")),
        normal_path=_optional_path(data_raw.get("normal_path")),
        attack_path=_optional_path(data_raw.get("attack_path")),
        attack_list_path=_optional_path(data_raw.get("attack_list_path")),
        merged_path=_optional_path(data_raw.get("merged_path")),
        test_size=float(data_raw.get("test_size", 0.20)),
        window_size=int(data_raw.get("window_size", 100)),
        window_stride=int(data_raw.get("window_stride", 50)),
        label_column=data_raw.get("label_column", "Normal/Attack"),
        timestamp_column=data_raw.get("timestamp_column", "Timestamp"),
        window_label_rule=data_raw.get("window_label_rule", "any_attack"),
        drop_timestamp=bool(data_raw.get("drop_timestamp", True)),
        drop_columns=tuple(str(column).strip() for column in data_raw.get("drop_columns", [])),
        scale_method=data_raw.get("scale_method", "standard"),
        processed_output_dir=Path(data_raw.get("processed_output_dir", "data/processed")),
    )
    model_raw = raw["model"]
    model = ModelConfig(
        name=model_raw.get("name", "ms_cnn_bilstm_attention"),
        conv_channels=int(model_raw["conv_channels"]),
        kernel_sizes=tuple(int(k) for k in model_raw["kernel_sizes"]),
        lstm_hidden=int(model_raw["lstm_hidden"]),
        lstm_layers=int(model_raw["lstm_layers"]),
        attention_heads=int(model_raw["attention_heads"]),
        dense_hidden=int(model_raw["dense_hidden"]),
        dropout=float(model_raw["dropout"]),
    )
    training_raw = raw["training"]
    training = TrainingConfig(
        epochs=int(training_raw["epochs"]),
        batch_size=int(training_raw["batch_size"]),
        learning_rate=float(training_raw["learning_rate"]),
        weight_decay=float(training_raw["weight_decay"]),
        loss=training_raw["loss"],
        early_stopping_patience=int(training_raw["early_stopping_patience"]),
        selection_metric=training_raw.get("selection_metric", "macro_f1"),
        threshold_strategy=training_raw.get("threshold_strategy", "argmax"),
        threshold_min_precision=float(training_raw.get("threshold_min_precision", 0.95)),
        threshold_min_recall=float(training_raw.get("threshold_min_recall", 0.40)),
        threshold_grid_step=float(training_raw.get("threshold_grid_step", 0.01)),
        focal_gamma=float(training_raw.get("focal_gamma", 2.0)),
        anomaly_threshold_percentile=float(training_raw.get("anomaly_threshold_percentile", 99.5)),
        anomaly_min_consecutive_windows=int(training_raw.get("anomaly_min_consecutive_windows", 1)),
        anomaly_merge_gap_windows=int(training_raw.get("anomaly_merge_gap_windows", 0)),
        anomaly_score_method=training_raw.get("anomaly_score_method", "global_mse"),
        anomaly_calibration_source=training_raw.get("anomaly_calibration_source", "normal_validation"),
    )
    return Config(experiment=experiment, data=data, model=model, training=training)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None
