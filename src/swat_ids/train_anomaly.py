from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from swat_ids.config import Config, load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_anomaly_model
from swat_ids.operational import (
    label_event_metrics,
    load_official_attack_list,
    official_event_metrics,
    window_timeline,
)
from swat_ids.postprocessing import postprocess_anomaly_predictions
from swat_ids.utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SWaT normal-only anomaly detector.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    run(config, config_path=config_path)


def run(config: Config, config_path: Path | None = None) -> None:
    if config.data.protocol != "standard_normal_attack":
        raise ValueError(
            "Normal-only anomaly detection must use protocol='standard_normal_attack' "
            "so training/validation are normal-only and testing uses the attack period."
        )

    set_seed(config.experiment.seed)
    output_dir = ensure_dir(config.experiment.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_swat(config.data, seed=config.experiment.seed)
    if np.any(bundle.y_train != 0) or np.any(bundle.y_val != 0):
        raise ValueError("Normal-only anomaly detection requires normal-only train and validation windows.")

    model = build_anomaly_model(config.model, input_features=bundle.input_features).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    train_loader = _loader(bundle.x_train, bundle.y_train, config.training.batch_size, shuffle=True)
    val_loader = _loader(bundle.x_val, bundle.y_val, config.training.batch_size, shuffle=False)
    test_loader = _loader(bundle.x_test, bundle.y_test, config.training.batch_size, shuffle=False)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            anomaly_loss=config.training.loss,
        )
        val_loss = _reconstruction_loss(
            model,
            val_loader,
            device,
            anomaly_loss=config.training.loss,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_reconstruction_loss": float(train_loss),
                "val_reconstruction_loss": float(val_loss),
            }
        )
        print(
            f"epoch={epoch:03d} train_recon_loss={train_loss:.6f} "
            f"val_recon_loss={val_loss:.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.training.anomaly_calibration_source != "normal_validation":
        raise ValueError(
            "swat-train-anomaly currently supports anomaly_calibration_source="
            "'normal_validation'. Use scripts/sweep_anomaly_score_methods.py for "
            "diagnostic burn-in calibration sweeps."
        )

    y_val, val_feature_errors = _feature_errors(model, val_loader, device, anomaly_loss=config.training.loss)
    y_test, test_feature_errors = _feature_errors(model, test_loader, device, anomaly_loss=config.training.loss)
    score_context = _score_context(val_feature_errors)
    val_scores = _score_from_feature_errors(
        val_feature_errors,
        method=config.training.anomaly_score_method,
        context=score_context,
    )
    test_scores = _score_from_feature_errors(
        test_feature_errors,
        method=config.training.anomaly_score_method,
        context=score_context,
    )
    threshold = float(np.percentile(val_scores, config.training.anomaly_threshold_percentile))
    pred_test_raw = (test_scores > threshold).astype(np.int64)
    pred_test = postprocess_anomaly_predictions(
        pred_test_raw,
        min_consecutive_windows=config.training.anomaly_min_consecutive_windows,
        merge_gap_windows=config.training.anomaly_merge_gap_windows,
    )
    metrics = classification_metrics(y_test, pred_test, bundle.class_names)

    split_meta = bundle.metadata["splits"]["test"]
    timeline = window_timeline(
        first_timestamp=split_meta["first_timestamp"],
        window_count=len(y_test),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )
    label_events = label_event_metrics(y_test, pred_test, timeline, stride=config.data.window_stride)
    official_events = None
    attack_list_path = config.data.attack_list_path or Path("data/raw/List_of_attacks_Final.xlsx")
    if attack_list_path.exists():
        official_events = official_event_metrics(
            attack_list=load_official_attack_list(attack_list_path),
            y_pred=pred_test,
            timeline=timeline,
            stride=config.data.window_stride,
        )

    metrics["history"] = history
    metrics["experiment"] = config.experiment.name
    metrics["model_name"] = config.model.name
    metrics["data_protocol"] = config.data.protocol
    metrics["window_size"] = config.data.window_size
    metrics["window_stride"] = config.data.window_stride
    metrics["window_label_rule"] = config.data.window_label_rule
    metrics["loss"] = config.training.loss
    metrics["selection_metric"] = "val_reconstruction_loss"
    metrics["best_selected_score"] = float(best_loss)
    metrics["validation_test_gap"] = None
    metrics["threshold_strategy"] = "normal_validation_percentile"
    metrics["anomaly_threshold_percentile"] = config.training.anomaly_threshold_percentile
    metrics["anomaly_min_consecutive_windows"] = config.training.anomaly_min_consecutive_windows
    metrics["anomaly_merge_gap_windows"] = config.training.anomaly_merge_gap_windows
    metrics["anomaly_score_method"] = config.training.anomaly_score_method
    metrics["anomaly_calibration_source"] = config.training.anomaly_calibration_source
    metrics["attack_threshold"] = threshold
    metrics["validation_error_stats"] = _error_stats(y_val, val_scores)
    metrics["test_error_stats"] = _error_stats(y_test, test_scores)
    metrics["label_derived_operational_metrics"] = label_events
    metrics["operational_metrics"] = official_events or label_events
    metrics["official_attack_list_used"] = str(attack_list_path) if official_events is not None else None
    metrics["device"] = str(device)
    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["latency"] = _measure_latency(model, test_loader, device)
    metrics["data_metadata"] = bundle.metadata

    write_json(output_dir / "metrics.json", metrics)
    _write_window_scores(
        output_dir / "window_scores.csv",
        y_test,
        pred_test,
        pred_test_raw,
        test_scores,
        threshold,
        timeline,
    )
    if config_path is not None:
        shutil.copyfile(config_path, output_dir / "config.toml")
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    anomaly_loss: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x_batch, _ in tqdm(loader, desc="train", leave=False):
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(model_input)
        loss = loss_fn(reconstruction, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x_batch)
        total_items += len(x_batch)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _reconstruction_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    anomaly_loss: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        errors = torch.mean((model(model_input) - target) ** 2, dim=(1, 2))
        total_loss += float(errors.sum().detach().cpu())
        total_items += len(x_batch)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _feature_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    anomaly_loss: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        errors = torch.mean((model(model_input) - target) ** 2, dim=1)
        scores.append(errors.cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(scores)


def _score_context(validation_feature_errors: np.ndarray) -> dict[str, np.ndarray]:
    median = np.median(validation_feature_errors, axis=0)
    q25 = np.percentile(validation_feature_errors, 25, axis=0)
    q75 = np.percentile(validation_feature_errors, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return {"median": median, "iqr": iqr}


def _score_from_feature_errors(
    feature_errors: np.ndarray,
    method: str,
    context: dict[str, np.ndarray],
) -> np.ndarray:
    if method == "global_mse":
        return feature_errors.mean(axis=1)
    if method == "max_feature":
        return feature_errors.max(axis=1)
    if method.startswith("top") and method.endswith("_feature_mean"):
        return _topk_mean(feature_errors, _topk_from_method(method))
    if method == "max_robust_z":
        return _robust_z(feature_errors, context).max(axis=1)
    if method.startswith("top") and method.endswith("_robust_z"):
        return _topk_mean(_robust_z(feature_errors, context), _topk_from_method(method))
    raise ValueError(f"Unsupported anomaly_score_method: {method!r}")


def _robust_z(feature_errors: np.ndarray, context: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum((feature_errors - context["median"]) / context["iqr"], 0.0)


def _topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(max(k, 1), values.shape[1])
    partition = np.partition(values, values.shape[1] - k, axis=1)
    return partition[:, -k:].mean(axis=1)


def _topk_from_method(method: str) -> int:
    digits = "".join(char for char in method.split("_", 1)[0] if char.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse top-k value from anomaly_score_method: {method!r}")
    return int(digits)


def _anomaly_input_target(
    x_batch: torch.Tensor,
    anomaly_loss: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if anomaly_loss == "reconstruction_mse":
        return x_batch, x_batch
    if anomaly_loss == "forecast_mse":
        if x_batch.shape[1] < 2:
            raise ValueError("forecast_mse requires window_size >= 2.")
        return x_batch[:, :-1, :], x_batch[:, 1:, :]
    raise ValueError(
        f"Unsupported anomaly loss: {anomaly_loss!r}. Expected "
        "'reconstruction_mse' or 'forecast_mse'."
    )


@torch.no_grad()
def _measure_latency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    sample_count = 0
    start = time.perf_counter()
    for idx, (x_batch, _) in enumerate(loader):
        if idx >= max_batches:
            break
        _ = model(x_batch.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        sample_count += len(x_batch)
    elapsed = time.perf_counter() - start
    return {
        "measured_samples": float(sample_count),
        "total_seconds": float(elapsed),
        "seconds_per_sample": float(elapsed / max(sample_count, 1)),
        "samples_per_second": float(sample_count / elapsed) if elapsed > 0 else 0.0,
    }


def _error_stats(y_true: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "all": _stats(errors),
        "normal": _stats(errors[y_true == 0]),
        "attack": _stats(errors[y_true == 1]),
    }
    return payload


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _write_window_scores(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_raw: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    timeline: list[dict[str, str | int | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index",
        "start_time",
        "end_time",
        "y_true",
        "y_pred",
        "y_pred_raw",
        "anomaly_score",
        "threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (truth, pred, raw_pred, score) in enumerate(zip(y_true, y_pred, y_pred_raw, scores)):
            writer.writerow(
                {
                    "window_index": idx,
                    "start_time": timeline[idx]["start"],
                    "end_time": timeline[idx]["end"],
                    "y_true": int(truth),
                    "y_pred": int(pred),
                    "y_pred_raw": int(raw_pred),
                    "anomaly_score": float(score),
                    "threshold": threshold,
                }
            )


if __name__ == "__main__":
    main()
