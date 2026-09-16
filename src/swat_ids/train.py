from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from swat_ids.config import Config, load_config
from swat_ids.data import load_swat
from swat_ids.losses import build_loss
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_model
from swat_ids.utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis IDS model.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    run(config, config_path=config_path)


def run(config: Config, config_path: Path | None = None) -> None:
    set_seed(config.experiment.seed)
    output_dir = ensure_dir(config.experiment.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_swat(config.data, seed=config.experiment.seed)
    if len(np.unique(bundle.y_train)) < bundle.num_classes:
        raise ValueError(
            "Supervised classifier training requires both normal and attack "
            "windows in the training split. Current training window labels are "
            f"{np.unique(bundle.y_train).tolist()}. Use a supervised labelled "
            "split or implement the normal-only forecasting/reconstruction "
            "track before training on normal-only data."
        )
    model = build_model(
        config.model,
        input_features=bundle.input_features,
        num_classes=bundle.num_classes,
    ).to(device)

    loss_fn = build_loss(
        config.training.loss,
        bundle.y_train,
        bundle.num_classes,
        focal_gamma=config.training.focal_gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    train_loader = _loader(bundle.x_train, bundle.y_train, config.training.batch_size, shuffle=True)
    val_loader = _loader(bundle.x_val, bundle.y_val, config.training.batch_size, shuffle=False)
    test_loader = _loader(bundle.x_test, bundle.y_test, config.training.batch_size, shuffle=False)

    best_score = -1.0
    best_state = None
    best_val_metrics = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        y_val, pred_val = _predict(model, val_loader, device)
        val_metrics = classification_metrics(y_val, pred_val, bundle.class_names)
        selected_score = _selection_score(val_metrics, config.training.selection_metric)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "selected_score": float(selected_score),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        if selected_score > best_score:
            best_score = selected_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_metrics = val_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val, val_probs = _predict_probs(model, val_loader, device)
    y_test, test_probs = _predict_probs(model, test_loader, device)
    threshold_info = _calibrate_threshold(
        y_true=y_val,
        probabilities=val_probs,
        class_names=bundle.class_names,
        strategy=config.training.threshold_strategy,
        min_precision=config.training.threshold_min_precision,
        min_recall=config.training.threshold_min_recall,
        grid_step=config.training.threshold_grid_step,
    )
    pred_test = _probability_predictions(
        test_probs,
        threshold=threshold_info["threshold"],
        strategy=config.training.threshold_strategy,
    )
    metrics = classification_metrics(y_test, pred_test, bundle.class_names)
    metrics["history"] = history
    metrics["best_validation"] = best_val_metrics
    metrics["calibrated_validation"] = threshold_info["validation_metrics"]
    metrics["selection_metric"] = config.training.selection_metric
    metrics["best_selected_score"] = float(best_score)
    metrics["threshold_strategy"] = config.training.threshold_strategy
    metrics["attack_threshold"] = threshold_info["threshold"]
    metrics["threshold_search"] = threshold_info
    metrics["validation_test_gap"] = (
        float(best_val_metrics["macro_f1"] - metrics["macro_f1"])
        if best_val_metrics is not None
        else None
    )
    metrics["experiment"] = config.experiment.name
    metrics["model_name"] = config.model.name
    metrics["data_protocol"] = config.data.protocol
    metrics["window_size"] = config.data.window_size
    metrics["window_stride"] = config.data.window_stride
    metrics["window_label_rule"] = config.data.window_label_rule
    metrics["loss"] = config.training.loss
    metrics["focal_gamma"] = config.training.focal_gamma
    metrics["device"] = str(device)
    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["latency"] = _measure_latency(model, test_loader, device)
    metrics["data_metadata"] = bundle.metadata

    write_json(output_dir / "metrics.json", metrics)
    if config_path is not None:
        shutil.copyfile(config_path, output_dir / "config.toml")
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


def _selection_score(metrics: dict, selection_metric: str) -> float:
    if selection_metric == "accuracy":
        return float(metrics["accuracy"])
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    if selection_metric == "weighted_f1":
        return float(metrics["weighted_f1"])
    raise ValueError(
        f"Unsupported selection_metric: {selection_metric!r}. "
        "Expected 'accuracy', 'macro_f1', or 'weighted_f1'."
    )


def _probability_predictions(
    probabilities: np.ndarray,
    threshold: float | None,
    strategy: str,
) -> np.ndarray:
    if strategy == "argmax" or threshold is None:
        return probabilities.argmax(axis=1)
    if probabilities.shape[1] != 2:
        raise ValueError("Threshold calibration is only supported for binary classification.")
    return (probabilities[:, 1] >= threshold).astype(np.int64)


def _calibrate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    strategy: str,
    min_precision: float,
    min_recall: float,
    grid_step: float,
) -> dict:
    if strategy == "argmax":
        predictions = probabilities.argmax(axis=1)
        return {
            "strategy": strategy,
            "threshold": None,
            "validation_metrics": classification_metrics(y_true, predictions, class_names),
            "candidate_count": 0,
        }
    if probabilities.shape[1] != 2:
        raise ValueError("Threshold calibration is only supported for binary classification.")
    if not 0.0 < grid_step <= 0.5:
        raise ValueError("threshold_grid_step must be in the range (0, 0.5].")

    candidates = np.arange(grid_step, 1.0, grid_step)
    scored = []
    for threshold in candidates:
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        metrics = classification_metrics(y_true, predictions, class_names)
        attack = metrics["per_class"]["attack"]
        scored.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(metrics["macro_f1"]),
                "attack_precision": float(attack["precision"]),
                "attack_recall": float(attack["recall"]),
                "attack_f1": float(attack["f1"]),
                "attack_fpr": float(attack["false_positive_rate"]),
                "metrics": metrics,
            }
        )

    if strategy == "binary_f1":
        selected = max(scored, key=lambda row: (row["attack_f1"], row["macro_f1"], -row["attack_fpr"]))
    elif strategy == "binary_macro_f1":
        selected = max(scored, key=lambda row: (row["macro_f1"], row["attack_f1"], -row["attack_fpr"]))
    elif strategy == "binary_recall_at_precision":
        feasible = [row for row in scored if row["attack_precision"] >= min_precision]
        selected = max(
            feasible or scored,
            key=lambda row: (row["attack_recall"], row["attack_f1"], -row["attack_fpr"]),
        )
    elif strategy == "binary_precision_at_recall":
        feasible = [row for row in scored if row["attack_recall"] >= min_recall]
        selected = max(
            feasible or scored,
            key=lambda row: (row["attack_precision"], row["attack_f1"], -row["attack_fpr"]),
        )
    else:
        raise ValueError(
            f"Unsupported threshold_strategy: {strategy!r}. Expected 'argmax', "
            "'binary_f1', 'binary_macro_f1', 'binary_recall_at_precision', or "
            "'binary_precision_at_recall'."
        )

    return {
        "strategy": strategy,
        "threshold": selected["threshold"],
        "validation_metrics": selected["metrics"],
        "selected_validation_attack_precision": selected["attack_precision"],
        "selected_validation_attack_recall": selected["attack_recall"],
        "selected_validation_attack_f1": selected["attack_f1"],
        "selected_validation_attack_fpr": selected["attack_fpr"],
        "candidate_count": len(scored),
        "min_precision": min_precision,
        "min_recall": min_recall,
        "grid_step": grid_step,
    }


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x_batch, y_batch in tqdm(loader, desc="train", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x_batch)
        total_items += len(x_batch)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(predictions)
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


@torch.no_grad()
def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_probs: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        y_probs.append(probabilities)
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_probs)


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


if __name__ == "__main__":
    main()
