from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_anomaly_model
from swat_ids.postprocessing import postprocess_anomaly_predictions
from swat_ids.train_anomaly import (
    _feature_errors,
    _loader,
    _score_context,
    _score_from_feature_errors,
)
from swat_ids.utils import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SWaT Objective 3 drift and perturbation behavior.")
    parser.add_argument("--config", required=True, help="Config used to train the selected anomaly model.")
    parser.add_argument("--model-path", required=True, help="Path to model.pt.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", nargs="*", type=float, default=[0.01, 0.03, 0.05])
    parser.add_argument("--sensor-dropout", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    bundle = load_swat(config.data, seed=config.experiment.seed)
    if np.any(bundle.y_train != 0) or np.any(bundle.y_val != 0):
        raise ValueError("SWaT Objective 3 drift script expects the normal-only anomaly protocol.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_anomaly_model(config.model, input_features=bundle.input_features).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    val_mid, val_late = _split_validation(bundle.x_val, bundle.y_val)
    mid_loader = _loader(val_mid[0], val_mid[1], config.training.batch_size, shuffle=False)
    late_loader = _loader(val_late[0], val_late[1], config.training.batch_size, shuffle=False)
    test_loader = _loader(bundle.x_test, bundle.y_test, config.training.batch_size, shuffle=False)

    _, mid_errors = _feature_errors(model, mid_loader, device, anomaly_loss=config.training.loss)
    y_late, late_errors = _feature_errors(model, late_loader, device, anomaly_loss=config.training.loss)
    y_test, test_errors = _feature_errors(model, test_loader, device, anomaly_loss=config.training.loss)
    context = _score_context(mid_errors)
    mid_scores = _score_from_feature_errors(mid_errors, config.training.anomaly_score_method, context)
    late_scores = _score_from_feature_errors(late_errors, config.training.anomaly_score_method, context)
    test_scores = _score_from_feature_errors(test_errors, config.training.anomaly_score_method, context)
    threshold = float(np.percentile(mid_scores, config.training.anomaly_threshold_percentile))

    late_pred_raw = (late_scores > threshold).astype(np.int64)
    late_pred = postprocess_anomaly_predictions(
        late_pred_raw,
        min_consecutive_windows=config.training.anomaly_min_consecutive_windows,
        merge_gap_windows=config.training.anomaly_merge_gap_windows,
    )
    test_pred_raw = (test_scores > threshold).astype(np.int64)
    test_pred = postprocess_anomaly_predictions(
        test_pred_raw,
        min_consecutive_windows=config.training.anomaly_min_consecutive_windows,
        merge_gap_windows=config.training.anomaly_merge_gap_windows,
    )

    clean_test_metrics = classification_metrics(y_test, test_pred, bundle.class_names)
    results: dict[str, Any] = {
        "config": args.config,
        "model_path": args.model_path,
        "seed": args.seed,
        "threshold_source": "first_half_of_normal_validation",
        "drift_test_source": "second_half_of_normal_validation",
        "threshold": threshold,
        "normal_drift": {
            "windows": int(len(y_late)),
            "false_positive_windows": int(late_pred.sum()),
            "false_positive_rate": float(late_pred.mean()) if len(late_pred) else 0.0,
            "raw_false_positive_rate": float(late_pred_raw.mean()) if len(late_pred_raw) else 0.0,
            "score_mean": float(np.mean(late_scores)),
            "score_p95": float(np.percentile(late_scores, 95)),
            "score_p99": float(np.percentile(late_scores, 99)),
        },
        "attack_period_clean": clean_test_metrics,
        "perturbations": [],
    }

    rng = np.random.default_rng(args.seed)
    for std in args.noise_std:
        perturbed = bundle.x_test + rng.normal(0.0, std, size=bundle.x_test.shape).astype(np.float32)
        metrics = _evaluate_anomaly(model, perturbed.astype(np.float32), bundle.y_test, config, context, threshold, device)
        results["perturbations"].append(_payload("gaussian_sensor_noise", std, clean_test_metrics, metrics))

    for rate in args.sensor_dropout:
        mask = rng.random(bundle.x_test.shape) >= rate
        perturbed = bundle.x_test * mask.astype(np.float32)
        metrics = _evaluate_anomaly(model, perturbed.astype(np.float32), bundle.y_test, config, context, threshold, device)
        results["perturbations"].append(_payload("sensor_dropout", rate, clean_test_metrics, metrics))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved SWaT Objective 3 evaluation to {output_path}")
    return 0


def _split_validation(x_val: np.ndarray, y_val: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    split = max(1, len(y_val) // 2)
    if split >= len(y_val):
        raise ValueError("Validation split is too small for drift evaluation.")
    return (x_val[:split], y_val[:split]), (x_val[split:], y_val[split:])


def _evaluate_anomaly(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    config: Any,
    context: dict[str, np.ndarray],
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    loader = _loader(x, y, config.training.batch_size, shuffle=False)
    y_true, feature_errors = _feature_errors(model, loader, device, anomaly_loss=config.training.loss)
    scores = _score_from_feature_errors(feature_errors, config.training.anomaly_score_method, context)
    raw = (scores > threshold).astype(np.int64)
    pred = postprocess_anomaly_predictions(
        raw,
        min_consecutive_windows=config.training.anomaly_min_consecutive_windows,
        merge_gap_windows=config.training.anomaly_merge_gap_windows,
    )
    metrics = classification_metrics(y_true, pred, ["normal", "attack"])
    metrics["raw_positive_rate"] = float(raw.mean()) if len(raw) else 0.0
    metrics["postprocessed_positive_rate"] = float(pred.mean()) if len(pred) else 0.0
    return metrics


def _payload(kind: str, value: float, clean: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": kind,
        "value": float(value),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy_drop": float(clean["accuracy"] - metrics["accuracy"]),
        "macro_f1_drop": float(clean["macro_f1"] - metrics["macro_f1"]),
        "weighted_f1_drop": float(clean["weighted_f1"] - metrics["weighted_f1"]),
        "per_class": metrics["per_class"],
        "raw_positive_rate": metrics.get("raw_positive_rate"),
        "postprocessed_positive_rate": metrics.get("postprocessed_positive_rate"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
