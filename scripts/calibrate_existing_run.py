from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_operational import _event_metrics, _window_timeline
from evaluate_operational import _load_official_attack_list, _official_event_metrics
from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a SWaT binary threshold on validation data and apply it to test data."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["macro_f1", "attack_f1", "recall_at_precision"],
        default="macro_f1",
    )
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--attack-list", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_swat(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, bundle.input_features, bundle.num_classes).to(device)
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location=device))

    y_val, val_probs = _predict_probs(model, _loader(bundle.x_val, bundle.y_val, args.batch_size), device)
    y_test, test_probs = _predict_probs(model, _loader(bundle.x_test, bundle.y_test, args.batch_size), device)

    val_rows = _score_thresholds(
        y_true=y_val,
        probabilities=val_probs,
        class_names=bundle.class_names,
        start=args.start,
        stop=args.stop,
        step=args.step,
    )
    selected = _select(val_rows, strategy=args.strategy, min_precision=args.min_precision)
    test_row = _score_single_threshold(
        y_true=y_test,
        probabilities=test_probs,
        class_names=bundle.class_names,
        threshold=selected["threshold"],
    )

    split_meta = bundle.metadata["splits"]["test"]
    timeline = _window_timeline(
        first_timestamp=split_meta["first_timestamp"],
        window_count=len(y_test),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )
    y_test_pred = (test_probs[:, 1] >= selected["threshold"]).astype(np.int64)
    if args.attack_list is not None:
        operational = _official_event_metrics(
            official_attack_list=_load_official_attack_list(args.attack_list),
            y_pred=y_test_pred,
            timeline=timeline,
            stride=config.data.window_stride,
        )
    else:
        operational = _event_metrics(y_test, y_test_pred, timeline, stride=config.data.window_stride)

    payload = {
        "run_dir": str(args.run_dir),
        "config": str(args.config),
        "strategy": args.strategy,
        "selected_threshold": selected["threshold"],
        "validation_selection": selected,
        "test_metrics": test_row,
        "operational_metrics": operational,
    }

    output_json = args.output_json or args.run_dir / f"calibrated_{args.strategy}.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        row = _summary_row(payload)
        exists = args.output_csv.exists()
        with args.output_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Wrote {args.output_csv}")

    print(json.dumps(_summary_row(payload), indent=2))
    return 0


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )


@torch.no_grad()
def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probabilities = []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y_batch.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _score_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    start: float,
    stop: float,
    step: float,
) -> list[dict]:
    return [
        _score_single_threshold(y_true, probabilities, class_names, threshold=round(float(threshold), 6))
        for threshold in np.arange(start, stop + step / 2, step)
    ]


def _score_single_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> dict:
    predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
    metrics = classification_metrics(y_true, predictions, class_names)
    attack = metrics["per_class"]["attack"]
    return {
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "attack_precision": attack["precision"],
        "attack_recall": attack["recall"],
        "attack_f1": attack["f1"],
        "attack_fpr": attack["false_positive_rate"],
    }


def _select(rows: list[dict], strategy: str, min_precision: float) -> dict:
    if strategy == "macro_f1":
        return max(rows, key=lambda row: (row["macro_f1"], row["attack_f1"], -row["attack_fpr"]))
    if strategy == "attack_f1":
        return max(rows, key=lambda row: (row["attack_f1"], row["macro_f1"], -row["attack_fpr"]))
    feasible = [row for row in rows if row["attack_precision"] >= min_precision]
    return max(
        feasible or rows,
        key=lambda row: (row["attack_recall"], row["attack_f1"], -row["attack_fpr"]),
    )


def _summary_row(payload: dict) -> dict:
    test = payload["test_metrics"]
    operational = payload["operational_metrics"]
    return {
        "run_dir": payload["run_dir"],
        "strategy": payload["strategy"],
        "selected_threshold": payload["selected_threshold"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "weighted_f1": test["weighted_f1"],
        "attack_precision": test["attack_precision"],
        "attack_recall": test["attack_recall"],
        "attack_f1": test["attack_f1"],
        "attack_fpr": test["attack_fpr"],
        "event_recall": operational["event_recall"],
        "missed_event_count": operational["missed_event_count"],
        "false_alarm_segment_count": operational["false_alarm_segment_count"],
        "false_alarm_windows": operational["false_alarm_windows"],
        "mean_detection_delay_seconds": operational["mean_detection_delay_seconds"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
