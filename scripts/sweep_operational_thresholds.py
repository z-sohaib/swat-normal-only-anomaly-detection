from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_operational import _event_metrics, _window_timeline
from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep binary attack thresholds for SWaT operational metrics."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_swat(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model, bundle.input_features, bundle.num_classes).to(device)
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location=device))

    loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle.x_test, dtype=torch.float32),
            torch.tensor(bundle.y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    y_true, probabilities = _predict_probs(model, loader, device)
    split_meta = bundle.metadata["splits"]["test"]
    timeline = _window_timeline(
        first_timestamp=split_meta["first_timestamp"],
        window_count=len(y_true),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )

    rows = []
    for threshold in np.arange(args.start, args.stop + args.step / 2, args.step):
        threshold = round(float(threshold), 6)
        y_pred = (probabilities[:, 1] >= threshold).astype(np.int64)
        point = classification_metrics(y_true, y_pred, bundle.class_names)
        event = _event_metrics(y_true, y_pred, timeline, stride=config.data.window_stride)
        attack = point["per_class"]["attack"]
        rows.append(
            {
                "threshold": threshold,
                "accuracy": point["accuracy"],
                "macro_f1": point["macro_f1"],
                "attack_precision": attack["precision"],
                "attack_recall": attack["recall"],
                "attack_f1": attack["f1"],
                "attack_fpr": attack["false_positive_rate"],
                "event_recall": event["event_recall"],
                "missed_event_count": event["missed_event_count"],
                "false_alarm_segment_count": event["false_alarm_segment_count"],
                "false_alarm_windows": event["false_alarm_windows"],
                "false_alarm_segments_per_hour": event["false_alarm_segments_per_hour"],
                "mean_detection_delay_seconds": event["mean_detection_delay_seconds"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_macro = max(rows, key=lambda row: row["macro_f1"])
    best_event = max(
        rows,
        key=lambda row: (
            row["event_recall"],
            -row["false_alarm_segment_count"],
            row["attack_f1"],
        ),
    )
    print(f"Wrote {args.output}")
    print(f"Best macro-F1 threshold: {best_macro}")
    print(f"Best event-recall threshold: {best_event}")
    return 0


@torch.no_grad()
def _predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    probabilities = []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(probabilities)


if __name__ == "__main__":
    raise SystemExit(main())
