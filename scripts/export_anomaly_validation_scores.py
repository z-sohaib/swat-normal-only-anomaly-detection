from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.models import build_anomaly_model
from swat_ids.operational import window_timeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SWaT anomaly validation scores.")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    for run_text in args.run:
        _export(Path(run_text), batch_size=args.batch_size)
    return 0


def _export(run_dir: Path, batch_size: int | None) -> None:
    config = load_config(run_dir / "config.toml")
    bundle = load_swat(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_anomaly_model(config.model, input_features=bundle.input_features).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    loader = _loader(bundle.x_val, bundle.y_val, batch_size or config.training.batch_size)
    y_true, errors = _score_windows(model, loader, device, anomaly_loss=config.training.loss)
    split_meta = bundle.metadata["splits"]["validation"]
    timeline = window_timeline(
        first_timestamp=split_meta["first_timestamp"],
        window_count=len(y_true),
        window_size=config.data.window_size,
        stride=config.data.window_stride,
    )
    _write_scores(run_dir / "validation_scores.csv", y_true, errors, timeline)
    print(f"Wrote {run_dir / 'validation_scores.csv'}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _score_windows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    anomaly_loss: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    scores = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        errors = torch.mean((model(model_input) - target) ** 2, dim=(1, 2))
        scores.append(errors.cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(scores)


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
    raise ValueError(f"Unsupported anomaly loss: {anomaly_loss!r}.")


def _write_scores(
    path: Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    timeline: list[dict[str, str | int | None]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_index", "start_time", "end_time", "y_true", "anomaly_score"],
        )
        writer.writeheader()
        for idx, (truth, score) in enumerate(zip(y_true, scores)):
            writer.writerow(
                {
                    "window_index": idx,
                    "start_time": timeline[idx]["start"],
                    "end_time": timeline[idx]["end"],
                    "y_true": int(truth),
                    "anomaly_score": float(score),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
