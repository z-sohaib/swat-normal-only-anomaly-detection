from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swat_ids.config import load_config
from swat_ids.data import load_swat
from swat_ids.metrics import classification_metrics
from swat_ids.models import build_anomaly_model
from swat_ids.operational import load_official_attack_list
from swat_ids.postprocessing import postprocess_anomaly_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep SWaT anomaly scores from per-feature reconstruction errors."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "global_mse",
            "max_feature",
            "top3_feature_mean",
            "top5_feature_mean",
            "top10_feature_mean",
            "max_robust_z",
            "top3_robust_z",
            "top5_robust_z",
            "top10_robust_z",
        ],
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[99.5, 99.7, 99.9, 99.95, 99.99],
    )
    parser.add_argument("--min-consecutive", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--merge-gaps", nargs="+", type=int, default=[0, 1, 2, 5])
    parser.add_argument(
        "--calibration-sources",
        nargs="+",
        default=["normal_validation"],
        choices=["normal_validation", "attack_burn_in", "validation_plus_burn_in"],
    )
    parser.add_argument("--output", default="runs/anomaly_score_method_sweep.csv")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for run_text in args.run:
        rows.extend(
            _sweep_run(
                run_dir=Path(run_text),
                methods=args.methods,
                percentiles=args.percentiles,
                min_consecutive_values=args.min_consecutive,
                merge_gap_values=args.merge_gaps,
                calibration_sources=args.calibration_sources,
                batch_size=args.batch_size,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, rows)
    print(f"Wrote {output}")
    return 0


def _sweep_run(
    run_dir: Path,
    methods: list[str],
    percentiles: list[float],
    min_consecutive_values: list[int],
    merge_gap_values: list[int],
    calibration_sources: list[str],
    batch_size: int | None,
) -> list[dict[str, Any]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.toml")
    bundle = load_swat(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_anomaly_model(config.model, input_features=bundle.input_features).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))

    effective_batch_size = batch_size or config.training.batch_size
    val_feature_errors = _feature_errors(
        model,
        _loader(bundle.x_val, bundle.y_val, effective_batch_size),
        device,
        anomaly_loss=config.training.loss,
    )[1]
    y_true, test_feature_errors = _feature_errors(
        model,
        _loader(bundle.x_test, bundle.y_test, effective_batch_size),
        device,
        anomaly_loss=config.training.loss,
    )
    score_context = _score_context(val_feature_errors)
    _, _, window_starts, window_ends, origin = _read_test_score_timeline(run_dir / "window_scores.csv")
    attack_list_path = config.data.attack_list_path or Path("data/raw/List_of_attacks_Final.xlsx")
    event_masks = _event_masks(
        attack_list=load_official_attack_list(attack_list_path),
        window_starts=window_starts,
        window_ends=window_ends,
        origin=origin,
    )
    burn_in_mask = _burn_in_mask(event_masks, window_ends)
    total_hours = _duration_hours(window_starts, window_ends)

    rows = []
    for method in methods:
        val_scores = _score_from_feature_errors(val_feature_errors, method, score_context)
        test_scores = _score_from_feature_errors(test_feature_errors, method, score_context)
        for calibration_source in calibration_sources:
            calibration_scores = _calibration_scores(
                source=calibration_source,
                validation_scores=val_scores,
                test_scores=test_scores,
                burn_in_mask=burn_in_mask,
            )
            for percentile in percentiles:
                threshold = float(np.percentile(calibration_scores, percentile))
                raw_pred = (test_scores > threshold).astype(np.int64)
                for min_consecutive in min_consecutive_values:
                    for merge_gap in merge_gap_values:
                        pred = postprocess_anomaly_predictions(
                            raw_pred,
                            min_consecutive_windows=min_consecutive,
                            merge_gap_windows=merge_gap,
                        )
                        point = classification_metrics(y_true, pred, ["normal", "attack"])
                        events = _fast_official_event_metrics(pred, event_masks, window_ends, total_hours)
                        rows.append(
                            _row(
                                run_dir=run_dir,
                                metrics=metrics,
                                method=method,
                                calibration_source=calibration_source,
                                percentile=percentile,
                                threshold=threshold,
                                min_consecutive=min_consecutive,
                                merge_gap=merge_gap,
                                point=point,
                                events=events,
                            )
                        )
    return rows


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def _feature_errors(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    anomaly_loss: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    errors = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        model_input, target = _anomaly_input_target(x_batch, anomaly_loss)
        reconstruction = model(model_input)
        batch_errors = torch.mean((reconstruction - target) ** 2, dim=1)
        errors.append(batch_errors.cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(errors)


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
        k = _topk_from_method(method)
        return _topk_mean(feature_errors, k)
    if method == "max_robust_z":
        robust = _robust_z(feature_errors, context)
        return robust.max(axis=1)
    if method.startswith("top") and method.endswith("_robust_z"):
        k = _topk_from_method(method)
        return _topk_mean(_robust_z(feature_errors, context), k)
    raise ValueError(f"Unsupported score method: {method}")


def _calibration_scores(
    source: str,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    burn_in_mask: np.ndarray,
) -> np.ndarray:
    if source == "normal_validation":
        return validation_scores
    burn_in_scores = test_scores[burn_in_mask]
    if len(burn_in_scores) == 0:
        raise ValueError("No pre-attack burn-in windows were found.")
    if source == "attack_burn_in":
        return burn_in_scores
    if source == "validation_plus_burn_in":
        return np.concatenate([validation_scores, burn_in_scores])
    raise ValueError(f"Unsupported calibration source: {source}")


def _robust_z(feature_errors: np.ndarray, context: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum((feature_errors - context["median"]) / context["iqr"], 0.0)


def _topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(max(k, 1), values.shape[1])
    partition = np.partition(values, values.shape[1] - k, axis=1)
    return partition[:, -k:].mean(axis=1)


def _topk_from_method(method: str) -> int:
    digits = "".join(char for char in method.split("_", 1)[0] if char.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse top-k value from method: {method}")
    return int(digits)


def _read_test_score_timeline(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime]:
    y_true = []
    scores = []
    starts = []
    ends = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            scores.append(float(row["anomaly_score"]))
            starts.append(datetime.fromisoformat(row["start_time"]))
            ends.append(datetime.fromisoformat(row["end_time"]))

    if not starts:
        raise ValueError(f"No window scores found in {path}")
    origin = starts[0]
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray([(value - origin).total_seconds() for value in starts], dtype=np.float64),
        np.asarray([(value - origin).total_seconds() for value in ends], dtype=np.float64),
        origin,
    )


def _event_masks(
    attack_list: dict[str, Any],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    origin: datetime,
) -> list[dict[str, Any]]:
    masks = []
    timeline = {"start": float(window_starts[0]), "end": float(window_ends[-1])}
    for event in attack_list["events"]:
        start = (event["start"] - origin).total_seconds()
        end = (event["end"] - origin).total_seconds()
        if start > timeline["end"] or end < timeline["start"]:
            continue
        clipped_start = max(start, timeline["start"])
        clipped_end = min(end, timeline["end"])
        mask = (window_starts <= clipped_end) & (clipped_start <= window_ends)
        masks.append({"start_seconds": clipped_start, "mask": mask})
    return masks


def _burn_in_mask(event_masks: list[dict[str, Any]], window_ends: np.ndarray) -> np.ndarray:
    if not event_masks:
        return np.ones_like(window_ends, dtype=bool)
    first_attack_start = min(float(event["start_seconds"]) for event in event_masks)
    return window_ends < first_attack_start


def _fast_official_event_metrics(
    y_pred: np.ndarray,
    event_masks: list[dict[str, Any]],
    window_ends: np.ndarray,
    total_hours: float,
) -> dict[str, Any]:
    predicted = y_pred.astype(np.int64)
    official_any = np.zeros_like(predicted, dtype=bool)
    detected = 0
    missed = 0
    delays = []
    for event in event_masks:
        mask = event["mask"]
        official_any |= mask
        hit_indices = np.flatnonzero((predicted == 1) & mask)
        if len(hit_indices) == 0:
            missed += 1
            continue
        detected += 1
        first_hit = int(hit_indices[0])
        delays.append(max(float(window_ends[first_hit] - event["start_seconds"]), 0.0))

    false_alarm_segments = 0
    for start, end in _segments(predicted):
        if not bool(official_any[start : end + 1].any()):
            false_alarm_segments += 1

    false_alarm_windows = int(((predicted == 1) & (~official_any)).sum())
    true_event_count = len(event_masks)
    return {
        "true_event_count": true_event_count,
        "detected_event_count": detected,
        "event_recall": detected / true_event_count if true_event_count else 0.0,
        "missed_event_count": missed,
        "false_alarm_segment_count": false_alarm_segments,
        "false_alarm_windows": false_alarm_windows,
        "false_alarm_segments_per_hour": false_alarm_segments / total_hours if total_hours > 0 else 0.0,
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
    }


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


def _duration_hours(window_starts: np.ndarray, window_ends: np.ndarray) -> float:
    return max(float(window_ends[-1] - window_starts[0]) / 3600.0, 0.0)


def _row(
    run_dir: Path,
    metrics: dict[str, Any],
    method: str,
    calibration_source: str,
    percentile: float,
    threshold: float,
    min_consecutive: int,
    merge_gap: int,
    point: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:
    attack = point["per_class"]["attack"]
    normal = point["per_class"]["normal"]
    return {
        "experiment": metrics.get("experiment", run_dir.name),
        "model_name": metrics.get("model_name", ""),
        "window_size": metrics.get("window_size", ""),
        "window_stride": metrics.get("window_stride", ""),
        "score_method": method,
        "calibration_source": calibration_source,
        "percentile": percentile,
        "threshold": threshold,
        "min_consecutive_windows": min_consecutive,
        "merge_gap_windows": merge_gap,
        "accuracy": point["accuracy"],
        "macro_f1": point["macro_f1"],
        "weighted_f1": point["weighted_f1"],
        "attack_precision": attack["precision"],
        "attack_recall": attack["recall"],
        "attack_f1": attack["f1"],
        "attack_fpr": attack["false_positive_rate"],
        "normal_recall": normal["recall"],
        "true_event_count": events["true_event_count"],
        "detected_event_count": events["detected_event_count"],
        "event_recall": events["event_recall"],
        "missed_event_count": events["missed_event_count"],
        "false_alarm_segment_count": events["false_alarm_segment_count"],
        "false_alarm_windows": events["false_alarm_windows"],
        "false_alarm_segments_per_hour": events["false_alarm_segments_per_hour"],
        "mean_detection_delay_seconds": events["mean_detection_delay_seconds"],
        "metrics_path": str(run_dir / "metrics.json"),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
