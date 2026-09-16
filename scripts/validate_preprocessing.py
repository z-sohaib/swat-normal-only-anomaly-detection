from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swat_ids.config import load_config
from swat_ids.data import load_swat


EXPECTED_SWAT = {
    "feature_count": 51,
    "normal_rows": 495000,
    "attack_period_rows": 449919,
    "attack_period_normal_rows": 395298,
    "attack_period_attack_rows": 54621,
    "combined_rows": 944919,
    "combined_normal_rows": 890298,
    "combined_attack_rows": 54621,
    "sampling_seconds": 1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SWaT Phase 2 preprocessing.")
    parser.add_argument("--config", required=True, help="Path to SWaT TOML config.")
    parser.add_argument(
        "--output-dir",
        default="runs/preprocessing",
        help="Directory for preprocessing validation artifacts.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_swat(config.data, seed=config.experiment.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": config.experiment.name,
        "data": {
            "protocol": config.data.protocol,
            "normal_path": str(config.data.normal_path) if config.data.normal_path else None,
            "attack_path": str(config.data.attack_path) if config.data.attack_path else None,
            "merged_path": str(config.data.merged_path) if config.data.merged_path else None,
            "validation_size": config.data.validation_size,
            "test_size": config.data.test_size,
            "window_size": config.data.window_size,
            "window_stride": config.data.window_stride,
            "window_label_rule": config.data.window_label_rule,
            "scale_method": config.data.scale_method,
        },
        "arrays": {
            "x_train_shape": list(bundle.x_train.shape),
            "x_val_shape": list(bundle.x_val.shape),
            "x_test_shape": list(bundle.x_test.shape),
            "y_train_shape": list(bundle.y_train.shape),
            "y_val_shape": list(bundle.y_val.shape),
            "y_test_shape": list(bundle.y_test.shape),
        },
        "metadata": bundle.metadata,
        "phase2_checks": _phase2_checks(bundle),
    }

    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_window_distribution(output_dir / "window_distribution.csv", bundle.metadata)
    _write_split_boundaries(output_dir / "split_boundaries.json", bundle.metadata)

    print(json.dumps(summary["phase2_checks"], indent=2))
    print(f"Wrote {output_dir / 'preprocessing_summary.json'}")
    print(f"Wrote {output_dir / 'window_distribution.csv'}")
    print(f"Wrote {output_dir / 'split_boundaries.json'}")
    return 0 if summary["phase2_checks"]["phase2_ready"] else 2


def _phase2_checks(bundle) -> dict:
    protocol = bundle.metadata["protocol"]
    feature_count = bundle.metadata["feature_count"]
    train_counts = bundle.metadata["splits"]["train"]["window_label_counts"]
    val_counts = bundle.metadata["splits"]["validation"]["window_label_counts"]
    test_counts = bundle.metadata["splits"]["test"]["window_label_counts"]
    train_point_counts = bundle.metadata["splits"]["train"]["point_label_counts"]
    val_point_counts = bundle.metadata["splits"]["validation"]["point_label_counts"]
    test_point_counts = bundle.metadata["splits"]["test"]["point_label_counts"]
    train_has_both_classes = train_counts["normal"] > 0 and train_counts["attack"] > 0
    val_has_any = val_counts["normal"] + val_counts["attack"] > 0
    test_has_any = test_counts["normal"] + test_counts["attack"] > 0
    train_val_normal_only = (
        train_point_counts["attack"] == 0
        and val_point_counts["attack"] == 0
        and train_point_counts["normal"] > 0
        and val_point_counts["normal"] > 0
    )
    test_attack_period_valid = (
        test_point_counts["normal"] == EXPECTED_SWAT["attack_period_normal_rows"]
        and test_point_counts["attack"] == EXPECTED_SWAT["attack_period_attack_rows"]
    )
    standard_anomaly_protocol = (
        protocol == "standard_normal_attack"
        and feature_count == EXPECTED_SWAT["feature_count"]
        and train_val_normal_only
        and test_attack_period_valid
        and test_has_any
    )
    supervised_protocol = protocol == "supervised_chronological" and train_has_both_classes and val_has_any and test_has_any
    timestamp_checks = _timestamp_checks(bundle.metadata)
    official_shape_checks = _official_shape_checks(bundle.metadata)
    return {
        "phase2_ready": bool((standard_anomaly_protocol or supervised_protocol) and timestamp_checks["all_splits_monotonic"]),
        "valid_for_standard_normal_only_anomaly_detection": bool(standard_anomaly_protocol),
        "valid_for_supervised_classifier": bool(train_has_both_classes and val_has_any and test_has_any),
        "train_has_both_classes": bool(train_has_both_classes),
        "train_validation_are_normal_only": bool(train_val_normal_only),
        "test_is_full_attack_period": bool(test_attack_period_valid),
        "feature_count_is_51": bool(feature_count == EXPECTED_SWAT["feature_count"]),
        "validation_has_windows": bool(val_has_any),
        "test_has_windows": bool(test_has_any),
        "official_shape_checks": official_shape_checks,
        "timestamp_checks": timestamp_checks,
        "notes": [
            "standard_normal_attack is the paper-comparable SWaT normal-only anomaly-detection protocol.",
            "It intentionally has normal-only train/validation labels and must not be used with supervised cross-entropy classifier training.",
            "supervised_chronological can be used for supervised classifier experiments only when train_has_both_classes is true.",
        ],
    }


def _official_shape_checks(metadata: dict) -> dict:
    splits = metadata["splits"]
    if metadata["protocol"] == "standard_normal_attack":
        standard_total = splits["train"]["point_rows"] + splits["validation"]["point_rows"]
        attack_total = splits["test"]["point_rows"]
        return {
            "protocol": "standard_normal_attack",
            "normal_file_rows_match_495000": bool(standard_total == EXPECTED_SWAT["normal_rows"]),
            "attack_period_rows_match_449919": bool(attack_total == EXPECTED_SWAT["attack_period_rows"]),
            "attack_period_normal_rows_match_395298": bool(
                splits["test"]["point_label_counts"]["normal"] == EXPECTED_SWAT["attack_period_normal_rows"]
            ),
            "attack_period_attack_rows_match_54621": bool(
                splits["test"]["point_label_counts"]["attack"] == EXPECTED_SWAT["attack_period_attack_rows"]
            ),
        }

    total_rows = sum(split["point_rows"] for split in splits.values())
    total_normal = sum(split["point_label_counts"]["normal"] for split in splits.values())
    total_attack = sum(split["point_label_counts"]["attack"] for split in splits.values())
    return {
        "protocol": metadata["protocol"],
        "combined_rows_match_944919": bool(total_rows == EXPECTED_SWAT["combined_rows"]),
        "combined_normal_rows_match_890298": bool(total_normal == EXPECTED_SWAT["combined_normal_rows"]),
        "combined_attack_rows_match_54621": bool(total_attack == EXPECTED_SWAT["combined_attack_rows"]),
    }


def _timestamp_checks(metadata: dict) -> dict:
    split_checks = {}
    for split, split_meta in metadata["splits"].items():
        split_checks[split] = {
            "monotonic": split_meta["timestamp_monotonic"],
            "estimated_sampling_seconds": split_meta["estimated_sampling_seconds"],
            "sampling_is_1hz": split_meta["estimated_sampling_seconds"] == EXPECTED_SWAT["sampling_seconds"],
        }
    return {
        "all_splits_monotonic": all(value["monotonic"] is True for value in split_checks.values()),
        "all_splits_look_1hz": all(value["sampling_is_1hz"] for value in split_checks.values()),
        "splits": split_checks,
    }


def _write_window_distribution(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "point_count", "window_count"])
        for split, split_meta in metadata["splits"].items():
            for label in ["normal", "attack"]:
                writer.writerow(
                    [
                        split,
                        label,
                        split_meta["point_label_counts"][label],
                        split_meta["window_label_counts"][label],
                    ]
                )


def _write_split_boundaries(path: Path, metadata: dict) -> None:
    boundaries = {
        split: {
            "source": split_meta["source"],
            "point_rows": split_meta["point_rows"],
            "window_count": split_meta["window_count"],
            "first_timestamp": split_meta["first_timestamp"],
            "last_timestamp": split_meta["last_timestamp"],
        }
        for split, split_meta in metadata["splits"].items()
    }
    path.write_text(json.dumps(boundaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
