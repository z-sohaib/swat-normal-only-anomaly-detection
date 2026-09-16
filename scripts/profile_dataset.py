from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_STANDARD = {
    "SWaT_Dataset_Normal_v1.csv": {
        "role": "standard normal training file",
        "expected_rows": 495000,
        "expected_labels": {"Normal": 495000},
    },
    "SWaT_Dataset_Attack_v0.csv": {
        "role": "standard attack-period test file",
        "expected_rows": 449919,
        "expected_labels": {"Normal": 395298, "Attack": 54621},
    },
}

EXPECTED_FEATURE_COLUMNS = 53
EXPECTED_LABEL_COLUMN = "Normal/Attack"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile SWaT raw CSV files.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing SWaT CSV files.")
    parser.add_argument("--output-dir", default="runs/profile", help="Directory for generated profile artifacts.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.csv"))
    profiles = [_profile_csv(path) for path in files]
    verdict = _build_verdict(profiles)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "expected_standard_files": EXPECTED_STANDARD,
        "profiles": profiles,
        "verdict": verdict,
    }

    (output_dir / "dataset_profile.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_class_distribution(output_dir / "class_distribution.csv", profiles)
    _write_file_summary(output_dir / "file_summary.csv", profiles)
    _maybe_write_class_distribution_plot(output_dir / "class_distribution.png", profiles)

    print(json.dumps(verdict, indent=2))
    print(f"Wrote {output_dir / 'dataset_profile.json'}")
    print(f"Wrote {output_dir / 'class_distribution.csv'}")
    print(f"Wrote {output_dir / 'file_summary.csv'}")
    return 0 if verdict["standard_protocol_ready"] else 2


def _profile_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(handle, dialect)
        header = [column.strip() for column in next(reader)]
        label_idx = _find_label_column(header)
        timestamp_idx = _find_timestamp_column(header)

        labels: Counter[str] = Counter()
        row_count = 0
        bad_width_rows = 0
        missing_cells = 0
        first_timestamp = None
        last_timestamp = None

        for row in reader:
            row_count += 1
            if len(row) != len(header):
                bad_width_rows += 1
            missing_cells += sum(1 for value in row if value == "")
            if label_idx is not None and label_idx < len(row):
                labels[row[label_idx].strip()] += 1
            if timestamp_idx is not None and timestamp_idx < len(row):
                timestamp = row[timestamp_idx].strip()
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp

    standard_expectation = EXPECTED_STANDARD.get(path.name)
    standard_match = False
    reasons: list[str] = []
    if standard_expectation:
        standard_match = row_count == standard_expectation["expected_rows"] and dict(labels) == standard_expectation["expected_labels"]
        if row_count != standard_expectation["expected_rows"]:
            reasons.append(
                f"row count is {row_count}, expected {standard_expectation['expected_rows']}"
            )
        if dict(labels) != standard_expectation["expected_labels"]:
            reasons.append(
                f"label counts are {dict(labels)}, expected {standard_expectation['expected_labels']}"
            )

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "delimiter": dialect.delimiter,
        "column_count": len(header),
        "columns": header,
        "first_columns": header[:8],
        "last_columns": header[-8:],
        "label_column": header[label_idx] if label_idx is not None else None,
        "timestamp_column": header[timestamp_idx] if timestamp_idx is not None else None,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "data_rows": row_count,
        "label_counts": dict(labels),
        "bad_width_rows": bad_width_rows,
        "missing_cells": missing_cells,
        "matches_standard_expectation": standard_match,
        "standard_mismatch_reasons": reasons,
    }


def _find_label_column(header: list[str]) -> int | None:
    for idx, column in enumerate(header):
        normalized = column.lower().replace(" ", "")
        if normalized in {"normal/attack", "normalattack", "attack", "label"}:
            return idx
    return None


def _find_timestamp_column(header: list[str]) -> int | None:
    for idx, column in enumerate(header):
        if column.lower().strip() in {"timestamp", "time", "date"}:
            return idx
    return None


def _build_verdict(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {profile["file"]: profile for profile in profiles}
    required_names = set(EXPECTED_STANDARD)
    present_required = required_names.intersection(by_name)
    missing_required = sorted(required_names - present_required)
    matched_required = [
        name for name in present_required if by_name[name]["matches_standard_expectation"]
    ]

    all_required_match = len(matched_required) == len(required_names)
    swat_shaped = all(
        profile["column_count"] == EXPECTED_FEATURE_COLUMNS
        and profile["label_column"] == EXPECTED_LABEL_COLUMN
        for profile in profiles
        if profile["file"] in required_names or profile["file"] == "merged.csv"
    )

    explanation = []
    if swat_shaped:
        explanation.append("Available CSV files have the expected SWaT physical-data schema: timestamp, 51 process variables, and Normal/Attack label.")
    if not all_required_match:
        explanation.append("The available CSV row/label counts do not match the standard Normal_v1 + full Attack_v0 protocol used by most SWaT papers.")
    if "merged.csv" in by_name:
        explanation.append("merged.csv appears to be a pre-combined file and should not be used as the primary raw source for the strict protocol.")

    return {
        "standard_protocol_ready": all_required_match,
        "swat_schema_confirmed": swat_shaped,
        "required_files_present": sorted(present_required),
        "required_files_missing": missing_required,
        "required_files_matching_standard_counts": sorted(matched_required),
        "explanation": explanation,
        "recommended_action": (
            "Use these files only if intentionally reproducing a preprocessed/merged variant. For the thesis standard protocol, obtain the full SWaT_Dataset_Normal_v1 and SWaT_Dataset_Attack_v0 files with expected row and label counts."
            if not all_required_match
            else "Files match the standard SWaT physical/historian protocol and are ready for Phase 2 preprocessing validation."
        ),
    }


def _write_class_distribution(path: Path, profiles: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "count", "percent"])
        for profile in profiles:
            total = sum(profile["label_counts"].values())
            for label, count in sorted(profile["label_counts"].items()):
                percent = (count / total * 100.0) if total else 0.0
                writer.writerow([profile["file"], label, count, f"{percent:.6f}"])


def _write_file_summary(path: Path, profiles: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "size_bytes",
                "data_rows",
                "column_count",
                "label_column",
                "timestamp_column",
                "first_timestamp",
                "last_timestamp",
                "matches_standard_expectation",
                "standard_mismatch_reasons",
            ]
        )
        for profile in profiles:
            writer.writerow(
                [
                    profile["file"],
                    profile["size_bytes"],
                    profile["data_rows"],
                    profile["column_count"],
                    profile["label_column"],
                    profile["timestamp_column"],
                    profile["first_timestamp"],
                    profile["last_timestamp"],
                    profile["matches_standard_expectation"],
                    " | ".join(profile["standard_mismatch_reasons"]),
                ]
            )


def _maybe_write_class_distribution_plot(path: Path, profiles: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = []
    counts = []
    colors = []
    for profile in profiles:
        for label, count in sorted(profile["label_counts"].items()):
            labels.append(f"{profile['file']}\n{label}")
            counts.append(count)
            colors.append("#2563eb" if label.lower() == "normal" else "#dc2626")

    if not labels:
        return

    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, counts, color=colors)
    plt.ylabel("Rows")
    plt.title("SWaT Raw CSV Label Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
