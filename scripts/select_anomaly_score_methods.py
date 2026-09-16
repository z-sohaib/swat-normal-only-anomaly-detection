from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Select strong SWaT anomaly score-method results.")
    parser.add_argument("--input", default="runs/anomaly_score_method_sweep.csv")
    parser.add_argument("--output", default="runs/anomaly_selected_score_methods.csv")
    parser.add_argument("--min-event-recall", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input))
    feasible = [row for row in rows if _float(row["event_recall"]) >= args.min_event_recall]
    ranked = sorted(
        feasible or rows,
        key=lambda row: (
            _float(row["false_alarm_segment_count"]),
            _float(row["false_alarm_windows"]),
            -_float(row["event_recall"]),
            -_float(row["attack_precision"]),
            _float(row["mean_detection_delay_seconds"], default=10**9),
            -_float(row["macro_f1"]),
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, ranked[: args.top_k])
    print(f"Wrote {output}")
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
