from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swat_ids.config import load_config
from swat_ids.train import run as run_supervised
from swat_ids.train_anomaly import run as run_anomaly


STABLE_DROP_COLUMNS = ["AIT201", "P201", "FIT601", "P601", "P602", "P603"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SWaT W&B sweep trial.")
    parser.add_argument("--base-config", required=True, help="Base TOML config to override.")
    parser.add_argument("--mode", required=True, choices=["supervised", "anomaly"])
    parser.add_argument("--project", default="pfe-thesis-swat")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-prefix", default="sweep")
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved TOML without training.")
    args, unknown = parser.parse_known_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. Run `python -m pip install -e .` from projects/swat "
            "or install it explicitly with `python -m pip install wandb`."
        ) from exc

    cli_params = _parse_unknown_args(unknown)
    with wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        config=cli_params,
        tags=["swat", args.mode, "hpo"],
    ) as wb_run:
        sweep_params = dict(wandb.config)
        base_path = Path(args.base_config)
        resolved = _resolved_config(base_path, sweep_params, wb_run.id, args.run_prefix)
        resolved_dir = Path(resolved["experiment"]["output_dir"])
        resolved_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = resolved_dir / "resolved_config.toml"
        resolved_path.write_text(_toml_dumps(resolved), encoding="utf-8")
        wandb.config.update(_flatten_config(resolved), allow_val_change=True)

        if args.dry_run:
            print(f"Dry run wrote {resolved_path}")
            return 0

        config = load_config(resolved_path)
        if args.mode == "supervised":
            run_supervised(config, config_path=resolved_path)
        else:
            run_anomaly(config, config_path=resolved_path)

        metrics_path = resolved_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        flat_metrics = _flatten_metrics(metrics)
        wandb.log(flat_metrics)
        wandb.summary.update(flat_metrics)
        wandb.save(str(metrics_path))
        wandb.save(str(resolved_path))
        model_path = resolved_dir / "model.pt"
        if model_path.exists():
            wandb.save(str(model_path), policy="now")
    return 0


def _resolved_config(base_path: Path, params: dict[str, Any], run_id: str, run_prefix: str) -> dict[str, Any]:
    with base_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    for key, value in params.items():
        if key.startswith("_") or value is None:
            continue
        if key == "data.drop_columns_preset":
            raw.setdefault("data", {})["drop_columns"] = _drop_columns_for_preset(str(value))
            continue
        _set_nested(raw, key, _normalize_value(key, value))

    base_name = str(raw["experiment"]["name"])
    sweep_name = f"{run_prefix}_{base_name}_{run_id}"
    raw["experiment"]["name"] = sweep_name
    raw["experiment"]["output_dir"] = f"runs/wandb/{base_name}/{run_id}"
    return raw


def _drop_columns_for_preset(preset: str) -> list[str]:
    if preset == "stable_45":
        return STABLE_DROP_COLUMNS
    if preset == "all_51":
        return []
    raise ValueError(f"Unsupported data.drop_columns_preset={preset!r}.")


def _set_nested(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        return
    current = raw
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _normalize_value(key: str, value: Any) -> Any:
    if key in {"model.kernel_sizes", "data.drop_columns"}:
        return _as_list(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        parsed = _parse_jsonish(value)
        return parsed
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = _parse_jsonish(value)
        if isinstance(parsed, list):
            return parsed
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [value]


def _parse_unknown_args(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    index = 0
    while index < len(items):
        item = items[index]
        if not item.startswith("--"):
            index += 1
            continue
        payload = item[2:]
        if "=" in payload:
            key, value = payload.split("=", 1)
        else:
            key = payload
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                value = items[index + 1]
                index += 1
            else:
                value = "true"
        params[key] = _parse_jsonish(value)
        index += 1
    return params


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(f"{prefix}.{child_key}" if prefix else child_key, child_value)
        else:
            flattened[prefix] = value

    visit("", raw)
    return flattened


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    attack = metrics.get("per_class", {}).get("attack", {})
    normal = metrics.get("per_class", {}).get("normal", {})
    operational = metrics.get("operational_metrics", {})
    latency = metrics.get("latency", {})
    detected_events = _float_or_none(operational.get("detected_event_count"))
    false_alarm_segments = _float_or_none(operational.get("false_alarm_segment_count"))
    event_precision = None
    if detected_events is not None and false_alarm_segments is not None:
        denominator = detected_events + false_alarm_segments
        event_precision = detected_events / denominator if denominator else 0.0
    event_recall = _float_or_none(operational.get("event_recall"))
    event_f1 = None
    if event_precision is not None and event_recall is not None:
        denominator = event_precision + event_recall
        event_f1 = 2 * event_precision * event_recall / denominator if denominator else 0.0

    return {
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_weighted_f1": metrics.get("weighted_f1"),
        "test_attack_precision": attack.get("precision"),
        "test_attack_recall": attack.get("recall"),
        "test_attack_f1": attack.get("f1"),
        "test_attack_fpr": attack.get("false_positive_rate"),
        "test_normal_precision": normal.get("precision"),
        "test_normal_recall": normal.get("recall"),
        "test_normal_f1": normal.get("f1"),
        "official_event_recall": event_recall,
        "official_event_precision": event_precision,
        "official_event_f1": event_f1,
        "false_alarm_segments": false_alarm_segments,
        "false_alarm_windows": operational.get("false_alarm_windows"),
        "false_alarm_segments_per_hour": operational.get("false_alarm_segments_per_hour"),
        "mean_detection_delay_seconds": operational.get("mean_detection_delay_seconds"),
        "parameter_count": metrics.get("parameter_count"),
        "seconds_per_sample": latency.get("seconds_per_sample"),
        "samples_per_second": latency.get("samples_per_second"),
        "best_selected_score": metrics.get("best_selected_score"),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _toml_dumps(raw: dict[str, Any]) -> str:
    sections = []
    for section, values in raw.items():
        sections.append(f"[{section}]")
        for key, value in values.items():
            sections.append(f"{key} = {_toml_value(value)}")
        sections.append("")
    return "\n".join(sections)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        print(f"Interrupted after {elapsed:.1f}s", file=sys.stderr)
        raise
