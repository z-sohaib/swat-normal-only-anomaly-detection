from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tomllib


REQUIRED_PACKAGES = ["torch", "pandas", "sklearn", "imblearn", "numpy", "tqdm"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check thesis experiment setup.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/binary.toml",
        ],
    )
    args = parser.parse_args()

    ok = True
    print("Dependency check")
    for package in REQUIRED_PACKAGES:
        installed = importlib.util.find_spec(package) is not None
        print(f"  {package:<10} {'OK' if installed else 'MISSING'}")
        ok = ok and installed

    print("\nDataset path check")
    for config_path in args.configs:
        path = Path(config_path)
        if not path.exists():
            print(f"  {config_path}: CONFIG MISSING")
            ok = False
            continue
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        data = raw["data"]
        expected_paths = [
            data.get("train_path"),
            data.get("test_path"),
            data.get("normal_path"),
            data.get("attack_path"),
        ]
        print(f"  {config_path}")
        for expected in [p for p in expected_paths if p]:
            exists = Path(expected).exists()
            print(f"    {expected}: {'OK' if exists else 'MISSING'}")
            ok = ok and exists

    if ok:
        print("\nSetup looks ready.")
        return 0
    print("\nSetup is incomplete. Install dependencies and place dataset files as shown above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
