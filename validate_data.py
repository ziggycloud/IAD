from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import load_config, materialize_paths
from realiad_dinomaly2.validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rtx3060ti_strict_upstream.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=("metadata", "sample", "full"),
        default="sample",
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = materialize_paths(load_config(args.config, args.set))
    output_dir = Path(config["experiment"]["output_dir"])
    result = validate_dataset(
        config,
        mode=args.mode,
        output_path=output_dir / "data_validation.json",
    )
    print(
        f"Data validation OK: {result['categories']} categories, "
        f"{result['totals']['train_images']} train views, "
        f"{result['totals']['test_images']} test views."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
