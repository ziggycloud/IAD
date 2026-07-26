from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.cache_prepare import prepare_train_cache
from realiad_dinomaly2.config import load_config, materialize_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable 1024x1024 cache for Real-IAD Variety train "
            "images listed by the official manifests."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rtx3060ti.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional deterministic prefix for a smoke test; omit for all train images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = materialize_paths(load_config(args.config, args.set))
    state = prepare_train_cache(config, max_images=args.max_images)
    print(
        "Train cache ready: "
        f"{state['processed']} processed, {state['written']} written, "
        f"{state['skipped_valid']} already valid; {state['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
