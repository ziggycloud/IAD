from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.testc_data import (  # noqa: E402
    discover_data_root,
    prepare_testc,
)


def parse_args() -> argparse.Namespace:
    data_root = discover_data_root(ROOT)
    parser = argparse.ArgumentParser(
        description="Build deterministic 50-seen/50-unseen Test_C with ground truth."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=data_root / "Real-IAD_Variety",
    )
    parser.add_argument(
        "--competition-train",
        type=Path,
        default=data_root / "competition" / "Train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root / "competition" / "Test_C",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "testc_protocol.json",
    )
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_testc(
        source_root=args.source_root,
        competition_train=args.competition_train,
        output_root=args.output,
        protocol_path=args.protocol,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
