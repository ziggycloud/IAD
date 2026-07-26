from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import load_config, materialize_paths
from realiad_dinomaly2.eval_engine import evaluate
from realiad_dinomaly2.runtime import atomic_write_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Dinomaly2 checkpoint with paper metrics."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rtx3060ti_strict_upstream.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto or a checkpoint path",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow diagnostic evaluation before training.total_steps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = materialize_paths(load_config(args.config, args.set))
    try:
        evaluate(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
        )
    except Exception as exc:
        error_path = (
            Path(config["experiment"]["output_dir"])
            / "evaluation"
            / "last_error.json"
        )
        atomic_write_json(
            error_path,
            {
                "status": "failed",
                "updated_at": utc_now(),
                "checkpoint": args.checkpoint,
                "allow_partial": args.allow_partial,
                "last_error": repr(exc),
                "traceback": traceback.format_exc(),
                "next_action": (
                    "检查 checkpoint/config；修复后重新运行 evaluate.ps1，"
                    "已完成类别会按签名续用"
                ),
            },
        )
        run_state_path = (
            Path(config["experiment"]["output_dir"]) / "run_state.json"
        )
        run_state: dict[str, object] = {}
        if run_state_path.is_file():
            try:
                run_state = json.loads(
                    run_state_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                run_state = {}
        run_state.update(
            {
                "status": "evaluation_failed",
                "updated_at": utc_now(),
                "last_error": repr(exc),
                "next_action": (
                    "读取 evaluation/last_error.json；修复后重新运行 "
                    "evaluate.ps1 续评"
                ),
            }
        )
        atomic_write_json(run_state_path, run_state)
        raise
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
