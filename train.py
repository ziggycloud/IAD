from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import load_config, materialize_paths
from realiad_dinomaly2.runtime import atomic_write_json, utc_now
from realiad_dinomaly2.train_engine import probe_batch, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Dinomaly2 on Real-IAD Variety with resume support."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rtx3060ti_strict_upstream.yaml"),
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, never, or a checkpoint path",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run the real CUDA batch-size probe and exit before training.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = materialize_paths(load_config(args.config, args.set))
    try:
        if args.probe_only:
            probe_batch(config)
        else:
            train(config, resume=args.resume)
    except FileExistsError:
        # A fresh-run guard must not overwrite the state of the existing run.
        raise
    except Exception as exc:
        # In torchrun mode only rank 0 owns shared logs/state/checkpoints.
        if int(os.environ.get("RANK", "0")) != 0:
            raise
        state_path = (
            Path(config["experiment"]["output_dir"]) / "run_state.json"
        )
        state: dict[str, object] = {}
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
        state.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "last_error": repr(exc),
                "traceback": traceback.format_exc(),
                "next_action": (
                    "读取 logs/train.log 与 traceback；修复后从 last.pt 续跑"
                ),
            }
        )
        atomic_write_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
