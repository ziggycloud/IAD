from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.bootstrap import ensure_iad_runtime  # noqa: E402


ensure_iad_runtime(ROOT, Path(__file__), sys.argv[1:])

from realiad_dinomaly2.clip_fusion import fit_clip_fusion  # noqa: E402
from realiad_dinomaly2.competition_data import (  # noqa: E402
    scan_competition_split,
)
from realiad_dinomaly2.competition_submission import (  # noqa: E402
    generate_competition_submission,
    resolve_competition_checkpoint,
)
from realiad_dinomaly2.config import (  # noqa: E402
    dump_resolved_config,
    load_config,
    materialize_paths,
)
from realiad_dinomaly2.runtime import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    utc_now,
)
from realiad_dinomaly2.normal_prior import fit_normal_prior  # noqa: E402
from realiad_dinomaly2.train_engine import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train category-generalized Dinomaly on competition Train, infer "
            "Test_A, and build a validated submission.zip."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "competition.yaml"),
        help="Competition YAML configuration path.",
    )
    parser.add_argument(
        "--resume", default="auto", help="auto, never, or a checkpoint path"
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto or the checkpoint used to generate the submission",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Audit Train/Test_A layout without loading the model.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a diagnostic ZIP from an incomplete checkpoint.",
    )
    parser.add_argument(
        "--active-timeout-seconds",
        type=int,
        default=300,
        help="Reject duplicate training when run_state.json is fresh.",
    )
    return parser.parse_args()


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _guard_against_active_training(
    output_dir: Path,
    timeout_seconds: int,
) -> None:
    if timeout_seconds < 0:
        raise ValueError("--active-timeout-seconds must be >= 0")
    state_path = output_dir / "run_state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    updated_at = _parse_timestamp(state.get("updated_at"))
    if state.get("status") != "training" or updated_at is None:
        return
    age = (datetime.now(timezone.utc) - updated_at).total_seconds()
    if age <= timeout_seconds:
        raise RuntimeError(
            f"run_state.json reports active training ({max(0, age):.0f}s old); "
            "refusing to start a duplicate trainer"
        )


def _write_state(
    output_dir: Path,
    *,
    status: str,
    next_action: str,
    **extra: Any,
) -> None:
    payload = {
        "status": status,
        "updated_at": utc_now(),
        "entrypoint": str(Path(__file__).resolve()),
        "arguments": list(sys.argv[1:]),
        "next_action": next_action,
        **extra,
    }
    atomic_write_json(output_dir / "competition_pipeline_state.json", payload)
    append_jsonl(
        output_dir / "logs" / "competition_pipeline.jsonl",
        {"event": "pipeline_state", **payload},
    )


def _audit_data(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    train_manifest = scan_competition_split(
        Path(dataset["train_dir"]),
        requested=dataset["categories"],
        limit=dataset.get("category_limit"),
    )
    test_manifest = scan_competition_split(
        Path(dataset["test_dir"]),
        requested=dataset["categories"],
        limit=dataset.get("category_limit"),
    )
    if train_manifest.categories != test_manifest.categories:
        only_train = sorted(
            set(train_manifest.categories) - set(test_manifest.categories)
        )
        only_test = sorted(
            set(test_manifest.categories) - set(train_manifest.categories)
        )
        raise ValueError(
            "Train/Test_A categories differ: "
            f"only_train={only_train}, only_test={only_test}"
        )
    category_limit = dataset.get("category_limit")
    expected_categories = int(
        category_limit
        if category_limit is not None
        else dataset.get("expected_categories", 50)
    )
    if len(train_manifest.categories) != expected_categories:
        raise ValueError(
            f"Expected {expected_categories} categories, found "
            f"{len(train_manifest.categories)}"
        )
    expected_train = int(
        dataset.get("expected_train_samples_per_category", 20)
    )
    expected_test = int(
        dataset.get("expected_test_samples_per_category", 15)
    )
    train_counts = {
        category: len(train_manifest.views_for_category(category)) // 5
        for category in train_manifest.categories
    }
    test_counts = {
        category: len(test_manifest.views_for_category(category)) // 5
        for category in test_manifest.categories
    }
    bad_train = {
        category: count
        for category, count in train_counts.items()
        if count != expected_train
    }
    bad_test = {
        category: count
        for category, count in test_counts.items()
        if count != expected_test
    }
    if bad_train or bad_test:
        raise ValueError(
            "Unexpected samples per category: "
            f"Train={bad_train}, Test_A={bad_test}"
        )
    return {
        "status": "valid",
        "audited_at": utc_now(),
        "train": train_manifest.summary(),
        "test": test_manifest.summary(),
        "same_categories": True,
        "train_samples_per_category": expected_train,
        "test_samples_per_category": expected_test,
        "category_names": list(train_manifest.categories),
    }


def main() -> int:
    args = parse_args()
    if args.skip_train and args.skip_inference and not args.validate_only:
        raise ValueError("--skip-train and --skip-inference leave no work to do")
    config = materialize_paths(load_config(args.config, args.set))
    if config["dataset"].get("type") != "competition_folders":
        raise ValueError(
            "run_competition_pipeline.py requires dataset.type=competition_folders"
        )
    output_dir = Path(config["experiment"]["output_dir"])
    rank = int(os.environ.get("RANK", "0"))
    is_primary = rank == 0

    try:
        audit = _audit_data(config)
        if is_primary:
            dump_resolved_config(
                config, output_dir / "competition_resolved_config.yaml"
            )
            atomic_write_json(output_dir / "competition_data_audit.json", audit)
            _write_state(
                output_dir,
                status="data_validated",
                next_action=(
                    "Train the model, or use --skip-train with a completed "
                    "checkpoint."
                ),
                audit=str(output_dir / "competition_data_audit.json"),
            )
        if args.validate_only:
            return 0

        if not args.skip_train:
            _guard_against_active_training(
                output_dir, args.active_timeout_seconds
            )
            if is_primary:
                _write_state(
                    output_dir,
                    status="training",
                    next_action="Rerun the same command to resume from last.pt.",
                )
            train(config, resume=args.resume)

        # torchrun workers stop here; rank 0 alone writes the shared package.
        if not is_primary:
            return 0
        checkpoint_path = resolve_competition_checkpoint(
            output_dir,
            args.checkpoint,
        )
        if bool(
            config["evaluation"].get("normal_prior", {}).get("enabled", False)
        ):
            _write_state(
                output_dir,
                status="fitting_normal_prior",
                next_action=(
                    "Fit or strictly validate Train-normal category/view priors."
                ),
                checkpoint=str(checkpoint_path),
            )
            fit_normal_prior(
                config,
                checkpoint_path,
                categories=audit["category_names"],
            )
        if bool(
            config["evaluation"].get("clip_fusion", {}).get("enabled", False)
        ):
            _write_state(
                output_dir,
                status="fitting_clip_fusion",
                next_action=(
                    "Fit or validate Train-normal Dinomaly/OpenCLIP fusion "
                    "scales without reading Test_A."
                ),
                checkpoint=str(checkpoint_path),
            )
            fit_clip_fusion(
                config,
                checkpoint_path,
                categories=audit["category_names"],
            )
        if args.skip_inference:
            _write_state(
                output_dir,
                status="trained",
                next_action=(
                    "Run with --skip-train to infer Test_A and create the ZIP."
                ),
            )
            return 0

        _write_state(
            output_dir,
            status="inferring",
            next_action=(
                "Inference resumes category by category if this process is "
                "interrupted."
            ),
        )
        result = generate_competition_submission(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
        )
        _write_state(
            output_dir,
            status=result["status"],
            next_action="Upload the generated submission.zip manually.",
            zip=result["zip"],
            submission_csv=result["submission_csv"],
            validation=result["validation"],
        )
    except KeyboardInterrupt:
        if is_primary:
            _write_state(
                output_dir,
                status="interrupted",
                next_action="Rerun the same command to resume.",
                last_error="KeyboardInterrupt",
            )
        return 130
    except Exception as exc:
        if is_primary:
            _write_state(
                output_dir,
                status="failed",
                next_action=(
                    "Inspect competition_pipeline_state.json and logs, fix the "
                    "reported issue, then rerun the same command."
                ),
                last_error=f"{exc!r}\n{traceback.format_exc()}",
            )
        raise
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
