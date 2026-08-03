from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _missing_runtime_modules() -> list[str]:
    required = ("torch", "sklearn", "yaml", "cv2")
    return [name for name in required if importlib.util.find_spec(name) is None]


def _python_in_prefix(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def _candidate_iad_pythons() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("IAD_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    project_parent = ROOT.parent
    prefixes = [
        project_parent / "IAD" / "data" / ".conda" / "iad",
        project_parent / "IAD" / "data" / ".conda" / "realiad-variety-py311",
        project_parent / "IAD" / "data.conda" / "realiad-variety-py311",
        Path(r"J:\project\IAD\data\.conda\iad"),
        Path(r"J:\project\IAD\data\.conda\realiad-variety-py311"),
        Path(r"J:\project\IAD\data.conda\realiad-variety-py311"),
    ]
    candidates.extend(_python_in_prefix(prefix) for prefix in prefixes)
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
        for raw_prefix in json.loads(result.stdout).get("envs", []):
            prefix = Path(raw_prefix)
            if prefix.name.casefold() in {"iad", "realiad-variety-py311"}:
                candidates.append(_python_in_prefix(prefix))
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _candidate_has_runtime(candidate: Path) -> bool:
    probe = (
        "import importlib.util,sys;"
        "mods=('torch','sklearn','yaml','cv2');"
        "sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
    )
    try:
        result = subprocess.run(
            [str(candidate), "-c", probe],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _ensure_iad_python() -> None:
    missing = _missing_runtime_modules()
    if not missing:
        return
    if os.environ.get("IAD_UNSEEN_PIPELINE_REEXEC") == "1":
        raise RuntimeError(
            f"Selected IAD Python is missing dependencies: {', '.join(missing)}"
        )
    current = Path(sys.executable).resolve()
    for candidate in _candidate_iad_pythons():
        if not candidate.is_file() or candidate.resolve() == current:
            continue
        if not _candidate_has_runtime(candidate):
            continue
        print(
            f"Current Python is missing {', '.join(missing)}; "
            f"re-launching with {candidate}",
            file=sys.stderr,
            flush=True,
        )
        os.environ["IAD_UNSEEN_PIPELINE_REEXEC"] = "1"
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
        )
    raise RuntimeError(
        "A complete IAD Python environment was not found. Set IAD_PYTHON to "
        "its python executable. Missing modules: " + ", ".join(missing)
    )


_ensure_iad_python()
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import (  # noqa: E402
    dump_resolved_config,
    load_config,
    materialize_paths,
)
from realiad_dinomaly2.data import discover_categories  # noqa: E402
from realiad_dinomaly2.eval_engine import evaluate  # noqa: E402
from realiad_dinomaly2.latency import (  # noqa: E402
    benchmark_single_frame_latency,
)
from realiad_dinomaly2.runtime import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    utc_now,
)
from realiad_dinomaly2.train_engine import train  # noqa: E402
from realiad_dinomaly2.unseen_protocol import (  # noqa: E402
    compute_unseen_evaluation_scores,
    generate_category_split,
    load_category_split,
    render_unseen_evaluation_report,
    save_category_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-click category-generalized Dinomaly training and deterministic "
            "50-seen/50-unseen evaluation."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "unseen_2x40gb.yaml"),
        help="YAML configuration path.",
    )
    parser.add_argument("--resume", default="auto", help="auto, never, or checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto or the checkpoint path used for both evaluations.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override protocol.seed; a persisted split can never silently change.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override.",
    )
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--active-timeout-seconds",
        type=int,
        default=300,
        help="Reject a second trainer when run_state.json is fresh.",
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


def _guard_against_active_training(output_dir: Path, timeout_seconds: int) -> None:
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
            f"run_state.json reports training and was updated {max(0, age):.0f}s "
            "ago; refusing to start a duplicate trainer."
        )


def _write_state(
    output_dir: Path,
    log_path: Path,
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
    atomic_write_json(output_dir / "unseen_pipeline_state.json", payload)
    append_jsonl(log_path, {"event": "pipeline_state", **payload})


def _materialize_protocol_split(
    config: dict[str, Any],
    split_seed: int,
    split_path: Path,
    *,
    is_primary: bool,
):
    categories = discover_categories(
        Path(config["dataset"]["json_dir"]),
        requested="all",
        limit=None,
    )
    generated = generate_category_split(categories, split_seed)
    if split_path.is_file():
        return load_category_split(
            split_path,
            expected_categories=categories,
            expected_seed=split_seed,
        )
    if is_primary:
        save_category_split(generated, split_path)
    return generated


def _metric_rows(evaluation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(evaluation_payload["evaluation_dir"]) / "per_category"
    rows = []
    for path in sorted(directory.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    expected = int(evaluation_payload["categories"])
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} per-category results in {directory}, got {len(rows)}"
        )
    return rows


def _checkpoint_path(output_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "auto":
        path = output_dir / "checkpoints" / "final_model.pt"
    else:
        path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def main() -> int:
    args = parse_args()
    if args.skip_train and args.skip_eval:
        raise ValueError("--skip-train and --skip-eval cannot be used together")

    config = materialize_paths(load_config(args.config, args.set))
    protocol_config = config.get("protocol", {})
    expected_counts = {
        "seen_categories": 50,
        "unseen_categories": 50,
        "unused_categories": 60,
    }
    mismatched_counts = {
        name: protocol_config.get(name)
        for name, expected in expected_counts.items()
        if int(protocol_config.get(name, expected)) != expected
    }
    if mismatched_counts:
        raise ValueError(
            "This scoring protocol is fixed at 50 seen / 50 unseen / 60 "
            f"unused; invalid overrides: {mismatched_counts}"
        )
    latency_threshold = float(
        config.get("latency", {}).get("threshold_seconds", 1.0)
    )
    if latency_threshold != 1.0:
        raise ValueError(
            "The official single-frame latency threshold is fixed at 1.0 second."
        )
    output_dir = Path(config["experiment"]["output_dir"])
    log_path = output_dir / "logs" / "unseen_pipeline.jsonl"
    rank = int(os.environ.get("RANK", "0"))
    is_primary = rank == 0
    split_seed = int(
        args.split_seed
        if args.split_seed is not None
        else config.get("protocol", {}).get(
            "seed", config["experiment"]["seed"]
        )
    )
    split_path = output_dir / "protocol" / "category_split.json"

    try:
        split = _materialize_protocol_split(
            config,
            split_seed,
            split_path,
            is_primary=is_primary,
        )
        config["dataset"]["categories"] = list(split.seen)
        config["dataset"]["category_limit"] = None
        if is_primary:
            dump_resolved_config(
                config,
                output_dir / "protocol" / "resolved_seen_config.yaml",
            )
            _write_state(
                output_dir,
                log_path,
                status="split_ready",
                next_action="Train only on the 50 seen categories.",
                split=str(split_path),
                split_seed=split_seed,
                seen_categories=len(split.seen),
                unseen_categories=len(split.unseen),
                unused_categories=len(split.unused),
            )
        if args.split_only:
            return 0

        if not args.skip_train:
            _guard_against_active_training(
                output_dir, args.active_timeout_seconds
            )
            if is_primary:
                _write_state(
                    output_dir,
                    log_path,
                    status="training",
                    next_action="Rerun the same command to resume from last.pt.",
                    split=str(split_path),
                )
            train(config, resume=args.resume)

        # torchrun workers finish after the synchronized training call; only
        # rank 0 performs the two resumable evaluations and writes the report.
        if not is_primary:
            return 0
        if args.skip_eval:
            _write_state(
                output_dir,
                log_path,
                status="trained",
                next_action="Rerun with --skip-train to evaluate seen and unseen.",
                split=str(split_path),
            )
            return 0

        _write_state(
            output_dir,
            log_path,
            status="evaluating_seen",
            next_action="Seen evaluation is resumable per category.",
            split=str(split_path),
        )
        seen_result = evaluate(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
            categories_override=list(split.seen),
            split_name="seen",
            publish_root_report=False,
        )
        _write_state(
            output_dir,
            log_path,
            status="evaluating_unseen",
            next_action="Unseen evaluation is resumable per category.",
            seen_metrics=str(
                Path(seen_result["evaluation_dir"]) / "metrics.json"
            ),
        )
        unseen_result = evaluate(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
            categories_override=list(split.unseen),
            split_name="unseen",
            publish_root_report=False,
        )

        scoring = compute_unseen_evaluation_scores(
            _metric_rows(seen_result),
            _metric_rows(unseen_result),
            split=split,
        )
        checkpoint_path = _checkpoint_path(output_dir, args.checkpoint)
        latency_summary = None
        if not args.skip_latency:
            _write_state(
                output_dir,
                log_path,
                status="benchmarking_latency",
                next_action="Run the configured single-frame latency benchmark.",
            )
            latency_summary = benchmark_single_frame_latency(
                config,
                checkpoint_path,
                category=split.unseen[0],
            )

        partial = (
            seen_result["status"] != "complete"
            or unseen_result["status"] != "complete"
        )
        latency_valid = (
            None if latency_summary is None else bool(latency_summary["valid"])
        )
        valid_submission = False if partial else latency_valid
        result_payload = {
            "status": "partial_diagnostic" if partial else "complete",
            "completed_at": utc_now(),
            "protocol": split.to_dict(),
            "checkpoint": str(checkpoint_path),
            "seen_evaluation": seen_result,
            "unseen_evaluation": unseen_result,
            "score": scoring,
            "latency": latency_summary,
            "valid_submission": valid_submission,
            "validity_note": (
                "Partial checkpoints are diagnostic only."
                if partial
                else (
                    "Latency was explicitly skipped; submission validity is unknown."
                    if latency_summary is None
                    else (
                        "Score and latency requirements completed."
                        if latency_valid
                        else "P95/mean single-frame latency exceeds the limit."
                    )
                )
            ),
        }
        result_dir = output_dir / "unseen_evaluation"
        atomic_write_json(result_dir / "metrics_and_score.json", result_payload)
        report = render_unseen_evaluation_report(
            scoring,
            split,
            latency_summary=latency_summary,
        )
        if partial:
            report_status = (
                "> **诊断结果：checkpoint 尚未完成全部训练，不能作为正式提交。**"
            )
        elif latency_summary is None:
            report_status = (
                "> **指标计算完成，但单帧延迟被显式跳过，提交有效性尚未判定。**"
            )
        elif not latency_valid:
            report_status = (
                "> **指标计算完成，但单帧延迟要求未通过，按规则本次结果无效。**"
            )
        else:
            report_status = "> **正式评估与单帧延迟检查均已完成并通过。**"
        report = report_status + "\n\n" + report
        (result_dir / "evaluation_report.md").write_text(
            report, encoding="utf-8", newline="\n"
        )
        reports_dir = ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "unseen_evaluation_report.md").write_text(
            report, encoding="utf-8", newline="\n"
        )
        final_status = (
            "partial_diagnostic"
            if partial
            else (
                "complete_latency_not_checked"
                if latency_valid is None
                else ("complete" if latency_valid else "invalid_latency")
            )
        )
        _write_state(
            output_dir,
            log_path,
            status=final_status,
            next_action=(
                "Read unseen_evaluation/evaluation_report.md and metrics_and_score.json."
            ),
            total_score=scoring["total_score"],
            valid_submission=valid_submission,
            report=str(result_dir / "evaluation_report.md"),
        )
        return 0 if final_status in {"complete", "complete_latency_not_checked"} else 2
    except KeyboardInterrupt:
        if is_primary:
            _write_state(
                output_dir,
                log_path,
                status="interrupted",
                next_action="Rerun the identical command to resume.",
                last_error="KeyboardInterrupt",
            )
        return 130
    except Exception as exc:
        if is_primary:
            _write_state(
                output_dir,
                log_path,
                status="failed",
                next_action=(
                    "Read unseen_pipeline_state.json and logs/unseen_pipeline.jsonl, "
                    "fix the error, then rerun the identical command."
                ),
                last_error=f"{exc!r}\n{traceback.format_exc()}",
            )
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
