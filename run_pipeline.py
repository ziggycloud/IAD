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

    # Known local prefixes plus the same layout relative to the repository.
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
        envs = json.loads(result.stdout).get("envs", [])
        for raw_prefix in envs:
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

    if os.environ.get("IAD_PIPELINE_REEXEC") == "1":
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
        os.environ["IAD_PIPELINE_REEXEC"] = "1"
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    raise RuntimeError(
        "A complete IAD Python environment was not found. Set IAD_PYTHON to "
        "its python executable, or run setup_env.ps1 first. Missing modules: "
        + ", ".join(missing)
    )


_ensure_iad_python()
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import load_config, materialize_paths
from realiad_dinomaly2.eval_engine import evaluate
from realiad_dinomaly2.runtime import atomic_write_json, utc_now
from realiad_dinomaly2.train_engine import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-click Dinomaly2 training and paper-protocol evaluation. "
            "Training resumes from last.pt by default."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "rtx3060ti_strict_upstream.yaml"),
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, never, or a training checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="auto or the checkpoint path used for evaluation.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override shared by training and evaluation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate an existing final checkpoint.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only and do not start evaluation.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow diagnostic evaluation of an incomplete checkpoint.",
    )
    parser.add_argument(
        "--active-timeout-seconds",
        type=int,
        default=300,
        help=(
            "Refuse to start training when run_state.json reports a training "
            "update newer than this many seconds."
        ),
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
    if state.get("status") != "training":
        return

    updated_at = _parse_timestamp(state.get("updated_at"))
    if updated_at is None:
        return
    age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    if age_seconds <= timeout_seconds:
        raise RuntimeError(
            "run_state.json was updated "
            f"{max(0, age_seconds):.0f}s ago and reports active training. "
            "Do not start a second process for the same output directory."
        )


def _write_pipeline_state(
    output_dir: Path,
    *,
    status: str,
    config_path: str,
    next_action: str,
    last_error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": utc_now(),
        "config": config_path,
        "next_action": next_action,
    }
    if last_error:
        payload["last_error"] = last_error
    atomic_write_json(output_dir / "pipeline_state.json", payload)


def main() -> int:
    args = parse_args()
    if args.skip_train and args.skip_eval:
        raise ValueError("--skip-train and --skip-eval cannot be used together")

    config = materialize_paths(load_config(args.config, args.set))
    output_dir = Path(config["experiment"]["output_dir"])
    config_path = str(Path(args.config).resolve())

    if not args.skip_train:
        _guard_against_active_training(
            output_dir,
            args.active_timeout_seconds,
        )

    try:
        if not args.skip_train:
            _write_pipeline_state(
                output_dir,
                status="training",
                config_path=config_path,
                next_action="Wait for training to finish; rerun this command to resume.",
            )
            train(config, resume=args.resume)

        if args.skip_eval:
            _write_pipeline_state(
                output_dir,
                status="trained",
                config_path=config_path,
                next_action="Run run_pipeline.py --skip-train to evaluate.",
            )
            return 0

        _write_pipeline_state(
            output_dir,
            status="evaluating",
            config_path=config_path,
            next_action="Wait for the resumable paper-protocol evaluation.",
        )
        evaluate(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
        )
        _write_pipeline_state(
            output_dir,
            status="complete",
            config_path=config_path,
            next_action="Read reports/evaluation_report.md and evaluation/latest.json.",
        )
    except KeyboardInterrupt:
        _write_pipeline_state(
            output_dir,
            status="interrupted",
            config_path=config_path,
            next_action="Rerun the same command to resume.",
            last_error="KeyboardInterrupt",
        )
        return 130
    except Exception as exc:
        _write_pipeline_state(
            output_dir,
            status="failed",
            config_path=config_path,
            next_action=(
                "Inspect run_state.json, logs, and evaluation/last_error.json; "
                "then rerun the same command."
            ),
            last_error=f"{exc!r}\n{traceback.format_exc()}",
        )
        raise
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
