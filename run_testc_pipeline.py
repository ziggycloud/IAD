from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.bootstrap import ensure_iad_runtime  # noqa: E402


ensure_iad_runtime(ROOT, Path(__file__), sys.argv[1:])

import yaml  # noqa: E402
import torch  # noqa: E402

from realiad_dinomaly2.competition_submission import (  # noqa: E402
    generate_competition_submission,
    resolve_competition_checkpoint,
)
from realiad_dinomaly2.config import (  # noqa: E402
    config_fingerprint,
    dump_resolved_config,
    load_config,
    materialize_paths,
)
from realiad_dinomaly2.latency import benchmark_single_frame_latency  # noqa: E402
from realiad_dinomaly2.normal_prior import fit_normal_prior  # noqa: E402
from realiad_dinomaly2.runtime import append_jsonl, atomic_write_json, utc_now  # noqa: E402
from realiad_dinomaly2.testc_data import (  # noqa: E402
    audit_competition_train,
    audit_testc,
    canonical_digest,
    discover_data_root,
    file_sha256,
    load_testc_protocol,
    prepare_testc,
)
from realiad_dinomaly2.testc_evaluation import evaluate_testc_submission  # noqa: E402
from realiad_dinomaly2.train_engine import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on competition Train and evaluate the exact Test_B categories on Test_C."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "testc.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--export-testc-predictions-zip",
        action="store_true",
        help=(
            "Optionally archive Test_C predictions for debugging. The archive "
            "is explicitly marked as not competition-submittable."
        ),
    )
    return parser.parse_args()


_TRAINING_DATASET_KEYS = (
    "type", "json_dir", "image_dir", "train_image_dir", "train_dir",
    "categories", "category_limit", "image_size", "crop_size", "train_mode",
    "image_label_policy", "missing_anomaly_mask_policy", "mask_resize_semantics",
)


def _restore_checkpoint_training_config(
    config: dict[str, Any], checkpoint_path: Path, explicit: Path | None
) -> tuple[dict[str, Any], Path]:
    """Use the original training YAML, preserving current Test_C inference fields."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fingerprint = checkpoint.get("config_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("checkpoint has no config_fingerprint")
    output_dir = Path(config["experiment"]["output_dir"])
    candidates = (
        [explicit.expanduser().resolve()] if explicit is not None else
        sorted(output_dir.glob("testc_runs/*/resolved_config.yaml"), reverse=True)
        + [output_dir / "resolved_config.yaml"]
    )
    for path in candidates:
        if not path.is_file():
            continue
        candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict) or config_fingerprint(candidate) != fingerprint:
            continue
        restored = copy.deepcopy(config)
        restored["model"] = copy.deepcopy(candidate["model"])
        restored["training"] = copy.deepcopy(candidate["training"])
        restored["experiment"]["seed"] = candidate["experiment"]["seed"]
        for key in _TRAINING_DATASET_KEYS:
            if key in candidate["dataset"]:
                restored["dataset"][key] = copy.deepcopy(candidate["dataset"][key])
        if config_fingerprint(restored) != fingerprint:
            raise RuntimeError(f"restored config from {path} still mismatches checkpoint")
        return restored, path
    raise ValueError(
        "No resolved training YAML matches this checkpoint; pass "
        "--checkpoint-config /path/to/resolved_config.yaml"
    )


def _load_wrapper(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "base_config" not in payload or "testc" not in payload:
        raise ValueError("Test_C config requires base_config and testc mappings")
    payload["_path"] = resolved
    return payload


def _resolve_data_root(wrapper: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = wrapper["testc"].get("data_root", "auto")
    if configured != "auto":
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()
    return discover_data_root(ROOT)


def _build_config(
    wrapper: dict[str, Any], data_root: Path, protocol: dict[str, Any], cli: list[str]
) -> dict[str, Any]:
    wrapper_path = Path(wrapper["_path"])
    base_path = Path(str(wrapper["base_config"]))
    if not base_path.is_absolute():
        base_path = wrapper_path.parent / base_path
    experiment = wrapper.get("experiment", {})
    defaults = [
        f"experiment.name={json.dumps(str(experiment.get('name', 'testc')))}",
        f"experiment.output_dir={json.dumps(str(experiment.get('output_dir', 'outputs/testc')))}",
        f"dataset.train_dir={json.dumps(str(data_root / 'competition' / 'Train'))}",
        f"dataset.test_dir={json.dumps(str(data_root / 'competition' / 'Test_C' / 'images'))}",
        f"dataset.categories={json.dumps(protocol['seen_categories'])}",
        "dataset.category_limit=null",
        f"dataset.test_categories={json.dumps(protocol['categories'])}",
        "dataset.test_category_limit=null",
        "dataset.require_same_categories=false",
        "dataset.expected_categories=50",
        "dataset.expected_test_categories=100",
        "dataset.expected_train_samples_per_category=20",
        "dataset.expected_test_samples_per_category=20",
    ]
    config = materialize_paths(load_config(base_path, [*defaults, *cli]))
    config["testc"] = copy.deepcopy(wrapper["testc"])
    config["testc"]["root"] = str(data_root / "competition" / "Test_C")
    config["testc"]["source_root"] = str(data_root / "Real-IAD_Variety")
    config["testc"]["protocol_file"] = str(ROOT / "configs" / "testc_protocol.json")
    config["latency"] = copy.deepcopy(wrapper.get("latency", {}))
    return config


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _run_manifest(
    config: dict[str, Any],
    protocol: dict[str, Any],
    testc_manifest: Path,
    *,
    write: bool,
) -> tuple[Path, dict[str, Any]]:
    git_sha = _git_value("rev-parse", "HEAD")
    config_digest = canonical_digest(config)
    signature = canonical_digest(
        {
            "git_sha": git_sha,
            "config_sha256": config_digest,
            "protocol_sha256": protocol["protocol_sha256"],
            "testc_manifest_sha256": file_sha256(testc_manifest),
        }
    )
    output_dir = Path(config["experiment"]["output_dir"])
    run_dir = output_dir / "testc_runs" / signature[:12]
    payload = {
        "run_id": signature[:12],
        "signature": signature,
        "created_at": utc_now(),
        "git": {
            "commit": git_sha,
            "branch": _git_value("branch", "--show-current"),
            "dirty": bool(_git_value("status", "--porcelain")),
            "status": _git_value("status", "--short"),
        },
        "arguments": list(sys.argv[1:]),
        "config_sha256": config_digest,
        "protocol_sha256": protocol["protocol_sha256"],
        "testc_manifest": str(testc_manifest),
        "testc_manifest_sha256": file_sha256(testc_manifest),
        "environment": _environment(),
        "status": "initialized",
    }
    if write:
        atomic_write_json(run_dir / "run_manifest.json", payload)
        dump_resolved_config(config, run_dir / "resolved_config.yaml")
    return run_dir, payload


def _update_manifest(run_dir: Path, payload: dict[str, Any], **changes: Any) -> None:
    payload.update(changes)
    payload["updated_at"] = utc_now()
    atomic_write_json(run_dir / "run_manifest.json", payload)
    append_jsonl(run_dir / "pipeline_events.jsonl", {"timestamp": utc_now(), **changes})


def _inference_rendezvous_path(output_dir: Path) -> Path:
    return output_dir / "testc_inference_rendezvous.json"


def _set_inference_rendezvous(
    path: Path, run_id: str, status: str, **details: Any
) -> None:
    atomic_write_json(
        path,
        {"run_id": run_id, "status": status, "updated_at": utc_now(), **details},
    )


def _wait_for_inference_rendezvous(
    path: Path, run_id: str, timeout_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = "missing"
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if payload.get("run_id") == run_id:
                last_status = str(payload.get("status", "unknown"))
                if last_status == "ready":
                    return
                if last_status == "failed":
                    raise RuntimeError(
                        "Test_C primary rank failed before distributed inference: "
                        f"{payload.get('error', 'see run_manifest.json')}"
                    )
        time.sleep(1)
    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for Test_C primary rank "
        f"to prepare inference artifacts ({last_status})"
    )


def main() -> int:
    args = parse_args()
    wrapper = _load_wrapper(args.config)
    data_root = _resolve_data_root(wrapper, args.data_root)
    protocol_path = ROOT / "configs" / "testc_protocol.json"
    protocol = load_testc_protocol(protocol_path)
    testc_root = data_root / "competition" / "Test_C"
    source_root = data_root / "Real-IAD_Variety"
    competition_train = data_root / "competition" / "Train"
    if args.prepare_only:
        result = prepare_testc(
            source_root=source_root,
            competition_train=competition_train,
            output_root=testc_root,
            protocol_path=protocol_path,
            seed=int(wrapper["testc"].get("sampling_seed", 20260909)),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    audit = audit_testc(testc_root, protocol_path)
    train_audit = audit_competition_train(competition_train, source_root, protocol_path)
    config = _build_config(wrapper, data_root, protocol, args.set)
    if args.skip_train:
        checkpoint_path = resolve_competition_checkpoint(
            Path(config["experiment"]["output_dir"]), args.checkpoint
        )
        config, restored_path = _restore_checkpoint_training_config(
            config, checkpoint_path, args.checkpoint_config
        )
        print(f"Restored checkpoint training semantics from {restored_path}")
    rank = int(os.environ.get("RANK", "0"))
    is_primary = rank == 0
    run_dir, run_manifest = _run_manifest(
        config, protocol, testc_root / "manifest.json", write=is_primary
    )
    if is_primary:
        _update_manifest(
            run_dir,
            run_manifest,
            status="data_validated",
            testc_audit=audit,
            train_audit=train_audit,
        )
    output_dir = Path(config["experiment"]["output_dir"])
    rendezvous_path = _inference_rendezvous_path(output_dir)
    rendezvous_timeout = int(
        config.get("testc", {}).get("inference_rendezvous_timeout_seconds", 7200)
    )
    if is_primary and not args.skip_eval:
        _set_inference_rendezvous(
            rendezvous_path, run_manifest["run_id"], "preparing"
        )
    try:
        if not args.skip_train:
            if is_primary:
                _update_manifest(run_dir, run_manifest, status="training")
            train(config, resume=args.resume)
            # The zero-shot trainer is deliberately single-process.  The
            # preceding Dinomaly phase has already torn down its DDP group;
            # starting this trainer on every torchrun rank would make every
            # worker resolve runtime.device (normally cuda:0), duplicating
            # the CLIP model on one device.  Non-primary ranks continue to
            # the category-sharded inference rendezvous below and wait there.
            if is_primary and bool(
                config.get("zero_shot", {}).get("enabled", False)
            ):
                from realiad_dinomaly2.zero_shot_engine import train_zero_shot

                _update_manifest(run_dir, run_manifest, status="training_zero_shot")
                train_zero_shot(config, resume=args.resume)
        if not is_primary:
            if args.skip_eval:
                return 0
            _wait_for_inference_rendezvous(
                rendezvous_path, run_manifest["run_id"], rendezvous_timeout
            )
            generate_competition_submission(
                config,
                checkpoint=args.checkpoint,
                allow_partial=args.allow_partial,
                artifact_kind="testc_evaluation",
                package_zip=args.export_testc_predictions_zip,
            )
            return 0
        checkpoint_path = resolve_competition_checkpoint(output_dir, args.checkpoint)
        _update_manifest(
            run_dir,
            run_manifest,
            status="checkpoint_ready",
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=file_sha256(checkpoint_path),
        )
        if bool(config["evaluation"].get("normal_prior", {}).get("enabled", False)):
            _update_manifest(run_dir, run_manifest, status="fitting_normal_prior")
            fit_normal_prior(config, checkpoint_path, categories=protocol["seen_categories"])
        if args.skip_eval:
            _update_manifest(run_dir, run_manifest, status="trained")
            return 0
        _set_inference_rendezvous(
            rendezvous_path,
            run_manifest["run_id"],
            "ready",
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=file_sha256(checkpoint_path),
        )
        _update_manifest(run_dir, run_manifest, status="inferring_testc")
        submission = generate_competition_submission(
            config,
            checkpoint=args.checkpoint,
            allow_partial=args.allow_partial,
            artifact_kind="testc_evaluation",
            package_zip=args.export_testc_predictions_zip,
        )
        _update_manifest(run_dir, run_manifest, status="scoring_testc")
        evaluation = evaluate_testc_submission(
            testc_root=testc_root,
            protocol_path=protocol_path,
            submission_result=submission,
            output_dir=output_dir,
            image_top_ratio=float(wrapper["testc"].get("image_top_ratio", 0.01)),
            recommended_checkpoint_steps=int(
                wrapper["testc"].get("recommended_checkpoint_steps", 6000)
            ),
        )
        latency = None
        if not args.skip_latency and bool(config.get("latency", {}).get("enabled", False)):
            _update_manifest(run_dir, run_manifest, status="benchmarking_latency")
            latency = benchmark_single_frame_latency(
                config,
                checkpoint_path,
                category=protocol["seen_categories"][0],
            )
        result = {
            "timestamp": utc_now(),
            "run_id": run_manifest["run_id"],
            "git_commit": run_manifest["git"]["commit"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "score": evaluation["score"],
            "latency": latency,
            "metrics": evaluation["report"],
        }
        append_jsonl(output_dir / "testc_model_registry.jsonl", result)
        _update_manifest(run_dir, run_manifest, status="complete", result=result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        if is_primary:
            if not args.skip_eval:
                _set_inference_rendezvous(
                    rendezvous_path,
                    run_manifest["run_id"],
                    "failed",
                    error="primary rank interrupted",
                )
            _update_manifest(run_dir, run_manifest, status="interrupted")
        return 130
    except Exception as exc:
        if is_primary:
            if not args.skip_eval:
                _set_inference_rendezvous(
                    rendezvous_path,
                    run_manifest["run_id"],
                    "failed",
                    error=repr(exc),
                )
            _update_manifest(
                run_dir,
                run_manifest,
                status="failed",
                error=f"{exc!r}\n{traceback.format_exc()}",
            )
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
