from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.bootstrap import ensure_iad_runtime  # noqa: E402


ensure_iad_runtime(ROOT, Path(__file__), sys.argv[1:])

import torch  # noqa: E402
import yaml  # noqa: E402

from realiad_dinomaly2.competition_submission import (  # noqa: E402
    generate_competition_submission,
    resolve_competition_checkpoint,
)
from realiad_dinomaly2.config import (  # noqa: E402
    config_fingerprint,
    load_config,
    materialize_paths,
)
from realiad_dinomaly2.testc_data import (  # noqa: E402
    discover_data_root,
    load_testc_protocol,
)


_TRAINING_DATASET_KEYS = (
    "type",
    "json_dir",
    "image_dir",
    "train_image_dir",
    "train_dir",
    "categories",
    "category_limit",
    "image_size",
    "crop_size",
    "train_mode",
    "image_label_policy",
    "missing_anomaly_mask_policy",
    "mask_resize_semantics",
)


def _load_wrapper(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or "base_config" not in payload
        or "testc" not in payload
    ):
        raise ValueError(
            "Test_B submission config requires the Test_C wrapper format"
        )
    payload["_path"] = resolved
    return payload


def _resolve_data_root(
    wrapper: dict[str, Any], explicit: Path | None
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = wrapper["testc"].get("data_root", "auto")
    if configured != "auto":
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()
    return discover_data_root(ROOT)


def _build_testb_config(
    wrapper: dict[str, Any],
    data_root: Path,
    testb_dir: Path | None,
    overrides: list[str],
) -> dict[str, Any]:
    wrapper_path = Path(wrapper["_path"])
    base_path = Path(str(wrapper["base_config"]))
    if not base_path.is_absolute():
        base_path = wrapper_path.parent / base_path
    protocol = load_testc_protocol(ROOT / "configs" / "testc_protocol.json")
    experiment = wrapper.get("experiment", {})
    test_root = (
        testb_dir.expanduser().resolve()
        if testb_dir is not None
        else (data_root / "competition" / "Test_B").resolve()
    )
    defaults = [
        f"experiment.name={json.dumps(str(experiment.get('name', 'testb')))}",
        (
            "experiment.output_dir="
            f"{json.dumps(str(experiment.get('output_dir', 'outputs/testc')))}"
        ),
        f"dataset.train_dir={json.dumps(str(data_root / 'competition' / 'Train'))}",
        f"dataset.test_dir={json.dumps(str(test_root))}",
        f"dataset.categories={json.dumps(protocol['seen_categories'])}",
        "dataset.category_limit=null",
        "dataset.test_categories=all",
        "dataset.test_category_limit=null",
        "dataset.require_same_categories=false",
        "dataset.expected_categories=50",
        "dataset.expected_test_categories=null",
        "dataset.expected_train_samples_per_category=20",
        "dataset.expected_test_samples_per_category=null",
    ]
    return materialize_paths(load_config(base_path, [*defaults, *overrides]))


def _restore_checkpoint_training_config(
    config: dict[str, Any],
    checkpoint_path: Path,
    explicit: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    fingerprint = checkpoint.get("config_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("seen checkpoint has no config_fingerprint")
    output_dir = Path(config["experiment"]["output_dir"])
    candidates = (
        [explicit.expanduser().resolve()]
        if explicit is not None
        else sorted(
            output_dir.glob("testc_runs/*/resolved_config.yaml"), reverse=True
        )
        + [
            output_dir / "resolved_config.yaml",
            output_dir / "competition_resolved_config.yaml",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            not isinstance(candidate, dict)
            or config_fingerprint(candidate) != fingerprint
        ):
            continue
        restored = copy.deepcopy(config)
        zero_shot_checkpoint = restored.get("zero_shot", {}).get("checkpoint")
        restored["model"] = copy.deepcopy(candidate["model"])
        restored["training"] = copy.deepcopy(candidate["training"])
        restored["experiment"]["seed"] = candidate["experiment"]["seed"]
        for key in _TRAINING_DATASET_KEYS:
            if key in candidate["dataset"]:
                restored["dataset"][key] = copy.deepcopy(
                    candidate["dataset"][key]
                )
        # The main checkpoint fingerprint excludes the independent zero-shot
        # branch. Restore its exact training semantics from the same run while
        # retaining the caller's best/final checkpoint choice.
        if "zero_shot" in candidate:
            restored["zero_shot"] = copy.deepcopy(candidate["zero_shot"])
            if zero_shot_checkpoint is not None:
                restored["zero_shot"]["checkpoint"] = zero_shot_checkpoint
        if config_fingerprint(restored) != fingerprint:
            raise RuntimeError(
                f"restored config from {path} still mismatches checkpoint"
            )
        return restored, path
    if config_fingerprint(config) == fingerprint:
        return config, None
    raise ValueError(
        "The seen checkpoint does not match the current training config "
        "and no matching resolved YAML was found; pass "
        "--checkpoint-config /path/to/resolved_config.yaml"
    )


def run_testb_submission(
    *,
    config_path: Path,
    data_root: Path | None,
    testb_dir: Path | None,
    checkpoint: str,
    checkpoint_config: Path | None,
    overrides: list[str],
    allow_partial: bool,
) -> dict[str, Any]:
    """Package Test_B using completed seen and zero-shot checkpoints only."""
    wrapper = _load_wrapper(config_path)
    resolved_data_root = _resolve_data_root(wrapper, data_root)
    config = _build_testb_config(
        wrapper, resolved_data_root, testb_dir, overrides
    )
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = resolve_competition_checkpoint(output_dir, checkpoint)
    config, restored_path = _restore_checkpoint_training_config(
        config, checkpoint_path, checkpoint_config
    )
    result = generate_competition_submission(
        config,
        checkpoint=str(checkpoint_path),
        allow_partial=allow_partial,
        artifact_kind="competition_submission",
        package_zip=True,
    )
    result["mode"] = "testb_submission_only"
    result["restored_checkpoint_config"] = (
        str(restored_path) if restored_path is not None else None
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use completed seen and AdaptCLIP checkpoints to build the Test_B "
            "submission ZIP. No training or evaluation is performed."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "testc.yaml"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--testb-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--checkpoint-config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_testb_submission(
        config_path=args.config,
        data_root=args.data_root,
        testb_dir=args.testb_dir,
        checkpoint=args.checkpoint,
        checkpoint_config=args.checkpoint_config,
        overrides=args.set,
        allow_partial=args.allow_partial,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
