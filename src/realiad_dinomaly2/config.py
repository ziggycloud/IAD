from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"配置覆盖必须是 key=value，收到：{item!r}")
        key, raw_value = item.split("=", 1)
        _deep_set(result, key.strip(), yaml.safe_load(raw_value))
    return result


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _validate(config: dict[str, Any]) -> None:
    required_sections = (
        "experiment",
        "dataset",
        "model",
        "training",
        "evaluation",
        "runtime",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"配置缺少 section：{', '.join(missing)}")

    dataset = config["dataset"]
    dataset_type = str(dataset.get("type", ""))
    if dataset_type not in {"realiad_variety", "competition_folders"}:
        raise ValueError(
            "dataset.type 必须是 realiad_variety 或 competition_folders"
        )
    if dataset_type == "competition_folders":
        for key in ("train_dir", "test_dir"):
            if not isinstance(dataset.get(key), (str, Path)):
                raise ValueError(f"dataset.{key} must be a path")
        for key in ("categories", "test_categories"):
            value = dataset.get(key, "all")
            if value != "all" and not isinstance(value, (list, tuple)):
                raise ValueError(f"dataset.{key} must be all or a category list")
        for key in ("category_limit", "test_category_limit"):
            value = dataset.get(key)
            if value is not None and int(value) <= 0:
                raise ValueError(f"dataset.{key} must be null or positive")
        if not isinstance(dataset.get("require_same_categories", True), bool):
            raise ValueError("dataset.require_same_categories must be boolean")
        for key in (
            "expected_categories",
            "expected_test_categories",
            "expected_train_samples_per_category",
            "expected_test_samples_per_category",
        ):
            value = dataset.get(key)
            if value is not None and int(value) <= 0:
                raise ValueError(f"dataset.{key} must be null or positive")
    image_size = int(dataset["image_size"])
    crop_size = int(dataset["crop_size"])
    if image_size <= 0 or crop_size <= 0 or crop_size > image_size:
        raise ValueError("需要满足 image_size >= crop_size > 0")
    if crop_size % 14 != 0:
        raise ValueError("DINOv2/14 要求 crop_size 能被 14 整除")

    train_image_dir = dataset.get("train_image_dir")
    if train_image_dir is not None and not isinstance(train_image_dir, (str, Path)):
        raise ValueError("dataset.train_image_dir must be a path or null")
    mask_semantics = dataset.get(
        "mask_resize_semantics",
        "upstream_bilinear_nonzero",
    )
    if mask_semantics not in {
        "upstream_bilinear_nonzero",
        "nearest_binary",
    }:
        raise ValueError(
            "dataset.mask_resize_semantics must be "
            "upstream_bilinear_nonzero or nearest_binary"
        )

    training = config["training"]
    if int(training["effective_batch_size"]) <= 0:
        raise ValueError("effective_batch_size 必须为正整数")
    micro = training["micro_batch_size"]
    if micro != "auto" and int(micro) <= 0:
        raise ValueError("micro_batch_size 必须为 auto 或正整数")
    if int(training["total_steps"]) <= 0:
        raise ValueError("total_steps 必须为正整数")

    if float(training.get("generalized_regularization_weight", 0.0)) < 0:
        raise ValueError("generalized_regularization_weight must be >= 0")

    evaluation = config["evaluation"]
    if not 0 < float(evaluation["image_top_ratio"]) <= 1:
        raise ValueError("image_top_ratio 必须位于 (0, 1]")
    if int(evaluation["metric_bins"]) < 5:
        raise ValueError("metric_bins 不能小于 5")
    submission = config.get("submission")
    if dataset_type == "competition_folders":
        if not isinstance(submission, dict):
            raise ValueError(
                "competition_folders 配置必须包含 submission section"
            )
        if int(submission.get("mask_size", 448)) != 448:
            raise ValueError("比赛要求 submission.mask_size 固定为 448")
        lower = float(submission.get("lower_quantile", 0.001))
        upper = float(submission.get("upper_quantile", 0.99999))
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "submission quantiles must satisfy 0 <= lower < upper <= 1"
            )
    cache = config.get("cache")
    if cache is not None:
        if "output_dir" not in cache:
            raise ValueError("cache.output_dir is required when cache is configured")
        if int(cache.get("max_side", 1024)) <= 0:
            raise ValueError("cache.max_side must be a positive integer")
        if int(cache.get("workers", 4)) <= 0:
            raise ValueError("cache.workers must be a positive integer")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件不是 YAML mapping：{config_path}")
    config = apply_overrides(raw, overrides)
    _validate(config)
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
    }
    return config


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config["dataset"]
    paths = {
        "output_dir": resolve_path(config["experiment"]["output_dir"]),
        "backbone_weights_dir": resolve_path(config["model"]["backbone_weights_dir"]),
    }
    if dataset.get("type") == "competition_folders":
        paths["train_dir"] = resolve_path(dataset["train_dir"])
        paths["test_dir"] = resolve_path(dataset["test_dir"])
    else:
        paths["json_dir"] = resolve_path(dataset["json_dir"])
        paths["image_dir"] = resolve_path(dataset["image_dir"])
        if dataset.get("train_image_dir"):
            paths["train_image_dir"] = resolve_path(
                dataset["train_image_dir"]
            )
    return paths


def materialize_paths(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    paths = resolved_paths(result)
    if result["dataset"].get("type") == "competition_folders":
        result["dataset"]["train_dir"] = str(paths["train_dir"])
        result["dataset"]["test_dir"] = str(paths["test_dir"])
    else:
        result["dataset"]["json_dir"] = str(paths["json_dir"])
        result["dataset"]["image_dir"] = str(paths["image_dir"])
    if "train_image_dir" in paths:
        result["dataset"]["train_image_dir"] = str(paths["train_image_dir"])
    result["experiment"]["output_dir"] = str(paths["output_dir"])
    result["model"]["backbone_weights_dir"] = str(paths["backbone_weights_dir"])
    if "cache" in result and "output_dir" in result["cache"]:
        result["cache"]["output_dir"] = str(resolve_path(result["cache"]["output_dir"]))
    return result


def semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must match when resuming a training run."""
    return {
        "dataset": {
            key: config["dataset"].get(key)
            for key in (
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
        },
        "model": copy.deepcopy(config["model"]),
        "training": {
            key: config["training"].get(key)
            for key in (
                "total_steps",
                "effective_batch_size",
                "amp",
                "amp_dtype",
                "learning_rate",
                "first_bottleneck_lr_scale",
                "weight_decay",
                "adam_betas",
                "adam_epsilon",
                "warmup_steps",
                "lr_step_offset",
                "final_lr_ratio",
                "loose_loss_warmup_steps",
                "loose_loss_final_discard",
                "generalized_regularization_weight",
                "gradient_clip_norm",
            )
        },
        "seed": config["experiment"]["seed"],
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        semantic_config(config),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dump_resolved_config(config: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = copy.deepcopy(config)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
