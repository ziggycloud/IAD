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
    if dataset.get("type") != "realiad_variety":
        raise ValueError("当前适配器仅支持 dataset.type=realiad_variety")
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

    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if float(training["adam_epsilon"]) <= 0:
        raise ValueError("training.adam_epsilon must be > 0")
    if float(training["gradient_clip_norm"]) <= 0:
        raise ValueError("training.gradient_clip_norm must be > 0")

    optimizer = training.get("optimizer", {})
    if not isinstance(optimizer, dict):
        raise ValueError("training.optimizer must be a mapping")
    optimizer_type = str(optimizer.get("type", "stable_adamw")).lower()
    if optimizer_type not in {"stable_adamw", "adamw", "adam"}:
        raise ValueError(
            "training.optimizer.type must be stable_adamw, adamw, or adam"
        )
    if float(optimizer.get("stable_clip_threshold", 1.0)) <= 0:
        raise ValueError("training.optimizer.stable_clip_threshold must be > 0")

    scheduler = training.get("scheduler", {})
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a mapping")
    scheduler_type = str(scheduler.get("type", "cosine")).lower()
    if scheduler_type not in {
        "cosine", "linear", "polynomial", "constant", "step", "multistep"
    }:
        raise ValueError(
            "training.scheduler.type must be cosine, linear, polynomial, "
            "constant, step, or multistep"
        )
    scheduler_warmup = int(
        scheduler.get("warmup_steps", training.get("warmup_steps", 0))
    )
    scheduler_min_ratio = float(
        scheduler.get("min_lr_ratio", training.get("final_lr_ratio", 1.0))
    )
    if scheduler_warmup < 0:
        raise ValueError("training.scheduler.warmup_steps must be >= 0")
    if not 0.0 <= scheduler_min_ratio <= 1.0:
        raise ValueError("training.scheduler.min_lr_ratio must be in [0, 1]")
    if scheduler_type == "polynomial" and float(
        scheduler.get("power", 1.0)
    ) <= 0:
        raise ValueError("training.scheduler.power must be > 0")
    if scheduler_type == "step" and int(
        scheduler.get("step_size", 1000)
    ) <= 0:
        raise ValueError("training.scheduler.step_size must be > 0")
    if scheduler_type in {"step", "multistep"} and not 0.0 < float(
        scheduler.get("gamma", 0.1)
    ) <= 1.0:
        raise ValueError("training.scheduler.gamma must be in (0, 1]")
    milestones = scheduler.get("milestones", [])
    if scheduler_type == "multistep" and (
        not isinstance(milestones, list)
        or any(int(value) <= 0 for value in milestones)
    ):
        raise ValueError(
            "training.scheduler.milestones must be a list of positive steps"
        )

    gradient_guard = training.get("gradient_guard", {})
    if not isinstance(gradient_guard, dict):
        raise ValueError("training.gradient_guard must be a mapping")
    skip_step_norm = gradient_guard.get("skip_step_norm")
    if skip_step_norm is not None and float(skip_step_norm) <= 0:
        raise ValueError("training.gradient_guard.skip_step_norm must be > 0 or null")
    if int(gradient_guard.get("max_consecutive_skips", 3)) < 0:
        raise ValueError(
            "training.gradient_guard.max_consecutive_skips must be >= 0"
        )

    if float(training.get("generalized_regularization_weight", 0.0)) < 0:
        raise ValueError("generalized_regularization_weight must be >= 0")

    evaluation = config["evaluation"]
    if not 0 < float(evaluation["image_top_ratio"]) <= 1:
        raise ValueError("image_top_ratio 必须位于 (0, 1]")
    if int(evaluation["metric_bins"]) < 5:
        raise ValueError("metric_bins 不能小于 5")
    gaussian_kernel_size = int(evaluation["gaussian_kernel_size"])
    if gaussian_kernel_size <= 0 or gaussian_kernel_size % 2 == 0:
        raise ValueError("evaluation.gaussian_kernel_size must be positive and odd")
    if float(evaluation["gaussian_sigma"]) < 0:
        raise ValueError("evaluation.gaussian_sigma must be >= 0")
    layer_weights = evaluation.get("anomaly_map_layer_weights")
    if layer_weights is not None:
        if (
            not isinstance(layer_weights, list)
            or not layer_weights
            or any(float(value) < 0 for value in layer_weights)
            or sum(float(value) for value in layer_weights) <= 0
        ):
            raise ValueError(
                "evaluation.anomaly_map_layer_weights must be a non-empty "
                "list of non-negative values with a positive sum"
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
    paths = {
        "json_dir": resolve_path(config["dataset"]["json_dir"]),
        "image_dir": resolve_path(config["dataset"]["image_dir"]),
        "output_dir": resolve_path(config["experiment"]["output_dir"]),
        "backbone_weights_dir": resolve_path(config["model"]["backbone_weights_dir"]),
    }
    if config["dataset"].get("train_image_dir"):
        paths["train_image_dir"] = resolve_path(
            config["dataset"]["train_image_dir"]
        )
    return paths


def materialize_paths(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    paths = resolved_paths(result)
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
    training_semantics = {
        key: config["training"].get(key)
        for key in (
            "total_steps", "effective_batch_size", "amp", "amp_dtype",
            "learning_rate", "first_bottleneck_lr_scale", "weight_decay",
            "adam_betas", "adam_epsilon", "warmup_steps", "lr_step_offset",
            "final_lr_ratio", "loose_loss_warmup_steps",
            "loose_loss_final_discard", "generalized_regularization_weight",
            "gradient_clip_norm",
        )
    }
    for optional_key in ("optimizer", "scheduler", "gradient_guard"):
        if optional_key in config["training"]:
            training_semantics[optional_key] = copy.deepcopy(
                config["training"][optional_key]
            )
    return {
        "dataset": {
            key: config["dataset"].get(key)
            for key in (
                "type",
                "json_dir",
                "image_dir",
                "train_image_dir",
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
        "training": training_semantics,
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
