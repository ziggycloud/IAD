"""Second-stage training for the independent unseen-category path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .competition_data import CompetitionObjectDataset, scan_competition_split
from .runtime import atomic_torch_save, resolve_device, setup_logger, setup_seed
from .zero_shot_model import (
    ZERO_SHOT_FORMAT_VERSION,
    LearnedZeroShotSegmenter,
    load_trainable_zero_shot_state_dict,
    trainable_zero_shot_state_dict,
    zero_shot_checkpoint_path,
    zero_shot_config_fingerprint,
    zero_shot_last_checkpoint_path,
)


def _learning_rate(step: int, config: dict[str, Any]) -> float:
    total = int(config["total_steps"])
    warmup = int(config.get("warmup_steps", 0))
    peak = float(config["learning_rate"])
    minimum = peak * float(config.get("min_lr_ratio", 0.05))
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def _payload(
    model: LearnedZeroShotSegmenter,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    completed_steps: int,
) -> dict[str, Any]:
    return {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        "config_fingerprint": zero_shot_config_fingerprint(config),
        "completed_steps": completed_steps,
        "model": trainable_zero_shot_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }


def train_zero_shot(config: dict[str, Any], resume: str = "auto") -> Path | None:
    if not bool(config.get("zero_shot", {}).get("enabled", False)):
        return None
    train_config = config["zero_shot"]["training"]
    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(int(config["experiment"]["seed"]) + 907, False)
    logger = setup_logger(
        "zero_shot_training",
        zero_shot_checkpoint_path(config).parent.parent / "train.log",
    )
    manifest = scan_competition_split(
        Path(config["dataset"]["train_dir"]),
        requested=config["dataset"]["categories"],
        limit=config["dataset"].get("category_limit"),
    )
    dataset = CompetitionObjectDataset(
        manifest.views,
        image_size=int(config["dataset"]["image_size"]),
        crop_size=int(config["dataset"]["crop_size"]),
        num_views=5,
        missing_view_policy="error",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.get("object_batch_size", 2)),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=bool(config["runtime"].get("pin_memory", True)),
        drop_last=True,
        persistent_workers=int(train_config.get("num_workers", 4)) > 0,
    )
    model = LearnedZeroShotSegmenter(config, device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target_group = (
            no_decay
            if name in {"prompt_delta", "layer_logits"} or parameter.ndim == 1
            else decay
        )
        target_group.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(train_config.get("weight_decay", 1e-4))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(train_config["learning_rate"]),
        betas=tuple(float(value) for value in train_config.get("adam_betas", [0.9, 0.999])),
    )
    last_path = zero_shot_last_checkpoint_path(config)
    completed_steps = 0
    if resume != "never" and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("config_fingerprint") != zero_shot_config_fingerprint(config):
            raise ValueError("zero-shot last checkpoint/config fingerprint mismatch")
        load_trainable_zero_shot_state_dict(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed_steps = int(saved["completed_steps"])
        logger.info("恢复 zero-shot 训练：step=%d", completed_steps)

    total_steps = int(train_config["total_steps"])
    iterator = iter(loader)
    model.train()
    for step in range(completed_steps + 1, total_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        object_images = batch["images"]
        object_batch, view_count = object_images.shape[:2]
        images = object_images.flatten(0, 1).to(device, non_blocking=True)
        categories = [
            str(category)
            for category in batch["category"]
            for _ in range(view_count)
        ]
        lr = _learning_rate(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
        requested_dtype = str(
            train_config.get("amp_dtype", "bfloat16")
        ).lower()
        amp_dtype = (
            torch.bfloat16
            if requested_dtype in {"bf16", "bfloat16"}
            else torch.float16
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(images, categories=categories)
            # The foreground target is an image-derived, detached background
            # estimate. Every foreground patch and every image is normal;
            # anomaly text acts only as a semantic negative anchor.
            foreground = output["color_foreground_probability"].detach()[:, 0]
            patch_target = torch.where(
                foreground >= float(train_config.get("foreground_threshold", 0.45)),
                torch.ones_like(foreground, dtype=torch.long),
                torch.zeros_like(foreground, dtype=torch.long),
            )
            patch_loss = F.cross_entropy(
                output["class_logits"], patch_target
            )
            global_target = torch.ones(
                images.shape[0], device=device, dtype=torch.long
            )
            image_loss = F.cross_entropy(
                output["global_logits"], global_target
            )
            margin = F.relu(
                float(train_config.get("normal_margin", 1.0))
                - output["global_logits"][:, 1]
                + output["global_logits"][:, 2]
            ).mean()
            normal_probability = output["global_logits"].softmax(-1)[:, 1]
            view_consistency = normal_probability.reshape(
                object_batch, view_count
            ).var(dim=1, unbiased=False).mean()
            clean_loss = output["semantic_probability"].mean()
            anchor_loss = model.prompt_delta.square().mean()
            loss = (
                float(train_config.get("patch_normal_weight", 1.0)) * patch_loss
                + float(train_config.get("image_normal_weight", 1.0)) * image_loss
                + float(train_config.get("normal_margin_weight", 0.25)) * margin
                + float(train_config.get("view_consistency_weight", 0.1))
                * view_consistency
                + float(train_config.get("clean_weight", 0.1)) * clean_loss
                + float(train_config.get("moe_balance_weight", 0.01))
                * output["moe_balance_loss"]
                + float(train_config.get("moe_diversity_weight", 0.01))
                * output["moe_diversity_loss"]
                + float(train_config.get("prompt_anchor_weight", 0.01))
                * anchor_loss
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(),
            float(train_config.get("gradient_clip_norm", 1.0)),
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite zero-shot gradient at step={step}")
        optimizer.step()
        if step == 1 or step % int(train_config.get("log_every", 20)) == 0:
            logger.info(
                "normal-only step %d/%d | loss %.5f | patch %.5f | "
                "image %.5f | margin %.5f | lr %.3e | grad %.3f",
                step,
                total_steps,
                loss.item(),
                patch_loss.item(),
                image_loss.item(),
                margin.item(),
                lr,
                float(grad_norm),
            )
        if step % int(train_config.get("checkpoint_every", 500)) == 0:
            atomic_torch_save(last_path, _payload(model, optimizer, config, step))

    final_path = zero_shot_checkpoint_path(config)
    atomic_torch_save(final_path, _payload(model, optimizer, config, total_steps))
    atomic_torch_save(last_path, _payload(model, optimizer, config, total_steps))
    logger.info("zero-shot 训练完成：%s", final_path)
    return final_path
