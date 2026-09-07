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
from .synthetic_anomaly import synthesize_defects
from .zero_shot_model import (
    ZERO_SHOT_FORMAT_VERSION,
    LearnedZeroShotSegmenter,
    load_trainable_zero_shot_state_dict,
    trainable_zero_shot_state_dict,
    zero_shot_checkpoint_path,
    zero_shot_config_fingerprint,
    zero_shot_final_checkpoint_path,
    zero_shot_last_checkpoint_path,
)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    pt = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2.0 * (probability * targets).flatten(1).sum(dim=1) + 1.0
    denominator = (probability + targets).flatten(1).sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


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
    *,
    best_metric: float,
    best_step: int,
    ema_loss: float | None,
) -> dict[str, Any]:
    return {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        "config_fingerprint": zero_shot_config_fingerprint(config),
        "completed_steps": completed_steps,
        "best_metric": best_metric,
        "best_step": best_step,
        "ema_loss": ema_loss,
        "selection": "minimum training-loss EMA after warmup",
        "model": trainable_zero_shot_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }


def train_zero_shot(config: dict[str, Any], resume: str = "auto") -> Path | None:
    if not bool(config.get("zero_shot", {}).get("enabled", False)):
        return None
    train_config = config["zero_shot"]["training"]
    synthesis_config = config["zero_shot"]["synthesis"]
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
    best_path = zero_shot_checkpoint_path(config)
    completed_steps = 0
    best_metric = float("inf")
    best_step = 0
    ema_loss: float | None = None
    if resume != "never" and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("config_fingerprint") != zero_shot_config_fingerprint(config):
            raise ValueError("zero-shot last checkpoint/config fingerprint mismatch")
        load_trainable_zero_shot_state_dict(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed_steps = int(saved["completed_steps"])
        best_metric = float(saved.get("best_metric", float("inf")))
        best_step = int(saved.get("best_step", 0))
        saved_ema = saved.get("ema_loss")
        ema_loss = None if saved_ema is None else float(saved_ema)
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
        images = batch["images"].flatten(0, 1).to(device, non_blocking=True)
        images, masks, labels = synthesize_defects(images, synthesis_config)
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
        use_feature_anomaly = torch.rand(()).item() < float(
            synthesis_config.get("feature_anomaly_probability", 0.5)
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            train_branch = "visual" if step % 2 else "textual"
            output = model(
                images,
                feature_anomaly_mask=masks if use_feature_anomaly else None,
                feature_noise_std=float(synthesis_config.get("feature_noise_std", 0.08)),
                train_branch=train_branch,
            )
            target = F.interpolate(
                masks, size=output["logits"].shape[-2:], mode="area"
            ).clamp(0.0, 1.0)
            focal_gamma = float(train_config.get("focal_gamma", 2.0))
            branch_logits = output[f"{train_branch}_margin"]
            branch_image_logits = output[f"{train_branch}_image_margin"]
            focal = _focal_loss(
                branch_logits,
                target,
                focal_gamma,
                float(train_config.get("focal_alpha", 0.75)),
            )
            positive = target.flatten(1).sum(dim=1) > 1e-6
            dice = (
                _dice_loss(branch_logits[positive], target[positive])
                if positive.any()
                else branch_logits.new_zeros(())
            )
            image_loss = F.binary_cross_entropy_with_logits(
                branch_image_logits, labels
            )
            clean = labels == 0
            clean_loss = (
                branch_logits[clean].sigmoid().mean()
                if clean.any()
                else branch_logits.new_zeros(())
            )
            # Keep AdaptCLIP-style alternating optimization strict: the text
            # prompt is regularized only on textual-adapter updates.
            anchor_loss = (
                model.prompt_delta.square().mean()
                if step % 2 == 0
                else output["logits"].new_zeros(())
            )
            loss = (
                float(train_config.get("focal_weight", 1.0)) * focal
                + float(train_config.get("dice_weight", 1.0)) * dice
                + float(train_config.get("image_weight", 0.25)) * image_loss
                + float(train_config.get("clean_weight", 0.2)) * clean_loss
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
        loss_value = float(loss.detach())
        ema_loss = (
            loss_value
            if ema_loss is None
            else 0.98 * ema_loss + 0.02 * loss_value
        )
        best_start = max(
            int(train_config.get("warmup_steps", 0)),
            max(1, total_steps // 10),
        )
        best_check_every = int(train_config.get("log_every", 20))
        if (
            step >= best_start
            and step % best_check_every == 0
            and ema_loss < best_metric - 1e-4
        ):
            best_metric = ema_loss
            best_step = step
            atomic_torch_save(
                best_path,
                _payload(
                    model,
                    optimizer,
                    config,
                    step,
                    best_metric=best_metric,
                    best_step=best_step,
                    ema_loss=ema_loss,
                ),
            )
            logger.info(
                "更新 zero-shot best：step=%d | ema_loss=%.6f | %s",
                best_step,
                best_metric,
                best_path,
            )
        if step == 1 or step % int(train_config.get("log_every", 20)) == 0:
            logger.info(
                "zero-shot step %d/%d | loss %.5f | focal %.5f | "
                "dice %.5f | positive %.3f | branch %s | lr %.3e | grad %.3e",
                step,
                total_steps,
                loss.item(),
                focal.item(),
                dice.item(),
                float(positive.float().mean()),
                train_branch,
                lr,
                float(grad_norm),
            )
        if step % int(train_config.get("checkpoint_every", 500)) == 0:
            atomic_torch_save(
                last_path,
                _payload(
                    model,
                    optimizer,
                    config,
                    step,
                    best_metric=best_metric,
                    best_step=best_step,
                    ema_loss=ema_loss,
                ),
            )

    final_path = zero_shot_final_checkpoint_path(config)
    final_payload = _payload(
        model,
        optimizer,
        config,
        total_steps,
        best_metric=best_metric,
        best_step=best_step,
        ema_loss=ema_loss,
    )
    atomic_torch_save(final_path, final_payload)
    atomic_torch_save(last_path, final_payload)
    if not best_path.is_file():
        best_metric = float(ema_loss) if ema_loss is not None else float("nan")
        best_step = total_steps
        final_payload["best_metric"] = best_metric
        final_payload["best_step"] = best_step
        final_payload["selection"] = "final fallback; no eligible EMA checkpoint"
        atomic_torch_save(best_path, final_payload)
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    selected["training_completed_steps"] = total_steps
    atomic_torch_save(best_path, selected)
    logger.info(
        "zero-shot 训练完成：final=%s | best=%s (step=%d, ema_loss=%.6f)",
        final_path,
        best_path,
        best_step,
        best_metric,
    )
    return best_path
