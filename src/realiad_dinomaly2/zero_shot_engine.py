"""Supervised auxiliary-category training used by official MoECLIP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .moeclip_data import build_moeclip_auxiliary_dataset
from .runtime import atomic_torch_save, resolve_device, setup_logger, setup_seed
from .zero_shot_model import (
    ZERO_SHOT_FORMAT_VERSION,
    MoECLIPSegmenter,
    load_trainable_zero_shot_state_dict,
    trainable_zero_shot_state_dict,
    zero_shot_checkpoint_path,
    zero_shot_config_fingerprint,
    zero_shot_final_checkpoint_path,
    zero_shot_last_checkpoint_path,
)


def _focal_loss(probability: torch.Tensor, target: torch.Tensor,
                gamma: float = 2.0, smooth: float = 1e-5) -> torch.Tensor:
    classes = probability.shape[1]
    one_hot = F.one_hot(target.long(), classes).permute(0, 3, 1, 2).to(probability.dtype)
    one_hot = one_hot.clamp(smooth / max(1, classes - 1), 1.0 - smooth)
    pt = (one_hot * probability).sum(1) + smooth
    return (-(1.0 - pt).pow(gamma) * pt.log()).mean()


def _dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.flatten(1)
    target = target.flatten(1)
    score = (2.0 * (probability * target).sum(1) + 1.0) / (
        probability.sum(1) + target.sum(1) + 1.0
    )
    return 1.0 - score.mean()


def _segmentation_loss(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = mask[:, 0].long()
    return (
        _focal_loss(probability, target)
        + _dice_loss(probability[:, 0], 1.0 - mask[:, 0])
        + _dice_loss(probability[:, 1], mask[:, 0])
    )


def _payload(model: MoECLIPSegmenter, optimizer: torch.optim.Optimizer,
             config: dict[str, Any], completed_steps: int, epoch: int,
             best_metric: float, best_epoch: int) -> dict[str, Any]:
    return {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        "config_fingerprint": zero_shot_config_fingerprint(config),
        "completed_steps": completed_steps,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection": "minimum supervised training loss (official protocol)",
        "model": trainable_zero_shot_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }


def train_zero_shot(config: dict[str, Any], resume: str = "auto") -> Path | None:
    zero_config = config.get("zero_shot", {})
    if not bool(zero_config.get("enabled", False)):
        return None
    if str(zero_config.get("backend", "")) != "moeclip_official":
        raise ValueError("zero_shot.backend must be moeclip_official")
    train_config = zero_config["training"]
    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(int(config["experiment"]["seed"]) + 907, False)
    logger = setup_logger(
        "moeclip_training", zero_shot_checkpoint_path(config).parent.parent / "train.log"
    )
    dataset = build_moeclip_auxiliary_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=bool(config["runtime"].get("pin_memory", True)),
        persistent_workers=int(train_config.get("num_workers", 4)) > 0,
    )
    model = MoECLIPSegmenter(config, device)
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=float(train_config.get("learning_rate", 5e-5)),
        betas=tuple(float(value) for value in train_config.get("adam_betas", [0.5, 0.999])),
    )
    milestones = [int(value) for value in train_config.get("lr_milestones", [16000, 32000])]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=float(train_config.get("lr_gamma", 0.5))
    )
    last_path = zero_shot_last_checkpoint_path(config)
    best_path = zero_shot_checkpoint_path(config)
    start_epoch = 0
    completed_steps = 0
    best_metric = float("inf")
    best_epoch = 0
    if resume != "never" and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("config_fingerprint") != zero_shot_config_fingerprint(config):
            raise ValueError("MoECLIP last checkpoint/config fingerprint mismatch")
        load_trainable_zero_shot_state_dict(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved.get("epoch", 0))
        completed_steps = int(saved.get("completed_steps", 0))
        best_metric = float(saved.get("best_metric", float("inf")))
        best_epoch = int(saved.get("best_epoch", 0))
        for _ in range(completed_steps):
            scheduler.step()
        logger.info("恢复 MoECLIP：epoch=%d step=%d", start_epoch, completed_steps)
    epochs = int(train_config.get("epochs", 20))
    balance_weight = float(train_config.get("balance_loss_weight", 0.01))
    etf_weight = float(train_config.get("etf_loss_weight", 0.01))
    use_amp = bool(train_config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(train_config.get("amp_dtype", "bfloat16")).lower() in {"bf16", "bfloat16"} else torch.float16
    log_every = int(train_config.get("log_every", 20))
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            categories = [str(value) for value in batch["class_name"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(images, categories=categories)
                image_loss = F.cross_entropy(output["image_logits"], labels)
                segmentation_loss = sum(
                    _segmentation_loss(probability, masks)
                    for probability in output["patch_probabilities"]
                )
                loss = (
                    image_loss + segmentation_loss
                    + balance_weight * output["moe_balance_loss"]
                    + etf_weight * output["moe_etf_loss"]
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed_steps += 1
            epoch_loss += float(loss.detach())
            if completed_steps == 1 or completed_steps % log_every == 0:
                logger.info(
                    "MoECLIP epoch %d/%d step %d | loss %.5f | image %.5f | "
                    "seg %.5f | balance %.3e | etf %.3e | lr %.3e",
                    epoch + 1, epochs, completed_steps, loss.item(), image_loss.item(),
                    segmentation_loss.item(), output["moe_balance_loss"].item(),
                    output["moe_etf_loss"].item(), optimizer.param_groups[0]["lr"],
                )
        mean_loss = epoch_loss / max(1, len(loader))
        payload = _payload(model, optimizer, config, completed_steps, epoch + 1,
                           best_metric, best_epoch)
        atomic_torch_save(last_path, payload)
        if mean_loss < best_metric:
            best_metric, best_epoch = mean_loss, epoch + 1
            payload["best_metric"], payload["best_epoch"] = best_metric, best_epoch
            atomic_torch_save(best_path, payload)
            logger.info("更新 MoECLIP best：epoch=%d loss=%.6f", best_epoch, best_metric)
    final_path = zero_shot_final_checkpoint_path(config)
    atomic_torch_save(
        final_path,
        _payload(model, optimizer, config, completed_steps, epochs, best_metric, best_epoch),
    )
    logger.info("MoECLIP 训练完成：final=%s best=%s", final_path, best_path)
    return best_path
