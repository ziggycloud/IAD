"""On-device synthetic industrial defects with exact pixel masks."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _foreground(rgb: torch.Tensor) -> torch.Tensor:
    """Estimate the object support from border colour without class labels."""
    height, width = rgb.shape[-2:]
    border = torch.cat(
        [
            rgb[:, :, 0, :],
            rgb[:, :, -1, :],
            rgb[:, :, :, 0],
            rgb[:, :, :, -1],
        ],
        dim=-1,
    )
    background = border.median(dim=-1).values[:, :, None, None]
    distance = (rgb - background).square().mean(dim=1, keepdim=True).sqrt()
    flat = distance.flatten(1)
    threshold = torch.maximum(
        flat.quantile(0.55, dim=1),
        torch.full((rgb.shape[0],), 0.045, device=rgb.device),
    )[:, None, None, None]
    support = distance > threshold
    # Dilate slightly so defects may touch object boundaries.
    support = F.max_pool2d(support.float(), 11, stride=1, padding=5) > 0
    if height < 11 or width < 11:
        return torch.ones_like(support)
    return support


def _blob_mask(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    min_area: float,
    max_area: float,
) -> torch.Tensor:
    low_h = max(4, height // 32)
    low_w = max(4, width // 32)
    noise = torch.rand(batch, 1, low_h, low_w, device=device)
    noise = F.interpolate(
        noise, size=(height, width), mode="bicubic", align_corners=False
    )
    area = torch.empty(batch, device=device).uniform_(min_area, max_area)
    flattened = noise.flatten(1)
    # torch.quantile does not pair a vector q with individual rows: for
    # input [B, HW] and q [B] it returns [B, B]. Select each sample's own
    # order statistic explicitly so the threshold remains [B, 1, 1, 1].
    ordered = flattened.sort(dim=1).values
    rank = ((1.0 - area) * (flattened.shape[1] - 1)).long()
    threshold = ordered.gather(1, rank[:, None]).view(batch, 1, 1, 1)
    return noise > threshold


def _scratch_mask(
    batch: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    xx = xx.float()[None]
    yy = yy.float()[None]
    x0 = torch.rand(batch, 1, 1, device=device) * width
    y0 = torch.rand(batch, 1, 1, device=device) * height
    angle = torch.rand(batch, 1, 1, device=device) * math.pi
    length = (0.12 + 0.55 * torch.rand(batch, 1, 1, device=device)) * min(
        height, width
    )
    x1 = x0 + angle.cos() * length
    y1 = y0 + angle.sin() * length
    vx, vy = x1 - x0, y1 - y0
    t = ((xx - x0) * vx + (yy - y0) * vy) / (vx.square() + vy.square() + 1e-6)
    t = t.clamp(0.0, 1.0)
    distance = ((xx - (x0 + t * vx)).square() + (yy - (y0 + t * vy)).square()).sqrt()
    thickness = 1.0 + torch.rand(batch, 1, 1, device=device) * 0.025 * min(
        height, width
    )
    return (distance < thickness)[:, None]


def _tiny_mask(
    batch: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Pins, LEDs and fuse defects often occupy far below 0.2% of the image."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    xx = xx.float()[None]
    yy = yy.float()[None]
    center_x = torch.rand(batch, 1, 1, device=device) * width
    center_y = torch.rand(batch, 1, 1, device=device) * height
    radius = (0.003 + 0.012 * torch.rand(batch, 1, 1, device=device)) * min(
        height, width
    )
    aspect = torch.empty(batch, 1, 1, device=device).uniform_(0.35, 1.0)
    distance = ((xx - center_x) / aspect).square() + (yy - center_y).square()
    return (distance < radius.square())[:, None]


def _ring_mask(
    batch: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    xx = xx.float()[None]
    yy = yy.float()[None]
    center_x = torch.rand(batch, 1, 1, device=device) * width
    center_y = torch.rand(batch, 1, 1, device=device) * height
    radius = (0.01 + 0.04 * torch.rand(batch, 1, 1, device=device)) * min(
        height, width
    )
    thickness = torch.maximum(
        torch.ones_like(radius),
        radius * torch.empty(batch, 1, 1, device=device).uniform_(0.15, 0.35),
    )
    distance = ((xx - center_x).square() + (yy - center_y).square()).sqrt()
    return ((distance > radius - thickness) & (distance < radius + thickness))[:, None]


def _hard_normal_augment(rgb: torch.Tensor, probability: float) -> torch.Tensor:
    output = rgb.clone()
    selected = (torch.rand(rgb.shape[0], 1, 1, 1, device=rgb.device) < probability).float()
    contrast = torch.empty(rgb.shape[0], 1, 1, 1, device=rgb.device).uniform_(0.75, 1.25)
    brightness = torch.empty(rgb.shape[0], 1, 1, 1, device=rgb.device).uniform_(-0.08, 0.08)
    changed = (output - 0.5) * contrast + 0.5 + brightness
    changed = changed + torch.randn_like(changed) * 0.015
    return (output * (1.0 - selected) + changed * selected).clamp(0.0, 1.0)


def _object_anomaly_visibility(
    batch: int,
    *,
    group_size: int,
    object_probability: float,
    view_probability: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample balanced object labels with potentially invisible defect views."""
    if group_size <= 0 or batch % group_size:
        raise ValueError("batch must be divisible by a positive object group size")
    object_count = batch // group_size
    object_anomalous = torch.rand(object_count, device=device) < object_probability
    visible = torch.rand(object_count, group_size, device=device) < view_probability
    # An anomalous object must have at least one visible synthetic defect, while
    # the remaining views may stay normal like Test_C's null-mask semantics.
    missing = object_anomalous & ~visible.any(dim=1)
    if bool(missing.any()):
        fallback_view = torch.randint(0, group_size, (int(missing.sum()),), device=device)
        visible[missing] = False
        visible[missing, fallback_view] = True
    visible &= object_anomalous[:, None]
    return visible.reshape(batch, 1, 1, 1)


def synthesize_defects(
    normalized_images: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return augmented images, binary masks and image-level labels."""
    mean = normalized_images.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = normalized_images.new_tensor(_STD).view(1, 3, 1, 1)
    rgb = (normalized_images.float() * std + mean).clamp(0.0, 1.0)
    batch, _, height, width = rgb.shape
    rgb = _hard_normal_augment(
        rgb, float(config.get("hard_normal_probability", 0.5))
    )
    support = _foreground(rgb)
    blob = _blob_mask(
        batch,
        height,
        width,
        rgb.device,
        float(config.get("min_area_ratio", 0.002)),
        float(config.get("max_area_ratio", 0.18)),
    )
    scratch = _scratch_mask(batch, height, width, rgb.device)
    tiny = _tiny_mask(batch, height, width, rgb.device)
    ring = _ring_mask(batch, height, width, rgb.device)
    selector = torch.rand(batch, 1, 1, 1, device=rgb.device)
    tiny_probability = float(config.get("tiny_probability", 0.0))
    ring_probability = float(config.get("ring_probability", 0.0))
    scratch_probability = float(config.get("scratch_probability", 0.35))
    mask = torch.where(selector < tiny_probability, tiny, blob)
    mask = torch.where(
        (selector >= tiny_probability)
        & (selector < tiny_probability + ring_probability),
        ring,
        mask,
    )
    mask = torch.where(
        (selector >= tiny_probability + ring_probability)
        & (
            selector
            < tiny_probability + ring_probability + scratch_probability
        ),
        scratch,
        mask,
    )
    mask &= support
    group_size = int(config.get("object_group_size", 1))
    if group_size > 1:
        anomalous = _object_anomaly_visibility(
            batch,
            group_size=group_size,
            object_probability=float(
                config.get("object_anomaly_probability", 0.5)
            ),
            view_probability=float(config.get("view_anomaly_probability", 0.6)),
            device=rgb.device,
        )
    else:
        anomalous = torch.rand(batch, 1, 1, 1, device=rgb.device) < float(
            config.get("anomaly_probability", 0.75)
        )
    mask &= anomalous

    # Mix four defect families: foreign texture, discoloration, missing
    # material and displaced material. They deliberately avoid class names.
    texture = torch.roll(rgb, shifts=1, dims=0)
    colour = torch.rand(batch, 3, 1, 1, device=rgb.device)
    border_colour = torch.cat(
        [rgb[:, :, 0, :], rgb[:, :, -1, :]], dim=-1
    ).median(dim=-1).values[:, :, None, None]
    shift_y = random.choice((-32, -16, 16, 32))
    shift_x = random.choice((-32, -16, 16, 32))
    displaced = torch.roll(rgb, shifts=(shift_y, shift_x), dims=(-2, -1))
    modes = torch.randint(0, 4, (batch, 1, 1, 1), device=rgb.device)
    source = torch.where(modes == 0, texture, colour.expand_as(rgb))
    source = torch.where(modes == 2, border_colour.expand_as(rgb), source)
    source = torch.where(modes == 3, displaced, source)
    alpha = torch.empty(batch, 1, 1, 1, device=rgb.device).uniform_(
        float(config.get("alpha_min", 0.45)),
        float(config.get("alpha_max", 0.95)),
    )
    soft_mask = F.avg_pool2d(mask.float(), 5, stride=1, padding=2) * alpha
    synthetic = rgb * (1.0 - soft_mask) + source * soft_mask
    labels = mask.flatten(1).any(dim=1).float()
    return (synthetic.clamp(0.0, 1.0) - mean) / std, mask.float(), labels
