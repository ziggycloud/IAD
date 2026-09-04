"""Frozen CLIP semantics for unseen competition categories.

This module is deliberately inference-only.  It does not become part of the
Dinomaly checkpoint and is only constructed when a test category is absent
from Train, preserving the original path exactly for seen categories.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import resolve_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _prompt_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"evaluation.unseen_clip.{name} must be non-empty")
    prompts = tuple(str(item).strip() for item in value)
    if any(not prompt for prompt in prompts):
        raise ValueError(
            f"evaluation.unseen_clip.{name} cannot contain empty prompts"
        )
    return prompts


class FrozenClipBrokenSegmenter(nn.Module):
    """Produce a dense generic broken-vs-normal probability map."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Unseen-category CLIP is enabled, but open_clip_torch is not "
                "installed. Run: pip install -r requirements.txt"
            ) from exc

        self.device = device
        self.temperature = float(config.get("temperature", 0.07))
        self.intermediate_layers = int(config.get("intermediate_layers", 4))
        self.prompt_aggregation = str(
            config.get("prompt_aggregation", "max")
        ).lower()
        model_name = str(config.get("model_name", "ViT-B-16"))
        pretrained = str(config.get("pretrained", "openai"))
        image_size = int(config.get("image_size", 448))
        self.image_size = image_size
        cache_dir = resolve_path(
            str(config.get("weights_dir", "third_party/OpenCLIP/weights"))
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=bool(
                config.get("force_quick_gelu", pretrained == "openai")
            ),
            force_image_size=image_size,
            cache_dir=str(cache_dir),
        )
        model.eval()
        model.requires_grad_(False)
        visual = getattr(model, "visual", None)
        if visual is None or not hasattr(visual, "forward_intermediates"):
            raise TypeError(
                f"OpenCLIP model {model_name!r} does not expose ViT "
                "forward_intermediates"
            )
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)

        preprocess = open_clip.get_model_preprocess_cfg(model)
        clip_mean = tuple(float(value) for value in preprocess["mean"])
        clip_std = tuple(float(value) for value in preprocess["std"])
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(clip_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(clip_std).view(1, 3, 1, 1),
            persistent=False,
        )

        normal_prompts = _prompt_list(
            config.get("normal_prompts"), "normal_prompts"
        )
        broken_prompts = _prompt_list(
            config.get("broken_prompts"), "broken_prompts"
        )
        self.register_buffer(
            "normal_text",
            self._encode_prompts(normal_prompts),
            persistent=False,
        )
        self.register_buffer(
            "broken_text",
            self._encode_prompts(broken_prompts),
            persistent=False,
        )
        # ``device=`` above moves the OpenCLIP child module only. These
        # normalization buffers are created afterwards and would otherwise
        # remain on CPU, causing the first CUDA input normalization to fail.
        self.to(device)

    @torch.no_grad()
    def _encode_prompts(
        self,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        features = self.model.encode_text(tokens).float()
        return F.normalize(features, dim=-1, eps=1e-6)

    def _aggregate_prompt_similarity(
        self,
        patches: torch.Tensor,
        prompts: torch.Tensor,
    ) -> torch.Tensor:
        similarities = torch.einsum("bhwc,pc->bhwp", patches, prompts)
        if self.prompt_aggregation == "max":
            return similarities.amax(dim=-1)
        if self.prompt_aggregation == "mean":
            return similarities.mean(dim=-1)
        raise ValueError(
            "evaluation.unseen_clip.prompt_aggregation must be max or mean"
        )

    def _project_patches(self, feature: torch.Tensor) -> torch.Tensor:
        # OpenCLIP ViTs expose the same projection used by the pooled token.
        patches = feature.float().permute(0, 2, 3, 1)
        projection = getattr(self.model.visual, "proj", None)
        if isinstance(projection, nn.Module):
            patches = projection(patches)
        elif isinstance(projection, torch.Tensor):
            patches = patches @ projection
        elif patches.shape[-1] != self.normal_text.shape[-1]:
            raise RuntimeError(
                "CLIP patch width does not match text width and visual.proj "
                "is unavailable"
            )
        return F.normalize(patches.float(), dim=-1, eps=1e-6)

    @torch.no_grad()
    def forward(self, dinov2_images: torch.Tensor) -> torch.Tensor:
        # Competition images arrive with ImageNet/DINO normalization. Restore
        # RGB [0, 1], then apply the exact normalization from this CLIP model.
        rgb = (
            dinov2_images.float() * self.imagenet_std + self.imagenet_mean
        ).clamp(0.0, 1.0)
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(
                rgb,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        clip_images = (rgb - self.clip_mean) / self.clip_std
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            output = self.model.visual.forward_intermediates(
                clip_images,
                indices=self.intermediate_layers,
                normalize_intermediates=True,
                intermediates_only=True,
                output_fmt="NCHW",
            )
        features = output.get("image_intermediates")
        if not isinstance(features, list) or not features:
            raise RuntimeError("OpenCLIP returned no spatial intermediates")
        projected = torch.stack(
            [self._project_patches(feature) for feature in features],
            dim=0,
        ).mean(dim=0)
        projected = F.normalize(projected, dim=-1, eps=1e-6)
        # Keep each prompt separate. Max aggregation acts like a union of
        # defect concepts: a patch matching any known defect description can
        # strongly activate the broken class instead of being diluted by an
        # average over unrelated defect types.
        normal = self._aggregate_prompt_similarity(
            projected, self.normal_text
        )
        broken = self._aggregate_prompt_similarity(
            projected, self.broken_text
        )
        logits = torch.stack([normal, broken], dim=1) / self.temperature
        return logits.softmax(dim=1)[:, 1:2]


def fuse_unseen_anomaly_map(
    reconstruction_map: torch.Tensor,
    broken_probability: torch.Tensor,
    *,
    reconstruction_gain: float,
    semantic_gain: float,
    semantic_scale_floor: float,
    broken_threshold: float,
    center_quantile: float,
    upper_quantile: float,
    global_retention: float,
) -> torch.Tensor:
    """Suppress broad category novelty and retain CLIP-supported defects."""

    if reconstruction_map.ndim != 4 or reconstruction_map.shape[1] != 1:
        raise ValueError("reconstruction_map must have shape [B, 1, H, W]")
    broken_probability = F.interpolate(
        broken_probability.float(),
        size=reconstruction_map.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    values = reconstruction_map.float().flatten(start_dim=1)
    minimum = values.amin(dim=1).view(-1, 1, 1, 1)
    center = torch.quantile(values, center_quantile, dim=1).view(-1, 1, 1, 1)
    upper = torch.quantile(values, upper_quantile, dim=1).view(-1, 1, 1, 1)
    eps = torch.finfo(torch.float32).eps
    # Keep cosine-distance units so unseen and seen object scores remain
    # comparable. Subtracting the image-wide floor/median removes category
    # novelty without turning the result into an arbitrary [0, 1] score.
    global_map = (reconstruction_map.float() - minimum).clamp_min(0.0)
    local_map = (reconstruction_map.float() - center).clamp_min(0.0)
    adjusted = (
        global_retention * global_map
        + (1.0 - global_retention) * local_map
    )
    semantic = (
        (broken_probability - broken_threshold)
        / max(1.0 - broken_threshold, eps)
    ).clamp(0.0, 1.0)
    # Aggressive mode: semantic evidence is additive rather than a convex
    # blend, and its scale cannot collapse merely because an unseen-category
    # reconstruction map is spatially flat.
    semantic_scale = (upper - center).clamp_min(semantic_scale_floor)
    return (
        reconstruction_gain * adjusted
        + semantic_gain * semantic * semantic_scale
    ).clamp_min(0.0)
