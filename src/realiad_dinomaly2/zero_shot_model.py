"""Trainable category-agnostic anomaly path with a frozen public CLIP base.

The local implementation borrows the dual-adapter and alternating-update idea
from AdaptCLIP, without loading its code or anomaly-specific checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import resolve_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ZERO_SHOT_FORMAT_VERSION = 3


def zero_shot_checkpoint_path(config: dict[str, Any]) -> Path:
    configured = config.get("zero_shot", {}).get("checkpoint")
    output_dir = Path(config["experiment"]["output_dir"])
    if configured is None:
        return output_dir / "zero_shot" / "checkpoints" / "final_model.pt"
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else output_dir / path


def zero_shot_last_checkpoint_path(config: dict[str, Any]) -> Path:
    return zero_shot_checkpoint_path(config).parent / "last.pt"


def zero_shot_config_fingerprint(config: dict[str, Any]) -> str:
    dataset = config["dataset"]
    zero_shot = config.get("zero_shot", {})
    payload = {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        # Inference thresholds and the checkpoint destination do not alter the
        # learned parameters and must not invalidate an otherwise compatible
        # training artifact.
        "zero_shot": {
            key: zero_shot.get(key)
            for key in ("model", "training", "synthesis")
        },
        "dataset": {
            key: dataset.get(key)
            for key in (
                "type",
                "train_dir",
                "categories",
                "category_limit",
                "image_size",
                "crop_size",
            )
        },
        "seed": config["experiment"]["seed"],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"zero_shot.model.{name} must be a non-empty list")
    prompts = tuple(str(item).strip() for item in value)
    if any(not prompt for prompt in prompts):
        raise ValueError(f"zero_shot.model.{name} contains an empty prompt")
    return prompts


class LearnedZeroShotSegmenter(nn.Module):
    """Dense normal-vs-broken segmenter built around a frozen OpenCLIP ViT."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "The zero-shot branch requires open_clip_torch. "
                "Run: pip install -r requirements.txt"
            ) from exc

        model_config = config["zero_shot"]["model"]
        self.device_spec = device
        self.image_size = int(model_config.get("image_size", 448))
        self.intermediate_layers = int(
            model_config.get("intermediate_layers", 4)
        )
        self.semantic_temperature = float(
            model_config.get("semantic_temperature", 0.07)
        )
        self.visual_fusion_weight = float(
            model_config.get("visual_fusion_weight", 0.65)
        )
        self.image_local_weight = float(
            model_config.get("image_local_weight", 0.35)
        )
        self.image_top_ratio = float(
            model_config.get("image_top_ratio", 0.01)
        )
        model_name = str(model_config.get("model_name", "ViT-B-16"))
        pretrained = str(model_config.get("pretrained", "openai"))
        cache_dir = resolve_path(
            str(
                model_config.get(
                    "weights_dir", "third_party/OpenCLIP/weights"
                )
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        clip, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=bool(
                model_config.get("force_quick_gelu", pretrained == "openai")
            ),
            force_image_size=self.image_size,
            cache_dir=str(cache_dir),
        )
        clip.eval()
        clip.requires_grad_(False)
        visual = getattr(clip, "visual", None)
        if visual is None or not hasattr(visual, "forward_intermediates"):
            raise TypeError(
                f"OpenCLIP model {model_name!r} does not expose spatial "
                "forward_intermediates"
            )
        self.clip = clip
        self.tokenizer = open_clip.get_tokenizer(model_name)

        preprocess = open_clip.get_model_preprocess_cfg(clip)
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
            model_config.get("normal_prompts"), "normal_prompts"
        )
        broken_prompts = _prompt_list(
            model_config.get("broken_prompts"), "broken_prompts"
        )
        normal_anchor = self._encode_prompts(normal_prompts).mean(dim=0)
        broken_anchor = self._encode_prompts(broken_prompts).mean(dim=0)
        prompt_anchors = F.normalize(
            torch.stack([normal_anchor, broken_anchor]), dim=-1, eps=1e-6
        )
        self.register_buffer("prompt_anchors", prompt_anchors, persistent=True)
        text_dim = int(prompt_anchors.shape[-1])
        self.prompt_delta = nn.Parameter(torch.zeros(2, text_dim))

        configured_weights = model_config.get(
            "intermediate_layer_weights", [0.1, 0.2, 0.3, 0.4]
        )
        weights = torch.tensor(configured_weights, dtype=torch.float32)
        if weights.numel() != self.intermediate_layers:
            raise ValueError(
                "zero_shot.model.intermediate_layer_weights must match "
                "intermediate_layers"
            )
        weights = weights / weights.sum()
        self.layer_logits = nn.Parameter(weights.clamp_min(1e-6).log())

        adapter_hidden = int(model_config.get("adapter_hidden_dim", 256))
        dropout = float(model_config.get("dropout", 0.1))
        self.local_visual_adapter = nn.Sequential(
            nn.LayerNorm(text_dim, eps=1e-6),
            nn.Linear(text_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, text_dim),
        )
        self.global_visual_adapter = nn.Sequential(
            nn.LayerNorm(text_dim, eps=1e-6),
            nn.Linear(text_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, text_dim),
        )
        self.textual_adapter = nn.Sequential(
            nn.LayerNorm(text_dim, eps=1e-6),
            nn.Linear(text_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, text_dim),
        )
        # All adapters start as identity residuals, preserving public CLIP's
        # original embedding geometry before auxiliary anomaly training.
        for adapter in (
            self.local_visual_adapter,
            self.global_visual_adapter,
            self.textual_adapter,
        ):
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.to(device)
        self.clip.eval()

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device_spec)
        return F.normalize(
            self.clip.encode_text(tokens).float(), dim=-1, eps=1e-6
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def learned_prompts(self) -> torch.Tensor:
        return F.normalize(
            self.prompt_anchors
            + self.prompt_delta
            + self.textual_adapter(self.prompt_anchors),
            dim=-1,
            eps=1e-6,
        )

    def _project_patches(self, feature: torch.Tensor) -> torch.Tensor:
        patches = feature.float().permute(0, 2, 3, 1)
        projection = getattr(self.clip.visual, "proj", None)
        if isinstance(projection, nn.Module):
            patches = projection(patches)
        elif isinstance(projection, torch.Tensor):
            patches = patches @ projection
        elif patches.shape[-1] != self.prompt_anchors.shape[-1]:
            raise RuntimeError(
                "CLIP patch width does not match text width and visual.proj "
                "is unavailable"
            )
        return F.normalize(patches.float(), dim=-1, eps=1e-6)

    def _clip_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        rgb = (
            images.float() * self.imagenet_std + self.imagenet_mean
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
            if images.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            output = self.clip.visual.forward_intermediates(
                clip_images,
                indices=self.intermediate_layers,
                normalize_intermediates=True,
                intermediates_only=True,
                output_fmt="NCHW",
            )
        features = output.get("image_intermediates")
        if not isinstance(features, list) or len(features) != self.intermediate_layers:
            raise RuntimeError("OpenCLIP returned unexpected spatial intermediates")
        return features

    def forward(
        self,
        images: torch.Tensor,
        *,
        feature_anomaly_mask: torch.Tensor | None = None,
        feature_noise_std: float = 0.0,
        train_branch: str | None = None,
    ) -> dict[str, torch.Tensor]:
        projected_layers = torch.stack(
            [self._project_patches(feature) for feature in self._clip_features(images)],
            dim=0,
        )
        layer_weights = self.layer_logits.softmax(dim=0).to(
            projected_layers.dtype
        )
        patches = torch.einsum("l,lbhwc->bhwc", layer_weights, projected_layers)
        if feature_anomaly_mask is not None and feature_noise_std > 0.0:
            feature_mask = F.interpolate(
                feature_anomaly_mask.float(),
                size=patches.shape[1:3],
                mode="area",
            ).permute(0, 2, 3, 1)
            patches = patches + (
                torch.randn_like(patches)
                * feature_mask
                * float(feature_noise_std)
            )
        global_feature = F.normalize(patches.mean(dim=(1, 2)), dim=-1, eps=1e-6)
        visual_patches = F.normalize(
            patches + self.local_visual_adapter(patches), dim=-1, eps=1e-6
        )
        visual_global = F.normalize(
            global_feature + self.global_visual_adapter(global_feature),
            dim=-1,
            eps=1e-6,
        )
        static_prompts = self.prompt_anchors.to(patches.dtype)
        learned_prompts = self.learned_prompts().to(patches.dtype)

        visual_logits = torch.einsum(
            "bhwc,kc->bkhw", visual_patches, static_prompts
        ) / self.semantic_temperature
        textual_logits = torch.einsum(
            "bhwc,kc->bkhw", patches, learned_prompts
        ) / self.semantic_temperature
        visual_global_logits = torch.einsum(
            "bc,kc->bk", visual_global, static_prompts
        ) / self.semantic_temperature
        textual_global_logits = torch.einsum(
            "bc,kc->bk", global_feature, learned_prompts
        ) / self.semantic_temperature

        visual_margin = visual_logits[:, 1:2] - visual_logits[:, 0:1]
        textual_margin = textual_logits[:, 1:2] - textual_logits[:, 0:1]
        visual_image_margin = (
            visual_global_logits[:, 1] - visual_global_logits[:, 0]
        )
        textual_image_margin = (
            textual_global_logits[:, 1] - textual_global_logits[:, 0]
        )
        # AdaptCLIP's central result: visual and textual representations are
        # optimized alternately, not jointly in one backward pass.
        if train_branch == "visual":
            textual_margin = textual_margin.detach()
            textual_image_margin = textual_image_margin.detach()
        elif train_branch == "textual":
            visual_margin = visual_margin.detach()
            visual_image_margin = visual_image_margin.detach()
        elif train_branch not in {None, "visual", "textual"}:
            raise ValueError(f"unsupported train_branch={train_branch!r}")

        logits = (
            self.visual_fusion_weight * visual_margin
            + (1.0 - self.visual_fusion_weight) * textual_margin
        )
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        flat = probability.flatten(1)
        top_count = max(1, int(round(flat.shape[1] * self.image_top_ratio)))
        local_probability = flat.topk(top_count, dim=1).values.mean(dim=1)
        local_logit = torch.logit(local_probability.clamp(1e-6, 1.0 - 1e-6))
        global_margin = (
            self.visual_fusion_weight * visual_image_margin
            + (1.0 - self.visual_fusion_weight) * textual_image_margin
        )
        image_logits = (
            (1.0 - self.image_local_weight) * global_margin
            + self.image_local_weight * local_logit
        )
        image_probability = image_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        return {
            "logits": logits.float(),
            "probability": probability.float(),
            "image_logits": image_logits.float(),
            "image_probability": image_probability.float(),
            "visual_margin": visual_margin.float(),
            "textual_margin": textual_margin.float(),
            "visual_image_margin": visual_image_margin.float(),
            "textual_image_margin": textual_image_margin.float(),
            "visual_logits": visual_logits.float(),
            "textual_logits": textual_logits.float(),
            "semantic_logit": logits.float(),
            "layer_weights": layer_weights.float(),
        }


def trainable_zero_shot_state_dict(
    model: LearnedZeroShotSegmenter,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("clip.")
    }


def load_trainable_zero_shot_state_dict(
    model: LearnedZeroShotSegmenter,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("clip.")
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Zero-shot checkpoint structure mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_zero_shot_segmenter(
    config: dict[str, Any], device: torch.device
) -> LearnedZeroShotSegmenter:
    checkpoint_path = zero_shot_checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Zero-shot checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != ZERO_SHOT_FORMAT_VERSION:
        raise ValueError("Zero-shot checkpoint format is incompatible")
    expected = zero_shot_config_fingerprint(config)
    if payload.get("config_fingerprint") != expected:
        raise ValueError("Zero-shot checkpoint/config fingerprint mismatch")
    model = LearnedZeroShotSegmenter(config, device)
    load_trainable_zero_shot_state_dict(model, payload["model"])
    model.eval()
    return model
