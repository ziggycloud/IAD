"""Normal-only unseen-category branch built on public frozen OpenCLIP weights.

No anomalous image, synthetic defect, CutPaste sample, or anomaly checkpoint is
used. Patch MoE adapters learn normal visual modes while inference builds a
robust category/view prototype from the unlabeled target category itself.
"""

from __future__ import annotations

import hashlib
import json
import re
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
        "zero_shot": {key: zero_shot.get(key) for key in ("model", "training")},
        "dataset": {
            key: dataset.get(key)
            for key in (
                "type", "train_dir", "categories", "category_limit",
                "image_size", "crop_size",
            )
        },
        "seed": config["experiment"]["seed"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"zero_shot.model.{name} must be a non-empty list")
    prompts = tuple(str(item).strip() for item in value)
    if any(not prompt for prompt in prompts):
        raise ValueError(f"zero_shot.model.{name} contains an empty prompt")
    return prompts


def readable_category(value: str) -> str:
    """Convert folder identifiers into conservative CLIP prompt nouns."""
    name = re.sub(r"[_\-]+", " ", str(value)).strip()
    name = re.sub(r"\b\d+\b", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "industrial object"


class PatchNormalityMoE(nn.Module):
    """Top-k low-rank experts with frozen orthogonal input projections."""

    def __init__(
        self,
        width: int,
        experts: int,
        rank: int,
        top_k: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")
        if experts * rank > width:
            raise ValueError(
                "moe_num_experts * moe_rank must not exceed CLIP width"
            )
        self.experts = experts
        self.top_k = top_k
        self.router = nn.Linear(width, experts)
        basis, _ = torch.linalg.qr(
            torch.randn(width, experts * rank), mode="reduced"
        )
        self.register_buffer(
            "expert_down",
            basis.T.reshape(experts, rank, width),
            persistent=True,
        )
        self.expert_up = nn.Parameter(torch.empty(experts, width, rank))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.normal_(self.expert_up, std=1e-4)
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)

    def forward(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = patches.shape
        flat = patches.reshape(-1, shape[-1])
        routing = self.router(flat).softmax(dim=-1)
        values, indices = routing.topk(self.top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gates = torch.zeros_like(routing).scatter(1, indices, values)
        residual = torch.zeros_like(flat)
        for expert in range(self.experts):
            selected = torch.nonzero(
                gates[:, expert] > 0, as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            hidden = F.gelu(
                F.linear(flat[selected], self.expert_down[expert])
            )
            update = F.linear(hidden, self.expert_up[expert])
            residual.index_add_(
                0, selected, update * gates[selected, expert, None]
            )
        importance = routing.mean(dim=0)
        balance = importance.var(unbiased=False) / importance.mean().square().clamp_min(1e-6)
        return (
            flat + self.residual_scale * residual
        ).reshape(shape), balance

    def diversity_loss(self) -> torch.Tensor:
        vectors = F.normalize(
            self.expert_up.flatten(1), dim=-1, eps=1e-6
        )
        gram = vectors @ vectors.T
        eye = torch.eye(
            self.experts, device=gram.device, dtype=gram.dtype
        )
        return ((gram - eye) * (1.0 - eye)).square().mean()


class LearnedZeroShotSegmenter(nn.Module):
    """Category-conditioned normal-only segmenter around frozen OpenCLIP."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "The zero-shot branch requires open_clip_torch. "
                "Run: pip install -r requirements.txt"
            ) from exc

        cfg = config["zero_shot"]["model"]
        self.device_spec = device
        self.image_size = int(cfg.get("image_size", 448))
        self.intermediate_layers = int(
            cfg.get("intermediate_layers", 4)
        )
        self.semantic_temperature = float(
            cfg.get("semantic_temperature", 0.07)
        )
        self.paa_scales = tuple(
            int(value) for value in cfg.get("paa_scales", [1, 3, 5])
        )
        if any(value <= 0 or value % 2 == 0 for value in self.paa_scales):
            raise ValueError(
                "paa_scales must contain positive odd integers"
            )
        model_name = str(cfg.get("model_name", "ViT-B-16"))
        pretrained = str(cfg.get("pretrained", "openai"))
        cache_dir = resolve_path(
            str(cfg.get("weights_dir", "third_party/OpenCLIP/weights"))
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            force_quick_gelu=bool(
                cfg.get("force_quick_gelu", pretrained == "openai")
            ),
            force_image_size=self.image_size,
            cache_dir=str(cache_dir),
        )
        clip.eval().requires_grad_(False)
        if not hasattr(
            getattr(clip, "visual", None), "forward_intermediates"
        ):
            raise TypeError(
                f"OpenCLIP model {model_name!r} has no spatial intermediates"
            )
        self.clip = clip
        self.tokenizer = open_clip.get_tokenizer(model_name)
        preprocess = open_clip.get_model_preprocess_cfg(clip)
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
            torch.tensor(preprocess["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(preprocess["std"]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.prompt_sets = (
            _prompt_list(
                cfg.get("background_prompts"), "background_prompts"
            ),
            _prompt_list(cfg.get("normal_prompts"), "normal_prompts"),
            _prompt_list(cfg.get("broken_prompts"), "broken_prompts"),
        )
        self.class_prompt_templates = tuple(
            _prompt_list(cfg.get(key), key)
            for key in (
                "class_background_prompts",
                "class_normal_prompts",
                "class_broken_prompts",
            )
        )
        generic = torch.stack(
            [self._encode_prompts(items).mean(0) for items in self.prompt_sets]
        )
        generic = F.normalize(generic, dim=-1, eps=1e-6)
        self.register_buffer("prompt_anchors", generic, persistent=True)
        width = int(generic.shape[-1])
        self.prompt_delta = nn.Parameter(torch.zeros(3, width))
        hidden = int(cfg.get("adapter_hidden_dim", 192))
        dropout = float(cfg.get("dropout", 0.1))
        self.textual_adapter = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        nn.init.zeros_(self.textual_adapter[-1].weight)
        nn.init.zeros_(self.textual_adapter[-1].bias)

        weights = torch.tensor(
            cfg.get(
                "intermediate_layer_weights", [0.1, 0.2, 0.3, 0.4]
            ),
            dtype=torch.float32,
        )
        if weights.numel() != self.intermediate_layers:
            raise ValueError(
                "intermediate_layer_weights must match intermediate_layers"
            )
        self.layer_logits = nn.Parameter(
            (weights / weights.sum()).clamp_min(1e-6).log()
        )
        self.patch_moe = PatchNormalityMoE(
            width=width,
            experts=int(cfg.get("moe_num_experts", 4)),
            rank=int(cfg.get("moe_rank", 32)),
            top_k=int(cfg.get("moe_top_k", 2)),
            residual_scale=float(cfg.get("moe_residual_scale", 0.1)),
        )
        compact_dim = int(cfg.get("prototype_dim", 64))
        if not 0 < compact_dim <= width:
            raise ValueError(
                "prototype_dim must be in [1, CLIP text width]"
            )
        compact_basis, _ = torch.linalg.qr(
            torch.randn(width, compact_dim), mode="reduced"
        )
        self.register_buffer(
            "compact_projection", compact_basis.T, persistent=True
        )
        self._category_anchor_cache: dict[str, torch.Tensor] = {}
        self.to(device)
        self.clip.eval()

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device_spec)
        return F.normalize(
            self.clip.encode_text(tokens).float(), dim=-1, eps=1e-6
        )

    @torch.no_grad()
    def _base_category_anchors(self, category: str) -> torch.Tensor:
        key = readable_category(category)
        cached = self._category_anchor_cache.get(key)
        if cached is not None:
            return cached
        anchors = []
        for generic, templates in zip(
            self.prompt_sets, self.class_prompt_templates, strict=True
        ):
            prompts = list(generic) + [
                template.format(class_name=key) for template in templates
            ]
            anchors.append(self._encode_prompts(prompts).mean(0))
        result = F.normalize(torch.stack(anchors), dim=-1, eps=1e-6)
        self._category_anchor_cache[key] = result
        return result

    def learned_prompts(self, categories: Sequence[str]) -> torch.Tensor:
        base = torch.stack(
            [self._base_category_anchors(value) for value in categories]
        )
        return F.normalize(
            base + self.prompt_delta + self.textual_adapter(base),
            dim=-1,
            eps=1e-6,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def _project_patches(self, feature: torch.Tensor) -> torch.Tensor:
        patches = feature.float().permute(0, 2, 3, 1)
        projection = getattr(self.clip.visual, "proj", None)
        if isinstance(projection, nn.Module):
            patches = projection(patches)
        elif isinstance(projection, torch.Tensor):
            patches = patches @ projection
        elif patches.shape[-1] != self.prompt_anchors.shape[-1]:
            raise RuntimeError("CLIP patch width does not match text width")
        return F.normalize(patches.float(), dim=-1, eps=1e-6)

    def _clip_features(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        rgb = (
            images.float() * self.imagenet_std + self.imagenet_mean
        ).clamp(0, 1)
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(
                rgb,
                (self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        clip_images = (rgb - self.clip_mean) / self.clip_std
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
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
            raise RuntimeError(
                "OpenCLIP returned unexpected spatial intermediates"
            )
        return features, rgb

    @staticmethod
    def _color_foreground(
        rgb: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        small = F.interpolate(
            rgb, size=size, mode="bilinear", align_corners=False
        )
        border = torch.cat(
            (
                small[:, :, 0, :],
                small[:, :, -1, :],
                small[:, :, :, 0],
                small[:, :, :, -1],
            ),
            dim=-1,
        )
        background = border.median(dim=-1).values[:, :, None, None]
        distance = (
            (small - background).square().mean(dim=1, keepdim=True).sqrt()
        )
        flat = distance.flatten(1)
        low = torch.quantile(flat, 0.20, dim=1)[:, None, None, None]
        high = torch.quantile(flat, 0.70, dim=1)[:, None, None, None]
        return torch.sigmoid(
            (distance - low) / (0.25 * (high - low).clamp_min(1e-4))
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        categories: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if categories is None:
            categories = ["industrial object"] * images.shape[0]
        if len(categories) != images.shape[0]:
            raise ValueError("categories length must match image batch")
        features, rgb = self._clip_features(images)
        layers = torch.stack(
            [self._project_patches(value) for value in features], dim=0
        )
        weights = self.layer_logits.softmax(0).to(layers.dtype)
        patches = torch.einsum("l,lbhwc->bhwc", weights, layers)
        patches, balance = self.patch_moe(patches)
        patches = F.normalize(patches, dim=-1, eps=1e-6)
        _, height, width, _ = patches.shape
        nchw = patches.permute(0, 3, 1, 2)
        pooled = [
            F.avg_pool2d(
                nchw, scale, stride=1, padding=scale // 2
            )
            for scale in self.paa_scales
        ]
        patches = F.normalize(
            torch.stack(pooled).mean(0).permute(0, 2, 3, 1),
            dim=-1,
            eps=1e-6,
        )
        prompts = self.learned_prompts(categories).to(patches.dtype)
        class_logits = torch.einsum(
            "bhwc,bkc->bkhw", patches, prompts
        ) / self.semantic_temperature
        class_probability = class_logits.softmax(dim=1)
        color_fg = self._color_foreground(rgb, (height, width))
        foreground = torch.sqrt(
            (color_fg * (1.0 - class_probability[:, 0:1])).clamp_min(0)
        )
        semantic_anomaly = class_probability[:, 2:3]
        probability = (foreground * semantic_anomaly).clamp(1e-6, 1 - 1e-6)
        global_feature = F.normalize(
            patches.mean((1, 2)), dim=-1, eps=1e-6
        )
        global_logits = torch.einsum(
            "bc,bkc->bk", global_feature, prompts
        ) / self.semantic_temperature
        compact = F.normalize(
            F.linear(patches, self.compact_projection), dim=-1, eps=1e-6
        )
        return {
            "logits": torch.logit(probability).float(),
            "probability": probability.float(),
            "semantic_probability": semantic_anomaly.float(),
            "class_logits": class_logits.float(),
            "global_logits": global_logits.float(),
            "foreground_probability": foreground.float(),
            "color_foreground_probability": color_fg.float(),
            "normal_features": compact.float(),
            "global_feature": global_feature.float(),
            "moe_balance_loss": balance.float(),
            "moe_diversity_loss": self.patch_moe.diversity_loss().float(),
            "layer_weights": weights.float(),
        }


@torch.no_grad()
def score_normal_only_category(
    normal_features: torch.Tensor,
    semantic_probability: torch.Tensor,
    foreground_probability: torch.Tensor,
    view_ids: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build robust target-category prototypes and return absolute maps/scores.

    Tensors are intentionally accepted on CPU: a 64-D FP16 cache keeps a full
    category small and avoids retaining CLIP activations on the GPU.
    """
    infer = config["zero_shot"].get("inference", {})
    features = F.normalize(normal_features.float(), dim=-1, eps=1e-6)
    semantic = semantic_probability.float().squeeze(1)
    foreground = foreground_probability.float().squeeze(1)
    result = torch.zeros_like(semantic)
    retain_ratio = float(infer.get("prototype_retain_ratio", 0.6))
    minimum = int(infer.get("prototype_min_samples", 3))
    foreground_threshold = float(infer.get("foreground_threshold", 0.35))
    distance_quantile = float(infer.get("normal_distance_quantile", 0.99))
    semantic_quantile = float(infer.get("normal_semantic_quantile", 0.95))
    prototype_gain = float(infer.get("prototype_gain", 4.0))
    semantic_gain = float(infer.get("semantic_gain", 0.5))
    decision_bias = float(infer.get("decision_bias", 0.5))
    foreground_power = float(infer.get("foreground_power", 1.5))

    for view_id in torch.unique(view_ids, sorted=True):
        indices = torch.nonzero(view_ids == view_id, as_tuple=False).flatten()
        current = features[indices]
        current_fg = foreground[indices]
        weights = current_fg[..., None]
        global_features = F.normalize(
            (current * weights).sum((1, 2))
            / weights.sum((1, 2)).clamp_min(1e-6),
            dim=-1,
            eps=1e-6,
        )
        center = F.normalize(
            global_features.median(dim=0).values, dim=-1, eps=1e-6
        )
        global_distance = 1.0 - global_features @ center
        global_distance = global_distance + 0.1 * semantic[indices].flatten(1).mean(1)
        keep_count = min(
            indices.numel(),
            max(minimum, int(round(indices.numel() * retain_ratio))),
        )
        selected = global_distance.argsort()[:keep_count]
        prototype = F.normalize(
            current[selected].mean(dim=0), dim=-1, eps=1e-6
        )
        distance = 1.0 - (current * prototype).sum(dim=-1)
        selected_fg = current_fg[selected] >= foreground_threshold
        reference_distance = distance[selected][selected_fg]
        if reference_distance.numel() < 16:
            reference_distance = distance[selected].flatten()
        distance_center = torch.quantile(
            reference_distance, distance_quantile
        )
        distance_median = reference_distance.median()
        distance_scale = (
            distance_center - distance_median
        ).clamp_min(1e-3)
        distance_z = (distance - distance_center) / distance_scale

        reference_semantic = semantic[indices][selected][selected_fg]
        if reference_semantic.numel() < 16:
            reference_semantic = semantic[indices][selected].flatten()
        semantic_center = torch.quantile(
            reference_semantic, semantic_quantile
        )
        semantic_scale = (
            semantic_center - reference_semantic.median()
        ).clamp_min(0.02)
        semantic_z = (
            semantic[indices] - semantic_center
        ) / semantic_scale
        logits = (
            prototype_gain * distance_z
            + semantic_gain * semantic_z
            - decision_bias
        )
        result[indices] = (
            current_fg.pow(foreground_power) * logits.sigmoid()
        ).clamp(0, 1)

    ratio = float(infer.get("image_top_ratio", 0.01))
    flat = result.flatten(1)
    count = max(1, int(round(flat.shape[1] * ratio)))
    image_scores = flat.topk(count, dim=1).values.mean(dim=1)
    return result[:, None], image_scores


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
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Zero-shot checkpoint structure mismatch: "
            f"missing={missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )


def load_zero_shot_segmenter(
    config: dict[str, Any], device: torch.device
) -> LearnedZeroShotSegmenter:
    path = zero_shot_checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Zero-shot checkpoint does not exist: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != ZERO_SHOT_FORMAT_VERSION:
        raise ValueError(
            "Zero-shot checkpoint format is incompatible; "
            "retrain the normal-only branch"
        )
    if payload.get("config_fingerprint") != zero_shot_config_fingerprint(config):
        raise ValueError("Zero-shot checkpoint/config fingerprint mismatch")
    model = LearnedZeroShotSegmenter(config, device)
    load_trainable_zero_shot_state_dict(model, payload["model"])
    return model.eval()
