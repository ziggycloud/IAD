"""MoECLIP unseen-category branch based on the official CVPR 2026 design."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import resolve_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ZERO_SHOT_FORMAT_VERSION = 5


def zero_shot_checkpoint_path(config: dict[str, Any]) -> Path:
    configured = config.get("zero_shot", {}).get("checkpoint")
    output_dir = Path(config["experiment"]["output_dir"])
    if configured is None:
        return output_dir / "zero_shot" / "checkpoints" / "best_model.pt"
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else output_dir / path


def zero_shot_last_checkpoint_path(config: dict[str, Any]) -> Path:
    return zero_shot_checkpoint_path(config).parent / "last.pt"


def zero_shot_final_checkpoint_path(config: dict[str, Any]) -> Path:
    return zero_shot_checkpoint_path(config).parent / "final_model.pt"


def zero_shot_config_fingerprint(config: dict[str, Any]) -> str:
    zero_shot = config.get("zero_shot", {})
    payload = {
        "format_version": ZERO_SHOT_FORMAT_VERSION,
        "model": zero_shot.get("model"),
        "training": zero_shot.get("training"),
        "seed": config["experiment"]["seed"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def readable_category(value: str) -> str:
    name = re.sub(r"[_\-]+", " ", str(value)).strip()
    return re.sub(r"\s+", " ", name) or "object"


class SimpleProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, relu: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, out_features, bias=False)]
        if relu:
            layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[0].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DetectionProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.depthwise = nn.Conv1d(
            in_features, in_features, 3, padding=1, groups=in_features,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_features, out_features, 1, bias=False)
        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm(value).transpose(1, 2)
        return self.pointwise(F.gelu(self.depthwise(value))).transpose(1, 2)


class FOFSLoRAExpert(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float,
                 fixed_down: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("down", fixed_down, persistent=True)
        self.up = nn.Parameter(torch.zeros(width, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = float(alpha) / float(rank)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(value), self.down), self.up) * self.scale


class PatchRoutedFOFSMoE(nn.Module):
    """Top-k LoRA experts with the paper's frozen orthogonal feature split."""

    def __init__(self, width: int, num_experts: int = 4, rank: int = 8,
                 alpha: float = 16.0, top_k: int = 2,
                 dropout: float = 0.05) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(width, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)
        experts = []
        for indices in torch.tensor_split(torch.arange(width), num_experts):
            if indices.numel() < rank:
                raise ValueError("each FOFS feature partition must cover moe_rank")
            basis, _ = torch.linalg.qr(torch.randn(indices.numel(), rank))
            down = torch.zeros(rank, width)
            down[:, indices] = basis.T
            experts.append(FOFSLoRAExpert(width, rank, alpha, down, dropout))
        self.experts = nn.ModuleList(experts)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = value.shape
        flat = value.reshape(-1, shape[-1])
        probabilities = self.router(flat).softmax(-1, dtype=torch.float32)
        weights, selected = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        all_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        rows = torch.arange(flat.shape[0], device=flat.device)[:, None]
        routed = (all_outputs[rows, selected] * weights.to(flat.dtype)[..., None]).sum(1)
        gate_sum = probabilities.sum(0)
        balance = (gate_sum.std() / gate_sum.mean().clamp_min(1e-6)).square()
        normalized = routed * flat.norm(dim=-1, keepdim=True) / (
            routed.norm(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.reshape(shape), balance, all_outputs


def etf_loss(expert_outputs: torch.Tensor) -> torch.Tensor:
    experts = expert_outputs.shape[1]
    if experts <= 1:
        return expert_outputs.new_zeros(())
    normalized = F.normalize(expert_outputs, dim=-1, eps=1e-6)
    gram = normalized @ normalized.transpose(1, 2)
    target = gram.new_full((experts, experts), -1.0 / (experts - 1))
    target.fill_diagonal_(1.0)
    return F.mse_loss(gram, target.expand_as(gram))


class MoECLIPSegmenter(nn.Module):
    """Patch-MoE, PAA, segmentation adapters and text adapter from MoECLIP."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        cfg = config["zero_shot"]["model"]
        mirror = str(cfg.get("hf_endpoint", "")).strip()
        if mirror:
            os.environ.setdefault("HF_ENDPOINT", mirror)
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("MoECLIP requires open_clip_torch") from exc
        self.device_spec = device
        self.image_size = int(cfg.get("image_size", 518))
        self.levels = tuple(int(x) for x in cfg.get("levels", [6, 12, 18, 24]))
        self.moe_layers = tuple(int(x) for x in cfg.get("moe_layers", [5, 11, 17, 23]))
        if tuple(x - 1 for x in self.levels) != self.moe_layers:
            raise ValueError("official MoECLIP requires moe_layers == levels - 1")
        self.paa_scales = tuple(int(x) for x in cfg.get("paa_scales", [1, 3, 5]))
        self.image_adapt_weight = float(cfg.get("image_adapt_weight", 0.1))
        inference = config["zero_shot"].get("inference", {})
        self.gaussian_kernel_size = int(
            inference.get("gaussian_kernel_size", 7)
        )
        self.gaussian_sigma = float(inference.get("gaussian_sigma", 1.0))
        model_name = str(cfg.get("model_name", "ViT-L-14-336"))
        pretrained = str(cfg.get("pretrained", "openai"))
        cache_dir = resolve_path(str(cfg.get("weights_dir", "third_party/OpenCLIP/weights")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        clip, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
            force_quick_gelu=bool(cfg.get("force_quick_gelu", True)),
            force_image_size=self.image_size, cache_dir=str(cache_dir),
        )
        clip.eval().requires_grad_(False)
        visual = clip.visual
        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if blocks is None or len(blocks) < max(self.levels):
            raise TypeError("selected OpenCLIP backbone is not a compatible ViT-L/14")
        self.clip = clip
        self.tokenizer = open_clip.get_tokenizer(model_name)
        preprocess = open_clip.get_model_preprocess_cfg(clip)
        for name, values in (
            ("imagenet_mean", IMAGENET_MEAN), ("imagenet_std", IMAGENET_STD),
            ("clip_mean", preprocess["mean"]), ("clip_std", preprocess["std"]),
        ):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1), persistent=False)
        width = int(visual.conv1.out_channels)
        text_width = int(clip.text_projection.shape[-1])
        self.moe_adapters = nn.ModuleList([
            PatchRoutedFOFSMoE(
                width, int(cfg.get("moe_num_experts", 4)),
                int(cfg.get("moe_rank", 8)), float(cfg.get("moe_lora_alpha", 16)),
                int(cfg.get("moe_top_k", 2)), float(cfg.get("moe_dropout", 0.05)),
            ) for _ in self.moe_layers
        ])
        self.seg_projections = nn.ModuleList([
            SimpleProjection(width, text_width, bool(cfg.get("relu", True)))
            for _ in self.levels
        ])
        self.det_projection = DetectionProjection(width, text_width)
        self.text_adapter = SimpleProjection(text_width, text_width, True)
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.to(device)
        self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        rgb = (images.float() * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(rgb, (self.image_size, self.image_size), mode="bicubic",
                                align_corners=False, antialias=True)
        return (rgb - self.clip_mean) / self.clip_std

    @staticmethod
    def _run_block(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
        try:
            result = block(value, attn_mask=None)
        except TypeError:
            result = block(value)
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _paa(value: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return value
        cls, patches = value[:, :1], value[:, 1:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("MoECLIP PAA requires a square patch grid")
        spatial = patches.reshape(value.shape[0], side, side, -1).permute(0, 3, 1, 2)
        spatial = F.avg_pool2d(spatial, scale, stride=1, padding=scale // 2)
        return torch.cat((cls, spatial.permute(0, 2, 3, 1).reshape(value.shape[0], -1, value.shape[-1])), 1)

    def _blur_patch_map(self, value: torch.Tensor) -> torch.Tensor:
        size = self.gaussian_kernel_size
        coordinates = torch.arange(
            size, device=value.device, dtype=value.dtype
        ) - (size - 1) / 2
        kernel = torch.exp(
            -coordinates.square() / (2.0 * self.gaussian_sigma ** 2)
        )
        kernel = kernel / kernel.sum()
        kernel_2d = (kernel[:, None] * kernel[None, :]).reshape(1, 1, size, size)
        padding = size // 2
        return F.conv2d(
            F.pad(value, (padding,) * 4, mode="reflect"), kernel_2d
        )

    def _visual_forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.clip.visual
        x = visual.conv1(images).reshape(images.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), 1)
        positional = visual.positional_embedding.to(x.dtype)
        if positional.shape[0] != x.shape[1]:
            raise ValueError("OpenCLIP positional embedding does not match image_size")
        x = visual.ln_pre(visual.patch_dropout(x + positional))
        transformer = visual.transformer
        batch_first = bool(getattr(transformer, "batch_first", False))
        hidden = x if batch_first else x.transpose(0, 1)
        captured: list[torch.Tensor] = []
        balance = x.new_zeros(())
        diversity = x.new_zeros(())
        for index, block in enumerate(transformer.resblocks):
            hidden = self._run_block(block, hidden)
            if index in self.moe_layers:
                adapter_index = self.moe_layers.index(index)
                batch_tokens = hidden if batch_first else hidden.transpose(0, 1)
                update, current_balance, expert_outputs = self.moe_adapters[adapter_index](batch_tokens)
                batch_tokens = self.image_adapt_weight * update + (1.0 - self.image_adapt_weight) * batch_tokens
                hidden = batch_tokens if batch_first else batch_tokens.transpose(0, 1)
                balance = balance + current_balance
                diversity = diversity + etf_loss(expert_outputs)
            if index + 1 in self.levels:
                captured.append(hidden if batch_first else hidden.transpose(0, 1))
        tokens = [self._paa(level, scale) for level in captured for scale in self.paa_scales]
        tokens = [visual.ln_post(value)[:, 1:] for value in tokens]
        projected = [
            F.normalize(self.seg_projections[index // len(self.paa_scales)](value), dim=-1, eps=1e-6)
            for index, value in enumerate(tokens)
        ]
        detection = F.normalize(
            self.det_projection(tokens[-len(self.paa_scales)]), dim=-1, eps=1e-6
        ).mean(1)
        return projected, detection, balance, diversity

    @torch.no_grad()
    def _encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        clip = self.clip
        value = clip.token_embedding(tokens).to(clip.transformer.get_cast_dtype())
        value = value + clip.positional_embedding.to(value.dtype)
        transformer = clip.transformer
        batch_first = bool(getattr(transformer, "batch_first", False))
        hidden = value if batch_first else value.transpose(0, 1)
        for block in transformer.resblocks:
            try:
                result = block(hidden, attn_mask=clip.attn_mask)
            except TypeError:
                result = block(hidden)
            hidden = result[0] if isinstance(result, tuple) else result
        value = hidden if batch_first else hidden.transpose(0, 1)
        value = clip.ln_final(value)
        return value[torch.arange(value.shape[0], device=value.device), tokens.argmax(-1)]

    @torch.no_grad()
    def _base_text(self, category: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = readable_category(category)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        states = (
            [key, f"a {key}", f"the {key}"],
            [f"a damaged {key}", f"a broken {key}", f"a {key} with flaw",
             f"a {key} with defect", f"a {key} with damage"],
        )
        features: list[torch.Tensor] = []
        for state_group in states:
            prompts = [template.format(state) for state in state_group
                       for template in ("{}.", "a photo of {}.")]
            tokens = self.tokenizer(prompts).to(self.device_spec)
            features.append(self._encode_text_tokens(tokens).float())
        result = (features[0], features[1])
        self._text_cache[key] = result
        return result

    def text_features(self, categories: Sequence[str]) -> torch.Tensor:
        rows = []
        for category in categories:
            states = self._base_text(category)
            rows.append(torch.stack([
                F.normalize(
                    F.normalize(self.text_adapter(state), dim=-1, eps=1e-6).mean(0),
                    dim=-1,
                    eps=1e-6,
                )
                for state in states
            ]))
        return torch.stack(rows)

    def forward(self, images: torch.Tensor, *, categories: Sequence[str] | None = None) -> dict[str, Any]:
        categories = list(categories or ["object"] * images.shape[0])
        if len(categories) != images.shape[0]:
            raise ValueError("categories length must match image batch")
        clip_images = self._prepare_images(images)
        patch_features, detection, balance, diversity = self._visual_forward(clip_images)
        text = self.text_features(categories).to(detection.dtype)
        patch_logits = [100.0 * torch.einsum("bld,bkd->bkl", feature, text)
                        for feature in patch_features]
        side = math.isqrt(patch_logits[0].shape[-1])
        patch_probabilities = [
            F.interpolate(
                logits.reshape(images.shape[0], 2, side, side),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=True,
            ).softmax(1)
            for logits in patch_logits
        ]
        image_logits = torch.einsum("bd,bkd->bk", detection, text)
        if self.training:
            anomaly_maps = torch.stack(
                [value[:, 1:2] for value in patch_probabilities]
            ).mean(0)
        else:
            raw_maps = []
            for logits in patch_logits:
                logits = logits.reshape(images.shape[0], 2, side, side)
                raw = (logits[:, 1:2] + 1.0 - logits[:, 0:1]) / 2.0
                raw = self._blur_patch_map(raw)
                raw_maps.append(F.interpolate(
                    raw, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=True,
                ))
            anomaly_maps = torch.stack(raw_maps).sum(0)
        image_score = (image_logits.float()[:, 1] + 1.0) / 2.0
        return {
            "patch_logits": [value.float() for value in patch_logits],
            "patch_probabilities": [value.float() for value in patch_probabilities],
            "image_logits": image_logits.float(),
            "probability": anomaly_maps.float(),
            "image_score": image_score.float(),
            "moe_balance_loss": balance.float(),
            "moe_etf_loss": diversity.float(),
        }


LearnedZeroShotSegmenter = MoECLIPSegmenter


def trainable_zero_shot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith("clip.")}


def load_trainable_zero_shot_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise ValueError(f"unknown MoECLIP checkpoint keys: {unknown[:5]}")
    current.update(state)
    model.load_state_dict(current, strict=True)


def load_zero_shot_segmenter(config: dict[str, Any], device: torch.device) -> MoECLIPSegmenter:
    path = zero_shot_checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"MoECLIP checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", -1)) != ZERO_SHOT_FORMAT_VERSION:
        raise ValueError("MoECLIP checkpoint format mismatch; retraining is required")
    if payload.get("config_fingerprint") != zero_shot_config_fingerprint(config):
        raise ValueError("MoECLIP checkpoint/config fingerprint mismatch")
    model = MoECLIPSegmenter(config, device)
    load_trainable_zero_shot_state_dict(model, payload["model"])
    return model.eval()
