from __future__ import annotations

import sys
import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import PROJECT_ROOT
from .generalized_model import (
    CompositionalNormalityAdapter,
    GeneralizedDinomaly,
    combine_auxiliary_losses,
)


UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Dinomaly2"


def _load_upstream_symbols() -> dict[str, Any]:
    if not UPSTREAM_ROOT.is_dir():
        raise FileNotFoundError(
            f"缺少 Dinomaly2 上游源码：{UPSTREAM_ROOT}。"
            "请先克隆 guojiajeremy/Dinomaly2。"
        )
    upstream = str(UPSTREAM_ROOT)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)

    from dinov2.models import vision_transformer as vision_transformer_dinov2
    from models.uad import Dinomaly
    from models.vision_transformer import (
        Attention,
        Block,
        LinearAttention2,
    )
    from optimizers.StableAdamW import StableAdamW

    return {
        "vision_transformer_dinov2": vision_transformer_dinov2,
        "Dinomaly": Dinomaly,
        "Attention": Attention,
        "Block": Block,
        "LinearAttention2": LinearAttention2,
        "StableAdamW": StableAdamW,
    }


def _load_dinov2_register_backbone(
    symbols: dict[str, Any],
    backbone_name: str,
    weights_dir: Path,
) -> tuple[nn.Module, Path, str]:
    if not backbone_name.startswith("dinov2reg_vit_"):
        raise ValueError(
            "蒸馏版仅保留 DINOv2-register backbone，收到："
            f"{backbone_name}"
        )
    try:
        _, _, size, patch = backbone_name.split("_")
    except ValueError as exc:
        raise ValueError(f"无法解析 backbone 名称：{backbone_name}") from exc
    patch_size = int(patch)
    if patch_size != 14:
        raise ValueError("当前 DINOv2-register 权重仅配置 patch size 14")
    compact = {
        "small": "s",
        "base": "b",
        "large": "l",
    }.get(size)
    if compact is None:
        raise ValueError(f"不支持的 backbone size：{size}")

    constructor = symbols["vision_transformer_dinov2"].__dict__[f"vit_{size}"]
    encoder = constructor(
        patch_size=patch_size,
        img_size=518,
        block_chunks=0,
        init_values=1e-8,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    filename = f"dinov2_vit{compact}{patch_size}_reg4_pretrain.pth"
    checkpoint_path = weights_dir / filename
    if not checkpoint_path.is_file():
        from torch.hub import download_url_to_file

        url = (
            "https://dl.fbaipublicfiles.com/dinov2/"
            f"dinov2_vit{compact}{patch_size}/{filename}"
        )
        download_url_to_file(url, str(checkpoint_path), progress=True)
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    incompatible = encoder.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DINOv2 backbone 权重结构不匹配："
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return encoder, checkpoint_path, digest.hexdigest()


def loose_constraint_layers(mode: int) -> tuple[list[list[int]], list[list[int]]]:
    mappings = {
        0: [[0], [1], [2], [3], [4], [5], [6], [7]],
        1: [[0, 1, 2, 3, 4, 5, 6, 7]],
        2: [[0, 1, 2, 3], [4, 5, 6, 7]],
        3: [[0, 1, 2], [3, 4, 5], [6, 7]],
        4: [[0, 1], [2, 3], [4, 5], [6, 7]],
        11: [[7]],
        12: [[3], [7]],
        14: [[1], [3], [5], [7]],
    }
    if mode not in mappings:
        raise ValueError(f"不支持 loose_constraint_groups={mode}")
    groups = mappings[mode]
    return groups, [list(group) for group in groups]


@dataclass
class ModelBundle:
    model: nn.Module
    bottleneck: nn.ModuleList
    decoder: nn.ModuleList
    embed_dim: int
    backbone_name: str
    backbone_weights_path: Path
    backbone_sha256: str

    @property
    def trainable(self) -> nn.ModuleList:
        return nn.ModuleList([self.bottleneck, self.decoder])


def build_model(config: dict[str, Any], device: torch.device) -> ModelBundle:
    symbols = _load_upstream_symbols()
    model_config = config["model"]
    backbone_name = str(model_config["backbone"])
    weights_dir = Path(model_config["backbone_weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    encoder, weights_path, weights_sha256 = _load_dinov2_register_backbone(
        symbols,
        backbone_name,
        weights_dir,
    )

    if "small" in backbone_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in backbone_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in backbone_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(
            f"backbone 名称必须包含 small/base/large：{backbone_name}"
        )

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    dropout = float(model_config["dropout"])
    architecture = str(model_config.get("architecture", "dinomaly2")).lower()
    generalized_names = {"generalized", "category_generalized", "cg_dinomaly"}
    if architecture in generalized_names:
        generalized = dict(model_config.get("generalized", {}))
        latent_dim = int(generalized.get("latent_dim", 256))
        if latent_dim <= 0:
            raise ValueError("model.generalized.latent_dim must be positive")
        bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, latent_dim),
                    nn.Dropout(p=dropout),
                ),
                CompositionalNormalityAdapter(
                    latent_dim=latent_dim,
                    output_dim=embed_dim,
                    num_special_tokens=1 + int(encoder.num_register_tokens),
                    num_references=int(
                        generalized.get("num_references", 256)
                    ),
                    reference_top_k=int(
                        generalized.get("reference_top_k", 16)
                    ),
                    reference_temperature=float(
                        generalized.get("reference_temperature", 0.07)
                    ),
                    router_temperature=float(
                        generalized.get("router_temperature", 1.0)
                    ),
                    router_top_k=int(generalized.get("router_top_k", 3)),
                    router_trim_ratio=float(
                        generalized.get("router_trim_ratio", 0.1)
                    ),
                    dropout=float(
                        generalized.get("expert_dropout", dropout)
                    ),
                ),
            ]
        )
    elif architecture in {"dinomaly", "dinomaly2", "baseline"}:
        bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, 256),
                    nn.Dropout(p=dropout),
                ),
                nn.Sequential(
                    nn.Linear(256, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(p=dropout),
                ),
            ]
        )
    else:
        raise ValueError(
            "model.architecture must be dinomaly2 or generalized, "
            f"got {architecture!r}"
        )

    decoder = nn.ModuleList()
    attention = (
        partial(symbols["LinearAttention2"], eps=1e-8)
        if bool(model_config["linear_attention"])
        else symbols["Attention"]
    )
    for _ in range(8):
        decoder.append(
            symbols["Block"](
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=attention,
            )
        )

    encoder_fusion, decoder_fusion = loose_constraint_layers(
        int(model_config["loose_constraint_groups"])
    )
    model_class = (
        GeneralizedDinomaly
        if architecture in generalized_names
        else symbols["Dinomaly"]
    )
    model = model_class(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=encoder_fusion,
        fuse_layer_decoder=decoder_fusion,
        context_aware_recenter=bool(model_config["context_aware_recenter"]),
    )
    model.init_weights()
    model.to(device)
    return ModelBundle(
        model=model,
        bottleneck=bottleneck,
        decoder=decoder,
        embed_dim=embed_dim,
        backbone_name=backbone_name,
        backbone_weights_path=weights_path,
        backbone_sha256=weights_sha256,
    )


def build_optimizer(bundle: ModelBundle, config: dict[str, Any]):
    symbols = _load_upstream_symbols()
    train_config = config["training"]
    base_lr = float(train_config["learning_rate"])
    first_lr = base_lr * float(train_config["first_bottleneck_lr_scale"])
    betas = tuple(float(value) for value in train_config["adam_betas"])
    optimizer = symbols["StableAdamW"](
        [
            {
                "params": bundle.bottleneck[0].parameters(),
                "lr": first_lr,
                "initial_lr": first_lr,
                "group_name": "bottleneck_first",
            },
            {
                "params": bundle.bottleneck[1].parameters(),
                "lr": base_lr,
                "initial_lr": base_lr,
                "group_name": "bottleneck_rest",
            },
            {
                "params": bundle.decoder.parameters(),
                "lr": base_lr,
                "initial_lr": base_lr,
                "group_name": "decoder",
            },
        ],
        lr=base_lr,
        betas=betas,
        weight_decay=float(train_config["weight_decay"]),
        amsgrad=False,
        eps=float(train_config["adam_epsilon"]),
    )
    return optimizer


def set_learning_rate(
    optimizer,
    completed_steps: int,
    total_steps: int,
    warmup_steps: int,
    final_ratio: float,
    step_offset: int = 1,
) -> list[float]:
    """Set the LR for the next optimizer step, following the paper schedule."""
    import math

    schedule_step = completed_steps + step_offset
    if schedule_step < 0:
        raise ValueError("schedule_step 不能为负数")
    if warmup_steps > 0 and schedule_step <= warmup_steps:
        factor = schedule_step / warmup_steps
    else:
        denominator = max(1, total_steps - warmup_steps)
        progress = min(
            1.0,
            max(0.0, (schedule_step - warmup_steps) / denominator),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = final_ratio + (1.0 - final_ratio) * cosine

    values: list[float] = []
    for group in optimizer.param_groups:
        initial_lr = float(group["initial_lr"])
        group["lr"] = initial_lr * factor
        values.append(float(group["lr"]))
    return values


def trainable_state_dict(bundle: ModelBundle) -> dict[str, Any]:
    return {
        "bottleneck": bundle.bottleneck.state_dict(),
        "decoder": bundle.decoder.state_dict(),
    }


def load_trainable_state_dict(
    bundle: ModelBundle,
    state: dict[str, Any],
    strict: bool = True,
) -> None:
    bundle.bottleneck.load_state_dict(state["bottleneck"], strict=strict)
    bundle.decoder.load_state_dict(state["decoder"], strict=strict)


def parameter_summary(bundle: ModelBundle) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in bundle.model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def auxiliary_losses(
    model: nn.Module,
    detach: bool = False,
) -> dict[str, torch.Tensor]:
    """Read optional architecture losses without special-casing the baseline.

    This deliberately returns an empty mapping for the original Dinomaly2
    model, making it safe for shared training and logging code.
    """

    unwrapped = getattr(model, "module", model)
    getter = getattr(unwrapped, "auxiliary_losses", None)
    if getter is None:
        return {}
    return getter(detach=detach)


def forward_with_regularization(
    model: nn.Module,
    images: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Forward either architecture and return a uniform auxiliary contract.

    GeneralizedDinomaly returns its per-replica terms through ``forward`` so
    ``DataParallel`` can gather graph-connected values.  Each gathered vector
    is reduced to one scalar here.  Baseline Dinomaly2 gets an exact device-side
    zero and an empty mapping.
    """

    unwrapped = getattr(model, "module", model)
    if isinstance(unwrapped, GeneralizedDinomaly):
        encoder_features, decoder_features, gathered = model(
            images,
            return_auxiliary=True,
        )
        reduced = {name: value.mean() for name, value in gathered.items()}
        regularizer = combine_auxiliary_losses(
            reduced,
            weights=weights,
            anchor=next(iter(reduced.values())),
        )
        return encoder_features, decoder_features, regularizer, reduced

    encoder_features, decoder_features = model(images)
    zero = decoder_features[0].new_zeros(())
    return encoder_features, decoder_features, zero, {}


def regularization_loss(
    model: nn.Module,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return a differentiable scalar, or zero for the baseline architecture."""

    unwrapped = getattr(model, "module", model)
    getter = getattr(unwrapped, "regularization_loss", None)
    if getter is not None:
        return getter(weights=weights)
    try:
        parameter = next(unwrapped.parameters())
    except StopIteration:
        return torch.tensor(0.0)
    return parameter.sum() * 0.0
