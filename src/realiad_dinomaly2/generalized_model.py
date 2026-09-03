"""Category-generalized Dinomaly building blocks.

The modules in this file deliberately do not consume category IDs.  A frozen
encoder feature is projected into a compact space, reconstructed from a
learnable bank of normal primitives, and then processed by soft-routed experts
that specialize by spatial scale rather than by object category.

The public model keeps Dinomaly's ``(encoder_features, decoder_features)``
return contract so the existing reconstruction loss and anomaly-map code can
be reused unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_AUXILIARY_WEIGHTS: dict[str, float] = {
    "reference_balance": 1.0,
    "reference_assignment_entropy": 0.01,
    "router_balance": 1.0,
    "router_entropy_penalty": 0.1,
    "expert_diversity": 0.1,
    "context_consistency": 0.01,
    "context_variance": 0.001,
    "visibility_balance": 0.001,
    "attention_entropy": 0.001,
}


def combine_auxiliary_losses(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine scalar (or DataParallel-gathered vector) auxiliary terms."""

    selected = dict(DEFAULT_AUXILIARY_WEIGHTS)
    if weights is not None:
        selected.update({name: float(value) for name, value in weights.items()})
    if anchor is None:
        if losses:
            anchor = next(iter(losses.values()))
        else:
            return torch.tensor(0.0)
    total = anchor.sum() * 0.0
    for name, value in losses.items():
        # DataParallel concatenates the per-replica [1] tensors to [num_gpus].
        # DDP and single-device execution both arrive here with one element.
        total = total + selected.get(name, 0.0) * value.mean()
    return total


def _robust_token_context(
    tokens: torch.Tensor,
    trim_ratio: float,
) -> torch.Tensor:
    """Return a category-agnostic context after dropping extreme tokens.

    Selection is based on token norm.  Defects tend to occupy a small number
    of patches, so excluding both tails makes the router less sensitive to an
    anomalous patch while retaining gradients through all selected tokens.
    """

    if tokens.ndim != 3:
        raise ValueError(f"expected [B, N, C] tokens, got {tuple(tokens.shape)}")
    token_count = tokens.shape[1]
    trim = int(token_count * max(0.0, min(float(trim_ratio), 0.45)))
    if trim == 0 or token_count - 2 * trim < 1:
        return tokens.mean(dim=1)

    # Sorting only determines the (non-differentiable) membership.  Gathered
    # token values retain their normal gradient path.
    scores = tokens.detach().float().square().mean(dim=-1)
    order = scores.argsort(dim=1)
    kept = order[:, trim : token_count - trim]
    kept = kept.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=kept).mean(dim=1)


@dataclass
class MultiViewContextOutput:
    """Outputs of the category-free multi-view context path."""

    object_context: torch.Tensor
    cross_view_context: torch.Tensor
    robust_view_context: torch.Tensor
    token_dispersion: torch.Tensor
    visibility_weights: torch.Tensor
    attention_weights: torch.Tensor
    attention_entropy_penalty: torch.Tensor


class SetAttentionBlock(nn.Module):
    """Self-attention over camera contexts with explicit missing-view masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("SetAttentionBlock dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        contexts: torch.Tensor,
        valid_view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if contexts.ndim != 3:
            raise ValueError("contexts must have shape [B, V, C]")
        if valid_view_mask.shape != contexts.shape[:2]:
            raise ValueError("valid_view_mask must have shape [B, V]")
        if bool((~valid_view_mask.any(dim=1)).any()):
            raise ValueError("every object must contain at least one valid view")
        normalized = self.input_norm(contexts)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_view_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        contexts = contexts + self.attention_dropout(attended)
        contexts = contexts + self.mlp(self.output_norm(contexts))
        contexts = contexts * valid_view_mask.unsqueeze(-1).to(contexts.dtype)
        return contexts, weights


class VisibilityAwareCrossViewFusion(nn.Module):
    """Produce per-view conditioning and a reliability-weighted set context."""

    def __init__(self, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("visibility temperature must be positive")
        self.temperature = float(temperature)
        self.reliability = nn.Sequential(
            nn.LayerNorm(dim * 3, eps=1e-6),
            nn.Linear(dim * 3, max(32, dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, dim // 2), 1),
        )
        self.cross_view_projection = nn.Sequential(
            nn.LayerNorm(dim * 2, eps=1e-6),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        robust_context: torch.Tensor,
        attended_context: torch.Tensor,
        dispersion: torch.Tensor,
        valid_view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.reliability(
            torch.cat([robust_context, attended_context, dispersion], dim=-1)
        ).squeeze(-1)
        logits = logits.float() / self.temperature
        logits = logits.masked_fill(~valid_view_mask, -torch.inf)
        visibility = F.softmax(logits, dim=1).to(attended_context.dtype)
        object_context = (
            attended_context * visibility.unsqueeze(-1)
        ).sum(dim=1)
        expanded = object_context.unsqueeze(1).expand_as(attended_context)
        cross_view = self.cross_view_projection(
            torch.cat([attended_context, expanded], dim=-1)
        )
        cross_view = cross_view * valid_view_mask.unsqueeze(-1).to(
            cross_view.dtype
        )
        return object_context, cross_view, visibility


class MultiViewContextEncoder(nn.Module):
    """Robust pooling + Set Transformer over a fixed set of camera views."""

    def __init__(
        self,
        dim: int,
        num_views: int = 5,
        num_layers: int = 2,
        num_heads: int = 6,
        trim_ratio: float = 0.1,
        view_embedding: bool = True,
        view_dropout_probability: float = 0.2,
        visibility_temperature: float = 1.0,
        dropout: float = 0.1,
        variance_target: float = 1.0,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= view_dropout_probability < 1.0:
            raise ValueError("view_dropout_probability must be in [0, 1)")
        self.dim = int(dim)
        self.num_views = int(num_views)
        self.trim_ratio = float(trim_ratio)
        self.view_dropout_probability = float(view_dropout_probability)
        self.variance_target = float(variance_target)
        self.view_embedding = (
            nn.Embedding(self.num_views, self.dim) if view_embedding else None
        )
        self.blocks = nn.ModuleList(
            [
                SetAttentionBlock(self.dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fusion = VisibilityAwareCrossViewFusion(
            self.dim,
            temperature=visibility_temperature,
        )
        self._last_auxiliary: dict[str, torch.Tensor] = {}

    def _normalize_inputs(
        self,
        patch_tokens: torch.Tensor,
        view_ids: torch.Tensor | None,
        valid_view_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 4 or patch_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected [B, V, N, {self.dim}] tokens, got "
                f"{tuple(patch_tokens.shape)}"
            )
        batch_size, view_count = patch_tokens.shape[:2]
        if view_count != self.num_views:
            raise ValueError(
                f"expected {self.num_views} views, got {view_count}"
            )
        if view_ids is None:
            view_ids = torch.arange(view_count, device=patch_tokens.device)
        view_ids = view_ids.to(device=patch_tokens.device, dtype=torch.long)
        if view_ids.ndim == 1:
            view_ids = view_ids.unsqueeze(0).expand(batch_size, -1)
        if view_ids.shape != (batch_size, view_count):
            raise ValueError("view_ids must have shape [V] or [B, V]")
        if bool(((view_ids < 0) | (view_ids >= self.num_views)).any()):
            raise ValueError("view_ids contain an out-of-range camera index")
        if valid_view_mask is None:
            valid_view_mask = torch.ones(
                (batch_size, view_count),
                dtype=torch.bool,
                device=patch_tokens.device,
            )
        else:
            valid_view_mask = valid_view_mask.to(
                device=patch_tokens.device,
                dtype=torch.bool,
            )
        if valid_view_mask.shape != (batch_size, view_count):
            raise ValueError("valid_view_mask must have shape [B, V]")
        if bool((~valid_view_mask.any(dim=1)).any()):
            raise ValueError("every object must contain at least one valid view")
        return view_ids, valid_view_mask

    def _encode_once(
        self,
        patch_tokens: torch.Tensor,
        view_ids: torch.Tensor,
        valid_view_mask: torch.Tensor,
    ) -> MultiViewContextOutput:
        batch_size, view_count, token_count, _ = patch_tokens.shape
        flat = patch_tokens.reshape(batch_size * view_count, token_count, self.dim)
        robust = _robust_token_context(flat, self.trim_ratio).reshape(
            batch_size, view_count, self.dim
        )
        centered = patch_tokens - robust.unsqueeze(2)
        dispersion = centered.float().square().mean(dim=2).sqrt().to(
            patch_tokens.dtype
        )
        contexts = robust
        if self.view_embedding is not None:
            contexts = contexts + self.view_embedding(view_ids).to(contexts.dtype)
        contexts = contexts * valid_view_mask.unsqueeze(-1).to(contexts.dtype)
        attention_weights = contexts.new_zeros(
            (batch_size, 1, view_count, view_count)
        )
        attention_entropy_terms: list[torch.Tensor] = []
        for block in self.blocks:
            contexts, attention_weights = block(contexts, valid_view_mask)
            attention = attention_weights.float().clamp_min(0.0)
            key_mask = valid_view_mask[:, None, None, :]
            attention = attention * key_mask
            attention = attention / attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            entropy = -(
                attention.clamp_min(1e-8)
                * attention.clamp_min(1e-8).log()
            ).sum(dim=-1)
            maximum = valid_view_mask.sum(dim=1).clamp_min(1).float().log()
            valid_queries = valid_view_mask[:, None, :].expand_as(entropy)
            attention_entropy_terms.append(
                (
                    entropy
                    / maximum[:, None, None].clamp_min(1e-8)
                    * valid_queries
                ).sum()
                / valid_queries.sum().clamp_min(1)
            )
        object_context, cross_view, visibility = self.fusion(
            robust,
            contexts,
            dispersion,
            valid_view_mask,
        )
        return MultiViewContextOutput(
            object_context=object_context,
            cross_view_context=cross_view,
            robust_view_context=robust,
            token_dispersion=dispersion,
            visibility_weights=visibility,
            attention_weights=attention_weights,
            attention_entropy_penalty=torch.stack(
                attention_entropy_terms
            ).mean(),
        )

    @staticmethod
    def _drop_views(
        valid_view_mask: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        keep = torch.rand(valid_view_mask.shape, device=valid_view_mask.device)
        keep = (keep >= probability) & valid_view_mask
        empty = ~keep.any(dim=1)
        if bool(empty.any()):
            for row in empty.nonzero(as_tuple=False).flatten().tolist():
                candidates = valid_view_mask[row].nonzero(as_tuple=False).flatten()
                keep[row, int(candidates[0])] = True
        return keep

    def forward(
        self,
        patch_tokens: torch.Tensor,
        view_ids: torch.Tensor | None = None,
        valid_view_mask: torch.Tensor | None = None,
    ) -> MultiViewContextOutput:
        view_ids, valid_view_mask = self._normalize_inputs(
            patch_tokens,
            view_ids,
            valid_view_mask,
        )
        output = self._encode_once(patch_tokens, view_ids, valid_view_mask)

        # Use all valid cross-view representations rather than only the object
        # batch dimension. The old object_context.var(dim=0) had exactly zero
        # gradient whenever the per-device micro batch was one.
        variance_samples = output.cross_view_context[valid_view_mask].float()
        context_variance = F.relu(
            self.variance_target
            - torch.sqrt(variance_samples.var(dim=0, unbiased=False) + 1e-4)
        ).mean()
        valid_float = valid_view_mask.float()
        observed = valid_float.sum(dim=0)
        target_usage = observed / observed.sum().clamp_min(1.0)
        actual_usage = output.visibility_weights.float().sum(dim=0)
        actual_usage = actual_usage / actual_usage.sum().clamp_min(1e-8)
        visibility_balance = (
            (actual_usage - target_usage).square().mean() * self.num_views
        )

        attention_entropy = output.attention_entropy_penalty

        context_consistency = output.object_context.float().sum() * 0.0
        if self.training and self.view_dropout_probability > 0.0:
            dropped_mask = self._drop_views(
                valid_view_mask,
                self.view_dropout_probability,
            )
            dropped = self._encode_once(patch_tokens, view_ids, dropped_mask)
            context_consistency = (
                1.0
                - F.cosine_similarity(
                    output.object_context,
                    dropped.object_context,
                    dim=-1,
                    eps=1e-6,
                )
            ).mean()
        self._last_auxiliary = {
            "context_consistency": context_consistency,
            "context_variance": context_variance,
            "visibility_balance": visibility_balance,
            "attention_entropy": attention_entropy,
        }
        return output

    def auxiliary_losses(
        self,
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        if detach:
            return {
                name: value.detach() for name, value in self._last_auxiliary.items()
            }
        return dict(self._last_auxiliary)


class CompositionalReferenceBank(nn.Module):
    """Cross-attend to normal primitives without an input-value shortcut."""

    def __init__(
        self,
        dim: int,
        num_references: int = 256,
        top_k: int = 16,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_references < 2:
            raise ValueError("num_references must be at least 2")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.dim = int(dim)
        self.num_references = int(num_references)
        self.top_k = min(int(top_k), self.num_references)
        self.temperature = float(temperature)

        self.references = nn.Parameter(torch.empty(num_references, dim))
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, dim)
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self._last_auxiliary: dict[str, torch.Tensor] = {}
        self.reset_reference_parameters()

    def reset_reference_parameters(self) -> None:
        nn.init.trunc_normal_(self.references, std=0.02, a=-0.06, b=0.06)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected [B, N, {self.dim}] tokens, got {tuple(tokens.shape)}"
            )

        queries = F.normalize(self.query_projection(tokens), dim=-1, eps=1e-6)
        keys = F.normalize(
            self.key_projection(self.references),
            dim=-1,
            eps=1e-6,
        )
        logits = torch.matmul(queries, keys.transpose(0, 1)) / self.temperature
        if self.top_k < self.num_references:
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, -torch.inf)
            logits = sparse_logits.scatter(-1, top_indices, top_values)

        # FP32 softmax avoids underflow with BF16 training.  Values are cast
        # back before matrix multiplication to preserve the configured AMP path.
        assignments = F.softmax(logits.float(), dim=-1).to(tokens.dtype)
        values = self.value_projection(self.references).to(tokens.dtype)
        reconstruction = torch.matmul(assignments, values)
        reconstruction = self.output_projection(reconstruction)
        reconstruction = self.output_norm(reconstruction)

        usage = assignments.float().mean(dim=(0, 1))
        uniform = 1.0 / self.num_references
        reference_balance = (
            usage.clamp_min(1e-8)
            * (usage.clamp_min(1e-8).log() - math.log(uniform))
        ).sum()
        assignment_entropy = -(
            assignments.float().clamp_min(1e-8)
            * assignments.float().clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        self._last_auxiliary = {
            "reference_balance": reference_balance,
            "reference_assignment_entropy": assignment_entropy,
        }
        return reconstruction

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        return dict(self._last_auxiliary)


class CategoryFreeRouter(nn.Module):
    """Softly route samples using robust visual context, never a category ID."""

    def __init__(
        self,
        dim: int,
        num_experts: int = 3,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        top_k: int = 3,
        trim_ratio: float = 0.1,
        dropout: float = 0.0,
        multi_view_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        hidden_dim = hidden_dim or max(32, dim // 2)
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        self.top_k = min(int(top_k), self.num_experts)
        self.trim_ratio = float(trim_ratio)
        self.multi_view_conditioning = bool(multi_view_conditioning)
        input_dim = dim * (3 if self.multi_view_conditioning else 2)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim, eps=1e-6),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        # Uniform routing at initialization gives every semantic-scale expert
        # a gradient with the default dense soft routing and makes the new path
        # stable at the start of training.  Sparse top-k remains an explicit
        # ablation option rather than the default, since a zero-initialized
        # sparse router could otherwise starve one expert on its first update.
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        if self.top_k < self.num_experts:
            # A zero-initialized sparse router deterministically selects the
            # same experts on tied logits and can permanently starve one path.
            nn.init.trunc_normal_(output.weight, std=0.01, a=-0.03, b=0.03)
        else:
            nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        view_context: torch.Tensor | None = None,
        cross_view_context: torch.Tensor | None = None,
        token_dispersion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected [B, N, {self.dim}] tokens, got "
                f"{tuple(patch_tokens.shape)}"
            )
        context = (
            _robust_token_context(patch_tokens, self.trim_ratio)
            if view_context is None
            else view_context
        )
        if context.shape != (patch_tokens.shape[0], self.dim):
            raise ValueError("view_context must have shape [B, C]")
        if token_dispersion is None:
            centered = patch_tokens - context.unsqueeze(1)
            token_dispersion = centered.float().square().mean(dim=1).sqrt()
            token_dispersion = token_dispersion.to(context.dtype)
        if token_dispersion.shape != context.shape:
            raise ValueError("token_dispersion must have shape [B, C]")
        router_inputs = [context]
        if self.multi_view_conditioning:
            if cross_view_context is None:
                cross_view_context = context
            if cross_view_context.shape != context.shape:
                raise ValueError("cross_view_context must have shape [B, C]")
            router_inputs.append(cross_view_context)
        router_inputs.append(token_dispersion)
        logits = self.network(torch.cat(router_inputs, dim=-1))
        logits = logits / self.temperature
        if self.top_k < self.num_experts:
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, -torch.inf)
            logits = sparse_logits.scatter(-1, top_indices, top_values)
        return F.softmax(logits.float(), dim=-1).to(patch_tokens.dtype)


class SpatialScaleExpert(nn.Module):
    """Depthwise spatial expert for local or component-scale structure."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.dim = int(dim)
        self.input_norm = nn.LayerNorm(dim, eps=1e-6)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )
        self.special_mlp = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        num_special_tokens: int,
    ) -> torch.Tensor:
        special = tokens[:, :num_special_tokens]
        patches = tokens[:, num_special_tokens:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError(
                "semantic-scale spatial experts require a square patch grid, "
                f"got {patches.shape[1]} patch tokens"
            )

        normalized = self.input_norm(patches)
        feature_map = normalized.transpose(1, 2).reshape(
            tokens.shape[0], self.dim, side, side
        )
        spatial = self.depthwise(feature_map)
        spatial = spatial + self.channel_mixer(spatial)
        patch_output = spatial.flatten(2).transpose(1, 2)
        # This residual is from the already reconstructed normal reference,
        # never from the raw encoder token.
        patch_output = patch_output + patches

        if num_special_tokens:
            special_output = special + self.special_mlp(special)
            return torch.cat([special_output, patch_output], dim=1)
        return patch_output


class GlobalScaleExpert(nn.Module):
    """Model global layout using robust context-conditioned modulation."""

    def __init__(self, dim: int, dropout: float, trim_ratio: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.trim_ratio = float(trim_ratio)
        self.token_norm = nn.LayerNorm(dim, eps=1e-6)
        self.token_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.modulation = nn.Linear(dim, dim * 2)

    def forward(
        self,
        tokens: torch.Tensor,
        num_special_tokens: int,
    ) -> torch.Tensor:
        patches = tokens[:, num_special_tokens:]
        context = _robust_token_context(patches, self.trim_ratio)
        scale, shift = self.modulation(context).chunk(2, dim=-1)
        transformed = self.token_mlp(self.token_norm(tokens))
        transformed = transformed * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1))
        transformed = transformed + 0.1 * shift.unsqueeze(1)
        return tokens + transformed


class CompositionalNormalityAdapter(nn.Module):
    """Normal-reference reconstruction plus local/component/global experts."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        num_special_tokens: int,
        num_references: int = 256,
        reference_top_k: int = 16,
        reference_temperature: float = 0.07,
        router_temperature: float = 1.0,
        router_top_k: int = 3,
        router_trim_ratio: float = 0.1,
        dropout: float = 0.0,
        multi_view_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if num_special_tokens < 0:
            raise ValueError("num_special_tokens cannot be negative")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.num_special_tokens = int(num_special_tokens)
        self.multi_view_conditioning = bool(multi_view_conditioning)
        self.reference_bank = CompositionalReferenceBank(
            dim=latent_dim,
            num_references=num_references,
            top_k=reference_top_k,
            temperature=reference_temperature,
        )
        self.router = CategoryFreeRouter(
            dim=latent_dim,
            num_experts=3,
            temperature=router_temperature,
            top_k=router_top_k,
            trim_ratio=router_trim_ratio,
            dropout=dropout,
            multi_view_conditioning=self.multi_view_conditioning,
        )
        self.experts = nn.ModuleList(
            [
                SpatialScaleExpert(
                    latent_dim,
                    kernel_size=3,
                    dilation=1,
                    dropout=dropout,
                ),
                SpatialScaleExpert(
                    latent_dim,
                    kernel_size=5,
                    dilation=2,
                    dropout=dropout,
                ),
                GlobalScaleExpert(
                    latent_dim,
                    dropout=dropout,
                    trim_ratio=router_trim_ratio,
                ),
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.output_projection = nn.Linear(latent_dim, output_dim)
        self.cross_view_film = (
            nn.Sequential(
                nn.LayerNorm(latent_dim * 2, eps=1e-6),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            )
            if self.multi_view_conditioning
            else None
        )
        self._last_auxiliary: dict[str, torch.Tensor] = {}

    def forward(
        self,
        tokens: torch.Tensor,
        view_context: torch.Tensor | None = None,
        cross_view_context: torch.Tensor | None = None,
        token_dispersion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected [B, N, {self.latent_dim}] tokens, got "
                f"{tuple(tokens.shape)}"
            )
        if tokens.shape[1] <= self.num_special_tokens:
            raise ValueError("input does not contain any patch tokens")

        # Routing reads encoder context, but the experts only receive the
        # normal-reference reconstruction.  There is no encoder-value shortcut.
        patch_tokens = tokens[:, self.num_special_tokens :]
        routing = self.router(
            patch_tokens,
            view_context=view_context,
            cross_view_context=cross_view_context,
            token_dispersion=token_dispersion,
        )
        normal_tokens = self.reference_bank(tokens)
        if self.cross_view_film is not None:
            if view_context is None or cross_view_context is None:
                raise ValueError(
                    "multi-view normality adapter requires view and cross-view context"
                )
            scale, shift = self.cross_view_film(
                torch.cat([view_context, cross_view_context], dim=-1)
            ).chunk(2, dim=-1)
            # Context can only softly modulate the normal-reference path.  Raw
            # encoder patch values are never added to decoder values.
            normal_tokens = normal_tokens * (
                1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)
            )
            normal_tokens = normal_tokens + 0.1 * torch.tanh(shift).unsqueeze(1)
        expert_outputs = torch.stack(
            [
                expert(normal_tokens, self.num_special_tokens)
                for expert in self.experts
            ],
            dim=1,
        )
        mixed = (
            expert_outputs
            * routing[:, :, None, None].to(expert_outputs.dtype)
        ).sum(dim=1)
        output = self.output_projection(self.output_norm(mixed))

        mean_route = routing.float().mean(dim=0)
        router_balance = (
            (mean_route - 1.0 / routing.shape[1]).square().mean()
            * routing.shape[1]
        )
        entropy = -(
            routing.float().clamp_min(1e-8)
            * routing.float().clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        # Penalize uncertain per-sample routing. Global router_balance separately
        # prevents all samples from collapsing onto the same expert.
        router_entropy_penalty = entropy / math.log(routing.shape[1])

        flattened = expert_outputs.float().flatten(start_dim=2)
        diversity_terms = []
        for left in range(flattened.shape[1]):
            for right in range(left + 1, flattened.shape[1]):
                similarity = F.cosine_similarity(
                    flattened[:, left],
                    flattened[:, right],
                    dim=-1,
                    eps=1e-6,
                )
                diversity_terms.append(similarity.square().mean())
        expert_diversity = torch.stack(diversity_terms).mean()

        self._last_auxiliary = self.reference_bank.auxiliary_losses()
        self._last_auxiliary.update(
            {
                "router_balance": router_balance,
                "router_entropy_penalty": router_entropy_penalty,
                "expert_diversity": expert_diversity,
            }
        )
        return output

    def auxiliary_losses(
        self,
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        losses = dict(self._last_auxiliary)
        if detach:
            return {name: value.detach() for name, value in losses.items()}
        return losses

    def regularization_loss(
        self,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        """Return a safe scalar auxiliary loss (zero before first forward)."""

        if not self._last_auxiliary:
            return self.output_projection.weight.sum() * 0.0
        return combine_auxiliary_losses(
            self._last_auxiliary,
            weights=weights,
            anchor=self.output_projection.weight,
        )


class GeneralizedDinomaly(nn.Module):
    """Dinomaly-compatible model with an optional true multi-view path."""

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.ModuleList,
        decoder: nn.ModuleList,
        target_layers: Sequence[int],
        fuse_layer_encoder: Sequence[Sequence[int]],
        fuse_layer_decoder: Sequence[Sequence[int]],
        fuse_layer_bottleneck: Sequence[int] = tuple(range(8)),
        mask_neighbor_size: int = 0,
        remove_class_token: bool = False,
        context_aware_recenter: bool = True,
        multi_view_context_encoder: MultiViewContextEncoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.multi_view_context_encoder = multi_view_context_encoder
        self.target_layers = list(target_layers)
        self.fuse_layer_encoder = [list(group) for group in fuse_layer_encoder]
        self.fuse_layer_decoder = [list(group) for group in fuse_layer_decoder]
        self.fuse_layer_bottleneck = list(fuse_layer_bottleneck)
        self.mask_neighbor_size = int(mask_neighbor_size)
        self.remove_class_token = bool(remove_class_token)
        self.context_aware_recenter = bool(context_aware_recenter)
        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = getattr(
                self.encoder, "n_storage_tokens", 0
            )

    @property
    def multi_view_enabled(self) -> bool:
        return self.multi_view_context_encoder is not None

    @staticmethod
    def fuse_feature(features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("cannot fuse an empty feature list")
        return torch.stack(list(features), dim=1).mean(dim=1)

    def init_weights(self) -> None:
        # Match the upstream Dinomaly initialization for trainable transformer
        # components while preserving the frozen pretrained encoder.
        trainable_modules = [self.bottleneck, self.decoder]
        if self.multi_view_context_encoder is not None:
            trainable_modules.append(self.multi_view_context_encoder)
        for module in trainable_modules:
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.trunc_normal_(
                        child.weight, std=0.01, a=-0.03, b=0.03
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.Conv2d):
                    nn.init.trunc_normal_(
                        child.weight, std=0.02, a=-0.06, b=0.06
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                    if child.weight is not None:
                        nn.init.ones_(child.weight)
                elif isinstance(child, nn.Embedding):
                    nn.init.trunc_normal_(child.weight, std=0.02, a=-0.06, b=0.06)
        for child in self.bottleneck.modules():
            if isinstance(child, CompositionalReferenceBank):
                child.reset_reference_parameters()
            elif isinstance(child, CategoryFreeRouter):
                child.reset_output_parameters()

    def _generate_mask(
        self,
        feature_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        h = w = feature_size
        neighbor = self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for row in range(h):
            for column in range(w):
                row_start = max(row - neighbor // 2, 0)
                row_end = min(row + neighbor // 2 + 1, h)
                col_start = max(column - neighbor // 2, 0)
                col_end = min(column + neighbor // 2 + 1, w)
                mask[row, column, row_start:row_end, col_start:col_end] = 0
        mask = mask.reshape(h * w, h * w)
        if self.remove_class_token:
            return mask
        special = 1 + int(self.encoder.num_register_tokens)
        expanded = torch.ones(h * w + special, h * w + special, device=device)
        expanded[special:, special:] = mask
        return expanded

    def forward(
        self,
        images: torch.Tensor,
        view_ids: torch.Tensor | None = None,
        valid_view_mask: torch.Tensor | None = None,
        return_auxiliary: bool = False,
        return_context: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor]]
        | tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            dict[str, torch.Tensor],
        ]
        | tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
        ]
    ):
        object_batch_size: int | None = None
        view_count: int | None = None
        if self.multi_view_enabled:
            if images.ndim != 5:
                raise ValueError(
                    "multi-view model expects images with shape [B, V, 3, H, W]"
                )
            object_batch_size, view_count = images.shape[:2]
            assert self.multi_view_context_encoder is not None
            if view_count != self.multi_view_context_encoder.num_views:
                raise ValueError(
                    f"expected {self.multi_view_context_encoder.num_views} views, "
                    f"got {view_count}"
                )
            flat_images = images.reshape(
                object_batch_size * view_count,
                *images.shape[2:],
            )
        else:
            if images.ndim != 4:
                raise ValueError(
                    "single-view model expects images with shape [B, 3, H, W]"
                )
            flat_images = images

        with torch.no_grad():
            tokens = self.encoder.prepare_tokens(flat_images)
        encoder_layers: list[torch.Tensor] = []
        for index, block in enumerate(self.encoder.blocks):
            if index > self.target_layers[-1]:
                break
            with torch.no_grad():
                tokens = block(tokens)
            if index in self.target_layers:
                encoder_layers.append(tokens)
        if len(encoder_layers) != len(self.target_layers):
            raise RuntimeError(
                "encoder did not produce all requested target layers: "
                f"expected {len(self.target_layers)}, got {len(encoder_layers)}"
            )

        special = 1 + int(self.encoder.num_register_tokens)
        patch_count = encoder_layers[0].shape[1] - special
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise RuntimeError(
                f"encoder produced a non-square patch grid ({patch_count} tokens)"
            )

        if self.remove_class_token:
            bottleneck_layers = [layer[:, special:] for layer in encoder_layers]
        else:
            bottleneck_layers = encoder_layers
        tokens = self.fuse_feature(
            [bottleneck_layers[index] for index in self.fuse_layer_bottleneck]
        ).detach()
        context_output: MultiViewContextOutput | None = None
        for block in self.bottleneck:
            if self.multi_view_enabled and isinstance(
                block, CompositionalNormalityAdapter
            ):
                assert object_batch_size is not None and view_count is not None
                assert self.multi_view_context_encoder is not None
                patch_tokens = tokens[:, block.num_special_tokens :].reshape(
                    object_batch_size,
                    view_count,
                    patch_count,
                    block.latent_dim,
                )
                context_output = self.multi_view_context_encoder(
                    patch_tokens,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                )
                tokens = block(
                    tokens,
                    view_context=context_output.robust_view_context.reshape(
                        object_batch_size * view_count,
                        block.latent_dim,
                    ),
                    cross_view_context=context_output.cross_view_context.reshape(
                        object_batch_size * view_count,
                        block.latent_dim,
                    ),
                    token_dispersion=context_output.token_dispersion.reshape(
                        object_batch_size * view_count,
                        block.latent_dim,
                    ),
                )
            else:
                tokens = block(tokens)
        if self.multi_view_enabled and context_output is None:
            raise RuntimeError("multi-view context was not connected to the adapter")

        attention_mask = None
        if self.mask_neighbor_size > 0:
            attention_mask = self._generate_mask(side, tokens.device)
        decoder_layers: list[torch.Tensor] = []
        for block in self.decoder:
            tokens = block(tokens, attn_mask=attention_mask)
            decoder_layers.append(tokens)
        decoder_layers.reverse()

        encoder_features = [
            self.fuse_feature([encoder_layers[index] for index in group])
            for group in self.fuse_layer_encoder
        ]
        decoder_features = [
            self.fuse_feature([decoder_layers[index] for index in group])
            for group in self.fuse_layer_decoder
        ]
        if not self.remove_class_token:
            decoder_features = [feature[:, special:] for feature in decoder_features]

        if self.context_aware_recenter:
            encoder_features = [
                F.layer_norm(
                    feature[:, special:] - feature[:, :1],
                    normalized_shape=(feature.shape[-1],),
                    eps=1e-8,
                )
                for feature in encoder_features
            ]
        else:
            encoder_features = [feature[:, special:] for feature in encoder_features]

        flat_batch_size = flat_images.shape[0]
        encoder_features = [
            feature.permute(0, 2, 1)
            .reshape(flat_batch_size, -1, side, side)
            .contiguous()
            for feature in encoder_features
        ]
        decoder_features = [
            feature.permute(0, 2, 1)
            .reshape(flat_batch_size, -1, side, side)
            .contiguous()
            for feature in decoder_features
        ]
        if self.multi_view_enabled:
            assert object_batch_size is not None and view_count is not None
            encoder_features = [
                feature.reshape(
                    object_batch_size,
                    view_count,
                    *feature.shape[1:],
                )
                for feature in encoder_features
            ]
            decoder_features = [
                feature.reshape(
                    object_batch_size,
                    view_count,
                    *feature.shape[1:],
                )
                for feature in decoder_features
            ]
        context_payload: dict[str, torch.Tensor] = {}
        if context_output is not None:
            context_payload = {
                "object_context": context_output.object_context,
                "cross_view_context": context_output.cross_view_context,
                "visibility_weights": context_output.visibility_weights,
                "attention_weights": context_output.attention_weights,
            }
        if not return_auxiliary:
            if return_context:
                return encoder_features, decoder_features, context_payload
            return encoder_features, decoder_features
        # DataParallel cannot reliably expose Python-side state mutated inside
        # replicas. Returning graph-connected [1] tensors makes gather preserve
        # every auxiliary term and avoids the scalar-gather warning.
        packed_auxiliary = {
            name: value.reshape(1)
            for name, value in self.auxiliary_losses(detach=False).items()
        }
        if return_context:
            return (
                encoder_features,
                decoder_features,
                packed_auxiliary,
                context_payload,
            )
        return encoder_features, decoder_features, packed_auxiliary

    def _normality_adapter(self) -> CompositionalNormalityAdapter | None:
        for module in self.bottleneck.modules():
            if isinstance(module, CompositionalNormalityAdapter):
                return module
        return None

    def auxiliary_losses(
        self,
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        adapter = self._normality_adapter()
        if adapter is not None:
            losses.update(adapter.auxiliary_losses(detach=detach))
        if self.multi_view_context_encoder is not None:
            losses.update(
                self.multi_view_context_encoder.auxiliary_losses(detach=detach)
            )
        return losses

    def regularization_loss(
        self,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        losses = self.auxiliary_losses(detach=False)
        if not losses:
            return next(self.parameters()).sum() * 0.0
        return combine_auxiliary_losses(
            losses,
            weights=weights,
            anchor=next(self.parameters()),
        )


class MultiViewGeneralizedDinomaly(GeneralizedDinomaly):
    """Explicit type for the five-view category-generalized architecture."""

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("multi_view_context_encoder") is None:
            raise ValueError(
                "MultiViewGeneralizedDinomaly requires multi_view_context_encoder"
            )
        super().__init__(*args, **kwargs)
