from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .generalized_model import combine_auxiliary_losses


class InformationDensityDownProjection(nn.Module):
    """Original Dinomaly down-projection with context-only capacity gating.

    The difficulty estimator never observes the center patch directly.  It
    predicts the normal reconstruction error from neighboring patches and the
    frozen encoder's special tokens, preventing a localized anomaly from
    granting itself a wider bottleneck merely because its own feature is OOD.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_special_tokens: int,
        dropout: float,
        hidden_dim: int = 64,
        channel_widths: Sequence[int] = (64, 128, 256),
        channel_thresholds: Sequence[float] = (0.33, 0.66),
        channel_temperature: float = 0.08,
        difficulty_z_threshold: float = 0.5,
        difficulty_temperature: float = 0.5,
        target_budget: float = 0.3,
        capacity_warmup_steps: int = 1500,
        capacity_ramp_steps: int = 1000,
        residual_ema_decay: float = 0.99,
        initial_expected_error: float = 0.1,
        emit_routing: bool = False,
        context_scales: Sequence[int] = (3, 5),
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channel_widths)
        thresholds = tuple(float(value) for value in channel_thresholds)
        scales = tuple(int(value) for value in context_scales)
        if not widths or widths[-1] != latent_dim:
            raise ValueError(
                "information-density channel_widths must end at latent_dim"
            )
        if widths[0] <= 0 or any(
            left >= right for left, right in zip(widths, widths[1:])
        ):
            raise ValueError(
                "information-density channel_widths must be strictly increasing"
            )
        if len(thresholds) != len(widths) - 1:
            raise ValueError(
                "channel_thresholds must contain one value per optional group"
            )
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("channel_thresholds must lie in (0, 1)")
        if any(
            left >= right
            for left, right in zip(thresholds, thresholds[1:])
        ):
            raise ValueError("channel_thresholds must be strictly increasing")
        if channel_temperature <= 0 or difficulty_temperature <= 0:
            raise ValueError("information-density temperatures must be positive")
        if not 0.0 < target_budget < 1.0:
            raise ValueError("information-density target_budget must lie in (0, 1)")
        if capacity_warmup_steps < 0 or capacity_ramp_steps < 0:
            raise ValueError("capacity warmup/ramp steps must be non-negative")
        if not 0.0 <= residual_ema_decay < 1.0:
            raise ValueError("residual_ema_decay must lie in [0, 1)")
        if initial_expected_error <= 0:
            raise ValueError("initial_expected_error must be positive")
        if len(scales) < 2 or any(
            scale < 3 or scale % 2 == 0 for scale in scales
        ):
            raise ValueError(
                "context_scales must contain at least two odd values >= 3"
            )
        if any(
            left >= right for left, right in zip(scales, scales[1:])
        ):
            raise ValueError("context_scales must be strictly increasing")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_special_tokens = int(num_special_tokens)
        self.channel_widths = widths
        self.channel_thresholds = thresholds
        self.channel_temperature = float(channel_temperature)
        self.difficulty_z_threshold = float(difficulty_z_threshold)
        self.difficulty_temperature = float(difficulty_temperature)
        self.target_budget = float(target_budget)
        self.capacity_warmup_steps = int(capacity_warmup_steps)
        self.capacity_ramp_steps = int(capacity_ramp_steps)
        self.residual_ema_decay = float(residual_ema_decay)
        self.initial_expected_error = float(initial_expected_error)
        self.emit_routing = bool(emit_routing)
        self.context_scales = scales
        self.training_step = 0

        self.projection = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(p=dropout)
        context_dim = input_dim * (len(scales) + 1)
        complexity_dim = len(scales) + 1
        self.context_norm = nn.LayerNorm(context_dim, eps=1e-6)
        self.complexity_encoder = nn.Sequential(
            nn.Linear(complexity_dim, hidden_dim),
            nn.GELU(),
        )
        self.estimator = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        initial_variance = max(initial_expected_error * 0.5, 1e-3) ** 2
        self.register_buffer(
            "residual_mean", torch.tensor(float(initial_expected_error))
        )
        self.register_buffer(
            "residual_variance", torch.tensor(float(initial_variance))
        )
        self.register_buffer(
            "residual_initialized", torch.tensor(False, dtype=torch.bool)
        )

        self._last_expected_error: torch.Tensor | None = None
        self._last_difficulty: torch.Tensor | None = None
        self._last_channel_mask: torch.Tensor | None = None
        self._last_complexity: torch.Tensor | None = None
        self._last_side: int | None = None
        self._last_auxiliary: dict[str, torch.Tensor] = {}

    def reset_density_parameters(self) -> None:
        """Restore a conservative expected-error prior after upstream init."""

        final = self.estimator[-1]
        assert isinstance(final, nn.Linear)
        inverse_softplus = math.log(math.expm1(self.initial_expected_error))
        nn.init.constant_(final.bias, inverse_softplus)

    def set_training_step(self, completed_steps: int) -> None:
        self.training_step = max(0, int(completed_steps))

    def _neighbor_statistics(
        self,
        patches: torch.Tensor,
        side: int,
        kernel_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, patch_count, channels = patches.shape
        spatial = patches.transpose(1, 2).reshape(
            batch_size, channels, side, side
        )
        window_area = float(kernel_size * kernel_size)
        padding = kernel_size // 2
        neighborhood_sum = (
            F.avg_pool2d(
                spatial,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=True,
            )
            * window_area
            - spatial
        )
        neighborhood_square_sum = (
            F.avg_pool2d(
                spatial.square(),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=True,
            )
            * window_area
            - spatial.square()
        )
        valid = spatial.new_ones((batch_size, 1, side, side))
        neighbor_count = (
            F.avg_pool2d(
                valid,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                count_include_pad=True,
            )
            * window_area
            - 1.0
        ).clamp_min_(1.0)
        context = neighborhood_sum / neighbor_count
        second_moment = neighborhood_square_sum / neighbor_count
        variance = (second_moment - context.square()).clamp_min_(0.0)
        token_context = context.flatten(2).transpose(1, 2).reshape(
            batch_size, patch_count, channels
        )
        token_variance = variance.mean(dim=1, keepdim=True).flatten(
            2
        ).transpose(1, 2)
        return token_context, token_variance

    def _difficulty_from_expected(
        self, expected_error: torch.Tensor
    ) -> torch.Tensor:
        mean = self.residual_mean.detach().to(expected_error)
        standard_deviation = self.residual_variance.detach().to(
            expected_error
        ).clamp_min(1e-6).sqrt()
        standardized = (expected_error - mean) / standard_deviation
        return torch.sigmoid(
            (standardized - self.difficulty_z_threshold)
            / self.difficulty_temperature
        )

    def _capacity_blend(self) -> float:
        if not self.training:
            return 1.0
        if self.training_step < self.capacity_warmup_steps:
            return 0.0
        if self.capacity_ramp_steps == 0:
            return 1.0
        elapsed = self.training_step - self.capacity_warmup_steps
        return min(1.0, max(0.0, elapsed / self.capacity_ramp_steps))

    def _nested_mask(self, difficulty: torch.Tensor) -> torch.Tensor:
        group_sizes = [self.channel_widths[0]] + [
            right - left
            for left, right in zip(
                self.channel_widths, self.channel_widths[1:]
            )
        ]
        groups = [
            difficulty.new_ones((*difficulty.shape[:-1], group_sizes[0]))
        ]
        for threshold, group_size in zip(
            self.channel_thresholds, group_sizes[1:], strict=True
        ):
            gate = torch.sigmoid(
                (difficulty - threshold) / self.channel_temperature
            )
            groups.append(gate.expand(*gate.shape[:-1], group_size))
        mask = torch.cat(groups, dim=-1)
        blend = self._capacity_blend()
        if blend < 1.0:
            mask = 1.0 - blend * (1.0 - mask)
        return mask

    def forward(
        self, tokens: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                "information-density projection expects [B, N, input_dim]"
            )
        patch_count = tokens.shape[1] - self.num_special_tokens
        side = math.isqrt(patch_count)
        if patch_count <= 0 or side * side != patch_count:
            raise ValueError(
                "information-density projection requires a square patch grid"
            )
        special = tokens[:, : self.num_special_tokens]
        patches = tokens[:, self.num_special_tokens :]
        contexts = []
        variances = []
        for scale in self.context_scales:
            context, variance = self._neighbor_statistics(
                patches, side, scale
            )
            contexts.append(context)
            variances.append(torch.tanh(torch.log1p(variance)))
        global_context = special.mean(dim=1, keepdim=True).expand(
            -1, patch_count, -1
        )
        scale_disagreement = (
            0.5
            * (
                1.0
                - F.cosine_similarity(
                    contexts[0], contexts[-1], dim=-1
                ).unsqueeze(-1)
            )
        ).clamp_(0.0, 1.0)
        complexity = torch.cat(
            [*variances, scale_disagreement], dim=-1
        )
        normalized_context = self.context_norm(
            torch.cat([*contexts, global_context], dim=-1)
        )
        encoded_complexity = self.complexity_encoder(complexity)
        estimator_input = torch.cat(
            [normalized_context, encoded_complexity], dim=-1
        )
        expected_error = F.softplus(self.estimator(estimator_input)) + 1e-6
        difficulty = self._difficulty_from_expected(expected_error)
        patch_mask = self._nested_mask(difficulty.detach())
        special_mask = patch_mask.new_ones(
            tokens.shape[0], self.num_special_tokens, self.latent_dim
        )
        channel_mask = torch.cat([special_mask, patch_mask], dim=1)

        self._last_expected_error = expected_error
        self._last_difficulty = difficulty
        self._last_channel_mask = channel_mask
        self._last_complexity = complexity
        self._last_side = side
        projected = self.projection(tokens) * channel_mask
        projected = self.dropout(projected)
        if self.emit_routing:
            return projected, difficulty
        return projected

    def update_auxiliary(self, residual_map: torch.Tensor) -> None:
        if self._last_expected_error is None or self._last_difficulty is None:
            raise RuntimeError("difficulty estimator was not run before its target")
        if residual_map.ndim != 3:
            raise ValueError("residual_map must have shape [B, H, W]")
        target = residual_map.detach().reshape(
            residual_map.shape[0], -1, 1
        )
        if target.shape != self._last_expected_error.shape:
            raise ValueError("difficulty target and prediction shapes differ")

        prediction = F.smooth_l1_loss(self._last_expected_error, target)
        difficulty = self._last_difficulty
        budget = F.relu(difficulty.mean() - self.target_budget).square()
        side = residual_map.shape[-1]
        spatial = difficulty.reshape(difficulty.shape[0], 1, side, side)
        smoothness = 0.5 * (
            (spatial[:, :, 1:] - spatial[:, :, :-1]).abs().mean()
            + (spatial[:, :, :, 1:] - spatial[:, :, :, :-1]).abs().mean()
        )
        mid_usage = torch.sigmoid(
            (difficulty - self.channel_thresholds[0])
            / self.channel_temperature
        ).mean()
        high_usage = torch.sigmoid(
            (difficulty - self.channel_thresholds[-1])
            / self.channel_temperature
        ).mean()
        self._last_auxiliary = {
            "difficulty_prediction": prediction,
            "difficulty_budget": budget,
            "difficulty_smoothness": smoothness,
            "difficulty_mean": difficulty.mean(),
            "capacity_mid_usage": mid_usage,
            "capacity_high_usage": high_usage,
        }
        if self._last_complexity is not None:
            self._last_auxiliary.update(
                {
                    "local_complexity_mean": self._last_complexity[
                        ..., :-1
                    ].mean(),
                    "context_scale_disagreement": self._last_complexity[
                        ..., -1
                    ].mean(),
                }
            )

        if self.training and self.training_step <= self.capacity_warmup_steps:
            with torch.no_grad():
                values = target.float()
                moments = torch.stack(
                    [
                        values.sum(),
                        values.square().sum(),
                        values.new_tensor(values.numel()),
                    ]
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(moments, op=dist.ReduceOp.SUM)
                batch_mean = moments[0] / moments[2].clamp_min(1.0)
                batch_variance = (
                    moments[1] / moments[2].clamp_min(1.0)
                    - batch_mean.square()
                ).clamp_min(1e-6)
                if not bool(self.residual_initialized):
                    self.residual_mean.copy_(batch_mean)
                    self.residual_variance.copy_(batch_variance)
                    self.residual_initialized.fill_(True)
                else:
                    decay = self.residual_ema_decay
                    self.residual_mean.lerp_(batch_mean, 1.0 - decay)
                    self.residual_variance.lerp_(batch_variance, 1.0 - decay)

    def auxiliary_losses(
        self, detach: bool = False
    ) -> dict[str, torch.Tensor]:
        if detach:
            return {
                name: value.detach()
                for name, value in self._last_auxiliary.items()
            }
        return dict(self._last_auxiliary)

    def expected_error_map(self) -> torch.Tensor:
        if self._last_expected_error is None or self._last_side is None:
            raise RuntimeError("no information-density prediction is available")
        return self._last_expected_error.transpose(1, 2).reshape(
            self._last_expected_error.shape[0],
            1,
            self._last_side,
            self._last_side,
        )

    def difficulty_map(self) -> torch.Tensor:
        if self._last_difficulty is None or self._last_side is None:
            raise RuntimeError("no information-density prediction is available")
        return self._last_difficulty.transpose(1, 2).reshape(
            self._last_difficulty.shape[0],
            1,
            self._last_side,
            self._last_side,
        )


class DifficultyReconstructionExpert(nn.Module):
    """One difficulty-specific reconstruction path."""

    def __init__(
        self,
        input_width: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_width <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("MoE expert dimensions must be positive")
        self.input_width = int(input_width)
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent[..., : self.input_width])


class DifficultyRoutedMoE(nn.Module):
    """Three reconstruction experts selected by predicted normal difficulty.

    Training begins with the high-capacity expert only, then ramps into soft
    routing.  Once the difficulty estimator is stable, top-1 sparse dispatch
    is used.  Reconstruction gradients never update routing decisions; the
    estimator remains governed by its explicit normal-difficulty objectives.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        num_special_tokens: int,
        dropout: float,
        expert_input_widths: Sequence[int] = (64, 128, 256),
        expert_hidden_dims: Sequence[int] = (1024, 2048, 4096),
        routing_centers: Sequence[float] = (0.17, 0.5, 0.83),
        routing_temperature: float = 0.1,
        routing_warmup_steps: int = 1500,
        routing_ramp_steps: int = 1000,
        hard_routing_start_step: int = 2500,
        target_load: Sequence[float] = (0.6, 0.3, 0.1),
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in expert_input_widths)
        hidden = tuple(int(value) for value in expert_hidden_dims)
        centers = tuple(float(value) for value in routing_centers)
        loads = tuple(float(value) for value in target_load)
        if len(widths) != 3 or len(hidden) != 3 or len(centers) != 3:
            raise ValueError("difficulty MoE requires exactly three experts")
        if widths[-1] != latent_dim or any(
            left >= right for left, right in zip(widths, widths[1:])
        ):
            raise ValueError(
                "expert_input_widths must increase and end at latent_dim"
            )
        if any(value <= 0 for value in hidden):
            raise ValueError("expert_hidden_dims must be positive")
        if any(not 0.0 < value < 1.0 for value in centers) or any(
            left >= right for left, right in zip(centers, centers[1:])
        ):
            raise ValueError("routing_centers must increase within (0, 1)")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        if routing_warmup_steps < 0 or routing_ramp_steps < 0:
            raise ValueError("MoE routing warmup/ramp must be non-negative")
        if hard_routing_start_step < routing_warmup_steps + routing_ramp_steps:
            raise ValueError(
                "hard routing must start after the soft-routing ramp"
            )
        if len(loads) != 3 or any(value <= 0 for value in loads):
            raise ValueError("target_load must contain three positive values")

        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.num_special_tokens = int(num_special_tokens)
        self.expert_input_widths = widths
        self.routing_temperature = float(routing_temperature)
        self.routing_warmup_steps = int(routing_warmup_steps)
        self.routing_ramp_steps = int(routing_ramp_steps)
        self.hard_routing_start_step = int(hard_routing_start_step)
        self.training_step = 0
        self.experts = nn.ModuleList(
            [
                DifficultyReconstructionExpert(
                    input_width=width,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                )
                for width, hidden_dim in zip(widths, hidden, strict=True)
            ]
        )
        self.register_buffer(
            "routing_centers",
            torch.tensor(centers, dtype=torch.float32),
        )
        normalized_load = torch.tensor(loads, dtype=torch.float32)
        normalized_load = normalized_load / normalized_load.sum()
        self.register_buffer("target_load", normalized_load)
        self._last_auxiliary: dict[str, torch.Tensor] = {}
        self._last_assignment: torch.Tensor | None = None

    def set_training_step(self, completed_steps: int) -> None:
        self.training_step = max(0, int(completed_steps))

    def routing_probabilities(
        self, difficulty: torch.Tensor
    ) -> torch.Tensor:
        if difficulty.ndim != 3 or difficulty.shape[-1] != 1:
            raise ValueError("difficulty must have shape [B, N, 1]")
        centers = self.routing_centers.to(difficulty).view(1, 1, 3)
        logits = -(difficulty - centers).abs() / self.routing_temperature
        return torch.softmax(logits, dim=-1)

    def _routing_blend(self) -> float:
        if not self.training:
            return 1.0
        if self.training_step < self.routing_warmup_steps:
            return 0.0
        if self.routing_ramp_steps == 0:
            return 1.0
        elapsed = self.training_step - self.routing_warmup_steps
        return min(1.0, max(0.0, elapsed / self.routing_ramp_steps))

    def _soft_reconstruct(
        self,
        patches: torch.Tensor,
        routing: torch.Tensor,
        blend: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack(
            [expert(patches) for expert in self.experts], dim=2
        )
        high_only = routing.new_zeros(routing.shape)
        high_only[..., -1] = 1.0
        effective = high_only + blend * (routing.detach() - high_only)
        reconstructed = (
            expert_outputs * effective.unsqueeze(-1)
        ).sum(dim=2)
        return reconstructed, expert_outputs

    @staticmethod
    def _distillation_loss(
        expert_outputs: torch.Tensor | None,
        routing: torch.Tensor,
    ) -> torch.Tensor:
        if expert_outputs is None:
            return routing.sum() * 0.0
        teacher = expert_outputs[..., -1, :].detach()
        terms = []
        for expert_index in (0, 1):
            error = F.smooth_l1_loss(
                expert_outputs[..., expert_index, :],
                teacher,
                reduction="none",
            ).mean(dim=-1)
            weight = routing[..., expert_index].detach()
            terms.append(
                (error * weight).sum() / weight.sum().clamp_min(1e-6)
            )
        return torch.stack(terms).mean()

    def _hard_reconstruct(
        self,
        patches: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        flat_patches = patches.reshape(-1, patches.shape[-1])
        flat_assignment = assignment.reshape(-1)
        reconstructed = patches.new_zeros(
            flat_patches.shape[0], self.output_dim
        )
        for expert_index, expert in enumerate(self.experts):
            selected = torch.nonzero(
                flat_assignment == expert_index, as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            expert_output = expert(flat_patches.index_select(0, selected))
            reconstructed = reconstructed.index_copy(
                0, selected, expert_output
            )
        return reconstructed.reshape(
            patches.shape[0], patches.shape[1], self.output_dim
        )

    def _update_auxiliary(
        self,
        routing: torch.Tensor,
        assignment: torch.Tensor,
        expert_outputs: torch.Tensor | None,
        distillation_scale: float,
    ) -> None:
        usage = routing.mean(dim=(0, 1))
        target = self.target_load.to(usage)
        load_balance = (
            target * (target.clamp_min(1e-8).log() - usage.clamp_min(1e-8).log())
        ).sum()
        route_entropy = -(
            routing.clamp_min(1e-8) * routing.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        hard_usage = F.one_hot(assignment, num_classes=3).to(routing).mean(
            dim=(0, 1)
        )
        self._last_auxiliary = {
            "moe_load_balance": load_balance,
            "moe_route_entropy": route_entropy,
            "moe_expert_distillation": distillation_scale
            * self._distillation_loss(expert_outputs, routing),
            "moe_soft_low_usage": usage[0],
            "moe_soft_mid_usage": usage[1],
            "moe_soft_high_usage": usage[2],
            "moe_hard_low_usage": hard_usage[0],
            "moe_hard_mid_usage": hard_usage[1],
            "moe_hard_high_usage": hard_usage[2],
        }

    def forward(
        self,
        routed: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(routed, tuple) or len(routed) != 2:
            raise TypeError("difficulty MoE expects (latent, difficulty)")
        latent, difficulty = routed
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError("MoE latent must have shape [B, N, latent_dim]")
        patch_count = latent.shape[1] - self.num_special_tokens
        if difficulty.shape != (latent.shape[0], patch_count, 1):
            raise ValueError("MoE difficulty shape does not match patch tokens")

        special = latent[:, : self.num_special_tokens]
        patches = latent[:, self.num_special_tokens :]
        routing = self.routing_probabilities(difficulty)
        assignment = routing.detach().argmax(dim=-1)
        hard = not self.training or (
            self.training_step >= self.hard_routing_start_step
        )
        expert_outputs = None
        distillation_scale = 0.0
        if hard:
            reconstructed_patches = self._hard_reconstruct(
                patches, assignment
            )
        else:
            blend = self._routing_blend()
            reconstructed_patches, expert_outputs = self._soft_reconstruct(
                patches, routing, blend
            )
            distillation_scale = 1.0 - blend
        reconstructed_special = self.experts[-1](special)
        self._last_assignment = assignment
        self._update_auxiliary(
            routing,
            assignment,
            expert_outputs,
            distillation_scale,
        )
        return torch.cat(
            [reconstructed_special, reconstructed_patches], dim=1
        )

    def auxiliary_losses(
        self, detach: bool = False
    ) -> dict[str, torch.Tensor]:
        if detach:
            return {
                name: value.detach()
                for name, value in self._last_auxiliary.items()
            }
        return dict(self._last_auxiliary)


class InformationDensityDinomaly(nn.Module):
    """Dinomaly-compatible wrapper that transports graph-connected losses."""

    supports_auxiliary_forward = True

    def __init__(
        self,
        base_model: nn.Module,
        calibration_blend: float = 0.35,
    ) -> None:
        super().__init__()
        if not 0.0 <= calibration_blend <= 1.0:
            raise ValueError("calibration_blend must lie in [0, 1]")
        self.base_model = base_model
        self.calibration_blend = float(calibration_blend)
        projection = self.base_model.bottleneck[0]
        if not isinstance(projection, InformationDensityDownProjection):
            raise TypeError(
                "InformationDensityDinomaly requires its adaptive down-projection"
            )

    @property
    def down_projection(self) -> InformationDensityDownProjection:
        projection = self.base_model.bottleneck[0]
        assert isinstance(projection, InformationDensityDownProjection)
        return projection

    def init_weights(self) -> None:
        self.base_model.init_weights()
        self.down_projection.reset_density_parameters()

    def set_training_step(self, completed_steps: int) -> None:
        self.down_projection.set_training_step(completed_steps)
        moe = self.difficulty_moe
        if moe is not None:
            moe.set_training_step(completed_steps)

    @property
    def difficulty_moe(self) -> DifficultyRoutedMoE | None:
        if len(self.base_model.bottleneck) < 2:
            return None
        reconstruction = self.base_model.bottleneck[1]
        if isinstance(reconstruction, DifficultyRoutedMoE):
            return reconstruction
        return None

    @staticmethod
    def _token_residual(
        encoder_features: Sequence[torch.Tensor],
        decoder_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        residuals = [
            1.0
            - F.cosine_similarity(
                encoder.detach(), decoder.detach(), dim=1
            )
            for encoder, decoder in zip(
                encoder_features, decoder_features, strict=True
            )
        ]
        return torch.stack(residuals, dim=1).mean(dim=1)

    def forward(
        self,
        images: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        encoder_features, decoder_features = self.base_model(images)
        if self.training or return_auxiliary:
            self.down_projection.update_auxiliary(
                self._token_residual(encoder_features, decoder_features)
            )
        if not return_auxiliary:
            return encoder_features, decoder_features
        packed = {
            name: value.reshape(1)
            for name, value in self.auxiliary_losses(detach=False).items()
        }
        return encoder_features, decoder_features, packed

    def auxiliary_losses(
        self, detach: bool = False
    ) -> dict[str, torch.Tensor]:
        losses = self.down_projection.auxiliary_losses(detach=detach)
        moe = self.difficulty_moe
        if moe is not None:
            losses.update(moe.auxiliary_losses(detach=detach))
        return losses

    def regularization_loss(
        self, weights: Mapping[str, float] | None = None
    ) -> torch.Tensor:
        losses = self.auxiliary_losses(detach=False)
        if not losses:
            return next(self.parameters()).sum() * 0.0
        return combine_auxiliary_losses(
            losses,
            weights=weights,
            anchor=next(self.parameters()),
        )

    def calibrate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        expected = self.down_projection.expected_error_map().detach().to(
            anomaly_map
        )
        difficulty = self.down_projection.difficulty_map().detach().to(
            anomaly_map
        )
        if expected.shape[-2:] != anomaly_map.shape[-2:]:
            expected = F.interpolate(
                expected,
                size=anomaly_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            difficulty = F.interpolate(
                difficulty,
                size=anomaly_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        correction = self.calibration_blend * expected * difficulty
        return (anomaly_map - correction).clamp_min(0.0)


def calibrate_information_density_map(
    model: nn.Module,
    anomaly_map: torch.Tensor,
) -> torch.Tensor:
    unwrapped = getattr(model, "module", model)
    calibrator = getattr(unwrapped, "calibrate_anomaly_map", None)
    if calibrator is None:
        return anomaly_map
    return calibrator(anomaly_map)


def set_information_density_step(model: nn.Module, completed_steps: int) -> None:
    unwrapped = getattr(model, "module", model)
    setter = getattr(unwrapped, "set_training_step", None)
    if setter is not None:
        setter(completed_steps)
