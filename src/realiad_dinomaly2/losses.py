from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F


def _scale_gradient(
    gradient: torch.Tensor,
    selection: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    expanded = selection.expand_as(gradient)
    gradient[expanded] *= factor
    return gradient


def reconstruction_loss(
    encoder_features: list[torch.Tensor],
    decoder_features: list[torch.Tensor],
    discard_rate: float,
    loose_loss: bool,
    discarded_gradient_factor: float = 0.1,
) -> torch.Tensor:
    if len(encoder_features) != len(decoder_features):
        raise ValueError("encoder/decoder feature groups 数量不一致")
    losses: list[torch.Tensor] = []
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
        strict=True,
    ):
        target = encoder_feature.detach()
        if loose_loss:
            with torch.no_grad():
                point_distance = (
                    1.0
                    - F.cosine_similarity(
                        target,
                        decoder_feature.detach(),
                        dim=1,
                    )
                ).unsqueeze(1)
                kept_fraction = max(0.0, min(1.0, 1.0 - discard_rate))
                top_count = max(
                    1,
                    int(point_distance.numel() * kept_fraction),
                )
                threshold = torch.topk(
                    point_distance.reshape(-1),
                    k=top_count,
                    largest=True,
                ).values[-1]
                discarded = point_distance < threshold
            decoder_feature.register_hook(
                partial(
                    _scale_gradient,
                    selection=discarded,
                    factor=discarded_gradient_factor,
                )
            )

        cosine = F.cosine_similarity(
            target.reshape(target.shape[0], -1),
            decoder_feature.reshape(decoder_feature.shape[0], -1),
            dim=1,
        )
        losses.append((1.0 - cosine).mean())
    return torch.stack(losses).mean()


def discard_rate_for_step(
    completed_steps: int,
    warmup_steps: int,
    final_rate: float,
) -> float:
    if warmup_steps <= 0:
        return final_rate
    return min(final_rate, final_rate * completed_steps / warmup_steps)


def anomaly_map(
    encoder_features: list[torch.Tensor],
    decoder_features: list[torch.Tensor],
    output_size: int | tuple[int, int],
    layer_weights: list[float] | None = None,
    align_corners: bool = True,
) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if layer_weights is None:
        weights = [1.0] * len(encoder_features)
    else:
        weights = [float(value) for value in layer_weights]
        if len(weights) != len(encoder_features):
            raise ValueError(
                "anomaly_map layer_weights length must match feature groups: "
                f"{len(weights)} != {len(encoder_features)}"
            )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("anomaly_map layer_weights must be non-negative")
    maps: list[torch.Tensor] = []
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
        strict=True,
    ):
        current = 1.0 - F.cosine_similarity(
            encoder_feature,
            decoder_feature,
            dim=1,
        )
        current = F.interpolate(
            current.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=align_corners,
        )
        maps.append(current)
    stacked = torch.cat(maps, dim=1)
    weight_tensor = stacked.new_tensor(weights).view(1, -1, 1, 1)
    return (stacked * weight_tensor).sum(dim=1, keepdim=True) / sum(weights)
