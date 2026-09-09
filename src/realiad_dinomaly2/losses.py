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
    valid_view_mask: torch.Tensor | None = None,
    selection_scope: str = "batch",
) -> torch.Tensor:
    if len(encoder_features) != len(decoder_features):
        raise ValueError("encoder/decoder feature groups 数量不一致")
    losses: list[torch.Tensor] = []
    if selection_scope not in {"batch", "object"}:
        raise ValueError("selection_scope must be batch or object")
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
        strict=True,
    ):
        if encoder_feature.shape != decoder_feature.shape:
            raise ValueError("encoder/decoder feature shapes do not match")
        group_ids: torch.Tensor | None = None
        if encoder_feature.ndim == 5:
            batch_size, view_count = encoder_feature.shape[:2]
            encoder_feature = encoder_feature.reshape(
                batch_size * view_count, *encoder_feature.shape[2:]
            )
            decoder_feature = decoder_feature.reshape(
                batch_size * view_count, *decoder_feature.shape[2:]
            )
            if valid_view_mask is not None:
                if valid_view_mask.shape != (batch_size, view_count):
                    raise ValueError("valid_view_mask must have shape [B, V]")
                selected = valid_view_mask.reshape(-1).to(
                    device=encoder_feature.device,
                    dtype=torch.bool,
                )
                if not bool(selected.any()):
                    raise ValueError("reconstruction loss has no valid views")
                encoder_feature = encoder_feature[selected]
                decoder_feature = decoder_feature[selected]
                group_ids = (
                    torch.arange(batch_size, device=selected.device)
                    .unsqueeze(1)
                    .expand(-1, view_count)
                    .reshape(-1)[selected]
                )
            else:
                group_ids = (
                    torch.arange(batch_size, device=encoder_feature.device)
                    .unsqueeze(1)
                    .expand(-1, view_count)
                    .reshape(-1)
                )
        elif encoder_feature.ndim != 4:
            raise ValueError("features must have shape [B,C,H,W] or [B,V,C,H,W]")
        elif selection_scope == "object":
            group_ids = torch.arange(
                encoder_feature.shape[0], device=encoder_feature.device
            )
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
                if selection_scope == "batch":
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
                else:
                    if group_ids is None:
                        raise RuntimeError("object-scoped Loose Loss has no group ids")
                    discarded = torch.zeros_like(point_distance, dtype=torch.bool)
                    for group_id in torch.unique(group_ids):
                        selected_group = group_ids == group_id
                        values = point_distance[selected_group]
                        top_count = max(1, int(values.numel() * kept_fraction))
                        threshold = torch.topk(
                            values.reshape(-1),
                            k=top_count,
                            largest=True,
                        ).values[-1]
                        discarded[selected_group] = values < threshold
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
    multi_view_shape: tuple[int, int] | None = None
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
        strict=True,
    ):
        if encoder_feature.shape != decoder_feature.shape:
            raise ValueError("encoder/decoder feature shapes do not match")
        if encoder_feature.ndim == 5:
            batch_size, view_count = encoder_feature.shape[:2]
            current_shape = (batch_size, view_count)
            if multi_view_shape is None:
                multi_view_shape = current_shape
            elif multi_view_shape != current_shape:
                raise ValueError("multi-view feature groups have inconsistent shapes")
            encoder_feature = encoder_feature.reshape(
                batch_size * view_count, *encoder_feature.shape[2:]
            )
            decoder_feature = decoder_feature.reshape(
                batch_size * view_count, *decoder_feature.shape[2:]
            )
        elif encoder_feature.ndim != 4:
            raise ValueError("features must have shape [B,C,H,W] or [B,V,C,H,W]")
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
    result = (stacked * weight_tensor).sum(dim=1, keepdim=True) / sum(weights)
    if multi_view_shape is not None:
        result = result.reshape(*multi_view_shape, *result.shape[1:])
    return result


def debias_unseen_novelty(
    anomaly_maps: torch.Tensor,
    *,
    baseline_quantile: float = 0.5,
    local_blend: float = 0.5,
    global_retention: float = 0.25,
) -> torch.Tensor:
    """Reduce uniform semantic novelty while retaining localized defects.

    This utility is available in both competition branches so files remain
    import-compatible during branch switches. The baseline branch does not
    enable it unless an explicit unseen-novelty configuration requests it.
    """

    if anomaly_maps.ndim not in {4, 5}:
        raise ValueError(
            "unseen novelty debias expects [B,1,H,W] or [B,V,1,H,W] maps"
        )
    if anomaly_maps.shape[-3] != 1:
        raise ValueError("unseen novelty debias requires a singleton channel")
    if not 0.0 <= float(baseline_quantile) <= 1.0:
        raise ValueError("baseline_quantile must be in [0, 1]")
    if not 0.0 <= float(local_blend) <= 1.0:
        raise ValueError("local_blend must be in [0, 1]")
    if not 0.0 <= float(global_retention) <= 1.0:
        raise ValueError("global_retention must be in [0, 1]")
    if anomaly_maps.numel() == 0 or float(local_blend) == 0.0:
        return anomaly_maps

    original_shape = anomaly_maps.shape
    maps_4d = anomaly_maps.reshape(-1, *original_shape[-3:])
    flattened = maps_4d.float().flatten(start_dim=2)
    baseline = torch.quantile(
        flattened,
        q=float(baseline_quantile),
        dim=2,
        keepdim=True,
    ).reshape(maps_4d.shape[0], 1, 1, 1)
    baseline = baseline.to(dtype=maps_4d.dtype)
    local_excess = (maps_4d - baseline).clamp_min(0.0)
    contrast_map = local_excess + float(global_retention) * baseline
    blend = float(local_blend)
    result = (1.0 - blend) * maps_4d + blend * contrast_map
    return result.reshape(original_shape)
