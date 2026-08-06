from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .config import config_fingerprint
from .losses import anomaly_map
from .runtime import amp_dtype, autocast_context, resolve_device


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency values cannot be empty")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latency(
    elapsed_seconds: Sequence[float],
    *,
    threshold_seconds: float = 1.0,
) -> dict[str, Any]:
    """Summarize single-frame timings with a reproducible validity rule."""

    if threshold_seconds <= 0:
        raise ValueError("threshold_seconds must be positive")
    values = [float(value) for value in elapsed_seconds]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("elapsed_seconds must contain finite non-negative values")
    mean = sum(values) / len(values)
    p95 = _percentile(values, 95.0)
    return {
        "sample_count": len(values),
        "mean_seconds": mean,
        "p50_seconds": _percentile(values, 50.0),
        "p95_seconds": p95,
        "max_seconds": max(values),
        "threshold_seconds": float(threshold_seconds),
        "criterion": "mean_seconds <= threshold and p95_seconds <= threshold",
        "valid": mean <= threshold_seconds and p95 <= threshold_seconds,
    }


@torch.inference_mode()
def benchmark_single_frame_latency(
    config: dict[str, Any],
    checkpoint: str | Path,
    category: str,
) -> dict[str, Any]:
    """Benchmark model + anomaly-map postprocessing for one real test image.

    Image decoding and resize/normalization happen before timing.  The measured
    path includes the frozen encoder, normality adapter, decoder, anomaly map,
    evaluation resize, Gaussian smoothing, and image-score reduction.
    """

    # Keep the pure latency summary importable in lightweight environments.
    from .data import RealIADMultiViewDataset, RealIADVarietyDataset
    from .metrics import GaussianFilter, top_ratio_mean
    from .modeling import build_model, load_trainable_state_dict
    from .normal_prior import load_normal_prior, normal_prior_path

    latency_config = config.get("latency", {})
    warmup = int(latency_config.get("warmup_iterations", 10))
    repeats = int(latency_config.get("measure_iterations", 50))
    threshold = float(latency_config.get("threshold_seconds", 1.0))
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup must be >= 0 and measure_iterations > 0")

    device = resolve_device(str(config["runtime"]["device"]))
    if device.type != "cuda":
        raise RuntimeError("official latency validation requires a CUDA device")
    dtype = (
        amp_dtype(config, device)
        if bool(config["evaluation"].get("amp", False))
        else None
    )
    dataset_config = config["dataset"]
    multi_view_config = dict(config["model"].get("multi_view", {}))
    multi_view_enabled = bool(multi_view_config.get("enabled", False))
    common_args = {
        "json_dir": Path(dataset_config["json_dir"]),
        "image_dir": Path(dataset_config["image_dir"]),
        "category": category,
        "phase": "test",
        "image_size": int(dataset_config["image_size"]),
        "crop_size": int(dataset_config["crop_size"]),
        "image_label_policy": str(dataset_config["image_label_policy"]),
        "missing_anomaly_mask_policy": str(
            dataset_config["missing_anomaly_mask_policy"]
        ),
        "mask_resize_semantics": str(
            dataset_config.get(
                "mask_resize_semantics",
                "upstream_bilinear_nonzero",
            )
        ),
    }
    if multi_view_enabled:
        dataset = RealIADMultiViewDataset(
            **common_args,
            num_views=int(multi_view_config.get("num_views", 5)),
            missing_view_policy=str(
                multi_view_config.get("missing_view_policy", "error")
            ),
            max_objects=1,
        )
    else:
        dataset = RealIADVarietyDataset(**common_args, max_items=1)
    sample = dataset[0]
    image_key = "images" if multi_view_enabled else "image"
    image = sample[image_key].unsqueeze(0).to(device)
    view_ids = (
        sample["view_ids"].unsqueeze(0).to(device)
        if multi_view_enabled
        else torch.tensor([int(sample["view_id"]) - 1], device=device)
    )
    valid_view_mask = (
        sample["valid_view_mask"].unsqueeze(0).to(device)
        if multi_view_enabled
        else torch.ones(1, dtype=torch.bool, device=device)
    )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_fingerprint = config_fingerprint(config)
    if payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("latency checkpoint does not match the experiment config")

    bundle = build_model(config, device)
    if (
        payload.get("backbone_sha256") is not None
        and payload["backbone_sha256"] != bundle.backbone_sha256
    ):
        raise ValueError("latency checkpoint backbone SHA-256 does not match")
    load_trainable_state_dict(bundle, payload["model"])
    bundle.model.eval()
    normal_prior = None
    if bool(config["evaluation"].get("normal_prior", {}).get("enabled", False)):
        normal_prior = load_normal_prior(
            normal_prior_path(config),
            config,
            checkpoint_path,
        )

    evaluation = config["evaluation"]
    resize_mask = int(evaluation["resize_mask"])
    gaussian = GaussianFilter(
        kernel_size=int(evaluation["gaussian_kernel_size"]),
        sigma=float(evaluation["gaussian_sigma"]),
    ).to(device)
    gaussian.eval()

    def run_once() -> None:
        with autocast_context(dtype, device):
            if multi_view_enabled:
                encoder_features, decoder_features = bundle.model(
                    image,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                )
            else:
                encoder_features, decoder_features = bundle.model(image)
            maps = anomaly_map(
                encoder_features,
                decoder_features,
                output_size=int(dataset_config["crop_size"]) // 14,
                layer_weights=evaluation.get("anomaly_map_layer_weights"),
                align_corners=bool(
                    evaluation.get("anomaly_map_align_corners", True)
                ),
            )
        if normal_prior is not None:
            maps = normal_prior.calibrate(
                maps,
                categories=[category],
                view_ids=view_ids,
                valid_view_mask=valid_view_mask,
                config=config,
            )
        if maps.ndim == 5:
            maps = maps.reshape(-1, *maps.shape[2:])
        maps = F.interpolate(
            maps.float(),
            size=(resize_mask, resize_mask),
            mode="bilinear",
            align_corners=False,
        )
        maps = gaussian(maps)
        top_ratio_mean(maps, ratio=float(evaluation["image_top_ratio"]))

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        run_once()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)

    result = summarize_latency(timings, threshold_seconds=threshold)
    result.update(
        {
            "status": "valid" if result["valid"] else "invalid_latency",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_properties(device).name,
            "category": category,
            "image_path": str(
                sample.get("image_path", sample.get("image_paths"))
            ),
            "warmup_iterations": warmup,
            "scope": (
                "model forward + anomaly map + metric resize + Gaussian + "
                "image score and optional Train-normal prior; disk "
                "decode/preprocessing excluded"
            ),
            "batch_unit": "object" if multi_view_enabled else "view",
            "equivalent_views": (
                int(multi_view_config.get("num_views", 5))
                if multi_view_enabled
                else 1
            ),
            "checkpoint": str(checkpoint_path),
        }
    )
    return result
