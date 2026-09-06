from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from .clip_semantic import FrozenClipBrokenSegmenter
from .competition_data import build_competition_train_dataset
from .config import config_fingerprint
from .runtime import (
    atomic_torch_save,
    atomic_write_json,
    resolve_device,
    utc_now,
)


CLIP_NORMAL_PRIOR_FORMAT_VERSION = 1


def clip_prior_config_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "unseen_clip": config["evaluation"].get("unseen_clip", {}),
        "train_dataset": {
            key: config["dataset"].get(key)
            for key in (
                "type",
                "train_dir",
                "categories",
                "category_limit",
                "image_size",
                "crop_size",
            )
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clip_normal_prior_path(config: dict[str, Any]) -> Path:
    clip_config = config["evaluation"].get("unseen_clip", {})
    prior_config = clip_config.get("normal_prior", {})
    configured = prior_config.get("artifact_path")
    output_dir = Path(config["experiment"]["output_dir"])
    if configured is None:
        return output_dir / "normal_prior" / "clip_normal_prior.pt"
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else output_dir / path


class ClipNormalPrior:
    """View-global CLIP responses fitted only from normal Train images."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.view_global = payload.get("view_global", {})

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.payload["metadata"])

    def calibrate(
        self,
        probability_maps: torch.Tensor,
        *,
        view_ids: torch.Tensor,
        valid_view_mask: torch.Tensor | None,
        config: dict[str, Any],
    ) -> torch.Tensor:
        original_ndim = probability_maps.ndim
        if original_ndim == 4:
            probability_maps = probability_maps.unsqueeze(1)
        if probability_maps.ndim != 5 or probability_maps.shape[2] != 1:
            raise ValueError("CLIP prior expects [B,V,1,H,W] maps")
        batch_size, view_count = probability_maps.shape[:2]
        view_ids = view_ids.to(probability_maps.device, dtype=torch.long)
        if view_ids.ndim == 1:
            if view_count == 1 and view_ids.numel() == batch_size:
                view_ids = view_ids.unsqueeze(1)
            elif view_ids.numel() == view_count:
                view_ids = view_ids.unsqueeze(0).expand(batch_size, -1)
        if view_ids.shape != (batch_size, view_count):
            raise ValueError("view_ids must have shape [B,V]")
        if valid_view_mask is None:
            valid_view_mask = torch.ones_like(view_ids, dtype=torch.bool)
        else:
            valid_view_mask = valid_view_mask.to(
                probability_maps.device, dtype=torch.bool
            )
            if valid_view_mask.ndim == 1:
                if view_count == 1 and valid_view_mask.numel() == batch_size:
                    valid_view_mask = valid_view_mask.unsqueeze(1)
                elif valid_view_mask.numel() == view_count:
                    valid_view_mask = valid_view_mask.unsqueeze(0).expand(
                        batch_size, -1
                    )
        if valid_view_mask.shape != (batch_size, view_count):
            raise ValueError("valid_view_mask must have shape [B,V]")

        prior_config = config["evaluation"]["unseen_clip"]["normal_prior"]
        threshold = float(prior_config.get("threshold", 2.0))
        temperature = float(prior_config.get("temperature", 0.5))
        blend = float(prior_config.get("blend", 0.8))
        eps = float(prior_config.get("eps", 1e-6))
        mad_floor = float(prior_config.get("mad_floor", 0.02))
        calibrated = probability_maps.clone()
        for batch_index in range(batch_size):
            for view_index in range(view_count):
                if not bool(valid_view_mask[batch_index, view_index]):
                    continue
                camera_id = str(int(view_ids[batch_index, view_index]))
                stats = self.view_global.get(camera_id)
                if stats is None:
                    continue
                current = probability_maps[batch_index, view_index]
                median = stats["median"].to(current.device, current.dtype)
                mad = stats["mad"].to(current.device, current.dtype)
                if median.shape != current.shape:
                    raise ValueError(
                        "CLIP prior resolution does not match probability map: "
                        f"{tuple(median.shape)} != {tuple(current.shape)}"
                    )
                scale = mad.clamp_min(mad_floor) + eps
                normalized_excess = (current - median) / scale
                gate = torch.sigmoid(
                    (normalized_excess - threshold) / temperature
                )
                calibrated[batch_index, view_index] = current * (
                    (1.0 - blend) + blend * gate
                )
        return calibrated[:, 0] if original_ndim == 4 else calibrated


def validate_clip_normal_prior(
    payload: dict[str, Any], config: dict[str, Any]
) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != CLIP_NORMAL_PRIOR_FORMAT_VERSION:
        raise ValueError("CLIP normal prior format_version is incompatible")
    if metadata.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("CLIP normal prior/config fingerprint mismatch")
    if metadata.get("clip_config_fingerprint") != clip_prior_config_fingerprint(
        config
    ):
        raise ValueError("CLIP normal prior/semantic config fingerprint mismatch")
    if metadata.get("source_split") != "Train" or metadata.get(
        "source_labels"
    ) != "normal_only":
        raise ValueError("CLIP normal prior is not marked Train-normal-only")


def load_clip_normal_prior(
    path: str | Path, config: dict[str, Any]
) -> ClipNormalPrior:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("CLIP normal prior artifact is not a mapping")
    validate_clip_normal_prior(payload, config)
    return ClipNormalPrior(payload)


def _loader(dataset, config: dict[str, Any]) -> DataLoader:
    prior_config = config["evaluation"]["unseen_clip"]["normal_prior"]
    workers = int(
        prior_config.get("num_workers", config["evaluation"]["num_workers"])
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            prior_config.get("batch_size", config["evaluation"]["batch_size"])
        ),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(config["runtime"]["pin_memory"]),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(
            config["runtime"]["persistent_workers"]
        )
        kwargs["prefetch_factor"] = int(config["runtime"]["prefetch_factor"])
    return DataLoader(**kwargs)


def _statistics(values: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if not values:
        raise ValueError("cannot fit CLIP prior without samples")
    stacked = torch.stack(values, dim=0).float()
    median = stacked.median(dim=0).values
    mad = (stacked - median).abs().median(dim=0).values
    return {"median": median, "mad": mad}


@torch.inference_mode()
def fit_clip_normal_prior(
    config: dict[str, Any],
    categories: Iterable[str],
    *,
    force: bool = False,
) -> ClipNormalPrior | None:
    """Fit a view-global CLIP prior from competition Train normals only."""

    clip_config = config["evaluation"].get("unseen_clip", {})
    prior_config = clip_config.get("normal_prior", {})
    if not bool(clip_config.get("enabled", False)) or not bool(
        prior_config.get("enabled", False)
    ):
        return None
    if config["dataset"].get("type") != "competition_folders":
        raise ValueError("CLIP normal prior currently requires competition_folders")
    artifact_path = clip_normal_prior_path(config)
    if artifact_path.is_file() and not force:
        try:
            return load_clip_normal_prior(artifact_path, config)
        except ValueError as exc:
            if not str(exc).startswith("CLIP normal prior/"):
                raise

    category_list = sorted(str(category) for category in categories)
    dataset_config = config["dataset"]
    multi_view_config = dict(config["model"].get("multi_view", {}))
    multi_view_enabled = bool(multi_view_config.get("enabled", False))
    dataset, _ = build_competition_train_dataset(
        train_dir=Path(dataset_config["train_dir"]),
        categories=category_list,
        category_limit=None,
        image_size=int(dataset_config["image_size"]),
        crop_size=int(dataset_config["crop_size"]),
        multi_view_enabled=multi_view_enabled,
        num_views=int(multi_view_config.get("num_views", 5)),
        missing_view_policy=str(
            multi_view_config.get("missing_view_policy", "error")
        ),
    )
    device = resolve_device(str(config["runtime"]["device"]))
    segmenter = FrozenClipBrokenSegmenter(clip_config, device).eval()
    global_values: dict[int, list[torch.Tensor]] = defaultdict(list)
    for batch in _loader(dataset, config):
        if "images" in batch:
            images = batch["images"].to(
                device, non_blocking=bool(config["runtime"]["pin_memory"])
            )
            view_ids = batch["view_ids"].to(device, dtype=torch.long)
            valid_view_mask = batch["valid_view_mask"].to(
                device, dtype=torch.bool
            )
            batch_size, view_count = images.shape[:2]
            flat_maps = segmenter(
                images.reshape(batch_size * view_count, *images.shape[2:])
            )
            maps = flat_maps.reshape(
                batch_size, view_count, *flat_maps.shape[1:]
            )
        else:
            images = batch["image"].to(
                device, non_blocking=bool(config["runtime"]["pin_memory"])
            )
            maps = segmenter(images).unsqueeze(1)
            view_ids = batch["view_id"].to(device, dtype=torch.long).unsqueeze(1)
            valid_view_mask = torch.ones_like(view_ids, dtype=torch.bool)

        maps_cpu = maps.float().cpu()
        view_ids_cpu = view_ids.cpu()
        valid_cpu = valid_view_mask.cpu()
        for batch_index in range(maps_cpu.shape[0]):
            for view_index in range(maps_cpu.shape[1]):
                if not bool(valid_cpu[batch_index, view_index]):
                    continue
                camera_id = int(view_ids_cpu[batch_index, view_index])
                global_values[camera_id].append(
                    maps_cpu[batch_index, view_index]
                )

    view_global = {
        str(view_id): _statistics(global_values[view_id])
        for view_id in sorted(global_values)
    }
    first_stats = next(iter(view_global.values()), None)
    if first_stats is None:
        raise ValueError("CLIP normal prior collected no valid views")
    height, width = first_stats["median"].shape[-2:]
    metadata = {
        "format_version": CLIP_NORMAL_PRIOR_FORMAT_VERSION,
        "created_at": utc_now(),
        "source_split": "Train",
        "source_labels": "normal_only",
        "config_fingerprint": config_fingerprint(config),
        "clip_config_fingerprint": clip_prior_config_fingerprint(config),
        "categories": category_list,
        "model_name": str(clip_config["model_name"]),
        "pretrained": str(clip_config["pretrained"]),
        "height": int(height),
        "width": int(width),
        "views": {
            str(view_id): len(global_values[view_id])
            for view_id in sorted(global_values)
        },
    }
    payload = {"metadata": metadata, "view_global": view_global}
    atomic_torch_save(artifact_path, payload)
    atomic_write_json(
        artifact_path.with_suffix(".json"),
        {**metadata, "artifact": str(artifact_path)},
    )
    return ClipNormalPrior(payload)
