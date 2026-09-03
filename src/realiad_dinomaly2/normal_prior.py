from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from .competition_data import build_competition_train_dataset
from .config import config_fingerprint
from .data import build_train_dataset
from .losses import anomaly_map
from .modeling import build_model, load_trainable_state_dict
from .runtime import (
    amp_dtype,
    atomic_torch_save,
    atomic_write_json,
    autocast_context,
    resolve_device,
    utc_now,
)


NORMAL_PRIOR_FORMAT_VERSION = 1


def _normal_prior_fingerprint(config: dict[str, Any]) -> str:
    """Hash only settings that change the fitted Train-normal statistics."""

    evaluation = config["evaluation"]
    prior = evaluation.get("normal_prior", {})
    payload = {
        "training_config_fingerprint": config_fingerprint(config),
        "evaluation_amp": bool(evaluation.get("amp", False)),
        "anomaly_map_layer_weights": evaluation.get("anomaly_map_layer_weights"),
        "anomaly_map_align_corners": bool(
            evaluation.get("anomaly_map_align_corners", True)
        ),
        "normal_prior": {
            "resolution": prior.get("resolution", "patch"),
            "category_view_enabled": bool(
                prior.get("category_view_enabled", True)
            ),
            "statistic": prior.get("statistic", "median_mad"),
            "mad_floor_ratio": float(prior.get("mad_floor_ratio", 0.05)),
            "eps": float(prior.get("eps", 1e-6)),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_prior_path(config: dict[str, Any]) -> Path:
    prior_config = config["evaluation"].get("normal_prior", {})
    configured = prior_config.get("artifact_path")
    output_dir = Path(config["experiment"]["output_dir"])
    if configured is None:
        return output_dir / "normal_prior" / "normal_prior.pt"
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else output_dir / path


class NormalPrior:
    """Validated Train-normal anomaly-map statistics."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.category_view = payload.get("category_view", {})
        self.view_global = payload.get("view_global", {})

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.payload["metadata"])

    def calibrate(
        self,
        raw_maps: torch.Tensor,
        *,
        categories: Sequence[str],
        view_ids: torch.Tensor,
        valid_view_mask: torch.Tensor | None,
        config: dict[str, Any],
    ) -> torch.Tensor:
        """Softly suppress repeatable normal response without rank flattening."""

        prior_config = config["evaluation"]["normal_prior"]
        original_ndim = raw_maps.ndim
        if original_ndim == 4:
            raw_maps = raw_maps.unsqueeze(1)
        if raw_maps.ndim != 5 or raw_maps.shape[2] != 1:
            raise ValueError("prior calibration expects [B,V,1,H,W] maps")
        batch_size, view_count = raw_maps.shape[:2]
        if len(categories) != batch_size:
            raise ValueError("categories must contain one entry per object")
        view_ids = view_ids.to(device=raw_maps.device, dtype=torch.long)
        if view_ids.ndim == 1:
            if view_ids.numel() == view_count:
                view_ids = view_ids.unsqueeze(0).expand(batch_size, -1)
            elif view_count == 1 and view_ids.numel() == batch_size:
                view_ids = view_ids.unsqueeze(1)
        if view_ids.shape != (batch_size, view_count):
            raise ValueError("view_ids must have shape [B,V]")
        if valid_view_mask is None:
            valid_view_mask = torch.ones(
                (batch_size, view_count),
                dtype=torch.bool,
                device=raw_maps.device,
            )
        else:
            valid_view_mask = valid_view_mask.to(
                device=raw_maps.device,
                dtype=torch.bool,
            )
            if valid_view_mask.ndim == 1:
                if valid_view_mask.numel() == view_count:
                    valid_view_mask = valid_view_mask.unsqueeze(0).expand(
                        batch_size, -1
                    )
                elif view_count == 1 and valid_view_mask.numel() == batch_size:
                    valid_view_mask = valid_view_mask.unsqueeze(1)
        if valid_view_mask.shape != (batch_size, view_count):
            raise ValueError("valid_view_mask must have shape [B,V]")
        calibrated = raw_maps.clone()
        threshold = float(prior_config.get("threshold", 2.0))
        temperature = float(prior_config.get("temperature", 0.5))
        blend = float(prior_config.get("blend", 0.8))
        eps = float(prior_config.get("eps", 1e-6))
        category_enabled = bool(
            prior_config.get("category_view_enabled", True)
        )
        fallback = str(prior_config.get("unseen_fallback", "view_global"))

        for batch_index, category in enumerate(categories):
            category_stats = self.category_view.get(str(category), {})
            for view_index in range(view_count):
                if not bool(valid_view_mask[batch_index, view_index]):
                    continue
                camera_id = str(int(view_ids[batch_index, view_index]))
                stats = category_stats.get(camera_id) if category_enabled else None
                if stats is None and fallback == "view_global":
                    stats = self.view_global.get(camera_id)
                if stats is None:
                    continue
                median = stats["median"].to(
                    device=raw_maps.device,
                    dtype=raw_maps.dtype,
                )
                mad = stats["mad"].to(
                    device=raw_maps.device,
                    dtype=raw_maps.dtype,
                )
                current = raw_maps[batch_index, view_index]
                if median.shape != current.shape:
                    raise ValueError(
                        "normal prior resolution does not match raw patch map: "
                        f"{tuple(median.shape)} != {tuple(current.shape)}"
                    )
                mad_floor = stats.get("mad_floor")
                if mad_floor is None:
                    # Backward compatibility is defensive only; current
                    # fingerprints force legacy artifacts to be rebuilt.
                    denominator = mad.clamp_min(eps)
                else:
                    denominator = mad.clamp_min(
                        mad_floor.to(device=raw_maps.device, dtype=raw_maps.dtype)
                    )
                normalized_excess = (current - median) / denominator
                gate = torch.sigmoid(
                    (normalized_excess - threshold) / temperature
                )
                calibrated[batch_index, view_index] = current * (
                    (1.0 - blend) + blend * gate
                )
        return calibrated[:, 0] if original_ndim == 4 else calibrated


def _prior_metadata(
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    categories: Sequence[str],
    side: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_config = config["evaluation"]["normal_prior"]
    return {
        "format_version": NORMAL_PRIOR_FORMAT_VERSION,
        "created_at": utc_now(),
        "source_split": "Train",
        "source_labels": "normal_only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_fingerprint": config_fingerprint(config),
        "normal_prior_fingerprint": _normal_prior_fingerprint(config),
        "categories": sorted(str(category) for category in categories),
        "num_views": int(config["model"].get("multi_view", {}).get("num_views", 5)),
        "resolution": "patch",
        "height": side,
        "width": side,
        "statistic": str(prior_config.get("statistic", "median_mad")),
        "threshold": float(prior_config.get("threshold", 2.0)),
        "temperature": float(prior_config.get("temperature", 0.5)),
        "blend": float(prior_config.get("blend", 0.8)),
        "eps": float(prior_config.get("eps", 1e-6)),
        "entries": entries,
    }


def validate_normal_prior(
    payload: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != NORMAL_PRIOR_FORMAT_VERSION:
        raise ValueError("normal prior format_version is incompatible")
    expected_fingerprint = config_fingerprint(config)
    if metadata.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("normal prior/config fingerprint mismatch")
    if metadata.get("normal_prior_fingerprint") != _normal_prior_fingerprint(
        config
    ):
        raise ValueError("normal prior/evaluation fingerprint mismatch")
    expected_checkpoint_sha = file_sha256(checkpoint_path)
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("normal prior/checkpoint fingerprint mismatch")
    if metadata.get("source_split") != "Train" or metadata.get(
        "source_labels"
    ) != "normal_only":
        raise ValueError("normal prior is not marked as Train-normal-only")


def load_normal_prior(
    path: str | Path,
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> NormalPrior:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("normal prior artifact is not a mapping")
    validate_normal_prior(payload, config, checkpoint_path)
    return NormalPrior(payload)

def _loader(dataset, config: dict[str, Any]) -> DataLoader:
    prior_config = config["evaluation"]["normal_prior"]
    workers = int(prior_config.get("num_workers", config["evaluation"]["num_workers"]))
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


def _statistics(
    values: list[torch.Tensor],
    *,
    mad_floor_ratio: float,
    eps: float,
) -> tuple[dict[str, torch.Tensor], int]:
    if not values:
        raise ValueError("cannot fit prior statistics without samples")
    stacked = torch.stack(values, dim=0).float()
    median = stacked.median(dim=0).values
    mad = (stacked - median).abs().median(dim=0).values
    positive = mad[mad > eps]
    robust_scale = positive.median() if positive.numel() else mad.new_tensor(eps)
    mad_floor = torch.maximum(
        mad.new_tensor(eps), robust_scale * float(mad_floor_ratio)
    )
    return {
        "median": median,
        "mad": mad,
        "mad_floor": mad_floor,
    }, int(stacked.shape[0])


@torch.inference_mode()
def fit_normal_prior(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    categories: Iterable[str],
    *,
    force: bool = False,
) -> NormalPrior | None:
    """Fit only from the configured normal Train split, never Test/Test_A."""

    prior_config = config["evaluation"].get("normal_prior", {})
    if not bool(prior_config.get("enabled", False)):
        return None
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    artifact_path = normal_prior_path(config)
    if artifact_path.is_file() and not force:
        try:
            return load_normal_prior(artifact_path, config, checkpoint_path)
        except ValueError as exc:
            # A completed/resumed training run may rewrite final_model.pt while
            # leaving a prior fitted from an earlier checkpoint in place. Do
            # not reuse that stale artifact, but also do not require users to
            # delete cache files manually. Other load/deserialization errors
            # still propagate instead of silently hiding a damaged artifact.
            if not str(exc).startswith("normal prior/"):
                raise

    category_list = sorted(str(category) for category in categories)
    multi_view_config = dict(config["model"].get("multi_view", {}))
    multi_view_enabled = bool(multi_view_config.get("enabled", False))
    num_views = int(multi_view_config.get("num_views", 5))
    dataset_config = config["dataset"]
    missing_view_policy = str(
        multi_view_config.get("missing_view_policy", "error")
    )
    if dataset_config.get("type") == "competition_folders":
        dataset, _ = build_competition_train_dataset(
            train_dir=Path(dataset_config["train_dir"]),
            categories=category_list,
            category_limit=None,
            image_size=int(dataset_config["image_size"]),
            crop_size=int(dataset_config["crop_size"]),
            multi_view_enabled=multi_view_enabled,
            num_views=num_views,
            missing_view_policy=missing_view_policy,
        )
    else:
        dataset = build_train_dataset(
            json_dir=Path(dataset_config["json_dir"]),
            image_dir=Path(
                dataset_config.get("train_image_dir")
                or dataset_config["image_dir"]
            ),
            categories=category_list,
            image_size=int(dataset_config["image_size"]),
            crop_size=int(dataset_config["crop_size"]),
            multi_view_enabled=multi_view_enabled,
            num_views=num_views,
            missing_view_policy=missing_view_policy,
        )

    device = resolve_device(str(config["runtime"]["device"]))
    dtype = (
        amp_dtype(config, device)
        if bool(config["evaluation"].get("amp", False))
        else None
    )
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected_fingerprint = config_fingerprint(config)
    if checkpoint_payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint/config fingerprint mismatch while fitting prior")
    bundle = build_model(config, device)
    expected_backbone = checkpoint_payload.get("backbone_sha256")
    if expected_backbone is not None and expected_backbone != bundle.backbone_sha256:
        raise ValueError("checkpoint backbone does not match while fitting prior")
    load_trainable_state_dict(bundle, checkpoint_payload["model"])
    bundle.model.eval()

    category_values: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    global_values: dict[int, list[torch.Tensor]] = defaultdict(list)
    side = int(config["dataset"]["crop_size"]) // 14
    layer_weights = config["evaluation"].get("anomaly_map_layer_weights")
    align_corners = bool(
        config["evaluation"].get("anomaly_map_align_corners", True)
    )
    for batch in _loader(dataset, config):
        if multi_view_enabled:
            images = batch["images"].to(device, non_blocking=True)
            view_ids = batch["view_ids"].to(device, non_blocking=True)
            valid_view_mask = batch["valid_view_mask"].to(
                device, non_blocking=True
            )
            categories_batch = [str(value) for value in batch["category"]]
            with autocast_context(dtype, device):
                encoder_features, decoder_features = bundle.model(
                    images,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                )
                maps = anomaly_map(
                    encoder_features,
                    decoder_features,
                    output_size=side,
                    layer_weights=layer_weights,
                    align_corners=align_corners,
                )
        else:
            images = batch["image"].to(device, non_blocking=True)
            categories_batch = [str(value) for value in batch["category"]]
            view_ids = batch["view_id"].to(device, dtype=torch.long)
            if dataset_config.get("type") != "competition_folders":
                # Real-IAD names cameras C1..C5; competition filenames are
                # already zero-based 0.png..4.png.
                view_ids = view_ids - 1
            valid_view_mask = torch.ones_like(view_ids, dtype=torch.bool)
            with autocast_context(dtype, device):
                encoder_features, decoder_features = bundle.model(images)
                maps = anomaly_map(
                    encoder_features,
                    decoder_features,
                    output_size=side,
                    layer_weights=layer_weights,
                    align_corners=align_corners,
                ).unsqueeze(1)
            view_ids = view_ids.unsqueeze(1)
            valid_view_mask = valid_view_mask.unsqueeze(1)

        maps = maps.detach().float().cpu()
        view_ids_cpu = view_ids.cpu()
        valid_cpu = valid_view_mask.cpu()
        for batch_index, category in enumerate(categories_batch):
            for view_index in range(maps.shape[1]):
                if not bool(valid_cpu[batch_index, view_index]):
                    continue
                camera_id = int(view_ids_cpu[batch_index, view_index])
                value = maps[batch_index, view_index]
                if bool(prior_config.get("category_view_enabled", True)):
                    category_values[category][camera_id].append(value)
                global_values[camera_id].append(value)

    category_view: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    view_global: dict[str, dict[str, torch.Tensor]] = {}
    entries: list[dict[str, Any]] = []
    mad_floor_ratio = float(prior_config.get("mad_floor_ratio", 0.05))
    eps = float(prior_config.get("eps", 1e-6))
    for category in sorted(category_values):
        category_view[category] = {}
        for view_id in sorted(category_values[category]):
            stats, sample_count = _statistics(
                category_values[category][view_id],
                mad_floor_ratio=mad_floor_ratio,
                eps=eps,
            )
            category_view[category][str(view_id)] = stats
            entries.append(
                {
                    "scope": "category_view",
                    "category": category,
                    "view_id": view_id,
                    "height": side,
                    "width": side,
                    "samples": sample_count,
                }
            )
    for view_id in sorted(global_values):
        stats, sample_count = _statistics(
            global_values[view_id],
            mad_floor_ratio=mad_floor_ratio,
            eps=eps,
        )
        view_global[str(view_id)] = stats
        entries.append(
            {
                "scope": "view_global",
                "category": None,
                "view_id": view_id,
                "height": side,
                "width": side,
                "samples": sample_count,
            }
        )
    checkpoint_sha = file_sha256(checkpoint_path)
    metadata = _prior_metadata(
        config,
        checkpoint_path,
        checkpoint_sha,
        category_list,
        side,
        entries,
    )
    payload = {
        "metadata": metadata,
        "category_view": category_view,
        "view_global": view_global,
    }
    atomic_torch_save(artifact_path, payload)
    atomic_write_json(
        artifact_path.with_suffix(".json"),
        {
            **metadata,
            "artifact": str(artifact_path),
            "category_view_entries": sum(
                len(value) for value in category_view.values()
            ),
            "view_global_entries": len(view_global),
        },
    )
    return NormalPrior(payload)
