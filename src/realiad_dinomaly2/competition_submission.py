from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .competition_data import (
    CompetitionFolderDataset,
    CompetitionManifest,
    CompetitionObjectDataset,
    CompetitionView,
    scan_competition_split,
)
from .config import config_fingerprint
from .clip_normal_prior import (
    clip_normal_prior_path,
    load_clip_normal_prior,
)
from .clip_semantic import (
    FrozenClipBrokenSegmenter,
    fuse_unseen_anomaly_map,
)
from .losses import anomaly_map
from .modeling import build_model, load_trainable_state_dict
from .normal_prior import (
    file_sha256,
    load_normal_prior,
    normal_prior_path,
)
from .runtime import (
    amp_dtype,
    atomic_write_json,
    autocast_context,
    resolve_device,
    setup_logger,
    setup_seed,
    utc_now,
)
from .zero_shot_model import (
    load_zero_shot_segmenter,
    zero_shot_checkpoint_path,
)


ZERO_SHOT_SCORING_VERSION = "mask_topk_v1"


def resolve_competition_checkpoint(
    output_dir: Path,
    checkpoint: str,
) -> Path:
    if checkpoint == "auto":
        best = output_dir / "checkpoints" / "best_model.pt"
        final = output_dir / "checkpoints" / "final_model.pt"
        path = best if best.is_file() else final
    else:
        path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Competition checkpoint does not exist: {path}")
    return path.resolve()


def _manifest_digest(manifest: CompetitionManifest) -> str:
    digest = hashlib.sha256()
    for view in manifest.views:
        stat = view.image_path.stat()
        relative = view.image_path.relative_to(manifest.root).as_posix()
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _submission_signature(
    config: dict[str, Any],
    checkpoint_path: Path,
    manifest: CompetitionManifest,
    *,
    unseen_clip_active: bool = False,
    zero_shot_active: bool = False,
) -> tuple[str, dict[str, Any]]:
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "config_fingerprint": config_fingerprint(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "test_root": str(manifest.root),
        "test_manifest_sha256": _manifest_digest(manifest),
        "evaluation": config["evaluation"],
        "submission": config["submission"],
    }
    if bool(config["evaluation"].get("normal_prior", {}).get("enabled", False)):
        prior_path = normal_prior_path(config)
        if not prior_path.is_file():
            raise FileNotFoundError(
                f"Competition normal prior does not exist: {prior_path}"
            )
        payload["normal_prior"] = {
            "path": str(prior_path),
            "sha256": file_sha256(prior_path),
        }
    clip_config = config["evaluation"].get("unseen_clip", {})
    clip_prior_config = clip_config.get("normal_prior", {})
    if unseen_clip_active and bool(clip_prior_config.get("enabled", False)):
        clip_prior = clip_normal_prior_path(config)
        if not clip_prior.is_file():
            raise FileNotFoundError(
                f"CLIP normal prior does not exist: {clip_prior}"
            )
        payload["clip_normal_prior"] = {
            "path": str(clip_prior),
            "sha256": file_sha256(clip_prior),
        }
    if zero_shot_active:
        backend = config["zero_shot"].get("backend", "synthetic")
        zero_path = zero_shot_checkpoint_path(config)
        if not zero_path.is_file():
            raise FileNotFoundError(
                f"Zero-shot checkpoint does not exist: {zero_path}"
            )
        payload["zero_shot"] = {
            "backend": backend,
            "config": config["zero_shot"],
            "checkpoint": str(zero_path),
            "sha256": file_sha256(zero_path),
            "scoring_version": ZERO_SHOT_SCORING_VERSION,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), payload


def _loader(
    dataset: Dataset[dict[str, Any]],
    config: dict[str, Any],
) -> DataLoader:
    submission = config["submission"]
    workers = int(submission.get("num_workers", config["evaluation"]["num_workers"]))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            submission.get("batch_size", config["evaluation"]["batch_size"])
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


def _histogram_quantile(
    arrays: Iterable[np.ndarray],
    quantile: float,
    lower: float,
    upper: float,
    bins: int = 65_536,
) -> float:
    if not lower < upper:
        return lower
    histogram = np.zeros(bins, dtype=np.int64)
    count = 0
    for array in arrays:
        current, _ = np.histogram(array, bins=bins, range=(lower, upper))
        histogram += current
        count += int(array.size)
    target = int(round(quantile * max(0, count - 1)))
    index = int(np.searchsorted(np.cumsum(histogram), target + 1))
    index = min(max(index, 0), bins - 1)
    return lower + (upper - lower) * index / bins


def _calibration_bounds(
    maps: list[np.ndarray],
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    minimum = min(float(array.min()) for array in maps)
    maximum = max(float(array.max()) for array in maps)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise FloatingPointError("Anomaly maps contain non-finite values")
    if not minimum < maximum:
        return minimum, minimum + 1e-7
    lower = _histogram_quantile(maps, lower_quantile, minimum, maximum)
    upper = _histogram_quantile(maps, upper_quantile, minimum, maximum)
    if not lower < upper:
        lower, upper = minimum, maximum
    return lower, upper


def _top_ratio_score(arrays: list[np.ndarray], ratio: float) -> float:
    flattened = np.concatenate([array.reshape(-1) for array in arrays])
    count = max(1, int(flattened.size * ratio))
    top_values = np.partition(flattened, flattened.size - count)[-count:]
    return float(top_values.mean(dtype=np.float64))


def _aggregate_object_score(
    arrays: list[np.ndarray],
    ratio: float,
    *,
    mode: str = "legacy_concat_topk",
    visibility: np.ndarray | None = None,
    softmax_temperature: float = 0.25,
    visibility_max_blend: float = 0.5,
) -> float:
    """YAML-selectable five-view classification aggregation."""

    if len(arrays) != 5:
        raise ValueError(f"object score requires five maps, got {len(arrays)}")
    if mode == "legacy_concat_topk":
        return _top_ratio_score(arrays, ratio)
    per_view = np.asarray(
        [_top_ratio_score([array], ratio) for array in arrays],
        dtype=np.float64,
    )
    if mode == "max":
        return float(per_view.max())
    if mode == "softmax":
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        logits = per_view / softmax_temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        return float(np.dot(weights, per_view))
    if mode == "visibility_aware":
        if visibility is None or visibility.shape != (5,):
            raise ValueError("visibility_aware aggregation requires five weights")
        weights = np.clip(visibility.astype(np.float64), 0.0, None)
        if not float(weights.sum()) > 0:
            weights = np.full(5, 0.2, dtype=np.float64)
        else:
            weights /= weights.sum()
        if not 0.0 <= visibility_max_blend <= 1.0:
            raise ValueError("visibility_max_blend must be in [0, 1]")
        weighted = float(np.dot(weights, per_view))
        # The max component preserves defects visible in only one camera.
        return float(
            visibility_max_blend * per_view.max()
            + (1.0 - visibility_max_blend) * weighted
        )
    raise ValueError(
        "object_score_aggregation must be legacy_concat_topk, max, softmax, "
        "or visibility_aware"
    )


def _zero_shot_object_score(
    arrays: list[np.ndarray],
    ratio: float,
    *,
    max_blend: float = 0.5,
) -> float:
    """Score unseen objects from the exact float maps written as masks."""
    if not arrays:
        raise ValueError("zero-shot object scoring requires at least one map")
    if not 0.0 <= max_blend <= 1.0:
        raise ValueError("max_blend must be in [0, 1]")
    per_view = np.asarray(
        [_top_ratio_score([array], ratio) for array in arrays],
        dtype=np.float64,
    )
    return float(
        max_blend * per_view.max()
        + (1.0 - max_blend) * per_view.mean()
    )


def _probability_like_score(raw_score: float) -> float:
    # Cosine-distance anomaly maps are non-negative. This strictly monotonic
    # transform keeps category-wise ranking while satisfying the [0, 1] schema.
    return float(-math.expm1(-max(0.0, raw_score)))


def _write_mask(
    anomaly: np.ndarray,
    path: Path,
    lower: float,
    upper: float,
) -> None:
    scaled = np.clip((anomaly - lower) / (upper - lower), 0.0, 1.0)
    encoded = np.rint(scaled * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="L").save(path, format="PNG", optimize=True)


def _category_masks_are_valid(
    category_root: Path,
    views: tuple[CompetitionView, ...],
    mask_size: int,
) -> bool:
    for view in views:
        path = category_root / view.sample / f"{view.view_id}_mask.png"
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (mask_size, mask_size):
                    return False
        except OSError:
            return False
    return True


def _read_category_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid category result: {path}")
    return rows


def validate_submission_layout(
    submission_root: Path,
    manifest: CompetitionManifest,
    mask_size: int = 448,
) -> dict[str, Any]:
    csv_path = submission_root / "submission.csv"
    mask_root = submission_root / "predicted_masks"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing submission.csv: {csv_path}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Missing predicted_masks: {mask_root}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["group_folder", "anomaly_score"]:
            raise ValueError(
                "submission.csv columns must be exactly "
                "group_folder,anomaly_score"
            )
        rows = list(reader)
    expected_groups = list(manifest.group_folders)
    actual_groups = [row["group_folder"] for row in rows]
    if actual_groups != expected_groups:
        missing = sorted(set(expected_groups) - set(actual_groups))
        extra = sorted(set(actual_groups) - set(expected_groups))
        raise ValueError(
            "submission.csv groups/order do not match Test split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for row in rows:
        score = float(row["anomaly_score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid anomaly_score for {row['group_folder']}: {score}"
            )

    expected_masks = {
        f"{view.category}/{view.sample}/{view.view_id}_mask.png"
        for view in manifest.views
    }
    actual_masks = {
        path.relative_to(mask_root).as_posix()
        for path in mask_root.rglob("*.png")
    }
    if actual_masks != expected_masks:
        missing = sorted(expected_masks - actual_masks)
        extra = sorted(actual_masks - expected_masks)
        raise ValueError(
            "predicted_masks does not exactly match Test split: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for relative in sorted(expected_masks):
        with Image.open(mask_root / Path(relative)) as image:
            if image.mode != "L":
                raise ValueError(f"Mask is not single-channel L: {relative}")
            if image.size != (mask_size, mask_size):
                raise ValueError(
                    f"Mask has wrong size {image.size}, expected "
                    f"{mask_size}x{mask_size}: {relative}"
                )
    return {
        "groups": len(rows),
        "masks": len(expected_masks),
        "mask_size": mask_size,
    }


def build_submission_zip(
    submission_root: Path,
    zip_path: Path,
    manifest: CompetitionManifest,
) -> Path:
    expected_members = ["submission.csv"] + [
        "predicted_masks/"
        f"{view.category}/{view.sample}/{view.view_id}_mask.png"
        for view in manifest.views
    ]
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for member in expected_members:
            archive.write(submission_root / Path(member), arcname=member)
    with zipfile.ZipFile(temporary, mode="r") as archive:
        if archive.namelist() != expected_members:
            raise RuntimeError("ZIP member validation failed")
        corrupted = archive.testzip()
        if corrupted is not None:
            raise RuntimeError(f"ZIP CRC validation failed: {corrupted}")
    os.replace(temporary, zip_path)
    return zip_path


@torch.no_grad()
def generate_competition_submission(
    config: dict[str, Any],
    checkpoint: str = "auto",
    allow_partial: bool = False,
) -> dict[str, Any]:
    # Keep package-layout validation importable in lightweight environments.
    from .metrics import GaussianFilter

    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = resolve_competition_checkpoint(output_dir, checkpoint)
    dataset_config = config["dataset"]
    manifest = scan_competition_split(
        Path(dataset_config["test_dir"]),
        requested=dataset_config.get(
            "test_categories", dataset_config["categories"]
        ),
        limit=dataset_config.get(
            "test_category_limit", dataset_config.get("category_limit")
        ),
    )
    train_manifest = scan_competition_split(
        Path(dataset_config["train_dir"]),
        requested=dataset_config["categories"],
        limit=dataset_config.get("category_limit"),
    )
    seen_categories = set(train_manifest.categories)
    unseen_categories = set(manifest.categories) - seen_categories
    zero_shot_active = bool(
        config.get("zero_shot", {}).get("enabled", False)
    ) and bool(unseen_categories)
    clip_config = config["evaluation"].get("unseen_clip", {})
    unseen_clip_active = (
        not zero_shot_active
        and bool(clip_config.get("enabled", False))
        and bool(unseen_categories)
    )
    signature, signature_inputs = _submission_signature(
        config,
        checkpoint_path,
        manifest,
        unseen_clip_active=unseen_clip_active,
        zero_shot_active=zero_shot_active,
    )
    run_dir = output_dir / "competition_submission" / signature[:12]
    submission_root = run_dir / "package"
    mask_root = submission_root / "predicted_masks"
    category_result_dir = run_dir / "per_category"
    logger = setup_logger(
        "competition_submission", run_dir / "inference.log"
    )
    atomic_write_json(
        run_dir / "metadata.json",
        {
            "signature": signature,
            "inputs": signature_inputs,
            "created_at": utc_now(),
            "manifest": manifest.summary(),
        },
    )

    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(
        int(config["experiment"]["seed"]),
        bool(config["runtime"]["deterministic"]),
    )
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
        raise ValueError(
            "Checkpoint does not match the competition model/training config"
        )
    completed_steps = int(checkpoint_payload.get("completed_steps", -1))
    training_completed_steps = int(
        checkpoint_payload.get("training_completed_steps", completed_steps)
    )
    total_steps = int(config["training"]["total_steps"])
    if training_completed_steps != total_steps and not allow_partial:
        raise ValueError(
            "Checkpoint is partial "
            f"({training_completed_steps}/{total_steps}); "
            "pass --allow-partial only for a diagnostic package"
        )

    bundle = build_model(config, device)
    expected_backbone = checkpoint_payload.get("backbone_sha256")
    if expected_backbone and expected_backbone != bundle.backbone_sha256:
        raise ValueError("Checkpoint backbone hash does not match local weights")
    load_trainable_state_dict(bundle, checkpoint_payload["model"])
    bundle.model.eval()

    submission = config["submission"]
    evaluation = config["evaluation"]
    multi_view_config = dict(config["model"].get("multi_view", {}))
    multi_view_enabled = bool(multi_view_config.get("enabled", False))
    normal_prior = None
    if bool(evaluation.get("normal_prior", {}).get("enabled", False)):
        normal_prior = load_normal_prior(
            normal_prior_path(config),
            config,
            checkpoint_path,
        )
    clip_prior = None
    clip_segmenter = None
    zero_shot_segmenter = None
    if unseen_clip_active and bool(
        clip_config.get("normal_prior", {}).get("enabled", False)
    ):
        clip_prior = load_clip_normal_prior(
            clip_normal_prior_path(config), config
        )
    mask_size = int(submission.get("mask_size", 448))
    lower_quantile = float(submission.get("lower_quantile", 0.001))
    upper_quantile = float(submission.get("upper_quantile", 0.99999))
    object_top_ratio = float(
        submission.get(
            "object_top_ratio", config["evaluation"]["object_top_ratio"]
        )
    )
    aggregation_mode = str(
        submission.get("object_score_aggregation", "legacy_concat_topk")
    )
    aggregation_temperature = float(
        submission.get("object_score_softmax_temperature", 0.25)
    )
    visibility_max_blend = float(
        submission.get("visibility_max_blend", 0.5)
    )
    gaussian = GaussianFilter(
        kernel_size=int(config["evaluation"]["gaussian_kernel_size"]),
        sigma=float(config["evaluation"]["gaussian_sigma"]),
    ).to(device)
    gaussian.eval()

    all_rows: list[dict[str, Any]] = []
    for index, category in enumerate(manifest.categories, start=1):
        category_views = manifest.views_for_category(category)
        result_path = category_result_dir / f"{category}.json"
        category_mask_root = mask_root / category
        if result_path.is_file() and _category_masks_are_valid(
            category_mask_root, category_views, mask_size
        ):
            rows = _read_category_rows(result_path)
            logger.info(
                "[%d/%d] resume completed category %s",
                index,
                len(manifest.categories),
                category,
            )
            all_rows.extend(rows)
            continue

        category_uses_clip = unseen_clip_active and category in unseen_categories
        category_uses_zero_shot = (
            zero_shot_active and category in unseen_categories
        )
        if category_uses_zero_shot and zero_shot_segmenter is None:
            zero_shot_segmenter = load_zero_shot_segmenter(config, device)
        if category_uses_clip and clip_segmenter is None:
            clip_segmenter = FrozenClipBrokenSegmenter(clip_config, device).eval()
        logger.info(
            "[%d/%d] infer category %s (%d views, route=%s)",
            index,
            len(manifest.categories),
            category,
            len(category_views),
            "zero_shot" if category_uses_zero_shot else "dinomaly",
        )
        if multi_view_enabled:
            dataset = CompetitionObjectDataset(
                category_views,
                image_size=int(dataset_config["image_size"]),
                crop_size=int(dataset_config["crop_size"]),
                num_views=int(multi_view_config.get("num_views", 5)),
                missing_view_policy=str(
                    multi_view_config.get("missing_view_policy", "error")
                ),
            )
        else:
            dataset = CompetitionFolderDataset(
                category_views,
                image_size=int(dataset_config["image_size"]),
                crop_size=int(dataset_config["crop_size"]),
            )
        maps: list[np.ndarray] = []
        visibility_by_group: dict[str, np.ndarray] = {}
        for batch in _loader(dataset, config):
            # Use the dataset payload as the final source of truth. This keeps
            # inference compatible with resumed runs whose local config/code
            # was updated between training and Test_B packaging.
            batch_is_multi_view = "images" in batch
            if batch_is_multi_view:
                images = batch["images"].to(
                    device,
                    non_blocking=bool(config["runtime"]["pin_memory"]),
                )
                view_ids = batch["view_ids"].to(device, non_blocking=True)
                valid_view_mask = batch["valid_view_mask"].to(
                    device, non_blocking=True
                )
                if category_uses_zero_shot:
                    assert zero_shot_segmenter is not None
                    batch_size, view_count = images.shape[:2]
                    flat_images = images.reshape(
                        batch_size * view_count, *images.shape[2:]
                    )
                    flat_categories = [
                        str(value)
                        for value in batch["category"]
                        for _ in range(view_count)
                    ]
                    zero_output = zero_shot_segmenter(
                        flat_images, categories=flat_categories
                    )
                    current = zero_output["probability"]
                    current = F.interpolate(
                        current,
                        size=(mask_size, mask_size),
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0).reshape(
                        batch_size, view_count, 1, mask_size, mask_size
                    )
                    group_folders = [
                        str(value) for value in batch["group_folder"]
                    ]
                    uniform = np.full(
                        view_count, 1.0 / view_count, dtype=np.float64
                    )
                    for batch_index, group_folder in enumerate(group_folders):
                        visibility_by_group[group_folder] = uniform.copy()
                        maps.extend(
                            array
                            for array in current[batch_index, :, 0]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    continue
                category_names = [str(value) for value in batch["category"]]
                with autocast_context(dtype, device):
                    (
                        encoder_features,
                        decoder_features,
                        context_output,
                    ) = bundle.model(
                        images,
                        view_ids=view_ids,
                        valid_view_mask=valid_view_mask,
                        return_context=True,
                    )
                    current = anomaly_map(
                        encoder_features,
                        decoder_features,
                        output_size=int(dataset_config["crop_size"]) // 14,
                        layer_weights=evaluation.get(
                            "anomaly_map_layer_weights"
                        ),
                        align_corners=bool(
                            evaluation.get("anomaly_map_align_corners", True)
                        ),
                    )
                if normal_prior is not None:
                    current = normal_prior.calibrate(
                        current,
                        categories=category_names,
                        view_ids=view_ids,
                        valid_view_mask=valid_view_mask,
                        config=config,
                    )
                batch_size, view_count = current.shape[:2]
                current = F.interpolate(
                    current.float().reshape(
                        batch_size * view_count,
                        *current.shape[2:],
                    ),
                    size=(mask_size, mask_size),
                    mode="bilinear",
                    align_corners=False,
                )
                current = gaussian(current).clamp_(min=0.0).reshape(
                    batch_size,
                    view_count,
                    1,
                    mask_size,
                    mask_size,
                )
                visibility = context_output["visibility_weights"].float().cpu()
                del encoder_features, decoder_features, context_output
                if category_uses_clip:
                    assert clip_segmenter is not None
                    flat_images = images.reshape(
                        batch_size * view_count, *images.shape[2:]
                    )
                    broken_probability = clip_segmenter(flat_images)
                    broken_probability = broken_probability.reshape(
                        batch_size,
                        view_count,
                        *broken_probability.shape[1:],
                    )
                    if clip_prior is not None:
                        broken_probability = clip_prior.calibrate(
                            broken_probability,
                            view_ids=view_ids,
                            valid_view_mask=valid_view_mask,
                            config=config,
                        )
                    current = fuse_unseen_anomaly_map(
                        current.reshape(
                            batch_size * view_count, 1, mask_size, mask_size
                        ),
                        broken_probability.reshape(
                            batch_size * view_count,
                            *broken_probability.shape[2:],
                        ),
                        reconstruction_gain=float(
                            clip_config.get("reconstruction_gain", 1.0)
                        ),
                        semantic_gain=float(
                            clip_config.get("semantic_gain", 1.0)
                        ),
                        semantic_scale_floor=float(
                            clip_config.get("semantic_scale_floor", 0.02)
                        ),
                        broken_threshold=float(
                            clip_config.get("broken_threshold", 0.5)
                        ),
                        upper_quantile=float(
                            clip_config.get("upper_quantile", 0.995)
                        ),
                        foreground_low_quantile=float(
                            clip_config.get("foreground_low_quantile", 0.2)
                        ),
                        foreground_high_quantile=float(
                            clip_config.get("foreground_high_quantile", 0.7)
                        ),
                        foreground_floor=float(
                            clip_config.get("foreground_floor", 0.0)
                        ),
                        foreground_dilation_kernel=int(
                            clip_config.get("foreground_dilation_kernel", 9)
                        ),
                        confidence_power=float(
                            clip_config.get("confidence_power", 2.0)
                        ),
                    ).reshape(
                        batch_size,
                        view_count,
                        1,
                        mask_size,
                        mask_size,
                    )
                    if bool(
                        clip_config.get("final_gaussian_smoothing", True)
                    ):
                        current = gaussian(
                            current.reshape(
                                batch_size * view_count,
                                1,
                                mask_size,
                                mask_size,
                            )
                        ).clamp_(min=0.0).reshape_as(current)
                group_folders = [str(value) for value in batch["group_folder"]]
                for batch_index, group_folder in enumerate(group_folders):
                    visibility_by_group[group_folder] = (
                        visibility[batch_index].numpy().astype(np.float64)
                    )
                    maps.extend(
                        array
                        for array in current[batch_index, :, 0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
            else:
                images = batch["image"].to(
                    device,
                    non_blocking=bool(config["runtime"]["pin_memory"]),
                )
                if category_uses_zero_shot:
                    assert zero_shot_segmenter is not None
                    zero_output = zero_shot_segmenter(
                        images,
                        categories=[
                            str(value) for value in batch["category"]
                        ],
                    )
                    current = zero_output["probability"]
                    current = F.interpolate(
                        current,
                        size=(mask_size, mask_size),
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)
                    maps.extend(
                        array
                        for array in current[:, 0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    continue
                category_names = [str(value) for value in batch["category"]]
                view_ids = batch["view_id"].to(device, dtype=torch.long)
                valid_view_mask = torch.ones_like(view_ids, dtype=torch.bool)
                with autocast_context(dtype, device):
                    encoder_features, decoder_features = bundle.model(images)
                    current = anomaly_map(
                        encoder_features,
                        decoder_features,
                        output_size=int(dataset_config["crop_size"]) // 14,
                        layer_weights=evaluation.get(
                            "anomaly_map_layer_weights"
                        ),
                        align_corners=bool(
                            evaluation.get("anomaly_map_align_corners", True)
                        ),
                    )
                if normal_prior is not None:
                    current = normal_prior.calibrate(
                        current,
                        categories=category_names,
                        view_ids=view_ids,
                        valid_view_mask=valid_view_mask,
                        config=config,
                    )
                current = F.interpolate(
                    current.float(),
                    size=(mask_size, mask_size),
                    mode="bilinear",
                    align_corners=False,
                )
                current = gaussian(current).clamp_(min=0.0)
                del encoder_features, decoder_features
                if category_uses_clip:
                    assert clip_segmenter is not None
                    broken_probability = clip_segmenter(images)
                    if clip_prior is not None:
                        broken_probability = clip_prior.calibrate(
                            broken_probability,
                            view_ids=view_ids,
                            valid_view_mask=valid_view_mask,
                            config=config,
                        )
                    current = fuse_unseen_anomaly_map(
                        current,
                        broken_probability,
                        reconstruction_gain=float(
                            clip_config.get("reconstruction_gain", 1.0)
                        ),
                        semantic_gain=float(
                            clip_config.get("semantic_gain", 1.0)
                        ),
                        semantic_scale_floor=float(
                            clip_config.get("semantic_scale_floor", 0.02)
                        ),
                        broken_threshold=float(
                            clip_config.get("broken_threshold", 0.5)
                        ),
                        upper_quantile=float(
                            clip_config.get("upper_quantile", 0.995)
                        ),
                        foreground_low_quantile=float(
                            clip_config.get("foreground_low_quantile", 0.2)
                        ),
                        foreground_high_quantile=float(
                            clip_config.get("foreground_high_quantile", 0.7)
                        ),
                        foreground_floor=float(
                            clip_config.get("foreground_floor", 0.0)
                        ),
                        foreground_dilation_kernel=int(
                            clip_config.get("foreground_dilation_kernel", 9)
                        ),
                        confidence_power=float(
                            clip_config.get("confidence_power", 2.0)
                        ),
                    )
                    if bool(
                        clip_config.get("final_gaussian_smoothing", True)
                    ):
                        current = gaussian(current).clamp_(min=0.0)
                maps.extend(
                    array
                    for array in current[:, 0].cpu().numpy().astype(np.float32)
                )
        if len(maps) != len(category_views):
            raise RuntimeError(
                f"Inference count mismatch for {category}: "
                f"{len(maps)} != {len(category_views)}"
            )

        if category_uses_zero_shot:
            # Learned probabilities share one absolute scale across unseen
            # categories. Per-category stretching would turn harmless noise
            # into bright false positives.
            lower, upper = 0.0, 1.0
        else:
            lower, upper = _calibration_bounds(
                maps, lower_quantile, upper_quantile
            )
        grouped_maps: dict[str, list[np.ndarray]] = defaultdict(list)
        for view, current in zip(category_views, maps, strict=True):
            grouped_maps[view.group_folder].append(current)
            _write_mask(
                current,
                category_mask_root
                / view.sample
                / f"{view.view_id}_mask.png",
                lower,
                upper,
            )
        rows = []
        for group_folder in dict.fromkeys(
            view.group_folder for view in category_views
        ):
            if category_uses_zero_shot:
                # A global CLIP head can be confident while every local mask
                # is empty. Score the same float maps used to write PNGs so a
                # black localization result can never produce a high score.
                raw_score = _zero_shot_object_score(
                    grouped_maps[group_folder],
                    object_top_ratio,
                    max_blend=visibility_max_blend,
                )
            else:
                raw_score = _aggregate_object_score(
                    grouped_maps[group_folder],
                    object_top_ratio,
                    mode=aggregation_mode,
                    visibility=visibility_by_group.get(
                        group_folder,
                        np.full(5, 0.2, dtype=np.float64),
                    ),
                    softmax_temperature=aggregation_temperature,
                    visibility_max_blend=visibility_max_blend,
                )
            rows.append(
                {
                    "group_folder": group_folder,
                    "anomaly_score": (
                        float(np.clip(raw_score, 0.0, 1.0))
                        if category_uses_zero_shot
                        else _probability_like_score(raw_score)
                    ),
                    "raw_score": raw_score,
                }
            )
        atomic_write_json(
            result_path,
            {
                "category": category,
                "completed_at": utc_now(),
                "views": len(category_views),
                "unseen_clip": category_uses_clip,
                "route": "zero_shot" if category_uses_zero_shot else "dinomaly",
                "calibration": {
                    "lower": lower,
                    "upper": upper,
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
                },
                "object_score_aggregation": {
                    "mode": aggregation_mode,
                    "top_ratio": object_top_ratio,
                    "softmax_temperature": aggregation_temperature,
                    "visibility_max_blend": visibility_max_blend,
                },
                "rows": rows,
            },
        )
        all_rows.extend(rows)

    expected_order = list(manifest.group_folders)
    row_by_group = {row["group_folder"]: row for row in all_rows}
    if set(row_by_group) != set(expected_order):
        raise RuntimeError("Per-category inference results are incomplete")
    csv_path = submission_root / "submission.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["group_folder", "anomaly_score"])
        for group_folder in expected_order:
            writer.writerow(
                [group_folder, f"{row_by_group[group_folder]['anomaly_score']:.10f}"]
            )

    validation = validate_submission_layout(
        submission_root, manifest, mask_size=mask_size
    )
    zip_path = build_submission_zip(
        submission_root,
        run_dir / "submission.zip",
        manifest,
    )
    result = {
        "status": "partial_diagnostic" if completed_steps != total_steps else "complete",
        "completed_at": utc_now(),
        "signature": signature,
        "checkpoint": str(checkpoint_path),
        "checkpoint_steps": completed_steps,
        "submission_root": str(submission_root),
        "submission_csv": str(csv_path),
        "zip": str(zip_path),
        "validation": validation,
    }
    atomic_write_json(run_dir / "result.json", result)
    atomic_write_json(
        output_dir / "competition_submission" / "latest.json",
        {
            "signature": signature,
            "result": str(run_dir / "result.json"),
            "zip": str(zip_path),
            "updated_at": utc_now(),
        },
    )
    logger.info("Competition submission ready: %s", zip_path)
    return result
