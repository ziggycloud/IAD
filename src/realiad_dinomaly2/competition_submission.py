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
from PIL import Image
from torch.utils.data import DataLoader

from .competition_data import (
    CompetitionFolderDataset,
    CompetitionManifest,
    CompetitionView,
    scan_competition_split,
)
from .config import config_fingerprint
from .losses import anomaly_map
from .modeling import build_model, load_trainable_state_dict
from .runtime import (
    amp_dtype,
    atomic_write_json,
    autocast_context,
    resolve_device,
    setup_logger,
    setup_seed,
    utc_now,
)


def resolve_competition_checkpoint(
    output_dir: Path,
    checkpoint: str,
) -> Path:
    path = (
        output_dir / "checkpoints" / "final_model.pt"
        if checkpoint == "auto"
        else Path(checkpoint).expanduser().resolve()
    )
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
) -> tuple[str, dict[str, Any]]:
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "config_fingerprint": config_fingerprint(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "test_root": str(manifest.root),
        "test_manifest_sha256": _manifest_digest(manifest),
        "submission": config["submission"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), payload


def _loader(
    dataset: CompetitionFolderDataset,
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
        requested=dataset_config["categories"],
        limit=dataset_config.get("category_limit"),
    )
    signature, signature_inputs = _submission_signature(
        config, checkpoint_path, manifest
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
    total_steps = int(config["training"]["total_steps"])
    if completed_steps != total_steps and not allow_partial:
        raise ValueError(
            f"Checkpoint is partial ({completed_steps}/{total_steps}); "
            "pass --allow-partial only for a diagnostic package"
        )

    bundle = build_model(config, device)
    expected_backbone = checkpoint_payload.get("backbone_sha256")
    if expected_backbone and expected_backbone != bundle.backbone_sha256:
        raise ValueError("Checkpoint backbone hash does not match local weights")
    load_trainable_state_dict(bundle, checkpoint_payload["model"])
    bundle.model.eval()

    submission = config["submission"]
    mask_size = int(submission.get("mask_size", 448))
    lower_quantile = float(submission.get("lower_quantile", 0.001))
    upper_quantile = float(submission.get("upper_quantile", 0.99999))
    object_top_ratio = float(
        submission.get(
            "object_top_ratio", config["evaluation"]["object_top_ratio"]
        )
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

        logger.info(
            "[%d/%d] infer category %s (%d views)",
            index,
            len(manifest.categories),
            category,
            len(category_views),
        )
        dataset = CompetitionFolderDataset(
            category_views,
            image_size=int(dataset_config["image_size"]),
            crop_size=int(dataset_config["crop_size"]),
        )
        maps: list[np.ndarray] = []
        for batch in _loader(dataset, config):
            images = batch["image"].to(
                device,
                non_blocking=bool(config["runtime"]["pin_memory"]),
            )
            with autocast_context(dtype, device):
                encoder_features, decoder_features = bundle.model(images)
                current = anomaly_map(
                    encoder_features,
                    decoder_features,
                    output_size=mask_size,
                    layer_weights=evaluation.get(
                        "anomaly_map_layer_weights"
                    ),
                    align_corners=bool(
                        evaluation.get("anomaly_map_align_corners", True)
                    ),
                )
            current = gaussian(current.to(dtype=torch.float32))
            current = current.clamp_(min=0.0)
            maps.extend(
                array
                for array in current[:, 0].cpu().numpy().astype(np.float32)
            )
        if len(maps) != len(category_views):
            raise RuntimeError(
                f"Inference count mismatch for {category}: "
                f"{len(maps)} != {len(category_views)}"
            )

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
            raw_score = _top_ratio_score(
                grouped_maps[group_folder], object_top_ratio
            )
            rows.append(
                {
                    "group_folder": group_folder,
                    "anomaly_score": _probability_like_score(raw_score),
                    "raw_score": raw_score,
                }
            )
        atomic_write_json(
            result_path,
            {
                "category": category,
                "completed_at": utc_now(),
                "views": len(category_views),
                "calibration": {
                    "lower": lower,
                    "upper": upper,
                    "lower_quantile": lower_quantile,
                    "upper_quantile": upper_quantile,
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
