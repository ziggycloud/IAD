from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm

from .data import (
    category_image_root,
    discover_categories,
    load_records,
    record_file,
)
from .runtime import append_jsonl, atomic_write_json, utc_now


EXPECTED_OFFICIAL_TRAIN_IMAGES = 19_955
STATE_FILENAME = "_cache_state.json"
PROGRESS_FILENAME = "_cache_progress.jsonl"


@dataclass(frozen=True)
class CacheItem:
    category: str
    manifest_path: str
    source: Path
    destination: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def collect_train_cache_items(
    json_dir: Path,
    image_dir: Path,
    output_dir: Path,
    categories: Iterable[str],
) -> list[CacheItem]:
    """Resolve the official train manifest to unique source/destination pairs."""
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    items: list[CacheItem] = []
    seen_sources: set[Path] = set()
    for category in categories:
        category_root = category_image_root(image_dir, category)
        for record in load_records(json_dir, category, "train"):
            if record.is_anomaly:
                raise ValueError(
                    f"Train manifest contains an anomalous record: "
                    f"{category}/{record.image_path}"
                )
            source = record_file(category_root, record.image_path).resolve()
            if not _is_relative_to(source, image_dir):
                raise ValueError(
                    f"Manifest path escapes dataset.image_dir: {record.image_path!r}"
                )
            if source in seen_sources:
                raise ValueError(f"Duplicate train image in manifest: {source}")
            seen_sources.add(source)
            relative = source.relative_to(image_dir)
            destination = output_dir / relative
            items.append(
                CacheItem(
                    category=category,
                    manifest_path=record.image_path,
                    source=source,
                    destination=destination,
                )
            )
    return sorted(items, key=lambda item: item.source.as_posix())


def cache_image_is_valid(path: Path, target_size: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            if image.size != (target_size, target_size):
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _format_for_destination(destination: Path, source_format: str | None) -> str:
    output_format = Image.registered_extensions().get(destination.suffix.lower())
    if output_format is not None:
        return output_format
    if source_format is not None:
        return source_format
    return "PNG"


def _write_resized_atomically(
    source: Path,
    destination: Path,
    target_size: int,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with Image.open(source) as image:
            # Match upstream Real-IAD Variety preparation: force a square and
            # use bicubic interpolation.  Mode is intentionally preserved; the
            # training loader performs its normal RGB conversion afterwards.
            resized = image.resize(
                (target_size, target_size),
                resample=Image.Resampling.BICUBIC,
            )
            output_format = _format_for_destination(destination, image.format)
            resized.save(temporary, format=output_format)
            resized.close()
        # Windows requires a writable handle for FlushFileBuffers/os.fsync.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _process_item(item: CacheItem, target_size: int) -> dict[str, Any]:
    try:
        if cache_image_is_valid(item.destination, target_size):
            return {
                "status": "skipped",
                "source": str(item.source),
                "destination": str(item.destination),
            }
        _write_resized_atomically(item.source, item.destination, target_size)
        if not cache_image_is_valid(item.destination, target_size):
            raise OSError("atomic output failed post-write validation")
        return {
            "status": "written",
            "source": str(item.source),
            "destination": str(item.destination),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": str(item.source),
            "destination": str(item.destination),
            "error": repr(exc),
        }


def _manifest_fingerprint(items: Iterable[CacheItem], target_size: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"square-bicubic:{target_size}\n".encode("ascii"))
    for item in items:
        digest.update(item.category.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.manifest_path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_train_cache(
    config: dict[str, Any],
    max_images: int | None = None,
) -> dict[str, Any]:
    dataset_config = config["dataset"]
    cache_config = config["cache"]
    json_dir = Path(dataset_config["json_dir"])
    image_dir = Path(dataset_config["image_dir"]).resolve()
    output_dir = Path(cache_config["output_dir"]).resolve()
    target_size = int(cache_config.get("max_side", 1024))
    workers = int(cache_config.get("workers", 4))
    if target_size <= 0:
        raise ValueError("cache.max_side must be a positive integer")
    if workers <= 0:
        raise ValueError("cache.workers must be a positive integer")
    if output_dir == image_dir or _is_relative_to(output_dir, image_dir):
        raise ValueError(
            "cache.output_dir must be outside dataset.image_dir to avoid "
            f"polluting the raw dataset: {output_dir}"
        )

    categories = discover_categories(
        json_dir,
        requested=dataset_config["categories"],
        limit=dataset_config.get("category_limit"),
    )
    all_items = collect_train_cache_items(
        json_dir=json_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        categories=categories,
    )
    official_selection = (
        dataset_config["categories"] == "all"
        and dataset_config.get("category_limit") is None
    )
    if official_selection and len(all_items) != EXPECTED_OFFICIAL_TRAIN_IMAGES:
        raise ValueError(
            f"Official train manifest should contain "
            f"{EXPECTED_OFFICIAL_TRAIN_IMAGES} images, found {len(all_items)}"
        )
    selected_items = all_items
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive when supplied")
        selected_items = all_items[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    progress_path = output_dir / PROGRESS_FILENAME
    started_at = utc_now()
    state: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "json_dir": str(json_dir),
        "source_image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "target_size": [target_size, target_size],
        "workers": workers,
        "categories": len(categories),
        "official_manifest_images": len(all_items),
        "selected_images": len(selected_items),
        "limited_run": max_images is not None,
        "manifest_fingerprint": _manifest_fingerprint(all_items, target_size),
        "processed": 0,
        "written": 0,
        "skipped_valid": 0,
        "failed": 0,
        "errors": [],
        "last_path": None,
        "next_action": (
            "Wait for completion. Re-running the same command safely resumes "
            "by validating and skipping completed images."
        ),
    }
    atomic_write_json(state_path, state)
    append_jsonl(
        progress_path,
        {
            "timestamp": started_at,
            "event": "cache_start",
            "selected_images": len(selected_items),
            "manifest_fingerprint": state["manifest_fingerprint"],
        },
    )

    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any]], CacheItem] = {}
    last_state_write = time.monotonic()
    try:
        futures = {
            executor.submit(_process_item, item, target_size): item
            for item in selected_items
        }
        with tqdm(
            total=len(selected_items),
            unit="image",
            desc="Preparing train cache",
        ) as progress:
            for future in as_completed(futures):
                result = future.result()
                state["processed"] += 1
                if result["status"] == "written":
                    state["written"] += 1
                elif result["status"] == "skipped":
                    state["skipped_valid"] += 1
                else:
                    state["failed"] += 1
                    state["errors"].append(
                        {
                            "source": result["source"],
                            "destination": result["destination"],
                            "error": result["error"],
                        }
                    )
                state["last_path"] = result["source"]
                progress.update(1)
                now = time.monotonic()
                if (
                    state["processed"] % 25 == 0
                    or state["processed"] == len(selected_items)
                    or now - last_state_write >= 5.0
                ):
                    state["updated_at"] = utc_now()
                    atomic_write_json(state_path, state)
                    append_jsonl(
                        progress_path,
                        {
                            "timestamp": state["updated_at"],
                            "event": "cache_progress",
                            "processed": state["processed"],
                            "selected_images": state["selected_images"],
                            "written": state["written"],
                            "skipped_valid": state["skipped_valid"],
                            "failed": state["failed"],
                            "last_path": state["last_path"],
                        },
                    )
                    last_state_write = now
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        state.update(
            {
                "status": "interrupted",
                "updated_at": utc_now(),
                "next_action": (
                    "Run prepare_cache.ps1 again; valid atomic outputs are "
                    "detected and skipped."
                ),
            }
        )
        atomic_write_json(state_path, state)
        append_jsonl(
            progress_path,
            {
                "timestamp": state["updated_at"],
                "event": "cache_interrupted",
                "processed": state["processed"],
            },
        )
        raise
    except BaseException as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        state.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "last_error": repr(exc),
                "next_action": (
                    "Inspect _cache_state.json, fix the error, then run "
                    "prepare_cache.ps1 again."
                ),
            }
        )
        atomic_write_json(state_path, state)
        raise
    else:
        executor.shutdown(wait=True)

    state.update(
        {
            "status": "completed" if state["failed"] == 0 else "failed",
            "updated_at": utc_now(),
            "next_action": (
                "Train with configs/rtx3060ti_strict_upstream_cached.yaml "
                "(or configs/rtx3060ti_cached.yaml for the BF16/paper-LR preset)."
                if state["failed"] == 0 and max_images is None
                else (
                    "This was a limited cache run; run without --max-images "
                    "before using the full cached training preset."
                    if state["failed"] == 0
                    else (
                        "Inspect errors, fix the affected files, and rerun; "
                        "valid cache files will be skipped."
                    )
                )
            ),
        }
    )
    atomic_write_json(state_path, state)
    append_jsonl(
        progress_path,
        {
            "timestamp": state["updated_at"],
            "event": "cache_complete",
            "status": state["status"],
            "processed": state["processed"],
            "written": state["written"],
            "skipped_valid": state["skipped_valid"],
            "failed": state["failed"],
        },
    )
    if state["failed"]:
        raise RuntimeError(
            f"Cache preparation failed for {state['failed']} image(s); "
            f"details: {state_path}"
        )
    return state
