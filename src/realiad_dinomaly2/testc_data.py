from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image

from .competition_data import scan_competition_split


EXPECTED_VIEWS = tuple(range(5))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_testc_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seen = tuple(str(value) for value in payload["seen_categories"])
    unseen = tuple(str(value) for value in payload["unseen_categories"])
    if len(seen) != 50 or len(unseen) != 50:
        raise ValueError("Test_C protocol must contain exactly 50 seen and 50 unseen categories")
    if len(set(seen)) != 50 or len(set(unseen)) != 50 or set(seen) & set(unseen):
        raise ValueError("Test_C protocol categories must be unique and disjoint")
    if int(payload.get("normal_objects_per_category", 10)) != 10:
        raise ValueError("Test_C protocol requires 10 normal objects per category")
    if int(payload.get("anomaly_objects_per_category", 10)) != 10:
        raise ValueError("Test_C protocol requires 10 anomaly objects per category")
    if int(payload.get("views_per_object", 5)) != 5:
        raise ValueError("Test_C protocol requires five views per object")
    payload["seen_categories"] = list(seen)
    payload["unseen_categories"] = list(unseen)
    payload["categories"] = sorted([*seen, *unseen])
    payload["protocol_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "protocol_sha256"}
    )
    return payload


def discover_data_root(project_root: Path) -> Path:
    candidates = (
        project_root / "data",
        project_root.parent / "UAD" / "data",
    )
    for candidate in candidates:
        if (candidate / "Real-IAD_Variety").is_dir() and (
            candidate / "competition" / "Train"
        ).is_dir():
            return candidate.resolve()
    return candidates[-1].resolve()


def _category_root(image_root: Path, category: str) -> Path:
    flat = image_root / category
    nested = flat / category
    if nested.is_dir():
        return nested
    if flat.is_dir():
        return flat
    raise FileNotFoundError(f"Missing source image directory for {category}: {flat}")


def _record_file(category_root: Path, relative: str) -> Path:
    return category_root.joinpath(*PurePosixPath(relative.replace("\\", "/")).parts)


def _view_id(image_path: str) -> int:
    name = PurePosixPath(image_path.replace("\\", "/")).name
    for view in range(1, 6):
        if f"_C{view}_" in name or f"_C0{view}_" in name:
            return view - 1
    raise ValueError(f"Cannot identify C1..C5 view from {image_path!r}")


@dataclass(frozen=True)
class SourceObject:
    category: str
    object_id: str
    anomaly_class: str
    records: tuple[dict[str, Any], ...]

    @property
    def object_label(self) -> int:
        return int(self.anomaly_class.upper() != "OK")


def _group_test_objects(json_path: Path, category: str) -> list[SourceObject]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = payload.get("test")
    if not isinstance(records, list):
        raise ValueError(f"{json_path} does not contain a test list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(records):
        if str(raw.get("category")) != category:
            raise ValueError(f"{json_path}:test[{index}] has a mismatched category")
        image_path = str(raw["image_path"]).replace("\\", "/")
        object_id = str(PurePosixPath(image_path).parent)
        item = dict(raw)
        item["image_path"] = image_path
        if item.get("mask_path") is not None:
            item["mask_path"] = str(item["mask_path"]).replace("\\", "/")
        item["view_id"] = _view_id(image_path)
        grouped[object_id].append(item)

    objects: list[SourceObject] = []
    for object_id, items in sorted(grouped.items()):
        anomaly_classes = {str(item["anomaly_class"]) for item in items}
        if len(anomaly_classes) != 1:
            raise ValueError(f"{category}/{object_id} mixes anomaly classes")
        by_view = {int(item["view_id"]): item for item in items}
        if len(by_view) != len(items) or tuple(sorted(by_view)) != EXPECTED_VIEWS:
            raise ValueError(f"{category}/{object_id} must contain exactly C1..C5")
        objects.append(
            SourceObject(
                category=category,
                object_id=object_id,
                anomaly_class=next(iter(anomaly_classes)),
                records=tuple(by_view[index] for index in EXPECTED_VIEWS),
            )
        )
    return objects


def _rng(seed: int, category: str, purpose: str) -> random.Random:
    material = f"{seed}:{category}:{purpose}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def select_category_objects(
    objects: Sequence[SourceObject],
    *,
    category: str,
    seed: int,
    normal_count: int = 10,
    anomaly_count: int = 10,
) -> list[SourceObject]:
    normal = [item for item in objects if item.object_label == 0]
    anomalous: dict[str, list[SourceObject]] = defaultdict(list)
    for item in objects:
        if item.object_label:
            anomalous[item.anomaly_class].append(item)
    if len(normal) < normal_count:
        raise ValueError(f"{category} has only {len(normal)} normal test objects")
    if sum(len(items) for items in anomalous.values()) < anomaly_count:
        raise ValueError(f"{category} does not have {anomaly_count} anomalous test objects")

    normal_rng = _rng(seed, category, "normal")
    normal_rng.shuffle(normal)
    selected = normal[:normal_count]

    defect_names = sorted(anomalous)
    defect_rng = _rng(seed, category, "defects")
    defect_rng.shuffle(defect_names)
    for defect_name, items in anomalous.items():
        _rng(seed, category, f"defect:{defect_name}").shuffle(items)
    offsets = {name: 0 for name in defect_names}
    while len(selected) < normal_count + anomaly_count:
        made_progress = False
        for defect_name in defect_names:
            offset = offsets[defect_name]
            items = anomalous[defect_name]
            if offset < len(items):
                selected.append(items[offset])
                offsets[defect_name] += 1
                made_progress = True
                if len(selected) == normal_count + anomaly_count:
                    break
        if not made_progress:
            raise RuntimeError(f"Unable to complete stratified anomaly sample for {category}")

    _rng(seed, category, "output-order").shuffle(selected)
    return selected


def _same_file(source: Path, destination: Path) -> bool:
    return destination.is_file() and destination.stat().st_size == source.stat().st_size


def _copy_resumable(source: Path, destination: Path) -> bool:
    if _same_file(source, destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return True


def prepare_testc(
    *,
    source_root: Path,
    competition_train: Path,
    output_root: Path,
    protocol_path: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    competition_train = competition_train.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    protocol = load_testc_protocol(protocol_path.expanduser().resolve())
    seed = int(protocol["sampling_seed"] if seed is None else seed)
    json_root = source_root / "realiadvariety_jsons"
    image_root = source_root / "realiadvariety_1024"
    if not json_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(f"Invalid Real-IAD Variety root: {source_root}")

    train_audit = audit_competition_train(
        competition_train, source_root, protocol_path
    )
    train_manifest = scan_competition_split(competition_train)
    seen = tuple(protocol["seen_categories"])
    unseen = tuple(protocol["unseen_categories"])
    if train_manifest.categories != tuple(sorted(seen)):
        raise ValueError("Competition Train categories do not exactly match Test_C seen categories")
    train_counts = {
        category: len(train_manifest.views_for_category(category)) // 5
        for category in train_manifest.categories
    }
    bad_train = {key: value for key, value in train_counts.items() if value != 20}
    if bad_train:
        raise ValueError(f"Competition Train must contain 20 objects per category: {bad_train}")

    signature_inputs = {
        "protocol_sha256": protocol["protocol_sha256"],
        "seed": seed,
        "source_root": str(source_root),
        "competition_train": str(competition_train),
        "competition_train_audit": train_audit,
        "normal_objects_per_category": 10,
        "anomaly_objects_per_category": 10,
    }
    signature = canonical_digest(signature_inputs)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("preparation_signature") != signature:
            raise ValueError(
                f"Existing Test_C has a different signature: {manifest_path}. "
                "Choose another output directory instead of overwriting it."
            )
        return audit_testc(output_root, protocol_path)

    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "preparation_progress.jsonl"
    samples: list[dict[str, Any]] = []
    copied_files = 0
    reused_files = 0
    for category in protocol["categories"]:
        category_root = _category_root(image_root, category)
        objects = _group_test_objects(json_root / f"{category}.json", category)
        selected = select_category_objects(
            objects,
            category=category,
            seed=seed,
        )
        partition = "seen" if category in seen else "unseen"
        for sample_index, source_object in enumerate(selected, start=1):
            sample_id = f"S{sample_index:04d}"
            views: list[dict[str, Any]] = []
            for view_id, record in enumerate(source_object.records):
                source_image = _record_file(category_root, record["image_path"])
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                relative_image = Path("images") / category / sample_id / f"{view_id}.png"
                if _copy_resumable(source_image, output_root / relative_image):
                    copied_files += 1
                else:
                    reused_files += 1
                relative_mask: Path | None = None
                if record.get("mask_path") is not None:
                    source_mask = _record_file(category_root, str(record["mask_path"]))
                    if not source_mask.is_file():
                        raise FileNotFoundError(source_mask)
                    relative_mask = (
                        Path("masks") / category / sample_id / f"{view_id}_mask.png"
                    )
                    if _copy_resumable(source_mask, output_root / relative_mask):
                        copied_files += 1
                    else:
                        reused_files += 1
                visible_defect = bool(source_object.object_label and relative_mask is not None)
                views.append(
                    {
                        "view_id": view_id,
                        "image_path": relative_image.as_posix(),
                        "mask_path": None if relative_mask is None else relative_mask.as_posix(),
                        "view_label": int(visible_defect),
                        "pixel_valid": True,
                        "source_image_path": str(record["image_path"]),
                        "source_mask_path": record.get("mask_path"),
                    }
                )
            sample = {
                "category": category,
                "partition": partition,
                "sample_id": sample_id,
                "group_folder": f"{category}/{sample_id}",
                "object_label": source_object.object_label,
                "anomaly_class": source_object.anomaly_class,
                "source_object_id": source_object.object_id,
                "views": views,
            }
            samples.append(sample)
            append_jsonl(
                progress_path,
                {
                    "timestamp": utc_now(),
                    "event": "sample_ready",
                    "category": category,
                    "sample_id": sample_id,
                    "object_label": source_object.object_label,
                },
            )

    manifest = {
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "preparation_signature": signature,
        "prepared_at": utc_now(),
        "seed": seed,
        "source_root": str(source_root),
        "competition_train": str(competition_train),
        "competition_train_audit": train_audit,
        "seen_categories": list(seen),
        "unseen_categories": list(unseen),
        "samples": samples,
    }
    atomic_write_json(manifest_path, manifest)
    audit = audit_testc(output_root, protocol_path)
    audit["copy"] = {"copied_files": copied_files, "reused_files": reused_files}
    atomic_write_json(output_root / "audit.json", audit)
    return audit


def audit_testc(output_root: Path, protocol_path: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    protocol = load_testc_protocol(protocol_path.expanduser().resolve())
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Test_C manifest protocol hash does not match the committed protocol")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Test_C manifest samples must be a list")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    invalid_views: list[str] = []
    image_count = 0
    mask_count = 0
    for sample in samples:
        category = str(sample["category"])
        by_category[category].append(sample)
        views = sample.get("views", [])
        if [int(view["view_id"]) for view in views] != list(EXPECTED_VIEWS):
            invalid_views.append(str(sample.get("group_folder")))
        for view in views:
            image_path = output_root / Path(str(view["image_path"]))
            if not image_path.is_file():
                missing.append(str(image_path))
            image_count += 1
            raw_mask = view.get("mask_path")
            if raw_mask is not None:
                mask_path = output_root / Path(str(raw_mask))
                if not mask_path.is_file():
                    missing.append(str(mask_path))
                mask_count += 1
    expected_categories = set(protocol["categories"])
    if set(by_category) != expected_categories:
        raise ValueError("Test_C manifest categories do not match the protocol")
    bad_counts: dict[str, Any] = {}
    for category, category_samples in by_category.items():
        normal = sum(int(item["object_label"]) == 0 for item in category_samples)
        anomaly = sum(int(item["object_label"]) == 1 for item in category_samples)
        if len(category_samples) != 20 or normal != 10 or anomaly != 10:
            bad_counts[category] = {
                "objects": len(category_samples),
                "normal": normal,
                "anomaly": anomaly,
            }
    if bad_counts or missing or invalid_views:
        raise ValueError(
            "Invalid Test_C: "
            f"bad_counts={bad_counts}, missing={missing[:10]}, invalid_views={invalid_views[:10]}"
        )
    scan = scan_competition_split(
        output_root / "images",
        requested=protocol["categories"],
    )
    if len(scan.group_folders) != 2000 or len(scan.views) != 10000:
        raise ValueError("Competition-style Test_C image scan has unexpected counts")
    return {
        "status": "valid",
        "audited_at": utc_now(),
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": protocol["protocol_sha256"],
        "categories": len(by_category),
        "seen_categories": len(protocol["seen_categories"]),
        "unseen_categories": len(protocol["unseen_categories"]),
        "objects": len(samples),
        "normal_objects": sum(int(item["object_label"]) == 0 for item in samples),
        "anomaly_objects": sum(int(item["object_label"]) == 1 for item in samples),
        "images": image_count,
        "masks": mask_count,
        "views_per_object": 5,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def audit_competition_train(
    competition_train: Path,
    source_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    competition_train = competition_train.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    protocol = load_testc_protocol(protocol_path.expanduser().resolve())
    manifest = scan_competition_split(competition_train)
    expected = tuple(sorted(protocol["seen_categories"]))
    if manifest.categories != expected:
        raise ValueError("Training categories must exactly equal the 50 Test_C seen categories")
    json_root = source_root / "realiadvariety_jsons"
    bad_counts: dict[str, int] = {}
    non_train_objects: dict[str, list[str]] = {}
    for category in manifest.categories:
        count = len(manifest.views_for_category(category)) // 5
        if count != 20:
            bad_counts[category] = count
        payload = json.loads((json_root / f"{category}.json").read_text(encoding="utf-8"))
        allowed = {
            PurePosixPath(str(item["image_path"]).replace("\\", "/")).parent.name
            for item in payload["train"]
            if str(item["anomaly_class"]).upper() == "OK"
        }
        train_samples = {
            view.sample for view in manifest.views_for_category(category)
        }
        invalid = sorted(train_samples - allowed)
        if invalid:
            non_train_objects[category] = invalid
        if any(str(item["anomaly_class"]).upper() != "OK" for item in payload["train"]):
            raise ValueError(f"Official JSON train split contains anomalies for {category}")
    if bad_counts or non_train_objects:
        raise ValueError(
            f"Competition Train leakage/count audit failed: counts={bad_counts}, "
            f"objects_not_in_json_train={non_train_objects}"
        )
    return {
        "status": "valid",
        "categories": 50,
        "objects": 1000,
        "views": 5000,
        "all_good": True,
        "json_phase": "train",
    }
