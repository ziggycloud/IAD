from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .data import (
    category_image_root,
    discover_categories,
    load_records,
    record_file,
)
from .runtime import atomic_write_json, utc_now


EXPECTED_FULL = {
    "categories": 160,
    "train_images": 19_955,
    "test_images": 178_995,
    "train_objects": 3_991,
    "test_objects": 35_799,
}


def _selected_for_io(records, mode: str):
    if mode == "metadata":
        return []
    if mode == "full":
        return records
    normal = [record for record in records if not record.is_anomaly]
    anomaly = [record for record in records if record.is_anomaly]
    selected = []
    for group in (normal, anomaly):
        if group:
            selected.append(group[0])
            if len(group) > 1:
                selected.append(group[-1])
    return selected


def validate_dataset(
    config: dict[str, Any],
    mode: str,
    output_path: Path,
) -> dict[str, Any]:
    if mode not in {"metadata", "sample", "full"}:
        raise ValueError("mode 必须为 metadata/sample/full")
    json_dir = Path(config["dataset"]["json_dir"])
    image_dir = Path(config["dataset"]["image_dir"])
    train_image_dir = Path(
        config["dataset"].get("train_image_dir") or image_dir
    )
    categories = discover_categories(
        json_dir,
        requested=config["dataset"]["categories"],
        limit=config["dataset"].get("category_limit"),
    )

    totals = Counter()
    defect_types = Counter()
    all_train_paths: set[tuple[str, str]] = set()
    all_test_paths: set[tuple[str, str]] = set()
    problems: list[str] = []
    warnings: list[str] = []
    per_category: list[dict[str, Any]] = []

    for category in categories:
        category_root = category_image_root(image_dir, category)
        train_category_root = category_image_root(train_image_dir, category)
        train_records = load_records(json_dir, category, "train")
        test_records = load_records(json_dir, category, "test")
        train_groups: dict[str, list[int]] = defaultdict(list)
        test_groups: dict[str, list[int]] = defaultdict(list)

        for record in train_records:
            totals["train_images"] += 1
            train_groups[record.object_id].append(record.view_id)
            key = (category, record.image_path)
            if key in all_train_paths:
                problems.append(f"重复 train path: {key}")
            all_train_paths.add(key)
            if record.is_anomaly:
                totals["train_anomalies"] += 1

        for record in test_records:
            totals["test_images"] += 1
            test_groups[record.object_id].append(record.view_id)
            key = (category, record.image_path)
            if key in all_test_paths:
                problems.append(f"重复 test path: {key}")
            all_test_paths.add(key)
            if record.is_anomaly:
                totals["test_object_anomaly_views"] += 1
                defect_types[record.anomaly_class] += 1
                if record.mask_path is None:
                    totals["test_anomaly_views_without_mask"] += 1
                else:
                    totals["test_anomaly_views_with_mask"] += 1
            else:
                totals["test_ok_views"] += 1

        for phase, groups in (("train", train_groups), ("test", test_groups)):
            for object_id, views in groups.items():
                if sorted(views) != [1, 2, 3, 4, 5]:
                    problems.append(
                        f"{category}/{phase}/{object_id} views={sorted(views)}"
                    )

        io_records = [
            (record, train_category_root)
            for record in _selected_for_io(train_records, mode)
        ]
        io_records.extend(
            (record, category_root)
            for record in _selected_for_io(test_records, mode)
        )
        for record, record_root in io_records:
            image_path = record_file(record_root, record.image_path)
            if not image_path.is_file():
                problems.append(f"缺图：{image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_size = image.size
            except Exception as exc:
                problems.append(f"图像解码失败：{image_path}: {exc!r}")
                continue
            if record.mask_path is not None:
                mask_path = record_file(record_root, record.mask_path)
                if not mask_path.is_file():
                    problems.append(f"缺 mask：{mask_path}")
                    continue
                try:
                    with Image.open(mask_path) as mask:
                        mask.load()
                        if mask.size != image_size:
                            warnings.append(
                                "图/mask 原始尺寸不同（评估时会分别缩放到相同尺寸）："
                                f"{image_path} {image_size} vs {mask_path} {mask.size}"
                            )
                except Exception as exc:
                    problems.append(f"mask 解码失败：{mask_path}: {exc!r}")

        per_category.append(
            {
                "category": category,
                "train_images": len(train_records),
                "test_images": len(test_records),
                "train_objects": len(train_groups),
                "test_objects": len(test_groups),
            }
        )
        totals["train_objects"] += len(train_groups)
        totals["test_objects"] += len(test_groups)

    overlap = all_train_paths.intersection(all_test_paths)
    if overlap:
        problems.append(f"train/test 路径交叉：{len(overlap)}")
    if totals["train_anomalies"]:
        problems.append(f"train 含异常视图：{totals['train_anomalies']}")

    full_official = (
        config["dataset"]["categories"] == "all"
        and config["dataset"].get("category_limit") is None
    )
    expected_checks: dict[str, bool] = {}
    if full_official:
        actual = {
            "categories": len(categories),
            "train_images": totals["train_images"],
            "test_images": totals["test_images"],
            "train_objects": totals["train_objects"],
            "test_objects": totals["test_objects"],
        }
        for key, expected in EXPECTED_FULL.items():
            expected_checks[key] = actual[key] == expected
            if actual[key] != expected:
                problems.append(
                    f"{key}={actual[key]}，官方预期 {expected}"
                )

    result = {
        "status": "ok" if not problems else "failed",
        "mode": mode,
        "checked_at": utc_now(),
        "json_dir": str(json_dir),
        "image_dir": str(image_dir),
        "train_image_dir": str(train_image_dir),
        "categories": len(categories),
        "totals": dict(totals),
        "defect_types": dict(sorted(defect_types.items())),
        "expected_full_checks": expected_checks,
        "problems": problems,
        "warnings": warnings,
        "per_category": per_category,
    }
    atomic_write_json(output_path, result)
    if problems:
        preview = "\n".join(problems[:20])
        raise RuntimeError(
            f"数据验证失败，共 {len(problems)} 个问题。前 20 个：\n{preview}"
        )
    return result
