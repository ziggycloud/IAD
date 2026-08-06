from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


VIEW_PATTERN = re.compile(r"_C0?([1-5])(?:_|\.|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    category: str
    anomaly_class: str
    image_path: str
    mask_path: str | None

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly_class.upper() != "OK"

    @property
    def object_id(self) -> str:
        return str(PurePosixPath(self.image_path).parent)

    @property
    def view_id(self) -> int:
        match = VIEW_PATTERN.search(PurePosixPath(self.image_path).name)
        return int(match.group(1)) if match else -1


def discover_categories(
    json_dir: Path,
    requested: str | Sequence[str] = "all",
    limit: int | None = None,
) -> list[str]:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON 目录不存在：{json_dir}")
    available = sorted(path.stem for path in json_dir.glob("*.json"))
    if not available:
        raise FileNotFoundError(f"JSON 目录中没有 .json：{json_dir}")

    if requested == "all":
        selected = available
    else:
        selected = list(requested)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"配置包含未知类别：{unknown}")
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("类别列表为空")
    return selected


def load_records(json_dir: Path, category: str, phase: str) -> list[Record]:
    if phase not in {"train", "test"}:
        raise ValueError(f"phase 必须是 train/test，收到 {phase!r}")
    json_path = json_dir / f"{category}.json"
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if phase not in payload or not isinstance(payload[phase], list):
        raise ValueError(f"{json_path} 缺少 list 字段 {phase!r}")

    records: list[Record] = []
    for index, item in enumerate(payload[phase]):
        try:
            record = Record(
                category=str(item["category"]),
                anomaly_class=str(item["anomaly_class"]),
                image_path=str(item["image_path"]).replace("\\", "/"),
                mask_path=(
                    None
                    if item.get("mask_path") is None
                    else str(item["mask_path"]).replace("\\", "/")
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{json_path}:{phase}[{index}] 字段不完整") from exc
        if record.category != category:
            raise ValueError(
                f"{json_path}:{phase}[{index}] category={record.category!r}，"
                f"预期 {category!r}"
            )
        records.append(record)
    return records


def category_image_root(image_dir: Path, category: str) -> Path:
    nested = image_dir / category / category
    flat = image_dir / category
    if nested.is_dir():
        return nested
    if flat.is_dir():
        return flat
    raise FileNotFoundError(
        f"类别图像目录不存在；已检查 {nested} 和 {flat}"
    )


def record_file(category_root: Path, relative_path: str) -> Path:
    return category_root.joinpath(*PurePosixPath(relative_path).parts)


def image_transform(image_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def mask_transform(
    image_size: int,
    crop_size: int,
    resize_semantics: str,
) -> transforms.Compose:
    if resize_semantics == "upstream_bilinear_nonzero":
        interpolation = InterpolationMode.BILINEAR
        antialias = True
    elif resize_semantics == "nearest_binary":
        interpolation = InterpolationMode.NEAREST
        antialias = None
    else:
        raise ValueError(
            "mask_resize_semantics 必须为 upstream_bilinear_nonzero "
            "或 nearest_binary"
        )
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=interpolation,
                antialias=antialias,
            ),
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
        ]
    )


def _stratified_limit(records: list[Record], max_items: int | None) -> list[Record]:
    if max_items is None or len(records) <= max_items:
        return records
    max_items = int(max_items)
    if max_items < 2:
        return records[:max_items]
    normal = [record for record in records if not record.is_anomaly]
    anomaly = [record for record in records if record.is_anomaly]
    normal_count = min(len(normal), max_items // 2)
    anomaly_count = min(len(anomaly), max_items - normal_count)
    if normal_count + anomaly_count < max_items:
        normal_count = min(len(normal), max_items - anomaly_count)
    selected = normal[:normal_count] + anomaly[:anomaly_count]
    return sorted(selected, key=lambda item: item.image_path)


class RealIADVarietyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        json_dir: Path,
        image_dir: Path,
        category: str,
        phase: str,
        image_size: int,
        crop_size: int,
        max_items: int | None = None,
        image_label_policy: str = "visible_defect",
        missing_anomaly_mask_policy: str = "include_as_normal_view",
        mask_resize_semantics: str = "upstream_bilinear_nonzero",
    ) -> None:
        self.category = category
        self.phase = phase
        self.category_root = category_image_root(image_dir, category)
        records = load_records(json_dir, category, phase)
        self.records = _stratified_limit(records, max_items)
        self.transform = image_transform(image_size, crop_size)
        self.gt_transform = mask_transform(
            image_size,
            crop_size,
            mask_resize_semantics,
        )
        self.crop_size = crop_size
        if image_label_policy not in {"visible_defect", "object_anomaly"}:
            raise ValueError(
                "image_label_policy 必须为 visible_defect 或 object_anomaly"
            )
        if missing_anomaly_mask_policy not in {
            "include_as_normal_view",
            "exclude_pixel_metrics",
        }:
            raise ValueError(
                "missing_anomaly_mask_policy 必须为 include_as_normal_view "
                "或 exclude_pixel_metrics"
            )
        self.image_label_policy = image_label_policy
        self.missing_anomaly_mask_policy = missing_anomaly_mask_policy
        self.mask_resize_semantics = mask_resize_semantics

        if phase == "train":
            anomalous = sum(record.is_anomaly for record in self.records)
            if anomalous:
                raise ValueError(
                    f"{category} train split 含 {anomalous} 个异常图像，"
                    "不符合无监督训练协议"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record_file(self.category_root, record.image_path)
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))

        image_label = int(record.is_anomaly)
        if self.image_label_policy == "visible_defect" and record.mask_path is None:
            image_label = 0
        item: dict[str, Any] = {
            "image": image,
            "label": image_label,
            "object_label": int(record.is_anomaly),
            "category": record.category,
            "object_id": record.object_id,
            "view_id": record.view_id,
            "image_path": str(image_path),
        }
        if self.phase == "test":
            if record.mask_path is None:
                mask = torch.zeros((1, self.crop_size, self.crop_size), dtype=torch.float32)
                pixel_valid = (
                    not record.is_anomaly
                    or self.missing_anomaly_mask_policy == "include_as_normal_view"
                )
            else:
                mask_path = record_file(self.category_root, record.mask_path)
                with Image.open(mask_path) as source:
                    mask = self.gt_transform(source.convert("L"))
                threshold = (
                    0.0
                    if self.mask_resize_semantics
                    == "upstream_bilinear_nonzero"
                    else 0.5
                )
                mask = (mask > threshold).to(torch.float32)
                pixel_valid = True
            item["mask"] = mask
            item["pixel_valid"] = pixel_valid
        return item


def group_records_by_object(
    records: Sequence[Record],
    *,
    num_views: int = 5,
    missing_view_policy: str = "error",
) -> list[tuple[str, tuple[int | None, ...]]]:
    """Group Real-IAD records into ordered, validated camera sets.

    Real-IAD camera names are one-based (``C1`` ... ``C5``), while the model
    contract is deliberately zero-based.  Returning record indices keeps the
    grouping deterministic and avoids copying manifest objects.
    """

    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if missing_view_policy not in {
        "error",
        "pad_and_mask",
        "drop_incomplete",
    }:
        raise ValueError(
            "missing_view_policy must be error, pad_and_mask, or "
            "drop_incomplete"
        )

    grouped: dict[str, list[int | None]] = {}
    for index, record in enumerate(records):
        camera_id = int(record.view_id)
        if not 1 <= camera_id <= num_views:
            raise ValueError(
                f"{record.category}/{record.object_id} has out-of-range "
                f"camera/view id {camera_id}; expected C1..C{num_views}"
            )
        view_id = camera_id - 1
        slots = grouped.setdefault(record.object_id, [None] * num_views)
        if slots[view_id] is not None:
            previous = records[int(slots[view_id])]
            raise ValueError(
                f"{record.category}/{record.object_id} contains duplicate "
                f"view {view_id}: {previous.image_path!r} and "
                f"{record.image_path!r}"
            )
        slots[view_id] = index

    result: list[tuple[str, tuple[int | None, ...]]] = []
    for object_id in sorted(grouped):
        slots = grouped[object_id]
        missing = [index for index, value in enumerate(slots) if value is None]
        if missing:
            if missing_view_policy == "error":
                category = records[next(value for value in slots if value is not None)].category
                raise ValueError(
                    f"{category}/{object_id} is missing views {missing}; "
                    f"policy={missing_view_policy}"
                )
            if missing_view_policy == "drop_incomplete":
                continue
        result.append((object_id, tuple(slots)))
    if not result:
        raise ValueError("multi-view grouping produced no usable objects")
    return result


class RealIADMultiViewDataset(Dataset[dict[str, Any]]):
    """One dataset item is one object with ordered camera views."""

    def __init__(
        self,
        json_dir: Path,
        image_dir: Path,
        category: str,
        phase: str,
        image_size: int,
        crop_size: int,
        *,
        num_views: int = 5,
        missing_view_policy: str = "error",
        max_objects: int | None = None,
        image_label_policy: str = "visible_defect",
        missing_anomaly_mask_policy: str = "include_as_normal_view",
        mask_resize_semantics: str = "upstream_bilinear_nonzero",
    ) -> None:
        self.base = RealIADVarietyDataset(
            json_dir=json_dir,
            image_dir=image_dir,
            category=category,
            phase=phase,
            image_size=image_size,
            crop_size=crop_size,
            max_items=None,
            image_label_policy=image_label_policy,
            missing_anomaly_mask_policy=missing_anomaly_mask_policy,
            mask_resize_semantics=mask_resize_semantics,
        )
        self.category = category
        self.phase = phase
        self.num_views = int(num_views)
        self.crop_size = int(crop_size)
        groups = group_records_by_object(
            self.base.records,
            num_views=self.num_views,
            missing_view_policy=missing_view_policy,
        )
        if max_objects is not None:
            groups = groups[: int(max_objects)]
        if not groups:
            raise ValueError(f"{category} has no usable multi-view objects")
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        object_id, slots = self.groups[index]
        loaded: list[dict[str, Any] | None] = [
            None if item_index is None else self.base[int(item_index)]
            for item_index in slots
        ]
        template = next(item for item in loaded if item is not None)
        images: list[torch.Tensor] = []
        labels: list[int] = []
        pixel_valid: list[bool] = []
        masks: list[torch.Tensor] = []
        image_paths: list[str] = []
        valid_views: list[bool] = []
        object_labels: list[int] = []
        for item in loaded:
            if item is None:
                images.append(torch.zeros_like(template["image"]))
                labels.append(0)
                object_labels.append(0)
                image_paths.append("")
                valid_views.append(False)
                if self.phase == "test":
                    masks.append(
                        torch.zeros(
                            (1, self.crop_size, self.crop_size),
                            dtype=torch.float32,
                        )
                    )
                    pixel_valid.append(False)
                continue
            images.append(item["image"])
            labels.append(int(item["label"]))
            object_labels.append(int(item["object_label"]))
            image_paths.append(str(item["image_path"]))
            valid_views.append(True)
            if self.phase == "test":
                masks.append(item["mask"])
                pixel_valid.append(bool(item["pixel_valid"]))

        result: dict[str, Any] = {
            "images": torch.stack(images, dim=0),
            "view_ids": torch.arange(self.num_views, dtype=torch.long),
            "valid_view_mask": torch.tensor(valid_views, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "object_label": max(object_labels),
            "category": self.category,
            "object_id": object_id,
            "image_paths": tuple(image_paths),
        }
        if self.phase == "test":
            result["masks"] = torch.stack(masks, dim=0)
            result["pixel_valid"] = torch.tensor(pixel_valid, dtype=torch.bool)
        return result


def build_train_dataset(
    json_dir: Path,
    image_dir: Path,
    categories: Iterable[str],
    image_size: int,
    crop_size: int,
    *,
    multi_view_enabled: bool = False,
    num_views: int = 5,
    missing_view_policy: str = "error",
) -> ConcatDataset[dict[str, Any]]:
    dataset_class = (
        RealIADMultiViewDataset if multi_view_enabled else RealIADVarietyDataset
    )
    datasets = [
        dataset_class(
            json_dir=json_dir,
            image_dir=image_dir,
            category=category,
            phase="train",
            image_size=image_size,
            crop_size=crop_size,
            **(
                {
                    "num_views": num_views,
                    "missing_view_policy": missing_view_policy,
                }
                if multi_view_enabled
                else {}
            ),
        )
        for category in categories
    ]
    return ConcatDataset(datasets)
