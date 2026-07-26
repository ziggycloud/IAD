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


def build_train_dataset(
    json_dir: Path,
    image_dir: Path,
    categories: Iterable[str],
    image_size: int,
    crop_size: int,
) -> ConcatDataset[dict[str, Any]]:
    datasets = [
        RealIADVarietyDataset(
            json_dir=json_dir,
            image_dir=image_dir,
            category=category,
            phase="train",
            image_size=image_size,
            crop_size=crop_size,
        )
        for category in categories
    ]
    return ConcatDataset(datasets)
