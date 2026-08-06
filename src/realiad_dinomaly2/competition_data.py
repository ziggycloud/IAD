from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import image_transform


SAMPLE_PATTERN = re.compile(r"^S\d+$", re.IGNORECASE)
EXPECTED_VIEW_IDS = tuple(range(5))


@dataclass(frozen=True)
class CompetitionView:
    category: str
    sample: str
    view_id: int
    image_path: Path

    @property
    def group_folder(self) -> str:
        return f"{self.category}/{self.sample}"


@dataclass(frozen=True)
class CompetitionManifest:
    root: Path
    categories: tuple[str, ...]
    views: tuple[CompetitionView, ...]

    @property
    def group_folders(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(view.group_folder for view in self.views)
        )

    def views_for_category(self, category: str) -> tuple[CompetitionView, ...]:
        return tuple(view for view in self.views if view.category == category)

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "categories": len(self.categories),
            "samples": len(self.group_folders),
            "views": len(self.views),
            "views_per_sample": len(EXPECTED_VIEW_IDS),
        }


def _selected_categories(
    available: list[str],
    requested: str | Sequence[str],
    limit: int | None,
) -> list[str]:
    if requested == "all":
        selected = available
    else:
        selected = list(requested)
        if len(selected) != len(set(selected)):
            raise ValueError("dataset.categories contains duplicates")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown competition categories: {unknown}")
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("Competition category selection is empty")
    return selected


def scan_competition_split(
    root: Path,
    requested: str | Sequence[str] = "all",
    limit: int | None = None,
) -> CompetitionManifest:
    """Validate and enumerate ``category/Sxxxx/0.png..4.png`` deterministically."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Competition split directory does not exist: {root}")
    available = sorted(path.name for path in root.iterdir() if path.is_dir())
    categories = _selected_categories(available, requested, limit)
    views: list[CompetitionView] = []
    expected_names = {f"{view_id}.png" for view_id in EXPECTED_VIEW_IDS}

    for category in categories:
        category_dir = root / category
        samples = sorted(path for path in category_dir.iterdir() if path.is_dir())
        if not samples:
            raise ValueError(f"Competition category has no samples: {category_dir}")
        for sample_dir in samples:
            if SAMPLE_PATTERN.fullmatch(sample_dir.name) is None:
                raise ValueError(
                    f"Competition sample must be named Sxxxx: {sample_dir}"
                )
            nested_directories = [path for path in sample_dir.iterdir() if path.is_dir()]
            if nested_directories:
                raise ValueError(
                    f"Competition sample contains nested directories: {sample_dir}"
                )
            png_names = {
                path.name for path in sample_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".png"
            }
            missing = sorted(expected_names - png_names)
            unexpected = sorted(png_names - expected_names)
            if missing or unexpected:
                raise ValueError(
                    f"Invalid five-view sample {sample_dir}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for view_id in EXPECTED_VIEW_IDS:
                views.append(
                    CompetitionView(
                        category=category,
                        sample=sample_dir.name,
                        view_id=view_id,
                        image_path=(sample_dir / f"{view_id}.png").resolve(),
                    )
                )

    return CompetitionManifest(
        root=root,
        categories=tuple(categories),
        views=tuple(views),
    )


class CompetitionFolderDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        views: Sequence[CompetitionView],
        image_size: int,
        crop_size: int,
    ) -> None:
        if not views:
            raise ValueError("CompetitionFolderDataset requires at least one view")
        self.views = tuple(views)
        self.transform = image_transform(image_size, crop_size)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> dict[str, Any]:
        view = self.views[index]
        with Image.open(view.image_path) as source:
            image = self.transform(source.convert("RGB"))
        return {
            "image": image,
            "category": view.category,
            "sample": view.sample,
            "group_folder": view.group_folder,
            "view_id": view.view_id,
            "image_path": str(view.image_path),
        }


class CompetitionObjectDataset(Dataset[dict[str, Any]]):
    """One strict competition sample with five ordered camera images."""

    def __init__(
        self,
        views: Sequence[CompetitionView],
        image_size: int,
        crop_size: int,
        *,
        num_views: int = 5,
        missing_view_policy: str = "error",
    ) -> None:
        if num_views != len(EXPECTED_VIEW_IDS):
            raise ValueError("competition requires exactly five views")
        if missing_view_policy not in {
            "error",
            "pad_and_mask",
            "drop_incomplete",
        }:
            raise ValueError("invalid missing_view_policy")
        grouped: dict[str, list[CompetitionView | None]] = {}
        for view in views:
            if not 0 <= int(view.view_id) < num_views:
                raise ValueError(
                    f"{view.group_folder} has out-of-range view {view.view_id}"
                )
            slots = grouped.setdefault(view.group_folder, [None] * num_views)
            if slots[view.view_id] is not None:
                raise ValueError(
                    f"{view.group_folder} contains duplicate view {view.view_id}"
                )
            slots[view.view_id] = view
        objects: list[tuple[str, tuple[CompetitionView | None, ...]]] = []
        for group_folder, slots in grouped.items():
            missing = [index for index, value in enumerate(slots) if value is None]
            if missing:
                if missing_view_policy == "error":
                    raise ValueError(
                        f"{group_folder} is missing competition views {missing}"
                    )
                if missing_view_policy == "drop_incomplete":
                    continue
            objects.append((group_folder, tuple(slots)))
        if not objects:
            raise ValueError("CompetitionObjectDataset has no usable objects")
        self.objects = tuple(objects)
        self.num_views = int(num_views)
        self.crop_size = int(crop_size)
        self.transform = image_transform(image_size, crop_size)

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_folder, slots = self.objects[index]
        template_view = next(view for view in slots if view is not None)
        images: list[torch.Tensor] = []
        valid: list[bool] = []
        image_paths: list[str] = []
        for view in slots:
            if view is None:
                # This branch is only reachable for an explicit pad_and_mask
                # ablation; the official competition scanner is strict.
                images.append(torch.zeros(3, self.crop_size, self.crop_size))
                valid.append(False)
                image_paths.append("")
                continue
            with Image.open(view.image_path) as source:
                images.append(self.transform(source.convert("RGB")))
            valid.append(True)
            image_paths.append(str(view.image_path))
        return {
            "images": torch.stack(images, dim=0),
            "view_ids": torch.arange(self.num_views, dtype=torch.long),
            "valid_view_mask": torch.tensor(valid, dtype=torch.bool),
            "category": template_view.category,
            "sample": template_view.sample,
            "group_folder": group_folder,
            "image_paths": tuple(image_paths),
        }


def build_competition_train_dataset(
    train_dir: Path,
    categories: str | Sequence[str],
    category_limit: int | None,
    image_size: int,
    crop_size: int,
    *,
    multi_view_enabled: bool = False,
    num_views: int = 5,
    missing_view_policy: str = "error",
) -> tuple[Dataset[dict[str, Any]], CompetitionManifest]:
    manifest = scan_competition_split(
        train_dir,
        requested=categories,
        limit=category_limit,
    )
    dataset_class = (
        CompetitionObjectDataset if multi_view_enabled else CompetitionFolderDataset
    )
    dataset = dataset_class(
        manifest.views,
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
    return dataset, manifest
