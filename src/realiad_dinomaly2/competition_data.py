from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


def build_competition_train_dataset(
    train_dir: Path,
    categories: str | Sequence[str],
    category_limit: int | None,
    image_size: int,
    crop_size: int,
) -> tuple[CompetitionFolderDataset, CompetitionManifest]:
    manifest = scan_competition_split(
        train_dir,
        requested=categories,
        limit=category_limit,
    )
    dataset = CompetitionFolderDataset(
        manifest.views,
        image_size=image_size,
        crop_size=crop_size,
    )
    return dataset, manifest
