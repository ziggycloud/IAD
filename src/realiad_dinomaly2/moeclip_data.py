"""Auxiliary supervised anomaly dataset used by the official MoECLIP protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import resolve_path
from .zero_shot_model import IMAGENET_MEAN, IMAGENET_STD


class MoECLIPAuxiliaryDataset(Dataset[dict[str, Any]]):
    """Read official-style JSONL entries with real labels and masks.

    Each line must contain ``image_path``, ``class_name`` and ``label``.
    An anomalous line (label=1) must additionally contain ``mask_path``.
    Paths are relative to ``root`` unless absolute.
    """

    def __init__(self, root: Path, metadata: Path, image_size: int,
                 augment: bool = True,
                 prompt_names: dict[str, str] | None = None) -> None:
        self.root = root.expanduser().resolve()
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.prompt_names = dict(prompt_names or {})
        self.geometric = transforms.Compose([
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=30)], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        if not self.root.is_dir():
            raise FileNotFoundError(f"MoECLIP auxiliary root does not exist: {self.root}")
        if not metadata.is_file():
            raise FileNotFoundError(f"MoECLIP metadata does not exist: {metadata}")
        self.samples: list[dict[str, Any]] = []
        with metadata.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                missing = {"image_path", "class_name", "label"} - set(item)
                if missing:
                    raise ValueError(f"metadata line {line_number} misses {sorted(missing)}")
                label = int(item["label"])
                if label not in (0, 1):
                    raise ValueError(f"metadata line {line_number} label must be 0 or 1")
                if label and not item.get("mask_path"):
                    raise ValueError(f"metadata line {line_number} anomaly has no mask_path")
                self.samples.append(item)
        if not self.samples:
            raise ValueError("MoECLIP auxiliary metadata is empty")
        labels = {int(item["label"]) for item in self.samples}
        if labels != {0, 1}:
            raise ValueError("faithful MoECLIP training requires both normal and anomalous images")

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        with Image.open(self._path(str(item["image_path"]))) as source:
            image = source.convert("RGB")
        if int(item["label"]):
            with Image.open(self._path(str(item["mask_path"]))) as source:
                mask = source.convert("L")
        else:
            mask = Image.new("L", image.size, 0)
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BICUBIC)
        mask = TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) != 0).float()
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        if self.augment:
            transformed = self.geometric(
                torch.cat((image_tensor, mask_tensor), dim=0)
            )
            image_tensor, mask_tensor = transformed[:3], transformed[3:4]
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "class_name": str(
                item.get(
                    "prompt_name",
                    self.prompt_names.get(
                        str(item["class_name"]), str(item["class_name"])
                    ),
                )
            ),
            "image_path": str(item["image_path"]),
        }


def build_moeclip_auxiliary_dataset(config: dict[str, Any]) -> MoECLIPAuxiliaryDataset:
    training = config["zero_shot"]["training"]
    auxiliary = training.get("auxiliary_dataset", {})
    root_value = auxiliary.get("root")
    metadata_value = auxiliary.get("metadata")
    if not root_value or not metadata_value:
        raise ValueError(
            "faithful MoECLIP training requires zero_shot.training.auxiliary_dataset.root "
            "and .metadata (JSONL with real anomaly masks)"
        )
    return MoECLIPAuxiliaryDataset(
        resolve_path(str(root_value)), resolve_path(str(metadata_value)),
        int(config["zero_shot"]["model"].get("image_size", 518)),
        bool(auxiliary.get("augment", True)),
        auxiliary.get("prompt_names", {}),
    )
