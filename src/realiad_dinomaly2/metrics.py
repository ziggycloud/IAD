from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from .config import PROJECT_ROOT


def binary_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.uint8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.size == 0:
        raise ValueError("指标输入为空")
    if np.unique(labels_array).size != 2:
        raise ValueError(
            "AUROC/F1max 要求同时包含 normal 和 anomaly；"
            f"实际 labels={np.unique(labels_array).tolist()}"
        )
    precision, recall, _ = precision_recall_curve(
        labels_array,
        scores_array,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return {
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "aupr": float(average_precision_score(labels_array, scores_array)),
        "f1max": float(np.max(f1)),
    }


def top_ratio_mean(anomaly_maps: torch.Tensor, ratio: float) -> torch.Tensor:
    flattened = anomaly_maps.flatten(1)
    count = max(1, int(flattened.shape[1] * ratio))
    return torch.topk(
        flattened,
        k=count,
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size: int, sigma: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel_size 必须是正奇数")
        coordinates = torch.arange(kernel_size, dtype=torch.float32)
        coordinates -= kernel_size // 2
        grid_y, grid_x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        kernel = torch.exp(-(grid_x.square() + grid_y.square()) / (2 * sigma**2))
        kernel /= kernel.sum()
        self.register_buffer("weight", kernel.reshape(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, anomaly_maps: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            anomaly_maps,
            self.weight,
            padding=self.padding,
        )


def _adeval_classes():
    upstream = PROJECT_ROOT / "third_party" / "Dinomaly2"
    upstream_string = str(upstream)
    if upstream_string not in sys.path:
        sys.path.insert(0, upstream_string)
    from adeval import EvalAccumulator, EvalAccumulatorCuda

    return EvalAccumulator, EvalAccumulatorCuda


class PixelMetricAccumulator:
    def __init__(
        self,
        device: torch.device,
        bins: int,
        score_lower: float,
        score_upper: float,
    ) -> None:
        cpu_class, cuda_class = _adeval_classes()
        self.device = device
        accumulator_class = cuda_class if device.type == "cuda" else cpu_class
        self.accumulator = accumulator_class(
            score_lower,
            score_upper,
            score_lower,
            score_upper,
            nstrips=bins,
        )
        self.image_count = 0

    def add(self, anomaly_maps: torch.Tensor, masks: torch.Tensor) -> None:
        if anomaly_maps.ndim != 3 or masks.ndim != 3:
            raise ValueError("pixel accumulator 需要 [N,H,W]")
        if anomaly_maps.shape != masks.shape:
            raise ValueError(
                f"anomaly/mask shape 不同：{anomaly_maps.shape} vs {masks.shape}"
            )
        if self.device.type == "cuda":
            self.accumulator.add_anomap_batch(
                anomaly_maps.to(self.device, dtype=torch.float32),
                masks.to(self.device, dtype=torch.uint8),
            )
        else:
            self.accumulator.add_anomap_batch(
                anomaly_maps.cpu().numpy().astype(np.float32),
                masks.cpu().numpy().astype(np.uint8),
            )
        self.image_count += int(anomaly_maps.shape[0])

    def summary(self) -> dict[str, float]:
        if self.image_count == 0:
            raise ValueError("没有可用于像素指标的图像")
        metrics = self.accumulator.summary()
        return {
            "auroc": float(metrics["p_auroc"]),
            "aupr": float(metrics["p_aupr"]),
            "f1max": float(metrics["p_f1max"]),
            "aupro": float(metrics["p_aupro"]),
        }


class SpoolingPixelMetricAccumulator:
    """Spool one category to disk, then use its exact score min/max.

    The released evaluator builds adeval's 1,000 thresholds from each
    category's observed anomaly-map range. A fixed theoretical [0, 2] range
    is much coarser. Memmaps preserve the released behavior without keeping
    hundreds of MiB of maps in RAM or moving the whole category to the GPU.
    """

    def __init__(
        self,
        device: torch.device,
        bins: int,
        capacity: int,
        height: int,
        width: int,
        scratch_dir: Path,
        stem: str,
        replay_batch_size: int = 8,
    ) -> None:
        if capacity <= 0:
            raise ValueError("pixel metric capacity 必须为正整数")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.bins = bins
        self.capacity = capacity
        self.height = height
        self.width = width
        self.replay_batch_size = max(1, int(replay_batch_size))
        self.map_path = scratch_dir / f"{stem}.maps.f32"
        self.mask_path = scratch_dir / f"{stem}.masks.u8"
        self.map_path.unlink(missing_ok=True)
        self.mask_path.unlink(missing_ok=True)
        shape = (capacity, height, width)
        self.maps: np.memmap | None = np.memmap(
            self.map_path,
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        self.masks: np.memmap | None = np.memmap(
            self.mask_path,
            dtype=np.uint8,
            mode="w+",
            shape=shape,
        )
        self.image_count = 0
        self.score_min = float("inf")
        self.score_max = float("-inf")

    def add(self, anomaly_maps: torch.Tensor, masks: torch.Tensor) -> None:
        if self.maps is None or self.masks is None:
            raise RuntimeError("pixel metric spool 已关闭")
        if anomaly_maps.ndim != 3 or masks.ndim != 3:
            raise ValueError("pixel spool 需要 [N,H,W]")
        if anomaly_maps.shape != masks.shape:
            raise ValueError(
                f"anomaly/mask shape 不同：{anomaly_maps.shape} vs {masks.shape}"
            )
        count = int(anomaly_maps.shape[0])
        end = self.image_count + count
        if end > self.capacity:
            raise ValueError(
                f"pixel spool 容量 {self.capacity} 不足，尝试写到 {end}"
            )
        map_array = (
            anomaly_maps.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        mask_array = (
            masks.detach()
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        self.maps[self.image_count : end] = map_array
        self.masks[self.image_count : end] = mask_array
        self.score_min = min(self.score_min, float(map_array.min()))
        self.score_max = max(self.score_max, float(map_array.max()))
        self.image_count = end

    def close(self, remove: bool = True) -> None:
        for name in ("maps", "masks"):
            array = getattr(self, name)
            if array is not None:
                array.flush()
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                setattr(self, name, None)
        if remove:
            self.map_path.unlink(missing_ok=True)
            self.mask_path.unlink(missing_ok=True)

    def summary(self) -> dict[str, float]:
        if self.image_count == 0:
            self.close()
            raise ValueError("没有可用于像素指标的图像")
        if self.maps is None or self.masks is None:
            raise RuntimeError("pixel metric spool 已关闭")
        lower = self.score_min
        upper = self.score_max
        if not lower < upper:
            delta = max(1e-7, abs(lower) * 1e-7)
            lower -= delta
            upper += delta
        accumulator = PixelMetricAccumulator(
            device=self.device,
            bins=self.bins,
            score_lower=lower,
            score_upper=upper,
        )
        try:
            for start in range(0, self.image_count, self.replay_batch_size):
                end = min(self.image_count, start + self.replay_batch_size)
                maps = torch.from_numpy(
                    np.asarray(self.maps[start:end])
                )
                masks = torch.from_numpy(
                    np.asarray(self.masks[start:end])
                )
                accumulator.add(maps, masks)
            result = accumulator.summary()
            result["score_min"] = self.score_min
            result["score_max"] = self.score_max
            return result
        finally:
            self.close()


class ObjectScoreAccumulator:
    """Compute top-ratio score across all five views without retaining a class."""

    def __init__(self, top_ratio: float, expected_views: int = 5) -> None:
        self.top_ratio = top_ratio
        self.expected_views = expected_views
        self.pending: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.pending_labels: dict[str, list[int]] = defaultdict(list)
        self.labels: list[int] = []
        self.scores: list[float] = []
        self.view_counts: list[int] = []

    def add(
        self,
        object_key: str,
        anomaly_map: torch.Tensor,
        image_label: int,
    ) -> None:
        self.pending[object_key].append(
            anomaly_map.detach().reshape(-1).to(dtype=torch.float32).clone()
        )
        self.pending_labels[object_key].append(int(image_label))
        if len(self.pending[object_key]) == self.expected_views:
            self._finish(object_key)

    def _finish(self, object_key: str) -> None:
        maps = self.pending.pop(object_key)
        labels = self.pending_labels.pop(object_key)
        flattened = torch.cat(maps)
        count = max(1, int(flattened.numel() * self.top_ratio))
        score = torch.topk(
            flattened,
            k=count,
            largest=True,
            sorted=False,
        ).values.mean()
        self.labels.append(max(labels))
        self.scores.append(float(score.cpu()))
        self.view_counts.append(len(maps))

    def finish_pending(self) -> None:
        for key in list(self.pending):
            self._finish(key)

    def summary(self) -> dict[str, float]:
        self.finish_pending()
        result = binary_metrics(self.labels, self.scores)
        result["objects"] = float(len(self.labels))
        result["min_views"] = float(min(self.view_counts))
        result["max_views"] = float(max(self.view_counts))
        return result


def mean_and_std(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        means[key] = float(values.mean())
        standard_deviations[key] = float(values.std(ddof=0))
    return means, standard_deviations
