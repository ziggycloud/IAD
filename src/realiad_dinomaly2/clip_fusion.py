from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .competition_data import (
    CompetitionFolderDataset,
    build_competition_train_dataset,
)
from .config import config_fingerprint
from .losses import anomaly_map
from .modeling import build_model, load_trainable_state_dict
from .normal_prior import file_sha256
from .runtime import (
    amp_dtype,
    atomic_torch_save,
    atomic_write_json,
    autocast_context,
    resolve_device,
    utc_now,
)


CLIP_FUSION_FORMAT_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_PROMPT_TEMPLATES = (
    "a photo of a {phrase}.",
    "a close-up photo of a {phrase}.",
    "a cropped industrial inspection photo of a {phrase}.",
    "a bright inspection image of a {phrase}.",
)
_NORMAL_PHRASES = (
    "flawless {category}",
    "undamaged {category}",
    "normal {category}",
    "intact {category}",
    "correctly assembled {category}",
    "{category} without any defect",
)
_ANOMALOUS_PHRASES = (
    "defective {category}",
    "damaged {category}",
    "anomalous {category}",
    "broken {category}",
    "deformed {category}",
    "bent or twisted {category}",
    "misaligned {category}",
    "cracked or scratched {category}",
    "{category} with a missing part",
    "{category} with an incorrect shape",
)


def _category_words(category: str) -> str:
    return " ".join(str(category).replace("-", "_").split("_")).lower()


def build_prompt_ensemble(category: str) -> dict[str, tuple[str, ...]]:
    """Build category-aware normal/abnormal CLIP prompt ensembles."""

    category_words = _category_words(category)

    def expand(phrases: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            template.format(phrase=phrase.format(category=category_words))
            for phrase in phrases
            for template in _PROMPT_TEMPLATES
        )

    return {
        "normal": expand(_NORMAL_PHRASES),
        "anomalous": expand(_ANOMALOUS_PHRASES),
    }


def clip_fusion_path(config: dict[str, Any]) -> Path:
    fusion_config = config["evaluation"].get("clip_fusion", {})
    configured = fusion_config.get("artifact_path")
    output_dir = Path(config["experiment"]["output_dir"])
    if configured is None:
        return output_dir / "clip_fusion" / "clip_fusion.pt"
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else output_dir / path


def _clip_config_fingerprint(config: dict[str, Any]) -> str:
    fusion_config = dict(config["evaluation"].get("clip_fusion", {}))
    fusion_config.pop("artifact_path", None)
    payload = {
        "crop_size": int(config["dataset"]["crop_size"]),
        "clip_fusion": fusion_config,
        "prompt_templates": _PROMPT_TEMPLATES,
        "normal_phrases": _NORMAL_PHRASES,
        "anomalous_phrases": _ANOMALOUS_PHRASES,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_clip_pretrained(config: dict[str, Any]) -> tuple[str, str]:
    fusion_config = config["evaluation"]["clip_fusion"]
    configured = fusion_config.get("weights_path")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OpenCLIP weights do not exist: {path}")
        return str(path), file_sha256(path)
    pretrained = str(fusion_config.get("pretrained", "openai"))
    identity = json.dumps(
        {
            "model_name": fusion_config.get("model_name"),
            "pretrained": pretrained,
        },
        sort_keys=True,
    )
    return pretrained, hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _import_open_clip():
    try:
        import open_clip  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "CLIP fusion requires open_clip_torch==3.3.0. Run setup_env.ps1 "
            "or pip install -r requirements.txt before this competition pipe."
        ) from exc
    return open_clip


class ClipSemanticEncoder:
    """Frozen OpenCLIP with category-aware, multi-level dense anomaly prompts."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        fusion_config = config["evaluation"]["clip_fusion"]
        open_clip = _import_open_clip()
        model_name = str(fusion_config.get("model_name", "ViT-L-14-336"))
        pretrained, self.weights_identity = _resolve_clip_pretrained(config)
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load OpenCLIP {model_name!r} with pretrained="
                f"{pretrained!r}. Set evaluation.clip_fusion.weights_path to "
                "a local checkpoint when the runtime cannot download weights."
            ) from exc
        visual = getattr(model, "visual", None)
        if visual is None or not hasattr(visual, "forward_intermediates"):
            raise TypeError(
                "clip_fusion requires an OpenCLIP ViT visual tower with "
                "forward_intermediates"
            )
        if getattr(visual, "proj", None) is None:
            raise TypeError("clip_fusion requires a CLIP visual projection")
        self.model = model.eval()
        self.visual = visual
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.image_size = int(fusion_config.get("image_size", 336))
        visual_size = getattr(self.visual, "image_size", self.image_size)
        if isinstance(visual_size, (tuple, list)):
            visual_size = int(visual_size[0])
        if int(visual_size) != self.image_size:
            raise ValueError(
                "clip_fusion.image_size must match the selected OpenCLIP model: "
                f"{self.image_size} != {visual_size}"
            )
        preprocess_config = getattr(self.visual, "preprocess_cfg", {})
        self.clip_mean = tuple(
            float(value)
            for value in preprocess_config.get("mean", _OPENAI_CLIP_MEAN)
        )
        self.clip_std = tuple(
            float(value)
            for value in preprocess_config.get("std", _OPENAI_CLIP_STD)
        )
        self.prompt_temperature = float(
            fusion_config.get("prompt_temperature", 0.05)
        )
        self.category_prompt_blend = float(
            fusion_config.get("category_prompt_blend", 0.8)
        )
        layers = [int(value) for value in fusion_config.get("feature_layers", [])]
        if not layers:
            raise ValueError("clip_fusion.feature_layers cannot be empty")
        block_count = len(self.visual.transformer.resblocks)
        if any(value < 1 or value > block_count for value in layers):
            raise ValueError(
                f"clip_fusion.feature_layers must be in [1, {block_count}]"
            )
        self.layer_indices = [value - 1 for value in layers]
        self.layer_weights = self._normalized_weights(
            fusion_config.get("layer_weights"),
            len(self.layer_indices),
            "layer_weights",
        )
        self.local_kernels = [
            int(value) for value in fusion_config.get("local_kernels", [1, 3, 5])
        ]
        self.local_weights = self._normalized_weights(
            fusion_config.get("local_weights"),
            len(self.local_kernels),
            "local_weights",
        )
        self.global_context_weight = float(
            fusion_config.get("global_context_weight", 0.15)
        )
        self._prototype_cache: dict[str, torch.Tensor] = {}
        self._generic_prototypes: torch.Tensor | None = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _normalized_weights(
        configured: object,
        count: int,
        name: str,
    ) -> tuple[float, ...]:
        values = (
            [1.0] * count
            if configured is None
            else [float(value) for value in configured]
        )
        if len(values) != count or any(value < 0 for value in values):
            raise ValueError(
                f"clip_fusion.{name} must contain {count} non-negative values"
            )
        total = sum(values)
        if total <= 0:
            raise ValueError(f"clip_fusion.{name} must have a positive sum")
        return tuple(value / total for value in values)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        pixels = (images.float() * std + mean).clamp_(0.0, 1.0)
        pixels = F.interpolate(
            pixels,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        clip_mean = pixels.new_tensor(self.clip_mean).view(1, 3, 1, 1)
        clip_std = pixels.new_tensor(self.clip_std).view(1, 3, 1, 1)
        return (pixels - clip_mean) / clip_std

    def _encode_prompt_state(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        encoded = self.model.encode_text(tokens, normalize=True).float()
        prototype = F.normalize(encoded.mean(dim=0), dim=0, eps=1e-8)
        return prototype

    def _prototypes(self, category: str) -> torch.Tensor:
        category = str(category)
        cached = self._prototype_cache.get(category)
        if cached is not None:
            return cached
        prompts = build_prompt_ensemble(category)
        category_states = torch.stack(
            [
                self._encode_prompt_state(prompts["normal"]),
                self._encode_prompt_state(prompts["anomalous"]),
            ],
            dim=0,
        )
        if self._generic_prototypes is None:
            generic = build_prompt_ensemble("object")
            self._generic_prototypes = torch.stack(
                [
                    self._encode_prompt_state(generic["normal"]),
                    self._encode_prompt_state(generic["anomalous"]),
                ],
                dim=0,
            )
        generic_states = self._generic_prototypes
        prototypes = F.normalize(
            self.category_prompt_blend * category_states
            + (1.0 - self.category_prompt_blend) * generic_states,
            dim=-1,
            eps=1e-8,
        )
        self._prototype_cache[category] = prototypes
        return prototypes

    def _batch_prototypes(self, categories: Sequence[str]) -> torch.Tensor:
        return torch.stack(
            [self._prototypes(str(category)) for category in categories],
            dim=0,
        )

    def _project_dense(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = feature_map.float().permute(0, 2, 3, 1)
        projection = self.visual.proj
        if isinstance(projection, nn.Linear):
            tokens = projection(tokens)
        else:
            tokens = tokens @ projection
        tokens = F.normalize(tokens.float(), dim=-1, eps=1e-8)
        return tokens.permute(0, 3, 1, 2).contiguous()

    def _semantic_probability(
        self,
        feature_map: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(feature_map.float(), dim=1, eps=1e-8)
        similarities = torch.einsum("bdhw,bkd->bkhw", normalized, prototypes)
        margin = similarities[:, 1:2] - similarities[:, 0:1]
        return torch.sigmoid(margin / self.prompt_temperature)

    @torch.inference_mode()
    def semantic_map(
        self,
        images: torch.Tensor,
        categories: Sequence[str],
        config: dict[str, Any],
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("clip semantic encoder expects images [B,3,H,W]")
        if len(categories) != images.shape[0]:
            raise ValueError("categories must contain one value per CLIP image")
        dtype = (
            amp_dtype(config, self.device)
            if bool(config["evaluation"].get("amp", False))
            else None
        )
        clip_images = self._preprocess(images)
        prototypes = self._batch_prototypes(categories)
        with autocast_context(dtype, self.device):
            output = self.visual.forward_intermediates(
                clip_images,
                indices=self.layer_indices,
                normalize_intermediates=True,
                intermediates_only=False,
                output_fmt="NCHW",
            )
        intermediates = output.get("image_intermediates")
        if not isinstance(intermediates, list) or len(intermediates) != len(
            self.layer_indices
        ):
            raise RuntimeError("OpenCLIP returned incompatible intermediate features")

        scale_maps: list[torch.Tensor] = []
        for layer_feature, layer_weight in zip(
            intermediates,
            self.layer_weights,
            strict=True,
        ):
            projected = self._project_dense(layer_feature)
            for kernel, local_weight in zip(
                self.local_kernels,
                self.local_weights,
                strict=True,
            ):
                if kernel == 1:
                    local = projected
                else:
                    local = F.avg_pool2d(
                        projected,
                        kernel_size=kernel,
                        stride=1,
                        padding=kernel // 2,
                        count_include_pad=False,
                    )
                probability = self._semantic_probability(local, prototypes)
                scale_maps.append(probability * (layer_weight * local_weight))
        semantic = torch.stack(scale_maps, dim=0).sum(dim=0)

        image_features = output.get("image_features")
        if not isinstance(image_features, torch.Tensor):
            raise RuntimeError("OpenCLIP did not return global image features")
        image_features = F.normalize(image_features.float(), dim=-1, eps=1e-8)
        global_similarity = torch.einsum(
            "bd,bkd->bk",
            image_features,
            prototypes,
        )
        global_margin = global_similarity[:, 1] - global_similarity[:, 0]
        global_probability = torch.sigmoid(
            global_margin / self.prompt_temperature
        ).view(-1, 1, 1, 1)
        context_factor = (
            1.0
            - self.global_context_weight
            + self.global_context_weight * global_probability
        )
        return (semantic * context_factor).clamp_(0.0, 1.0)


def _tail_statistics(
    values: Sequence[torch.Tensor],
    shoulder_quantile: float,
    tail_quantile: float,
    eps: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("cannot fit CLIP fusion statistics without values")
    flattened = torch.cat([value.float().reshape(-1) for value in values])
    median = float(torch.quantile(flattened, 0.5).item())
    shoulder = float(torch.quantile(flattened, shoulder_quantile).item())
    tail = float(torch.quantile(flattened, tail_quantile).item())
    return {
        "median": median,
        "shoulder": max(shoulder, median + eps),
        "tail": max(tail, shoulder + eps, median + 2.0 * eps),
    }


def _normalize_view_ids(
    view_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    normalized = view_ids.to(device=device, dtype=torch.long).reshape(-1)
    if normalized.numel() != batch_size:
        raise ValueError("view_ids must contain one value per competition image")
    return normalized


def fuse_anomaly_evidence(
    reconstruction_map: torch.Tensor,
    semantic_map: torch.Tensor,
    reconstruction_stats: dict[str, float],
    semantic_stats: dict[str, float],
    *,
    reconstruction_floor: float,
    semantic_weight: float,
    agreement_weight: float,
    gate_temperature: float,
    max_normalized_score: float,
    eps: float,
) -> torch.Tensor:
    """Fuse calibrated reconstruction and semantic evidence for one image."""

    if reconstruction_map.shape != semantic_map.shape:
        raise ValueError("reconstruction and CLIP maps must have identical shapes")

    def normalized_tail(
        values: torch.Tensor,
        stats: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        median = float(stats["median"])
        shoulder = float(stats["shoulder"])
        tail = float(stats["tail"])
        score = ((values.float() - median) / max(tail - median, eps)).clamp(
            min=0.0,
            max=max_normalized_score,
        )
        gate_scale = max(tail - shoulder, eps) * gate_temperature
        gate = torch.sigmoid((values.float() - shoulder) / gate_scale)
        return score, gate

    reconstruction, _ = normalized_tail(
        reconstruction_map,
        reconstruction_stats,
    )
    semantic, semantic_gate = normalized_tail(semantic_map, semantic_stats)
    gated_reconstruction = reconstruction * (
        reconstruction_floor + (1.0 - reconstruction_floor) * semantic_gate
    )
    agreement = torch.sqrt((reconstruction * semantic).clamp_min(0.0) + eps)
    fused = (
        gated_reconstruction
        + semantic_weight * semantic
        + agreement_weight * agreement
    )
    return fused.to(reconstruction_map.dtype)


class ClipFusion:
    """Validated Train-normal calibration plus frozen zero-shot CLIP branch."""

    def __init__(
        self,
        payload: dict[str, Any],
        semantic_encoder: ClipSemanticEncoder,
    ) -> None:
        self.payload = payload
        self.category_view = payload.get("category_view", {})
        self.semantic_encoder = semantic_encoder

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.payload["metadata"])

    def calibrate(
        self,
        raw_maps: torch.Tensor,
        images: torch.Tensor,
        *,
        categories: Sequence[str],
        view_ids: torch.Tensor,
        valid_view_mask: torch.Tensor | None,
        config: dict[str, Any],
    ) -> torch.Tensor:
        if raw_maps.ndim != 4 or raw_maps.shape[1] != 1:
            raise ValueError("CLIP fusion expects reconstruction maps [B,1,H,W]")
        if images.ndim != 4 or images.shape[0] != raw_maps.shape[0]:
            raise ValueError("CLIP fusion image/map batch mismatch")
        batch_size = int(raw_maps.shape[0])
        if len(categories) != batch_size:
            raise ValueError("categories must contain one value per image")
        view_ids = _normalize_view_ids(view_ids, batch_size, raw_maps.device)
        if valid_view_mask is None:
            valid_view_mask = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=raw_maps.device,
            )
        else:
            valid_view_mask = valid_view_mask.to(
                device=raw_maps.device,
                dtype=torch.bool,
            ).reshape(-1)
        if valid_view_mask.numel() != batch_size:
            raise ValueError("valid_view_mask must contain one value per image")

        semantic = self.semantic_encoder.semantic_map(images, categories, config)
        semantic = F.interpolate(
            semantic.float(),
            size=raw_maps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fusion_config = config["evaluation"]["clip_fusion"]
        calibrated = raw_maps.clone()
        for index, category_value in enumerate(categories):
            if not bool(valid_view_mask[index]):
                continue
            category = str(category_value)
            camera_id = str(int(view_ids[index]))
            stats = self.category_view.get(category, {}).get(camera_id)
            if stats is None:
                raise KeyError(
                    f"CLIP fusion has no Train calibration for "
                    f"category={category!r}, view={camera_id}"
                )
            calibrated[index] = fuse_anomaly_evidence(
                raw_maps[index],
                semantic[index],
                stats["reconstruction"],
                stats["semantic"],
                reconstruction_floor=float(
                    fusion_config.get("reconstruction_floor", 0.35)
                ),
                semantic_weight=float(fusion_config.get("semantic_weight", 0.3)),
                agreement_weight=float(
                    fusion_config.get("agreement_weight", 0.35)
                ),
                gate_temperature=float(
                    fusion_config.get("gate_temperature", 0.5)
                ),
                max_normalized_score=float(
                    fusion_config.get("max_normalized_score", 4.0)
                ),
                eps=float(fusion_config.get("eps", 1e-6)),
            )
        return calibrated


def validate_clip_fusion(
    payload: dict[str, Any],
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != CLIP_FUSION_FORMAT_VERSION:
        raise ValueError("CLIP fusion format_version is incompatible")
    if metadata.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("CLIP fusion/model config fingerprint mismatch")
    if metadata.get("clip_config_sha256") != _clip_config_fingerprint(config):
        raise ValueError("CLIP fusion/evaluation config fingerprint mismatch")
    if metadata.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("CLIP fusion/Dinomaly checkpoint fingerprint mismatch")
    _, expected_weights_identity = _resolve_clip_pretrained(config)
    if metadata.get("clip_weights_identity") != expected_weights_identity:
        raise ValueError("CLIP fusion/OpenCLIP weights identity mismatch")
    if metadata.get("source_split") != "Train" or metadata.get(
        "source_labels"
    ) != "normal_only":
        raise ValueError("CLIP fusion is not marked as Train-normal-only")
    category_view = payload.get("category_view")
    if not isinstance(category_view, dict) or not category_view:
        raise ValueError("CLIP fusion category/view calibration is missing")


def load_clip_fusion(
    path: str | Path,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
) -> ClipFusion:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("CLIP fusion artifact is not a mapping")
    validate_clip_fusion(payload, config, checkpoint_path)
    semantic_encoder = ClipSemanticEncoder(config, device)
    if semantic_encoder.weights_identity != payload["metadata"].get(
        "clip_weights_identity"
    ):
        raise ValueError("CLIP fusion weights identity does not match artifact")
    return ClipFusion(payload, semantic_encoder)


def _loader(dataset, config: dict[str, Any]) -> DataLoader:
    fusion_config = config["evaluation"]["clip_fusion"]
    workers = int(
        fusion_config.get("num_workers", config["evaluation"]["num_workers"])
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(
            fusion_config.get(
                "batch_size",
                config["evaluation"]["batch_size"],
            )
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


def _calibration_subset(
    dataset: CompetitionFolderDataset,
    samples_per_group: int,
) -> Subset:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, view in enumerate(dataset.views):
        grouped[(view.category, int(view.view_id))].append(index)
    selected: list[int] = []
    for indices in grouped.values():
        count = min(samples_per_group, len(indices))
        if count <= 0:
            continue
        if count == len(indices):
            selected.extend(indices)
            continue
        positions = torch.linspace(0, len(indices) - 1, count).round().long()
        selected.extend(indices[int(position)] for position in positions)
    return Subset(dataset, sorted(selected))


@torch.inference_mode()
def fit_clip_fusion(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    categories: Iterable[str],
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    """Fit fusion scales from competition Train normals; never inspect Test_A."""

    fusion_config = config["evaluation"].get("clip_fusion", {})
    if not bool(fusion_config.get("enabled", False)):
        return None
    if config["dataset"].get("type") != "competition_folders":
        raise ValueError("CLIP fusion is restricted to the competition pipe")
    if bool(config["model"].get("multi_view", {}).get("enabled", False)):
        raise ValueError("CLIP fusion v1 requires independent competition views")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    artifact_path = clip_fusion_path(config)
    if artifact_path.is_file() and not force:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("CLIP fusion artifact is not a mapping")
        validate_clip_fusion(payload, config, checkpoint_path)
        return payload

    category_list = sorted(str(category) for category in categories)
    dataset_config = config["dataset"]
    dataset, _ = build_competition_train_dataset(
        train_dir=Path(dataset_config["train_dir"]),
        categories=category_list,
        category_limit=None,
        image_size=int(dataset_config["image_size"]),
        crop_size=int(dataset_config["crop_size"]),
        multi_view_enabled=False,
        num_views=5,
        missing_view_policy="error",
    )
    if not isinstance(dataset, CompetitionFolderDataset):
        raise TypeError("CLIP fusion requires independent competition images")
    calibration_dataset = _calibration_subset(
        dataset,
        int(fusion_config.get("calibration_samples_per_group", 8)),
    )

    device = resolve_device(str(config["runtime"]["device"]))
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
    if checkpoint_payload.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("checkpoint/config fingerprint mismatch while fitting CLIP")
    bundle = build_model(config, device)
    expected_backbone = checkpoint_payload.get("backbone_sha256")
    if expected_backbone is not None and expected_backbone != bundle.backbone_sha256:
        raise ValueError("checkpoint backbone does not match while fitting CLIP")
    load_trainable_state_dict(bundle, checkpoint_payload["model"])
    bundle.model.eval()
    semantic_encoder = ClipSemanticEncoder(config, device)

    reconstruction_values: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(
        list
    )
    semantic_values: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    side = int(dataset_config["crop_size"]) // 14
    layer_weights = config["evaluation"].get("anomaly_map_layer_weights")
    align_corners = bool(
        config["evaluation"].get("anomaly_map_align_corners", True)
    )
    for batch in tqdm(
        _loader(calibration_dataset, config),
        desc="Fitting Train-normal CLIP fusion",
        unit="batch",
    ):
        images = batch["image"].to(
            device,
            non_blocking=bool(config["runtime"]["pin_memory"]),
        )
        categories_batch = [str(value) for value in batch["category"]]
        view_ids = batch["view_id"].tolist()
        with autocast_context(dtype, device):
            encoder_features, decoder_features = bundle.model(images)
            reconstruction = anomaly_map(
                encoder_features,
                decoder_features,
                output_size=side,
                layer_weights=layer_weights,
                align_corners=align_corners,
            )
        semantic = semantic_encoder.semantic_map(
            images,
            categories_batch,
            config,
        )
        semantic = F.interpolate(
            semantic.float(),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        )
        reconstruction = reconstruction.detach().float().cpu()
        semantic = semantic.detach().float().cpu()
        for index, (category, view_id) in enumerate(
            zip(categories_batch, view_ids, strict=True)
        ):
            key = (category, int(view_id))
            reconstruction_values[key].append(reconstruction[index])
            semantic_values[key].append(semantic[index])

    shoulder_quantile = float(fusion_config.get("shoulder_quantile", 0.95))
    tail_quantile = float(fusion_config.get("tail_quantile", 0.995))
    eps = float(fusion_config.get("eps", 1e-6))
    category_view: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    entries: list[dict[str, Any]] = []
    for category, view_id in sorted(reconstruction_values):
        key = (category, view_id)
        reconstruction_stats = _tail_statistics(
            reconstruction_values[key],
            shoulder_quantile,
            tail_quantile,
            eps,
        )
        semantic_stats = _tail_statistics(
            semantic_values[key],
            shoulder_quantile,
            tail_quantile,
            eps,
        )
        entry = {
            "reconstruction": reconstruction_stats,
            "semantic": semantic_stats,
            "samples": len(reconstruction_values[key]),
        }
        category_view[category][str(view_id)] = entry
        entries.append(
            {
                "category": category,
                "view_id": view_id,
                **entry,
            }
        )

    metadata = {
        "format_version": CLIP_FUSION_FORMAT_VERSION,
        "created_at": utc_now(),
        "source_split": "Train",
        "source_labels": "normal_only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_fingerprint": config_fingerprint(config),
        "clip_config_sha256": _clip_config_fingerprint(config),
        "clip_weights_identity": semantic_encoder.weights_identity,
        "clip_model_name": str(fusion_config.get("model_name")),
        "clip_pretrained": str(fusion_config.get("pretrained")),
        "categories": category_list,
        "calibration_samples_per_group": int(
            fusion_config.get("calibration_samples_per_group", 8)
        ),
        "shoulder_quantile": shoulder_quantile,
        "tail_quantile": tail_quantile,
        "entries": entries,
    }
    payload = {
        "metadata": metadata,
        "category_view": dict(category_view),
    }
    atomic_torch_save(artifact_path, payload)
    atomic_write_json(
        artifact_path.with_suffix(".json"),
        {
            **metadata,
            "artifact": str(artifact_path),
            "category_view_entries": len(entries),
        },
    )
    return payload
