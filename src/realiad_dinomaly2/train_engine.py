from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .config import (
    config_fingerprint,
    dump_resolved_config,
)
from .competition_data import build_competition_train_dataset
from .data import build_train_dataset, discover_categories
from .losses import discard_rate_for_step, reconstruction_loss
from .modeling import (
    ModelBundle,
    build_model,
    build_optimizer,
    forward_with_regularization,
    load_trainable_state_dict,
    parameter_summary,
    set_learning_rate,
    trainable_state_dict,
)
from .information_density_model import set_information_density_step
from .runtime import (
    amp_dtype,
    append_jsonl,
    atomic_torch_save,
    atomic_write_json,
    autocast_context,
    environment_summary,
    make_grad_scaler,
    resolve_device,
    setup_logger,
    setup_seed,
    utc_now,
)


UPSTREAM_COMMIT = "1745c613a7079117798fdba42c6664d9f45820ce"
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DistributedContext:
    """Process/device topology used by the training loop."""

    strategy: str
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device_ids: tuple[int, ...] = ()
    backend: str | None = None
    process_group_initialized: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def is_ddp(self) -> bool:
        return self.strategy == "ddp"


def _initialize_distributed_context(config: dict[str, Any]) -> DistributedContext:
    """Initialize torchrun DDP or single-process DataParallel.

    ``effective_batch_size`` is always global. ``world_size`` means worker
    processes for DDP and participating GPUs for DataParallel.
    """

    runtime = config["runtime"]
    requested = str(runtime.get("multi_gpu_strategy", "single")).lower()
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if env_world_size > 1:
        if requested == "data_parallel":
            raise ValueError(
                "runtime.multi_gpu_strategy=data_parallel must be launched "
                "with plain python, not torchrun"
            )
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        configured = torch.device(str(runtime["device"]))
        if configured.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun requested CUDA, but CUDA is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank}, but only "
                    f"{torch.cuda.device_count()} CUDA devices are visible"
                )
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        else:
            device = configured

        configured_backend = str(
            runtime.get("distributed_backend", "auto")
        ).lower()
        if configured_backend == "auto":
            use_nccl = (
                device.type == "cuda"
                and os.name != "nt"
                and dist.is_nccl_available()
            )
            backend = "nccl" if use_nccl else "gloo"
        else:
            backend = configured_backend
        dist.init_process_group(backend=backend, init_method="env://")
        return DistributedContext(
            strategy="ddp",
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=env_world_size,
            device_ids=(local_rank,) if device.type == "cuda" else (),
            backend=backend,
            process_group_initialized=True,
        )

    if requested == "ddp":
        raise RuntimeError(
            "runtime.multi_gpu_strategy=ddp requires torchrun (WORLD_SIZE > 1)"
        )
    configured_device_ids = runtime.get("device_ids")
    use_data_parallel = requested == "data_parallel" or (
        requested == "auto"
        and configured_device_ids is not None
        and len(configured_device_ids) >= 2
    )
    if use_data_parallel:
        device_ids = tuple(
            int(value)
            for value in runtime.get(
                "device_ids", range(torch.cuda.device_count())
            )
        )
        if not torch.cuda.is_available() or len(device_ids) < 2:
            raise RuntimeError(
                "data_parallel requires at least two CUDA devices in "
                "runtime.device_ids"
            )
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("runtime.device_ids must not contain duplicates")
        available = torch.cuda.device_count()
        if any(index < 0 or index >= available for index in device_ids):
            raise RuntimeError(
                f"runtime.device_ids={list(device_ids)}, but only {available} "
                "CUDA devices are visible"
            )
        device = torch.device("cuda", device_ids[0])
        torch.cuda.set_device(device)
        return DistributedContext(
            strategy="data_parallel",
            device=device,
            world_size=len(device_ids),
            device_ids=device_ids,
        )

    if requested not in {"single", "none", "auto"}:
        raise ValueError(f"unsupported multi_gpu_strategy={requested!r}")
    return DistributedContext(
        strategy="single",
        device=resolve_device(str(runtime["device"])),
    )


def _cleanup_distributed_context(context: DistributedContext) -> None:
    if context.process_group_initialized and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(context: DistributedContext) -> None:
    if context.is_ddp and dist.is_initialized():
        dist.barrier()


def _rank_logger(name: str, log_path: Path, is_primary: bool):
    if is_primary:
        return setup_logger(name, log_path)
    logger = logging.getLogger(f"{name}.worker.{os.getpid()}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _build_model_for_context(
    config: dict[str, Any],
    context: DistributedContext,
) -> ModelBundle:
    """Avoid concurrent first-time backbone downloads under torchrun."""

    if not context.is_ddp:
        return build_model(config, context.device)
    bundle: ModelBundle | None = None
    if context.is_primary:
        bundle = build_model(config, context.device)
    _barrier(context)
    if not context.is_primary:
        bundle = build_model(config, context.device)
    _barrier(context)
    assert bundle is not None
    return bundle


class DeterministicIterationBatchSampler(Sampler[list[int]]):
    """Iteration-addressable distributed sampler with exact resume.

    Epoch shuffling and rank partitioning are delegated to
    :class:`DistributedSampler`. ``world_size=1`` preserves the old order.
    """

    def __init__(
        self,
        dataset_size: int,
        micro_batch_size: int,
        total_optimizer_steps: int,
        accumulation_steps: int,
        start_optimizer_step: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"invalid sampler rank/world_size={rank}/{world_size}"
            )
        samples_per_rank = dataset_size // world_size
        if samples_per_rank < micro_batch_size:
            raise ValueError(
                f"dataset has {dataset_size} samples, only {samples_per_rank} "
                f"per rank, but micro batch is {micro_batch_size}"
            )
        self.dataset_size = dataset_size
        self.micro_batch_size = micro_batch_size
        self.steps_per_epoch = samples_per_rank // micro_batch_size
        self.total_micro_steps = total_optimizer_steps * accumulation_steps
        self.start_micro_step = start_optimizer_step * accumulation_steps
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __len__(self) -> int:
        return self.total_micro_steps - self.start_micro_step

    def __iter__(self) -> Iterator[list[int]]:
        micro_step = self.start_micro_step
        cached_epoch = -1
        permutation: list[int] = []
        sampler = DistributedSampler(
            range(self.dataset_size),
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            drop_last=True,
        )
        while micro_step < self.total_micro_steps:
            epoch = micro_step // self.steps_per_epoch
            batch_in_epoch = micro_step % self.steps_per_epoch
            if epoch != cached_epoch:
                sampler.set_epoch(epoch)
                permutation = list(iter(sampler))
                cached_epoch = epoch
            start = batch_in_epoch * self.micro_batch_size
            yield permutation[start : start + self.micro_batch_size]
            micro_step += 1


@dataclass
class BatchChoice:
    micro_batch_size: int
    accumulation_steps: int
    effective_batch_size: int
    trials: list[dict[str, Any]]
    world_size: int = 1

    @property
    def global_micro_batch_size(self) -> int:
        return self.micro_batch_size * self.world_size


def _fixed_batch_choice(
    effective_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> BatchChoice:
    global_micro = micro_batch_size * world_size
    if effective_batch_size % global_micro:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"micro_batch_size={micro_batch_size} * world_size={world_size}"
        )
    return BatchChoice(
        micro_batch_size=micro_batch_size,
        accumulation_steps=effective_batch_size // global_micro,
        effective_batch_size=effective_batch_size,
        trials=[],
        world_size=world_size,
    )


def _trainable_parameters(bundle: ModelBundle) -> list[torch.nn.Parameter]:
    return [
        parameter
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    ]


def _multi_view_settings(config: dict[str, Any]) -> tuple[bool, int, str]:
    settings = dict(config["model"].get("multi_view", {}))
    return (
        bool(settings.get("enabled", False)),
        int(settings.get("num_views", 5)),
        str(settings.get("missing_view_policy", "error")),
    )


def _auxiliary_weights(config: dict[str, Any]) -> dict[str, float] | None:
    weights = dict(
        config["model"].get("generalized", {}).get("auxiliary_weights", {})
    )
    weights.update(
        {
            str(name): float(value)
            for name, value in config["training"]
            .get("multi_view_auxiliary_weights", {})
            .items()
        }
    )
    weights.update(
        {
            str(name): float(value)
            for name, value in config["model"]
            .get("information_density", {})
            .get("auxiliary_weights", {})
            .items()
        }
    )
    return weights or None


def _regularization_weight(config: dict[str, Any]) -> float:
    architecture = str(
        config["model"].get("architecture", "dinomaly2")
    ).lower()
    if architecture in {
        "information_density",
        "information_density_dinomaly2",
        "adaptive_dinomaly2",
    }:
        return float(
            config["training"].get(
                "information_density_regularization_weight", 0.0
            )
        )
    return float(
        config["training"].get("generalized_regularization_weight", 0.0)
    )


def _batch_model_inputs(
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    enabled, _, _ = _multi_view_settings(config)
    image_key = "images" if enabled else "image"
    images = batch[image_key].to(
        device,
        non_blocking=bool(config["runtime"]["pin_memory"]),
    )
    if not enabled:
        return images, None, None
    view_ids = batch["view_ids"].to(device, non_blocking=True)
    valid_view_mask = batch["valid_view_mask"].to(device, non_blocking=True)
    return images, view_ids, valid_view_mask


def _optimizer_group_gradient_norms(optimizer) -> dict[str, float]:
    """Return pre-clipping L2 norms, grouped for spike diagnosis."""
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if norms:
            total = torch.stack(norms).norm(2)
            value = float(total.detach().cpu())
        else:
            value = 0.0
        result[str(group.get("group_name", f"group_{index}"))] = value
    return result


def _gradient_step_should_be_skipped(
    grad_norm: float,
    gradient_guard: dict[str, Any],
) -> bool:
    if not math.isfinite(grad_norm):
        return True
    threshold = gradient_guard.get("skip_step_norm")
    return threshold is not None and grad_norm > float(threshold)


def _run_probe(
    bundle: ModelBundle,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None,
    candidate: int,
) -> dict[str, Any]:
    optimizer = build_optimizer(bundle, config)
    scaler = make_grad_scaler(dtype, device)
    crop_size = int(config["dataset"]["crop_size"])
    multi_view_enabled, num_views, _ = _multi_view_settings(config)
    synthetic_shape = (
        (candidate, num_views, 3, crop_size, crop_size)
        if multi_view_enabled
        else (candidate, 3, crop_size, crop_size)
    )
    synthetic = torch.randn(
        *synthetic_shape,
        device=device,
    )
    view_ids = (
        torch.arange(num_views, device=device).unsqueeze(0).expand(candidate, -1)
        if multi_view_enabled
        else None
    )
    valid_view_mask = (
        torch.ones((candidate, num_views), dtype=torch.bool, device=device)
        if multi_view_enabled
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    regularization_weight = _regularization_weight(config)
    auxiliary_weights = _auxiliary_weights(config)
    with autocast_context(dtype, device):
        (
            encoder_features,
            decoder_features,
            regularizer,
            auxiliary,
        ) = forward_with_regularization(
            bundle.model,
            synthetic,
            weights=auxiliary_weights,
            view_ids=view_ids,
            valid_view_mask=valid_view_mask,
        )
        reconstruction = reconstruction_loss(
            encoder_features,
            decoder_features,
            discard_rate=0.0,
            loose_loss=bool(config["model"]["loose_loss"]),
            valid_view_mask=valid_view_mask,
        )
        loss = (
            reconstruction + regularization_weight * regularizer
            if regularization_weight != 0.0
            else reconstruction
        )
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    clip_grad_norm_(
        _trainable_parameters(bundle),
        max_norm=float(config["training"]["gradient_clip_norm"]),
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    result = {
        "batch_size": candidate,
        "batch_unit": "objects" if multi_view_enabled else "views",
        "equivalent_view_batch_size": candidate * num_views,
        "status": "ok",
        "loss": float(loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction.detach().cpu()),
        "regularization_loss": float(regularizer.detach().cpu()),
        "auxiliary_losses": {
            name: float(value.detach().mean().cpu())
            for name, value in auxiliary.items()
        },
        "seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }

    del (
        encoder_features,
        decoder_features,
        reconstruction,
        regularizer,
        auxiliary,
        loss,
        synthetic,
        view_ids,
        valid_view_mask,
        optimizer,
        scaler,
    )
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.init_weights()
    torch.cuda.empty_cache()
    return result


def choose_batch_size(
    bundle: ModelBundle,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None,
    logger,
    output_dir: Path,
    world_size: int = 1,
) -> BatchChoice:
    training = config["training"]
    multi_view_enabled, num_views, _ = _multi_view_settings(config)
    effective = int(training["effective_batch_size"])
    requested = training["micro_batch_size"]
    if requested != "auto":
        micro = int(requested)
        return _fixed_batch_choice(
            effective_batch_size=effective,
            micro_batch_size=micro,
            world_size=world_size,
        )
    if device.type != "cuda":
        raise ValueError("micro_batch_size=auto 目前仅支持 CUDA")

    original_trainable_state = {
        module_name: {
            key: value.detach().cpu().clone()
            for key, value in module_state.items()
        }
        for module_name, module_state in trainable_state_dict(bundle).items()
    }
    original_rng_state = _rng_state()

    def restore_probe_state() -> None:
        bundle.model.zero_grad(set_to_none=True)
        load_trainable_state_dict(bundle, original_trainable_state)
        _restore_rng_state(original_rng_state)
        torch.cuda.empty_cache()

    candidates = sorted(
        {
            int(value)
            for value in training["batch_candidates"]
            if int(value) > 0
            and effective % (int(value) * world_size) == 0
        },
        reverse=True,
    )
    if not candidates:
        raise ValueError("batch_candidates 中没有 effective batch 的正整数因子")

    headroom = int(training["memory_headroom_mb"]) * 1024 * 1024
    index = 0 if device.index is None else device.index
    torch.cuda.empty_cache()
    free_before, _ = torch.cuda.mem_get_info(index)
    reserved_before = torch.cuda.memory_reserved(device)
    allowed_reserved = free_before + reserved_before - headroom
    trials: list[dict[str, Any]] = []

    logger.info(
        "开始显存探测：candidates=%s, 预留=%d MiB",
        candidates,
        int(training["memory_headroom_mb"]),
    )
    for candidate in candidates:
        try:
            trial = _run_probe(
                bundle=bundle,
                config=config,
                device=device,
                dtype=dtype,
                candidate=candidate,
            )
            within_budget = trial["peak_reserved_bytes"] <= allowed_reserved
            trial["within_budget"] = within_budget
            trials.append(trial)
            logger.info(
                "显存探测 batch=%d：peak reserved %.1f MiB，预算内=%s",
                candidate,
                trial["peak_reserved_bytes"] / (1024**2),
                within_budget,
            )
            if within_budget:
                choice = BatchChoice(
                    micro_batch_size=candidate,
                    accumulation_steps=effective // (candidate * world_size),
                    effective_batch_size=effective,
                    trials=trials,
                    world_size=world_size,
                )
                atomic_write_json(
                    output_dir / "batch_tuning.json",
                    {
                        "selected_micro_batch_size": choice.micro_batch_size,
                        "accumulation_steps": choice.accumulation_steps,
                        "effective_batch_size": choice.effective_batch_size,
                        "batch_unit": (
                            "objects" if multi_view_enabled else "views"
                        ),
                        "effective_view_batch_size": (
                            choice.effective_batch_size
                            * (num_views if multi_view_enabled else 1)
                        ),
                        "world_size": choice.world_size,
                        "global_micro_batch_size": (
                            choice.global_micro_batch_size
                        ),
                        "allowed_reserved_bytes": allowed_reserved,
                        "trials": trials,
                        "updated_at": utc_now(),
                    },
                )
                restore_probe_state()
                return choice
        except torch.OutOfMemoryError as exc:
            trials.append(
                {
                    "batch_size": candidate,
                    "status": "oom",
                    "error": str(exc),
                }
            )
            logger.warning("显存探测 batch=%d：OOM", candidate)
            bundle.model.zero_grad(set_to_none=True)
            bundle.model.init_weights()
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                restore_probe_state()
                raise
            trials.append(
                {
                    "batch_size": candidate,
                    "status": "oom",
                    "error": str(exc),
                }
            )
            logger.warning("显存探测 batch=%d：OOM", candidate)
            bundle.model.zero_grad(set_to_none=True)
            bundle.model.init_weights()
            torch.cuda.empty_cache()
        except BaseException:
            restore_probe_state()
            raise

    restore_probe_state()
    atomic_write_json(
        output_dir / "batch_tuning.json",
        {
            "selected_micro_batch_size": None,
            "trials": trials,
            "updated_at": utc_now(),
        },
    )
    raise RuntimeError("所有 batch_candidates 均 OOM 或没有保留足够显存")


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device so that optimizer tensors
    # are immediately usable. PyTorch's CPU RNG setter, however, only accepts a
    # CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [generator_state.cpu() for generator_state in state["cuda"]]
        )


def save_checkpoint(
    path: Path,
    bundle: ModelBundle,
    optimizer,
    scaler,
    config: dict[str, Any],
    completed_steps: int,
    batch_choice: BatchChoice,
    rng_states: list[dict[str, Any]] | None = None,
) -> None:
    multi_view_enabled, num_views, _ = _multi_view_settings(config)
    atomic_torch_save(
        path,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "backbone_sha256": bundle.backbone_sha256,
            "created_at": utc_now(),
            "config_fingerprint": config_fingerprint(config),
            "completed_steps": completed_steps,
            "micro_batch_size": batch_choice.micro_batch_size,
            "accumulation_steps": batch_choice.accumulation_steps,
            "effective_batch_size": batch_choice.effective_batch_size,
            "batch_unit": "objects" if multi_view_enabled else "views",
            "effective_view_batch_size": (
                batch_choice.effective_batch_size * num_views
            ),
            "global_micro_batch_size": batch_choice.global_micro_batch_size,
            "world_size": batch_choice.world_size,
            "model": trainable_state_dict(bundle),
            "optimizer": optimizer.state_dict(),
            "scheduler": {
                "completed_steps": completed_steps,
                "config": dict(config["training"].get("scheduler", {})),
            },
            "scaler": scaler.state_dict(),
            "rng_state": _rng_state(),
            "rng_states": rng_states,
        },
    )


def _gather_rng_states(
    context: DistributedContext,
) -> list[dict[str, Any]] | None:
    local_state = _rng_state()
    if not context.is_ddp:
        return [local_state]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * context.world_size if context.is_primary else None
    )
    dist.gather_object(local_state, gathered, dst=0)
    if not context.is_primary:
        return None
    assert gathered is not None
    return [state for state in gathered if state is not None]


def _save_checkpoint_for_all_ranks(
    path: Path,
    bundle: ModelBundle,
    optimizer,
    scaler,
    config: dict[str, Any],
    completed_steps: int,
    batch_choice: BatchChoice,
    context: DistributedContext,
) -> None:
    rng_states = _gather_rng_states(context)
    if context.is_primary:
        save_checkpoint(
            path,
            bundle=bundle,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            completed_steps=completed_steps,
            batch_choice=batch_choice,
            rng_states=rng_states,
        )
    _barrier(context)


def _resolve_resume(
    output_dir: Path,
    resume: str | None,
    auto_resume: bool,
) -> Path | None:
    if resume is None or resume == "auto":
        candidate = output_dir / "checkpoints" / "last.pt"
        return candidate if auto_resume and candidate.is_file() else None
    if resume == "never":
        checkpoint_dir = output_dir / "checkpoints"
        stale = [
            path
            for path in (
                checkpoint_dir / "last.pt",
                checkpoint_dir / "final_model.pt",
            )
            if path.is_file()
        ]
        if stale:
            rendered = ", ".join(str(path) for path in stale)
            raise FileExistsError(
                "--resume never 表示新实验，但输出目录已有 checkpoint："
                f"{rendered}。为防止评估误读旧模型，请改用新的 "
                "experiment.output_dir，或先手动归档这些文件。"
            )
        return None
    path = Path(resume).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint 不存在：{path}")
    return path


def _dataloader(
    dataset,
    config: dict[str, Any],
    batch_choice: BatchChoice,
    completed_steps: int,
    context: DistributedContext,
) -> DataLoader:
    runtime = config["runtime"]
    training = config["training"]
    is_ddp = context.is_ddp
    sampler = DeterministicIterationBatchSampler(
        dataset_size=len(dataset),
        micro_batch_size=(
            batch_choice.micro_batch_size
            if is_ddp
            else batch_choice.global_micro_batch_size
        ),
        total_optimizer_steps=int(training["total_steps"]),
        accumulation_steps=batch_choice.accumulation_steps,
        start_optimizer_step=completed_steps,
        seed=int(config["experiment"]["seed"]),
        rank=context.rank if is_ddp else 0,
        world_size=context.world_size if is_ddp else 1,
    )
    workers = int(runtime["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(runtime["pin_memory"]),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(runtime["persistent_workers"])
        kwargs["prefetch_factor"] = int(runtime["prefetch_factor"])
    return DataLoader(**kwargs)


def _choose_batch_for_context(
    bundle: ModelBundle,
    config: dict[str, Any],
    context: DistributedContext,
    dtype: torch.dtype | None,
    logger,
    output_dir: Path,
) -> BatchChoice:
    if not context.is_ddp:
        return choose_batch_size(
            bundle=bundle,
            config=config,
            device=context.device,
            dtype=dtype,
            logger=logger,
            output_dir=output_dir,
            world_size=context.world_size,
        )

    payload: list[Any] = [None]
    original_error: BaseException | None = None
    if context.is_primary:
        try:
            payload[0] = (
                "ok",
                choose_batch_size(
                    bundle=bundle,
                    config=config,
                    device=context.device,
                    dtype=dtype,
                    logger=logger,
                    output_dir=output_dir,
                    world_size=context.world_size,
                ),
            )
        except BaseException as exc:
            original_error = exc
            payload[0] = ("error", repr(exc))
    dist.broadcast_object_list(payload, src=0)
    status, value = payload[0]
    if status != "ok":
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank 0 batch-size selection failed: {value}")
    return value


def _wrap_parallel_model(
    bundle: ModelBundle,
    context: DistributedContext,
) -> None:
    if context.strategy == "data_parallel":
        bundle.model = torch.nn.DataParallel(
            bundle.model,
            device_ids=list(context.device_ids),
            output_device=context.device_ids[0],
        )
    elif context.is_ddp:
        kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if context.device.type == "cuda":
            kwargs.update(
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        bundle.model = DistributedDataParallel(bundle.model, **kwargs)


def _distributed_mean(value: float, context: DistributedContext) -> float:
    if not context.is_ddp:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= context.world_size
    return float(tensor.cpu())


def _distributed_max(value: float, context: DistributedContext) -> float:
    if not context.is_ddp:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def _assert_finite_on_all_ranks(
    loss: torch.Tensor,
    context: DistributedContext,
    step_index: int,
) -> None:
    finite = bool(torch.isfinite(loss))
    if context.is_ddp:
        flag = torch.tensor(
            1 if finite else 0,
            dtype=torch.int32,
            device=context.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        finite = bool(flag.item())
    if not finite:
        raise FloatingPointError(
            f"step={step_index} produced a non-finite loss on at least one rank"
        )


def _train_impl(
    config: dict[str, Any],
    resume: str | None,
    context: DistributedContext,
) -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    if context.is_primary:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _barrier(context)
    resume_path = _resolve_resume(
        output_dir,
        resume,
        bool(config["training"]["auto_resume"]),
    )
    logger = _rank_logger(
        "dinomaly2_train", log_dir / "train.log", context.is_primary
    )
    progress_path = log_dir / "progress.jsonl"
    state_path = output_dir / "run_state.json"
    if context.is_primary:
        dump_resolved_config(config, output_dir / "resolved_config.yaml")

    device = context.device
    setup_seed(
        int(config["experiment"]["seed"]) + context.rank,
        bool(config["runtime"]["deterministic"]),
    )
    dtype = amp_dtype(config, device)
    environment = environment_summary(device)
    environment["parallel"] = {
        "strategy": context.strategy,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "device_ids": list(context.device_ids),
        "backend": context.backend,
    }
    logger.info("环境：%s", environment)

    dataset_config = config["dataset"]
    dataset_type = str(dataset_config.get("type", "realiad_variety"))
    multi_view_enabled, num_views, missing_view_policy = _multi_view_settings(config)
    if dataset_type == "competition_folders":
        image_dir = Path(dataset_config["train_dir"])
        dataset, competition_manifest = build_competition_train_dataset(
            train_dir=image_dir,
            categories=dataset_config["categories"],
            category_limit=dataset_config.get("category_limit"),
            image_size=int(dataset_config["image_size"]),
            crop_size=int(dataset_config["crop_size"]),
            multi_view_enabled=multi_view_enabled,
            num_views=num_views,
            missing_view_policy=missing_view_policy,
        )
        categories = list(competition_manifest.categories)
    else:
        json_dir = Path(dataset_config["json_dir"])
        raw_image_dir = Path(dataset_config["image_dir"])
        configured_train_dir = dataset_config.get("train_image_dir")
        image_dir = (
            Path(configured_train_dir)
            if configured_train_dir
            else raw_image_dir
        )
        if configured_train_dir and not image_dir.is_dir():
            message = (
                "Configured dataset.train_image_dir does not exist: "
                f"{image_dir}. Run prepare_cache.ps1 first, or use "
                "the matching non-cached config (for example "
                "configs/rtx3060ti_strict_upstream.yaml) to train from raw "
                "images."
            )
            logger.error(message)
            if context.is_primary:
                atomic_write_json(
                    state_path,
                    {
                        "status": "failed",
                        "updated_at": utc_now(),
                        "experiment": config["experiment"]["name"],
                        "config": config["_meta"]["config_path"],
                        "output_dir": str(output_dir),
                        "completed_steps": 0,
                        "total_steps": int(config["training"]["total_steps"]),
                        "last_error": message,
                        "next_action": (
                            "Run prepare_cache.ps1 to completion, then restart "
                            "training; or select the raw-image config."
                        ),
                    },
                )
            raise FileNotFoundError(message)
        categories = discover_categories(
            json_dir,
            requested=dataset_config["categories"],
            limit=dataset_config.get("category_limit"),
        )
        dataset = build_train_dataset(
            json_dir=json_dir,
            image_dir=image_dir,
            categories=categories,
            image_size=int(dataset_config["image_size"]),
            crop_size=int(dataset_config["crop_size"]),
            multi_view_enabled=multi_view_enabled,
            num_views=num_views,
            missing_view_policy=missing_view_policy,
        )
    logger.info(
        "训练 manifest：%d 类，%d 个%s（等效 %d 张正常视图）",
        len(categories),
        len(dataset),
        "对象" if multi_view_enabled else "视图",
        len(dataset) * (num_views if multi_view_enabled else 1),
    )

    state: dict[str, Any] = {
        "status": "initializing",
        "updated_at": utc_now(),
        "experiment": config["experiment"]["name"],
        "config": config["_meta"]["config_path"],
        "output_dir": str(output_dir),
        "train_image_dir": str(image_dir),
        "completed_steps": 0,
        "total_steps": int(config["training"]["total_steps"]),
        "next_action": "构建 Dinomaly2 并进行显存探测",
        "environment": environment,
    }
    if context.is_primary:
        atomic_write_json(state_path, state)

    bundle = _build_model_for_context(config, context)
    logger.info("模型参数：%s", parameter_summary(bundle))
    checkpoint: dict[str, Any] | None = None
    completed_steps = 0
    if resume_path is not None:
        logger.info("加载断点：%s", resume_path)
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint format_version 不兼容")
        expected = config_fingerprint(config)
        if checkpoint.get("config_fingerprint") != expected:
            raise ValueError(
                "checkpoint 与当前模型/数据/训练语义配置不一致；"
                "请使用原配置或显式 --resume never 开新实验"
            )
        load_trainable_state_dict(bundle, checkpoint["model"])
        completed_steps = int(checkpoint["completed_steps"])
        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is not None:
            if int(scheduler_state.get("completed_steps", -1)) != completed_steps:
                raise ValueError("checkpoint scheduler step is inconsistent")
            if dict(scheduler_state.get("config", {})) != dict(
                config["training"].get("scheduler", {})
            ):
                raise ValueError("checkpoint scheduler config is inconsistent")
        checkpoint_world_size = int(checkpoint.get("world_size", 1))
        if checkpoint_world_size != context.world_size:
            raise ValueError(
                "checkpoint world_size does not match the current run: "
                f"{checkpoint_world_size} != {context.world_size}"
            )
        batch_choice = BatchChoice(
            micro_batch_size=int(checkpoint["micro_batch_size"]),
            accumulation_steps=int(checkpoint["accumulation_steps"]),
            effective_batch_size=int(checkpoint["effective_batch_size"]),
            trials=[],
            world_size=checkpoint_world_size,
        )
        stored_global_micro = int(
            checkpoint.get(
                "global_micro_batch_size",
                batch_choice.global_micro_batch_size,
            )
        )
        if stored_global_micro != batch_choice.global_micro_batch_size:
            raise ValueError(
                "checkpoint global_micro_batch_size is inconsistent with "
                "micro_batch_size * world_size"
            )
        expected_choice = _fixed_batch_choice(
            effective_batch_size=int(config["training"]["effective_batch_size"]),
            micro_batch_size=batch_choice.micro_batch_size,
            world_size=context.world_size,
        )
        if (
            expected_choice.accumulation_steps
            != batch_choice.accumulation_steps
        ):
            raise ValueError(
                "checkpoint accumulation_steps is inconsistent with the "
                "current global batch configuration"
            )
    else:
        batch_choice = _choose_batch_for_context(
            bundle=bundle,
            config=config,
            context=context,
            dtype=dtype,
            logger=logger,
            output_dir=output_dir,
        )

    _wrap_parallel_model(bundle, context)
    optimizer = build_optimizer(bundle, config)
    scaler = make_grad_scaler(dtype, device)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        rank_states = checkpoint.get("rng_states")
        if rank_states and context.rank < len(rank_states):
            _restore_rng_state(rank_states[context.rank])
        else:
            _restore_rng_state(checkpoint["rng_state"])

    total_steps = int(config["training"]["total_steps"])
    if completed_steps > total_steps:
        raise ValueError(
            f"checkpoint 已完成 {completed_steps} 步，超过配置 {total_steps}"
        )
    logger.info(
        "运行 batch（单位=%s）：per-GPU micro=%d, global micro=%d, "
        "accumulate=%d, effective=%d, equivalent views=%d；从 step=%d 开始",
        "objects" if multi_view_enabled else "views",
        batch_choice.micro_batch_size,
        batch_choice.global_micro_batch_size,
        batch_choice.accumulation_steps,
        batch_choice.effective_batch_size,
        batch_choice.effective_batch_size * (
            num_views if multi_view_enabled else 1
        ),
        completed_steps,
    )
    protocol_notes: list[str] = []
    if (
        batch_choice.accumulation_steps > 1
        and bool(config["model"]["loose_loss"])
    ):
        note = (
            "micro-batch 小于 effective batch 时，Loose Loss 的 top-k 阈值"
            "按每个 micro-batch 计算；梯度累积保持有效 batch，但该阈值"
            "语义是 8GB fallback 近似，不是 batch-16 严格等价。"
        )
        protocol_notes.append(note)
        logger.warning(note)
    state.update(
        {
            "status": "training",
            "updated_at": utc_now(),
            "completed_steps": completed_steps,
            "micro_batch_size": batch_choice.micro_batch_size,
            "global_micro_batch_size": batch_choice.global_micro_batch_size,
            "world_size": context.world_size,
            "multi_gpu_strategy": context.strategy,
            "accumulation_steps": batch_choice.accumulation_steps,
            "effective_batch_size": batch_choice.effective_batch_size,
            "batch_unit": "objects" if multi_view_enabled else "views",
            "effective_view_batch_size": (
                batch_choice.effective_batch_size
                * (num_views if multi_view_enabled else 1)
            ),
            "next_action": "继续训练；中断后重新运行 train.ps1 将自动续跑",
            "protocol_notes": protocol_notes,
        }
    )
    if context.is_primary:
        atomic_write_json(state_path, state)

    if completed_steps == total_steps:
        logger.info("checkpoint 已达到 total_steps，直接写出最终权重")
    else:
        loader = _dataloader(
            dataset,
            config,
            batch_choice,
            completed_steps,
            context,
        )
        iterator = iter(loader)
        bundle.model.train()
        trainable_parameters = _trainable_parameters(bundle)
        log_every = int(config["training"]["log_every"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        regularization_weight = _regularization_weight(config)
        auxiliary_weights = _auxiliary_weights(config)
        started = time.perf_counter()
        interval_started = started
        consecutive_skipped_steps = 0

        try:
            for step_index in range(completed_steps, total_steps):
                set_information_density_step(bundle.model, step_index)
                scheduler_config = dict(
                    config["training"].get("scheduler", {})
                )
                learning_rates = set_learning_rate(
                    optimizer=optimizer,
                    completed_steps=step_index,
                    total_steps=total_steps,
                    warmup_steps=int(
                        scheduler_config.get(
                            "warmup_steps",
                            config["training"].get("warmup_steps", 0),
                        )
                    ),
                    final_ratio=float(
                        scheduler_config.get(
                            "min_lr_ratio",
                            config["training"].get("final_lr_ratio", 1.0),
                        )
                    ),
                    step_offset=int(
                        scheduler_config.get(
                            "step_offset",
                            config["training"].get("lr_step_offset", 1),
                        )
                    ),
                    scheduler_config=scheduler_config,
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                accumulated_reconstruction = 0.0
                accumulated_regularization = 0.0
                accumulated_auxiliary: dict[str, float] = {}
                discard_rate = discard_rate_for_step(
                    completed_steps=step_index,
                    warmup_steps=int(
                        config["training"]["loose_loss_warmup_steps"]
                    ),
                    final_rate=float(
                        config["training"]["loose_loss_final_discard"]
                    ),
                )

                for accumulation_index in range(
                    batch_choice.accumulation_steps
                ):
                    batch = next(iterator)
                    images, view_ids, valid_view_mask = _batch_model_inputs(
                        batch,
                        config,
                        device,
                    )
                    synchronize = (
                        not context.is_ddp
                        or accumulation_index
                        == batch_choice.accumulation_steps - 1
                    )
                    synchronization_context = (
                        nullcontext()
                        if synchronize
                        else bundle.model.no_sync()
                    )
                    with synchronization_context:
                        with autocast_context(dtype, device):
                            (
                                encoder_features,
                                decoder_features,
                                regularizer,
                                auxiliary,
                            ) = forward_with_regularization(
                                bundle.model,
                                images,
                                weights=auxiliary_weights,
                                view_ids=view_ids,
                                valid_view_mask=valid_view_mask,
                            )
                            reconstruction = reconstruction_loss(
                                encoder_features,
                                decoder_features,
                                discard_rate=discard_rate,
                                loose_loss=bool(config["model"]["loose_loss"]),
                                valid_view_mask=valid_view_mask,
                            )
                            raw_loss = (
                                reconstruction
                                + regularization_weight * regularizer
                                if regularization_weight != 0.0
                                else reconstruction
                            )
                            loss = raw_loss / batch_choice.accumulation_steps
                        _assert_finite_on_all_ranks(
                            raw_loss, context, step_index
                        )
                        scaler.scale(loss).backward()
                    accumulated_loss += float(raw_loss.detach().cpu())
                    accumulated_reconstruction += float(
                        reconstruction.detach().cpu()
                    )
                    accumulated_regularization += float(
                        regularizer.detach().cpu()
                    )
                    for name, value in auxiliary.items():
                        accumulated_auxiliary[name] = (
                            accumulated_auxiliary.get(name, 0.0)
                            + float(value.detach().mean().cpu())
                        )

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradient_clip_norm = float(
                    config["training"]["gradient_clip_norm"]
                )
                grad_norm = clip_grad_norm_(
                    trainable_parameters,
                    max_norm=gradient_clip_norm,
                    error_if_nonfinite=False,
                )
                grad_norm_value = float(
                    grad_norm.detach().cpu()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                )
                grad_norm_value = _distributed_max(grad_norm_value, context)
                gradient_guard = dict(
                    config["training"].get("gradient_guard", {})
                )
                if not math.isfinite(grad_norm_value) and bool(
                    gradient_guard.get("fail_on_nonfinite", True)
                ):
                    raise FloatingPointError(
                        f"step={step_index} produced non-finite gradients"
                    )
                skip_optimizer_step = _gradient_step_should_be_skipped(
                    grad_norm_value,
                    gradient_guard,
                )
                next_completed_steps = step_index + 1
                scheduled_log = (
                    next_completed_steps == 1
                    or next_completed_steps % log_every == 0
                    or next_completed_steps == total_steps
                )
                gradient_group_norms = (
                    _optimizer_group_gradient_norms(optimizer)
                    if scheduled_log or skip_optimizer_step
                    else {}
                )
                if skip_optimizer_step:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    consecutive_skipped_steps += 1
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    consecutive_skipped_steps = 0
                completed_steps = step_index + 1

                should_log = scheduled_log or skip_optimizer_step
                if should_log:
                    mean_loss = _distributed_mean(
                        accumulated_loss
                        / batch_choice.accumulation_steps,
                        context,
                    )
                    mean_reconstruction = _distributed_mean(
                        accumulated_reconstruction
                        / batch_choice.accumulation_steps,
                        context,
                    )
                    mean_regularization = _distributed_mean(
                        accumulated_regularization
                        / batch_choice.accumulation_steps,
                        context,
                    )
                    mean_auxiliary = {
                        name: _distributed_mean(
                            value / batch_choice.accumulation_steps,
                            context,
                        )
                        for name, value in sorted(
                            accumulated_auxiliary.items()
                        )
                    }
                    now = time.perf_counter()
                    elapsed = now - started
                    mean_step_seconds = elapsed / max(
                        1,
                        completed_steps
                        - int(checkpoint["completed_steps"] if checkpoint else 0),
                    )
                    eta_seconds = mean_step_seconds * (
                        total_steps - completed_steps
                    )
                    payload = {
                        "timestamp": utc_now(),
                        "event": "train_step",
                        "step": completed_steps,
                        "total_steps": total_steps,
                        "loss": mean_loss,
                        "reconstruction_loss": mean_reconstruction,
                        "regularization_loss": mean_regularization,
                        "generalized_regularization_weight": (
                            regularization_weight
                        ),
                        "auxiliary_losses": mean_auxiliary,
                        # clip_grad_norm_ returns the norm before clipping.
                        "grad_norm": grad_norm_value,
                        "grad_norm_pre_clip": grad_norm_value,
                        "gradient_clip_norm": gradient_clip_norm,
                        "gradient_clip_scale": (
                            min(1.0, gradient_clip_norm / grad_norm_value)
                            if math.isfinite(grad_norm_value)
                            and grad_norm_value > 0
                            else 0.0
                        ),
                        "gradient_group_norms": gradient_group_norms,
                        "optimizer_step_skipped": skip_optimizer_step,
                        "consecutive_skipped_steps": (
                            consecutive_skipped_steps
                        ),
                        "learning_rates": learning_rates,
                        "discard_rate": discard_rate,
                        "micro_batch_size": batch_choice.micro_batch_size,
                        "global_micro_batch_size": (
                            batch_choice.global_micro_batch_size
                        ),
                        "batch_unit": (
                            "objects" if multi_view_enabled else "views"
                        ),
                        "effective_view_batch_size": (
                            batch_choice.effective_batch_size
                            * (num_views if multi_view_enabled else 1)
                        ),
                        "world_size": context.world_size,
                        "accumulation_steps": batch_choice.accumulation_steps,
                        "eta_seconds": eta_seconds,
                    }
                    if device.type == "cuda":
                        payload["gpu_allocated_bytes"] = (
                            torch.cuda.memory_allocated(device)
                        )
                        payload["gpu_reserved_bytes"] = (
                            torch.cuda.memory_reserved(device)
                        )
                    if context.is_primary:
                        append_jsonl(progress_path, payload)
                    logger.info(
                        "step %d/%d | loss %.6f | lr %.3e | grad_preclip %.4f "
                        "| clipped_to %.4f | skipped %s | ETA %.1fh",
                        completed_steps,
                        total_steps,
                        payload["loss"],
                        learning_rates[-1],
                        payload["grad_norm"],
                        gradient_clip_norm,
                        skip_optimizer_step,
                        eta_seconds / 3600,
                    )
                    state.update(
                        {
                            "status": "training",
                            "updated_at": utc_now(),
                            "completed_steps": completed_steps,
                            "last_loss": payload["loss"],
                            "last_reconstruction_loss": mean_reconstruction,
                            "last_regularization_loss": mean_regularization,
                            "last_auxiliary_losses": mean_auxiliary,
                            "eta_seconds": eta_seconds,
                            "next_action": (
                                "继续训练；中断后重新运行 train.ps1 将自动续跑"
                            ),
                        }
                    )
                    if context.is_primary:
                        atomic_write_json(state_path, state)
                    interval_started = now

                max_consecutive_skips = int(
                    gradient_guard.get("max_consecutive_skips", 3)
                )
                if (
                    skip_optimizer_step
                    and max_consecutive_skips > 0
                    and consecutive_skipped_steps >= max_consecutive_skips
                ):
                    raise FloatingPointError(
                        "gradient guard stopped training after "
                        f"{consecutive_skipped_steps} consecutive skipped "
                        f"optimizer steps at step={completed_steps}; "
                        f"last pre-clip norm={grad_norm_value:.6g}"
                    )

                if (
                    completed_steps % checkpoint_every == 0
                    or completed_steps == total_steps
                ):
                    logger.info("保存断点 step=%d", completed_steps)
                    _save_checkpoint_for_all_ranks(
                        checkpoint_dir / "last.pt",
                        bundle=bundle,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        completed_steps=completed_steps,
                        batch_choice=batch_choice,
                        context=context,
                    )
        except KeyboardInterrupt:
            logger.warning("收到中断，正在保存可续跑断点")
            if context.is_primary:
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    bundle=bundle,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    completed_steps=completed_steps,
                    batch_choice=batch_choice,
                )
            state.update(
                {
                    "status": "interrupted",
                    "updated_at": utc_now(),
                    "completed_steps": completed_steps,
                    "next_action": "重新运行 train.ps1 自动续跑",
                }
            )
            if context.is_primary:
                atomic_write_json(state_path, state)
            raise
        except BaseException as exc:
            state.update(
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "completed_steps": completed_steps,
                    "last_error": repr(exc),
                    "next_action": "读取 train.log，修复后用 last.pt 续跑",
                }
            )
            if context.is_primary:
                atomic_write_json(state_path, state)
            raise

    final_path = checkpoint_dir / "final_model.pt"
    if context.is_primary:
        atomic_torch_save(
            final_path,
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "upstream_commit": UPSTREAM_COMMIT,
                "backbone_sha256": bundle.backbone_sha256,
                "created_at": utc_now(),
                "config_fingerprint": config_fingerprint(config),
                "completed_steps": total_steps,
                "scheduler": {
                    "completed_steps": total_steps,
                    "config": dict(config["training"].get("scheduler", {})),
                },
                "model": trainable_state_dict(bundle),
                "backbone": bundle.backbone_name,
                "world_size": context.world_size,
                "effective_batch_size": batch_choice.effective_batch_size,
                "batch_unit": "objects" if multi_view_enabled else "views",
                "effective_view_batch_size": (
                    batch_choice.effective_batch_size
                    * (num_views if multi_view_enabled else 1)
                ),
                "global_micro_batch_size": (
                    batch_choice.global_micro_batch_size
                ),
            },
        )
    _barrier(context)
    state.update(
        {
            "status": "trained",
            "updated_at": utc_now(),
            "completed_steps": total_steps,
            "checkpoint": str(final_path),
            "next_action": "运行 evaluate.ps1 计算论文七项指标",
        }
    )
    if context.is_primary:
        atomic_write_json(state_path, state)
        append_jsonl(
            progress_path,
            {
                "timestamp": utc_now(),
                "event": "training_complete",
                "step": total_steps,
                "checkpoint": str(final_path),
            },
        )
    logger.info("训练完成：%s", final_path)
    return state


def train(config: dict[str, Any], resume: str | None = "auto") -> dict[str, Any]:
    """Train with automatic torchrun/DDP or Windows DataParallel setup."""

    context = _initialize_distributed_context(config)
    try:
        return _train_impl(config, resume, context)
    finally:
        _cleanup_distributed_context(context)


def _probe_batch_impl(
    config: dict[str, Any],
    context: DistributedContext,
) -> dict[str, Any]:
    """Run the real forward/backward/optimizer memory probe without training."""
    output_dir = Path(config["experiment"]["output_dir"])
    logger = _rank_logger(
        "dinomaly2_batch_probe",
        output_dir / "logs" / "batch_probe.log",
        context.is_primary,
    )
    state_path = output_dir / "run_state.json"
    if context.is_primary:
        dump_resolved_config(config, output_dir / "resolved_config.yaml")
    device = context.device
    setup_seed(
        int(config["experiment"]["seed"]) + context.rank,
        bool(config["runtime"]["deterministic"]),
    )
    dtype = amp_dtype(config, device)
    if context.is_primary:
        atomic_write_json(
            state_path,
            {
                "status": "batch_probing",
                "updated_at": utc_now(),
                "experiment": config["experiment"]["name"],
                "config": config["_meta"]["config_path"],
                "output_dir": str(output_dir),
                "completed_steps": 0,
                "total_steps": int(config["training"]["total_steps"]),
                "next_action": "等待显存探测完成",
                "environment": environment_summary(device),
            },
        )
    bundle = _build_model_for_context(config, context)
    logger.info("模型参数：%s", parameter_summary(bundle))
    choice = _choose_batch_for_context(
        bundle=bundle,
        config=config,
        context=context,
        dtype=dtype,
        logger=logger,
        output_dir=output_dir,
    )
    multi_view_enabled, num_views, _ = _multi_view_settings(config)
    result = {
        "status": "batch_tuned",
        "updated_at": utc_now(),
        "experiment": config["experiment"]["name"],
        "config": config["_meta"]["config_path"],
        "output_dir": str(output_dir),
        "completed_steps": 0,
        "total_steps": int(config["training"]["total_steps"]),
        "micro_batch_size": choice.micro_batch_size,
        "global_micro_batch_size": choice.global_micro_batch_size,
        "world_size": choice.world_size,
        "accumulation_steps": choice.accumulation_steps,
        "effective_batch_size": choice.effective_batch_size,
        "batch_unit": "objects" if multi_view_enabled else "views",
        "effective_view_batch_size": (
            choice.effective_batch_size * (num_views if multi_view_enabled else 1)
        ),
        "next_action": "运行 train.ps1；训练启动时会重新确认当前可用显存",
        "environment": environment_summary(device),
    }
    if context.is_primary:
        atomic_write_json(state_path, result)
    logger.info(
        "显存探测完成：micro=%d, accumulate=%d, effective=%d",
        choice.micro_batch_size,
        choice.accumulation_steps,
        choice.effective_batch_size,
    )
    return result


def probe_batch(config: dict[str, Any]) -> dict[str, Any]:
    context = _initialize_distributed_context(config)
    try:
        return _probe_batch_impl(config, context)
    finally:
        _cleanup_distributed_context(context)
