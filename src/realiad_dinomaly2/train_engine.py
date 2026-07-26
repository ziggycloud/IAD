from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler

from .config import (
    config_fingerprint,
    dump_resolved_config,
)
from .data import build_train_dataset, discover_categories
from .losses import discard_rate_for_step, reconstruction_loss
from .modeling import (
    ModelBundle,
    build_model,
    build_optimizer,
    load_trainable_state_dict,
    parameter_summary,
    set_learning_rate,
    trainable_state_dict,
)
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


class DeterministicIterationBatchSampler(Sampler[list[int]]):
    """Iteration-addressable shuffle sampler, allowing exact data-order resume."""

    def __init__(
        self,
        dataset_size: int,
        micro_batch_size: int,
        total_optimizer_steps: int,
        accumulation_steps: int,
        start_optimizer_step: int,
        seed: int,
    ) -> None:
        if dataset_size < micro_batch_size:
            raise ValueError(
                f"训练集仅 {dataset_size} 张，micro batch={micro_batch_size}"
            )
        self.dataset_size = dataset_size
        self.micro_batch_size = micro_batch_size
        self.steps_per_epoch = dataset_size // micro_batch_size
        self.total_micro_steps = total_optimizer_steps * accumulation_steps
        self.start_micro_step = start_optimizer_step * accumulation_steps
        self.seed = seed

    def __len__(self) -> int:
        return self.total_micro_steps - self.start_micro_step

    def __iter__(self) -> Iterator[list[int]]:
        micro_step = self.start_micro_step
        cached_epoch = -1
        permutation: list[int] = []
        while micro_step < self.total_micro_steps:
            epoch = micro_step // self.steps_per_epoch
            batch_in_epoch = micro_step % self.steps_per_epoch
            if epoch != cached_epoch:
                generator = torch.Generator()
                generator.manual_seed(self.seed + epoch)
                permutation = torch.randperm(
                    self.dataset_size,
                    generator=generator,
                ).tolist()
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


def _trainable_parameters(bundle: ModelBundle) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in (bundle.bottleneck, bundle.decoder)
        for parameter in module.parameters()
    ]


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
    synthetic = torch.randn(
        candidate,
        3,
        crop_size,
        crop_size,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with autocast_context(dtype, device):
        encoder_features, decoder_features = bundle.model(synthetic)
        loss = reconstruction_loss(
            encoder_features,
            decoder_features,
            discard_rate=0.0,
            loose_loss=bool(config["model"]["loose_loss"]),
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
        "status": "ok",
        "loss": float(loss.detach().cpu()),
        "seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }

    del encoder_features, decoder_features, loss, synthetic, optimizer, scaler
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
) -> BatchChoice:
    training = config["training"]
    effective = int(training["effective_batch_size"])
    requested = training["micro_batch_size"]
    if requested != "auto":
        micro = int(requested)
        if effective % micro:
            raise ValueError(
                f"effective_batch_size={effective} 不能被 micro_batch_size={micro} 整除"
            )
        return BatchChoice(
            micro_batch_size=micro,
            accumulation_steps=effective // micro,
            effective_batch_size=effective,
            trials=[],
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
            if int(value) > 0 and effective % int(value) == 0
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
                    accumulation_steps=effective // candidate,
                    effective_batch_size=effective,
                    trials=trials,
                )
                atomic_write_json(
                    output_dir / "batch_tuning.json",
                    {
                        "selected_micro_batch_size": choice.micro_batch_size,
                        "accumulation_steps": choice.accumulation_steps,
                        "effective_batch_size": choice.effective_batch_size,
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
) -> None:
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
            "model": trainable_state_dict(bundle),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": _rng_state(),
        },
    )


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
) -> DataLoader:
    runtime = config["runtime"]
    training = config["training"]
    sampler = DeterministicIterationBatchSampler(
        dataset_size=len(dataset),
        micro_batch_size=batch_choice.micro_batch_size,
        total_optimizer_steps=int(training["total_steps"]),
        accumulation_steps=batch_choice.accumulation_steps,
        start_optimizer_step=completed_steps,
        seed=int(config["experiment"]["seed"]),
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


def train(config: dict[str, Any], resume: str | None = "auto") -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _resolve_resume(
        output_dir,
        resume,
        bool(config["training"]["auto_resume"]),
    )
    logger = setup_logger("dinomaly2_train", log_dir / "train.log")
    progress_path = log_dir / "progress.jsonl"
    state_path = output_dir / "run_state.json"
    dump_resolved_config(config, output_dir / "resolved_config.yaml")

    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(
        int(config["experiment"]["seed"]),
        bool(config["runtime"]["deterministic"]),
    )
    dtype = amp_dtype(config, device)
    environment = environment_summary(device)
    logger.info("环境：%s", environment)

    json_dir = Path(config["dataset"]["json_dir"])
    raw_image_dir = Path(config["dataset"]["image_dir"])
    configured_train_dir = config["dataset"].get("train_image_dir")
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
            "configs/rtx3060ti_strict_upstream.yaml) to train from raw images."
        )
        logger.error(message)
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
        requested=config["dataset"]["categories"],
        limit=config["dataset"].get("category_limit"),
    )
    dataset = build_train_dataset(
        json_dir=json_dir,
        image_dir=image_dir,
        categories=categories,
        image_size=int(config["dataset"]["image_size"]),
        crop_size=int(config["dataset"]["crop_size"]),
    )
    logger.info(
        "训练 manifest：%d 类，%d 张正常视图",
        len(categories),
        len(dataset),
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
    atomic_write_json(state_path, state)

    bundle = build_model(config, device)
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
        batch_choice = BatchChoice(
            micro_batch_size=int(checkpoint["micro_batch_size"]),
            accumulation_steps=int(checkpoint["accumulation_steps"]),
            effective_batch_size=int(checkpoint["effective_batch_size"]),
            trials=[],
        )
    else:
        batch_choice = choose_batch_size(
            bundle=bundle,
            config=config,
            device=device,
            dtype=dtype,
            logger=logger,
            output_dir=output_dir,
        )

    optimizer = build_optimizer(bundle, config)
    scaler = make_grad_scaler(dtype, device)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        _restore_rng_state(checkpoint["rng_state"])

    total_steps = int(config["training"]["total_steps"])
    if completed_steps > total_steps:
        raise ValueError(
            f"checkpoint 已完成 {completed_steps} 步，超过配置 {total_steps}"
        )
    logger.info(
        "运行 batch：micro=%d, accumulate=%d, effective=%d；从 step=%d 开始",
        batch_choice.micro_batch_size,
        batch_choice.accumulation_steps,
        batch_choice.effective_batch_size,
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
            "accumulation_steps": batch_choice.accumulation_steps,
            "effective_batch_size": batch_choice.effective_batch_size,
            "next_action": "继续训练；中断后重新运行 train.ps1 将自动续跑",
            "protocol_notes": protocol_notes,
        }
    )
    atomic_write_json(state_path, state)

    if completed_steps == total_steps:
        logger.info("checkpoint 已达到 total_steps，直接写出最终权重")
    else:
        loader = _dataloader(dataset, config, batch_choice, completed_steps)
        iterator = iter(loader)
        bundle.model.train()
        trainable_parameters = _trainable_parameters(bundle)
        log_every = int(config["training"]["log_every"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        started = time.perf_counter()
        interval_started = started

        try:
            for step_index in range(completed_steps, total_steps):
                learning_rates = set_learning_rate(
                    optimizer=optimizer,
                    completed_steps=step_index,
                    total_steps=total_steps,
                    warmup_steps=int(config["training"]["warmup_steps"]),
                    final_ratio=float(config["training"]["final_lr_ratio"]),
                    step_offset=int(
                        config["training"].get("lr_step_offset", 1)
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                discard_rate = discard_rate_for_step(
                    completed_steps=step_index,
                    warmup_steps=int(
                        config["training"]["loose_loss_warmup_steps"]
                    ),
                    final_rate=float(
                        config["training"]["loose_loss_final_discard"]
                    ),
                )

                for _ in range(batch_choice.accumulation_steps):
                    batch = next(iterator)
                    images = batch["image"].to(
                        device,
                        non_blocking=bool(config["runtime"]["pin_memory"]),
                    )
                    with autocast_context(dtype, device):
                        encoder_features, decoder_features = bundle.model(images)
                        raw_loss = reconstruction_loss(
                            encoder_features,
                            decoder_features,
                            discard_rate=discard_rate,
                            loose_loss=bool(config["model"]["loose_loss"]),
                        )
                        loss = raw_loss / batch_choice.accumulation_steps
                    if not torch.isfinite(raw_loss):
                        raise FloatingPointError(
                            f"step={step_index} 出现非有限 loss={raw_loss.item()}"
                        )
                    scaler.scale(loss).backward()
                    accumulated_loss += float(raw_loss.detach().cpu())

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(
                    trainable_parameters,
                    max_norm=float(config["training"]["gradient_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                completed_steps = step_index + 1

                should_log = (
                    completed_steps == 1
                    or completed_steps % log_every == 0
                    or completed_steps == total_steps
                )
                if should_log:
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
                        "loss": accumulated_loss
                        / batch_choice.accumulation_steps,
                        "grad_norm": float(
                            grad_norm.detach().cpu()
                            if torch.is_tensor(grad_norm)
                            else grad_norm
                        ),
                        "learning_rates": learning_rates,
                        "discard_rate": discard_rate,
                        "micro_batch_size": batch_choice.micro_batch_size,
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
                    append_jsonl(progress_path, payload)
                    logger.info(
                        "step %d/%d | loss %.6f | lr %.3e | grad %.4f | ETA %.1fh",
                        completed_steps,
                        total_steps,
                        payload["loss"],
                        learning_rates[-1],
                        payload["grad_norm"],
                        eta_seconds / 3600,
                    )
                    state.update(
                        {
                            "status": "training",
                            "updated_at": utc_now(),
                            "completed_steps": completed_steps,
                            "last_loss": payload["loss"],
                            "eta_seconds": eta_seconds,
                            "next_action": (
                                "继续训练；中断后重新运行 train.ps1 将自动续跑"
                            ),
                        }
                    )
                    atomic_write_json(state_path, state)
                    interval_started = now

                if (
                    completed_steps % checkpoint_every == 0
                    or completed_steps == total_steps
                ):
                    logger.info("保存断点 step=%d", completed_steps)
                    save_checkpoint(
                        checkpoint_dir / "last.pt",
                        bundle=bundle,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        completed_steps=completed_steps,
                        batch_choice=batch_choice,
                    )
        except KeyboardInterrupt:
            logger.warning("收到中断，正在保存可续跑断点")
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
            atomic_write_json(state_path, state)
            raise

    final_path = checkpoint_dir / "final_model.pt"
    atomic_torch_save(
        final_path,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "backbone_sha256": bundle.backbone_sha256,
            "created_at": utc_now(),
            "config_fingerprint": config_fingerprint(config),
            "completed_steps": total_steps,
            "model": trainable_state_dict(bundle),
            "backbone": bundle.backbone_name,
        },
    )
    state.update(
        {
            "status": "trained",
            "updated_at": utc_now(),
            "completed_steps": total_steps,
            "checkpoint": str(final_path),
            "next_action": "运行 evaluate.ps1 计算论文七项指标",
        }
    )
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


def probe_batch(config: dict[str, Any]) -> dict[str, Any]:
    """Run the real forward/backward/optimizer memory probe without training."""
    output_dir = Path(config["experiment"]["output_dir"])
    logger = setup_logger(
        "dinomaly2_batch_probe",
        output_dir / "logs" / "batch_probe.log",
    )
    state_path = output_dir / "run_state.json"
    dump_resolved_config(config, output_dir / "resolved_config.yaml")
    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(
        int(config["experiment"]["seed"]),
        bool(config["runtime"]["deterministic"]),
    )
    dtype = amp_dtype(config, device)
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
    bundle = build_model(config, device)
    logger.info("模型参数：%s", parameter_summary(bundle))
    choice = choose_batch_size(
        bundle=bundle,
        config=config,
        device=device,
        dtype=dtype,
        logger=logger,
        output_dir=output_dir,
    )
    result = {
        "status": "batch_tuned",
        "updated_at": utc_now(),
        "experiment": config["experiment"]["name"],
        "config": config["_meta"]["config_path"],
        "output_dir": str(output_dir),
        "completed_steps": 0,
        "total_steps": int(config["training"]["total_steps"]),
        "micro_batch_size": choice.micro_batch_size,
        "accumulation_steps": choice.accumulation_steps,
        "effective_batch_size": choice.effective_batch_size,
        "next_action": "运行 train.ps1；训练启动时会重新确认当前可用显存",
        "environment": environment_summary(device),
    }
    atomic_write_json(state_path, result)
    logger.info(
        "显存探测完成：micro=%d, accumulate=%d, effective=%d",
        choice.micro_batch_size,
        choice.accumulation_steps,
        choice.effective_batch_size,
    )
    return result
