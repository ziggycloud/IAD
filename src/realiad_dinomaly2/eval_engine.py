from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, config_fingerprint
from .data import (
    RealIADMultiViewDataset,
    RealIADVarietyDataset,
    discover_categories,
)
from .losses import anomaly_map
from .metrics import (
    GaussianFilter,
    ObjectScoreAccumulator,
    SpoolingPixelMetricAccumulator,
    binary_metrics,
    mean_and_std,
    top_ratio_mean,
)
from .modeling import build_model, load_trainable_state_dict
from .normal_prior import (
    NormalPrior,
    file_sha256,
    load_normal_prior,
    normal_prior_path,
)
from .runtime import (
    amp_dtype,
    append_jsonl,
    atomic_write_json,
    autocast_context,
    environment_summary,
    resolve_device,
    setup_logger,
    setup_seed,
    utc_now,
)


PAPER_METRIC_KEYS = [
    "i_auroc",
    "i_aupr",
    "i_f1max",
    "p_auroc",
    "p_aupr",
    "p_f1max",
    "p_aupro",
]
DIAGNOSTIC_KEYS = [
    "o_auroc",
    "o_aupr",
    "o_f1max",
]


def _update_root_run_state(
    output_dir: Path,
    updates: dict[str, Any],
) -> None:
    path = output_dir / "run_state.json"
    state: dict[str, Any] = {}
    if path.is_file():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    state.update(updates)
    atomic_write_json(path, state)


def _resolve_checkpoint(output_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "auto":
        candidate = output_dir / "checkpoints" / "final_model.pt"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            "未找到 final_model.pt；正式评估默认只接受完整训练模型。"
            "请先完成训练。若仅做诊断，可显式指定 last.pt 并添加 "
            "--allow-partial。"
        )
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    return path


def _evaluation_signature(
    config: dict[str, Any],
    checkpoint_path: Path,
    categories: list[str],
) -> tuple[str, dict[str, Any]]:
    stat = checkpoint_path.stat()
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "config_fingerprint": config_fingerprint(config),
        "evaluation": config["evaluation"],
        "categories": categories,
        "image_label_policy": config["dataset"]["image_label_policy"],
        "missing_anomaly_mask_policy": config["dataset"][
            "missing_anomaly_mask_policy"
        ],
    }
    if bool(config["evaluation"].get("normal_prior", {}).get("enabled", False)):
        prior_path = normal_prior_path(config)
        if not prior_path.is_file():
            raise FileNotFoundError(
                f"normal prior artifact does not exist: {prior_path}"
            )
        payload["normal_prior"] = {
            "path": str(prior_path),
            "sha256": file_sha256(prior_path),
        }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hashlib.sha256(encoded).hexdigest()
    return signature, payload


def _loader(dataset, config: dict[str, Any]) -> DataLoader:
    evaluation = config["evaluation"]
    workers = int(evaluation["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(evaluation["batch_size"]),
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


@torch.no_grad()
def evaluate_category(
    category: str,
    bundle,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None,
    logger,
    scratch_dir: Path,
    normal_prior: NormalPrior | None = None,
) -> dict[str, Any]:
    dataset_config = config["dataset"]
    evaluation = config["evaluation"]
    multi_view_config = dict(config["model"].get("multi_view", {}))
    multi_view_enabled = bool(multi_view_config.get("enabled", False))
    common_dataset_args = {
        "json_dir": Path(dataset_config["json_dir"]),
        "image_dir": Path(dataset_config["image_dir"]),
        "category": category,
        "phase": "test",
        "image_size": int(dataset_config["image_size"]),
        "crop_size": int(dataset_config["crop_size"]),
        "image_label_policy": str(dataset_config["image_label_policy"]),
        "missing_anomaly_mask_policy": str(
            dataset_config["missing_anomaly_mask_policy"]
        ),
        "mask_resize_semantics": str(
            dataset_config.get(
                "mask_resize_semantics",
                "upstream_bilinear_nonzero",
            )
        ),
    }
    if multi_view_enabled:
        max_images = evaluation.get("max_test_images_per_category")
        max_objects = (
            None
            if max_images is None
            else max(1, int(max_images) // int(multi_view_config.get("num_views", 5)))
        )
        dataset = RealIADMultiViewDataset(
            **common_dataset_args,
            num_views=int(multi_view_config.get("num_views", 5)),
            missing_view_policy=str(
                multi_view_config.get("missing_view_policy", "error")
            ),
            max_objects=max_objects,
        )
    else:
        dataset = RealIADVarietyDataset(
            **common_dataset_args,
            max_items=evaluation.get("max_test_images_per_category"),
        )
    loader = _loader(dataset, config)
    resize_mask = int(evaluation["resize_mask"])
    pixel_accumulator = SpoolingPixelMetricAccumulator(
        device=device,
        bins=int(evaluation["metric_bins"]),
        capacity=(
            len(dataset) * int(multi_view_config.get("num_views", 5))
            if multi_view_enabled
            else len(dataset)
        ),
        height=resize_mask,
        width=resize_mask,
        scratch_dir=scratch_dir,
        stem=category,
        replay_batch_size=max(1, int(evaluation["batch_size"])),
    )
    object_accumulator = ObjectScoreAccumulator(
        top_ratio=float(evaluation["object_top_ratio"]),
        expected_views=5,
        mode=str(
            evaluation.get("object_score_aggregation", "legacy_concat_topk")
        ),
        softmax_temperature=float(
            evaluation.get("object_score_softmax_temperature", 0.25)
        ),
    )
    gaussian = GaussianFilter(
        kernel_size=int(evaluation["gaussian_kernel_size"]),
        sigma=float(evaluation["gaussian_sigma"]),
    ).to(device)
    gaussian.eval()

    image_labels: list[int] = []
    image_scores: list[float] = []
    image_count = 0
    pixel_count = 0
    started = time.perf_counter()
    bundle.model.eval()

    for batch in loader:
        if multi_view_enabled:
            images = batch["images"].to(
                device,
                non_blocking=bool(config["runtime"]["pin_memory"]),
            )
            view_ids = batch["view_ids"].to(device, non_blocking=True)
            valid_view_mask = batch["valid_view_mask"].to(
                device, non_blocking=True
            )
            category_names = [str(value) for value in batch["category"]]
            with autocast_context(dtype, device):
                encoder_features, decoder_features = bundle.model(
                    images,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                )
                patch_maps = anomaly_map(
                    encoder_features,
                    decoder_features,
                    output_size=int(dataset_config["crop_size"]) // 14,
                    layer_weights=evaluation.get("anomaly_map_layer_weights"),
                    align_corners=bool(
                        evaluation.get("anomaly_map_align_corners", True)
                    ),
                )
            if normal_prior is not None:
                patch_maps = normal_prior.calibrate(
                    patch_maps,
                    categories=category_names,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                    config=config,
                )
            batch_size, view_count = patch_maps.shape[:2]
            flat_maps = patch_maps.float().reshape(
                batch_size * view_count,
                *patch_maps.shape[2:],
            )
            labels_tensor = batch["labels"].to(dtype=torch.uint8).reshape(-1)
            masks = batch["masks"].to(device, non_blocking=True).reshape(
                batch_size * view_count,
                *batch["masks"].shape[2:],
            )
            pixel_valid = batch["pixel_valid"].to(dtype=torch.bool).reshape(-1)
            valid_flat = valid_view_mask.reshape(-1)
            object_ids = [str(value) for value in batch["object_id"]]
            flat_object_ids = [
                object_ids[object_index]
                for object_index in range(batch_size)
                for _ in range(view_count)
            ]
            flat_categories = [
                category_names[object_index]
                for object_index in range(batch_size)
                for _ in range(view_count)
            ]
        else:
            images = batch["image"].to(
                device,
                non_blocking=bool(config["runtime"]["pin_memory"]),
            )
            category_names = [str(value) for value in batch["category"]]
            view_ids = batch["view_id"].to(device, dtype=torch.long) - 1
            valid_view_mask = torch.ones_like(view_ids, dtype=torch.bool)
            with autocast_context(dtype, device):
                encoder_features, decoder_features = bundle.model(images)
                patch_maps = anomaly_map(
                    encoder_features,
                    decoder_features,
                    output_size=int(dataset_config["crop_size"]) // 14,
                    layer_weights=evaluation.get("anomaly_map_layer_weights"),
                    align_corners=bool(
                        evaluation.get("anomaly_map_align_corners", True)
                    ),
                )
            if normal_prior is not None:
                patch_maps = normal_prior.calibrate(
                    patch_maps,
                    categories=category_names,
                    view_ids=view_ids,
                    valid_view_mask=valid_view_mask,
                    config=config,
                )
            flat_maps = patch_maps.float()
            labels_tensor = batch["label"].to(dtype=torch.uint8)
            masks = batch["mask"].to(device, non_blocking=True)
            pixel_valid = batch["pixel_valid"].to(dtype=torch.bool)
            valid_flat = valid_view_mask
            flat_object_ids = [str(value) for value in batch["object_id"]]
            flat_categories = category_names

        flat_maps = F.interpolate(
            flat_maps,
            size=(resize_mask, resize_mask),
            mode="bilinear",
            align_corners=False,
        )
        flat_maps = gaussian(flat_maps)
        masks = F.interpolate(
            masks,
            size=(resize_mask, resize_mask),
            mode="nearest",
        )
        masks = (masks > 0.5).to(torch.uint8)
        scores_tensor = top_ratio_mean(
            flat_maps,
            ratio=float(evaluation["image_top_ratio"]),
        )
        valid_cpu = valid_flat.to(device="cpu", dtype=torch.bool)
        image_labels.extend(
            int(value) for value in labels_tensor[valid_cpu].tolist()
        )
        image_scores.extend(
            float(value) for value in scores_tensor.cpu()[valid_cpu].tolist()
        )

        metric_valid = pixel_valid & valid_cpu
        if bool(metric_valid.any()):
            valid_device = metric_valid.to(device)
            pixel_accumulator.add(
                flat_maps[valid_device, 0],
                masks[valid_device, 0],
            )
            pixel_count += int(metric_valid.sum())

        for index in valid_cpu.nonzero(as_tuple=False).flatten().tolist():
            object_accumulator.add(
                object_key=f"{flat_categories[index]}/{flat_object_ids[index]}",
                anomaly_map=flat_maps[index, 0],
                image_label=int(labels_tensor[index]),
            )
        image_count += int(valid_cpu.sum())

    image_result = binary_metrics(image_labels, image_scores)
    pixel_result = pixel_accumulator.summary()
    object_result = object_accumulator.summary()
    elapsed = time.perf_counter() - started
    row: dict[str, Any] = {
        "category": category,
        "test_images": image_count,
        "pixel_metric_images": pixel_count,
        "objects": int(object_result["objects"]),
        "seconds": elapsed,
        "i_auroc": image_result["auroc"],
        "i_aupr": image_result["aupr"],
        "i_f1max": image_result["f1max"],
        "p_auroc": pixel_result["auroc"],
        "p_aupr": pixel_result["aupr"],
        "p_f1max": pixel_result["f1max"],
        "p_aupro": pixel_result["aupro"],
        "pixel_score_min": pixel_result["score_min"],
        "pixel_score_max": pixel_result["score_max"],
        "o_auroc": object_result["auroc"],
        "o_aupr": object_result["aupr"],
        "o_f1max": object_result["f1max"],
        "min_views_per_object": int(object_result["min_views"]),
        "max_views_per_object": int(object_result["max_views"]),
    }
    logger.info(
        "%s | I-ROC %.4f I-PR %.4f | P-ROC %.4f P-PR %.4f P-PRO %.4f | %.1fs",
        category,
        row["i_auroc"],
        row["i_aupr"],
        row["p_auroc"],
        row["p_aupr"],
        row["p_aupro"],
        elapsed,
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _report_text(
    config: dict[str, Any],
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    macro: dict[str, float],
    std: dict[str, float],
    is_partial: bool,
    completed_steps: int,
) -> str:
    worst_pixel = sorted(rows, key=lambda row: row["p_aupr"])[:5]
    i_p_gap = macro["i_auroc"] - macro["p_aupr"]
    category_spread = std["p_aupr"]
    object_gain = macro["o_auroc"] - macro["i_auroc"]

    suggestions: list[str] = []
    if i_p_gap > 0.25:
        suggestions.append(
            "图像识别明显强于像素精定位。280 输入经 ViT/14 后仅有 20 x 20 token，"
            "优先尝试高分辨率滑窗或轻量多尺度定位头。"
        )
    if category_spread > 0.08:
        suggestions.append(
            "类别间 P-PR 波动较大，统一 decoder 仍可能有容量-多样性冲突；"
            "可尝试类别条件 bottleneck 或小型 mixture-of-experts。"
        )
    if object_gain > 0.01:
        suggestions.append(
            "五视角对象分数优于独立视角，说明跨视角信息有价值；"
            "可把目前仅在分数端的聚合前移为 token/feature 级融合。"
        )
    else:
        suggestions.append(
            "当前特征级跨视角融合没有带来稳定的对象级收益；应检查融合权重"
            "是否削弱了仅在单一视角可见的异常证据。"
        )

    metric_order = (
        "I-ROC",
        "I-PR",
        "I-F1max",
        "P-ROC",
        "P-PR",
        "P-F1max",
        "P-PRO",
    )
    values = [_percentage(macro[key]) for key in PAPER_METRIC_KEYS]
    lines = [
        "# Dinomaly2 在 Real-IAD Variety 上的评估报告",
        "",
        (
            f"状态：{'中间断点诊断' if is_partial else '完整评估'}已完成"
            f"（训练 {completed_steps:,}/{int(config['training']['total_steps']):,} "
            f"steps；{len(rows)} 个类别，宏平均）。"
        ),
        "",
        "## 实测结果（%）",
        "",
        "| " + " | ".join(metric_order) + " |",
        "|" + "|".join(["---:"] * len(metric_order)) + "|",
        "| " + " | ".join(values) + " |",
        "",
        "评估按官方 split 逐类别计算后宏平均；P-PRO 积分到 30% FPR。",
        "无 mask 的异常物体视角按上游 Dinomaly2 语义视为 view-normal，"
        "对象标签取五个视角标签的最大值。",
        "",
        "## 运行配置",
        "",
        f"- Checkpoint：`{checkpoint_path}`",
        f"- Backbone：`{config['model']['backbone']}`",
        (
            f"- 输入：{config['dataset']['image_size']} -> "
            f"{config['dataset']['crop_size']}"
        ),
        (
            f"- 图像/对象分数：top "
            f"{100 * float(config['evaluation']['image_top_ratio']):g}% / "
            f"{100 * float(config['evaluation']['object_top_ratio']):g}%"
        ),
        "",
        "## 最弱的 5 个类别（按 P-PR）",
        "",
        "| 类别 | I-ROC | P-ROC | P-PR | P-PRO |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in worst_pixel:
        lines.append(
            "| {category} | {i} | {p} | {pr} | {pro} |".format(
                category=row["category"],
                i=_percentage(row["i_auroc"]),
                p=_percentage(row["p_auroc"]),
                pr=_percentage(row["p_aupr"]),
                pro=_percentage(row["p_aupro"]),
            )
        )
    lines.extend(
        [
            "",
            "## 当前架构最值得改进之处",
            "",
        ]
    )
    for index, suggestion in enumerate(suggestions, start=1):
        lines.append(f"{index}. {suggestion}")
    lines.extend(
        [
            "",
            "论文表 3 的 Dinomaly（非 Dinomaly2）参考值为 "
            "85.4 / 97.2 / 94.5 / 91.5 / 42.8 / 45.8 / 75.6；"
            "模型和训练配置不同，只用于定位量级，不能视作同配置复现目标。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(
    config: dict[str, Any],
    checkpoint: str = "auto",
    allow_partial: bool = False,
    categories_override: list[str] | None = None,
    split_name: str | None = None,
    publish_root_report: bool = True,
) -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoint_path = _resolve_checkpoint(output_dir, checkpoint)
    device = resolve_device(str(config["runtime"]["device"]))
    setup_seed(
        int(config["experiment"]["seed"]),
        bool(config["runtime"]["deterministic"]),
    )
    dtype = (
        amp_dtype(config, device)
        if bool(config["evaluation"].get("amp", False))
        else None
    )
    categories = discover_categories(
        Path(config["dataset"]["json_dir"]),
        requested=(
            categories_override
            if categories_override is not None
            else config["dataset"]["categories"]
        ),
        limit=(
            None
            if categories_override is not None
            else config["dataset"].get("category_limit")
        ),
    )
    signature, signature_payload = _evaluation_signature(
        config,
        checkpoint_path,
        categories,
    )
    evaluation_dir = output_dir / "evaluation" / signature[:12]
    category_dir = evaluation_dir / "per_category"
    scratch_dir = evaluation_dir / "scratch"
    category_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("dinomaly2_eval", evaluation_dir / "evaluate.log")
    progress_path = evaluation_dir / "progress.jsonl"
    state_path = evaluation_dir / "eval_state.json"
    atomic_write_json(
        evaluation_dir / "metadata.json",
        {
            "signature": signature,
            "inputs": signature_payload,
            "created_at": utc_now(),
        },
    )
    atomic_write_json(
        output_dir / "evaluation" / "latest.json",
        {
            "signature": signature,
            "evaluation_dir": str(evaluation_dir),
            "updated_at": utc_now(),
        },
    )
    _update_root_run_state(
        output_dir,
        {
            "status": "evaluating",
            "updated_at": utc_now(),
            "completed_categories": 0,
            "total_categories": len(categories),
            "evaluation_dir": str(evaluation_dir),
            "next_action": (
                "继续评估；中断后重新运行 evaluate.ps1 将跳过已完成类别"
            ),
        },
    )

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected = config_fingerprint(config)
    if checkpoint_payload.get("config_fingerprint") != expected:
        raise ValueError("checkpoint 与当前模型/训练语义配置不一致")
    completed_steps = int(checkpoint_payload.get("completed_steps", -1))
    total_steps = int(config["training"]["total_steps"])
    is_partial = completed_steps != total_steps
    if is_partial and not allow_partial:
        raise ValueError(
            f"checkpoint 仅完成 {completed_steps}/{total_steps} steps。"
            "正式论文指标拒绝评估中间断点；如仅做诊断，请显式添加 "
            "--allow-partial。"
        )
    bundle = build_model(config, device)
    expected_backbone_sha256 = checkpoint_payload.get("backbone_sha256")
    if (
        expected_backbone_sha256 is not None
        and expected_backbone_sha256 != bundle.backbone_sha256
    ):
        raise ValueError(
            "checkpoint 所用 DINOv2 backbone SHA-256 与当前权重不一致"
        )
    load_trainable_state_dict(bundle, checkpoint_payload["model"])
    normal_prior = None
    if bool(config["evaluation"].get("normal_prior", {}).get("enabled", False)):
        normal_prior = load_normal_prior(
            normal_prior_path(config),
            config,
            checkpoint_path,
        )
    logger.info(
        "评估 %d 类；checkpoint step=%s；环境=%s",
        len(categories),
        checkpoint_payload.get("completed_steps"),
        environment_summary(device),
    )

    resume_enabled = bool(config["evaluation"]["resume"])
    rows: list[dict[str, Any]] = []
    for index, category in enumerate(categories, start=1):
        result_path = category_dir / f"{category}.json"
        if resume_enabled and result_path.is_file():
            with result_path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
            logger.info("[%d/%d] 跳过已完成 %s", index, len(categories), category)
        else:
            logger.info("[%d/%d] 开始 %s", index, len(categories), category)
            row = evaluate_category(
                category=category,
                bundle=bundle,
                config=config,
                device=device,
                dtype=dtype,
                logger=logger,
                scratch_dir=scratch_dir,
                normal_prior=normal_prior,
            )
            atomic_write_json(result_path, row)
            append_jsonl(
                progress_path,
                {
                    "timestamp": utc_now(),
                    "event": "category_complete",
                    "index": index,
                    "total_categories": len(categories),
                    **row,
                },
            )
        rows.append(row)
        atomic_write_json(
            state_path,
            {
                "status": "evaluating",
                "updated_at": utc_now(),
                "completed_categories": index,
                "total_categories": len(categories),
                "current_or_last_category": category,
                "next_action": "重新运行 evaluate.ps1 将跳过已完成类别",
            },
        )
        _update_root_run_state(
            output_dir,
            {
                "status": "evaluating",
                "updated_at": utc_now(),
                "completed_categories": index,
                "total_categories": len(categories),
                "current_or_last_category": category,
                "evaluation_dir": str(evaluation_dir),
                "next_action": (
                    "继续评估；中断后重新运行 evaluate.ps1 将跳过已完成类别"
                ),
            },
        )

    rows.sort(key=lambda row: row["category"])
    macro, standard_deviation = mean_and_std(
        rows,
        PAPER_METRIC_KEYS + DIAGNOSTIC_KEYS,
    )
    metrics_payload = {
        "status": "partial_diagnostic" if is_partial else "complete",
        "split_name": split_name,
        "completed_at": utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_steps": checkpoint_payload.get("completed_steps"),
        "training_total_steps": total_steps,
        "categories": len(rows),
        "macro_average": macro,
        "category_standard_deviation": standard_deviation,
        "paper_metric_order": PAPER_METRIC_KEYS,
        "evaluation_dir": str(evaluation_dir),
        "metrics_per_category": str(
            evaluation_dir / "metrics_per_category.csv"
        ),
    }
    atomic_write_json(evaluation_dir / "metrics.json", metrics_payload)
    _write_csv(evaluation_dir / "metrics_per_category.csv", rows)
    report = _report_text(
        config=config,
        checkpoint_path=checkpoint_path,
        rows=rows,
        macro=macro,
        std=standard_deviation,
        is_partial=is_partial,
        completed_steps=completed_steps,
    )
    report_path = evaluation_dir / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    if (
        publish_root_report
        and split_name is None
        and "smoke" not in str(config["experiment"]["name"]).lower()
        and not is_partial
    ):
        (PROJECT_ROOT / "reports" / "evaluation_report.md").write_text(
            report,
            encoding="utf-8",
            newline="\n",
        )
    atomic_write_json(
        state_path,
        {
            "status": "partial_diagnostic" if is_partial else "complete",
            "updated_at": utc_now(),
            "completed_categories": len(categories),
            "total_categories": len(categories),
            "metrics": str(evaluation_dir / "metrics.json"),
            "report": str(report_path),
            "next_action": "阅读评估报告并据最弱类别开展架构消融",
        },
    )
    _update_root_run_state(
        output_dir,
        {
            "status": (
                "partial_diagnostic" if is_partial else "complete"
            ),
            "updated_at": utc_now(),
            "completed_categories": len(categories),
            "total_categories": len(categories),
            "metrics": str(evaluation_dir / "metrics.json"),
            "report": str(report_path),
            "next_action": (
                "阅读评估报告并据最弱类别开展架构消融"
            ),
        },
    )
    logger.info("评估完成：%s", metrics_payload["macro_average"])
    return metrics_payload
