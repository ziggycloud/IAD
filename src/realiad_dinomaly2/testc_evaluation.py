from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .runtime import append_jsonl, atomic_write_json, utc_now
from .testc_data import canonical_digest, file_sha256, load_testc_protocol


SCORE_WEIGHTS = {"classification": 0.3, "segmentation": 0.5, "zero_shot": 0.2}


def _binary_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    labels_array = np.asarray(list(labels), dtype=np.uint8)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    if labels_array.size == 0 or np.unique(labels_array).size != 2:
        raise ValueError("Binary metrics require both normal and anomalous samples")
    precision, recall, thresholds = precision_recall_curve(labels_array, scores_array)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    index = int(np.argmax(f1))
    threshold = (
        float(thresholds[index])
        if index < len(thresholds)
        else float(np.nextafter(scores_array.max(), math.inf))
    )
    return {
        "auroc": float(roc_auc_score(labels_array, scores_array)),
        "ap": float(average_precision_score(labels_array, scores_array)),
        "f1max": float(f1[index]),
        "f1_threshold": threshold,
    }


class PixelHistogram:
    """Exact binary metrics for 8-bit submitted masks without retaining pixels."""

    def __init__(self) -> None:
        self.positive = np.zeros(256, dtype=np.int64)
        self.negative = np.zeros(256, dtype=np.int64)

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.dtype != np.uint8 or prediction.shape != target.shape:
            raise ValueError("Pixel predictions must be uint8 and match the GT shape")
        positive = target.astype(bool, copy=False)
        self.positive += np.bincount(prediction[positive], minlength=256)
        self.negative += np.bincount(prediction[~positive], minlength=256)

    def summary(self) -> dict[str, float]:
        positives = int(self.positive.sum())
        negatives = int(self.negative.sum())
        if positives == 0 or negatives == 0:
            raise ValueError("Pixel metrics require positive and negative GT pixels")
        tp = np.concatenate(([0], np.cumsum(self.positive[::-1], dtype=np.int64)))
        fp = np.concatenate(([0], np.cumsum(self.negative[::-1], dtype=np.int64)))
        recall = tp / positives
        fpr = fp / negatives
        precision = np.divide(
            tp,
            tp + fp,
            out=np.ones_like(tp, dtype=np.float64),
            where=(tp + fp) > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        best = int(np.argmax(f1))
        # index 0 represents a threshold above 255; index 1 represents 255.
        threshold_byte = 256 - best
        return {
            "auroc": float(np.trapz(recall, fpr)),
            "ap": float(np.sum(np.diff(recall) * precision[1:])),
            "f1max": float(f1[best]),
            "f1_threshold": float(threshold_byte / 255.0),
            "positive_pixels": float(positives),
            "negative_pixels": float(negatives),
        }


def _top_ratio_score(array: np.ndarray, ratio: float) -> float:
    values = array.reshape(-1)
    count = max(1, int(values.size * ratio))
    start = values.size - count
    return float(np.partition(values, start)[start:].mean() / 255.0)


def _read_submission_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["group_folder", "anomaly_score"]:
            raise ValueError(f"Unexpected submission columns: {reader.fieldnames}")
        rows = {str(row["group_folder"]): float(row["anomaly_score"]) for row in reader}
    if len(rows) == 0:
        raise ValueError("Submission score CSV is empty")
    return rows


def _category_metrics(
    *,
    category: str,
    samples: list[dict[str, Any]],
    score_by_group: dict[str, float],
    submission_root: Path,
    testc_root: Path,
    image_top_ratio: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    object_labels: list[int] = []
    object_scores: list[float] = []
    view_labels: list[int] = []
    view_scores: list[float] = []
    max_view_object_scores: list[float] = []
    mean_view_object_scores: list[float] = []
    pixel = PixelHistogram()
    visible_anomaly_views = 0
    missing_anomaly_masks = 0
    for sample in samples:
        group_folder = str(sample["group_folder"])
        if group_folder not in score_by_group:
            raise ValueError(f"Missing submitted object score for {group_folder}")
        object_labels.append(int(sample["object_label"]))
        object_scores.append(float(score_by_group[group_folder]))
        current_view_scores: list[float] = []
        for view in sample["views"]:
            view_id = int(view["view_id"])
            prediction_path = (
                submission_root
                / "predicted_masks"
                / category
                / str(sample["sample_id"])
                / f"{view_id}_mask.png"
            )
            with Image.open(prediction_path) as source:
                prediction = np.asarray(source.convert("L"), dtype=np.uint8)
            raw_mask = view.get("mask_path")
            if raw_mask is None:
                target = np.zeros_like(prediction, dtype=np.uint8)
                if int(sample["object_label"]):
                    missing_anomaly_masks += 1
            else:
                with Image.open(testc_root / Path(str(raw_mask))) as source:
                    mask_image = source.convert("L").resize(
                        (prediction.shape[1], prediction.shape[0]),
                        resample=Image.Resampling.NEAREST,
                    )
                    target = (np.asarray(mask_image, dtype=np.uint8) > 0).astype(np.uint8)
                visible_anomaly_views += int(target.any())
            if bool(view.get("pixel_valid", True)):
                pixel.add(prediction, target)
            current_score = _top_ratio_score(prediction, image_top_ratio)
            current_view_scores.append(current_score)
            view_labels.append(int(view["view_label"]))
            view_scores.append(current_score)
        max_view_object_scores.append(max(current_view_scores))
        mean_view_object_scores.append(float(np.mean(current_view_scores)))

    classification = _binary_metrics(object_labels, object_scores)
    per_view = _binary_metrics(view_labels, view_scores)
    max_view = _binary_metrics(object_labels, max_view_object_scores)
    mean_view = _binary_metrics(object_labels, mean_view_object_scores)
    segmentation = pixel.summary()
    normal_scores = [
        score for label, score in zip(object_labels, object_scores, strict=True)
        if label == 0
    ]
    anomaly_scores = [
        score for label, score in zip(object_labels, object_scores, strict=True)
        if label == 1
    ]
    pixel_count = segmentation["positive_pixels"] + segmentation["negative_pixels"]
    return {
        "category": category,
        "partition": str(samples[0]["partition"]),
        "objects": len(samples),
        "normal_objects": sum(value == 0 for value in object_labels),
        "anomaly_objects": sum(value == 1 for value in object_labels),
        "views": len(view_labels),
        "visible_anomaly_views": visible_anomaly_views,
        "missing_anomaly_masks": missing_anomaly_masks,
        "seconds": time.perf_counter() - started,
        "c_auroc": classification["auroc"],
        "c_ap": classification["ap"],
        "c_f1max": classification["f1max"],
        "c_f1_threshold": classification["f1_threshold"],
        "p_auroc": segmentation["auroc"],
        "p_ap": segmentation["ap"],
        "p_f1max": segmentation["f1max"],
        "p_f1_threshold": segmentation["f1_threshold"],
        "view_auroc": per_view["auroc"],
        "view_ap": per_view["ap"],
        "view_f1max": per_view["f1max"],
        "diag_max_view_auroc": max_view["auroc"],
        "diag_mean_view_auroc": mean_view["auroc"],
        "normal_object_score_mean": float(np.mean(normal_scores)),
        "anomaly_object_score_mean": float(np.mean(anomaly_scores)),
        "object_score_margin": float(np.mean(anomaly_scores) - np.mean(normal_scores)),
        "pixel_positive_fraction": float(segmentation["positive_pixels"] / pixel_count),
        "object_score_min": float(min(object_scores)),
        "object_score_max": float(max(object_scores)),
        "object_score_mean": float(np.mean(object_scores)),
    }


def _mean(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def compute_testc_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen = [row for row in rows if row["partition"] == "seen"]
    unseen = [row for row in rows if row["partition"] == "unseen"]
    if len(seen) != 50 or len(unseen) != 50:
        raise ValueError("Test_C score requires exactly 50 seen and 50 unseen rows")
    seen_macro = _mean(
        seen,
        ["c_auroc", "c_ap", "c_f1max", "p_auroc", "p_ap", "p_f1max"],
    )
    unseen_macro = _mean(
        unseen,
        ["c_auroc", "c_ap", "c_f1max", "p_auroc", "p_ap", "p_f1max"],
    )
    s_cls = float(np.mean([seen_macro["c_auroc"], seen_macro["c_ap"]]))
    s_seg = float(
        np.mean([seen_macro["p_auroc"], seen_macro["p_ap"], seen_macro["p_f1max"]])
    )
    s_zs = float(
        np.mean(
            [
                unseen_macro["c_auroc"],
                unseen_macro["c_ap"],
                unseen_macro["p_auroc"],
                unseen_macro["p_ap"],
                unseen_macro["p_f1max"],
            ]
        )
    )
    total = 100.0 * (
        SCORE_WEIGHTS["classification"] * s_cls
        + SCORE_WEIGHTS["segmentation"] * s_seg
        + SCORE_WEIGHTS["zero_shot"] * s_zs
    )
    return {
        "formula": "100 * (0.3 * S_cls + 0.5 * S_seg + 0.2 * S_zs)",
        "weights": SCORE_WEIGHTS,
        "s_cls": 100.0 * s_cls,
        "s_seg": 100.0 * s_seg,
        "s_zs": 100.0 * s_zs,
        "total_score": total,
        "seen_macro": seen_macro,
        "unseen_macro": unseen_macro,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _diagnostics(
    score: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    submission_result: dict[str, Any],
    recommended_checkpoint_steps: int | None,
    recommended_zero_shot_steps: int | None,
) -> dict[str, Any]:
    seen = [row for row in rows if row["partition"] == "seen"]
    unseen = [row for row in rows if row["partition"] == "unseen"]
    warnings: list[str] = []
    completed = submission_result.get(
        "training_completed_steps", submission_result.get("checkpoint_steps")
    )
    if completed is not None and recommended_checkpoint_steps is not None:
        if int(completed) < recommended_checkpoint_steps:
            warnings.append(
                f"Dinomaly training completed {completed} steps; the protocol "
                f"recommends at least {recommended_checkpoint_steps}"
            )
    zero_completed = submission_result.get("zero_shot_training_completed_steps")
    if zero_completed is not None and recommended_zero_shot_steps is not None:
        if int(zero_completed) < recommended_zero_shot_steps:
            warnings.append(
                f"zero-shot training completed {zero_completed} steps; the protocol "
                f"recommends at least {recommended_zero_shot_steps}"
            )
    return {
        "quality_warnings": warnings,
        "score_contribution_points": {
            "classification": SCORE_WEIGHTS["classification"] * score["s_cls"],
            "segmentation": SCORE_WEIGHTS["segmentation"] * score["s_seg"],
            "zero_shot": SCORE_WEIGHTS["zero_shot"] * score["s_zs"],
        },
        "classification_minus_pixel_ap_points": {
            partition: 100.0 * (
                np.mean([row["c_auroc"] for row in current])
                - np.mean([row["p_ap"] for row in current])
            )
            for partition, current in (("seen", seen), ("unseen", unseen))
        },
        "object_aggregation_auroc": {
            partition: {
                "submitted": float(np.mean([row["c_auroc"] for row in current])),
                "max_view": float(np.mean([row["diag_max_view_auroc"] for row in current])),
                "mean_view": float(np.mean([row["diag_mean_view_auroc"] for row in current])),
            }
            for partition, current in (("seen", seen), ("unseen", unseen))
        },
        "category_standard_deviation": {
            partition: {
                metric: float(np.std([row[metric] for row in current], ddof=1))
                for metric in ("c_auroc", "c_ap", "p_auroc", "p_ap", "p_f1max")
            }
            for partition, current in (("seen", seen), ("unseen", unseen))
        },
    }


def _report(
    score: dict[str, Any], rows: list[dict[str, Any]], diagnostics: dict[str, Any]
) -> str:
    weakest = sorted(rows, key=lambda row: float(row["p_ap"]))[:10]
    lines = [
        "# Test_C evaluation",
        "",
        f"- Total score: {score['total_score']:.4f}",
        f"- S_cls: {score['s_cls']:.4f}",
        f"- S_seg: {score['s_seg']:.4f}",
        f"- S_zs: {score['s_zs']:.4f}",
        "- Protocol: 50 seen + 50 unseen categories; 10 normal + 10 anomalous objects per category.",
        "",
        "## Lowest pixel AP categories",
        "",
        "| Category | Split | C-AUROC | C-AP | P-AUROC | P-AP | P-F1max | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in weakest:
        lines.append(
            "| {category} | {partition} | {c_auroc:.4f} | {c_ap:.4f} | "
            "{p_auroc:.4f} | {p_ap:.4f} | {p_f1max:.4f} | {seconds:.2f} |".format(**row)
        )
    seen = [row for row in rows if row["partition"] == "seen"]
    unseen = [row for row in rows if row["partition"] == "unseen"]
    aggregation = diagnostics["object_aggregation_auroc"]
    contributions = diagnostics["score_contribution_points"]
    lines.extend(
        [
            "",
            "## Bottleneck diagnostics",
            "",
            f"- Weighted score contributions: classification {contributions['classification']:.2f}, "
            f"segmentation {contributions['segmentation']:.2f}, zero-shot {contributions['zero_shot']:.2f} points.",
            f"- Seen classification minus pixel AP: "
            f"{100.0 * (np.mean([r['c_auroc'] for r in seen]) - np.mean([r['p_ap'] for r in seen])):.2f} points.",
            f"- Unseen classification minus pixel AP: "
            f"{100.0 * (np.mean([r['c_auroc'] for r in unseen]) - np.mean([r['p_ap'] for r in unseen])):.2f} points.",
            f"- Seen object AUROC (submitted / max-view / mean-view): "
            f"{aggregation['seen']['submitted']:.4f} / {aggregation['seen']['max_view']:.4f} / {aggregation['seen']['mean_view']:.4f}.",
            f"- Unseen object AUROC (submitted / max-view / mean-view): "
            f"{aggregation['unseen']['submitted']:.4f} / {aggregation['unseen']['max_view']:.4f} / {aggregation['unseen']['mean_view']:.4f}.",
            f"- Mean category evaluation time: {np.mean([r['seconds'] for r in rows]):.2f} seconds.",
            "",
        ]
    )
    if diagnostics["quality_warnings"]:
        lines.extend(["## Quality warnings", ""])
        lines.extend(f"- {warning}" for warning in diagnostics["quality_warnings"])
        lines.append("")
    return "\n".join(lines)


def evaluate_testc_submission(
    *,
    testc_root: Path,
    protocol_path: Path,
    submission_result: dict[str, Any],
    output_dir: Path,
    image_top_ratio: float,
    recommended_checkpoint_steps: int | None = None,
    recommended_zero_shot_steps: int | None = None,
) -> dict[str, Any]:
    testc_root = testc_root.expanduser().resolve()
    protocol = load_testc_protocol(protocol_path.expanduser().resolve())
    manifest_path = testc_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_category[str(sample["category"])].append(sample)
    submission_root = Path(submission_result["submission_root"]).resolve()
    submission_csv = Path(submission_result["submission_csv"]).resolve()
    score_by_group = _read_submission_scores(submission_csv)
    expected_groups = {str(sample["group_folder"]) for sample in samples}
    if set(score_by_group) != expected_groups:
        raise ValueError("Submission object rows do not exactly match Test_C manifest")

    signature = canonical_digest(
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "manifest_sha256": file_sha256(manifest_path),
            "submission_signature": submission_result.get("signature"),
            "image_top_ratio": image_top_ratio,
        }
    )
    run_dir = output_dir.expanduser().resolve() / "testc_evaluation" / signature[:12]
    category_dir = run_dir / "per_category"
    progress_path = run_dir / "eval_progress.jsonl"
    rows: list[dict[str, Any]] = []
    for index, category in enumerate(protocol["categories"], start=1):
        result_path = category_dir / f"{category}.json"
        if result_path.is_file():
            row = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            row = _category_metrics(
                category=category,
                samples=by_category[category],
                score_by_group=score_by_group,
                submission_root=submission_root,
                testc_root=testc_root,
                image_top_ratio=image_top_ratio,
            )
            atomic_write_json(result_path, row)
            append_jsonl(
                progress_path,
                {
                    "timestamp": utc_now(),
                    "event": "category_complete",
                    "index": index,
                    "total_categories": 100,
                    **row,
                },
            )
        rows.append(row)
    rows.sort(key=lambda row: str(row["category"]))
    score = compute_testc_score(rows)
    diagnostics = _diagnostics(
        score,
        rows,
        submission_result=submission_result,
        recommended_checkpoint_steps=recommended_checkpoint_steps,
        recommended_zero_shot_steps=recommended_zero_shot_steps,
    )
    payload = {
        "status": "complete",
        "completed_at": utc_now(),
        "signature": signature,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "submission_result": submission_result,
        "score": score,
        "diagnostics": diagnostics,
        "metrics_per_category": str(run_dir / "metrics_per_category.csv"),
        "report": str(run_dir / "evaluation_report.md"),
    }
    _write_csv(run_dir / "metrics_per_category.csv", rows)
    atomic_write_json(run_dir / "metrics_per_category.json", rows)
    atomic_write_json(run_dir / "metrics_and_score.json", payload)
    (run_dir / "evaluation_report.md").write_text(
        _report(score, rows, diagnostics), encoding="utf-8", newline="\n"
    )
    atomic_write_json(
        output_dir / "testc_evaluation" / "latest.json",
        {
            "signature": signature,
            "run_dir": str(run_dir),
            "metrics": str(run_dir / "metrics_and_score.json"),
            "updated_at": utc_now(),
        },
    )
    return payload
