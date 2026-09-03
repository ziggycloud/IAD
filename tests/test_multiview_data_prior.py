from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.config import config_fingerprint
from realiad_dinomaly2.data import Record, group_records_by_object
from realiad_dinomaly2.normal_prior import (
    NORMAL_PRIOR_FORMAT_VERSION,
    NormalPrior,
    _normal_prior_fingerprint,
    file_sha256,
    fit_normal_prior,
    validate_normal_prior,
)


def record(object_id: str, view: int) -> Record:
    return Record(
        category="part",
        anomaly_class="OK",
        image_path=f"OK/{object_id}/image_C{view}.png",
        mask_path=None,
    )


def minimal_config() -> dict:
    return {
        "experiment": {"seed": 7, "output_dir": "unused"},
        "dataset": {
            "type": "realiad_variety",
            "json_dir": "json",
            "image_dir": "images",
            "train_image_dir": None,
            "categories": ["seen"],
            "category_limit": None,
            "image_size": 28,
            "crop_size": 28,
            "train_mode": "all_views",
            "image_label_policy": "visible_defect",
            "missing_anomaly_mask_policy": "include_as_normal_view",
            "mask_resize_semantics": "nearest_binary",
        },
        "model": {
            "architecture": "category_generalized",
            "multi_view": {
                "enabled": True,
                "num_views": 5,
                "context_dim": 12,
            },
        },
        "training": {
            "total_steps": 1,
            "effective_batch_size": 1,
            "amp": False,
            "amp_dtype": "float32",
            "learning_rate": 1e-3,
            "first_bottleneck_lr_scale": 1.0,
            "weight_decay": 0.0,
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-8,
            "loose_loss_warmup_steps": 0,
            "loose_loss_final_discard": 0.0,
            "generalized_regularization_weight": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "evaluation": {
            "normal_prior": {
                "enabled": True,
                "resolution": "patch",
                "category_view_enabled": True,
                "unseen_fallback": "view_global",
                "statistic": "median_mad",
                "threshold": 2.0,
                "temperature": 0.5,
                "blend": 0.8,
                "eps": 1e-6,
            }
        },
        "runtime": {"pin_memory": False},
    }


class MultiViewGroupingTests(unittest.TestCase):
    def test_grouping_is_object_then_zero_based_camera_order(self) -> None:
        records = [
            record("S0002", 5),
            record("S0001", 3),
            record("S0001", 1),
            record("S0002", 1),
            record("S0001", 5),
            record("S0001", 2),
            record("S0002", 4),
            record("S0001", 4),
            record("S0002", 3),
            record("S0002", 2),
        ]
        groups = group_records_by_object(records)
        self.assertEqual([name for name, _ in groups], ["OK/S0001", "OK/S0002"])
        first_records = [records[index].view_id for index in groups[0][1]]
        second_records = [records[index].view_id for index in groups[1][1]]
        self.assertEqual(first_records, [1, 2, 3, 4, 5])
        self.assertEqual(second_records, [1, 2, 3, 4, 5])

    def test_duplicate_missing_and_out_of_range_views_are_explicit(self) -> None:
        complete = [record("S0001", view) for view in range(1, 6)]
        with self.assertRaisesRegex(ValueError, "duplicate view 0"):
            group_records_by_object(complete + [record("S0001", 1)])
        with self.assertRaisesRegex(ValueError, "missing views \[4\]"):
            group_records_by_object(complete[:-1], missing_view_policy="error")
        padded = group_records_by_object(
            complete[:-1], missing_view_policy="pad_and_mask"
        )
        self.assertIsNone(padded[0][1][4])
        with self.assertRaisesRegex(ValueError, "no usable objects"):
            group_records_by_object(
                complete[:-1], missing_view_policy="drop_incomplete"
            )
        invalid = Record("part", "OK", "OK/S0001/no_camera.png", None)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            group_records_by_object([invalid])


class NormalPriorTests(unittest.TestCase):
    def test_unseen_category_falls_back_to_view_global_and_preserves_order(self) -> None:
        config = minimal_config()
        stats = {
            "median": torch.zeros(1, 2, 2),
            "mad": torch.ones(1, 2, 2),
        }
        prior = NormalPrior(
            {
                "metadata": {},
                "category_view": {"seen": {"0": stats}},
                "view_global": {str(view): stats for view in range(5)},
            }
        )
        raw = torch.arange(1.0, 5.0).reshape(1, 1, 1, 2, 2).repeat(
            1, 5, 1, 1, 1
        )
        calibrated = prior.calibrate(
            raw,
            categories=["unseen"],
            view_ids=torch.arange(5),
            valid_view_mask=torch.ones(1, 5, dtype=torch.bool),
            config=config,
        )
        flattened = calibrated[0, 0, 0].flatten()
        self.assertTrue(torch.all(flattened[1:] > flattened[:-1]))
        self.assertTrue(torch.all(calibrated > 0))
        self.assertTrue(torch.all(calibrated <= raw))

    def test_single_view_calibration_accepts_batched_one_dimensional_masks(self) -> None:
        config = minimal_config()
        stats = {
            "median": torch.zeros(1, 2, 2),
            "mad": torch.ones(1, 2, 2),
        }
        prior = NormalPrior(
            {
                "metadata": {},
                "category_view": {},
                "view_global": {"0": stats, "1": stats},
            }
        )
        raw = torch.ones(2, 1, 2, 2)
        calibrated = prior.calibrate(
            raw,
            categories=["a", "b"],
            view_ids=torch.tensor([0, 1]),
            valid_view_mask=torch.ones(2, dtype=torch.bool),
            config=config,
        )
        self.assertEqual(calibrated.shape, raw.shape)
        self.assertTrue(torch.all(calibrated > 0))

    def test_checkpoint_fingerprint_mismatch_is_rejected(self) -> None:
        config = minimal_config()
        checkpoint = ROOT / "outputs" / f"unittest_prior_{os.getpid()}.pt"
        try:
            checkpoint.write_bytes(b"checkpoint-a")
            payload = {
                "metadata": {
                    "format_version": NORMAL_PRIOR_FORMAT_VERSION,
                    "config_fingerprint": config_fingerprint(config),
                    "normal_prior_fingerprint": _normal_prior_fingerprint(config),
                    "checkpoint_sha256": "not-the-checkpoint",
                    "source_split": "Train",
                    "source_labels": "normal_only",
                }
            }
            with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
                validate_normal_prior(payload, config, checkpoint)
            payload["metadata"]["checkpoint_sha256"] = file_sha256(checkpoint)
            validate_normal_prior(payload, config, checkpoint)
        finally:
            checkpoint.unlink(missing_ok=True)

    def test_prior_fitter_is_hardwired_to_train_dataset(self) -> None:
        source = inspect.getsource(fit_normal_prior)
        self.assertIn("build_train_dataset", source)
        self.assertIn("build_competition_train_dataset", source)
        self.assertIn('dataset_config["train_dir"]', source)
        self.assertNotIn('dataset_config["test_dir"]', source)
        self.assertNotIn("RealIADVarietyDataset", source)


if __name__ == "__main__":
    unittest.main()
