from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.clip_fusion import (  # noqa: E402
    build_prompt_ensemble,
    fit_clip_fusion,
    fuse_anomaly_evidence,
)


class ClipFusionTests(unittest.TestCase):
    def test_prompt_ensemble_is_binary_category_aware_and_fine_grained(self) -> None:
        prompts = build_prompt_ensemble("PLCC_socket")
        self.assertEqual(set(prompts), {"normal", "anomalous"})
        self.assertGreater(len(prompts["normal"]), 10)
        self.assertGreater(len(prompts["anomalous"]), 10)
        self.assertTrue(
            all(
                "plcc socket" in prompt
                for values in prompts.values()
                for prompt in values
            )
        )
        self.assertTrue(any("twisted" in value for value in prompts["anomalous"]))
        self.assertTrue(any("missing" in value for value in prompts["anomalous"]))

    def test_semantic_evidence_suppresses_normal_reconstruction_response(self) -> None:
        reconstruction = torch.full((1, 2, 2), 0.4)
        semantic_normal = torch.zeros_like(reconstruction)
        semantic_anomaly = torch.full_like(reconstruction, 0.4)
        stats = {"median": 0.0, "shoulder": 0.2, "tail": 0.4}
        normal_fused = fuse_anomaly_evidence(
            reconstruction,
            semantic_normal,
            stats,
            stats,
            reconstruction_floor=0.35,
            semantic_weight=0.3,
            agreement_weight=0.35,
            gate_temperature=0.5,
            max_normalized_score=4.0,
            eps=1e-6,
        )
        anomaly_fused = fuse_anomaly_evidence(
            reconstruction,
            semantic_anomaly,
            stats,
            stats,
            reconstruction_floor=0.35,
            semantic_weight=0.3,
            agreement_weight=0.35,
            gate_temperature=0.5,
            max_normalized_score=4.0,
            eps=1e-6,
        )
        self.assertLess(float(normal_fused.mean()), 0.5)
        self.assertGreater(float(anomaly_fused.mean()), 1.4)

    def test_semantic_only_branch_can_rescue_a_weak_reconstruction_signal(self) -> None:
        stats = {"median": 0.0, "shoulder": 0.2, "tail": 0.4}
        fused = fuse_anomaly_evidence(
            torch.zeros(1, 2, 2),
            torch.full((1, 2, 2), 0.4),
            stats,
            stats,
            reconstruction_floor=0.35,
            semantic_weight=0.3,
            agreement_weight=0.35,
            gate_temperature=0.5,
            max_normalized_score=4.0,
            eps=1e-6,
        )
        self.assertGreater(float(fused.mean()), 0.29)

    def test_fitter_is_hardwired_to_competition_train(self) -> None:
        source = inspect.getsource(fit_clip_fusion)
        self.assertIn("build_competition_train_dataset", source)
        self.assertIn('dataset_config["train_dir"]', source)
        self.assertNotIn('dataset_config["test_dir"]', source)


if __name__ == "__main__":
    unittest.main()
