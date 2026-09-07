from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.clip_normal_prior import (  # noqa: E402
    ClipNormalPrior,
    clip_prior_config_fingerprint,
)
from realiad_dinomaly2.clip_semantic import (  # noqa: E402
    fuse_unseen_anomaly_map,
)
from realiad_dinomaly2.config import load_config  # noqa: E402


def _fuse(
    reconstruction_map: torch.Tensor,
    broken_probability: torch.Tensor,
) -> torch.Tensor:
    return fuse_unseen_anomaly_map(
        reconstruction_map,
        broken_probability,
        reconstruction_gain=1.0,
        semantic_gain=1.0,
        semantic_scale_floor=0.02,
        broken_threshold=0.5,
        upper_quantile=0.995,
        foreground_low_quantile=0.2,
        foreground_high_quantile=0.7,
        foreground_floor=0.0,
        foreground_dilation_kernel=1,
        confidence_power=2.0,
    )


class ClipSemanticFusionTests(unittest.TestCase):
    def test_clip_cannot_create_anomaly_without_dino_support(self) -> None:
        reconstruction = torch.zeros(1, 1, 8, 8)
        broken = torch.ones(1, 1, 2, 2)

        fused = _fuse(reconstruction, broken)

        torch.testing.assert_close(fused, torch.zeros_like(fused))

    def test_confident_clip_adds_evidence_inside_dino_support(self) -> None:
        reconstruction = torch.zeros(1, 1, 8, 8)
        reconstruction[:, :, 2:6, 2:6] = 0.2
        broken = torch.ones(1, 1, 2, 2)

        fused = _fuse(reconstruction, broken)

        self.assertGreater(
            float(fused[:, :, 3:5, 3:5].mean()),
            float(reconstruction[:, :, 3:5, 3:5].mean()),
        )
        self.assertEqual(float(fused[:, :, 0, 0].max()), 0.0)

    def test_uncertain_clip_probability_adds_no_evidence(self) -> None:
        reconstruction = torch.rand(1, 1, 8, 8)
        broken = torch.full((1, 1, 2, 2), 0.5)

        fused = _fuse(reconstruction, broken)

        torch.testing.assert_close(fused, reconstruction)


class ClipNormalPriorTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "evaluation": {
                "unseen_clip": {
                    "normal_prior": {
                        "threshold": 2.0,
                        "temperature": 0.5,
                        "blend": 0.8,
                        "eps": 1e-6,
                    }
                }
            }
        }

    def test_view_global_prior_suppresses_expected_normal_response(self) -> None:
        stats = {
            "median": torch.full((1, 2, 2), 0.6),
            "mad": torch.full((1, 2, 2), 0.05),
        }
        prior = ClipNormalPrior(
            {"metadata": {}, "view_global": {"0": stats, "1": stats}}
        )
        maps = torch.full((1, 2, 1, 2, 2), 0.6)

        calibrated = prior.calibrate(
            maps,
            view_ids=torch.tensor([[0, 1]]),
            valid_view_mask=torch.tensor([[True, True]]),
            config=self._config(),
        )

        self.assertEqual(calibrated.shape, maps.shape)
        self.assertTrue(torch.all(calibrated < maps * 0.25))

    def test_missing_view_prior_leaves_map_unchanged(self) -> None:
        prior = ClipNormalPrior({"metadata": {}, "view_global": {}})
        maps = torch.rand(2, 1, 2, 2)

        calibrated = prior.calibrate(
            maps,
            view_ids=torch.tensor([3, 4]),
            valid_view_mask=None,
            config=self._config(),
        )

        torch.testing.assert_close(calibrated, maps)


class ClipCompetitionConfigTests(unittest.TestCase):
    def test_new_competition_config_uses_trained_zero_shot_route(self) -> None:
        config = load_config(ROOT / "configs" / "competition.yaml")

        self.assertTrue(config["model"]["multi_view"]["enabled"])
        self.assertTrue(config["evaluation"]["normal_prior"]["enabled"])
        self.assertFalse(config["evaluation"]["unseen_clip"]["enabled"])
        self.assertTrue(config["zero_shot"]["enabled"])
        self.assertEqual(config["zero_shot"]["route"], "unseen_only")
        self.assertEqual(
            config["zero_shot"]["training"]["scheduler"], "cosine"
        )
        self.assertEqual(
            config["evaluation"]["unseen_clip"][
                "intermediate_layer_weights"
            ],
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_clip_prior_fingerprint_changes_with_prompts(self) -> None:
        config = load_config(ROOT / "configs" / "competition.yaml")
        before = clip_prior_config_fingerprint(config)
        config["evaluation"]["unseen_clip"]["broken_prompts"].append(
            "a deliberately different defect prompt"
        )

        after = clip_prior_config_fingerprint(config)

        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
