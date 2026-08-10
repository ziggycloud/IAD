from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.information_density_model import (
    DifficultyRoutedMoE,
    InformationDensityDinomaly,
    InformationDensityDownProjection,
    calibrate_information_density_map,
)
from realiad_dinomaly2.modeling import (
    ModelBundle,
    build_optimizer,
    forward_with_regularization,
    load_trainable_state_dict,
    trainable_state_dict,
)


class _TinyDinomaly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = nn.ModuleList(
            [
                InformationDensityDownProjection(
                    input_dim=8,
                    latent_dim=6,
                    num_special_tokens=2,
                    dropout=0.0,
                    hidden_dim=4,
                    channel_widths=(2, 4, 6),
                    channel_thresholds=(0.33, 0.66),
                    capacity_warmup_steps=3,
                    capacity_ramp_steps=2,
                    initial_expected_error=0.1,
                ),
                nn.Linear(6, 8),
            ]
        )
        self.decoder = nn.ModuleList([nn.Linear(8, 8)])

    def init_weights(self) -> None:
        for module in self.bottleneck.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        target = tokens[:, 2:].transpose(1, 2).reshape(-1, 8, 4, 4)
        reconstructed = tokens
        for module in self.bottleneck:
            reconstructed = module(reconstructed)
        decoded = reconstructed[:, 2:].transpose(1, 2).reshape(-1, 8, 4, 4)
        return [target], [decoded]


class _TinyMoEDinomaly(_TinyDinomaly):
    def __init__(self) -> None:
        super().__init__()
        self.bottleneck = nn.ModuleList(
            [
                InformationDensityDownProjection(
                    input_dim=8,
                    latent_dim=6,
                    num_special_tokens=2,
                    dropout=0.0,
                    hidden_dim=4,
                    channel_widths=(2, 4, 6),
                    channel_thresholds=(0.33, 0.66),
                    capacity_warmup_steps=3,
                    capacity_ramp_steps=2,
                    initial_expected_error=0.1,
                    emit_routing=True,
                ),
                DifficultyRoutedMoE(
                    latent_dim=6,
                    output_dim=8,
                    num_special_tokens=2,
                    dropout=0.0,
                    expert_input_widths=(2, 4, 6),
                    expert_hidden_dims=(4, 6, 8),
                    routing_centers=(0.17, 0.5, 0.83),
                    routing_temperature=0.1,
                    routing_warmup_steps=3,
                    routing_ramp_steps=2,
                    hard_routing_start_step=5,
                    target_load=(0.6, 0.3, 0.1),
                ),
            ]
        )


class InformationDensityTests(unittest.TestCase):
    def _make_model(self) -> InformationDensityDinomaly:
        model = InformationDensityDinomaly(
            _TinyDinomaly(), calibration_blend=0.35
        )
        model.init_weights()
        return model

    def _make_moe_model(self) -> InformationDensityDinomaly:
        model = InformationDensityDinomaly(
            _TinyMoEDinomaly(), calibration_blend=0.35
        )
        model.init_weights()
        return model

    def test_center_token_cannot_raise_its_own_difficulty(self) -> None:
        model = self._make_model().eval()
        tokens = torch.randn(1, 18, 8)
        model(tokens)
        before = model.down_projection.expected_error_map()[0, 0, 1, 1]
        changed = tokens.clone()
        changed[:, 2 + 5] += 100.0
        model(changed)
        after = model.down_projection.expected_error_map()[0, 0, 1, 1]
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=0.0))

    def test_warmup_preserves_full_original_capacity(self) -> None:
        model = self._make_model().train()
        model.set_training_step(0)
        model(torch.randn(2, 18, 8))
        mask = model.down_projection._last_channel_mask
        assert mask is not None
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))

        model.eval()
        model(torch.randn(2, 18, 8))
        active = model.down_projection._last_channel_mask
        assert active is not None
        self.assertTrue(
            torch.equal(active[..., :2], torch.ones_like(active[..., :2]))
        )
        self.assertTrue(bool((active[:, 2:, 4:] < 1.0).any()))

    def test_multiscale_context_measures_local_complexity(self) -> None:
        model = self._make_model()
        projection = model.down_projection
        constant = torch.ones(1, 16, 8)
        _, constant_variance = projection._neighbor_statistics(
            constant, side=4, kernel_size=3
        )
        heterogeneous = constant.clone()
        heterogeneous[:, ::2] = -1.0
        _, heterogeneous_variance = projection._neighbor_statistics(
            heterogeneous, side=4, kernel_size=3
        )
        self.assertTrue(
            torch.allclose(
                constant_variance, torch.zeros_like(constant_variance)
            )
        )
        self.assertGreater(float(heterogeneous_variance.mean()), 0.0)

        model.train()
        model(torch.randn(2, 18, 8))
        auxiliary = model.auxiliary_losses(detach=True)
        self.assertIn("local_complexity_mean", auxiliary)
        self.assertIn("context_scale_disagreement", auxiliary)

    def test_auxiliary_supervision_reaches_estimator(self) -> None:
        model = self._make_model().train()
        encoder, decoder, regularizer, auxiliary = forward_with_regularization(
            model,
            torch.randn(2, 18, 8),
            weights={
                "difficulty_prediction": 1.0,
                "difficulty_budget": 0.1,
                "difficulty_smoothness": 0.01,
            },
        )
        reconstruction = 1.0 - F.cosine_similarity(
            encoder[0].flatten(1), decoder[0].flatten(1), dim=1
        ).mean()
        (reconstruction + 0.1 * regularizer).backward()
        final = model.down_projection.estimator[-1]
        assert isinstance(final, nn.Linear)
        self.assertIsNotNone(final.weight.grad)
        self.assertTrue(torch.isfinite(final.weight.grad).all())
        self.assertIn("difficulty_prediction", auxiliary)
        self.assertIn("capacity_high_usage", auxiliary)

    def test_calibration_is_bounded_and_checkpoint_contains_estimator(self) -> None:
        model = self._make_model().eval()
        model(torch.randn(2, 18, 8))
        raw = torch.full((2, 1, 4, 4), 0.5)
        calibrated = calibrate_information_density_map(model, raw)
        self.assertTrue(torch.all(calibrated >= 0.0))
        self.assertTrue(torch.all(calibrated <= raw))
        keys = set(model.down_projection.state_dict())
        self.assertIn("estimator.2.weight", keys)
        self.assertIn("complexity_encoder.0.weight", keys)
        self.assertIn("residual_mean", keys)
        self.assertIn("residual_variance", keys)

    def test_optimizer_and_checkpoint_cover_every_density_parameter(self) -> None:
        model = self._make_model()
        base = model.base_model
        bundle = ModelBundle(
            model=model,
            bottleneck=base.bottleneck,
            decoder=base.decoder,
            embed_dim=8,
            backbone_name="tiny",
            backbone_weights_path=Path("tiny.pt"),
            backbone_sha256="tiny-sha",
        )
        config = {
            "training": {
                "learning_rate": 3e-4,
                "first_bottleneck_lr_scale": 0.2,
                "adam_betas": [0.9, 0.999],
                "weight_decay": 1e-4,
                "adam_epsilon": 1e-8,
                "optimizer": {"type": "adamw", "amsgrad": False},
            }
        }
        optimizer = build_optimizer(bundle, config)
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            [
                "bottleneck_first",
                "difficulty_estimator",
                "bottleneck_rest",
                "decoder",
            ],
        )
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimized, expected)

        model(torch.randn(2, 18, 8))
        saved = copy.deepcopy(trainable_state_dict(bundle))
        final = model.down_projection.estimator[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.fill_(99.0)
            model.down_projection.residual_mean.fill_(99.0)
        load_trainable_state_dict(bundle, saved)
        self.assertFalse(torch.all(final.weight == 99.0))
        self.assertNotEqual(
            float(model.down_projection.residual_mean), 99.0
        )

    def test_three_experts_follow_low_mid_high_difficulty(self) -> None:
        model = self._make_moe_model()
        moe = model.difficulty_moe
        assert moe is not None
        difficulty = torch.tensor([[[0.05], [0.5], [0.95]]])
        assignment = moe.routing_probabilities(difficulty).argmax(dim=-1)
        self.assertEqual(assignment.tolist(), [[0, 1, 2]])

    def test_moe_warmup_uses_original_high_capacity_path(self) -> None:
        model = self._make_moe_model().train()
        moe = model.difficulty_moe
        assert moe is not None
        model.set_training_step(0)
        latent = torch.randn(2, 6, 6)
        difficulty = torch.rand(2, 4, 1)
        actual = moe((latent, difficulty))
        expected = moe.experts[-1](latent)
        self.assertTrue(torch.allclose(actual, expected))

    def test_distillation_pretrains_low_and_mid_experts(self) -> None:
        model = self._make_moe_model().train()
        moe = model.difficulty_moe
        assert moe is not None
        model.set_training_step(0)
        latent = torch.randn(2, 6, 6)
        difficulty = torch.rand(2, 4, 1)
        moe((latent, difficulty))
        auxiliary = moe.auxiliary_losses(detach=False)
        distillation = auxiliary["moe_expert_distillation"]
        self.assertGreater(float(distillation.detach()), 0.0)
        distillation.backward()
        for expert_index in (0, 1):
            first = moe.experts[expert_index].network[0]
            assert isinstance(first, nn.Linear)
            self.assertIsNotNone(first.weight.grad)
            self.assertGreater(float(first.weight.grad.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        model.set_training_step(3)
        moe((latent, difficulty))
        ramp_start = moe.auxiliary_losses(
            detach=True
        )["moe_expert_distillation"]
        model.set_training_step(4)
        moe((latent, difficulty))
        ramp_middle = moe.auxiliary_losses(
            detach=True
        )["moe_expert_distillation"]
        self.assertTrue(torch.allclose(ramp_middle, 0.5 * ramp_start))

        model.set_training_step(5)
        moe((latent, difficulty))
        sparse_distillation = moe.auxiliary_losses(
            detach=True
        )["moe_expert_distillation"]
        self.assertEqual(float(sparse_distillation), 0.0)

    def test_hard_routing_dispatches_only_selected_expert(self) -> None:
        model = self._make_moe_model().eval()
        moe = model.difficulty_moe
        assert moe is not None
        with torch.no_grad():
            for expert_index, expert in enumerate(moe.experts):
                for parameter in expert.parameters():
                    parameter.zero_()
                final = expert.network[3]
                assert isinstance(final, nn.Linear)
                final.bias.fill_(float(expert_index + 1))
        latent = torch.randn(1, 5, 6)
        difficulty = torch.tensor([[[0.05], [0.5], [0.95]]])
        reconstructed = moe((latent, difficulty))
        self.assertTrue(torch.equal(reconstructed[:, :2], torch.full_like(
            reconstructed[:, :2], 3.0
        )))
        self.assertEqual(reconstructed[0, 2:, 0].tolist(), [1.0, 2.0, 3.0])

    def test_reconstruction_cannot_manipulate_moe_difficulty_router(self) -> None:
        model = self._make_moe_model().train()
        model.set_training_step(4)
        encoder, decoder = model(torch.randn(2, 18, 8))
        reconstruction = 1.0 - F.cosine_similarity(
            encoder[0].flatten(1), decoder[0].flatten(1), dim=1
        ).mean()
        reconstruction.backward()
        final = model.down_projection.estimator[-1]
        assert isinstance(final, nn.Linear)
        self.assertIsNone(final.weight.grad)

        model.zero_grad(set_to_none=True)
        _, _, regularizer, auxiliary = forward_with_regularization(
            model,
            torch.randn(2, 18, 8),
            weights={
                "difficulty_prediction": 1.0,
                "moe_load_balance": 0.1,
                "moe_route_entropy": 0.01,
            },
        )
        regularizer.backward()
        self.assertIsNotNone(final.weight.grad)
        self.assertIn("moe_load_balance", auxiliary)
        self.assertIn("moe_hard_high_usage", auxiliary)

    def test_moe_checkpoint_contains_all_three_experts(self) -> None:
        model = self._make_moe_model()
        keys = set(model.state_dict())
        for expert_index in range(3):
            self.assertIn(
                "base_model.bottleneck.1.experts."
                f"{expert_index}.network.0.weight",
                keys,
            )


if __name__ == "__main__":
    unittest.main()
