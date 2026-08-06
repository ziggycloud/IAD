from __future__ import annotations

import copy
import inspect
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.generalized_model import (
    CategoryFreeRouter,
    CompositionalNormalityAdapter,
    CompositionalReferenceBank,
    GeneralizedDinomaly,
    MultiViewContextEncoder,
    MultiViewGeneralizedDinomaly,
)
from realiad_dinomaly2.losses import anomaly_map, reconstruction_loss
from realiad_dinomaly2.modeling import (
    ModelBundle,
    auxiliary_losses,
    forward_with_regularization,
    load_trainable_state_dict,
    regularization_loss,
    trainable_state_dict,
)
from realiad_dinomaly2.train_engine import BatchChoice, save_checkpoint


class ReferenceAndRouterTests(unittest.TestCase):
    def test_reference_bank_preserves_token_shape(self) -> None:
        bank = CompositionalReferenceBank(
            dim=16,
            num_references=12,
            top_k=4,
            temperature=0.2,
        )
        tokens = torch.randn(3, 18, 16, requires_grad=True)
        reconstructed = bank(tokens)
        self.assertEqual(reconstructed.shape, tokens.shape)
        auxiliary = bank.auxiliary_losses()
        self.assertEqual(
            set(auxiliary),
            {"reference_balance", "reference_assignment_entropy"},
        )
        self.assertTrue(all(value.ndim == 0 for value in auxiliary.values()))
        reconstructed.square().mean().backward()
        self.assertIsNotNone(bank.references.grad)

    def test_default_router_is_dense_soft_and_has_no_category_argument(self) -> None:
        router = CategoryFreeRouter(
            dim=16,
            num_experts=3,
            temperature=0.7,
        )
        routes = router(torch.randn(4, 16, 16))
        self.assertEqual(routes.shape, (4, 3))
        self.assertTrue(torch.allclose(routes.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.equal((routes > 0).sum(dim=1), torch.full((4,), 3)))
        self.assertNotIn("category", inspect.signature(router.forward).parameters)
        self.assertNotIn("category_id", inspect.signature(router.forward).parameters)


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + 0.01 * self.projection(tokens)


class _DecoderBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attn_mask
        return tokens + self.projection(tokens)


class _TinyEncoder(nn.Module):
    num_register_tokens = 1

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(3, dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([_EncoderBlock(dim), _EncoderBlock(dim)])

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        patches = F.adaptive_avg_pool2d(images, output_size=(4, 4))
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.patch_projection(patches)
        special = torch.cat([self.class_token, self.register_token], dim=1)
        special = special.expand(images.shape[0], -1, -1)
        return torch.cat([special, patches], dim=1)


class GeneralizedModelContractTests(unittest.TestCase):
    def _make_model(self) -> GeneralizedDinomaly:
        embed_dim = 24
        latent_dim = 12
        bottleneck = nn.ModuleList(
            [
                nn.Linear(embed_dim, latent_dim),
                CompositionalNormalityAdapter(
                    latent_dim=latent_dim,
                    output_dim=embed_dim,
                    num_special_tokens=2,
                    num_references=10,
                    reference_top_k=4,
                    router_top_k=3,
                    dropout=0.0,
                ),
            ]
        )
        model = GeneralizedDinomaly(
            encoder=_TinyEncoder(embed_dim),
            bottleneck=bottleneck,
            decoder=nn.ModuleList(
                [_DecoderBlock(embed_dim), _DecoderBlock(embed_dim)]
            ),
            target_layers=[0, 1],
            fuse_layer_encoder=[[0], [1]],
            fuse_layer_decoder=[[0], [1]],
            fuse_layer_bottleneck=[0, 1],
            context_aware_recenter=True,
        )
        model.init_weights()
        return model

    def _make_multi_view_model(self) -> MultiViewGeneralizedDinomaly:
        embed_dim = 24
        latent_dim = 12
        context = MultiViewContextEncoder(
            dim=latent_dim,
            num_views=5,
            num_layers=1,
            num_heads=3,
            trim_ratio=0.1,
            view_dropout_probability=0.4,
            dropout=0.0,
        )
        bottleneck = nn.ModuleList(
            [
                nn.Linear(embed_dim, latent_dim),
                CompositionalNormalityAdapter(
                    latent_dim=latent_dim,
                    output_dim=embed_dim,
                    num_special_tokens=2,
                    num_references=10,
                    reference_top_k=4,
                    router_top_k=3,
                    dropout=0.0,
                    multi_view_conditioning=True,
                ),
            ]
        )
        model = MultiViewGeneralizedDinomaly(
            encoder=_TinyEncoder(embed_dim),
            bottleneck=bottleneck,
            decoder=nn.ModuleList(
                [_DecoderBlock(embed_dim), _DecoderBlock(embed_dim)]
            ),
            target_layers=[0, 1],
            fuse_layer_encoder=[[0], [1]],
            fuse_layer_decoder=[[0], [1]],
            fuse_layer_bottleneck=[0, 1],
            context_aware_recenter=True,
            multi_view_context_encoder=context,
        )
        model.init_weights()
        return model

    def test_forward_matches_existing_loss_contract(self) -> None:
        model = self._make_model()
        images = torch.randn(2, 3, 8, 8)
        self.assertEqual(float(regularization_loss(model).detach()), 0.0)

        encoder_features, decoder_features = model(images)
        self.assertEqual(len(encoder_features), 2)
        self.assertEqual(len(decoder_features), 2)
        for encoder_feature, decoder_feature in zip(
            encoder_features, decoder_features, strict=True
        ):
            self.assertEqual(encoder_feature.shape, (2, 24, 4, 4))
            self.assertEqual(decoder_feature.shape, encoder_feature.shape)

        reconstruction = reconstruction_loss(
            encoder_features,
            decoder_features,
            discard_rate=0.0,
            loose_loss=False,
        )
        auxiliary = regularization_loss(model)
        self.assertTrue(torch.isfinite(reconstruction))
        self.assertTrue(torch.isfinite(auxiliary))
        detached_auxiliary = auxiliary_losses(model, detach=True)
        self.assertEqual(
            set(detached_auxiliary),
            {
                "reference_balance",
                "reference_assignment_entropy",
                "router_balance",
                "router_entropy_penalty",
                "expert_diversity",
            },
        )
        self.assertGreaterEqual(
            float(detached_auxiliary["router_entropy_penalty"]), 0.0
        )
        (reconstruction + 0.01 * auxiliary).backward()
        adapter = model.bottleneck[1]
        assert isinstance(adapter, CompositionalNormalityAdapter)
        self.assertIsNotNone(adapter.reference_bank.references.grad)
        self.assertIsNotNone(adapter.router.network[-1].weight.grad)
        for expert in adapter.experts:
            self.assertTrue(
                any(parameter.grad is not None for parameter in expert.parameters())
            )

    def test_forward_can_transport_auxiliary_losses_for_data_parallel(self) -> None:
        model = self._make_model()
        images = torch.randn(2, 3, 8, 8)
        default_output = model(images)
        self.assertEqual(len(default_output), 2)

        encoder_features, decoder_features, packed = model(
            images,
            return_auxiliary=True,
        )
        self.assertEqual(len(encoder_features), 2)
        self.assertEqual(len(decoder_features), 2)
        self.assertTrue(packed)
        self.assertTrue(all(value.shape == (1,) for value in packed.values()))

        _, _, regularizer, reduced = forward_with_regularization(model, images)
        self.assertEqual(regularizer.ndim, 0)
        self.assertTrue(torch.isfinite(regularizer))
        self.assertTrue(all(value.ndim == 0 for value in reduced.values()))

    def test_init_weights_resets_new_spatial_and_reference_parameters(self) -> None:
        model = self._make_model()
        adapter = model.bottleneck[1]
        assert isinstance(adapter, CompositionalNormalityAdapter)
        spatial = adapter.experts[0]
        depthwise = spatial.depthwise
        with torch.no_grad():
            adapter.reference_bank.references.fill_(123.0)
            depthwise.weight.fill_(123.0)
        model.init_weights()
        self.assertFalse(
            torch.all(adapter.reference_bank.references == 123.0)
        )
        self.assertFalse(torch.all(depthwise.weight == 123.0))

    def test_checkpoint_state_contains_every_new_component(self) -> None:
        model = self._make_model()
        state_keys = set(model.bottleneck.state_dict())
        expected_prefixes = {
            "1.reference_bank.",
            "1.router.",
            "1.experts.0.",
            "1.experts.1.",
            "1.experts.2.",
            "1.output_projection.",
        }
        for prefix in expected_prefixes:
            self.assertTrue(
                any(key.startswith(prefix) for key in state_keys),
                msg=f"checkpoint is missing {prefix}",
            )

    def test_multi_view_checkpoint_restores_module_optimizer_and_scheduler(self) -> None:
        model = self._make_multi_view_model()
        context = model.multi_view_context_encoder
        assert context is not None
        bundle = ModelBundle(
            model=model,
            bottleneck=model.bottleneck,
            decoder=model.decoder,
            embed_dim=24,
            backbone_name="tiny",
            backbone_weights_path=Path("tiny.pt"),
            backbone_sha256="tiny-sha",
            multi_view_context=context,
        )
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )

        class Scaler:
            @staticmethod
            def state_dict():
                return {"scale": 1.0}

        config = {
            "experiment": {"seed": 3},
            "dataset": {
                "type": "realiad_variety",
                "json_dir": "json",
                "image_dir": "images",
                "categories": ["part"],
                "image_size": 8,
                "crop_size": 8,
            },
            "model": {
                "architecture": "category_generalized",
                "multi_view": {"enabled": True, "num_views": 5},
            },
            "training": {
                "total_steps": 10,
                "effective_batch_size": 2,
                "scheduler": {"type": "cosine", "warmup_steps": 2},
            },
        }
        checkpoint_path = (
            ROOT / "outputs" / f"unittest_multiview_checkpoint_{os.getpid()}.pt"
        )
        try:
            save_checkpoint(
                checkpoint_path,
                bundle,
                optimizer,
                Scaler(),
                config,
                completed_steps=4,
                batch_choice=BatchChoice(1, 2, 2, [], 1),
            )
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertIn("multi_view_context", payload["model"])
            self.assertIn("optimizer", payload)
            self.assertEqual(payload["scheduler"]["completed_steps"], 4)
            saved = copy.deepcopy(payload["model"])
            assert context.view_embedding is not None
            with torch.no_grad():
                context.view_embedding.weight.fill_(99.0)
            load_trainable_state_dict(bundle, saved)
            self.assertFalse(
                torch.all(context.view_embedding.weight.detach() == 99.0)
            )
            restored_optimizer = torch.optim.Adam(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=1e-3,
            )
            restored_optimizer.load_state_dict(payload["optimizer"])
            self.assertEqual(
                len(restored_optimizer.param_groups),
                len(optimizer.param_groups),
            )
        finally:
            checkpoint_path.unlink(missing_ok=True)

    def test_inference_signature_has_no_category_id(self) -> None:
        model = self._make_model()
        parameters = inspect.signature(model.forward).parameters
        self.assertNotIn("category", parameters)
        self.assertNotIn("category_id", parameters)

    def test_true_multi_view_shape_and_independent_maps(self) -> None:
        model = self._make_multi_view_model().eval()
        images = torch.randn(2, 5, 3, 8, 8)
        view_ids = torch.arange(5).expand(2, -1)
        valid = torch.ones(2, 5, dtype=torch.bool)
        encoder_features, decoder_features = model(
            images,
            view_ids=view_ids,
            valid_view_mask=valid,
        )
        for encoder_feature, decoder_feature in zip(
            encoder_features, decoder_features, strict=True
        ):
            self.assertEqual(encoder_feature.shape, (2, 5, 24, 4, 4))
            self.assertEqual(decoder_feature.shape, encoder_feature.shape)
        maps = anomaly_map(encoder_features, decoder_features, output_size=8)
        self.assertEqual(maps.shape, (2, 5, 1, 8, 8))
        self.assertTrue(torch.isfinite(maps).all())

    def test_valid_mask_blocks_missing_views(self) -> None:
        context = MultiViewContextEncoder(
            dim=12,
            num_views=5,
            num_layers=1,
            num_heads=3,
            view_dropout_probability=0.0,
            dropout=0.0,
        ).eval()
        tokens = torch.randn(2, 5, 16, 12)
        valid = torch.tensor(
            [[True, True, True, True, False], [True, False, True, False, True]]
        )
        output = context(tokens, torch.arange(5), valid)
        self.assertTrue(torch.equal(output.visibility_weights[~valid], torch.zeros(3)))
        self.assertTrue(
            torch.equal(output.cross_view_context[~valid], torch.zeros(3, 12))
        )
        invalid_keys = (~valid)[:, None, None, :].expand_as(
            output.attention_weights
        )
        self.assertTrue(
            torch.allclose(
                output.attention_weights.masked_select(invalid_keys),
                torch.zeros_like(
                    output.attention_weights.masked_select(invalid_keys)
                ),
            )
        )

    def test_view_ids_select_the_corresponding_embedding(self) -> None:
        context = MultiViewContextEncoder(
            dim=12,
            num_views=5,
            num_layers=1,
            num_heads=3,
            view_dropout_probability=0.0,
            dropout=0.0,
        ).eval()
        assert context.view_embedding is not None
        ids = torch.tensor([4, 3, 2, 1, 0])
        selected = context.view_embedding(ids)
        self.assertTrue(
            torch.equal(selected[0], context.view_embedding.weight[4])
        )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            context(torch.randn(1, 5, 4, 12), torch.tensor([0, 1, 2, 3, 5]))

    def test_view_dropout_consistency_is_differentiable(self) -> None:
        torch.manual_seed(7)
        model = self._make_multi_view_model().train()
        images = torch.randn(2, 5, 3, 8, 8)
        valid = torch.ones(2, 5, dtype=torch.bool)
        encoder_features, decoder_features, packed = model(
            images,
            view_ids=torch.arange(5),
            valid_view_mask=valid,
            return_auxiliary=True,
        )
        loss = reconstruction_loss(
            encoder_features,
            decoder_features,
            discard_rate=0.0,
            loose_loss=False,
            valid_view_mask=valid,
        ) + packed["context_consistency"].mean()
        loss.backward()
        context = model.multi_view_context_encoder
        assert context is not None and context.view_embedding is not None
        self.assertIsNotNone(context.view_embedding.weight.grad)
        self.assertTrue(torch.isfinite(context.view_embedding.weight.grad).all())

    def test_single_view_signal_is_not_hard_zeroed(self) -> None:
        model = self._make_multi_view_model().eval()
        images = torch.zeros(1, 5, 3, 8, 8)
        images[:, 2] = 5.0
        encoder_features, decoder_features = model(
            images,
            view_ids=torch.arange(5),
            valid_view_mask=torch.ones(1, 5, dtype=torch.bool),
        )
        maps = anomaly_map(encoder_features, decoder_features, output_size=8)
        self.assertGreater(float(maps[:, 2].detach().abs().max()), 0.0)

    def test_multi_view_auxiliary_contract_is_data_parallel_gatherable(self) -> None:
        model = self._make_multi_view_model().train()
        _, _, regularizer, auxiliary = forward_with_regularization(
            model,
            torch.randn(2, 5, 3, 8, 8),
            view_ids=torch.arange(5).expand(2, -1),
            valid_view_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        self.assertEqual(regularizer.ndim, 0)
        self.assertTrue(
            {
                "context_consistency",
                "context_variance",
                "visibility_balance",
                "attention_entropy",
            }.issubset(auxiliary)
        )
        self.assertTrue(all(value.ndim == 0 for value in auxiliary.values()))

    def test_uniform_helper_returns_zero_for_baseline(self) -> None:
        class Baseline(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(()))

            def forward(self, images: torch.Tensor):
                feature = images[:, :1] * self.scale
                return [feature], [feature]

        baseline = Baseline()
        _, _, regularizer, auxiliary = forward_with_regularization(
            baseline,
            torch.randn(2, 3, 4, 4),
        )
        self.assertEqual(float(regularizer), 0.0)
        self.assertEqual(regularizer.device, baseline.scale.device)
        self.assertEqual(auxiliary, {})


if __name__ == "__main__":
    unittest.main()
