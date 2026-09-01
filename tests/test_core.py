from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.data import Record, mask_transform
from realiad_dinomaly2.metrics import (
    GaussianFilter,
    SpoolingPixelMetricAccumulator,
    binary_metrics,
    top_ratio_mean,
)
from realiad_dinomaly2.losses import anomaly_map, debias_unseen_novelty
from realiad_dinomaly2.modeling import set_learning_rate
from realiad_dinomaly2.train_engine import (
    DeterministicIterationBatchSampler,
    _gradient_step_should_be_skipped,
)


class DataTests(unittest.TestCase):
    def test_view_parser_accepts_c03(self) -> None:
        record = Record(
            category="access_card",
            anomaly_class="OK",
            image_path="OK/S0001/access_card__0001_OK_C03_stamp.png",
            mask_path=None,
        )
        self.assertEqual(record.view_id, 3)
        self.assertEqual(record.object_id, "OK/S0001")

    def test_upstream_mask_resize_semantics(self) -> None:
        source_array = np.zeros((4, 4), dtype=np.uint8)
        source_array[1:3, 1:3] = 255
        source = Image.fromarray(source_array, mode="L")
        upstream = transforms.Compose(
            [
                transforms.Resize((8, 8)),
                transforms.CenterCrop(8),
                transforms.ToTensor(),
            ]
        )(source).bool()
        adapted = (
            mask_transform(
                image_size=8,
                crop_size=8,
                resize_semantics="upstream_bilinear_nonzero",
            )(source)
            > 0
        )
        self.assertTrue(torch.equal(adapted, upstream))


class SamplerTests(unittest.TestCase):
    def test_resume_is_exact_suffix(self) -> None:
        common = dict(
            dataset_size=12,
            micro_batch_size=3,
            total_optimizer_steps=4,
            accumulation_steps=2,
            seed=7,
        )
        complete = list(
            DeterministicIterationBatchSampler(
                start_optimizer_step=0,
                **common,
            )
        )
        resumed = list(
            DeterministicIterationBatchSampler(
                start_optimizer_step=2,
                **common,
            )
        )
        self.assertEqual(resumed, complete[4:])


class MetricTests(unittest.TestCase):
    def test_zero_sigma_disables_gaussian_smoothing(self) -> None:
        maps = torch.rand(2, 1, 5, 5)
        filtered = GaussianFilter(kernel_size=5, sigma=0.0)(maps)
        self.assertTrue(torch.equal(filtered, maps))

    def test_binary_metrics_perfect(self) -> None:
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result, {"auroc": 1.0, "aupr": 1.0, "f1max": 1.0})

    def test_top_ratio_uses_floor_like_upstream(self) -> None:
        maps = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
        score = top_ratio_mean(maps, ratio=0.019)
        self.assertEqual(float(score), 99.0)

    def test_pixel_spool_matches_upstream_dynamic_bounds(self) -> None:
        upstream = ROOT / "third_party" / "Dinomaly2"
        sys.path.insert(0, str(upstream))
        from adeval import EvalAccumulator

        generator = np.random.default_rng(3)
        maps = generator.uniform(0.2, 0.4, size=(4, 8, 8)).astype(np.float32)
        masks = np.zeros((4, 8, 8), dtype=np.uint8)
        masks[2:, 2:6, 3:7] = 1
        direct = EvalAccumulator(
            float(maps.min()),
            float(maps.max()),
            float(maps.min()),
            float(maps.max()),
            nstrips=100,
        )
        direct.add_anomap_batch(maps, masks)
        expected = direct.summary()

        spool = SpoolingPixelMetricAccumulator(
            device=torch.device("cpu"),
            bins=100,
            capacity=4,
            height=8,
            width=8,
            scratch_dir=ROOT / "outputs",
            stem="unittest_pixel_spool",
            replay_batch_size=2,
        )
        spool.add(torch.from_numpy(maps[:2]), torch.from_numpy(masks[:2]))
        spool.add(torch.from_numpy(maps[2:]), torch.from_numpy(masks[2:]))
        actual = spool.summary()

        self.assertAlmostEqual(actual["auroc"], expected["p_auroc"])
        self.assertAlmostEqual(actual["aupr"], expected["p_aupr"])
        self.assertAlmostEqual(actual["f1max"], expected["p_f1max"])
        self.assertAlmostEqual(actual["aupro"], expected["p_aupro"])


class ScheduleTests(unittest.TestCase):
    def test_warmup_and_final_ratio(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD(
            [{"params": [parameter], "lr": 2.0, "initial_lr": 2.0}]
        )
        first = set_learning_rate(
            optimizer,
            completed_steps=0,
            total_steps=1000,
            warmup_steps=100,
            final_ratio=0.1,
        )
        final = set_learning_rate(
            optimizer,
            completed_steps=999,
            total_steps=1000,
            warmup_steps=100,
            final_ratio=0.1,
        )
        self.assertAlmostEqual(first[0], 0.02)
        self.assertAlmostEqual(final[0], 0.2)

    def test_upstream_scheduler_starts_at_zero(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD(
            [{"params": [parameter], "lr": 2.0, "initial_lr": 2.0}]
        )
        first = set_learning_rate(
            optimizer,
            completed_steps=0,
            total_steps=1000,
            warmup_steps=100,
            final_ratio=1.0,
            step_offset=0,
        )
        self.assertEqual(first[0], 0.0)

    def test_yaml_linear_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD(
            [{"params": [parameter], "lr": 2.0, "initial_lr": 2.0}]
        )
        middle = set_learning_rate(
            optimizer,
            completed_steps=49,
            total_steps=100,
            warmup_steps=0,
            final_ratio=0.1,
            scheduler_config={"type": "linear"},
        )
        self.assertAlmostEqual(middle[0], 1.1)


class StabilityTests(unittest.TestCase):
    def test_gradient_guard_skips_catastrophic_norms(self) -> None:
        guard = {"skip_step_norm": 1.0}
        self.assertFalse(_gradient_step_should_be_skipped(0.9, guard))
        self.assertTrue(_gradient_step_should_be_skipped(40.0, guard))
        self.assertTrue(_gradient_step_should_be_skipped(float("nan"), guard))

    def test_anomaly_map_layer_weights(self) -> None:
        encoder = [
            torch.tensor([[[[1.0]], [[0.0]]]]),
            torch.tensor([[[[1.0]], [[0.0]]]]),
        ]
        decoder = [
            torch.tensor([[[[0.0]], [[1.0]]]]),
            torch.tensor([[[[1.0]], [[0.0]]]]),
        ]
        result = anomaly_map(
            encoder,
            decoder,
            output_size=2,
            layer_weights=[0.25, 0.75],
            align_corners=False,
        )
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.25)))

    def test_unseen_novelty_debias_suppresses_uniform_shift_not_local_peak(self) -> None:
        maps = torch.full((1, 1, 4, 4), 0.4)
        maps[0, 0, 1, 2] = 1.0
        result = debias_unseen_novelty(
            maps,
            baseline_quantile=0.5,
            local_blend=1.0,
            global_retention=0.25,
        )
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(result[0, 0, 1, 2]), 0.7, places=6)
        self.assertAlmostEqual(
            float(result[0, 0, 1, 2] - result[0, 0, 0, 0]),
            float(maps[0, 0, 1, 2] - maps[0, 0, 0, 0]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
