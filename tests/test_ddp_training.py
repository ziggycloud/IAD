from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import realiad_dinomaly2.train_engine as train_engine
from realiad_dinomaly2.train_engine import (
    DeterministicIterationBatchSampler,
    _fixed_batch_choice,
    _initialize_distributed_context,
)


class GlobalBatchTests(unittest.TestCase):
    def test_single_gpu_keeps_legacy_batch_semantics(self) -> None:
        choice = _fixed_batch_choice(16, 4, 1)
        self.assertEqual(choice.global_micro_batch_size, 4)
        self.assertEqual(choice.accumulation_steps, 4)

    def test_two_gpu_batch_is_global(self) -> None:
        choice = _fixed_batch_choice(64, 16, 2)
        self.assertEqual(choice.global_micro_batch_size, 32)
        self.assertEqual(choice.accumulation_steps, 2)

    def test_global_batch_must_be_divisible(self) -> None:
        with self.assertRaises(ValueError):
            _fixed_batch_choice(48, 16, 2)


class DistributedSamplingTests(unittest.TestCase):
    def test_ranks_are_disjoint_and_cover_the_epoch(self) -> None:
        common = dict(
            dataset_size=16,
            micro_batch_size=2,
            total_optimizer_steps=4,
            accumulation_steps=1,
            start_optimizer_step=0,
            seed=11,
            world_size=2,
        )
        rank_zero = list(
            DeterministicIterationBatchSampler(rank=0, **common)
        )
        rank_one = list(
            DeterministicIterationBatchSampler(rank=1, **common)
        )
        zero_indices = {index for batch in rank_zero for index in batch}
        one_indices = {index for batch in rank_one for index in batch}
        self.assertFalse(zero_indices & one_indices)
        self.assertEqual(zero_indices | one_indices, set(range(16)))

    def test_distributed_resume_is_exact_suffix_on_each_rank(self) -> None:
        for rank in (0, 1):
            common = dict(
                dataset_size=24,
                micro_batch_size=3,
                total_optimizer_steps=6,
                accumulation_steps=2,
                seed=5,
                rank=rank,
                world_size=2,
            )
            complete = list(
                DeterministicIterationBatchSampler(
                    start_optimizer_step=0, **common
                )
            )
            resumed = list(
                DeterministicIterationBatchSampler(
                    start_optimizer_step=3, **common
                )
            )
            self.assertEqual(resumed, complete[6:])


class StrategySelectionTests(unittest.TestCase):
    def test_plain_python_auto_uses_configured_gpus(self) -> None:
        config = {
            "runtime": {
                "device": "cuda:0",
                "multi_gpu_strategy": "auto",
                "device_ids": [0, 1],
            }
        }
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(torch.cuda, "set_device") as set_device,
        ):
            context = _initialize_distributed_context(config)
        self.assertEqual(context.strategy, "data_parallel")
        self.assertEqual(context.world_size, 2)
        self.assertEqual(context.device_ids, (0, 1))
        set_device.assert_called_once_with(torch.device("cuda:0"))

    def test_legacy_config_remains_single_process(self) -> None:
        config = {"runtime": {"device": "cpu"}}
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            context = _initialize_distributed_context(config)
        self.assertEqual(context.strategy, "single")
        self.assertEqual(context.world_size, 1)

    def test_torchrun_environment_initializes_gloo_for_cpu(self) -> None:
        config = {
            "runtime": {
                "device": "cpu",
                "multi_gpu_strategy": "auto",
            }
        }
        environment = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch.object(train_engine.dist, "is_available", return_value=True),
            mock.patch.object(
                train_engine.dist, "init_process_group"
            ) as initialize,
        ):
            context = _initialize_distributed_context(config)
        self.assertEqual(context.strategy, "ddp")
        self.assertEqual(context.rank, 1)
        self.assertEqual(context.world_size, 2)
        self.assertEqual(context.backend, "gloo")
        initialize.assert_called_once_with(backend="gloo", init_method="env://")


if __name__ == "__main__":
    unittest.main()
