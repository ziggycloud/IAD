from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.synthetic_anomaly import _blob_mask  # noqa: E402


class SyntheticAnomalyTests(unittest.TestCase):
    def test_blob_mask_uses_one_quantile_per_batch_item(self) -> None:
        torch.manual_seed(907)

        mask = _blob_mask(
            batch=10,
            height=448,
            width=448,
            device=torch.device("cpu"),
            min_area=0.01,
            max_area=0.10,
        )

        self.assertEqual(mask.shape, (10, 1, 448, 448))
        area = mask.float().flatten(1).mean(1)
        self.assertTrue(torch.all(area > 0.005))
        self.assertTrue(torch.all(area < 0.15))


if __name__ == "__main__":
    unittest.main()
