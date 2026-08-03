from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.latency import summarize_latency


class LatencySummaryTests(unittest.TestCase):
    def test_summary_uses_mean_and_interpolated_p95(self) -> None:
        result = summarize_latency([0.1, 0.2, 0.3, 0.4], threshold_seconds=0.5)
        self.assertAlmostEqual(result["mean_seconds"], 0.25)
        self.assertAlmostEqual(result["p50_seconds"], 0.25)
        self.assertAlmostEqual(result["p95_seconds"], 0.385)
        self.assertTrue(result["valid"])

    def test_p95_failure_invalidates_submission(self) -> None:
        result = summarize_latency([0.2] * 18 + [2.0] * 2, threshold_seconds=1.0)
        self.assertLess(result["mean_seconds"], 1.0)
        self.assertGreater(result["p95_seconds"], 1.0)
        self.assertFalse(result["valid"])

    def test_invalid_inputs_are_rejected(self) -> None:
        for values in ([], [-0.1], [math.inf], [math.nan]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    summarize_latency(values)
        with self.assertRaises(ValueError):
            summarize_latency([0.1], threshold_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
