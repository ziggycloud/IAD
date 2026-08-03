from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.unseen_protocol import (
    compute_unseen_evaluation_scores,
    generate_category_split,
    load_category_split,
    load_or_create_category_split,
    render_unseen_evaluation_report,
    save_category_split,
)


def category_names() -> list[str]:
    return [f"category_{index:03d}" for index in range(160)]


def metric_rows(prefix: str, values: tuple[float, ...]) -> list[dict[str, float | str]]:
    return [
        {
            "category": f"{prefix}_{index:03d}",
            "i_auroc": values[0],
            "i_aupr": values[1],
            "p_auroc": values[2],
            "p_aupr": values[3],
            "p_f1max": values[4],
        }
        for index in range(50)
    ]


class CategorySplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_sorted_and_complete(self) -> None:
        categories = category_names()
        first = generate_category_split(categories, seed=2026)
        reordered = generate_category_split(reversed(categories), seed=2026)
        different = generate_category_split(categories, seed=2027)

        self.assertEqual(first, reordered)
        self.assertNotEqual(first.seen, different.seen)
        self.assertEqual(len(first.seen), 50)
        self.assertEqual(len(first.unseen), 50)
        self.assertEqual(len(first.unused), 60)
        self.assertEqual(first.seen, tuple(sorted(first.seen)))
        self.assertEqual(first.unseen, tuple(sorted(first.unseen)))
        self.assertEqual(first.unused, tuple(sorted(first.unused)))
        self.assertFalse(set(first.seen) & set(first.unseen))
        self.assertFalse(set(first.seen) & set(first.unused))
        self.assertFalse(set(first.unseen) & set(first.unused))
        self.assertEqual(set(first.all_categories), set(categories))

    def test_json_round_trip_and_mismatch_rejection(self) -> None:
        categories = category_names()
        split = generate_category_split(categories, seed=17)
        path = ROOT / "outputs" / f"unittest_category_split_{os.getpid()}.json"
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            save_category_split(split, path)
            self.assertEqual(
                load_category_split(
                    path,
                    expected_categories=categories,
                    expected_seed=17,
                ),
                split,
            )
            self.assertEqual(
                load_or_create_category_split(categories, 17, path),
                split,
            )
            with self.assertRaisesRegex(ValueError, "requested seed"):
                load_or_create_category_split(categories, 18, path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["seen"] = list(reversed(payload["seen"]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sorted"):
                load_category_split(path)
        finally:
            path.unlink(missing_ok=True)
            temporary_path.unlink(missing_ok=True)


class UnseenScoreTests(unittest.TestCase):
    def test_confirmed_weighted_score_and_native_metric_names(self) -> None:
        seen = metric_rows("seen", (0.8, 0.6, 0.7, 0.5, 0.9))
        unseen = metric_rows("unseen", (0.4, 0.6, 0.5, 0.3, 0.7))
        result = compute_unseen_evaluation_scores(seen, unseen)

        self.assertAlmostEqual(result["components"]["S_cls"]["score"], 70.0)
        self.assertAlmostEqual(result["components"]["S_seg"]["score"], 70.0)
        self.assertAlmostEqual(result["components"]["S_zs"]["score"], 50.0)
        self.assertAlmostEqual(result["total_score"], 66.0)
        self.assertAlmostEqual(
            result["seen"]["macro_metrics"]["I-ROC"], 80.0
        )

    def test_report_metric_aliases_are_supported(self) -> None:
        seen = {
            f"seen_{index:03d}": {
                "I-ROC": 0.8,
                "I-PR": 0.6,
                "P-ROC": 0.7,
                "P-PR": 0.5,
                "P-F1max": 0.9,
            }
            for index in range(50)
        }
        unseen = {
            f"unseen_{index:03d}": {
                "I_AUROC": 0.4,
                "I_AUPR": 0.6,
                "P_AUROC": 0.5,
                "P_AUPR": 0.3,
                "P_F1MAX": 0.7,
            }
            for index in range(50)
        }
        result = compute_unseen_evaluation_scores(seen, unseen)
        self.assertAlmostEqual(result["total_score"], 66.0)

    def test_wrong_count_and_overlap_are_rejected(self) -> None:
        seen = metric_rows("seen", (0.8, 0.6, 0.7, 0.5, 0.9))
        unseen = metric_rows("unseen", (0.4, 0.6, 0.5, 0.3, 0.7))
        with self.assertRaisesRegex(ValueError, "requires 50 seen"):
            compute_unseen_evaluation_scores(seen[:-1], unseen)

        unseen[0]["category"] = seen[0]["category"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            compute_unseen_evaluation_scores(seen, unseen)


class UnseenReportTests(unittest.TestCase):
    def make_result(self):
        split = generate_category_split(category_names(), seed=2026)
        seen = {
            category: {
                "i_auroc": 0.8,
                "i_aupr": 0.6,
                "p_auroc": 0.7,
                "p_aupr": 0.5,
                "p_f1max": 0.9,
            }
            for category in split.seen
        }
        unseen = {
            category: {
                "i_auroc": 0.4,
                "i_aupr": 0.6,
                "p_auroc": 0.5,
                "p_aupr": 0.3,
                "p_f1max": 0.7,
            }
            for category in split.unseen
        }
        return split, compute_unseen_evaluation_scores(seen, unseen, split=split)

    def test_markdown_contains_metrics_formulas_score_and_latency(self) -> None:
        split, result = self.make_result()
        report = render_unseen_evaluation_report(
            result,
            split,
            latency_seconds=0.75,
        )

        self.assertIn("| Seen | 80.00 | 60.00 | 70.00 | 50.00 | 90.00 |", report)
        self.assertIn("| Unseen | 40.00 | 60.00 | 50.00 | 30.00 | 70.00 |", report)
        self.assertIn("S_cls = mean(Seen I-ROC, Seen I-PR)", report)
        self.assertIn("总分 = 0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs", report)
        self.assertIn("| **总分** |  |  | **66.00** |", report)
        self.assertIn("| 0.750 s/图 | ≤ 1.000 s/图 | **通过** |", report)

    def test_missing_and_over_limit_latency_are_explicit(self) -> None:
        split, result = self.make_result()
        missing = render_unseen_evaluation_report(result, split)
        failed = render_unseen_evaluation_report(
            result,
            split,
            latency_seconds=1.01,
        )
        self.assertIn("| 未提供 | ≤ 1.000 s/图 | **未判定** |", missing)
        self.assertIn("| 1.010 s/图 | ≤ 1.000 s/图 | **不通过** |", failed)

    def test_latency_summary_displays_aggregates_and_trusts_valid(self) -> None:
        split, result = self.make_result()
        report = render_unseen_evaluation_report(
            result,
            split,
            latency_summary={
                "mean_seconds": 0.4,
                "p95_seconds": 0.7,
                "max_seconds": 0.9,
                "threshold_seconds": 1.0,
                # A strict evaluator may invalidate the run for reasons that
                # cannot be recovered from rounded aggregates.
                "valid": False,
            },
        )
        self.assertIn("| Mean | P95 | Max | 限制 | 严格判定 |", report)
        self.assertIn(
            "| 0.400 s/图 | 0.700 s/图 | 0.900 s/图 | ≤ 1.000 s/图 | **不通过** |",
            report,
        )

    def test_two_latency_forms_cannot_be_mixed(self) -> None:
        split, result = self.make_result()
        with self.assertRaisesRegex(ValueError, "cannot be provided together"):
            render_unseen_evaluation_report(
                result,
                split,
                latency_seconds=0.5,
                latency_summary={
                    "mean_seconds": 0.4,
                    "p95_seconds": 0.7,
                    "max_seconds": 0.9,
                    "threshold_seconds": 1.0,
                    "valid": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
