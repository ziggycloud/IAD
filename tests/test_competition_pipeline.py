from __future__ import annotations

import csv
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.competition_data import (  # noqa: E402
    CompetitionFolderDataset,
    CompetitionObjectDataset,
    scan_competition_split,
)
from realiad_dinomaly2.competition_submission import (  # noqa: E402
    _aggregate_object_score,
    _top_ratio_score,
    build_submission_zip,
    validate_submission_layout,
)
from realiad_dinomaly2.config import load_config, materialize_paths  # noqa: E402


def _write_sample(root: Path, category: str, sample: str) -> None:
    sample_dir = root / category / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    for view_id in range(5):
        array = np.full((12, 12, 3), 20 + view_id, dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(sample_dir / f"{view_id}.png")


@contextmanager
def _temporary_directory():
    scratch_root = ROOT / "outputs" / "_test_tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    path = scratch_root / f"competition-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class CompetitionDataTests(unittest.TestCase):
    def test_scan_and_dataset_preserve_five_view_topology(self) -> None:
        with _temporary_directory() as root:
            _write_sample(root, "part_b", "S0002")
            _write_sample(root, "part_a", "S0001")
            manifest = scan_competition_split(root)
            self.assertEqual(manifest.categories, ("part_a", "part_b"))
            self.assertEqual(
                manifest.group_folders, ("part_a/S0001", "part_b/S0002")
            )
            self.assertEqual(
                [view.view_id for view in manifest.views[:5]], list(range(5))
            )
            dataset = CompetitionFolderDataset(
                manifest.views, image_size=14, crop_size=14
            )
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (3, 14, 14))
            self.assertEqual(item["group_folder"], "part_a/S0001")
            object_dataset = CompetitionObjectDataset(
                manifest.views, image_size=14, crop_size=14
            )
            object_item = object_dataset[0]
            self.assertEqual(tuple(object_item["images"].shape), (5, 3, 14, 14))
            self.assertEqual(object_item["view_ids"].tolist(), list(range(5)))
            self.assertTrue(object_item["valid_view_mask"].all())
            means = object_item["images"].mean(dim=(1, 2, 3))
            self.assertTrue((means[1:] > means[:-1]).all())

    def test_scan_rejects_missing_view(self) -> None:
        with _temporary_directory() as root:
            _write_sample(root, "part", "S0001")
            (root / "part" / "S0001" / "4.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                scan_competition_split(root)

    def test_competition_config_materializes_folder_paths(self) -> None:
        config = materialize_paths(load_config(ROOT / "configs" / "competition.yaml"))
        self.assertEqual(config["dataset"]["type"], "competition_folders")
        self.assertTrue(Path(config["dataset"]["train_dir"]).is_absolute())
        self.assertTrue(Path(config["dataset"]["test_dir"]).is_absolute())
        self.assertEqual(config["model"]["architecture"], "dinomaly2")
        self.assertFalse(config["model"]["multi_view"]["enabled"])
        self.assertEqual(config["training"]["effective_batch_size"], 12)

    def test_object_score_modes_keep_single_view_anomaly(self) -> None:
        maps = [np.zeros((4, 4), dtype=np.float32) for _ in range(5)]
        maps[3][0, 0] = 10.0
        legacy = _aggregate_object_score(
            maps, 0.1, mode="legacy_concat_topk"
        )
        self.assertEqual(legacy, _top_ratio_score(maps, 0.1))
        score = _aggregate_object_score(
            maps,
            0.1,
            mode="visibility_aware",
            visibility=np.asarray([1.0, 1.0, 1.0, 0.0, 1.0]),
            visibility_max_blend=0.5,
        )
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(
            score,
            0.5 * _aggregate_object_score(maps, 0.1, mode="max"),
        )


class CompetitionPackageTests(unittest.TestCase):
    def test_validator_and_zip_have_exact_official_layout(self) -> None:
        with _temporary_directory() as root:
            test_root = root / "Test_A"
            _write_sample(test_root, "part", "S0001")
            manifest = scan_competition_split(test_root)
            package = root / "package"
            package.mkdir()
            with (package / "submission.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["group_folder", "anomaly_score"])
                writer.writerow(["part/S0001", "0.123"])
            mask_dir = package / "predicted_masks" / "part" / "S0001"
            mask_dir.mkdir(parents=True)
            for view_id in range(5):
                Image.new("L", (448, 448), color=view_id).save(
                    mask_dir / f"{view_id}_mask.png"
                )

            summary = validate_submission_layout(package, manifest)
            self.assertEqual(summary, {"groups": 1, "masks": 5, "mask_size": 448})
            zip_path = build_submission_zip(
                package, root / "submission.zip", manifest
            )
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist()[0], "submission.csv")
                self.assertEqual(len(archive.namelist()), 6)
                self.assertTrue(
                    all(
                        not name.startswith("package/")
                        for name in archive.namelist()
                    )
                )


if __name__ == "__main__":
    unittest.main()
