from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realiad_dinomaly2.cache_prepare import (
    STATE_FILENAME,
    cache_image_is_valid,
    prepare_train_cache,
)


class CachePreparationTests(unittest.TestCase):
    @contextmanager
    def _temporary_directory(self):
        scratch_root = ROOT / "outputs" / "_test_tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
        path = scratch_root / f"cache-{uuid.uuid4().hex}"
        path.mkdir()
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _fixture(self, root: Path) -> tuple[dict, Path]:
        json_dir = root / "json"
        raw_dir = root / "raw"
        cache_dir = root / "cache"
        category = "nested_part"
        relative_paths = [
            "OK/S0001/nested_part_0001_OK_C1.png",
            "OK/S0001/nested_part_0001_OK_C2.png",
        ]
        category_root = raw_dir / category / category
        for index, relative in enumerate(relative_paths):
            destination = category_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (21 + index, 13 + index),
                color=(40 + index, 80, 120),
            ).save(destination)
        json_dir.mkdir(parents=True)
        payload = {
            "train": [
                {
                    "category": category,
                    "anomaly_class": "OK",
                    "image_path": relative,
                    "mask_path": None,
                }
                for relative in relative_paths
            ],
            "test": [],
        }
        (json_dir / f"{category}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        config = {
            "dataset": {
                "json_dir": str(json_dir),
                "image_dir": str(raw_dir),
                "categories": [category],
                "category_limit": None,
            },
            "cache": {
                "output_dir": str(cache_dir),
                "max_side": 32,
                "workers": 2,
            },
        }
        expected = cache_dir / category / category / relative_paths[0]
        return config, expected

    def test_cache_preserves_nested_tree_and_resumes(self) -> None:
        with self._temporary_directory() as temporary:
            config, expected = self._fixture(Path(temporary))
            first = prepare_train_cache(config)
            self.assertEqual(first["written"], 2)
            self.assertEqual(first["skipped_valid"], 0)
            self.assertTrue(cache_image_is_valid(expected, 32))

            second = prepare_train_cache(config)
            self.assertEqual(second["written"], 0)
            self.assertEqual(second["skipped_valid"], 2)
            state = json.loads(
                (Path(config["cache"]["output_dir"]) / STATE_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["processed"], 2)

    def test_corrupt_existing_output_is_replaced(self) -> None:
        with self._temporary_directory() as temporary:
            config, expected = self._fixture(Path(temporary))
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"not an image")
            state = prepare_train_cache(config)
            self.assertEqual(state["written"], 2)
            self.assertTrue(cache_image_is_valid(expected, 32))


if __name__ == "__main__":
    unittest.main()
