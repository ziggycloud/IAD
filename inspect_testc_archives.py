#!/usr/bin/env python3
"""Inspect split Test_C ZIP archives without extracting them.

Example:
    python inspect_testc_archives.py --archives-dir data/competition/Test_C \
        --json-preview-chars 300

The output is JSON, so it can be saved or pasted back unchanged when the
archive layout needs to be mapped into the Test_C preparation pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> list[object]:
    return [int(item) if item.isdigit() else item.casefold()
            for item in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the internal layout of independently extractable Test_C ZIP files."
    )
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-archives", type=int, default=0,
        help="Expected ZIP count; 0 disables the check (default: 0).",
    )
    parser.add_argument(
        "--sample-paths", type=int, default=20,
        help="Maximum example paths per file type in each archive (default: 20).",
    )
    parser.add_argument(
        "--json-preview-chars", type=int, default=400,
        help="Maximum decoded characters shown from each JSON file (default: 400).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional JSON report path. stdout is always printed.",
    )
    return parser.parse_args()


def archives_in(directory: Path, expected: int) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"archives directory does not exist: {directory}")
    archives = sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.casefold() == ".zip"),
        key=natural_key,
    )
    if not archives:
        raise FileNotFoundError(f"no ZIP archives found in {directory}")
    if expected and len(archives) != expected:
        raise ValueError(f"expected {expected} ZIP files, found {len(archives)}")
    return archives


def _path_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name.replace("\\", "/"))
    return tuple(part for part in path.parts if part not in {"", "."})


def _json_summary(source: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": info.filename, "bytes": info.file_size}
    try:
        raw = source.read(info)
        text = raw.decode("utf-8-sig")
        payload = json.loads(text)
        result["json_type"] = type(payload).__name__
        if isinstance(payload, dict):
            result["keys"] = sorted(str(key) for key in payload)[:50]
            for key in ("train", "test", "validation", "val", "images", "annotations"):
                value = payload.get(key)
                if isinstance(value, list):
                    result[f"{key}_count"] = len(value)
        elif isinstance(payload, list):
            result["item_count"] = len(payload)
        result["preview"] = text[:limit]
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_archive(archive: Path, sample_limit: int, json_limit: int) -> tuple[dict[str, Any], set[str]]:
    with zipfile.ZipFile(archive) as source:
        infos = [info for info in source.infolist() if not info.is_dir()]
        paths = [info.filename for info in infos]
        top_level = sorted({_path_parts(name)[0] for name in paths if _path_parts(name)})
        suffixes: dict[str, int] = defaultdict(int)
        examples: dict[str, list[str]] = defaultdict(list)
        json_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            suffix = PurePosixPath(info.filename).suffix.casefold() or "[no suffix]"
            suffixes[suffix] += 1
            if len(examples[suffix]) < sample_limit:
                examples[suffix].append(info.filename)
            if suffix == ".json":
                json_infos.append(info)
        image_examples = [
            info.filename for info in infos
            if PurePosixPath(info.filename).suffix.casefold() in IMAGE_SUFFIXES
        ][:sample_limit]
        report = {
            "archive": archive.name,
            "bytes": archive.stat().st_size,
            "file_count": len(infos),
            "top_level": top_level,
            "suffix_counts": dict(sorted(suffixes.items())),
            "path_samples_by_suffix": dict(sorted(examples.items())),
            "image_path_samples": image_examples,
            "json_files": [_json_summary(source, info, json_limit) for info in json_infos],
        }
        return report, set(paths)


def main() -> int:
    args = parse_args()
    archives = archives_in(args.archives_dir, args.expected_archives)
    reports = []
    all_paths: dict[str, list[str]] = defaultdict(list)
    for archive in archives:
        report, paths = inspect_archive(archive, args.sample_paths, args.json_preview_chars)
        reports.append(report)
        for path in paths:
            all_paths[path].append(archive.name)

    duplicates = {
        path: names for path, names in sorted(all_paths.items()) if len(names) > 1
    }
    payload = {
        "archives_dir": str(args.archives_dir.expanduser().resolve()),
        "archive_count": len(archives),
        "archives": reports,
        "cross_archive_duplicate_paths": duplicates,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
