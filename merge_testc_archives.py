#!/usr/bin/env python3
"""Safely merge independently split Real-IAD Variety ZIP archives.

The Test_C evaluators first need a source tree like::

    <output-root>/
      realiadvariety_1024/
      realiadvariety_jsons/

They then build the competition-shaped Test_C from the official JSON test
split via ``prepare_testc.py``. This program intentionally does *not* sample
or rename Test_C objects; it only restores the downloaded source archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


IMAGE_ROOT = "realiadvariety_1024"
JSON_ROOT = "realiadvariety_jsons"
KNOWN_ROOTS = {
    "realiadvariety_1024": IMAGE_ROOT,
    "realiadvariety_jsons": JSON_ROOT,
    "real-iad_variety_jsons": JSON_ROOT,
    "real_iad_variety_jsons": JSON_ROOT,
}
WRAPPER_NAMES = {
    "real-iad_variety_jsons",
    "real_iad_variety_jsons",
    "realiad_variety_jsons",
    "realiadvariety_jsons",
}


@dataclass(frozen=True)
class Member:
    archive: Path
    name: str
    target: Path
    size: int
    crc: int


def natural_key(path: Path) -> list[object]:
    return [int(item) if item.isdigit() else item.casefold()
            for item in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split Test_C Real-IAD Variety ZIP archives."
    )
    parser.add_argument(
        "--archives-dir", type=Path, required=True,
        help="Directory containing the seven independently extractable ZIP files.",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Destination Real-IAD_Variety root (created if absent).",
    )
    parser.add_argument(
        "--expected-archives", type=int, default=7,
        help="Expected number of ZIP files; use 0 to disable the count check.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate layouts and print the plan without writing files.",
    )
    return parser.parse_args()


def _safe_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {member_name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise ValueError(f"empty ZIP member path: {member_name!r}")
    return parts


def _target_for(member_name: str) -> Path | None:
    parts = _safe_parts(member_name)
    lowered = [part.casefold() for part in parts]

    # Archives may contain a top-level Real-IAD_Variety directory, or may put
    # the two required roots directly at their ZIP root.
    for index, part in enumerate(lowered):
        if part in KNOWN_ROOTS:
            root = KNOWN_ROOTS[part]
            suffix = list(parts[index + 1:])
            if root == JSON_ROOT and suffix and suffix[0].casefold() in WRAPPER_NAMES:
                suffix.pop(0)
            if not suffix:
                return None
            return Path(root, *suffix)
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _same_member(archive: Path, member_name: str, target: Path) -> bool:
    if not target.is_file():
        return False
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member_name)
        if target.stat().st_size != info.file_size:
            return False
        digest = hashlib.sha256()
        with source.open(info) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest() == _sha256_file(target)


def _archives(directory: Path, expected: int) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"archives directory does not exist: {directory}")
    split_volumes = sorted(directory.glob("*.z[0-9][0-9]"))
    if split_volumes:
        raise ValueError(
            "Found .z01-style multi-volume ZIP parts. This script expects "
            "seven independently extractable .zip archives, not one spanned ZIP."
        )
    archives = sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.casefold() == ".zip"),
        key=natural_key,
    )
    if not archives:
        raise FileNotFoundError(f"no .zip archives found in {directory}")
    if expected > 0 and len(archives) != expected:
        raise ValueError(
            f"expected {expected} ZIP archives, found {len(archives)}: "
            f"{[path.name for path in archives]}"
        )
    return archives


def plan(archives: list[Path]) -> tuple[list[Member], dict[str, int]]:
    members: list[Member] = []
    seen: dict[Path, Member] = {}
    ignored = 0
    for archive in archives:
        try:
            source = zipfile.ZipFile(archive)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid ZIP archive: {archive}") from exc
        with source:
            for info in source.infolist():
                if info.is_dir():
                    continue
                target = _target_for(info.filename)
                if target is None:
                    ignored += 1
                    continue
                current = Member(archive, info.filename, target, info.file_size, info.CRC)
                previous = seen.get(target)
                if previous is not None:
                    if (previous.size, previous.crc) != (current.size, current.crc):
                        raise ValueError(
                            "archive collision with different content: "
                            f"{target} from {previous.archive.name} and {archive.name}"
                        )
                    continue
                seen[target] = current
                members.append(current)
    roots = {item.target.parts[0] for item in members}
    missing_roots = {IMAGE_ROOT, JSON_ROOT} - roots
    if missing_roots:
        raise ValueError(
            "ZIP layout does not expose required roots "
            f"{sorted(missing_roots)}. Inspect the archive top-level names."
        )
    return members, {"ignored_members": ignored, "planned_members": len(members)}


def extract(members: list[Member], output_root: Path, dry_run: bool) -> dict[str, int]:
    output_root = output_root.expanduser().resolve()
    copied = reused = 0
    handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for member in members:
            target = output_root / member.target
            if target.exists():
                if _same_member(member.archive, member.name, target):
                    reused += 1
                    continue
                raise FileExistsError(
                    f"refusing to overwrite conflicting file: {target}"
                )
            if dry_run:
                copied += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handles.setdefault(member.archive, zipfile.ZipFile(member.archive))
            temporary = target.with_suffix(target.suffix + ".partial")
            try:
                with source.open(member.name) as reader, temporary.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            copied += 1
    finally:
        for source in handles.values():
            source.close()
    return {"copied_files": copied, "reused_files": reused}


def main() -> int:
    args = parse_args()
    archives = _archives(args.archives_dir, args.expected_archives)
    members, counts = plan(archives)
    result = {
        "archives": [str(path) for path in archives],
        "output_root": str(args.output_root.expanduser().resolve()),
        **counts,
        **extract(members, args.output_root, args.dry_run),
        "required_roots": [IMAGE_ROOT, JSON_ROOT],
        "next_command": (
            "python prepare_testc.py --source-root "
            f"{args.output_root.expanduser().resolve()}"
        ),
    }
    if not args.dry_run:
        report = args.output_root.expanduser().resolve() / "merge_testc_archives_report.json"
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["report"] = str(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
