from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


RUNTIME_MODULES = ("torch", "sklearn", "yaml", "cv2", "PIL")


def _python_in_prefix(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def _candidate_pythons(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("IAD_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    parent = project_root.parent
    prefixes = [
        parent / "IAD" / "data" / ".conda" / "iad",
        parent / "IAD" / "data" / ".conda" / "realiad-variety-py311",
        parent / "IAD" / "data.conda" / "realiad-variety-py311",
        Path(r"J:\project\IAD\data\.conda\iad"),
        Path(r"J:\project\IAD\data\.conda\realiad-variety-py311"),
        Path(r"J:\project\IAD\data.conda\realiad-variety-py311"),
    ]
    candidates.extend(_python_in_prefix(prefix) for prefix in prefixes)
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
        for raw_prefix in json.loads(result.stdout).get("envs", []):
            prefix = Path(raw_prefix)
            if prefix.name.casefold() in {"iad", "realiad-variety-py311"}:
                candidates.append(_python_in_prefix(prefix))
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _has_runtime(candidate: Path) -> bool:
    probe = (
        "import importlib.util,sys;"
        f"mods={RUNTIME_MODULES!r};"
        "sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
    )
    try:
        result = subprocess.run(
            [str(candidate), "-c", probe],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def ensure_iad_runtime(
    project_root: Path,
    script_path: Path,
    arguments: list[str],
) -> None:
    missing = [
        name for name in RUNTIME_MODULES if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return
    marker = "IAD_COMPETITION_PIPELINE_REEXEC"
    if os.environ.get(marker) == "1":
        raise RuntimeError(
            "Selected IAD Python is missing dependencies: " + ", ".join(missing)
        )
    current = Path(sys.executable).resolve()
    for candidate in _candidate_pythons(project_root):
        if not candidate.is_file() or candidate.resolve() == current:
            continue
        if not _has_runtime(candidate):
            continue
        print(
            f"Current Python is missing {', '.join(missing)}; "
            f"re-launching with {candidate}",
            file=sys.stderr,
            flush=True,
        )
        os.environ[marker] = "1"
        os.execv(
            str(candidate),
            [str(candidate), str(script_path.resolve()), *arguments],
        )
    raise RuntimeError(
        "A complete IAD Python environment was not found. Set IAD_PYTHON to "
        "its python executable. Missing modules: " + ", ".join(missing)
    )
