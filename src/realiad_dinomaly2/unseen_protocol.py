"""Deterministic seen/unseen protocol and scoring for Real-IAD Variety.

The protocol deliberately persists the sampled category split.  A saved split is
treated as part of the experiment definition: changing the seed or category
universe raises an error instead of silently creating a different benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_NAME = "realiad-variety-seen50-unseen50-v1"
TOTAL_CATEGORIES = 160
SEEN_CATEGORIES = 50
UNSEEN_CATEGORIES = 50
UNUSED_CATEGORIES = 60

_METRIC_ALIASES = {
    "i_auroc": {
        "iauroc",
        "iroc",
        "imageauroc",
        "imagelevelauroc",
    },
    "i_aupr": {
        "iaupr",
        "ipr",
        "iap",
        "imageaupr",
        "imageap",
        "imagelevelaupr",
    },
    "p_auroc": {
        "pauroc",
        "proc",
        "pixelauroc",
        "pixellevelauroc",
    },
    "p_aupr": {
        "paupr",
        "ppr",
        "pap",
        "pixelaupr",
        "pixelap",
        "pixellevelaupr",
    },
    "p_f1max": {
        "pf1max",
        "pixelf1max",
        "pixellevelf1max",
    },
}
_DISPLAY_NAMES = {
    "i_auroc": "I-ROC",
    "i_aupr": "I-PR",
    "p_auroc": "P-ROC",
    "p_aupr": "P-PR",
    "p_f1max": "P-F1max",
}
_CATEGORY_ALIASES = {"category", "class", "classname"}


def _category_digest(categories: Iterable[str]) -> str:
    canonical = "\n".join(sorted(categories)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_category_names(categories: Iterable[Any]) -> tuple[str, ...]:
    values = tuple(categories)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Every category must be a non-empty string.")
    if any(value != value.strip() for value in values):
        raise ValueError("Category names must not contain surrounding whitespace.")
    if len(values) != len(set(values)):
        raise ValueError("Category names must be unique.")
    return values


@dataclass(frozen=True)
class CategorySplit:
    """A validated, immutable 50/50/60 category split."""

    seed: int
    seen: tuple[str, ...]
    unseen: tuple[str, ...]
    unused: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

        groups = {
            "seen": _validate_category_names(self.seen),
            "unseen": _validate_category_names(self.unseen),
            "unused": _validate_category_names(self.unused),
        }
        for name, values in groups.items():
            object.__setattr__(self, name, values)
        expected_counts = {
            "seen": SEEN_CATEGORIES,
            "unseen": UNSEEN_CATEGORIES,
            "unused": UNUSED_CATEGORIES,
        }
        for name, values in groups.items():
            if len(values) != expected_counts[name]:
                raise ValueError(
                    f"{name} must contain exactly {expected_counts[name]} "
                    f"categories, got {len(values)}."
                )
            if tuple(sorted(values)) != values:
                raise ValueError(f"{name} categories must be lexicographically sorted.")

        seen = set(self.seen)
        unseen = set(self.unseen)
        unused = set(self.unused)
        if seen & unseen or seen & unused or unseen & unused:
            raise ValueError("seen, unseen, and unused categories must be disjoint.")
        if len(seen | unseen | unused) != TOTAL_CATEGORIES:
            raise ValueError(
                f"The split must cover exactly {TOTAL_CATEGORIES} categories."
            )

    @property
    def all_categories(self) -> tuple[str, ...]:
        return tuple(sorted((*self.seen, *self.unseen, *self.unused)))

    @property
    def categories_sha256(self) -> str:
        return _category_digest(self.all_categories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "seed": self.seed,
            "total_categories": TOTAL_CATEGORIES,
            "counts": {
                "seen": SEEN_CATEGORIES,
                "unseen": UNSEEN_CATEGORIES,
                "unused": UNUSED_CATEGORIES,
            },
            "categories_sha256": self.categories_sha256,
            "seen": list(self.seen),
            "unseen": list(self.unseen),
            "unused": list(self.unused),
        }


def generate_category_split(
    categories: Iterable[str],
    seed: int,
) -> CategorySplit:
    """Sample a deterministic split after canonicalizing the category universe."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    values = _validate_category_names(categories)
    if len(values) != TOTAL_CATEGORIES:
        raise ValueError(
            f"Expected exactly {TOTAL_CATEGORIES} categories, got {len(values)}."
        )

    shuffled = sorted(values)
    random.Random(seed).shuffle(shuffled)
    return CategorySplit(
        seed=seed,
        seen=tuple(sorted(shuffled[:SEEN_CATEGORIES])),
        unseen=tuple(
            sorted(shuffled[SEEN_CATEGORIES : SEEN_CATEGORIES + UNSEEN_CATEGORIES])
        ),
        unused=tuple(sorted(shuffled[SEEN_CATEGORIES + UNSEEN_CATEGORIES :])),
    )


def save_category_split(split: CategorySplit, path: str | Path) -> Path:
    """Atomically persist a validated category split as deterministic JSON."""

    # Reconstructing runs all validation even if a caller bypassed the frozen
    # dataclass through low-level mutation.
    validated = CategorySplit(split.seed, split.seen, split.unseen, split.unused)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validated.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_category_split(
    path: str | Path,
    *,
    expected_categories: Iterable[str] | None = None,
    expected_seed: int | None = None,
) -> CategorySplit:
    """Load JSON and reject schema, ordering, seed, or universe mismatches."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read a valid category split from {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Category split JSON must contain an object.")

    expected_keys = {
        "protocol",
        "seed",
        "total_categories",
        "counts",
        "categories_sha256",
        "seen",
        "unseen",
        "unused",
    }
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        extra = sorted(set(payload) - expected_keys)
        raise ValueError(f"Invalid category split schema; missing={missing}, extra={extra}.")
    if payload["protocol"] != PROTOCOL_NAME:
        raise ValueError(f"Unsupported category split protocol: {payload['protocol']!r}.")
    if payload["total_categories"] != TOTAL_CATEGORIES:
        raise ValueError("Invalid total_categories in category split JSON.")
    if payload["counts"] != {
        "seen": SEEN_CATEGORIES,
        "unseen": UNSEEN_CATEGORIES,
        "unused": UNUSED_CATEGORIES,
    }:
        raise ValueError("Invalid category counts in category split JSON.")
    if not all(isinstance(payload[name], list) for name in ("seen", "unseen", "unused")):
        raise ValueError("seen, unseen, and unused must be JSON arrays.")

    split = CategorySplit(
        seed=payload["seed"],
        seen=tuple(payload["seen"]),
        unseen=tuple(payload["unseen"]),
        unused=tuple(payload["unused"]),
    )
    if payload["categories_sha256"] != split.categories_sha256:
        raise ValueError("Category digest does not match the persisted split.")
    if expected_seed is not None and split.seed != expected_seed:
        raise ValueError(
            f"Persisted seed {split.seed} does not match requested seed {expected_seed}."
        )
    if expected_categories is not None:
        expected = tuple(sorted(_validate_category_names(expected_categories)))
        if expected != split.all_categories:
            raise ValueError("Persisted split does not match the requested category universe.")
    return split


def load_or_create_category_split(
    categories: Iterable[str],
    seed: int,
    path: str | Path,
) -> CategorySplit:
    """Reuse an exact persisted split, or create it once when absent."""

    category_values = tuple(categories)
    destination = Path(path)
    if destination.exists():
        return load_category_split(
            destination,
            expected_categories=category_values,
            expected_seed=seed,
        )
    split = generate_category_split(category_values, seed)
    save_category_split(split, destination)
    return split


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _resolve_metric(row: Mapping[str, Any], metric_name: str, category: str) -> float:
    matches = [
        value
        for key, value in row.items()
        if _normalized_key(key) in _METRIC_ALIASES[metric_name]
    ]
    if not matches:
        raise ValueError(
            f"Category {category!r} is missing {_DISPLAY_NAMES[metric_name]}."
        )
    if len(matches) > 1 and any(value != matches[0] for value in matches[1:]):
        raise ValueError(
            f"Category {category!r} has conflicting aliases for "
            f"{_DISPLAY_NAMES[metric_name]}."
        )
    value = matches[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{_DISPLAY_NAMES[metric_name]} for {category!r} must be numeric."
        )
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"{_DISPLAY_NAMES[metric_name]} for {category!r} must be in [0, 1]."
        )
    return numeric


def _row_category(row: Mapping[str, Any]) -> str:
    matches = [
        value
        for key, value in row.items()
        if _normalized_key(key) in _CATEGORY_ALIASES
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0].strip():
        raise ValueError("Each metric row must contain exactly one non-empty category field.")
    return matches[0]


def _coerce_rows(
    rows: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    if isinstance(rows, Mapping):
        iterator = []
        for category, metrics in rows.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Metrics for category {category!r} must be a mapping.")
            row = dict(metrics)
            row["category"] = category
            iterator.append(row)
    else:
        iterator = list(rows)

    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("Every metric row must be a mapping.")
        category = _row_category(row)
        if category in normalized:
            raise ValueError(f"Duplicate metric row for category {category!r}.")
        normalized[category] = {
            name: _resolve_metric(row, name, category) for name in _METRIC_ALIASES
        }
    return normalized


def _macro(rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        name: sum(row[name] for row in rows.values()) / len(rows)
        for name in _METRIC_ALIASES
    }


def compute_unseen_evaluation_scores(
    seen_rows: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    unseen_rows: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    split: CategorySplit | None = None,
) -> dict[str, Any]:
    """Compute the confirmed 30%/50%/20% benchmark score.

    Input metrics use the native evaluator scale (fractions in ``[0, 1]``).
    Returned component metrics and scores are percentages in ``[0, 100]``.
    """

    seen = _coerce_rows(seen_rows)
    unseen = _coerce_rows(unseen_rows)
    if len(seen) != SEEN_CATEGORIES or len(unseen) != UNSEEN_CATEGORIES:
        raise ValueError(
            f"Scoring requires {SEEN_CATEGORIES} seen and {UNSEEN_CATEGORIES} "
            f"unseen categories, got {len(seen)} and {len(unseen)}."
        )
    overlap = set(seen) & set(unseen)
    if overlap:
        raise ValueError(f"Seen and unseen metric rows overlap: {sorted(overlap)}.")
    if split is not None:
        if set(seen) != set(split.seen):
            raise ValueError("Seen metric categories do not exactly match the saved split.")
        if set(unseen) != set(split.unseen):
            raise ValueError("Unseen metric categories do not exactly match the saved split.")

    seen_macro = _macro(seen)
    unseen_macro = _macro(unseen)
    s_cls = 100.0 * (seen_macro["i_auroc"] + seen_macro["i_aupr"]) / 2.0
    s_seg = 100.0 * (
        seen_macro["p_auroc"] + seen_macro["p_aupr"] + seen_macro["p_f1max"]
    ) / 3.0
    s_zs = 100.0 * sum(unseen_macro.values()) / 5.0
    component_values = {"S_cls": s_cls, "S_seg": s_seg, "S_zs": s_zs}
    weights = {"S_cls": 0.3, "S_seg": 0.5, "S_zs": 0.2}

    def display_macro(macro: Mapping[str, float]) -> dict[str, float]:
        return {
            _DISPLAY_NAMES[name]: 100.0 * macro[name] for name in _DISPLAY_NAMES
        }

    component_specs = {
        "S_cls": ("seen", ["I-ROC", "I-PR"]),
        "S_seg": ("seen", ["P-ROC", "P-PR", "P-F1max"]),
        "S_zs": (
            "unseen",
            ["I-ROC", "I-PR", "P-ROC", "P-PR", "P-F1max"],
        ),
    }
    components = {}
    for name, value in component_values.items():
        scope, metrics = component_specs[name]
        components[name] = {
            "scope": scope,
            "metrics": metrics,
            "score": value,
            "weight": weights[name],
            "weighted_points": weights[name] * value,
        }

    return {
        "score_scale": "0-100",
        "input_metric_scale": "0-1",
        "formula": "total_score = 0.3*S_cls + 0.5*S_seg + 0.2*S_zs",
        "seen": {
            "category_count": len(seen),
            "categories": sorted(seen),
            "macro_metrics": display_macro(seen_macro),
        },
        "unseen": {
            "category_count": len(unseen),
            "categories": sorted(unseen),
            "macro_metrics": display_macro(unseen_macro),
        },
        "components": components,
        "total_score": sum(
            weights[name] * component_values[name] for name in component_values
        ),
    }


def _validated_report_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise ValueError(f"{label} must be in [0, 100].")
    return numeric


def render_unseen_evaluation_report(
    scoring_result: Mapping[str, Any],
    split: CategorySplit,
    *,
    latency_seconds: float | None = None,
    latency_summary: Mapping[str, Any] | None = None,
) -> str:
    """Render a concise Chinese Markdown report without performing file I/O.

    ``latency_seconds`` keeps compatibility with a single measured latency.
    ``latency_summary`` accepts ``mean_seconds``, ``p95_seconds``,
    ``max_seconds``, ``threshold_seconds``, and ``valid``.  The two latency forms
    are mutually exclusive.  In summary mode, ``valid`` is the authoritative
    strict-evaluation result and is never inferred again from an aggregate.
    """

    if not isinstance(scoring_result, Mapping):
        raise ValueError("scoring_result must be a mapping.")
    if scoring_result.get("score_scale") != "0-100":
        raise ValueError("scoring_result must use the 0-100 score scale.")

    macro_values: dict[str, dict[str, float]] = {}
    for scope_name, expected_categories in (
        ("seen", split.seen),
        ("unseen", split.unseen),
    ):
        scope = scoring_result.get(scope_name)
        if not isinstance(scope, Mapping):
            raise ValueError(f"scoring_result is missing the {scope_name} section.")
        categories = scope.get("categories")
        if not isinstance(categories, (list, tuple)) or tuple(sorted(categories)) != (
            expected_categories
        ):
            raise ValueError(
                f"{scope_name} report categories do not exactly match the saved split."
            )
        if scope.get("category_count") != len(expected_categories):
            raise ValueError(f"Invalid {scope_name} category_count in scoring_result.")
        macro = scope.get("macro_metrics")
        if not isinstance(macro, Mapping):
            raise ValueError(f"scoring_result is missing {scope_name} macro metrics.")
        if set(macro) != set(_DISPLAY_NAMES.values()):
            raise ValueError(f"Invalid {scope_name} macro metric fields.")
        macro_values[scope_name] = {
            name: _validated_report_score(macro[name], f"{scope_name} {name}")
            for name in _DISPLAY_NAMES.values()
        }
    components = scoring_result.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("scoring_result is missing score components.")
    weights = {"S_cls": 0.3, "S_seg": 0.5, "S_zs": 0.2}
    component_scores: dict[str, float] = {}
    for name, expected_weight in weights.items():
        component = components.get(name)
        if not isinstance(component, Mapping):
            raise ValueError(f"scoring_result is missing {name}.")
        if component.get("weight") != expected_weight:
            raise ValueError(f"{name} must use weight {expected_weight}.")
        component_scores[name] = _validated_report_score(
            component.get("score"), name
        )

    calculated_total = sum(
        weights[name] * component_scores[name] for name in component_scores
    )
    reported_total = _validated_report_score(
        scoring_result.get("total_score"), "total_score"
    )
    if not math.isclose(calculated_total, reported_total, abs_tol=1e-8):
        raise ValueError("total_score is inconsistent with the weighted components.")

    if latency_seconds is not None and latency_summary is not None:
        raise ValueError(
            "latency_seconds and latency_summary cannot be provided together."
        )

    latency_table: list[str]
    if latency_summary is not None:
        if not isinstance(latency_summary, Mapping):
            raise ValueError("latency_summary must be a mapping or None.")
        required_latency_fields = {
            "mean_seconds",
            "p95_seconds",
            "max_seconds",
            "threshold_seconds",
            "valid",
        }
        missing = required_latency_fields - set(latency_summary)
        if missing:
            raise ValueError(
                f"latency_summary is missing fields: {sorted(missing)}."
            )
        latency_values: dict[str, float] = {}
        for field in (
            "mean_seconds",
            "p95_seconds",
            "max_seconds",
            "threshold_seconds",
        ):
            value = latency_summary[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"latency_summary.{field} must be numeric.")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"latency_summary.{field} must be finite and non-negative."
                )
            latency_values[field] = numeric
        if latency_values["threshold_seconds"] <= 0.0:
            raise ValueError("latency_summary.threshold_seconds must be positive.")
        valid = latency_summary["valid"]
        if not isinstance(valid, bool):
            raise ValueError("latency_summary.valid must be a boolean.")
        latency_status = "通过" if valid else "不通过"
        latency_table = [
            "| Mean | P95 | Max | 限制 | 严格判定 |",
            "|---:|---:|---:|---:|:---:|",
            (
                f"| {latency_values['mean_seconds']:.3f} s/图 "
                f"| {latency_values['p95_seconds']:.3f} s/图 "
                f"| {latency_values['max_seconds']:.3f} s/图 "
                f"| ≤ {latency_values['threshold_seconds']:.3f} s/图 "
                f"| **{latency_status}** |"
            ),
        ]
    elif latency_seconds is None:
        latency_text = "未提供"
        latency_status = "未判定"
        latency_table = [
            "| 实测延迟 | 要求 | 判定 |",
            "|---:|---:|:---:|",
            f"| {latency_text} | ≤ 1.000 s/图 | **{latency_status}** |",
        ]
    else:
        if isinstance(latency_seconds, bool) or not isinstance(
            latency_seconds, (int, float)
        ):
            raise ValueError("latency_seconds must be numeric or None.")
        latency_value = float(latency_seconds)
        if not math.isfinite(latency_value) or latency_value < 0.0:
            raise ValueError("latency_seconds must be finite and non-negative.")
        latency_text = f"{latency_value:.3f} s/图"
        latency_status = "通过" if latency_value <= 1.0 else "不通过"
        latency_table = [
            "| 实测延迟 | 要求 | 判定 |",
            "|---:|---:|:---:|",
            f"| {latency_text} | ≤ 1.000 s/图 | **{latency_status}** |",
        ]

    metric_names = ("I-ROC", "I-PR", "P-ROC", "P-PR", "P-F1max")
    lines = [
        "# Seen/Unseen 无监督异常检测评估报告",
        "",
        (
            f"类别划分：随机种子 `{split.seed}`；Seen `{len(split.seen)}` 类，"
            f"Unseen `{len(split.unseen)}` 类，Unused `{len(split.unused)}` 类。"
        ),
        f"划分校验值（SHA-256）：`{split.categories_sha256}`。",
        "",
        "## 指标宏平均（%）",
        "",
        "| 测试组 | " + " | ".join(metric_names) + " |",
        "|---|" + "---:|" * len(metric_names),
    ]
    for scope_name, display_name in (("seen", "Seen"), ("unseen", "Unseen")):
        values = macro_values[scope_name]
        lines.append(
            f"| {display_name} | "
            + " | ".join(f"{values[name]:.2f}" for name in metric_names)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 加权评分（满分 100）",
            "",
            "- `S_cls = mean(Seen I-ROC, Seen I-PR)`",
            "- `S_seg = mean(Seen P-ROC, Seen P-PR, Seen P-F1max)`",
            (
                "- `S_zs = mean(Unseen I-ROC, Unseen I-PR, Unseen P-ROC, "
                "Unseen P-PR, Unseen P-F1max)`"
            ),
            "- `总分 = 0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs`",
            "",
            "| 分项 | 得分 | 权重 | 加权分 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("S_cls", "S_seg", "S_zs"):
        score = component_scores[name]
        weight = weights[name]
        lines.append(f"| {name} | {score:.2f} | {weight:.1f} | {score * weight:.2f} |")
    lines.extend(
        [
            f"| **总分** |  |  | **{reported_total:.2f}** |",
            "",
            "## 单帧延迟",
            "",
        ]
    )
    lines.extend(latency_table)
    lines.append("")
    return "\n".join(lines)
