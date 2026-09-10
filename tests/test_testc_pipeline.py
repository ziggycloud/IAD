from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from realiad_dinomaly2.testc_data import (
    SourceObject,
    canonical_digest,
    load_testc_protocol,
    select_category_objects,
)
from realiad_dinomaly2.testc_evaluation import compute_testc_score
from realiad_dinomaly2.losses import reconstruction_loss
from realiad_dinomaly2.zero_shot_engine import _map_object_logits
from realiad_dinomaly2.synthetic_anomaly import _object_anomaly_visibility


ROOT = Path(__file__).resolve().parents[1]


def _object(category: str, index: int, defect: str) -> SourceObject:
    records = tuple(
        {
            "category": category,
            "image_path": f"{category}/test/{index:04d}_C{view + 1}_x.png",
            "mask_path": None,
            "anomaly_class": defect,
            "view_id": view,
        }
        for view in range(5)
    )
    return SourceObject(category, f"{category}/test/{index:04d}", defect, records)


def test_fixed_protocol_has_disjoint_50_seen_and_50_unseen() -> None:
    protocol = load_testc_protocol(ROOT / "configs" / "testc_protocol.json")
    assert len(protocol["seen_categories"]) == 50
    assert len(protocol["unseen_categories"]) == 50
    assert len(protocol["categories"]) == 100
    assert set(protocol["seen_categories"]).isdisjoint(protocol["unseen_categories"])


def test_sampling_is_deterministic_balanced_and_defect_stratified() -> None:
    objects = [_object("part", i, "OK") for i in range(20)]
    objects += [_object("part", 100 + i, f"NG_{i % 3}") for i in range(30)]
    first = select_category_objects(objects, category="part", seed=20260909)
    second = select_category_objects(objects, category="part", seed=20260909)
    assert [item.object_id for item in first] == [item.object_id for item in second]
    assert Counter(item.object_label for item in first) == {0: 10, 1: 10}
    selected_defects = Counter(item.anomaly_class for item in first if item.object_label)
    assert max(selected_defects.values()) - min(selected_defects.values()) <= 1
    assert all(len(item.records) == 5 for item in first)


def test_score_formula_uses_category_macro_averages() -> None:
    rows = []
    for partition in ("seen", "unseen"):
        for index in range(50):
            rows.append(
                {
                    "category": f"{partition}_{index}",
                    "partition": partition,
                    "c_auroc": 0.8,
                    "c_ap": 0.6,
                    "c_f1max": 0.5,
                    "p_auroc": 0.7,
                    "p_ap": 0.5,
                    "p_f1max": 0.3,
                }
            )
    score = compute_testc_score(rows)
    assert score["s_cls"] == pytest.approx(70.0)
    assert score["s_seg"] == pytest.approx(50.0)
    assert score["s_zs"] == pytest.approx(58.0)
    assert score["total_score"] == pytest.approx(57.6)


def test_run_signature_changes_with_material_input() -> None:
    assert canonical_digest({"seed": 1, "manifest": "a"}) == canonical_digest(
        {"manifest": "a", "seed": 1}
    )
    assert canonical_digest({"seed": 1}) != canonical_digest({"seed": 2})


def test_protocol_rejects_wrong_category_count(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "seen_categories": ["only_one"],
                "unseen_categories": [],
                "normal_objects_per_category": 10,
                "anomaly_objects_per_category": 10,
                "views_per_object": 5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="50 seen and 50 unseen"):
        load_testc_protocol(path)


def test_object_scoped_loose_loss_is_micro_batch_invariant() -> None:
    encoder = torch.randn(2, 5, 4, 2, 2)
    decoder_values = encoder + 0.2 * torch.randn_like(encoder)
    decoder = decoder_values.clone().requires_grad_(True)
    together = reconstruction_loss(
        [encoder], [decoder], discard_rate=0.5, loose_loss=True,
        selection_scope="object",
    )
    together.backward()
    together_gradient = decoder.grad.detach().clone()
    decoder_separate = decoder_values.clone().requires_grad_(True)
    separate = sum(
        reconstruction_loss(
            [encoder[index:index + 1]], [decoder_separate[index:index + 1]],
            discard_rate=0.5, loose_loss=True, selection_scope="object",
        )
        for index in range(2)
    ) / 2
    separate.backward()
    assert float(together.detach()) == pytest.approx(float(separate.detach()), abs=1e-6)
    assert torch.allclose(together_gradient, decoder_separate.grad, atol=1e-6)


def test_zero_shot_map_object_loss_matches_five_view_submission_rule() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5])
    logits = torch.logit(probabilities).reshape(10, 1, 1, 1)
    labels = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    object_logits, object_labels = _map_object_logits(
        logits, labels, views_per_object=5, top_ratio=1.0, max_blend=0.5
    )
    assert object_logits.sigmoid().tolist() == pytest.approx([0.64, 0.4])
    assert object_labels.tolist() == [1.0, 0.0]


def test_zero_shot_map_object_loss_fuses_global_frame_evidence() -> None:
    local = torch.full((5, 1, 1, 1), -1.3862944)
    global_logits = torch.full((5,), 1.3862944)
    labels = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
    object_logits, object_labels = _map_object_logits(
        local,
        labels,
        global_logits=global_logits,
        global_weight=0.5,
        views_per_object=5,
        top_ratio=1.0,
        max_blend=0.0,
    )
    assert object_logits.sigmoid().item() == pytest.approx(0.5)
    assert object_labels.tolist() == [1.0]


def test_synthetic_object_visibility_balances_at_object_level() -> None:
    visible = _object_anomaly_visibility(
        10, group_size=5, object_probability=1.0, view_probability=0.0,
        device=torch.device("cpu"),
    ).reshape(2, 5)
    assert visible.any(dim=1).all()
    normal = _object_anomaly_visibility(
        10, group_size=5, object_probability=0.0, view_probability=1.0,
        device=torch.device("cpu"),
    )
    assert not normal.any()
