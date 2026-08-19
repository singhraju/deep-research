"""Tests for pattern visuals (drill-down paths + reimbursement payer table)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from ui import pattern_visuals
from ui.pattern_visuals import (
    build_drill_down_paths,
    render_drill_down_paths,
    render_payer_summary_table,
    render_policy_links_expander,
    render_recommended_actions,
)


# ---------- drill-down path builder ----------


SAMPLE_CARDS: List[Dict[str, Any]] = [
    {
        "card_id": "c1",
        "source_entity": {"type": "states", "name": "CO"},
        "dimensions": {
            "service_area_state": "CO",
            "product_description": "HMO",
            "facility_type": "ACUTE HOSPITAL",
            "pa_required_code": "Y",
        },
        "context_dimensions": {
            "er_admit_indicator": "Y",
            "rendering_provider_name": "UCHEALTH UNIVERSITY OF COLORADO HOSPITAL",
            "drg_name": "Infectious and Parasitic Diseases with O.R. Procedures with Other Variations",
            "primary_diagnosis_name": "LIVER TRANSPLANT FAILURE",
        },
        "metrics": {
            "value": {"delta": 19_100_000},
            "explainer": {"admissions": {"delta": 485}},
        },
    },
    {
        "card_id": "c2",
        "source_entity": {"type": "providers", "name": "UCI MEDICAL CENTER"},
        "dimensions": {
            "service_area_state": "CA",
            "product_description": "HMO",
            "facility_type": "ACUTE HOSPITAL",
            "pa_required_code": "N",
        },
        "context_dimensions": {
            "er_admit_indicator": "Y",
            "drg_name": "Craniotomy with Major Device Implant or Acute Complex CNS Principal Diagnosis",
            "primary_diagnosis_name": "SEPSIS UNSPECIFIED ORGANISM",
        },
        "metrics": {
            "value": {"delta": 12_900_000},
            "explainer": {"admissions": {"delta": 120}},
        },
    },
    {
        "card_id": "c3-tech",
        "source_entity": {"type": "states", "name": "UNKNOWN"},  # filtered out
        "dimensions": {"service_area_state": "AUTH_CODE_DISTRIBUTION"},
        "metrics": {"value": {"delta": 1_000}},
    },
    {
        "card_id": "c4-dup",  # same business path as c1, dedupes
        "source_entity": {"type": "states", "name": "CO"},
        "dimensions": {
            "service_area_state": "CO",
            "product_description": "HMO",
            "facility_type": "ACUTE HOSPITAL",
            "pa_required_code": "Y",
        },
        "context_dimensions": {
            "er_admit_indicator": "Y",
            "rendering_provider_name": "UCHEALTH UNIVERSITY OF COLORADO HOSPITAL",
            "drg_name": "Infectious and Parasitic Diseases with O.R. Procedures with Other Variations",
            "primary_diagnosis_name": "LIVER TRANSPLANT FAILURE",
        },
        "metrics": {
            "value": {"delta": 500_000},
            "explainer": {"admissions": {"delta": 15}},
        },
    },
]


def test_build_drill_down_paths_basic():
    pattern = {"source_card_ids": ["c1", "c2", "c3-tech", "c4-dup", "missing"]}
    paths = build_drill_down_paths(pattern, SAMPLE_CARDS)

    # 'c3-tech' filtered, 'missing' ignored, 'c4-dup' merges into 'c1'.
    assert len(paths) == 2

    top = paths[0]
    assert "[State: CO]" in top["path"]
    assert "HMO (ER)" in top["path"]
    assert "Acute Hospital" not in top["path"]  # facility_type as-typed, not title-cased here
    assert "ACUTE HOSPITAL" in top["path"]
    assert "PA Y" in top["path"]
    assert "UCHEALTH" in top["path"]
    assert "LIVER TRANSPLANT FAILURE" in top["path"]
    assert top["delta"] == pytest.approx(19_600_000.0)  # 19.1M + 0.5M
    assert top["admissions"] == pytest.approx(500.0)  # 485 + 15
    assert top["count"] == 2

    second = paths[1]
    assert "[Provider: UCI MEDICAL CENTER]" in second["path"]
    assert second["delta"] == pytest.approx(12_900_000.0)
    assert second["count"] == 1


def test_build_drill_down_paths_no_source_cards():
    assert build_drill_down_paths({}, SAMPLE_CARDS) == []
    assert build_drill_down_paths({"source_card_ids": []}, SAMPLE_CARDS) == []


def test_build_drill_down_paths_truncates_long_clinical_strings():
    long_drg = "X" * 60
    long_diag = "Y" * 50
    cards = [
        {
            "card_id": "c1",
            "source_entity": {"type": "states", "name": "CA"},
            "dimensions": {"service_area_state": "CA", "product_description": "PPO"},
            "context_dimensions": {"drg_name": long_drg, "primary_diagnosis_name": long_diag},
            "metrics": {"value": {"delta": 100}},
        }
    ]
    paths = build_drill_down_paths({"source_card_ids": ["c1"]}, cards)
    assert len(paths) == 1
    text = paths[0]["path"]
    assert ("X" * 42 + "...") in text  # drg truncated
    assert ("Y" * 32 + "...") in text  # diagnosis truncated


def test_build_drill_down_paths_requires_two_components():
    cards = [
        {
            "card_id": "c1",
            "source_entity": {"type": "states", "name": "CA"},
            # Empty dimensions → no filter component, only the entity → one component → dropped.
            "dimensions": {},
            "context_dimensions": {},
            "metrics": {"value": {"delta": 100}},
        }
    ]
    assert build_drill_down_paths({"source_card_ids": ["c1"]}, cards) == []


def test_build_drill_down_paths_sorted_by_absolute_delta():
    cards = [
        {
            "card_id": "small",
            "source_entity": {"type": "states", "name": "AL"},
            "dimensions": {"service_area_state": "AL", "product_description": "PPO"},
            "context_dimensions": {},
            "metrics": {"value": {"delta": 1_000}},
        },
        {
            "card_id": "big-neg",
            "source_entity": {"type": "states", "name": "TX"},
            "dimensions": {"service_area_state": "TX", "product_description": "EPO"},
            "context_dimensions": {},
            "metrics": {"value": {"delta": -500_000}},  # bigger by absolute value
        },
    ]
    paths = build_drill_down_paths({"source_card_ids": ["small", "big-neg"]}, cards)
    assert [p["delta"] for p in paths] == [-500_000.0, 1_000.0]


# ---------- streamlit-facing renderers ----------


def _patch_streamlit(monkeypatch) -> Dict[str, List[Any]]:
    calls: Dict[str, List[Any]] = {"markdown": [], "caption": [], "info": [], "json": []}

    def _record(name):
        def _fn(*args, **kwargs):
            calls[name].append((args, kwargs))
        return _fn

    monkeypatch.setattr(pattern_visuals.st, "markdown", _record("markdown"))
    monkeypatch.setattr(pattern_visuals.st, "caption", _record("caption"))
    monkeypatch.setattr(pattern_visuals.st, "info", _record("info"))
    monkeypatch.setattr(pattern_visuals.st, "json", _record("json"))
    return calls


def test_render_drill_down_paths_emits_panel(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    pattern = {
        "source_card_ids": ["c1"],
        "impact_summary": {"estimated_delta": "+$52.6M"},
    }
    rendered = render_drill_down_paths(pattern, SAMPLE_CARDS)
    assert rendered is True
    assert calls["markdown"], "should emit at least one markdown block"
    blob = "".join(args[0] for args, _ in calls["markdown"])
    assert "DRILL-DOWN PATHS" in blob
    assert "[State: CO]" in blob
    assert "+$52.6M" in blob
    assert "Δ$19.1M" in blob  # formatted impact


def test_render_drill_down_paths_returns_false_when_no_paths(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    assert render_drill_down_paths({}, []) is False
    assert calls["markdown"] == []


def test_render_payer_summary_table_renders(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    summary_table = {
        "title": "Payer Policy Summary",
        "subtitle": "Cesarean PA/Stay",
        "columns": [
            {"id": "payer_org", "label": "Payer Organization", "type": "text"},
            {"id": "bundle", "label": "Global Maternity Bundle", "type": "text"},
            {"id": "appeals", "label": "Appeals Process\n(Documented)", "type": "badge"},
        ],
        "rows": [
            {
                "payer_org": "Elevance Health",
                "bundle": "Subsequent multi-birth deliveries pay 50%.",
                "appeals": "Documented",
            },
            {
                "payer_org": "Cigna",
                "bundle": "Global maternity CPTs bundle antepartum.",
                "appeals": "-",
            },
        ],
    }
    assert render_payer_summary_table(summary_table) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    assert "Payer Policy Summary" in body
    assert "Elevance Health" in body
    assert "Cigna" in body
    assert "Global Maternity Bundle" in body
    # Newline in label becomes <br>
    assert "Appeals Process<br>(Documented)" in body
    # Badge styling appears for the populated badge cell
    assert "Documented" in body
    # "-" should normalize to em-dash
    assert "—" in body


def test_render_payer_summary_table_skips_empty(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    assert render_payer_summary_table(None) is False
    assert render_payer_summary_table({}) is False
    assert render_payer_summary_table({"columns": [], "rows": []}) is False
    assert calls["markdown"] == []


def test_render_recommended_actions_renders_callout(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "priority": "High",
            "description": "Apply MPPR edit to deny separate reimbursement for 59400, 59409.",
        },
        {
            "rank": 2,
            "priority": "Medium",
            "description": "On multiple-birth claims, require modifier 59 for vaginal deliveries.",
        },
    ]
    assert render_recommended_actions(actions) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    assert "Recommended Action" in body
    assert "Recommendation 1" in body
    assert "Recommendation 2" in body
    assert "Apply MPPR edit" in body
    assert "modifier 59" in body
    # Priority chips
    assert "HIGH" in body or "High" in body


def test_render_recommended_actions_returns_false_when_empty(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    assert render_recommended_actions(None) is False
    assert render_recommended_actions([]) is False
    assert render_recommended_actions([{}]) is False
    assert calls["markdown"] == []


def test_render_policy_links_expander_renders(monkeypatch):
    expander_cm = MagicMock()
    expander_cm.__enter__ = MagicMock(return_value=expander_cm)
    expander_cm.__exit__ = MagicMock(return_value=False)
    expander_mock = MagicMock(return_value=expander_cm)
    monkeypatch.setattr(pattern_visuals.st, "expander", expander_mock)

    markdown_calls: List[Any] = []
    monkeypatch.setattr(
        pattern_visuals.st,
        "markdown",
        lambda *a, **kw: markdown_calls.append((a, kw)),
    )

    policies = [
        {
            "payer_name": "Cigna",
            "policy_title": "Ambulance Services",
            "policy_url": "https://example.com/cigna-ambulance.pdf",
            "effective_date": "07/15/2010",
        },
        {
            "payer_name": "Elevance Health",
            "title": "Tracheostomy Supplies",
            "policy_url": "https://example.com/elevance-tracheostomy.pdf",
        },
        {
            "payer_name": "Skipped",
            "policy_title": "",  # empty title filtered out
        },
    ]
    assert render_policy_links_expander(policies) is True
    expander_mock.assert_called_once()
    args, kwargs = expander_mock.call_args
    label = args[0] if args else kwargs.get("label", "")
    assert "2" in label  # only 2 valid policies (Skipped dropped)
    assert kwargs.get("expanded") is False

    body = "".join(args[0] for args, _ in markdown_calls)
    assert "[Ambulance Services](https://example.com/cigna-ambulance.pdf)" in body
    assert "[Tracheostomy Supplies](https://example.com/elevance-tracheostomy.pdf)" in body
    assert "Cigna" in body
    assert "Elevance Health" in body
    # Effective date rendered when present
    assert "07/15/2010" in body


def test_render_policy_links_expander_returns_false_when_empty(monkeypatch):
    monkeypatch.setattr(pattern_visuals.st, "expander", MagicMock())
    assert render_policy_links_expander(None) is False
    assert render_policy_links_expander([]) is False
    assert render_policy_links_expander([{"policy_title": ""}]) is False
