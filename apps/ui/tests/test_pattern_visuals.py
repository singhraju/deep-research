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


def test_render_recommended_actions_renders_cited_policies(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "priority": "Medium",
            "description": "Deny A0427 without origin/destination modifiers.",
            "citation": [
                "Cigna · Ambulance Services — https://ex.com/cigna.pdf",
                "United Health · Hospital Based Ambulance Policy, Facility "
                "— https://ex.com/uhc-hba.pdf",
                "Nonmatching · Ghost Policy — https://ex.com/nomatch.pdf",
            ],
        }
    ]
    policies = [
        {
            "policy_url": "https://ex.com/cigna.pdf",
            "policy_title": "Ambulance Services",
            "payer_name": "Cigna",
            "effective_date": "07/15/2010",
            "edit_rule_facts": {
                "target_codes": ["A0427", "A0429"],
                "required_modifiers": ["QM", "QN"],
                "action_types": ["allow_with_conditions"],
            },
        },
        {
            "policy_url": "https://ex.com/uhc-hba.pdf",
            "policy_title": "Hospital Based Ambulance Policy, Facility",
            "payer_name": "United Health",
            "effective_date": "",
            "edit_rule_facts": {
                "target_codes": ["A0427"],
                "required_modifiers": ["QM"],
                "action_types": [],
            },
        },
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    # Hyperlinked policy titles for both matched citations
    assert 'href="https://ex.com/cigna.pdf"' in body
    assert ">Ambulance Services</a>" in body
    assert 'href="https://ex.com/uhc-hba.pdf"' in body
    assert "Hospital Based Ambulance Policy, Facility" in body
    # Supporting columns present
    assert "Cigna" in body
    assert "United Health" in body
    assert "07/15/2010" in body
    assert "A0427" in body
    assert "QM" in body
    assert "allow_with_conditions" in body
    # Nonmatching citation still surfaces via synthesis so cited policies
    # (e.g. quarantined records missing from individual_policies) aren't
    # silently dropped from the UI.
    assert "Ghost Policy" in body
    assert 'href="https://ex.com/nomatch.pdf"' in body


def test_render_recommended_actions_synthesizes_missing_citation(monkeypatch):
    """A citation whose URL isn't in ``policies`` still renders as a row.

    Guards the fix where cited-but-quarantined policies (e.g. Elevance) used to
    disappear from the recommendation table entirely. The row is synthesized
    from the citation's ``<payer> · <title> — <url>`` structure so the reader
    can still click through to the source.
    """

    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "description": "Some recommendation.",
            "citation": [
                "Elevance Health (external) · Ambulance Transportation "
                "— https://www.anthem.com/policy/C-1900.pdf"
            ],
        }
    ]
    policies = [
        {"policy_url": "https://ex.com/other.pdf", "policy_title": "Other"}
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    assert "Some recommendation." in body
    # Table rendered with the synthesized Elevance citation row.
    assert "<table" in body
    assert "Elevance Health (external)" in body
    assert 'href="https://www.anthem.com/policy/C-1900.pdf"' in body
    assert ">Ambulance Transportation</a>" in body


def test_render_recommended_actions_matches_by_title_when_url_differs(
    monkeypatch,
):
    """When the citation's URL isn't in ``policies`` but the payer + title
    match a real policy, fall back to that record (with its edit_rule_facts)
    rather than synthesising a bare row."""

    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "description": "Deny A0427 without modifiers.",
            "citation": [
                "Elevance Health · Ambulance Transportation "
                "— https://cited-but-different.example.com/policy.pdf"
            ],
        }
    ]
    policies = [
        {
            # A different URL than what the LLM cited — historically this
            # meant the policy disappeared from the recommendation table.
            "policy_url": "https://real.example.com/elevance.pdf",
            "policy_title": "Ambulance Transportation",
            "payer_name": "Elevance Health (external)",
            "effective_date": "01/01/2020",
            "edit_rule_facts": {
                "target_codes": ["A0427", "A0429"],
                "required_modifiers": ["QM"],
                "action_types": ["deny"],
            },
        },
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    # Matched to the real record — edit_rule_facts and effective_date come
    # from ``policies``, not the (bare) citation string.
    assert "01/01/2020" in body
    assert "A0427" in body
    assert "QM" in body
    assert "deny" in body
    # Link goes to the real policy_url, not the URL from the citation.
    assert 'href="https://real.example.com/elevance.pdf"' in body


def test_render_recommended_actions_splits_sub_recommendations(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "priority": "Medium",
            "description": (
                "Recommendation 1: For A0427, deny without modifiers. "
                "Peer: CareFirst requires two-digit modifiers.\n\n"
                "Recommendation 2: For A0427, require A0425 mileage pairing. "
                "Peer: Cigna ties payment to mileage pairing.\n\n"
                "IA — Implement precertification. Cigna states ambulance "
                "transportation codes may require precertification."
            ),
            "citation": [
                "CareFirst · Ambulance Services — https://ex.com/cf.pdf",
                "Cigna · Ambulance Services — https://ex.com/cigna.pdf",
            ],
        }
    ]
    policies = [
        {
            "policy_url": "https://ex.com/cf.pdf",
            "policy_title": "Ambulance Services",
            "payer_name": "CareFirst",
            "effective_date": "01/01/2008",
            "edit_rule_facts": {
                "target_codes": ["A0427"],
                "required_modifiers": ["DH", "EH"],
                "action_types": ["allow_with_conditions"],
            },
        },
        {
            "policy_url": "https://ex.com/cigna.pdf",
            "policy_title": "Ambulance Services",
            "payer_name": "Cigna",
            "effective_date": "07/15/2010",
            "edit_rule_facts": {
                "target_codes": ["A0427"],
                "required_modifiers": [],
                "action_types": ["allow_with_conditions"],
            },
        },
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])

    # Outer "Recommendation 1" header appears once (no duplication with sub-rec heading).
    assert body.count("Recommendation 1:") == 1
    # Sub-rec labels are rendered.
    assert "Recommendation 2:" in body
    assert "IA" in body
    # Each sub-rec has its own table (Rec 1 → CareFirst, Rec 2 → Cigna, IA → Cigna).
    assert body.count("<table") == 3
    # Rec 1 table contains CareFirst but not Cigna. Grab the Rec 1 section.
    rec1_start = body.index("Recommendation 1:")
    rec2_start = body.index("Recommendation 2:")
    rec1_section = body[rec1_start:rec2_start]
    assert "CareFirst" in rec1_section
    assert "Cigna" not in rec1_section
    # Rec 2 table contains Cigna but not CareFirst.
    ia_start = body.index("IA", rec2_start)
    rec2_section = body[rec2_start:ia_start]
    assert "Cigna" in rec2_section
    assert "CareFirst" not in rec2_section


def test_render_recommended_actions_attributes_via_evidence(monkeypatch):
    """A policy is attributed to a sub-rec whose text contains one of the
    policy's evidence values, even if the payer isn't named in that sub-rec."""

    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "priority": "Medium",
            "description": (
                "Recommendation 1: Deny unless transport is filed with "
                "modifier QL.\n\n"
                "Recommendation 2: Reimburse max 1 unit per trip (each way)."
            ),
            "citation": [
                "Elevance Health (external) · Ambulance Reimbursement Policy "
                "— https://ex.com/elevance.pdf",
                "Cigna · Ambulance Services — https://ex.com/cigna.pdf",
            ],
            "evidence": [
                'Elevance Health (external) · Ambulance Reimbursement Policy: '
                '"unless transport is filed with modifier QL" [pairing_conditions]',
                'Cigna · Ambulance Services: '
                '"max 1 unit per trip (each way)" [utilization_limits]',
                # Low-signal target_codes evidence must be ignored — otherwise
                # every policy with A0427 evidence would land in every sub-rec.
                'Cigna · Ambulance Services: "A0427" [target_codes]',
            ],
        }
    ]
    policies = [
        {
            "policy_url": "https://ex.com/elevance.pdf",
            "policy_title": "Ambulance Reimbursement Policy",
            "payer_name": "Elevance Health (external)",
        },
        {
            "policy_url": "https://ex.com/cigna.pdf",
            "policy_title": "Ambulance Services",
            "payer_name": "Cigna",
        },
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])

    rec1_start = body.index("Recommendation 1:")
    rec2_start = body.index("Recommendation 2:")
    rec1_section = body[rec1_start:rec2_start]
    rec2_section = body[rec2_start:]

    # Elevance shows under Rec 1 via evidence match, even though "Elevance"
    # isn't in the Rec 1 text.
    assert "Ambulance Reimbursement Policy" in rec1_section
    # Cigna appears under Rec 2 (via 'max 1 unit per trip' evidence) but NOT
    # under Rec 1 (the shared A0427 evidence is a low-signal field).
    assert "Cigna" in rec2_section
    assert "Cigna" not in rec1_section
    # No fallback block needed when every citation is attributed somewhere.
    assert "Additional referenced policies" not in body


def test_render_recommended_actions_short_evidence_value_word_boundary(monkeypatch):
    """A 1-char evidence value like 'D' should match 'D,' but not 'D-SNP'."""

    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "description": (
                "Recommendation 1: modifiers from D, E, G.\n\n"
                "IA — plans include Blue Medicare (HMO-POS D-SNP)."
            ),
            "citation": [
                "Foo · Modifier Policy — https://ex.com/m.pdf",
            ],
            "evidence": [
                'Foo · Modifier Policy: "D" [required_modifiers]',
            ],
        }
    ]
    policies = [
        {
            "policy_url": "https://ex.com/m.pdf",
            "policy_title": "Modifier Policy",
            "payer_name": "Foo",
        }
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])

    rec1_start = body.index("Recommendation 1:")
    ia_start = body.index("IA", rec1_start + 1)
    rec1_section = body[rec1_start:ia_start]
    ia_section = body[ia_start:]

    # "D," in Rec 1 is a standalone token → attribute Foo to Rec 1.
    assert "Modifier Policy" in rec1_section
    # "D" inside "D-SNP" in IA is not a real modifier token → Foo must NOT
    # land in IA (its payer "Foo" isn't in IA either, so no fallback).
    assert "Modifier Policy" not in ia_section


def test_render_recommended_actions_shows_additional_policies_when_unmatched(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "description": (
                "Recommendation 1: Peer: Cigna requires modifiers.\n\n"
                "Recommendation 2: Peer: Cigna requires mileage."
            ),
            "citation": [
                "Cigna · Ambulance Services — https://ex.com/cigna.pdf",
                "Elevance · Base Policy — https://ex.com/elevance.pdf",
            ],
        }
    ]
    policies = [
        {
            "policy_url": "https://ex.com/cigna.pdf",
            "policy_title": "Ambulance Services",
            "payer_name": "Cigna",
        },
        {
            "policy_url": "https://ex.com/elevance.pdf",
            "policy_title": "Base Policy",
            "payer_name": "Elevance Health (external)",
        },
    ]
    assert render_recommended_actions(actions, policies) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    # Elevance isn't mentioned by name in any sub-rec, so falls into the extras.
    assert "Additional referenced policies" in body
    assert "Base Policy" in body


def test_render_recommended_actions_no_policies_backward_compatible(monkeypatch):
    calls = _patch_streamlit(monkeypatch)
    actions = [
        {
            "rank": 1,
            "priority": "High",
            "description": "Legacy call without policies argument.",
            "citation": ["Foo · Bar — https://ex.com/x.pdf"],
        }
    ]
    # Called positionally with a single arg (the legacy signature).
    assert render_recommended_actions(actions) is True
    body = "".join(args[0] for args, _ in calls["markdown"])
    assert "Legacy call without policies argument." in body
    # Citations always render — synthesised from the citation string when no
    # ``policies`` argument is provided.
    assert 'href="https://ex.com/x.pdf"' in body
    assert ">Bar</a>" in body
    assert "Foo" in body


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


# ---------- drill-down chart label helpers ----------


def test_wrap_drill_path_for_label_breaks_at_arrow_boundaries():
    from ui.pattern_visuals import _wrap_drill_path_for_label

    path = "[State: CA] → HMO + Ambulance + Sacramento → SEPSIS UNSPECIFIED ORGANISM"
    wrapped = _wrap_drill_path_for_label(path, max_line_len=30)
    # Each line breaks at a → boundary — no line stays over the limit.
    for line in wrapped.split("<br>"):
        assert len(line) <= 40  # some slack for the arrow token itself
    # → tokens are preserved when the path spans multiple hops.
    assert "→" in wrapped
    assert wrapped.count("<br>") >= 1


def test_wrap_drill_path_for_label_returns_dash_for_empty_input():
    from ui.pattern_visuals import _wrap_drill_path_for_label

    assert _wrap_drill_path_for_label("") == "—"
    assert _wrap_drill_path_for_label(None) == "—"


def test_wrap_drill_path_for_label_truncates_when_exceeding_max_lines():
    from ui.pattern_visuals import _wrap_drill_path_for_label

    long_path = " → ".join(f"segment_{i}_that_is_long" for i in range(12))
    wrapped = _wrap_drill_path_for_label(long_path, max_line_len=20, max_lines=3)
    assert wrapped.count("<br>") == 2  # exactly 3 lines
    assert wrapped.endswith("…")


def test_format_bar_detail_text_includes_all_three_pieces_when_present():
    from ui.pattern_visuals import _format_bar_detail_text

    text = _format_bar_detail_text(delta=165_000, admits=12, count=3)
    assert "+$165.0K" in text
    assert "+12 admits" in text
    assert "3 cells" in text


def test_format_bar_detail_text_omits_zero_admits_and_single_cell():
    from ui.pattern_visuals import _format_bar_detail_text

    text = _format_bar_detail_text(delta=-42_000, admits=0, count=1)
    assert "-$42.0K" in text
    assert "admits" not in text
    assert "cells" not in text


def test_build_pattern_breakdown_figure_bar_text_shows_delta_admits_cells():
    """The static chart must surface Δ paid, admits, and cell count on the bar,
    not only in the hover tooltip (which never appears in an exported PDF)."""

    from ui.pattern_visuals import build_pattern_breakdown_figure

    cards = [
        {
            "card_id": "c1",
            "source_entity": {"type": "states", "name": "CA"},
            "dimensions": {"service_area_state": "CA", "product_description": "HMO"},
            "context_dimensions": {"drg_name": "AMBULANCE TRANSPORT"},
            "metrics": {
                "value": {"delta": 165_000},
                "explainer": {"admissions": {"delta": 12}},
            },
        },
        {
            "card_id": "c2",  # dedupes into c1's aggregated path → count becomes 2
            "source_entity": {"type": "states", "name": "CA"},
            "dimensions": {"service_area_state": "CA", "product_description": "HMO"},
            "context_dimensions": {"drg_name": "AMBULANCE TRANSPORT"},
            "metrics": {
                "value": {"delta": 35_000},
                "explainer": {"admissions": {"delta": 4}},
            },
        },
    ]
    fig = build_pattern_breakdown_figure(
        {"source_card_ids": ["c1", "c2"]}, cards
    )
    assert fig is not None
    bar = fig.data[0]
    text = "".join(bar.text)
    assert "+$200.0K" in text  # aggregated delta
    assert "+16 admits" in text  # aggregated admits
    assert "2 cells" in text  # aggregated count
    # Y-axis label carries the rank chip and wrapped drill path.
    y_label = "".join(bar.y)
    assert "#1" in y_label
    assert "CA" in y_label
    assert "HMO" in y_label
    assert "AMBULANCE" in y_label
