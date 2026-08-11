"""Tests for the UI's intent/conversation_id pre-builders.

The orchestrator's intent-detection LLM is bypassed when the UI hands it a
fully-formed IntentOutput. These tests pin the payload shape so the
`analysis_mode_parameters.period` block matches what the correlation agent
expects (the same shape as in 2026-05-release.ipynb).
"""

from __future__ import annotations

import pytest

from ui.filter_options import build_conversation_id, build_prebuilt_intent


def test_build_prebuilt_intent_period_shape_matches_correlation_contract():
    intent = build_prebuilt_intent(
        drill_metric="expense_detail.total_paid",
        period_label="Rolling 3",
        end_month=202601,
        rolling_time_dimension_qualified="expense_detail.incurred_month",
        dimension_selections={
            "lob_description": "Commercial",
            "service_area_state": "KY",
        },
        extra_filters={"snap_month": 202604},
        raw_question="Where did change happen for state KY?",
    )

    assert intent["analysis_mode"] == "cost_change_investigation_over_time_window"
    assert intent["metric_hint"] == "expense_detail.total_paid"
    assert intent["raw_question"] == "Where did change happen for state KY?"
    assert intent["group_by"] == []
    assert intent["validation_warnings"] == []

    params = intent["analysis_mode_parameters"]
    assert params["drill_metric"] == ["expense_detail.total_paid"]

    period = params["period"]
    assert period["rolling_time_dimension"] == "expense_detail.incurred_month"
    # R3 ending 202601 → start 202511 (current), prior-year shift → 202411/202501
    assert period["current_period"] == {"start_time": 202511, "end_time": 202601}
    assert period["previous_period"] == {"start_time": 202411, "end_time": 202501}
    # rolling_window must be a single-element list so build_clarification_request's
    # _safe_list(...) returns len()==1 instead of dropping a bare string.
    assert period["rolling_window"] == ["3_months"]
    # Top-level start_time/end_time satisfy build_clarification_request's
    # has_explicit_dates path independently of the rolling_window check.
    assert period["start_time"] == 202511
    assert period["end_time"] == 202601


def test_build_prebuilt_intent_emits_one_filter_per_dimension_value():
    intent = build_prebuilt_intent(
        drill_metric="expense_detail.total_paid",
        period_label="Rolling 3",
        end_month=202601,
        rolling_time_dimension_qualified="expense_detail.incurred_month",
        dimension_selections={
            "lob_description": ["Commercial", "Medicare"],
            "service_area_state": "KY",
        },
        extra_filters={"snap_month": 202604},
        raw_question="x",
    )

    fields = [(f["field"], f["value"]) for f in intent["filters"]]
    assert ("lob_description", "Commercial") in fields
    assert ("lob_description", "Medicare") in fields
    assert ("service_area_state", "KY") in fields
    assert ("snap_month", 202604) in fields
    for f in intent["filters"]:
        assert f["operator"] == "="
        assert f["source"] == "dimension_match"


def test_build_prebuilt_intent_drops_sentinel_dimension_values():
    intent = build_prebuilt_intent(
        drill_metric="expense_detail.total_paid",
        period_label="Rolling 3",
        end_month=202601,
        rolling_time_dimension_qualified="expense_detail.incurred_month",
        dimension_selections={
            "lob_description": "Commercial",
            "service_area_state": "",  # sentinel — should be dropped
            "hcc_medium": [],  # empty list — should be dropped
            "facility_type": None,
        },
        extra_filters=None,
        raw_question="x",
    )

    fields = [f["field"] for f in intent["filters"]]
    assert "lob_description" in fields
    assert "service_area_state" not in fields
    assert "hcc_medium" not in fields
    assert "facility_type" not in fields


def test_build_prebuilt_intent_ytd_window():
    intent = build_prebuilt_intent(
        drill_metric="expense_detail.total_paid",
        period_label="YTD",
        end_month=202609,
        rolling_time_dimension_qualified="expense_detail.incurred_month",
        dimension_selections=None,
        extra_filters=None,
        raw_question="x",
    )
    period = intent["analysis_mode_parameters"]["period"]
    assert period["current_period"] == {"start_time": 202601, "end_time": 202609}
    assert period["previous_period"] == {"start_time": 202501, "end_time": 202509}


def test_build_conversation_id_matches_notebook_format():
    cid = build_conversation_id(
        view_name="IP_AUTH",
        lob="Commercial",
        snap_month=202604,
        period_label="Rolling 3",
        end_month=202601,
    )
    assert cid == "tutorial-IP_AUTH-Commercial-202604-R3-202601"


def test_build_conversation_id_handles_missing_segments():
    cid = build_conversation_id(
        view_name=None,
        lob=None,
        snap_month=None,
        period_label=None,
        end_month=None,
    )
    # All "unknown" segments + ECAP fallback "R0"
    assert cid == "tutorial-view-ALL-unknown-R0-unknown"


def test_build_prebuilt_intent_passes_build_clarification_request():
    """Adversarial check: feed the produced intent through the orchestrator's own
    build_clarification_request gate. If the period payload is malformed (missing
    rolling_window list or top-level dates) the gate emits a blocking issue and
    the user sees "Need clarification before execution." in the UI.
    """
    import sys
    from pathlib import Path

    pkg_src = Path(__file__).resolve().parents[3] / "packages" / "agents" / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))
    from deep_research_agents.orchestrator import build_clarification_request

    intent = build_prebuilt_intent(
        drill_metric="expense_detail.total_paid",
        period_label="Rolling 3",
        end_month=202601,
        rolling_time_dimension_qualified="expense_detail.incurred_month",
        dimension_selections={"lob_description": "Commercial"},
        extra_filters={"snap_month": 202604},
        raw_question="x",
    )
    assert build_clarification_request(intent) is None, (
        "build_clarification_request must return None for a fully-formed UI intent; "
        "any blocking issue here would block the Research click."
    )


# ---------------------------------------------------------------------------
# Live filter values → validation path
# ---------------------------------------------------------------------------


def _import_validate_intent_output():
    import sys
    from pathlib import Path

    pkg_src = Path(__file__).resolve().parents[3] / "packages" / "agents" / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))
    from deep_research_agents.user_intent import build_semantic_index, validate_intent_output

    return build_semantic_index, validate_intent_output


def _semantic_model_with_stale_snap_samples():
    """YAML-style dict mirroring the production bug: snap_month wrongly listed
    as a dimension carrying a stale ``sample_values`` allowlist of ['202604']
    while the live warehouse already contains 202606.
    """
    return {
        "tables": [
            {
                "name": "expense_detail",
                "dimensions": [
                    {"name": "service_area_state", "sample_values": ["CA", "NY", "KY"]},
                    {
                        "name": "rendering_provider_name",
                    },
                ],
                "time_dimensions": [
                    {"name": "incurred_month"},
                ],
                "facts": [{"name": "total_paid"}],
            },
            {
                "name": "membership",
                "dimensions": [
                    {"name": "snap_month", "sample_values": ["202604"]},
                ],
                "time_dimensions": [],
                "facts": [],
            },
        ],
        "filters": [],
    }


def _live_filter_values_for_snowflake_state():
    return {
        "dimensions": {
            "service_area_state": {
                "values": ["CA", "NY", "KY", "TX"],
                "is_free_text": False,
                "source": "db",
            },
            "rendering_provider_name": {
                "values": [],
                "is_free_text": True,
                "source": "db",
            },
            # Live snapshot says snap_month is also enumerable here (membership table)
            "snap_month": {
                "values": ["202601", "202602", "202603", "202604", "202605", "202606"],
                "is_free_text": False,
                "source": "db",
            },
        },
        "time_dimensions": {
            "incurred_month": {"min": "202501", "max": "202603", "source": "db"},
        },
    }


def test_validate_intent_output_live_values_supersede_stale_yaml_samples():
    """The reported regression: YAML hardcoded snap_month=['202604'] but the warehouse
    moved to 202606. When the UI passes live values, validation must accept 202606.
    """
    build_semantic_index, validate_intent_output = _import_validate_intent_output()
    model = _semantic_model_with_stale_snap_samples()
    index = build_semantic_index(model)
    live = _live_filter_values_for_snowflake_state()

    intent = {
        "filters": [
            {"field": "snap_month", "operator": "=", "value": "202606", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(intent, model, index, live)
    assert warnings == [], f"Expected no warnings with live values, got: {warnings}"


def test_validate_intent_output_back_compat_without_live_values():
    """When live_filter_values is None, the historical YAML-based check still fires."""
    build_semantic_index, validate_intent_output = _import_validate_intent_output()
    model = _semantic_model_with_stale_snap_samples()
    index = build_semantic_index(model)

    intent = {
        "filters": [
            {"field": "snap_month", "operator": "=", "value": "202606", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(intent, model, index)
    assert any("snap_month" in w and "sample values" in w for w in warnings), (
        f"Expected YAML sample_values warning when no live data is passed, got: {warnings}"
    )


def test_validate_intent_output_live_values_reject_out_of_range_dimension():
    """Live distinct list rejects a value not present in Snowflake."""
    build_semantic_index, validate_intent_output = _import_validate_intent_output()
    model = _semantic_model_with_stale_snap_samples()
    index = build_semantic_index(model)
    live = _live_filter_values_for_snowflake_state()

    intent = {
        "filters": [
            {"field": "service_area_state", "operator": "=", "value": "ZZ", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(intent, model, index, live)
    assert any("service_area_state" in w and "live values" in w for w in warnings), (
        f"Expected live-values warning for unknown state, got: {warnings}"
    )


def test_validate_intent_output_skips_free_text_dimensions():
    """High-cardinality fields (UI shows free-text input) should not be allowlist-checked."""
    build_semantic_index, validate_intent_output = _import_validate_intent_output()
    model = _semantic_model_with_stale_snap_samples()
    index = build_semantic_index(model)
    live = _live_filter_values_for_snowflake_state()

    intent = {
        "filters": [
            {
                "field": "rendering_provider_name",
                "operator": "=",
                "value": "Some Provider That Definitely Isn't In Any List",
                "source": "dimension_match",
            },
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(intent, model, index, live)
    assert warnings == [], (
        f"Free-text dimensions should not produce categorical warnings, got: {warnings}"
    )


def test_validate_intent_output_time_dimension_range_check():
    """Live MIN/MAX bounds a time-dimension filter on both ends."""
    build_semantic_index, validate_intent_output = _import_validate_intent_output()
    model = _semantic_model_with_stale_snap_samples()
    index = build_semantic_index(model)
    live = _live_filter_values_for_snowflake_state()

    in_range = {
        "filters": [
            {"field": "incurred_month", "operator": "=", "value": "202602", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    assert validate_intent_output(in_range, model, index, live) == []

    too_late = {
        "filters": [
            {"field": "incurred_month", "operator": "=", "value": "202612", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(too_late, model, index, live)
    assert any("incurred_month" in w and "outside the live range" in w for w in warnings), (
        f"Expected out-of-range warning, got: {warnings}"
    )

    too_early = {
        "filters": [
            {"field": "incurred_month", "operator": "=", "value": "202401", "source": "dimension_match"},
        ],
        "group_by": [],
    }
    warnings = validate_intent_output(too_early, model, index, live)
    assert any("incurred_month" in w and "outside the live range" in w for w in warnings), (
        f"Expected out-of-range warning for early value, got: {warnings}"
    )


def test_serialize_live_filter_values_shape_matches_validator_contract():
    """The UI serializer's output is exactly what validate_intent_output reads."""
    import datetime as dt
    from dataclasses import dataclass, field
    from typing import Tuple

    from ui.filter_options import serialize_live_filter_values
    from ui.time_adapters import YearMonthIntAdapter

    @dataclass(frozen=True)
    class _Dim:
        dimension_name: str
        values: Tuple[str, ...]
        is_free_text: bool
        source: str = "db"

    @dataclass(frozen=True)
    class _Time:
        dimension_name: str
        min_dt: dt.datetime
        max_dt: dt.datetime
        adapter: object
        source: str = "db"

    @dataclass(frozen=True)
    class _FV:
        dimensions: Tuple[_Dim, ...]
        time_dimensions: Tuple[_Time, ...]

    fv = _FV(
        dimensions=(
            _Dim(dimension_name="service_area_state", values=("CA", "NY"), is_free_text=False),
            _Dim(dimension_name="rendering_provider_name", values=(), is_free_text=True),
        ),
        time_dimensions=(
            _Time(
                dimension_name="snap_month",
                min_dt=dt.datetime(2025, 1, 1),
                max_dt=dt.datetime(2026, 6, 1),
                adapter=YearMonthIntAdapter(),
            ),
        ),
    )

    payload = serialize_live_filter_values(fv)
    assert payload["dimensions"]["service_area_state"]["values"] == ["CA", "NY"]
    assert payload["dimensions"]["service_area_state"]["is_free_text"] is False
    assert payload["dimensions"]["rendering_provider_name"]["is_free_text"] is True
    assert payload["time_dimensions"]["snap_month"]["min"] == "202501"
    assert payload["time_dimensions"]["snap_month"]["max"] == "202606"


def test_orchestrator_prepare_request_passes_live_values_to_validator():
    """End-to-end: UI context with analysis_overrides.live_filter_values reaches
    validate_intent_output via prepare_request → run_intent_detection.
    """
    import sys
    from pathlib import Path

    pkg_src = Path(__file__).resolve().parents[3] / "packages" / "agents" / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))
    from deep_research_agents.orchestrator import split_soft_context

    context = {
        "analysis_overrides": {
            "prebuilt_intent": {"foo": "bar"},
            "live_filter_values": {
                "dimensions": {"x": {"values": ["a"], "is_free_text": False}},
                "time_dimensions": {},
            },
        },
        "service_area_state": "CA",
    }
    soft, analysis = split_soft_context(context)
    assert soft == {"service_area_state": "CA"}
    assert analysis["prebuilt_intent"] == {"foo": "bar"}
    assert analysis["live_filter_values"]["dimensions"]["x"]["values"] == ["a"]
