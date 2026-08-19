"""Unit tests for ui.filter_options.build_context_from_filters."""

import datetime as dt

from ui.filter_options import build_context_from_filters


def test_dimensions_pass_through_and_drop_sentinels():
    ctx = build_context_from_filters(
        dimension_selections={
            "service_area_state": "NY",
            "lob_description": "Commercial",
            "hcc_medium": "All",
            "facility_type": "Select...",
            "drg_name": "",
        },
        metric_name="expense_detail.total_paid",
    )
    assert ctx["service_area_state"] == "NY"
    assert ctx["lob_description"] == "Commercial"
    assert "hcc_medium" not in ctx
    assert "facility_type" not in ctx
    assert "drg_name" not in ctx
    assert ctx["drill_metric"] == "expense_detail.total_paid"


def test_multiselect_single_pick_emits_scalar():
    ctx = build_context_from_filters(
        dimension_selections={"service_area_state": ["NY"]},
    )
    assert ctx["service_area_state"] == "NY"


def test_multiselect_multi_pick_emits_list():
    ctx = build_context_from_filters(
        dimension_selections={"service_area_state": ["NY", "CA", "TX"]},
    )
    assert ctx["service_area_state"] == ["NY", "CA", "TX"]


def test_multiselect_empty_drops_key():
    ctx = build_context_from_filters(
        dimension_selections={"service_area_state": [], "lob_description": None},
    )
    assert "service_area_state" not in ctx
    assert "lob_description" not in ctx


def test_multiselect_drops_sentinel_picks():
    ctx = build_context_from_filters(
        dimension_selections={"service_area_state": ["All", "Select...", ""]},
    )
    assert "service_area_state" not in ctx


def test_principal_time_dimension_maps_to_end_time():
    ctx = build_context_from_filters(
        dimension_selections={},
        time_selections={"incurred_month": 202604},
        principal_time_dimension="incurred_month",
    )
    assert ctx["end_time"] == 202604
    assert ctx["incurred_month"] == 202604


def test_period_payload_expanded_with_rolling_three():
    ctx = build_context_from_filters(
        dimension_selections={"service_area_state": "NY"},
        time_selections={"incurred_month": 202604},
        metric_name="expense_detail.total_paid",
        period_label="Rolling 3",
        principal_time_dimension="incurred_month",
    )
    assert ctx["rolling_window"] == "3_months"
    assert ctx["period"]["current_period"] == {"start_time": 202602, "end_time": 202604}
    assert ctx["period"]["previous_period"] == {"start_time": 202502, "end_time": 202504}
    assert ctx["start_time"] == 202602


def test_datetime_principal_dim_yields_yyyymm_end_time():
    ctx = build_context_from_filters(
        time_selections={"incurred_month": dt.datetime(2026, 6, 1)},
        period_label="Rolling 6",
        principal_time_dimension="incurred_month",
    )
    assert ctx["end_time"] == 202606
    assert ctx["rolling_window"] == "6_months"


def test_empty_inputs_yield_empty_context():
    assert build_context_from_filters() == {}
