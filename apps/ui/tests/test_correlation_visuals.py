"""Tests for correlation visuals (waterfall + interaction matrix renderers)."""

from __future__ import annotations

import pytest

from ui.correlation_visuals import (
    _build_dim_pairs,
    _resolve_dim_label,
    build_correlation_visuals_html,
    build_matrix_section_html,
    format_compact_money,
    format_pct_points,
    format_percent,
    format_signed_integer,
    format_signed_money,
)


# Sample mirrors the screenshot the analytics team distributes (r6 commercial IP).
SAMPLE_CORRELATION_SUMMARY = {
    "metric_label": "expense_detail.total_paid",
    "baseline_value": 6_600_000,
    "comparison_value": 9_900_000,
    "drill_path": [
        {
            "level": 1,
            "dimension": "er_admit_indicator",
            "top_segments": [
                {"value": "N", "aligned_delta": 3_000_000},
                {"value": "Y", "aligned_delta": 600_000},
            ],
        },
        {
            "level": 2,
            "dimension": "pa_required_code",
            "top_segments": [
                {"value": "N", "aligned_delta": -100_000},
            ],
        },
    ],
    "interaction_matrix": {
        "summary": {"status": "success"},
        "operational": {
            "selected_cells": [
                {
                    "dimension_values": {
                        "service_area_state": "CA",
                        "product_description": "PPO",
                        "facility_type": "PHYSICAL REHAB",
                        "pa_required_code": "Y",
                    },
                    "delta_value": 710_200,
                    "share_of_positive_delta": 0.106,
                    "share_of_net_delta": 0.217,
                    "explainer_metrics": {
                        "expense_detail.claim_count": {"delta": 15, "baseline": 100},
                        "expense_detail.total_admissions": {"delta": 14, "baseline": 80},
                        "expense_detail.avg_paid_per_admit": {"delta": 17_400},
                        "expense_detail.paid_ratio": {"delta": 0.016},
                    },
                },
                {
                    "dimension_values": {
                        "service_area_state": "CA",
                        "product_description": "EPO",
                        "facility_type": "PHYSICAL REHAB",
                        "pa_required_code": "Y",
                    },
                    "delta_value": 623_900,
                    "share_of_positive_delta": 0.093,
                    "share_of_net_delta": 0.191,
                    "explainer_metrics": {
                        "expense_detail.claim_count": {"delta": 18, "baseline": 120},
                        "expense_detail.total_admissions": {"delta": 13, "baseline": 70},
                        "expense_detail.avg_paid_per_admit": {"delta": 20_600},
                        "expense_detail.paid_ratio": {"delta": -0.012},
                    },
                },
            ]
        },
        "clinical": {
            "selected_cells": [
                {
                    "dimension_values": {"drg_name": "ENC AFTERCARE FLW SURGERY NEOPLASM"},
                    "delta_value": 434_900,
                    "share_of_net_delta": 0.133,
                    "explainer_metrics": {
                        "expense_detail.claim_count": {"delta": 12, "baseline": 50},
                        "expense_detail.total_admissions": {"delta": 10, "baseline": 40},
                        "expense_detail.avg_paid_per_admit": {"delta": 29_200},
                        "expense_detail.paid_ratio": {"delta": 0.02},
                    },
                },
            ],
            "offset_cells_preview": [
                {
                    "dimension_values": {"drg_name": "ENC SURG AFTERCARE FOLLOW SURG NS"},
                    "delta_value": -44_200,
                    "share_of_net_delta": -0.014,
                    "explainer_metrics": {
                        "expense_detail.claim_count": {"delta": -2, "baseline": 30},
                        "expense_detail.total_admissions": {"delta": -2, "baseline": 25},
                        "expense_detail.avg_paid_per_admit": {"delta": -22_100},
                        "expense_detail.paid_ratio": {"delta": -0.015},
                    },
                },
            ],
        },
    },
}


# ---------- formatters ----------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1_700_000, "$1.7M"),
        (-1_700_000, "$1.7M"),  # compact_money strips sign
        (44_200, "$44K"),
        (200, "$200"),
        (0, "$0"),
        (None, "—"),
        (float("nan"), "—"),
    ],
)
def test_format_compact_money(value, expected):
    assert format_compact_money(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1_700_000, "+$1.7M"),
        (-1_500_000, "-$1.5M"),
        (710_200, "+$710.2K"),
        (None, "—"),
    ],
)
def test_format_signed_money(value, expected):
    assert format_signed_money(value) == expected


def test_format_percent_and_pct_points():
    assert format_percent(0.261) == "26.1%"
    assert format_pct_points(0.016) == "+1.6 pts"
    assert format_pct_points(-0.012) == "-1.2 pts"
    assert format_pct_points(None) == "—"


def test_format_signed_integer():
    assert format_signed_integer(15) == "+15"
    assert format_signed_integer(-1) == "-1"
    assert format_signed_integer(1234) == "+1,234"
    assert format_signed_integer(None) == "—"


# ---------- top-level renderer ----------


def test_build_correlation_visuals_html_full_payload():
    html = build_correlation_visuals_html(SAMPLE_CORRELATION_SUMMARY)
    assert html is not None

    # Waterfall section
    assert "Waterfall Chart: expense_detail.total_paid" in html
    assert "Baseline" in html and "Comparison" in html
    assert "Level 1: er_admit_indicator" in html
    assert "Level 2: pa_required_code" in html
    assert "<svg" in html

    # KPI strip
    assert "Selected Operational Δ Paid" in html
    assert "Share of +Δ" in html
    assert "Share Net Δ" in html  # table column header
    assert "Selected Clinical Δ Paid" in html
    # KPI values from sample (710,200 + 623,900 = 1,334,100 → $1.3M signed)
    assert "+$1.3M" in html  # operational sum
    assert "+$434.9K" in html  # clinical sum (only positive cells)

    # Operational table — stacked, one labeled dimension per line
    assert "Operational Concentration" in html
    # friendly dimension labels are surfaced next to each value
    assert ">State</span>" in html
    assert ">Product</span>" in html
    assert ">Facility Type</span>" in html
    assert ">Prior Auth</span>" in html
    # values still render, and the plain-text tooltip carries the labeled path
    assert "State: CA · Product: PPO · Facility Type: Physical Rehab · Prior Auth: PA Required" in html
    assert "State: CA · Product: EPO · Facility Type: Physical Rehab · Prior Auth: PA Required" in html
    assert "+$710.2K" in html
    assert "+$623.9K" in html
    assert "Volume Plus Paid Ratio" in html  # signal label for first cell

    # Heatmap (selected cells matrix)
    assert ">EPO<" in html
    assert ">PPO<" in html
    assert ">CA<" in html

    # Clinical detail table — DRG value labeled with its dimension name
    assert "Clinical Detail Within Selected Operational Cells" in html
    assert ">DRG</span>" in html
    assert "DRG: ENC AFTERCARE FLW SURGERY NEOPLASM" in html  # tooltip
    assert "ENC AFTERCARE FLW SURGERY NEOPLASM" in html
    assert "ENC SURG AFTERCARE FOLLOW SURG NS" in html
    assert "Increase" in html
    assert "Offset" in html


def test_build_correlation_visuals_html_no_drill_returns_none():
    assert build_correlation_visuals_html({}) is None


def test_build_correlation_visuals_html_waterfall_only():
    payload = {
        "metric_label": "metric.x",
        "baseline_value": 100.0,
        "comparison_value": 150.0,
        "drill_path": [
            {
                "level": 1,
                "dimension": "dim",
                "top_segments": [{"value": "A", "aligned_delta": 50.0}],
            }
        ],
    }
    html = build_correlation_visuals_html(payload)
    assert html is not None
    assert "Waterfall Chart" in html
    assert "Detected Interactions" not in html  # no matrix


def test_build_correlation_visuals_html_skips_matrix_when_status_not_success():
    payload = dict(SAMPLE_CORRELATION_SUMMARY)
    payload["interaction_matrix"] = {
        **payload["interaction_matrix"],
        "summary": {"status": "skipped"},
    }
    html = build_correlation_visuals_html(payload)
    assert html is not None
    assert "Waterfall Chart" in html
    assert "Detected Interactions" not in html


def test_signal_derivation_via_html_unit_cost():
    payload = {
        "metric_label": "m",
        "baseline_value": 100.0,
        "comparison_value": 200.0,
        "drill_path": [
            {"level": 1, "dimension": "d", "top_segments": [{"value": "v", "aligned_delta": 100.0}]}
        ],
        "interaction_matrix": {
            "summary": {"status": "success"},
            "operational": {
                "selected_cells": [
                    {
                        "dimension_values": {"service_area_state": "CA", "product_description": "PPO"},
                        "delta_value": 1.0,
                        "explainer_metrics": {
                            # Small admit move, big avg paid move, small paid ratio → unit_cost
                            "expense_detail.total_admissions": {"delta": 1, "baseline": 1000},
                            "expense_detail.avg_paid_per_admit": {"delta": 5000},
                            "expense_detail.paid_ratio": {"delta": 0.001},
                        },
                    }
                ]
            },
            "clinical": {"selected_cells": [], "offset_cells_preview": []},
        },
    }
    html = build_correlation_visuals_html(payload)
    assert html is not None
    assert "Unit Cost" in html


# ---------- labeled drill-path (business-friendly dimension labels) ----------


def test_resolve_dim_label_override_and_fallback():
    # curated friendly override wins
    assert _resolve_dim_label("place_of_service_code") == "Place of Service"
    assert _resolve_dim_label("claim_line_revenue_code") == "Revenue Code"
    assert _resolve_dim_label("provider_speciality_code_description") == "Provider Specialty"
    # unknown key falls back to a humanized key
    assert _resolve_dim_label("some_brand_new_field") == "Some Brand New Field"


def test_build_dim_pairs_labels_order_and_null_drop():
    cell = {
        "dimension_values": {
            "provider_speciality_code_description": "Ground Ambulance",
            "type_of_bill_code": "<NULL>",
            "claim_line_revenue_code": "UNK",
            "place_of_service_code": "41",
            "line_diagnosis_code_1": "F99",
            "ndc_label_number": "<NULL>",
        }
    }
    dim_order = [
        "provider_speciality_code_description",
        "type_of_bill_code",
        "claim_line_revenue_code",
        "place_of_service_code",
        "line_diagnosis_code_1",
    ]
    pairs = _build_dim_pairs(cell, dim_order)
    # <NULL> dims drop the whole pair — no orphan "Bill Type"/"Drug (NDC)" label
    assert pairs == [
        ("Provider Specialty", "Ground Ambulance"),
        ("Revenue Code", "UNK"),
        ("Place of Service", "41"),
        ("Diagnosis", "F99"),
    ]


def _main_view_matrix_payload():
    """Payload shaped like the real pi_wgs_mad interaction matrix (with schema)."""
    return {
        "metric_label": "wgs_mad.total_paid",
        "baseline_value": 6_600_000,
        "comparison_value": 9_900_000,
        "drill_path": [
            {"level": 1, "dimension": "funding_type",
             "top_segments": [{"value": "ASO", "aligned_delta": 3_000_000}]}
        ],
        "interaction_matrix": {
            "summary": {"status": "success"},
            "schema": {
                "operational_dimensions": [
                    {"name": "funding_type", "label": "funding type: If user looks for FINR ..."},
                    {"name": "in_network_code", "label": "IN NETWORK CODE:"},
                    {"name": "src_provider_zip_code", "label": "SOURCE PROVIDER ZIP CODE ..."},
                    {"name": "provider_state_code", "label": "PROVIDER STATE CODE ..."},
                ],
                "clinical_dimensions": [
                    {"name": "provider_speciality_code_description", "label": "Provider Specialty Code Description"},
                    {"name": "type_of_bill_code", "label": "Type of Bill Code ..."},
                    {"name": "claim_line_revenue_code", "label": "Revenue code: should always have 4 digits"},
                    {"name": "place_of_service_code", "label": "Place of Service Code"},
                    {"name": "line_diagnosis_code_1", "label": "Primary line level Dx code ..."},
                ],
            },
            "operational": {
                "selected_cells": [
                    {
                        "dimension_values": {
                            "funding_type": "ASO",
                            "in_network_code": "OUT",
                            "src_provider_zip_code": "93637",
                            "provider_state_code": "CA",
                        },
                        "delta_value": 710_200,
                        "share_of_positive_delta": 0.1,
                        "share_of_net_delta": 0.2,
                        "explainer_metrics": {"wgs_mad.claim_count": {"delta": 15, "baseline": 100}},
                    }
                ]
            },
            "clinical": {
                "selected_cells": [
                    {
                        "dimension_values": {
                            "provider_speciality_code_description": "Ground Ambulance",
                            "type_of_bill_code": "<NULL>",
                            "claim_line_revenue_code": "UNK",
                            "place_of_service_code": "41",
                            "line_diagnosis_code_1": "F99",
                            "ndc_label_number": "<NULL>",
                        },
                        "delta_value": 32_200,
                        "share_of_net_delta": 0.05,
                        "explainer_metrics": {"wgs_mad.claim_count": {"delta": 5, "baseline": 20}},
                    }
                ],
                "offset_cells_preview": [],
            },
        },
    }


def test_matrix_section_labels_clinical_drill_path():
    html = build_matrix_section_html(_main_view_matrix_payload())
    assert html is not None
    # friendly labels rendered next to each value (verbose schema labels NOT used)
    for label in ("Provider Specialty", "Revenue Code", "Place of Service", "Diagnosis"):
        assert f">{label}</span>" in html
    for value in ("Ground Ambulance", "UNK", "41", "F99"):
        assert f">{value}</span>" in html
    # <NULL> dims produce no label
    assert ">Bill Type</span>" not in html
    assert ">Drug (NDC)</span>" not in html
    # tooltip carries the full labeled plain-text path
    assert "Provider Specialty: Ground Ambulance · Revenue Code: UNK · Place of Service: 41 · Diagnosis: F99" in html


def test_matrix_section_labels_operational_drill_path():
    html = build_matrix_section_html(_main_view_matrix_payload())
    assert html is not None
    for label, value in [
        ("Funding Type", "ASO"),
        ("Network Status", "OUT"),
        ("Provider ZIP", "93637"),
        ("Provider State", "CA"),
    ]:
        assert f">{label}</span>" in html
        assert f">{value}</span>" in html
    # the verbose schema label must not leak into the UI
    assert "If user looks for FINR" not in html
