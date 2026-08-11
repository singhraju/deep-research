"""Tests for the executive PDF report builder."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from ui import pdf_report


# A representative orchestrator output covering the sections the PDF renders
# (executive summary, one pattern with drill cards + reimbursement, and a
# handful of recommendations across priority buckets). Kept small enough for
# assertions while exercising every branch of :func:`build_pdf_report`.
SAMPLE_OUTPUT: Dict[str, Any] = {
    "question": "Why did inpatient cost per admit spike in Q2?",
    "analysis": {
        "metadata": {
            "correlation_summary": {
                "executive_summary": (
                    "Cost per admit rose **+6.1%** vs prior year, driven by two "
                    "concentrated states.\n\n"
                    "- CO acute-hospital HMO admits with PA drove +$19M.\n"
                    "- CA provider concentration drove +$12.9M."
                ),
                "metric_label": "Cost per Admit",
                "baseline_value": 320_000_000,
                "comparison_value": 352_000_000,
                "drill_path": [
                    {
                        "level": 1,
                        "dimension": "service_area_state",
                        "top_segments": [
                            {"value": "CO", "aligned_delta": 19_100_000},
                            {"value": "CA", "aligned_delta": 12_900_000},
                        ],
                    }
                ],
                "interaction_matrix": {
                    "summary": {"status": "success"},
                    "schema": {
                        "operational_dims": [
                            {"name": "service_area_state", "label": "State"},
                            {"name": "product_description", "label": "Product"},
                        ],
                        "clinical_dims": [
                            {"name": "drg_name", "label": "DRG"},
                        ],
                        "explainer_metrics": [
                            {"name": "admissions", "label": "Admits", "role": "count"},
                        ],
                    },
                    "operational": {
                        "selected_cells": [
                            {
                                "dimension_values": {
                                    "service_area_state": "CO",
                                    "product_description": "HMO",
                                },
                                "delta_value": 19_100_000,
                                "share_of_positive_delta": 0.59,
                                "share_of_net_delta": 0.6,
                                "explainer_metrics": {
                                    "admissions": {"delta": 485, "baseline": 4200},
                                },
                            },
                            {
                                "dimension_values": {
                                    "service_area_state": "CA",
                                    "product_description": "HMO",
                                },
                                "delta_value": 12_900_000,
                                "share_of_positive_delta": 0.41,
                                "share_of_net_delta": 0.4,
                                "explainer_metrics": {
                                    "admissions": {"delta": 120, "baseline": 3900},
                                },
                            },
                        ],
                    },
                    "clinical": {
                        "selected_cells": [
                            {
                                "dimension_values": {"drg_name": "SEPSIS"},
                                "delta_value": 7_400_000,
                                "share_of_net_delta": 0.23,
                                "explainer_metrics": {
                                    "admissions": {"delta": 210, "baseline": 1500},
                                },
                            }
                        ],
                        "offset_cells_preview": [],
                    },
                },
            }
        }
    },
    "research": {
        "pattern_summary": {
            "cards": [
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
                        "rendering_provider_name": "UCHEALTH",
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
                    },
                    "context_dimensions": {"er_admit_indicator": "Y"},
                    "metrics": {
                        "value": {"delta": 12_900_000},
                        "explainer": {"admissions": {"delta": 120}},
                    },
                },
            ]
        },
        "business_patterns": [
            {
                "pattern_rank": 1,
                "pattern_title": "Elective admits concentrated in CO",
                "what_is_impacting": "Colorado HMO admits with prior auth",
                "impact_summary": {
                    "direction": "INCREASE",
                    "estimated_delta": "+$32.0M",
                },
                "why_it_matters": (
                    "The concentration is unusual vs peer states and suggests policy "
                    "leakage we can address with an MPPR edit."
                ),
                "evidence_summary": [
                    "CO drove +$19.1M with 485 extra admits",
                    "CA driver adds another +$12.9M",
                ],
                "recommended_next_step": "Meet with policy team to align on edit rollout.",
                "source_card_ids": ["c1", "c2"],
            }
        ],
        "reimbursement_by_pattern": {
            "1": {
                "elevance_executive_summary": (
                    "Elevance policies allow **modifier 59** for separate reimbursement; "
                    "peer payers require prior auth."
                ),
                "formatted_output": {
                    "summary_table": {
                        "title": "Payer Policy Summary",
                        "subtitle": "Cesarean PA / Stay",
                        "columns": [
                            {"id": "payer_org", "label": "Payer", "type": "text"},
                            {"id": "bundle", "label": "Bundle", "type": "text"},
                            {"id": "appeals", "label": "Appeals", "type": "badge"},
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
                    },
                    "recommended_action": [
                        {
                            "rank": 1,
                            "priority": "High",
                            "description": "Apply MPPR edit to deny separate reimbursement for 59400.",
                        }
                    ],
                    "individual_policies": [
                        {
                            "policy_title": "MPPR Global Maternity Edit",
                            "payer_name": "Elevance Health",
                            "policy_url": "https://example.com/policy/1",
                            "effective_date": "2026-01-01",
                        }
                    ],
                },
            }
        },
        "recommendations": [
            {
                "rank": 1,
                "priority": "High",
                "description": "Deploy MPPR edit for 59400/59409.",
                "owner": "Payment Integrity",
                "eta": "4 weeks",
                "evidence": ["Elevance policy allows modifier 59"],
            },
            {
                "rank": 2,
                "priority": "Medium",
                "description": "Add modifier 59 requirement to vaginal deliveries.",
            },
            {
                "rank": 3,
                "priority": "Low",
                "description": "Publish payer comparison to provider network.",
            },
        ],
    },
}


def test_is_available_matches_reportlab_import():
    # This test just documents current behaviour — reportlab is a project dep,
    # so is_available should be True in CI.
    assert pdf_report.is_available() is True


def test_build_pdf_report_returns_pdf_bytes():
    pdf_bytes = pdf_report.build_pdf_report(SAMPLE_OUTPUT)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1024, "PDF should carry real content, not just headers"
    assert pdf_bytes.startswith(b"%PDF-"), "Output should be a valid PDF stream"


def test_build_pdf_report_embeds_report_content(monkeypatch):
    # Skip the plotly→PNG step so the assertions look at the fallback path where
    # drill-down data is emitted as a text table (searchable in the PDF stream).
    monkeypatch.setattr(pdf_report, "_figure_to_png_bytes", lambda fig: None)
    pdf_bytes = pdf_report.build_pdf_report(SAMPLE_OUTPUT)

    # ReportLab wraps content in compressed streams by default; opening the PDF
    # in text mode misses the words. Use pypdf if available, otherwise fall back
    # to a permissive substring scan of the decoded stream.
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover — pypdf not required in the base env
        text = pdf_bytes.decode("latin-1", errors="ignore")
    else:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "Deep Research Report" in text
    assert "Elective admits concentrated in CO" in text
    assert "Final Recommendations" in text
    # The SQL Queries and Technical Details tabs are intentionally skipped.
    assert "SQL Queries" not in text
    assert "Technical Details" not in text


def test_build_pdf_report_includes_correlation_visuals(monkeypatch):
    """PDF must carry the KPI strip, waterfall (or text fallback), and matrix tables."""

    # Force the text fallback so we can search the PDF for the drill segment
    # labels — kaleido rendering would embed a PNG that pypdf can't read text
    # from.
    monkeypatch.setattr(pdf_report, "_figure_to_png_bytes", lambda fig: None)
    pdf_bytes = pdf_report.build_pdf_report(SAMPLE_OUTPUT)

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover
        pytest.skip("pypdf not installed; skipping text-level assertions")
    import io as _io

    reader = PdfReader(_io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # KPI strip labels + values.
    assert "Correlation Overview" in text
    assert "Total Impact" in text
    assert "Top Driver" in text
    assert "Baseline" in text and "Comparison" in text
    assert "Detected Interactions" in text

    # Waterfall content (fallback table lists baseline/comparison + drill segments).
    assert "Cost Cascade" in text
    assert "CO" in text and "CA" in text

    # Interaction matrix — operational + clinical sections.
    assert "Operational Concentration" in text
    assert "Clinical Detail" in text
    assert "SEPSIS" in text
    # drill-path values are labeled with their friendly dimension name
    assert "DRG" in text


def test_find_correlation_summary_falls_back_to_alternate_locations():
    """The lookup helper must find the payload whether it's nested under
    analysis.metadata, at the top level, or under research."""

    payload = {"baseline_value": 1000, "comparison_value": 1100}

    # Primary path wins even when alternates are also populated.
    primary = {
        "analysis": {"metadata": {"correlation_summary": payload}},
        "correlation_summary": {"marker": "top_level"},
        "research": {"correlation_summary": {"marker": "research"}},
    }
    assert pdf_report._find_correlation_summary(primary) is payload

    # Top-level fallback when analysis.metadata has no correlation_summary.
    top_level = {"analysis": {"metadata": {}}, "correlation_summary": payload}
    assert pdf_report._find_correlation_summary(top_level) is payload

    # Research fallback when both primary + top-level are empty.
    research = {"research": {"correlation_summary": payload}}
    assert pdf_report._find_correlation_summary(research) is payload

    # Nothing anywhere → empty mapping (falsy) so the section short-circuits.
    assert pdf_report._find_correlation_summary({"analysis": {}}) == {}


def test_build_pdf_report_without_patterns_or_recommendations():
    output = {"question": "Empty run", "research": {}}
    pdf_bytes = pdf_report.build_pdf_report(output)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_build_pdf_report_type_error_on_non_mapping():
    with pytest.raises(TypeError):
        pdf_report.build_pdf_report(["not a dict"])  # type: ignore[arg-type]


def test_suggested_filename_slugifies_question_and_ends_in_pdf():
    name = pdf_report.suggested_filename({"question": "Why did *cost/admit* spike?"})
    assert name.endswith(".pdf")
    # Non-alphanumerics collapse to underscores; timestamp is appended.
    assert "why_did" in name
    assert "spike" in name


def test_suggested_filename_falls_back_when_question_missing():
    name = pdf_report.suggested_filename({})
    assert name.endswith(".pdf")
    assert name.startswith("deep_research_")


def test_normalize_priority_variants():
    assert pdf_report._normalize_priority("high") == "HIGH"
    assert pdf_report._normalize_priority("H") == "HIGH"
    assert pdf_report._normalize_priority("Med") == "MEDIUM"
    assert pdf_report._normalize_priority(None) == "MEDIUM"
    assert pdf_report._normalize_priority("nonsense") == "NONSENSE"


def test_markdown_to_para_html_handles_bold_italic_and_links():
    result = pdf_report._markdown_to_para_html(
        "See **bold**, *italic*, and [link](https://example.com)."
    )
    assert "<b>bold</b>" in result
    assert "<i>italic</i>" in result
    assert 'href="https://example.com"' in result
    assert "&lt;" not in result  # nothing to escape here
