"""Tests for the notebook-based PatternAgent business synthesis flow."""

from __future__ import annotations

from typing import Any, Dict
from deep_research_agents.pattern_agent import DEFAULT_SEMANTIC_CONFIG_PATH
from deep_research_agents.pattern_agent import BusinessPatternSchema
from deep_research_agents.pattern_agent import PatternAgent
from deep_research_agents.pattern_agent import PatternSynthesisSchema
from deep_research_agents.pattern_agent import build_pattern_cards_and_groups


def _build_operational_cell(product: str, facility: str, delta_value: float, admissions_delta: float, unit_delta: float) -> Dict[str, Any]:
    return {
        "cell_id": f"{product}-{facility}-{int(delta_value)}",
        "dimension_values": {
            "product_description": product,
            "facility_type": facility,
        },
        "baseline_value": max(delta_value, 1.0),
        "comparison_value": max(delta_value, 1.0) + delta_value,
        "delta_value": delta_value,
        "share_of_positive_delta": 0.45,
        "share_of_net_delta": 0.40,
        "raw_row_count_baseline": 8,
        "raw_row_count_comparison": 12,
        "explainer_metrics": {
            "expense_detail.total_admissions": {
                "baseline": 8,
                "comparison": 8 + admissions_delta,
                "delta": admissions_delta,
            },
            "expense_detail.avg_paid_per_admit": {
                "baseline": 100000,
                "comparison": 100000 + unit_delta,
                "delta": unit_delta,
            },
        },
    }


def _build_validation_level() -> Dict[str, Any]:
    return {
        "level": 1,
        "dimension": "pa_required_code",
        "top_segments": [
            {
                "value": "Y",
                "delta_value": 900000,
                "baseline_value": 200000,
                "comparison_value": 1100000,
                "raw_row_count_baseline": 5,
                "raw_row_count_comparison": 11,
            }
        ],
        "bottom_segments": [
            {
                "value": "-3",
                "delta_value": -500000,
                "baseline_value": 800000,
                "comparison_value": 300000,
                "raw_row_count_baseline": 6,
                "raw_row_count_comparison": 3,
            }
        ],
    }


def _sample_correlation_results(*, small_deltas: bool = False) -> Dict[str, Any]:
    delta_one = 800 if small_deltas else 1_800_000
    delta_two = 700 if small_deltas else 1_250_000
    return {
        "states": {
            "CO": {
                "output": {
                    "root_metric": "expense_detail.total_paid",
                    "baseline_value": 2_000_000,
                    "comparison_value": 3_800_000,
                    "delta_value": delta_one,
                    "drill_path": [_build_validation_level()],
                    "interaction_matrix": {
                        "operational": {
                            "selected_cells": [
                                _build_operational_cell("HMO", "ACUTE HOSPITAL", delta_one, 4, 25000),
                            ]
                        },
                        "clinical": {
                            "selected_cells": [
                                {
                                    "cell_id": "clinical-co",
                                    "dimension_values": {"drg_name": "IP Med/Surg"},
                                    "baseline_value": 300000,
                                    "comparison_value": 520000,
                                    "delta_value": 220000,
                                    "share_of_positive_delta": 0.10,
                                    "raw_row_count_baseline": 3,
                                    "raw_row_count_comparison": 4,
                                    "explainer_metrics": {
                                        "expense_detail.total_admissions": {
                                            "baseline": 3,
                                            "comparison": 4,
                                            "delta": 1,
                                        }
                                    },
                                }
                            ]
                        },
                    },
                }
            },
            "ME": {
                "output": {
                    "root_metric": "expense_detail.total_paid",
                    "baseline_value": 1_500_000,
                    "comparison_value": 2_750_000,
                    "delta_value": delta_two,
                    "drill_path": [_build_validation_level()],
                    "interaction_matrix": {
                        "operational": {
                            "selected_cells": [
                                _build_operational_cell("HMO", "ACUTE HOSPITAL", delta_two, 3, 18000),
                            ]
                        },
                        "clinical": {
                            "selected_cells": [
                                {
                                    "cell_id": "clinical-me",
                                    "dimension_values": {"drg_name": "Residential treatment center"},
                                    "baseline_value": 250000,
                                    "comparison_value": 430000,
                                    "delta_value": 180000,
                                    "share_of_positive_delta": 0.08,
                                    "raw_row_count_baseline": 2,
                                    "raw_row_count_comparison": 3,
                                    "explainer_metrics": {
                                        "expense_detail.total_admissions": {
                                            "baseline": 2,
                                            "comparison": 3,
                                            "delta": 1,
                                        }
                                    },
                                }
                            ]
                        },
                    },
                }
            },
        },
        "providers": {},
        "drgs": {},
    }


def _base_context(*, small_deltas: bool = False) -> Dict[str, Any]:
    return {
        "anomaly_context": {},
        "deep_dive_report": {},
        "filters": [
            {"field": "snap_month", "operator": "=", "value": 202604, "source": "base_context"},
            {"field": "lob_description", "operator": "=", "value": "Commercial", "source": "base_context"},
        ],
        "correlation_results": _sample_correlation_results(small_deltas=small_deltas),
    }


def _build_analytical_output() -> Dict[str, Any]:
    return build_pattern_cards_and_groups(
        correlation_results=_base_context()["correlation_results"],
        semantic_config_path=str(DEFAULT_SEMANTIC_CONFIG_PATH),
    )


def _format_expected_delta(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{sign}${absolute / 1_000_000:.1f}M"
    if absolute >= 1_000:
        rounded = int(round(absolute / 1_000.0) * 1_000)
        return f"{sign}${rounded:,.0f}"
    return f"{sign}${absolute:,.0f}"


def _select_broad_group_and_focus_card() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    analytical_output = _build_analytical_output()
    cards_by_id = {str(card["card_id"]): card for card in analytical_output["cards"]}
    broad_group = next(group for group in analytical_output["groups"] if int(group["impact"]["card_count"]) > 1)
    group_cards = [cards_by_id[str(card_id)] for card_id in broad_group["card_ids"]]
    focus_card = max(group_cards, key=lambda card: abs(float(card["metrics"]["value"]["delta"] or 0.0)))
    return analytical_output, broad_group, focus_card


class _StructuredLLM:
    def __init__(self, response: PatternSynthesisSchema, include_raw: bool) -> None:
        self._response = response
        self._include_raw = include_raw

    def invoke(self, _messages: list[dict[str, str]]) -> object:
        if self._include_raw:
            return {
                "parsed": self._response,
                "raw": {"usage": {"input_tokens": 12, "output_tokens": 7}},
            }
        return self._response


class _CustomLLM:
    def __init__(self, response: PatternSynthesisSchema) -> None:
        self._response = response

    def with_structured_output(self, _schema: object, **kwargs: object) -> _StructuredLLM:
        return _StructuredLLM(self._response, include_raw=bool(kwargs.get("include_raw")))


def test_pattern_agent_stub_output_builds_business_patterns_from_groups() -> None:
    agent = PatternAgent(test_mode=True)
    result = agent(conversation_id="abc123", context=_base_context())

    assert result["agent"] == "pattern_agent"
    assert result["conversation_id"] == "abc123"
    assert result["visual_component"] == {}
    assert "patterns" not in result["output"]

    groups = result["output"]["groups"]
    cards = result["output"]["cards"]
    business_patterns = result["output"]["business_patterns"]

    assert groups
    assert cards
    assert business_patterns
    assert result["output"]["quality_checks"]["patterns_returned"] == len(business_patterns)
    assert business_patterns[0]["source_group_ids"]
    assert business_patterns[0]["source_card_ids"]
    assert any(pattern["pattern_type"] == "coding_mapping_validation" for pattern in business_patterns)
    assert business_patterns[0]["top_pattern"] == groups[0]["pattern_name"]


def test_pattern_agent_returns_empty_business_patterns_when_no_groups_pass_threshold() -> None:
    agent = PatternAgent(test_mode=True)
    context = _base_context(small_deltas=True)
    context["min_abs_delta"] = 2_000_000

    result = agent(conversation_id="no_groups", context=context)

    assert result["status"] == "success"
    assert result["output"]["groups"] == []
    assert result["output"]["business_patterns"] == []
    assert result["output"]["quality_checks"]["patterns_returned"] == 0
    assert any("no analytical groups" in note.lower() for note in result["output"]["quality_checks"]["notes"])


def test_pattern_agent_validation_fails_for_unknown_trace_ids_from_structured_output() -> None:
    response = PatternSynthesisSchema(
        business_patterns=[
            BusinessPatternSchema(
                pattern_rank=1,
                top_pattern="Invalid trace pattern",
                pattern_type="utilization_growth",
                what_is_impacting="Commercial HMO / Acute Hospital",
                priority_entities={"states": ["CO"], "providers": [], "products": ["HMO"], "facility_types": ["ACUTE HOSPITAL"], "clinical_categories": []},
                key_driver_codes=["HMO"],
                impact_summary={
                    "primary_metric": "total_paid",
                    "direction": "increase",
                    "estimated_delta": "+$1.8M",
                    "volume_signal": "admissions up",
                    "unit_cost_signal": "unknown",
                },
                pattern_details="Synthetic invalid pattern for validation.",
                why_it_matters="Synthetic invalid pattern for validation.",
                recommended_next_step="Synthetic invalid pattern for validation.",
                validation_needed=False,
                validation_reason=None,
                downstream_routes=["um_operations_agent"],
                evidence_summary=["Synthetic invalid evidence."],
                source_group_ids=["missing-group"],
                source_card_ids=["missing-card"],
            )
        ]
    )
    agent = PatternAgent(test_mode=True)
    agent.llm = _CustomLLM(response)

    result = agent(conversation_id="invalid_trace", context=_base_context())

    assert result["status"] == "failed"
    assert result["tokens"]["input"] == 12
    assert result["tokens"]["output"] == 7
    assert any("unknown source_group_ids" in error.lower() for error in result["validation"]["errors"])
    assert any("unknown source_card_ids" in error.lower() for error in result["validation"]["errors"])


def test_pattern_agent_uses_selected_cards_for_pattern_attribution() -> None:
    _, broad_group, focus_card = _select_broad_group_and_focus_card()
    expected_delta = _format_expected_delta(float(focus_card["metrics"]["value"]["delta"]))

    response = PatternSynthesisSchema(
        business_patterns=[
            BusinessPatternSchema(
                pattern_rank=1,
                top_pattern="Focused utilization story",
                pattern_type="utilization_growth",
                what_is_impacting="Commercial HMO / Acute Hospital",
                priority_entities={
                    "states": ["CO", "ME"],
                    "providers": ["UNRELATED HEALTH SYSTEM"],
                    "products": ["HMO"],
                    "facility_types": ["ACUTE HOSPITAL"],
                    "clinical_categories": ["UNRELATED CATEGORY"],
                },
                key_driver_codes=["HMO", "ACUTE HOSPITAL"],
                impact_summary={
                    "primary_metric": "total_paid",
                    "direction": "increase",
                    "estimated_delta": expected_delta,
                    "volume_signal": "admissions up",
                    "unit_cost_signal": "unknown",
                },
                pattern_details="Synthetic pattern for testing LLM response preservation.",
                why_it_matters="Synthetic pattern for testing LLM response preservation.",
                recommended_next_step="Synthetic pattern for testing LLM response preservation.",
                validation_needed=False,
                validation_reason=None,
                downstream_routes=["um_operations_agent"],
                evidence_summary=["Synthetic evidence summary."],
                source_group_ids=[str(broad_group["group_id"])],
                source_card_ids=[str(focus_card["card_id"])],
            )
        ]
    )
    agent = PatternAgent(test_mode=True)
    agent.llm = _CustomLLM(response)

    result = agent(conversation_id="coverage_regression", context=_base_context())

    business_pattern = result["output"]["business_patterns"][0]
    quality_checks = result["output"]["quality_checks"]

    # Verify the agent preserves the LLM's structured output
    assert result["status"] == "success"
    assert business_pattern["source_group_ids"] == [str(broad_group["group_id"])]
    assert business_pattern["source_card_ids"] == [str(focus_card["card_id"])]
    # The agent preserves the LLM's priority_entities as-is
    assert business_pattern["priority_entities"]["states"] == ["CO", "ME"]
    assert business_pattern["priority_entities"]["products"] == ["HMO"]
    assert business_pattern["priority_entities"]["facility_types"] == ["ACUTE HOSPITAL"]
    assert business_pattern["priority_entities"]["providers"] == ["UNRELATED HEALTH SYSTEM"]
    assert business_pattern["impact_summary"]["estimated_delta"] == expected_delta
    assert quality_checks["groups_consumed"] == 1
    assert quality_checks["ungrouped_group_ids"]
