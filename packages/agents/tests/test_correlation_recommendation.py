from __future__ import annotations

from typing import Any, Dict, List

from deep_research_agents.correlation_recommendation import (
    CorrelationRecommendationsSchema,
    create_correlation_recommendations,
    normalize_legacy_recommendations,
)


class StubRecommendationLLM:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def with_structured_output(self, schema, **kwargs):
        class _StructuredLLM:
            def __init__(self, payload: Dict[str, Any], include_raw: bool) -> None:
                self._payload = payload
                self._include_raw = include_raw
                self._schema = schema

            def invoke(self, messages: List[Dict[str, str]]):
                _ = messages
                parsed = self._schema(**self._payload)
                if self._include_raw:
                    return {
                        "parsed": parsed,
                        "raw": {
                            "response_metadata": {
                                "token_usage": {
                                    "prompt_tokens": 17,
                                    "completion_tokens": 9,
                                }
                            }
                        },
                    }
                return parsed

        return _StructuredLLM(self.payload, include_raw=bool(kwargs.get("include_raw")))


class FailingRecommendationLLM:
    def with_structured_output(self, schema, **kwargs):
        class _StructuredLLM:
            def invoke(self, messages: List[Dict[str, str]]):
                _ = schema, kwargs, messages
                raise RuntimeError("synthetic llm failure")

        return _StructuredLLM()


def _sample_root_trend() -> Dict[str, Any]:
    return {
        "metric_name": "expense_detail.total_paid",
        "baseline_value": 15000.0,
        "comparison_value": 55000.0,
        "delta_value": 40000.0,
        "delta_pct": 2.6667,
    }


def _sample_drill_path() -> List[Dict[str, Any]]:
    return [
        {
            "level": 1,
            "dimension": "hcc_medium",
            "dimension_label": "HCC Medium",
            "top_segments": [
                {
                    "value": "IP OB Dlvry/Well NB",
                    "delta_value": 40000.0,
                    "contribution_pct_total": 1.0,
                    "aligned_contribution_pct_of_aligned_delta": 1.0,
                }
            ],
            "bottom_segments": [],
        },
        {
            "level": 2,
            "dimension": "pa_required_code",
            "dimension_label": "PA Required",
            "top_segments": [
                {
                    "value": "N",
                    "delta_value": 29500.0,
                    "contribution_pct_total": 0.74,
                    "aligned_contribution_pct_of_aligned_delta": 0.74,
                }
            ],
            "bottom_segments": [
                {
                    "value": "Y",
                    "delta_value": -4500.0,
                    "opposing_share": 1.0,
                }
            ],
        },
    ]


def _sample_explainer_metrics() -> Dict[str, Any]:
    return {
        "root": [
            {
                "period_bucket": "baseline",
                "expense_detail_claim_count": 100.0,
                "expense_detail_total_admissions": 100.0,
                "expense_detail_avg_paid_per_admit": 150.0,
                "expense_detail_avg_allowed_per_admit": 190.0,
                "expense_detail_paid_ratio": 0.79,
            },
            {
                "period_bucket": "comparison",
                "expense_detail_claim_count": 101.0,
                "expense_detail_total_admissions": 101.0,
                "expense_detail_avg_paid_per_admit": 545.0,
                "expense_detail_avg_allowed_per_admit": 610.0,
                "expense_detail_paid_ratio": 0.89,
            },
        ]
    }


def _sample_interaction_matrix() -> Dict[str, Any]:
    return {
        "operational": {
            "selected_cells": [
                {
                    "cell_id": "op_001",
                    "delta_value": 25000.0,
                    "share_of_positive_delta": 0.62,
                    "dimension_values": {
                        "pa_required_code": "N",
                        "product_description": "PPO",
                        "facility_type": "ACUTE HOSPITAL",
                        "service_area_state": "CA",
                    },
                }
            ]
        },
        "clinical": {
            "selected_cells": [
                {
                    "cell_id": "cl_001",
                    "delta_value": 12000.0,
                    "dimension_values": {
                        "drg_name": "DRG A",
                        "primary_diagnosis_name": "POST-TERM PREGNANCY",
                    },
                }
            ],
            "offset_cells_preview": [
                {
                    "cell_id": "cl_off_001",
                    "delta_value": -4000.0,
                    "dimension_values": {
                        "primary_diagnosis_name": "DX B",
                    },
                }
            ],
        },
    }


def test_normalize_legacy_recommendations_parses_stringified_items() -> None:
    payload = {
        "items": [
            {
                "text": '{"priority":1,"action":"Review CA PPO acute hospital unit cost increases.","rationale":"$8.8M increase with admissions nearly flat and paid per admission up about $1.0K.","cell_ids":["op_001"],"review_area":"unit_cost"}',
                "cell_ids": ["op_001"],
            }
        ],
        "source": "llm",
    }

    normalized = normalize_legacy_recommendations(payload)

    assert normalized["recommended_action"][0]["description"] == "Review CA PPO acute hospital unit cost increases."
    assert normalized["recommended_action"][0]["rank"] == 1
    assert normalized["recommended_action"][0]["priority"] == "LOW"
    assert normalized["recommended_action"][0]["evidence"][0] == "Interaction cell reference: op_001"
    assert normalized["recommended_action"][0]["story_alignment"][0] == "Why: $8.8M increase with admissions nearly flat and paid per admission up about $1.0K."


def test_create_correlation_recommendations_uses_structured_llm_output() -> None:
    llm = StubRecommendationLLM(
        {
            "recommended_action": [
                {
                    "rank": 1,
                    "priority": "HIGH",
                    "category": "Operational",
                    "description": "Prioritize prior-auth review in the concentrated no-auth growth pocket.",
                    "evidence": [
                        "Drill path: pa_required_code=N contributed $29.5K.",
                        "Operational interaction cell op_001 changed $25.0K; dimensions: pa_required_code=N, product_description=PPO, facility_type=ACUTE HOSPITAL, service_area_state=CA.",
                    ],
                    "story_alignment": [
                        "Why: Root paid increased by $40.0K and interaction cell op_001 concentrated the increase while a smaller clinical offset remained.",
                        "research_consideration: Analyst to review variance driven by services requiring prior auth vs no prior auth.",
                        "cost_of_care_suggestion: Confirm prior auth was obtained when required.",
                    ],
                    "peer_benchmarking": [
                        "Peer benchmark on pa_required_code: top segment N increased $29.5K while comparator Y moved $-4.5K.",
                    ],
                    "citation": [],
                }
            ],
            "summary": {
                "overall_pattern": "Concentrated",
                "primary_next_action": "Prioritize prior-auth review in the concentrated no-auth growth pocket.",
                "do_not_overgeneralize": "Do not generalize the finding beyond the concentrated no-auth segment.",
            },
        }
    )

    payload = create_correlation_recommendations(
        metric_name="expense_detail.total_paid",
        root_trend=_sample_root_trend(),
        drill_path=_sample_drill_path(),
        explainer_metrics=_sample_explainer_metrics(),
        interaction_matrix=_sample_interaction_matrix(),
        interaction_summary={"text": "", "source": "disabled"},
        prior_recommendations={"recommended_action": [], "source": "empty"},
        llm=llm,
    )

    assert payload["source"] == "llm"
    assert payload["recommended_action"][0]["description"].startswith("Prioritize prior-auth review")
    assert payload["recommended_action"][0]["story_alignment"][1].startswith("research_consideration: Analyst to review variance driven")
    assert payload["recommended_action"][0]["priority"] == "HIGH"
    assert payload["summary"]["overall_pattern"] == "Concentrated"
    assert payload["llm_tokens"]["input"] == 17
    assert payload["llm_tokens"]["output"] == 9
    assert payload["llm_tokens"]["breakdown"]["recommendations"] == {"input": 17, "output": 9}


def test_create_correlation_recommendations_falls_back_to_deterministic_payload() -> None:
    payload = create_correlation_recommendations(
        metric_name="expense_detail.total_paid",
        root_trend=_sample_root_trend(),
        drill_path=_sample_drill_path(),
        explainer_metrics=_sample_explainer_metrics(),
        interaction_matrix=_sample_interaction_matrix(),
        interaction_summary={"text": "", "source": "disabled"},
        prior_recommendations={"recommended_action": [], "source": "empty"},
        llm=FailingRecommendationLLM(),
    )

    parsed = CorrelationRecommendationsSchema(**payload)

    assert parsed.source == "deterministic"
    assert parsed.recommended_action
    assert parsed.recommended_action[0].description
    assert parsed.recommended_action[0].category == "Clinical"
    assert parsed.recommended_action[0].priority == "HIGH"
    assert any("op_001" in item for item in parsed.recommended_action[0].evidence)
    assert any("Peer benchmark" in item for item in parsed.recommended_action[0].peer_benchmarking)
    assert parsed.summary.primary_next_action == parsed.recommended_action[0].description
    assert payload["llm_tokens"]["input"] == 0
    assert payload["llm_tokens"]["output"] == 0
    assert payload["llm_tokens"]["breakdown"]["recommendations"] == {"input": 0, "output": 0}
