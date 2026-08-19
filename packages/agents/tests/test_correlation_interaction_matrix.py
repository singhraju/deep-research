from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from deep_research_agents.correlation_agent import build_semantic_catalog
from deep_research_agents.correlation_interaction_matrix import (
    _resolve_stage_dimensions,
    build_interaction_aggregate_query,
    execute_interaction_matrix,
    pivot_interaction_comparison,
    render_or_filter_groups,
)


class StubSnowparkHelper:
    def __init__(self, dataframes: Iterable[pd.DataFrame]) -> None:
        self._dataframes = list(dataframes)
        self.queries: List[str] = []

    def execute_query_and_return_pandas_df(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        if not self._dataframes:
            raise AssertionError("No stubbed DataFrames remaining for query execution.")
        return self._dataframes.pop(0)


class StubLLM:
    def __init__(self) -> None:
        self.summary_response = "Operational interactions concentrated in two cells, with one downstream clinical pocket standing out."

    def invoke(self, messages):
        class _Response:
            def __init__(self, content: str) -> None:
                self.content = content

        _ = messages
        return _Response(self.summary_response)


def _build_semantic_model() -> dict:
    return {
        "name": "interaction_test_model",
        "tables": [
            {
                "name": "expense_detail",
                "description": "Minimal interaction-matrix test table.",
                "base_table": {
                    "database": "TEST_DB",
                    "schema": "TEST_SCHEMA",
                    "table": "EXPENSE_DETAIL",
                },
                "time_dimensions": [
                    {
                        "name": "incurred_month",
                        "expr": "incurred_month",
                        "data_type": "number",
                    }
                ],
                "dimensions": [
                    {"name": "pa_required_code", "expr": "pa_required_code", "data_type": "string"},
                    {"name": "rendering_hospital_system", "expr": "rendering_hospital_system", "data_type": "string"},
                    {"name": "product_description", "expr": "product_description", "data_type": "string"},
                    {"name": "facility_type", "expr": "facility_type", "data_type": "string"},
                    {"name": "mbu_cls_short_description", "expr": "mbu_cls_short_description", "data_type": "string"},
                    {"name": "service_area_state", "expr": "service_area_state", "data_type": "string"},
                    {"name": "lob_code", "expr": "lob_code", "data_type": "string"},
                    {"name": "drg_name", "expr": "drg_name", "data_type": "string"},
                    {"name": "primary_diagnosis_name", "expr": "primary_diagnosis_name", "data_type": "string"},
                    {"name": "hcc_medium", "expr": "hcc_medium", "data_type": "string"},
                ],
                "facts": [
                    {"name": "paid_amount", "expr": "paid_amount", "data_type": "number"},
                    {"name": "allowed_amount", "expr": "allowed_amount", "data_type": "number"},
                    {"name": "admit_count", "expr": "admit_count", "data_type": "number"},
                    {"name": "claim_number", "expr": "claim_number", "data_type": "string"},
                ],
                "metrics": [
                    {"name": "expense_detail.total_paid", "expr": "SUM({expense_detail.paid_amount})"},
                    {"name": "expense_detail.claim_count", "expr": "COUNT(DISTINCT {expense_detail.claim_number})"},
                    {"name": "expense_detail.total_admissions", "expr": "SUM({expense_detail.admit_count})"},
                    {"name": "expense_detail.avg_paid_per_admit", "expr": "SUM({expense_detail.paid_amount}) / NULLIF(SUM({expense_detail.admit_count}), 0)"},
                    {"name": "expense_detail.total_allowed", "expr": "SUM({expense_detail.allowed_amount})"},
                    {"name": "expense_detail.avg_allowed_per_admit", "expr": "SUM({expense_detail.allowed_amount}) / NULLIF(SUM({expense_detail.admit_count}), 0)"},
                    {"name": "expense_detail.paid_ratio", "expr": "SUM({expense_detail.paid_amount}) / NULLIF(SUM({expense_detail.allowed_amount}), 0)"},
                ],
            }
        ],
    }


def test_stage_dimensions_respect_hard_filters_and_carry_through() -> None:
    catalog = build_semantic_catalog(_build_semantic_model())
    stage_config = {
        "dimensions": ["pa_required_code", "product_description"],
        "carry_through_dimensions": ["service_area_state", "lob_code"],
    }
    filters = [
        {"field": "pa_required_code", "operator": "=", "value": "N", "source": "dimension_match"},
        {"field": "service_area_state", "operator": "=", "value": "CO", "source": "dimension_match"},
        {"field": "lob_code", "operator": "in", "value": ["A", "B"], "source": "dimension_match"},
    ]

    eligible, excluded = _resolve_stage_dimensions(stage_config, catalog, filters)

    assert eligible == ["product_description", "service_area_state", "lob_code"]
    assert excluded == ["pa_required_code"]


def test_render_or_filter_groups_preserves_or_of_and_shape() -> None:
    catalog = build_semantic_catalog(_build_semantic_model())
    alias_map = {"expense_detail": "ed"}
    sql = render_or_filter_groups(
        [
            [
                {"field": "pa_required_code", "operator": "=", "value": "N", "source": "dimension_match"},
                {"field": "product_description", "operator": "=", "value": "PPO", "source": "dimension_match"},
            ],
            [
                {"field": "pa_required_code", "operator": "=", "value": "Y", "source": "dimension_match"},
                {"field": "product_description", "operator": "=", "value": "HMO", "source": "dimension_match"},
            ],
        ],
        catalog,
        alias_map,
        ["expense_detail"],
    )

    assert " OR " in sql
    assert "ed.pa_required_code = 'N'" in sql
    assert "ed.product_description = 'PPO'" in sql
    assert "ed.pa_required_code = 'Y'" in sql
    assert "ed.product_description = 'HMO'" in sql
    assert "IN ('N', 'Y')" not in sql


def test_pivot_interaction_comparison_handles_multi_key_rows() -> None:
    df = pd.DataFrame(
        {
            "pa_required_code": ["N", "N", "Y", "Y"],
            "product_description": ["PPO", "PPO", "HMO", "HMO"],
            "period_bucket": ["baseline", "comparison", "baseline", "comparison"],
            "metric_value": [10.0, 30.0, 5.0, 25.0],
            "raw_row_count": [1, 2, 1, 3],
            "expense_detail_claim_count": [1, 2, 1, 3],
        }
    )

    pivot = pivot_interaction_comparison(df, ["pa_required_code", "product_description"], 40.0)

    assert list(pivot[["pa_required_code", "product_description"]].iloc[0]) == ["N", "PPO"]
    assert pivot.iloc[0]["delta_value"] == 20.0
    assert pivot.iloc[0]["share_of_positive_delta"] == 0.5
    assert pivot.iloc[0]["expense_detail_claim_count_delta"] == 1.0
    assert pivot.iloc[0]["artifact_row_ref"] == 1


def test_execute_interaction_matrix_returns_traceable_cells_and_artifacts(tmp_path: Path) -> None:
    catalog = build_semantic_catalog(_build_semantic_model())
    metric = catalog["metrics_by_name"]["expense_detail.total_paid"]
    helper = StubSnowparkHelper(
        [
            pd.DataFrame(
                {
                    "pa_required_code": ["N", "N", "Y", "Y"],
                    "rendering_hospital_system": ["Cedars", "Cedars", "Emory", "Emory"],
                    "product_description": ["PPO", "PPO", "HMO", "HMO"],
                    "facility_type": ["Acute", "Acute", "Acute", "Acute"],
                    "mbu_cls_short_description": ["Small Group", "Small Group", "Small Group", "Small Group"],
                    "service_area_state": ["CO", "CO", "CO", "CO"],
                    "lob_code": ["L1", "L1", "L2", "L2"],
                    "period_bucket": ["baseline", "comparison", "baseline", "comparison"],
                    "metric_value": [10000.0, 30000.0, 5000.0, 25000.0],
                    "raw_row_count": [2, 4, 1, 3],
                    "expense_detail_claim_count": [2, 4, 1, 3],
                    "expense_detail_total_admissions": [1, 2, 1, 2],
                    "expense_detail_avg_paid_per_admit": [10000.0, 15000.0, 5000.0, 12500.0],
                    "expense_detail_total_allowed": [12000.0, 35000.0, 7000.0, 28000.0],
                    "expense_detail_avg_allowed_per_admit": [12000.0, 17500.0, 7000.0, 14000.0],
                    "expense_detail_paid_ratio": [0.83, 0.86, 0.71, 0.89],
                }
            ),
            pd.DataFrame(
                {
                    "drg_name": ["DRG A", "DRG A", "DRG B", "DRG B"],
                    "primary_diagnosis_name": ["DX A", "DX A", "DX B", "DX B"],
                    "hcc_medium": ["IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB"],
                    "mbu_cls_short_description": ["Small Group", "Small Group", "Small Group", "Small Group"],
                    "period_bucket": ["baseline", "comparison", "baseline", "comparison"],
                    "metric_value": [5000.0, 18000.0, 9000.0, 4000.0],
                    "raw_row_count": [1, 2, 2, 1],
                    "expense_detail_claim_count": [1, 2, 2, 1],
                    "expense_detail_total_admissions": [1, 2, 1, 1],
                    "expense_detail_avg_paid_per_admit": [5000.0, 9000.0, 9000.0, 4000.0],
                    "expense_detail_total_allowed": [6000.0, 20000.0, 10000.0, 5000.0],
                    "expense_detail_avg_allowed_per_admit": [6000.0, 10000.0, 10000.0, 5000.0],
                    "expense_detail_paid_ratio": [0.83, 0.90, 0.90, 0.80],
                }
            ),
        ]
    )
    llm = StubLLM()
    run_dir = tmp_path / "run"
    queries_dir = run_dir / "queries"
    aggregates_dir = run_dir / "aggregates"
    summary_dir = run_dir / "summary"
    queries_dir.mkdir(parents=True)
    aggregates_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    manifest = {"files": []}
    warnings: List[str] = []
    intent = {
        "analysis_mode_parameters": {
            "explainer_metrics": [
                "expense_detail.claim_count",
                "expense_detail.total_admissions",
                "expense_detail.avg_paid_per_admit",
            ],
            "interaction_matrix": {
                "enabled": True,
                "trigger_rules": {
                    "interaction_stage": {
                        "min_abs_net_delta": 1000,
                        "min_drill_path_depth": 2,
                        "run_when_repeated_delta_ratio": 0.95,
                        "run_when_low_volume": False,
                    },
                    "clinical_stage": {
                        "require_selected_operational_cells": True,
                        "min_selected_operational_positive_delta": 1000,
                    },
                },
                "categories": {
                    "operational": {
                        "dimensions": [
                            "pa_required_code",
                            "rendering_hospital_system",
                            "product_description",
                            "facility_type",
                            "mbu_cls_short_description",
                        ],
                        "carry_through_dimensions": ["service_area_state", "lob_code"],
                    },
                    "clinical": {
                        "dimensions": ["drg_name", "primary_diagnosis_name", "hcc_medium"],
                        "carry_through_dimensions": ["mbu_cls_short_description"],
                    },
                },
            },
        }
    }
    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [15000.0, 55000.0],
            "raw_row_count": [3, 7],
        }
    )
    drill_path = [
        {
            "level": 1,
            "dimension": "hcc_medium",
            "top_segments": [{"value": "IP OB Dlvry/Well NB", "delta_value": 40000.0}],
        },
        {
            "level": 2,
            "dimension": "er_admit_indicator",
            "top_segments": [{"value": "Y", "delta_value": 39000.0}],
        },
    ]

    result = execute_interaction_matrix(
        intent=intent,
        catalog=catalog,
        metric_name="expense_detail.total_paid",
        metric=metric,
        primary_table="expense_detail",
        period_window={
            "time_dimension": "expense_detail.incurred_month",
            "start_time": 202501,
            "end_time": 202512,
            "baseline_start_time": 202401,
            "baseline_end_time": 202412,
            "comparison_strategy": "prior_year_same_window",
            "baseline_months": [202401],
            "comparison_months": [202501],
        },
        filters=[],
        drill_path=drill_path,
        root_summary_df=root_summary_df,
        delta_value=40000.0,
        queries_dir=queries_dir,
        aggregates_dir=aggregates_dir,
        summary_dir=summary_dir,
        run_dir=run_dir,
        snowflake_helper=helper,
        llm=llm,
        manifest=manifest,
        warnings=warnings,
        disable_summary_creation=False,
    )

    assert result["summary"]["status"] == "success"
    assert result["operational"]["selected_cells"][0]["cell_id"] == "op_001"
    assert result["operational"]["selected_cells"][0]["dimension_values"]["pa_required_code"] == "N"
    assert len(result["operational"]["selected_cell_filter_groups"]) == 2
    assert result["clinical"]["selected_cells"][0]["cell_id"] == "cl_001"
    assert result["clinical"]["offset_cells_preview"][0]["cell_id"] == "cl_off_001"
    assert result["interaction_summary"]["source"] == "llm"
    assert result["recommended_action"]["recommended_action"] == []
    assert "summary" not in result["recommended_action"]
    assert result["operational"]["artifact_paths"]["delta"] == ""
    assert result["operational"]["artifact_paths"]["full_matrix"] == ""
    assert result["clinical"]["artifact_paths"]["delta"] == ""
    assert result["clinical"]["artifact_paths"]["full_matrix"] == ""
    assert (summary_dir / "interaction_summary.json").exists()
    assert (summary_dir / "interaction_recommendations.json").exists()
    recommendations_payload = json.loads((summary_dir / "interaction_recommendations.json").read_text(encoding="utf-8"))
    assert "summary" not in recommendations_payload
    assert any(path.startswith("queries/interaction_matrix/") for path in manifest["files"])
    assert not any(path.startswith("aggregates/interaction_matrix/") for path in manifest["files"])


def test_execute_interaction_matrix_disable_summary_creation_skips_summary_artifact(tmp_path: Path) -> None:
    catalog = build_semantic_catalog(_build_semantic_model())
    metric = catalog["metrics_by_name"]["expense_detail.total_paid"]
    helper = StubSnowparkHelper(
        [
            pd.DataFrame(
                {
                    "pa_required_code": ["N", "N", "Y", "Y"],
                    "rendering_hospital_system": ["Cedars", "Cedars", "Emory", "Emory"],
                    "product_description": ["PPO", "PPO", "HMO", "HMO"],
                    "facility_type": ["Acute", "Acute", "Acute", "Acute"],
                    "mbu_cls_short_description": ["Small Group", "Small Group", "Small Group", "Small Group"],
                    "service_area_state": ["CO", "CO", "CO", "CO"],
                    "lob_code": ["L1", "L1", "L2", "L2"],
                    "period_bucket": ["baseline", "comparison", "baseline", "comparison"],
                    "metric_value": [10000.0, 30000.0, 5000.0, 25000.0],
                    "raw_row_count": [2, 4, 1, 3],
                    "expense_detail_claim_count": [2, 4, 1, 3],
                    "expense_detail_total_admissions": [1, 2, 1, 2],
                    "expense_detail_avg_paid_per_admit": [10000.0, 15000.0, 5000.0, 12500.0],
                    "expense_detail_total_allowed": [12000.0, 35000.0, 7000.0, 28000.0],
                    "expense_detail_avg_allowed_per_admit": [12000.0, 17500.0, 7000.0, 14000.0],
                    "expense_detail_paid_ratio": [0.83, 0.86, 0.71, 0.89],
                }
            ),
            pd.DataFrame(
                {
                    "drg_name": ["DRG A", "DRG A", "DRG B", "DRG B"],
                    "primary_diagnosis_name": ["DX A", "DX A", "DX B", "DX B"],
                    "hcc_medium": ["IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB", "IP OB Dlvry/Well NB"],
                    "mbu_cls_short_description": ["Small Group", "Small Group", "Small Group", "Small Group"],
                    "period_bucket": ["baseline", "comparison", "baseline", "comparison"],
                    "metric_value": [5000.0, 18000.0, 9000.0, 4000.0],
                    "raw_row_count": [1, 2, 2, 1],
                    "expense_detail_claim_count": [1, 2, 2, 1],
                    "expense_detail_total_admissions": [1, 2, 1, 1],
                    "expense_detail_avg_paid_per_admit": [5000.0, 9000.0, 9000.0, 4000.0],
                    "expense_detail_total_allowed": [6000.0, 20000.0, 10000.0, 5000.0],
                    "expense_detail_avg_allowed_per_admit": [6000.0, 10000.0, 10000.0, 5000.0],
                    "expense_detail_paid_ratio": [0.83, 0.90, 0.90, 0.80],
                }
            ),
        ]
    )
    llm = StubLLM()
    run_dir = tmp_path / "run"
    queries_dir = run_dir / "queries"
    aggregates_dir = run_dir / "aggregates"
    summary_dir = run_dir / "summary"
    queries_dir.mkdir(parents=True)
    aggregates_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    manifest = {"files": []}
    warnings: List[str] = []
    intent = {
        "analysis_mode_parameters": {
            "explainer_metrics": [
                "expense_detail.claim_count",
                "expense_detail.total_admissions",
                "expense_detail.avg_paid_per_admit",
            ],
            "interaction_matrix": {
                "enabled": True,
                "trigger_rules": {
                    "interaction_stage": {
                        "min_abs_net_delta": 1000,
                        "min_drill_path_depth": 2,
                        "run_when_repeated_delta_ratio": 0.95,
                        "run_when_low_volume": False,
                    },
                    "clinical_stage": {
                        "require_selected_operational_cells": True,
                        "min_selected_operational_positive_delta": 1000,
                    },
                },
                "categories": {
                    "operational": {
                        "dimensions": [
                            "pa_required_code",
                            "rendering_hospital_system",
                            "product_description",
                            "facility_type",
                            "mbu_cls_short_description",
                        ],
                        "carry_through_dimensions": ["service_area_state", "lob_code"],
                    },
                    "clinical": {
                        "dimensions": ["drg_name", "primary_diagnosis_name", "hcc_medium"],
                        "carry_through_dimensions": ["mbu_cls_short_description"],
                    },
                },
            },
        }
    }
    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [15000.0, 55000.0],
            "raw_row_count": [3, 7],
        }
    )
    drill_path = [
        {
            "level": 1,
            "dimension": "hcc_medium",
            "top_segments": [{"value": "IP OB Dlvry/Well NB", "delta_value": 40000.0}],
        },
        {
            "level": 2,
            "dimension": "er_admit_indicator",
            "top_segments": [{"value": "Y", "delta_value": 39000.0}],
        },
    ]

    result = execute_interaction_matrix(
        intent=intent,
        catalog=catalog,
        metric_name="expense_detail.total_paid",
        metric=metric,
        primary_table="expense_detail",
        period_window={
            "time_dimension": "expense_detail.incurred_month",
            "start_time": 202501,
            "end_time": 202512,
            "baseline_start_time": 202401,
            "baseline_end_time": 202412,
            "comparison_strategy": "prior_year_same_window",
            "baseline_months": [202401],
            "comparison_months": [202501],
        },
        filters=[],
        drill_path=drill_path,
        root_summary_df=root_summary_df,
        delta_value=40000.0,
        queries_dir=queries_dir,
        aggregates_dir=aggregates_dir,
        summary_dir=summary_dir,
        run_dir=run_dir,
        snowflake_helper=helper,
        llm=llm,
        manifest=manifest,
        warnings=warnings,
        disable_summary_creation=True,
    )

    assert result["summary"]["status"] == "success"
    assert result["interaction_summary"]["text"] == ""
    assert result["interaction_summary"]["source"] == "disabled"
    assert result["recommended_action"]["source"] == "empty"
    assert not (summary_dir / "interaction_summary.json").exists()
    assert (summary_dir / "interaction_recommendations.json").exists()
    recommendations_payload = json.loads((summary_dir / "interaction_recommendations.json").read_text(encoding="utf-8"))
    assert "summary" not in recommendations_payload
    assert not any(path.endswith("summary/interaction_summary.json") for path in manifest["files"])
    assert any(path.endswith("summary/interaction_recommendations.json") for path in manifest["files"])


def test_build_interaction_aggregate_query_includes_multiple_dimensions_and_explainers() -> None:
    catalog = build_semantic_catalog(_build_semantic_model())
    metric = catalog["metrics_by_name"]["expense_detail.total_paid"]
    sql = build_interaction_aggregate_query(
        catalog=catalog,
        metric=metric,
        filters=[],
        period_window={
            "time_dimension": "expense_detail.incurred_month",
            "start_time": 202501,
            "end_time": 202512,
            "baseline_start_time": 202401,
            "baseline_end_time": 202412,
            "comparison_strategy": "prior_year_same_window",
            "baseline_months": [202401],
            "comparison_months": [202501],
        },
        primary_table="expense_detail",
        dimension_names=["pa_required_code", "product_description"],
        explainer_metric_names=["expense_detail.claim_count"],
    )

    assert "AS pa_required_code" in sql
    assert "AS product_description" in sql
    assert "AS expense_detail_claim_count" in sql
    assert "GROUP BY" in sql
    assert "ORDER BY pa_required_code, product_description, period_bucket" in sql
