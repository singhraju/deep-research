"""Tests for correlation agent execution using the AgentBase wrapper."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest
import yaml
import deep_research_agents.correlation_agent as correlation_agent_module
import deep_research_agents.correlation_interaction_matrix as correlation_interaction_matrix
import deep_research_agents.correlation_recommendation as correlation_recommendation

from deep_research_agents.correlation_agent import (
    APP_VERSION,
    AgentConfigurationError,
    CorrelationAgent,
    DEFAULT_STOP_RULES,
    build_semantic_catalog,
    correlation_llm_tokens_for_step,
    empty_correlation_llm_tokens,
    get_candidate_dimensions,
)


class StubSnowparkHelper:
    """Lightweight SnowparkHelper stub returning predefined DataFrames."""

    def __init__(self, dataframes: Iterable[pd.DataFrame]) -> None:
        self._dataframes = list(dataframes)
        self.queries: List[str] = []

    def execute_query_and_return_pandas_df(self, query: str) -> pd.DataFrame:
        """Return the next stub DataFrame for each query."""
        self.queries.append(query)
        if not self._dataframes:
            raise AssertionError("No stubbed DataFrames remaining for query execution.")
        return self._dataframes.pop(0)

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None


def _write_minimal_semantic_model(target_path: Path) -> None:
    """Write a minimal semantic model YAML for correlation tests."""
    semantic_model = {
        "name": "test_semantic_model",
        "tables": [
            {
                "name": "claims_expense",
                "description": "Minimal claims expense table for tests.",
                "base_table": {
                    "database": "TEST_DB",
                    "schema": "TEST_SCHEMA",
                    "table": "CLAIMS_EXPENSE",
                },
                "time_dimensions": [
                    {
                        "name": "incurred_month",
                        "expr": "incurred_month",
                        "data_type": "number",
                        "description": "Incurred month.",
                    }
                ],
                "dimensions": [
                    {
                        "name": "procedure_name",
                        "expr": "procedure_name",
                        "data_type": "string",
                        "description": "Procedure name.",
                    }
                ],
                "facts": [
                    {
                        "name": "total_paid",
                        "expr": "total_paid",
                        "data_type": "number",
                        "description": "Total paid amount.",
                    }
                ],
                "metrics": [
                    {
                        "name": "claims_expense.total_paid",
                        "expr": "SUM({claims_expense.total_paid})",
                        "description": "Total paid metric.",
                    }
                ],
            }
        ],
    }

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(semantic_model, handle, sort_keys=False)


def test_correlation_agent_executes_sample_intent(tmp_path: Path) -> None:
    """Execute a sample intent using the AgentBase-powered correlation runner."""
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper(
        [root_summary_df, baseline_extract_df, comparison_extract_df]
    )
    output_root = tmp_path / "correlation_runs"

    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(output_root),
        test_mode=True,
    )

    stop_rules = copy.deepcopy(DEFAULT_STOP_RULES)
    sample_request = {
        "conversation_id": "test_conversation",
        "query": "Find what changes in HCPCS for ALCOHL&/RX SRVC; SUB-AC DTOX RES IP?",
        "context": {
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "explainer_metrics": [
                    "claims_expense.claim_count",
                    "claims_expense.total_admissions",
                ],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {
                        "start_time": 202509,
                        "end_time": 202511,
                    },
                    "previous_period": {
                        "start_time": 202409,
                        "end_time": 202411,
                    },
                },
                "exclude_if_filtered": True,
                "stop_rules": stop_rules,
            },
            "filters": [
                {
                    "field": "procedure_name",
                    "operator": "=",
                    "value": "ALCOHL&/RX SRVC; SUB-AC DTOX RES IP",
                    "source": "dimension_match",
                }
            ],
            "metric_hint": "claims_expense.total_paid",
        },
    }

    result = agent.execute(**sample_request, output_root=str(output_root))

    assert result["agent"] == "correlation_agent"
    assert result["conversation_id"] == "test_conversation"
    assert result["status"] == "success"
    assert result["job_id"]
    assert result["visual_component"] == {}
    assert result["validation"]["is_valid"] is True
    assert result["validation"]["checks"] == []
    assert result["validation"]["warnings"]
    assert result["validation"]["errors"] == []
    assert result["execution"]["start_time"]
    assert result["execution"]["end_time"]
    assert result["execution"]["duration_ms"] >= 0
    assert result["execution"]["version"] == APP_VERSION

    output = result["output"]
    assert output["root_metric"] == "claims_expense.total_paid"
    assert output["baseline_value"] == 1000.0
    assert output["comparison_value"] == 1500.0
    assert output["delta_value"] == 500.0
    assert output["drill_path"] == []
    assert "min_abs_delta" in " ".join(output.get("warnings", []))
    assert Path(output["manifest_path"]).exists()
    assert Path(output["executive_summary_path"]).exists()
    assert not Path(output["run_dir"]).joinpath("aggregates").exists()
    assert output["narrative_summary"] == ""
    assert output["executive_summary"] == ""
    assert output["executive_summary_source"] == "disabled"
    assert output_root.exists()


def test_correlation_agent_save_parquet_enabled_creates_aggregate_folder(tmp_path: Path) -> None:
    """When save_parquet is enabled via request/context, aggregate parquet artifacts are written."""
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper([root_summary_df, baseline_extract_df, comparison_extract_df])
    output_root = tmp_path / "correlation_runs"
    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(output_root),
        test_mode=True,
    )

    result = agent.execute(
        conversation_id="test_conversation",
        query="Find cost changes",
        context={
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {
                        "start_time": 202509,
                        "end_time": 202511,
                    },
                    "previous_period": {
                        "start_time": 202409,
                        "end_time": 202411,
                    },
                },
                "stop_rules": copy.deepcopy(DEFAULT_STOP_RULES),
            },
            "save_parquet": True,
        },
        output_root=str(output_root),
    )

    output = result["output"]
    assert Path(output["run_dir"]).joinpath("aggregates").exists()
    manifest_path = Path(output["manifest_path"])
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "root_baseline.parquet" in manifest_text or "root_baseline.json" in manifest_text


def test_correlation_agent_explicitly_enables_summary_creation(tmp_path: Path) -> None:
    """When disable_summary_creation is explicitly false, summary text fields are produced."""
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper([root_summary_df, baseline_extract_df, comparison_extract_df])
    output_root = tmp_path / "correlation_runs"
    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(output_root),
        test_mode=True,
    )

    result = agent.execute(
        conversation_id="test_conversation",
        query="Find cost changes",
        context={
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {
                        "start_time": 202509,
                        "end_time": 202511,
                    },
                    "previous_period": {
                        "start_time": 202409,
                        "end_time": 202411,
                    },
                },
                "stop_rules": copy.deepcopy(DEFAULT_STOP_RULES),
            },
            "disable_summary_creation": False,
        },
        output_root=str(output_root),
    )

    output = result["output"]
    assert output["narrative_summary"]
    assert output["executive_summary"]
    assert output["executive_summary_source"] in {"deterministic", "llm"}

    executive_summary_path = Path(output["executive_summary_path"])
    assert executive_summary_path.exists()
    executive_payload = json.loads(executive_summary_path.read_text(encoding="utf-8"))
    assert executive_payload["narrative_summary"]
    assert executive_payload["executive_summary"]
    assert executive_payload["executive_summary_source"] in {"deterministic", "llm"}


def test_correlation_agent_generates_recommendations_when_summaries_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper([root_summary_df, baseline_extract_df, comparison_extract_df])
    output_root = tmp_path / "correlation_runs"
    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(output_root),
        test_mode=True,
    )

    def _stub_execute_interaction_matrix(**kwargs):
        return {
            "enabled": True,
            "summary": {"status": "success"},
            "operational": {
                "artifact_paths": {"sql": "", "delta": "", "full_matrix": ""},
                "selected_cells": [{"cell_id": "op_001", "delta_value": 500.0, "dimension_values": {"procedure_name": "PROC"}}],
            },
            "clinical": {
                "artifact_paths": {"sql": "", "delta": "", "full_matrix": ""},
                "selected_cells": [],
                "offset_cells_preview": [],
            },
            "interaction_summary": {"text": "", "source": "disabled"},
            "recommended_action": {"recommended_action": [], "summary": {"overall_pattern": "Needs validation", "primary_next_action": "", "do_not_overgeneralize": ""}, "source": "empty"},
        }

    def _stub_create_correlation_recommendations(**kwargs):
        assert kwargs["interaction_summary"]["source"] == "disabled"
        assert kwargs["drill_path"] == []
        assert kwargs["interaction_matrix"]["operational"]["selected_cells"][0]["cell_id"] == "op_001"
        return {
            "recommended_action": [
                {
                    "rank": 1,
                    "priority": "HIGH",
                    "category": "Operational",
                    "description": "Review the top operational cell.",
                    "evidence": [
                        "Operational interaction cell op_001 changed $500; dimensions: procedure_name=PROC.",
                    ],
                    "story_alignment": [
                        "Why: The top operational cell concentrated the increase.",
                        "research_consideration: Analyst to review claim drivers.",
                        "cost_of_care_suggestion: Claims processing opportunity.",
                    ],
                    "peer_benchmarking": [],
                    "citation": [],
                }
            ],
            "summary": {
                "overall_pattern": "Concentrated",
                "primary_next_action": "Review the top operational cell.",
                "do_not_overgeneralize": "Do not generalize beyond the concentrated segment.",
            },
            "source": "deterministic",
        }

    monkeypatch.setattr(correlation_interaction_matrix, "execute_interaction_matrix", _stub_execute_interaction_matrix)
    monkeypatch.setattr(correlation_recommendation, "create_correlation_recommendations", _stub_create_correlation_recommendations)

    result = agent.execute(
        conversation_id="test_conversation",
        query="Find cost changes",
        context={
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {
                        "start_time": 202509,
                        "end_time": 202511,
                    },
                    "previous_period": {
                        "start_time": 202409,
                        "end_time": 202411,
                    },
                },
                "stop_rules": copy.deepcopy(DEFAULT_STOP_RULES),
                "interaction_matrix": {"enabled": True, "preview_limits": {"max_recommendations": 3}},
            },
            "disable_summary_creation": True,
        },
        output_root=str(output_root),
    )

    output = result["output"]
    assert output["narrative_summary"] == ""
    assert output["executive_summary"] == ""
    assert output["executive_summary_source"] == "disabled"
    assert result["recommended_action"][0]["description"] == "Review the top operational cell."
    assert "recommended_action" not in output

    recommendations_path = Path(output["run_dir"]) / "summary" / "interaction_recommendations.json"
    assert recommendations_path.exists()
    recommendations_payload = json.loads(recommendations_path.read_text(encoding="utf-8"))
    assert recommendations_payload["source"] == "deterministic"
    assert recommendations_payload["recommended_action"][0]["description"] == "Review the top operational cell."
    assert "summary" not in recommendations_payload

    executive_summary_path = Path(output["executive_summary_path"])
    executive_summary_payload = json.loads(executive_summary_path.read_text(encoding="utf-8"))
    assert executive_summary_payload["recommended_action"][0]["description"] == "Review the top operational cell."


def test_correlation_agent_aggregates_llm_tokens_with_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper([root_summary_df, baseline_extract_df, comparison_extract_df])
    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(tmp_path / "correlation_runs"),
        llm=object(),
        test_mode=True,
    )

    monkeypatch.setattr(
        correlation_agent_module,
        "generate_executive_summary",
        lambda llm, input_text: ("Executive summary from llm.", correlation_llm_tokens_for_step("executive_summary", 11, 7)),
    )
    monkeypatch.setattr(
        correlation_interaction_matrix,
        "execute_interaction_matrix",
        lambda **kwargs: {
            "enabled": True,
            "summary": {"status": "success"},
            "operational": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}, "selected_cells": []},
            "clinical": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}, "selected_cells": [], "offset_cells_preview": []},
            "interaction_summary": {"text": "Interaction summary from llm.", "source": "llm"},
            "recommended_action": {"recommended_action": [], "source": "empty"},
            "llm_tokens": correlation_llm_tokens_for_step("interaction_summary", 5, 3),
        },
    )
    monkeypatch.setattr(
        correlation_recommendation,
        "create_correlation_recommendations",
        lambda **kwargs: {
            "recommended_action": [
                {
                    "rank": 1,
                    "priority": "HIGH",
                    "category": "Operational",
                    "description": "Review the growth pocket.",
                    "evidence": [],
                    "story_alignment": [],
                    "peer_benchmarking": [],
                    "citation": [],
                }
            ],
            "summary": {
                "overall_pattern": "Concentrated",
                "primary_next_action": "Review the growth pocket.",
                "do_not_overgeneralize": "Validate before scaling.",
            },
            "source": "llm",
            "llm_tokens": correlation_llm_tokens_for_step("recommendations", 13, 2),
        },
    )

    result = agent.execute(
        conversation_id="test_conversation",
        query="Find cost changes",
        context={
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {"start_time": 202509, "end_time": 202511},
                    "previous_period": {"start_time": 202409, "end_time": 202411},
                },
                "stop_rules": copy.deepcopy(DEFAULT_STOP_RULES),
                "interaction_matrix": {"enabled": True, "preview_limits": {"max_recommendations": 3}},
            },
            "disable_summary_creation": False,
        },
    )

    assert result["tokens"]["input"] == 29
    assert result["tokens"]["output"] == 12
    assert result["tokens"]["breakdown"]["executive_summary"] == {"input": 11, "output": 7}
    assert result["tokens"]["breakdown"]["interaction_summary"] == {"input": 5, "output": 3}
    assert result["tokens"]["breakdown"]["recommendations"] == {"input": 13, "output": 2}
    assert result["output"]["llm_tokens"]["input"] == 29
    assert result["output"]["llm_tokens"]["output"] == 12
    assert result["output"]["llm_tokens"]["breakdown"]["recommendations"] == {"input": 13, "output": 2}


def test_correlation_agent_preserves_partial_llm_token_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    root_summary_df = pd.DataFrame(
        {
            "period_bucket": ["baseline", "comparison"],
            "metric_value": [1000.0, 1500.0],
            "raw_row_count": [10, 12],
        }
    )
    baseline_extract_df = pd.DataFrame(
        {
            "incurred_month": [202409],
            "metric_value": [1000.0],
            "raw_row_count": [10],
        }
    )
    comparison_extract_df = pd.DataFrame(
        {
            "incurred_month": [202509],
            "metric_value": [1500.0],
            "raw_row_count": [12],
        }
    )

    helper = StubSnowparkHelper([root_summary_df, baseline_extract_df, comparison_extract_df])
    agent = CorrelationAgent(
        yaml_path=str(yaml_path),
        snowflake_helper=helper,
        output_root=str(tmp_path / "correlation_runs"),
        llm=object(),
        test_mode=True,
    )

    monkeypatch.setattr(
        correlation_agent_module,
        "generate_executive_summary",
        lambda llm, input_text: ("Executive summary from llm.", correlation_llm_tokens_for_step("executive_summary", 4, 1)),
    )
    monkeypatch.setattr(
        correlation_interaction_matrix,
        "execute_interaction_matrix",
        lambda **kwargs: {
            "enabled": True,
            "summary": {"status": "success"},
            "operational": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}, "selected_cells": []},
            "clinical": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}, "selected_cells": [], "offset_cells_preview": []},
            "interaction_summary": {"text": "", "source": "deterministic"},
            "recommended_action": {"recommended_action": [], "source": "empty"},
            "llm_tokens": empty_correlation_llm_tokens(),
        },
    )
    monkeypatch.setattr(
        correlation_recommendation,
        "create_correlation_recommendations",
        lambda **kwargs: {
            "recommended_action": [],
            "summary": {
                "overall_pattern": "Needs validation",
                "primary_next_action": "",
                "do_not_overgeneralize": "",
            },
            "source": "deterministic",
            "llm_tokens": correlation_llm_tokens_for_step("recommendations", 6, 2),
        },
    )

    result = agent.execute(
        conversation_id="test_conversation",
        query="Find cost changes",
        context={
            "analysis_mode_parameters": {
                "name": "cost_change_investigation_over_time_window",
                "drill_metric": ["claims_expense.total_paid"],
                "period": {
                    "rolling_time_dimension": "claims_expense.incurred_month",
                    "current_period": {"start_time": 202509, "end_time": 202511},
                    "previous_period": {"start_time": 202409, "end_time": 202411},
                },
                "stop_rules": copy.deepcopy(DEFAULT_STOP_RULES),
                "interaction_matrix": {"enabled": True},
            },
            "disable_summary_creation": False,
        },
    )

    assert result["tokens"]["input"] == 10
    assert result["tokens"]["output"] == 3
    assert result["tokens"]["breakdown"]["executive_summary"] == {"input": 4, "output": 1}
    assert result["tokens"]["breakdown"]["interaction_summary"] == {"input": 0, "output": 0}
    assert result["tokens"]["breakdown"]["recommendations"] == {"input": 6, "output": 2}


def test_prepare_state_save_parquet_request_precedence_over_analysis_mode(tmp_path: Path) -> None:
    """Request/context save_parquet should override analysis_mode_parameters save_parquet."""
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    agent = CorrelationAgent(yaml_path=str(yaml_path), snowflake_helper=None, test_mode=True)

    state = agent.prepare_state(
        conversation_id="conversation_1",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "save_parquet": True,
            },
            "save_parquet": False,
        },
    )
    assert state["intent"]["save_parquet"] is False

    overridden_state = agent.prepare_state(
        conversation_id="conversation_2",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "save_parquet": False,
            },
            "save_parquet": False,
        },
        save_parquet=True,
    )
    assert overridden_state["intent"]["save_parquet"] is True


def test_prepare_state_disable_summary_creation_request_precedence_over_analysis_mode(tmp_path: Path) -> None:
    """Request/context disable_summary_creation should override analysis_mode_parameters value."""
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    agent = CorrelationAgent(yaml_path=str(yaml_path), snowflake_helper=None, test_mode=True)

    state = agent.prepare_state(
        conversation_id="conversation_1",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "disable_summary_creation": False,
            },
            "disable_summary_creation": True,
        },
    )
    assert state["intent"]["disable_summary_creation"] is True

    overridden_state = agent.prepare_state(
        conversation_id="conversation_2",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "disable_summary_creation": True,
            },
            "disable_summary_creation": True,
        },
        disable_summary_creation=False,
    )
    assert overridden_state["intent"]["disable_summary_creation"] is False


def test_prepare_state_generate_recommendations_request_precedence_over_analysis_mode(tmp_path: Path) -> None:
    yaml_path = tmp_path / "semantic_model.yaml"
    _write_minimal_semantic_model(yaml_path)

    agent = CorrelationAgent(yaml_path=str(yaml_path), snowflake_helper=None, test_mode=True)

    state = agent.prepare_state(
        conversation_id="conversation_1",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "generate_recommendations": False,
            },
            "generate_recommendations": True,
        },
    )
    assert state["intent"]["generate_recommendations"] is True

    overridden_state = agent.prepare_state(
        conversation_id="conversation_2",
        query="Check trend",
        context={
            "analysis_mode_parameters": {
                "drill_metric": ["claims_expense.total_paid"],
                "generate_recommendations": True,
            },
            "generate_recommendations": True,
        },
        generate_recommendations=False,
    )
    assert overridden_state["intent"]["generate_recommendations"] is False


def test_candidate_dimensions_respect_drill_dimension_allowlist() -> None:
    semantic_model = {
        "name": "test_semantic_model",
        "tables": [
            {
                "name": "claims_expense",
                "description": "Claims expense table.",
                "base_table": {
                    "database": "TEST_DB",
                    "schema": "TEST_SCHEMA",
                    "table": "CLAIMS_EXPENSE",
                },
                "time_dimensions": [
                    {
                        "name": "incurred_month",
                        "expr": "incurred_month",
                        "data_type": "number",
                        "description": "Incurred month.",
                    }
                ],
                "dimensions": [
                    {
                        "name": "lob_description",
                        "expr": "lob_description",
                        "data_type": "string",
                        "description": "Line of business description.",
                    },
                    {
                        "name": "procedure_name",
                        "expr": "procedure_name",
                        "data_type": "string",
                        "description": "Procedure name.",
                    },
                ],
            }
        ],
    }
    catalog = build_semantic_catalog(semantic_model)
    intent = {
        "analysis_mode_parameters": {
            "period": {"rolling_time_dimension": "claims_expense.incurred_month"},
            "drill_dimensions": ["lob_description"],
        },
        "filters": [],
    }

    candidate_dimensions = get_candidate_dimensions(intent, catalog, "claims_expense")

    assert candidate_dimensions == ["lob_description"]


def test_candidate_dimensions_raise_on_missing_allowlist() -> None:
    semantic_model = {
        "name": "test_semantic_model",
        "tables": [
            {
                "name": "claims_expense",
                "description": "Claims expense table.",
                "base_table": {
                    "database": "TEST_DB",
                    "schema": "TEST_SCHEMA",
                    "table": "CLAIMS_EXPENSE",
                },
                "time_dimensions": [
                    {
                        "name": "incurred_month",
                        "expr": "incurred_month",
                        "data_type": "number",
                        "description": "Incurred month.",
                    }
                ],
                "dimensions": [
                    {
                        "name": "lob_description",
                        "expr": "lob_description",
                        "data_type": "string",
                        "description": "Line of business description.",
                    }
                ],
            }
        ],
    }
    catalog = build_semantic_catalog(semantic_model)
    intent = {
        "analysis_mode_parameters": {
            "period": {"rolling_time_dimension": "claims_expense.incurred_month"},
            "drill_dimensions": ["lob_description", "missing_dimension"],
        },
        "filters": [],
    }

    with pytest.raises(AgentConfigurationError, match="missing_dimension"):
        get_candidate_dimensions(intent, catalog, "claims_expense")


def test_prepare_state_defaults_drill_dimensions(tmp_path: Path) -> None:
    semantic_model = {
        "name": "test_semantic_model",
        "tables": [
            {
                "name": "claims_expense",
                "description": "Claims expense table.",
                "base_table": {
                    "database": "TEST_DB",
                    "schema": "TEST_SCHEMA",
                    "table": "CLAIMS_EXPENSE",
                },
                "time_dimensions": [
                    {
                        "name": "incurred_month",
                        "expr": "incurred_month",
                        "data_type": "number",
                        "description": "Incurred month.",
                    }
                ],
                "dimensions": [
                    {
                        "name": "procedure_name",
                        "expr": "procedure_name",
                        "data_type": "string",
                        "description": "Procedure name.",
                    }
                ],
                "facts": [
                    {
                        "name": "total_paid",
                        "expr": "total_paid",
                        "data_type": "number",
                        "description": "Total paid amount.",
                    }
                ],
                "metrics": [
                    {
                        "name": "claims_expense.total_paid",
                        "expr": "SUM({claims_expense.total_paid})",
                        "description": "Total paid metric.",
                    }
                ],
            }
        ],
        "analysis_modes": [
            {
                "name": "cost_change_investigation_over_time_window",
                "drill_dimensions": ["procedure_name"],
            }
        ],
    }

    yaml_path = tmp_path / "semantic_model.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(semantic_model, handle, sort_keys=False)

    agent = CorrelationAgent(yaml_path=str(yaml_path), snowflake_helper=None, test_mode=True)
    state = agent.prepare_state(
        conversation_id="test_conversation",
        query="Test",
        context={"analysis_mode_parameters": {"drill_metric": ["claims_expense.total_paid"]}},
    )

    assert state["intent"]["analysis_mode_parameters"]["drill_dimensions"] == ["procedure_name"]
