from __future__ import annotations

import copy
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from deep_research_utils.cache_utils import get_token_cache_obj

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - older LangGraph versions
    try:
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore
    except ImportError:  # pragma: no cover - allow import in environments without the package
        InMemorySaver = None  # type: ignore

try:
    from deep_research_utils.logger_config import get_logger
    from deep_research_utils.app_constant import AppConstants

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover - local/dev fallback
    logger = logging.getLogger(__name__)
    AppConstants = None  # type: ignore

try:
    from deep_research_core.base_agent import AgentExecutionError, AgentConfigurationError
except ImportError:  # pragma: no cover - allow running as a script
    # Define minimal exception classes if import fails
    class AgentExecutionError(Exception):  # type: ignore
        """Raised when agent execution fails."""
        pass
    
    class AgentConfigurationError(Exception):  # type: ignore
        """Raised when agent is misconfigured."""
        pass

try:
    from deep_research_agents.user_intent import (  # Reuse the existing intent capability instead of rewriting it.
        AnalysisModeDefinition,
        FilterCondition,
        IntentOutput,
        build_llm,
        build_semantic_index,
        identify_intent_and_filters,
        load_semantic_yaml,
        summarize_semantic_model,
        validate_intent_output,
    )
except ImportError:  # pragma: no cover - allow running as a script
    from user_intent import (  # type: ignore
        AnalysisModeDefinition,
        FilterCondition,
        IntentOutput,
        build_llm,
        build_semantic_index,
        identify_intent_and_filters,
        load_semantic_yaml,
        summarize_semantic_model,
        validate_intent_output,
    )

try:
    from deep_research_agents.correlation_agent import build_app as build_correlation_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from correlation_agent import build_app as build_correlation_app  # type: ignore
    except ImportError:
        build_correlation_app = None  # type: ignore[assignment]

try:
    from deep_research_agents.policy_hypothesis_agent import build_app as build_policy_hypothesis_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from policy_hypothesis_agent import build_app as build_policy_hypothesis_app  # type: ignore
    except ImportError:
        build_policy_hypothesis_app = None  # type: ignore[assignment]

try:
    from deep_research_agents.mandate_hypothesis_agent import build_app as build_mandate_hypothesis_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from mandate_hypothesis_agent import build_app as build_mandate_hypothesis_app  # type: ignore
    except ImportError:
        build_mandate_hypothesis_app = None  # type: ignore[assignment]

try:
    from deep_research_agents.pattern_agent import build_app as build_pattern_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from pattern_agent import build_app as build_pattern_app  # type: ignore
    except ImportError:
        build_pattern_app = None  # type: ignore[assignment]

try:
    from deep_research_agents.reimbursement_agent import build_app as build_reimbursement_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from reimbursement_agent import build_app as build_reimbursement_app  # type: ignore
    except ImportError:
        build_reimbursement_app = None  # type: ignore[assignment]

try:
    from deep_research_agents.recommendation_dtr_agent import build_app as build_recommendation_dtr_app
except ImportError:  # pragma: no cover - fallback to script imports or missing optional dependency
    try:
        from recommendation_dtr_agent import build_app as build_recommendation_dtr_app  # type: ignore
    except ImportError:
        build_recommendation_dtr_app = None  # type: ignore[assignment]

try:
    from deep_research_utils.snowflake_helper import SnowparkHelper
except ImportError:  # pragma: no cover - optional dependency for local correlation execution
    SnowparkHelper = None  # type: ignore[assignment]

# ============================================================
# Module: Parent Orchestrator
# ============================================================
#
# Why this file exists:
# - user_intent.py already resolves filters + analysis_mode.
# - This file wraps that capability in a parent LangGraph workflow.
# - Routing stays centralized in one orchestrator node instead of letting
#   every downstream node decide global control flow.
#
# Design choices:
# 1. The graph topology is compiled once.
#    We do NOT mutate the graph structure per request at runtime.
#    Instead, the router chooses which already-registered node/subgraph
#    should execute next.
#
# 2. UI context is treated as SOFT context.
#    If the user question explicitly implies a different value, the
#    extracted question filter wins and the UI context is dropped for
#    that field.
#
# 3. Every analysis mode returns a strict JSON-serializable contract.
#    For now, the built-in mode handlers are planning stubs. They prove
#    the orchestration path and output shape without pretending to have
#    executed the real SQL/drill-down logic yet.
#
# 4. Memory is thread-scoped via LangGraph checkpoints when a checkpointer
#    is provided. The persistent state only stores dynamic turn data.
#    Static assets like the semantic model and LLM client are captured in
#    closures so they do not bloat checkpoint storage.
#
# Future extension points:
# - Replace stub analysis handlers with real mode subgraphs/agents.
# - Add follow-up routing after analysis.
# - Add interrupt-based clarification if you want resumable HITL flow.
# - Fan out to multiple analysis modes later with Send when needed.
#

AnalysisStatus = Literal["planned", "success", "needs_clarification", "error"]
ContextResolutionType = Literal["matched", "kept_ui_context", "question_override"]

ANALYSIS_CONTEXT_KEYS = {
    "analysis_overrides",
    "drill_metric",
    "period",
    "rolling_window",
    "start_time",
    "end_time",
    "prebuilt_intent",
    "conversation_id",
    "live_filter_values",
    "semantic_roles",
}


class ContextResolution(TypedDict, total=False):
    """Explains how a soft UI filter was handled."""

    field: str
    ui_value: Any
    resolved_value: Any
    resolution: ContextResolutionType
    reason: str


class ClarificationRequest(TypedDict, total=False):
    """Structured clarification payload returned when routing cannot continue safely."""

    reason: str
    blocking_issues: List[str]
    questions: List[str]
    suggested_defaults: Dict[str, Any]


class AnalysisContract(TypedDict, total=False):
    """Strict JSON contract that every analysis mode must emit."""

    contract_version: str
    status: AnalysisStatus
    analysis_mode: Optional[str]
    question: str
    scope: Dict[str, Any]
    selected_configuration: Dict[str, Any]
    execution_plan: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    next_actions: List[str]
    metadata: Dict[str, Any]


class ReportContract(TypedDict, total=False):
    contract_version: str
    status: AnalysisStatus
    analysis_mode: Optional[str]
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class VisualContract(TypedDict, total=False):
    contract_version: str
    status: AnalysisStatus
    analysis_mode: Optional[str]
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SummaryContract(TypedDict, total=False):
    contract_version: str
    status: AnalysisStatus
    headline: str
    bullets: List[str]
    metadata: Dict[str, Any]


class HypothesisOutput(TypedDict, total=False):
    """Optional hypothesis bundle produced after successful correlation execution."""

    policy: List[Dict[str, Any]]
    mandate: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ResearchOutput(TypedDict, total=False):
    """Bundle produced by the pattern → reimbursement → recommendation chain."""

    business_patterns: List[Dict[str, Any]]
    pattern_summary: Dict[str, Any]
    reimbursement_by_pattern: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class FinalOutput(TypedDict, total=False):
    contract_version: str
    status: AnalysisStatus
    question: str
    context_resolution: List[ContextResolution]
    intent: Optional[IntentOutput]
    clarification_request: Optional[ClarificationRequest]
    analysis: Optional[AnalysisContract]
    report: Optional[ReportContract]
    visuals: Optional[VisualContract]
    summary: Optional[SummaryContract]
    hypotheses: Optional[HypothesisOutput]
    research: Optional[ResearchOutput]
    recent_step_summaries: List[str]
    conversation_summary: str
    last_completed_stage: str


class OrchestratorState(TypedDict, total=False):
    """Dynamic per-thread state for the parent orchestrator graph."""

    question: str
    context: Optional[Dict[str, Any]]
    conversation_id: Optional[str]
    prebuilt_intent: Optional[Dict[str, Any]]

    # Split from raw context so UI filters can remain soft.
    soft_context_filters: Dict[str, Any]
    analysis_context: Dict[str, Any]
    bypass_intent_detection: bool
    live_filter_values: Optional[Dict[str, Any]]

    # Outputs from orchestration.
    intent: IntentOutput
    context_resolution: List[ContextResolution]
    clarification_request: ClarificationRequest
    analysis_contract: AnalysisContract
    report_contract: ReportContract
    visual_contract: VisualContract
    summary_contract: SummaryContract
    hypotheses: HypothesisOutput
    business_patterns: List[Dict[str, Any]]
    pattern_summary: Dict[str, Any]
    reimbursement_by_pattern: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    research: ResearchOutput
    final_output: FinalOutput

    # Control / observability.
    route: str
    last_completed_stage: str
    step_summaries: Annotated[List[str], lambda left, right: _append_bounded(left, right, max_items=8)]
    errors: List[str]
    conversation_summary: str


AnalysisModeHandler = Callable[[OrchestratorState], AnalysisContract]
CorrelationRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]
HypothesisRunner = Callable[..., List[Dict[str, Any]]]


# =============================
# Generic helpers
# =============================

def _append_bounded(
    left: Optional[List[str]],
    right: Optional[List[str]],
    *,
    max_items: int = 8,
) -> List[str]:
    merged = list(left or [])
    if right:
        merged.extend(str(item) for item in right if item is not None)
    return merged[-max_items:]


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_non_empty(values: Any) -> Optional[Any]:
    if isinstance(values, list):
        for value in values:
            if value is not None and value != "":
                return value
        return None
    if values is None or values == "":
        return None
    return values


def _normalize_node_fragment(text: str) -> str:
    fragment = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return fragment or "unknown"


def analysis_node_name(mode_name: str) -> str:
    return f"analysis__{_normalize_node_fragment(mode_name)}"


def _ensure_jsonable(payload: Dict[str, Any]) -> None:
    json.dumps(payload, ensure_ascii=False)


def _copy_jsonable(payload: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def split_soft_context(context: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split incoming UI context into:
    - soft_context_filters: filter-like fields from the UI
    - analysis_context: analysis-specific hints (period, metric override, etc.)
    """
    if not context:
        return {}, {}

    filter_context: Dict[str, Any] = {}
    analysis_context: Dict[str, Any] = {}

    for key, value in context.items():
        if key == "analysis_overrides" and isinstance(value, Mapping):
            analysis_context.update(value)
        elif key in ANALYSIS_CONTEXT_KEYS:
            analysis_context[key] = value
        else:
            filter_context[key] = value

    return filter_context, analysis_context


def _dedupe_filters(filters: List[FilterCondition]) -> List[FilterCondition]:
    seen = set()
    deduped: List[FilterCondition] = []
    for filter_cond in filters:
        key = (
            filter_cond["field"],
            filter_cond["operator"],
            filter_cond["value"],
            filter_cond["source"],
        )
        if key not in seen:
            deduped.append(filter_cond)
            seen.add(key)
    return deduped


def merge_soft_context_filters(
    extracted_filters: List[FilterCondition],
    soft_context_filters: Optional[Dict[str, Any]],
) -> Tuple[List[FilterCondition], List[ContextResolution]]:
    """
    Merge UI context as soft defaults.

    Rule:
    - If the question already produced a filter for a UI field, keep the
      question-derived filter and treat it as an explicit override.
    - Otherwise, add the UI filter.
    """
    merged = list(extracted_filters)
    resolutions: List[ContextResolution] = []
    soft_context_filters = soft_context_filters or {}

    extracted_by_field: Dict[str, List[FilterCondition]] = {}
    for filter_cond in extracted_filters:
        extracted_by_field.setdefault(filter_cond["field"], []).append(filter_cond)

    for field, ui_value in soft_context_filters.items():
        ui_value_str = str(ui_value)
        existing = extracted_by_field.get(field, [])

        if not existing:
            merged.append(
                {
                    "field": field,
                    "operator": "=",
                    "value": ui_value_str,
                    "source": "dimension_match",
                }
            )
            resolutions.append(
                {
                    "field": field,
                    "ui_value": ui_value,
                    "resolved_value": ui_value,
                    "resolution": "kept_ui_context",
                    "reason": "No explicit question filter was found for this field, so the UI context was kept.",
                }
            )
            continue

        existing_values = {item["value"] for item in existing}
        if ui_value_str in existing_values:
            resolutions.append(
                {
                    "field": field,
                    "ui_value": ui_value,
                    "resolved_value": ui_value,
                    "resolution": "matched",
                    "reason": "The UI context already matched the question-derived filter.",
                }
            )
            continue

        # Any explicit extracted filter on the same field wins over soft UI context.
        resolutions.append(
            {
                "field": field,
                "ui_value": ui_value,
                "resolved_value": sorted(existing_values),
                "resolution": "question_override",
                "reason": "The question explicitly implied a different value for this field, so the UI context was dropped.",
            }
        )

    return _dedupe_filters(merged), resolutions


def _build_empty_intent(question: str) -> IntentOutput:
    return {
        "metric_hint": None,
        "group_by": [],
        "filters": [],
        "analysis_mode": None,
        "analysis_mode_parameters": None,
        "raw_question": question,
        "validation_warnings": [],
    }


# =============================
# Clarification rules
# =============================

def build_clarification_request(intent: IntentOutput) -> Optional[ClarificationRequest]:
    """
    Decide whether the orchestrator should stop and ask for clarification.

    v1 rules are intentionally strict because every analysis mode must emit a
    concrete contract and we allow only one mode per turn.
    """
    blocking_issues: List[str] = []
    questions: List[str] = []
    suggested_defaults: Dict[str, Any] = {}

    analysis_mode = intent.get("analysis_mode")
    confidence_score = intent.get("analysis_mode_confidence")  # Add this if available in intent
    if not analysis_mode:
        msg = "No analysis_mode was selected from the question."
        if confidence_score is not None:
            msg += f" (LLM confidence: {confidence_score:.2f})"
        blocking_issues.append(msg)
        questions.append("What kind of analysis do you want me to run for this question?")
        logger.warning(f"Analysis mode missing. {msg}")

    for warning in _safe_list(intent.get("validation_warnings")):
        blocking_issues.append(str(warning))

    params = _safe_dict(intent.get("analysis_mode_parameters"))

    drill_metric_options = _safe_list(params.get("drill_metric"))
    if analysis_mode and len(drill_metric_options) != 1:
        blocking_issues.append(
            f"Expected exactly one drill metric after intent resolution, but found {len(drill_metric_options)}."
        )
        questions.append("Which metric should I use for the analysis?")
        if drill_metric_options:
            suggested_defaults["drill_metric_options"] = drill_metric_options

    period = _safe_dict(params.get("period"))
    rolling_window_options = _safe_list(period.get("rolling_window"))
    has_explicit_dates = bool(period.get("start_time") or period.get("end_time"))
    if analysis_mode and not has_explicit_dates and len(rolling_window_options) != 1:
        blocking_issues.append(
            f"Expected exactly one rolling window or explicit dates, but found {len(rolling_window_options)} rolling window options."
        )
        questions.append("Which time window should I compare?")
        if rolling_window_options:
            suggested_defaults["rolling_window_options"] = rolling_window_options

    if not blocking_issues:
        return None

    return {
        "reason": "The request could not be routed to a concrete executable analysis contract without clarification.",
        "blocking_issues": blocking_issues,
        "questions": questions,
        "suggested_defaults": suggested_defaults,
    }


# =============================
# Mode handler defaults
# =============================

def default_generic_analysis_handler(state: OrchestratorState) -> AnalysisContract:
    """
    Generic planning stub used for any mode without a dedicated handler.

    This is intentionally honest: the orchestrator only plans the execution
    contract here. Real SQL/drill-down execution should replace this handler.
    """
    intent = state["intent"]
    params = _safe_dict(intent.get("analysis_mode_parameters"))

    contract: AnalysisContract = {
        "contract_version": "1.0",
        "status": "planned",
        "analysis_mode": intent.get("analysis_mode"),
        "question": intent.get("raw_question", state["question"]),
        "scope": {
            "filters": _copy_jsonable({"filters": intent.get("filters", [])}).get("filters", []),
            "group_by": list(intent.get("group_by", [])),
            "metric_hint": intent.get("metric_hint"),
            "context_resolution": _copy_jsonable({"items": state.get("context_resolution", [])}).get("items", []),
        },
        "selected_configuration": _copy_jsonable(params),
        "execution_plan": [
            {
                "step": 1,
                "node": "analysis_mode_executor",
                "description": "Invoke the concrete mode executor with the resolved scope and configuration.",
            },
            {
                "step": 2,
                "node": "report_builder",
                "description": "Build a strict downstream report contract from the executor result.",
            },
            {
                "step": 3,
                "node": "visual_node",
                "description": "Produce visualization recommendations based on the report contract.",
            },
            {
                "step": 4,
                "node": "summarizer",
                "description": "Generate a concise summary contract for the UI response layer.",
            },
        ],
        "findings": [],
        "artifacts": [],
        "next_actions": [
            "Attach a real mode-specific executor or subgraph to replace this generic planning stub."
        ],
        "metadata": {
            "planned_by": "orchestrator",
            "is_stub": True,
        },
    }
    _ensure_jsonable(contract)
    return contract


# =============================
# Correlation agent integration
# =============================

def _format_drill_path_summary(drill_path: List[Dict[str, Any]]) -> str:
    """Create a short, human-readable drill-path summary."""
    if not drill_path:
        return ""
    segments: List[str] = []
    for node in drill_path:
        dimension_label = node.get("dimension_label") or node.get("dimension")
        if not dimension_label:
            continue
        values = []
        for segment in _safe_list(node.get("top_segments")):
            if isinstance(segment, dict):
                value = str(segment.get("value") or "").strip()
                if value:
                    values.append(value)
        if not values:
            segments.append(str(dimension_label))
        else:
            segments.append(f"{dimension_label}: {', '.join(values)}")
    return " -> ".join(segments)


def _build_correlation_findings(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Translate correlation run output into findings for the analysis contract."""
    drill_path = [item for item in _safe_list(result.get("drill_path")) if isinstance(item, dict)]
    interaction_summary = _safe_dict(result.get("interaction_summary"))
    raw_recommendations = result.get("recommended_action")
    if isinstance(raw_recommendations, list):
        recommendations = {"recommended_action": raw_recommendations}
    else:
        recommendations = _safe_dict(raw_recommendations or result.get("recommendations"))
    warnings = [str(item) for item in _safe_list(result.get("warnings")) if item is not None]

    # Normalize legacy correlation recommendation payloads
    if isinstance(recommendations, str):
        recommendations = {"text": recommendations}

    findings: List[Dict[str, Any]] = [
        {
            "id": "root_delta",
            "metric": result.get("root_metric"),
            "baseline_value": result.get("baseline_value"),
            "comparison_value": result.get("comparison_value"),
            "delta_value": result.get("delta_value"),
            "delta_pct": result.get("delta_pct"),
        }
    ]

    if drill_path:
        findings.append(
            {
                "id": "drill_path",
                "summary": _format_drill_path_summary(drill_path),
                "path": drill_path,
            }
        )

    narrative_summary = result.get("narrative_summary")
    if narrative_summary:
        findings.append({"id": "narrative_summary", "summary": narrative_summary})

    interaction_text = str(interaction_summary.get("text") or "").strip()
    if interaction_text:
        findings.append(
            {
                "id": "interaction_summary",
                "summary": interaction_text,
                "source": interaction_summary.get("source"),
            }
        )

    try:
        try:
            from deep_research_agents.correlation_recommendation import normalize_legacy_recommendations
        except ImportError:
            from correlation_recommendation import normalize_legacy_recommendations  # type: ignore
        recommendations = normalize_legacy_recommendations(recommendations)
    except Exception:
        if isinstance(raw_recommendations, list):
            recommendations = {"recommended_action": [item for item in raw_recommendations if isinstance(item, dict)]}
        else:
            recommendations = _safe_dict(raw_recommendations or result.get("recommendations"))

    recommendation_items = [item for item in _safe_list(recommendations.get("recommended_action")) if isinstance(item, dict)]
    if recommendation_items:
        findings.append(
            {
                "id": "interaction_recommendations",
                "items": recommendation_items,
                "source": recommendations.get("source"),
            }
        )

    if warnings:
        findings.append({"id": "warnings", "items": warnings})

    return findings


def _build_correlation_artifacts(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expose file-system artifacts created by the correlation agent."""
    artifacts: List[Dict[str, Any]] = []
    run_dir = result.get("run_dir")
    interaction_matrix = _safe_dict(result.get("interaction_matrix"))
    if run_dir:
        artifacts.append({"type": "run_directory", "path": run_dir})
    manifest_path = result.get("manifest_path")
    if manifest_path:
        artifacts.append({"type": "manifest", "path": manifest_path})
    summary_path = result.get("executive_summary_path")
    if summary_path:
        artifacts.append({"type": "executive_summary", "path": summary_path})
    for stage_name in ("operational", "clinical"):
        stage_payload = _safe_dict(interaction_matrix.get(stage_name))
        artifact_paths = _safe_dict(stage_payload.get("artifact_paths"))
        for artifact_name, relative_path in artifact_paths.items():
            path_text = str(relative_path or "").strip()
            if path_text:
                artifacts.append({"type": f"interaction_{stage_name}_{artifact_name}", "path": path_text})
    return artifacts


def build_correlation_analysis_handler(
    correlation_runner: CorrelationRunner,
    *,
    output_root: Optional[str] = None,
) -> AnalysisModeHandler:
    """Wrap the correlation agent so it emits a full analysis contract."""

    def _handler(state: OrchestratorState) -> AnalysisContract:
        intent = state["intent"]
        params = _safe_dict(intent.get("analysis_mode_parameters"))
        period = _safe_dict(params.get("period"))

        selected_drill_metric = _first_non_empty(params.get("drill_metric")) or intent.get("metric_hint")
        selected_window = _first_non_empty(period.get("rolling_window"))

        selected_configuration = {
            "drill_metric": selected_drill_metric,
            "explainer_metrics": _safe_list(params.get("explainer_metrics")),
            "period": {
                "rolling_time_dimension": period.get("rolling_time_dimension"),
                "rolling_window": [selected_window] if selected_window else _safe_list(period.get("rolling_window")),
                "start_time": period.get("start_time"),
                "end_time": period.get("end_time"),
                "baseline_start_time": period.get("baseline_start_time"),
                "baseline_end_time": period.get("baseline_end_time"),
                "comparison_strategy": period.get("comparison_strategy"),
            },
            "stop_rules": _copy_jsonable(_safe_dict(params.get("stop_rules"))),
        }

        execution_plan = [
            {
                "step": 1,
                "node": "correlation_root_summary",
                "description": "Compute baseline vs comparison totals for the selected metric and scope.",
            },
            {
                "step": 2,
                "node": "correlation_period_extracts",
                "description": "Extract monthly baseline/comparison time series for validation.",
            },
            {
                "step": 3,
                "node": "correlation_dimension_scan",
                "description": "Score candidate dimensions and pick the strongest contributor.",
            },
            {
                "step": 4,
                "node": "correlation_drill_path",
                "description": "Iteratively drill into top contributors until stop rules are met.",
            },
            {
                "step": 5,
                "node": "correlation_artifacts",
                "description": "Persist summary, manifest, and aggregates for downstream consumers.",
            },
        ]

        scope = {
            "filters": _copy_jsonable({"filters": intent.get("filters", [])}).get("filters", []),
            "group_by": list(intent.get("group_by", [])),
            "metric_hint": intent.get("metric_hint"),
            "context_resolution": _copy_jsonable({"items": state.get("context_resolution", [])}).get("items", []),
        }

        try:
            # Extract yaml_path from intent if provided (for runtime YAML selection)
            yaml_path = intent.get("yaml_path")
            
            correlation_request = {
                "intent": intent,
                "conversation_id": state.get("conversation_id"),
                "query": state.get("question") or intent.get("raw_question"),
            }
            
            # Add yaml_path to request if provided
            if yaml_path:
                correlation_request["yaml_path"] = yaml_path
            
            result = correlation_runner(correlation_request)
            job_id = result.get("job_id") if isinstance(result, dict) else None
            if isinstance(result, dict) and "output" in result:
                result_payload = _copy_jsonable({"result": result.get("output")}).get("result", {})
            else:
                result_payload = _copy_jsonable({"result": result}).get("result", {})
            if job_id and "run_id" not in result_payload:
                result_payload["run_id"] = job_id
            warnings = [str(item) for item in _safe_list(result_payload.get("warnings")) if item is not None]

            status: AnalysisStatus = "success"
            findings = _build_correlation_findings(result_payload)
            artifacts = _build_correlation_artifacts(result_payload)
            next_actions = [
                "Review the executive summary artifact for the narrative output.",
                "Inspect the manifest to locate aggregate files for deeper drill-down.",
            ]
            metadata = _copy_jsonable(
                {
                    "executed_by": "correlation_agent",
                    "run_id": result_payload.get("run_id") or job_id,
                    "job_id": job_id,
                    "period_window": result_payload.get("period_window"),
                    "warnings": warnings,
                    "output_root": output_root,
                    "correlation_summary": result_payload,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in integration scenarios
            logger.exception(
                "Correlation agent failed for mode %s",
                intent.get("analysis_mode"),
                exc_info=exc,
            )
            # Raise exception instead of returning error in response body
            raise AgentExecutionError(
                f"Correlation analysis failed for mode '{intent.get('analysis_mode')}': {str(exc)}"
            ) from exc

        contract: AnalysisContract = {
            "contract_version": "1.0",
            "status": status,
            "analysis_mode": intent.get("analysis_mode"),
            "question": intent.get("raw_question", state["question"]),
            "scope": scope,
            "selected_configuration": _copy_jsonable(selected_configuration),
            "execution_plan": execution_plan,
            "findings": findings,
            "artifacts": artifacts,
            "next_actions": next_actions,
            "metadata": metadata,
        }
        _ensure_jsonable(contract)
        return contract

    return _handler


def _extract_correlation_summary(analysis_contract: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the full correlation payload from the analysis contract metadata."""
    metadata = _safe_dict(analysis_contract.get("metadata"))
    summary = metadata.get("correlation_summary")
    if isinstance(summary, dict):
        return summary
    return None


def _should_run_hypotheses(analysis_contract: Mapping[str, Any]) -> bool:
    """Return True when the correlation analysis succeeded and includes full payload."""
    if not analysis_contract:
        return False
    if analysis_contract.get("status") != "success":
        return False
    metadata = _safe_dict(analysis_contract.get("metadata"))
    if metadata.get("executed_by") != "correlation_agent":
        return False
    return isinstance(metadata.get("correlation_summary"), dict)


def _invoke_hypothesis_runner(
    runner: HypothesisRunner,
    *,
    intent: IntentOutput,
    correlation_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Safely invoke a hypothesis agent, returning an empty list on errors."""
    try:
        result = runner(
            intent=intent,
            correlation_summary=correlation_summary,
            executive_summary=correlation_summary.get("executive_summary"),
            executive_summary_path=correlation_summary.get("executive_summary_path"),
            period_window=correlation_summary.get("period_window"),
        )
    except Exception as exc:  # pragma: no cover - integration safety net
        logger.warning("Hypothesis runner failed.", exc_info=exc)
        return []
    return _safe_list(result)


def default_cost_change_handler(state: OrchestratorState) -> AnalysisContract:
    """Planning stub for cost_change_investigation_over_time_window."""
    intent = state["intent"]
    params = _safe_dict(intent.get("analysis_mode_parameters"))
    period = _safe_dict(params.get("period"))

    selected_drill_metric = _first_non_empty(params.get("drill_metric")) or intent.get("metric_hint")
    selected_window = _first_non_empty(period.get("rolling_window"))

    contract: AnalysisContract = {
        "contract_version": "1.0",
        "status": "planned",
        "analysis_mode": intent.get("analysis_mode"),
        "question": intent.get("raw_question", state["question"]),
        "scope": {
            "filters": _copy_jsonable({"filters": intent.get("filters", [])}).get("filters", []),
            "group_by": list(intent.get("group_by", [])),
            "metric_hint": intent.get("metric_hint"),
            "context_resolution": _copy_jsonable({"items": state.get("context_resolution", [])}).get("items", []),
        },
        "selected_configuration": {
            "drill_metric": selected_drill_metric,
            "explainer_metrics": _safe_list(params.get("explainer_metrics")),
            "period": {
                "rolling_time_dimension": period.get("rolling_time_dimension"),
                "rolling_window": [selected_window] if selected_window else _safe_list(period.get("rolling_window")),
                "start_time": period.get("start_time"),
                "end_time": period.get("end_time"),
            },
            "stop_rules": _copy_jsonable(_safe_dict(params.get("stop_rules"))),
        },
        "execution_plan": [
            {
                "step": 1,
                "node": "time_window_builder",
                "description": "Resolve prior/current comparable periods from the selected rolling window or explicit dates.",
            },
            {
                "step": 2,
                "node": "baseline_delta_sql",
                "description": "Compute current vs prior totals for the selected drill metric within the resolved scope.",
            },
            {
                "step": 3,
                "node": "dimension_ranking_sql",
                "description": "Rank candidate dimensions by absolute and percentage contribution to the metric delta.",
            },
            {
                "step": 4,
                "node": "iterative_drill_subgraph",
                "description": "Drill into top contributors until the configured stop rules are hit.",
            },
            {
                "step": 5,
                "node": "structured_findings_builder",
                "description": "Emit strict JSON findings for report, visual, and summary nodes.",
            },
        ],
        "findings": [],
        "artifacts": [],
        "next_actions": [
            "Attach the real drill-down / SQL executor to this mode handler.",
            "Keep the output schema stable when replacing this planning stub with the real executor.",
        ],
        "metadata": {
            "planned_by": "orchestrator",
            "is_stub": True,
            "mode_family": "cost_change_investigation",
        },
    }
    _ensure_jsonable(contract)
    return contract


def build_default_analysis_mode_handlers() -> Dict[str, AnalysisModeHandler]:
    """Default registry. Add new modes here or pass them in from the caller."""
    return {
        "cost_change_investigation_over_time_window": default_cost_change_handler,
    }


# =============================
# Shared downstream builders
# =============================

def validate_analysis_contract(contract: AnalysisContract) -> None:
    required_keys = {
        "contract_version",
        "status",
        "analysis_mode",
        "question",
        "scope",
        "selected_configuration",
        "execution_plan",
        "findings",
        "artifacts",
        "next_actions",
        "metadata",
    }
    missing = sorted(required_keys - set(contract.keys()))
    if missing:
        raise ValueError(f"Analysis contract is missing required keys: {missing}")
    _ensure_jsonable(contract)


def build_report_from_analysis(state: OrchestratorState) -> ReportContract:
    analysis = state["analysis_contract"]
    report: ReportContract = {
        "contract_version": "1.0",
        "status": analysis["status"],
        "analysis_mode": analysis.get("analysis_mode"),
        "sections": [
            {
                "id": "request",
                "title": "Request",
                "content": {
                    "question": analysis.get("question"),
                    "analysis_mode": analysis.get("analysis_mode"),
                },
            },
            {
                "id": "scope",
                "title": "Scope",
                "content": _copy_jsonable(_safe_dict(analysis.get("scope"))),
            },
            {
                "id": "selected_configuration",
                "title": "Selected Configuration",
                "content": _copy_jsonable(_safe_dict(analysis.get("selected_configuration"))),
            },
            {
                "id": "execution_plan",
                "title": "Execution Plan",
                "content": _copy_jsonable({"items": _safe_list(analysis.get("execution_plan"))}).get("items", []),
            },
            {
                "id": "findings",
                "title": "Findings",
                "content": _copy_jsonable({"items": _safe_list(analysis.get("findings"))}).get("items", []),
            },
        ],
        "metadata": {
            "generated_by": "build_report_contract",
            "is_stub": bool(_safe_dict(analysis.get("metadata")).get("is_stub", False)),
        },
    }
    _ensure_jsonable(report)
    return report


def build_visuals_from_analysis(state: OrchestratorState) -> VisualContract:
    analysis = state["analysis_contract"]
    selected_config = _safe_dict(analysis.get("selected_configuration"))
    period = _safe_dict(selected_config.get("period"))
    scope = _safe_dict(analysis.get("scope"))

    visual_contract: VisualContract = {
        "contract_version": "1.0",
        "status": analysis["status"],
        "analysis_mode": analysis.get("analysis_mode"),
        "recommendations": [
            {
                "chart_id": "metric_trend",
                "title": "Metric trend over time",
                "kind": "line",
                "x": period.get("rolling_time_dimension") or "time",
                "y": selected_config.get("drill_metric") or scope.get("metric_hint"),
                "reason": "Useful for comparing the prior and current time windows for the selected metric.",
            },
            {
                "chart_id": "top_contributors",
                "title": "Top contributors to change",
                "kind": "bar",
                "x": "dimension_value",
                "y": "abs_delta",
                "reason": "Useful after drill-down to show which segments explain the metric delta.",
            },
        ],
        "metadata": {
            "generated_by": "build_visual_contract",
            "group_by": list(scope.get("group_by", [])) if isinstance(scope.get("group_by"), list) else [],
        },
    }
    _ensure_jsonable(visual_contract)
    return visual_contract


def build_summary_from_analysis(state: OrchestratorState) -> SummaryContract:
    analysis = state["analysis_contract"]
    selected_config = _safe_dict(analysis.get("selected_configuration"))
    scope = _safe_dict(analysis.get("scope"))

    summary: SummaryContract = {
        "contract_version": "1.0",
        "status": analysis["status"],
        "headline": (
            f"Routed to {analysis.get('analysis_mode')} and built a strict execution contract."
        ),
        "bullets": [
            f"Question: {analysis.get('question')}",
            f"Metric: {selected_config.get('drill_metric') or scope.get('metric_hint')}",
            f"Filters resolved: {len(_safe_list(scope.get('filters')))}",
            f"Execution steps planned: {len(_safe_list(analysis.get('execution_plan')))}",
            "Current mode handler is a planning stub until the real executor is attached.",
        ],
        "metadata": {
            "generated_by": "build_summary_contract",
            "is_stub": bool(_safe_dict(analysis.get("metadata")).get("is_stub", False)),
        },
    }
    _ensure_jsonable(summary)
    return summary


def build_conversation_summary(state: OrchestratorState) -> str:
    recent_steps = state.get("step_summaries", [])[-4:]
    if not recent_steps:
        return f"Most recent question: {state['question']}"
    return " | ".join(recent_steps)


# =============================
# Graph factory
# =============================

def build_app(
    yaml_path: str,
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
    analysis_mode_handlers: Optional[Dict[str, AnalysisModeHandler]] = None,
    checkpointer: Optional[Any] = None,
    *,
    correlation_runner: Optional[CorrelationRunner] = None,
    snowflake_helper: Optional[Any] = None,
    snowflake_helper_builder: Optional[Callable[[], Any]] = None,
    correlation_output_root: str = "correlation_runs",
    enable_correlation_execution: bool = True,
):
    """
    Build the parent orchestrator graph.

    Notes:
    - The graph is compiled once and routed at runtime via a central router node.
    - Static assets stay outside graph state to keep persisted checkpoints small.
    - v1 supports exactly one analysis_mode per turn.
    - Provide a correlation_runner or Snowflake helper to execute the correlation agent
      for cost_change_investigation_over_time_window; otherwise the handler remains a stub.
    - Set enable_correlation_execution=False to keep the stub handler even if the
      correlation agent is available (useful for UI demos without Snowflake).
    - The resolved LLM (or llm_builder) is forwarded to the correlation agent for executive summaries.
    - When correlation analysis succeeds, policy and mandate hypothesis agents run in parallel.
    """
    logger.info("Building orchestrator app from YAML: %s", yaml_path)

    semantic_model = load_semantic_yaml(yaml_path)
    semantic_index = build_semantic_index(semantic_model)
    semantic_summary = summarize_semantic_model(semantic_model)

    def _unwrap_llm(result: Any) -> Any:
        # ``user_intent.build_llm`` returns ``(llm, ehap)`` for retry-utility support.
        # Downstream agents expect just the llm; passing the tuple through makes
        # ``self.llm`` an unserializable composite that crashes LangGraph's
        # msgpack checkpointer on EHAPBase.
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result

    llm_client = _unwrap_llm(llm) if llm is not None else None
    if llm_client is None and llm_builder is not None:
        raw_builder = llm_builder
        llm_builder = lambda: _unwrap_llm(raw_builder())
        try:
            llm_client = llm_builder()
            logger.info("LLM built using provided builder")
        except Exception as exc:
            logger.warning("LLM builder failed", exc_info=exc)
            llm_client = None

    if llm_client is None:
        try:
            llm_client = _unwrap_llm(build_llm())
            logger.info("LLM built using default build_llm()")
        except Exception as exc:  # pragma: no cover - depends on local auth/env
            logger.warning("LLM unavailable for orchestrator; using rule-based fallback inside user_intent.", exc_info=exc)
            llm_client = None

    # When a builder is available, hand sub-agents the builder *only* — passing
    # ``llm=llm_client`` makes ``AgentBase.__init__`` skip EHAP wiring entirely
    # (it short-circuits to ``self.ehap = None``), which silently disables the
    # token-refresh machinery in ``ehap_retry.structured_llm_invoke``. With the
    # builder alone, each sub-agent constructs its own ``EHAPBase`` and the
    # 401/proactive-expiry retry path stays active across long-lived sessions
    # (e.g. cached Streamlit orchestrators).
    forwarded_sub_agent_llm = None if llm_builder is not None else llm_client

    handlers = build_default_analysis_mode_handlers()
    resolved_correlation_runner = correlation_runner
    if not enable_correlation_execution:
        if resolved_correlation_runner is not None:
            logger.info("Correlation execution disabled; ignoring provided runner.")
        resolved_correlation_runner = None
    elif resolved_correlation_runner is None and build_correlation_app is not None:
        if snowflake_helper is None and snowflake_helper_builder is None:
            logger.info(
                "No Snowflake helper provided; correlation agent will attempt credential-based initialization."
            )
        resolved_correlation_runner = build_correlation_app(
            yaml_path=yaml_path,
            snowflake_helper=snowflake_helper,
            snowflake_helper_builder=snowflake_helper_builder,
            output_root=correlation_output_root,
            llm=forwarded_sub_agent_llm,
            llm_builder=llm_builder,
        )
    elif resolved_correlation_runner is None and build_correlation_app is None:
        logger.warning(
            "Correlation agent requested but unavailable; install correlation_agent dependencies to enable execution."
        )

    if resolved_correlation_runner is not None:
        handlers["cost_change_investigation_over_time_window"] = build_correlation_analysis_handler(
            resolved_correlation_runner,
            output_root=correlation_output_root,
        )
    if analysis_mode_handlers:
        handlers.update(analysis_mode_handlers)

    policy_hypothesis_runner: Optional[HypothesisRunner] = None
    if build_policy_hypothesis_app is not None:
        try:
            policy_hypothesis_runner = build_policy_hypothesis_app(
                llm=forwarded_sub_agent_llm,
                llm_builder=llm_builder,
            )
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Policy hypothesis agent unavailable.", exc_info=exc)

    mandate_hypothesis_runner: Optional[HypothesisRunner] = None
    if build_mandate_hypothesis_app is not None:
        try:
            mandate_hypothesis_runner = build_mandate_hypothesis_app(
                llm=forwarded_sub_agent_llm,
                llm_builder=llm_builder,
            )
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Mandate hypothesis agent unavailable.", exc_info=exc)

    pattern_runner: Optional[Any] = None
    if build_pattern_app is not None:
        try:
            pattern_runner = build_pattern_app(llm=forwarded_sub_agent_llm, llm_builder=llm_builder)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Pattern agent unavailable.", exc_info=exc)

    reimbursement_runner: Optional[Any] = None
    if build_reimbursement_app is not None:
        try:
            reimbursement_runner = build_reimbursement_app(
                snowflake_helper=snowflake_helper,
                snowflake_helper_builder=snowflake_helper_builder,
                llm=forwarded_sub_agent_llm,
                llm_builder=llm_builder,
            )
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Reimbursement agent unavailable.", exc_info=exc)

    recommendation_runner: Optional[Any] = None
    if build_recommendation_dtr_app is not None:
        try:
            recommendation_runner = build_recommendation_dtr_app(
                llm=forwarded_sub_agent_llm,
                llm_builder=llm_builder,
            )
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Recommendation agent unavailable.", exc_info=exc)

    registered_analysis_nodes: Dict[str, str] = {
        mode_name: analysis_node_name(mode_name)
        for mode_name in handlers
    }

    def prepare_request(state: OrchestratorState) -> Dict[str, Any]:
        question = str(state.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")

        soft_context_filters, analysis_context = split_soft_context(state.get("context"))

        # Pre-built intent path: if the UI handed us a fully-formed intent
        # (via analysis_overrides.prebuilt_intent or top-level prebuilt_intent),
        # honor it and skip the LLM intent-detection step entirely.
        prebuilt_intent_payload = analysis_context.pop("prebuilt_intent", None)
        if not isinstance(prebuilt_intent_payload, dict):
            prebuilt_intent_payload = None

        # Top-level state fallback — if the caller wrote ``prebuilt_intent``
        # directly into initial_state (e.g. from ``run(prebuilt_intent=...)``)
        # the dict travels here as ``state["prebuilt_intent"]`` regardless of
        # the context envelope. Honor it the same way.
        if prebuilt_intent_payload is None:
            top_level = state.get("prebuilt_intent")
            if isinstance(top_level, dict):
                prebuilt_intent_payload = top_level

        if prebuilt_intent_payload is not None:
            initial_intent: IntentOutput = {
                "metric_hint": prebuilt_intent_payload.get("metric_hint"),
                "group_by": list(prebuilt_intent_payload.get("group_by", []) or []),
                "filters": [
                    dict(f) for f in prebuilt_intent_payload.get("filters", []) or []
                    if isinstance(f, dict)
                ],
                "analysis_mode": prebuilt_intent_payload.get("analysis_mode"),
                "analysis_mode_parameters": _safe_dict(prebuilt_intent_payload.get("analysis_mode_parameters")),
                "raw_question": str(prebuilt_intent_payload.get("raw_question") or question),
                "validation_warnings": list(prebuilt_intent_payload.get("validation_warnings", []) or []),
            }
            bypass = True
            step_note = (
                "Prepared new turn with pre-built intent — bypassing intent-detection LLM. "
                f"Question: {question}"
            )
            logger.info(
                "prepare_request: BYPASS intent-detection — analysis_mode=%s filters=%d period=%s",
                initial_intent["analysis_mode"],
                len(initial_intent["filters"]),
                _safe_dict(initial_intent["analysis_mode_parameters"]).get("period"),
            )
        else:
            initial_intent = _build_empty_intent(question)
            bypass = False
            step_note = f"Prepared new turn for question: {question}"
            logger.info("prepare_request: no prebuilt intent — intent-detection LLM will run.")

        conversation_id = state.get("conversation_id") or analysis_context.pop("conversation_id", None)
        live_filter_values = analysis_context.pop("live_filter_values", None)
        if not isinstance(live_filter_values, dict):
            live_filter_values = None

        return {
            "soft_context_filters": soft_context_filters,
            "analysis_context": analysis_context,
            "conversation_id": conversation_id,
            "bypass_intent_detection": bypass,
            "live_filter_values": live_filter_values,
            "intent": initial_intent,
            "context_resolution": [],
            "clarification_request": {},
            "analysis_contract": {},
            "report_contract": {},
            "visual_contract": {},
            "summary_contract": {},
            "hypotheses": {"policy": [], "mandate": [], "metadata": {}},
            "business_patterns": [],
            "pattern_summary": {},
            "reimbursement_by_pattern": {},
            "recommendations": [],
            "research": {},
            "final_output": {},
            "route": "",
            "errors": [],
            "last_completed_stage": "prepare_request",
            "step_summaries": [step_note],
        }

    def run_intent_detection(state: OrchestratorState) -> Dict[str, Any]:
        live_filter_values = state.get("live_filter_values") if isinstance(state.get("live_filter_values"), dict) else None

        # Bypass path — UI handed us a fully-formed intent in prepare_request.
        # Skip the LLM intent-resolution call and the soft-merge step (UI
        # filters are authoritative here).
        if state.get("bypass_intent_detection"):
            existing = state.get("intent") or _build_empty_intent(state["question"])
            existing["validation_warnings"] = validate_intent_output(
                existing, semantic_model, semantic_index, live_filter_values
            )
            logger.info(
                "run_intent_detection: BYPASS active — skipping LLM. analysis_mode=%s filters=%d live_values=%s",
                existing.get("analysis_mode"),
                len(existing.get("filters", [])),
                bool(live_filter_values),
            )
            return {
                "intent": existing,
                "context_resolution": [],
                "last_completed_stage": "run_intent_detection",
                "step_summaries": [
                    (
                        "Intent bypass: using pre-built intent. "
                        f"analysis_mode={existing.get('analysis_mode')}, "
                        f"filters={len(existing.get('filters', []))}."
                    )
                ],
            }

        child_state = {
            "question": state["question"],
            # Only pass analysis-level hints here.
            # UI filters are merged later with soft precedence rules.
            "context": state.get("analysis_context") or None,
            "semantic_model": semantic_model,
            "semantic_index": semantic_index,
            "semantic_summary": semantic_summary,
            "llm": llm_client,
        }

        result = identify_intent_and_filters(child_state)["result"]
        merged_filters, context_resolution = merge_soft_context_filters(
            extracted_filters=result.get("filters", []),
            soft_context_filters=state.get("soft_context_filters"),
        )
        result["filters"] = merged_filters
        result["validation_warnings"] = validate_intent_output(
            result, semantic_model, semantic_index, live_filter_values
        )

        return {
            "intent": result,
            "context_resolution": context_resolution,
            "last_completed_stage": "run_intent_detection",
            "step_summaries": [
                (
                    "Intent resolved: "
                    f"analysis_mode={result.get('analysis_mode')}, "
                    f"filters={len(result.get('filters', []))}, "
                    f"group_by={len(result.get('group_by', []))}."
                )
            ],
        }

    def decide_next_step(state: OrchestratorState) -> Dict[str, Any]:
        intent = state["intent"]
        clarification_request = build_clarification_request(intent)

        if clarification_request is not None:
            route = "clarification"
            step_note = "Routing to clarification because the current request is still ambiguous or invalid."
        else:
            mode_name = str(intent.get("analysis_mode"))
            route = registered_analysis_nodes.get(mode_name, "analysis__generic")
            step_note = f"Routing to {route}."

        return {
            "route": route,
            "clarification_request": clarification_request or {},
            "last_completed_stage": "decide_next_step",
            "step_summaries": [step_note],
        }

    def clarification_node(state: OrchestratorState) -> Dict[str, Any]:
        clarification_request = _safe_dict(state.get("clarification_request"))
        final_output: FinalOutput = {
            "contract_version": "1.0",
            "status": "needs_clarification",
            "question": state["question"],
            "context_resolution": state.get("context_resolution", []),
            "intent": state.get("intent"),
            "clarification_request": clarification_request,
            "analysis": None,
            "report": None,
            "visuals": None,
            "summary": {
                "contract_version": "1.0",
                "status": "needs_clarification",
                "headline": "Need clarification before executing analysis.",
                "bullets": clarification_request.get("blocking_issues", []),
                "metadata": {"generated_by": "clarification_node"},
            },
            "hypotheses": None,
            "recent_step_summaries": state.get("step_summaries", []),
            "conversation_summary": build_conversation_summary(state),
            "last_completed_stage": "clarification",
        }
        return {
            "final_output": final_output,
            "conversation_summary": final_output["conversation_summary"],
            "last_completed_stage": "clarification",
            "step_summaries": ["Built clarification response contract."],
        }

    def build_analysis_mode_node(mode_name: str, handler: AnalysisModeHandler):
        def _node(state: OrchestratorState) -> Dict[str, Any]:
            contract = handler(state)
            validate_analysis_contract(contract)
            return {
                "analysis_contract": contract,
                "last_completed_stage": f"analysis::{mode_name}",
                "step_summaries": [
                    f"Built analysis contract for mode: {mode_name}."
                ],
            }

        _node.__name__ = f"run_{analysis_node_name(mode_name)}"
        return _node

    def generic_analysis_node(state: OrchestratorState) -> Dict[str, Any]:
        contract = default_generic_analysis_handler(state)
        validate_analysis_contract(contract)
        return {
            "analysis_contract": contract,
            "last_completed_stage": "analysis::generic",
            "step_summaries": [
                "Built generic analysis contract because no dedicated handler was registered for the selected mode."
            ],
        }

    def build_report_contract_node(state: OrchestratorState) -> Dict[str, Any]:
        report_contract = build_report_from_analysis(state)
        return {
            "report_contract": report_contract,
            "last_completed_stage": "build_report_contract",
            "step_summaries": ["Built report contract from analysis contract."],
        }

    def build_visual_contract_node(state: OrchestratorState) -> Dict[str, Any]:
        visual_contract = build_visuals_from_analysis(state)
        return {
            "visual_contract": visual_contract,
            "last_completed_stage": "build_visual_contract",
            "step_summaries": ["Built visual contract from analysis contract."],
        }

    def build_summary_contract_node(state: OrchestratorState) -> Dict[str, Any]:
        summary_contract = build_summary_from_analysis(state)
        return {
            "summary_contract": summary_contract,
            "last_completed_stage": "build_summary_contract",
            "step_summaries": ["Built summary contract from analysis contract."],
        }

    def run_hypothesis_agents(state: OrchestratorState) -> Dict[str, Any]:
        analysis_contract = _safe_dict(state.get("analysis_contract"))
        intent = state.get("intent") or _build_empty_intent(state.get("question", ""))

        hypotheses: HypothesisOutput = {"policy": [], "mandate": [], "metadata": {}}
        if not _should_run_hypotheses(analysis_contract):
            return {
                "hypotheses": hypotheses,
                "last_completed_stage": "hypotheses::skipped",
                "step_summaries": ["Skipped hypothesis agents (no successful correlation output)."],
            }

        correlation_summary = _extract_correlation_summary(analysis_contract)
        if not correlation_summary:
            return {
                "hypotheses": hypotheses,
                "last_completed_stage": "hypotheses::skipped",
                "step_summaries": ["Skipped hypothesis agents (missing correlation summary payload)."],
            }

        runners: List[Tuple[str, HypothesisRunner]] = []
        if policy_hypothesis_runner is not None:
            runners.append(("policy", policy_hypothesis_runner))
        if mandate_hypothesis_runner is not None:
            runners.append(("mandate", mandate_hypothesis_runner))

        if not runners:
            return {
                "hypotheses": hypotheses,
                "last_completed_stage": "hypotheses::skipped",
                "step_summaries": ["Skipped hypothesis agents (no runners available)."],
            }

        results: Dict[str, List[Dict[str, Any]]] = {"policy": [], "mandate": []}
        with ThreadPoolExecutor(max_workers=len(runners)) as executor:
            futures = {
                executor.submit(
                    _invoke_hypothesis_runner,
                    runner,
                    intent=intent,
                    correlation_summary=correlation_summary,
                ): label
                for label, runner in runners
            }
            for future, label in futures.items():
                try:
                    results[label] = future.result()
                except Exception as exc:  # pragma: no cover - safety net
                    logger.warning("Hypothesis agent failed for %s.", label, exc_info=exc)
                    results[label] = []

        hypotheses = {
            "policy": results.get("policy", []),
            "mandate": results.get("mandate", []),
            "metadata": {
                "executed_by": "hypothesis_agents",
                "correlation_run_id": correlation_summary.get("run_id"),
            },
        }

        step_notes = [
            f"Generated {len(hypotheses['policy'])} policy hypotheses.",
            f"Generated {len(hypotheses['mandate'])} mandate hypotheses.",
        ]
        return {
            "hypotheses": hypotheses,
            "last_completed_stage": "hypotheses",
            "step_summaries": step_notes,
        }

    def run_pattern_agent(state: OrchestratorState) -> Dict[str, Any]:
        analysis_contract = _safe_dict(state.get("analysis_contract"))
        if pattern_runner is None:
            return {
                "business_patterns": [],
                "pattern_summary": {},
                "last_completed_stage": "pattern_agent::skipped",
                "step_summaries": ["Skipped pattern agent (runner unavailable)."],
            }
        if not _should_run_hypotheses(analysis_contract):
            return {
                "business_patterns": [],
                "pattern_summary": {},
                "last_completed_stage": "pattern_agent::skipped",
                "step_summaries": ["Skipped pattern agent (correlation analysis incomplete)."],
            }

        correlation_summary = _extract_correlation_summary(analysis_contract) or {}
        conversation_id = state.get("conversation_id") or ""
        question = state.get("question") or ""
        analysis_context = _safe_dict(state.get("analysis_context"))
        semantic_roles = analysis_context.get("semantic_roles")
        pattern_context: Dict[str, Any] = {"correlation_results": correlation_summary}
        if isinstance(semantic_roles, dict) and semantic_roles:
            pattern_context["semantic_roles"] = semantic_roles

        try:
            pattern_result = pattern_runner(
                conversation_id=conversation_id,
                query=question,
                context=pattern_context,
                semantic_config_path=yaml_path,
            )
        except Exception as exc:  # pragma: no cover - exercised in integration scenarios
            logger.exception("Pattern agent failed", exc_info=exc)
            return {
                "business_patterns": [],
                "pattern_summary": {},
                "last_completed_stage": "pattern_agent::error",
                "step_summaries": [f"Pattern agent failed: {exc}"],
                "errors": [f"pattern_agent: {exc}"],
            }

        pattern_output = _safe_dict(
            pattern_result.get("output") if isinstance(pattern_result, dict) else None
        ) or _safe_dict(pattern_result)
        business_patterns = list(pattern_output.get("business_patterns", []) or [])
        pattern_summary = {
            "executive_summary": pattern_output.get("executive_summary"),
            "quality_checks": pattern_output.get("quality_checks"),
            "cards": pattern_output.get("cards", []),
            "groups": pattern_output.get("groups", []),
            "stats": pattern_output.get("stats", {}),
        }

        return {
            "business_patterns": business_patterns,
            "pattern_summary": pattern_summary,
            "last_completed_stage": "pattern_agent",
            "step_summaries": [
                f"Pattern agent produced {len(business_patterns)} business patterns."
            ],
        }

    def run_reimbursement_fanout(state: OrchestratorState) -> Dict[str, Any]:
        patterns = list(state.get("business_patterns") or [])
        if reimbursement_runner is None or not patterns:
            return {
                "reimbursement_by_pattern": {},
                "last_completed_stage": "reimbursement::skipped",
                "step_summaries": [
                    "Skipped reimbursement fan-out ("
                    + ("runner unavailable" if reimbursement_runner is None else "no patterns")
                    + ")."
                ],
            }

        conversation_id = state.get("conversation_id") or ""
        question = state.get("question") or ""
        pattern_summary = _safe_dict(state.get("pattern_summary"))
        cards = list(pattern_summary.get("cards") or [])
        analysis_context = _safe_dict(state.get("analysis_context"))
        semantic_roles = analysis_context.get("semantic_roles")
        semantic_roles_dict: Dict[str, str] = (
            {str(k): str(v) for k, v in semantic_roles.items() if k and v}
            if isinstance(semantic_roles, dict)
            else {}
        )

        def _process(pattern: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
            rank = pattern.get("pattern_rank")
            reimbursement_context: Dict[str, Any] = {"pattern": pattern, "cards": cards}
            if semantic_roles_dict:
                reimbursement_context["semantic_roles"] = semantic_roles_dict
            try:
                result = reimbursement_runner.execute(
                    conversation_id=conversation_id,
                    query=question,
                    context=reimbursement_context,
                )
                return rank, result if isinstance(result, dict) else _safe_dict(result)
            except Exception as exc:  # pragma: no cover - integration path
                logger.exception("Reimbursement agent failed for pattern %s", rank, exc_info=exc)
                return rank, {"error": str(exc), "pattern_rank": rank}

        results: Dict[str, Any] = {}
        worker_count = max(1, min(8, len(patterns)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process, pattern) for pattern in patterns]
            for future in futures:
                rank, payload = future.result()
                if payload is None:
                    continue
                results[str(rank)] = payload

        return {
            "reimbursement_by_pattern": results,
            "last_completed_stage": "reimbursement",
            "step_summaries": [
                f"Reimbursement fan-out completed for {len(results)} pattern(s)."
            ],
        }

    def run_recommendation_agent(state: OrchestratorState) -> Dict[str, Any]:
        patterns = list(state.get("business_patterns") or [])
        reimbursement_by_pattern = _safe_dict(state.get("reimbursement_by_pattern"))
        pattern_summary = _safe_dict(state.get("pattern_summary"))
        analysis_contract = _safe_dict(state.get("analysis_contract"))
        correlation_summary = _extract_correlation_summary(analysis_contract) or {}

        def _build_research_bundle(
            recommendations: List[Dict[str, Any]],
            extra_metadata: Optional[Dict[str, Any]] = None,
        ) -> ResearchOutput:
            metadata: Dict[str, Any] = {
                "executed_by": "research_chain",
                "correlation_run_id": correlation_summary.get("run_id"),
                "pattern_count": len(patterns),
                "reimbursement_count": len(reimbursement_by_pattern),
                "recommendation_count": len(recommendations),
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            return {
                "business_patterns": patterns,
                "pattern_summary": pattern_summary,
                "reimbursement_by_pattern": reimbursement_by_pattern,
                "recommendations": recommendations,
                "metadata": metadata,
            }

        if recommendation_runner is None or not patterns:
            skip_reason = "runner unavailable" if recommendation_runner is None else "no patterns"
            return {
                "recommendations": [],
                "research": _build_research_bundle(
                    [],
                    {"recommendation_status": "skipped", "skip_reason": skip_reason},
                ),
                "last_completed_stage": "recommendation::skipped",
                "step_summaries": [f"Skipped recommendation agent ({skip_reason})."],
            }

        combined: List[Dict[str, Any]] = []
        for pattern in patterns:
            rank = pattern.get("pattern_rank")
            entry = dict(pattern)
            entry["reimbursement"] = reimbursement_by_pattern.get(str(rank))
            combined.append(entry)

        try:
            result = recommendation_runner.execute(patterns_data=combined)
        except Exception as exc:  # pragma: no cover - integration path
            logger.exception("Recommendation agent failed", exc_info=exc)
            return {
                "recommendations": [],
                "research": _build_research_bundle(
                    [],
                    {"recommendation_status": "error", "error": str(exc)},
                ),
                "last_completed_stage": "recommendation::error",
                "step_summaries": [f"Recommendation agent failed: {exc}"],
                "errors": [f"recommendation_agent: {exc}"],
            }

        # Extract recommendations with backward compatibility
        # New format: recommendations are in output.recommendations
        # Legacy format: recommendations are at top level
        if "output" in result and isinstance(result.get("output"), dict):
            recommendations = list(_safe_dict(result["output"]).get("recommendations", []))
        else:
            recommendations = list(_safe_dict(result).get("recommendations", []))

        research_bundle = _build_research_bundle(recommendations)

        return {
            "recommendations": recommendations,
            "research": research_bundle,
            "last_completed_stage": "recommendation",
            "step_summaries": [
                f"Generated {len(recommendations)} recommendations."
            ],
        }

    def finalize_node(state: OrchestratorState) -> Dict[str, Any]:
        analysis_contract = state.get("analysis_contract") or None
        report_contract = state.get("report_contract") or None
        visual_contract = state.get("visual_contract") or None
        summary_contract = state.get("summary_contract") or None
        hypotheses = state.get("hypotheses") or None
        research = state.get("research") or None

        status: AnalysisStatus
        if state.get("clarification_request"):
            status = "needs_clarification"
        elif analysis_contract:
            status = analysis_contract.get("status", "planned")
        else:
            status = "error"

        conversation_summary = build_conversation_summary(state)
        final_output: FinalOutput = {
            "contract_version": "1.0",
            "status": status,
            "question": state["question"],
            "context_resolution": state.get("context_resolution", []),
            "intent": state.get("intent"),
            "clarification_request": state.get("clarification_request") or None,
            "analysis": analysis_contract,
            "report": report_contract,
            "visuals": visual_contract,
            "summary": summary_contract,
            "hypotheses": hypotheses,
            "research": research,
            "recent_step_summaries": state.get("step_summaries", []),
            "conversation_summary": conversation_summary,
            "last_completed_stage": "finalize",
        }
        _ensure_jsonable(final_output)
        return {
            "final_output": final_output,
            "conversation_summary": conversation_summary,
            "last_completed_stage": "finalize",
            "step_summaries": ["Finalized orchestrator output."],
        }

    def route_from_decision(state: OrchestratorState) -> str:
        return state.get("route", "clarification")

    graph = StateGraph(OrchestratorState)
    graph.add_node("prepare_request", prepare_request)
    graph.add_node("run_intent_detection", run_intent_detection)
    graph.add_node("decide_next_step", decide_next_step)
    graph.add_node("clarification", clarification_node)
    graph.add_node("analysis__generic", generic_analysis_node)
    graph.add_node("build_report_contract", build_report_contract_node)
    graph.add_node("build_visual_contract", build_visual_contract_node)
    graph.add_node("build_summary_contract", build_summary_contract_node)
    graph.add_node("run_hypothesis_agents", run_hypothesis_agents)
    graph.add_node("run_pattern_agent", run_pattern_agent)
    graph.add_node("run_reimbursement_fanout", run_reimbursement_fanout)
    graph.add_node("run_recommendation_agent", run_recommendation_agent)
    graph.add_node("finalize", finalize_node)

    for mode_name, handler in handlers.items():
        graph.add_node(analysis_node_name(mode_name), build_analysis_mode_node(mode_name, handler))

    graph.add_edge(START, "prepare_request")
    graph.add_edge("prepare_request", "run_intent_detection")
    graph.add_edge("run_intent_detection", "decide_next_step")
    graph.add_conditional_edges("decide_next_step", route_from_decision)

    graph.add_edge("clarification", "finalize")
    graph.add_edge("analysis__generic", "build_report_contract")

    # After the correlation analysis node, run the research chain
    # (pattern → reimbursement fan-out → recommendation) instead of the
    # legacy hypothesis fan-out. The hypothesis node is kept registered
    # for tests that import it directly but is no longer wired in.
    for mode_name in handlers:
        node_name = analysis_node_name(mode_name)
        if mode_name == "cost_change_investigation_over_time_window":
            graph.add_edge(node_name, "run_pattern_agent")
        else:
            graph.add_edge(node_name, "build_report_contract")

    graph.add_edge("run_pattern_agent", "run_reimbursement_fanout")
    graph.add_edge("run_reimbursement_fanout", "run_recommendation_agent")
    graph.add_edge("run_recommendation_agent", "build_report_contract")
    graph.add_edge("run_hypothesis_agents", "build_report_contract")

    graph.add_edge("build_report_contract", "build_visual_contract")
    graph.add_edge("build_visual_contract", "build_summary_contract")
    graph.add_edge("build_summary_contract", "finalize")
    graph.add_edge("finalize", END)

    if checkpointer is None and InMemorySaver is not None:
        checkpointer = InMemorySaver()

    app = graph.compile(checkpointer=checkpointer)

    def run(
        question: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prebuilt_intent: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FinalOutput:
        """
        Execute one turn through the parent orchestrator.

        Example:
            result = run(
                question="Find what changes in HCPCS 99292? Use total paid amount as the metric and filter for Texas.",
                context={"hcc_medium": "IP BH", "lob_description": "Commercial", "period": "Rolling 3"},
                thread_id="demo-thread-1",
                conversation_id="tutorial-IP_AUTH-Commercial-202604-R3-202601",
            )
        """
        invoke_config = copy.deepcopy(config) if config else {}
        if thread_id:
            invoke_config.setdefault("configurable", {})["thread_id"] = thread_id
        elif checkpointer is not None:
            invoke_config.setdefault("configurable", {}).setdefault("thread_id", "orchestrator-default-thread")

        initial_state: OrchestratorState = {
            "question": question,
            "context": context or {},
        }
        if conversation_id:
            initial_state["conversation_id"] = conversation_id
        if isinstance(prebuilt_intent, dict):
            initial_state["prebuilt_intent"] = prebuilt_intent
        out = app.invoke(initial_state, config=invoke_config)
        return out["final_output"]

    # Expose internals for testing/debugging without changing the callable pattern.
    run.graph = app  # type: ignore[attr-defined]
    run.analysis_mode_handlers = handlers  # type: ignore[attr-defined]
    run.semantic_model = semantic_model  # type: ignore[attr-defined]
    return run


def build_orchestrator(*args: Any, **kwargs: Any):
    """Alias kept for readability at call sites."""
    return build_app(*args, **kwargs)


def build_snowflake_helper_from_env() -> "SnowparkHelper":
    """Build a SnowparkHelper from environment variables for local runs."""
    from dotenv import load_dotenv
    load_dotenv(".env")
    if SnowparkHelper is None:
        raise RuntimeError(
            "SnowparkHelper is unavailable; install deep_research_utils to enable Snowflake access."
        )

    # Use CredentialProvider
    creds = CredentialProvider.get_instance()
    snowflake_creds = creds.get_snowflake_credentials()
    
    # Build helper with auto-detected connection type
    helper = SnowparkHelper(
        batch_size=10000,        # Larger batches
        max_workers=6,           # Parallel processing
        enable_metrics=True,     # Performance tracking
        connection_pool_size=4,  # Connection pooling
        **snowflake_creds
    )
    return helper


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     yaml_path = "/Users/AH45807/project/deep-research/configs/ecap_semantic_view_with_samples.yaml"
#
#     helper = build_snowflake_helper_from_env()
#     try:
#         run = build_app(
#             yaml_path=yaml_path,
#             snowflake_helper=helper,
#             correlation_output_root="correlation_runs",
#         )
#
#         question = "Find what changed in HCPCS H0010?"
#         context = {
#             "period": "Rolling 3",
#             "drill_metric": ["claims_expense.total_paid"]
#         }
#
#         result = run(
#             question=question,
#             context=context,
#             thread_id="demo-thread-1",
#         )
#
#         from pprint import pprint
#
#         pprint(result)
#     finally:
#         helper.close()
# # python packages/core/src/deep_research_core/orchestrator.py
