from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, TypedDict

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from deep_research_core.base_agent import (
    AgentBase,
    AgentConfigurationError,
    AgentExecutionError,
    CredentialProvider,
)

from deep_research_utils.app_constant import AppConstants

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from deep_research_utils.snowflake_helper import SnowparkHelper
except ImportError:  # pragma: no cover - allows local editing without dependency wiring
    SnowparkHelper = object  # type: ignore[misc,assignment]

try:
    from deep_research_utils.subset_table_manager import SubsetTableManager
except ImportError:  # pragma: no cover
    SubsetTableManager = None  # type: ignore[misc,assignment]


def _load_app_version() -> str:
    """Load application version from package metadata."""
    try:
        from deep_research_agents import __version__ as package_version

        return str(package_version)
    except ImportError:
        return "unknown"


APP_VERSION = _load_app_version()


FilterSource = Literal["dimension_match", "named_filter"]


class FilterCondition(TypedDict):
    field: str
    operator: str
    value: Any
    source: FilterSource


class AnalysisModeDefinition(TypedDict, total=False):
    name: str
    aliases: List[str]
    description: str
    drill_metric: List[str]
    explainer_metrics: List[str]
    interaction_matrix: Dict[str, Any]
    period: Dict[str, Any]
    drill_dimensions: List[str]
    exclude_if_filtered: bool
    stop_rules: Dict[str, Any]
    save_parquet: bool
    disable_summary_creation: bool
    generate_recommendations: bool


class IntentPayload(TypedDict, total=False):
    analysis_mode: Optional[str]
    analysis_mode_parameters: Optional[AnalysisModeDefinition]
    filters: List[FilterCondition]
    group_by: List[str]
    metric_hint: Optional[str]
    raw_question: str
    validation_warnings: List[str]
    save_parquet: bool
    disable_summary_creation: bool
    generate_recommendations: bool


class CorrelationContext(TypedDict, total=False):
    analysis_mode_parameters: AnalysisModeDefinition
    filters: List[FilterCondition]
    metric_hint: Optional[str]
    analysis_mode: Optional[str]
    group_by: List[str]
    validation_warnings: List[str]
    save_parquet: bool
    disable_summary_creation: bool
    generate_recommendations: bool


class CorrelationRequest(TypedDict, total=False):
    conversation_id: str
    query: str
    context: CorrelationContext
    save_parquet: bool
    disable_summary_creation: bool
    generate_recommendations: bool


class PeriodWindow(TypedDict):
    """Normalized time window payload for correlation comparisons.
    
    Time values (start_time, end_time, baseline_start_time, baseline_end_time) can be:
    - int (for YYYYMM format like 202501)
    - str (for ISO date strings like '2025-01-01')
    - datetime objects
    
    The data_type field indicates the semantic data type from the YAML config.
    """

    time_dimension: str
    start_time: Any
    end_time: Any
    baseline_start_time: Any
    baseline_end_time: Any
    comparison_strategy: str
    baseline_months: List[Any]
    comparison_months: List[Any]
    data_type: Optional[str]


class FieldDefinition(TypedDict, total=False):
    name: str
    table_name: str
    expr: str
    data_type: str
    description: str
    label: str
    kind: str
    synonyms: List[str]


class MetricDefinition(TypedDict, total=False):
    name: str
    expr: str
    description: str
    dependency_tables: List[str]
    primary_table: Optional[str]


class RelationshipDefinition(TypedDict, total=False):
    name: str
    left_table: str
    right_table: str
    relationship_columns: List[Dict[str, str]]


class TableDefinition(TypedDict, total=False):
    name: str
    description: str
    database: str
    schema: str
    table: str
    qualified_name: str


class SemanticCatalog(TypedDict):
    tables: Dict[str, TableDefinition]
    fields_by_name: Dict[str, List[FieldDefinition]]
    table_fields: Dict[str, Dict[str, FieldDefinition]]
    metrics_by_name: Dict[str, MetricDefinition]
    relationships: List[RelationshipDefinition]


class CandidateSelection(TypedDict, total=False):
    dimension: str
    dimension_label: str
    folder_name: str
    top_segments: List["SegmentSummary"]
    bottom_segments: List["SegmentSummary"]
    top_aligned_share: float
    top_aligned_delta: float
    top_total_delta: float
    query_path: str
    aggregate_paths: Dict[str, str]


class SegmentSummary(TypedDict, total=False):
    value: str
    baseline_value: float
    comparison_value: float
    delta_value: float
    contribution_pct_total: float
    contribution_pct_parent: float
    aligned_delta: float
    aligned_contribution_pct_total: float
    aligned_contribution_pct_parent: float
    aligned_contribution_pct_of_aligned_delta: float
    raw_row_count_baseline: float
    raw_row_count_comparison: float
    opposing_share: float


class PathNodeSummary(TypedDict, total=False):
    level: int
    dimension: str
    dimension_label: str
    folder_name: str
    top_segments: List[SegmentSummary]
    bottom_segments: List[SegmentSummary]
    parent_context: str


class CorrelationRunResult(TypedDict, total=False):
    run_id: str
    run_dir: str
    manifest_path: str
    executive_summary_path: str
    period_window: PeriodWindow
    root_metric: str
    baseline_value: float
    comparison_value: float
    delta_value: float
    delta_pct: Optional[float]
    drill_path: List[PathNodeSummary]
    narrative_summary: str
    executive_summary: str
    executive_summary_source: str
    interaction_matrix: Dict[str, Any]
    interaction_summary: Dict[str, Any]
    recommended_action: List[Dict[str, Any]]
    llm_tokens: Dict[str, Any]
    warnings: List[str]


class CorrelationAgentResponse(TypedDict, total=False):
    job_id: str
    conversation_id: str
    agent: str
    status: str
    recommended_action: List[Dict[str, Any]]
    visual_component: Dict[str, Any]
    output: CorrelationRunResult
    explanation: Dict[str, Any]
    validation: Dict[str, Any]
    tokens: Dict[str, int]
    execution: Dict[str, Any]


class GraphState(TypedDict, total=False):
    conversation_id: str
    query: str
    job_id: str
    start_time: str
    yaml_path: str
    intent: IntentPayload
    semantic_model: Dict[str, Any]
    catalog: SemanticCatalog
    snowflake_helper: Optional[SnowparkHelper]
    output_root: str
    llm: Optional[Any]
    input_tokens: int
    output_tokens: int
    llm_tokens: Dict[str, Any]
    result: CorrelationRunResult


class PeriodWindowSchema(BaseModel):
    """Schema for correlation period window output.
    
    Supports flexible time value types (int, str, datetime).
    """

    model_config = ConfigDict(extra="ignore")

    time_dimension: Optional[str] = None
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    baseline_start_time: Optional[Any] = None
    baseline_end_time: Optional[Any] = None
    comparison_strategy: Optional[str] = None
    baseline_months: List[Any] = Field(default_factory=list)
    comparison_months: List[Any] = Field(default_factory=list)
    data_type: Optional[str] = None


class SegmentSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value: Optional[str] = None
    baseline_value: Optional[float] = None
    comparison_value: Optional[float] = None
    delta_value: Optional[float] = None
    contribution_pct_total: Optional[float] = None
    contribution_pct_parent: Optional[float] = None
    aligned_delta: Optional[float] = None
    aligned_contribution_pct_total: Optional[float] = None
    aligned_contribution_pct_parent: Optional[float] = None
    aligned_contribution_pct_of_aligned_delta: Optional[float] = None
    raw_row_count_baseline: Optional[float] = None
    raw_row_count_comparison: Optional[float] = None
    opposing_share: Optional[float] = None


class PathNodeSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: Optional[int] = None
    dimension: Optional[str] = None
    dimension_label: Optional[str] = None
    folder_name: Optional[str] = None
    top_segments: List[SegmentSummarySchema] = Field(default_factory=list)
    bottom_segments: List[SegmentSummarySchema] = Field(default_factory=list)
    parent_context: Optional[str] = None


class CorrelationRunResultSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    executive_summary_path: Optional[str] = None
    period_window: Optional[PeriodWindowSchema] = None
    root_metric: Optional[str] = None
    baseline_value: Optional[float] = None
    comparison_value: Optional[float] = None
    delta_value: Optional[float] = None
    delta_pct: Optional[float] = None
    drill_path: List[PathNodeSummarySchema] = Field(default_factory=list)
    narrative_summary: Optional[str] = None
    executive_summary: Optional[str] = None
    executive_summary_source: Optional[str] = None
    interaction_matrix: Dict[str, Any] = Field(default_factory=dict)
    interaction_summary: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: List[Dict[str, Any]] = Field(default_factory=list)
    llm_tokens: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class CorrelationTokensSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: int = 0
    output: int = 0
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class CorrelationExecutionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: int = 0
    version: str = APP_VERSION


class CorrelationValidationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_valid: bool = True
    checks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ExecutiveSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""


class CorrelationAgentResponseSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent: str
    status: str
    recommended_action: List[Dict[str, Any]] = Field(default_factory=list)
    visual_component: Dict[str, Any] = Field(default_factory=dict)
    output: CorrelationRunResultSchema
    explanation: Dict[str, Any] = Field(default_factory=dict)
    validation: CorrelationValidationSchema = Field(default_factory=CorrelationValidationSchema)
    tokens: CorrelationTokensSchema
    execution: CorrelationExecutionSchema


DEFAULT_SEMANTIC_MODEL_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "correlation_pattern"
    / "coc_ecap_ip_auth_sematic_view_with_samples.yaml"
)
DEFAULT_STOP_RULES = {
    "max_depth": 4,
    "top_k_per_level": 5,
    "max_abs_delta": 1e12,
    "min_abs_delta": 50000.0,
    "min_row_count": 20,
    "min_contribution_pct": 0.10,
    "min_incremental_gain_pct": 0.05,
}


def _resolve_semantic_model_path(yaml_path: Optional[str]) -> str:
    """Resolve and validate the semantic model path for correlation execution.
    
    Args:
        yaml_path: Path to YAML file (relative or absolute). If None, uses DEFAULT_SEMANTIC_MODEL_PATH.
    
    Returns:
        Absolute path to the YAML file.
    
    Raises:
        AgentConfigurationError: If path is invalid, outside allowed directory, or file doesn't exist.
    """
    project_root = Path(__file__).resolve().parents[4]
    allowed_dir = project_root / "configs" / "correlation_pattern"
    
    if yaml_path is None:
        candidate = DEFAULT_SEMANTIC_MODEL_PATH
    else:
        candidate = Path(yaml_path).expanduser()
        
        # Resolve relative paths against project root
        if not candidate.is_absolute():
            candidate = project_root / candidate
    
    resolved = candidate.resolve()
    
    # Security: Ensure path is within allowed directory
    try:
        resolved.relative_to(allowed_dir)
    except ValueError:
        raise AgentConfigurationError(
            f"YAML path must be within {allowed_dir}. "
            f"Attempted to access: {resolved}"
        )
    
    if not resolved.exists():
        raise AgentConfigurationError(
            f"Semantic model YAML not found at {resolved}. "
            f"Original path: {yaml_path}"
        )
    
    return str(resolved)


EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """You are a senior healthcare cost-of-care analyst who translates complex claim analytics into concise executive-ready insights.

Your job:
- Summarize total cost movement with supporting volume/intensity context.
- Highlight the most material contributors, authorization detail, and offsetting declines without over-claiming causality.
- Collapse multi-level drill-downs into a business-readable narrative using only provided facts.

Output rules:
- Write 3-4 concise sentences.
- Use only the provided segment values and facts; do not hard-code or infer values.
- If a downstream clinical pocket is present, mention it in the final sentence.
- If authorization, product/plan, business segment, site-of-care, or geography nodes are present, include the largest material contributors.
- Do not list every waterfall level; summarize the path into business-readable concentrations.
- Use cautious language ("suggesting", "pointing to", "concentrated in", "warranting review").

Tone:
- Executive-facing
- Direct and measured
- No fluff or speculation
"""

EXECUTIVE_SUMMARY_USER_PROMPT_TEMPLATE = """Create a 3-4 sentence executive summary for cost-of-care leadership using only the structured facts below.
If a fact is missing, omit it.
If a downstream clinical pocket is present, mention it in the final sentence.
If authorization, product/plan, business segment, site-of-care, or geography nodes are present, include the largest material contributors.
Do not list every waterfall level; collapse the path into a business-readable narrative.

Facts:
{input_text}
"""
NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\}")

DIMENSION_FOLDER_ALIASES: Dict[str, str] = {
    "service_area_state": "state",
    "rendering_provider_name": "provider",
    "rendering_hospital_system": "hospital",
    "lob_description": "lob",
    "product_description": "product",
    "product_code": "product",
    "procedure_code": "procedure",
    "procedure_name": "procedure",
    "drg_name": "drg",
    "drg_code": "drg",
    "primary_diagnosis_name": "diagnosis",
    "primary_diagnosis_code": "diagnosis",
    "facility_type": "facility",
    "hcc_medium": "hcc_medium",
    "hcc_high": "hcc_high",
    "hcc_low": "hcc_low",
}

ROOT_SUMMARY_SQL_NAME = "root_summary.sql"
BASELINE_EXTRACT_SQL_NAME = "baseline_extract.sql"
COMPARISON_EXTRACT_SQL_NAME = "comparison_extract.sql"
RUN_CONFIG_YAML_NAME = "run_config.yaml"
MANIFEST_JSON_NAME = "manifest.json"
EXECUTIVE_SUMMARY_JSON_NAME = "executive_summary.json"
DATA_QUALITY_JSON_NAME = "data_quality.json"
GUARDRAILS_JSON_NAME = "guardrails_applied.json"


# =============================
# Generic helpers
# =============================

CORRELATION_LLM_TOKEN_STEPS = (
    "executive_summary",
    "interaction_summary",
    "recommendations",
)


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_token_usage(raw_response: Any) -> tuple[int, int]:
    usage: Dict[str, Any] = {}
    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    elif isinstance(raw_response, dict):
        nested_metadata = raw_response.get("response_metadata") if isinstance(raw_response.get("response_metadata"), dict) else {}
        usage = (
            raw_response.get("token_usage")
            or raw_response.get("usage")
            or nested_metadata.get("token_usage")
            or nested_metadata.get("usage")
            or {}
        )

    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def empty_correlation_llm_tokens() -> Dict[str, Any]:
    return {
        "input": 0,
        "output": 0,
        "breakdown": {
            step_name: {"input": 0, "output": 0}
            for step_name in CORRELATION_LLM_TOKEN_STEPS
        },
    }


def merge_correlation_llm_tokens(*token_payloads: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = empty_correlation_llm_tokens()
    for token_payload in token_payloads:
        payload = _safe_dict(token_payload)
        merged["input"] += int(payload.get("input", 0) or 0)
        merged["output"] += int(payload.get("output", 0) or 0)
        payload_breakdown = _safe_dict(payload.get("breakdown"))
        merged_breakdown = _safe_dict(merged.get("breakdown"))
        step_names = set(CORRELATION_LLM_TOKEN_STEPS) | set(payload_breakdown.keys())
        for step_name in step_names:
            step_payload = _safe_dict(payload_breakdown.get(step_name))
            merged_step = _safe_dict(merged_breakdown.get(step_name))
            merged_breakdown[step_name] = {
                "input": int(merged_step.get("input", 0) or 0) + int(step_payload.get("input", 0) or 0),
                "output": int(merged_step.get("output", 0) or 0) + int(step_payload.get("output", 0) or 0),
            }
        merged["breakdown"] = merged_breakdown
    return merged


def correlation_llm_tokens_for_step(step_name: str, input_tokens: int = 0, output_tokens: int = 0) -> Dict[str, Any]:
    normalized_input = int(input_tokens or 0)
    normalized_output = int(output_tokens or 0)
    breakdown = {
        name: {"input": 0, "output": 0}
        for name in CORRELATION_LLM_TOKEN_STEPS
    }
    breakdown[str(step_name)] = {
        "input": normalized_input,
        "output": normalized_output,
    }
    return {
        "input": normalized_input,
        "output": normalized_output,
        "breakdown": breakdown,
    }


def _recommended_action_items(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = _safe_dict(value).get("recommended_action")
    return [item for item in _safe_list(raw_items) if isinstance(item, dict)]


def _interaction_recommendations_artifact(value: Any) -> Dict[str, Any]:
    payload = _safe_dict(value)
    source = str(payload.get("source") or "empty")
    if isinstance(value, list):
        source = "empty" if not _recommended_action_items(value) else source
    return {
        "recommended_action": _recommended_action_items(value),
        "source": source,
    }


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _resolve_save_parquet(intent: IntentPayload) -> bool:
    context_override = _coerce_bool(intent.get("save_parquet"), default=None)
    if context_override is not None:
        return context_override
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    mode_value = _coerce_bool(mode_parameters.get("save_parquet"), default=None)
    if mode_value is not None:
        return mode_value
    return False


def _resolve_disable_summary_creation(intent: IntentPayload) -> bool:
    context_override = _coerce_bool(intent.get("disable_summary_creation"), default=None)
    if context_override is not None:
        return context_override
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    mode_value = _coerce_bool(mode_parameters.get("disable_summary_creation"), default=None)
    if mode_value is not None:
        return mode_value
    return True


def _resolve_generate_recommendations(intent: IntentPayload) -> bool:
    context_override = _coerce_bool(intent.get("generate_recommendations"), default=None)
    if context_override is not None:
        return context_override
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    mode_value = _coerce_bool(mode_parameters.get("generate_recommendations"), default=None)
    if mode_value is not None:
        return mode_value
    return False


def _context_from_intent(intent: IntentPayload) -> Dict[str, Any]:
    return {
        "analysis_mode_parameters": _safe_dict(intent.get("analysis_mode_parameters")),
        "filters": list(intent.get("filters", [])),
        "metric_hint": intent.get("metric_hint"),
        "analysis_mode": intent.get("analysis_mode"),
        "group_by": list(intent.get("group_by", [])),
        "validation_warnings": list(intent.get("validation_warnings", [])),
        "save_parquet": _coerce_bool(intent.get("save_parquet"), default=False),
        "disable_summary_creation": _coerce_bool(intent.get("disable_summary_creation"), default=True),
        "generate_recommendations": _coerce_bool(intent.get("generate_recommendations"), default=False),
    }


def _build_intent_from_request(query: str, context: Optional[Mapping[str, Any]]) -> IntentPayload:
    context_dict = _safe_dict(context)
    mode_parameters = _safe_dict(context_dict.get("analysis_mode_parameters"))
    analysis_mode = str(context_dict.get("analysis_mode") or "").strip()
    analysis_mode = analysis_mode or "cost_change_investigation_over_time_window"
    metric_hint_value = context_dict.get("metric_hint")
    metric_hint = str(metric_hint_value).strip() if metric_hint_value is not None else None
    group_by = [
        str(item).strip()
        for item in _safe_list(context_dict.get("group_by"))
        if str(item).strip()
    ]
    validation_warnings = [
        str(item)
        for item in _safe_list(context_dict.get("validation_warnings"))
        if item is not None
    ]
    context_save_parquet = _coerce_bool(context_dict.get("save_parquet"), default=None)
    mode_save_parquet = _coerce_bool(mode_parameters.get("save_parquet"), default=None)
    save_parquet = context_save_parquet if context_save_parquet is not None else (mode_save_parquet if mode_save_parquet is not None else False)
    context_disable_summary_creation = _coerce_bool(context_dict.get("disable_summary_creation"), default=None)
    mode_disable_summary_creation = _coerce_bool(mode_parameters.get("disable_summary_creation"), default=None)
    disable_summary_creation = (
        context_disable_summary_creation
        if context_disable_summary_creation is not None
        else (mode_disable_summary_creation if mode_disable_summary_creation is not None else True)
    )
    context_generate_recommendations = _coerce_bool(context_dict.get("generate_recommendations"), default=None)
    mode_generate_recommendations = _coerce_bool(mode_parameters.get("generate_recommendations"), default=None)
    generate_recommendations = (
        context_generate_recommendations
        if context_generate_recommendations is not None
        else (mode_generate_recommendations if mode_generate_recommendations is not None else False)
    )
    return {
        "analysis_mode": analysis_mode,
        "analysis_mode_parameters": mode_parameters,
        "filters": [item for item in _safe_list(context_dict.get("filters")) if isinstance(item, dict)],
        "group_by": group_by,
        "metric_hint": metric_hint,
        "raw_question": query,
        "validation_warnings": validation_warnings,
        "save_parquet": save_parquet,
        "disable_summary_creation": disable_summary_creation,
        "generate_recommendations": generate_recommendations,
    }


def normalize(text: Any) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"[^a-z0-9_ ]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("_")


def _find_analysis_mode_definition(
    semantic_model: Mapping[str, Any],
    analysis_mode: str,
) -> Dict[str, Any]:
    if not analysis_mode:
        return {}
    normalized_name = normalize(analysis_mode)
    for raw_mode in _safe_list(semantic_model.get("analysis_modes")):
        mode_dict = _safe_dict(raw_mode)
        name = str(mode_dict.get("name", "")).strip()
        if not name:
            continue
        if normalize(name) == normalized_name:
            return mode_dict
        for alias in _safe_list(mode_dict.get("aliases")):
            alias_name = str(alias).strip()
            if alias_name and normalize(alias_name) == normalized_name:
                return mode_dict
    return {}


def _apply_analysis_mode_defaults(
    intent: IntentPayload,
    semantic_model: Mapping[str, Any],
) -> IntentPayload:
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    analysis_mode = str(intent.get("analysis_mode") or "").strip()
    defaults = _find_analysis_mode_definition(semantic_model, analysis_mode)
    if not defaults:
        return intent

    drill_dimensions = [
        str(item).strip()
        for item in _safe_list(defaults.get("drill_dimensions"))
        if str(item).strip()
    ]
    explainer_metrics = [
        str(item).strip()
        for item in _safe_list(defaults.get("explainer_metrics"))
        if str(item).strip()
    ]
    interaction_matrix = copy.deepcopy(_safe_dict(defaults.get("interaction_matrix")))
    stop_rules = copy.deepcopy(_safe_dict(defaults.get("stop_rules")))
    default_save_parquet = _coerce_bool(defaults.get("save_parquet"), default=None)
    default_disable_summary_creation = _coerce_bool(defaults.get("disable_summary_creation"), default=None)
    default_generate_recommendations = _coerce_bool(defaults.get("generate_recommendations"), default=None)
    if (
        not drill_dimensions
        and not explainer_metrics
        and not interaction_matrix
        and not stop_rules
        and default_save_parquet is None
        and default_disable_summary_creation is None
        and default_generate_recommendations is None
    ):
        return intent

    if not isinstance(intent.get("analysis_mode_parameters"), dict):
        intent["analysis_mode_parameters"] = {}
        mode_parameters = intent["analysis_mode_parameters"]

    if "drill_dimensions" not in mode_parameters or mode_parameters.get("drill_dimensions") is None:
        mode_parameters["drill_dimensions"] = drill_dimensions
    if "explainer_metrics" not in mode_parameters or mode_parameters.get("explainer_metrics") is None:
        mode_parameters["explainer_metrics"] = explainer_metrics
    if "interaction_matrix" not in mode_parameters or mode_parameters.get("interaction_matrix") is None:
        mode_parameters["interaction_matrix"] = interaction_matrix
    if stop_rules and ("stop_rules" not in mode_parameters or mode_parameters.get("stop_rules") is None):
        mode_parameters["stop_rules"] = stop_rules
    if default_save_parquet is not None and ("save_parquet" not in mode_parameters or mode_parameters.get("save_parquet") is None):
        mode_parameters["save_parquet"] = default_save_parquet
    if default_disable_summary_creation is not None and (
        "disable_summary_creation" not in mode_parameters or mode_parameters.get("disable_summary_creation") is None
    ):
        mode_parameters["disable_summary_creation"] = default_disable_summary_creation
    if default_generate_recommendations is not None and (
        "generate_recommendations" not in mode_parameters or mode_parameters.get("generate_recommendations") is None
    ):
        mode_parameters["generate_recommendations"] = default_generate_recommendations
    return intent


def slugify(text: Any) -> str:
    value = normalize(text)
    return value or "unknown"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_semantic_yaml(path: str) -> Dict[str, Any]:
    """Load semantic YAML file (loaded fresh per request based on yaml_path payload)."""
    logger.info("🔍 Loading semantic YAML from: %s", path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    logger.info("✅ Semantic YAML loaded successfully: %s", payload.get('name', 'unknown'))
    return payload if isinstance(payload, dict) else {}


def parse_yyyymm(value: Any) -> Tuple[int, int]:
    value_str = str(value)
    if not re.match(r"^\d{6}$", value_str):
        raise ValueError(f"Expected YYYYMM integer/string, received: {value}")
    year = int(value_str[:4])
    month = int(value_str[4:])
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month in YYYYMM value: {value}")
    return year, month


def make_yyyymm(year: int, month: int) -> int:
    return int(f"{year:04d}{month:02d}")


def shift_yyyymm_by_year(value: int, years: int) -> int:
    year, month = parse_yyyymm(value)
    return make_yyyymm(year + years, month)


def expand_yyyymm_window(start_value: int, end_value: int) -> List[int]:
    start_year, start_month = parse_yyyymm(start_value)
    end_year, end_month = parse_yyyymm(end_value)
    months: List[int] = []
    year = start_year
    month = start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        months.append(make_yyyymm(year, month))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return months


def format_time_literal(value: Any, data_type: Optional[str]) -> str:
    """Format a time value for SQL based on its data_type.
    
    Uses Snowflake-compatible DATE/TIMESTAMP literal syntax.
    
    Args:
        value: The time value (int, str, date, or datetime)
        data_type: The semantic data type ('number', 'date', 'timestamp', etc.)
    
    Returns:
        SQL-formatted literal string
    
    Examples:
        format_time_literal(202501, 'number') -> '202501'
        format_time_literal('2025-01-01', 'date') -> "DATE '2025-01-01'"
        format_time_literal(datetime(2025, 1, 1), 'timestamp') -> "TIMESTAMP '2025-01-01 00:00:00'"
    """
    # Handle explicit number type
    if data_type and data_type.lower() == 'number':
        return str(value)
    
    # Handle datetime objects
    if isinstance(value, datetime):
        if data_type and 'timestamp' in data_type.lower():
            return f"TIMESTAMP '{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        # Default to DATE for datetime objects
        return f"DATE '{value.strftime('%Y-%m-%d')}'"
    
    # Handle string values
    if isinstance(value, str):
        # Check for timestamp indicator
        if data_type and 'timestamp' in data_type.lower():
            return f"TIMESTAMP '{value}'"
        
        # Extract date part if it includes time component
        date_part = value.split('T')[0].split(' ')[0]
        
        # Check if it looks like an ISO date (YYYY-MM-DD)
        if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
            # Always use DATE syntax for ISO date strings to avoid Snowflake type errors
            return f"DATE '{date_part}'"
        
        # If data_type explicitly says 'date', use DATE syntax
        if data_type and 'date' in data_type.lower():
            return f"DATE '{date_part}'"
        
        # Fallback for other strings (non-date patterns)
        return f"'{value}'"
    
    # Fallback for other types (int, etc.)
    return str(value)


def calculate_month_aligned_exclusive_end(start_value: Any, end_value: Any, data_type: Optional[str]) -> Any:
    """Calculate the exclusive upper bound for a half-open range [start, exclusive_end).
    
    For YYYYMM integers: increment the month (202512 -> 202601)
    For date strings: parse, add 1 month, return ISO string
    For datetime objects: add 1 month, return datetime
    
    Args:
        start_value: Start of the period
        end_value: End of the period (inclusive)
        data_type: The semantic data type
    
    Returns:
        The first moment of the month after end_value
    """
    if data_type and data_type.lower() == 'number':
        year, month = parse_yyyymm(end_value)
        month += 1
        if month > 12:
            year += 1
            month = 1
        return make_yyyymm(year, month)
    
    if isinstance(end_value, str):
        from dateutil import parser as dateutil_parser
        end_dt = dateutil_parser.parse(end_value)
        end_month_start = datetime(end_dt.year, end_dt.month, 1)
        next_month = end_month_start.month + 1
        next_year = end_month_start.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        return f"{next_year:04d}-{next_month:02d}-01"
    
    if isinstance(end_value, datetime):
        next_month = end_value.month + 1
        next_year = end_value.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        return datetime(next_year, next_month, 1)
    
    raise ValueError(f"Unsupported time value type for exclusive end calculation: {type(end_value)}")


def shift_time_by_year(value: Any, years: int, data_type: Optional[str]) -> Any:
    """Shift a time value by a number of years, preserving the month and day.
    
    Generalizes shift_yyyymm_by_year to support dates and datetimes.
    
    Args:
        value: The time value to shift
        years: Number of years to shift (can be negative)
        data_type: The semantic data type
    
    Returns:
        The shifted time value in the same type as the input
    """
    if data_type and data_type.lower() == 'number':
        return shift_yyyymm_by_year(int(value), years)
    
    if isinstance(value, str):
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(value)
        shifted_dt = datetime(dt.year + years, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        if 'timestamp' in (data_type or '').lower():
            return shifted_dt.strftime('%Y-%m-%d %H:%M:%S')
        return shifted_dt.strftime('%Y-%m-%d')
    
    if isinstance(value, datetime):
        return datetime(value.year + years, value.month, value.day, value.hour, value.minute, value.second, value.microsecond, value.tzinfo)
    
    raise ValueError(f"Unsupported time value type for year shift: {type(value)}")


def expand_time_window(start_value: Any, end_value: Any, data_type: Optional[str]) -> List[Any]:
    """Expand a time window into a list of month values.
    
    Generalizes expand_yyyymm_window to support dates and datetimes.
    
    Args:
        start_value: Start of the period
        end_value: End of the period (inclusive)
        data_type: The semantic data type
    
    Returns:
        List of month values in the same type as the input
    """
    if data_type and data_type.lower() == 'number':
        return expand_yyyymm_window(int(start_value), int(end_value))
    
    if isinstance(start_value, str):
        from dateutil import parser as dateutil_parser
        start_dt = dateutil_parser.parse(start_value)
        end_dt = dateutil_parser.parse(end_value)
        months: List[str] = []
        current = datetime(start_dt.year, start_dt.month, 1)
        end_month_start = datetime(end_dt.year, end_dt.month, 1)
        
        while current <= end_month_start:
            if 'timestamp' in (data_type or '').lower():
                months.append(current.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                months.append(current.strftime('%Y-%m-%d'))
            
            current_month = current.month + 1
            current_year = current.year
            if current_month > 12:
                current_month = 1
                current_year += 1
            current = datetime(current_year, current_month, 1)
        
        return months
    
    if isinstance(start_value, datetime):
        months: List[datetime] = []
        current = datetime(start_value.year, start_value.month, 1)
        end_month_start = datetime(end_value.year, end_value.month, 1)
        
        while current <= end_month_start:
            months.append(current)
            current_month = current.month + 1
            current_year = current.year
            if current_month > 12:
                current_month = 1
                current_year += 1
            current = datetime(current_year, current_month, 1)
        
        return months
    
    return []


def escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def to_python_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def maybe_number(value: str) -> bool:
    return bool(NUMERIC_PATTERN.match(str(value).strip()))


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def first_non_empty(values: Sequence[Optional[str]]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if str(value).strip():
            return str(value).strip()
    return None


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def write_yaml(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_text(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def write_dataframe(path: Path, df: pd.DataFrame) -> str:
    """
    Best-effort artifact writer.
    Prefers parquet, falls back to json if parquet dependencies are unavailable.
    Returns the actual written path.
    """
    target_path = path
    try:
        df.to_parquet(target_path, index=False)
        return str(target_path)
    except Exception as exc:
        fallback_path = target_path.with_suffix(".json")
        logger.warning(
            "Failed to write parquet at %s; wrote JSON fallback to %s. Error: %s",
            target_path,
            fallback_path,
            exc,
        )
        df.to_json(fallback_path, orient="records", indent=2)
        return str(fallback_path)


# =============================
# Semantic catalog construction
# =============================

def parse_metric_dependencies(expr: str) -> List[str]:
    dependencies: Set[str] = set()
    for table_name, _ in PLACEHOLDER_PATTERN.findall(expr or ""):
        dependencies.add(table_name)
    return sorted(dependencies)


def build_semantic_catalog(semantic_model: Dict[str, Any]) -> SemanticCatalog:
    tables: Dict[str, TableDefinition] = {}
    fields_by_name: Dict[str, List[FieldDefinition]] = {}
    table_fields: Dict[str, Dict[str, FieldDefinition]] = {}
    metrics_by_name: Dict[str, MetricDefinition] = {}
    relationships: List[RelationshipDefinition] = []

    def register_field(table_name: str, kind: str, raw_field: Mapping[str, Any]) -> None:
        field_name = str(raw_field.get("name", "")).strip()
        expr = str(raw_field.get("expr", "")).strip()
        if not field_name or not expr:
            return

        label = str(raw_field.get("label") or raw_field.get("display_name") or "").strip()
        field_def: FieldDefinition = {
            "name": field_name,
            "table_name": table_name,
            "expr": expr,
            "data_type": str(raw_field.get("data_type", "")).strip(),
            "description": str(raw_field.get("description", "")).strip(),
            "label": label,
            "kind": kind,
            "synonyms": [str(item).strip() for item in _safe_list(raw_field.get("synonyms")) if str(item).strip()],
        }
        table_fields.setdefault(table_name, {})[field_name] = field_def
        fields_by_name.setdefault(field_name, []).append(field_def)

    for raw_table in _safe_list(semantic_model.get("tables")):
        table_dict = _safe_dict(raw_table)
        table_name = str(table_dict.get("name", "")).strip()
        if not table_name:
            continue

        base_table = _safe_dict(table_dict.get("base_table"))
        database = str(base_table.get("database", "")).strip()
        schema = str(base_table.get("schema", "")).strip()
        physical_table = str(base_table.get("table", "")).strip()
        qualified_name = ".".join([part for part in (database, schema, physical_table) if part])

        tables[table_name] = {
            "name": table_name,
            "description": str(table_dict.get("description", "")).strip(),
            "database": database,
            "schema": schema,
            "table": physical_table,
            "qualified_name": qualified_name,
        }

        for kind in ("dimensions", "time_dimensions", "facts"):
            for field in _safe_list(table_dict.get(kind)):
                register_field(table_name, kind[:-1] if kind.endswith("s") else kind, _safe_dict(field))

        for raw_metric in _safe_list(table_dict.get("metrics")):
            metric_dict = _safe_dict(raw_metric)
            metric_name = str(metric_dict.get("name", "")).strip()
            expr = str(metric_dict.get("expr", "")).strip()
            if not metric_name or not expr:
                continue
            dependencies = parse_metric_dependencies(expr) or [table_name]
            metrics_by_name[metric_name] = {
                "name": metric_name,
                "expr": expr,
                "description": str(metric_dict.get("description", "")).strip(),
                "dependency_tables": dependencies,
                "primary_table": table_name,
            }

    for raw_metric in _safe_list(semantic_model.get("metrics")):
        metric_dict = _safe_dict(raw_metric)
        metric_name = str(metric_dict.get("name", "")).strip()
        expr = str(metric_dict.get("expr", "")).strip()
        if not metric_name or not expr:
            continue
        dependencies = parse_metric_dependencies(expr)
        metrics_by_name[metric_name] = {
            "name": metric_name,
            "expr": expr,
            "description": str(metric_dict.get("description", "")).strip(),
            "dependency_tables": dependencies,
            "primary_table": dependencies[0] if len(dependencies) == 1 else None,
        }

    for raw_relationship in _safe_list(semantic_model.get("relationships")):
        rel_dict = _safe_dict(raw_relationship)
        if not rel_dict:
            continue
        relationships.append(
            {
                "name": str(rel_dict.get("name", "")).strip(),
                "left_table": str(rel_dict.get("left_table", "")).strip(),
                "right_table": str(rel_dict.get("right_table", "")).strip(),
                "relationship_columns": [
                    {
                        "left_column": str(_safe_dict(item).get("left_column", "")).strip(),
                        "right_column": str(_safe_dict(item).get("right_column", "")).strip(),
                    }
                    for item in _safe_list(rel_dict.get("relationship_columns"))
                ],
            }
        )

    catalog = {
        "tables": tables,
        "fields_by_name": fields_by_name,
        "table_fields": table_fields,
        "metrics_by_name": metrics_by_name,
        "relationships": relationships,
    }
    
    logger.info("📊 Semantic catalog built: %d tables, %d metrics, %d fields", 
                len(tables), len(metrics_by_name), len(fields_by_name))
    logger.info("📈 Available metrics: %s", list(metrics_by_name.keys()))
    
    return catalog


# =============================
# Field and metric resolution
# =============================

def resolve_metric_name(intent: IntentPayload, catalog: SemanticCatalog) -> str:
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    drill_metrics = [str(item).strip() for item in _safe_list(mode_parameters.get("drill_metric")) if str(item).strip()]
    metric_hint = str(intent.get("metric_hint") or "").strip()

    for candidate in drill_metrics + ([metric_hint] if metric_hint else []):
        if candidate in catalog["metrics_by_name"]:
            return candidate

    if metric_hint:
        metric_hint_norm = normalize(metric_hint).replace("_", " ")
        for metric_name in catalog["metrics_by_name"]:
            metric_norm = normalize(metric_name).replace("_", " ")
            if metric_hint_norm == metric_norm or metric_hint_norm in metric_norm or metric_norm in metric_hint_norm:
                return metric_name

    raise KeyError("Unable to resolve drill metric from intent payload.")


def resolve_field(
    catalog: SemanticCatalog,
    field_name: str,
    preferred_tables: Optional[Sequence[str]] = None,
) -> Optional[FieldDefinition]:
    candidates = catalog["fields_by_name"].get(field_name, [])
    if not candidates:
        return None

    preferred = list(preferred_tables or [])
    for table_name in preferred:
        for candidate in candidates:
            if candidate["table_name"] == table_name:
                return candidate

    return candidates[0]


def resolve_time_field(period_window: PeriodWindow, catalog: SemanticCatalog) -> FieldDefinition:
    qualified_name = period_window["time_dimension"]
    if "." in qualified_name:
        table_name, field_name = qualified_name.split(".", 1)
        field = catalog["table_fields"].get(table_name, {}).get(field_name)
        if field:
            return field
    fallback = resolve_field(catalog, qualified_name, None)
    if fallback is None:
        raise KeyError(f"Unable to resolve time dimension: {qualified_name}")
    return fallback


def default_primary_table(metric: MetricDefinition, period_window: PeriodWindow, catalog: SemanticCatalog) -> str:
    time_field = resolve_time_field(period_window, catalog)
    if time_field.get("table_name"):
        return str(time_field["table_name"])
    if metric.get("primary_table"):
        return str(metric["primary_table"])
    dependencies = metric.get("dependency_tables") or []
    if dependencies:
        return str(dependencies[0])
    raise KeyError(f"Unable to determine primary table for metric {metric.get('name')}")


def build_alias_map(table_names: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    used: Set[str] = set()
    for index, table_name in enumerate(table_names, start=1):
        parts = [part for part in table_name.split("_") if part]
        alias = "".join(part[0] for part in parts) or f"t{index}"
        alias = alias[:6]
        if alias in used:
            alias = f"{alias}{index}"
        aliases[table_name] = alias
        used.add(alias)
    return aliases


def render_expression(expr: str, alias_map: Dict[str, str], catalog: SemanticCatalog) -> str:
    def replacer(match: re.Match[str]) -> str:
        table_name = match.group(1)
        field_name = match.group(2)
        field_def = catalog["table_fields"].get(table_name, {}).get(field_name)
        if not field_def:
            raise KeyError(f"Unknown field placeholder: {table_name}.{field_name}")
        alias = alias_map[table_name]
        return f"{alias}.{field_def['expr']}"

    return PLACEHOLDER_PATTERN.sub(replacer, expr)


def determine_required_tables(
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    dimensions: Sequence[str],
    period_window: PeriodWindow,
    catalog: SemanticCatalog,
    primary_table: str,
) -> List[str]:
    required: List[str] = [primary_table]
    seen: Set[str] = {primary_table}

    def add_table(table_name: Optional[str]) -> None:
        if not table_name:
            return
        if table_name not in seen:
            required.append(table_name)
            seen.add(table_name)

    for table_name in _safe_list(metric.get("dependency_tables")):
        add_table(str(table_name))

    time_field = resolve_time_field(period_window, catalog)
    add_table(time_field.get("table_name"))

    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]
    for filter_condition in filters:
        if filter_condition["operator"].lower().strip() == "named_filter":
            for table_name in parse_metric_dependencies(filter_condition["value"]):
                add_table(table_name)
            continue
        field_def = resolve_field(catalog, filter_condition["field"], preferred_tables=preferred_tables)
        add_table(field_def.get("table_name") if field_def else None)

    for dimension in dimensions:
        field_def = resolve_field(catalog, dimension, preferred_tables=[primary_table]) or resolve_field(catalog, dimension)
        add_table(field_def.get("table_name") if field_def else None)

    return required


def find_relationship(
    catalog: SemanticCatalog,
    left_table: str,
    right_table: str,
) -> Optional[Tuple[RelationshipDefinition, bool]]:
    for relationship in catalog["relationships"]:
        if relationship.get("left_table") == left_table and relationship.get("right_table") == right_table:
            return relationship, False
        if relationship.get("left_table") == right_table and relationship.get("right_table") == left_table:
            return relationship, True
    return None


def build_from_clause(
    catalog: SemanticCatalog,
    required_tables: Sequence[str],
    primary_table: str,
    alias_map: Dict[str, str],
) -> str:
    if primary_table not in catalog["tables"]:
        raise KeyError(f"Unknown primary table: {primary_table}")

    joined_tables: Set[str] = {primary_table}
    clauses = [f"FROM {catalog['tables'][primary_table]['qualified_name']} {alias_map[primary_table]}"]
    pending = [table_name for table_name in required_tables if table_name != primary_table]

    while pending:
        progress = False
        for table_name in list(pending):
            for joined in list(joined_tables):
                rel_info = find_relationship(catalog, joined, table_name)
                if rel_info is None:
                    continue
                relationship, reversed_direction = rel_info
                left_alias = alias_map[joined]
                right_alias = alias_map[table_name]
                conditions: List[str] = []
                for column_pair in _safe_list(relationship.get("relationship_columns")):
                    pair = _safe_dict(column_pair)
                    left_column = str(pair.get("left_column", "")).strip()
                    right_column = str(pair.get("right_column", "")).strip()
                    if not left_column or not right_column:
                        continue
                    if reversed_direction:
                        conditions.append(f"{left_alias}.{right_column} = {right_alias}.{left_column}")
                    else:
                        conditions.append(f"{left_alias}.{left_column} = {right_alias}.{right_column}")
                clauses.append(
                    f"LEFT JOIN {catalog['tables'][table_name]['qualified_name']} {right_alias}"
                    f" ON {' AND '.join(conditions)}"
                )
                joined_tables.add(table_name)
                pending.remove(table_name)
                progress = True
                break
            if progress:
                break
        if not progress:
            raise ValueError(f"Unable to join tables with available relationships: {pending}")

    return "\n".join(clauses)


def render_filter_clause(
    filter_condition: FilterCondition,
    catalog: SemanticCatalog,
    alias_map: Dict[str, str],
    preferred_tables: Sequence[str],
) -> str:
    operator = filter_condition["operator"].lower().strip()

    if operator == "named_filter":
        return render_expression(filter_condition["value"], alias_map, catalog)

    field_def = resolve_field(catalog, filter_condition["field"], preferred_tables=preferred_tables)
    if field_def is None:
        raise KeyError(f"Unknown filter field: {filter_condition['field']}")

    alias = alias_map[field_def["table_name"]]
    field_sql = f"{alias}.{field_def['expr']}"
    raw_filter_value = filter_condition["value"]
    raw_value = str(raw_filter_value).strip()
    data_type = str(field_def.get("data_type", "")).lower().strip()

    def literal(value: str) -> str:
        if data_type in {"number", "integer", "float"} and maybe_number(value):
            return value
        if maybe_number(value) and data_type != "string":
            return value
        return f"'{escape_sql_string(value)}'"

    if operator in {"=", ">", "<", ">=", "<="}:
        if operator == "=" and isinstance(raw_filter_value, (list, tuple, set)):
            values = [str(item).strip() for item in raw_filter_value if str(item).strip()]
            if not values:
                raise ValueError(f"IN filter expects at least one value: {filter_condition}")
            return f"{field_sql} IN ({', '.join(literal(item) for item in values)})"
        return f"{field_sql} {operator} {literal(raw_value)}"
    if operator == "like":
        return f"{field_sql} LIKE {literal(raw_value)}"
    if operator == "in":
        if isinstance(raw_filter_value, (list, tuple, set)):
            values = [str(item).strip() for item in raw_filter_value if str(item).strip()]
        else:
            values = [item.strip() for item in raw_value.split(",") if item.strip()]
        if not values:
            raise ValueError(f"IN filter expects at least one value: {filter_condition}")
        return f"{field_sql} IN ({', '.join(literal(item) for item in values)})"
    if operator == "between":
        pieces = [item.strip() for item in re.split(r"\band\b", raw_value, flags=re.IGNORECASE) if item.strip()]
        if len(pieces) != 2:
            raise ValueError(f"BETWEEN filter expects 'value1 and value2': {filter_condition}")
        return f"{field_sql} BETWEEN {literal(pieces[0])} AND {literal(pieces[1])}"

    raise ValueError(f"Unsupported filter operator: {filter_condition['operator']}")


def _normalize_filter_values(filter_condition: FilterCondition) -> List[Any]:
    """Normalize filter values into a flat list for OR-style grouping."""
    value = filter_condition["value"]
    operator = str(filter_condition.get("operator", "")).lower().strip()
    if isinstance(value, (list, tuple, set)):
        return [item for item in value]
    if operator == "in" and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _filter_value_key(value: Any) -> str:
    """Create a stable key for deduplicating filter values."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return str(value)


def _normalize_dimension_name(name: str) -> str:
    """Normalize a drill dimension name for comparison."""
    if "." in name:
        return name.split(".", 1)[1]
    return name


def normalize_filters_for_sql(filters: Sequence[FilterCondition]) -> List[FilterCondition]:
    """Collapse repeated field filters into a single IN clause to avoid AND conflicts."""
    grouped: Dict[Tuple[str, FilterSource], Dict[str, Any]] = {}
    ordered_entries: List[Tuple[str, Any]] = []

    for filter_condition in filters:
        operator = str(filter_condition.get("operator", "")).lower().strip()
        if operator in {"=", "in"}:
            key = (filter_condition["field"], filter_condition["source"])
            if key not in grouped:
                grouped[key] = {
                    "template": filter_condition,
                    "operators": set(),
                    "values": [],
                }
                ordered_entries.append(("group", key))
            grouped[key]["operators"].add(operator)
            grouped[key]["values"].extend(_normalize_filter_values(filter_condition))
        else:
            ordered_entries.append(("single", filter_condition))

    merged: List[FilterCondition] = []
    emitted_groups: Set[Tuple[str, FilterSource]] = set()

    for entry_type, entry in ordered_entries:
        if entry_type == "single":
            merged.append(entry)
            continue
        if entry in emitted_groups:
            continue
        emitted_groups.add(entry)
        group = grouped[entry]
        values: List[Any] = []
        seen: Set[str] = set()
        for value in group["values"]:
            key = _filter_value_key(value)
            if key in seen:
                continue
            seen.add(key)
            values.append(value)

        template = group["template"]
        if len(values) == 1 and group["operators"] == {"="}:
            merged.append({**template, "operator": "=", "value": values[0]})
        else:
            merged.append({**template, "operator": "in", "value": values})

    return merged


# =============================
# Period handling
# =============================

def resolve_period_window(intent: IntentPayload, catalog: SemanticCatalog) -> PeriodWindow:
    """Resolve current/previous period windows from analysis mode parameters.
    
    Supports multiple time formats (YYYYMM int, date, datetime) based on the
    semantic catalog's data_type for the time dimension.
    """

    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    period_dict = _safe_dict(mode_parameters.get("period"))
    current_period = _safe_dict(period_dict.get("current_period"))
    previous_period = _safe_dict(period_dict.get("previous_period"))
    
    # Require time_dimension from config - no hardcoded fallback
    time_dimension = str(
        period_dict.get("time_dimension")
        or period_dict.get("rolling_time_dimension")
        or ""
    ).strip()
    if not time_dimension:
        raise AgentConfigurationError(
            "time_dimension or rolling_time_dimension must be specified in the analysis_mode_parameters.period configuration. "
            "Check your semantic view YAML for the correct time dimension field (e.g., 'expense_detail.incurred_month')."
        )

    # Lookup data_type from catalog for the time dimension
    data_type = None
    time_field = None
    if "." in time_dimension:
        table_name, field_name = time_dimension.split(".", 1)
        time_field = catalog.get("table_fields", {}).get(table_name, {}).get(field_name)
    else:
        candidates = catalog.get("fields_by_name", {}).get(time_dimension, [])
        if candidates:
            time_field = candidates[0]
    
    if time_field:
        data_type = time_field.get("data_type")
    
    if not data_type:
        logger.warning(
            f"Could not determine data_type for time dimension '{time_dimension}'. "
            "Defaulting to 'number' for backward compatibility."
        )
        data_type = "number"

    # Require comparison dates from config - no hardcoded fallback
    comparison_start_value = (
        current_period.get("start_time")
        or period_dict.get("start_time")
    )
    if not comparison_start_value:
        raise AgentConfigurationError(
            "start_time must be specified in current_period or period configuration. "
            "Provide a time value (e.g., 202601 for YYYYMM or '2025-01-01' for date) in your analysis_mode_parameters.period.current_period.start_time."
        )
    
    comparison_end_value = (
        current_period.get("end_time")
        or period_dict.get("end_time")
    )
    if not comparison_end_value:
        raise AgentConfigurationError(
            "end_time must be specified in current_period or period configuration. "
            "Provide a time value (e.g., 202603 for YYYYMM or '2025-03-01' for date) in your analysis_mode_parameters.period.current_period.end_time."
        )
    
    # Preserve the native type - don't cast to int
    comparison_start = comparison_start_value
    comparison_end = comparison_end_value
    
    # Calculate baseline from comparison if not explicitly provided
    baseline_start_explicit = (
        previous_period.get("start_time")
        or period_dict.get("baseline_start_time")
    )
    baseline_start = (
        baseline_start_explicit
        if baseline_start_explicit is not None
        else shift_time_by_year(comparison_start, -1, data_type)
    )
    
    baseline_end_explicit = (
        previous_period.get("end_time")
        or period_dict.get("baseline_end_time")
    )
    baseline_end = (
        baseline_end_explicit
        if baseline_end_explicit is not None
        else shift_time_by_year(comparison_end, -1, data_type)
    )
    
    comparison_strategy = str(
        period_dict.get("comparison_strategy") or "prior_year_same_window"
    ).strip()

    return {
        "time_dimension": time_dimension,
        "start_time": comparison_start,
        "end_time": comparison_end,
        "baseline_start_time": baseline_start,
        "baseline_end_time": baseline_end,
        "comparison_strategy": comparison_strategy,
        "baseline_months": expand_time_window(baseline_start, baseline_end, data_type),
        "comparison_months": expand_time_window(comparison_start, comparison_end, data_type),
        "data_type": data_type,
    }


# =============================
# SQL builders
# =============================

def period_case_sql(time_expr: str, period_window: PeriodWindow) -> str:
    """Generate SQL CASE expression using half-open range pattern.
    
    Uses >= start AND < exclusive_end to avoid midnight cutoff issues.
    """
    data_type = period_window.get("data_type")
    
    # Calculate exclusive upper bounds (first day of month after the end month)
    baseline_exclusive_end = calculate_month_aligned_exclusive_end(
        period_window['baseline_start_time'],
        period_window['baseline_end_time'],
        data_type
    )
    comparison_exclusive_end = calculate_month_aligned_exclusive_end(
        period_window['start_time'],
        period_window['end_time'],
        data_type
    )
    
    # Format literals based on data_type
    baseline_start_lit = format_time_literal(period_window['baseline_start_time'], data_type)
    baseline_end_lit = format_time_literal(baseline_exclusive_end, data_type)
    comparison_start_lit = format_time_literal(period_window['start_time'], data_type)
    comparison_end_lit = format_time_literal(comparison_exclusive_end, data_type)
    
    return (
        "CASE "
        f"WHEN {time_expr} >= {baseline_start_lit} AND {time_expr} < {baseline_end_lit} THEN 'baseline' "
        f"WHEN {time_expr} >= {comparison_start_lit} AND {time_expr} < {comparison_end_lit} THEN 'comparison' "
        "ELSE NULL END"
    )


def period_filter_sql(time_expr: str, period_window: PeriodWindow) -> str:
    """Generate SQL filter using half-open range pattern.
    
    Uses >= start AND < exclusive_end to avoid midnight cutoff issues.
    """
    data_type = period_window.get("data_type")
    
    # Calculate exclusive upper bounds
    baseline_exclusive_end = calculate_month_aligned_exclusive_end(
        period_window['baseline_start_time'],
        period_window['baseline_end_time'],
        data_type
    )
    comparison_exclusive_end = calculate_month_aligned_exclusive_end(
        period_window['start_time'],
        period_window['end_time'],
        data_type
    )
    
    # Format literals based on data_type
    baseline_start_lit = format_time_literal(period_window['baseline_start_time'], data_type)
    baseline_end_lit = format_time_literal(baseline_exclusive_end, data_type)
    comparison_start_lit = format_time_literal(period_window['start_time'], data_type)
    comparison_end_lit = format_time_literal(comparison_exclusive_end, data_type)
    
    return (
        f"(({time_expr} >= {baseline_start_lit} AND {time_expr} < {baseline_end_lit}) "
        f"OR ({time_expr} >= {comparison_start_lit} AND {time_expr} < {comparison_end_lit}))"
    )


def build_root_summary_query(
    catalog: SemanticCatalog,
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    period_window: PeriodWindow,
    primary_table: str,
) -> str:
    required_tables = determine_required_tables(metric, filters, [], period_window, catalog, primary_table)
    alias_map = build_alias_map(required_tables)
    from_clause = build_from_clause(catalog, required_tables, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    metric_sql = render_expression(metric["expr"], alias_map, catalog)
    where_clauses = [period_filter_sql(time_expr, period_window)]
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))
    period_case = period_case_sql(time_expr, period_window)

    return f"""
SELECT
  {period_case} AS period_bucket,
  {metric_sql} AS metric_value,
  COUNT(*) AS raw_row_count
{from_clause}
WHERE {' AND '.join(where_clauses)}
GROUP BY {period_case}
ORDER BY period_bucket
""".strip()


def build_period_extract_query(
    catalog: SemanticCatalog,
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    period_window: PeriodWindow,
    primary_table: str,
    period_label: str,
) -> str:
    if period_label not in {"baseline", "comparison"}:
        raise ValueError(f"Unknown period label: {period_label}")

    required_tables = determine_required_tables(metric, filters, [], period_window, catalog, primary_table)
    alias_map = build_alias_map(required_tables)
    from_clause = build_from_clause(catalog, required_tables, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    metric_sql = render_expression(metric["expr"], alias_map, catalog)
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]

    data_type = period_window.get("data_type")
    
    if period_label == "baseline":
        start_lit = format_time_literal(period_window['baseline_start_time'], data_type)
        exclusive_end = calculate_month_aligned_exclusive_end(
            period_window['baseline_start_time'],
            period_window['baseline_end_time'],
            data_type
        )
        end_lit = format_time_literal(exclusive_end, data_type)
        time_window_clause = f"{time_expr} >= {start_lit} AND {time_expr} < {end_lit}"
    else:
        start_lit = format_time_literal(period_window['start_time'], data_type)
        exclusive_end = calculate_month_aligned_exclusive_end(
            period_window['start_time'],
            period_window['end_time'],
            data_type
        )
        end_lit = format_time_literal(exclusive_end, data_type)
        time_window_clause = f"{time_expr} >= {start_lit} AND {time_expr} < {end_lit}"

    where_clauses = [time_window_clause]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))

    return f"""
SELECT
  {time_expr} AS incurred_month,
  {metric_sql} AS metric_value,
  COUNT(*) AS raw_row_count
{from_clause}
WHERE {' AND '.join(where_clauses)}
GROUP BY {time_expr}
ORDER BY incurred_month
""".strip()


def build_dimension_aggregate_query(
    catalog: SemanticCatalog,
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    period_window: PeriodWindow,
    primary_table: str,
    dimension_name: str,
) -> str:
    required_tables = determine_required_tables(metric, filters, [dimension_name], period_window, catalog, primary_table)
    alias_map = build_alias_map(required_tables)
    from_clause = build_from_clause(catalog, required_tables, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    metric_sql = render_expression(metric["expr"], alias_map, catalog)
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]

    field_def = resolve_field(catalog, dimension_name, preferred_tables=[primary_table]) or resolve_field(catalog, dimension_name)
    if field_def is None:
        raise KeyError(f"Unknown dimension field: {dimension_name}")
    dimension_expr = f"{alias_map[field_def['table_name']]}.{field_def['expr']}"
    dimension_sql = f"COALESCE(CAST({dimension_expr} AS VARCHAR), '<NULL>')"

    where_clauses = [period_filter_sql(time_expr, period_window)]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))
    period_case = period_case_sql(time_expr, period_window)

    return f"""
SELECT
  {dimension_sql} AS dimension_value,
  {period_case} AS period_bucket,
  {metric_sql} AS metric_value,
  COUNT(*) AS raw_row_count
{from_clause}
WHERE {' AND '.join(where_clauses)}
GROUP BY {dimension_sql}, {period_case}
ORDER BY dimension_value, period_bucket
""".strip()


def build_explainer_query(
    catalog: SemanticCatalog,
    metric_names: Sequence[str],
    filters: Sequence[FilterCondition],
    period_window: PeriodWindow,
    primary_table: str,
) -> str:
    if not metric_names:
        raise ValueError("Explainer query requires at least one metric name")

    metrics = [catalog["metrics_by_name"][metric_name] for metric_name in metric_names]
    required_table_set: Set[str] = {primary_table}
    for metric in metrics:
        for table_name in determine_required_tables(metric, filters, [], period_window, catalog, primary_table):
            required_table_set.add(table_name)
    required_tables = [primary_table] + [table_name for table_name in sorted(required_table_set) if table_name != primary_table]
    alias_map = build_alias_map(required_tables)
    from_clause = build_from_clause(catalog, required_tables, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    period_case = period_case_sql(time_expr, period_window)
    where_clauses = [period_filter_sql(time_expr, period_window)]
    preferred_tables = [primary_table] + [table for metric in metrics for table in _safe_list(metric.get("dependency_tables"))]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))

    rendered_metrics = []
    for metric_name in metric_names:
        metric = catalog["metrics_by_name"][metric_name]
        rendered_metrics.append(f"{render_expression(metric['expr'], alias_map, catalog)} AS {slugify(metric_name)}")

    return f"""
SELECT
  {period_case} AS period_bucket,
  {', '.join(rendered_metrics)}
{from_clause}
WHERE {' AND '.join(where_clauses)}
GROUP BY {period_case}
ORDER BY period_bucket
""".strip()


# =============================
# Data shaping and scoring
# =============================

def reshape_root_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float, Optional[float]]:
    normalized = normalize_dataframe_columns(df)
    if normalized.empty:
        empty = pd.DataFrame(
            [
                {"period_bucket": "baseline", "metric_value": 0.0, "raw_row_count": 0},
                {"period_bucket": "comparison", "metric_value": 0.0, "raw_row_count": 0},
            ]
        )
        return empty, 0.0, 0.0, 0.0, None

    metric_col = "metric_value"
    row_count_col = "raw_row_count"
    baseline_value = to_python_float(normalized.loc[normalized["period_bucket"] == "baseline", metric_col].sum())
    comparison_value = to_python_float(normalized.loc[normalized["period_bucket"] == "comparison", metric_col].sum())
    delta_value = comparison_value - baseline_value
    delta_pct = (delta_value / baseline_value) if baseline_value else None
    return normalized, baseline_value, comparison_value, delta_value, delta_pct


def pivot_dimension_comparison(
    dimension_df: pd.DataFrame,
    total_delta: float,
    parent_delta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = normalize_dataframe_columns(dimension_df)
    if normalized.empty:
        empty = pd.DataFrame(columns=["dimension_value", "metric_value", "raw_row_count"])
        delta_empty = pd.DataFrame(
            columns=[
                "dimension_value",
                "baseline_value",
                "comparison_value",
                "delta_value",
                "raw_row_count_baseline",
                "raw_row_count_comparison",
                "contribution_pct_total",
                "contribution_pct_parent",
                "aligned_delta",
                "aligned_contribution_pct_total",
                "aligned_contribution_pct_parent",
            ]
        )
        return empty, empty, delta_empty

    baseline = normalized.loc[normalized["period_bucket"] == "baseline", ["dimension_value", "metric_value", "raw_row_count"]].copy()
    comparison = normalized.loc[normalized["period_bucket"] == "comparison", ["dimension_value", "metric_value", "raw_row_count"]].copy()
    baseline.columns = ["dimension_value", "metric_value", "raw_row_count"]
    comparison.columns = ["dimension_value", "metric_value", "raw_row_count"]

    pivot = baseline.rename(columns={"metric_value": "baseline_value", "raw_row_count": "raw_row_count_baseline"}).merge(
        comparison.rename(columns={"metric_value": "comparison_value", "raw_row_count": "raw_row_count_comparison"}),
        on="dimension_value",
        how="outer",
    )
    for column in ["baseline_value", "comparison_value", "raw_row_count_baseline", "raw_row_count_comparison"]:
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce").fillna(0.0)

    pivot["delta_value"] = pivot["comparison_value"] - pivot["baseline_value"]
    pivot["aligned_delta"] = pivot["delta_value"].where(pivot["delta_value"].apply(sign) == sign(total_delta), 0.0)
    aligned_total_delta = float(pivot["aligned_delta"].sum())

    if total_delta:
        pivot["contribution_pct_total"] = pivot["delta_value"] / total_delta
        pivot["aligned_contribution_pct_total"] = pivot["aligned_delta"] / total_delta
    else:
        pivot["contribution_pct_total"] = 0.0
        pivot["aligned_contribution_pct_total"] = 0.0

    if aligned_total_delta:
        pivot["aligned_contribution_pct_of_aligned_delta"] = pivot["aligned_delta"] / aligned_total_delta
    else:
        pivot["aligned_contribution_pct_of_aligned_delta"] = 0.0

    if parent_delta:
        pivot["contribution_pct_parent"] = pivot["delta_value"] / parent_delta
        pivot["aligned_contribution_pct_parent"] = pivot["aligned_delta"] / parent_delta
    else:
        pivot["contribution_pct_parent"] = 0.0
        pivot["aligned_contribution_pct_parent"] = 0.0

    pivot = pivot.sort_values(
        by=["aligned_contribution_pct_total", "aligned_delta", "delta_value"],
        ascending=False,
    ).reset_index(drop=True)

    return baseline, comparison, pivot


def choose_best_segment(
    delta_df: pd.DataFrame,
    total_delta: float,
    parent_delta: float,
    min_row_count: int,
) -> Optional[pd.Series]:
    if delta_df.empty:
        return None

    filtered = delta_df.loc[
        (delta_df["raw_row_count_baseline"] >= min_row_count)
        | (delta_df["raw_row_count_comparison"] >= min_row_count)
    ].copy()
    if filtered.empty:
        filtered = delta_df.copy()

    aligned = filtered.loc[filtered["aligned_delta"].apply(sign) == sign(total_delta)]
    if not aligned.empty:
        aligned = aligned.sort_values(
            by=["aligned_contribution_pct_total", "aligned_delta", "contribution_pct_parent"],
            ascending=False,
        )
        return aligned.iloc[0]

    filtered = filtered.sort_values(by=["delta_value"], ascending=False)
    return filtered.iloc[0] if not filtered.empty else None


def _segment_value(value: Any) -> str:
    """Normalize raw dimension values for summary output."""
    if value is None:
        return "<NULL>"
    text = str(value).strip()
    return text if text else "<NULL>"


def _segment_summary_from_row(row: pd.Series, *, opposing_share: Optional[float] = None) -> SegmentSummary:
    """Convert a delta row into a serializable segment summary."""
    summary: SegmentSummary = {
        "value": _segment_value(row.get("dimension_value")),
        "baseline_value": to_python_float(row.get("baseline_value")),
        "comparison_value": to_python_float(row.get("comparison_value")),
        "delta_value": to_python_float(row.get("delta_value")),
        "contribution_pct_total": to_python_float(row.get("contribution_pct_total")),
        "contribution_pct_parent": to_python_float(row.get("contribution_pct_parent")),
        "aligned_delta": to_python_float(row.get("aligned_delta")),
        "aligned_contribution_pct_total": to_python_float(row.get("aligned_contribution_pct_total")),
        "aligned_contribution_pct_parent": to_python_float(row.get("aligned_contribution_pct_parent")),
        "aligned_contribution_pct_of_aligned_delta": to_python_float(
            row.get("aligned_contribution_pct_of_aligned_delta")
        ),
        "raw_row_count_baseline": to_python_float(row.get("raw_row_count_baseline")),
        "raw_row_count_comparison": to_python_float(row.get("raw_row_count_comparison")),
    }
    if opposing_share is not None:
        summary["opposing_share"] = opposing_share
    return summary


def _top_segment_values(node: PathNodeSummary) -> List[str]:
    """Extract the selected values from a drill-path node."""
    values: List[str] = []
    for segment in node.get("top_segments", []):
        value = str(segment.get("value") or "").strip()
        if value:
            values.append(value)
    return values


def _sum_segment_field(segments: Sequence[SegmentSummary], field: str) -> float:
    """Sum a numeric field across segment summaries."""
    return float(sum(to_python_float(segment.get(field)) for segment in segments))


def _select_top_segments(
    delta_df: pd.DataFrame,
    *,
    level: int,
    min_row_count: int,
    min_contribution_pct: float,
    min_incremental_gain_pct: float,
    top_k_limit: int,
) -> Tuple[List[SegmentSummary], float]:
    """Select top positive contributors using cumulative share and guardrails."""
    if delta_df.empty:
        return [], 0.0

    filtered = delta_df.loc[
        (delta_df["raw_row_count_baseline"] >= min_row_count)
        | (delta_df["raw_row_count_comparison"] >= min_row_count)
    ].copy()
    if filtered.empty:
        return [], 0.0

    positive_total_delta = float(filtered.loc[filtered["delta_value"] > 0, "delta_value"].sum())
    if positive_total_delta <= 0.0:
        return [], 0.0

    aligned = filtered.loc[filtered["delta_value"] > 0].copy()
    if aligned.empty:
        return [], 0.0

    aligned = aligned.loc[aligned["contribution_pct_total"].abs() >= min_contribution_pct]
    if level > 1:
        aligned = aligned.loc[aligned["contribution_pct_parent"].abs() >= min_incremental_gain_pct]
    if aligned.empty:
        return [], 0.0

    aligned = aligned.sort_values(by=["delta_value"], ascending=False)

    top_segments: List[SegmentSummary] = []
    cumulative_share = 0.0
    for _, row in aligned.iterrows():
        share = to_python_float(row.get("delta_value")) / positive_total_delta if positive_total_delta else 0.0
        if share <= 0:
            continue
        summary = _segment_summary_from_row(row)
        summary["aligned_delta"] = summary.get("delta_value", 0.0)
        summary["aligned_contribution_pct_total"] = summary.get("contribution_pct_total", 0.0)
        summary["aligned_contribution_pct_parent"] = summary.get("contribution_pct_parent", 0.0)
        summary["aligned_contribution_pct_of_aligned_delta"] = share
        top_segments.append(summary)
        cumulative_share += share
        if cumulative_share >= 0.80 or len(top_segments) >= top_k_limit:
            break

    return top_segments, cumulative_share


def _select_bottom_segments(
    delta_df: pd.DataFrame,
    *,
    max_segments: int,
) -> List[SegmentSummary]:
    """Select negative contributors by share of the negative change."""
    if delta_df.empty:
        return []

    opposite = delta_df.loc[delta_df["delta_value"] < 0].copy()
    if opposite.empty:
        return []

    negative_total_delta_abs = float(opposite["delta_value"].abs().sum())
    if negative_total_delta_abs == 0.0:
        return []

    opposite["opposing_share"] = opposite["delta_value"].abs() / negative_total_delta_abs
    opposite = opposite.sort_values(by=["opposing_share", "delta_value"], ascending=False)

    bottom_segments: List[SegmentSummary] = []
    cumulative_share = 0.0
    for _, row in opposite.iterrows():
        share = to_python_float(row.get("opposing_share"))
        bottom_segments.append(_segment_summary_from_row(row, opposing_share=share))
        cumulative_share += share
        if cumulative_share >= 0.80 or len(bottom_segments) >= max_segments:
            break

    return bottom_segments


def humanize_metric_name(metric_name: str) -> str:
    if "." in metric_name:
        metric_name = metric_name.split(".", 1)[1]
    return metric_name.replace("_", " ")


def humanize_dimension_name(dimension_name: str) -> str:
    return dimension_name.replace("_", " ")


def derive_dimension_label(field_def: Optional[FieldDefinition], fallback_name: str) -> str:
    if field_def:
        label = str(field_def.get("label") or "").strip()
        if label:
            return label
        description = str(field_def.get("description") or "").strip()
        if description:
            first_sentence = description.split(".", 1)[0].strip()
            if len(first_sentence) > 60:
                shortened = re.split(r"\s+(?:is|are)\s+", first_sentence, flags=re.IGNORECASE)
                candidate = shortened[0].strip() if shortened else ""
                if candidate:
                    return candidate
            if first_sentence:
                return first_sentence
    return humanize_dimension_name(fallback_name)


def resolve_dimension_label(catalog: SemanticCatalog, table_name: str, dimension_name: str) -> str:
    field_def = catalog["table_fields"].get(table_name, {}).get(dimension_name)
    return derive_dimension_label(field_def, dimension_name)


def format_value(metric_name: str, value: float) -> str:
    metric_lower = metric_name.lower()
    abs_value = abs(value)

    if any(token in metric_lower for token in ["rate", "pct", "ratio"]):
        return f"{value * 100:.1f}%"

    if any(token in metric_lower for token in ["paid", "allowed", "billed", "pmpm", "avg"]):
        if abs_value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        if abs_value >= 1_000:
            return f"${value / 1_000:.1f}K"
        return f"${value:,.0f}"

    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def folder_token_for_dimension(dimension_name: str) -> str:
    return DIMENSION_FOLDER_ALIASES.get(dimension_name, slugify(dimension_name))


def build_nested_path(
    base_dir: Path,
    drill_path: Sequence[Dict[str, Any]],
    dimension_name: str,
    filename: str
) -> Path:
    """
    Build nested directory path for drill-down levels.
    
    Args:
        base_dir: Base directory (queries_dir or aggregates_dir)
        drill_path: Current drill-down path from root
        dimension_name: Current dimension being analyzed
        filename: File name (e.g., "health_service_code.sql")
    
    Returns:
        Full path with nested structure
    
    Example:
        drill_path = [
            {"dimension": "place_of_service_code", ...},
            {"dimension": "health_service_code", ...}
        ]
        dimension_name = "claim_its_host_code"
        
        Returns: base_dir/level_3/place_of_service_code/health_service_code/claim_its_host_code.sql
    """
    if not drill_path:
        level_dir = base_dir / "level_1"
        return level_dir / filename
    
    level_num = len(drill_path) + 1
    level_dir = base_dir / f"level_{level_num}"
    
    for node in drill_path:
        parent_folder = folder_token_for_dimension(node["dimension"])
        level_dir = level_dir / parent_folder
    
    return level_dir / filename


def build_scope_label(filters: Sequence[FilterCondition]) -> str:
    procedure_values: List[str] = []
    for filter_condition in filters:
        if filter_condition["field"] != "procedure_code":
            continue
        operator = str(filter_condition.get("operator", "")).lower().strip()
        raw_value = filter_condition.get("value")
        if operator in {"=", "in"}:
            if isinstance(raw_value, (list, tuple, set)):
                procedure_values = [str(item).strip() for item in raw_value if str(item).strip()]
            elif isinstance(raw_value, str):
                procedure_values = [item.strip() for item in raw_value.split(",") if item.strip()]
            elif raw_value is not None:
                procedure_values = [str(raw_value).strip()]
        if procedure_values:
            break

    if procedure_values:
        return f"For CPT {', '.join(procedure_values)} in the scoped population"
    return "For the scoped population"


def format_dimension_context(node: PathNodeSummary) -> str:
    label = str(node.get("dimension_label") or node.get("dimension") or "").strip()
    values = _top_segment_values(node)
    if not label and not values:
        return "<UNKNOWN>"
    if not label:
        return ", ".join(values) if values else "<UNKNOWN>"
    if not values:
        return label
    return f"{label} in [{', '.join(values)}]"


def build_context_phrase(path_nodes: Sequence[PathNodeSummary]) -> str:
    values = [format_dimension_context(node) for node in path_nodes]
    values = [value for value in values if value and value != "<UNKNOWN>"]
    if not values:
        return "the scoped population"
    return " / ".join(values)


CLAIM_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "claim_count",
    "claims_count",
    "total_claims",
)
ADMISSION_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "total_admissions",
    "admission_count",
    "admit_count",
    "admissions",
)
PAID_PER_ADMIT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "avg_paid_per_admit",
    "paid_per_admit",
    "paid_per_admission",
    "average_paid_per_admit",
)


def _format_signed_value(metric_name: str, value: float) -> str:
    """Format a metric delta with an explicit sign prefix."""
    formatted = format_value(metric_name, abs(value))
    if value > 0:
        return f"+{formatted}"
    if value < 0:
        return f"-{formatted}"
    return formatted


def _format_signed_pct(value: Optional[float]) -> Optional[str]:
    """Format a percentage delta with an explicit sign prefix."""
    if value is None:
        return None
    prefix = "+" if value > 0 else "-" if value < 0 else ""
    return f"{prefix}{format_pct(abs(value))}"


def _extract_explainer_root_records(explainer_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return explainer root records as a list of dictionaries."""
    payload = _safe_dict(explainer_payload)
    records = _safe_list(payload.get("root"))
    return [record for record in records if isinstance(record, dict)]


def _select_explainer_metric_key(
    records: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[str],
) -> Optional[str]:
    """Pick the first matching metric key from explainer records."""
    if not records:
        return None
    available_keys = {str(key) for record in records for key in record.keys()}
    normalized_candidates = [slugify(candidate) for candidate in candidate_metrics]
    for candidate in normalized_candidates:
        if candidate in available_keys:
            return candidate
    for candidate in normalized_candidates:
        for key in available_keys:
            if key.endswith(candidate) or candidate in key:
                return key
    return None


def _find_period_metric_value(
    records: Sequence[Mapping[str, Any]],
    period_bucket: str,
    metric_key: str,
) -> Optional[Any]:
    """Fetch a metric value for a given period bucket from explainer records."""
    for record in records:
        bucket = str(record.get("period_bucket") or "").strip().lower()
        if bucket == period_bucket:
            return record.get(metric_key)
    return None


def _extract_explainer_metric_change(
    explainer_payload: Mapping[str, Any],
    candidate_metrics: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Extract baseline/comparison/delta values for an explainer metric."""
    records = _extract_explainer_root_records(explainer_payload)
    metric_key = _select_explainer_metric_key(records, candidate_metrics)
    if not metric_key:
        return None
    baseline_raw = _find_period_metric_value(records, "baseline", metric_key)
    comparison_raw = _find_period_metric_value(records, "comparison", metric_key)
    if baseline_raw is None and comparison_raw is None:
        return None
    baseline_value = to_python_float(baseline_raw)
    comparison_value = to_python_float(comparison_raw)
    delta_value = comparison_value - baseline_value
    delta_pct = (delta_value / baseline_value) if baseline_value else None
    return {
        "metric_key": metric_key,
        "baseline": baseline_value,
        "comparison": comparison_value,
        "delta": delta_value,
        "delta_pct": delta_pct,
    }


def _format_metric_change_summary(label: str, metric_name: str, change: Mapping[str, Any]) -> str:
    """Format a structured metric change summary line."""
    baseline = format_value(metric_name, to_python_float(change.get("baseline")))
    comparison = format_value(metric_name, to_python_float(change.get("comparison")))
    delta_text = _format_signed_value(metric_name, to_python_float(change.get("delta")))
    delta_pct_text = _format_signed_pct(change.get("delta_pct"))
    if delta_pct_text:
        return f"{label}: {baseline} -> {comparison} ({delta_text}, {delta_pct_text})."
    return f"{label}: {baseline} -> {comparison} ({delta_text})."


def _format_metric_movement(label: str, metric_name: str, change: Mapping[str, Any]) -> str:
    """Format a natural-language metric movement phrase."""
    baseline = format_value(metric_name, to_python_float(change.get("baseline")))
    comparison = format_value(metric_name, to_python_float(change.get("comparison")))
    delta_text = _format_signed_value(metric_name, to_python_float(change.get("delta")))
    delta_pct_text = _format_signed_pct(change.get("delta_pct"))
    suffix = f" ({delta_text}{f', {delta_pct_text}' if delta_pct_text else ''})"
    return f"{label} moved from {baseline} to {comparison}{suffix}"


def _join_phrases(phrases: Sequence[str]) -> str:
    """Join phrases with commas and an Oxford-style conjunction."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def _format_segment_summary(
    metric_name: str,
    segment: SegmentSummary,
    *,
    share_field: str,
) -> Optional[str]:
    """Format a short segment summary with delta and share."""
    value = str(segment.get("value") or "").strip()
    if not value:
        return None
    delta_value = to_python_float(segment.get("delta_value"))
    delta_text = _format_signed_value(metric_name, delta_value)
    share_value = to_python_float(segment.get(share_field))
    share_text = format_pct(abs(share_value)) if share_value else None
    if share_text:
        return f"{value} ({delta_text}, {share_text} of delta)"
    return f"{value} ({delta_text})"


def _format_segment_list(
    metric_name: str,
    segments: Sequence[SegmentSummary],
    *,
    share_field: str,
    limit: int = 2,
) -> List[str]:
    """Return formatted segment summaries limited to the top entries."""
    formatted: List[str] = []
    for segment in list(segments)[:limit]:
        summary = _format_segment_summary(metric_name, segment, share_field=share_field)
        if summary:
            formatted.append(summary)
    return formatted


def _find_path_node_by_dimension(
    drill_path: Sequence[PathNodeSummary],
    dimension_name: str,
) -> Optional[PathNodeSummary]:
    """Find a drill-path node by its dimension name."""
    target = _normalize_dimension_name(dimension_name).lower().strip()
    for node in drill_path:
        node_name = _normalize_dimension_name(str(node.get("dimension") or "")).lower().strip()
        if node_name and node_name == target:
            return node
    return None


def _dimension_role_label(dimension_name: str) -> str:
    """Map a dimension name to a semantic role label."""
    normalized = normalize(_normalize_dimension_name(dimension_name))
    tokens = {token for token in normalized.split("_") if token}

    if "authorization" in tokens or "auth" in tokens or "pa" in tokens or "prior_authorization" in normalized:
        return "authorization concentration"
    if any(token in tokens for token in {"product", "plan", "benefit"}):
        return "product/plan concentration"
    if "place_of_service" in normalized or any(token in tokens for token in {"facility", "pos"}):
        return "site-of-care concentration"
    if any(token in tokens for token in {"state", "region", "market", "county", "geo", "geography"}):
        return "geographic concentration"
    if any(token in tokens for token in {"diagnosis", "dx", "condition", "clinical"}):
        return "downstream clinical pocket"
    if any(token in tokens for token in {"provider", "hospital", "system", "rendering"}):
        return "provider/facility-system concentration"
    if any(token in tokens for token in {"group", "segment", "lob", "business", "mbu"}):
        return "business segment concentration"
    return "drill-path contributor"


def _segment_share_details(segment: SegmentSummary, share_fields: Sequence[Tuple[str, str]]) -> List[str]:
    details: List[str] = []
    for field_name, label in share_fields:
        share_value = to_python_float(segment.get(field_name))
        if share_value:
            details.append(f"{label} {format_pct(abs(share_value))}")
    return details


def _format_segment_detail(
    metric_name: str,
    segment: SegmentSummary,
    *,
    share_fields: Sequence[Tuple[str, str]],
) -> Optional[str]:
    value = str(segment.get("value") or "").strip()
    if not value:
        return None
    delta_text = _format_signed_value(metric_name, to_python_float(segment.get("delta_value")))
    details = [delta_text]
    details.extend(_segment_share_details(segment, share_fields))
    return f"{value} ({', '.join(details)})"


def _format_top_segments_for_dimension(
    metric_name: str,
    node: PathNodeSummary,
    *,
    limit: int = 3,
) -> List[str]:
    """Format top segments with delta and share details."""
    formatted: List[str] = []
    share_fields = (
        ("aligned_contribution_pct_of_aligned_delta", "share of positive delta"),
        ("aligned_contribution_pct_parent", "share of parent"),
        ("contribution_pct_total", "share of total"),
    )
    for segment in list(node.get("top_segments", []))[:limit]:
        summary = _format_segment_detail(metric_name, segment, share_fields=share_fields)
        if summary:
            formatted.append(summary)
    return formatted


def _format_bottom_segments_for_dimension(
    metric_name: str,
    node: PathNodeSummary,
    *,
    limit: int = 3,
) -> List[str]:
    """Format bottom segments with delta and share details."""
    formatted: List[str] = []
    share_fields = (
        ("opposing_share", "share of negative delta"),
        ("contribution_pct_parent", "share of parent"),
        ("contribution_pct_total", "share of total"),
    )
    for segment in list(node.get("bottom_segments", []))[:limit]:
        summary = _format_segment_detail(metric_name, segment, share_fields=share_fields)
        if summary:
            formatted.append(summary)
    return formatted


def _primary_segment(segments: Sequence[SegmentSummary]) -> Optional[SegmentSummary]:
    if not segments:
        return None
    return max(segments, key=lambda segment: abs(to_python_float(segment.get("delta_value"))))


def _segment_share_value(segment: Optional[SegmentSummary]) -> float:
    if not segment:
        return 0.0
    candidate_fields = [
        "aligned_contribution_pct_of_aligned_delta",
        "aligned_contribution_pct_parent",
        "contribution_pct_parent",
        "contribution_pct_total",
        "opposing_share",
    ]
    shares = [abs(to_python_float(segment.get(field))) for field in candidate_fields]
    return max(shares) if shares else 0.0


def _is_material_segment(segment: Optional[SegmentSummary], *, threshold: float = 0.1) -> bool:
    if not segment:
        return False
    return _segment_share_value(segment) >= threshold


def _no_auth_value(value: str) -> bool:
    normalized = normalize(value)
    tokens = {token for token in normalized.split("_") if token}
    if normalized in {"n", "no", "none", "na", "n_a", "false", "0"}:
        return True
    if "not" in tokens and ("required" in tokens or "require" in tokens):
        return True
    if "no" in tokens and any(token in tokens for token in {"auth", "authorization", "pa"}):
        return True
    if "not_required" in normalized:
        return True
    return False


def _format_authorization_detail(metric_name: str, node: PathNodeSummary) -> Optional[str]:
    segment = _primary_segment(node.get("top_segments", []))
    if not segment:
        return None
    raw_value = str(segment.get("value") or "").strip()
    if not raw_value:
        return None
    value_label = "claims not requiring prior authorization" if _no_auth_value(raw_value) else f"authorization category {raw_value}"
    delta_text = _format_signed_value(metric_name, to_python_float(segment.get("delta_value")))
    return f"{value_label} ({delta_text})"


def _build_drill_path_detail_lines(
    drill_path: Sequence[PathNodeSummary],
    metric_name: str,
) -> List[str]:
    lines: List[str] = []
    for node in drill_path:
        dimension_name = str(node.get("dimension") or "").strip()
        dimension_label = str(node.get("dimension_label") or dimension_name).strip()
        if not dimension_name and not dimension_label:
            continue
        role_label = _dimension_role_label(dimension_name)
        lines.append(
            f"{role_label} — dimension: {dimension_name or '<unknown>'} | label: {dimension_label or '<unknown>'}"
        )
        top_segments = _format_top_segments_for_dimension(metric_name, node, limit=3)
        if top_segments:
            lines.append(f"  - Top segments: {', '.join(top_segments)}")
        bottom_segments = _format_bottom_segments_for_dimension(metric_name, node, limit=3)
        if bottom_segments:
            lines.append(f"  - Bottom/offset segments: {', '.join(bottom_segments)}")
    return lines


def _find_nodes_by_role(
    drill_path: Sequence[PathNodeSummary],
    role_label: str,
) -> List[PathNodeSummary]:
    return [node for node in drill_path if _dimension_role_label(str(node.get("dimension") or "")) == role_label]


ROLE_HIGHLIGHT_ORDER: Tuple[str, ...] = (
    "authorization concentration",
    "product/plan concentration",
    "business segment concentration",
    "site-of-care concentration",
    "geographic concentration",
    "provider/facility-system concentration",
)


def _node_primary_delta(node: PathNodeSummary) -> float:
    segment = _primary_segment(node.get("top_segments", []))
    if not segment:
        return 0.0
    return abs(to_python_float(segment.get("delta_value")))


def _select_role_node(
    drill_path: Sequence[PathNodeSummary],
    role_label: str,
    *,
    deepest: bool = False,
) -> Optional[PathNodeSummary]:
    nodes = _find_nodes_by_role(drill_path, role_label)
    if not nodes:
        return None
    if deepest:
        return nodes[-1]
    return max(nodes, key=_node_primary_delta)


def _collect_role_highlights(
    drill_path: Sequence[PathNodeSummary],
    metric_name: str,
) -> List[Dict[str, Any]]:
    highlights: List[Dict[str, Any]] = []
    for role_label in ROLE_HIGHLIGHT_ORDER:
        node = _select_role_node(drill_path, role_label)
        if not node:
            continue
        primary_segment = _primary_segment(node.get("top_segments", []))
        if not primary_segment:
            continue
        if role_label != "authorization concentration" and not _is_material_segment(primary_segment):
            continue
        dimension_name = str(node.get("dimension") or "").strip()
        dimension_label = str(node.get("dimension_label") or dimension_name).strip()
        segments = _format_top_segments_for_dimension(metric_name, node, limit=3)
        if not segments:
            continue
        highlight: Dict[str, Any] = {
            "role": role_label,
            "dimension": dimension_name,
            "dimension_label": dimension_label,
            "segments": segments,
            "primary_segment": primary_segment,
            "node": node,
        }
        if role_label == "authorization concentration":
            highlight["authorization_detail"] = _format_authorization_detail(metric_name, node)
        highlights.append(highlight)
    return highlights


def _clinical_pocket_detail(metric_name: str, drill_path: Sequence[PathNodeSummary]) -> Optional[Dict[str, Any]]:
    node = _select_role_node(drill_path, "downstream clinical pocket", deepest=True)
    if not node:
        return None
    segment = _primary_segment(node.get("top_segments", []))
    if not segment:
        return None
    value = str(segment.get("value") or "").strip()
    if not value:
        return None
    dimension_name = str(node.get("dimension") or "").strip()
    dimension_label = str(node.get("dimension_label") or dimension_name).strip()
    delta_text = _format_signed_value(metric_name, to_python_float(segment.get("delta_value")))
    return {
        "dimension": dimension_name,
        "dimension_label": dimension_label,
        "value": value,
        "delta_text": delta_text,
    }


def _clinical_pocket_sentence(metric_name: str, drill_path: Sequence[PathNodeSummary]) -> Optional[str]:
    detail = _clinical_pocket_detail(metric_name, drill_path)
    if not detail:
        return None
    value = detail["value"]
    delta_text = detail["delta_text"]
    return (
        f"A deeper clinical drill-down identified {value} as a downstream contributor "
        f"({delta_text}), warranting follow-up on related cost drivers."
    )


def _concentration_phrase(metric_name: str, node: PathNodeSummary) -> Optional[str]:
    role = _dimension_role_label(str(node.get("dimension") or ""))
    dimension_label = str(node.get("dimension_label") or node.get("dimension") or "").strip()
    if role == "authorization concentration":
        auth_detail = _format_authorization_detail(metric_name, node)
        if auth_detail:
            return f"{role} in {auth_detail}"
    segments = _format_top_segments_for_dimension(metric_name, node, limit=3)
    if not segments:
        return None
    if not dimension_label:
        return f"{role} ({', '.join(segments)})"
    return f"{role} in {dimension_label} ({', '.join(segments)})"


def _rank_contributor_nodes(drill_path: Sequence[PathNodeSummary]) -> List[PathNodeSummary]:
    candidates: List[PathNodeSummary] = []
    for node in drill_path:
        role = _dimension_role_label(str(node.get("dimension") or ""))
        if role == "downstream clinical pocket":
            continue
        segment = _primary_segment(node.get("top_segments", []))
        if not segment:
            continue
        if role != "authorization concentration" and not _is_material_segment(segment):
            continue
        candidates.append(node)
    return sorted(candidates, key=_node_primary_delta, reverse=True)


def build_executive_summary_input(
    filters: Sequence[FilterCondition],
    metric_name: str,
    baseline_value: float,
    comparison_value: float,
    delta_value: float,
    delta_pct: Optional[float],
    drill_path: Sequence[PathNodeSummary],
    explainer_payload: Mapping[str, Any],
) -> str:
    """Build a structured input for executive summary generation."""
    metric_label = humanize_metric_name(metric_name)
    scope_label = build_scope_label(filters)
    delta_text = _format_signed_value(metric_name, delta_value)
    delta_pct_text = _format_signed_pct(delta_pct)
    delta_details = f"{delta_text}{f', {delta_pct_text}' if delta_pct_text else ''}"
    total_line = (
        f"{scope_label}, total {metric_label}: {format_value(metric_name, baseline_value)} -> "
        f"{format_value(metric_name, comparison_value)} ({delta_details})."
    )

    lines = [total_line]

    claim_change = _extract_explainer_metric_change(explainer_payload, CLAIM_COUNT_METRIC_CANDIDATES)
    admission_change = _extract_explainer_metric_change(explainer_payload, ADMISSION_COUNT_METRIC_CANDIDATES)
    paid_per_admit_change = _extract_explainer_metric_change(explainer_payload, PAID_PER_ADMIT_METRIC_CANDIDATES)

    volume_lines: List[str] = []
    if claim_change:
        volume_lines.append(_format_metric_change_summary("Claim volume", "claim_count", claim_change))
    if admission_change:
        volume_lines.append(_format_metric_change_summary("Admissions", "admission_count", admission_change))
    if admission_change and paid_per_admit_change:
        volume_lines.append(
            _format_metric_change_summary("Paid per admission", "avg_paid_per_admit", paid_per_admit_change)
        )
    if volume_lines:
        lines.append("Root volume/intensity metrics:")
        lines.extend([f"- {line}" for line in volume_lines])

    role_highlights = _collect_role_highlights(drill_path, metric_name)
    if role_highlights:
        lines.append("Role highlights:")
        for highlight in role_highlights:
            role = highlight["role"]
            dimension_name = highlight.get("dimension") or "<unknown>"
            dimension_label = highlight.get("dimension_label") or "<unknown>"
            segments = ", ".join(highlight.get("segments", []))
            lines.append(f"- {role}: {dimension_label} ({dimension_name}) -> {segments}")
            auth_detail = highlight.get("authorization_detail")
            if auth_detail:
                lines.append(f"  - Authorization detail: {auth_detail}")

    drill_details = _build_drill_path_detail_lines(drill_path, metric_name)
    if drill_details:
        lines.append("Drill-path detail by dimension:")
        lines.extend([f"- {line}" for line in drill_details])

    clinical_detail = _clinical_pocket_detail(metric_name, drill_path)
    if clinical_detail:
        lines.append(
            "Clinical pocket detail: "
            f"{clinical_detail['dimension_label']} ({clinical_detail['dimension']}) -> "
            f"{clinical_detail['value']} ({clinical_detail['delta_text']})"
        )

    return "\n".join(lines)


def normalize_executive_summary(text: str, *, max_sentences: int = 4) -> str:
    """Normalize the LLM response to a concise multi-sentence summary."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
    if not sentences:
        return ""
    summary = " ".join(sentences[:max_sentences]).strip()
    if summary and not summary.endswith((".", "!", "?")):
        summary = f"{summary}."
    return summary


def generate_executive_summary(
    llm: Any,
    input_text: str,
    ehap: Optional[Any] = None,
    llm_reinitializer: Optional[Any] = None,
) -> tuple[Optional[str], Dict[str, Any]]:
    """Generate an executive-ready summary using the configured LLM with token retry support."""
    messages = [
        {"role": "system", "content": EXECUTIVE_SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXECUTIVE_SUMMARY_USER_PROMPT_TEMPLATE.format(input_text=input_text),
        },
    ]
    try:
        # Use structured_llm_invoke with token retry support if ehap is available
        if ehap is not None and llm_reinitializer is not None:
            from deep_research_utils.ehap_retry import structured_llm_invoke
            
            result_schema, _ = structured_llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                schema=ExecutiveSummarySchema,
                llm_reinitializer=llm_reinitializer,
            )
            # Token usage tracking not available with retry utility
            input_tokens, output_tokens = 0, 0
        else:
            # Fallback to old behavior for backward compatibility
            try:
                structured_llm = llm.with_structured_output(ExecutiveSummarySchema, include_raw=True)
                response = structured_llm.invoke(messages)
                raw_response = response.get("raw") if isinstance(response, dict) else response
            except TypeError:
                structured_llm = llm.with_structured_output(ExecutiveSummarySchema)
                response = structured_llm.invoke(messages)
                raw_response = response
            input_tokens, output_tokens = _extract_token_usage(raw_response)
            if isinstance(response, dict) and "parsed" in response:
                result_schema = response.get("parsed") or ExecutiveSummarySchema()
            else:
                result_schema = response
        content = str(result_schema.summary or "").strip()
        if not content:
            return None, correlation_llm_tokens_for_step("executive_summary", input_tokens, output_tokens)
        normalized = normalize_executive_summary(content)
        return normalized or None, correlation_llm_tokens_for_step("executive_summary", input_tokens, output_tokens)
    except Exception as exc:
        logger.warning("Executive summary LLM call failed; using deterministic narrative.", exc_info=exc)
        return None, empty_correlation_llm_tokens()


def _build_fallback_executive_summary(
    filters: Sequence[FilterCondition],
    metric_name: str,
    baseline_value: float,
    comparison_value: float,
    delta_value: float,
    delta_pct: Optional[float],
    drill_path: Sequence[PathNodeSummary],
    explainer_payload: Mapping[str, Any],
) -> str:
    metric_label = humanize_metric_name(metric_name)
    scope_label = build_scope_label(filters)
    delta_text = _format_signed_value(metric_name, delta_value)
    delta_pct_text = _format_signed_pct(delta_pct)
    delta_details = f"{delta_text}{f', {delta_pct_text}' if delta_pct_text else ''}"
    sentences = [
        (
            f"{scope_label}, {metric_label} moved from {format_value(metric_name, baseline_value)} "
            f"to {format_value(metric_name, comparison_value)} ({delta_details})."
        )
    ]

    claim_change = _extract_explainer_metric_change(explainer_payload, CLAIM_COUNT_METRIC_CANDIDATES)
    admission_change = _extract_explainer_metric_change(explainer_payload, ADMISSION_COUNT_METRIC_CANDIDATES)
    paid_per_admit_change = _extract_explainer_metric_change(explainer_payload, PAID_PER_ADMIT_METRIC_CANDIDATES)

    volume_phrases: List[str] = []
    if claim_change:
        volume_phrases.append(_format_metric_movement("Claim volume", "claim_count", claim_change))
    if admission_change:
        volume_phrases.append(_format_metric_movement("Admissions", "admission_count", admission_change))
    if admission_change and paid_per_admit_change:
        volume_phrases.append(
            _format_metric_movement("Paid per admission", "avg_paid_per_admit", paid_per_admit_change)
        )
    if volume_phrases:
        sentences.append(f"Volume/intensity context: {_join_phrases(volume_phrases)}.")

    ranked_nodes = _rank_contributor_nodes(drill_path)
    if ranked_nodes:
        primary_phrase = _concentration_phrase(metric_name, ranked_nodes[0])
        secondary_phrase = None
        if len(ranked_nodes) > 1:
            secondary_phrase = _concentration_phrase(metric_name, ranked_nodes[1])
        auth_node = _select_role_node(drill_path, "authorization concentration")
        auth_phrase = None
        if auth_node and auth_node not in ranked_nodes[:2]:
            auth_phrase = _concentration_phrase(metric_name, auth_node)

        if primary_phrase and secondary_phrase:
            sentence = f"The change was concentrated in {primary_phrase}, with additional concentration in {secondary_phrase}."
        elif primary_phrase:
            sentence = f"The change was concentrated in {primary_phrase}."
        else:
            sentence = ""

        if sentence:
            if auth_phrase and auth_phrase not in sentence:
                sentence = f"{sentence} Authorization detail pointed to {auth_phrase}."
            sentences.append(sentence.strip())
        elif auth_phrase:
            sentences.append(f"Authorization detail pointed to {auth_phrase}.")

    clinical_sentence = _clinical_pocket_sentence(metric_name, drill_path)
    if clinical_sentence:
        sentences.append(clinical_sentence)
    else:
        offset_detail = None
        offset_node = None
        for node in drill_path:
            segment = _primary_segment(node.get("bottom_segments", []))
            if not segment:
                continue
            if offset_detail is None or abs(to_python_float(segment.get("delta_value"))) > abs(
                to_python_float(offset_detail.get("delta_value"))
            ):
                offset_detail = segment
                offset_node = node
        if offset_detail is not None and offset_node is not None and _is_material_segment(offset_detail):
            dimension_label = str(offset_node.get("dimension_label") or offset_node.get("dimension") or "").strip()
            value = str(offset_detail.get("value") or "").strip()
            delta_text = _format_signed_value(metric_name, to_python_float(offset_detail.get("delta_value")))
            if dimension_label and value:
                sentences.append(
                    f"Offsetting declines were observed in {dimension_label} ({value}, {delta_text})."
                )

    return " ".join(sentences[:4])


def _segment_readout(metric_name: str, segment: SegmentSummary, *, is_offset: bool) -> str:
    """Generate a deterministic business readout for a segment."""
    delta_value = to_python_float(segment.get("delta_value"))
    direction = "increase" if delta_value >= 0 else "decrease"
    impact = format_value(metric_name, abs(delta_value))
    if is_offset:
        return f"{direction.title()} of {impact} contributing to the negative change."
    return f"{direction.title()} of {impact} contributing to the positive change."


def build_narrative_summary(
    filters: Sequence[FilterCondition],
    metric_name: str,
    delta_value: float,
    drill_path: Sequence[PathNodeSummary],
) -> str:
    metric_label = humanize_metric_name(metric_name)
    direction = "increased" if delta_value >= 0 else "decreased"
    scope_label = build_scope_label(filters)
    lines = [
        f"{scope_label}. {metric_label} {direction} by {format_value(metric_name, abs(delta_value))} versus prior year.",
    ]

    if not drill_path:
        return "\n".join(lines)

    lines.append("")
    lines.append("Waterfall path:")
    for index, node in enumerate(drill_path, start=1):
        dimension_label = str(node.get("dimension_label") or node.get("dimension") or "").strip()
        level_label = node.get("level") or index
        parent_context = "the total population" if index == 1 else build_context_phrase(drill_path[: index - 1])
        top_segments = node.get("top_segments", [])
        bottom_segments = node.get("bottom_segments", [])
        top_share = _sum_segment_field(top_segments, "aligned_contribution_pct_of_aligned_delta")
        bottom_share = _sum_segment_field(bottom_segments, "opposing_share")

        lines.append(f"- Level {level_label} — {dimension_label}")
        lines.append(
            f"  - Within {parent_context}, the following contributors explain {format_pct(abs(top_share))} of the positive change:"
        )
        lines.append("  - Top contributors:")
        for segment_index, segment in enumerate(top_segments, start=1):
            lines.append(f"    {segment_index}. {segment.get('value')}")
            lines.append(
                f"       - Impact: {format_value(metric_name, to_python_float(segment.get('delta_value')))}"
            )
            lines.append(
                "       - Share of positive change: "
                f"{format_pct(abs(to_python_float(segment.get('aligned_contribution_pct_of_aligned_delta'))))}"
            )
            if index == 1:
                lines.append(
                    "       - Share of total net change: "
                    f"{format_pct(abs(to_python_float(segment.get('contribution_pct_total'))))}"
                )
            else:
                lines.append(
                    "       - Share of parent change: "
                    f"{format_pct(abs(to_python_float(segment.get('contribution_pct_parent'))))}"
                )
            lines.append(f"       - Business readout: {_segment_readout(metric_name, segment, is_offset=False)}")

        if bottom_segments:
            lines.append("  - Bottom contributors / declines:")
            lines.append(
                f"    - Negative segments explain {format_pct(abs(bottom_share))} of the negative change:"
            )
            for segment_index, segment in enumerate(bottom_segments, start=1):
                lines.append(f"    {segment_index}. {segment.get('value')}")
                lines.append(
                    f"       - Impact: {format_value(metric_name, to_python_float(segment.get('delta_value')))}"
                )
                lines.append(
                    "       - Share of negative change: "
                    f"{format_pct(abs(to_python_float(segment.get('opposing_share'))))}"
                )
                if index == 1:
                    lines.append(
                        "       - Share of total net change: "
                        f"{format_pct(abs(to_python_float(segment.get('contribution_pct_total'))))}"
                    )
                else:
                    lines.append(
                        "       - Share of parent change: "
                        f"{format_pct(abs(to_python_float(segment.get('contribution_pct_parent'))))}"
                    )
                lines.append(f"       - Business readout: {_segment_readout(metric_name, segment, is_offset=True)}")

    return "\n".join(lines)
# Candidate dimensions and execution
# =============================

def get_candidate_dimensions(
    intent: IntentPayload,
    catalog: SemanticCatalog,
    primary_table: str,
) -> List[str]:
    """Return candidate drill-down dimensions for the requested analysis."""
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    period_dict = _safe_dict(mode_parameters.get("period"))
    time_dimension_name = _normalize_dimension_name(
        str(period_dict.get("time_dimension") or period_dict.get("rolling_time_dimension") or "").strip()
    )
    exclude_if_filtered = bool(mode_parameters.get("exclude_if_filtered", True))
    allowed_dimensions = [
        _normalize_dimension_name(str(item).strip())
        for item in _safe_list(mode_parameters.get("drill_dimensions"))
        if str(item).strip()
    ]
    allowed_set = set(allowed_dimensions)
    filtered_fields = {filter_condition["field"] for filter_condition in intent.get("filters", [])}
    candidates: List[str] = []
    primary_dimensions = {
        field_name
        for field_name, field_def in catalog["table_fields"].get(primary_table, {}).items()
        if field_def.get("kind") == "dimension"
    }

    for field_name, field_def in catalog["table_fields"].get(primary_table, {}).items():
        if field_def.get("kind") != "dimension":
            continue
        if field_name == "claim_number":
            continue
        if time_dimension_name and field_name == time_dimension_name:
            continue
        if exclude_if_filtered and field_name in filtered_fields:
            continue
        candidates.append(field_name)

    if allowed_set:
        candidates = [candidate for candidate in candidates if candidate in allowed_set]
        missing = sorted(allowed_set - primary_dimensions)
        if missing:
            raise AgentConfigurationError(
                "Requested drill_dimensions are not present on primary table "
                f"'{primary_table}': {', '.join(missing)}"
            )

    unique_candidates: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def extract_columns_from_expression(expr: str) -> Set[str]:
    """
    Extract column names from a SQL expression.
    
    Handles common SQL patterns like:
    - Simple columns: column_name
    - Function calls: SUM(column_name)
    - Arithmetic: column1 + column2
    - CASE statements
    
    Args:
        expr: SQL expression string
    
    Returns:
        Set of column names found in expression
    """
    columns = set()
    
    # Remove common SQL keywords and functions
    expr_cleaned = re.sub(r'\b(SUM|AVG|COUNT|MIN|MAX|COALESCE|CAST|CASE|WHEN|THEN|ELSE|END|AS|AND|OR|NOT)\b', '', expr, flags=re.IGNORECASE)
    
    # Remove operators and parentheses
    expr_cleaned = re.sub(r'[+\-*/()=<>!,]', ' ', expr_cleaned)
    
    # Remove string literals in single quotes
    expr_cleaned = re.sub(r"'[^']*'", '', expr_cleaned)
    
    # Remove numeric literals
    expr_cleaned = re.sub(r'\b\d+(\.\d+)?\b', '', expr_cleaned)
    
    # Extract remaining identifiers (column names)
    potential_columns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr_cleaned)
    
    # Filter out common SQL keywords that might remain
    sql_keywords = {
        'varchar', 'integer', 'float', 'date', 'timestamp', 'boolean',
        'null', 'true', 'false', 'from', 'where', 'group', 'by', 'order',
        'having', 'limit', 'offset', 'join', 'left', 'right', 'inner',
        'outer', 'on', 'using', 'distinct', 'all', 'any', 'some', 'exists'
    }
    
    for col in potential_columns:
        if col.lower() not in sql_keywords and len(col) > 1:
            columns.add(col)
    
    return columns


def _strip_table_prefix(expr: str) -> str:
    """
    Strip table prefix from column expression.
    
    Converts 'TABLE.COLUMN' or 'table.column' to just 'COLUMN' or 'column'.
    Handles quoted identifiers and preserves case.
    
    Args:
        expr: Column expression (e.g., 'WGS_MAD.LOB_DESC' or 'status')
    
    Returns:
        Column name without table prefix (e.g., 'LOB_DESC' or 'status')
    """
    if not expr:
        return expr
    
    # If contains a dot, take the part after the last dot
    if '.' in expr:
        parts = expr.split('.')
        return parts[-1].strip()
    
    return expr.strip()


def _strip_table_prefixes_from_sql(sql: str) -> str:
    """
    Strip table prefixes from all column references in SQL.
    
    Converts references like 'wgs_mad.LOB_DESC' to just 'LOB_DESC'.
    Handles multiple table prefixes and preserves the rest of the SQL.
    
    Args:
        sql: SQL string with potential table prefixes
    
    Returns:
        SQL string with table prefixes removed
    """
    if not sql:
        return sql
    
    # Pattern to match table_name.column_name (word characters, dots)
    # Match: word.word but not DATE '...' or other SQL constructs
    import re
    
    # Replace patterns like "table.column" with just "column"
    # But avoid matching things like DATE '2024-01-01' or numeric decimals
    # Look for word characters followed by dot followed by word characters
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    def replacer(match):
        # Return just the column name (second group)
        return match.group(2)
    
    result = re.sub(pattern, replacer, sql)
    return result


def determine_required_columns(
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    candidate_dimensions: List[str],
    period_window: PeriodWindow,
    catalog: SemanticCatalog,
    primary_table: str,
) -> List[str]:
    """
    Determine all columns required for correlation analysis.
    
    Extracts columns from:
    - Metric expressions
    - User filter fields
    - Config drill dimensions
    - Time dimension
    - JOIN keys (if multi-table metric)
    
    Args:
        metric: Metric definition with expression
        filters: User-provided filter conditions
        candidate_dimensions: Drill-down dimensions from config
        period_window: Period window with time dimension
        catalog: Semantic catalog
        primary_table: Primary table name
    
    Returns:
        Deduplicated list of required column names (without table prefixes)
    """
    columns = set()
    
    # 1. Extract columns from metric expression
    metric_expr = str(metric.get("expr", ""))
    metric_columns = extract_columns_from_expression(metric_expr)
    
    # Resolve to actual column expressions and strip table prefixes
    for col_name in metric_columns:
        field_def = resolve_field(catalog, col_name, preferred_tables=[primary_table])
        if field_def:
            clean_col = _strip_table_prefix(field_def["expr"])
            columns.add(clean_col)
    
    # 2. Extract columns from user filters
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]
    for filter_condition in filters:
        field_name = filter_condition.get("field", "")
        if field_name:
            field_def = resolve_field(catalog, field_name, preferred_tables=preferred_tables)
            if field_def:
                clean_col = _strip_table_prefix(field_def["expr"])
                columns.add(clean_col)
    
    # 3. Extract columns from drill dimensions (from config)
    for dimension_name in candidate_dimensions:
        field_def = resolve_field(catalog, dimension_name, preferred_tables=[primary_table])
        if not field_def:
            field_def = resolve_field(catalog, dimension_name)
        if field_def:
            clean_col = _strip_table_prefix(field_def["expr"])
            columns.add(clean_col)
    
    # 4. Add time dimension column
    time_field = resolve_time_field(period_window, catalog)
    if time_field:
        clean_col = _strip_table_prefix(time_field["expr"])
        columns.add(clean_col)
    
    # 5. Add any dependency table JOIN keys if metric spans multiple tables
    # For simplicity, we'll include all fields from primary table that are referenced
    # in relationships (this is conservative but safe)
    required_tables = determine_required_tables(metric, filters, candidate_dimensions, period_window, catalog, primary_table)
    
    if len(required_tables) > 1:
        # Multi-table query - need JOIN keys
        for relationship in catalog.get("relationships", []):
            left_table = relationship.get("left_table")
            right_table = relationship.get("right_table")
            
            # If this relationship involves our primary table and a required table
            if (left_table == primary_table and right_table in required_tables) or \
               (right_table == primary_table and left_table in required_tables):
                # Add relationship columns from primary table side
                for col_pair in _safe_list(relationship.get("relationship_columns")):
                    pair = _safe_dict(col_pair)
                    if left_table == primary_table:
                        col_name = _strip_table_prefix(str(pair.get("left_column", "")))
                        columns.add(col_name)
                    elif right_table == primary_table:
                        col_name = _strip_table_prefix(str(pair.get("right_column", "")))
                        columns.add(col_name)
    
    # Filter out empty strings and return as list
    return [col for col in sorted(columns) if col]


def execute_sql_to_df(snowflake_helper: SnowparkHelper, query: str) -> pd.DataFrame:
    logger.debug("Executing SQL (%s chars)", len(query))
    df = snowflake_helper.execute_query_and_return_pandas_df(query)
    return normalize_dataframe_columns(df)


def build_run_id(intent: IntentPayload) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    question_slug = slugify(intent.get("raw_question") or "correlation_run")[:40]
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{question_slug}_{suffix}"


def summarize_filters(filters: Sequence[FilterCondition]) -> List[Dict[str, Any]]:
    return [
        {
            "field": filter_condition["field"],
            "operator": filter_condition["operator"],
            "value": filter_condition["value"],
            "source": filter_condition["source"],
        }
        for filter_condition in filters
    ]


def execute_correlation(state: GraphState) -> Dict[str, Any]:
    intent = state["intent"]
    catalog = state["catalog"]
    snowflake_helper = state.get("snowflake_helper")
    output_root = state.get("output_root") or "correlation_runs"
    llm = state.get("llm")
    llm_tokens = merge_correlation_llm_tokens(state.get("llm_tokens"))
    if snowflake_helper is None:
        raise RuntimeError("Snowflake helper is required to execute correlation analysis.")

    # Log raw period from UI
    mode_params = intent.get("analysis_mode_parameters", {})
    period_raw = mode_params.get("period", {})
    logger.info(f"Raw period from UI: {period_raw}")
    
    period_window = resolve_period_window(intent, catalog)
    logger.info(f"Period window resolved: baseline={period_window.get('baseline_start_time')} to {period_window.get('baseline_end_time')}, comparison={period_window.get('start_time')} to {period_window.get('end_time')}")
    metric_name = resolve_metric_name(intent, catalog)
    metric = catalog["metrics_by_name"][metric_name]
    primary_table = default_primary_table(metric, period_window, catalog)
    filters = normalize_filters_for_sql(copy.deepcopy(intent.get("filters", [])))
    candidate_dimensions = get_candidate_dimensions(intent, catalog, primary_table)
    stop_rules = copy.deepcopy(DEFAULT_STOP_RULES)
    stop_rules.update(_safe_dict(_safe_dict(intent.get("analysis_mode_parameters")).get("stop_rules")))
    save_parquet = _resolve_save_parquet(intent)
    disable_summary_creation = _resolve_disable_summary_creation(intent)
    generate_recommendations = _resolve_generate_recommendations(intent)

    run_id = build_run_id(intent)
    run_dir = ensure_dir(Path(output_root) / run_id)
    aggregates_dir = run_dir / "aggregates"
    if save_parquet:
        ensure_dir(aggregates_dir)
    quality_dir = ensure_dir(run_dir / "quality")
    queries_dir = ensure_dir(run_dir / "queries")
    summary_dir = ensure_dir(run_dir / "summary")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_question": intent.get("raw_question"),
        "root_metric": metric_name,
        "analysis_mode": intent.get("analysis_mode"),
        "period_window": period_window,
        "files": [],
    }
    warnings: List[str] = []
    
    # Create subset table for improved performance
    subset_table_name = None
    subset_manager = None
    original_qualified_name = None
    use_subset_table = SubsetTableManager is not None and AppConstants.CORRELATION_SUBSET_CLEANUP_ENABLED
    
    if use_subset_table:
        try:
            logger.info(f"Creating subset table with SELECT * for performance optimization")
            
            # Get base table qualified name
            base_table_def = catalog["tables"].get(primary_table)
            if base_table_def:
                base_table_qualified = base_table_def.get("qualified_name", primary_table)
            else:
                base_table_qualified = primary_table
            
            # Build WHERE clauses for subset
            time_field = resolve_time_field(period_window, catalog)
            time_expr = time_field["expr"]
            time_filter = period_filter_sql(time_expr, period_window)
            
            # Strip table prefixes from time filter
            time_filter_clean = _strip_table_prefixes_from_sql(time_filter)
            where_clauses = [time_filter_clean]
            
            # Add user filter conditions
            alias_map_temp = {primary_table: primary_table}
            preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]
            for filter_condition in filters:
                try:
                    filter_sql = render_filter_clause(filter_condition, catalog, alias_map_temp, preferred_tables)
                    # Strip table prefixes from filter SQL
                    filter_sql_clean = _strip_table_prefixes_from_sql(filter_sql)
                    where_clauses.append(filter_sql_clean)
                except Exception as e:
                    logger.warning(f"Could not add filter to subset table: {e}")
            
            # Strip table prefix from time column for clustering
            time_column_clean = _strip_table_prefix(time_expr)
            
            # Create subset table manager and table with SELECT *
            subset_manager = SubsetTableManager(snowflake_helper)
            subset_table_name = subset_manager.create_subset_table(
                base_table_qualified_name=base_table_qualified,
                required_columns=None,  # Use SELECT * to include all columns
                where_clauses=where_clauses,
                cluster_column=time_column_clean
            )
            
            # Save original qualified_name and temporarily update catalog to point to subset
            # This allows all query building functions to work with the logical table name
            # while actually querying the subset table
            original_qualified_name = catalog["tables"][primary_table]["qualified_name"]
            catalog["tables"][primary_table]["qualified_name"] = subset_table_name
            logger.info(f"✅ Using subset table for correlation analysis: {subset_table_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create subset table, falling back to full table: {e}")
            subset_table_name = None
            subset_manager = None
            use_subset_table = False
            original_qualified_name = None
    
    # Root summary and monthly extracts.
    root_summary_sql = build_root_summary_query(catalog, metric, filters, period_window, primary_table)
    baseline_extract_sql = build_period_extract_query(catalog, metric, filters, period_window, primary_table, "baseline")
    comparison_extract_sql = build_period_extract_query(catalog, metric, filters, period_window, primary_table, "comparison")

    write_text(queries_dir / ROOT_SUMMARY_SQL_NAME, root_summary_sql)
    write_text(queries_dir / BASELINE_EXTRACT_SQL_NAME, baseline_extract_sql)
    write_text(queries_dir / COMPARISON_EXTRACT_SQL_NAME, comparison_extract_sql)
    manifest["files"].extend(
            [
                str((queries_dir / ROOT_SUMMARY_SQL_NAME).relative_to(run_dir)),
                str((queries_dir / BASELINE_EXTRACT_SQL_NAME).relative_to(run_dir)),
                str((queries_dir / COMPARISON_EXTRACT_SQL_NAME).relative_to(run_dir)),
            ]
        )

    root_summary_df = execute_sql_to_df(snowflake_helper, root_summary_sql)
    baseline_extract_df = execute_sql_to_df(snowflake_helper, baseline_extract_sql)
    comparison_extract_df = execute_sql_to_df(snowflake_helper, comparison_extract_sql)

    root_summary_df, baseline_value, comparison_value, delta_value, delta_pct = reshape_root_summary(root_summary_df)

    if save_parquet:
        baseline_path = write_dataframe(aggregates_dir / "root_baseline.parquet", baseline_extract_df)
        comparison_path = write_dataframe(aggregates_dir / "root_comparison.parquet", comparison_extract_df)
        root_summary_path = write_dataframe(aggregates_dir / "root_summary.parquet", root_summary_df)
        manifest["files"].extend(
            [
                str(Path(baseline_path).relative_to(run_dir)),
                str(Path(comparison_path).relative_to(run_dir)),
                str(Path(root_summary_path).relative_to(run_dir)),
            ]
        )

    if abs(delta_value) < float(stop_rules["min_abs_delta"]):
        warnings.append(
            f"Root delta {delta_value:,.2f} is below min_abs_delta={stop_rules['min_abs_delta']}; drill-down stopped early."
        )

    drill_path: List[PathNodeSummary] = []
    explored_dimensions: List[Dict[str, Any]] = []
    current_filters = copy.deepcopy(filters)
    available_dimensions = list(candidate_dimensions)
    parent_delta = delta_value
    max_depth = int(stop_rules["max_depth"])
    min_contribution_pct = float(stop_rules["min_contribution_pct"])
    min_incremental_gain_pct = float(stop_rules["min_incremental_gain_pct"])
    min_row_count = int(stop_rules["min_row_count"])
    top_k_limit = int(stop_rules["top_k_per_level"])
    bottom_k_limit = 5

    for level in range(1, max_depth + 1):
        if abs(parent_delta) < float(stop_rules["min_abs_delta"]):
            break
        if not available_dimensions:
            break

        level_results: List[CandidateSelection] = []
        for dimension_name in available_dimensions:
            dimension_label = resolve_dimension_label(catalog, primary_table, dimension_name)
            folder_name = folder_token_for_dimension(dimension_name)
            
            sql_filename = f"{folder_name}.sql"
            query_path = build_nested_path(queries_dir, drill_path, dimension_name, sql_filename)
            
            query_path.parent.mkdir(parents=True, exist_ok=True)
            
            dimension_sql = build_dimension_aggregate_query(
                catalog=catalog,
                metric=metric,
                filters=current_filters,
                period_window=period_window,
                primary_table=primary_table,
                dimension_name=dimension_name,
            )
            write_text(query_path, dimension_sql)
            dimension_df = execute_sql_to_df(snowflake_helper, dimension_sql)
            baseline_df, comparison_df, delta_df = pivot_dimension_comparison(dimension_df, total_delta=delta_value, parent_delta=parent_delta)
            manifest["files"].append(str(Path(query_path).relative_to(run_dir)))

            aggregate_paths = {
                "baseline": "",
                "comparison": "",
                "delta": "",
            }
            if save_parquet:
                aggregate_dir = build_nested_path(aggregates_dir, drill_path, dimension_name, folder_name)
                aggregate_dir.mkdir(parents=True, exist_ok=True)
                written_baseline = write_dataframe(aggregate_dir / "baseline.parquet", baseline_df)
                written_comparison = write_dataframe(aggregate_dir / "comparison.parquet", comparison_df)
                written_delta = write_dataframe(aggregate_dir / "delta.parquet", delta_df)
                aggregate_paths = {
                    "baseline": str(Path(written_baseline).relative_to(run_dir)),
                    "comparison": str(Path(written_comparison).relative_to(run_dir)),
                    "delta": str(Path(written_delta).relative_to(run_dir)),
                }
                manifest["files"].extend(
                    [
                        aggregate_paths["baseline"],
                        aggregate_paths["comparison"],
                        aggregate_paths["delta"],
                    ]
                )

            top_segments, top_aligned_share = _select_top_segments(
                delta_df,
                level=level,
                min_row_count=min_row_count,
                min_contribution_pct=min_contribution_pct,
                min_incremental_gain_pct=min_incremental_gain_pct,
                top_k_limit=top_k_limit,
            )
            if not top_segments:
                explored_dimensions.append(
                    {
                        "level": level,
                        "dimension": dimension_name,
                        "dimension_label": dimension_label,
                        "selected": None,
                        "reason": "no_top_segments",
                        "folder_name": folder_name,
                        "nested_path": str(query_path.relative_to(queries_dir)),
                    }
                )
                continue

            bottom_segments = _select_bottom_segments(
                delta_df,
                max_segments=bottom_k_limit,
            )

            top_aligned_delta = _sum_segment_field(top_segments, "aligned_delta")
            top_total_delta = _sum_segment_field(top_segments, "delta_value")

            selection: CandidateSelection = {
                "dimension": dimension_name,
                "dimension_label": dimension_label,
                "folder_name": folder_name,
                "nested_path": str(query_path.relative_to(queries_dir)),
                "top_segments": top_segments,
                "bottom_segments": bottom_segments,
                "top_aligned_share": top_aligned_share,
                "top_aligned_delta": top_aligned_delta,
                "top_total_delta": top_total_delta,
                "query_path": str(query_path.relative_to(run_dir)),
                "aggregate_paths": aggregate_paths,
            }
            level_results.append(selection)
            explored_dimensions.append(
                {
                    "level": level,
                    "dimension": dimension_name,
                    "dimension_label": dimension_label,
                    "selected": [segment.get("value") for segment in top_segments],
                    "folder_name": folder_name,
                    "nested_path": str(query_path.relative_to(queries_dir)),
                    "top_aligned_share": top_aligned_share,
                    "top_total_delta": top_total_delta,
                }
            )

        if not level_results:
            break

        level_results = sorted(
            level_results,
            key=lambda item: (
                abs(item.get("top_aligned_share", 0.0)),
                abs(item.get("top_aligned_delta", 0.0)),
                abs(item.get("top_total_delta", 0.0)),
            ),
            reverse=True,
        )
        chosen = level_results[0]

        if not chosen.get("top_segments"):
            warnings.append(f"Stopped at level {level}: no top contributors met guardrails.")
            break

        node_summary: PathNodeSummary = {
            "level": level,
            "dimension": chosen["dimension"],
            "dimension_label": chosen.get("dimension_label") or humanize_dimension_name(chosen["dimension"]),
            "folder_name": chosen["folder_name"],
            "top_segments": chosen.get("top_segments", []),
            "bottom_segments": chosen.get("bottom_segments", []),
            "parent_context": build_context_phrase(drill_path) if drill_path else "root",
        }
        drill_path.append(node_summary)

        selected_values = [segment.get("value") for segment in chosen.get("top_segments", []) if segment.get("value")]
        current_filters.append(
            {
                "field": chosen["dimension"],
                "operator": "in",
                "value": selected_values,
                "source": "dimension_match",
            }
        )
        available_dimensions = [item for item in available_dimensions if item != chosen["dimension"]]
        parent_delta = chosen.get("top_total_delta", parent_delta)

    explainer_metrics = [
        metric_name
        for metric_name in [str(item).strip() for item in _safe_list(_safe_dict(intent.get("analysis_mode_parameters")).get("explainer_metrics"))]
        if metric_name in catalog["metrics_by_name"]
    ]
    explainer_payload: Dict[str, Any] = {}
    if explainer_metrics:
        try:
            root_explainer_sql = build_explainer_query(catalog, explainer_metrics, filters, period_window, primary_table)
            write_text(queries_dir / "root_explainers.sql", root_explainer_sql)
            manifest["files"].append(str((queries_dir / "root_explainers.sql").relative_to(run_dir)))
            root_explainer_df = execute_sql_to_df(snowflake_helper, root_explainer_sql)
            explainer_payload["root"] = root_explainer_df.to_dict(orient="records")

            node_explainers: List[Dict[str, Any]] = []
            path_filters = copy.deepcopy(filters)
            for node in drill_path:
                path_filters.append(
                    {
                        "field": node["dimension"],
                        "operator": "in",
                        "value": _top_segment_values(node),
                        "source": "dimension_match",
                    }
                )
                sql_name = f"{node['folder_name']}_explainers.sql"
                node_sql = build_explainer_query(catalog, explainer_metrics, path_filters, period_window, primary_table)
                write_text(queries_dir / sql_name, node_sql)
                manifest["files"].append(str((queries_dir / sql_name).relative_to(run_dir)))
                node_df = execute_sql_to_df(snowflake_helper, node_sql)
                node_explainers.append(
                    {
                        "folder_name": node["folder_name"],
                        "dimension": node["dimension"],
                        "selected_values": _top_segment_values(node),
                        "metrics": node_df.to_dict(orient="records"),
                    }
                )
            explainer_payload["path_nodes"] = node_explainers
        except Exception as exc:
            warnings.append(f"Explainer metric execution failed: {exc}")
            logger.warning("Explainer query execution failed", exc_info=exc)

    interaction_matrix_payload: Dict[str, Any] = {}
    interaction_summary_payload: Dict[str, Any] = {}
    recommended_action_payload: Dict[str, Any] = {}
    interaction_config = _safe_dict(_safe_dict(intent.get("analysis_mode_parameters")).get("interaction_matrix"))
    if interaction_config:
        try:
            try:
                from deep_research_agents.correlation_interaction_matrix import execute_interaction_matrix
            except ImportError:
                from correlation_interaction_matrix import execute_interaction_matrix  # type: ignore

            interaction_result = execute_interaction_matrix(
                intent=intent,
                catalog=catalog,
                metric_name=metric_name,
                metric=metric,
                primary_table=primary_table,
                period_window=period_window,
                filters=filters,
                drill_path=drill_path,
                root_summary_df=root_summary_df,
                delta_value=delta_value,
                queries_dir=queries_dir,
                aggregates_dir=aggregates_dir,
                summary_dir=summary_dir,
                run_dir=run_dir,
                save_parquet=save_parquet,
                disable_summary_creation=disable_summary_creation,
                generate_recommendations=generate_recommendations,
                snowflake_helper=snowflake_helper,
                llm=llm,
                manifest=manifest,
                warnings=warnings,
            )
            interaction_matrix_payload = {
                key: value
                for key, value in interaction_result.items()
                if key not in {"interaction_summary", "recommended_action", "llm_tokens"}
            }
            interaction_summary_payload = _safe_dict(interaction_result.get("interaction_summary"))
            recommended_action_payload = _safe_dict(interaction_result.get("recommended_action"))
            llm_tokens = merge_correlation_llm_tokens(llm_tokens, interaction_result.get("llm_tokens"))
        except Exception as exc:
            warnings.append(f"Interaction matrix execution failed: {exc}")
            logger.warning("Interaction matrix execution failed", exc_info=exc)

    if interaction_config and generate_recommendations:
        try:
            try:
                from deep_research_agents.correlation_recommendation import create_correlation_recommendations
            except ImportError:
                from correlation_recommendation import create_correlation_recommendations  # type: ignore

            interaction_preview_limits = _safe_dict(interaction_config.get("preview_limits"))
            recommended_action_payload = create_correlation_recommendations(
                metric_name=metric_name,
                root_trend={
                    "metric_name": metric_name,
                    "baseline_value": baseline_value,
                    "comparison_value": comparison_value,
                    "delta_value": delta_value,
                    "delta_pct": delta_pct,
                },
                drill_path=drill_path,
                explainer_metrics=explainer_payload,
                interaction_matrix=interaction_matrix_payload,
                interaction_summary=interaction_summary_payload,
                prior_recommendations=recommended_action_payload,
                llm=llm,
                max_recommendations=int(interaction_preview_limits.get("max_recommendations", 5) or 5),
                ehap=self.ehap,
                llm_reinitializer=self._initialize_llm,
            )
            llm_tokens = merge_correlation_llm_tokens(llm_tokens, _safe_dict(recommended_action_payload.get("llm_tokens")))
            write_json(summary_dir / "interaction_recommendations.json", _interaction_recommendations_artifact(recommended_action_payload))
            if str((summary_dir / "interaction_recommendations.json").relative_to(run_dir)) not in manifest["files"]:
                manifest["files"].append(str((summary_dir / "interaction_recommendations.json").relative_to(run_dir)))
        except Exception as exc:
            warnings.append(f"Recommendation synthesis failed: {exc}")
            logger.warning("Correlation recommendation synthesis failed", exc_info=exc)

    run_config = {
        "run_id": run_id,
        "raw_question": intent.get("raw_question"),
        "root_metric": metric_name,
        "filters": summarize_filters(filters),
        "period_window": period_window,
        "analysis_mode": intent.get("analysis_mode"),
        "stop_rules": stop_rules,
        "save_parquet": save_parquet,
        "disable_summary_creation": disable_summary_creation,
        "generate_recommendations": generate_recommendations,
        "candidate_dimensions": candidate_dimensions,
    }
    write_yaml(queries_dir / RUN_CONFIG_YAML_NAME, run_config)
    manifest["files"].append(str((queries_dir / RUN_CONFIG_YAML_NAME).relative_to(run_dir)))

    data_quality = {
        "root_summary": root_summary_df.to_dict(orient="records"),
        "baseline_months": period_window["baseline_months"],
        "comparison_months": period_window["comparison_months"],
        "root_baseline_row_count": int(root_summary_df.loc[root_summary_df["period_bucket"] == "baseline", "raw_row_count"].sum()) if not root_summary_df.empty else 0,
        "root_comparison_row_count": int(root_summary_df.loc[root_summary_df["period_bucket"] == "comparison", "raw_row_count"].sum()) if not root_summary_df.empty else 0,
        "validation_warnings_from_intent": intent.get("validation_warnings", []),
        "execution_warnings": warnings,
    }
    guardrails = {
        "comparison_strategy": period_window["comparison_strategy"],
        "period_window": {
            "comparison_start": period_window["start_time"],
            "comparison_end": period_window["end_time"],
            "baseline_start": period_window["baseline_start_time"],
            "baseline_end": period_window["baseline_end_time"],
        },
        "stop_rules": stop_rules,
        "notes": [
            "time_dimension and start_time/end_time are required in the configuration; no defaults are applied.",
            "If previous_period is missing, the agent shifts current_period back one year to build the baseline window.",
            "The agent uses deterministic SQL generation; no LLM is used inside the correlation executor.",
        ],
    }
    write_json(quality_dir / DATA_QUALITY_JSON_NAME, data_quality)
    write_json(quality_dir / GUARDRAILS_JSON_NAME, guardrails)
    manifest["files"].extend(
        [
            str((quality_dir / DATA_QUALITY_JSON_NAME).relative_to(run_dir)),
            str((quality_dir / GUARDRAILS_JSON_NAME).relative_to(run_dir)),
        ]
    )

    narrative_summary = ""
    executive_summary_text = ""
    executive_summary_source = "disabled" if disable_summary_creation else "empty"
    if not disable_summary_creation:
        narrative_summary = build_narrative_summary(filters, metric_name, delta_value, drill_path)
        executive_summary_text = _build_fallback_executive_summary(
            filters=filters,
            metric_name=metric_name,
            baseline_value=baseline_value,
            comparison_value=comparison_value,
            delta_value=delta_value,
            delta_pct=delta_pct,
            drill_path=drill_path,
            explainer_payload=explainer_payload,
        )
        executive_summary_source = "deterministic"
        if llm is not None:
            input_text = build_executive_summary_input(
                filters=filters,
                metric_name=metric_name,
                baseline_value=baseline_value,
                comparison_value=comparison_value,
                delta_value=delta_value,
                delta_pct=delta_pct,
                drill_path=drill_path,
                explainer_payload=explainer_payload,
            )
            summary, executive_summary_tokens = generate_executive_summary(
                llm, input_text, ehap=self.ehap, llm_reinitializer=self._initialize_llm
            )
            llm_tokens = merge_correlation_llm_tokens(llm_tokens, executive_summary_tokens)
            if summary:
                executive_summary_text = summary
                executive_summary_source = "llm"
    executive_summary = {
        "run_id": run_id,
        "raw_question": intent.get("raw_question"),
        "root_metric": metric_name,
        "metric_label": humanize_metric_name(metric_name),
        "baseline_value": baseline_value,
        "comparison_value": comparison_value,
        "delta_value": delta_value,
        "delta_pct": delta_pct,
        "baseline_value_formatted": format_value(metric_name, baseline_value),
        "comparison_value_formatted": format_value(metric_name, comparison_value),
        "delta_value_formatted": format_value(metric_name, delta_value),
        "period_window": period_window,
        "scope_filters": summarize_filters(filters),
        "drill_path": drill_path,
        "dimensions_explored": explored_dimensions,
        "explainer_metrics": explainer_payload,
        "interaction_matrix": interaction_matrix_payload,
        "interaction_summary": interaction_summary_payload,
        "recommended_action": _recommended_action_items(recommended_action_payload),
        "llm_tokens": llm_tokens,
        "narrative_summary": narrative_summary,
        "executive_summary": executive_summary_text,
        "executive_summary_source": executive_summary_source,
        "story_stub": executive_summary_text,
    }
    write_json(summary_dir / EXECUTIVE_SUMMARY_JSON_NAME, executive_summary)
    manifest["files"].append(str((summary_dir / EXECUTIVE_SUMMARY_JSON_NAME).relative_to(run_dir)))

    write_json(run_dir / MANIFEST_JSON_NAME, manifest)

    result: CorrelationRunResult = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / MANIFEST_JSON_NAME),
        "executive_summary_path": str(summary_dir / EXECUTIVE_SUMMARY_JSON_NAME),
        "period_window": period_window,
        "root_metric": metric_name,
        "baseline_value": baseline_value,
        "comparison_value": comparison_value,
        "delta_value": delta_value,
        "delta_pct": delta_pct,
        "drill_path": drill_path,
        "narrative_summary": narrative_summary,
        "executive_summary": executive_summary_text,
        "executive_summary_source": executive_summary_source,
        "interaction_matrix": interaction_matrix_payload,
        "interaction_summary": interaction_summary_payload,
        "recommended_action": _recommended_action_items(recommended_action_payload),
        "llm_tokens": llm_tokens,
        "warnings": warnings,
    }
    
    # Restore original qualified_name and cleanup subset table before returning
    if subset_manager and subset_table_name:
        logger.info(f"Cleaning up subset table: {subset_table_name}")
        # Restore original qualified_name in catalog
        if original_qualified_name and primary_table in catalog["tables"]:
            catalog["tables"][primary_table]["qualified_name"] = original_qualified_name
        # Drop the temporary subset table
        subset_manager.cleanup_table(subset_table_name, ignore_errors=True)
    
    return {
        "result": result,
        "input_tokens": int(llm_tokens.get("input", 0) or 0),
        "output_tokens": int(llm_tokens.get("output", 0) or 0),
        "llm_tokens": llm_tokens,
    }


# =============================
# Agent class
# =============================

class CorrelationAgent(AgentBase):
    """
    Agent for executing correlation drill-down analyses using a semantic model.

    This agent wraps the deterministic correlation pipeline in the AgentBase
    lifecycle, providing standardized logging, resource management, and
    optional LLM summaries.

    Args:
        yaml_path: Path to the semantic model YAML.
        snowflake_helper: Optional SnowparkHelper instance.
        snowflake_helper_builder: Optional builder for SnowparkHelper.
        output_root: Directory where run artifacts are written.
        llm: Optional LLM instance for executive summaries.
        llm_builder: Optional LLM builder callable.
        **kwargs: Additional arguments passed to AgentBase.
    """

    api_response_model = CorrelationAgentResponseSchema

    def __init__(
        self,
        yaml_path: Optional[str] = None,
        snowflake_helper: Optional[SnowparkHelper] = None,
        snowflake_helper_builder: Optional[Callable[[], SnowparkHelper]] = None,
        output_root: str = "correlation_runs",
        llm: Optional[Any] = None,
        llm_builder: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize CorrelationAgent.
        
        Args:
            yaml_path: Optional default path to semantic model YAML.
                      Can be overridden per-request in prepare_state.
            snowflake_helper: Optional SnowparkHelper instance.
            snowflake_helper_builder: Optional builder for SnowparkHelper.
            output_root: Directory where run artifacts are written.
            llm: Optional LLM instance for executive summaries.
            llm_builder: Optional LLM builder callable.
            **kwargs: Additional arguments passed to AgentBase.
        """
        self.output_root = output_root
        
        kwargs.setdefault("llm_summary_mode", "auto")

        super().__init__(
            agent_name="correlation",
            state_class=GraphState,
            llm=llm,
            llm_builder=llm_builder,
            **kwargs,
        )
        
        # Resolve default yaml_path from parameter, config, or fallback
        if yaml_path is None:
            resolved_path = self._get_semantic_path_from_config()
            self.yaml_path = _resolve_semantic_model_path(resolved_path)
        else:
            self.yaml_path = _resolve_semantic_model_path(yaml_path)

        self.snowflake_helper = self._init_snowflake(
            snowflake_helper,
            snowflake_helper_builder,
        )
    
    def _get_semantic_path_from_config(self) -> str:
        """Get semantic model path from .ini configuration."""
        from pathlib import Path
        
        try:
            # Get semantic_config_path from agent config
            config_path = self.config('semantic_config_path')
            
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[4]
            return str(project_root / config_path)
        except KeyError:
            # Fallback to default if not in config
            self.logger.warning(
                f"semantic_config_path not found in config for environment '{AppConstants.ENV}'. "
                f"Using default path."
            )
            return str(DEFAULT_SEMANTIC_MODEL_PATH)

    @property
    def node_name(self) -> str:
        return "execute_correlation"

    def _load_static_resources(self) -> Dict[str, Any]:
        """Load static resources (none for correlation agent - YAML loaded per-request).
        
        Returns:
            Empty dict - semantic model and catalog are loaded per-request in prepare_state.
        """
        return {}

    def _init_snowflake(
        self,
        helper: Optional[SnowparkHelper],
        builder: Optional[Callable[[], SnowparkHelper]],
    ) -> Optional[SnowparkHelper]:
        if helper is not None:
            return helper

        if builder is not None:
            try:
                return builder()
            except Exception as exc:
                self.logger.error("Snowflake builder failed: %s", exc, exc_info=exc)
                raise AgentConfigurationError("Snowflake builder failed") from exc

        try:
            creds = CredentialProvider.get_instance().get_snowflake_credentials()
        except Exception as exc:
            self.logger.warning(
                "Snowflake credentials not configured; provide a helper to execute the agent.",
                exc_info=exc,
            )
            return None

        try:
            return SnowparkHelper(
                batch_size=10000,
                max_workers=6,
                enable_metrics=True,
                connection_pool_size=4,
                **creds,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to initialize Snowflake helper (connection may have failed); provide a helper to execute the agent.",
                exc_info=exc,
            )
            return None

    def create_stub_llm(self) -> Any:
        class _StubLLM:
            class _StructuredLLM:
                def invoke(self, messages: List[Dict[str, str]]) -> ExecutiveSummarySchema:
                    _ = messages
                    return ExecutiveSummarySchema(summary="Stub executive summary.")

            def with_structured_output(self, _schema: object, **kwargs: object) -> "_StubLLM._StructuredLLM":
                _ = kwargs
                return self._StructuredLLM()

        return _StubLLM()

    def execute(self, **kwargs: Any) -> Any:
        if "intent" in kwargs:
            intent = kwargs.pop("intent")
            if not isinstance(intent, dict):
                raise ValueError("intent must be a mapping when provided.")
            query = str(kwargs.pop("query", "") or intent.get("raw_question") or "").strip()
            if not query:
                raise ValueError("query is required to execute correlation analysis.")
            conversation_id = str(kwargs.pop("conversation_id", "") or "")
            context_payload = _context_from_intent(intent)
            context = _safe_dict(kwargs.pop("context", None))
            if context:
                context_payload.update(context)
            save_parquet = kwargs.pop("save_parquet", None)
            if save_parquet is not None:
                context_payload["save_parquet"] = save_parquet
            disable_summary_creation = kwargs.pop("disable_summary_creation", None)
            if disable_summary_creation is not None:
                context_payload["disable_summary_creation"] = disable_summary_creation
            generate_recommendations = kwargs.pop("generate_recommendations", None)
            if generate_recommendations is not None:
                context_payload["generate_recommendations"] = generate_recommendations
            return super().execute(
                conversation_id=conversation_id,
                query=query,
                context=context_payload,
                **kwargs,
            )
        return super().execute(**kwargs)

    def node_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.snowflake_helper is None:
            raise AgentExecutionError("Snowflake helper is required to execute correlation analysis.")
        state_payload = dict(state)
        state_payload["snowflake_helper"] = self.snowflake_helper
        state_payload["llm"] = self.llm
        return execute_correlation(state_payload)  # type: ignore[arg-type]

    def prepare_state(
        self,
        conversation_id: str,
        query: str,
        context: Optional[CorrelationContext] = None,
        yaml_path: Optional[str] = None,
        save_parquet: Optional[bool] = None,
        disable_summary_creation: Optional[bool] = None,
        generate_recommendations: Optional[bool] = None,
        **kwargs: Any,
    ) -> GraphState:
        """Prepare state for correlation execution.
        
        Args:
            conversation_id: Conversation identifier
            query: User query/question
            context: Optional correlation context
            yaml_path: Optional runtime path to semantic model YAML (overrides default)
            save_parquet: Whether to save parquet files
            disable_summary_creation: Whether to disable summary creation
            generate_recommendations: Whether to generate recommendations
            **kwargs: Additional state fields (output_root, job_id, etc.)
        
        Returns:
            GraphState with semantic model and catalog loaded from yaml_path
        """
        if not query:
            raise ValueError("query is required to execute correlation analysis.")
        
        # Resolve and validate yaml_path (use default if not provided)
        resolved_yaml_path = _resolve_semantic_model_path(yaml_path or self.yaml_path)
        
        # Load semantic model and catalog fresh for this request
        semantic_model = load_semantic_yaml(resolved_yaml_path)
        catalog = build_semantic_catalog(semantic_model)
        
        intent = _build_intent_from_request(query, context)
        save_parquet_override = _coerce_bool(save_parquet, default=None)
        if save_parquet_override is not None:
            intent["save_parquet"] = save_parquet_override
        disable_summary_creation_override = _coerce_bool(disable_summary_creation, default=None)
        if disable_summary_creation_override is not None:
            intent["disable_summary_creation"] = disable_summary_creation_override
        generate_recommendations_override = _coerce_bool(generate_recommendations, default=None)
        if generate_recommendations_override is not None:
            intent["generate_recommendations"] = generate_recommendations_override
        
        # Apply analysis mode defaults using the loaded semantic model
        intent = _apply_analysis_mode_defaults(intent, semantic_model)
        
        output_root = kwargs.get("output_root") or self.output_root
        job_id = str(kwargs.get("job_id") or uuid.uuid4().hex)
        start_time = datetime.now(timezone.utc).isoformat()
        
        return {
            "conversation_id": str(conversation_id or ""),
            "query": query,
            "job_id": job_id,
            "start_time": start_time,
            "yaml_path": resolved_yaml_path,
            "intent": intent,
            "semantic_model": semantic_model,
            "catalog": catalog,
            "output_root": output_root,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_tokens": empty_correlation_llm_tokens(),
        }

    def extract_result(self, graph_output: Dict[str, Any]) -> CorrelationAgentResponse:
        result_payload = dict(graph_output.get("result") or {})
        recommended_action = _recommended_action_items(result_payload.pop("recommended_action", []))
        conversation_id = str(graph_output.get("conversation_id") or "")
        job_id = str(result_payload.pop("run_id", "") or graph_output.get("job_id") or uuid.uuid4().hex)
        start_time = graph_output.get("start_time")
        end_time = datetime.now(timezone.utc).isoformat()
        start_dt = _parse_iso_datetime(start_time)
        end_dt = _parse_iso_datetime(end_time)
        duration_ms = 0
        if start_dt is not None and end_dt is not None:
            duration_ms = max(int((end_dt - start_dt).total_seconds() * 1000), 0)
        output_warnings = [str(item) for item in _safe_list(result_payload.get("warnings")) if item is not None]
        llm_tokens = merge_correlation_llm_tokens(
            graph_output.get("llm_tokens") or result_payload.get("llm_tokens")
        )
        tokens = {
            "input": int(llm_tokens.get("input", graph_output.get("input_tokens", 0)) or 0),
            "output": int(llm_tokens.get("output", graph_output.get("output_tokens", 0)) or 0),
            "breakdown": _safe_dict(llm_tokens.get("breakdown")),
        }
        return {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "agent": "correlation_agent",
            "status": "success",
            "recommended_action": recommended_action,
            "visual_component": {},
            "output": result_payload,
            "explanation": {},
            "validation": {
                "is_valid": True,
                "checks": [],
                "warnings": output_warnings,
                "errors": [],
            },
            "tokens": tokens,
            "execution": {
                "start_time": str(start_time or ""),
                "end_time": end_time,
                "duration_ms": duration_ms,
                "version": APP_VERSION,
            },
        }

    def validate_result(self, result: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        if "result" not in result:
            warnings.append("Missing 'result' key in graph output.")
        return warnings


# =============================
# Graph factory
# =============================

def build_app(
    yaml_path: str,
    snowflake_helper: Optional[SnowparkHelper] = None,
    snowflake_helper_builder: Optional[Callable[[], SnowparkHelper]] = None,
    output_root: str = "correlation_runs",
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
    **kwargs: Any,
) -> Callable[[Mapping[str, Any]], CorrelationAgentResponse]:
    """
    Build the correlation agent runner.

    Args:
        yaml_path: Path to the semantic model YAML.
        snowflake_helper: Optional SnowparkHelper instance.
        snowflake_helper_builder: Optional SnowparkHelper builder.
        output_root: Directory where run artifacts are written.
        llm: Optional LLM instance for executive summaries.
        llm_builder: Optional LLM builder callable.
        **kwargs: Additional AgentBase arguments (e.g., test_mode, debug).

    Returns:
        Callable that executes the correlation analysis for a given request payload.
    """
    agent = CorrelationAgent(
        yaml_path=yaml_path,
        snowflake_helper=snowflake_helper,
        snowflake_helper_builder=snowflake_helper_builder,
        output_root=output_root,
        llm=llm,
        llm_builder=llm_builder,
        **kwargs,
    )

    def run(payload: Mapping[str, Any]) -> CorrelationAgentResponse:
        if not isinstance(payload, Mapping):
            raise ValueError("Correlation runner expects a mapping payload.")
        if "intent" in payload:
            return agent.execute(
                intent=payload.get("intent"),
                conversation_id=payload.get("conversation_id"),
                query=payload.get("query"),
                context=payload.get("context"),
                output_root=payload.get("output_root"),
                save_parquet=payload.get("save_parquet"),
                disable_summary_creation=payload.get("disable_summary_creation"),
                generate_recommendations=payload.get("generate_recommendations"),
            )
        if "context" in payload or "query" in payload or "conversation_id" in payload:
            return agent.execute(
                conversation_id=payload.get("conversation_id"),
                query=payload.get("query"),
                context=payload.get("context"),
                output_root=payload.get("output_root"),
                save_parquet=payload.get("save_parquet"),
                disable_summary_creation=payload.get("disable_summary_creation"),
                generate_recommendations=payload.get("generate_recommendations"),
            )
        return agent.execute(intent=payload)

    return run