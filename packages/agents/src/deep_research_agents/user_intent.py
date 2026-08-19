from __future__ import annotations

import copy
import json
import logging
import re
from typing import (
    Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, TypedDict,
)

import yaml
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tenacity import stop_after_attempt, retry_if_exception_type, retry

from deep_research_core.app_exceptions import LLMInvocationError
from deep_research_utils.app_constant import AppConstants
from deep_research_utils.cache_utils import get_token_cache_obj


try:
    from deep_research_utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# =============================
# Module: User Intent Resolution
# =============================
#
# This module provides question resolution for analytics questions using a two-tier approach:
#
# 1. LLM-FIRST RESOLUTION (Primary)
#    - Uses a language model to understand the question and extract semantic information
#    - Leverages a semantic model (YAML-based) to provide context about available dimensions,
#      filters, and metrics
#    - Produces structured output with filters, group-by fields, and analysis mode selection
#
# 2. RULE-BASED FALLBACK (Secondary)
#    - Deterministic pattern matching when LLM is unavailable or fails
#    - Extracts filters from named filters and dimension values defined in the semantic model
#    - Fully schema-driven with no hardcoded assumptions
#
# KEY COMPONENTS:
#
# Semantic Model:
#   - YAML file defining dimensions, filters, and their synonyms
#   - Supports dimension values with aliases for schema-driven value matching
#   - Enables state field detection and other heuristic extraction
#
# Semantic Index:
#   - In-memory lookup structure built from the semantic model
#   - Maps aliases to canonical names for fast phrase matching
#   - Tracks likely state fields and dimension value synonyms
#
# Intent Output:
#   - Structured result containing:
#     * metric_hint: Suggested metric if detected
#     * group_by: Dimensions to group results by
#     * filters: Extracted filter conditions with field, operator, value, source
#     * analysis_mode: Selected analysis mode name (if any)
#     * analysis_mode_parameters: Selected analysis mode configuration (if any)
#     * raw_question: Original user question
#
# WORKFLOW:
#   1. Load semantic model from YAML
#   2. Build semantic index for fast lookups
#   3. Summarize semantic model for LLM context
#   4. For each question:
#      a. Try LLM-based resolution with semantic context
#      b. Fall back to rule-based resolution if LLM fails
#      c. Select analysis mode
#      d. Return structured output
#
# USAGE:
#   app = build_app(yaml_path="/path/to/semantic_model.yaml")
#   result = app(question="Find what changes in Virginia?")
#   # result contains: filters, group_by, analysis_mode, etc.


# =============================
# Typed schemas
# =============================

FilterSource = Literal["dimension_match", "named_filter"]


class FilterCondition(TypedDict):
    field: str
    operator: str
    value: str
    source: FilterSource


class AnalysisModeDefinition(TypedDict, total=False):
    """Definition payload for a single analysis mode from the semantic model."""

    name: str
    aliases: List[str]
    description: str
    drill_metric: List[str]
    explainer_metrics: List[str]
    period: Dict[str, Any]
    drill_dimensions: List[str]
    exclude_if_filtered: bool
    stop_rules: Dict[str, Any]


class IntentOutput(TypedDict):
    metric_hint: Optional[str]
    group_by: List[str]
    filters: List[FilterCondition]
    analysis_mode: Optional[str]
    analysis_mode_parameters: Optional[AnalysisModeDefinition]
    raw_question: str
    validation_warnings: List[str]


# =============================
# Pydantic schemas for LLM structured output
# =============================

class FilterConditionSchema(BaseModel):
    """Pydantic model for filter conditions from LLM output."""
    field: str = Field(description="Field name to filter on")
    operator: str = Field(
        default="=",
        description="""Filter operator. Supported operators:
        - '=' : Exact match (e.g., state = 'Virginia')
        - '>' : Greater than (e.g., amount > 100)
        - '<' : Less than (e.g., amount < 100)
        - '>=' : Greater than or equal to
        - '<=' : Less than or equal to
        - 'in' : Value in list (e.g., state in ['VA', 'MD'])
        - 'between' : Value between two values (e.g., date between '2024-01-01' and '2024-12-31')
        - 'like' : Pattern matching (e.g., name like '%John%')
        - 'named_filter' : Reference to a predefined filter from semantic model
        """
    )
    value: str = Field(description="Filter value as string. For 'in' operator, use comma-separated values. For 'between', use 'value1 and value2' format.")
    source: FilterSource = Field(default="dimension_match", description="Source of the filter: dimension_match or named_filter")


class IntentOutputSchema(BaseModel):
    """Pydantic model for filter/group-by extraction output from LLM."""

    metric_hint: Optional[str] = Field(default=None, description="Suggested metric name if detected")
    group_by: List[str] = Field(default_factory=list, description="Dimensions or fields to group by")
    filters: List[FilterConditionSchema] = Field(default_factory=list, description="Extracted filter conditions")
    raw_question: str = Field(description="Original user question")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings about mismatches with semantic model")


class AnalysisModeSelectionSchema(BaseModel):
    """Pydantic model for analysis mode selection output from LLM."""

    analysis_mode: Optional[str] = Field(
        default=None,
        description="Name of the selected analysis mode (must match one of the provided names), or null if none apply",
    )


class SemanticIndex(TypedDict):
    dimension_alias_to_name: Dict[str, str]
    filter_alias_to_def: Dict[str, Dict[str, str]]
    known_dimension_names: List[str]
    known_filter_names: List[str]
    likely_state_fields: List[str]
    dimension_value_aliases: Dict[str, List[Dict[str, str]]]


class GraphState(TypedDict, total=False):
    question: str
    context: Optional[Dict[str, str]]
    semantic_model: Dict[str, Any]
    semantic_index: SemanticIndex
    semantic_summary: Dict[str, Any]
    llm: Any
    ehap: Any  # EHAP instance for token retry
    result: IntentOutput


# =============================
# Constants
# =============================

FILTER_SOURCE_VALUES: Set[str] = {"dimension_match", "named_filter"}

ANALYSIS_CONTEXT_KEYS: Set[str] = {
    "analysis_overrides",
    "drill_metric",
    "period",
    "rolling_window",
    "start_time",
    "end_time",
}

TREND_PATTERNS: Tuple[str, ...] = (
    "change",
    "changes",
    "trend",
    "trends",
    "over time",
    "movement",
    "moved",
    "delta",
    "increase",
    "decrease",
    "growth",
    "decline",
)

FIELD_KEY_HINTS = {
    "name",
    "field",
    "column",
    "dimension",
    "metric",
    "measure",
    "attribute",
    "id",
    "code",
}
FILTER_KEY_HINTS = {"filter", "filters"}
SYNONYM_KEY_HINTS = {"synonyms", "aliases", "alias"}


# =============================
# YAML loading + normalization
# =============================

def load_semantic_yaml(path: str) -> Dict[str, Any]:
    logger.debug(f"Loading semantic YAML from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    result = payload if isinstance(payload, dict) else {}
    logger.debug(f"Loaded semantic model with {len(result)} top-level keys")
    return result


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def phrase_in_text(phrase: str, text: str) -> bool:
    """
    Safer phrase matching than raw substring checks.
    Works on normalized text and respects token boundaries.
    """
    p = normalize(phrase)
    t = normalize(text)
    if not p or not t:
        return False
    pattern = rf"(?<![a-z0-9_]){re.escape(p)}(?![a-z0-9_])"
    return re.search(pattern, t) is not None


# =============================
# Semantic indexing
# =============================

def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _register_alias(
    alias_map: Dict[str, str],
    alias: str,
    canonical_name: str,
) -> None:
    alias_norm = normalize(alias)
    if alias_norm:
        alias_map[alias_norm] = canonical_name


def _register_dimension_value_alias(
    dimension_value_aliases: Dict[str, List[Dict[str, str]]],
    field_name: str,
    alias: str,
    canonical_value: str,
) -> None:
    alias_norm = normalize(alias)
    if not alias_norm:
        return

    entries = dimension_value_aliases.setdefault(alias_norm, [])
    record = {"field": field_name, "value": canonical_value}

    if record not in entries:
        entries.append(record)


def _index_dimension_values(
    dimension_value_aliases: Dict[str, List[Dict[str, str]]],
    field_name: str,
    raw_values: Any,
) -> None:
    """
    Index dimension values if present in the semantic model.

    Supported examples:
      sample_values:
        - name: VA
          synonyms: ["Virginia"]
        - name: CA
          synonyms: ["California"]

      sample_values:
        - VA
        - CA

      sample_values:
        VA:
          synonyms: ["Virginia"]
        CA:
          synonyms: ["California"]
    """
    if isinstance(raw_values, list):
        for item in raw_values:
            if isinstance(item, str):
                canonical_value = item.strip()
                if canonical_value:
                    _register_dimension_value_alias(
                        dimension_value_aliases,
                        field_name,
                        canonical_value,
                        canonical_value,
                    )

            elif isinstance(item, dict):
                item_dict = _safe_dict(item)
                canonical_value = str(item_dict.get("name", "")).strip()
                if not canonical_value:
                    continue

                _register_dimension_value_alias(
                    dimension_value_aliases,
                    field_name,
                    canonical_value,
                    canonical_value,
                )

                for syn in _safe_list(item_dict.get("synonyms")):
                    syn_str = str(syn).strip()
                    if syn_str:
                        _register_dimension_value_alias(
                            dimension_value_aliases,
                            field_name,
                            syn_str,
                            canonical_value,
                        )

    elif isinstance(raw_values, dict):
        for raw_key, raw_val in raw_values.items():
            canonical_value = str(raw_key).strip()
            if not canonical_value:
                continue

            _register_dimension_value_alias(
                dimension_value_aliases,
                field_name,
                canonical_value,
                canonical_value,
            )

            if isinstance(raw_val, dict):
                for syn in _safe_list(raw_val.get("synonyms")):
                    syn_str = str(syn).strip()
                    if syn_str:
                        _register_dimension_value_alias(
                            dimension_value_aliases,
                            field_name,
                            syn_str,
                            canonical_value,
                        )


def build_semantic_index(semantic_model: Dict[str, Any]) -> SemanticIndex:
    """
    Build a lookup index from the YAML:
    - dimensions and their synonyms
    - named filters and their synonyms
    - likely state fields for heuristic extraction
    - dimension values and their synonyms (schema-driven geography)
    """
    logger.debug("Building semantic index from semantic model")
    dimension_alias_to_name: Dict[str, str] = {}
    filter_alias_to_def: Dict[str, Dict[str, str]] = {}
    known_dimension_names: Set[str] = set()
    known_filter_names: Set[str] = set()
    likely_state_fields: Set[str] = set()
    dimension_value_aliases: Dict[str, List[Dict[str, str]]] = {}

    def maybe_track_state_field(field_name: str) -> None:
        n = normalize(field_name)
        if "state" in n:
            likely_state_fields.add(field_name)

    for flt in _safe_list(semantic_model.get("filters")):
        flt = _safe_dict(flt)
        name = str(flt.get("name", "")).strip()
        expr = str(flt.get("expr", "")).strip()
        if not name:
            continue

        known_filter_names.add(name)
        filter_alias_to_def[normalize(name)] = {"name": name, "expr": expr}

        for syn in _safe_list(flt.get("synonyms")):
            syn_str = str(syn).strip()
            if syn_str:
                filter_alias_to_def[normalize(syn_str)] = {"name": name, "expr": expr}

    for table in _safe_list(semantic_model.get("tables")):
        table = _safe_dict(table)

        for dim_group_key in ("dimensions", "time_dimensions"):
            for dim in _safe_list(table.get(dim_group_key)):
                dim = _safe_dict(dim)
                canonical_name = str(dim.get("name", "")).strip()
                if not canonical_name:
                    continue

                known_dimension_names.add(canonical_name)
                maybe_track_state_field(canonical_name)
                _register_alias(dimension_alias_to_name, canonical_name, canonical_name)

                for syn in _safe_list(dim.get("synonyms")):
                    syn_str = str(syn).strip()
                    if syn_str:
                        _register_alias(dimension_alias_to_name, syn_str, canonical_name)

                _index_dimension_values(
                    dimension_value_aliases=dimension_value_aliases,
                    field_name=canonical_name,
                    raw_values=dim.get("sample_values"),
                )

        for flt in _safe_list(table.get("filters")):
            flt = _safe_dict(flt)
            name = str(flt.get("name", "")).strip()
            expr = str(flt.get("expr", "")).strip()
            if not name:
                continue

            known_filter_names.add(name)
            filter_alias_to_def[normalize(name)] = {"name": name, "expr": expr}

            for syn in _safe_list(flt.get("synonyms")):
                syn_str = str(syn).strip()
                if syn_str:
                    filter_alias_to_def[normalize(syn_str)] = {"name": name, "expr": expr}

    result = {
        "dimension_alias_to_name": dimension_alias_to_name,
        "filter_alias_to_def": filter_alias_to_def,
        "known_dimension_names": sorted(known_dimension_names),
        "known_filter_names": sorted(known_filter_names),
        "likely_state_fields": sorted(likely_state_fields),
        "dimension_value_aliases": dimension_value_aliases,
    }
    logger.debug(
        f"Semantic index built: {len(dimension_alias_to_name)} dimension aliases, "
        f"{len(filter_alias_to_def)} filter aliases, "
        f"{len(known_dimension_names)} dimensions, "
        f"{len(known_filter_names)} filters, "
        f"{len(likely_state_fields)} state fields"
    )
    return result


# =============================
# Schema-agnostic summarization
# =============================

def _format_path(path_parts: Sequence[str]) -> str:
    formatted = ""
    for part in path_parts:
        if part.startswith("["):
            formatted += part
        else:
            formatted = f"{formatted}.{part}" if formatted else part
    return formatted


def summarize_semantic_model(
    semantic_model: Any,
    *,
    max_entries: int = 200,
    max_depth: int = 6,
    max_value_length: int = 120,
) -> Dict[str, Any]:
    """
    Create a compact, schema-agnostic summary of any semantic model structure.
    Optimized for LLM prompting and lower token usage.
    """
    logger.debug(f"Summarizing semantic model (max_entries={max_entries}, max_depth={max_depth})")
    field_candidates: Set[str] = set()
    filter_candidates: Set[str] = set()
    synonym_terms: Set[str] = set()
    string_paths: List[Dict[str, str]] = []
    keys: Set[str] = set()

    def capture_string(path: List[str], value: str, key_hint: Optional[str]) -> None:
        if len(string_paths) >= max_entries:
            return

        value = value.strip()
        if not value:
            return

        if len(value) > max_value_length:
            value = value[:max_value_length] + "..."

        string_paths.append({"path": _format_path(path), "value": value})

        if key_hint and key_hint in FIELD_KEY_HINTS:
            field_candidates.add(value)

        if key_hint and any(hint in key_hint for hint in FILTER_KEY_HINTS):
            filter_candidates.add(value)

    def walk(node: Any, path: List[str], depth: int) -> None:
        if depth > max_depth or len(string_paths) >= max_entries:
            return

        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                key_lower = key_str.lower()
                keys.add(key_str)
                next_path = path + [key_str]

                if isinstance(value, (str, int, float, bool)):
                    capture_string(next_path, str(value), key_lower)
                elif isinstance(value, list):
                    if key_lower in SYNONYM_KEY_HINTS:
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                synonym_terms.add(item.strip())
                    for idx, item in enumerate(value):
                        if len(string_paths) >= max_entries:
                            break
                        walk(item, next_path + [f"[{idx}]"], depth + 1)
                elif isinstance(value, dict):
                    walk(value, next_path, depth + 1)

        elif isinstance(node, list):
            for idx, item in enumerate(node):
                if len(string_paths) >= max_entries:
                    break
                walk(item, path + [f"[{idx}]"], depth + 1)

    walk(semantic_model, [], 0)

    result = {
        "model_type": type(semantic_model).__name__,
        "field_candidates": sorted(field_candidates)[:max_entries],
        "filter_candidates": sorted(filter_candidates)[:max_entries],
        "synonym_terms": sorted(synonym_terms)[:max_entries],
        "string_paths": string_paths,
        "keys": sorted(keys)[:max_entries],
    }
    logger.debug(
        f"Semantic model summary: {len(field_candidates)} field candidates, "
        f"{len(filter_candidates)} filter candidates, "
        f"{len(synonym_terms)} synonym terms, {len(string_paths)} string paths"
    )
    return result


# =============================
# Defensive serialization
# =============================

def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _to_jsonable(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 100,
    _depth: int = 0,
) -> Any:
    if _depth >= max_depth:
        return "<max_depth_reached>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["<truncated>"] = f"<{len(value) - max_items} more keys>"
                break
            out[str(k)] = _to_jsonable(v, max_depth=max_depth, max_items=max_items, _depth=_depth + 1)
        return out

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out_list: List[Any] = []
        for item in seq[:max_items]:
            out_list.append(_to_jsonable(item, max_depth=max_depth, max_items=max_items, _depth=_depth + 1))
        if len(seq) > max_items:
            out_list.append(f"<truncated: {len(seq) - max_items} more items>")
        return out_list

    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return _to_jsonable(value.model_dump(), max_depth=max_depth, max_items=max_items, _depth=_depth + 1)
        except Exception:
            return str(value)

    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value), max_depth=max_depth, max_items=max_items, _depth=_depth + 1)
        except Exception:
            return str(value)

    return str(value)


def _serialize_for_llm(value: Any, *, max_chars: int = 25_000) -> str:
    payload = _to_jsonable(value)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return _truncate_text(text, max_chars)


# =============================
# LLM configuration
# =============================

INTENT_RESOLUTION_SYSTEM_PROMPT = """You are a question resolver for analytics questions that will be used to generate SQL queries.

You will receive:
1. a user question
2. a compact semantic model summary with sample values
3. a canonical semantic index with dimension_value_aliases

Analyze the question and extract relevant semantic information needed for query generation.

CRITICAL RULES FOR FILTER VALUES:
- **ALWAYS use exact values from sample_values when available** - this is critical for SQL query accuracy
- Check the semantic_summary.string_paths for sample_values arrays
- If a field has sample_values, your filter value MUST exactly match one of them
- For geographic references (states, cities), use the canonical abbreviation if that's what's in sample_values
- For codes (LOB, product, procedure), use the exact code format from sample_values
- The dimension_value_aliases may provide mappings from user terms to canonical values - use these mappings

Field and Filter Name Rules:
- Prefer canonical field/filter names present in the provided semantic context
- Use semantic synonyms only to map back to canonical names
- If uncertain, keep filters empty

Supported filter operators:
- '=' : Exact match (e.g., state = 'VA', procedure_code = '99291')
- '>' : Greater than (e.g., amount > 100)
- '<' : Less than (e.g., amount < 100)
- '>=' : Greater than or equal to (e.g., year >= 2024)
- '<=' : Less than or equal to (e.g., year <= 2024)
- 'in' : Value in list (e.g., state in ['VA', 'MD', 'PA'])
- 'between' : Value between range (e.g., date between '2024-01-01' and '2024-12-31')
- 'like' : Pattern matching (e.g., diagnosis like '%diabetes%')
- 'named_filter' : Reference to a predefined named filter from the semantic model

Examples showing CORRECT use of sample values:
- "in Virginia" + sample_values=['VA', 'MD', ...] → {field: 'service_area_state', operator: '=', value: 'VA'}  ✓
- "in Virginia" → {field: 'service_area_state', operator: '=', value: 'Virginia'}  ✗ WRONG - not in sample values
- "Commercial LOB" + sample_values=['Commercial', 'Medicare', 'Medicaid'] → {field: 'lob_description', operator: '=', value: 'Commercial'}  ✓
- "procedure code 99291" → {field: 'procedure_code', operator: '=', value: '99291'}  ✓
- "IP BH claims" + sample_values=['IP BH', 'IP Med/Surg', ...] → {field: 'hcc_medium', operator: '=', value: 'IP BH'}  ✓
- "2025 vs 2024" → {field: 'incurred_month', operator: 'between', value: '202401 and 202512'}  ✓
"""

ANALYSIS_MODE_SELECTION_SYSTEM_PROMPT = """You are selecting a single analysis mode for an analytics question.

You will receive:
1. the user question
2. context filters (UI selections + extracted filters)
3. available analysis modes (name + description + key parameters)

Rules:
- Pick at most one analysis mode name from the provided list.
- Return null if none apply.
- Choose the most specific matching mode when multiple seem plausible.
- Do not invent names.

Note: Today we only select a single analysis mode; in the future this can expand to chains of modes running sequentially or in parallel.
"""


def build_llm(
    model_name: Optional[str] = None,
    reasoning_effort: str = "medium",
    summary_mode: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Build LLM client factory with automatic token refresh.
    
    Returns a factory function that creates ChatOpenAI clients with fresh tokens.
    This ensures long-running processes always use valid EHAP tokens.
    
    Args:
        model_name: LLM model name (defaults to EHAP_LLM_MODEL)
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        summary_mode: Summary mode ("auto", "detailed", or None)
        
    Returns:
        Factory function that returns ChatOpenAI client with current token
    """
    from deep_research_utils import EHAPBase  # type: ignore
    
    model_name = model_name or AppConstants.EHAP_LLM_MODEL
    
    # Create EHAP instance for token management
    ehap = EHAPBase(
        base_url=AppConstants.EHAP_BASE_URL,
        client_id=AppConstants.EHAP_CLIENT_ID,
        client_secret=AppConstants.EHAP_CLIENT_SECRET,
        verify=AppConstants.SSL_CERT_FILE or False,
    )
    
    def _create_llm() -> ChatOpenAI:
        """Factory function that creates LLM with fresh token."""
        token = ehap.get_token()  # Always get fresh token
        
        try:
            logger.debug(f"Creating LLM client: model={model_name}, reasoning_effort={reasoning_effort}")
            client = ChatOpenAI(
                base_url=AppConstants.OPENAI_BASE_URL,
                model=model_name,
                api_key=token,
                extra_body={
                    "reasoning_effort": reasoning_effort,
                    "summary": summary_mode,
                },
                http_client=AppConstants.http_client_,
                http_async_client=AppConstants.http_async_client_,
            )
            return client
        except Exception as e:
            logger.error(f"ERROR in making OpenAI client: {str(e)}")
            raise
    
    # Return tuple of (llm, ehap) for retry utility support
    return _create_llm(), ehap


# =============================
# Context filters and validation
# =============================

def merge_context_filters(
    existing_filters: List[FilterCondition],
    context: Optional[Dict[str, str]],
) -> List[FilterCondition]:
    """
    Merge UI context filters with extracted filters.

    Args:
        existing_filters: Filters extracted from the user question
        context: Optional dict of field->value pairs from UI filters

    Returns:
        Combined list of filters with context filters added
    """
    if not context:
        logger.debug("No context filters to merge")
        return existing_filters

    logger.info(f"Merging {len(context)} context filters with {len(existing_filters)} extracted filters")

    combined = list(existing_filters)
    for field, value in context.items():
        # Preserve list/tuple/set values so render_filter_clause can emit IN (...).
        # Stringify scalars for backward compat with previous behavior.
        if isinstance(value, (list, tuple, set)):
            normalized_value: Any = [str(item) for item in value]
        else:
            normalized_value = str(value)
        context_filter: FilterCondition = {
            "field": field,
            "operator": "=",
            "value": normalized_value,
            "source": "dimension_match",
        }
        combined.append(context_filter)
        logger.debug(f"Added context filter: {field} = {normalized_value}")

    return _dedupe_filters(combined)


def split_context_filters(
    context: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Split incoming context into filterable fields vs analysis-mode hints."""
    if not context:
        logger.debug("split_context_filters: No context provided")
        return None, {}

    logger.debug(f"split_context_filters: Incoming context: {context}")
    filter_context: Dict[str, Any] = {}
    analysis_context: Dict[str, Any] = {}
    for key, value in context.items():
        if key == "analysis_overrides" and isinstance(value, Mapping):
            logger.debug(f"split_context_filters: Adding analysis_overrides: {value}")
            analysis_context.update(value)
        elif key in ANALYSIS_CONTEXT_KEYS:
            logger.debug(f"split_context_filters: Adding to analysis_context: {key} = {value}")
            analysis_context[key] = value
        else:
            logger.debug(f"split_context_filters: Adding to filter_context: {key} = {value}")
            filter_context[key] = value

    logger.info(f"split_context_filters: Split into filter_context={filter_context}, analysis_context={analysis_context}")
    return (filter_context or None), analysis_context


def validate_intent_output(
    intent: IntentOutput,
    semantic_model: Dict[str, Any],
    semantic_index: SemanticIndex,
    live_filter_values: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Validate the resolved output against the semantic model.

    Checks:
    1. All dimension/time_dimension/fact names in filters and group_by exist in YAML
    2. Filter values are valid for their field

    Per-filter validation precedence (when ``live_filter_values`` is provided):
      a. If the field appears under ``live_filter_values["time_dimensions"]``, the
         filter value is range-checked against the live MIN/MAX from Snowflake.
      b. Else if the field appears under ``live_filter_values["dimensions"]``,
         the filter value is checked against the live distinct list — unless the
         live entry is flagged ``is_free_text`` (high cardinality), in which case
         no categorical check runs.
      c. Else the historic YAML ``sample_values`` allowlist applies (back-compat
         for callers that don't pass live data).

    ``live_filter_values`` should be shaped like::

        {
            "dimensions": {
                "service_area_state": {"values": ["CA", "NY"], "is_free_text": False},
                "rendering_provider_name": {"values": [], "is_free_text": True},
            },
            "time_dimensions": {
                "snap_month": {"min": "202501", "max": "202606"},
            },
        }

    Returns:
        List of validation warning messages
    """
    warnings: List[str] = []
    logger.debug(
        "Validating resolved output with %s filters and %s group_by fields",
        len(intent["filters"]),
        len(intent["group_by"]),
    )

    # Build sets of valid field names from semantic model
    valid_dimensions: Set[str] = set()
    valid_time_dimensions: Set[str] = set()
    valid_facts: Set[str] = set()
    field_to_samples: Dict[str, List[str]] = {}

    for table in _safe_list(semantic_model.get("tables")):
        table = _safe_dict(table)

        # Collect dimensions
        for dim in _safe_list(table.get("dimensions")):
            dim = _safe_dict(dim)
            name = str(dim.get("name", "")).strip()
            if name:
                valid_dimensions.add(name)

                # Collect sample values if available
                sample_values = dim.get("sample_values")
                if sample_values:
                    if isinstance(sample_values, list):
                        field_to_samples[name] = [str(v).strip() for v in sample_values if str(v).strip()]

        # Collect time dimensions
        for time_dim in _safe_list(table.get("time_dimensions")):
            time_dim = _safe_dict(time_dim)
            name = str(time_dim.get("name", "")).strip()
            if name:
                valid_time_dimensions.add(name)

        # Collect facts
        for fact in _safe_list(table.get("facts")):
            fact = _safe_dict(fact)
            name = str(fact.get("name", "")).strip()
            if name:
                valid_facts.add(name)

    all_valid_fields = valid_dimensions | valid_time_dimensions | valid_facts

    logger.debug(f"Found {len(valid_dimensions)} dimensions, {len(valid_time_dimensions)} time dimensions, {len(valid_facts)} facts")

    live_dim_index: Dict[str, Dict[str, Any]] = {}
    live_time_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(live_filter_values, dict):
        raw_dims = live_filter_values.get("dimensions")
        if isinstance(raw_dims, dict):
            live_dim_index = {str(k): v for k, v in raw_dims.items() if isinstance(v, dict)}
        raw_times = live_filter_values.get("time_dimensions")
        if isinstance(raw_times, dict):
            live_time_index = {str(k): v for k, v in raw_times.items() if isinstance(v, dict)}

    # Validate group_by fields
    for field in intent["group_by"]:
        if field not in all_valid_fields:
            warning = f"group_by field '{field}' not found in semantic model (dimensions/time_dimensions/facts)"
            warnings.append(warning)
            logger.warning(warning)

    # Validate filter fields and values
    for filter_cond in intent["filters"]:
        field = filter_cond["field"]
        value = filter_cond["value"]
        source = filter_cond["source"]

        # Skip validation for named_filters as they reference filter names, not dimension names
        if source == "named_filter":
            # Check if the filter name exists
            if field not in semantic_index["known_filter_names"]:
                warning = f"named_filter '{field}' not found in semantic model filters"
                warnings.append(warning)
                logger.warning(warning)
            continue

        # Validate field name exists
        if field not in all_valid_fields:
            warning = f"filter field '{field}' not found in semantic model (dimensions/time_dimensions/facts)"
            warnings.append(warning)
            logger.warning(warning)
            continue

        if filter_cond["operator"] == "in":
            filter_values = [v.strip() for v in str(value).split(",")]
        else:
            filter_values = [value]

        if field in live_time_index:
            entry = live_time_index[field]
            min_val = entry.get("min")
            max_val = entry.get("max")
            for fval in filter_values:
                if fval in (None, ""):
                    continue
                if not _value_within_range(fval, min_val, max_val):
                    warning = (
                        f"filter value '{fval}' for time dimension '{field}' is outside "
                        f"the live range [{min_val}..{max_val}]"
                    )
                    warnings.append(warning)
                    logger.warning(warning)
            continue

        if field in live_dim_index:
            entry = live_dim_index[field]
            if entry.get("is_free_text"):
                continue
            allowed_raw = entry.get("values") or []
            if not isinstance(allowed_raw, list):
                allowed_raw = list(allowed_raw)
            allowed = [str(v).strip() for v in allowed_raw if str(v).strip()]
            if not allowed:
                continue
            for fval in filter_values:
                fval_str = str(fval).strip() if fval is not None else ""
                if fval_str and fval_str not in allowed:
                    warning = (
                        f"filter value '{fval_str}' for field '{field}' is not one of the "
                        f"live values from Snowflake (sample: {allowed[:5]}{'…' if len(allowed) > 5 else ''})"
                    )
                    warnings.append(warning)
                    logger.warning(warning)
            continue

        # Back-compat: YAML sample_values allowlist when no live values were supplied.
        if field in field_to_samples:
            sample_values = field_to_samples[field]
            for fval in filter_values:
                if fval and fval not in sample_values:
                    warning = f"filter value '{fval}' for field '{field}' not in sample values: {sample_values}"
                    warnings.append(warning)
                    logger.warning(warning)

    logger.info(f"Validation complete: {len(warnings)} warning(s) found")
    return warnings


def _value_within_range(value: Any, min_val: Any, max_val: Any) -> bool:
    """Range-check ``value`` against ``[min_val, max_val]``.

    Coerces both sides to int when all three look numeric (handles YYYYMM as int
    or str); otherwise falls back to string comparison. Returns True when either
    bound is missing — the validator should not fabricate a constraint the UI
    didn't supply.
    """
    if min_val in (None, "") or max_val in (None, ""):
        return True
    try:
        v_int = int(str(value).strip())
        lo_int = int(str(min_val).strip())
        hi_int = int(str(max_val).strip())
        return lo_int <= v_int <= hi_int
    except (TypeError, ValueError):
        v_str = str(value).strip()
        lo_str = str(min_val).strip()
        hi_str = str(max_val).strip()
        return lo_str <= v_str <= hi_str


# =============================
# Helper functions for rule-based fallback
# =============================

def _dedupe_filters(filters: List[FilterCondition]) -> List[FilterCondition]:
    """Deduplicate filter conditions based on field, operator, value, and source."""
    out: List[FilterCondition] = []
    seen: Set[Tuple[str, str, Any, str]] = set()

    for f in filters:
        raw_value = f["value"]
        # List/set values are unhashable — collapse to a sorted tuple of strings so
        # dedupe still works for context filters that came in as multi-select picks.
        if isinstance(raw_value, list):
            value_key: Any = ("__list__", tuple(str(item) for item in raw_value))
        elif isinstance(raw_value, set):
            value_key = ("__set__", tuple(sorted(str(item) for item in raw_value)))
        elif isinstance(raw_value, tuple):
            value_key = ("__tuple__", tuple(str(item) for item in raw_value))
        else:
            value_key = raw_value
        key = (f["field"], f["operator"], value_key, f["source"])
        if key not in seen:
            out.append(f)
            seen.add(key)

    return out


def resolve_intent_with_llm(
    question: str,
    semantic_summary: Dict[str, Any],
    semantic_index: SemanticIndex,
    llm: Any,
    ehap: Optional[Any] = None,
) -> Optional[IntentOutput]:
    logger.info(f"resolve_intent_with_llm: Processing question: {question[:100]}...")
    
    # Handle case where llm might be a tuple (defensive programming)
    if isinstance(llm, tuple):
        logger.warning(f"resolve_intent_with_llm: llm is tuple {type(llm)}, extracting first element")
        original_tuple = llm
        llm = llm[0]
        if ehap is None and len(original_tuple) > 1:
            ehap = original_tuple[1]

    # Extract sample values for easier LLM access
    field_sample_values: Dict[str, List[str]] = {}
    for path_item in semantic_summary.get("string_paths", []):
        path = path_item.get("path", "")
        if "sample_values" in path:
            # Extract field name from path like "tables[0].dimensions[5].sample_values[2]"
            parts = path.split(".")
            for i, part in enumerate(parts):
                if "dimensions" in part or "time_dimensions" in part:
                    # Get the name from the previous parts
                    name_path = ".".join(parts[:i+1] + ["name"])
                    # Find corresponding name in string_paths
                    for name_item in semantic_summary.get("string_paths", []):
                        if name_item.get("path") == name_path:
                            field_name = name_item.get("value")
                            if field_name:
                                if field_name not in field_sample_values:
                                    field_sample_values[field_name] = []
                                value = path_item.get("value")
                                if value and value not in field_sample_values[field_name]:
                                    field_sample_values[field_name].append(value)
                            break

    llm_context = {
        "semantic_summary": semantic_summary,
        "canonical_index": {
            "known_dimension_names": semantic_index["known_dimension_names"],
            "known_filter_names": semantic_index["known_filter_names"],
            "likely_state_fields": semantic_index["likely_state_fields"],
            "dimension_value_aliases": semantic_index["dimension_value_aliases"],
        },
        "field_sample_values": field_sample_values,  # Explicitly provide field->sample_values mapping
    }

    system_prompt = INTENT_RESOLUTION_SYSTEM_PROMPT

    user_prompt = (
        f"Question: {question}\n\n"
        f"Semantic context JSON: {_serialize_for_llm(llm_context)}"
    )

    logger.debug(f"System prompt length: {len(system_prompt)} characters")
    logger.debug(f"User prompt length: {len(user_prompt)} characters")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Use structured output with Pydantic model
        if ehap is not None:
            from deep_research_utils.ehap_retry import structured_llm_invoke
            result_schema, _ = structured_llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                schema=IntentOutputSchema,
                llm_reinitializer=lambda: build_llm()[0],
            )
        else:
            # Fallback to old behavior for backward compatibility
            structured_llm = llm.with_structured_output(IntentOutputSchema)
            result_schema: IntentOutputSchema = structured_llm.invoke(messages)

        logger.debug("LLM structured output received for filters/group_by")

        # Convert Pydantic model to TypedDict for backward compatibility
        result: IntentOutput = {
            "metric_hint": result_schema.metric_hint,
            "group_by": list(result_schema.group_by),
            "filters": [
                {
                    "field": fc.field,
                    "operator": fc.operator,
                    "value": fc.value,
                    "source": fc.source,
                }
                for fc in result_schema.filters
            ],
            "analysis_mode": None,
            "analysis_mode_parameters": None,
            "raw_question": result_schema.raw_question,
            "validation_warnings": result_schema.validation_warnings,
        }

        logger.info(f"resolve_intent_with_llm: LLM extraction succeeded - filters={result['filters']}, group_by={result['group_by']}, metric_hint={result['metric_hint']}")
        return result

    except Exception as e:
        logger.warning(f"Failed to get structured output from LLM: {e}")
        return None


# =============================
# Rule-based fallback
# =============================

def match_named_filters(question: str, semantic_index: SemanticIndex) -> List[FilterCondition]:
    logger.debug(f"Matching named filters for question: {question[:100]}...")
    matches: List[FilterCondition] = []

    for alias, flt in semantic_index["filter_alias_to_def"].items():
        if phrase_in_text(alias, question):
            logger.debug(f"Matched named filter: {flt['name']}")
            matches.append(
                {
                    "field": flt["name"],
                    "operator": "named_filter",
                    "value": flt["expr"],
                    "source": "named_filter",
                }
            )

    logger.debug(f"Found {len(matches)} named filter matches")
    return _dedupe_filters(matches)


def resolve_state_field(semantic_index: SemanticIndex) -> Optional[str]:
    preferred_aliases = ["service_area_state", "service_state", "state"]

    for alias in preferred_aliases:
        canonical = semantic_index["dimension_alias_to_name"].get(normalize(alias))
        if canonical:
            return canonical

    if semantic_index["likely_state_fields"]:
        return semantic_index["likely_state_fields"][0]

    return None


def extract_semantic_dimension_value_filters(
    question: str,
    semantic_index: SemanticIndex,
    *,
    candidate_fields: Optional[Set[str]] = None,
) -> List[FilterCondition]:
    """
    Extract filters only from dimension values defined in the semantic model.

    Example:
      question: "Find what changes in Virginia?"
      semantic model state dimension values:
        VA -> ["Virginia"]

    returns:
      [{"field": "service_area_state", "operator": "=", "value": "VA", "source": "dimension_match"}]
    """
    logger.debug(f"Extracting semantic dimension value filters for question: {question[:100]}...")
    matches: List[FilterCondition] = []

    for alias, records in semantic_index["dimension_value_aliases"].items():
        if not phrase_in_text(alias, question):
            continue

        for record in records:
            field_name = record["field"]
            if candidate_fields is not None and field_name not in candidate_fields:
                continue

            matches.append(
                {
                    "field": field_name,
                    "operator": "=",
                    "value": record["value"],
                    "source": "dimension_match",
                }
            )

    result = _dedupe_filters(matches)
    logger.debug(f"Found {len(result)} dimension value filter matches")
    return result


def extract_state_filter(question: str, semantic_index: SemanticIndex) -> List[FilterCondition]:
    """
    State extraction is fully semantic-model driven.
    No hardcoded geography table or state regex is used.
    """
    logger.debug(f"Extracting state filter for question: {question[:100]}...")
    state_field = resolve_state_field(semantic_index)
    if not state_field:
        logger.debug("No state field found in semantic index")
        return []

    q_norm = normalize(question)
    mentions_state_concept = any(
        candidate in q_norm
        for candidate in ("state", "service state", "service area state")
    )

    candidate_fields = {state_field}

    matches = extract_semantic_dimension_value_filters(
        question,
        semantic_index,
        candidate_fields=candidate_fields,
    )

    if matches:
        logger.debug(f"Found {len(matches)} state filter matches")
        return matches

    # If the user clearly references the state field but the model contains no state values,
    # we do not guess. Keep fallback deterministic and schema-driven.
    if mentions_state_concept:
        logger.debug("User mentioned state concept but no state values found in model")
        return []

    logger.debug("No state filters extracted")
    return []


def extract_group_by(
    question: str,
    filters: List[FilterCondition],
    semantic_index: SemanticIndex,
) -> List[str]:
    q = normalize(question)
    group_by: List[str] = []

    if any(pattern in q for pattern in TREND_PATTERNS):
        state_field = resolve_state_field(semantic_index)
        if state_field and any(f["field"] == state_field for f in filters):
            group_by.append(state_field)

    return group_by


# =============================
# Analysis mode selection
# =============================

def _analysis_mode_tokens(text: str) -> Set[str]:
    return set(normalize(text).replace("_", " ").split())


def _match_metric_candidate(candidate: Optional[str], metrics: Sequence[str]) -> Optional[str]:
    if not candidate or not metrics:
        return None

    candidate_norm = normalize(candidate).replace("_", " ").replace(".", " ")
    candidate_tokens = set(candidate_norm.split())
    best_score = 0
    best_metric: Optional[str] = None

    for metric in metrics:
        metric_norm = normalize(metric).replace("_", " ").replace(".", " ")
        if metric_norm == candidate_norm:
            return metric

        metric_tokens = set(metric_norm.split())
        score = len(candidate_tokens & metric_tokens)
        if score > best_score:
            best_score = score
            best_metric = metric

    return best_metric if best_score > 0 else None


def _match_rolling_window(candidate: Optional[str], options: Sequence[str]) -> Optional[str]:
    if not candidate or not options:
        return None

    candidate_tokens = _analysis_mode_tokens(candidate) - {"rolling", "roll", "window", "period"}
    best_score = 0
    best_option: Optional[str] = None

    for option in options:
        option_tokens = _analysis_mode_tokens(option)
        score = len(candidate_tokens & option_tokens)
        if score > best_score:
            best_score = score
            best_option = option

    return best_option if best_score > 0 else None


def extract_analysis_modes(semantic_model: Dict[str, Any]) -> List[AnalysisModeDefinition]:
    """Extract analysis mode definitions from the semantic model."""
    modes: List[AnalysisModeDefinition] = []
    for raw_mode in _safe_list(semantic_model.get("analysis_modes")):
        mode_dict = _safe_dict(raw_mode)
        name = str(mode_dict.get("name", "")).strip()
        if not name:
            continue
        modes.append(copy.deepcopy(mode_dict))
    return modes


def build_analysis_mode_index(
    analysis_modes: List[AnalysisModeDefinition],
) -> Dict[str, AnalysisModeDefinition]:
    """Build a normalized lookup for analysis mode names and aliases."""
    index: Dict[str, AnalysisModeDefinition] = {}
    for mode in analysis_modes:
        name = str(mode.get("name", "")).strip()
        if not name:
            continue
        index[normalize(name)] = mode
        for alias in _safe_list(mode.get("aliases")):
            alias_str = str(alias).strip()
            if alias_str:
                index[normalize(alias_str)] = mode
    return index


def match_analysis_mode_name(
    candidate: Optional[str],
    analysis_mode_index: Dict[str, AnalysisModeDefinition],
) -> Optional[str]:
    """Normalize and align a candidate name to a known analysis mode name."""
    if not candidate:
        return None
    candidate_norm = normalize(candidate)
    if candidate_norm in analysis_mode_index:
        return analysis_mode_index[candidate_norm].get("name")

    candidate_underscore = candidate_norm.replace(" ", "_")
    if candidate_underscore in analysis_mode_index:
        return analysis_mode_index[candidate_underscore].get("name")

    for name_norm, mode in analysis_mode_index.items():
        if candidate_norm and (candidate_norm in name_norm or name_norm in candidate_norm):
            return mode.get("name")

    candidate_tokens = _analysis_mode_tokens(candidate)
    best_score = 0
    best_name: Optional[str] = None
    for name_norm, mode in analysis_mode_index.items():
        name_tokens = _analysis_mode_tokens(name_norm)
        score = len(candidate_tokens & name_tokens)
        if score > best_score:
            best_score = score
            best_name = mode.get("name")
    return best_name


def rule_based_analysis_mode_selection(
    question: str,
    analysis_modes: List[AnalysisModeDefinition],
) -> Optional[str]:
    """Lightweight fallback for analysis mode selection without an LLM."""
    if not analysis_modes:
        return None

    q_tokens = _analysis_mode_tokens(question)
    trend_tokens = {"change", "changes", "trend", "trends", "delta", "increase", "decrease"}
    best_score = 0
    best_name: Optional[str] = None

    for mode in analysis_modes:
        name = str(mode.get("name", "")).strip()
        if not name:
            continue
        name_tokens = _analysis_mode_tokens(name)
        score = len(q_tokens & name_tokens)

        description = str(mode.get("description", "")).strip()
        if description:
            desc_tokens = _analysis_mode_tokens(description)
            score += len(q_tokens & desc_tokens & trend_tokens)

        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_score > 0 else None



@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(LLMInvocationError),
    retry_error_callback=lambda retry_state: None,
)
def resolve_analysis_mode_with_llm(
    question: str,
    filters: List[FilterCondition],
    context_filters: Optional[Dict[str, Any]],
    analysis_context: Optional[Dict[str, Any]],
    analysis_modes: List[AnalysisModeDefinition],
    llm: Any,
    ehap: Optional[Any] = None,
) -> Optional[str]:
    """Use the LLM to select the best analysis mode for the question."""
    # Handle case where llm might be a tuple (defensive programming)
    if isinstance(llm, tuple):
        logger.warning(f"resolve_analysis_mode_with_llm: llm is tuple {type(llm)}, extracting first element")
        original_tuple = llm
        llm = llm[0]
        if ehap is None and len(original_tuple) > 1:
            ehap = original_tuple[1]
    if not analysis_modes:
        return None

    llm_context = {
        "analysis_modes": analysis_modes,
        "context_filters": context_filters or {},
        "analysis_hints": analysis_context or {},
        "extracted_filters": filters,
    }
    user_prompt = (
        f"Question: {question}\n\n"
        f"Analysis mode context JSON: {_serialize_for_llm(llm_context, max_chars=12_000)}"
    )

    messages = [
        {"role": "system", "content": ANALYSIS_MODE_SELECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if ehap is not None:
            from deep_research_utils.ehap_retry import structured_llm_invoke
            result_schema, _ = structured_llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                schema=AnalysisModeSelectionSchema,
                llm_reinitializer=lambda: build_llm()[0],
            )
        else:
            # Fallback to old behavior for backward compatibility
            structured_llm = llm.with_structured_output(AnalysisModeSelectionSchema)
            result_schema: AnalysisModeSelectionSchema = structured_llm.invoke(messages)
        return result_schema.analysis_mode
    except Exception as exc:
        logger.warning("Failed to select analysis mode with LLM.", exc_info=exc)
        # llm, ehap = build_llm()
        raise LLMInvocationError() from exc


def resolve_analysis_mode(
    question: str,
    filters: List[FilterCondition],
    context_filters: Optional[Dict[str, Any]],
    analysis_context: Optional[Dict[str, Any]],
    semantic_model: Dict[str, Any],
    llm: Optional[Any],
    ehap: Optional[Any] = None,
) -> Tuple[Optional[str], Optional[AnalysisModeDefinition]]:
    """
    Resolve analysis mode for the question.

    Note: This returns a single analysis mode today. In the future, this can expand to
    a list for chained analysis modes running sequentially or in parallel.
    """
    analysis_modes = extract_analysis_modes(semantic_model)
    if not analysis_modes:
        return None, None

    analysis_mode_index = build_analysis_mode_index(analysis_modes)
    analysis_mode_candidate: Optional[str] = None

    if llm is not None:
        analysis_mode_candidate = resolve_analysis_mode_with_llm(
            question=question,
            filters=filters,
            context_filters=context_filters,
            analysis_context=analysis_context,
            analysis_modes=analysis_modes,
            llm=llm,
            ehap=ehap,
        )

    analysis_mode_name = match_analysis_mode_name(analysis_mode_candidate, analysis_mode_index)
    if analysis_mode_name is None:
        analysis_mode_name = rule_based_analysis_mode_selection(question, analysis_modes)

    if not analysis_mode_name:
        return None, None

    resolved_mode = analysis_mode_index.get(normalize(analysis_mode_name))
    if not resolved_mode:
        return None, None

    resolved_mode = copy.deepcopy(resolved_mode)
    analysis_context = analysis_context or {}

    drill_metric_options = _safe_list(resolved_mode.get("drill_metric"))
    metric_candidate = analysis_context.get("drill_metric")
    if metric_candidate is None:
        metric_candidate = question
    matched_metric = _match_metric_candidate(str(metric_candidate), drill_metric_options)
    if matched_metric:
        resolved_mode["drill_metric"] = [matched_metric]

    period = _safe_dict(resolved_mode.get("period"))
    rolling_window_options = _safe_list(period.get("rolling_window"))
    period_candidate: Optional[str] = None
    period_override = analysis_context.get("period")
    if isinstance(period_override, Mapping):
        period_candidate = period_override.get("rolling_window")  # type: ignore[assignment]
    elif isinstance(period_override, str):
        period_candidate = period_override
    period_candidate = period_candidate or analysis_context.get("rolling_window")
    if period_candidate is None:
        period_candidate = question
    matched_window = _match_rolling_window(str(period_candidate), rolling_window_options)
    if matched_window:
        updated_period = dict(period)
        updated_period["rolling_window"] = [matched_window]
        resolved_mode["period"] = updated_period

    start_time = analysis_context.get("start_time")
    end_time = analysis_context.get("end_time")
    if start_time or end_time:
        updated_period = dict(resolved_mode.get("period", {}))
        if start_time:
            updated_period["start_time"] = start_time
        if end_time:
            updated_period["end_time"] = end_time
        resolved_mode["period"] = updated_period

    return analysis_mode_name, resolved_mode


def rule_based_intent_resolution(
    question: str,
    semantic_index: SemanticIndex,
) -> IntentOutput:
    logger.info(f"Resolving filters with rule-based logic for question: {question[:100]}...")

    filters: List[FilterCondition] = []
    filters.extend(match_named_filters(question, semantic_index))
    filters.extend(extract_state_filter(question, semantic_index))
    filters.extend(extract_semantic_dimension_value_filters(question, semantic_index))
    filters = _dedupe_filters(filters)

    result = {
        "metric_hint": None,
        "group_by": extract_group_by(question, filters, semantic_index),
        "filters": filters,
        "analysis_mode": None,
        "analysis_mode_parameters": None,
        "raw_question": question,
        "validation_warnings": [],
    }
    logger.info(
        "Rule-based resolution complete: filters=%s, group_by=%s",
        len(filters),
        len(result["group_by"]),
    )
    return result


# =============================
# LangGraph node
# =============================

def identify_intent_and_filters(state: GraphState) -> Dict[str, Any]:
    """
    Resolve filters using LLM-first logic and fall back to deterministic rules.
    Also merges context filters, selects analysis mode, and validates the final output.
    """
    question = state["question"]
    context = state.get("context")
    logger.info(f"=" * 80)
    logger.info(f"identify_intent_and_filters: START - Question: {question[:100]}...")
    logger.info(f"identify_intent_and_filters: Raw context received: {context}")

    semantic_model = state.get("semantic_model", {})
    semantic_index = state.get("semantic_index") or build_semantic_index(semantic_model)
    semantic_summary = state.get("semantic_summary") or summarize_semantic_model(semantic_model)
    llm = state.get("llm")
    ehap = state.get("ehap")

    result: Optional[IntentOutput] = None

    if llm is not None:
        logger.info("identify_intent_and_filters: LLM available, attempting LLM-based filter extraction")
        try:
            llm_result = resolve_intent_with_llm(
                question=question,
                semantic_summary=semantic_summary,
                semantic_index=semantic_index,
                llm=llm,
                ehap=ehap,
            )
            if llm_result is not None:
                logger.info(f"identify_intent_and_filters: LLM extraction returned {len(llm_result['filters'])} filters")
                result = llm_result
        except Exception as exc:
            logger.warning("identify_intent_and_filters: LLM filter extraction failed; falling back to rule-based logic.", exc_info=exc)

    if result is None:
        logger.info("identify_intent_and_filters: Using rule-based filter extraction")
        result = rule_based_intent_resolution(question, semantic_index)
        logger.info(f"identify_intent_and_filters: Rule-based extraction returned {len(result['filters'])} filters")

    logger.info(f"identify_intent_and_filters: Filters BEFORE context merge: {result['filters']}")
    
    filter_context, analysis_context = split_context_filters(context)
    logger.info(f"identify_intent_and_filters: After split - filter_context={filter_context}, analysis_context={analysis_context}")

    # Merge context filters if provided
    filters_before_merge = list(result["filters"])
    result["filters"] = merge_context_filters(result["filters"], filter_context)
    logger.info(f"identify_intent_and_filters: Filters AFTER context merge: {result['filters']}")
    logger.info(f"identify_intent_and_filters: Filter merge added {len(result['filters']) - len(filters_before_merge)} new filters")

    # Resolve analysis mode using question + context
    logger.info(f"identify_intent_and_filters: Resolving analysis mode...")
    analysis_mode_name, analysis_mode_params = resolve_analysis_mode(
        question=question,
        filters=result["filters"],
        context_filters=filter_context,
        analysis_context=analysis_context,
        semantic_model=semantic_model,
        llm=llm,
        ehap=ehap,
    )
    result["analysis_mode"] = analysis_mode_name
    result["analysis_mode_parameters"] = analysis_mode_params
    logger.info(f"identify_intent_and_filters: Analysis mode resolved: {analysis_mode_name}")

    # Validate the final output
    # The AI is responsible for extracting exact values from YAML sample_values
    validation_warnings = validate_intent_output(result, semantic_model, semantic_index)
    result["validation_warnings"] = validation_warnings
    logger.info(f"identify_intent_and_filters: Validation warnings: {validation_warnings}")

    logger.info(f"identify_intent_and_filters: FINAL RESULT - analysis_mode={result['analysis_mode']}, filters={result['filters']}, group_by={result['group_by']}")
    logger.info(f"=" * 80)

    return {"result": result}


# =============================
# Graph factory
# =============================

def build_app(
    yaml_path: str,
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
):
    """
    Build a LangGraph app that resolves filters and analysis modes using:
    1. LLM-first extraction
    2. deterministic rule-based fallback
    """
    logger.info(f"Building question resolution app from YAML: {yaml_path}")
    semantic_model = load_semantic_yaml(yaml_path)
    semantic_index = build_semantic_index(semantic_model)
    semantic_summary = summarize_semantic_model(semantic_model)

    llm_client = llm
    if llm_client is None and llm_builder is not None:
        try:
            logger.debug("Building LLM client using provided builder")
            llm_client = llm_builder()
            logger.info("LLM client built successfully from builder")
        except Exception as exc:
            logger.warning("LLM builder failed", exc_info=exc)
            llm_client = None

    ehap_client = None
    if llm_client is None:
        try:
            logger.debug("Attempting to build default LLM client")
            llm_client, ehap_client = build_llm()
            logger.info("LLM client built successfully")
        except Exception as exc:
            logger.warning("LLM unavailable; using rule-based filter extraction.", exc_info=exc)
            llm_client = None
            ehap_client = None

    graph = StateGraph(GraphState)
    graph.add_node("identify_intent_and_filters", identify_intent_and_filters)
    graph.add_edge(START, "identify_intent_and_filters")
    graph.add_edge("identify_intent_and_filters", END)

    app = graph.compile()
    logger.info("Question resolution app built and compiled successfully")

    def run(question: str, context: Optional[Dict[str, Any]] = None) -> IntentOutput:
        """
        Run question resolution for a given question.

        Args:
            question: The user's question
            context: Optional dict of UI filter field->value pairs to merge with extracted filters
                    Example: {"hcc_medium_desc": "IP BH", "lob_description": "Commercial"}

        Returns:
            IntentOutput with filters, group_by, analysis mode, and validation warnings
        """
        nonlocal llm_client  # Allow modification of outer scope variable

        logger.info(f"Running question resolution for question: {question[:100]}...")
        if context:
            logger.info(f"UI context provided: {context}")

        state: GraphState = {
            "question": question,
            "context": context,
            "semantic_model": semantic_model,
            "semantic_index": semantic_index,
            "semantic_summary": semantic_summary,
            "llm": llm_client,
            "ehap": ehap_client,
        }
        logger.debug("Invoking graph ...")
        out = app.invoke(state)
        result = out["result"]
        logger.info(
            "Question resolution complete: analysis_mode=%s, warnings=%s",
            result.get("analysis_mode"),
            len(result["validation_warnings"]),
        )
        return result

    return run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run = build_app(
        "configs/ecap_semantic_view_with_samples.yaml"
    )

    question = "Find what changes in HCPCS 99292? Use total paid amount as the metric and filter for Texas."
    # Optional context - comment out to test without context (like UI without selections)
    context={
        "hcc_medium": "IP BH",
        "lob_description": "Commercial",
        # "drill_metric": "claims_expense.total_allowed",
        "period": "Rolling 3",
    }
    # To test without context (matching UI with no selections):
    # context = None
    result = run(question, context=context)

    from pprint import pprint
    pprint(result)
