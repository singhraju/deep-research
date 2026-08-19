from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from deep_research_agents.correlation_agent import (
    FilterCondition,
    MetricDefinition,
    PathNodeSummary,
    PeriodWindow,
    SemanticCatalog,
    _extract_token_usage,
    _normalize_dimension_name,
    _safe_dict,
    _safe_list,
    build_alias_map,
    build_from_clause,
    correlation_llm_tokens_for_step,
    determine_required_tables,
    empty_correlation_llm_tokens,
    execute_sql_to_df,
    folder_token_for_dimension,
    format_value,
    normalize_dataframe_columns,
    period_case_sql,
    period_filter_sql,
    render_expression,
    render_filter_clause,
    resolve_dimension_label,
    resolve_field,
    resolve_time_field,
    sign,
    slugify,
    summarize_filters,
    to_python_float,
    write_dataframe,
    write_json,
    write_text,
)

INTERACTION_SUMMARY_SYSTEM_PROMPT = """
You are a senior healthcare cost-of-care analytics writer preparing an executive-facing readout.

Summarize interaction-matrix results using only the provided facts.

Rules:
- Write in plain business language for analytics leaders and business stakeholders.
- Be concise, grounded, and non-causal.
- Say "concentrated in", "associated with", "points to", or "warrants review"; do not say "caused by" or "proves".
- Do not expose raw database field names unless no readable label is available.
- Translate obvious codes into readable wording.
- Format dollars compactly, e.g. $8.8M.
- Format percentages as whole percentages unless precision is needed.
- Mention no more than 3 operational cells and 2 clinical cells.
- Mention offsets only if they materially affect the interpretation.
- Output 2 to 4 sentences only.
"""

INTERACTION_SUMMARY_USER_PROMPT_TEMPLATE = """
Create an executive-facing summary of the interaction matrix using only the facts below.

Include:
1. The main operational concentration.
2. Whether the movement appears concentrated or broad-based.
3. The top clinical pocket, if present.
4. Any material offset, if relevant.

Facts:
{input_text}
"""

DEFAULT_INTERACTION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "trigger_rules": {
        "interaction_stage": {
            "min_abs_net_delta": 1000.0,
            "min_drill_path_depth": 2,
            "run_when_repeated_delta_ratio": 0.95,
            "run_when_low_volume": True,
        },
        "clinical_stage": {
            "require_selected_operational_cells": True,
            "min_selected_operational_positive_delta": 1000.0,
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
    "selection_rules": {
        "operational": {
            "min_positive_delta": 10000.0,
            "min_share_of_positive_delta": 0.10,
            "max_cumulative_share_of_positive_delta": 0.80,
            "max_k": 5,
            "always_include_min_share_of_net_delta": 0.50,
        },
        "clinical": {
            "min_positive_delta": 5000.0,
            "min_share_of_positive_delta": 0.05,
            "max_cumulative_share_of_positive_delta": 0.90,
            "max_k": 10,
            "include_offsets": True,
            "max_offset_k": 3,
        },
    },
    "preview_limits": {
        "max_operational_cells": 5,
        "max_clinical_cells": 10,
        "max_recommendations": 5,
    },
    "artifact_limits": {
        "max_rows_json_preview": 1000,
    },
}

PREFERRED_EXPLAINER_SUFFIXES = {
    "claim_count",
    "total_admissions",
    "avg_paid_per_admit",
    "total_allowed",
    "avg_allowed_per_admit",
    "paid_ratio",
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(_safe_dict(merged.get(key)), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_stage_dimension_list(stage_config: Mapping[str, Any]) -> Tuple[List[str], Set[str]]:
    base = [
        _normalize_dimension_name(str(item).strip())
        for item in _safe_list(stage_config.get("dimensions"))
        if str(item).strip()
    ]
    carry = [
        _normalize_dimension_name(str(item).strip())
        for item in _safe_list(stage_config.get("carry_through_dimensions"))
        if str(item).strip()
    ]
    combined: List[str] = []
    seen: Set[str] = set()
    for item in base + carry:
        if item and item not in seen:
            combined.append(item)
            seen.add(item)
    return combined, set(carry)


def _normalize_filter_values(filter_condition: Mapping[str, Any]) -> List[Any]:
    value = filter_condition.get("value")
    if isinstance(value, (list, tuple, set)):
        return [item for item in value]
    if str(filter_condition.get("operator") or "").lower().strip() == "in" and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _is_hard_filtered(filter_condition: Mapping[str, Any]) -> bool:
    operator = str(filter_condition.get("operator") or "").lower().strip()
    if operator == "named_filter":
        return False
    values = [item for item in _normalize_filter_values(filter_condition) if item is not None and str(item).strip()]
    if operator in {"=", "in"} and len(values) == 1:
        return True
    return False


def _metric_column_name(metric_name: str) -> str:
    return slugify(metric_name)


def _resolve_optional_explainer_metrics(
    catalog: SemanticCatalog,
    configured_metric_names: Sequence[str],
) -> List[str]:
    selected: List[str] = []
    seen: Set[str] = set()
    for metric_name in configured_metric_names:
        if metric_name in catalog["metrics_by_name"] and metric_name not in seen:
            selected.append(metric_name)
            seen.add(metric_name)
    for metric_name in catalog["metrics_by_name"]:
        suffix = metric_name.split(".", 1)[-1]
        if suffix in PREFERRED_EXPLAINER_SUFFIXES and metric_name not in seen:
            selected.append(metric_name)
            seen.add(metric_name)
    return selected


def build_interaction_aggregate_query(
    catalog: SemanticCatalog,
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    period_window: PeriodWindow,
    primary_table: str,
    dimension_names: Sequence[str],
    explainer_metric_names: Sequence[str] = (),
) -> str:
    explainer_metrics = [catalog["metrics_by_name"][name] for name in explainer_metric_names if name in catalog["metrics_by_name"]]
    required_table_set: List[str] = determine_required_tables(metric, filters, list(dimension_names), period_window, catalog, primary_table)
    for extra_metric in explainer_metrics:
        for table_name in determine_required_tables(extra_metric, filters, list(dimension_names), period_window, catalog, primary_table):
            if table_name not in required_table_set:
                required_table_set.append(table_name)
    alias_map = build_alias_map(required_table_set)
    from_clause = build_from_clause(catalog, required_table_set, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    period_case = period_case_sql(time_expr, period_window)
    metric_sql = render_expression(metric["expr"], alias_map, catalog)
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]

    dimension_selects: List[str] = []
    group_by_items: List[str] = []
    order_by_items: List[str] = []
    for dimension_name in dimension_names:
        field_def = resolve_field(catalog, dimension_name, preferred_tables=[primary_table]) or resolve_field(catalog, dimension_name)
        if field_def is None:
            raise KeyError(f"Unknown dimension field: {dimension_name}")
        dimension_expr = f"{alias_map[field_def['table_name']]}.{field_def['expr']}"
        alias_name = slugify(dimension_name)
        dimension_sql = f"COALESCE(CAST({dimension_expr} AS VARCHAR), '<NULL>')"
        dimension_selects.append(f"  {dimension_sql} AS {alias_name}")
        group_by_items.append(dimension_sql)
        order_by_items.append(alias_name)

    explainer_selects: List[str] = []
    for metric_name in explainer_metric_names:
        if metric_name not in catalog["metrics_by_name"]:
            continue
        explainer_selects.append(
            f"  {render_expression(catalog['metrics_by_name'][metric_name]['expr'], alias_map, catalog)} AS {_metric_column_name(metric_name)}"
        )

    where_clauses = [period_filter_sql(time_expr, period_window)]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))

    select_items = dimension_selects + [f"  {period_case} AS period_bucket", f"  {metric_sql} AS metric_value", "  COUNT(*) AS raw_row_count"] + explainer_selects
    group_by_sql = ", ".join(group_by_items + [period_case]) if group_by_items else period_case
    order_by_sql = ", ".join(order_by_items + ["period_bucket"]) if order_by_items else "period_bucket"

    return (
        "SELECT\n"
        + ",\n".join(select_items)
        + f"\n{from_clause}\nWHERE {' AND '.join(where_clauses)}\nGROUP BY {group_by_sql}\nORDER BY {order_by_sql}"
    )


def render_or_filter_groups(
    filter_groups: List[List[FilterCondition]],
    catalog: SemanticCatalog,
    alias_map: Dict[str, str],
    preferred_tables: Sequence[str],
) -> str:
    rendered_groups: List[str] = []
    for group in filter_groups:
        clauses = [render_filter_clause(filter_condition, catalog, alias_map, preferred_tables) for filter_condition in group]
        if clauses:
            rendered_groups.append(f"({' AND '.join(clauses)})")
    if not rendered_groups:
        return "1=1"
    return f"({' OR '.join(rendered_groups)})"


def build_interaction_aggregate_query_with_groups(
    catalog: SemanticCatalog,
    metric: MetricDefinition,
    filters: Sequence[FilterCondition],
    filter_groups: List[List[FilterCondition]],
    period_window: PeriodWindow,
    primary_table: str,
    dimension_names: Sequence[str],
    explainer_metric_names: Sequence[str] = (),
) -> str:
    explainer_metrics = [catalog["metrics_by_name"][name] for name in explainer_metric_names if name in catalog["metrics_by_name"]]
    flat_group_filters = [item for group in filter_groups for item in group]
    required_table_set: List[str] = determine_required_tables(metric, list(filters) + flat_group_filters, list(dimension_names), period_window, catalog, primary_table)
    for extra_metric in explainer_metrics:
        for table_name in determine_required_tables(extra_metric, list(filters) + flat_group_filters, list(dimension_names), period_window, catalog, primary_table):
            if table_name not in required_table_set:
                required_table_set.append(table_name)
    alias_map = build_alias_map(required_table_set)
    from_clause = build_from_clause(catalog, required_table_set, primary_table, alias_map)
    time_field = resolve_time_field(period_window, catalog)
    time_expr = f"{alias_map[time_field['table_name']]}.{time_field['expr']}"
    period_case = period_case_sql(time_expr, period_window)
    metric_sql = render_expression(metric["expr"], alias_map, catalog)
    preferred_tables = [primary_table] + [str(item) for item in _safe_list(metric.get("dependency_tables"))]

    dimension_selects: List[str] = []
    group_by_items: List[str] = []
    order_by_items: List[str] = []
    for dimension_name in dimension_names:
        field_def = resolve_field(catalog, dimension_name, preferred_tables=[primary_table]) or resolve_field(catalog, dimension_name)
        if field_def is None:
            raise KeyError(f"Unknown dimension field: {dimension_name}")
        dimension_expr = f"{alias_map[field_def['table_name']]}.{field_def['expr']}"
        alias_name = slugify(dimension_name)
        dimension_sql = f"COALESCE(CAST({dimension_expr} AS VARCHAR), '<NULL>')"
        dimension_selects.append(f"  {dimension_sql} AS {alias_name}")
        group_by_items.append(dimension_sql)
        order_by_items.append(alias_name)

    explainer_selects: List[str] = []
    for metric_name in explainer_metric_names:
        if metric_name not in catalog["metrics_by_name"]:
            continue
        explainer_selects.append(
            f"  {render_expression(catalog['metrics_by_name'][metric_name]['expr'], alias_map, catalog)} AS {_metric_column_name(metric_name)}"
        )

    where_clauses = [period_filter_sql(time_expr, period_window)]
    for filter_condition in filters:
        where_clauses.append(render_filter_clause(filter_condition, catalog, alias_map, preferred_tables))
    where_clauses.append(render_or_filter_groups(filter_groups, catalog, alias_map, preferred_tables))

    select_items = dimension_selects + [f"  {period_case} AS period_bucket", f"  {metric_sql} AS metric_value", "  COUNT(*) AS raw_row_count"] + explainer_selects
    group_by_sql = ", ".join(group_by_items + [period_case]) if group_by_items else period_case
    order_by_sql = ", ".join(order_by_items + ["period_bucket"]) if order_by_items else "period_bucket"

    return (
        "SELECT\n"
        + ",\n".join(select_items)
        + f"\n{from_clause}\nWHERE {' AND '.join(where_clauses)}\nGROUP BY {group_by_sql}\nORDER BY {order_by_sql}"
    )


def pivot_interaction_comparison(
    df: pd.DataFrame,
    dimension_columns: List[str],
    total_net_delta: float,
) -> pd.DataFrame:
    normalized = normalize_dataframe_columns(df)
    if normalized.empty:
        return pd.DataFrame(columns=dimension_columns + ["baseline_value", "comparison_value", "delta_value"])

    numeric_columns = [col for col in normalized.columns if col not in set(dimension_columns + ["period_bucket"])]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)

    baseline = normalized.loc[normalized["period_bucket"] == "baseline"].copy()
    comparison = normalized.loc[normalized["period_bucket"] == "comparison"].copy()
    rename_map_baseline = {col: f"{col}_baseline" for col in numeric_columns}
    rename_map_comparison = {col: f"{col}_comparison" for col in numeric_columns}
    baseline = baseline[dimension_columns + numeric_columns].rename(columns=rename_map_baseline)
    comparison = comparison[dimension_columns + numeric_columns].rename(columns=rename_map_comparison)
    pivot = baseline.merge(comparison, on=dimension_columns, how="outer")
    for column in pivot.columns:
        if column not in dimension_columns:
            pivot[column] = pd.to_numeric(pivot[column], errors="coerce").fillna(0.0)

    pivot["baseline_value"] = pivot.get("metric_value_baseline", 0.0)
    pivot["comparison_value"] = pivot.get("metric_value_comparison", 0.0)
    pivot["raw_row_count_baseline"] = pivot.get("raw_row_count_baseline", 0.0)
    pivot["raw_row_count_comparison"] = pivot.get("raw_row_count_comparison", 0.0)
    pivot["delta_value"] = pivot["comparison_value"] - pivot["baseline_value"]
    gross_positive_delta = float(pivot.loc[pivot["delta_value"] > 0, "delta_value"].sum())
    gross_negative_delta = float(pivot.loc[pivot["delta_value"] < 0, "delta_value"].abs().sum())
    net_abs = abs(total_net_delta)
    pivot["gross_positive_delta"] = gross_positive_delta
    pivot["gross_negative_delta"] = gross_negative_delta
    pivot["share_of_positive_delta"] = pivot["delta_value"].apply(lambda value: (value / gross_positive_delta) if gross_positive_delta and value > 0 else 0.0)
    pivot["share_of_negative_delta"] = pivot["delta_value"].apply(lambda value: (abs(value) / gross_negative_delta) if gross_negative_delta and value < 0 else 0.0)
    pivot["share_of_net_delta"] = pivot["delta_value"].apply(lambda value: (value / net_abs) if net_abs else 0.0)

    for column in numeric_columns:
        if column in {"metric_value", "raw_row_count"}:
            continue
        pivot[f"{column}_delta"] = pivot.get(f"{column}_comparison", 0.0) - pivot.get(f"{column}_baseline", 0.0)

    pivot = pivot.sort_values(by=["delta_value", "comparison_value"], ascending=False).reset_index(drop=True)
    pivot["artifact_row_ref"] = pivot.index + 1
    return pivot


def _resolve_interaction_config(intent: Mapping[str, Any]) -> Dict[str, Any]:
    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    raw = _safe_dict(mode_parameters.get("interaction_matrix"))
    return _deep_merge(DEFAULT_INTERACTION_CONFIG, raw)


def _available_dimension_names(catalog: SemanticCatalog) -> Set[str]:
    names: Set[str] = set()
    for table_fields in catalog["table_fields"].values():
        for field_name, field_def in table_fields.items():
            if field_def.get("kind") == "dimension":
                names.add(field_name)
    return names


def _hard_filtered_fields(filters: Sequence[FilterCondition]) -> Set[str]:
    fields: Set[str] = set()
    for filter_condition in filters:
        if not isinstance(filter_condition, dict):
            continue
        field_name = str(filter_condition.get("field") or "")
        if not field_name:
            continue
        if _is_hard_filtered(filter_condition):
            fields.add(_normalize_dimension_name(field_name))
    return fields


def _resolve_stage_dimensions(
    stage_config: Mapping[str, Any],
    catalog: SemanticCatalog,
    filters: Sequence[FilterCondition],
) -> Tuple[List[str], List[str]]:
    configured, carry_set = _normalize_stage_dimension_list(stage_config)
    available = _available_dimension_names(catalog)
    hard_filtered = _hard_filtered_fields(filters)

    eligible: List[str] = []
    excluded: List[str] = []
    for item in configured:
        if item not in available:
            continue
        if item in hard_filtered and item not in carry_set:
            excluded.append(item)
            continue
        eligible.append(item)
    return eligible, excluded


def _build_dimension_values(
    row: Mapping[str, Any],
    dimension_column_map: Sequence[Tuple[str, str]],
) -> Dict[str, str]:
    return {
        original_name: str(row.get(alias_name) or "<NULL>").strip() or "<NULL>"
        for original_name, alias_name in dimension_column_map
    }


def _cell_payload(
    row: Mapping[str, Any],
    *,
    stage: str,
    cell_id: str,
    dimension_column_map: Sequence[Tuple[str, str]],
    explainer_metric_names: Sequence[str],
) -> Dict[str, Any]:
    payload = {
        "cell_id": cell_id,
        "stage": stage,
        "dimension_values": _build_dimension_values(row, dimension_column_map),
        "baseline_value": to_python_float(row.get("baseline_value")),
        "comparison_value": to_python_float(row.get("comparison_value")),
        "delta_value": to_python_float(row.get("delta_value")),
        "share_of_positive_delta": to_python_float(row.get("share_of_positive_delta")),
        "share_of_net_delta": to_python_float(row.get("share_of_net_delta")),
        "raw_row_count_baseline": to_python_float(row.get("raw_row_count_baseline")),
        "raw_row_count_comparison": to_python_float(row.get("raw_row_count_comparison")),
        "artifact_row_ref": int(to_python_float(row.get("artifact_row_ref"))),
    }
    explainers: Dict[str, Dict[str, float]] = {}
    for metric_name in explainer_metric_names:
        column = _metric_column_name(metric_name)
        if f"{column}_baseline" not in row and f"{column}_comparison" not in row:
            continue
        explainers[metric_name] = {
            "baseline": to_python_float(row.get(f"{column}_baseline")),
            "comparison": to_python_float(row.get(f"{column}_comparison")),
            "delta": to_python_float(row.get(f"{column}_delta")),
        }
    if explainers:
        payload["explainer_metrics"] = explainers
    return payload


def _select_cells(
    pivot_df: pd.DataFrame,
    *,
    stage: str,
    rules: Mapping[str, Any],
    explainer_metric_names: Sequence[str],
    dimension_column_map: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if pivot_df.empty:
        return [], []
    min_positive_delta = float(rules.get("min_positive_delta", 0.0) or 0.0)
    min_share = float(rules.get("min_share_of_positive_delta", 0.0) or 0.0)
    max_cumulative = float(rules.get("max_cumulative_share_of_positive_delta", 1.0) or 1.0)
    max_k = int(rules.get("max_k", 5) or 5)
    always_include_share = float(rules.get("always_include_min_share_of_net_delta", 0.0) or 0.0)

    positives = pivot_df.loc[pivot_df["delta_value"] > 0].copy().sort_values(by=["delta_value", "share_of_positive_delta"], ascending=False)
    selected_rows: List[Tuple[int, Mapping[str, Any]]] = []
    selected_indices: Set[int] = set()
    for index, row in positives.iterrows():
        if to_python_float(row.get("share_of_net_delta")) >= always_include_share > 0:
            selected_rows.append((index, row))
            selected_indices.add(index)
    cumulative = float(sum(to_python_float(row.get("share_of_positive_delta")) for _, row in selected_rows))
    for index, row in positives.iterrows():
        if index in selected_indices:
            continue
        if to_python_float(row.get("delta_value")) < min_positive_delta:
            continue
        if to_python_float(row.get("share_of_positive_delta")) < min_share:
            continue
        if len(selected_rows) >= max_k:
            break
        if cumulative >= max_cumulative and selected_rows:
            break
        selected_rows.append((index, row))
        selected_indices.add(index)
        cumulative += to_python_float(row.get("share_of_positive_delta"))

    selected_cells = [
        _cell_payload(row, stage=stage, cell_id=f"{'op' if stage == 'operational' else 'cl'}_{position:03d}", dimension_column_map=dimension_column_map, explainer_metric_names=explainer_metric_names)
        for position, (_, row) in enumerate(selected_rows, start=1)
    ]

    offset_cells: List[Dict[str, Any]] = []
    if bool(rules.get("include_offsets", False)):
        max_offset_k = int(rules.get("max_offset_k", 0) or 0)
        negatives = pivot_df.loc[pivot_df["delta_value"] < 0].copy().sort_values(by=["delta_value"], ascending=True).head(max_offset_k)
        offset_cells = [
            _cell_payload(row, stage=stage, cell_id=f"{'op' if stage == 'operational' else 'cl'}_off_{position:03d}", dimension_column_map=dimension_column_map, explainer_metric_names=explainer_metric_names)
            for position, (_, row) in enumerate(negatives.iterrows(), start=1)
        ]
    return selected_cells, offset_cells


def _group_signature(group: List[FilterCondition], exclude_field: str) -> Tuple[Tuple[str, str], ...]:
    items: List[Tuple[str, str]] = []
    for filter_condition in group:
        field_name = str(filter_condition.get("field") or "")
        if field_name == exclude_field:
            continue
        items.append((field_name, json.dumps(filter_condition.get("value"), sort_keys=True, ensure_ascii=False, default=str)))
    return tuple(sorted(items))


def _safe_merge_filter_groups(filter_groups: List[List[FilterCondition]]) -> List[List[FilterCondition]]:
    groups = [copy.deepcopy(group) for group in filter_groups]
    changed = True
    while changed:
        changed = False
        for dimension_name in list({str(item.get("field") or "") for group in groups for item in group if item.get("field")}):
            buckets: Dict[Tuple[Tuple[str, str], ...], List[List[FilterCondition]]] = {}
            for group in groups:
                buckets.setdefault(_group_signature(group, dimension_name), []).append(group)
            for _, bucket in buckets.items():
                if len(bucket) < 2:
                    continue
                merged_values: List[Any] = []
                template: Optional[FilterCondition] = None
                survivor: List[FilterCondition] = []
                base_signature: Optional[Tuple[Tuple[str, str], ...]] = None
                for group in bucket:
                    dimension_filters = [item for item in group if str(item.get("field") or "") == dimension_name]
                    if len(dimension_filters) != 1:
                        survivor = []
                        template = None
                        break
                    if template is None:
                        template = copy.deepcopy(dimension_filters[0])
                        base_signature = _group_signature(group, dimension_name)
                        survivor = [copy.deepcopy(item) for item in group if str(item.get("field") or "") != dimension_name]
                    elif _group_signature(group, dimension_name) != base_signature:
                        survivor = []
                        template = None
                        break
                    merged_values.extend(_normalize_filter_values(dimension_filters[0]))
                if template is None:
                    continue
                unique_values: List[Any] = []
                seen: Set[str] = set()
                for value in merged_values:
                    key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
                    if key not in seen:
                        seen.add(key)
                        unique_values.append(value)
                merged_group = list(survivor)
                merged_group.append({**template, "operator": "in" if len(unique_values) > 1 else "=", "value": unique_values if len(unique_values) > 1 else unique_values[0]})
                groups = [group for group in groups if group not in bucket] + [merged_group]
                changed = True
    return groups


def _selected_cells_to_filter_groups(selected_cells: Sequence[Mapping[str, Any]]) -> List[List[FilterCondition]]:
    groups: List[List[FilterCondition]] = []
    for cell in selected_cells:
        dimension_values = _safe_dict(cell.get("dimension_values"))
        group: List[FilterCondition] = []
        for field_name, value in dimension_values.items():
            group.append(
                {
                    "field": field_name,
                    "operator": "=",
                    "value": value,
                    "source": "dimension_match",
                }
            )
        if group:
            groups.append(group)
    return _safe_merge_filter_groups(groups)


def _deterministic_interaction_summary(
    metric_name: str,
    operational_cells: Sequence[Mapping[str, Any]],
    clinical_cells: Sequence[Mapping[str, Any]],
) -> str:
    if not operational_cells:
        return ""
    top_operational = operational_cells[0]
    op_values = ", ".join(f"{key}={value}" for key, value in _safe_dict(top_operational.get("dimension_values")).items())
    text = (
        f"The strongest operational interaction cell was {op_values} with {format_value(metric_name, to_python_float(top_operational.get('delta_value')))} of change"
        f" ({to_python_float(top_operational.get('share_of_positive_delta')) * 100:.0f}% of gross positive delta)."
    )
    if clinical_cells:
        top_clinical = clinical_cells[0]
        cl_values = ", ".join(f"{key}={value}" for key, value in _safe_dict(top_clinical.get("dimension_values")).items())
        text += f" Within selected operational cells, the strongest clinical interaction was {cl_values} with {format_value(metric_name, to_python_float(top_clinical.get('delta_value')))} of change."
    return text


def _llm_interaction_summary(llm: Any, payload: Mapping[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": INTERACTION_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": INTERACTION_SUMMARY_USER_PROMPT_TEMPLATE.format(input_text=json.dumps(payload, indent=2, ensure_ascii=False))},
            ]
        )
        input_tokens, output_tokens = _extract_token_usage(response)
        content = str(getattr(response, "content", response)).strip()
        return (
            " ".join(content.split()) if content else None,
            correlation_llm_tokens_for_step("interaction_summary", input_tokens, output_tokens),
        )
    except Exception as exc:
        logger.warning("Interaction summary LLM call failed.", exc_info=exc)
        return None, empty_correlation_llm_tokens()


def _build_trigger_summary(
    config: Mapping[str, Any],
    delta_value: float,
    drill_path: Sequence[PathNodeSummary],
    root_summary_df: pd.DataFrame,
) -> Dict[str, Any]:
    interaction_rules = _safe_dict(_safe_dict(config.get("trigger_rules")).get("interaction_stage"))
    repeated_ratio = 0.0
    if delta_value:
        for node in drill_path:
            top_delta = sum(to_python_float(item.get("delta_value")) for item in _safe_list(node.get("top_segments")))
            repeated_ratio = max(repeated_ratio, abs(top_delta) / abs(delta_value))
    root_counts = {
        "baseline": int(root_summary_df.loc[root_summary_df["period_bucket"] == "baseline", "raw_row_count"].sum()) if not root_summary_df.empty else 0,
        "comparison": int(root_summary_df.loc[root_summary_df["period_bucket"] == "comparison", "raw_row_count"].sum()) if not root_summary_df.empty else 0,
    }
    low_volume = min(root_counts.values()) <= 25 if root_counts else False
    interaction_stage_passed = (
        abs(delta_value) >= float(interaction_rules.get("min_abs_net_delta", 0.0) or 0.0)
        and len(drill_path) >= int(interaction_rules.get("min_drill_path_depth", 0) or 0)
        and (
            repeated_ratio >= float(interaction_rules.get("run_when_repeated_delta_ratio", 0.0) or 0.0)
            or (bool(interaction_rules.get("run_when_low_volume", False)) and low_volume)
        )
    )
    return {
        "interaction_stage": {
            "passed": interaction_stage_passed,
            "min_abs_net_delta": float(interaction_rules.get("min_abs_net_delta", 0.0) or 0.0),
            "min_drill_path_depth": int(interaction_rules.get("min_drill_path_depth", 0) or 0),
            "observed_drill_path_depth": len(drill_path),
            "observed_repeated_delta_ratio": repeated_ratio,
            "run_when_repeated_delta_ratio": float(interaction_rules.get("run_when_repeated_delta_ratio", 0.0) or 0.0),
            "run_when_low_volume": bool(interaction_rules.get("run_when_low_volume", False)),
            "observed_low_volume": low_volume,
            "root_row_counts": root_counts,
        }
    }


def execute_interaction_matrix(
    *,
    intent: Mapping[str, Any],
    catalog: SemanticCatalog,
    metric_name: str,
    metric: MetricDefinition,
    primary_table: str,
    period_window: PeriodWindow,
    filters: Sequence[FilterCondition],
    drill_path: Sequence[PathNodeSummary],
    root_summary_df: pd.DataFrame,
    delta_value: float,
    queries_dir: Path,
    aggregates_dir: Path,
    summary_dir: Path,
    run_dir: Path,
    snowflake_helper: Any,
    llm: Any,
    manifest: Dict[str, Any],
    warnings: List[str],
    save_parquet: bool = False,
    disable_summary_creation: bool = True,
    generate_recommendations: bool = True,
) -> Dict[str, Any]:
    config = _resolve_interaction_config(intent)
    if not bool(config.get("enabled", False)):
        return {"enabled": False, "summary": {"status": "disabled"}, "llm_tokens": empty_correlation_llm_tokens()}

    trigger_summary = _build_trigger_summary(config, delta_value, drill_path, root_summary_df)
    if not _safe_dict(trigger_summary.get("interaction_stage")).get("passed"):
        return {"enabled": True, "summary": {"status": "not_triggered", **trigger_summary}, "llm_tokens": empty_correlation_llm_tokens()}

    interaction_queries_dir = Path(queries_dir) / "interaction_matrix"
    interaction_aggregates_dir = Path(aggregates_dir) / "interaction_matrix"
    interaction_summary_dir = Path(summary_dir)
    interaction_queries_dir.mkdir(parents=True, exist_ok=True)
    if save_parquet:
        interaction_aggregates_dir.mkdir(parents=True, exist_ok=True)

    categories = _safe_dict(config.get("categories"))
    operational_dimensions, operational_excluded = _resolve_stage_dimensions(_safe_dict(categories.get("operational")), catalog, filters)
    clinical_dimensions, clinical_excluded = _resolve_stage_dimensions(_safe_dict(categories.get("clinical")), catalog, filters)
    if not operational_dimensions:
        warnings.append("Interaction matrix skipped: no eligible operational dimensions resolved.")
        return {"enabled": True, "summary": {"status": "no_operational_dimensions", **trigger_summary}, "llm_tokens": empty_correlation_llm_tokens()}

    mode_parameters = _safe_dict(intent.get("analysis_mode_parameters"))
    configured_explainers = [str(item).strip() for item in _safe_list(mode_parameters.get("explainer_metrics")) if str(item).strip()]
    interaction_explainers = _resolve_optional_explainer_metrics(catalog, configured_explainers)

    operational_sql = build_interaction_aggregate_query(
        catalog=catalog,
        metric=metric,
        filters=filters,
        period_window=period_window,
        primary_table=primary_table,
        dimension_names=operational_dimensions,
        explainer_metric_names=interaction_explainers,
    )
    operational_sql_path = interaction_queries_dir / "operational.sql"
    write_text(operational_sql_path, operational_sql)
    operational_df = execute_sql_to_df(snowflake_helper, operational_sql)
    operational_pivot = pivot_interaction_comparison(operational_df, [slugify(item) for item in operational_dimensions], delta_value)
    operational_full_path = ""
    operational_delta_path = ""
    if save_parquet:
        operational_full_path = write_dataframe(interaction_aggregates_dir / "operational_full_matrix.parquet", operational_pivot)
        operational_delta_path = write_dataframe(interaction_aggregates_dir / "operational_delta.parquet", operational_pivot)
    operational_dimension_map = [(item, slugify(item)) for item in operational_dimensions]

    operational_rules = _safe_dict(_safe_dict(config.get("selection_rules")).get("operational"))
    selected_operational_cells, _ = _select_cells(
        operational_pivot,
        stage="operational",
        rules=operational_rules,
        explainer_metric_names=interaction_explainers,
        dimension_column_map=operational_dimension_map,
    )
    selected_operational_positive_delta = sum(max(0.0, to_python_float(item.get("delta_value"))) for item in selected_operational_cells)

    filter_groups = _selected_cells_to_filter_groups(selected_operational_cells)
    clinical_trigger_rules = _safe_dict(_safe_dict(config.get("trigger_rules")).get("clinical_stage"))
    clinical_stage_passed = (
        (not bool(clinical_trigger_rules.get("require_selected_operational_cells", True)) or bool(filter_groups))
        and selected_operational_positive_delta >= float(clinical_trigger_rules.get("min_selected_operational_positive_delta", 0.0) or 0.0)
    )

    clinical_sql_path = interaction_queries_dir / "clinical.sql"
    clinical_full_path = ""
    clinical_delta_path = ""
    clinical_pivot = pd.DataFrame()
    selected_clinical_cells: List[Dict[str, Any]] = []
    clinical_offset_cells: List[Dict[str, Any]] = []
    if clinical_dimensions and filter_groups and clinical_stage_passed:
        clinical_sql = build_interaction_aggregate_query_with_groups(
            catalog=catalog,
            metric=metric,
            filters=filters,
            filter_groups=filter_groups,
            period_window=period_window,
            primary_table=primary_table,
            dimension_names=clinical_dimensions,
            explainer_metric_names=interaction_explainers,
        )
        write_text(clinical_sql_path, clinical_sql)
        clinical_df = execute_sql_to_df(snowflake_helper, clinical_sql)
        clinical_pivot = pivot_interaction_comparison(clinical_df, [slugify(item) for item in clinical_dimensions], delta_value)
        if save_parquet:
            clinical_full_path = write_dataframe(interaction_aggregates_dir / "clinical_full_matrix.parquet", clinical_pivot)
            clinical_delta_path = write_dataframe(interaction_aggregates_dir / "clinical_delta.parquet", clinical_pivot)
        clinical_dimension_map = [(item, slugify(item)) for item in clinical_dimensions]
        clinical_rules = _safe_dict(_safe_dict(config.get("selection_rules")).get("clinical"))
        selected_clinical_cells, clinical_offset_cells = _select_cells(
            clinical_pivot,
            stage="clinical",
            rules=clinical_rules,
            explainer_metric_names=interaction_explainers,
            dimension_column_map=clinical_dimension_map,
        )

    manifest["files"].append(str(operational_sql_path.relative_to(run_dir)))
    if operational_full_path:
        manifest["files"].append(str(Path(operational_full_path).relative_to(run_dir)))
    if operational_delta_path:
        manifest["files"].append(str(Path(operational_delta_path).relative_to(run_dir)))
    if clinical_sql_path.exists():
        manifest["files"].append(str(clinical_sql_path.relative_to(run_dir)))
    if clinical_full_path:
        manifest["files"].append(str(Path(clinical_full_path).relative_to(run_dir)))
    if clinical_delta_path:
        manifest["files"].append(str(Path(clinical_delta_path).relative_to(run_dir)))

    preview_limits = _safe_dict(config.get("preview_limits"))
    summary_payload = {
        "metric_name": metric_name,
        "operational_cells": selected_operational_cells[: int(preview_limits.get("max_operational_cells", 5) or 5)],
        "clinical_cells": selected_clinical_cells[: int(preview_limits.get("max_clinical_cells", 10) or 10)],
        "clinical_offset_cells": clinical_offset_cells[: int(preview_limits.get("max_clinical_cells", 10) or 10)],
    }
    interaction_summary_text = ""
    interaction_summary_source = "disabled" if disable_summary_creation else "empty"
    interaction_llm_tokens = empty_correlation_llm_tokens()
    if not disable_summary_creation:
        interaction_summary_text = _deterministic_interaction_summary(metric_name, selected_operational_cells, selected_clinical_cells)
        interaction_summary_source = "deterministic" if interaction_summary_text else "empty"
        if llm is not None and summary_payload["operational_cells"]:
            llm_summary, interaction_llm_tokens = _llm_interaction_summary(llm, summary_payload)
            if llm_summary:
                interaction_summary_text = llm_summary
                interaction_summary_source = "llm"

    interaction_summary_artifact = {
        "text": interaction_summary_text,
        "source": interaction_summary_source,
    }
    interaction_recommendation_artifact = {
        "recommended_action": [],
        "source": "empty" if generate_recommendations else "disabled",
    }
    if not disable_summary_creation:
        write_json(interaction_summary_dir / "interaction_summary.json", interaction_summary_artifact)
    if generate_recommendations:
        write_json(interaction_summary_dir / "interaction_recommendations.json", interaction_recommendation_artifact)
    if not disable_summary_creation:
        manifest["files"].append(str((interaction_summary_dir / "interaction_summary.json").relative_to(run_dir)))
    if generate_recommendations:
        manifest["files"].append(str((interaction_summary_dir / "interaction_recommendations.json").relative_to(run_dir)))

    return {
        "enabled": True,
        "summary": {
            "status": "success",
            **trigger_summary,
            "clinical_stage": {
                "passed": clinical_stage_passed,
                "require_selected_operational_cells": bool(clinical_trigger_rules.get("require_selected_operational_cells", True)),
                "min_selected_operational_positive_delta": float(clinical_trigger_rules.get("min_selected_operational_positive_delta", 0.0) or 0.0),
                "observed_selected_operational_positive_delta": selected_operational_positive_delta,
            },
            "excluded_dimensions": {
                "operational": operational_excluded,
                "clinical": clinical_excluded,
            },
        },
        "operational": {
            "dimensions": operational_dimensions,
            "carry_through_dimensions": [
                str(item).strip()
                for item in _safe_list(_safe_dict(categories.get("operational")).get("carry_through_dimensions"))
                if str(item).strip()
            ],
            "selected_cell_filter_groups": [summarize_filters(group) for group in filter_groups],
            "selected_cells": selected_operational_cells,
            "top_cells_preview": selected_operational_cells[: int(preview_limits.get("max_operational_cells", 5) or 5)],
            "artifact_paths": {
                "sql": str(operational_sql_path.relative_to(run_dir)),
                "delta": str(Path(operational_delta_path).relative_to(run_dir)) if operational_delta_path else "",
                "full_matrix": str(Path(operational_full_path).relative_to(run_dir)) if operational_full_path else "",
            },
        },
        "clinical": {
            "dimensions": clinical_dimensions,
            "carry_through_dimensions": [
                str(item).strip()
                for item in _safe_list(_safe_dict(categories.get("clinical")).get("carry_through_dimensions"))
                if str(item).strip()
            ],
            "selected_cells": selected_clinical_cells,
            "top_cells_preview": selected_clinical_cells[: int(preview_limits.get("max_clinical_cells", 10) or 10)],
            "offset_cells_preview": clinical_offset_cells[: int(preview_limits.get("max_clinical_cells", 10) or 10)],
            "artifact_paths": {
                "sql": str(clinical_sql_path.relative_to(run_dir)) if clinical_sql_path.exists() else "",
                "delta": str(Path(clinical_delta_path).relative_to(run_dir)) if clinical_delta_path else "",
                "full_matrix": str(Path(clinical_full_path).relative_to(run_dir)) if clinical_full_path else "",
            },
        },
        "interaction_summary": interaction_summary_artifact,
        "recommended_action": interaction_recommendation_artifact,
        "llm_tokens": interaction_llm_tokens,
    }
