from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field

from deep_research_core.base_agent import AgentBase, AgentExecutionError

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def build_pattern_cards_and_groups(
    correlation_results: str | Path | Mapping[str, Any],
    semantic_config_path: str | Path,
    analysis_mode_name: str = "cost_change_investigation_over_time_window",
    *,
    min_abs_delta: Optional[float] = None,
    max_top_cards_per_group: int = 5,
    semantic_roles: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    correlation_json = _load_json_or_dict(correlation_results)
    semantic = _load_yaml(semantic_config_path)
    semantic_cfg = _build_semantic_card_config(
        semantic, analysis_mode_name, semantic_roles=semantic_roles
    )

    if min_abs_delta is not None:
        semantic_cfg["thresholds"]["min_abs_delta"] = float(min_abs_delta)

    cards: list[dict[str, Any]] = []

    for run_ref in _iter_correlation_runs(correlation_json):
        run_cards = _build_cards_for_run(run_ref, semantic_cfg)
        cards.extend(run_cards)

    cards = [card for card in cards if _card_passes_threshold(card, semantic_cfg)]
    groups = _group_cards(cards, semantic_cfg, max_top_cards_per_group=max_top_cards_per_group)

    return {
        "cards": cards,
        "groups": groups,
        "stats": {
            "num_cards": len(cards),
            "num_groups": len(groups),
            "cards_by_stage": dict(Counter(card["stage"] for card in cards)),
            "cards_by_type": dict(Counter(card["card_type"] for card in cards)),
            "cards_by_driver": dict(Counter(card["driver_type"] for card in cards)),
            "groups_by_stage": dict(Counter(group["stage"] for group in groups)),
            "groups_by_driver_group": dict(Counter(group["driver_group"] for group in groups)),
        },
        "semantic_summary": {
            "analysis_mode": analysis_mode_name,
            "stage_dimensions": semantic_cfg["stage_dimensions"],
            "stage_carry_through_dimensions": semantic_cfg["stage_carry_through_dimensions"],
            "drill_dimensions": semantic_cfg["drill_dimensions"],
            "explainer_metrics": semantic_cfg["explainer_metrics"],
            "root_metric_candidates": semantic_cfg["root_metric_candidates"],
            "thresholds": semantic_cfg["thresholds"],
        },
    }


def _build_semantic_card_config(
    semantic: Mapping[str, Any],
    analysis_mode_name: str,
    *,
    semantic_roles: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    mode = _find_analysis_mode(semantic, analysis_mode_name)
    dim_catalog = _build_dimension_catalog(semantic)
    metric_catalog = _build_metric_catalog(semantic)

    interaction = mode.get("interaction_matrix", {}) or {}
    categories = interaction.get("categories", {}) or {}
    selection_rules = interaction.get("selection_rules", {}) or {}
    trigger_rules = interaction.get("trigger_rules", {}) or {}
    stop_rules = mode.get("stop_rules", {}) or {}

    stage_dimensions = {
        stage: list(spec.get("dimensions", []) or [])
        for stage, spec in categories.items()
        if isinstance(spec, Mapping)
    }
    stage_carry = {
        stage: list(spec.get("carry_through_dimensions", []) or [])
        for stage, spec in categories.items()
        if isinstance(spec, Mapping)
    }

    thresholds = {
        "min_abs_delta": float(stop_rules.get("min_abs_delta", 1000.0)),
        "min_row_count": float(stop_rules.get("min_row_count", 1.0)),
        "min_contribution_pct": float(stop_rules.get("min_contribution_pct", 0.05)),
        "low_baseline_value": 1000.0,
        "low_baseline_rows": 3.0,
        "small_n_rows": 5.0,
        "small_n_admissions": 3.0,
        "unit_cost_material_abs": 1000.0,
        "unit_cost_material_pct": 0.10,
        "high_repeated_delta_ratio": float(
            ((trigger_rules.get("interaction_stage", {}) or {}).get("run_when_repeated_delta_ratio", 0.95))
        ),
        "operational_min_positive_delta": float(
            ((selection_rules.get("operational", {}) or {}).get("min_positive_delta", 1000.0))
        ),
        "clinical_min_positive_delta": float(
            ((selection_rules.get("clinical", {}) or {}).get("min_positive_delta", 500.0))
        ),
        "operational_min_share_of_positive_delta": float(
            ((selection_rules.get("operational", {}) or {}).get("min_share_of_positive_delta", 0.05))
        ),
        "clinical_min_share_of_positive_delta": float(
            ((selection_rules.get("clinical", {}) or {}).get("min_share_of_positive_delta", 0.05))
        ),
    }

    return {
        "view_name": semantic.get("name"),
        "mode_name": mode.get("name"),
        "dimension_catalog": dim_catalog,
        "metric_catalog": metric_catalog,
        "stage_dimensions": stage_dimensions,
        "stage_carry_through_dimensions": stage_carry,
        "stages": tuple(stage_dimensions.keys()) or ("operational", "clinical"),
        "drill_dimensions": list(mode.get("drill_dimensions", []) or []),
        "drill_metric": list(mode.get("drill_metric", []) or []),
        "explainer_metrics": list(mode.get("explainer_metrics", []) or []),
        "root_metric_candidates": _root_metric_candidates(mode, metric_catalog),
        "thresholds": thresholds,
        "dimension_roles": _resolve_dimension_roles(dim_catalog, semantic_roles),
        "metric_aliases": _infer_metric_aliases(metric_catalog, mode),
    }


def _find_analysis_mode(semantic: Mapping[str, Any], analysis_mode_name: str) -> Mapping[str, Any]:
    for mode in semantic.get("analysis_modes", []) or []:
        if mode.get("name") == analysis_mode_name:
            return mode
    available = [mode.get("name") for mode in semantic.get("analysis_modes", []) or []]
    raise ValueError(f"Analysis mode not found: {analysis_mode_name}. Available: {available}")


def _build_dimension_catalog(semantic: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    for table in semantic.get("tables", []) or []:
        table_name = table.get("name")

        for section in ("dimensions", "time_dimensions"):
            for dim in table.get(section, []) or []:
                name = dim.get("name")
                if not name:
                    continue

                catalog[name] = {
                    "name": name,
                    "table": table_name,
                    "expr": dim.get("expr"),
                    "description": dim.get("description"),
                    "data_type": dim.get("data_type"),
                    "synonyms": list(dim.get("synonyms", []) or []),
                    "sample_values": list(dim.get("sample_values", []) or []),
                    "is_enum": bool(dim.get("is_enum", False)),
                    "section": section,
                }

    return catalog


def _build_metric_catalog(semantic: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    for table in semantic.get("tables", []) or []:
        for metric in table.get("metrics", []) or []:
            name = metric.get("name")
            if not name:
                continue

            catalog[name] = {
                "name": name,
                "table": table.get("name"),
                "scope": "table",
                "expr": metric.get("expr"),
                "description": metric.get("description"),
                "synonyms": list(metric.get("synonyms", []) or []),
            }

    for metric in semantic.get("metrics", []) or []:
        name = metric.get("name")
        if not name:
            continue

        catalog[name] = {
            "name": name,
            "table": None,
            "scope": "view",
            "expr": metric.get("expr"),
            "description": metric.get("description"),
            "synonyms": list(metric.get("synonyms", []) or []),
        }

    return catalog


def _root_metric_candidates(mode: Mapping[str, Any], metric_catalog: Mapping[str, Any]) -> list[str]:
    candidates = []
    candidates.extend(mode.get("drill_metric", []) or [])

    for metric_name in metric_catalog:
        normed = _norm(metric_name)
        if any(token in normed for token in ("total_paid", "paid_amount", "expense", "cost")):
            candidates.append(metric_name)

    return list(dict.fromkeys(candidates))


# Map the LLM classifier's fine-grained taxonomy onto the short role names the
# rest of pattern_agent (and downstream consumers) already reason about.
# Multiple classifier roles collapse into one legacy role — e.g. procedure_code
# and procedure_name both flow into "procedure". Legacy roles missing from the
# taxonomy (admission_source) stay keyword-inferred.
_CLASSIFIER_ROLE_TO_LEGACY: dict[str, str] = {
    "state": "state",
    "zip": "zip",
    "county": "geography",
    "city": "geography",
    "region": "geography",
    "geography": "geography",
    "provider": "provider",
    "provider_name": "provider",
    "provider_type": "provider",
    "provider_specialty": "provider",
    "provider_id": "provider",
    "product": "product",
    "plan": "product",
    "contract": "product",
    "facility": "facility",
    "place_of_service": "facility",
    "authorization": "auth",
    "line_of_business": "line_of_business",
    "clinical_category": "clinical_category",
    "service_category": "clinical_category",
    "drg_code": "drg",
    "drg_name": "drg",
    "diagnosis_code": "diagnosis",
    "diagnosis_name": "diagnosis",
    "procedure_code": "procedure",
    "procedure_name": "procedure",
    "revenue_code": "procedure",
    "network": "network",
    "time": "time",
    "date": "time",
    "month": "time",
    "quarter": "time",
    "year": "time",
}


def _resolve_dimension_roles(
    dim_catalog: Mapping[str, Mapping[str, Any]],
    semantic_roles: Optional[Mapping[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Return {legacy_role -> (dim_names,)} for the current YAML.

    When ``semantic_roles`` is supplied (dim -> classifier role), invert it
    into the {role -> tuple[dim,]} shape the rest of pattern_agent uses, then
    merge with the legacy keyword output so any dimension the classifier
    marked ``other`` still gets a chance at keyword matching. Classifier
    assignments win on collisions.

    When ``semantic_roles`` is None (backward-compat / legacy path), fall
    back to the original keyword-based inference. Remove that branch once
    every checked-in YAML has a companion JSON.
    """

    keyword_roles = _infer_dimension_roles(dim_catalog)
    if not semantic_roles:
        return keyword_roles

    accumulator: dict[str, list[str]] = {role: list(dims) for role, dims in keyword_roles.items()}
    classifier_assigned: set[str] = set()

    for dim_name, classifier_role in semantic_roles.items():
        if not classifier_role or classifier_role == "other":
            continue
        legacy_role = _CLASSIFIER_ROLE_TO_LEGACY.get(str(classifier_role))
        if legacy_role is None:
            continue
        bucket = accumulator.setdefault(legacy_role, [])
        if dim_name not in bucket:
            bucket.append(str(dim_name))
        classifier_assigned.add(str(dim_name))

    # Strip legacy keyword hits for dims the classifier claimed authoritatively
    # under a different role. Prevents provider_state_code from staying in the
    # "geography" bucket after the classifier put it in "state".
    for role in list(accumulator.keys()):
        accumulator[role] = [
            dim
            for dim in accumulator[role]
            if dim not in classifier_assigned
            or _CLASSIFIER_ROLE_TO_LEGACY.get(str(semantic_roles.get(dim, ""))) == role
        ]

    return {role: tuple(dict.fromkeys(dims)) for role, dims in accumulator.items() if dims}


def _infer_dimension_roles(dim_catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    # Role priority order: `state` and `zip` are listed BEFORE `geography` and
    # `provider` so a dimension name like `provider_state_code` or
    # `src_provider_zip_code` resolves to the more specific role. The
    # downstream `_role_for_dimension` iterates this dict and returns the
    # first match, so ordering matters.
    role_keywords = {
        "state": ("state code", "provider state", "rendering state", "service state", "member state", "brand state"),
        "zip": ("zip", "postal"),
        "geography": ("market", "region", "county", "service area", "geo"),
        "provider": ("provider", "hospital", "rendering", "facility name", "vendor", "supplier"),
        "product": ("product", "plan", "benefit"),
        "facility": ("facility type", "facility", "site of care", "place of service", "pos"),
        "auth": ("auth", "authorization", "pa required", "prior authorization", "pa_required"),
        "admission_source": ("er admit", "emergency", "admission source"),
        "line_of_business": ("lob", "line of business", "business line"),
        "clinical_category": ("hcc", "health care category", "service category", "service line", "clinical category"),
        "drg": ("drg", "diagnosis related group"),
        "diagnosis": ("diagnosis", "diag", "dx", "icd"),
        "procedure": ("procedure", "proc", "cpt"),
        "network": ("network", "in network", "out of network"),
        "time": ("month", "date", "time", "period"),
    }

    roles: dict[str, list[str]] = {role: [] for role in role_keywords}

    for dim_name, meta in dim_catalog.items():
        haystack = _search_text(
            dim_name,
            meta.get("expr"),
            meta.get("description"),
            " ".join(meta.get("synonyms", []) or []),
        )

        for role, keywords in role_keywords.items():
            if any(_norm(keyword) in haystack for keyword in keywords):
                roles[role].append(dim_name)

    return {role: tuple(values) for role, values in roles.items() if values}


def _infer_metric_aliases(
    metric_catalog: Mapping[str, Mapping[str, Any]],
    mode: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    canonical_keywords = {
        "claim_count": ("claim_count", "claim count", "claims", "number of claims"),
        "admissions": ("admission", "admit", "total admissions"),
        "avg_paid_per_admit": ("avg paid per admit", "average paid per admit", "paid per admission", "avg_paid"),
        "avg_allowed_per_admit": ("avg allowed per admit", "average allowed per admit", "allowed per admission", "avg_allowed"),
        "total_paid": ("total paid", "paid amount", "expense", "cost"),
        "total_allowed": ("total allowed", "allowed amount"),
        "paid_ratio": ("paid ratio", "paid to allowed", "payment ratio"),
        "pmpm": ("pmpm", "per member per month"),
    }

    aliases: dict[str, list[str]] = {canonical: [] for canonical in canonical_keywords}

    all_metric_names = set(metric_catalog.keys())
    all_metric_names.update(mode.get("explainer_metrics", []) or [])
    all_metric_names.update(mode.get("drill_metric", []) or [])

    for metric_name in all_metric_names:
        meta = metric_catalog.get(metric_name, {})
        haystack = _search_text(
            metric_name,
            meta.get("expr"),
            meta.get("description"),
            " ".join(meta.get("synonyms", []) or []),
        )

        for canonical, keywords in canonical_keywords.items():
            if any(_norm(keyword) in haystack for keyword in keywords):
                aliases[canonical].append(metric_name)

    return {key: tuple(dict.fromkeys(values)) for key, values in aliases.items() if values}


@dataclass(frozen=True)
class RunRef:
    group_type: str
    entity_name: str
    run: Mapping[str, Any]


def _iter_correlation_runs(data: Mapping[str, Any]) -> Iterable[RunRef]:
    if _looks_like_run(data):
        yield RunRef("root", str(data.get("job_id") or "root"), data)
        return

    for group_type, entity_map in data.items():
        if not isinstance(entity_map, Mapping):
            continue

        if _looks_like_run(entity_map):
            yield RunRef(str(group_type), str(entity_map.get("job_id") or group_type), entity_map)
            continue

        for entity_name, run in entity_map.items():
            if isinstance(run, Mapping) and _looks_like_run(run):
                yield RunRef(str(group_type), str(entity_name), run)


def _looks_like_run(obj: Mapping[str, Any]) -> bool:
    output = obj.get("output", obj)
    return isinstance(output, Mapping) and any(
        key in output for key in ("drill_path", "interaction_matrix", "root_metric", "baseline_value", "comparison_value")
    )


def _build_cards_for_run(run_ref: RunRef, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = run_ref.run.get("output", run_ref.run)
    if not isinstance(output, Mapping):
        return []

    run_ctx = _run_context(run_ref, output, cfg)
    cards: list[dict[str, Any]] = []
    cards.extend(_interaction_cell_cards(run_ctx, cfg))
    cards.extend(_drill_validation_cards(run_ctx, cfg))
    return cards


def _run_context(run_ref: RunRef, output: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    interaction = output.get("interaction_matrix", {}) or {}
    summary = interaction.get("summary", {}) or {}
    drill_path = list(output.get("drill_path", []) or [])

    return {
        "run_id": str(run_ref.run.get("job_id") or output.get("run_dir") or run_ref.entity_name),
        "conversation_id": run_ref.run.get("conversation_id"),
        "source_entity": {"type": run_ref.group_type, "name": run_ref.entity_name},
        "root_metric": output.get("root_metric"),
        "period_window": output.get("period_window"),
        "root_values": {
            "baseline": _to_float(output.get("baseline_value")),
            "comparison": _to_float(output.get("comparison_value")),
            "delta": _to_float(output.get("delta_value")),
            "delta_pct": _to_float(output.get("delta_pct")),
        },
        "drill_path": drill_path,
        "dominant_drill_context": _dominant_drill_context(drill_path),
        "interaction_matrix": interaction,
        "run_flags": _run_flags(summary, drill_path, cfg),
    }


def _interaction_cell_cards(ctx: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    interaction = ctx.get("interaction_matrix", {}) or {}
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()

    for stage in cfg["stages"]:
        stage_obj = interaction.get(stage, {}) or {}
        if not isinstance(stage_obj, Mapping):
            continue

        selected = stage_obj.get("selected_cells", []) or []
        top_preview = stage_obj.get("top_cells_preview", []) or []
        source_name = "selected_cells" if selected else "top_cells_preview"
        cells = selected if selected else top_preview

        for cell in cells:
            if not isinstance(cell, Mapping):
                continue

            card = _make_interaction_card(ctx, stage, cell, source_name, cfg)
            dedupe_key = _hash_obj(
                {
                    "run_id": ctx.get("run_id"),
                    "stage": stage,
                    "cell_id": cell.get("cell_id"),
                    "raw_dims": cell.get("dimension_values") or {},
                }
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cards.append(card)

    return cards


def _make_interaction_card(
    ctx: Mapping[str, Any],
    stage: str,
    cell: Mapping[str, Any],
    source_name: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    raw_dims = dict(cell.get("dimension_values") or {})
    dims = dict(raw_dims)
    context_dims = dict(ctx.get("dominant_drill_context") or {})
    dimension_sources = {key: "interaction_cell" for key in raw_dims}
    for key in context_dims:
        if key not in dimension_sources:
            dimension_sources[key] = "drill_context"
    canonical = _canonical_dimensions(dims, cfg)
    metrics = _cell_metrics(cell, cfg)
    flags = _card_flags(metrics, dims, canonical, ctx, cfg)
    flags = [flag for flag in flags if flag != "auth_or_mapping_shift"]

    driver = _driver_type(
        stage,
        metrics,
        flags,
        canonical,
        card_type="interaction_cell",
        source_name=source_name,
        cfg=cfg,
    )

    return {
        "card_id": _hash_obj(
            {
                "run_id": ctx.get("run_id"),
                "stage": stage,
                "cell_id": cell.get("cell_id"),
                "raw_dims": raw_dims,
            }
        ),
        "card_type": "interaction_cell",
        "stage": stage,
        "source_entity": ctx.get("source_entity"),
        "source_trace": {
            "run_id": ctx.get("run_id"),
            "conversation_id": ctx.get("conversation_id"),
            "source": source_name,
            "cell_id": cell.get("cell_id"),
            "artifact_row_ref": cell.get("artifact_row_ref"),
        },
        "root_metric": ctx.get("root_metric"),
        "period_window": ctx.get("period_window"),
        "dimensions": dims,
        "context_dimensions": context_dims,
        "dimension_sources": dimension_sources,
        "canonical_dimensions": canonical,
        "metrics": metrics,
        "driver_type": driver,
        "flags": flags,
        "filters": _filters_from_dims(
            raw_dims,
            source="interaction_cell",
            context_dims=context_dims,
            context_source="drill_context",
        ),
        "downstream_routes": _routes(stage, driver, flags, canonical),
        "rank_score": _rank_score(metrics, flags),
        "business_terms": _business_terms_for_card(stage, canonical, driver),
    }


def _drill_validation_cards(ctx: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    for level in ctx.get("drill_path", []) or []:
        dim = str(level.get("dimension") or "")
        dim_role = _role_for_dimension(dim, cfg)

        if dim_role != "auth" and not _contains_mapping_terms(dim):
            continue

        top_segments = level.get("top_segments", []) or []
        bottom_segments = level.get("bottom_segments", []) or []

        positive_delta = sum(max(0.0, _to_float(segment.get("delta_value")) or 0.0) for segment in top_segments)
        negative_delta = sum(min(0.0, _to_float(segment.get("delta_value")) or 0.0) for segment in bottom_segments)

        if positive_delta <= 0 or abs(negative_delta) <= 0:
            continue

        segment_summary = {
            "positive_segments": [
                {
                    "value": segment.get("value"),
                    "delta": _to_float(segment.get("delta_value")),
                    "baseline": _to_float(segment.get("baseline_value")),
                    "comparison": _to_float(segment.get("comparison_value")),
                    "baseline_rows": _to_float(segment.get("raw_row_count_baseline")),
                    "comparison_rows": _to_float(segment.get("raw_row_count_comparison")),
                }
                for segment in top_segments
            ],
            "negative_segments": [
                {
                    "value": segment.get("value"),
                    "delta": _to_float(segment.get("delta_value")),
                    "baseline": _to_float(segment.get("baseline_value")),
                    "comparison": _to_float(segment.get("comparison_value")),
                    "baseline_rows": _to_float(segment.get("raw_row_count_baseline")),
                    "comparison_rows": _to_float(segment.get("raw_row_count_comparison")),
                }
                for segment in bottom_segments
            ],
        }

        dims = {dim: "AUTH_CODE_DISTRIBUTION"}
        context_dims = dict(ctx.get("dominant_drill_context") or {})
        canonical = _canonical_dimensions(dims, cfg)
        metrics = {
            "value": {"baseline": None, "comparison": None, "delta": positive_delta + negative_delta},
            "share_of_positive_delta": None,
            "share_of_net_delta": None,
            "raw_row_count": {"baseline": None, "comparison": None, "delta": None},
            "explainer": {},
        }
        flags = ["auth_or_mapping_shift", "requires_validation_before_action"]

        cards.append(
            {
                "card_id": _hash_obj(
                    {
                        "run_id": ctx.get("run_id"),
                        "dimension": dim,
                        "card_type": "auth_distribution_validation",
                    }
                ),
                "card_type": "drill_validation_summary",
                "stage": "operational",
                "source_entity": ctx.get("source_entity"),
                "source_trace": {
                    "run_id": ctx.get("run_id"),
                    "conversation_id": ctx.get("conversation_id"),
                    "source": "drill_path",
                    "drill_level": level.get("level"),
                    "dimension": dim,
                },
                "root_metric": ctx.get("root_metric"),
                "period_window": ctx.get("period_window"),
                "dimensions": dims,
                "context_dimensions": context_dims,
                "dimension_sources": {
                    **{dim: "drill_path_distribution"},
                    **{key: "drill_context" for key in context_dims if key != dim},
                },
                "canonical_dimensions": canonical,
                "metrics": metrics,
                "segment_summary": segment_summary,
                "driver_type": "coding_or_mapping_validation",
                "flags": flags,
                "filters": _filters_from_dims(
                    dims,
                    source="drill_path_distribution",
                    context_dims=context_dims,
                    context_source="drill_context",
                ),
                "downstream_routes": _routes("operational", "coding_or_mapping_validation", flags, canonical),
                "rank_score": abs(positive_delta) + abs(negative_delta),
                "business_terms": ["auth-to-claim validation", "coding validation"],
            }
        )

    return cards


def _cell_metrics(cell: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "value": {
            "baseline": _to_float(cell.get("baseline_value")),
            "comparison": _to_float(cell.get("comparison_value")),
            "delta": _to_float(cell.get("delta_value")),
        },
        "share_of_positive_delta": _to_float(cell.get("share_of_positive_delta")),
        "share_of_net_delta": _to_float(cell.get("share_of_net_delta")),
        "raw_row_count": {
            "baseline": _to_float(cell.get("raw_row_count_baseline")),
            "comparison": _to_float(cell.get("raw_row_count_comparison")),
            "delta": _delta(
                _to_float(cell.get("raw_row_count_comparison")),
                _to_float(cell.get("raw_row_count_baseline")),
            ),
        },
        "explainer": _normalize_explainer_metrics(cell.get("explainer_metrics", {}) or {}, cfg),
    }


def _drill_segment_metrics(segment: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _to_float(segment.get("baseline_value"))
    comparison = _to_float(segment.get("comparison_value"))
    delta = _to_float(segment.get("aligned_delta"))
    if delta is None:
        delta = _to_float(segment.get("delta_value"))

    rows_base = _to_float(segment.get("raw_row_count_baseline"))
    rows_comp = _to_float(segment.get("raw_row_count_comparison"))

    return {
        "value": {"baseline": baseline, "comparison": comparison, "delta": delta},
        "share_of_positive_delta": _to_float(segment.get("aligned_contribution_pct_of_aligned_delta")),
        "share_of_net_delta": _to_float(segment.get("contribution_pct_total")),
        "raw_row_count": {"baseline": rows_base, "comparison": rows_comp, "delta": _delta(rows_comp, rows_base)},
        "explainer": {},
    }


def _normalize_explainer_metrics(
    explainer_metrics: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {}
    raw_by_norm = {_norm(key): value for key, value in explainer_metrics.items() if isinstance(value, Mapping)}

    for canonical, aliases in cfg.get("metric_aliases", {}).items():
        for raw_norm, raw_val in raw_by_norm.items():
            if any(_norm(alias) in raw_norm or raw_norm.endswith(_norm(alias)) for alias in aliases):
                out[canonical] = {
                    "baseline": _to_float(raw_val.get("baseline")),
                    "comparison": _to_float(raw_val.get("comparison")),
                    "delta": _to_float(raw_val.get("delta")),
                }
                break

    fallback_patterns = {
        "claim_count": ("claim_count", "claims"),
        "admissions": ("total_admissions", "admissions", "admits"),
        "avg_paid_per_admit": ("avg_paid_per_admit", "paid_per_admit"),
        "avg_allowed_per_admit": ("avg_allowed_per_admit", "allowed_per_admit"),
        "total_allowed": ("total_allowed", "allowed"),
        "paid_ratio": ("paid_ratio", "paid_to_allowed"),
    }

    for canonical, patterns in fallback_patterns.items():
        if canonical in out:
            continue
        for raw_norm, raw_val in raw_by_norm.items():
            if any(pattern in raw_norm for pattern in patterns):
                out[canonical] = {
                    "baseline": _to_float(raw_val.get("baseline")),
                    "comparison": _to_float(raw_val.get("comparison")),
                    "delta": _to_float(raw_val.get("delta")),
                }
                break

    return out


def _run_flags(summary: Mapping[str, Any], drill_path: list[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[str]:
    flags = set()
    stage = summary.get("interaction_stage", {}) or {}
    thresholds = cfg["thresholds"]

    repeated_ratio = _to_float(stage.get("observed_repeated_delta_ratio"))
    if repeated_ratio is not None and repeated_ratio > thresholds["high_repeated_delta_ratio"]:
        flags.add("repeated_delta_concentration")

    if stage.get("observed_low_volume") is True:
        flags.add("low_volume_run")

    if _detect_auth_code_shift(drill_path, cfg):
        flags.add("auth_or_mapping_shift")

    return sorted(flags)


def _detect_auth_code_shift(drill_path: list[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    for level in drill_path:
        dim = str(level.get("dimension") or "")
        if _role_for_dimension(dim, cfg) != "auth":
            continue

        positive = sum(max(0.0, _to_float(segment.get("delta_value")) or 0.0) for segment in level.get("top_segments", []) or [])
        negative = sum(min(0.0, _to_float(segment.get("delta_value")) or 0.0) for segment in level.get("bottom_segments", []) or [])

        if positive > 0 and abs(negative) > 0:
            return True

    return False


def _card_flags(
    metrics: Mapping[str, Any],
    dims: Mapping[str, Any],
    canonical: Mapping[str, list[Any]],
    ctx: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> list[str]:
    flags = set(ctx.get("run_flags", []) or [])
    thresholds = cfg["thresholds"]

    value = metrics.get("value", {}) or {}
    rows = metrics.get("raw_row_count", {}) or {}

    baseline_value = _to_float(value.get("baseline"))
    comparison_value = _to_float(value.get("comparison"))
    baseline_rows = _to_float(rows.get("baseline"))
    comparison_rows = _to_float(rows.get("comparison"))

    if comparison_value and (baseline_value is None or abs(baseline_value) <= thresholds["low_baseline_value"]):
        flags.add("low_baseline_value")

    if baseline_rows is not None and baseline_rows <= thresholds["low_baseline_rows"] and comparison_rows and comparison_rows > baseline_rows:
        flags.add("low_baseline_rows")

    if comparison_rows is not None and comparison_rows <= thresholds["small_n_rows"]:
        flags.add("small_n_rows")

    admissions = _metric(metrics, "admissions")
    if admissions:
        comp_admissions = _to_float(admissions.get("comparison"))
        if comp_admissions is not None and comp_admissions <= thresholds["small_n_admissions"]:
            flags.add("small_n_admissions")

    if any(_is_unmapped_or_unknown(value_item) for value_item in _flatten_values(dims.values())):
        flags.add("unmapped_or_unknown_value")

    if "auth" in canonical:
        flags.add("auth_dimension_present")

    return sorted(flags)


def _driver_type(
    stage: str,
    metrics: Mapping[str, Any],
    flags: list[str],
    canonical: Mapping[str, list[Any]],
    *,
    card_type: str,
    source_name: str,
    cfg: Mapping[str, Any],
) -> str:
    if card_type.startswith("drill_validation") and "auth" in canonical:
        return "coding_or_mapping_validation"

    if card_type.startswith("drill_validation") and "unmapped_or_unknown_value" in flags:
        return "mapping_validation"

    if "unmapped_or_unknown_value" in flags and card_type != "interaction_cell":
        return "mapping_validation"

    admissions_delta = _metric_delta(metrics, "admissions")
    claim_delta = _metric_delta(metrics, "claim_count")

    avg_paid = _metric(metrics, "avg_paid_per_admit")
    avg_allowed = _metric(metrics, "avg_allowed_per_admit")

    unit_delta = _first_non_null(
        _to_float((avg_paid or {}).get("delta")),
        _to_float((avg_allowed or {}).get("delta")),
    )

    volume_up = (admissions_delta is not None and admissions_delta > 0) or (claim_delta is not None and claim_delta > 0)
    volume_flat_or_down = (admissions_delta is not None and admissions_delta <= 0) or (claim_delta is not None and claim_delta <= 0)
    unit_up = _is_unit_cost_up(unit_delta, avg_paid or avg_allowed, cfg)

    if stage == "clinical" and ("small_n_admissions" in flags or "low_baseline_value" in flags):
        return "clinical_case_mix_validation"

    if stage == "clinical":
        if volume_up and unit_up:
            return "clinical_mixed_volume_intensity"
        if volume_up:
            return "clinical_volume_or_case_mix"
        if unit_up:
            return "clinical_intensity_or_case_mix"
        return "clinical_case_mix"

    if volume_up and unit_up:
        return "mixed_volume_unit_cost"

    if volume_up:
        return "volume_led"

    if volume_flat_or_down and unit_up:
        return "unit_cost_led"

    if unit_up:
        return "unit_cost_led"

    if "low_baseline_value" in flags:
        return "low_baseline_emergence"

    return "mix_or_other"


def _is_unit_cost_up(delta: Optional[float], metric: Optional[Mapping[str, Any]], cfg: Mapping[str, Any]) -> bool:
    if delta is None:
        return False

    thresholds = cfg["thresholds"]
    baseline = _to_float((metric or {}).get("baseline"))

    if delta > thresholds["unit_cost_material_abs"]:
        return True

    if baseline not in (None, 0):
        return (delta / abs(baseline)) >= thresholds["unit_cost_material_pct"]

    return False


def _group_cards(
    cards: list[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    max_top_cards_per_group: int,
) -> list[dict[str, Any]]:
    grouping_stats = _build_grouping_stats(cards)
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)

    for card in cards:
        buckets[_group_key(card, cfg, grouping_stats)].append(card)

    groups = [
        _summarize_group(group_key, group_cards, cfg, max_top_cards_per_group=max_top_cards_per_group)
        for group_key, group_cards in buckets.items()
    ]
    groups.sort(key=lambda group: abs(group["impact"]["total_delta"]), reverse=True)

    for index, group in enumerate(groups, start=1):
        group["pattern_rank"] = index

    return groups


def _build_grouping_stats(cards: list[Mapping[str, Any]]) -> dict[str, Any]:
    role_value_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    role_value_entities: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for card in cards:
        stage = str(card.get("stage") or "unknown")
        source_entity = card.get("source_entity", {}) or {}
        entity_name = str(source_entity.get("name") or "unknown")

        for role, values in (card.get("canonical_dimensions", {}) or {}).items():
            for value in values:
                value_str = str(value)
                role_value_counts[(stage, role)][value_str] += 1
                role_value_entities[(stage, role, value_str)].add(entity_name)

    return {
        "role_value_counts": role_value_counts,
        "role_value_entities": role_value_entities,
    }


def _group_key(
    card: Mapping[str, Any],
    cfg: Mapping[str, Any],
    grouping_stats: Mapping[str, Any],
) -> tuple[Any, ...]:
    stage = str(card.get("stage") or "unknown")
    if stage == "clinical":
        return _clinical_group_key(card, cfg, grouping_stats)
    return _operational_group_key(card, cfg, grouping_stats)


def _clinical_group_key(
    card: Mapping[str, Any],
    cfg: Mapping[str, Any],
    grouping_stats: Mapping[str, Any],
) -> tuple[Any, ...]:
    canonical = card.get("canonical_dimensions", {}) or {}
    driver_group = _driver_group(str(card.get("driver_type") or "unknown"))
    mechanism = _clinical_mechanism(card)

    key_parts: list[Any] = ["clinical", driver_group]
    clinical_category = _top_values_for_role(canonical, "clinical_category", max_values=1)
    if clinical_category:
        key_parts.append(("clinical_category", tuple(clinical_category)))
    else:
        key_parts.append(("clinical_category", ("CLINICAL_UNCLASSIFIED",)))

    key_parts.append(("clinical_mechanism", (mechanism,)))

    recurring_drgs = _recurring_values_for_role(
        card=card,
        role="drg",
        grouping_stats=grouping_stats,
        min_cards=3,
        min_entities=2,
        max_values=1,
    )
    if recurring_drgs:
        key_parts.append(("recurring_drg", tuple(recurring_drgs)))

    return tuple(key_parts)


def _operational_group_key(
    card: Mapping[str, Any],
    cfg: Mapping[str, Any],
    grouping_stats: Mapping[str, Any],
) -> tuple[Any, ...]:
    stage = str(card.get("stage") or "operational")
    driver_group = _driver_group(str(card.get("driver_type") or "unknown"))
    canonical = card.get("canonical_dimensions", {}) or {}

    configured_dims = cfg.get("stage_dimensions", {}).get(stage, []) or []
    configured_roles = [_role_for_dimension(dim, cfg) or "other" for dim in configured_dims]

    excluded_primary_roles = {"geography", "provider"}
    max_roles = 2

    key_parts: list[Any] = [stage, driver_group]
    used_roles: set[str] = set()

    if driver_group in {"coding_validation", "mapping_validation"}:
        if "auth" in canonical:
            return (stage, driver_group, ("auth", ("AUTH_CODE_DISTRIBUTION",)))
        if "facility" in canonical:
            return (stage, driver_group, ("facility", ("MAPPING_OR_UNKNOWN_BUCKET",)))

    for role in configured_roles:
        if role in used_roles or role in excluded_primary_roles:
            continue
        if role == "auth":
            continue

        values = _top_values_for_role(canonical, role, max_values=2)
        if not values:
            continue

        key_parts.append((role, tuple(values)))
        used_roles.add(role)
        if len(used_roles) >= max_roles:
            break

    if len(key_parts) == 2:
        for role in ("product", "facility", "clinical_category", "admission_source", "line_of_business", "auth"):
            values = _top_values_for_role(canonical, role, max_values=2)
            if values:
                key_parts.append((role, tuple(values)))
                break

    return tuple(key_parts)


def _clinical_mechanism(card: Mapping[str, Any]) -> str:
    driver = str(card.get("driver_type") or "")
    flags = set(card.get("flags", []) or [])
    metrics = card.get("metrics", {}) or {}

    admissions_delta = _metric_delta(metrics, "admissions")
    claim_delta = _metric_delta(metrics, "claim_count")

    avg_paid = _metric(metrics, "avg_paid_per_admit")
    avg_allowed = _metric(metrics, "avg_allowed_per_admit")
    unit_delta = _first_non_null(
        _to_float((avg_paid or {}).get("delta")),
        _to_float((avg_allowed or {}).get("delta")),
    )

    volume_up = (admissions_delta is not None and admissions_delta > 0) or (claim_delta is not None and claim_delta > 0)
    unit_up = unit_delta is not None and unit_delta > 0

    if "small_n_admissions" in flags or "low_baseline_value" in flags:
        return "new_or_rare_high_cost_cases"
    if "mixed" in driver or (volume_up and unit_up):
        return "volume_and_intensity"
    if "volume" in driver or volume_up:
        return "volume_or_case_mix"
    if "intensity" in driver or "unit_cost" in driver or unit_up:
        return "intensity_or_reimbursement"
    return "case_mix_shift"


def _recurring_values_for_role(
    *,
    card: Mapping[str, Any],
    role: str,
    grouping_stats: Mapping[str, Any],
    min_cards: int,
    min_entities: int,
    max_values: int,
) -> list[str]:
    stage = str(card.get("stage") or "unknown")
    canonical = card.get("canonical_dimensions", {}) or {}
    values = [str(value) for value in canonical.get(role, []) or []]
    if not values:
        return []

    role_value_counts = grouping_stats.get("role_value_counts", {})
    role_value_entities = grouping_stats.get("role_value_entities", {})
    recurring: list[str] = []

    for value in values:
        card_count = role_value_counts.get((stage, role), Counter()).get(value, 0)
        entity_count = len(role_value_entities.get((stage, role, value), set()))
        if card_count >= min_cards or entity_count >= min_entities:
            recurring.append(value)

    return sorted(recurring)[:max_values]


def _summarize_group(
    group_key: tuple[Any, ...],
    members: list[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    max_top_cards_per_group: int,
) -> dict[str, Any]:
    dims_by_role: dict[str, Counter] = defaultdict(Counter)
    entities_by_type: dict[str, Counter] = defaultdict(Counter)
    flags = Counter()
    driver_types = Counter()
    routes = Counter()

    total_delta = 0.0
    total_abs_delta = 0.0

    for card in members:
        delta = _to_float(((card.get("metrics", {}) or {}).get("value", {}) or {}).get("delta")) or 0.0
        total_delta += delta
        total_abs_delta += abs(delta)

        for role, values in (card.get("canonical_dimensions", {}) or {}).items():
            for value in values:
                dims_by_role[role][str(value)] += 1

        source_entity = card.get("source_entity", {}) or {}
        if source_entity.get("type") and source_entity.get("name"):
            entities_by_type[str(source_entity["type"])][str(source_entity["name"])] += 1

        flags.update(card.get("flags", []) or [])
        driver_types.update([str(card.get("driver_type"))])

        for route, enabled in (card.get("downstream_routes", {}) or {}).items():
            if enabled:
                routes[route] += 1

    top_cards = sorted(members, key=lambda card: card.get("rank_score") or 0.0, reverse=True)
    top_compact_cards = [_compact_card(card) for card in top_cards[:max_top_cards_per_group]]

    stage = str(group_key[0])
    driver_group = str(group_key[1])

    return {
        "group_id": _hash_obj({"group_key": group_key}),
        "pattern_rank": None,
        "pattern_name": _pattern_name(group_key, dims_by_role),
        "stage": stage,
        "driver_group": driver_group,
        "group_key": group_key,
        "impact": {"total_delta": total_delta, "total_abs_delta": total_abs_delta, "card_count": len(members)},
        "what_is_impacting": _what_is_impacting(dims_by_role),
        "impacted_entities": {
            entity_type: [name for name, _ in counter.most_common(10)]
            for entity_type, counter in entities_by_type.items()
        },
        "top_dimensions": {
            role: [value for value, _ in counter.most_common(10)]
            for role, counter in dims_by_role.items()
        },
        "driver_types": dict(driver_types),
        "flags": [flag for flag, _ in flags.most_common()],
        "downstream_routes": [route for route, _ in routes.most_common()],
        "card_ids": [card["card_id"] for card in top_cards],
        "top_cards": top_compact_cards,
        "business_story_seed": _business_story_seed(stage, driver_group, dims_by_role, top_compact_cards),
    }


def _driver_group(driver: str) -> str:
    if "coding" in driver:
        return "coding_validation"
    if "mapping" in driver:
        return "mapping_validation"
    if "clinical" in driver:
        return "clinical_case_mix"
    if "mixed" in driver:
        return "mixed_volume_unit_cost"
    if "unit_cost" in driver:
        return "unit_cost"
    if "volume" in driver:
        return "utilization"
    if "baseline" in driver:
        return "low_baseline_emergence"
    return "mix_or_other"


def _pattern_name(group_key: tuple[Any, ...], dims_by_role: Mapping[str, Counter]) -> str:
    stage = str(group_key[0]).title()
    driver = str(group_key[1]).replace("_", " ")

    label_parts: list[str] = []
    for part in group_key[2:]:
        if isinstance(part, tuple) and len(part) == 2:
            _, values = part
            label_parts.extend(str(value) for value in values)

    if not label_parts:
        for role in ("product", "facility", "clinical_category", "drg", "diagnosis", "auth"):
            if role in dims_by_role:
                label_parts.extend([value for value, _ in dims_by_role[role].most_common(2)])
                break

    label = " / ".join(label_parts[:3]) if label_parts else "unclassified segment"
    return f"{stage} {driver}: {label}"


def _what_is_impacting(dims_by_role: Mapping[str, Counter]) -> str:
    priority = ("clinical_category", "facility", "product", "auth", "admission_source", "drg", "diagnosis")
    parts: list[str] = []

    for role in priority:
        if role in dims_by_role:
            parts.extend([value for value, _ in dims_by_role[role].most_common(2)])

    return " / ".join(parts[:5]) if parts else "Unclassified business segment"


def _business_story_seed(
    stage: str,
    driver_group: str,
    dims_by_role: Mapping[str, Counter],
    top_cards: list[Mapping[str, Any]],
) -> str:
    impacted = _what_is_impacting(dims_by_role)
    # Combine `state` + `geography` for the story seed's "Priority markets"
    # line. State was pulled out of `geography` to keep ZIP codes from
    # leaking into pattern.priority_entities.states; the story-seed text
    # is still about geographic scope, so we surface both here.
    geos = [
        value for value, _ in dims_by_role.get("state", Counter()).most_common(5)
    ] + [
        value for value, _ in dims_by_role.get("geography", Counter()).most_common(5)
    ]
    providers = [value for value, _ in dims_by_role.get("provider", Counter()).most_common(5)]

    geo_text = f" Priority markets: {', '.join(geos)}." if geos else ""
    provider_text = f" Priority providers: {', '.join(providers)}." if providers else ""

    if driver_group == "coding_validation":
        return f"{impacted} shows a coding/auth mapping validation pattern before business action.{geo_text}{provider_text}"
    if driver_group == "mapping_validation":
        return f"{impacted} includes unmapped or unknown buckets that should be reconciled before operational interpretation.{geo_text}{provider_text}"
    if driver_group == "utilization":
        return f"{impacted} appears primarily volume-led, with admissions or claim counts driving the increase.{geo_text}{provider_text}"
    if driver_group == "unit_cost":
        return f"{impacted} appears unit-cost-led, with paid or allowed per admission increasing more than volume.{geo_text}{provider_text}"
    if driver_group == "mixed_volume_unit_cost":
        return f"{impacted} shows both utilization and unit-cost pressure, so review volume, level of care, and reimbursement together.{geo_text}{provider_text}"
    if driver_group == "clinical_case_mix":
        return f"{impacted} points to a focused clinical case-mix pattern rather than a broad clinical trend.{geo_text}{provider_text}"
    return f"{impacted} has a material movement that should be reviewed with the attached cards.{geo_text}{provider_text}"


def _canonical_dimensions(dims: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)

    for dim_name, value in dims.items():
        role = _role_for_dimension(str(dim_name), cfg) or "other"
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is None:
                continue
            if item not in out[role]:
                out[role].append(item)

    return dict(out)


def _role_for_dimension(dim_name: str, cfg: Mapping[str, Any]) -> Optional[str]:
    normalized_name = _norm(dim_name)

    for role, names in cfg.get("dimension_roles", {}).items():
        for name in names:
            normalized_candidate = _norm(name)
            if normalized_name == normalized_candidate or normalized_name.endswith(normalized_candidate) or normalized_candidate in normalized_name:
                return role

    # `state` and `zip` are ordered ahead of `geography`/`provider` so a
    # dimension name like `provider_state_code` or `src_provider_zip_code`
    # doesn't accidentally end up in the broader geography or provider
    # bucket (state used to be pooled with ZIP under `geography`, which
    # let ZIP-code values leak into pattern.priority_entities.states).
    fallback = {
        "state": ("_state_code", "_state", "provider_state", "rendering_state", "member_brand_state"),
        "zip": ("zip", "postal"),
        "geography": ("market", "region", "county"),
        "provider": ("provider", "hospital", "rendering"),
        "product": ("product", "plan", "benefit"),
        "facility": ("facility", "site_of_care", "place_of_service"),
        "auth": ("auth", "pa_required", "prior_authorization"),
        "admission_source": ("er_admit", "emergency", "admission_source"),
        "line_of_business": ("lob", "line_of_business"),
        "clinical_category": ("hcc", "clinical", "service_category"),
        "drg": ("drg",),
        "diagnosis": ("diagnosis", "diag", "dx"),
    }

    for role, tokens in fallback.items():
        if any(token in normalized_name for token in tokens):
            return role

    return None


def _dominant_drill_context(drill_path: list[Mapping[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}

    for level in drill_path:
        dim = level.get("dimension")
        top_segments = level.get("top_segments", []) or []
        if not dim or not top_segments:
            continue

        best = max(
            top_segments,
            key=lambda segment: abs(_to_float(segment.get("aligned_delta")) or _to_float(segment.get("delta_value")) or 0.0),
        )
        context[str(dim)] = best.get("value")

    return context


def _top_values_for_role(canonical: Mapping[str, list[Any]], role: str, max_values: int) -> list[str]:
    values = canonical.get(role, []) or []
    return sorted({str(value) for value in values if value is not None})[:max_values]


def _routes(stage: str, driver: str, flags: list[str], canonical: Mapping[str, list[Any]]) -> dict[str, bool]:
    flag_text = " ".join(flags)
    return {
        "clinical_agent": stage == "clinical" or any(role in canonical for role in ("clinical_category", "drg", "diagnosis", "procedure")),
        "um_operations_agent": "volume" in driver or "admission_source" in canonical or "auth" in canonical,
        "reimbursement_policy_agent": "unit_cost" in driver or "facility" in canonical or "auth" in canonical,
        "network_financial_agent": "provider" in canonical or "unit_cost" in driver,
        "data_quality_agent": any(token in flag_text for token in ("mapping", "unknown", "baseline", "repeated")) or "coding" in driver,
    }


def _business_terms_for_card(stage: str, canonical: Mapping[str, list[Any]], driver: str) -> list[str]:
    terms: list[str] = []

    if "auth" in canonical:
        terms.append("auth-to-claim validation")
    if "facility" in canonical:
        terms.append("site-of-care")
    if "admission_source" in canonical:
        terms.append("admission source")
    if "provider" in canonical:
        terms.append("provider economics")
    if "clinical_category" in canonical or "drg" in canonical or "diagnosis" in canonical:
        terms.append("case-mix")
    if "unit_cost" in driver:
        terms.append("unit cost")
    if "volume" in driver:
        terms.append("utilization")
    if stage == "clinical":
        terms.append("clinical review")

    return list(dict.fromkeys(terms))


def _compact_card(card: Mapping[str, Any]) -> dict[str, Any]:
    metrics = card.get("metrics", {}) or {}
    value = metrics.get("value", {}) or {}
    explainer = metrics.get("explainer", {}) or {}

    return {
        "card_id": card.get("card_id"),
        "card_type": card.get("card_type"),
        "stage": card.get("stage"),
        "source_entity": card.get("source_entity"),
        "driver_type": card.get("driver_type"),
        "delta": value.get("delta"),
        "share_of_positive_delta": metrics.get("share_of_positive_delta"),
        "dimensions": card.get("dimensions"),
        "key_metrics": {
            "claim_count_delta": (explainer.get("claim_count", {}) or {}).get("delta"),
            "admissions_delta": (explainer.get("admissions", {}) or {}).get("delta"),
            "avg_paid_per_admit_delta": (explainer.get("avg_paid_per_admit", {}) or {}).get("delta"),
            "avg_allowed_per_admit_delta": (explainer.get("avg_allowed_per_admit", {}) or {}).get("delta"),
            "paid_ratio_delta": (explainer.get("paid_ratio", {}) or {}).get("delta"),
        },
        "flags": card.get("flags"),
        "business_terms": card.get("business_terms"),
    }


def _filters_from_dims(
    dims: Mapping[str, Any],
    *,
    source: str,
    context_dims: Optional[Mapping[str, Any]] = None,
    context_source: str = "drill_context",
) -> list[dict[str, Any]]:
    placeholder_values = {"AUTH_CODE_DISTRIBUTION"}
    filters: list[dict[str, Any]] = []
    seen: set[str] = set()

    def normalized_values(value: Any) -> list[Any]:
        values = value if isinstance(value, list) else [value]
        cleaned: list[Any] = []
        for item in values:
            text = str(item or "").strip()
            if not text or text in placeholder_values:
                continue
            cleaned.append(item)
        return cleaned

    def add_filters(source_label: str, payload: Mapping[str, Any]) -> None:
        for field, value in payload.items():
            normalized_field = str(field or "").strip()
            if not normalized_field or normalized_field in seen:
                continue
            values = normalized_values(value)
            if not values:
                continue
            seen.add(normalized_field)
            filters.append(
                {
                    "field": normalized_field,
                    "operator": "in" if len(values) > 1 else "=",
                    "value": values if len(values) > 1 else values[0],
                    "source": source_label,
                }
            )

    add_filters(source, dims)
    if context_dims:
        add_filters(context_source, context_dims)
    return filters


def _card_passes_threshold(card: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    delta = _to_float(((card.get("metrics", {}) or {}).get("value", {}) or {}).get("delta"))
    if delta is None:
        return False
    return abs(delta) >= cfg["thresholds"]["min_abs_delta"]


def _rank_score(metrics: Mapping[str, Any], flags: list[str]) -> float:
    delta = abs(_to_float((metrics.get("value", {}) or {}).get("delta")) or 0.0)
    share = abs(_to_float(metrics.get("share_of_positive_delta")) or 0.0)

    penalty = 1.0
    if "small_n_rows" in flags or "small_n_admissions" in flags:
        penalty *= 0.75
    if "low_baseline_value" in flags:
        penalty *= 0.90

    return delta * (1.0 + min(share, 2.0)) * penalty


def _load_json_or_dict(obj: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    path = Path(obj)
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML root must be a mapping. Got: {type(loaded)}")
    return loaded


def _metric(metrics: Mapping[str, Any], canonical_name: str) -> Optional[Mapping[str, Any]]:
    value = (metrics.get("explainer", {}) or {}).get(canonical_name)
    return value if isinstance(value, Mapping) else None


def _metric_delta(metrics: Mapping[str, Any], canonical_name: str) -> Optional[float]:
    metric = _metric(metrics, canonical_name)
    return _to_float(metric.get("delta")) if metric else None


def _detect_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    return _detect_number(value)


def _delta(comparison: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if comparison is None or baseline is None:
        return None
    return comparison - baseline


def _first_non_null(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _flatten_values(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if isinstance(value, list):
            out.extend(value)
        else:
            out.append(value)
    return out


def _is_unmapped_or_unknown(value: Any) -> bool:
    text = _norm(value)
    return (
        text in {"", "none", "null", "unknown", "not_mapped", "notmapped", "unmapped", "-3", "3"}
        or "not_mapped" in text
        or "unknown" in text
        or "unmapped" in text
    )


def _contains_mapping_terms(value: Any) -> bool:
    text = _norm(value)
    return any(token in text for token in ("mapping", "mapped", "unknown", "auth", "authorization", "pa_required"))


def _search_text(*parts: Any) -> str:
    return " ".join(_norm(part) for part in parts if part is not None)


def _norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _hash_obj(obj: Any) -> str:
    return blake2b(repr(obj).encode("utf-8", errors="ignore"), digest_size=10).hexdigest()


PATTERN_AGENT_VERSION = "3.1.0"
PATTERN_AGENT_NAME = "pattern_agent"
DEFAULT_ANALYSIS_MODE_NAME = "cost_change_investigation_over_time_window"
MAX_PATTERNS_DEFAULT = 8
MAX_TOP_CARDS_PER_GROUP_DEFAULT = 5
DEFAULT_USE_CASE_NAME = "Cost of Care Pattern Analysis for Health Care Insurance"
DEFAULT_BUSINESS_QUESTION = "What are the key patterns for executive to look at?"
_MODULE_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _MODULE_PATH.parents[4] if len(_MODULE_PATH.parents) > 4 else _MODULE_PATH.parent
DEFAULT_SEMANTIC_CONFIG_PATH = _PROJECT_ROOT / "configs" / "correlation_pattern" / "coc_ecap_ip_auth_sematic_view_with_samples.yaml"

PATTERN_SYSTEM_PROMPT = """You are a Business Pattern Agent.

Your job is to convert analytical pattern groups into a concise business-facing pattern table. You are not the correlation agent, clinical agent, reimbursement agent, or SQL agent. Do not rerun analysis. Do not invent facts. Use only the supplied groups, cards, metrics, flags, and source traces.

Goal:
Collapse analytical groups into 5-8 crisp business patterns. Prefer 8 patterns when separate business mechanisms would otherwise be hidden. Do not compress below 8 solely for brevity if oncology, maternity/C-section, provider economics, site-of-care, validation, utilization, and rare high-cost outliers are all materially present.

Rules:
1. Produce business-friendly patterns, not technical readouts.
2. Do not mention internal terms such as drill path, interaction matrix, selected cell, cell id, group key, aligned delta, source trace, or explainer metrics in business-facing fields.
3. Rank patterns by business materiality, recurrence across entities, actionability, and validation importance.
4. Merge clinical groups only when they share the same business mechanism and likely downstream action. DRGs and diagnoses should usually be key drivers inside a broader clinical pattern, but do not merge oncology/chemotherapy episodes, maternity/C-section episodes, and rare ultra-high-cost Med/Surg outliers into one catch-all pattern when each is material.
5. Treat PA/auth/coding/mapping issues as gating validation patterns. If present, create one consolidated validation pattern unless the evidence clearly supports multiple distinct validation issues.
6. Separate these business mechanisms when evidence supports them:
   - utilization or admission growth
   - unit cost or reimbursement pressure
   - mixed volume and unit-cost pressure
   - PA/auth/coding/mapping validation
   - site-of-care or facility shift
   - focused clinical case-mix
   - provider or network economics
   - low-baseline or small-volume emergence
7. Special clinical split rule: keep inpatient oncology/chemotherapy, maternity/C-section, and rare catastrophic Med/Surg/transplant/ECMO/trach outlier patterns separate when they appear as material drivers, because their business actions differ.
8. Aim to cover all analytical groups, but narrative clarity takes priority over forced compression. It is better to return 8 focused patterns than 7 patterns that hide a distinct business lever. If you fold/merge multiple analytical groups into one business pattern story, include all of their group_ids in source_group_ids. Avoid reusing the same group across multiple patterns unless the evidence clearly supports it.
9. source_group_ids define the business scope of the pattern. source_card_ids are representative evidence, not the full scope. Do not shrink estimated_delta, priority_entities, key_driver_codes, or the business narrative to only the selected source_card_ids when the pattern intentionally uses multiple source_group_ids.
10. Preserve traceability using group_ids and card_ids, but keep traceability out of the business narrative.
11. Use rounded numbers:
   - dollars: nearest $0.1M when large, otherwise nearest $1K
   - percentages: nearest whole percent or one decimal if needed
   - counts: whole numbers
12. Be direct. If a pattern needs validation before action, say so.
13. Avoid vague recommendations such as "review trends" or "monitor closely." Tie every pattern to a concrete business lever.

Output must be valid JSON only. No markdown.
"""

PATTERN_USER_PROMPT_TEMPLATE = """
Convert the supplied analytical groups into a crisp business pattern table.

Context:
- Use case: {use_case_name}
- Business question: {business_question}
- Desired number of business patterns: 5 to 8; prefer 8 when distinct business mechanisms are present.
- Audience: business leaders plus downstream clinical, reimbursement policy, UM operations, network, and data-quality agents

Input:
You will receive JSON with:
- groups: analytical grouped findings
- cards: optional atomic evidence cards
- stats: summary counts

Instructions:
1. Collapse the analytical groups into 5-8 business patterns.
2. Each pattern should tell a clear story:
   - what changed
   - where it changed
   - why it matters
   - what business action or downstream review is appropriate
3. Do not list every state, provider, DRG, or diagnosis unless it materially supports the story.
4. For clinical patterns, group by clinical category and mechanism first. Put DRGs/diagnoses in key_driver_codes. Do not merge oncology, maternity/C-section, and rare catastrophic Med/Surg outliers into one pattern if each is material.
5. For PA/auth mapping, create a gating validation pattern if the evidence shows PA code movement, unmapped buckets, or classification shifts.
6. Assign analytical groups to patterns via source_group_ids to maximize coverage. If you fold/merge multiple analytical groups into one business pattern story, include all of their group_ids in source_group_ids (avoid leaving groups unassigned unless they are clearly immaterial or redundant). Avoid reusing the same group across multiple patterns unless necessary.
7. Keep source_card_ids as the minimal representative evidence set. Use source_group_ids to preserve pattern scope. Do not derive estimated_delta, priority_entities, key_driver_codes, or pattern scope only from selected cards when the pattern spans multiple analytical groups.
8. For each pattern, include only the strongest 2-4 evidence bullets.
9. Use group_ids and card_ids for traceability.
10. Do not invent additional facts, actions, or metrics not supported by the input.

Return JSON using this schema:
{op_schema}

Analytical groups JSON:
{groups_json}
"""

PATTERN_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "business_patterns": [
        {
            "pattern_rank": 1,
            "top_pattern": "Short business-friendly pattern name",
            "pattern_type": "utilization_growth | unit_cost_pressure | mixed_volume_unit_cost | coding_mapping_validation | site_of_care_shift | clinical_case_mix | provider_network_economics | low_baseline_emergence",
            "what_is_impacting": "Business area impacted, e.g., Commercial HMO / Acute Hospital / IP Med/Surg",
            "priority_entities": {
                "states": [],
                "providers": [],
                "products": [],
                "facility_types": [],
                "clinical_categories": [],
            },
            "key_driver_codes": [
                "Only the most important PA flags, DRGs, diagnoses, facility types, products, or providers"
            ],
            "impact_summary": {
                "primary_metric": "total_paid",
                "direction": "increase | decrease | mixed",
                "estimated_delta": "Rounded business-readable impact, e.g., +$14.2M",
                "volume_signal": "admissions up / claims up / flat / down / not material / unknown",
                "unit_cost_signal": "paid per admit up / allowed per admit up / flat / down / mixed / unknown",
            },
            "pattern_details": "2-4 sentence business story. No technical jargon.",
            "why_it_matters": "One sentence explaining business importance.",
            "recommended_next_step": "Concrete next step tied to business lever.",
            "validation_needed": True,
            "validation_reason": "Required only if validation_needed=true; otherwise null.",
            "downstream_routes": [
                "clinical_agent",
                "reimbursement_policy_agent",
                "um_operations_agent",
                "network_financial_agent",
                "data_quality_agent",
            ],
            "evidence_summary": ["2-4 crisp evidence bullets with rounded numbers"],
            "source_group_ids": [],
            "source_card_ids": [],
        }
    ],
    "executive_summary": {
        "headline": "One sentence summary of the overall story",
        "primary_business_message": "2-3 sentence synthesis",
        "recommended_focus_order": ["First priority", "Second priority", "Third priority"],
    },
    "quality_checks": {
        "patterns_returned": 0,
        "groups_consumed": 0,
        "ungrouped_group_ids": [],
        "notes": [],
    },
}


class PatternAgentInput(TypedDict, total=False):
    conversation_id: str
    context: Dict[str, Any]
    job_id: str
    query: str
    max_patterns: int
    analysis_mode_name: str
    semantic_config_path: str
    min_abs_delta: Optional[float]
    max_top_cards_per_group: int


class PatternPriorityEntities(TypedDict):
    states: List[str]
    providers: List[str]
    products: List[str]
    facility_types: List[str]
    clinical_categories: List[str]


class PatternImpactSummary(TypedDict):
    primary_metric: str
    direction: Literal["increase", "decrease", "mixed"]
    estimated_delta: str
    volume_signal: str
    unit_cost_signal: str


class BusinessPattern(TypedDict):
    pattern_rank: int
    top_pattern: str
    pattern_type: str
    what_is_impacting: str
    priority_entities: PatternPriorityEntities
    key_driver_codes: List[str]
    impact_summary: PatternImpactSummary
    pattern_details: str
    why_it_matters: str
    recommended_next_step: str
    validation_needed: bool
    validation_reason: Optional[str]
    downstream_routes: List[str]
    evidence_summary: List[str]
    source_group_ids: List[str]
    source_card_ids: List[str]


class PatternExecutiveSummary(TypedDict):
    headline: str
    primary_business_message: str
    recommended_focus_order: List[str]


class PatternQualityChecks(TypedDict):
    patterns_returned: int
    groups_consumed: int
    ungrouped_group_ids: List[str]
    notes: List[str]


class PatternAgentOutput(TypedDict):
    job_id: str
    conversation_id: Optional[str]
    agent: str
    status: Literal["success", "partial_success", "failed"]
    output: Dict[str, Any]
    visual_component: Dict[str, Any]
    explanation: Dict[str, Any]
    validation: Dict[str, Any]
    tokens: Dict[str, Any]
    execution: Dict[str, Any]


class PatternGraphState(TypedDict, total=False):
    conversation_id: Optional[str]
    context: Dict[str, Any]
    query: str
    job_id: str
    llm: Any
    correlation_results: Dict[str, Any]
    semantic_config_path: str
    analysis_mode_name: str
    min_abs_delta: Optional[float]
    max_patterns: int
    max_top_cards_per_group: int
    semantic_roles: Optional[Dict[str, str]]
    analytical_output: Dict[str, Any]
    llm_output: Dict[str, Any]
    result: Dict[str, Any]
    start_time: str
    input_tokens: int
    output_tokens: int
    token_breakdown: Dict[str, Any]


class PatternPriorityEntitiesSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    states: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    facility_types: List[str] = Field(default_factory=list)
    clinical_categories: List[str] = Field(default_factory=list)


class PatternImpactSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_metric: str = "total_paid"
    direction: Literal["increase", "decrease", "mixed"] = "increase"
    estimated_delta: str = "$0"
    volume_signal: str = "unknown"
    unit_cost_signal: str = "unknown"


class BusinessPatternSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pattern_rank: int = 1
    top_pattern: str = ""
    pattern_type: str = "utilization_growth"
    what_is_impacting: str = ""
    priority_entities: PatternPriorityEntitiesSchema = Field(default_factory=PatternPriorityEntitiesSchema)
    key_driver_codes: List[str] = Field(default_factory=list)
    impact_summary: PatternImpactSummarySchema = Field(default_factory=PatternImpactSummarySchema)
    pattern_details: str = ""
    why_it_matters: str = ""
    recommended_next_step: str = ""
    validation_needed: bool = False
    validation_reason: Optional[str] = None
    downstream_routes: List[str] = Field(default_factory=list)
    evidence_summary: List[str] = Field(default_factory=list)
    source_group_ids: List[str] = Field(default_factory=list)
    source_card_ids: List[str] = Field(default_factory=list)


class PatternExecutiveSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    headline: str = ""
    primary_business_message: str = ""
    recommended_focus_order: List[str] = Field(default_factory=list)


class PatternQualityChecksSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patterns_returned: int = 0
    groups_consumed: int = 0
    ungrouped_group_ids: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PatternSynthesisSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    business_patterns: List[BusinessPatternSchema] = Field(default_factory=list)
    executive_summary: PatternExecutiveSummarySchema = Field(default_factory=PatternExecutiveSummarySchema)
    quality_checks: PatternQualityChecksSchema = Field(default_factory=PatternQualityChecksSchema)


class PatternOutputSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    business_patterns: List[BusinessPatternSchema] = Field(default_factory=list)
    executive_summary: PatternExecutiveSummarySchema = Field(default_factory=PatternExecutiveSummarySchema)
    quality_checks: PatternQualityChecksSchema = Field(default_factory=PatternQualityChecksSchema)
    cards: List[Dict[str, Any]] = Field(default_factory=list)
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    semantic_summary: Dict[str, Any] = Field(default_factory=dict)


class PatternValidationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_valid: bool
    checks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class PatternTokensSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: int = 0
    output: int = 0
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class PatternExecutionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: int = 0
    version: str = PATTERN_AGENT_VERSION


class PatternAgentApiResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    conversation_id: Optional[str]
    agent: str
    status: Literal["success", "partial_success", "failed"]
    output: PatternOutputSchema
    visual_component: Dict[str, Any] = Field(default_factory=dict)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    validation: PatternValidationSchema
    tokens: PatternTokensSchema
    execution: PatternExecutionSchema


class PatternAgent(AgentBase):
    api_response_model = PatternAgentApiResponse

    def __init__(self, agent_name: str = PATTERN_AGENT_NAME, llm_timeout: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(agent_name=agent_name, state_class=PatternGraphState, llm_timeout=llm_timeout, **kwargs)

    @property
    def node_name(self) -> str:
        return "build_business_patterns"

    def create_stub_llm(self) -> Any:
        class _StubStructuredLLM:
            def __init__(self, include_raw: bool) -> None:
                self._include_raw = include_raw

            def invoke(self, messages: List[Dict[str, str]]) -> Any:
                analytical_output = _extract_analytical_json_from_messages(messages)
                response = _build_stub_response(analytical_output)
                if self._include_raw:
                    return {
                        "parsed": response,
                        "raw": {
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                            }
                        },
                    }
                return response

        class _StubLLM:
            def with_structured_output(self, _schema: Any, **kwargs: Any) -> _StubStructuredLLM:
                return _StubStructuredLLM(include_raw=bool(kwargs.get("include_raw")))

        return _StubLLM()

    def prepare_state(
        self,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        query: Optional[str] = None,
        max_patterns: Optional[int] = None,
        analysis_mode_name: Optional[str] = None,
        semantic_config_path: Optional[str] = None,
        min_abs_delta: Optional[float] = None,
        max_top_cards_per_group: Optional[int] = None,
        **kwargs: Any,
    ) -> PatternGraphState:
        if not conversation_id:
            raise ValueError("conversation_id is required for PatternAgent execution.")

        context = context or {}
        if not isinstance(context, dict):
            raise ValueError("context must be an object for PatternAgent execution.")

        correlation_results = context.get("correlation_results")
        if isinstance(correlation_results, str):
            correlation_results = json.loads(correlation_results)
        if not isinstance(correlation_results, dict):
            raise ValueError("context.correlation_results is required and must be an object.")

        semantic_candidate = (
            semantic_config_path
            or context.get("semantic_config_path")
            or context.get("semantic_view_path")
            or kwargs.get("semantic_config_path")
            or kwargs.get("semantic_view_path")
            or str(DEFAULT_SEMANTIC_CONFIG_PATH)
        )
        mode_name = (
            analysis_mode_name
            or context.get("analysis_mode_name")
            or kwargs.get("analysis_mode_name")
            or DEFAULT_ANALYSIS_MODE_NAME
        )
        normalized_query = query or context.get("business_question") or DEFAULT_BUSINESS_QUESTION
        normalized_max_patterns = _coerce_positive_int(max_patterns or context.get("max_patterns"), MAX_PATTERNS_DEFAULT)
        normalized_top_cards = _coerce_positive_int(
            max_top_cards_per_group or context.get("max_top_cards_per_group"),
            MAX_TOP_CARDS_PER_GROUP_DEFAULT,
        )

        raw_semantic_roles = (
            context.get("semantic_roles")
            or kwargs.get("semantic_roles")
        )
        normalized_semantic_roles: Optional[Dict[str, str]] = None
        if isinstance(raw_semantic_roles, Mapping):
            normalized_semantic_roles = {
                str(k): str(v) for k, v in raw_semantic_roles.items() if k and v
            }

        return {
            "conversation_id": conversation_id,
            "context": context,
            "query": str(normalized_query),
            "job_id": job_id or kwargs.get("job_id") or uuid.uuid4().hex,
            "llm": self.llm,
            "correlation_results": correlation_results,
            "semantic_config_path": str(semantic_candidate),
            "analysis_mode_name": str(mode_name),
            "min_abs_delta": min_abs_delta if min_abs_delta is not None else context.get("min_abs_delta"),
            "max_patterns": normalized_max_patterns,
            "max_top_cards_per_group": normalized_top_cards,
            "semantic_roles": normalized_semantic_roles,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "input_tokens": 0,
            "output_tokens": 0,
            "token_breakdown": {},
            "llm": self.llm,
        }

    def node_function(self, state: PatternGraphState) -> Dict[str, Any]:
        llm = state.get("llm") or self.llm
        if llm is None:
            raise AgentExecutionError("LLM is required to build business pattern output.")

        analytical_output = build_pattern_cards_and_groups(
            correlation_results=state.get("correlation_results") or {},
            semantic_config_path=state.get("semantic_config_path") or str(DEFAULT_SEMANTIC_CONFIG_PATH),
            analysis_mode_name=state.get("analysis_mode_name") or DEFAULT_ANALYSIS_MODE_NAME,
            min_abs_delta=_coerce_optional_float(state.get("min_abs_delta")),
            max_top_cards_per_group=_coerce_positive_int(
                state.get("max_top_cards_per_group"),
                MAX_TOP_CARDS_PER_GROUP_DEFAULT,
            ),
            semantic_roles=state.get("semantic_roles"),
        )

        groups = analytical_output.get("groups", []) or []
        if not groups:
            empty_result = PatternSynthesisSchema(
                business_patterns=[],
                executive_summary=PatternExecutiveSummarySchema(
                    headline="No material grouped patterns met the configured threshold.",
                    primary_business_message="The analytical grouping step did not produce material pattern candidates for business synthesis.",
                    recommended_focus_order=[],
                ),
                quality_checks=PatternQualityChecksSchema(
                    patterns_returned=0,
                    groups_consumed=0,
                    ungrouped_group_ids=[],
                    notes=["No analytical groups met the threshold before LLM synthesis."],
                ),
            )
            return {
                "analytical_output": analytical_output,
                "llm_output": empty_result.model_dump(),
                "result": _merge_output_payload(empty_result.model_dump(), analytical_output),
                "input_tokens": 0,
                "output_tokens": 0,
                "token_breakdown": {},
            }

        messages = [
            {"role": "system", "content": PATTERN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PATTERN_USER_PROMPT_TEMPLATE.format(
                    use_case_name=state.get("context", {}).get("use_case_name") or DEFAULT_USE_CASE_NAME,
                    business_question=state.get("query") or DEFAULT_BUSINESS_QUESTION,
                    op_schema=json.dumps(PATTERN_SCHEMA_EXAMPLE, indent=2),
                    groups_json=json.dumps(analytical_output, indent=2, default=str),
                ),
            },
        ]

        raw_response: Any = None
        try:
            # Use structured_llm_invoke with token retry support
            from deep_research_utils.ehap_retry import structured_llm_invoke
            
            parsed, updated_llm = structured_llm_invoke(
                llm=self.llm,
                ehap=self.ehap,
                messages=messages,
                schema=PatternSynthesisSchema,
                llm_reinitializer=self._initialize_llm,
                timeout=self.llm_timeout,
            )
            # Update self.llm to capture any token refresh that occurred
            self.llm = updated_llm
            # Token usage tracking not available with retry utility
            input_tokens, output_tokens = 0, 0
        except Exception as exc:
            raise AgentExecutionError("Failed to synthesize business pattern output.") from exc

        result_schema = _coerce_synthesis_schema(parsed)
        result_schema = _reconcile_pattern_attribution(
            result_schema,
            analytical_output,
            max_patterns=_coerce_positive_int(state.get("max_patterns"), MAX_PATTERNS_DEFAULT),
        )
        llm_output = _finalize_quality_checks(result_schema, analytical_output).model_dump()

        return {
            "analytical_output": analytical_output,
            "llm_output": llm_output,
            "result": _merge_output_payload(llm_output, analytical_output),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_breakdown": _extract_usage_breakdown(raw_response),
        }

    def extract_result(self, graph_output: Dict[str, Any]) -> PatternAgentOutput:
        analytical_output = _safe_dict(graph_output.get("analytical_output"))
        merged_output = _safe_dict(graph_output.get("result"))
        validation = _validate_output_payload(merged_output, analytical_output)
        status = _resolve_status(validation)
        start_time = graph_output.get("start_time")
        end_time = datetime.now(timezone.utc).isoformat()
        duration_ms = _duration_ms(start_time, end_time)

        return {
            "job_id": str(graph_output.get("job_id") or uuid.uuid4().hex),
            "conversation_id": graph_output.get("conversation_id"),
            "agent": PATTERN_AGENT_NAME,
            "status": status,
            "output": merged_output,
            "visual_component": {},
            "explanation": {
                "summary": f"Built {len(merged_output.get('business_patterns', []))} business patterns from {len(analytical_output.get('groups', []))} analytical groups.",
                "group_count": len(analytical_output.get("groups", [])),
                "card_count": len(analytical_output.get("cards", [])),
            },
            "validation": validation,
            "tokens": {
                "input": int(graph_output.get("input_tokens", 0) or 0),
                "output": int(graph_output.get("output_tokens", 0) or 0),
                "breakdown": dict(graph_output.get("token_breakdown") or {}),
            },
            "execution": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "version": PATTERN_AGENT_VERSION,
            },
        }

    def handle_execution_error(self, exc: Exception, **kwargs: Any) -> PatternAgentOutput:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "job_id": str(kwargs.get("job_id") or uuid.uuid4().hex),
            "conversation_id": kwargs.get("conversation_id"),
            "agent": PATTERN_AGENT_NAME,
            "status": "failed",
            "output": {
                "business_patterns": [],
                "executive_summary": {
                    "headline": "",
                    "primary_business_message": "",
                    "recommended_focus_order": [],
                },
                "quality_checks": {
                    "patterns_returned": 0,
                    "groups_consumed": 0,
                    "ungrouped_group_ids": [],
                    "notes": [],
                },
                "cards": [],
                "groups": [],
                "stats": {},
                "semantic_summary": {},
            },
            "visual_component": {},
            "explanation": {"error": str(exc)},
            "validation": {
                "is_valid": False,
                "checks": [],
                "warnings": [],
                "errors": [str(exc)],
            },
            "tokens": {"input": 0, "output": 0, "breakdown": {}},
            "execution": {
                "start_time": now,
                "end_time": now,
                "duration_ms": 0,
                "version": PATTERN_AGENT_VERSION,
            },
        }


def _coerce_synthesis_schema(value: Any) -> PatternSynthesisSchema:
    if isinstance(value, PatternSynthesisSchema):
        return value
    if isinstance(value, BaseModel):
        return PatternSynthesisSchema.model_validate(value.model_dump())
    if isinstance(value, dict):
        return PatternSynthesisSchema.model_validate(value)
    raise AgentExecutionError("Structured business pattern response could not be validated.")


def _merge_output_payload(llm_output: Mapping[str, Any], analytical_output: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "business_patterns": list(llm_output.get("business_patterns") or []),
        "executive_summary": dict(llm_output.get("executive_summary") or {}),
        "quality_checks": dict(llm_output.get("quality_checks") or {}),
        "cards": list(analytical_output.get("cards") or []),
        "groups": list(analytical_output.get("groups") or []),
        "stats": dict(analytical_output.get("stats") or {}),
        "semantic_summary": dict(analytical_output.get("semantic_summary") or {}),
    }


def _validate_output_payload(output: Mapping[str, Any], analytical_output: Mapping[str, Any]) -> Dict[str, Any]:
    checks: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []

    patterns = output.get("business_patterns") or []
    groups = analytical_output.get("groups") or []
    cards = analytical_output.get("cards") or []
    group_ids = {str(group.get("group_id")) for group in groups if group.get("group_id")}
    card_ids = {str(card.get("card_id")) for card in cards if card.get("card_id")}

    checks.append(f"Generated {len(patterns)} business patterns from {len(groups)} analytical groups.")

    for pattern in patterns:
        source_group_ids = [str(item) for item in pattern.get("source_group_ids") or [] if item]
        source_card_ids = [str(item) for item in pattern.get("source_card_ids") or [] if item]
        if not source_group_ids:
            warnings.append(f"Pattern '{pattern.get('top_pattern', 'unknown')}' has no source_group_ids.")
        if any(group_id not in group_ids for group_id in source_group_ids):
            errors.append(f"Pattern '{pattern.get('top_pattern', 'unknown')}' references unknown source_group_ids.")
        if any(card_id not in card_ids for card_id in source_card_ids):
            errors.append(f"Pattern '{pattern.get('top_pattern', 'unknown')}' references unknown source_card_ids.")

    quality_checks = _safe_dict(output.get("quality_checks"))
    if int(quality_checks.get("patterns_returned") or 0) != len(patterns):
        warnings.append("quality_checks.patterns_returned did not match the business pattern count.")
    if len(groups) and not patterns:
        warnings.append("Analytical groups were present but no business patterns were returned.")

    return {
        "is_valid": not errors,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def _resolve_status(validation: Mapping[str, Any]) -> Literal["success", "partial_success", "failed"]:
    if validation.get("errors"):
        return "failed"
    if validation.get("warnings"):
        return "partial_success"
    return "success"


def _duration_ms(start_time: Optional[str], end_time: Optional[str]) -> int:
    if not start_time or not end_time:
        return 0
    try:
        start_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(int((end_dt - start_dt).total_seconds() * 1000), 0)


def _extract_token_usage(raw_response: Any) -> tuple[int, int]:
    usage: Dict[str, Any] = {}
    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    elif isinstance(raw_response, dict):
        usage = raw_response.get("token_usage") or raw_response.get("usage") or {}

    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def _extract_usage_breakdown(raw_response: Any) -> Dict[str, Any]:
    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        return dict(response_metadata.get("token_usage") or response_metadata.get("usage") or {})
    if isinstance(raw_response, dict):
        return dict(raw_response.get("token_usage") or raw_response.get("usage") or {})
    return {}


def _extract_analytical_json_from_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    marker = "Analytical groups JSON:"
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if marker not in content:
            continue
        raw_json = content.split(marker, 1)[1].strip()
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
    return {}


def _build_stub_response(analytical_output: Mapping[str, Any]) -> PatternSynthesisSchema:
    groups = [group for group in analytical_output.get("groups", []) or [] if isinstance(group, dict)]
    business_patterns: List[BusinessPatternSchema] = []

    for index, group in enumerate(groups[: min(len(groups), MAX_PATTERNS_DEFAULT)], start=1):
        pattern_type = _pattern_type_from_group(group)
        impacted_entities = _safe_dict(group.get("impacted_entities"))
        top_dimensions = _safe_dict(group.get("top_dimensions"))
        source_group_id = str(group.get("group_id") or "")
        source_card_ids = [str(item) for item in group.get("card_ids", [])[:5] if item]
        validation_needed = pattern_type == "coding_mapping_validation" or "requires_validation_before_action" in (group.get("flags") or [])
        validation_reason = "Validate coding or mapping movement before business action." if validation_needed else None
        business_patterns.append(
            BusinessPatternSchema(
                pattern_rank=index,
                top_pattern=str(group.get("pattern_name") or f"Pattern {index}"),
                pattern_type=pattern_type,
                what_is_impacting=str(group.get("what_is_impacting") or "Unclassified business segment"),
                priority_entities=PatternPriorityEntitiesSchema(
                    states=[str(item) for item in impacted_entities.get("states", [])[:3]],
                    providers=[str(item) for item in impacted_entities.get("providers", [])[:3]],
                    products=[str(item) for item in top_dimensions.get("product", [])[:3]],
                    facility_types=[str(item) for item in top_dimensions.get("facility", [])[:3]],
                    clinical_categories=[str(item) for item in top_dimensions.get("clinical_category", [])[:3]],
                ),
                key_driver_codes=_collect_key_driver_codes(group),
                impact_summary=PatternImpactSummarySchema(
                    primary_metric="total_paid",
                    direction=_direction_from_delta(_to_float(group.get("impact", {}).get("total_delta"))),
                    estimated_delta=_format_currency(_to_float(group.get("impact", {}).get("total_delta"))),
                    volume_signal=_volume_signal_from_group(group),
                    unit_cost_signal=_unit_cost_signal_from_group(group),
                ),
                pattern_details=str(group.get("business_story_seed") or ""),
                why_it_matters=_build_why_it_matters(group),
                recommended_next_step=_build_next_step(pattern_type),
                validation_needed=validation_needed,
                validation_reason=validation_reason,
                downstream_routes=[str(item) for item in group.get("downstream_routes", [])],
                evidence_summary=_build_evidence_summary(group),
                source_group_ids=[source_group_id] if source_group_id else [],
                source_card_ids=source_card_ids,
            )
        )

    executive_summary = PatternExecutiveSummarySchema(
        headline=_build_headline(business_patterns),
        primary_business_message=_build_primary_message(business_patterns),
        recommended_focus_order=[pattern.top_pattern for pattern in business_patterns[:3]],
    )
    used_group_ids = {group_id for pattern in business_patterns for group_id in pattern.source_group_ids}
    quality_checks = PatternQualityChecksSchema(
        patterns_returned=len(business_patterns),
        groups_consumed=len(used_group_ids),
        ungrouped_group_ids=[str(group.get("group_id")) for group in groups if str(group.get("group_id")) not in used_group_ids],
        notes=[],
    )
    return PatternSynthesisSchema(
        business_patterns=business_patterns,
        executive_summary=executive_summary,
        quality_checks=quality_checks,
    )


def _reconcile_pattern_attribution(
    result_schema: PatternSynthesisSchema,
    analytical_output: Mapping[str, Any],
    *,
    max_patterns: int,
) -> PatternSynthesisSchema:
    groups = [group for group in analytical_output.get("groups", []) or [] if isinstance(group, dict)]
    cards = [card for card in analytical_output.get("cards", []) or [] if isinstance(card, dict)]
    if not groups:
        return result_schema

    if not result_schema.business_patterns:
        fallback = _build_stub_response(analytical_output)
        if max_patterns > 0:
            fallback.business_patterns = list(fallback.business_patterns[:max_patterns])
            fallback.executive_summary = PatternExecutiveSummarySchema(
                headline=_build_headline(fallback.business_patterns),
                primary_business_message=_build_primary_message(fallback.business_patterns),
                recommended_focus_order=[pattern.top_pattern for pattern in fallback.business_patterns[:3]],
            )
        fallback.quality_checks = PatternQualityChecksSchema(
            patterns_returned=fallback.quality_checks.patterns_returned,
            groups_consumed=fallback.quality_checks.groups_consumed,
            ungrouped_group_ids=list(fallback.quality_checks.ungrouped_group_ids),
            notes=_unique_preserve_order(
                [
                    *list(fallback.quality_checks.notes),
                    "LLM returned no business patterns; deterministic fallback built patterns from analytical groups.",
                ]
            ),
        )
        return fallback

    group_by_id = {
        str(group.get("group_id")): group
        for group in groups
        if str(group.get("group_id") or "")
    }
    card_by_id = {
        str(card.get("card_id")): card
        for card in cards
        if str(card.get("card_id") or "")
    }
    card_group_ids: Dict[str, List[str]] = defaultdict(list)
    for group in groups:
        group_id = str(group.get("group_id") or "")
        if not group_id:
            continue
        for card_id in group.get("card_ids", []) or []:
            normalized_card_id = str(card_id or "")
            if normalized_card_id and group_id not in card_group_ids[normalized_card_id]:
                card_group_ids[normalized_card_id].append(group_id)

    inferred_card_patterns = 0
    for pattern in result_schema.business_patterns:
        if _reconcile_pattern_with_cards(pattern, group_by_id, card_by_id, card_group_ids):
            inferred_card_patterns += 1

    result_schema.quality_checks = PatternQualityChecksSchema(
        patterns_returned=result_schema.quality_checks.patterns_returned,
        groups_consumed=result_schema.quality_checks.groups_consumed,
        ungrouped_group_ids=list(result_schema.quality_checks.ungrouped_group_ids),
        notes=_unique_preserve_order(
            [
                *list(result_schema.quality_checks.notes),
                *(
                    [f"Inferred supporting card traceability for {inferred_card_patterns} pattern(s) from pattern-group alignment."]
                    if inferred_card_patterns
                    else []
                ),
            ]
        ),
    )
    return result_schema


def _finalize_quality_checks(
    result_schema: PatternSynthesisSchema,
    analytical_output: Mapping[str, Any],
) -> PatternSynthesisSchema:
    groups = [group for group in analytical_output.get("groups", []) or [] if isinstance(group, dict)]
    valid_group_ids = {str(group.get("group_id")) for group in groups if str(group.get("group_id") or "")}
    used_group_ids = {
        str(group_id)
        for pattern in result_schema.business_patterns
        for group_id in pattern.source_group_ids
        if str(group_id) in valid_group_ids
    }
    notes = list(result_schema.quality_checks.notes)
    if not result_schema.business_patterns and groups:
        notes.append("LLM returned no business patterns even though analytical groups were available.")
    ungrouped_group_ids = [str(group.get("group_id")) for group in groups if str(group.get("group_id")) not in used_group_ids]
    if result_schema.business_patterns and ungrouped_group_ids:
        notes.append(
            f"Left {len(ungrouped_group_ids)} analytical groups unassigned to preserve clean narrative attribution."
        )
    result_schema.quality_checks = PatternQualityChecksSchema(
        patterns_returned=len(result_schema.business_patterns),
        groups_consumed=len(used_group_ids),
        ungrouped_group_ids=ungrouped_group_ids,
        notes=_unique_preserve_order(notes),
    )
    return result_schema


def _pattern_type_from_group(group: Mapping[str, Any]) -> str:
    driver_group = str(group.get("driver_group") or "")
    top_dimensions = _safe_dict(group.get("top_dimensions"))
    if driver_group in {"coding_validation", "mapping_validation"}:
        return "coding_mapping_validation"
    if driver_group == "mixed_volume_unit_cost":
        return "mixed_volume_unit_cost"
    if driver_group == "unit_cost":
        return "unit_cost_pressure"
    if driver_group == "clinical_case_mix":
        return "clinical_case_mix"
    if driver_group == "low_baseline_emergence":
        return "low_baseline_emergence"
    if top_dimensions.get("facility"):
        return "site_of_care_shift"
    if top_dimensions.get("provider"):
        return "provider_network_economics"
    return "utilization_growth"


def _pattern_family(pattern_type: str) -> str:
    family_map = {
        "coding_mapping_validation": "validation",
        "clinical_case_mix": "clinical",
        "low_baseline_emergence": "emergence",
        "mixed_volume_unit_cost": "mixed",
        "provider_network_economics": "economics",
        "site_of_care_shift": "site_of_care",
        "unit_cost_pressure": "unit_cost",
        "utilization_growth": "utilization",
    }
    return family_map.get(str(pattern_type or ""), str(pattern_type or ""))


def _pattern_descriptor_values(pattern: BusinessPatternSchema) -> List[str]:
    values = [
        str(pattern.top_pattern or ""),
        str(pattern.what_is_impacting or ""),
        str(pattern.pattern_type or ""),
    ]
    values.extend(str(item) for item in pattern.key_driver_codes if str(item))
    values.extend(_pattern_priority_values(pattern))
    values.extend(str(route) for route in pattern.downstream_routes if str(route))
    values.extend(str(item) for item in pattern.evidence_summary if str(item))
    return values


def _descriptor_tokens(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        normalized = _norm(value)
        for token in normalized.split("_"):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _pattern_priority_values(pattern: BusinessPatternSchema) -> List[str]:
    return [
        *[str(item) for item in pattern.priority_entities.states if str(item)],
        *[str(item) for item in pattern.priority_entities.providers if str(item)],
        *[str(item) for item in pattern.priority_entities.products if str(item)],
        *[str(item) for item in pattern.priority_entities.facility_types if str(item)],
        *[str(item) for item in pattern.priority_entities.clinical_categories if str(item)],
    ]


def _groups_for_pattern(
    pattern: BusinessPatternSchema,
    group_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    groups: List[Mapping[str, Any]] = []
    seen: set[str] = set()
    for group_id in pattern.source_group_ids:
        normalized_group_id = str(group_id)
        if normalized_group_id in seen or normalized_group_id not in group_by_id:
            continue
        seen.add(normalized_group_id)
        groups.append(group_by_id[normalized_group_id])
    return groups


def _reconcile_pattern_with_cards(
    pattern: BusinessPatternSchema,
    group_by_id: Mapping[str, Mapping[str, Any]],
    card_by_id: Mapping[str, Mapping[str, Any]],
    card_group_ids: Mapping[str, List[str]],
) -> bool:
    raw_group_ids = [str(group_id) for group_id in pattern.source_group_ids if str(group_id)]
    raw_card_ids = [str(card_id) for card_id in pattern.source_card_ids if str(card_id)]
    explicit_group_ids = [str(group_id) for group_id in pattern.source_group_ids if str(group_id) in group_by_id]
    groups = [group_by_id[group_id] for group_id in explicit_group_ids]
    selected_cards, inferred_cards = _support_cards_for_pattern(pattern, groups, card_by_id)
    inferred_group_ids = _group_ids_for_cards(selected_cards, card_group_ids)

    selected_card_ids = [str(card.get("card_id")) for card in selected_cards if str(card.get("card_id") or "")]
    pattern.source_card_ids = raw_card_ids if any(card_id not in card_by_id for card_id in raw_card_ids) else selected_card_ids
    if any(group_id not in group_by_id for group_id in raw_group_ids):
        pattern.source_group_ids = raw_group_ids
    else:
        pattern.source_group_ids = _unique_preserve_order([*explicit_group_ids, *inferred_group_ids])

    if selected_cards:
        pattern.downstream_routes = _downstream_routes_from_cards(selected_cards, groups, pattern.downstream_routes)

        # Cards are representative evidence. They should fill gaps, not narrow a broader LLM pattern.
        derived_key_driver_codes = _collect_card_key_driver_codes(selected_cards)
        if not pattern.key_driver_codes and derived_key_driver_codes:
            pattern.key_driver_codes = derived_key_driver_codes

        # Drop LLM-provided priority_entities.states entries that don't
        # appear in any source card's `state`-role bucket. The prompt schema
        # doesn't validate `states`, so the LLM occasionally echoes program
        # names or ZIP codes it saw in the analytical input (e.g. "BCC STATE
        # SPONSORED PROGRAM", "95816") — trusting only card-attested values
        # eliminates those without a hand-authored US-state allow-list.
        card_state_values = {
            str(value)
            for card in selected_cards
            for value in (_safe_dict(card.get("canonical_dimensions")).get("state") or [])
            if str(value)
        }
        if card_state_values:
            pattern.priority_entities.states = [
                s for s in pattern.priority_entities.states if str(s) in card_state_values
            ]

        derived_priority_entities = _priority_entities_from_cards(selected_cards)
        pattern.priority_entities = _merge_priority_entities(
            pattern.priority_entities,
            derived_priority_entities,
            _priority_entities_from_groups(groups),
        )

        # Preserve LLM-provided impact if it is non-zero/non-empty. Only derive from cards/groups when missing.
        if _impact_summary_missing(pattern.impact_summary):
            pattern.impact_summary = _impact_summary_from_groups_or_cards(
                groups=groups,
                cards=selected_cards,
                current_summary=pattern.impact_summary,
            )

        if not pattern.evidence_summary:
            pattern.evidence_summary = _build_evidence_summary_from_cards(selected_cards)

        derived_impacting = _what_is_impacting(_build_dims_by_role_from_cards(selected_cards))
        if (
            (not pattern.what_is_impacting or pattern.what_is_impacting == "Unclassified business segment")
            and derived_impacting
            and derived_impacting != "Unclassified business segment"
        ):
            pattern.what_is_impacting = derived_impacting

    validation_needed = bool(pattern.validation_needed) or any(_group_requires_validation(group) for group in groups) or any(
        _card_requires_validation(card) for card in selected_cards
    )
    pattern.validation_needed = validation_needed
    if validation_needed and not pattern.validation_reason:
        pattern.validation_reason = (
            "Validate PA/auth, coding, mapping, or unmapped classification movement before assigning business ownership."
        )
    if not pattern.what_is_impacting and groups:
        pattern.what_is_impacting = str(groups[0].get("what_is_impacting") or "")
    if not pattern.pattern_details and groups:
        pattern.pattern_details = str(groups[0].get("business_story_seed") or "")
    return inferred_cards


def _support_cards_for_pattern(
    pattern: BusinessPatternSchema,
    groups: List[Mapping[str, Any]],
    card_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[List[Mapping[str, Any]], bool]:
    raw_card_ids = [str(card_id) for card_id in pattern.source_card_ids if str(card_id)]
    explicit_card_ids = [str(card_id) for card_id in pattern.source_card_ids if str(card_id) in card_by_id]
    if raw_card_ids and not explicit_card_ids:
        return [], False
    if explicit_card_ids:
        return [card_by_id[card_id] for card_id in explicit_card_ids], False

    candidate_cards = _candidate_cards_for_pattern(groups, card_by_id)
    if not candidate_cards:
        return [], False
    return _select_support_cards(pattern, candidate_cards), True


def _candidate_cards_for_pattern(
    groups: List[Mapping[str, Any]],
    card_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    candidate_ids: List[str] = []
    for group in groups:
        candidate_ids.extend(str(card_id) for card_id in group.get("card_ids", []) or [] if str(card_id) in card_by_id)
    if candidate_ids:
        return [card_by_id[card_id] for card_id in _unique_preserve_order(candidate_ids)]
    return list(card_by_id.values())


def _select_support_cards(
    pattern: BusinessPatternSchema,
    candidate_cards: List[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    scored_cards = [
        (
            _card_pattern_score(card, pattern),
            abs(_card_delta(card)),
            _to_float(card.get("rank_score")) or 0.0,
            card,
        )
        for card in candidate_cards
    ]
    scored_cards.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if not scored_cards:
        return []

    best_score = scored_cards[0][0]
    if best_score <= 0:
        return [item[3] for item in scored_cards[: min(3, len(scored_cards))]]

    min_score = max(5, best_score - 2)
    selected_cards = [item[3] for item in scored_cards if item[0] >= min_score][:6]
    return selected_cards or [scored_cards[0][3]]


def _card_pattern_score(card: Mapping[str, Any], pattern: BusinessPatternSchema) -> int:
    card_pattern_type = _pattern_type_from_card(card)
    pattern_type = str(pattern.pattern_type or "")
    score = 0

    if card_pattern_type == pattern_type:
        score += 10
    if _pattern_family(card_pattern_type) == _pattern_family(pattern_type):
        score += 4
    if str(card.get("stage") or "") == "clinical" and pattern_type == "clinical_case_mix":
        score += 3
    if _card_requires_validation(card) and pattern.validation_needed:
        score += 3

    card_routes = _card_routes(card)
    pattern_routes = {str(route) for route in pattern.downstream_routes if str(route)}
    score += len(card_routes.intersection(pattern_routes)) * 2

    card_tokens = _descriptor_tokens(_card_descriptor_values(card))
    pattern_tokens = _descriptor_tokens(_pattern_descriptor_values(pattern))
    score += min(len(card_tokens.intersection(pattern_tokens)), 8)

    card_priority = set(_card_priority_values(card))
    pattern_priority = set(_pattern_priority_values(pattern))
    score += min(len(card_priority.intersection(pattern_priority)), 4)

    card_key_drivers = set(_collect_card_key_driver_codes([card]))
    pattern_key_drivers = {str(item) for item in pattern.key_driver_codes if str(item)}
    score += min(len(card_key_drivers.intersection(pattern_key_drivers)), 4)
    return score


def _pattern_type_from_card(card: Mapping[str, Any]) -> str:
    driver_type = str(card.get("driver_type") or "")
    canonical = _safe_dict(card.get("canonical_dimensions"))
    source_entity = _safe_dict(card.get("source_entity"))
    flags = {str(flag) for flag in card.get("flags", []) or [] if str(flag)}

    if "coding" in driver_type or "mapping" in driver_type or "auth_or_mapping_shift" in flags:
        return "coding_mapping_validation"
    if str(card.get("stage") or "") == "clinical":
        return "clinical_case_mix"
    if "mixed" in driver_type:
        return "mixed_volume_unit_cost"
    if "unit_cost" in driver_type:
        return "unit_cost_pressure"
    if "low_baseline" in driver_type:
        return "low_baseline_emergence"
    if canonical.get("facility"):
        return "site_of_care_shift"
    if canonical.get("provider") or str(source_entity.get("type") or "") == "providers":
        return "provider_network_economics"
    return "utilization_growth"


def _card_descriptor_values(card: Mapping[str, Any]) -> List[str]:
    source_entity = _safe_dict(card.get("source_entity"))
    canonical = _safe_dict(card.get("canonical_dimensions"))
    dimensions = _safe_dict(card.get("dimensions"))
    values = [
        str(card.get("stage") or ""),
        str(card.get("driver_type") or ""),
        str(source_entity.get("name") or ""),
        str(source_entity.get("type") or ""),
    ]
    values.extend(str(item) for item in card.get("business_terms", []) or [] if str(item))
    for role_values in canonical.values():
        values.extend(str(item) for item in role_values or [] if str(item))
    for dim_values in dimensions.values():
        if isinstance(dim_values, list):
            values.extend(str(item) for item in dim_values if str(item))
        elif str(dim_values):
            values.append(str(dim_values))
    return values


def _card_priority_values(card: Mapping[str, Any]) -> List[str]:
    canonical = _safe_dict(card.get("canonical_dimensions"))
    source_entity = _safe_dict(card.get("source_entity"))
    values = [
        # state and geography are both surfaced — state was split out of
        # geography so ZIP codes stop leaking into priority_entities.states;
        # priority-value use still wants both flavors of geographic context.
        *[str(item) for item in canonical.get("state", []) or [] if str(item)],
        *[str(item) for item in canonical.get("geography", []) or [] if str(item)],
        *[str(item) for item in canonical.get("provider", []) or [] if str(item)],
        *[str(item) for item in canonical.get("product", []) or [] if str(item)],
        *[str(item) for item in canonical.get("facility", []) or [] if str(item)],
        *[str(item) for item in canonical.get("clinical_category", []) or [] if str(item)],
    ]
    if str(source_entity.get("type") or "") == "providers" and str(source_entity.get("name") or "").strip():
        values.append(str(source_entity.get("name")))
    return values


def _group_ids_for_cards(
    cards: List[Mapping[str, Any]],
    card_group_ids: Mapping[str, List[str]],
) -> List[str]:
    group_ids: List[str] = []
    for card in cards:
        card_id = str(card.get("card_id") or "")
        group_ids.extend(str(group_id) for group_id in card_group_ids.get(card_id, []) if str(group_id))
    return _unique_preserve_order(group_ids)


def _build_dims_by_role_from_cards(cards: List[Mapping[str, Any]]) -> Mapping[str, Counter]:
    dims_by_role: dict[str, Counter] = defaultdict(Counter)
    for card in cards:
        canonical = _safe_dict(card.get("canonical_dimensions"))
        for role, values in canonical.items():
            for value in values or []:
                dims_by_role[str(role)][str(value)] += 1
    return dims_by_role


def _priority_entities_from_cards(cards: List[Mapping[str, Any]]) -> PatternPriorityEntitiesSchema:
    dims_by_role = _build_dims_by_role_from_cards(cards)

    # `state` role holds only actual state dimensions (provider_state_code,
    # prov_service_state, rendering_state, member_brand_state). It used to
    # be `geography`, which mixed state + zip + market + region and let
    # ZIP codes leak into pattern.priority_entities.states.
    state_counter = Counter(dims_by_role.get("state", Counter()))
    provider_counter = Counter(dims_by_role.get("provider", Counter()))

    for card in cards:
        source_entity = _safe_dict(card.get("source_entity"))
        source_type = str(source_entity.get("type") or "").lower()
        source_name = str(source_entity.get("name") or "").strip()

        if not source_name:
            continue

        if source_type in {"state", "states", "market", "markets"}:
            state_counter[source_name] += 1

        if source_type in {"provider", "providers"}:
            provider_counter[source_name] += 1

    return PatternPriorityEntitiesSchema(
        states=[value for value, _ in state_counter.most_common(5)],
        providers=[value for value, _ in provider_counter.most_common(5)],
        products=[value for value, _ in dims_by_role.get("product", Counter()).most_common(5)],
        facility_types=[value for value, _ in dims_by_role.get("facility", Counter()).most_common(5)],
        clinical_categories=[value for value, _ in dims_by_role.get("clinical_category", Counter()).most_common(5)],
    )


def _collect_card_key_driver_codes(cards: List[Mapping[str, Any]]) -> List[str]:
    ordered_roles = ["auth", "drg", "diagnosis", "facility", "product", "provider", "clinical_category"]
    values: List[str] = []
    for card in cards:
        canonical = _safe_dict(card.get("canonical_dimensions"))
        source_entity = _safe_dict(card.get("source_entity"))
        for role in ordered_roles:
            for item in canonical.get(role, [])[:2]:
                cleaned = str(item).strip()
                if cleaned and cleaned not in values:
                    values.append(cleaned)
        source_type = str(source_entity.get("type") or "")
        source_name = str(source_entity.get("name") or "").strip()
        source_role = {"drgs": "drg", "providers": "provider"}.get(source_type)
        if source_role in ordered_roles and source_name and source_name not in values:
            values.append(source_name)
    return values[:10]


def _impact_summary_from_cards(
    cards: List[Mapping[str, Any]],
    current_summary: PatternImpactSummarySchema,
) -> PatternImpactSummarySchema:
    total_delta = sum(_card_delta(card) for card in cards)
    return PatternImpactSummarySchema(
        primary_metric=str(current_summary.primary_metric or "total_paid"),
        direction=_direction_from_delta(total_delta),
        estimated_delta=_format_currency(total_delta),
        volume_signal=_volume_signal_from_cards(cards),
        unit_cost_signal=_unit_cost_signal_from_cards(cards),
    )


def _impact_summary_missing(summary: PatternImpactSummarySchema) -> bool:
    estimated_delta = str(summary.estimated_delta or "").strip()
    return not estimated_delta or not re.search(r"[1-9]", estimated_delta)


def _impact_summary_from_groups_or_cards(
    *,
    groups: List[Mapping[str, Any]],
    cards: List[Mapping[str, Any]],
    current_summary: PatternImpactSummarySchema,
) -> PatternImpactSummarySchema:
    if groups:
        total_delta = sum(
            _to_float(_safe_dict(group.get("impact")).get("total_delta")) or 0.0
            for group in groups
        )

        volume_signal = _volume_signal_from_cards(cards) if cards else _volume_signal_from_groups(groups)
        unit_cost_signal = _unit_cost_signal_from_cards(cards) if cards else _unit_cost_signal_from_groups(groups)

        return PatternImpactSummarySchema(
            primary_metric=str(current_summary.primary_metric or "total_paid"),
            direction=_direction_from_delta(total_delta),
            estimated_delta=_format_currency(total_delta),
            volume_signal=volume_signal,
            unit_cost_signal=unit_cost_signal,
        )

    return _impact_summary_from_cards(cards, current_summary)


def _volume_signal_from_groups(groups: List[Mapping[str, Any]]) -> str:
    signals = [_volume_signal_from_group(group) for group in groups]
    if any(signal == "admissions up" for signal in signals):
        return "admissions up"
    if any(signal == "claims up" for signal in signals):
        return "claims up"
    if any(signal == "down" for signal in signals):
        return "down"
    if any(signal == "flat" for signal in signals):
        return "flat"
    if any(signal == "not material" for signal in signals):
        return "not material"
    return "unknown"


def _unit_cost_signal_from_groups(groups: List[Mapping[str, Any]]) -> str:
    signals = [_unit_cost_signal_from_group(group) for group in groups]
    if any(signal == "paid per admit up" for signal in signals):
        return "paid per admit up"
    if any(signal == "allowed per admit up" for signal in signals):
        return "allowed per admit up"
    if any(signal == "mixed" for signal in signals):
        return "mixed"
    if any(signal == "down" for signal in signals):
        return "down"
    if any(signal == "flat" for signal in signals):
        return "flat"
    return "unknown"


def _priority_entities_from_groups(groups: List[Mapping[str, Any]]) -> PatternPriorityEntitiesSchema:
    states = Counter()
    providers = Counter()
    products = Counter()
    facility_types = Counter()
    clinical_categories = Counter()

    for group in groups:
        impacted_entities = _safe_dict(group.get("impacted_entities"))
        top_dimensions = _safe_dict(group.get("top_dimensions"))

        for value in impacted_entities.get("states", []) or []:
            states[str(value)] += 1
        # `state` role instead of the broader `geography` bucket — see
        # note in _priority_entities_from_cards for why this matters.
        for value in top_dimensions.get("state", []) or []:
            states[str(value)] += 1

        for value in impacted_entities.get("providers", []) or []:
            providers[str(value)] += 1
        for value in top_dimensions.get("provider", []) or []:
            providers[str(value)] += 1

        for value in top_dimensions.get("product", []) or []:
            products[str(value)] += 1
        for value in top_dimensions.get("facility", []) or []:
            facility_types[str(value)] += 1
        for value in top_dimensions.get("clinical_category", []) or []:
            clinical_categories[str(value)] += 1

    return PatternPriorityEntitiesSchema(
        states=[value for value, _ in states.most_common(5)],
        providers=[value for value, _ in providers.most_common(5)],
        products=[value for value, _ in products.most_common(5)],
        facility_types=[value for value, _ in facility_types.most_common(5)],
        clinical_categories=[value for value, _ in clinical_categories.most_common(5)],
    )


def _merge_priority_entities(
    current: PatternPriorityEntitiesSchema,
    *derived_items: PatternPriorityEntitiesSchema,
) -> PatternPriorityEntitiesSchema:
    def merge_values(existing: List[str], derived: Iterable[str]) -> List[str]:
        return _unique_preserve_order([*existing, *[str(item) for item in derived if str(item)]])[:5]

    states = list(current.states)
    providers = list(current.providers)
    products = list(current.products)
    facility_types = list(current.facility_types)
    clinical_categories = list(current.clinical_categories)

    for derived in derived_items:
        states = merge_values(states, derived.states)
        providers = merge_values(providers, derived.providers)
        products = merge_values(products, derived.products)
        facility_types = merge_values(facility_types, derived.facility_types)
        clinical_categories = merge_values(clinical_categories, derived.clinical_categories)

    return PatternPriorityEntitiesSchema(
        states=states,
        providers=providers,
        products=products,
        facility_types=facility_types,
        clinical_categories=clinical_categories,
    )


def _downstream_routes_from_cards(
    cards: List[Mapping[str, Any]],
    groups: List[Mapping[str, Any]],
    existing_routes: List[str],
) -> List[str]:
    routes = list(existing_routes)
    for card in cards:
        routes.extend(_card_routes(card))
    if not routes:
        for group in groups:
            routes.extend(str(route) for route in group.get("downstream_routes", []) or [] if str(route))
    return _unique_preserve_order(routes)


def _build_evidence_summary_from_cards(cards: List[Mapping[str, Any]]) -> List[str]:
    bullets: List[str] = []
    ranked_cards = sorted(cards, key=lambda card: abs(_card_delta(card)), reverse=True)
    for card in ranked_cards[:4]:
        delta = _format_currency(_card_delta(card))
        source_entity = _safe_dict(card.get("source_entity"))
        dimensions = _safe_dict(card.get("dimensions"))
        dimension_text = ", ".join(f"{key}={value}" for key, value in list(dimensions.items())[:2])
        entity_text = str(source_entity.get("name") or "")
        prefix = f"{entity_text} " if entity_text else ""
        bullets.append(
            f"{prefix}{str(card.get('stage') or 'Unknown').title()} {str(card.get('driver_type') or 'movement').replace('_', ' ')} contributed about {delta}{f' in {dimension_text}' if dimension_text else ''}."
        )
    return bullets


def _card_requires_validation(card: Mapping[str, Any]) -> bool:
    flags = {str(flag) for flag in card.get("flags", []) or [] if str(flag)}
    driver_type = str(card.get("driver_type") or "")
    card_type = str(card.get("card_type") or "")

    return (
        card_type == "drill_validation_summary"
        or "coding" in driver_type
        or "mapping" in driver_type
        or "requires_validation_before_action" in flags
        or "auth_or_mapping_shift" in flags
        or "unmapped_or_unknown_value" in flags
    )


def _card_routes(card: Mapping[str, Any]) -> set[str]:
    return {
        str(route)
        for route, enabled in (_safe_dict(card.get("downstream_routes"))).items()
        if enabled and str(route)
    }


def _card_delta(card: Mapping[str, Any]) -> float:
    metrics = _safe_dict(card.get("metrics"))
    value = _safe_dict(metrics.get("value"))
    return _to_float(value.get("delta")) or 0.0


def _volume_signal_from_cards(cards: List[Mapping[str, Any]]) -> str:
    admissions_deltas = [_metric_delta(_safe_dict(card.get("metrics")), "admissions") for card in cards]
    admissions_values = [delta for delta in admissions_deltas if delta is not None]
    if admissions_values:
        total_admissions = sum(admissions_values)
        if total_admissions > 0:
            return "admissions up"
        if total_admissions < 0:
            return "down"
        return "flat"

    claim_deltas = [_metric_delta(_safe_dict(card.get("metrics")), "claim_count") for card in cards]
    claim_values = [delta for delta in claim_deltas if delta is not None]
    if claim_values:
        total_claims = sum(claim_values)
        if total_claims > 0:
            return "claims up"
        if total_claims < 0:
            return "down"
        return "flat"
    return "unknown"


def _unit_cost_signal_from_cards(cards: List[Mapping[str, Any]]) -> str:
    paid_deltas = [_metric_delta(_safe_dict(card.get("metrics")), "avg_paid_per_admit") for card in cards]
    paid_values = [delta for delta in paid_deltas if delta is not None]
    if paid_values:
        if all(delta > 0 for delta in paid_values):
            return "paid per admit up"
        if all(delta < 0 for delta in paid_values):
            return "down"
        if all(delta == 0 for delta in paid_values):
            return "flat"
        return "mixed"

    allowed_deltas = [_metric_delta(_safe_dict(card.get("metrics")), "avg_allowed_per_admit") for card in cards]
    allowed_values = [delta for delta in allowed_deltas if delta is not None]
    if allowed_values:
        if all(delta > 0 for delta in allowed_values):
            return "allowed per admit up"
        if all(delta < 0 for delta in allowed_values):
            return "down"
        if all(delta == 0 for delta in allowed_values):
            return "flat"
        return "mixed"
    return "unknown"


def _group_requires_validation(group: Mapping[str, Any]) -> bool:
    return _pattern_type_from_group(group) == "coding_mapping_validation" or "requires_validation_before_action" in {
        str(flag) for flag in group.get("flags", []) or [] if str(flag)
    }


def _collect_key_driver_codes(group: Mapping[str, Any]) -> List[str]:
    top_dimensions = _safe_dict(group.get("top_dimensions"))
    ordered_roles = ["auth", "drg", "diagnosis", "facility", "product", "provider", "clinical_category"]
    values: List[str] = []
    for role in ordered_roles:
        for item in top_dimensions.get(role, [])[:2]:
            cleaned = str(item).strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values[:6]


def _direction_from_delta(delta: Optional[float]) -> Literal["increase", "decrease", "mixed"]:
    if delta is None:
        return "mixed"
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "mixed"


def _format_currency(value: Optional[float]) -> str:
    if value is None:
        return "$0"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{sign}${absolute / 1_000_000:.1f}M"
    if absolute >= 1_000:
        rounded = int(round(absolute / 1_000.0) * 1_000)
        return f"{sign}${rounded:,.0f}"
    return f"{sign}${absolute:,.0f}"


def _volume_signal_from_group(group: Mapping[str, Any]) -> str:
    driver_group = str(group.get("driver_group") or "")
    if driver_group in {"utilization", "mixed_volume_unit_cost"}:
        return "admissions up"
    if driver_group == "low_baseline_emergence":
        return "not material"
    return "unknown"


def _unit_cost_signal_from_group(group: Mapping[str, Any]) -> str:
    driver_group = str(group.get("driver_group") or "")
    if driver_group in {"unit_cost", "mixed_volume_unit_cost"}:
        return "paid per admit up"
    return "unknown"


def _build_why_it_matters(group: Mapping[str, Any]) -> str:
    delta = _format_currency(_to_float(group.get("impact", {}).get("total_delta")))
    stage = str(group.get("stage") or "business")
    return f"This {stage} pattern concentrates approximately {delta} of movement into a focused business segment."


def _build_next_step(pattern_type: str) -> str:
    mapping = {
        "coding_mapping_validation": "Validate coding and mapping movement before assigning operational ownership.",
        "unit_cost_pressure": "Review reimbursement, pricing, and site-of-care drivers tied to the affected segment.",
        "mixed_volume_unit_cost": "Review volume growth and reimbursement together before setting the intervention plan.",
        "clinical_case_mix": "Route the pattern to clinical review for focused case-mix validation.",
        "site_of_care_shift": "Assess whether care can be redirected to a lower-cost site when clinically appropriate.",
        "provider_network_economics": "Review provider-level economics and contract position for the named entities.",
        "low_baseline_emergence": "Confirm whether the low-baseline segment is becoming material enough to require a targeted response.",
        "utilization_growth": "Review utilization management levers for the affected segment and priority entities.",
    }
    return mapping.get(pattern_type, "Review the highest-impact business lever tied to this pattern.")


def _build_evidence_summary(group: Mapping[str, Any]) -> List[str]:
    bullets: List[str] = []
    impact = _safe_dict(group.get("impact"))
    if impact:
        bullets.append(
            f"Grouped impact is {_format_currency(_to_float(impact.get('total_delta')))} across {int(impact.get('card_count') or 0)} supporting cards."
        )
    for card in group.get("top_cards", [])[:3]:
        if not isinstance(card, dict):
            continue
        delta = _format_currency(_to_float(card.get("delta")))
        dimensions = _safe_dict(card.get("dimensions"))
        dimension_text = ", ".join(f"{key}={value}" for key, value in list(dimensions.items())[:2])
        bullets.append(
            f"{card.get('stage', 'Unknown').title()} driver {card.get('driver_type', 'unknown')} contributed about {delta}{f' in {dimension_text}' if dimension_text else ''}."
        )
    return bullets[:4]


def _build_headline(patterns: List[BusinessPatternSchema]) -> str:
    if not patterns:
        return "No material business patterns were identified."
    first = patterns[0]
    return f"Top pattern: {first.top_pattern}."


def _build_primary_message(patterns: List[BusinessPatternSchema]) -> str:
    if not patterns:
        return "No grouped analytical patterns were available for executive synthesis."
    snippets = [pattern.why_it_matters for pattern in patterns[:2] if pattern.why_it_matters]
    return " ".join(snippets)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return normalized if normalized > 0 else default


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None



def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    items: List[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return items


def build_app(**kwargs: Any) -> PatternAgent:
    return PatternAgent(**kwargs)


__all__ = [
    "BusinessPatternSchema",
    "PatternAgent",
     "PatternAgentApiResponse",
     "PatternOutputSchema",
     "PatternPriorityEntitiesSchema",
     "PatternQualityChecksSchema",
     "PatternSynthesisSchema",
     "build_app",
]
