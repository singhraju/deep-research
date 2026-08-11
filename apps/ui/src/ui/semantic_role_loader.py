"""Streamlit-cached loader for dimension semantic-role mappings.

Reads the companion JSON file emitted by
:func:`deep_research_utils.semantic_role_classifier.classify_dimension_semantic_roles`
alongside a semantic-view YAML. When the JSON is missing or its
``source_hash`` no longer matches the current YAML's drill_dimensions block,
falls back to invoking the LLM classifier on the fly.

Design contract with the UI (see plan LLM-driven semantic role classifier):
* Companion JSONs are committed to git — cold startup should be a pure file
  read with zero LLM calls in the common case.
* Any regeneration path emits a visible warning through the diagnostics
  sidebar so YAML/JSON drift is caught in review.
* Same cache lifecycle as ``load_all_filter_values`` — the "Refresh filter
  values" button also clears this cache.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from deep_research_utils.semantic_role_classifier import (
    classify_dimension_semantic_roles,
    companion_json_path,
    load_companion_json,
    source_hash_for_yaml,
)
from ui.semantic_schema import SemanticSchema

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


RoleSource = str  # "companion_json", "regenerated", "unavailable"


@dataclass(frozen=True)
class SemanticRoles:
    """Loaded semantic-role snapshot for the active YAML/analysis mode."""

    dimension_roles: Dict[str, str]
    view_name: Optional[str]
    analysis_mode: Optional[str]
    generated_at: Optional[str]
    llm_model: Optional[str]
    source_hash: Optional[str]
    source: RoleSource
    warning: Optional[str] = None

    def role_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for role in self.dimension_roles.values():
            counts[role] = counts.get(role, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def dims_flagged_other(self) -> list[str]:
        return sorted(
            name for name, role in self.dimension_roles.items() if role == "other"
        )


def _build_classifier_llm() -> Any:
    """Build the LLM used by the classifier.

    The classifier endpoint may differ from the main-app LLM (the plan calls
    out a dedicated EHAP endpoint that permits models outside the default
    allowlist). ``SEMANTIC_ROLE_LLM_MODEL`` env var overrides which model
    ``build_llm`` provisions; if unset, the standard app model is used.
    """

    from deep_research_agents.user_intent import build_llm

    override = os.environ.get("SEMANTIC_ROLE_LLM_MODEL")
    llm, _ehap = build_llm(model_name=override) if override else build_llm()
    return llm


@st.cache_data(ttl=3600, show_spinner="Loading semantic role mappings...")
def load_semantic_roles(
    schema: SemanticSchema,
    force_regenerate: bool = False,
) -> SemanticRoles:
    """Return the semantic-role snapshot for the active YAML/analysis mode.

    Resolution order:
      1. If a companion JSON exists at the canonical path AND its
         ``source_hash`` matches the current YAML's drill_dimensions hash
         AND ``force_regenerate`` is False -> load from disk. No LLM call.
      2. Otherwise call :func:`classify_dimension_semantic_roles`, write the
         companion JSON, and return the fresh mapping. The returned
         ``warning`` field carries a message the diagnostics panel surfaces.
      3. If regeneration fails for any reason, return an empty mapping with
         ``source="unavailable"`` and a warning describing the failure.
         Downstream agents must treat missing roles as "fall back to
         legacy heuristics".
    """

    yaml_path = Path(schema.yaml_path)
    analysis_mode = schema.analysis_mode_name
    target = companion_json_path(yaml_path, analysis_mode)

    try:
        current_hash = source_hash_for_yaml(yaml_path, analysis_mode)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not compute source_hash for %s: %s", yaml_path, exc)
        return SemanticRoles(
            dimension_roles={},
            view_name=schema.view_name,
            analysis_mode=analysis_mode,
            generated_at=None,
            llm_model=None,
            source_hash=None,
            source="unavailable",
            warning=f"source_hash computation failed: {exc}",
        )

    cached = None if force_regenerate else load_companion_json(target)
    if cached is not None:
        cached_hash = str(cached.get("source_hash") or "")
        if cached_hash == current_hash:
            return SemanticRoles(
                dimension_roles=dict(cached.get("dimension_roles") or {}),
                view_name=cached.get("view_name") or schema.view_name,
                analysis_mode=cached.get("analysis_mode") or analysis_mode,
                generated_at=cached.get("generated_at"),
                llm_model=cached.get("llm_model"),
                source_hash=cached_hash,
                source="companion_json",
            )
        logger.warning(
            "Companion JSON at %s has stale source_hash (was %s, now %s); regenerating via LLM.",
            target,
            cached_hash,
            current_hash,
        )

    reason = (
        "forced regeneration"
        if force_regenerate
        else ("stale companion JSON" if cached is not None else "missing companion JSON")
    )

    try:
        llm = _build_classifier_llm()
        payload = classify_dimension_semantic_roles(
            yaml_path,
            scope_to_analysis_mode=analysis_mode,
            llm=llm,
        )
    except Exception as exc:
        logger.error(
            "Semantic role classification failed for %s (mode=%s): %s",
            yaml_path,
            analysis_mode,
            exc,
        )
        return SemanticRoles(
            dimension_roles={},
            view_name=schema.view_name,
            analysis_mode=analysis_mode,
            generated_at=None,
            llm_model=None,
            source_hash=current_hash,
            source="unavailable",
            warning=(
                f"Semantic role classification failed ({reason}): {exc}. "
                "Agents will fall back to legacy keyword/regex inference."
            ),
        )

    warning = (
        f"Regenerated semantic roles via LLM ({reason}). Commit the updated "
        f"{target.name} so future sessions skip the LLM call."
    )
    return SemanticRoles(
        dimension_roles=dict(payload.get("dimension_roles") or {}),
        view_name=payload.get("view_name") or schema.view_name,
        analysis_mode=payload.get("analysis_mode") or analysis_mode,
        generated_at=payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        llm_model=payload.get("llm_model"),
        source_hash=payload.get("source_hash") or current_hash,
        source="regenerated",
        warning=warning,
    )
