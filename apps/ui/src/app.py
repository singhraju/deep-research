"""Streamlit UI for the deep-research orchestrator."""

from __future__ import annotations

import datetime as dt
import logging
import os
import traceback
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

import urllib.parse

import streamlit as st
from deep_research_utils.app_constant import AppConstants
from deep_research_utils.snowflake_helper import SnowparkHelper

logger = logging.getLogger(__name__)

from ui.correlation_visuals import render_correlation_visuals, format_period_window
from ui.pattern_visuals import (
    render_drill_down_paths,
    render_pattern_storyboard,
    render_payer_summary_table,
    render_policy_links_expander,
    render_recommended_actions,
)
from ui import pdf_report
from ui.filter_options import (
    build_conversation_id,
    build_prebuilt_intent,
    serialize_live_filter_values,
)
from ui.filter_values_loader import DimensionValues, FilterValues, TimeDimensionRange, load_all_filter_values
from ui.semantic_role_loader import SemanticRoles, load_semantic_roles
from ui.semantic_schema import SemanticSchema, load_semantic_schema
from ui.theme import inject_global_theme

_ORCHESTRATOR_IMPORT_ERROR: Optional[str] = None
OrchestratorConfig = None  # type: ignore[assignment]
build_orchestrator = None  # type: ignore[assignment]
try:
    from ui.orchestrator_client import OrchestratorConfig, build_orchestrator  # type: ignore[no-redef]
except Exception as _import_exc:  # ModuleNotFoundError, ImportError, etc.
    _ORCHESTRATOR_IMPORT_ERROR = f"{type(_import_exc).__name__}: {_import_exc}"
    logger.error(
        "Failed to import orchestrator_client at startup — chat will be disabled. %s",
        _ORCHESTRATOR_IMPORT_ERROR,
    )


def _agent_availability() -> tuple[List[str], Dict[str, str]]:
    """Return (available, missing) agents — empty lists if the package itself failed to import."""
    try:
        from deep_research_agents import available_agents, missing_agents  # type: ignore[no-redef]
        return available_agents(), missing_agents()
    except Exception as exc:
        return [], {"deep_research_agents": f"{type(exc).__name__}: {exc}"}

APP_ROOT = Path(__file__).resolve().parents[3]

PERIOD_OPTIONS: List[str] = ["Rolling 3", "Rolling 6", "Rolling 12", "YTD"]
DEFAULT_PERIOD = "Rolling 3"

_FRIENDLY_STAGE_LABELS: Dict[str, str] = {
    "prepare_request": "Reading your question",
    "run_intent_detection": "Understanding what to analyze",
    "decide_next_step": "Choosing the right analysis",
    "clarification": "Asking for clarification",
    "analysis::generic": "Running analysis",
    "analysis::cost_change_investigation_over_time_window": "Investigating cost changes",
    "run_hypothesis_agents": "Forming hypotheses",
    "hypotheses": "Forming hypotheses",
    "hypotheses::skipped": "Skipping hypotheses (not applicable)",
    "pattern_agent": "Finding business patterns",
    "pattern_agent::skipped": "Skipping patterns (not applicable)",
    "pattern_agent::error": "Pattern step ran into an issue",
    "reimbursement": "Comparing payment patterns",
    "reimbursement::skipped": "Skipping payment comparison",
    "recommendation": "Building recommendations",
    "recommendation::skipped": "Skipping recommendations",
    "recommendation::error": "Recommendation step ran into an issue",
    "build_report_contract": "Drafting the report",
    "build_visual_contract": "Preparing charts",
    "build_summary_contract": "Writing the summary",
    "finalize": "Wrapping up",
}


def _friendly_stage_label(stage: str) -> str:
    if stage in _FRIENDLY_STAGE_LABELS:
        return _FRIENDLY_STAGE_LABELS[stage]
    if stage.startswith("analysis::"):
        return "Running analysis"
    return stage.replace("::", " ").replace("_", " ").strip().capitalize()


# @st.cache_data(ttl=3600)
def _load_lob_filter_values() -> List[str]:
    """Environment-aware function to load LOB values using Semantic View Approach"""
    database = AppConstants.SNOWFLAKE_DATABASE
    schema = 'COC_DTI_STG'
    table = 'WORK_ELEVATE_COC_CLAIM_DETAIL'

    # Construct environment-aware SQL query using semantic view configuration
    table_name = f"{database}.{schema}.{table}"
    query = f"SELECT DISTINCT LOB_SHRT_DESC FROM {table_name} WHERE LOB_SHRT_DESC IS NOT NULL ORDER BY LOB_SHRT_DESC"
    logger.info(f"Executing semantic view-based query: {query}")
    print(f"Executing semantic view-based query: {query}")

    try:
        sf = SnowparkHelper()
        
        result = sf.session.sql(query).collect()
        
        # Extract values from result
        lob_values = [row[0] for row in result]
        logger.info(f"LOB filter values loaded from db: {lob_values}")
        print(f"LOB filter values loaded from db: {lob_values}")
        return lob_values
        
    except Exception as e:
        logger.error(f"Failed to load LOB values from database: {e}")
        print(f"Failed to load LOB values from database: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Fallback to empty list
        return []


def _resolve_semantic_config() -> tuple[Path, Optional[str]]:
    """Read configs/{ENVIRONMENT}.ini and resolve the active semantic YAML.

    Returns ``(yaml_path, error_message)``. ``error_message`` is None on success.
    """

    try:
        config = AppConstants.load_agent_config()
        semantic_config_path = config.get("correlation", "semantic_config_path")
    except FileNotFoundError as exc:
        return APP_ROOT / "configs" / "local" / "missing.yaml", (
            f"Environment config file not found for ENVIRONMENT='{AppConstants.ENV}': {exc}"
        )
    except Exception as exc:  # pragma: no cover - configparser edge cases
        return APP_ROOT / "configs" / "local" / "missing.yaml", (
            f"Failed to read [correlation].semantic_config_path from configs/{AppConstants.ENV}.ini: {exc}"
        )
    return APP_ROOT / semantic_config_path, None


DEFAULT_MODEL_PATH, _ENV_LOAD_ERROR = _resolve_semantic_config()
DEFAULT_OUTPUT_ROOT = Path(AppConstants.CORRELATION_OUTPUT_ROOT)

# Set SSL_CERT_FILE environment variable for the application
os.environ["SSL_CERT_FILE"] = str(APP_ROOT / "cacert.pem")

ASSETS_DIR = Path(__file__).parent / "assets"
_ICON_SVG = (ASSETS_DIR / "icon.svg").read_text(encoding="utf-8")
_AGENTS_LOADER_SVG = (ASSETS_DIR / "agents_loader.svg").read_text(encoding="utf-8")


_AGENTS_LOADER_WIDTH_PX = 220
# SVG viewBox is 120 × 80 → natural rendered height at width=W is W * 80/120.
_AGENTS_LOADER_HEIGHT_PX = int(_AGENTS_LOADER_WIDTH_PX * 80 / 120)
# Iframe needs a small buffer beyond the SVG height so the flex center has
# room to breathe and nothing clips at sub-pixel rounding.
_AGENTS_LOADER_IFRAME_HEIGHT = _AGENTS_LOADER_HEIGHT_PX + 24


def _agents_loader_html() -> str:
    """Standalone HTML for ``st.iframe`` using data URI.

    The iframe is sized to the SVG's natural height so the trend-line and
    arrowhead aren't cropped — early versions hard-coded 110px which clipped
    the top of the recovery arrow for a 240×160 chart.
    """
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden;}'
        f'.wrap{{display:flex;justify-content:center;align-items:center;'
        f'width:100%;min-height:{_AGENTS_LOADER_IFRAME_HEIGHT}px;}}'
        f'.wrap svg{{width:{_AGENTS_LOADER_WIDTH_PX}px;height:auto;display:block;}}'
        '</style></head>'
        f'<body><div class="wrap">{_AGENTS_LOADER_SVG}</div></body></html>'
    )


def _header_icon_html() -> str:
    sized = _ICON_SVG.replace(
        "<svg ",
        '<svg style="width:44px;height:44px;flex:0 0 auto;" ',
        1,
    )
    return (
        '<div style="display:flex;align-items:center;gap:12px;margin:0 0 4px 0;">'
        f'{sized}'
        '<h1 style="margin:0;font-weight:800;line-height:1.1;">Deep Research</h1>'
        '</div>'
    )


def main() -> None:
    """Render the Deep Research Streamlit interface."""

    st.set_page_config(
        page_title="Deep Research",
        page_icon=_ICON_SVG,
        layout="wide",
    )
    inject_global_theme()
    _init_session_state()

    st.markdown(_header_icon_html(), unsafe_allow_html=True)
    st.caption("Ask healthcare analytics questions.")

    if _ENV_LOAD_ERROR:
        st.error(_ENV_LOAD_ERROR)
        return

    try:
        schema = load_semantic_schema(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Semantic YAML not found at: {DEFAULT_MODEL_PATH}")
        return
    except Exception as exc:
        st.error(f"Failed to parse semantic YAML: {exc}")
        return

    filter_values = load_all_filter_values(schema)
    selections = _render_filters(schema, filter_values)

    # Snapshot live filter values for the orchestrator's validator. Cached for
    # the session by ``load_all_filter_values``; refreshing the cache (sidebar
    # "Refresh filter values" button) updates this snapshot too.
    st.session_state.live_filter_values = serialize_live_filter_values(filter_values)

    force_regenerate_roles = bool(st.session_state.pop("_force_regenerate_semantic_roles", False))
    semantic_roles = load_semantic_roles(schema, force_regenerate=force_regenerate_roles)
    st.session_state.semantic_roles = semantic_roles.dimension_roles
    st.session_state.semantic_roles_meta = semantic_roles

    runtime_settings = {
        "model_path": str(DEFAULT_MODEL_PATH),
        "enable_llm": True,
        "enable_snowflake": True,
        "schema_view_name": schema.view_name,
        "principal_time_dimension": schema.principal_time_dimension,
    }
    st.session_state.runtime_settings = runtime_settings

    _render_diagnostics_sidebar(runtime_settings, schema, filter_values, semantic_roles)
    _render_chat_interface(selections, runtime_settings)
    _render_latest_output()


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", f"ui-{uuid.uuid4().hex}")
    st.session_state.setdefault("last_output", None)
    st.session_state.setdefault("orchestrator_error", None)


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


_PRIMARY_DIM_NAMES = {"lob_description"}  # LOB loaded dynamically from DB

_DIM_CATEGORIES: Dict[str, str] = {
    "service_area_state": "Geography",
    "rendering_provider_name": "Geography",
    "facility_type": "Geography",
    "drg_name": "Clinical",
    "primary_diagnosis_name": "Clinical",
    "hcc_medium": "Clinical",
    "product_description": "Clinical",
    "pa_required_code": "Utilization",
    "er_admit_indicator": "Utilization",
}
_CATEGORY_ORDER = ["Geography", "Clinical", "Utilization", "Other"]


def _categorize_dimension(name: str) -> str:
    return _DIM_CATEGORIES.get(name, "Other")


def _format_filter_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
        if len(items) <= 2:
            return ", ".join(items)
        return f"{items[0]}, {items[1]} +{len(items) - 2}"
    return str(value)


def _build_narrative(
    metric_value: str,
    period_value: str,
    metric_descriptions: Dict[str, str],
    dimension_selections: Dict[str, Any],
) -> str:
    metric_short = metric_value.split(".")[-1].replace("_", " ") if metric_value else "metric"
    parts = [f"**{metric_short.title()}** over **{period_value}**"]
    lob = dimension_selections.get("lob_description")
    if lob:
        parts.append(f"for **{_format_filter_value(lob)}**")
    other = [
        f"{_humanize(name)}={_format_filter_value(value)}"
        for name, value in dimension_selections.items()
        if name != "lob_description"
    ]
    if other:
        parts.append("filtered by " + ", ".join(other))
    return " ".join(parts) + "."


def _render_filters(schema: SemanticSchema, values: FilterValues) -> Dict[str, Any]:
    """Narrative header + 3-chip primary strip + popover for the rest of the dimensions.

    Preserves the legacy return-dict shape so the orchestrator contract is
    untouched: ``metric_name``, ``period_label``, ``dimension_selections``,
    ``time_selections``, ``principal_time_dimension``,
    ``principal_time_dimension_qualified``, ``analysis_mode_name``,
    ``view_name``.
    """

    metric_choices = [m.name for m in schema.metrics] or [schema.default_drill_metric or ""]
    metric_descriptions = {m.name: m.description for m in schema.metrics}
    default_metric_index = (
        metric_choices.index(schema.default_drill_metric)
        if schema.default_drill_metric in metric_choices
        else 0
    )

    time_ranges_by_name = {tr.dimension_name: tr for tr in values.time_dimensions}
    principal_range = time_ranges_by_name.get(schema.principal_time_dimension or "")
    if principal_range is None and values.time_dimensions:
        principal_range = values.time_dimensions[0]
    snap_range = time_ranges_by_name.get("snap_month")

    dim_value_map: Dict[str, DimensionValues] = {dv.dimension_name: dv for dv in values.dimensions}
    schema_dims = [d for d in schema.dimensions if d.name in dim_value_map]
    primary_dims = [d for d in schema_dims if d.name in _PRIMARY_DIM_NAMES]
    secondary_dims = [d for d in schema_dims if d.name not in _PRIMARY_DIM_NAMES]

    dimension_selections: Dict[str, Any] = {}
    time_selections: Dict[str, Any] = {}

    with st.container(border=True):
        st.markdown("##### Scope")

        primary_cols = st.columns([1.4, 1, 1, 1.2, 1.2])
        with primary_cols[0]:
            metric_value = st.selectbox(
                "Metric",
                options=metric_choices,
                index=default_metric_index,
                help="The metric whose change the orchestrator should explain.",
                format_func=lambda name: f"{name} — {metric_descriptions.get(name, '')}".rstrip(" —"),
                key="filter_metric",
            )
        with primary_cols[1]:
            period_value = st.selectbox(
                "Period",
                options=PERIOD_OPTIONS,
                index=PERIOD_OPTIONS.index(DEFAULT_PERIOD),
                help="Rolling window (R3/R6/R12) or YTD anchor for the current-vs-prior-year comparison.",
                key="filter_period",
            )
        with primary_cols[2]:
            if snap_range is not None:
                snap_value = _render_time_widget(snap_range)
                if snap_value is not None:
                    time_selections["snap_month"] = snap_value
            else:
                st.caption("snap_month not in schema")
        with primary_cols[3]:
            if principal_range is not None:
                end_value = _render_time_widget(principal_range)
                if end_value is not None:
                    time_selections[principal_range.dimension_name] = end_value
        with primary_cols[4]:
            # LOB Filter - Simple direct database query
            lob_values = _load_lob_filter_values()
            lob_options = ["All"] + sorted(lob_values)
            
            lob_value = st.selectbox(
                "Line of Business",
                options=lob_options,
                index=0,  # Default to "All"
                help="Select the Line of Business to filter by. Values loaded directly from database.",
                key="filter_lob",
            )
            if lob_value and lob_value != "All":
                dimension_selections["lob_description"] = lob_value

        # Extra time dims (beyond snap + principal) — keep them rendered, but as compact captions in the same row.
        rendered_time_names = {"snap_month"} | (
            {principal_range.dimension_name} if principal_range else set()
        )
        extra_time_dims = [t for t in values.time_dimensions if t.dimension_name not in rendered_time_names]
        if extra_time_dims:
            extra_cols = st.columns(min(4, len(extra_time_dims)))
            for col, extra in zip(extra_cols, extra_time_dims):
                with col:
                    value = _render_time_widget(extra)
                    if value is not None:
                        time_selections[extra.dimension_name] = value

        if secondary_dims:
            categorized: Dict[str, List[Any]] = {category: [] for category in _CATEGORY_ORDER}
            for dim in secondary_dims:
                categorized[_categorize_dimension(dim.name)].append(dim)
            active_categories = [c for c in _CATEGORY_ORDER if categorized[c]]

            # Count currently-active secondary filters by reading session_state
            # (the multiselects below write to keys we control).
            active_count = sum(
                1 for d in secondary_dims if st.session_state.get(f"flt_{d.name}")
            )
            popover_label = (
                f"More filters · {active_count} active" if active_count else "More filters"
            )
            with st.popover(popover_label, width="content"):
                if len(active_categories) > 1:
                    tabs = st.tabs(active_categories)
                    for tab, category in zip(tabs, active_categories):
                        with tab:
                            for dim in categorized[category]:
                                dv = dim_value_map[dim.name]
                                selected = _render_dimension_widget(
                                    dim_name=dim.name,
                                    dim_description=dim.description,
                                    dv=dv,
                                    key=f"flt_{dim.name}",
                                )
                                if selected is not None:
                                    dimension_selections[dim.name] = selected
                elif active_categories:
                    for dim in categorized[active_categories[0]]:
                        dv = dim_value_map[dim.name]
                        selected = _render_dimension_widget(
                            dim_name=dim.name,
                            dim_description=dim.description,
                            dv=dv,
                            key=f"flt_{dim.name}",
                        )
                        if selected is not None:
                            dimension_selections[dim.name] = selected

        narrative = _build_narrative(metric_value, period_value, metric_descriptions, dimension_selections)
        st.caption(narrative)

    return {
        "metric_name": metric_value,
        "period_label": period_value,
        "dimension_selections": dimension_selections,
        "time_selections": time_selections,
        "principal_time_dimension": schema.principal_time_dimension,
        "principal_time_dimension_qualified": schema.principal_time_dimension_qualified,
        "principal_time_range": principal_range,
        "analysis_mode_name": schema.analysis_mode_name,
        "view_name": schema.view_name,
    }


def _render_dimension_widget(
    *,
    dim_name: str,
    dim_description: str,
    dv: DimensionValues,
    key: Optional[str] = None,
) -> Optional[Any]:
    label = _humanize(dim_name)
    help_text = dim_description or None

    if dv.is_free_text:
        text_value = st.text_input(
            label,
            value="",
            placeholder=dim_description[:80] if dim_description else "Type to filter…",
            help=help_text,
            key=key,
        )
        return text_value or None

    chosen = st.multiselect(
        label,
        options=list(dv.values),
        default=[],
        help=help_text,
        placeholder="All",
        key=key,
    )
    return chosen or None


def _render_time_widget(time_range: TimeDimensionRange) -> Optional[Any]:
    adapter = time_range.adapter
    label = _humanize(time_range.dimension_name)
    help_text = "Anchor month/date for the rolling comparison."

    if adapter.widget_kind == "month":
        try:
            months = adapter.enumerate_range(adapter.serialize(time_range.min_dt), adapter.serialize(time_range.max_dt))
        except Exception:
            months = []
        if not months:
            months = [time_range.max_dt]
        labels = [adapter.format_for_display(m) for m in months]
        default_index = len(labels) - 1
        choice = st.selectbox(label, options=labels, index=default_index, help=help_text)
        chosen_dt = months[labels.index(choice)]
        return adapter.serialize(chosen_dt)

    if adapter.widget_kind == "date":
        default_date = dt.date(time_range.max_dt.year, time_range.max_dt.month, time_range.max_dt.day)
        chosen_date = st.date_input(
            label,
            value=default_date,
            min_value=dt.date(time_range.min_dt.year, time_range.min_dt.month, time_range.min_dt.day),
            max_value=default_date,
            help=help_text,
        )
        if isinstance(chosen_date, (list, tuple)):
            chosen_date = chosen_date[0]
        return adapter.serialize(dt.datetime(chosen_date.year, chosen_date.month, chosen_date.day))

    if adapter.widget_kind in {"datetime", "datetime_tz"}:
        date_col, time_col = st.columns([2, 1])
        with date_col:
            chosen_date = st.date_input(
                f"{label} (date)",
                value=time_range.max_dt.date(),
                min_value=time_range.min_dt.date(),
                max_value=time_range.max_dt.date(),
                help=help_text,
            )
        with time_col:
            chosen_time = st.time_input(
                f"{label} (time)",
                value=time_range.max_dt.time(),
            )
        combined = dt.datetime.combine(chosen_date, chosen_time)
        if adapter.widget_kind == "datetime_tz" and time_range.max_dt.tzinfo is not None:
            combined = combined.replace(tzinfo=time_range.max_dt.tzinfo)
        return adapter.serialize(combined)

    typed = st.text_input(label, value="", help=help_text)
    return typed or None


DEFAULT_RESEARCH_QUERY = "Where did change happen?"

# Chat UI is hidden for now — the page is driven by filter selections + the
# Research button. The orchestrator wiring (session_state.messages, thread_id,
# etc.) stays intact so this can flip back to True later without rework.
_CHAT_ENABLED = False


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_research_payload(selections: Dict[str, Any], raw_question: str) -> Dict[str, Any]:
    """Translate UI selections into the orchestrator's prebuilt-intent payload."""

    time_selections: Dict[str, Any] = selections.get("time_selections") or {}
    principal_name = selections.get("principal_time_dimension")
    principal_range = selections.get("principal_time_range")
    
    # Preserve native time value (int, date, or datetime) - don't coerce to int
    # Serialize datetime/date objects to ISO strings for JSON compatibility
    end_month_raw = time_selections.get(principal_name) if principal_name else None
    if end_month_raw is not None:
        if isinstance(end_month_raw, dt.datetime):
            end_month = end_month_raw.strftime('%Y-%m-%d')
        elif hasattr(end_month_raw, 'isoformat'):
            end_month = end_month_raw.isoformat()
        else:
            end_month = end_month_raw
    else:
        end_month = None
    
    # Extract data_type from the principal time dimension's adapter
    principal_time_data_type = None
    if principal_range is not None:
        principal_time_data_type = getattr(principal_range, 'adapter', None)
        if principal_time_data_type is not None:
            # Get the data_type by checking if it's a number/YYYYMM adapter or date adapter
            adapter_name = getattr(principal_time_data_type, 'name', '')
            if 'yyyymm' in adapter_name or adapter_name == 'yyyymm_int':
                principal_time_data_type = 'number'
            elif 'date' in adapter_name:
                principal_time_data_type = 'date'
            elif 'timestamp' in adapter_name:
                principal_time_data_type = 'timestamp'
            else:
                principal_time_data_type = None
    
    snap_month = _coerce_int(time_selections.get("snap_month"))
    period_label = selections.get("period_label")

    # snap_month rides as an explicit filter alongside dimension selections.
    extra_filters: Dict[str, Any] = {}
    if snap_month is not None:
        extra_filters["snap_month"] = snap_month

    intent = build_prebuilt_intent(
        drill_metric=selections.get("metric_name"),
        period_label=period_label,
        end_month=end_month,
        rolling_time_dimension_qualified=selections.get("principal_time_dimension_qualified"),
        dimension_selections=selections.get("dimension_selections"),
        extra_filters=extra_filters,
        raw_question=raw_question,
        analysis_mode=selections.get("analysis_mode_name") or "cost_change_investigation_over_time_window",
        principal_time_data_type=principal_time_data_type,
    )

    dim_selections = selections.get("dimension_selections") or {}
    lob_value = dim_selections.get("lob_description")
    if isinstance(lob_value, (list, tuple)):
        lob_value = lob_value[0] if lob_value else None
    conversation_id = build_conversation_id(
        view_name=selections.get("view_name"),
        lob=lob_value,
        snap_month=snap_month,
        period_label=period_label,
        end_month=end_month,
    )

    analysis_overrides: Dict[str, Any] = {"prebuilt_intent": intent}
    live_values = st.session_state.get("live_filter_values")
    if isinstance(live_values, dict) and (live_values.get("dimensions") or live_values.get("time_dimensions")):
        analysis_overrides["live_filter_values"] = live_values

    context = {"analysis_overrides": analysis_overrides}
    semantic_roles = st.session_state.get("semantic_roles")
    if isinstance(semantic_roles, dict) and semantic_roles:
        context["semantic_roles"] = semantic_roles
    return {"context": context, "conversation_id": conversation_id, "intent": intent}


def _render_chat_interface(selections: Dict[str, Any], runtime_settings: Dict[str, Any]) -> None:
    if _CHAT_ENABLED:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.markdown(
        """
        <div style="text-align: center; font-size: 0.7em; margin-top: 16px; margin-bottom: 8px; color: #666;">
            <strong>Disclaimer:</strong> While this employee-assist tool supports informed decision-making, associates hold final responsibility for the accuracy of their work.
        </div>
        """,
        unsafe_allow_html=True
    )

    if _CHAT_ENABLED:
        typed_question = st.text_input(
            "Optional query (used for narrative only — filters drive the analysis)",
            value=st.session_state.get("pending_question", ""),
            key="pending_question",
            placeholder="Where did change happen for state KY?",
        )
        button_cols = st.columns([1, 1, 4])
        research_clicked = button_cols[0].button("Research", type="primary")
        reset_clicked = button_cols[1].button("New conversation")
    else:
        typed_question = ""
        button_cols = st.columns([1, 5])
        research_clicked = button_cols[0].button("Research", type="primary", width="stretch")
        reset_clicked = False

    if reset_clicked:
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex}"
        st.session_state.last_output = None
        st.session_state.pending_question = ""
        return

    if not research_clicked:
        return

    question = (typed_question or "").strip() or DEFAULT_RESEARCH_QUERY
    st.session_state.messages.append({"role": "user", "content": question})

    payload = _build_research_payload(selections, question)
    context = payload["context"]
    conversation_id = payload["conversation_id"]
    st.session_state.thread_id = conversation_id

    logger.info(
        "UI app.py: Research click — conversation_id=%s intent=%s",
        conversation_id,
        payload["intent"],
    )
    # Stash the prebuilt intent on session_state so the streaming helper can
    # ride it through to the orchestrator initial_state.
    st.session_state.last_prebuilt_intent = payload["intent"]

    orchestrator = _get_orchestrator(runtime_settings)
    if orchestrator is None:
        error_detail = st.session_state.get("orchestrator_error", "Unknown error")
        full_trace = st.session_state.get("orchestrator_traceback", "")

        error_msg = f"""❌ **Orchestrator is unavailable**

**Error Details:**
```
{error_detail}
```

**Troubleshooting:**
- Verify all environment variables are set
- Check YAML config file exists: `{runtime_settings['model_path']}`
- Review container logs for detailed stack trace
"""
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

        if full_trace:
            with st.expander("🔍 Full Stack Trace (for debugging)", expanded=False):
                st.code(full_trace, language="python")

        return

    try:
        output = _run_orchestrator_with_progress(
            orchestrator,
            question,
            context,
            conversation_id,
        )
        st.session_state.last_output = output
        summary_text = _summary_as_markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": summary_text})
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.session_state.orchestrator_error = str(exc)
        st.session_state.messages.append(
            {"role": "assistant", "content": f"❌ Execution failed: {str(exc)}"}
        )


def _run_orchestrator_with_progress(
    orchestrator: Any,
    question: str,
    context: Dict[str, Any],
    thread_id: str,
) -> Dict[str, Any]:
    """Run orchestrator with real-time progress updates using streaming."""

    prebuilt_intent = st.session_state.get("last_prebuilt_intent")

    graph = getattr(orchestrator, "graph", None)
    if graph is None:
        return orchestrator(
            question=question,
            context=context,
            thread_id=thread_id,
            conversation_id=thread_id,
            prebuilt_intent=prebuilt_intent,
        )

    initial_state = {
        "question": question,
        "context": context or {},
        "conversation_id": thread_id,
    }
    if isinstance(prebuilt_intent, dict):
        initial_state["prebuilt_intent"] = prebuilt_intent

    config = {
        "configurable": {"thread_id": thread_id}
    }
    
    with st.status("Working on your question...", expanded=True) as status:
        loader_slot = st.empty()
        with loader_slot.container():
            html_content = _agents_loader_html()
            data_uri = f"data:text/html;charset=utf-8,{urllib.parse.quote(html_content)}"
            st.iframe(data_uri, height=_AGENTS_LOADER_IFRAME_HEIGHT)
        st.write("🔍 Starting analysis...")

        last_output = None
        seen_stages: set[str] = set()
        technical_log: List[str] = []

        try:
            for chunk in graph.stream(initial_state, config=config):
                for node_name, node_output in chunk.items():
                    if node_name == "__end__":
                        continue

                    last_completed = node_output.get("last_completed_stage", "") or ""

                    step_summaries = node_output.get("step_summaries", [])
                    if step_summaries and isinstance(step_summaries, list):
                        for summary in step_summaries:
                            if summary:
                                prefix = f"[{last_completed}] " if last_completed else ""
                                technical_log.append(f"{prefix}{summary}")

                    if last_completed and last_completed not in seen_stages:
                        seen_stages.add(last_completed)
                        st.write(f"✓ {_friendly_stage_label(last_completed)}")

                    last_output = node_output

            if technical_log:
                with st.expander("Show technical details", expanded=False):
                    for line in technical_log:
                        st.text(line)

            final_output = last_output.get("final_output") if last_output else None

            if final_output:
                loader_slot.empty()
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                return final_output
            else:
                loader_slot.empty()
                st.warning("Orchestrator completed but no final output was produced.")
                status.update(label="⚠️ Incomplete", state="error", expanded=False)
                return {}

        except Exception as stream_exc:
            loader_slot.empty()
            st.error(f"Streaming error: {stream_exc}")
            if technical_log:
                with st.expander("Show technical details", expanded=False):
                    for line in technical_log:
                        st.text(line)
            status.update(label="❌ Failed", state="error", expanded=False)
            return orchestrator(
                question=question,
                context=context,
                thread_id=thread_id,
                conversation_id=thread_id,
                prebuilt_intent=prebuilt_intent,
            )


def _render_latest_output() -> None:
    output = st.session_state.last_output
    if not output:
        return

    st.divider()
    header_cols = st.columns([6, 2])
    with header_cols[0]:
        st.subheader("Deep Research Results")
    with header_cols[1]:
        if not output.get("clarification_request"):
            _render_pdf_download_button(output)

    if output.get("clarification_request"):
        _render_clarification(output)
        return

    # Display time periods if available
    analysis = output.get("analysis") or {}
    metadata = analysis.get("metadata") or {}
    correlation_summary = metadata.get("correlation_summary") or {}
    period_window = correlation_summary.get("period_window")
    
    period_text = format_period_window(period_window)
    if period_text:
        st.caption(f"📅 {period_text}")
        st.write("")

    _render_tabbed_results(output)


def _render_pdf_download_button(output: Dict[str, Any]) -> None:
    """Render the "Download PDF" button next to the results header.

    Builds the PDF lazily on click via the ``data`` callback so pages that
    never trigger a download don't pay the render cost.
    """

    if not pdf_report.is_available():
        st.caption("Install `reportlab` to enable PDF export.")
        return

    # Include the builder function id so hot-reloading pdf_report.py
    # invalidates any previously cached bytes (a stale reference would keep
    # serving the pre-reload PDF even after we've fixed something in the
    # module).
    signature = (id(output), id(pdf_report.build_pdf_report))
    cache_key = "_pdf_cache"
    cache = st.session_state.get(cache_key) or {}
    if cache.get("signature") != signature:
        try:
            pdf_bytes = pdf_report.build_pdf_report(output)
        except Exception as exc:  # noqa: BLE001 — surface the reason in the UI
            logger.exception("PDF export failed")
            st.caption(f"PDF export failed: {exc}")
            return
        cache = {"signature": signature, "bytes": pdf_bytes}
        st.session_state[cache_key] = cache

    st.download_button(
        label="⬇ Download PDF",
        data=cache["bytes"],
        file_name=pdf_report.suggested_filename(output),
        mime="application/pdf",
        help="Executive-friendly summary of the analysis (skips SQL and technical details).",
        width="stretch",
    )


def _render_clarification(output: Dict[str, Any]) -> None:
    clarification = output.get("clarification_request") or {}
    st.warning("Need clarification before execution.")
    if clarification.get("blocking_issues"):
        st.write("**Blocking issues:**")
        for issue in clarification.get("blocking_issues", []):
            st.write(f"- {issue}")
    if clarification.get("questions"):
        st.write("**Questions:**")
        for question in clarification.get("questions", []):
            st.write(f"- {question}")


def _render_tabbed_results(output: Dict[str, Any]) -> None:
    research = output.get("research") or {}
    patterns: List[Dict[str, Any]] = list(research.get("business_patterns") or [])
    reimbursement_by_pattern: Dict[str, Any] = research.get("reimbursement_by_pattern") or {}
    pattern_cards: List[Dict[str, Any]] = list(
        (research.get("pattern_summary") or {}).get("cards") or []
    )

    pattern_tab_labels = [
        f"Pattern {p.get('pattern_rank', idx + 1)}" for idx, p in enumerate(patterns)
    ]
    tab_labels = (
        ["Correlation Summary"]
        + pattern_tab_labels
        + ["Recommendations", "SQL Queries", "Technical Details"]
    )
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_correlation_summary(output)

    for idx, pattern in enumerate(patterns):
        with tabs[1 + idx]:
            rank = str(pattern.get("pattern_rank", idx + 1))
            _render_pattern_tab(
                pattern, reimbursement_by_pattern.get(rank), pattern_cards
            )

    sql_tab_index = 1 + len(patterns)
    with tabs[sql_tab_index]:
        _render_recommendations(research.get("recommendations") or [])

    with tabs[sql_tab_index + 1]:
        _render_sql_queries(output)

    with tabs[sql_tab_index + 2]:
        _render_technical_details(output)


def _render_correlation_summary(output: Dict[str, Any]) -> None:
    analysis = output.get("analysis") or {}
    metadata = analysis.get("metadata") or {}
    correlation_summary = metadata.get("correlation_summary") or {}

    question = output.get("question", "Analysis")
    exec_summary_text = correlation_summary.get("executive_summary")

    title_parts = ["Root Cause Analysis"]
    if question:
        title_parts.append(question)

    st.markdown(f"## {': '.join(title_parts)}")
    st.divider()

    if exec_summary_text:
        st.markdown("### Summary")
        st.markdown(exec_summary_text)
        st.write("")

    rendered_visuals = render_correlation_visuals(correlation_summary)

    research = output.get("research") or {}
    patterns = research.get("business_patterns") or []
    recommendations = research.get("recommendations") or []
    if patterns or recommendations:
        st.markdown("### Downstream Outputs")
        st.write(
            f"- **Business patterns:** {len(patterns)}\n"
            f"- **Recommendations:** {len(recommendations)}"
        )

    if not exec_summary_text and not patterns and not rendered_visuals:
        st.info("No executive summary or patterns available for this analysis.")


def _render_pattern_tab(
    pattern: Dict[str, Any],
    reimbursement: Optional[Dict[str, Any]],
    cards: Optional[List[Dict[str, Any]]] = None,
) -> None:
    rank = pattern.get("pattern_rank") or pattern.get("rank") or ""
    storyboard_rendered = render_pattern_storyboard(pattern, cards or [], rank=rank)
    if not storyboard_rendered:
        # Defensive fallback to the legacy bullet view if there's nothing for the storyboard
        # to render (e.g. a pattern with no source cards and no evidence).
        st.markdown(f"### {pattern.get('pattern_title') or 'Pattern'}")
        render_drill_down_paths(pattern, cards or [])

    details = pattern.get("pattern_details")
    if details:
        with st.expander("Pattern details", expanded=False):
            st.markdown(details)

    st.divider()
    st.markdown("### Reimbursement Policy Findings")
    if reimbursement is None:
        st.info("No reimbursement output produced for this pattern.")
        return

    if reimbursement.get("error"):
        st.error(f"Reimbursement agent failed: {reimbursement['error']}")
        return

    formatted_output = reimbursement.get("formatted_output") or reimbursement.get("output") or {}
    if not isinstance(formatted_output, dict):
        formatted_output = {}

    elevance_summary = reimbursement.get("elevance_executive_summary") or formatted_output.get(
        "elevance_executive_summary"
    )
    if elevance_summary:
        st.markdown("#### 🏢 Elevance Executive Summary")
        st.markdown(elevance_summary)

    summary_table = formatted_output.get("summary_table")
    rendered_table = render_payer_summary_table(summary_table)

    recommended_actions = (
        reimbursement.get("recommended_action")
        or reimbursement.get("recommendations")
        or formatted_output.get("recommended_action")
        or []
    )
    individual_policies = (
        formatted_output.get("individual_policies")
        or reimbursement.get("individual_policies")
        or formatted_output.get("reimbursement_policies")
        or reimbursement.get("reimbursement_policies")
        or []
    )
    render_recommended_actions(recommended_actions, individual_policies)
    render_policy_links_expander(individual_policies)

    if not (rendered_table or recommended_actions or individual_policies or elevance_summary):
        st.info("No reimbursement details available for this pattern.")

    with st.expander("Raw reimbursement output", expanded=False):
        st.json(reimbursement)


_PRIORITY_BUCKETS = ("HIGH", "MEDIUM", "LOW")
_PRIORITY_COLORS = {
    "HIGH": "#B91C1C",
    "MEDIUM": "#B45309",
    "LOW": "#047857",
}


def _normalize_priority(value: Any) -> str:
    text = "" if value is None else str(value).strip().upper()
    if text in {"H", "HIGH", "URGENT", "CRITICAL"}:
        return "HIGH"
    if text in {"M", "MED", "MEDIUM"}:
        return "MEDIUM"
    if text in {"L", "LOW", "MINOR"}:
        return "LOW"
    return text or "MEDIUM"


def _render_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    st.markdown("### Final Recommendations")
    if not recommendations:
        st.info("No recommendations were generated for this analysis.")
        return

    bucketed: Dict[str, List[Dict[str, Any]]] = {p: [] for p in _PRIORITY_BUCKETS}
    other: List[Dict[str, Any]] = []
    for rec in recommendations:
        pri = _normalize_priority(rec.get("priority"))
        if pri in bucketed:
            bucketed[pri].append(rec)
        else:
            other.append(rec)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric("Total recommendations", str(len(recommendations)))
    for col, pri in zip(kpi_cols[1:], _PRIORITY_BUCKETS):
        with col:
            with st.container(border=True):
                st.metric(f"{pri.title()} priority", str(len(bucketed[pri])))

    st.markdown("")
    columns = st.columns(3, gap="medium")
    for col, pri in zip(columns, _PRIORITY_BUCKETS):
        with col:
            color = _PRIORITY_COLORS[pri]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"margin-bottom:8px;'>"
                f"<span style='color:{color};font-weight:700;letter-spacing:0.04em;'>{pri}</span>"
                f"<span style='color:#5B6776;font-size:0.85em;'>{len(bucketed[pri])} item(s)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if not bucketed[pri]:
                st.caption("No items in this priority.")
                continue
            for rec in bucketed[pri]:
                _render_recommendation_card(rec, color)

    if other:
        st.markdown("##### Other / Unclassified")
        for rec in other:
            _render_recommendation_card(rec, _PRIORITY_COLORS["MEDIUM"])


def _render_recommendation_card(rec: Dict[str, Any], color: str) -> None:
    rank = rec.get("rank")
    title = rec.get("description") or rec.get("title") or (f"Recommendation {rank}".strip())
    short_title = title if len(title) <= 140 else title[:137] + "…"
    evidence = rec.get("evidence") or []
    peer_benchmarking = rec.get("peer_benchmarking") or []
    owner = rec.get("owner") or rec.get("assignee")
    eta = rec.get("eta") or rec.get("eta_weeks")

    with st.container(border=True):
        rank_chip = (
            f"<span style='background:{color};color:#FFF;border-radius:999px;"
            f"padding:2px 10px;font-size:0.75em;font-weight:600;'>#{rank}</span> "
            if rank
            else ""
        )
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:8px;'>{rank_chip}"
            f"<span style='font-weight:600;color:#1A2436;line-height:1.4;'>{short_title}</span></div>",
            unsafe_allow_html=True,
        )
        meta_bits: List[str] = []
        if owner:
            meta_bits.append(f"Owner: {owner}")
        if eta:
            meta_bits.append(f"ETA: {eta}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        if evidence or peer_benchmarking:
            with st.expander("Evidence & peer benchmarking", expanded=False):
                if evidence:
                    st.markdown("**Evidence**")
                    for item in evidence:
                        st.markdown(f"- {item}")
                if peer_benchmarking:
                    st.markdown("**Peer benchmarking**")
                    for item in peer_benchmarking:
                        st.markdown(f"- {item}")


_SQL_KEY_QUERIES = {
    "root_summary.sql",
    "root_explainers.sql",
    "baseline_extract.sql",
    "comparison_extract.sql",
}


def _infer_sql_stage(filepath: str) -> str:
    """
    Infer analysis stage from file path.
    
    Args:
        filepath: Relative path from queries folder (e.g., "level_2/place_of_service/health_service.sql")
    """
    normalized = filepath.replace("\\", "/")
    
    if "/level_" in normalized or normalized.startswith("level_"):
        return "Drill-down"
    
    stem = normalized.split("/")[-1].replace(".sql", "")
    
    if stem.startswith("root_"):
        return "Root"
    if stem.startswith("baseline") or stem.startswith("comparison"):
        return "Extract"
    if "drill" in stem:
        return "Drill-down"
    if "interaction" in stem or "matrix" in stem:
        return "Interaction"
    if "pattern" in stem:
        return "Pattern"
    if "reimbursement" in stem:
        return "Reimbursement"
    return "Other"


def _collect_sql_records(sql_files: List[Path], queries_folder: Path) -> List[Dict[str, Any]]:
    """
    Collect SQL file records with relative path information.
    
    Args:
        sql_files: List of SQL file paths
        queries_folder: Base queries directory for computing relative paths
    """
    records: List[Dict[str, Any]] = []
    for sql_file in sql_files:
        try:
            relative_path = sql_file.relative_to(queries_folder)
            relative_path_str = str(relative_path).replace("\\", "/")
        except ValueError:
            relative_path_str = sql_file.name
        
        try:
            content = sql_file.read_text()
        except Exception as exc:
            records.append({
                "id": str(sql_file),
                "title": sql_file.stem.replace("_", " ").title(),
                "stage": _infer_sql_stage(relative_path_str),
                "lines": 0,
                "filename": sql_file.name,
                "relative_path": relative_path_str,
                "sql": "",
                "is_key": sql_file.name in _SQL_KEY_QUERIES,
                "error": str(exc),
            })
            continue
        records.append({
            "id": str(sql_file),
            "title": sql_file.stem.replace("_", " ").title(),
            "stage": _infer_sql_stage(relative_path_str),
            "lines": content.count("\n") + 1,
            "filename": sql_file.name,
            "relative_path": relative_path_str,
            "sql": content,
            "is_key": sql_file.name in _SQL_KEY_QUERIES,
            "error": None,
        })
    return records


def _render_sql_queries(output: Dict[str, Any]) -> None:
    st.markdown("### Generated SQL Queries")

    analysis = output.get("analysis") or {}
    metadata = analysis.get("metadata") or {}
    correlation_summary = metadata.get("correlation_summary") or {}
    run_id = correlation_summary.get("run_id")
    run_dir = correlation_summary.get("run_dir")

    sql_files: List[Path] = []
    queries_folder = None
    if run_dir:
        queries_folder = Path(run_dir) / "queries"
        if queries_folder.exists() and queries_folder.is_dir():
            sql_files = sorted(queries_folder.glob("**/*.sql"))
    elif run_id:
        queries_folder = DEFAULT_OUTPUT_ROOT / run_id / "queries"
        if queries_folder.exists() and queries_folder.is_dir():
            sql_files = sorted(queries_folder.glob("**/*.sql"))

    if sql_files and queries_folder:
        _render_sql_catalog(_collect_sql_records(sql_files, queries_folder))
        return

    artifacts = analysis.get("artifacts", [])
    sql_artifacts = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("type") in {"sql", "query"}
    ] if isinstance(artifacts, list) else []

    if not sql_artifacts:
        st.info("No SQL queries were generated for this analysis.")
        return

    records: List[Dict[str, Any]] = []
    for idx, artifact in enumerate(sql_artifacts, 1):
        sql_content = artifact.get("content", "") or ""
        artifact_name = artifact.get("name", f"Query {idx}")
        records.append({
            "id": str(artifact.get("id", f"artifact_{idx}")),
            "title": artifact_name,
            "stage": str(artifact.get("stage") or "Artifact"),
            "lines": sql_content.count("\n") + 1 if sql_content else 0,
            "filename": str(artifact.get("filename", f"{artifact_name}.sql")),
            "sql": sql_content,
            "is_key": False,
            "error": None,
        })
    _render_sql_catalog(records)


def _render_sql_catalog(records: List[Dict[str, Any]]) -> None:
    if not records:
        st.info("No SQL queries were generated for this analysis.")
        return

    stages = sorted({r["stage"] for r in records})
    stage_counts = {s: sum(1 for r in records if r["stage"] == s) for s in stages}
    chip_html = " ".join(
        f"<span style='display:inline-block;background:#F1F4F9;color:#1A2436;"
        f"border:1px solid #D6DBE4;padding:4px 12px;border-radius:999px;"
        f"margin-right:6px;font-size:0.85em;'>{s} <b>{stage_counts[s]}</b></span>"
        for s in stages
    )
    st.markdown(
        f"**Analysis backed by {len(records)} SQL queries across {len(stages)} stages.**",
    )
    st.markdown(chip_html, unsafe_allow_html=True)

    with st.expander("Browse queries", expanded=False):
        filter_cols = st.columns([2, 3, 1])
        with filter_cols[0]:
            picked_stages = st.multiselect(
                "Stages",
                stages,
                default=stages,
                key="sql_stage_filter",
            )
        with filter_cols[1]:
            search = st.text_input(
                "Search",
                placeholder="filename or SQL text…",
                key="sql_search",
            )
        with filter_cols[2]:
            key_only = st.toggle("Key only", value=False, key="sql_key_only")

        needle = (search or "").strip().lower()
        filtered = [
            r
            for r in records
            if r["stage"] in picked_stages
            and (not key_only or r["is_key"])
            and (
                not needle
                or needle in r["filename"].lower()
                or needle in r["sql"].lower()
                or needle in r["title"].lower()
            )
        ]
        if not filtered:
            st.caption("No queries match the current filters.")
            return

        # Sort: key queries first, then by stage + filename
        filtered.sort(key=lambda r: (not r["is_key"], r["stage"], r["filename"]))

        # Resolve the active selection by id so it survives filter changes.
        current_id = st.session_state.get("selected_sql_id")
        if not any(r["id"] == current_id for r in filtered):
            current_id = filtered[0]["id"]
            st.session_state["selected_sql_id"] = current_id

        rail_col, detail_col = st.columns([2, 3], gap="medium")
        with rail_col:
            st.caption(f"{len(filtered)} queries · click any row to load its SQL")
            list_box = st.container(height=420)
            with list_box:
                for r in filtered:
                    is_active = r["id"] == current_id
                    star = "⭐ " if r["is_key"] else ""
                    label = f"{star}{r['stage']} · {r['title']}  ·  {r['lines']} lines"
                    if st.button(
                        label,
                        key=f"sql_pick_{r['id']}",
                        width="stretch",
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state["selected_sql_id"] = r["id"]
                        st.rerun()

        record = next(
            (r for r in filtered if r["id"] == current_id),
            filtered[0],
        )
        with detail_col:
            st.markdown(f"**{record['stage']} · {record['title']}**")
            if record.get('relative_path') and record['relative_path'] != record['filename']:
                st.caption(f"`{record['relative_path']}` · {record['lines']} lines")
            else:
                st.caption(f"`{record['filename']}` · {record['lines']} lines")
            if record["error"]:
                st.error(f"Error reading file: {record['error']}")
            else:
                st.code(record["sql"], language="sql")
                st.download_button(
                    "Download .sql",
                    record["sql"],
                    file_name=record["filename"],
                    mime="text/plain",
                    key=f"sql_download_{record['id']}",
                )


def _render_technical_details(output: Dict[str, Any]) -> None:
    st.markdown("### Technical Details")

    with st.expander("Resolved Intent", expanded=False):
        st.json(output.get("intent", {}))

    with st.expander("Full Analysis Contract", expanded=False):
        st.json(output.get("analysis", {}))

    research = output.get("research") or {}
    with st.expander("Business Patterns (raw)", expanded=False):
        st.json(research.get("business_patterns") or [])

    with st.expander("Reimbursement by Pattern", expanded=False):
        st.json(research.get("reimbursement_by_pattern") or {})

    with st.expander("Recommendations (raw)", expanded=False):
        st.json(research.get("recommendations") or [])

    with st.expander("Report Contract", expanded=False):
        st.json(output.get("report", {}))

    with st.expander("Visual Recommendations", expanded=False):
        st.json(output.get("visuals", {}))


def _get_orchestrator(runtime_settings: Dict[str, Any]) -> Optional[Any]:
    if _ORCHESTRATOR_IMPORT_ERROR or OrchestratorConfig is None or build_orchestrator is None:
        _, missing = _agent_availability()
        detail = _ORCHESTRATOR_IMPORT_ERROR or "orchestrator_client unavailable"
        if missing:
            detail += "\nMissing agents: " + ", ".join(f"{k} ({v})" for k, v in missing.items())
        st.session_state.orchestrator_error = detail
        return None

    config = OrchestratorConfig(
        yaml_path=runtime_settings["model_path"],
        enable_llm=runtime_settings["enable_llm"],
        enable_snowflake=runtime_settings["enable_snowflake"],
        correlation_output_root=str(DEFAULT_OUTPUT_ROOT),
    )

    try:
        return build_orchestrator(config)
    except Exception as exc:
        error_msg = str(exc)
        full_traceback = traceback.format_exc()
        logger.error("Failed to build orchestrator: %s", error_msg, exc_info=True)
        logger.error("Full traceback:\n%s", full_traceback)
        st.session_state.orchestrator_error = error_msg
        st.session_state.orchestrator_traceback = full_traceback
        return None


def _summary_as_markdown(output: Dict[str, Any]) -> str:
    summary = output.get("summary") or {}
    if summary:
        bullets = summary.get("bullets", []) or []
        bullet_text = "\n".join(f"- {item}" for item in bullets)
        headline = summary.get("headline", "Orchestrator summary")
        return f"{headline}\n{bullet_text}" if bullet_text else headline

    if output.get("clarification_request"):
        return "Need clarification before continuing."

    return "Orchestrator run completed."


def _render_diagnostics_sidebar(
    runtime_settings: Dict[str, Any],
    schema: SemanticSchema,
    filter_values: FilterValues,
    semantic_roles: SemanticRoles,
) -> None:
    """Render diagnostics panel in sidebar to help debug configuration issues."""
    with st.sidebar:
        st.caption(
            f"Filter values loaded in **{filter_values.fetch_seconds:.2f}s** · "
            f"Snowflake: {'✅' if filter_values.snowflake_enabled else '❌ (YAML fallback)'}"
        )
        if st.button("🔄 Refresh filter values", key="refresh_filter_values", width="stretch"):
            load_all_filter_values.clear()
            load_semantic_roles.clear()
            st.rerun()

        with st.expander("🔧 System Diagnostics", expanded=False):
            st.subheader("Environment")
            st.write(f"**ENVIRONMENT:** `{AppConstants.ENV}`")
            st.write(f"**Semantic view:** `{schema.view_name}`")
            yaml_path = runtime_settings["model_path"]
            yaml_exists = Path(yaml_path).exists()
            st.write(f"**YAML Config:** {'✅' if yaml_exists else '❌'}")
            st.code(yaml_path, language="text")

            st.subheader("Agents")
            available, missing = _agent_availability()
            if _ORCHESTRATOR_IMPORT_ERROR:
                st.error(f"orchestrator_client failed to import: {_ORCHESTRATOR_IMPORT_ERROR}")
            for name in available:
                st.write(f"✅ `{name}`")
            for name, err in missing.items():
                st.write(f"❌ `{name}` — {err}")
            if not available and not missing:
                st.warning("Could not enumerate agents — deep_research_agents package unavailable.")

            st.subheader("Filter values")
            st.write(
                f"Loaded in **{filter_values.fetch_seconds:.2f}s** · "
                f"Snowflake: {'✅' if filter_values.snowflake_enabled else '❌ (using YAML fallback)'}"
            )
            for w in filter_values.warnings:
                st.warning(w)
            for dv in filter_values.dimensions:
                badge = {"db": "✅", "yaml_fallback": "⚠️", "synthetic": "🟡", "unknown": "❌"}.get(dv.source, "·")
                line = f"{badge} `{dv.dimension_name}` — {dv.source}"
                if dv.is_free_text:
                    line += " (free-text)"
                if dv.warning:
                    line += f" — {dv.warning}"
                st.write(line)
            for tr in filter_values.time_dimensions:
                badge = {"db": "✅", "yaml_fallback": "⚠️", "synthetic": "🟡", "unknown": "❌"}.get(tr.source, "·")
                st.write(f"{badge} `{tr.dimension_name}` — {tr.source} ({tr.adapter.name})")
                if tr.warning:
                    st.caption(tr.warning)

            st.subheader("Semantic roles")
            role_source_badge = {
                "companion_json": "✅",
                "regenerated": "🟡",
                "unavailable": "❌",
            }.get(semantic_roles.source, "·")
            st.write(
                f"{role_source_badge} Source: `{semantic_roles.source}` · "
                f"Dimensions classified: **{len(semantic_roles.dimension_roles)}**"
            )
            if semantic_roles.llm_model:
                st.caption(f"Last generator: `{semantic_roles.llm_model}`")
            if semantic_roles.generated_at:
                st.caption(f"Generated at: `{semantic_roles.generated_at}`")
            if semantic_roles.warning:
                st.warning(semantic_roles.warning)
            counts = semantic_roles.role_counts()
            if counts:
                summary_bits = [f"{count} {role}" for role, count in counts.items()]
                st.caption("Role distribution: " + ", ".join(summary_bits))
            flagged_other = semantic_roles.dims_flagged_other()
            if flagged_other:
                st.warning(
                    f"{len(flagged_other)} dimension(s) fell back to `other` — review: "
                    + ", ".join(f"`{name}`" for name in flagged_other[:8])
                    + ("..." if len(flagged_other) > 8 else "")
                )
            if st.button(
                "♻️ Regenerate semantic roles",
                key="regenerate_semantic_roles",
                width="stretch",
                help="Force a fresh LLM classification and overwrite the companion JSON.",
            ):
                load_semantic_roles.clear()
                st.session_state["_force_regenerate_semantic_roles"] = True
                st.rerun()

            st.subheader("Environment Variables")
            if runtime_settings["enable_llm"]:
                st.write("**LLM Variables:**")
                llm_vars = [
                    "EHAP_BASE_URL",
                    "EHAP_CLIENT_ID",
                    "EHAP_CLIENT_SECRET",
                    "EHAP_LLM_MODEL",
                    "DEEP_RESEARCH_LLM_MODEL",
                ]
                for var in llm_vars:
                    is_set = bool(os.environ.get(var))
                    value_preview = "***" if is_set else "NOT SET"
                    st.write(f"  {'✅' if is_set else '❌'} `{var}`: {value_preview}")

            if runtime_settings["enable_snowflake"]:
                st.write("**Snowflake Variables:**")
                sf_vars = [
                    "SNOWFLAKE_ACCOUNT",
                    "SNOWFLAKE_USER",
                    "SNOWFLAKE_SECRET",
                    "SNOWFLAKE_WAREHOUSE",
                    "SNOWFLAKE_DATABASE",
                    "SNOWFLAKE_SCHEMA",
                ]
                for var in sf_vars:
                    is_set = bool(os.environ.get(var))
                    value_preview = "***" if is_set else "NOT SET"
                    st.write(f"  {'✅' if is_set else '❌'} `{var}`: {value_preview}")

            if st.session_state.get("orchestrator_error"):
                st.subheader("Last Error")
                st.error(st.session_state.orchestrator_error)


if __name__ == "__main__":
    main()
