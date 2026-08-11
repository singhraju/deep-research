"""Translate dynamic UI filter selections into orchestrator context.

The widget-rendering side lives in ``app.py``; the value-loading side lives in
``filter_values_loader.py``. This module owns the mapping from user picks to
the context dict the orchestrator expects, plus a few period-payload helpers
that are shared by both surfaces.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ui.filter_values_loader import FilterValues

logger = logging.getLogger(__name__)

_SENTINEL_VALUES = {"All", "Select...", "", None}


def _coerce_dimension_value(value: Any) -> Any:
    """Normalize a single dimension selection into the shape the correlation agent wants.

    - ``None`` / empty / sentinel string → return ``None`` (caller drops the key).
    - ``list``/``tuple``/``set`` → drop sentinels, then return a single scalar
      when only one survives, or a list when 2+ remain. ``render_filter_clause``
      emits ``field = value`` for scalars and ``field IN (...)`` for lists.
    - everything else → returned as-is (scalar string, int, etc.).
    """

    if isinstance(value, (list, tuple, set)):
        cleaned = [item for item in value if item not in _SENTINEL_VALUES]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return list(cleaned)
    if value in _SENTINEL_VALUES:
        return None
    return value


def build_context_from_filters(
    *,
    dimension_selections: Mapping[str, Any] | None = None,
    time_selections: Mapping[str, Any] | None = None,
    metric_name: Optional[str] = None,
    period_label: Optional[str] = None,
    principal_time_dimension: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate dynamic filter selections into an orchestrator context dict.

    - Each dimension selection that is not a sentinel ("All"/"Select.../empty)
      becomes ``context[dimension_name] = value``.
    - The principal time dimension's value is mapped to ``context["end_time"]``
      so existing orchestrator code keeps working unchanged.
    - ``metric_name`` becomes ``context["drill_metric"]``.
    - ``period_label`` is expanded into the legacy period payload when an end
      time is available, using the existing ECAP helpers.
    """

    context: Dict[str, Any] = {}

    dimension_selections = dimension_selections or {}
    for dim_name, raw_value in dimension_selections.items():
        coerced = _coerce_dimension_value(raw_value)
        if coerced is None:
            continue
        context[dim_name] = coerced

    time_selections = time_selections or {}
    end_time_value: Optional[int] = None
    for dim_name, value in time_selections.items():
        if value is None:
            continue
        context[dim_name] = value
        if dim_name == principal_time_dimension:
            if isinstance(value, int):
                end_time_value = value
            elif isinstance(value, datetime):
                end_time_value = value.year * 100 + value.month
            else:
                parsed = _parse_incurred_month(str(value))
                if parsed is not None:
                    end_time_value = parsed

    if end_time_value is not None:
        context.setdefault("end_time", end_time_value)

    if metric_name and metric_name not in _SENTINEL_VALUES:
        context["drill_metric"] = metric_name

    if period_label and period_label not in _SENTINEL_VALUES:
        rolling_window = _period_to_rolling_window(period_label)
        if rolling_window:
            context["rolling_window"] = rolling_window

        period_payload = _build_period_payload(period_label, end_time_value, rolling_window)
        if period_payload:
            context["period"] = period_payload
            current_period = period_payload.get("current_period") or {}
            if current_period.get("start_time") is not None:
                context.setdefault("start_time", current_period.get("start_time"))
            if current_period.get("end_time") is not None:
                context["end_time"] = current_period.get("end_time")
        else:
            context.setdefault("period", period_label)

    logger.info("build_context_from_filters -> %s", context)
    return context


# ---------------------------------------------------------------------------
# Period payload helpers (preserved verbatim — orchestrator depends on them)
# ---------------------------------------------------------------------------


def _period_to_rolling_window(period: str) -> Optional[str]:
    normalized = period.lower().replace("rolling", "").strip()
    tokens = normalized.split()
    if not tokens:
        return None
    if tokens[0].isdigit():
        return f"{tokens[0]}_months"
    return None


def _period_to_ecap_code(period: str) -> Optional[str]:
    """Map UI period labels to ECAP trend codes (R3, R6, R12, YTD)."""
    normalized = period.strip().lower()
    if normalized.startswith("rolling"):
        tokens = normalized.replace("rolling", "").strip().split()
        if tokens and tokens[0].isdigit():
            return f"R{tokens[0]}"
    if normalized in {"ytd", "year to date"}:
        return "YTD"
    if normalized in {"monthly", "month"}:
        return "R1"
    return None


def get_ecap_start_month(trnd_tm_prd_cd: str, trnd_tm_prd_end_mnth_nbr: Any, data_type: Optional[str] = None) -> Any:
    """Compute start month for a given ECAP time period.
    
    Supports both YYYYMM integer format and date strings based on data_type.
    
    Args:
        trnd_tm_prd_cd: Period code like 'R3', 'R6', 'R12', 'YTD'
        trnd_tm_prd_end_mnth_nbr: End month (YYYYMM int or ISO date string)
        data_type: Semantic data type ('number' for YYYYMM, 'date' for dates)
    
    Returns:
        Start month in the same format as the input
    """
    from ui.time_adapters import add_months, first_day_of_month
    
    # Parse end_date based on data_type
    if data_type and data_type.lower() == 'number':
        end_date = datetime.strptime(str(trnd_tm_prd_end_mnth_nbr), "%Y%m")
        is_number_type = True
    elif isinstance(trnd_tm_prd_end_mnth_nbr, str):
        from dateutil import parser as dateutil_parser
        end_date = dateutil_parser.parse(trnd_tm_prd_end_mnth_nbr)
        is_number_type = False
    elif isinstance(trnd_tm_prd_end_mnth_nbr, datetime):
        end_date = trnd_tm_prd_end_mnth_nbr
        is_number_type = False
    else:
        # Fallback for backward compatibility
        end_date = datetime.strptime(str(trnd_tm_prd_end_mnth_nbr), "%Y%m")
        is_number_type = True
    
    # Calculate start_date
    if trnd_tm_prd_cd.startswith("R"):
        months_back = int(trnd_tm_prd_cd[1:]) - 1
        start_date = add_months(first_day_of_month(end_date), -months_back)
    elif trnd_tm_prd_cd == "YTD":
        start_date = datetime(end_date.year, 1, 1)
    else:
        raise ValueError(f"Unsupported trnd_tm_prd_cd: {trnd_tm_prd_cd}")
    
    # Return in the same format as input
    if is_number_type:
        return int(start_date.strftime("%Y%m"))
    else:
        return start_date.strftime("%Y-%m-%d")


def convert_current_ecap_time_to_previous_year(
    current_period_start: Any,
    current_period_end: Any,
    data_type: Optional[str] = None,
) -> tuple[Any, Any]:
    """Convert ECAP period to previous year period.
    
    Supports both YYYYMM integer format and date strings based on data_type.
    
    Args:
        current_period_start: Start of current period
        current_period_end: End of current period
        data_type: Semantic data type ('number' for YYYYMM, 'date' for dates)
    
    Returns:
        Tuple of (previous_start, previous_end) in the same format as inputs
    """

    def shift_to_previous_year_int(period_value: int) -> int:
        year = period_value // 100
        month = period_value % 100
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month in period: {period_value}")
        return (year - 1) * 100 + month
    
    def shift_to_previous_year_date(date_str: str) -> str:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(date_str)
        prev_year_dt = datetime(dt.year - 1, dt.month, dt.day)
        return prev_year_dt.strftime("%Y-%m-%d")
    
    # Determine format and shift accordingly
    if data_type and data_type.lower() == 'number':
        return (
            shift_to_previous_year_int(int(current_period_start)),
            shift_to_previous_year_int(int(current_period_end))
        )
    elif isinstance(current_period_start, str):
        return (
            shift_to_previous_year_date(current_period_start),
            shift_to_previous_year_date(current_period_end)
        )
    else:
        # Fallback for backward compatibility
        return (
            shift_to_previous_year_int(int(current_period_start)),
            shift_to_previous_year_int(int(current_period_end))
        )


def _build_period_payload(
    period_label: str,
    end_month: Any,
    rolling_window: Optional[str],
    data_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build period payload for correlation agent.
    
    Supports both YYYYMM integer format and date strings based on data_type.
    
    Args:
        period_label: Period label like 'Rolling 3', 'YTD'
        end_month: End month (YYYYMM int or ISO date string)
        rolling_window: Rolling window code
        data_type: Semantic data type ('number' for YYYYMM, 'date' for dates)
    
    Returns:
        Period payload dict with current_period and previous_period
    """
    if end_month is None:
        return None

    period_code = _period_to_ecap_code(period_label)
    if period_code is None:
        return None

    start_month = get_ecap_start_month(period_code, end_month, data_type)
    previous_start, previous_end = convert_current_ecap_time_to_previous_year(start_month, end_month, data_type)

    payload: Dict[str, Any] = {
        "current_period": {
            "start_time": start_month,
            "end_time": end_month,
        },
        "previous_period": {
            "start_time": previous_start,
            "end_time": previous_end,
        },
    }
    if rolling_window:
        payload["rolling_window"] = rolling_window
    return payload


def _parse_incurred_month(label: str) -> Optional[int]:
    try:
        parsed = datetime.strptime(label, "%Y %B")
        return int(parsed.strftime("%Y%m"))
    except ValueError:
        stripped = label.strip()
        if stripped.isdigit() and len(stripped) == 6:
            return int(stripped)
        return None


# ---------------------------------------------------------------------------
# Pre-built intent (bypass the orchestrator's intent-detection LLM step)
# ---------------------------------------------------------------------------


def _coerce_filter_value(value: Any) -> Any:
    """Strip sentinels and normalize list selections for filter emission."""
    if isinstance(value, (list, tuple, set)):
        cleaned = [item for item in value if item not in _SENTINEL_VALUES]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return list(cleaned)
    if value in _SENTINEL_VALUES:
        return None
    return value


def _filter_conditions(
    dimension_selections: Mapping[str, Any] | None,
    extra_filters: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build the orchestrator's ``filters`` list from UI selections.

    Mirrors the shape ``user_intent.FilterCondition`` produces: each entry
    has ``field``, ``operator``, ``value``, ``source``. Multi-select dimensions
    emit one row per value with ``operator = "="`` (the orchestrator's
    correlation handler treats repeated equality filters as an IN-set).
    """

    conditions: List[Dict[str, Any]] = []

    def _append(field: str, raw_value: Any) -> None:
        value = _coerce_filter_value(raw_value)
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                conditions.append({
                    "field": field,
                    "operator": "=",
                    "value": item,
                    "source": "dimension_match",
                })
        else:
            conditions.append({
                "field": field,
                "operator": "=",
                "value": value,
                "source": "dimension_match",
            })

    for field, raw_value in (dimension_selections or {}).items():
        _append(field, raw_value)
    for field, raw_value in (extra_filters or {}).items():
        _append(field, raw_value)
    return conditions


def build_prebuilt_intent(
    *,
    drill_metric: Optional[str],
    period_label: Optional[str],
    end_month: Any,
    rolling_time_dimension_qualified: Optional[str],
    dimension_selections: Mapping[str, Any] | None,
    extra_filters: Mapping[str, Any] | None = None,
    raw_question: str,
    analysis_mode: str = "cost_change_investigation_over_time_window",
    principal_time_data_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a fully-formed IntentOutput dict from UI selections.

    Matches ``user_intent.IntentOutput`` so the orchestrator can write it
    directly into state without invoking the LLM intent-resolution step.

    - ``drill_metric``: the metric the user picked in the UI (already in
      ``table.column`` form, e.g. ``expense_detail.total_paid``).
    - ``period_label``: one of "Rolling 3", "Rolling 6", "Rolling 12", "YTD".
    - ``end_month``: YYYYMM int end anchor for the current window.
    - ``rolling_time_dimension_qualified``: ``table.column`` reference taken
      from the semantic YAML (e.g. ``expense_detail.incurred_month``).
    - ``dimension_selections``: ``{field: value-or-list}`` from the UI rows.
    - ``extra_filters``: additional ``{field: value}`` pairs not surfaced as
      curated dimensions (snap_month, hcc_high, lob_description, ...).
    - ``raw_question``: the user's free-text query (falls back to a generic
      "Where did change happen?" upstream when empty).
    """

    rolling_window = _period_to_rolling_window(period_label or "")
    period_payload: Optional[Dict[str, Any]] = None
    if period_label and end_month is not None:
        period_payload = _build_period_payload(period_label, end_month, rolling_window, principal_time_data_type)
    if period_payload is None:
        period_payload = {
            "current_period": {"start_time": None, "end_time": end_month},
            "previous_period": {"start_time": None, "end_time": None},
        }
    if rolling_time_dimension_qualified:
        period_payload["rolling_time_dimension"] = rolling_time_dimension_qualified

    # Normalize ``rolling_window`` to a single-element list. The orchestrator's
    # ``build_clarification_request`` runs ``_safe_list(period.rolling_window)``
    # and demands exactly one entry; if it's still a bare string the call
    # collapses to ``[]`` and the request is rejected with "0 rolling window
    # options". The correlation handler at orchestrator.py:747 also tolerates
    # the list form.
    if isinstance(period_payload.get("rolling_window"), str):
        period_payload["rolling_window"] = [period_payload["rolling_window"]]
    elif rolling_window and not period_payload.get("rolling_window"):
        period_payload["rolling_window"] = [rolling_window]

    # Surface the current window's start/end at the top of ``period`` too —
    # ``build_clarification_request`` only treats explicit dates as satisfied
    # when ``period.start_time``/``period.end_time`` are populated. We keep
    # the nested ``current_period`` / ``previous_period`` for the correlation
    # agent's ``resolve_period_window``, which reads either form.
    current_period = period_payload.get("current_period") or {}
    if current_period.get("start_time") is not None:
        period_payload.setdefault("start_time", current_period.get("start_time"))
    if current_period.get("end_time") is not None:
        period_payload.setdefault("end_time", current_period.get("end_time"))

    analysis_mode_parameters: Dict[str, Any] = {
        "name": analysis_mode,
        "drill_metric": [drill_metric] if drill_metric else [],
        "period": period_payload,
    }

    filters = _filter_conditions(dimension_selections, extra_filters)

    intent: Dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "analysis_mode_parameters": analysis_mode_parameters,
        "filters": filters,
        "group_by": [],
        "metric_hint": drill_metric,
        "raw_question": raw_question,
        "validation_warnings": [],
    }
    logger.info(
        "build_prebuilt_intent -> mode=%s metric=%s filters=%d period=%s",
        analysis_mode,
        drill_metric,
        len(filters),
        {k: v for k, v in period_payload.items() if k != "rolling_window"},
    )
    return intent


def build_conversation_id(
    *,
    view_name: Optional[str],
    lob: Optional[str],
    snap_month: Optional[int],
    period_label: Optional[str],
    end_month: Optional[int],
) -> str:
    """Build the deterministic conversation_id format from the release notebook.

    Format: ``tutorial-{view}-{lob}-{snap}-{ecap}-{end}``. Falls back to
    "unknown" for any missing segment so downstream code never sees ``None``.
    """

    ecap = _period_to_ecap_code(period_label or "") or "R0"
    parts = [
        "tutorial",
        str(view_name or "view").replace(" ", "_"),
        str(lob or "ALL").replace(" ", "_"),
        str(snap_month) if snap_month is not None else "unknown",
        ecap,
        str(end_month) if end_month is not None else "unknown",
    ]
    return "-".join(parts)


# ---------------------------------------------------------------------------
# Live filter values → orchestrator validation payload
# ---------------------------------------------------------------------------


def _coerce_time_bound(adapter: Any, dt_value: Any) -> Optional[str]:
    """Serialize a TimeAdapter's parsed bound back to the wire form Snowflake uses."""
    if dt_value is None:
        return None
    try:
        serialized = adapter.serialize(dt_value)
    except Exception:
        return None
    if serialized is None:
        return None
    return str(serialized)


def serialize_live_filter_values(filter_values: "FilterValues") -> Dict[str, Any]:
    """Project a :class:`FilterValues` into the dict shape the orchestrator expects.

    Output shape (see ``validate_intent_output`` for the contract)::

        {
            "dimensions": {
                "service_area_state": {"values": ["CA", ...], "is_free_text": False, "source": "db"},
                "rendering_provider_name": {"values": [], "is_free_text": True, "source": "db"},
            },
            "time_dimensions": {
                "snap_month": {"min": "202501", "max": "202606", "source": "db"},
            },
        }

    Free-text and high-cardinality dimensions still appear in the map so the
    orchestrator can distinguish ``"no live data, skip categorical check"``
    from ``"this field isn't in the schema"``.
    """
    dims_out: Dict[str, Dict[str, Any]] = {}
    for dv in getattr(filter_values, "dimensions", ()) or ():
        dims_out[dv.dimension_name] = {
            "values": list(dv.values),
            "is_free_text": bool(dv.is_free_text),
            "source": dv.source,
        }

    times_out: Dict[str, Dict[str, Any]] = {}
    for tr in getattr(filter_values, "time_dimensions", ()) or ():
        adapter = tr.adapter
        min_val = _coerce_time_bound(adapter, tr.min_dt)
        max_val = _coerce_time_bound(adapter, tr.max_dt)
        if min_val is None and max_val is None:
            continue
        times_out[tr.dimension_name] = {
            "min": min_val,
            "max": max_val,
            "source": tr.source,
        }

    return {"dimensions": dims_out, "time_dimensions": times_out}
