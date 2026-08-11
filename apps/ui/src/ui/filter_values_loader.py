"""Live filter-value loader backed by Snowflake.

Uses :class:`SnowparkHelper` to pull DISTINCT dimension values and MIN/MAX
time-dimension ranges so the UI dropdowns reflect what is actually in the
warehouse on each session. Falls back to the YAML's ``sample_values`` when
Snowflake is unreachable, the query fails, or the column is too high-cardinality
to enumerate safely.

Caching is session-scoped via ``@st.cache_data(ttl=3600)``: one fan-out of
queries per Streamlit session, refreshed hourly.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import streamlit as st

from ui.semantic_schema import FilterDimension, SemanticSchema, TimeFilterDimension
from ui.time_adapters import (
    TimeAdapter,
    YearMonthIntAdapter,
    default_adapter_for_yaml_type,
    select_adapter,
    synthetic_recent_range,
)

try:
    from deep_research_utils.logger_config import get_logger  # noqa: WPS433
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - logging fallback when utils unavailable
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

ValueSource = Literal["db", "yaml_fallback", "synthetic", "unknown"]

_DISTINCT_HARD_LIMIT = 1000
_DISTINCT_SOFT_LIMIT = 500

_UI_TIME_ALLOWLIST = {"snap_month"}


def _is_ui_relevant_time_dim(tdim: "TimeFilterDimension", schema: "SemanticSchema") -> bool:
    """UI probes MIN/MAX only for the principal time dim and the snapshot selector.

    Why: the YAML declares many time_dimensions (claim_line_*, um_*, ...) but the
    Scope panel only surfaces snap_month + principal as first-class widgets, so
    probing the rest wastes a serial Snowflake round-trip per dim on first render.
    """
    if schema.principal_time_dimension and tdim.name == schema.principal_time_dimension:
        return True
    return tdim.name in _UI_TIME_ALLOWLIST


@dataclass(frozen=True)
class DimensionValues:
    """Resolved filter values for a single dimension."""

    dimension_name: str
    values: Tuple[str, ...]
    is_free_text: bool
    source: ValueSource
    warning: Optional[str] = None


@dataclass(frozen=True)
class TimeDimensionRange:
    """Resolved MIN/MAX + adapter for a single time dimension."""

    dimension_name: str
    min_dt: dt.datetime
    max_dt: dt.datetime
    adapter: TimeAdapter
    source: ValueSource
    warning: Optional[str] = None


@dataclass(frozen=True)
class FilterValues:
    """All live values for a SemanticSchema, plus diagnostic metadata."""

    dimensions: Tuple[DimensionValues, ...]
    time_dimensions: Tuple[TimeDimensionRange, ...]
    snowflake_enabled: bool
    fetch_seconds: float
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@st.cache_resource(show_spinner=False)
def _get_snowpark():
    """Return a process-singleton SnowparkHelper, or None if unavailable."""
    try:
        from deep_research_utils.snowflake_helper import SnowparkHelper  # noqa: WPS433

        return SnowparkHelper()
    except Exception as exc:  # pragma: no cover - env-dependent
        logger.warning("Snowflake helper unavailable: %s", exc)
        return None


def _snowflake_enabled() -> bool:
    from deep_research_utils.app_constant import AppConstants  # noqa: WPS433

    return bool(AppConstants.SNOWFLAKE_ACCOUNT and (AppConstants.SNOWFLAKE_USER or AppConstants.SNOWFLAKE_SECRET))


def _qualified_table(dim: FilterDimension | TimeFilterDimension) -> Optional[str]:
    ref = dim.primary_source
    if ref is None:
        return None
    parts = [p for p in (ref.database, ref.schema, ref.physical_table) if p]
    if not parts:
        return None
    return ".".join(parts)


def _yaml_fallback_dimension(dim: FilterDimension, warning: str) -> DimensionValues:
    return DimensionValues(
        dimension_name=dim.name,
        values=dim.sample_values,
        is_free_text=not dim.sample_values,
        source="yaml_fallback",
        warning=warning,
    )


def _execute(sf, query: str, *, purpose: str, dimension_name: str):
    """Run a query, returning a Pandas DataFrame. Logs INFO + raises on failure."""

    logger.info(
        "filter-loader SQL [%s] dimension=%s | %s",
        purpose,
        dimension_name,
        " ".join(query.split()),
    )
    start = time.monotonic()
    try:
        df = sf.execute_query_and_return_pandas_df(query)
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.warning(
            "filter-loader SQL [%s] FAILED dimension=%s after %.2fs: %s",
            purpose,
            dimension_name,
            elapsed,
            exc,
        )
        raise
    elapsed = time.monotonic() - start
    rows = 0 if df is None else len(df)
    logger.info(
        "filter-loader SQL [%s] dimension=%s returned %d rows in %.2fs",
        purpose,
        dimension_name,
        rows,
        elapsed,
    )
    return df


def _pull_distinct(sf, dim: FilterDimension, table_fqn: str) -> List[str]:
    expr = dim.primary_source.expr  # type: ignore[union-attr]
    query = (
        f"SELECT DISTINCT {expr} AS value "
        f"FROM {table_fqn} "
        f"WHERE {expr} IS NOT NULL "
        f"ORDER BY {expr} "
        f"LIMIT {_DISTINCT_HARD_LIMIT}"
    )
    df = _execute(sf, query, purpose="distinct", dimension_name=dim.name)
    if df is None or df.empty:
        return []
    column = df.columns[0]
    return [str(value) for value in df[column].tolist() if value is not None]


def _count_distinct(sf, dim: FilterDimension, table_fqn: str) -> int:
    expr = dim.primary_source.expr  # type: ignore[union-attr]
    query = f"SELECT COUNT(DISTINCT {expr}) AS cnt FROM {table_fqn}"
    df = _execute(sf, query, purpose="count_distinct", dimension_name=dim.name)
    if df is None or df.empty:
        return 0
    return int(df.iloc[0, 0] or 0)


def _load_one_dimension(dim: FilterDimension) -> DimensionValues:
    if not _snowflake_enabled():
        return _yaml_fallback_dimension(dim, "Snowflake credentials not set")

    sf = _get_snowpark()
    if sf is None:
        return _yaml_fallback_dimension(dim, "Snowflake helper failed to initialize")

    table_fqn = _qualified_table(dim)
    if not table_fqn:
        return _yaml_fallback_dimension(dim, "No base table for this dimension")

    yaml_hints_safe = dim.is_enum or bool(dim.sample_values)

    try:
        if yaml_hints_safe:
            values = _pull_distinct(sf, dim, table_fqn)
        else:
            count = _count_distinct(sf, dim, table_fqn)
            if count == 0:
                return _yaml_fallback_dimension(dim, "Probe returned 0 distinct values")
            if count > _DISTINCT_SOFT_LIMIT:
                return DimensionValues(
                    dimension_name=dim.name,
                    values=(),
                    is_free_text=True,
                    source="db",
                    warning=f"{count} distinct values — rendered as free-text input",
                )
            values = _pull_distinct(sf, dim, table_fqn)
    except Exception as exc:
        logger.warning("DISTINCT pull failed for %s: %s", dim.name, exc)
        return _yaml_fallback_dimension(dim, f"Query failed: {exc.__class__.__name__}")

    if len(values) > _DISTINCT_SOFT_LIMIT:
        return DimensionValues(
            dimension_name=dim.name,
            values=(),
            is_free_text=True,
            source="db",
            warning=f"{len(values)} distinct values — rendered as free-text input",
        )

    return DimensionValues(
        dimension_name=dim.name,
        values=tuple(values),
        is_free_text=not values,
        source="db",
    )


def _load_one_time_dimension(dim: TimeFilterDimension) -> TimeDimensionRange:
    if not _snowflake_enabled():
        return _synthetic_time_range(dim, "Snowflake credentials not set")

    sf = _get_snowpark()
    if sf is None:
        return _synthetic_time_range(dim, "Snowflake helper failed to initialize")

    table_fqn = _qualified_table(dim)
    if not table_fqn:
        return _synthetic_time_range(dim, "No base table for this time dimension")

    expr = dim.primary_source.expr  # type: ignore[union-attr]
    try:
        query = (
            f"SELECT MIN({expr}) AS min_val, MAX({expr}) AS max_val "
            f"FROM {table_fqn} WHERE {expr} IS NOT NULL"
        )
        df = _execute(sf, query, purpose="min_max", dimension_name=dim.name)
    except Exception as exc:
        return _synthetic_time_range(dim, f"Query failed: {exc.__class__.__name__}")

    if df is None or df.empty:
        return _synthetic_time_range(dim, "Probe returned no rows")

    min_val = df.iloc[0, 0]
    max_val = df.iloc[0, 1]
    adapter = select_adapter(min_val, max_val, dim.data_type)
    if adapter is None:
        return _synthetic_time_range(dim, "No adapter matched DB values")

    try:
        min_dt = adapter.parse(min_val)
        max_dt = adapter.parse(max_val)
    except Exception as exc:
        logger.warning("Adapter parse failed for %s: %s", dim.name, exc)
        return _synthetic_time_range(dim, f"Adapter parse failed: {exc.__class__.__name__}")

    return TimeDimensionRange(
        dimension_name=dim.name,
        min_dt=min_dt,
        max_dt=max_dt,
        adapter=adapter,
        source="db",
    )


def _synthetic_time_range(dim: TimeFilterDimension, warning: str) -> TimeDimensionRange:
    adapter = default_adapter_for_yaml_type(dim.data_type)
    min_dt, max_dt = synthetic_recent_range(months_back=24)
    return TimeDimensionRange(
        dimension_name=dim.name,
        min_dt=min_dt,
        max_dt=max_dt,
        adapter=adapter,
        source="synthetic",
        warning=warning,
    )


@st.cache_data(ttl=3600, show_spinner="Loading filter values from warehouse...")
def load_all_filter_values(schema: SemanticSchema) -> FilterValues:
    """Fan out DISTINCT and MIN/MAX queries for every dimension, cached for 1 hour."""

    start = time.monotonic()
    sf_enabled = _snowflake_enabled()
    warnings: List[str] = []

    selected_tdims = tuple(
        t for t in schema.time_dimensions if _is_ui_relevant_time_dim(t, schema)
    )

    logger.info(
        "filter-loader: initial load for view='%s' | dimensions=%d time_dimensions=%d selected_time_dimensions=%d snowflake_enabled=%s",
        schema.view_name,
        len(schema.dimensions),
        len(schema.time_dimensions),
        len(selected_tdims),
        sf_enabled,
    )

    dim_values = tuple(_load_one_dimension(dim) for dim in schema.dimensions)
    time_ranges = tuple(_load_one_time_dimension(tdim) for tdim in selected_tdims)

    if not sf_enabled:
        warnings.append("Snowflake disabled — all filters using YAML sample_values")

    elapsed = time.monotonic() - start
    logger.info(
        "filter-loader: initial load complete in %.2fs | db=%d yaml_fallback=%d synthetic=%d",
        elapsed,
        sum(1 for dv in dim_values if dv.source == "db"),
        sum(1 for dv in dim_values if dv.source == "yaml_fallback"),
        sum(1 for tr in time_ranges if tr.source == "synthetic"),
    )
    return FilterValues(
        dimensions=dim_values,
        time_dimensions=time_ranges,
        snowflake_enabled=sf_enabled,
        fetch_seconds=elapsed,
        warnings=tuple(warnings),
    )
