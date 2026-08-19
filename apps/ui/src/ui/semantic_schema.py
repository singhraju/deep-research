"""Parse correlation_pattern YAML into a UI-friendly schema view.

Pure parsing — no Streamlit and no Snowflake imports — so this module is easy
to unit-test and can be reused by other surfaces (CLI, tests, notebooks).

The UI cares about three things:
  * which dimensions to render as filters (curated by ``analysis_modes[*].drill_dimensions``)
  * which time-dimension to anchor period selection on (intersection across all tables)
  * which metrics to expose in the Metric dropdown (``drill_metric`` + ``explainer_metrics``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class TableRef:
    """Pointer back to a base table that exposes a given column."""

    table_name: str
    database: str
    schema: str
    physical_table: str
    expr: str


@dataclass(frozen=True)
class FilterDimension:
    """A dimension surfaced as a UI filter."""

    name: str
    description: str
    data_type: str
    is_enum: bool
    sample_values: Tuple[str, ...]
    synonyms: Tuple[str, ...]
    source_tables: Tuple[TableRef, ...]

    @property
    def primary_source(self) -> Optional[TableRef]:
        return self.source_tables[0] if self.source_tables else None


@dataclass(frozen=True)
class TimeFilterDimension:
    """A time-typed dimension shared across every table in the YAML."""

    name: str
    description: str
    data_type: str
    synonyms: Tuple[str, ...]
    source_tables: Tuple[TableRef, ...]

    @property
    def primary_source(self) -> Optional[TableRef]:
        return self.source_tables[0] if self.source_tables else None


@dataclass(frozen=True)
class MetricOption:
    """A metric the user can pick as the drill-down target."""

    name: str
    description: str
    expr: str
    is_drill_metric: bool


@dataclass(frozen=True)
class SemanticSchema:
    """Everything the UI needs to render filters for a YAML."""

    yaml_path: str
    view_name: str
    description: str
    dimensions: Tuple[FilterDimension, ...]
    time_dimensions: Tuple[TimeFilterDimension, ...]
    metrics: Tuple[MetricOption, ...]
    default_drill_metric: Optional[str]
    principal_time_dimension: Optional[str] = field(default=None)
    principal_time_dimension_qualified: Optional[str] = field(default=None)
    analysis_mode_name: Optional[str] = field(default=None)


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _build_table_ref(table_def: Dict[str, Any], expr: str) -> TableRef:
    base = table_def.get("base_table") or {}
    return TableRef(
        table_name=str(table_def.get("name") or ""),
        database=str(base.get("database") or ""),
        schema=str(base.get("schema") or ""),
        physical_table=str(base.get("table") or ""),
        expr=expr,
    )


def _curated_dimension_names(yaml_doc: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for mode in yaml_doc.get("analysis_modes") or []:
        for name in mode.get("drill_dimensions") or []:
            if name and name not in seen:
                seen[name] = None
    return list(seen.keys())


def _collect_metrics_index(yaml_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup of metric name → metric definition.

    Top-level ``metrics`` use their bare ``name`` (e.g. ``paid_pmpm``).
    Per-table ``metrics`` use the dotted ``name`` already encoded in the YAML
    (e.g. ``expense_detail.total_paid``).
    """

    index: Dict[str, Dict[str, Any]] = {}
    for metric in yaml_doc.get("metrics") or []:
        name = metric.get("name")
        if name:
            index[str(name)] = metric
    for table in yaml_doc.get("tables") or []:
        for metric in table.get("metrics") or []:
            name = metric.get("name")
            if name:
                index[str(name)] = metric
    return index


def _curated_metric_names(yaml_doc: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    """Return (curated metric names in display order, default drill metric)."""

    drill_metrics: List[str] = []
    explainer_metrics: List[str] = []
    default: Optional[str] = None
    for mode in yaml_doc.get("analysis_modes") or []:
        for name in mode.get("drill_metric") or []:
            if name and name not in drill_metrics:
                drill_metrics.append(str(name))
                if default is None:
                    default = str(name)
        for name in mode.get("explainer_metrics") or []:
            if name and name not in explainer_metrics and name not in drill_metrics:
                explainer_metrics.append(str(name))
    return drill_metrics + explainer_metrics, default


def _parse_dimensions(
    yaml_doc: Dict[str, Any],
    curated_names: List[str],
) -> List[FilterDimension]:
    accumulator: Dict[str, Dict[str, Any]] = {
        name: {
            "description": "",
            "data_type": "string",
            "is_enum": False,
            "sample_values": [],
            "synonyms": [],
            "source_tables": [],
        }
        for name in curated_names
    }

    for table_def in yaml_doc.get("tables") or []:
        for dim in table_def.get("dimensions") or []:
            dim_name = dim.get("name")
            if not dim_name or dim_name not in accumulator:
                continue
            entry = accumulator[dim_name]
            if not entry["description"]:
                entry["description"] = str(dim.get("description") or "")
            if dim.get("data_type"):
                entry["data_type"] = str(dim["data_type"])
            entry["is_enum"] = bool(entry["is_enum"] or dim.get("is_enum"))
            for value in dim.get("sample_values") or []:
                if value is None:
                    continue
                str_val = str(value)
                if str_val not in entry["sample_values"]:
                    entry["sample_values"].append(str_val)
            for syn in dim.get("synonyms") or []:
                if syn and syn not in entry["synonyms"]:
                    entry["synonyms"].append(str(syn))
            entry["source_tables"].append(
                _build_table_ref(table_def, str(dim.get("expr") or dim_name))
            )

    return [
        FilterDimension(
            name=name,
            description=entry["description"],
            data_type=entry["data_type"],
            is_enum=entry["is_enum"],
            sample_values=tuple(entry["sample_values"]),
            synonyms=tuple(entry["synonyms"]),
            source_tables=tuple(entry["source_tables"]),
        )
        for name, entry in accumulator.items()
        if entry["source_tables"]
    ]


def _parse_time_dimensions(yaml_doc: Dict[str, Any]) -> List[TimeFilterDimension]:
    """Collect every time dimension surfaced by any table in the YAML.

    Prior behaviour required a time dim to appear in EVERY table's
    ``time_dimensions:`` list (set intersection). That dropped fields like
    ``snap_month`` which the YAML only declares as a time dim on one table
    (``expense_detail``) even though the other table tracks the same column
    under its ``dimensions:`` list. The UI needs ``snap_month`` available as a
    time widget so the Research payload can carry the correct snapshot anchor;
    using the union keeps backwards compatibility for tables that share names
    and surfaces single-table time dims that the prior gate hid.
    """

    tables = yaml_doc.get("tables") or []
    if not tables:
        return []

    accumulator: Dict[str, Dict[str, Any]] = {}

    for table_def in tables:
        for tdim in table_def.get("time_dimensions") or []:
            name = tdim.get("name")
            if not name:
                continue
            entry = accumulator.setdefault(
                name,
                {
                    "description": "",
                    "data_type": "string",
                    "synonyms": [],
                    "source_tables": [],
                },
            )
            if not entry["description"]:
                entry["description"] = str(tdim.get("description") or "")
            if tdim.get("data_type"):
                entry["data_type"] = str(tdim["data_type"])
            for syn in tdim.get("synonyms") or []:
                if syn and syn not in entry["synonyms"]:
                    entry["synonyms"].append(str(syn))
            entry["source_tables"].append(
                _build_table_ref(table_def, str(tdim.get("expr") or name))
            )

    return [
        TimeFilterDimension(
            name=name,
            description=entry["description"],
            data_type=entry["data_type"],
            synonyms=tuple(entry["synonyms"]),
            source_tables=tuple(entry["source_tables"]),
        )
        for name, entry in accumulator.items()
    ]


def _parse_metrics(yaml_doc: Dict[str, Any]) -> Tuple[List[MetricOption], Optional[str]]:
    metric_index = _collect_metrics_index(yaml_doc)
    curated_names, default = _curated_metric_names(yaml_doc)
    drill_set = set()
    for mode in yaml_doc.get("analysis_modes") or []:
        for name in mode.get("drill_metric") or []:
            if name:
                drill_set.add(str(name))

    metrics: List[MetricOption] = []
    for name in curated_names:
        meta = metric_index.get(name, {})
        metrics.append(
            MetricOption(
                name=name,
                description=str(meta.get("description") or ""),
                expr=str(meta.get("expr") or ""),
                is_drill_metric=name in drill_set,
            )
        )
    return metrics, default


def _pick_principal_time_dimension(
    yaml_doc: Dict[str, Any],
    time_dimensions: List[TimeFilterDimension],
) -> Tuple[Optional[str], Optional[str]]:
    """Use the first analysis_mode's ``period.rolling_time_dimension`` when present.

    The YAML stores it as ``table.column`` (e.g., ``expense_detail.incurred_month``);
    we drop the table prefix for ``principal_time_dimension`` (used to address the
    UI widget) and keep the full ``table.column`` form as the qualified name
    (used inside the correlation payload's ``period.rolling_time_dimension``).
    Falls back to the first shared time dimension if no preference is recorded.
    """

    shared_names = {td.name for td in time_dimensions}
    for mode in yaml_doc.get("analysis_modes") or []:
        period = mode.get("period") or {}
        rolling = period.get("rolling_time_dimension")
        if not rolling:
            continue
        bare = rolling.split(".", 1)[-1]
        if bare in shared_names:
            return bare, str(rolling)
    if time_dimensions:
        return time_dimensions[0].name, None
    return None, None


def _pick_analysis_mode_name(yaml_doc: Dict[str, Any]) -> Optional[str]:
    for mode in yaml_doc.get("analysis_modes") or []:
        name = mode.get("name")
        if name:
            return str(name)
    return None


def load_semantic_schema(yaml_path: str | Path) -> SemanticSchema:
    """Parse a correlation_pattern YAML into a ``SemanticSchema``."""

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as handle:
        doc: Dict[str, Any] = yaml.safe_load(handle) or {}

    curated_names = _curated_dimension_names(doc)
    dimensions = _parse_dimensions(doc, curated_names)
    time_dimensions = _parse_time_dimensions(doc)
    metrics, default_drill_metric = _parse_metrics(doc)
    principal_time, principal_time_qualified = _pick_principal_time_dimension(doc, time_dimensions)
    analysis_mode_name = _pick_analysis_mode_name(doc)

    return SemanticSchema(
        yaml_path=str(path),
        view_name=str(doc.get("name") or path.stem),
        description=str(doc.get("description") or ""),
        dimensions=tuple(dimensions),
        time_dimensions=tuple(time_dimensions),
        metrics=tuple(metrics),
        default_drill_metric=default_drill_metric,
        principal_time_dimension=principal_time,
        principal_time_dimension_qualified=principal_time_qualified,
        analysis_mode_name=analysis_mode_name,
    )
