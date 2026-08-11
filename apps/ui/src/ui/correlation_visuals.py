"""Render correlation results (waterfall + interaction matrix) in Streamlit.

Mirrors the static HTML viewer shipped with correlation reports so the in-app
output looks the same as the standalone HTML the analytics team distributes.
The expected data shape is the ``correlation_summary`` payload at
``output['analysis']['metadata']['correlation_summary']`` — same JSON the static
viewer consumes (``baseline_value``, ``comparison_value``, ``drill_path``,
``interaction_matrix``).
"""

from __future__ import annotations

import datetime as dt
import html
import math
import urllib.parse
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st

from deep_research_utils.metric_role import classify_metric, humanize_metric_name

_POSITIVE_COLORS = ["#28a745", "#20c997", "#17a2b8", "#48c9b0", "#52b788"]
_NEGATIVE_COLORS = ["#dc3545", "#e83e8c", "#fd7e14", "#e74c3c", "#c0392b"]

_SIGNAL_LABELS = {
    "volume_plus_paid_ratio": "Volume Plus Paid Ratio",
    "volume": "Volume",
    "unit_cost": "Unit Cost",
    "paid_ratio": "Paid Ratio",
    "mixed": "Mixed",
}


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def format_compact_money(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"${abs_val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"${round(abs_val / 1_000)}K"
    return f"${round(abs_val)}"


def format_signed_money(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    sign = "+" if val >= 0 else "-"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{sign}${abs_val / 1_000:.1f}K"
    return f"{sign}${abs_val:.2f}"


def format_currency(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:.2f}"


def format_percent(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def format_pct_points(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    pts = val * 100
    sign = "+" if pts >= 0 else ""
    return f"{sign}{pts:.1f} pts"


def format_signed_integer(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "—"
    rounded = int(round(val))
    sign = "+" if rounded >= 0 else ""
    return f"{sign}{rounded:,}"


def format_period_window(period_window: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Format baseline and comparison periods for display.
    
    Args:
        period_window: Dictionary with baseline_months and comparison_months arrays
        
    Returns:
        Formatted string like "Baseline: Jan 2025 - Mar 2025 | Comparison: Jan 2024 - Mar 2024"
        or None if data is missing or malformed
    """
    if not period_window:
        return None
    
    baseline_months = period_window.get("baseline_months", [])
    comparison_months = period_window.get("comparison_months", [])
    
    if not baseline_months or not comparison_months:
        return None
    
    def _format_month(value: Any) -> Optional[str]:
        """Format a single month value to 'Mon YYYY' format."""
        if value is None:
            return None
            
        # Handle YYYYMM integer format (e.g., 202501)
        if isinstance(value, int):
            year = value // 100
            month = value % 100
            if 1 <= month <= 12:
                return dt.date(year, month, 1).strftime("%b %Y")
            return None
        
        # Handle string date formats
        if isinstance(value, str):
            try:
                # Try parsing as ISO date
                parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
                return parsed.strftime("%b %Y")
            except (ValueError, AttributeError):
                # Try YYYYMM string format
                try:
                    if len(value) == 6 and value.isdigit():
                        year = int(value[:4])
                        month = int(value[4:6])
                        if 1 <= month <= 12:
                            return dt.date(year, month, 1).strftime("%b %Y")
                except (ValueError, AttributeError):
                    pass
                return None
        
        # Handle datetime objects
        if isinstance(value, (dt.date, dt.datetime)):
            return value.strftime("%b %Y")
        
        return None
    
    # Extract start and end from arrays
    baseline_start = _format_month(baseline_months[0]) if baseline_months else None
    baseline_end = _format_month(baseline_months[-1]) if baseline_months else None
    comparison_start = _format_month(comparison_months[0]) if comparison_months else None
    comparison_end = _format_month(comparison_months[-1]) if comparison_months else None
    
    if not all([baseline_start, baseline_end, comparison_start, comparison_end]):
        return None
    
    return f"Baseline: {baseline_start} - {baseline_end} | Comparison: {comparison_start} - {comparison_end}"


def _pa_label(value: Optional[str]) -> str:
    if not value or value == "<NULL>":
        return "Unknown"
    if value == "N":
        return "No PA"
    if value == "Y":
        return "PA Required"
    return value


def _title_case_facility(value: Optional[str]) -> str:
    if not value or value == "<NULL>":
        return "Unknown"
    if value == "ACUTE HOSPITAL":
        return "Acute Hospital"
    return " ".join(word.capitalize() for word in value.split(" "))


# Optional per-key value formatters for legacy dimension vocabularies. Applied
# only when the key name matches; unknown keys fall through to str(value).
_DIM_VALUE_FORMATTERS: Dict[str, Callable[[str], str]] = {
    "pa_required_code": lambda v: _pa_label(v),
    "facility_type": lambda v: _title_case_facility(v),
}


def _build_dim_label_map(dim_specs: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Build dimension name -> friendly label map from schema specs.
    
    Extracts labels from the interaction_matrix.schema dimension specs.
    Falls back to humanized dimension names when labels are not available.
    """
    label_map: Dict[str, str] = {}
    for spec in dim_specs:
        name = str(spec.get("name", "")).strip()
        label = str(spec.get("label", "")).strip()
        if name:
            label_map[name] = label or str(name).replace("_", " ").title()
    return label_map


def _resolve_dim_label(key: str, label_map: Optional[Dict[str, str]] = None) -> str:
    """Business-friendly label for a raw dimension key.

    Resolution order: provided label_map (from schema), else humanized key.
    """
    if label_map and key in label_map:
        return label_map[key]
    return str(key).replace("_", " ").title()


def _format_dim_value(key: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "<NULL>":
        return None
    formatter = _DIM_VALUE_FORMATTERS.get(key)
    if formatter is not None:
        text = formatter(text)
    if not text or text == "Unknown":
        return None
    return text


def _build_dim_pairs(
    cell: Mapping[str, Any],
    dim_order: Sequence[str] = (),
    label_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Return ``(friendly_label, formatted_value)`` pairs for a cell's dimensions.

    ``dim_order`` optionally provides the preferred column order (from the
    interaction-matrix schema); any additional keys the cell carries are
    appended in insertion order so nothing is silently dropped. Values that
    are null/``<NULL>``/``Unknown`` are filtered by ``_format_dim_value`` and
    drop the whole pair, so no orphan label is ever emitted.
    """
    dims = cell.get("dimension_values") or {}
    if not isinstance(dims, Mapping):
        return []
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for key in dim_order:
        if key in dims:
            seen.add(key)
            formatted = _format_dim_value(key, dims[key])
            if formatted:
                pairs.append((_resolve_dim_label(key, label_map), formatted))
    for key, value in dims.items():
        if key in seen:
            continue
        formatted = _format_dim_value(key, value)
        if formatted:
            pairs.append((_resolve_dim_label(key, label_map), formatted))
    return pairs


def _build_dim_label(
    cell: Mapping[str, Any],
    dim_order: Sequence[str] = (),
    label_map: Optional[Dict[str, str]] = None,
) -> str:
    """Plain-text ``Label: value · …`` join of a cell's dimensions.

    Used for the HTML ``title=`` tooltip / alt text and any non-HTML consumer;
    the tables render the stacked HTML/PDF form via ``_build_dim_pairs``.
    """
    return " · ".join(f"{label}: {value}" for label, value in _build_dim_pairs(cell, dim_order, label_map))


def _render_dim_pairs_html(pairs: Sequence[Tuple[str, str]]) -> str:
    """Render ``(label, value)`` pairs as a stacked two-column HTML grid.

    Each label and value is individually escaped, so the returned string is
    safe to insert raw into a ``<td>`` (do NOT wrap it in ``_esc`` again).
    """
    if not pairs:
        return "—"
    body = "".join(
        f'<span style="color:#6B7280; white-space:nowrap;">{_esc(label)}</span>'
        f'<span style="color:#212529;">{_esc(value)}</span>'
        for label, value in pairs
    )
    return (
        '<div style="display:grid; grid-template-columns:max-content 1fr; '
        'column-gap:14px; row-gap:3px; line-height:1.35;">' + body + "</div>"
    )


def _get_metric_delta(cell: Mapping[str, Any], metric_name: str) -> Optional[float]:
    explainers = cell.get("explainer_metrics") or {}
    metric = explainers.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    return _coerce_float(metric.get("delta"))


def _get_metric_baseline(cell: Mapping[str, Any], metric_name: str) -> Optional[float]:
    explainers = cell.get("explainer_metrics") or {}
    metric = explainers.get(metric_name)
    if not isinstance(metric, Mapping):
        return None
    return _coerce_float(metric.get("baseline"))


def _find_metric_by_role(
    explainer_specs: Sequence[Mapping[str, Any]], role: str
) -> Optional[str]:
    for spec in explainer_specs:
        if str(spec.get("role", "")) == role:
            name = str(spec.get("name", "")).strip()
            if name:
                return name
    return None


def _derive_signal(
    cell: Mapping[str, Any],
    explainer_specs: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Classify a cell's movement by consulting the config's own explainer
    metrics. ``explainer_specs`` comes from ``interaction_matrix.schema`` and
    tells us which metric fills the "volume", "unit cost", "ratio" roles.
    """
    volume_key = _find_metric_by_role(explainer_specs, "admissions") or _find_metric_by_role(explainer_specs, "count")
    unit_cost_key = _find_metric_by_role(explainer_specs, "currency")
    ratio_key = _find_metric_by_role(explainer_specs, "ratio")

    volume_delta = _get_metric_delta(cell, volume_key) if volume_key else None
    unit_cost_delta = _get_metric_delta(cell, unit_cost_key) if unit_cost_key else None
    ratio_delta = _get_metric_delta(cell, ratio_key) if ratio_key else None
    volume_baseline = _get_metric_baseline(cell, volume_key) if volume_key else None

    volume_delta = volume_delta or 0.0
    unit_cost_delta = unit_cost_delta or 0.0
    ratio_delta = ratio_delta or 0.0
    volume_baseline = volume_baseline or 1.0

    volume_material = (
        abs(volume_delta) >= 50
        or (abs(volume_delta) / max(volume_baseline, 1)) >= 0.05
    )
    unit_cost_material = abs(unit_cost_delta) >= 500
    ratio_material = abs(ratio_delta) >= 0.01

    if volume_material and ratio_material:
        return "volume_plus_paid_ratio"
    if volume_material and not unit_cost_material and not ratio_material:
        return "volume"
    if unit_cost_material and not volume_material:
        return "unit_cost"
    if ratio_material and not volume_material:
        return "paid_ratio"
    return "mixed"


def _signal_label(signal: str) -> str:
    return _SIGNAL_LABELS.get(signal, signal.replace("_", " ").title())


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _build_waterfall_chart_html(correlation_summary: Mapping[str, Any]) -> Optional[str]:
    drill_path = _safe_list(correlation_summary.get("drill_path"))
    baseline = _coerce_float(correlation_summary.get("baseline_value"))
    comparison = _coerce_float(correlation_summary.get("comparison_value"))
    if baseline is None or comparison is None or not drill_path:
        return None

    metric_label = (
        correlation_summary.get("metric_label")
        or correlation_summary.get("root_metric")
        or "Metric"
    )

    points: List[Dict[str, Any]] = []
    cumulative = baseline
    points.append({
        "label": "Baseline",
        "value": baseline,
        "cumulative": cumulative,
        "is_total": True,
    })

    for level in drill_path:
        if not isinstance(level, Mapping):
            continue
        top_segments = _safe_list(level.get("top_segments"))[:5]
        if not top_segments:
            continue
        segments: List[Dict[str, Any]] = []
        for seg in top_segments:
            if not isinstance(seg, Mapping):
                continue
            delta = _coerce_float(seg.get("aligned_delta"))
            if delta is None:
                delta = _coerce_float(seg.get("delta_value")) or 0.0
            segments.append({
                "value": str(seg.get("value", "")),
                "delta": delta,
            })
        if not segments:
            continue
        total_delta = sum(s["delta"] for s in segments)
        cumulative += total_delta
        points.append({
            "label": f"Level {level.get('level', '?')}: {level.get('dimension', '')}",
            "total": total_delta,
            "cumulative": cumulative,
            "segments": segments,
            "is_total": False,
        })

    points.append({
        "label": "Comparison",
        "value": comparison,
        "cumulative": comparison,
        "is_total": True,
    })

    svg_width = max(1000, len(points) * 120)
    svg_height = 500
    margin_top, margin_right, margin_bottom, margin_left = 40, 40, 120, 80
    chart_width = svg_width - margin_left - margin_right
    chart_height = svg_height - margin_top - margin_bottom

    all_values: List[float] = []
    for p in points:
        if p["is_total"]:
            all_values.append(p["cumulative"])
        else:
            all_values.append(p["cumulative"])
            all_values.append(p["cumulative"] - p["total"])
    if not all_values:
        return None
    max_value = max(all_values)
    min_value = min(all_values)
    span = max_value - min_value
    y_scale = chart_height / span if span > 0 else 0
    bar_width = min(80.0, chart_width / (len(points) + 1))
    bar_spacing = chart_width / (len(points) + 1)

    parts: List[str] = []
    parts.append(
        f'<svg viewBox="0 0 {svg_width} {svg_height}" style="width:100%; max-height:500px;">'
        '<defs><linearGradient id="totalGradient" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" style="stop-color:#667eea;stop-opacity:1"/>'
        '<stop offset="100%" style="stop-color:#764ba2;stop-opacity:1"/>'
        '</linearGradient></defs>'
    )

    for index, point in enumerate(points):
        cx = margin_left + (index + 0.5) * bar_spacing

        if point["is_total"]:
            value = point["value"]
            bar_height = abs(value) * y_scale
            y = margin_top + (max_value - point["cumulative"]) * y_scale
            parts.append(
                f'<rect x="{cx - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="url(#totalGradient)" stroke="#fff" '
                f'stroke-width="2" rx="4"><title>{_esc(point["label"])}: '
                f'{_esc(format_currency(value))}</title></rect>'
            )
            label_y = svg_height - margin_bottom + 15
            parts.append(
                f'<text x="{cx:.1f}" y="{label_y}" font-size="11" fill="#495057" '
                f'text-anchor="end" transform="rotate(-45, {cx:.1f}, {label_y})">'
                f'{_esc(point["label"])}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y - 10:.1f}" font-size="12" font-weight="600" '
                f'fill="#212529" text-anchor="middle">{_esc(format_currency(value))}</text>'
            )
        else:
            total_delta = point["total"]
            segments = point["segments"]
            base_cumulative = point["cumulative"] - total_delta
            is_positive = total_delta >= 0
            colors = _POSITIVE_COLORS if is_positive else _NEGATIVE_COLORS

            if is_positive:
                running = base_cumulative
                for seg_index, segment in enumerate(segments):
                    seg_height = max(3.0, abs(segment["delta"]) * y_scale)
                    seg_top = margin_top + (max_value - running - segment["delta"]) * y_scale
                    color = colors[seg_index % len(colors)]
                    parts.append(
                        f'<rect x="{cx - bar_width / 2:.1f}" y="{seg_top:.1f}" '
                        f'width="{bar_width:.1f}" height="{seg_height:.1f}" '
                        f'fill="{color}" stroke="#fff" stroke-width="1" rx="2">'
                        f'<title>{_esc(segment["value"])}: '
                        f'{_esc(format_signed_money(segment["delta"]))}</title></rect>'
                    )
                    running += segment["delta"]
            else:
                running = base_cumulative
                for seg_index, segment in enumerate(segments):
                    seg_height = max(3.0, abs(segment["delta"]) * y_scale)
                    seg_top = margin_top + (max_value - running) * y_scale
                    color = colors[seg_index % len(colors)]
                    parts.append(
                        f'<rect x="{cx - bar_width / 2:.1f}" y="{seg_top:.1f}" '
                        f'width="{bar_width:.1f}" height="{seg_height:.1f}" '
                        f'fill="{color}" stroke="#fff" stroke-width="1" rx="2">'
                        f'<title>{_esc(segment["value"])}: '
                        f'{_esc(format_signed_money(segment["delta"]))}</title></rect>'
                    )
                    running += segment["delta"]

            connector_y = margin_top + (max_value - point["cumulative"]) * y_scale
            parts.append(
                f'<text x="{cx:.1f}" y="{connector_y - 10:.1f}" font-size="12" '
                f'font-weight="600" fill="#212529" text-anchor="middle">'
                f'{_esc(format_currency(point["cumulative"]))}</text>'
            )
            label_y = svg_height - margin_bottom + 15
            parts.append(
                f'<text x="{cx:.1f}" y="{label_y}" font-size="11" fill="#495057" '
                f'text-anchor="end" transform="rotate(-45, {cx:.1f}, {label_y})">'
                f'{_esc(point["label"])}</text>'
            )

    parts.append("</svg>")

    title = f"Waterfall Chart: {_esc(metric_label)}"
    return (
        '<div style="background:white; border-radius:8px; padding:20px; '
        'box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:20px;">'
        f'<div style="text-align:center; font-size:1.2em; font-weight:600; '
        f'color:#212529; margin-bottom:8px;">📈 {title}</div>'
        '<p style="text-align:center; color:#6c757d; font-size:0.9em; margin-bottom:15px;">'
        'Each level bar is stacked showing top contributors (up to 5). '
        'Hover over segments for details.</p>'
        + "".join(parts)
        + "</div>"
    )


def _sum_field(cells: Sequence[Mapping[str, Any]], field: str) -> float:
    total = 0.0
    for cell in cells:
        val = _coerce_float(cell.get(field))
        if val is not None:
            total += val
    return total


def _heatmap_title(
    cells: Sequence[Mapping[str, Any]],
    dim_specs: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Assemble a title from any invariant dimension values across the cells.

    Falls back to "Selected cells" when no dimension is constant across all
    cells or when the cell list is empty. Uses dimension display labels from
    ``dim_specs`` when available.
    """
    if not cells:
        return "Selected cells"
    first_dims = cells[0].get("dimension_values") or {}
    label_by_name = {str(spec.get("name", "")): str(spec.get("label", "")) for spec in dim_specs}
    invariant_parts: List[str] = []
    for key, first_value in first_dims.items():
        if first_value in (None, "", "<NULL>", "Unknown"):
            continue
        same = all(
            (cell.get("dimension_values") or {}).get(key) == first_value
            for cell in cells
        )
        if same:
            display = str(first_value)
            invariant_parts.append(display)
            if len(invariant_parts) >= 2:
                break
    if invariant_parts:
        return f"{' · '.join(invariant_parts)} selected cells"
    return "Selected cells"


def _build_operational_heatmap_html(
    cells: Sequence[Mapping[str, Any]],
    operational_dims: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Render a 2D heatmap using the first two operational dimensions as axes.

    ``operational_dims`` is the ordered list from ``interaction_matrix.schema``.
    Falls back to the first two keys present in ``cells[0].dimension_values``
    when the schema hint is unavailable (e.g. older payloads).
    """
    if not cells:
        return ""

    dim_names = [str(spec.get("name", "")) for spec in operational_dims if spec.get("name")]
    if len(dim_names) < 2:
        first_dims = cells[0].get("dimension_values") or {}
        for key in first_dims.keys():
            if key not in dim_names:
                dim_names.append(str(key))
            if len(dim_names) >= 2:
                break
    if len(dim_names) < 2:
        return ""

    row_key, col_key = dim_names[0], dim_names[1]
    label_map = _build_dim_label_map(operational_dims)
    row_label = _resolve_dim_label(row_key, label_map)

    grid: Dict[str, Dict[str, float]] = {}
    row_totals: Dict[str, float] = {}
    col_totals: Dict[str, float] = {}

    for cell in cells:
        dims = cell.get("dimension_values") or {}
        row = str(dims.get(row_key) or "Unknown")
        col = str(dims.get(col_key) or "Unknown")
        delta = _coerce_float(cell.get("delta_value")) or 0.0
        grid.setdefault(row, {})[col] = grid.get(row, {}).get(col, 0.0) + delta
        row_totals[row] = row_totals.get(row, 0.0) + delta
        col_totals[col] = col_totals.get(col, 0.0) + delta

    if not grid or not col_totals:
        return ""

    rows = sorted(grid.keys())
    cols = sorted(col_totals.keys())
    grand_total = sum(row_totals.values())
    max_value = max([abs(v) for v in row_totals.values()] + [abs(v) for v in col_totals.values()] + [1.0])

    title = _heatmap_title(cells, operational_dims)
    html_parts: List[str] = []
    html_parts.append(
        '<div style="background:#f8f9fa; border-radius:6px; padding:15px; min-width:240px;">'
        f'<div style="font-weight:600; color:#212529; margin-bottom:10px;">{_esc(title)}</div>'
        '<table style="width:100%; border-collapse:collapse; font-size:0.9em;">'
        '<thead><tr style="background:#e9ecef;">'
        f'<th style="padding:6px 8px; text-align:left;">{_esc(row_label)}</th>'
    )
    for col in cols:
        html_parts.append(
            f'<th style="padding:6px 8px; text-align:right;">{_esc(col)}</th>'
        )
    html_parts.append('<th style="padding:6px 8px; text-align:right;"><strong>Total</strong></th></tr></thead><tbody>')

    for row in rows:
        html_parts.append(
            f'<tr><th style="padding:6px 8px; text-align:left; background:#fff;">{_esc(row)}</th>'
        )
        for col in cols:
            value = grid.get(row, {}).get(col, 0.0)
            intensity = abs(value) / max_value if max_value > 0 else 0
            bg = ""
            if value > 0 and intensity > 0.5:
                bg = "background:#a8e6cf;"
            elif value > 0:
                bg = "background:#d4edda;"
            elif value < 0:
                bg = "background:#f8d7da;"
            text = format_compact_money(value) if value else "—"
            html_parts.append(
                f'<td style="padding:6px 8px; text-align:right; {bg}">{text}</td>'
            )
        html_parts.append(
            f'<td style="padding:6px 8px; text-align:right;"><strong>{format_compact_money(row_totals[row])}</strong></td>'
            "</tr>"
        )

    html_parts.append('<tr style="background:#e9ecef;"><th style="padding:6px 8px; text-align:left;"><strong>Total</strong></th>')
    for col in cols:
        html_parts.append(
            f'<td style="padding:6px 8px; text-align:right;"><strong>{format_compact_money(col_totals[col])}</strong></td>'
        )
    html_parts.append(
        f'<td style="padding:6px 8px; text-align:right;"><strong>{format_compact_money(grand_total)}</strong></td></tr>'
        "</tbody></table></div>"
    )
    return "".join(html_parts)


_SIGNAL_COLORS = {
    "volume_plus_paid_ratio": "#c8e6c9",
    "volume": "#bbdefb",
    "unit_cost": "#ffcdd2",
    "paid_ratio": "#fff9c4",
    "mixed": "#e1bee7",
}


def _signal_badge(signal: str) -> str:
    bg = _SIGNAL_COLORS.get(signal, "#e0e0e0")
    return (
        f'<span style="display:inline-block; padding:3px 8px; border-radius:10px; '
        f'background:{bg}; font-size:0.8em; color:#212529;">{_esc(_signal_label(signal))}</span>'
    )


def _build_kpi_strip_html(
    operational_cells: Sequence[Mapping[str, Any]],
    clinical_cells: Sequence[Mapping[str, Any]],
) -> str:
    op_delta = _sum_field(operational_cells, "delta_value")
    share_pos = _sum_field(operational_cells, "share_of_positive_delta")
    clinical_delta = sum(
        _coerce_float(c.get("delta_value")) or 0.0
        for c in clinical_cells
        if (_coerce_float(c.get("delta_value")) or 0.0) > 0
    )

    def tile(label: str, value: str, color: str = "#212529") -> str:
        return (
            '<div style="flex:1; min-width:140px; background:white; padding:12px 16px; '
            'border-radius:6px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">'
            f'<div style="color:#6c757d; font-size:0.85em; margin-bottom:4px;">{_esc(label)}</div>'
            f'<div style="font-size:1.5em; font-weight:600; color:{color};">{_esc(value)}</div>'
            "</div>"
        )

    op_color = "#28a745" if op_delta >= 0 else "#dc3545"
    cl_color = "#28a745" if clinical_delta >= 0 else "#dc3545"
    return (
        '<div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;">'
        + tile("Selected Operational Δ Paid", format_signed_money(op_delta), op_color)
        + tile("Share of +Δ", format_percent(share_pos))
        + tile("Selected Clinical Δ Paid", format_signed_money(clinical_delta), cl_color)
        + "</div>"
    )


def _formatter_for_role(role: str) -> Callable[[Any], str]:
    if role == "count" or role == "admissions":
        return format_signed_integer
    if role == "ratio":
        return format_pct_points
    return format_signed_money


def _build_metric_columns(
    explainer_specs: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str, Callable[[Any], str]]]:
    """Return [(metric_key, header_label, cell_formatter), ...] for the dynamic
    metric columns of the Operational / Clinical tables."""
    columns: List[Tuple[str, str, Callable[[Any], str]]] = []
    for spec in explainer_specs:
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        role = str(spec.get("role") or classify_metric(name))
        label = str(spec.get("label") or "").strip() or humanize_metric_name(name)
        formatter = _formatter_for_role(role)
        columns.append((name, f"{label} Δ", formatter))
    return columns


def _build_operational_table_html(
    cells: Sequence[Mapping[str, Any]],
    dim_order: Sequence[str] = (),
    explainer_specs: Sequence[Mapping[str, Any]] = (),
    dim_specs: Sequence[Mapping[str, Any]] = (),
) -> str:
    deltas = [abs(_coerce_float(c.get("delta_value")) or 0.0) for c in cells]
    max_delta = max(deltas) if deltas else 0.0
    metric_columns = _build_metric_columns(explainer_specs)
    label_map = _build_dim_label_map(dim_specs)

    rows: List[str] = []
    header_parts = [
        '<table style="width:100%; border-collapse:collapse; font-size:0.9em; background:white;">'
        '<thead><tr style="background:#f1f3f5; border-bottom:2px solid #dee2e6;">'
        '<th style="padding:8px; text-align:left;">Rank</th>'
        '<th style="padding:8px; text-align:left;">Interaction</th>'
        '<th style="padding:8px; text-align:right;">Δ Paid</th>'
        '<th style="padding:8px; text-align:right;">Share +Δ</th>'
        '<th style="padding:8px; text-align:right;">Share Net Δ</th>'
    ]
    for _, header_label, _formatter in metric_columns:
        header_parts.append(
            f'<th style="padding:8px; text-align:right;">{_esc(header_label)}</th>'
        )
    header_parts.append('<th style="padding:8px; text-align:left;">Signal</th>')
    header_parts.append("</tr></thead><tbody>")
    rows.append("".join(header_parts))

    for idx, cell in enumerate(cells):
        pairs = _build_dim_pairs(cell, dim_order, label_map)
        tooltip = _esc(_build_dim_label(cell, dim_order, label_map))
        delta = _coerce_float(cell.get("delta_value")) or 0.0
        bar_width = (abs(delta) / max_delta * 100) if max_delta > 0 else 0
        bar_color = "#28a745" if delta >= 0 else "#dc3545"
        signal = _derive_signal(cell, explainer_specs)
        row_parts = [
            f'<tr style="border-bottom:1px solid #e9ecef;">'
            f'<td style="padding:8px; vertical-align:top;">{idx + 1}</td>'
            f'<td style="padding:8px; vertical-align:top;" title="{tooltip}">{_render_dim_pairs_html(pairs)}</td>'
            f'<td style="padding:8px; text-align:right; position:relative;">'
            f'<div style="position:relative;">'
            f'<div style="position:absolute; inset:0; background:{bar_color}; opacity:0.15; '
            f'width:{bar_width:.1f}%;"></div>'
            f'<span style="position:relative;">{_esc(format_signed_money(delta))}</span>'
            f'</div></td>'
            f'<td style="padding:8px; text-align:right;">{_esc(format_percent(cell.get("share_of_positive_delta", 0)))}</td>'
            f'<td style="padding:8px; text-align:right;">{_esc(format_percent(cell.get("share_of_net_delta", 0)))}</td>'
        ]
        for metric_name, _header, formatter in metric_columns:
            row_parts.append(
                f'<td style="padding:8px; text-align:right;">'
                f'{_esc(formatter(_get_metric_delta(cell, metric_name)))}</td>'
            )
        row_parts.append(f'<td style="padding:8px;">{_signal_badge(signal)}</td></tr>')
        rows.append("".join(row_parts))
    rows.append("</tbody></table>")
    return "".join(rows)


def _build_clinical_table_html(
    clinical_cells: Sequence[Mapping[str, Any]],
    offset_cells: Sequence[Mapping[str, Any]],
    dim_order: Sequence[str] = (),
    explainer_specs: Sequence[Mapping[str, Any]] = (),
    dim_specs: Sequence[Mapping[str, Any]] = (),
) -> str:
    metric_columns = _build_metric_columns(explainer_specs)
    label_map = _build_dim_label_map(dim_specs)

    rows: List[str] = []
    header_parts = [
        '<table style="width:100%; border-collapse:collapse; font-size:0.9em; background:white;">'
        '<thead><tr style="background:#f1f3f5; border-bottom:2px solid #dee2e6;">'
        '<th style="padding:8px; text-align:left;">Type</th>'
        '<th style="padding:8px; text-align:left;">Clinical Cell</th>'
        '<th style="padding:8px; text-align:right;">Δ Paid</th>'
        '<th style="padding:8px; text-align:right;">Share Net Δ</th>'
    ]
    for _, header_label, _formatter in metric_columns:
        header_parts.append(
            f'<th style="padding:8px; text-align:right;">{_esc(header_label)}</th>'
        )
    header_parts.append('<th style="padding:8px; text-align:left;">Signal</th>')
    header_parts.append("</tr></thead><tbody>")
    rows.append("".join(header_parts))

    def add_row(cell: Mapping[str, Any], type_label: str, type_color: str) -> None:
        pairs = _build_dim_pairs(cell, dim_order, label_map)
        tooltip = _esc(_build_dim_label(cell, dim_order, label_map))
        delta = _coerce_float(cell.get("delta_value")) or 0.0
        signal = _derive_signal(cell, explainer_specs)
        delta_text = format_signed_money(delta) if type_label == "Increase" else format_compact_money(delta)
        row_parts = [
            f'<tr style="border-bottom:1px solid #e9ecef;">'
            f'<td style="padding:8px;"><span style="color:{type_color}; font-weight:600;">{type_label}</span></td>'
            f'<td style="padding:8px; vertical-align:top;" title="{tooltip}">{_render_dim_pairs_html(pairs)}</td>'
            f'<td style="padding:8px; text-align:right;">{_esc(delta_text)}</td>'
            f'<td style="padding:8px; text-align:right;">{_esc(format_percent(cell.get("share_of_net_delta", 0)))}</td>'
        ]
        for metric_name, _header, formatter in metric_columns:
            row_parts.append(
                f'<td style="padding:8px; text-align:right;">'
                f'{_esc(formatter(_get_metric_delta(cell, metric_name)))}</td>'
            )
        row_parts.append(f'<td style="padding:8px;">{_signal_badge(signal)}</td></tr>')
        rows.append("".join(row_parts))

    for cell in clinical_cells:
        add_row(cell, "Increase", "#28a745")
    for cell in offset_cells:
        add_row(cell, "Offset", "#dc3545")

    rows.append("</tbody></table>")
    return "".join(rows)


def _infer_dim_specs_from_cells(
    cells: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Fallback dim spec list built from the first non-empty cell's own keys.

    Used when the correlation payload predates the ``interaction_matrix.schema``
    hint (e.g. cached results from before this rollout, or unit-test payloads).
    """
    for cell in cells:
        dims = cell.get("dimension_values") if isinstance(cell, Mapping) else None
        if isinstance(dims, Mapping) and dims:
            return [
                {"name": key, "label": str(key).replace("_", " ").title()}
                for key in dims.keys()
            ]
    return []


def _infer_explainer_specs_from_cells(
    cells: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    for cell in cells:
        explainers = cell.get("explainer_metrics") if isinstance(cell, Mapping) else None
        if isinstance(explainers, Mapping) and explainers:
            return [
                {
                    "name": key,
                    "label": humanize_metric_name(key),
                    "role": classify_metric(key),
                }
                for key in explainers.keys()
            ]
    return []


def _extract_matrix_schema(
    matrix: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]], List[str], List[str]]:
    """Pull the interaction-matrix schema hint out of the correlation payload.

    Returns ``(operational_dim_specs, clinical_dim_specs, explainer_specs,
    operational_dim_order, clinical_dim_order)``. Falls back to inferring from
    the cells themselves when the payload lacks a ``schema`` key.
    """
    schema = _safe_dict(matrix.get("schema"))
    op_specs: List[Mapping[str, Any]] = [
        _safe_dict(item) for item in _safe_list(schema.get("operational_dimensions"))
    ]
    cl_specs: List[Mapping[str, Any]] = [
        _safe_dict(item) for item in _safe_list(schema.get("clinical_dimensions"))
    ]
    ex_specs: List[Mapping[str, Any]] = [
        _safe_dict(item) for item in _safe_list(schema.get("explainer_metrics"))
    ]

    if not op_specs or not ex_specs:
        op_cells = _safe_list(_safe_dict(matrix.get("operational")).get("selected_cells"))
        if not op_specs:
            op_specs = _infer_dim_specs_from_cells(op_cells)
        if not ex_specs:
            ex_specs = _infer_explainer_specs_from_cells(op_cells)
    if not cl_specs:
        cl_cells = _safe_list(_safe_dict(matrix.get("clinical")).get("selected_cells"))
        cl_specs = _infer_dim_specs_from_cells(cl_cells)

    op_order = [str(item.get("name", "")) for item in op_specs if item.get("name")]
    cl_order = [str(item.get("name", "")) for item in cl_specs if item.get("name")]
    return op_specs, cl_specs, ex_specs, op_order, cl_order


def build_correlation_visuals_html(correlation_summary: Mapping[str, Any]) -> Optional[str]:
    """Return self-contained HTML for the waterfall + interaction sections.

    Returns ``None`` when there is no waterfall data to render.
    """

    waterfall_html = _build_waterfall_chart_html(correlation_summary)
    matrix = _safe_dict(correlation_summary.get("interaction_matrix"))
    operational = _safe_dict(matrix.get("operational"))
    clinical = _safe_dict(matrix.get("clinical"))
    operational_cells = _safe_list(operational.get("selected_cells"))
    clinical_cells = _safe_list(clinical.get("selected_cells"))
    offset_cells = _safe_list(clinical.get("offset_cells_preview"))
    op_specs, cl_specs, ex_specs, op_order, cl_order = _extract_matrix_schema(matrix)

    summary_status = _safe_dict(matrix.get("summary")).get("status")
    interactions_available = summary_status == "success" and operational_cells

    pieces: List[str] = []
    if waterfall_html:
        pieces.append(waterfall_html)

    if interactions_available:
        pieces.append(
            '<div style="background:#f8f9fa; border-radius:8px; padding:20px; margin-bottom:20px;">'
            '<div style="font-size:1.2em; font-weight:600; color:#212529; margin-bottom:6px;">'
            '🔍 Detected Interactions</div>'
            '<div style="color:#6c757d; font-size:0.9em; margin-bottom:14px;">'
            'Operational cross-product cells selected from the waterfall, followed by clinical '
            'detail within the selected operational pocket.</div>'
            + _build_kpi_strip_html(operational_cells, clinical_cells)
            + '<div style="font-weight:600; color:#212529; margin:12px 0 8px;">Operational Concentration</div>'
            '<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">'
            f'<div style="flex:2; min-width:520px; overflow-x:auto;">{_build_operational_table_html(operational_cells, op_order, ex_specs, op_specs)}</div>'
            f'<div style="flex:1; min-width:260px;">{_build_operational_heatmap_html(operational_cells, op_specs)}</div>'
            "</div>"
        )
        if clinical_cells or offset_cells:
            pieces.append(
                '<div style="font-weight:600; color:#212529; margin:18px 0 8px;">'
                'Clinical Detail Within Selected Operational Cells</div>'
                + _build_clinical_table_html(clinical_cells, offset_cells, cl_order, ex_specs, cl_specs)
            )
        pieces.append("</div>")

    if not pieces:
        return None

    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        + "".join(pieces)
        + "</div>"
    )


def _estimate_height(correlation_summary: Mapping[str, Any]) -> int:
    height = 60
    if correlation_summary.get("drill_path") and correlation_summary.get("baseline_value") is not None:
        height += 560
    matrix = _safe_dict(correlation_summary.get("interaction_matrix"))
    operational = _safe_list(_safe_dict(matrix.get("operational")).get("selected_cells"))
    clinical = _safe_list(_safe_dict(matrix.get("clinical")).get("selected_cells"))
    offsets = _safe_list(_safe_dict(matrix.get("clinical")).get("offset_cells_preview"))
    if operational:
        height += 240 + len(operational) * 44
    if clinical or offsets:
        height += 130 + (len(clinical) + len(offsets)) * 44
    return min(height, 2400)


def _estimate_matrix_height(correlation_summary: Mapping[str, Any]) -> int:
    """Height for the matrix-only HTML used by the redesigned renderer.

    Drill-path cells now render one dimension per line, so each row's height
    scales with the number of (non-null) dimensions rather than being a flat
    per-row constant.
    """

    def _row_height(cells: Sequence[Mapping[str, Any]]) -> int:
        # ~20px per stacked dimension line + ~24px vertical padding; floor at 1.
        lines = max((len(_build_dim_pairs(c)) for c in cells), default=1) or 1
        return 24 + lines * 20

    height = 60
    matrix = _safe_dict(correlation_summary.get("interaction_matrix"))
    operational = _safe_list(_safe_dict(matrix.get("operational")).get("selected_cells"))
    clinical = _safe_list(_safe_dict(matrix.get("clinical")).get("selected_cells"))
    offsets = _safe_list(_safe_dict(matrix.get("clinical")).get("offset_cells_preview"))
    if operational:
        height += 240 + len(operational) * _row_height(operational)
    if clinical or offsets:
        height += 130 + (len(clinical) + len(offsets)) * _row_height(list(clinical) + list(offsets))
    return min(height, 2400)


# ---------------------------------------------------------------------------
# Exec-grade redesigned renderers (Plotly waterfall + Streamlit KPI cards)
# ---------------------------------------------------------------------------


def build_kpi_data(correlation_summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Distill the correlation summary into the 5 KPI-card values.

    Returns ``None`` when there is no baseline/comparison data to render.
    """

    baseline = _coerce_float(correlation_summary.get("baseline_value"))
    comparison = _coerce_float(correlation_summary.get("comparison_value"))
    if baseline is None and comparison is None:
        return None

    baseline = baseline or 0.0
    comparison = comparison or 0.0
    impact = comparison - baseline
    impact_pct = (impact / baseline) if baseline else None

    matrix = _safe_dict(correlation_summary.get("interaction_matrix"))
    operational_cells = _safe_list(_safe_dict(matrix.get("operational")).get("selected_cells"))
    clinical_cells = _safe_list(_safe_dict(matrix.get("clinical")).get("selected_cells"))

    top_driver_name = "—"
    top_driver_delta: float = 0.0
    for level in _safe_list(correlation_summary.get("drill_path")):
        if not isinstance(level, Mapping):
            continue
        dimension = str(level.get("dimension", ""))
        for seg in _safe_list(level.get("top_segments"))[:5]:
            if not isinstance(seg, Mapping):
                continue
            delta = _coerce_float(seg.get("aligned_delta"))
            if delta is None:
                delta = _coerce_float(seg.get("delta_value")) or 0.0
            if abs(delta) > abs(top_driver_delta):
                value_text = str(seg.get("value", ""))
                top_driver_name = f"{dimension}: {value_text}" if dimension else value_text or "—"
                top_driver_delta = delta

    return {
        "impact": impact,
        "impact_pct": impact_pct,
        "metric_label": correlation_summary.get("metric_label") or "Metric",
        "baseline": baseline,
        "comparison": comparison,
        "top_driver_name": top_driver_name,
        "top_driver_delta": top_driver_delta,
        "n_interactions": len(operational_cells) + len(clinical_cells),
    }


def render_kpi_strip(correlation_summary: Mapping[str, Any]) -> bool:
    """Render the 4-card executive KPI hero. Returns True if rendered."""

    data = build_kpi_data(correlation_summary)
    if data is None:
        return False

    cards = st.columns(4)

    def _truncate(text: str, limit: int = 28) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"

    with cards[0]:
        with st.container(border=True):
            delta = (
                f"{data['impact_pct'] * 100:+.1f}% vs prior"
                if data["impact_pct"] is not None
                else None
            )
            st.metric(
                "Total Impact",
                format_signed_money(data["impact"]),
                delta,
                delta_color="inverse",
            )
    with cards[1]:
        with st.container(border=True):
            st.metric(
                "Top Driver",
                _truncate(data["top_driver_name"]),
                format_signed_money(data["top_driver_delta"]) if data["top_driver_delta"] else None,
                delta_color="inverse",
            )
    with cards[2]:
        with st.container(border=True):
            st.metric(
                "Baseline → Comparison",
                f"{format_compact_money(data['baseline'])} → {format_compact_money(data['comparison'])}",
            )
    with cards[3]:
        with st.container(border=True):
            st.metric(
                "Detected Interactions",
                str(data["n_interactions"]),
                help="Operational + clinical cells the matrix flagged.",
            )
    return True


_WATERFALL_POSITIVE_SHADES = [
    "#B5364B",  # primary brand red
    "#C25668",
    "#CE7585",
    "#DA94A3",
    "#E6B3C0",
]
_WATERFALL_NEGATIVE_SHADES = [
    "#1A8754",
    "#3CA071",
    "#5EB98E",
    "#80D2AB",
    "#A2EBC8",
]


def _waterfall_segment_color(delta: float, idx: int) -> str:
    if delta < 0:
        return _WATERFALL_NEGATIVE_SHADES[idx % len(_WATERFALL_NEGATIVE_SHADES)]
    if delta > 0:
        return _WATERFALL_POSITIVE_SHADES[idx % len(_WATERFALL_POSITIVE_SHADES)]
    return "#94A3B8"


def _build_waterfall_rows(
    drill_path: Sequence[Any],
    baseline: float,
    comparison: float,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Walk drill_path and produce the row spec the renderer iterates over.

    Each row carries: label, type (``total``/``level``/``residual``), segments
    list (with name + delta), and the cumulative_start/cumulative_end positions
    used to place stacked-bar bases.
    """

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "label": "Baseline",
            "type": "total",
            "segments": [{"name": "Baseline", "delta": baseline}],
            "cumulative_start": 0.0,
            "cumulative_end": baseline,
            "level_total": baseline,
        }
    )

    cumulative = baseline
    for level in drill_path:
        if not isinstance(level, Mapping):
            continue
        raw_segments = _safe_list(level.get("top_segments"))
        all_segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            if not isinstance(seg, Mapping):
                continue
            delta = _coerce_float(seg.get("aligned_delta"))
            if delta is None:
                delta = _coerce_float(seg.get("delta_value")) or 0.0
            name = str(seg.get("value") or "—")
            all_segments.append({"name": name, "delta": delta})
        if not all_segments:
            continue

        # Sort by absolute contribution so the visual stack puts the biggest
        # piece adjacent to the running total — matches the legacy SVG order.
        all_segments.sort(key=lambda s: abs(s["delta"]), reverse=True)
        top_segments = all_segments[:top_n]
        remainder = all_segments[top_n:]
        residual_delta = sum(s["delta"] for s in remainder)
        if remainder and abs(residual_delta) > 0:
            top_segments.append(
                {"name": f"Others in level ({len(remainder)})", "delta": residual_delta}
            )

        level_total = sum(s["delta"] for s in top_segments)
        rows.append(
            {
                "label": f"Level {level.get('level', '?')}: {level.get('dimension', '') or 'segment'}",
                "type": "level",
                "segments": top_segments,
                "cumulative_start": cumulative,
                "cumulative_end": cumulative + level_total,
                "level_total": level_total,
            }
        )
        cumulative += level_total

    residual = comparison - cumulative
    # Only inject "All other" when the residual is materially non-zero
    # (avoid noise when drill_path already explains the full delta).
    if abs(residual) >= max(1.0, 0.01 * abs(comparison - baseline)):
        rows.append(
            {
                "label": "All other",
                "type": "residual",
                "segments": [{"name": "All other", "delta": residual}],
                "cumulative_start": cumulative,
                "cumulative_end": cumulative + residual,
                "level_total": residual,
            }
        )
        cumulative += residual

    rows.append(
        {
            "label": "Comparison",
            "type": "total",
            "segments": [{"name": "Comparison", "delta": comparison}],
            "cumulative_start": 0.0,
            "cumulative_end": comparison,
            "level_total": comparison,
        }
    )
    return rows


def build_waterfall_figure(correlation_summary: Mapping[str, Any]):
    """Return a Plotly Figure for the horizontal cost-cascade, or None when not enough data.

    Each drill-path level is rendered as a horizontal **stacked bar of its top
    contributors**: top-N segments plus an "Others in level" residual. Every
    segment has its own hover tooltip carrying the dimension value and signed
    delta. Dotted connector lines join adjacent rows at the running total. The
    biggest single segment gets a callout annotation, and a residual
    ``All other`` row reconciles drill totals to the comparison value when
    they don't sum cleanly.
    """

    drill_path = _safe_list(correlation_summary.get("drill_path"))
    baseline = _coerce_float(correlation_summary.get("baseline_value"))
    comparison = _coerce_float(correlation_summary.get("comparison_value"))
    if baseline is None or comparison is None or not drill_path:
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - plotly listed in pyproject
        return None

    rows = _build_waterfall_rows(drill_path, baseline, comparison, top_n=5)
    if len(rows) <= 2:  # only baseline + comparison, no levels rendered
        return None

    labels = [r["label"] for r in rows]

    fig = go.Figure()

    # Baseline / Comparison: single absolute bars from 0 → value.
    # Level rows: a stack of per-segment bars (each with its own base).
    # All bars at the same y use ``barmode="overlay"`` so the ``base`` offset
    # is honored without re-stacking them at x=0.
    for row in rows:
        label = row["label"]
        if row["type"] == "total":
            seg = row["segments"][0]
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[seg["delta"]],
                    base=[0.0],
                    orientation="h",
                    marker={"color": "#1F3A68"},
                    text=[format_compact_money(seg["delta"])],
                    textposition="outside",
                    textfont={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
                    hovertemplate=(
                        f"<b>{label}</b><br>{format_compact_money(seg['delta'])}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            continue

        if row["type"] == "residual":
            seg = row["segments"][0]
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[seg["delta"]],
                    base=[row["cumulative_start"]],
                    orientation="h",
                    marker={"color": "#5B6776", "line": {"color": "#FFFFFF", "width": 1}},
                    text=[format_signed_money(seg["delta"])],
                    textposition="outside",
                    textfont={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
                    hovertemplate=(
                        f"<b>All other</b><br>Δ {format_signed_money(seg['delta'])}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            continue

        # Level row: stack segments by walking the running cumulative.
        running = row["cumulative_start"]
        for idx, seg in enumerate(row["segments"]):
            delta = seg["delta"]
            color = _waterfall_segment_color(delta, idx)
            seg_name = seg["name"] if seg["name"] else "—"
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[delta],
                    base=[running],
                    orientation="h",
                    marker={"color": color, "line": {"color": "#FFFFFF", "width": 1}},
                    hovertemplate=(
                        f"<b>{html.escape(seg_name)}</b><br>"
                        f"Δ {format_signed_money(delta)}<br>"
                        f"Level: {html.escape(label)}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            running += delta

        # Level total label outside the stack, anchored on the trailing edge.
        if row["level_total"] >= 0:
            fig.add_annotation(
                x=row["cumulative_end"],
                y=label,
                text=f"<b>{format_signed_money(row['level_total'])}</b>",
                showarrow=False,
                xanchor="left",
                xshift=6,
                font={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
            )
        else:
            fig.add_annotation(
                x=row["cumulative_end"],
                y=label,
                text=f"<b>{format_signed_money(row['level_total'])}</b>",
                showarrow=False,
                xanchor="right",
                xshift=-6,
                font={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
            )

    # Dotted connector lines join the running total at the boundary between
    # each pair of adjacent rows. Baseline → first level → ... → Comparison.
    for i in range(len(rows) - 1):
        upper = rows[i]
        lower = rows[i + 1]
        if lower["type"] == "total" and lower["label"] == "Comparison":
            # Don't connect to Comparison since it floats from 0 separately;
            # leave a visual gap so the reader sees Comparison is independent.
            continue
        boundary_x = upper["cumulative_end"]
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=boundary_x,
            x1=boundary_x,
            y0=upper["label"],
            y1=lower["label"],
            line={"color": "#94A3B8", "dash": "dot", "width": 1},
            layer="below",
        )

    # Callout on the biggest single segment across all levels.
    big_info: Optional[Dict[str, Any]] = None
    for row in rows:
        if row["type"] != "level":
            continue
        running = row["cumulative_start"]
        for seg in row["segments"]:
            d = seg["delta"]
            mid_x = running + d / 2.0
            if big_info is None or abs(d) > abs(big_info["delta"]):
                big_info = {
                    "delta": d,
                    "x": mid_x,
                    "y": row["label"],
                    "name": seg["name"],
                }
            running += d
    if big_info is not None and abs(big_info["delta"]) > 0:
        fig.add_annotation(
            x=big_info["x"],
            y=big_info["y"],
            text=(
                f"<b>{html.escape(str(big_info['name']))}</b> drives "
                f"{format_signed_money(big_info['delta'])}"
            ),
            showarrow=True,
            arrowhead=2,
            ax=80,
            ay=-36,
            font={"size": 12, "color": "#1A2436"},
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#D6DBE4",
            borderwidth=1,
            borderpad=4,
        )

    metric_label = correlation_summary.get("metric_label") or "Metric"
    fig.update_layout(
        title={
            "text": f"Cost cascade · {metric_label}",
            "font": {"size": 16, "color": "#1A2436", "family": "Inter, sans-serif"},
            "x": 0.0,
            "xanchor": "left",
        },
        height=max(380, 90 + 56 * len(rows)),
        margin={"l": 12, "r": 12, "t": 60, "b": 24},
        showlegend=False,
        barmode="overlay",
        bargap=0.22,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis={
            "automargin": True,
            "title": "",
            "autorange": "reversed",
            "categoryorder": "array",
            "categoryarray": labels,
        },
        xaxis={
            "title": "",
            "showgrid": True,
            "gridcolor": "#F1F4F9",
            "zerolinecolor": "#D6DBE4",
        },
        font={"family": "Inter, -apple-system, sans-serif", "size": 13, "color": "#1A2436"},
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#D6DBE4",
            "font": {"family": "Inter, sans-serif", "color": "#1A2436", "size": 12},
        },
    )
    return fig


def render_waterfall_chart(correlation_summary: Mapping[str, Any]) -> bool:
    """Render the Plotly cost-cascade waterfall. Returns True if rendered."""

    fig = build_waterfall_figure(correlation_summary)
    if fig is None:
        return False
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
        key="correlation_waterfall",
    )
    return True


def build_matrix_section_html(correlation_summary: Mapping[str, Any]) -> Optional[str]:
    """Return ONLY the Detected Interactions / Operational / Clinical block as HTML.

    Reuses the existing analyst-grade table + heatmap builders. Returns ``None``
    when no matrix data is available.
    """

    matrix = _safe_dict(correlation_summary.get("interaction_matrix"))
    operational_cells = _safe_list(_safe_dict(matrix.get("operational")).get("selected_cells"))
    clinical_cells = _safe_list(_safe_dict(matrix.get("clinical")).get("selected_cells"))
    offset_cells = _safe_list(_safe_dict(matrix.get("clinical")).get("offset_cells_preview"))
    op_specs, cl_specs, ex_specs, op_order, cl_order = _extract_matrix_schema(matrix)
    summary_status = _safe_dict(matrix.get("summary")).get("status")
    if not (summary_status == "success" and operational_cells):
        return None

    pieces: List[str] = []
    pieces.append(
        '<div style="background:#F1F4F9; border-radius:10px; padding:20px; margin-bottom:20px;">'
        '<div style="font-size:1.05em; font-weight:600; color:#1A2436; margin-bottom:6px;">'
        '🔍 Detected Interactions</div>'
        '<div style="color:#5B6776; font-size:0.85em; margin-bottom:14px;">'
        'Operational cross-product cells selected from the waterfall, followed by clinical '
        'detail within the selected operational pocket.</div>'
        + _build_kpi_strip_html(operational_cells, clinical_cells)
        + '<div style="font-weight:600; color:#1A2436; margin:12px 0 8px;">Operational Concentration</div>'
        '<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">'
        f'<div style="flex:2; min-width:520px; overflow-x:auto;">{_build_operational_table_html(operational_cells, op_order, ex_specs, op_specs)}</div>'
        f'<div style="flex:1; min-width:260px;">{_build_operational_heatmap_html(operational_cells, op_specs)}</div>'
        "</div>"
    )
    if clinical_cells or offset_cells:
        pieces.append(
            '<div style="font-weight:600; color:#1A2436; margin:18px 0 8px;">'
            'Clinical Detail Within Selected Operational Cells</div>'
            + _build_clinical_table_html(clinical_cells, offset_cells, cl_order, ex_specs, cl_specs)
        )
    pieces.append("</div>")
    return (
        '<div style="font-family:Inter,-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        + "".join(pieces)
        + "</div>"
    )


def render_correlation_visuals(correlation_summary: Mapping[str, Any]) -> bool:
    """Render KPI strip + Plotly waterfall + interaction matrix. Returns True if anything was rendered."""

    rendered_any = False
    rendered_any |= render_kpi_strip(correlation_summary)
    rendered_any |= render_waterfall_chart(correlation_summary)

    matrix_html = build_matrix_section_html(correlation_summary)
    if matrix_html:
        data_uri = f"data:text/html;charset=utf-8,{urllib.parse.quote(matrix_html)}"
        st.iframe(data_uri, height=_estimate_matrix_height(correlation_summary))
        rendered_any = True

    return rendered_any
