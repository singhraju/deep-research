"""Render pattern tab visuals (drill-down paths + reimbursement payer table + recommended actions).

Mirrors the static HTML viewers shipped with the analytics team's pattern and
reimbursement reports. Inputs:
- ``pattern``: a business pattern dict from ``output['research']['business_patterns']``
- ``cards``: the card list from ``output['research']['pattern_summary']['cards']``
- ``reimbursement``: ``output['research']['reimbursement_by_pattern'][rank]``
  whose ``formatted_output`` provides ``summary_table`` and ``individual_policies``,
  and which carries ``recommended_action`` plus ``elevance_executive_summary``.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import streamlit as st

_TECHNICAL_VALUES = {
    "AUTH_CODE_DISTRIBUTION",
    "UNMAPPED",
    "-3",
    "-2",
    "-1",
    "NULL",
    "UNDEFINED",
    "",
}

_ENTITY_TYPE_LABELS = {
    "states": "State",
    "providers": "Provider",
    "products": "Product",
    "drg": "DRG",
    "diagnosis": "Dx",
}

_NUMERIC_CODE_RE = re.compile(r"^-?\d+$")


def _is_valid_business_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    upper = text.upper()
    if upper in _TECHNICAL_VALUES:
        return False
    if upper.startswith("UNKNOWN"):
        return False
    if _NUMERIC_CODE_RE.match(upper):
        return False
    return True


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


def _format_currency(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "$0.00"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:.2f}"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def build_drill_down_paths(
    pattern: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return up to ``top_n`` aggregated drill-down paths, sorted by absolute Δ.

    Each entry has ``path``, ``delta``, ``admissions``, ``count``.
    Mirrors ``buildDrillDownPaths`` in the static pattern viewer.
    """

    source_card_ids = list(pattern.get("source_card_ids") or [])
    if not source_card_ids:
        return []

    card_by_id: Dict[Any, Mapping[str, Any]] = {}
    for card in cards:
        if isinstance(card, Mapping):
            cid = card.get("card_id")
            if cid is not None:
                card_by_id[cid] = card

    raw_paths: List[Dict[str, Any]] = []
    for card_id in source_card_ids:
        card = card_by_id.get(card_id)
        if not isinstance(card, Mapping):
            continue

        dimensions = card.get("dimensions") or {}
        context = card.get("context_dimensions") or {}
        source_entity = card.get("source_entity") or {}

        components: List[str] = []

        # Section 1: source entity ([State: CO], [Provider: UCI ...]).
        entity_type = source_entity.get("type")
        entity_name = source_entity.get("name")
        if entity_type and entity_name and _is_valid_business_value(entity_name):
            label = _ENTITY_TYPE_LABELS.get(str(entity_type), str(entity_type))
            components.append(f"[{label}: {entity_name}]")

        # Section 2: operational drill-down filters.
        filters: List[str] = []
        state = dimensions.get("service_area_state")
        if _is_valid_business_value(state):
            filters.append(str(state))

        product = dimensions.get("product_description")
        if _is_valid_business_value(product):
            product_label = str(product)
            if context.get("er_admit_indicator") == "Y":
                product_label += " (ER)"
            filters.append(product_label)

        facility = dimensions.get("facility_type")
        if facility and (facility == "Not Mapped" or _is_valid_business_value(facility)):
            filters.append(str(facility))

        pa_code = dimensions.get("pa_required_code")
        if _is_valid_business_value(pa_code):
            pa_label = (
                "PA Y" if pa_code == "Y"
                else "PA N" if pa_code == "N"
                else str(pa_code)
            )
            filters.append(pa_label)

        hcc = context.get("hcc_medium")
        if _is_valid_business_value(hcc):
            filters.append(str(hcc))

        if filters:
            components.append(" + ".join(filters))

        # Section 3: clinical context (provider | drg | diagnosis).
        clinical: List[str] = []
        provider_name = context.get("rendering_provider_name")
        if _is_valid_business_value(provider_name):
            clinical.append(str(provider_name))

        drg_name = context.get("drg_name")
        if _is_valid_business_value(drg_name):
            drg_text = str(drg_name)
            if len(drg_text) > 45:
                drg_text = drg_text[:42] + "..."
            clinical.append(drg_text)

        diag = context.get("primary_diagnosis_name")
        if _is_valid_business_value(diag):
            diag_text = str(diag)
            if len(diag_text) > 35:
                diag_text = diag_text[:32] + "..."
            clinical.append(diag_text)

        if clinical:
            components.append(" | ".join(clinical))

        if len(components) < 2:
            continue

        metrics = card.get("metrics") or {}
        value_block = metrics.get("value") if isinstance(metrics, Mapping) else None
        delta = _coerce_float((value_block or {}).get("delta")) if isinstance(value_block, Mapping) else None
        explainer = metrics.get("explainer") if isinstance(metrics, Mapping) else None
        admissions_block = (explainer or {}).get("admissions") if isinstance(explainer, Mapping) else None
        admits = _coerce_float((admissions_block or {}).get("delta")) if isinstance(admissions_block, Mapping) else None

        raw_paths.append(
            {
                "path": " → ".join(components),
                "delta": delta or 0.0,
                "admissions": admits or 0.0,
            }
        )

    # Aggregate duplicates.
    aggregated: Dict[str, Dict[str, Any]] = {}
    for entry in raw_paths:
        key = entry["path"]
        bucket = aggregated.setdefault(
            key,
            {"path": key, "delta": 0.0, "admissions": 0.0, "count": 0},
        )
        bucket["delta"] += entry["delta"]
        bucket["admissions"] += entry["admissions"]
        bucket["count"] += 1

    sorted_paths = sorted(
        aggregated.values(), key=lambda item: abs(item["delta"]), reverse=True
    )[:top_n]
    return sorted_paths


def _drill_path_impact_html(entry: Mapping[str, Any]) -> str:
    delta = entry.get("delta") or 0.0
    admits = entry.get("admissions") or 0.0
    count = entry.get("count") or 0
    impact = _format_currency(delta)
    admits_part = ""
    if admits:
        sign = "+" if admits > 0 else ""
        admits_part = f" ({sign}{int(round(admits))} admits)"
    count_part = f" [{count} cells]" if count > 1 else ""
    return (
        f'<span style="display:inline-block; padding:3px 10px; border-radius:12px; '
        f'background:#28a745; color:white; font-size:0.85em; font-weight:600; '
        f'margin-left:6px;">Δ{_esc(impact)}{_esc(admits_part)}{_esc(count_part)}</span>'
    )


def render_drill_down_paths(
    pattern: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]
) -> bool:
    """Render the drill-down paths panel for a pattern. Returns True if rendered."""

    paths = build_drill_down_paths(pattern, cards)
    if not paths:
        return False

    impact_summary = pattern.get("impact_summary") or {}
    estimated_delta = impact_summary.get("estimated_delta")

    parts: List[str] = []
    parts.append(
        '<div style="font-size:0.75em; font-weight:700; color:#6c757d; '
        'letter-spacing:1px; margin-bottom:8px;">DRILL-DOWN PATHS</div>'
    )
    parts.append('<div style="display:flex; flex-direction:column; gap:8px;">')
    for entry in paths:
        parts.append(
            '<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); '
            'border-left:3px solid #667eea; padding:10px 12px; border-radius:6px; '
            'font-size:0.9em; line-height:1.6; color:#212529;">'
            f'{_esc(entry["path"])}{_drill_path_impact_html(entry)}'
            "</div>"
        )
    parts.append("</div>")

    if estimated_delta:
        parts.append(
            '<div style="margin-top:14px;">'
            '<div style="font-size:0.75em; font-weight:700; color:#6c757d; '
            'letter-spacing:1px; margin-bottom:4px;">IMPACT</div>'
            f'<div style="font-size:1.4em; font-weight:700; color:#28a745;">'
            f"{_esc(estimated_delta)}</div></div>"
        )

    st.markdown("".join(parts), unsafe_allow_html=True)
    return True


def render_payer_summary_table(summary_table: Optional[Mapping[str, Any]]) -> bool:
    """Render the payer × policy-column summary table. Returns True if rendered."""

    if not isinstance(summary_table, Mapping):
        return False
    columns = list(summary_table.get("columns") or [])
    rows = list(summary_table.get("rows") or [])
    if not columns or not rows:
        return False

    title = summary_table.get("title") or "Payer Policy Summary"
    subtitle = summary_table.get("subtitle")

    if title:
        st.markdown(f"#### {title}")
    if subtitle:
        st.caption(subtitle)

    header_cells: List[str] = []
    for col in columns:
        label = col.get("label", col.get("id", ""))
        # Labels may contain newlines from the agent (e.g. "Appeals Process\n(Documented)") — keep them as <br>.
        label_html = _esc(label).replace("\n", "<br>")
        header_cells.append(
            f'<th style="background:#667eea; color:white; padding:12px 14px; '
            f'text-align:left; font-size:0.9em; font-weight:600; '
            f'border-right:1px solid rgba(255,255,255,0.2);">{label_html}</th>'
        )

    body_rows: List[str] = []
    for row_idx, row in enumerate(rows):
        bg = "#ffffff" if row_idx % 2 == 0 else "#f8f9fa"
        cells: List[str] = []
        for col_idx, col in enumerate(columns):
            col_id = col.get("id")
            value = row.get(col_id, "—") if col_id else "—"
            value_text = "" if value is None else str(value)
            if not value_text or value_text == "-":
                value_text = "—"
            if col_idx == 0:
                # Payer column: emphasized link-style label (matches the source viewer).
                cells.append(
                    f'<td style="padding:12px 14px; vertical-align:top; '
                    f'border-bottom:1px solid #e9ecef; color:#667eea; '
                    f'font-weight:600; font-size:0.9em;">{_esc(value_text)}</td>'
                )
            elif col.get("type") == "badge" and value_text != "—":
                cells.append(
                    f'<td style="padding:12px 14px; vertical-align:top; '
                    f'border-bottom:1px solid #e9ecef; font-size:0.9em;">'
                    f'<span style="background:#e7f3ff; color:#0066cc; padding:3px 10px; '
                    f'border-radius:12px; font-size:0.85em;">{_esc(value_text)}</span></td>'
                )
            else:
                cells.append(
                    f'<td style="padding:12px 14px; vertical-align:top; '
                    f'border-bottom:1px solid #e9ecef; font-size:0.9em; color:#212529; '
                    f'line-height:1.5;">{_esc(value_text)}</td>'
                )
        body_rows.append(f'<tr style="background:{bg};">{"".join(cells)}</tr>')

    table_html = (
        '<div style="overflow-x:auto; border-radius:6px; '
        'box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:16px;">'
        '<table style="width:100%; border-collapse:collapse; background:white;">'
        f'<thead><tr>{"".join(header_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    return True


def render_recommended_actions(
    actions: Optional[Sequence[Mapping[str, Any]]],
) -> bool:
    """Render the lavender 'Recommended Action' callout. Returns True if rendered."""

    if not actions:
        return False

    items: List[str] = []
    for idx, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping):
            continue
        description = action.get("description") or action.get("title") or ""
        if not description:
            continue
        rank = action.get("rank", idx)
        priority = (action.get("priority") or "").strip()
        priority_chip = ""
        if priority:
            priority_lc = priority.lower()
            color = {
                "high": "#dc3545",
                "medium": "#fd7e14",
                "low": "#198754",
            }.get(priority_lc, "#6c757d")
            priority_chip = (
                f'<span style="margin-left:8px; padding:2px 8px; border-radius:10px; '
                f'background:{color}; color:white; font-size:0.7em; font-weight:600; '
                f'text-transform:uppercase;">{_esc(priority)}</span>'
            )
        items.append(
            f'<div style="margin-bottom:14px;">'
            f'<strong>Recommendation {_esc(rank)}:</strong>{priority_chip} '
            f'{_esc(description)}</div>'
        )

    if not items:
        return False

    block = (
        '<div style="background:#e8eaf6; border-left:4px solid #667eea; '
        'padding:18px 20px; border-radius:6px; margin-bottom:16px;">'
        '<div style="color:#667eea; font-weight:600; margin-bottom:12px; '
        'font-size:0.95em;">💡 Recommended Action:</div>'
        + "".join(items)
        + "</div>"
    )
    st.markdown(block, unsafe_allow_html=True)
    return True


def _normalize_policy_entry(policy: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    title = (
        policy.get("policy_title")
        or policy.get("title")
        or policy.get("policy_title_text")
        or ""
    )
    title = str(title).strip()
    if not title:
        return None
    payer = (
        policy.get("payer_name")
        or policy.get("payer")
        or policy.get("payer_org")
        or ""
    )
    return {
        "title": title,
        "payer": str(payer),
        "url": str(policy.get("policy_url") or policy.get("external_link") or ""),
        "effective": str(policy.get("effective_date") or ""),
    }


def render_policy_links_expander(
    policies: Optional[Sequence[Mapping[str, Any]]],
) -> bool:
    """Render the collapsible list of referenced policies (closed by default)."""

    if not policies:
        return False

    normalized: List[Dict[str, Any]] = []
    for policy in policies:
        if not isinstance(policy, Mapping):
            continue
        entry = _normalize_policy_entry(policy)
        if entry is not None:
            normalized.append(entry)
    if not normalized:
        return False

    with st.expander(f"📚 Referenced Policies ({len(normalized)})", expanded=False):
        for policy in normalized:
            payer = policy["payer"]
            title = policy["title"]
            url = policy["url"]
            effective = policy["effective"]
            link_md = f"[{title}]({url})" if url else title
            payer_part = f"**{payer}** — " if payer else ""
            effective_part = f" · _Effective: {effective}_" if effective else ""
            st.markdown(f"- {payer_part}{link_md}{effective_part}")
    return True


# ---------------------------------------------------------------------------
# Pattern storyboard (hero icicle + evidence card)
# ---------------------------------------------------------------------------


_PRIORITY_BORDER = {
    "INCREASE": "#B5364B",
    "DECREASE": "#1A8754",
    "STABLE": "#5B6776",
}


def _truncate(text: str, limit: int = 48) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _format_signed_int(value: Any) -> str:
    val = _coerce_float(value)
    if val is None or val == 0:
        return "0"
    rounded = int(round(val))
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded:,}"


def _format_signed_money(value: Any) -> str:
    val = _coerce_float(value)
    if val is None:
        return "$0"
    sign = "+" if val >= 0 else "-"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{sign}${abs_val / 1_000:.1f}K"
    return f"{sign}${abs_val:.2f}"


def _path_short_label(path: str) -> str:
    """Extract a short y-axis label from a full drill-down path string."""

    parts = [p.strip() for p in (path or "").split("→") if p.strip()]
    if not parts:
        return "—"
    head = parts[0]
    return _truncate(head, 36)


def build_pattern_breakdown_figure(
    pattern: Mapping[str, Any], cards: Sequence[Mapping[str, Any]]
):
    """Return a Plotly horizontal bar of source-card contributions, or None.

    Each aggregated drill path (from :func:`build_drill_down_paths`) becomes
    one bar, sorted by absolute Δ paid. Color encodes direction (red = cost
    up, green = cost down). The full path, admits delta, and contributing-cell
    count are surfaced in the hover tooltip.
    """

    paths = build_drill_down_paths(pattern, cards, top_n=10)
    if not paths:
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - plotly listed in pyproject
        return None

    # build_drill_down_paths already sorts by |delta| desc, but be defensive.
    ordered = sorted(paths, key=lambda p: abs(p.get("delta", 0.0)), reverse=True)

    labels: List[str] = []
    deltas: List[float] = []
    colors: List[str] = []
    customdata: List[List[Any]] = []

    for idx, entry in enumerate(ordered, start=1):
        delta = float(entry.get("delta") or 0.0)
        admits = float(entry.get("admissions") or 0.0)
        count = int(entry.get("count") or 1)
        short = _path_short_label(entry["path"])
        labels.append(f"#{idx} · {short}")
        deltas.append(delta)
        if delta > 0:
            colors.append("#B5364B")
        elif delta < 0:
            colors.append("#1A8754")
        else:
            colors.append("#94A3B8")
        hover_path = (entry["path"] or "").replace(" → ", "<br>→ ")
        customdata.append([
            hover_path,
            _format_signed_money(delta),
            _format_signed_int(admits),
            count,
        ])

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=deltas,
            orientation="h",
            marker={"color": colors, "line": {"color": "#FFFFFF", "width": 1}},
            text=[_format_signed_money(d) for d in deltas],
            textposition="outside",
            textfont={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
            hovertemplate=(
                "<b>Drill path</b><br>%{customdata[0]}<br><br>"
                "Δ paid: %{customdata[1]}<br>"
                "Admits Δ: %{customdata[2]}<br>"
                "Contributing cells: %{customdata[3]}<extra></extra>"
            ),
            customdata=customdata,
            showlegend=False,
        )
    )
    fig.update_layout(
        height=max(220, 60 + 48 * len(ordered)),
        margin={"l": 12, "r": 96, "t": 8, "b": 28},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        yaxis={"autorange": "reversed", "title": "", "automargin": True},
        xaxis={
            "title": "Δ paid",
            "showgrid": True,
            "gridcolor": "#F1F4F9",
            "zerolinecolor": "#D6DBE4",
        },
        font={"family": "Inter, sans-serif", "size": 12, "color": "#1A2436"},
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#D6DBE4",
            "font": {"family": "Inter, sans-serif", "color": "#1A2436", "size": 12},
            "align": "left",
        },
    )
    return fig


def render_pattern_storyboard(
    pattern: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    rank: Any = None,
) -> bool:
    """Render the hero storyboard for a pattern tab: ranked drill paths + evidence card.

    Returns True when at least one panel emitted output. Falls back to the
    legacy bullet-list rendering when there are no aggregated paths AND no
    narrative payload to show.
    """

    impact = pattern.get("impact_summary") or {}
    direction = str(impact.get("direction") or "").upper()
    border_color = _PRIORITY_BORDER.get(direction, "#1F3A68")
    estimated_delta = impact.get("estimated_delta")

    title = (
        pattern.get("pattern_title")
        or pattern.get("top_pattern")
        or pattern.get("what_is_impacting")
        or "Pattern"
    )

    header_cols = st.columns([5, 2, 2])
    with header_cols[0]:
        st.markdown(f"### {title}")
        what_is_impacting = pattern.get("what_is_impacting")
        if what_is_impacting:
            st.caption(what_is_impacting)
    with header_cols[1]:
        if direction:
            st.markdown(
                f"<span style='display:inline-block;padding:6px 14px;border-radius:999px;"
                f"background:{border_color};color:#FFFFFF;font-weight:600;font-size:0.8rem;"
                f"letter-spacing:0.04em;'>{_esc(direction)}</span>",
                unsafe_allow_html=True,
            )
    with header_cols[2]:
        if estimated_delta:
            st.metric("Impact", str(estimated_delta))

    fig = build_pattern_breakdown_figure(pattern, cards)
    evidence = list(pattern.get("evidence_summary") or [])
    why = pattern.get("why_it_matters")
    next_step = pattern.get("recommended_next_step")

    if fig is None and not (evidence or why or next_step):
        return False

    left, right = st.columns([3, 2], gap="medium")
    with left:
        if fig is not None:
            n_bars = len(fig.data[0].y) if fig.data else 0
            caption = (
                "**Drill-down contributions** · "
                f"{n_bars} aggregated path{'s' if n_bars != 1 else ''}, ranked by |Δ paid| · "
                "hover for the full path"
            )
            st.markdown(caption)
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False},
                key=f"pattern_breakdown_{rank}",
            )
        else:
            st.info("No source cards available for this pattern's drill hierarchy.")
            render_drill_down_paths(pattern, cards)

    with right:
        # Native bordered container + a thin colored stripe on top encodes
        # the pattern direction without depending on the deprecated
        # streamlit-extras.stylable_container.
        with st.container(border=True):
            st.markdown(
                f"<div style='height:4px;border-radius:4px;background:{border_color};"
                f"margin:-4px -4px 12px -4px;'></div>",
                unsafe_allow_html=True,
            )
            _render_evidence_body(why, evidence, next_step, rank)

    return True


def _render_evidence_body(
    why: Optional[str],
    evidence: Sequence[Any],
    next_step: Optional[str],
    rank: Any,
) -> None:
    if why:
        st.markdown("**Why it matters**")
        st.write(why)
    if evidence:
        st.markdown("**Evidence**")
        for bullet in evidence:
            st.markdown(f"- {bullet}")
    if next_step:
        st.markdown("**Recommended next step**")
        st.write(next_step)
        st.button(
            "Mark as planned",
            key=f"pattern_action_{rank}",
            type="primary",
            width="stretch",
        )
