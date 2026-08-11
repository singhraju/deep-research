"""Build a PDF report of the deep-research analysis for executive readers.

Mirrors the on-screen tabs, minus SQL Queries and Technical Details:

- Cover with the user's question and generation timestamp
- Correlation Summary (executive summary + downstream counts)
- One section per Business Pattern with drill-down bar chart image, evidence,
  and reimbursement findings
- Final Recommendations grouped by priority

Callers use :func:`build_pdf_report` to turn the orchestrator's ``output`` dict
into PDF bytes suitable for ``st.download_button``. Plotly chart images are
included when ``kaleido`` is available; otherwise the drill-down data falls
back to a compact table so the report still renders end-to-end.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover — surfaced to the UI via is_available()
    _REPORTLAB_AVAILABLE = False

from ui.correlation_visuals import (
    _build_dim_pairs,
    _build_metric_columns,
    _derive_signal,
    _extract_matrix_schema,
    _get_metric_delta,
    _signal_label,
    build_kpi_data,
    build_waterfall_figure,
    format_compact_money,
    format_percent,
    format_period_window,
    format_signed_money,
)
from ui.pattern_visuals import build_drill_down_paths, build_pattern_breakdown_figure


_PRIORITY_BUCKETS = ("HIGH", "MEDIUM", "LOW")
_PRIORITY_COLORS_HEX = {
    "HIGH": "#B91C1C",
    "MEDIUM": "#B45309",
    "LOW": "#047857",
}
_DIRECTION_COLORS_HEX = {
    "INCREASE": "#B5364B",
    "DECREASE": "#1A8754",
    "STABLE": "#5B6776",
}


def is_available() -> bool:
    """Return True when reportlab is importable and PDF export can run."""

    return _REPORTLAB_AVAILABLE


# ---------------------------------------------------------------------------
# Value formatting helpers (kept independent of Streamlit / HTML rendering)
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


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


def _normalize_priority(value: Any) -> str:
    text = "" if value is None else str(value).strip().upper()
    if text in {"H", "HIGH", "URGENT", "CRITICAL"}:
        return "HIGH"
    if text in {"M", "MED", "MEDIUM"}:
        return "MEDIUM"
    if text in {"L", "LOW", "MINOR"}:
        return "LOW"
    return text or "MEDIUM"


# ---------------------------------------------------------------------------
# Chart image conversion via kaleido (optional)
# ---------------------------------------------------------------------------


def _figure_to_png_bytes(fig: Any) -> Optional[bytes]:
    """Convert a Plotly figure to PNG bytes via kaleido. Returns None on failure."""

    if fig is None:
        return None
    try:
        image_bytes = fig.to_image(format="png", scale=2)
    except Exception as exc:  # kaleido missing, browser missing, etc.
        logger.warning("Plotly to_image failed — chart will be omitted: %s", exc)
        return None
    return image_bytes


# ---------------------------------------------------------------------------
# ReportLab styles
# ---------------------------------------------------------------------------


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {}

    styles["title"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_LEFT,
        spaceAfter=6,
        textColor=colors.HexColor("#1A2436"),
    )
    styles["subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#5B6776"),
        spaceAfter=12,
    )
    styles["h1"] = ParagraphStyle(
        "ReportH1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=6,
        spaceAfter=8,
        textColor=colors.HexColor("#1A2436"),
    )
    styles["h2"] = ParagraphStyle(
        "ReportH2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#1F3A68"),
    )
    styles["h3"] = ParagraphStyle(
        "ReportH3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.HexColor("#1A2436"),
    )
    styles["body"] = ParagraphStyle(
        "ReportBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#212934"),
        spaceAfter=4,
    )
    styles["bullet"] = ParagraphStyle(
        "ReportBullet",
        parent=styles["body"],
        leftIndent=14,
        bulletIndent=2,
        spaceAfter=2,
    )
    styles["caption"] = ParagraphStyle(
        "ReportCaption",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#5B6776"),
        spaceAfter=6,
    )
    styles["badge_increase"] = ParagraphStyle(
        "BadgeIncrease",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#B5364B"),
    )
    styles["badge_decrease"] = ParagraphStyle(
        "BadgeDecrease",
        parent=styles["badge_increase"],
        textColor=colors.HexColor("#1A8754"),
    )
    styles["badge_stable"] = ParagraphStyle(
        "BadgeStable",
        parent=styles["badge_increase"],
        textColor=colors.HexColor("#5B6776"),
    )
    return styles


# ---------------------------------------------------------------------------
# Markdown → ReportLab paragraph helpers
# ---------------------------------------------------------------------------


def _escape_para(text: Any) -> str:
    """Escape angle brackets and ampersands so Paragraph doesn't parse them as tags."""

    if text is None:
        return ""
    s = str(text)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s


def _dim_pairs_pdf_markup(pairs: Sequence[Tuple[str, str]]) -> str:
    """Render dimension ``(label, value)`` pairs as stacked ReportLab markup.

    One dimension per line (``<br/>``-separated), the label in muted grey. Both
    label and value are escaped, so the result is safe to pass straight to a
    ``Paragraph`` — do NOT wrap it in ``_escape_para`` again.
    """
    if not pairs:
        return "—"
    return "<br/>".join(
        f'<font color="#6B7280">{_escape_para(label)}:</font> {_escape_para(value)}'
        for label, value in pairs
    )


def _markdown_to_para_html(text: Any) -> str:
    """Bare-minimum markdown to ReportLab inline-XML conversion.

    Handles the small subset our on-screen renderers emit: ``**bold**``,
    ``*italic*``, and inline links via ``[label](url)`` become ``<b>``,
    ``<i>``, and ``<link>`` tags. Everything else is escaped. Paragraph
    breaks come from callers splitting on ``\n\n``.
    """

    if text is None:
        return ""
    escaped = _escape_para(text)

    import re

    # Links: [label](url) → <link href="url" color="...">label</link>
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: (
            f'<link href="{m.group(2)}" color="#1F3A68"><u>{m.group(1)}</u></link>'
        ),
        escaped,
    )
    # Bold: **text**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # Italic: *text* (careful not to eat the ** we just consumed)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    # Newlines within a block → <br/>
    escaped = escaped.replace("\n", "<br/>")
    return escaped


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_markdown_to_para_html(text), style)


def _markdown_paragraphs(
    text: Any, style: ParagraphStyle
) -> List[Any]:
    """Split a markdown blob into blocks (paragraphs and bullet lists)."""

    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    flow: List[Any] = []
    # Split on blank lines to get logical blocks.
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        bullets: List[str] = []
        paras: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(("- ", "* ")):
                # Flush any accumulated non-bullet lines first.
                if paras:
                    flow.append(_p(" ".join(paras), style))
                    paras = []
                bullets.append(stripped[2:].strip())
            else:
                if bullets:
                    for bullet in bullets:
                        flow.append(Paragraph(f"• {_markdown_to_para_html(bullet)}", style))
                    bullets = []
                paras.append(line)
        if paras:
            flow.append(_p(" ".join(paras), style))
        if bullets:
            for bullet in bullets:
                flow.append(Paragraph(f"• {_markdown_to_para_html(bullet)}", style))
    return flow


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_cover(
    output: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = []
    question = str(output.get("question") or "Deep Research Analysis").strip() or "Deep Research Analysis"
    story.append(Paragraph("Deep Research Report", styles["title"]))
    story.append(Paragraph(_markdown_to_para_html(question), styles["subtitle"]))
    generated = dt.datetime.now().strftime("%B %d, %Y · %H:%M")
    story.append(Paragraph(f"Generated {generated}", styles["caption"]))
    
    # Add time period information if available
    analysis = output.get("analysis") or {}
    metadata = analysis.get("metadata") or {}
    correlation_summary = metadata.get("correlation_summary") or {}
    period_window = correlation_summary.get("period_window")
    
    period_text = format_period_window(period_window)
    if period_text:
        story.append(Paragraph(period_text, styles["caption"]))
    
    story.append(Spacer(1, 0.1 * inch))
    return story


def _build_correlation_summary_section(
    output: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = []
    analysis = output.get("analysis") or {}
    metadata = analysis.get("metadata") or {}
    correlation_summary = metadata.get("correlation_summary") or {}
    exec_summary = correlation_summary.get("executive_summary")
    research = output.get("research") or {}
    patterns = list(research.get("business_patterns") or [])
    recommendations = list(research.get("recommendations") or [])

    story.append(Paragraph("Executive Summary", styles["h1"]))
    if exec_summary:
        story.extend(_markdown_paragraphs(exec_summary, styles["body"]))
    else:
        story.append(Paragraph("No executive summary was produced for this analysis.", styles["caption"]))

    if patterns or recommendations:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("At a glance", styles["h3"]))
        data = [
            ["Business patterns", str(len(patterns))],
            ["Recommendations", str(len(recommendations))],
        ]
        table = Table(data, colWidths=[2.4 * inch, 1.2 * inch], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5B6776")),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1A2436")),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
    return story


def _build_kpi_strip_table(
    correlation_summary: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> Optional[Table]:
    """Four-card KPI strip mirroring the on-screen ``render_kpi_strip``."""

    data = build_kpi_data(correlation_summary)
    if data is None:
        return None

    def _truncate(text: str, limit: int = 30) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"

    impact_delta = (
        f"{data['impact_pct'] * 100:+.1f}% vs prior"
        if data["impact_pct"] is not None
        else "—"
    )
    top_driver_delta = (
        format_signed_money(data["top_driver_delta"])
        if data["top_driver_delta"]
        else "—"
    )
    range_text = (
        f"{format_compact_money(data['baseline'])} → "
        f"{format_compact_money(data['comparison'])}"
    )

    kpi_label_style = ParagraphStyle(
        "KpiLabel",
        parent=styles["caption"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#5B6776"),
        spaceAfter=2,
    )
    kpi_value_style = ParagraphStyle(
        "KpiValue",
        parent=styles["h3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1A2436"),
        spaceAfter=0,
    )
    kpi_sub_style = ParagraphStyle(
        "KpiSub",
        parent=styles["caption"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#5B6776"),
        spaceAfter=0,
    )

    def _cell(label: str, value: str, sub: str) -> List[Paragraph]:
        return [
            Paragraph(_escape_para(label), kpi_label_style),
            Paragraph(_escape_para(value), kpi_value_style),
            Paragraph(_escape_para(sub), kpi_sub_style),
        ]

    cells = [
        _cell("Total Impact", format_signed_money(data["impact"]), impact_delta),
        _cell(
            "Top Driver",
            _truncate(str(data["top_driver_name"] or "—")),
            top_driver_delta,
        ),
        _cell("Baseline → Comparison", range_text, str(data["metric_label"] or "")),
        _cell(
            "Detected Interactions",
            str(data["n_interactions"]),
            "Operational + clinical cells",
        ),
    ]

    table = Table([cells], colWidths=[1.65 * inch] * 4, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DBE4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _build_waterfall_text_fallback(
    correlation_summary: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> Optional[Table]:
    """Text-only stand-in for the horizontal waterfall bar chart.

    Used when kaleido can't render the Plotly figure (missing browser, headless
    CI). Shows the same drill-path segments the chart would surface.
    """

    drill_path = correlation_summary.get("drill_path") or []
    baseline = _coerce_float(correlation_summary.get("baseline_value"))
    comparison = _coerce_float(correlation_summary.get("comparison_value"))
    if not isinstance(drill_path, Sequence) or baseline is None or comparison is None:
        return None

    rows: List[List[Any]] = [
        [
            Paragraph("<b>Level</b>", styles["body"]),
            Paragraph("<b>Top drivers</b>", styles["body"]),
            Paragraph("<b>Δ paid</b>", styles["body"]),
        ]
    ]
    rows.append(
        [
            Paragraph("Baseline", styles["body"]),
            Paragraph("—", styles["body"]),
            Paragraph(_escape_para(format_compact_money(baseline)), styles["body"]),
        ]
    )
    for level in drill_path:
        if not isinstance(level, Mapping):
            continue
        dim = str(level.get("dimension") or "")
        segments = level.get("top_segments") or []
        seg_labels: List[str] = []
        level_total = 0.0
        for seg in list(segments)[:5]:
            if not isinstance(seg, Mapping):
                continue
            delta = _coerce_float(seg.get("aligned_delta"))
            if delta is None:
                delta = _coerce_float(seg.get("delta_value")) or 0.0
            level_total += delta
            seg_labels.append(
                f"{seg.get('value', '')} ({format_signed_money(delta)})"
            )
        if not seg_labels:
            continue
        rows.append(
            [
                Paragraph(_escape_para(f"L{level.get('level', '?')} · {dim}"), styles["body"]),
                Paragraph(_escape_para(", ".join(seg_labels)), styles["body"]),
                Paragraph(_escape_para(format_signed_money(level_total)), styles["body"]),
            ]
        )
    rows.append(
        [
            Paragraph("Comparison", styles["body"]),
            Paragraph("—", styles["body"]),
            Paragraph(_escape_para(format_compact_money(comparison)), styles["body"]),
        ]
    )

    if len(rows) <= 3:
        return None

    table = Table(rows, colWidths=[1.4 * inch, 4.0 * inch, 1.2 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F9")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#94A3B8")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _build_interaction_matrix_tables(
    correlation_summary: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
    doc_width: float,
) -> List[Any]:
    """Operational + clinical concentration tables from the interaction matrix."""

    matrix = correlation_summary.get("interaction_matrix") or {}
    if not isinstance(matrix, Mapping):
        return []

    operational = matrix.get("operational") if isinstance(matrix.get("operational"), Mapping) else {}
    clinical = matrix.get("clinical") if isinstance(matrix.get("clinical"), Mapping) else {}
    op_cells = list(operational.get("selected_cells") or [])
    cl_cells = list(clinical.get("selected_cells") or [])
    offset_cells = list(clinical.get("offset_cells_preview") or [])
    summary_status = ""
    if isinstance(matrix.get("summary"), Mapping):
        summary_status = str(matrix.get("summary").get("status") or "")
    if summary_status != "success" or not op_cells:
        return []

    _, _, ex_specs, op_order, cl_order = _extract_matrix_schema(matrix)
    metric_cols = _build_metric_columns(ex_specs)

    story: List[Any] = []

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Detected Interactions", styles["h2"]))
    story.append(
        Paragraph(
            "Operational cross-product cells selected from the waterfall, followed by "
            "clinical detail within the selected operational pocket.",
            styles["caption"],
        )
    )

    story.append(Paragraph("Operational Concentration", styles["h3"]))

    # Operational table: Rank · Interaction · Δ Paid · Share +Δ · Share Net Δ ·
    # <one column per explainer metric> · Signal.
    header_cells: List[Any] = [
        Paragraph("<b>Rank</b>", styles["body"]),
        Paragraph("<b>Interaction</b>", styles["body"]),
        Paragraph("<b>Δ Paid</b>", styles["body"]),
        Paragraph("<b>Share +Δ</b>", styles["body"]),
        Paragraph("<b>Share Net Δ</b>", styles["body"]),
    ]
    for _name, header, _fmt in metric_cols:
        header_cells.append(Paragraph(f"<b>{_escape_para(header)}</b>", styles["body"]))
    header_cells.append(Paragraph("<b>Signal</b>", styles["body"]))
    op_rows: List[List[Any]] = [header_cells]

    for idx, cell in enumerate(op_cells, start=1):
        if not isinstance(cell, Mapping):
            continue
        cell_markup = _dim_pairs_pdf_markup(_build_dim_pairs(cell, op_order))
        delta = _coerce_float(cell.get("delta_value")) or 0.0
        row: List[Any] = [
            Paragraph(str(idx), styles["body"]),
            Paragraph(cell_markup, styles["body"]),
            Paragraph(_escape_para(format_signed_money(delta)), styles["body"]),
            Paragraph(
                _escape_para(format_percent(cell.get("share_of_positive_delta", 0))),
                styles["body"],
            ),
            Paragraph(
                _escape_para(format_percent(cell.get("share_of_net_delta", 0))),
                styles["body"],
            ),
        ]
        for metric_name, _header, formatter in metric_cols:
            row.append(
                Paragraph(
                    _escape_para(formatter(_get_metric_delta(cell, metric_name))),
                    styles["body"],
                )
            )
        row.append(
            Paragraph(
                _escape_para(_signal_label(_derive_signal(cell, ex_specs))),
                styles["body"],
            )
        )
        op_rows.append(row)

    n_cols = len(op_rows[0])
    # Give Interaction ~40% of available width; distribute the rest evenly.
    interaction_width = doc_width * 0.34
    other_width = (doc_width - interaction_width) / max(1, n_cols - 1)
    op_widths = [other_width] * n_cols
    op_widths[1] = interaction_width
    op_table = Table(op_rows, colWidths=op_widths, hAlign="LEFT", repeatRows=1)
    op_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F9")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#94A3B8")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (2, 1), (-2, -1), "RIGHT"),
            ]
        )
    )
    story.append(op_table)

    # Clinical table
    if cl_cells or offset_cells:
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            Paragraph("Clinical Detail Within Selected Operational Cells", styles["h3"])
        )

        cl_header: List[Any] = [
            Paragraph("<b>Type</b>", styles["body"]),
            Paragraph("<b>Clinical cell</b>", styles["body"]),
            Paragraph("<b>Δ Paid</b>", styles["body"]),
            Paragraph("<b>Share Net Δ</b>", styles["body"]),
        ]
        for _name, header, _fmt in metric_cols:
            cl_header.append(Paragraph(f"<b>{_escape_para(header)}</b>", styles["body"]))
        cl_header.append(Paragraph("<b>Signal</b>", styles["body"]))
        cl_rows: List[List[Any]] = [cl_header]

        def _append_clinical_row(cell: Mapping[str, Any], type_label: str) -> None:
            cell_markup = _dim_pairs_pdf_markup(_build_dim_pairs(cell, cl_order, cl_label_map))
            delta = _coerce_float(cell.get("delta_value")) or 0.0
            delta_text = (
                format_signed_money(delta)
                if type_label == "Increase"
                else format_compact_money(delta)
            )
            row: List[Any] = [
                Paragraph(_escape_para(type_label), styles["body"]),
                Paragraph(cell_markup, styles["body"]),
                Paragraph(_escape_para(delta_text), styles["body"]),
                Paragraph(
                    _escape_para(format_percent(cell.get("share_of_net_delta", 0))),
                    styles["body"],
                ),
            ]
            for metric_name, _header, formatter in metric_cols:
                row.append(
                    Paragraph(
                        _escape_para(formatter(_get_metric_delta(cell, metric_name))),
                        styles["body"],
                    )
                )
            row.append(
                Paragraph(
                    _escape_para(_signal_label(_derive_signal(cell, ex_specs))),
                    styles["body"],
                )
            )
            cl_rows.append(row)

        for cell in cl_cells:
            if isinstance(cell, Mapping):
                _append_clinical_row(cell, "Increase")
        for cell in offset_cells:
            if isinstance(cell, Mapping):
                _append_clinical_row(cell, "Offset")

        n_cl_cols = len(cl_rows[0])
        cl_interaction_width = doc_width * 0.34
        cl_other_width = (doc_width - cl_interaction_width) / max(1, n_cl_cols - 1)
        cl_widths = [cl_other_width] * n_cl_cols
        cl_widths[1] = cl_interaction_width
        cl_table = Table(cl_rows, colWidths=cl_widths, hAlign="LEFT", repeatRows=1)
        cl_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F9")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#94A3B8")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (2, 1), (-2, -1), "RIGHT"),
                ]
            )
        )
        story.append(cl_table)

    return story


def _find_correlation_summary(output: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the correlation summary payload from wherever it lives on ``output``.

    Primary path: ``output.analysis.metadata.correlation_summary`` (what the
    orchestrator emits and the Streamlit UI reads). Falls back to
    ``output.correlation_summary`` and ``output.research.correlation_summary``
    so alternate shipping shapes still light up the section.
    """

    analysis = output.get("analysis") if isinstance(output.get("analysis"), Mapping) else {}
    metadata = analysis.get("metadata") if isinstance(analysis.get("metadata"), Mapping) else {}
    primary = metadata.get("correlation_summary")
    if isinstance(primary, Mapping) and primary:
        return primary

    top_level = output.get("correlation_summary")
    if isinstance(top_level, Mapping) and top_level:
        return top_level

    research = output.get("research") if isinstance(output.get("research"), Mapping) else {}
    research_level = research.get("correlation_summary")
    if isinstance(research_level, Mapping) and research_level:
        return research_level

    return {}


def _build_correlation_visuals_section(
    output: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
    doc_width: float,
) -> List[Any]:
    """KPI strip + horizontal waterfall + interaction matrix — the correlation tab's core visuals."""

    correlation_summary = _find_correlation_summary(output)
    if not correlation_summary:
        logger.info("PDF: correlation_summary not found on output; skipping visuals section.")
        return []

    story: List[Any] = []

    kpi_table = _build_kpi_strip_table(correlation_summary, styles)
    if kpi_table is not None:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Correlation Overview", styles["h2"]))
        story.append(kpi_table)
    else:
        logger.info(
            "PDF: KPI strip skipped (baseline_value=%s, comparison_value=%s).",
            correlation_summary.get("baseline_value"),
            correlation_summary.get("comparison_value"),
        )

    # Horizontal waterfall bar chart (Plotly → PNG via kaleido, text fallback).
    fig = build_waterfall_figure(correlation_summary)
    image_bytes = _figure_to_png_bytes(fig) if fig is not None else None
    if image_bytes or fig is not None:
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Cost Cascade", styles["h3"]))
    if image_bytes:
        buf = io.BytesIO(image_bytes)
        img = Image(buf)
        aspect = img.imageHeight / img.imageWidth if img.imageWidth else 0.5
        img.drawWidth = doc_width
        img.drawHeight = doc_width * aspect
        story.append(img)
    elif fig is None:
        logger.info(
            "PDF: waterfall figure skipped (drill_path=%s items, baseline=%s, comparison=%s).",
            len(correlation_summary.get("drill_path") or []),
            correlation_summary.get("baseline_value"),
            correlation_summary.get("comparison_value"),
        )
        fallback = _build_waterfall_text_fallback(correlation_summary, styles)
        if fallback is not None:
            story.append(Spacer(1, 0.12 * inch))
            story.append(Paragraph("Cost Cascade", styles["h3"]))
            story.append(fallback)
    else:
        # Figure built but kaleido couldn't rasterize it — use the text fallback.
        fallback = _build_waterfall_text_fallback(correlation_summary, styles)
        if fallback is not None:
            story.append(fallback)

    matrix_flow = _build_interaction_matrix_tables(correlation_summary, styles, doc_width)
    if matrix_flow:
        story.extend(matrix_flow)
    else:
        matrix = correlation_summary.get("interaction_matrix") or {}
        op_cells = []
        summary_status = ""
        if isinstance(matrix, Mapping):
            operational = matrix.get("operational")
            if isinstance(operational, Mapping):
                op_cells = list(operational.get("selected_cells") or [])
            if isinstance(matrix.get("summary"), Mapping):
                summary_status = str(matrix.get("summary").get("status") or "")
        logger.info(
            "PDF: interaction matrix skipped (status=%r, operational cells=%d).",
            summary_status,
            len(op_cells),
        )

    return story


def _direction_badge_para(direction: str, styles: Dict[str, ParagraphStyle]) -> Optional[Paragraph]:
    d = (direction or "").strip().upper()
    if not d:
        return None
    color_key = {
        "INCREASE": "badge_increase",
        "DECREASE": "badge_decrease",
    }.get(d, "badge_stable")
    return Paragraph(_escape_para(d), styles[color_key])


def _build_drill_down_bar_table(
    pattern: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    styles: Dict[str, ParagraphStyle],
) -> Optional[Table]:
    """Text-only fallback for the pattern drill-down chart."""

    paths = build_drill_down_paths(pattern, cards, top_n=10)
    if not paths:
        return None

    rows: List[List[Any]] = [
        [
            Paragraph("<b>Rank</b>", styles["body"]),
            Paragraph("<b>Drill-down path</b>", styles["body"]),
            Paragraph("<b>Δ paid</b>", styles["body"]),
            Paragraph("<b>Cells</b>", styles["body"]),
        ]
    ]
    for idx, entry in enumerate(paths, start=1):
        rows.append(
            [
                Paragraph(f"#{idx}", styles["body"]),
                Paragraph(_markdown_to_para_html(entry.get("path", "—")), styles["body"]),
                Paragraph(_escape_para(_format_signed_money(entry.get("delta"))), styles["body"]),
                Paragraph(str(entry.get("count") or 1), styles["body"]),
            ]
        )
    table = Table(
        rows,
        colWidths=[0.5 * inch, 4.6 * inch, 1.0 * inch, 0.5 * inch],
        hAlign="LEFT",
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F9")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#C5CBD8")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _pattern_impact_line(
    pattern: Mapping[str, Any],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    impact = pattern.get("impact_summary") or {}
    direction = str(impact.get("direction") or "").upper()
    estimated_delta = impact.get("estimated_delta")
    fragments: List[str] = []
    if direction:
        color = _DIRECTION_COLORS_HEX.get(direction, "#5B6776")
        fragments.append(
            f'<font color="{color}"><b>{_escape_para(direction)}</b></font>'
        )
    if estimated_delta:
        fragments.append(f"Impact: <b>{_escape_para(estimated_delta)}</b>")
    if not fragments:
        return []
    return [Paragraph(" &nbsp;·&nbsp; ".join(fragments), styles["body"])]


def _build_pattern_section(
    pattern: Mapping[str, Any],
    reimbursement: Optional[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
    styles: Dict[str, ParagraphStyle],
    *,
    doc_width: float,
) -> List[Any]:
    rank = pattern.get("pattern_rank") or pattern.get("rank") or ""
    title = (
        pattern.get("pattern_title")
        or pattern.get("top_pattern")
        or pattern.get("what_is_impacting")
        or "Pattern"
    )
    heading_text = f"Pattern {rank}: {title}" if rank else f"Pattern: {title}"

    story: List[Any] = [Paragraph(_markdown_to_para_html(heading_text), styles["h1"])]

    what = pattern.get("what_is_impacting")
    if what:
        story.append(Paragraph(_markdown_to_para_html(what), styles["caption"]))

    story.extend(_pattern_impact_line(pattern, styles))

    # Drill-down chart (image if kaleido; text table otherwise).
    fig = build_pattern_breakdown_figure(pattern, cards)
    image_bytes = _figure_to_png_bytes(fig) if fig is not None else None
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph("Drill-down Contributions", styles["h2"]))
    if image_bytes:
        buf = io.BytesIO(image_bytes)
        img = Image(buf)
        # Scale to fit page width while preserving aspect ratio.
        max_width = doc_width
        aspect = img.imageHeight / img.imageWidth if img.imageWidth else 0.5
        img.drawWidth = max_width
        img.drawHeight = max_width * aspect
        story.append(img)
    else:
        table = _build_drill_down_bar_table(pattern, cards, styles)
        if table is not None:
            story.append(table)
        else:
            story.append(
                Paragraph(
                    "No source-card contributions were available for this pattern.",
                    styles["caption"],
                )
            )

    # Evidence card
    why = pattern.get("why_it_matters")
    evidence = list(pattern.get("evidence_summary") or [])
    next_step = pattern.get("recommended_next_step")
    if why or evidence or next_step:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Evidence", styles["h2"]))
        if why:
            story.append(Paragraph("<b>Why it matters</b>", styles["h3"]))
            story.extend(_markdown_paragraphs(why, styles["body"]))
        if evidence:
            story.append(Paragraph("<b>Key evidence</b>", styles["h3"]))
            for item in evidence:
                story.append(
                    Paragraph(f"• {_markdown_to_para_html(item)}", styles["bullet"])
                )
        if next_step:
            story.append(Paragraph("<b>Recommended next step</b>", styles["h3"]))
            story.extend(_markdown_paragraphs(next_step, styles["body"]))

    # Pattern details, if present.
    details = pattern.get("pattern_details")
    if details:
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph("Pattern Details", styles["h2"]))
        story.extend(_markdown_paragraphs(details, styles["body"]))

    # Reimbursement Policy Findings
    story.extend(_build_reimbursement_section(reimbursement, styles))

    return story


def _build_reimbursement_section(
    reimbursement: Optional[Mapping[str, Any]],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = []
    if not reimbursement:
        return story
    if reimbursement.get("error"):
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Reimbursement Policy Findings", styles["h2"]))
        story.append(
            Paragraph(
                f"Reimbursement agent failed: {_escape_para(reimbursement.get('error'))}",
                styles["caption"],
            )
        )
        return story

    formatted = reimbursement.get("formatted_output") or reimbursement.get("output") or {}
    if not isinstance(formatted, Mapping):
        formatted = {}

    elevance = reimbursement.get("elevance_executive_summary") or formatted.get(
        "elevance_executive_summary"
    )
    summary_table = formatted.get("summary_table")
    actions = (
        reimbursement.get("recommended_action")
        or reimbursement.get("recommendations")
        or formatted.get("recommended_action")
        or []
    )
    policies = (
        formatted.get("individual_policies")
        or reimbursement.get("individual_policies")
        or formatted.get("reimbursement_policies")
        or reimbursement.get("reimbursement_policies")
        or []
    )

    if not any([elevance, summary_table, actions, policies]):
        return story

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Reimbursement Policy Findings", styles["h2"]))

    if elevance:
        story.append(Paragraph("Elevance Executive Summary", styles["h3"]))
        story.extend(_markdown_paragraphs(elevance, styles["body"]))

    payer_table = _build_payer_summary_table(summary_table, styles)
    if payer_table is not None:
        story.append(Spacer(1, 0.05 * inch))
        title_text = "Payer Policy Summary"
        subtitle_text = None
        if isinstance(summary_table, Mapping):
            title_text = str(summary_table.get("title") or title_text)
            subtitle_text = summary_table.get("subtitle")
        story.append(Paragraph(_markdown_to_para_html(title_text), styles["h3"]))
        if subtitle_text:
            story.append(Paragraph(_markdown_to_para_html(subtitle_text), styles["caption"]))
        story.append(payer_table)

    if actions:
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph("Recommended Actions", styles["h3"]))
        for idx, action in enumerate(actions, start=1):
            if not isinstance(action, Mapping):
                continue
            rank = action.get("rank", idx)
            priority = action.get("priority") or ""
            description = action.get("description") or action.get("title") or ""
            if not description:
                continue
            head = f"<b>Recommendation {_escape_para(rank)}"
            if priority:
                head += f" · {_escape_para(priority).upper()}"
            head += ":</b> "
            story.append(Paragraph(head + _markdown_to_para_html(description), styles["body"]))

    if policies:
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph("Referenced Policies", styles["h3"]))
        for policy in policies:
            if not isinstance(policy, Mapping):
                continue
            title = str(
                policy.get("policy_title")
                or policy.get("title")
                or policy.get("policy_title_text")
                or ""
            ).strip()
            if not title:
                continue
            payer = str(
                policy.get("payer_name")
                or policy.get("payer")
                or policy.get("payer_org")
                or ""
            ).strip()
            url = str(policy.get("policy_url") or policy.get("external_link") or "").strip()
            effective = str(policy.get("effective_date") or "").strip()
            parts: List[str] = []
            if payer:
                parts.append(f"<b>{_escape_para(payer)}</b>")
            if url:
                parts.append(
                    f'<link href="{_escape_para(url)}" color="#1F3A68"><u>{_escape_para(title)}</u></link>'
                )
            else:
                parts.append(_escape_para(title))
            if effective:
                parts.append(f"<i>Effective {_escape_para(effective)}</i>")
            story.append(Paragraph("• " + " — ".join(parts), styles["bullet"]))

    return story


def _build_payer_summary_table(
    summary_table: Any,
    styles: Dict[str, ParagraphStyle],
) -> Optional[Table]:
    if not isinstance(summary_table, Mapping):
        return None
    columns = list(summary_table.get("columns") or [])
    rows = list(summary_table.get("rows") or [])
    if not columns or not rows:
        return None

    header_cells = [
        Paragraph(
            f"<b>{_markdown_to_para_html(col.get('label') or col.get('id') or '')}</b>",
            styles["body"],
        )
        for col in columns
    ]
    body_rows: List[List[Any]] = [header_cells]
    for row in rows:
        cells: List[Any] = []
        for col in columns:
            col_id = col.get("id")
            value = row.get(col_id, "") if col_id else ""
            value_text = "" if value is None else str(value)
            if not value_text or value_text == "-":
                value_text = "—"
            cells.append(Paragraph(_markdown_to_para_html(value_text), styles["body"]))
        body_rows.append(cells)

    # Distribute width evenly; first column slightly wider to hold payer names.
    n_cols = len(columns)
    if n_cols <= 0:
        return None
    total = 6.6  # inches available inside our default 0.75" margins on Letter
    first_weight = 1.3
    other_weight = (total - (total * first_weight / (first_weight + n_cols - 1))) / max(1, n_cols - 1)
    first_width = total * first_weight / (first_weight + n_cols - 1)
    col_widths = [first_width * inch] + [other_weight * inch] * (n_cols - 1)

    table = Table(body_rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667EEA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
            ]
        )
    )
    return table


def _build_recommendations_section(
    recommendations: Sequence[Mapping[str, Any]],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = [Paragraph("Final Recommendations", styles["h1"])]
    if not recommendations:
        story.append(Paragraph("No recommendations were generated for this analysis.", styles["caption"]))
        return story

    bucketed: Dict[str, List[Mapping[str, Any]]] = {p: [] for p in _PRIORITY_BUCKETS}
    other: List[Mapping[str, Any]] = []
    for rec in recommendations:
        if not isinstance(rec, Mapping):
            continue
        pri = _normalize_priority(rec.get("priority"))
        if pri in bucketed:
            bucketed[pri].append(rec)
        else:
            other.append(rec)

    counts = [
        ["Total", str(len(recommendations))],
        ["High priority", str(len(bucketed["HIGH"]))],
        ["Medium priority", str(len(bucketed["MEDIUM"]))],
        ["Low priority", str(len(bucketed["LOW"]))],
    ]
    counts_table = Table(counts, colWidths=[1.6 * inch, 1.0 * inch], hAlign="LEFT")
    counts_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5B6776")),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(counts_table)

    for pri in _PRIORITY_BUCKETS:
        items = bucketed[pri]
        color = _PRIORITY_COLORS_HEX[pri]
        story.append(Spacer(1, 0.1 * inch))
        header_html = (
            f'<font color="{color}"><b>{pri}</b></font> · {len(items)} item(s)'
        )
        story.append(Paragraph(header_html, styles["h2"]))
        if not items:
            story.append(Paragraph("No items in this priority.", styles["caption"]))
            continue
        for rec in items:
            story.extend(_build_recommendation_card(rec, color, styles))

    if other:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Other / Unclassified", styles["h2"]))
        for rec in other:
            story.extend(_build_recommendation_card(rec, _PRIORITY_COLORS_HEX["MEDIUM"], styles))

    return story


def _build_recommendation_card(
    rec: Mapping[str, Any],
    color: str,
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    rank = rec.get("rank")
    title = rec.get("description") or rec.get("title") or (f"Recommendation {rank}" if rank else "Recommendation")
    owner = rec.get("owner") or rec.get("assignee")
    eta = rec.get("eta") or rec.get("eta_weeks")
    evidence = rec.get("evidence") or []
    peer = rec.get("peer_benchmarking") or []

    lines: List[Any] = []
    head = ""
    if rank:
        head += f'<font color="{color}"><b>#{_escape_para(rank)}</b></font>&nbsp; '
    head += f"<b>{_markdown_to_para_html(title)}</b>"
    lines.append(Paragraph(head, styles["body"]))

    meta_bits: List[str] = []
    if owner:
        meta_bits.append(f"Owner: {_escape_para(owner)}")
    if eta:
        meta_bits.append(f"ETA: {_escape_para(eta)}")
    if meta_bits:
        lines.append(Paragraph(" · ".join(meta_bits), styles["caption"]))

    if evidence:
        lines.append(Paragraph("<b>Evidence</b>", styles["h3"]))
        for item in evidence:
            lines.append(Paragraph(f"• {_markdown_to_para_html(item)}", styles["bullet"]))
    if peer:
        lines.append(Paragraph("<b>Peer benchmarking</b>", styles["h3"]))
        for item in peer:
            lines.append(Paragraph(f"• {_markdown_to_para_html(item)}", styles["bullet"]))

    lines.append(Spacer(1, 0.06 * inch))
    return [KeepTogether(lines)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_pdf_report(output: Mapping[str, Any]) -> bytes:
    """Turn the orchestrator's ``output`` dict into a PDF report (bytes).

    Skips the SQL Queries and Technical Details tabs — the report is meant for
    business analysts and executives who plan to reuse content in slide decks.
    """

    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is required to export the report as PDF. "
            "Install it with `uv add reportlab kaleido`."
        )

    if not isinstance(output, Mapping):
        raise TypeError("build_pdf_report expects a mapping (the orchestrator output dict).")

    styles = _build_styles()
    research = output.get("research") or {}
    patterns: List[Mapping[str, Any]] = list(research.get("business_patterns") or [])
    reimbursement_by_pattern = research.get("reimbursement_by_pattern") or {}
    if not isinstance(reimbursement_by_pattern, Mapping):
        reimbursement_by_pattern = {}
    cards: List[Mapping[str, Any]] = list(
        (research.get("pattern_summary") or {}).get("cards") or []
    )
    recommendations: List[Mapping[str, Any]] = list(research.get("recommendations") or [])

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Deep Research Report",
        author="Deep Research",
    )
    doc_width = doc.width  # already accounts for margins

    story: List[Any] = []
    story.extend(_build_cover(output, styles))
    story.extend(_build_correlation_summary_section(output, styles))
    story.extend(_build_correlation_visuals_section(output, styles, doc_width))

    for idx, pattern in enumerate(patterns):
        rank = str(pattern.get("pattern_rank", idx + 1))
        story.append(PageBreak())
        story.extend(
            _build_pattern_section(
                pattern,
                reimbursement_by_pattern.get(rank) if isinstance(reimbursement_by_pattern, Mapping) else None,
                cards,
                styles,
                doc_width=doc_width,
            )
        )

    story.append(PageBreak())
    story.extend(_build_recommendations_section(recommendations, styles))

    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    return buffer.getvalue()


def _add_footer(canvas_obj: Any, doc_obj: Any) -> None:
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.HexColor("#8A93A2"))
    footer = f"Deep Research  ·  Page {doc_obj.page}"
    canvas_obj.drawRightString(LETTER[0] - 0.75 * inch, 0.4 * inch, footer)
    canvas_obj.restoreState()


def suggested_filename(output: Mapping[str, Any]) -> str:
    """Return a friendly filename for the download button."""

    question = str(output.get("question") or "deep_research").strip() or "deep_research"
    slug = "".join(ch if ch.isalnum() else "_" for ch in question).strip("_").lower()
    slug = slug[:60] or "deep_research"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    return f"{slug}_{stamp}.pdf"
