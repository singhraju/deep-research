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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


# Optional per-key display formatters — apply only when the config's dimension
# vocabulary happens to include one of these legacy keys. Any other key falls
# through to a plain string.
def _pretty_dim_value(key: str, value: str, extras: Mapping[str, Any]) -> str:
    if key == "pa_required_code":
        if value == "Y":
            return "PA Y"
        if value == "N":
            return "PA N"
    if key == "product_description" and extras.get("er_admit_indicator") == "Y":
        return f"{value} (ER)"
    return value


def _format_dim_values(
    dims: Mapping[str, Any],
    *,
    extras: Optional[Mapping[str, Any]] = None,
    truncate_at: Optional[int] = None,
) -> List[str]:
    """Return display strings for every valid entry in ``dims``, config-agnostic.

    ``extras`` is passed to ``_pretty_dim_value`` so legacy niceties like
    ``product_description (ER)`` still fire when the sibling flag exists.
    """
    extras = extras or {}
    out: List[str] = []
    if not isinstance(dims, Mapping):
        return out
    for key, value in dims.items():
        if not _is_valid_business_value(value):
            continue
        text = _pretty_dim_value(str(key), str(value), extras)
        if truncate_at is not None and len(text) > truncate_at:
            text = text[: max(0, truncate_at - 3)] + "..."
        out.append(text)
    return out


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

        # Section 2: operational drill-down filters — iterate every dim the
        # correlation agent emitted for this card, format optional keys with a
        # nicer display, and drop technical/unknown values.
        filters = _format_dim_values(dimensions, extras=context)
        if filters:
            components.append(" + ".join(filters))

        # Section 3: clinical context — same treatment on context_dimensions.
        clinical = _format_dim_values(context, truncate_at=45)
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


_CITATION_URL_RE = re.compile(r"https?://\S+")

# Citations from the reimbursement agent are shaped
# ``"<payer> · <title> — <url>"`` (em-dash) — see reimbursement_agent's
# citation formatter. The URL segment is optional so we still get a payer/title
# pair even when the LLM omitted the link.
_CITATION_STRUCT_RE = re.compile(
    r"^\s*(?P<payer>[^·]+?)\s*·\s*(?P<title>.+?)\s*(?:[—\-–]\s*(?P<url>https?://\S+))?\s*$"
)


def _parse_citation_entries(
    action: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Parse ``citation`` strings into structured ``{payer, title, url}`` dicts.

    Falls back to raw URL extraction when the ``<payer> · <title> — <url>``
    shape doesn't match (older callouts). Deduped in-order so the caller can
    iterate deterministically.
    """

    citations = action.get("citation") or action.get("citations") or []
    parsed: List[Dict[str, str]] = []
    seen: set = set()
    for entry in citations:
        if not isinstance(entry, str):
            continue
        text = entry.strip()
        if not text:
            continue
        match = _CITATION_STRUCT_RE.match(text)
        if match:
            payer = match.group("payer").strip()
            title = match.group("title").strip()
            url = (match.group("url") or "").rstrip(").,;")
        else:
            url_match = _CITATION_URL_RE.search(text)
            url = url_match.group(0).rstrip(").,;") if url_match else ""
            payer = ""
            title = ""
        key = (payer.lower(), title.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        parsed.append({"payer": payer, "title": title, "url": url})
    return parsed


def _extract_citation_urls(action: Mapping[str, Any]) -> List[str]:
    """Return cited policy URLs from a recommendation, preserving order.

    Retained for callers that only need URLs — prefer
    :func:`_parse_citation_entries` when payer/title fallback lookups matter.
    """

    urls: List[str] = []
    seen: set = set()
    for entry in _parse_citation_entries(action):
        url = entry.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _join_facts(values: Any, *, limit: int = 6) -> str:
    """Comma-join edit_rule_fact values with a '+N more' tail when long."""

    if not values:
        return "—"
    if isinstance(values, str):
        return values
    items = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not items:
        return "—"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" +{len(items) - limit} more"


def _format_edit_rule_facts(policy: Mapping[str, Any]) -> Dict[str, str]:
    """Return short display strings for the key edit_rule_facts columns."""

    facts = policy.get("edit_rule_facts") or {}
    if not isinstance(facts, Mapping):
        facts = {}
    return {
        "target_codes": _join_facts(facts.get("target_codes")),
        "required_modifiers": _join_facts(facts.get("required_modifiers")),
        "action_types": _join_facts(facts.get("action_types")),
    }


_SUBREC_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(Recommendation\s+\d+|IA)\s*([:—–\-])\s+",
    re.IGNORECASE,
)


def _split_recommendation_description(
    description: str,
) -> List[Dict[str, str]]:
    """Split a recommendation description into sub-recommendations.

    Recognizes headings like ``Recommendation 1:`` and ``IA —``. Returns a list
    of ``{"label", "sep", "text"}`` dicts. When no headings are found, returns a
    single entry with an empty label so callers can fall back to legacy display.
    """

    text = description.strip()
    matches = list(_SUBREC_HEADING_RE.finditer(text))
    if not matches:
        return [{"label": "", "sep": "", "text": text}]

    parts: List[Dict[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(
            {
                "label": match.group(1).strip(),
                "sep": match.group(2),
                "text": text[start:end].strip(),
            }
        )
    return parts


def _payer_base_name(payer_name: str) -> str:
    """Strip trailing parenthetical qualifiers, e.g. 'Elevance Health (internal)'."""

    return re.sub(r"\s*\([^)]*\)\s*", " ", payer_name).strip()


# Tokens that would produce false positives if used alone as a payer identifier.
# We keep the full base name too — this list only guards the "first significant
# token" candidate that catches suffix-coded forms like ``Highmark_BCBS``.
_PAYER_GENERIC_TOKENS = frozenset(
    {
        "the",
        "of",
        "in",
        "and",
        "a",
        "an",
        "health",
        "healthcare",
        "plan",
        "plans",
        "insurance",
        "group",
        "care",
        "medical",
        "blue",
        "cross",
        "shield",
        "bcbs",
    }
)


def _payer_match_candidates(payer_name: str) -> List[str]:
    """Return distinct substrings to look for when attributing by payer name.

    Covers the common shapes that appear in orchestrator payloads:

    * ``Elevance Health (external)`` → parens stripped → also try ``Elevance``
    * ``Highmark_BCBS`` / ``CareFirst_BCBS`` → also try ``Highmark`` / ``CareFirst``
      (LLM recommendations usually drop the ``_BCBS`` suffix)
    * ``Horizon NJ Health - New Jersey`` → also try ``Horizon NJ Health``
      (the piece before the state qualifier)

    Ordered most-specific first so a hit on the fully-qualified name is
    preferred over the shortened forms.
    """

    base = _payer_base_name(payer_name)
    if not base:
        return []

    seen: set = set()
    candidates: List[str] = []

    def _add(candidate: str) -> None:
        cand = candidate.strip()
        if not cand:
            return
        key = cand.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cand)

    _add(base)
    # Underscores → spaces so "Highmark_BCBS" also matches "Highmark BCBS".
    _add(base.replace("_", " "))
    # Segment before the first ``-`` or ``_`` separator: "Horizon NJ Health -
    # New Jersey" → "Horizon NJ Health", "Highmark_BCBS" → "Highmark".
    head = re.split(r"[_\-]", base, maxsplit=1)[0].strip()
    if head:
        _add(head)
    # First significant token (skips generic words like "the"/"health"/"blue"
    # so we don't accidentally match unrelated payers by a common suffix).
    for token in re.split(r"[\s_\-]+", base):
        if not token:
            continue
        if len(token) < 4:
            continue
        if token.lower() in _PAYER_GENERIC_TOKENS:
            continue
        _add(token)
        break

    return candidates


_EVIDENCE_RE = re.compile(r'^(.+?)\s+·\s+(.+?):\s+"(.+?)"\s+\[(.+?)\]\s*$')

# Evidence in these fields is usually the shared CPT/HCPCS code list (A0427,
# A0429, …) that appears in every sub-recommendation, so it's useless for
# attribution — skip it when deciding which sub-rec a policy belongs to.
_LOW_SIGNAL_EVIDENCE_FIELDS = {"target_codes", "related_codes"}


def _parse_evidence_entries(entries: Sequence[Any]) -> List[Dict[str, str]]:
    """Parse ``evidence`` strings into structured dicts.

    Each recommended-action's ``evidence`` list carries entries shaped
    ``"Payer · Policy Title: \"value\" [field_name]"``. Silently skips entries
    that don't match this shape.
    """

    parsed: List[Dict[str, str]] = []
    for entry in entries or []:
        if not isinstance(entry, str):
            continue
        match = _EVIDENCE_RE.match(entry.strip())
        if not match:
            continue
        parsed.append(
            {
                "payer": match.group(1).strip(),
                "title": match.group(2).strip(),
                "value": match.group(3).strip(),
                "field": match.group(4).strip(),
            }
        )
    return parsed


def _value_appears_in_text(value: str, text: str) -> bool:
    """Return True when ``value`` appears meaningfully in ``text``.

    Values 4+ chars use plain substring match. Shorter values require the
    surrounding characters to be neither alphanumeric nor a hyphen — so ``"D"``
    matches ``"modifiers from D, E, ..."`` (isolated token) but not ``"D-SNP"``
    (which is a compound acronym, not a modifier code).
    """

    if not value or not text:
        return False
    value_lc = value.lower()
    text_lc = text.lower()
    if len(value) >= 4:
        return value_lc in text_lc
    pattern = r"(?<![\w-])" + re.escape(value_lc) + r"(?![\w-])"
    return bool(re.search(pattern, text_lc))


def _policy_lookup_key(payer: str, title: str) -> Tuple[str, str]:
    return (payer.strip().lower(), title.strip().lower())


def _attribute_cited_policies(
    cited: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, str]],
    sub_recs: Sequence[Mapping[str, str]],
) -> Tuple[List[List[Mapping[str, Any]]], List[Mapping[str, Any]]]:
    """Assign each cited policy to the sub-recommendation(s) it supports.

    Returns ``(per_subrec_matches, unmatched)``.

    Rules:

    1. **Evidence-based**: a policy is attributed to a sub-rec whenever one of
       its evidence values (excluding low-signal fields like ``target_codes``)
       appears in that sub-rec's text. This is the strong signal — the
       recommendation directly quotes the policy's extracted rule.
    2. **Payer-name (additive)**: for any sub-rec whose text mentions one of
       the payer's identifying substrings (see
       :func:`_payer_match_candidates`), the policy is also attributed there
       — even if evidence already matched a different sub-rec. This catches
       peer mentions like ``"Peer: Cigna requires ..."`` while still allowing
       the same policy to appear under a different sub-rec that quotes one of
       its edit rules verbatim.
    3. Anything still unmatched is returned in the unmatched list for the
       "Additional referenced policies" fallback block.
    """

    evidence_by_policy: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    for ev in evidence:
        if ev.get("field") in _LOW_SIGNAL_EVIDENCE_FIELDS:
            continue
        key = _policy_lookup_key(ev.get("payer", ""), ev.get("title", ""))
        evidence_by_policy.setdefault(key, []).append(ev)

    per_subrec: List[List[Mapping[str, Any]]] = [[] for _ in sub_recs]
    unmatched: List[Mapping[str, Any]] = []

    for policy in cited:
        payer = str(
            policy.get("payer_name") or policy.get("payer") or ""
        ).strip()
        title = str(
            policy.get("policy_title") or policy.get("title") or ""
        ).strip()
        key = _policy_lookup_key(payer, title)
        policy_ev = evidence_by_policy.get(key, [])

        matched_indexes: List[int] = []
        for idx, sub_rec in enumerate(sub_recs):
            text = sub_rec.get("text", "")
            if any(
                _value_appears_in_text(ev.get("value", ""), text)
                for ev in policy_ev
            ):
                matched_indexes.append(idx)

        # Additive payer-name attribution: fill in any sub-rec that names the
        # payer but wasn't already picked up by an evidence match. This keeps
        # peer-list mentions (``"Peer: Cigna, ..."``) from being dropped when
        # the same policy also has a verbatim evidence hit elsewhere.
        candidates = _payer_match_candidates(payer)
        if candidates:
            candidates_lc = [cand.lower() for cand in candidates]
            for idx, sub_rec in enumerate(sub_recs):
                if idx in matched_indexes:
                    continue
                text_lc = sub_rec.get("text", "").lower()
                if any(cand in text_lc for cand in candidates_lc):
                    matched_indexes.append(idx)

        if matched_indexes:
            matched_indexes.sort()
            for idx in matched_indexes:
                per_subrec[idx].append(policy)
        else:
            unmatched.append(policy)

    return per_subrec, unmatched


def _normalize_title_key(title: str) -> str:
    """Loosely normalise a policy title for fuzzy lookup.

    Lowercase, collapse whitespace, strip common trailing qualifiers so
    ``"Ambulance Services (Commercial)"`` and ``"Ambulance Services"`` collide.
    """

    text = re.sub(r"\s+", " ", (title or "").strip().lower())
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
    return text


def _payer_key(payer: str) -> str:
    return _payer_base_name(payer or "").lower()


def _synthesize_citation_policy(entry: Mapping[str, str]) -> Dict[str, Any]:
    """Build a minimal policy record from a parsed citation entry.

    Used when a citation names a payer/title/url that isn't in
    ``individual_policies`` — the recommendation still needs a row so the
    reader can see (and click through to) the cited source.
    """

    title = entry.get("title") or entry.get("url") or ""
    return {
        "policy_title": title,
        "payer_name": entry.get("payer", ""),
        "policy_url": entry.get("url", ""),
        "effective_date": "",
        "edit_rule_facts": {},
    }


def _cited_policies_for_action(
    action: Mapping[str, Any],
    policy_by_url: Mapping[str, Mapping[str, Any]],
    policies: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Mapping[str, Any]]:
    """Return the policy records cited by ``action`` in citation order.

    Lookup order per citation:

    1. Exact ``policy_url`` match against ``policy_by_url`` (the strongest
       signal — same document).
    2. Normalised ``(payer, title)`` match against ``policies`` — catches
       cases where the LLM's cited URL differs from the ingested policy's
       stored URL, or the policy was quarantined from ``individual_policies``
       but is still referenced in ``policies``.
    3. Normalised title-only match — a last-resort fallback for the same
       policy under a slightly-different payer label (e.g. ``"Elevance"``
       vs ``"Elevance Health (external)"``).
    4. Synthesise a minimal record from the citation itself so the row still
       appears with the payer + linked title. Prevents the LLM from citing a
       policy that then silently disappears from the UI.
    """

    entries = _parse_citation_entries(action)
    if not entries:
        return []

    by_payer_title: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    by_title: Dict[str, Mapping[str, Any]] = {}
    for policy in policies or []:
        if not isinstance(policy, Mapping):
            continue
        payer_field = str(
            policy.get("payer_name") or policy.get("payer") or ""
        )
        title_field = str(
            policy.get("policy_title") or policy.get("title") or ""
        )
        title_key = _normalize_title_key(title_field)
        if not title_key:
            continue
        by_payer_title.setdefault(
            (_payer_key(payer_field), title_key), policy
        )
        by_title.setdefault(title_key, policy)

    cited: List[Mapping[str, Any]] = []
    seen_ids: set = set()

    def _remember(policy: Mapping[str, Any]) -> None:
        # Dedup on URL when available, else on (payer, title) identity so
        # synthesized-and-then-matched records don't double up.
        ident = (
            str(policy.get("policy_url") or policy.get("external_link") or ""),
            str(policy.get("payer_name") or policy.get("payer") or ""),
            str(policy.get("policy_title") or policy.get("title") or ""),
        )
        if ident in seen_ids:
            return
        seen_ids.add(ident)
        cited.append(policy)

    for entry in entries:
        url = entry.get("url", "")
        title_key = _normalize_title_key(entry.get("title", ""))
        payer_lookup = _payer_key(entry.get("payer", ""))

        policy: Optional[Mapping[str, Any]] = None
        if url:
            policy = policy_by_url.get(url)
        if policy is None and title_key and payer_lookup:
            policy = by_payer_title.get((payer_lookup, title_key))
        if policy is None and title_key:
            policy = by_title.get(title_key)
        if policy is None:
            policy = _synthesize_citation_policy(entry)

        _remember(policy)

    return cited


def _render_policies_table(policies: Sequence[Mapping[str, Any]]) -> str:
    """Render an HTML table for a list of policy records. Empty string if none."""

    if not policies:
        return ""

    rows: List[str] = []
    for policy in policies:
        url = str(
            policy.get("policy_url") or policy.get("external_link") or ""
        ).strip()
        title = str(
            policy.get("policy_title") or policy.get("title") or url
        ).strip()
        payer = str(
            policy.get("payer_name") or policy.get("payer") or ""
        ).strip()
        effective = str(policy.get("effective_date") or "").strip() or "—"
        facts = _format_edit_rule_facts(policy)
        if url:
            policy_link = (
                f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#3730a3; text-decoration:underline;">{_esc(title)}</a>'
            )
        else:
            policy_link = _esc(title)
        cell = (
            'style="padding:6px 10px; border-bottom:1px solid #e5e7eb; '
            'font-size:0.82em; vertical-align:top; color:#1f2937;"'
        )
        rows.append(
            "<tr>"
            f'<td {cell}>{policy_link}</td>'
            f'<td {cell}>{_esc(payer or "—")}</td>'
            f'<td {cell}>{_esc(effective)}</td>'
            f'<td {cell}>{_esc(facts["target_codes"])}</td>'
            f'<td {cell}>{_esc(facts["required_modifiers"])}</td>'
            f'<td {cell}>{_esc(facts["action_types"])}</td>'
            "</tr>"
        )

    if not rows:
        return ""

    header_cell = (
        'style="padding:6px 10px; text-align:left; font-size:0.78em; '
        'text-transform:uppercase; letter-spacing:0.03em; color:#4b5563; '
        'border-bottom:2px solid #c7d2fe; background:#f5f6ff;"'
    )
    return (
        '<div style="margin:8px 0 14px 0; background:#ffffff; '
        'border:1px solid #d6d9f0; border-radius:4px; overflow:hidden;">'
        '<table style="width:100%; border-collapse:collapse;">'
        "<thead><tr>"
        f'<th {header_cell}>Policy</th>'
        f'<th {header_cell}>Payer</th>'
        f'<th {header_cell}>Effective Date</th>'
        f'<th {header_cell}>Target Codes</th>'
        f'<th {header_cell}>Required Modifiers</th>'
        f'<th {header_cell}>Action</th>'
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def _priority_chip_html(priority: str) -> str:
    """Return the coloured priority chip span, or '' if no priority."""

    priority = (priority or "").strip()
    if not priority:
        return ""
    color = {
        "high": "#dc3545",
        "medium": "#fd7e14",
        "low": "#198754",
    }.get(priority.lower(), "#6c757d")
    return (
        f'<span style="margin-left:8px; padding:2px 8px; border-radius:10px; '
        f'background:{color}; color:white; font-size:0.7em; font-weight:600; '
        f'text-transform:uppercase;">{_esc(priority)}</span>'
    )


def _render_action_with_subrecs(
    rank: Any,
    priority_chip: str,
    sub_recs: Sequence[Mapping[str, str]],
    cited_policies: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, str]],
) -> str:
    """Render one outer action as a stack of labelled sub-recommendation blocks.

    Each sub-rec gets its own policy table, filled by
    :func:`_attribute_cited_policies` (evidence-value matches first, payer
    name fallback second). Policies not attributed to any sub-rec fall into
    an 'Additional referenced policies' block at the bottom.
    """

    header = (
        f'<div style="margin-bottom:10px;">'
        f'<strong>Recommendation {_esc(rank)}</strong>{priority_chip}'
        f'</div>'
    )

    per_subrec, unmatched = _attribute_cited_policies(
        cited_policies, evidence, sub_recs
    )

    body_parts: List[str] = []
    for entry, matches in zip(sub_recs, per_subrec):
        label = entry.get("label", "")
        sep = entry.get("sep") or ":"
        text = entry.get("text", "")
        sub_table = _render_policies_table(matches)
        label_html = (
            f'<strong>{_esc(label)}{_esc(sep)}</strong> ' if label else ""
        )
        body_parts.append(
            f'<div style="margin-bottom:14px;">'
            f'{label_html}{_esc(text)}'
            f'{sub_table}'
            f'</div>'
        )

    if unmatched:
        body_parts.append(
            '<div style="margin-bottom:14px;">'
            '<div style="font-size:0.82em; color:#4b5563; font-weight:600; '
            'margin-bottom:4px;">Additional referenced policies</div>'
            f'{_render_policies_table(unmatched)}'
            '</div>'
        )

    return header + "".join(body_parts)


def render_recommended_actions(
    actions: Optional[Sequence[Mapping[str, Any]]],
    policies: Optional[Sequence[Mapping[str, Any]]] = None,
) -> bool:
    """Render the lavender 'Recommended Action' callout. Returns True if rendered.

    When ``policies`` is provided, each recommendation's cited policies (matched
    by URL in its ``citation`` list) are rendered as a compact table below the
    recommendation text, with the ``policy_title`` as a hyperlink to
    ``policy_url`` plus payer, effective date, and key edit_rule_facts.

    When a recommendation's ``description`` packs multiple sub-recommendations
    (``Recommendation 1: ...``, ``IA — ...`` etc.), the description is split
    and each sub-recommendation gets its own filtered policy table so it's
    clear which policies were referenced for which sub-rec.
    """

    if not actions:
        return False

    policy_by_url: Dict[str, Mapping[str, Any]] = {}
    for policy in policies or []:
        if not isinstance(policy, Mapping):
            continue
        url = policy.get("policy_url") or policy.get("external_link")
        if url:
            policy_by_url.setdefault(str(url), policy)

    items: List[str] = []
    for idx, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping):
            continue
        description = action.get("description") or action.get("title") or ""
        if not description:
            continue
        rank = action.get("rank", idx)
        priority_chip = _priority_chip_html(action.get("priority") or "")
        cited = _cited_policies_for_action(action, policy_by_url, policies)
        evidence_parsed = _parse_evidence_entries(action.get("evidence") or [])
        sub_recs = _split_recommendation_description(description)
        has_sub_recs = any(entry.get("label") for entry in sub_recs)

        if has_sub_recs:
            items.append(
                _render_action_with_subrecs(
                    rank, priority_chip, sub_recs, cited, evidence_parsed
                )
            )
        else:
            table = _render_policies_table(cited)
            items.append(
                f'<div style="margin-bottom:14px;">'
                f'<strong>Recommendation {_esc(rank)}:</strong>{priority_chip} '
                f'{_esc(description)}'
                f'{table}'
                f'</div>'
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


def _wrap_drill_path_for_label(
    path: str, max_line_len: int = 55, max_lines: int = 4
) -> str:
    """Wrap a ``→``-delimited drill path for a multi-line plot axis label.

    Splits at ``→`` boundaries so each hop stays readable; falls back to
    truncation with an ellipsis when the path exceeds ``max_lines``. Returns
    a string with ``<br>`` separators — Plotly renders that as line breaks
    inside axis labels and kaleido preserves the same breaks in the PNG.
    """

    parts = [p.strip() for p in (path or "").split("→") if p.strip()]
    if not parts:
        return "—"
    lines: List[str] = []
    current = ""
    for i, part in enumerate(parts):
        candidate = part if i == 0 else f"→ {part}"
        if current and len(current) + 1 + len(candidate) > max_line_len:
            lines.append(current)
            current = candidate
        else:
            current = f"{current} {candidate}".strip() if current else candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        tail_limit = max(1, max_line_len - 1)
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][:tail_limit].rstrip() + "…"]
    return "<br>".join(lines)


def _format_bar_detail_text(delta: float, admits: float, count: int) -> str:
    """Compose the on-bar impact caption: ``+$165K · +12 admits · 3 cells``."""

    parts: List[str] = [f"<b>{_format_signed_money(delta)}</b>"]
    if admits:
        parts.append(f"{_format_signed_int(admits)} admits")
    if count > 1:
        parts.append(f"{count} cells")
    return "  ·  ".join(parts)


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
    bar_texts: List[str] = []
    customdata: List[List[Any]] = []
    label_line_counts: List[int] = []

    for idx, entry in enumerate(ordered, start=1):
        delta = float(entry.get("delta") or 0.0)
        admits = float(entry.get("admissions") or 0.0)
        count = int(entry.get("count") or 1)
        wrapped_path = _wrap_drill_path_for_label(entry["path"])
        # Rank on its own line above the wrapped path so the numbering stays
        # readable when the path itself spans multiple lines.
        label = f"<b>#{idx}</b><br>{wrapped_path}"
        labels.append(label)
        label_line_counts.append(label.count("<br>") + 1)
        deltas.append(delta)
        if delta > 0:
            colors.append("#B5364B")
        elif delta < 0:
            colors.append("#1A8754")
        else:
            colors.append("#94A3B8")
        bar_texts.append(_format_bar_detail_text(delta, admits, count))
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
            text=bar_texts,
            textposition="outside",
            cliponaxis=False,
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
    # Row height scales with the tallest wrapped label so multi-hop paths and
    # single-line labels both breathe. ~18px per line of text plus padding.
    per_row_height = 24 + 18 * max(label_line_counts)
    fig.update_layout(
        height=max(240, 60 + per_row_height * len(ordered)),
        margin={"l": 12, "r": 160, "t": 8, "b": 32},
        bargap=0.35,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        yaxis={
            "autorange": "reversed",
            "title": "",
            "automargin": True,
            "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": "#1A2436"},
        },
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
        uniformtext={"mode": "hide", "minsize": 10},
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
