from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from deep_research_agents.correlation_agent import (
    _extract_token_usage,
    _json_default,
    correlation_llm_tokens_for_step,
    empty_correlation_llm_tokens,
)


RECOMMENDATION_SYSTEM_PROMPT = """
You are a senior healthcare Cost of Care analytics advisor creating executive-ready next-action recommendations.

Your job is to convert drill-down output and interaction-matrix output into business decision-tree recommendations.

Use ONLY the facts provided in the input JSON. Do not invent causes, policies, owners, contracts, providers, or operational details. Use cautious language: "warrants review", "validate whether", "points to", "compare against", "may indicate". Do not say "caused by", "proves", or "will fix".

Generate recommendations using this structure:
1. Rank: numeric ordering of the recommended action.
2. Priority: HIGH, MEDIUM, or LOW.
3. Category: Policy, Operational, Clinical, or Financial.
4. Description: a short leadership-friendly recommendation.
5. Evidence: detailed strings describing the exact drill-down, interaction, metric, and offset facts supporting the recommendation.
6. Story Alignment: strings formatted as `Why: ...`, `research_consideration: ...`, and `cost_of_care_suggestion: ...`.
7. Peer Benchmarking: strings comparing top_segments against bottom_segments when comparator segments exist.
8. Citation: always an empty list.

Decision-tree mapping rules:

A. Provider / Network / Contract
Use when provider, hospital system, facility, network status, reimbursement, paid ratio, allowed/admit, or paid/admit signals are material.
Research Considerations:
- Analyst to determine Provider drivers.
- Analyst to review Contract Provisions.
- Analyst to review Provider distribution.
- Analyst to review Contract Relativity.
- Analyst to review In-Network vs Out-of-Network status.
- Analyst to review provider unit cost outliers.
- Analyst to review Reimbursement Method.
Cost of Care Suggestions:
- Contract remediation opportunity.
- Contract compliance concern.
- Alternative management options.
- Establish site-of-care opportunity.
- Confirm claims are processing per contractual obligation.

B. Member / Geography / Product Mix
Use when geography, product, benefit plan, member concentration, or market divergence is material.
Research Considerations:
- Analyst to determine Member drivers.
- Analyst to review geolocation of member concerns.
- Analyst to review segment, sub-segment, or benefit plan driving variance.
- Analyst to review whole health drivers.
- Analyst to review clinical engagement.
Cost of Care Suggestions:
- Market-specific intervention.
- Benefit design or product mix review.
- Social programming opportunity.
- Improve clinical engagement opportunity.
- Assess care navigation opportunity.

C. Diagnosis / Clinical Mix
Use when DRG, diagnosis, HCC, clinical category, severity, or clinical offsets are material.
Research Considerations:
- Analyst to determine Diagnosis drivers.
- Analyst to review admitting diagnosis.
- Analyst to review discharge diagnosis.
- Analyst to review distribution of cases / DRG mix.
- Analyst to review severity of cases / DRG mix.
- Analyst to review Maternal Child Health KPIs when OB-related.
Cost of Care Suggestions:
- Improve clinical KPIs.
- Assess root cause of admission increases.
- Review UM ROI model for PAV opportunities.
- Program Integrity opportunity.
- Avoid broad clinical action if offsets show a narrow issue.

D. Service / Site of Care / Length of Stay
Use when facility type, acute vs sub-acute, site of care, length of stay, or duplicate services are material.
Research Considerations:
- Analyst to review level of care, acute vs sub-acute.
- Analyst to review what else is being billed during stay.
- Analyst to review length of stay by diagnosis.
- Analyst to review admission source or discharge location.
Cost of Care Suggestions:
- Establish site-of-care opportunity.
- Review initial review KPIs.
- Review concurrent review KPIs.
- Improve discharge planning.
- Develop recurring recovery or edit process for inappropriate billing.

E. PAV / Prior Auth / Auth-to-Claim
Use when prior authorization, PA required vs not required, authorization overrides, denials, authorized level, or auth-to-claim mismatch is material.
Research Considerations:
- Analyst to review variance driven by services requiring prior auth vs no prior auth.
- Analyst to review lack of information denial rate.
- Analyst to review nurse approval / denial / referral rates.
- Analyst to review MD approval / denial rates.
- Analyst to review Auth to Claim match.
- Analyst to review Auth Overrides.
- Analyst to review Provider exclusions.
- Analyst to review Authorized level of care against actual.
Cost of Care Suggestions:
- Confirm prior auth was obtained when required.
- Claims processing opportunity.
- Contract remediation opportunity.
- Confirm level authorized was level paid.
- Improve nurse referral rate to MDs.
- Audit decision tree where approvals increased or denials decreased.

F. Claim Frequency / Processing / Payment Integrity
Use when row counts, interim/final/first claims, billed charges, auto/manual adjudication, outlier status, or high-cost claims are material.
Research Considerations:
- Analyst to review claim frequency/status.
- Analyst to determine volume of Interim Claims.
- Analyst to determine volume of Final Claims.
- Analyst to determine volume of First Claims.
- Analyst to review billed charges.
- Analyst to review outlier status.
- Analyst to review auto adjudication vs manually processed rate.
- Analyst to review impact of high-cost claims.
Cost of Care Suggestions:
- Claims processing opportunity.
- Develop recurring recovery process.
- Confirm final claim adjudicated per contractual obligation.
- Ensure manual adjudication is not overwriting key edits.
- Ensure itemized bill review.
- Program Integrity opportunity.

Signal interpretation rules:
- If paid increased while admissions/claims are flat or nearly flat, prioritize unit cost, reimbursement, contract, paid ratio, allowed/admit, or provider economics.
- If paid increased and admissions/claims increased materially while paid/admit is flat, prioritize utilization, admission growth, diagnosis, site-of-care, or market review.
- If paid, admissions, and paid/admit all increased, recommend a combined volume + intensity review.
- If baseline is very small and comparison jumps sharply, include data validation or mapping validation before operational action.
- If no-prior-auth segments increase while auth-coded segments decline, recommend prior-auth/PAV and auth-to-claim research.
- If a facility type dominates the increase, recommend site-of-care and level-of-care review.
- If geography has positive hotspots and negative comparators, recommend market-specific review and use lower-growth geographies as comparators.
- If provider/hospital positive and negative segments exist, recommend comparing high-growth providers against lower-growth comparable providers.
- If clinical positive cells have related negative/offset cells, avoid broad clinical conclusions and recommend targeted clinical review.
- If drill-down and interaction matrix agree, confidence should increase. If they conflict, recommend reconciliation before action.
- Do not recommend generic monitoring.

Return strict JSON only. Maximum 5 recommendations. Prioritize recommendations that are specific, quantified, and actionable for executive leadership.
""".strip()

RECOMMENDATION_USER_PROMPT_TEMPLATE = """
Create executive-ready Cost of Care recommendations from the JSON inputs below.

Inputs:
1. executive_summary_json: contains root trend, drill_path, positive and negative drill-down segments, explainer metrics, interaction matrix, and prior recommendations.
2. interaction_summary_json: contains the matrix narrative when available and may be empty or disabled.
3. interaction_recommendations_json: contains prior vanilla recommendations, which should be improved, not copied.

Instructions:
- Treat executive_summary_json as the authoritative structured input for recommendation generation.
- If interaction_summary_json is empty or disabled, rely on drill_path, explainer_metrics, and interaction_matrix from executive_summary_json.
- Combine drill-down and interaction-matrix evidence.
- Use both positive contributors and negative/offset contributors when they change the interpretation.
- First infer the Research Consideration using the business decision-tree mapping.
- Then map it to a Cost of Care Suggestion.
- Include a concise Why with quantified facts.
- Prefer recommendations that distinguish unit cost vs utilization vs authorization/PAV vs product/geography mix vs clinical mix vs payment integrity.
- Do not simply pick the top matrix cells.
- Do not overstate causality.
- Do not include unsupported recommendations.

Output strict JSON in this schema:

{
  "recommended_action": [
    {
      "rank": 1,
      "priority": "HIGH | MEDIUM | LOW",
      "category": "Policy | Operational | Clinical | Financial",
      "description": "Short leadership-friendly recommendation.",
      "evidence": [
        "Detailed supporting evidence strings built from drill-down, interaction, metric, and offset facts."
      ],
      "story_alignment": [
        "Why: Quantified explanation using drill-down and/or interaction-matrix facts.",
        "research_consideration: Business decision-tree research consideration.",
        "cost_of_care_suggestion: Mapped cost-of-care suggestion."
      ],
      "peer_benchmarking": [
        "Comparator statements built from top_segments versus bottom_segments."
      ],
      "citation": []
    }
  ],
  "summary": {
    "overall_pattern": "Concentrated | Broad-based | Mixed | Needs validation",
    "primary_next_action": "Single most important next action for leadership.",
    "do_not_overgeneralize": "What leadership should avoid concluding too broadly."
  }
}

JSON inputs:
executive_summary_json:
{{EXECUTIVE_SUMMARY_JSON}}

interaction_summary_json:
{{INTERACTION_SUMMARY_JSON}}

interaction_recommendations_json:
{{INTERACTION_RECOMMENDATIONS_JSON}}
""".strip()

CLAIM_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "claim_count",
    "claims_count",
    "total_claims",
)
ADMISSION_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "total_admissions",
    "admission_count",
    "admit_count",
    "admissions",
)
PAID_PER_ADMIT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "avg_paid_per_admit",
    "paid_per_admit",
    "paid_per_admission",
)
ALLOWED_PER_ADMIT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "avg_allowed_per_admit",
    "allowed_per_admit",
    "allowed_per_admission",
)
PAID_RATIO_METRIC_CANDIDATES: Tuple[str, ...] = (
    "paid_ratio",
    "pay_ratio",
)

CATEGORY_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "provider_network_contract",
        "keywords": {
            "rendering_provider_name",
            "billing_provider_name",
            "rendering_hospital_system",
            "hospital_system",
            "provider",
            "facility_name",
            "network_status",
            "reimbursement_method",
            "paid_ratio",
            "avg_allowed_per_admit",
            "avg_paid_per_admit",
        },
        "research_considerations": [
            "Analyst to determine Provider drivers.",
            "Analyst to review Contract Provisions.",
            "Analyst to review Provider distribution.",
            "Analyst to review Contract Relativity.",
            "Analyst to review In-Network vs Out-of-Network status.",
            "Analyst to review provider unit cost outliers.",
            "Analyst to review Reimbursement Method.",
        ],
        "suggestions": [
            "Contract remediation opportunity.",
            "Contract compliance concern.",
            "Alternative management options.",
            "Establish site-of-care opportunity.",
            "Confirm claims are processing per contractual obligation.",
        ],
        "review_area": "Network",
    },
    {
        "name": "member_geography_product",
        "keywords": {
            "state_name",
            "service_area_state",
            "product_description",
            "benefit_plan",
            "segment",
            "sub_segment",
            "lob_code",
            "mbu_cls_short_description",
            "market",
            "member",
            "geography",
        },
        "research_considerations": [
            "Analyst to determine Member drivers.",
            "Analyst to review geolocation of member concerns.",
            "Analyst to review segment, sub-segment, or benefit plan driving variance.",
            "Analyst to review whole health drivers.",
            "Analyst to review clinical engagement.",
        ],
        "suggestions": [
            "Market-specific intervention.",
            "Benefit design or product mix review.",
            "Social programming opportunity.",
            "Improve clinical engagement opportunity.",
            "Assess care navigation opportunity.",
        ],
        "review_area": "Mixed",
    },
    {
        "name": "diagnosis_clinical_mix",
        "keywords": {
            "drg_name",
            "primary_diagnosis_name",
            "diagnosis",
            "hcc_medium",
            "severity",
            "clinical_category",
            "ob",
            "maternal",
        },
        "research_considerations": [
            "Analyst to determine Diagnosis drivers.",
            "Analyst to review admitting diagnosis.",
            "Analyst to review discharge diagnosis.",
            "Analyst to review distribution of cases / DRG mix.",
            "Analyst to review severity of cases / DRG mix.",
            "Analyst to review Maternal Child Health KPIs when OB-related.",
        ],
        "suggestions": [
            "Improve clinical KPIs.",
            "Assess root cause of admission increases.",
            "Review UM ROI model for PAV opportunities.",
            "Program Integrity opportunity.",
            "Avoid broad clinical action if offsets show a narrow issue.",
        ],
        "review_area": "Clinical",
    },
    {
        "name": "service_site_of_care",
        "keywords": {
            "facility_type",
            "site_of_care",
            "length_of_stay",
            "los",
            "admission_source",
            "discharge_location",
            "acute",
            "sub_acute",
        },
        "research_considerations": [
            "Analyst to review level of care, acute vs sub-acute.",
            "Analyst to review what else is being billed during stay.",
            "Analyst to review length of stay by diagnosis.",
            "Analyst to review admission source or discharge location.",
        ],
        "suggestions": [
            "Establish site-of-care opportunity.",
            "Review initial review KPIs.",
            "Review concurrent review KPIs.",
            "Improve discharge planning.",
            "Develop recurring recovery or edit process for inappropriate billing.",
        ],
        "review_area": "Ops",
    },
    {
        "name": "pav_prior_auth",
        "keywords": {
            "pa_required_code",
            "prior_auth",
            "prior_authorization",
            "auth_to_claim",
            "auth_override",
            "authorized_level",
            "denial",
            "authorization",
        },
        "research_considerations": [
            "Analyst to review variance driven by services requiring prior auth vs no prior auth.",
            "Analyst to review lack of information denial rate.",
            "Analyst to review nurse approval / denial / referral rates.",
            "Analyst to review MD approval / denial rates.",
            "Analyst to review Auth to Claim match.",
            "Analyst to review Auth Overrides.",
            "Analyst to review Provider exclusions.",
            "Analyst to review Authorized level of care against actual.",
        ],
        "suggestions": [
            "Confirm prior auth was obtained when required.",
            "Claims processing opportunity.",
            "Contract remediation opportunity.",
            "Confirm level authorized was level paid.",
            "Improve nurse referral rate to MDs.",
            "Audit decision tree where approvals increased or denials decreased.",
        ],
        "review_area": "Ops",
    },
    {
        "name": "claim_frequency_processing",
        "keywords": {
            "claim_frequency",
            "claim_status",
            "interim",
            "final",
            "first",
            "billed_charges",
            "auto_adjudication",
            "manual_adjudication",
            "outlier",
            "high_cost",
            "raw_row_count",
            "claim_count",
        },
        "research_considerations": [
            "Analyst to review claim frequency/status.",
            "Analyst to determine volume of Interim Claims.",
            "Analyst to determine volume of Final Claims.",
            "Analyst to determine volume of First Claims.",
            "Analyst to review billed charges.",
            "Analyst to review outlier status.",
            "Analyst to review auto adjudication vs manually processed rate.",
            "Analyst to review impact of high-cost claims.",
        ],
        "suggestions": [
            "Claims processing opportunity.",
            "Develop recurring recovery process.",
            "Confirm final claim adjudicated per contractual obligation.",
            "Ensure manual adjudication is not overwriting key edits.",
            "Ensure itemized bill review.",
            "Program Integrity opportunity.",
        ],
        "review_area": "Ops",
    },
)


class RecommendationEvidenceSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: List[str] = Field(default_factory=list)


class RecommendedActionItemSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int = 1
    priority: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    category: Literal["Policy", "Operational", "Clinical", "Financial"] = "Operational"
    description: str = ""
    evidence: List[str] = Field(default_factory=list)
    story_alignment: List[str] = Field(default_factory=list)
    peer_benchmarking: List[str] = Field(default_factory=list)
    citation: List[str] = Field(default_factory=list)


class RecommendationSummarySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    overall_pattern: str = "Needs validation"
    primary_next_action: str = ""
    do_not_overgeneralize: str = ""


class CorrelationRecommendationsSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recommended_action: List[RecommendedActionItemSchema] = Field(default_factory=list)
    summary: RecommendationSummarySchema = Field(default_factory=RecommendationSummarySchema)
    source: str = "empty"

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        payload = _safe_dict(value)
        raw_items = payload.get("recommended_action")
        if not raw_items:
            raw_items = payload.get("items") or payload.get("recommendations")
        payload["recommended_action"] = [
            _normalize_recommended_action_item(item.model_dump(mode='json') if hasattr(item, "model_dump") else item, index)
            for index, item in enumerate(_safe_list(raw_items), start=1)
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
        return payload


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slugify(text: str) -> str:
    normalized = []
    previous_underscore = False
    for char in str(text or "").strip().lower():
        if char.isalnum():
            normalized.append(char)
            previous_underscore = False
        else:
            if not previous_underscore:
                normalized.append("_")
                previous_underscore = True
    return "".join(normalized).strip("_")


def _compact_currency(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _compact_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}K"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.0f}%"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _dimension_value_pairs(nodes: Sequence[Mapping[str, Any]], limit: int = 6) -> List[str]:
    pairs: List[str] = []
    for node in nodes:
        dimension = str(node.get("dimension") or "").strip()
        for segment in _safe_list(node.get("top_segments")):
            segment_dict = _safe_dict(segment)
            raw_value = str(segment_dict.get("value") or "").strip()
            if dimension and raw_value:
                pairs.append(f"{dimension}={raw_value}")
                if len(pairs) >= limit:
                    return pairs
    return pairs


def _offset_dimension_pairs(nodes: Sequence[Mapping[str, Any]], limit: int = 4) -> List[str]:
    pairs: List[str] = []
    for node in nodes:
        dimension = str(node.get("dimension") or "").strip()
        for segment in _safe_list(node.get("bottom_segments")):
            segment_dict = _safe_dict(segment)
            raw_value = str(segment_dict.get("value") or "").strip()
            if dimension and raw_value:
                pairs.append(f"{dimension}={raw_value}")
                if len(pairs) >= limit:
                    return pairs
    return pairs


def _extract_explainer_root_records(explainer_metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [record for record in _safe_list(_safe_dict(explainer_metrics).get("root")) if isinstance(record, dict)]


def _select_metric_key(records: Sequence[Mapping[str, Any]], candidate_metrics: Sequence[str]) -> Optional[str]:
    if not records:
        return None
    available_keys = {str(key) for record in records for key in record.keys()}
    normalized_candidates = [_slugify(candidate) for candidate in candidate_metrics]
    for candidate in normalized_candidates:
        if candidate in available_keys:
            return candidate
    for candidate in normalized_candidates:
        for key in available_keys:
            if key.endswith(candidate) or candidate in key:
                return key
    return None


def _find_period_metric_value(records: Sequence[Mapping[str, Any]], period_bucket: str, metric_key: str) -> Optional[Any]:
    for record in records:
        bucket = str(record.get("period_bucket") or "").strip().lower()
        if bucket == period_bucket:
            return record.get(metric_key)
    return None


def _extract_explainer_metric_change(explainer_metrics: Mapping[str, Any], candidate_metrics: Sequence[str]) -> Optional[Dict[str, Any]]:
    records = _extract_explainer_root_records(explainer_metrics)
    metric_key = _select_metric_key(records, candidate_metrics)
    if not metric_key:
        return None
    baseline_raw = _find_period_metric_value(records, "baseline", metric_key)
    comparison_raw = _find_period_metric_value(records, "comparison", metric_key)
    if baseline_raw is None and comparison_raw is None:
        return None
    baseline_value = _to_float(baseline_raw)
    comparison_value = _to_float(comparison_raw)
    delta_value = comparison_value - baseline_value
    delta_pct = (delta_value / baseline_value) if baseline_value else None
    return {
        "metric_key": metric_key,
        "baseline": baseline_value,
        "comparison": comparison_value,
        "delta": delta_value,
        "delta_pct": delta_pct,
    }


def _metric_summary_lines(explainer_metrics: Mapping[str, Any]) -> List[str]:
    summaries: List[str] = []
    metric_map = (
        ("claims", CLAIM_COUNT_METRIC_CANDIDATES),
        ("admissions", ADMISSION_COUNT_METRIC_CANDIDATES),
        ("paid/admit", PAID_PER_ADMIT_METRIC_CANDIDATES),
        ("allowed/admit", ALLOWED_PER_ADMIT_METRIC_CANDIDATES),
        ("paid ratio", PAID_RATIO_METRIC_CANDIDATES),
    )
    for label, candidates in metric_map:
        change = _extract_explainer_metric_change(explainer_metrics, candidates)
        if not change:
            continue
        delta_value = _to_float(change.get("delta"))
        if label == "paid ratio":
            summaries.append(f"{label} {_format_pct(change.get('baseline'))} to {_format_pct(change.get('comparison'))}")
        elif label in {"claims", "admissions"}:
            summaries.append(
                f"{label} {_compact_number(_to_float(change.get('baseline')))} to {_compact_number(_to_float(change.get('comparison')))}"
            )
        else:
            summaries.append(
                f"{label} {_compact_currency(_to_float(change.get('baseline')))} to {_compact_currency(_to_float(change.get('comparison')))}"
            )
        if delta_value == 0:
            summaries[-1] = f"{summaries[-1]} (flat)"
    return summaries


def _normalize_priority(value: Any) -> str:
    cleaned = str(value or "").strip().upper()
    if cleaned in {"HIGH", "MEDIUM", "LOW"}:
        return cleaned
    return "LOW"


def _normalize_rank(rank_value: Any, priority_value: Any, default: int) -> int:
    for candidate in (rank_value, priority_value):
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            return max(int(candidate), 1)
        if isinstance(candidate, str) and candidate.strip().isdigit():
            return max(int(candidate.strip()), 1)
    return max(default, 1)


def _normalize_category(value: Any, fallback: str = "Operational") -> str:
    cleaned = str(value or "").strip()
    category_map = {
        "Policy": "Policy",
        "Operational": "Operational",
        "Clinical": "Clinical",
        "Financial": "Financial",
        "Ops": "Operational",
        "Network": "Financial",
        "Mixed": fallback,
    }
    return category_map.get(cleaned, fallback)


def _story_alignment_lines(why: str, research_consideration: str, cost_of_care_suggestion: str) -> List[str]:
    lines: List[str] = []
    if why:
        lines.append(f"Why: {why}")
    if research_consideration:
        lines.append(f"research_consideration: {research_consideration}")
    if cost_of_care_suggestion:
        lines.append(f"cost_of_care_suggestion: {cost_of_care_suggestion}")
    return lines


def _legacy_evidence_to_list(evidence_payload: Any, cell_ids: Sequence[str], caveat: str) -> List[str]:
    evidence_dict = _safe_dict(evidence_payload)
    evidence_items = [str(value).strip() for value in _safe_list(evidence_payload) if str(value).strip()]
    if evidence_items:
        if caveat:
            evidence_items.append(f"Caveat: {caveat}")
        return evidence_items
    for drill_dimension in _safe_list(evidence_dict.get("drill_dimensions")):
        text = str(drill_dimension).strip()
        if text:
            evidence_items.append(f"Drill segment: {text}")
    for cell_id in [str(value).strip() for value in _safe_list(evidence_dict.get("interaction_cell_ids")) if str(value).strip()] or list(cell_ids):
        evidence_items.append(f"Interaction cell reference: {cell_id}")
    for cell_id in [str(value).strip() for value in _safe_list(evidence_dict.get("offset_cell_ids")) if str(value).strip()]:
        evidence_items.append(f"Offset cell reference: {cell_id}")
    for metric in _safe_list(evidence_dict.get("key_metrics")):
        text = str(metric).strip()
        if text:
            evidence_items.append(f"Metric movement: {text}")
    if caveat:
        evidence_items.append(f"Caveat: {caveat}")
    return evidence_items


def _normalize_recommended_action_item(item: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item_dict = _safe_dict(item)
    text = str(item_dict.get("text") or "").strip()
    parsed: Dict[str, Any] = {}
    if text.startswith("{") and text.endswith("}"):
        try:
            maybe_payload = json.loads(text)
            if isinstance(maybe_payload, dict):
                parsed = maybe_payload
        except Exception:
            parsed = {}
    combined = dict(parsed)
    combined.update(item_dict)
    description = _normalize_text(
        combined.get("description")
        or combined.get("executive_recommendation")
        or combined.get("action")
        or text
    )
    why = _normalize_text(combined.get("why") or combined.get("rationale"))
    research_consideration = _normalize_text(combined.get("research_consideration"))
    cost_of_care_suggestion = _normalize_text(combined.get("cost_of_care_suggestion"))
    caveat = _normalize_text(combined.get("caveat"))
    cell_ids = [str(cell_id).strip() for cell_id in _safe_list(combined.get("cell_ids")) if str(cell_id).strip()]
    story_alignment = [str(value).strip() for value in _safe_list(combined.get("story_alignment")) if str(value).strip()]
    if not story_alignment:
        story_alignment = _story_alignment_lines(why, research_consideration, cost_of_care_suggestion)
    evidence = _legacy_evidence_to_list(combined.get("evidence"), cell_ids, caveat)
    peer_benchmarking = [str(value).strip() for value in _safe_list(combined.get("peer_benchmarking")) if str(value).strip()]
    citation = [str(value).strip() for value in _safe_list(combined.get("citation")) if str(value).strip()]
    priority_source = combined.get("priority") if isinstance(combined.get("priority"), str) else combined.get("confidence")
    return {
        "rank": _normalize_rank(combined.get("rank"), combined.get("priority"), index),
        "priority": _normalize_priority(priority_source or combined.get("confidence")),
        "category": _normalize_category(combined.get("category"), fallback=_normalize_category(combined.get("review_area"))),
        "description": description,
        "evidence": evidence,
        "story_alignment": story_alignment,
        "peer_benchmarking": peer_benchmarking,
        "citation": citation,
    }


def normalize_legacy_recommendations(payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload_dict = _safe_dict(payload)
    normalized = CorrelationRecommendationsSchema(**payload_dict).model_dump(mode='json')
    normalized["source"] = str(payload_dict.get("source") or normalized.get("source") or "empty")
    return normalized


def _collect_dimension_keys(drill_path: Sequence[Mapping[str, Any]], interaction_matrix: Mapping[str, Any]) -> List[str]:
    keys: List[str] = []
    for node in drill_path:
        dimension = str(node.get("dimension") or "").strip()
        if dimension:
            keys.append(dimension)
    for stage_name in ("operational", "clinical"):
        stage_payload = _safe_dict(interaction_matrix.get(stage_name))
        for cell in _safe_list(stage_payload.get("selected_cells")) + _safe_list(stage_payload.get("offset_cells_preview")):
            for key in _safe_dict(_safe_dict(cell).get("dimension_values")).keys():
                keys.append(str(key).strip())
    return [key for key in keys if key]


def _choose_category(drill_path: Sequence[Mapping[str, Any]], interaction_matrix: Mapping[str, Any], metric_lines: Sequence[str]) -> Dict[str, Any]:
    dimension_keys = {_slugify(key) for key in _collect_dimension_keys(drill_path, interaction_matrix)}
    metric_keys = {_slugify(line.split()[0]) for line in metric_lines}
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for rule in CATEGORY_RULES:
        score = sum(1 for key in dimension_keys | metric_keys if key in rule["keywords"])
        if score:
            candidates.append((score, rule))
    if not candidates:
        return dict(CATEGORY_RULES[5])
    candidates.sort(key=lambda item: item[0], reverse=True)
    return dict(candidates[0][1])


def _category_to_action_category(category_name: str) -> str:
    category_map = {
        "provider_network_contract": "Financial",
        "diagnosis_clinical_mix": "Clinical",
        "service_site_of_care": "Operational",
        "pav_prior_auth": "Operational",
        "claim_frequency_processing": "Operational",
        "member_geography_product": "Operational",
    }
    return category_map.get(category_name, "Operational")


def _format_delta_pct(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return ""
    return f"{delta_pct * 100:.0f}%"


def _segment_evidence_lines(drill_path: Sequence[Mapping[str, Any]], limit: int = 4) -> List[str]:
    evidence: List[str] = []
    for node in drill_path:
        dimension = str(node.get("dimension_label") or node.get("dimension") or "").strip()
        for segment in _safe_list(node.get("top_segments"))[:2]:
            segment_dict = _safe_dict(segment)
            value = str(segment_dict.get("value") or "").strip()
            delta_value = _to_float(segment_dict.get("delta_value"))
            share = _to_float(
                segment_dict.get("aligned_contribution_pct_of_aligned_delta")
                or segment_dict.get("contribution_pct_total")
            )
            if dimension and value:
                detail = f"Drill path: {dimension}={value} contributed {_compact_currency(delta_value)}"
                if share:
                    detail = f"{detail} ({share * 100:.0f}% of aligned/total delta)"
                evidence.append(detail + ".")
                if len(evidence) >= limit:
                    return evidence
        for segment in _safe_list(node.get("bottom_segments"))[:1]:
            segment_dict = _safe_dict(segment)
            value = str(segment_dict.get("value") or "").strip()
            delta_value = _to_float(segment_dict.get("delta_value"))
            if dimension and value:
                evidence.append(
                    f"Offset comparator: {dimension}={value} moved {_compact_currency(delta_value)} and offsets the positive pocket."
                )
                if len(evidence) >= limit:
                    return evidence
    return evidence


def _cell_detail_text(stage_label: str, cell: Mapping[str, Any]) -> str:
    cell_dict = _safe_dict(cell)
    cell_id = str(cell_dict.get("cell_id") or "unknown_cell").strip()
    delta_value = _compact_currency(_to_float(cell_dict.get("delta_value")))
    share_value = _to_float(cell_dict.get("share_of_positive_delta"))
    dimensions = ", ".join(
        f"{key}={value}"
        for key, value in list(_safe_dict(cell_dict.get("dimension_values")).items())[:4]
        if str(value).strip()
    )
    detail = f"{stage_label} interaction cell {cell_id} changed {delta_value}"
    if share_value:
        detail = f"{detail} ({share_value * 100:.0f}% of positive delta)"
    if dimensions:
        detail = f"{detail}; dimensions: {dimensions}"
    return detail + "."


def _build_evidence_lines(
    *,
    metric_name: str,
    root_trend: Mapping[str, Any],
    drill_path: Sequence[Mapping[str, Any]],
    operational_cells: Sequence[Mapping[str, Any]],
    clinical_cells: Sequence[Mapping[str, Any]],
    offset_cells: Sequence[Mapping[str, Any]],
    metric_lines: Sequence[str],
    caveat: str,
) -> List[str]:
    evidence: List[str] = []
    baseline_value = _to_float(root_trend.get("baseline_value"))
    comparison_value = _to_float(root_trend.get("comparison_value"))
    delta_value = _to_float(root_trend.get("delta_value"))
    delta_pct = root_trend.get("delta_pct")
    root_line = (
        f"Root trend for {metric_name}: {_compact_currency(baseline_value)} to {_compact_currency(comparison_value)} "
        f"(delta {_compact_currency(delta_value)}"
    )
    pct_text = _format_delta_pct(delta_pct if isinstance(delta_pct, (int, float)) else None)
    if pct_text:
        root_line = f"{root_line}, {pct_text}"
    evidence.append(root_line + ").")
    evidence.extend(_segment_evidence_lines(drill_path))
    for cell in operational_cells[:2]:
        evidence.append(_cell_detail_text("Operational", cell))
    for cell in clinical_cells[:1]:
        evidence.append(_cell_detail_text("Clinical", cell))
    for cell in offset_cells[:1]:
        evidence.append(_cell_detail_text("Clinical offset", cell))
    for metric_line in metric_lines[:4]:
        evidence.append(f"Metric signal: {metric_line}.")
    if caveat:
        evidence.append(f"Caveat: {caveat}")
    return evidence


def _build_peer_benchmarking(drill_path: Sequence[Mapping[str, Any]]) -> List[str]:
    comparisons: List[str] = []
    for node in drill_path:
        dimension = str(node.get("dimension_label") or node.get("dimension") or "").strip()
        top_segments = [segment for segment in _safe_list(node.get("top_segments")) if isinstance(segment, dict)]
        bottom_segments = [segment for segment in _safe_list(node.get("bottom_segments")) if isinstance(segment, dict)]
        if not dimension or not top_segments or not bottom_segments:
            continue
        lead_top = _safe_dict(top_segments[0])
        lead_bottom = _safe_dict(bottom_segments[0])
        top_value = str(lead_top.get("value") or "").strip()
        bottom_value = str(lead_bottom.get("value") or "").strip()
        if top_value and bottom_value:
            comparisons.append(
                f"Peer benchmark on {dimension}: top segment {top_value} increased {_compact_currency(_to_float(lead_top.get('delta_value')))} while comparator {bottom_value} moved {_compact_currency(_to_float(lead_bottom.get('delta_value')))}."
            )
    return comparisons


def _root_caveat(root_trend: Mapping[str, Any], drill_path: Sequence[Mapping[str, Any]], offset_cells: Sequence[Mapping[str, Any]]) -> str:
    baseline_value = _to_float(root_trend.get("baseline_value"))
    comparison_value = _to_float(root_trend.get("comparison_value"))
    delta_value = _to_float(root_trend.get("delta_value"))
    if baseline_value and baseline_value < 1000 and comparison_value > baseline_value * 2:
        return "Baseline volume appears small relative to the jump, so validate mapping and period alignment before operational action."
    if offset_cells:
        return "Negative offset segments are present, so avoid broad conclusions until positive and negative pockets are reconciled."
    if not drill_path:
        return "Drill-path concentration is limited, so validate whether the pattern is broad-based before acting."
    if delta_value == 0:
        return "Net movement is limited, so confirm whether the observed pockets are material enough for action."
    return "Use the interaction cells and drill path together rather than treating any single segment as conclusive."


def _build_why(
    *,
    metric_name: str,
    root_trend: Mapping[str, Any],
    drill_path: Sequence[Mapping[str, Any]],
    operational_cells: Sequence[Mapping[str, Any]],
    clinical_cells: Sequence[Mapping[str, Any]],
    offset_cells: Sequence[Mapping[str, Any]],
    metric_lines: Sequence[str],
) -> str:
    clauses: List[str] = []
    delta_value = _to_float(root_trend.get("delta_value"))
    if delta_value:
        clauses.append(f"root paid increased by {_compact_currency(delta_value)}")
    if drill_path:
        lead_node = _safe_dict(drill_path[0])
        lead_segment = _safe_dict(_safe_list(lead_node.get("top_segments"))[0] if _safe_list(lead_node.get("top_segments")) else {})
        lead_dimension = str(lead_node.get("dimension") or "").strip()
        lead_value = str(lead_segment.get("value") or "").strip()
        lead_delta = _to_float(lead_segment.get("delta_value"))
        if lead_dimension and lead_value and lead_delta:
            clauses.append(f"drill-down concentrated in {lead_dimension}={lead_value} ({_compact_currency(lead_delta)})")
    if operational_cells:
        top_cell = _safe_dict(operational_cells[0])
        clauses.append(
            f"interaction cell {top_cell.get('cell_id')} contributed {_compact_currency(_to_float(top_cell.get('delta_value')))}"
        )
    if clinical_cells:
        top_clinical = _safe_dict(clinical_cells[0])
        clauses.append(
            f"clinical cell {top_clinical.get('cell_id')} added {_compact_currency(_to_float(top_clinical.get('delta_value')))}"
        )
    if metric_lines:
        clauses.append("; ".join(metric_lines[:2]))
    if offset_cells:
        top_offset = _safe_dict(offset_cells[0])
        clauses.append(
            f"offset cell {top_offset.get('cell_id')} declined {_compact_currency(abs(_to_float(top_offset.get('delta_value'))))}"
        )
    return "; ".join([clause for clause in clauses if clause]).strip().rstrip(";") + "."


def _infer_signal(metric_lines: Sequence[str], explainer_metrics: Mapping[str, Any]) -> str:
    claim_change = _extract_explainer_metric_change(explainer_metrics, CLAIM_COUNT_METRIC_CANDIDATES)
    admission_change = _extract_explainer_metric_change(explainer_metrics, ADMISSION_COUNT_METRIC_CANDIDATES)
    paid_per_admit_change = _extract_explainer_metric_change(explainer_metrics, PAID_PER_ADMIT_METRIC_CANDIDATES)
    claims_delta_pct = claim_change.get("delta_pct") if claim_change else None
    admits_delta_pct = admission_change.get("delta_pct") if admission_change else None
    ppa_delta_pct = paid_per_admit_change.get("delta_pct") if paid_per_admit_change else None
    volume_flat = any(value is not None and abs(value) <= 0.03 for value in (claims_delta_pct, admits_delta_pct))
    volume_up = any(value is not None and value >= 0.05 for value in (claims_delta_pct, admits_delta_pct))
    intensity_up = ppa_delta_pct is not None and ppa_delta_pct >= 0.05
    if volume_flat and intensity_up:
        return "unit_cost"
    if volume_up and not intensity_up:
        return "utilization"
    if volume_up and intensity_up:
        return "combined_volume_intensity"
    if metric_lines:
        return "mixed"
    return "validation"


def _recommendation_text(category_name: str, signal_name: str) -> str:
    if category_name == "pav_prior_auth":
        return "Validate prior-auth and auth-to-claim patterns before broad operational changes."
    if category_name == "service_site_of_care":
        return "Review acute site-of-care concentration and related utilization controls."
    if category_name == "diagnosis_clinical_mix":
        return "Focus clinical review on the narrowed diagnosis pocket rather than the full category."
    if category_name == "provider_network_contract" and signal_name == "unit_cost":
        return "Compare provider economics and reimbursement terms in the concentrated growth pockets."
    if category_name == "member_geography_product":
        return "Prioritize market and product review in the concentrated growth segments."
    if signal_name == "validation":
        return "Validate the concentrated pattern before assigning operational ownership."
    if signal_name == "utilization":
        return "Review utilization growth before assuming a unit-cost-only issue."
    if signal_name == "combined_volume_intensity":
        return "Run a combined volume and intensity review in the concentrated segments."
    return "Review the concentrated cost-of-care pattern using drill-down and interaction evidence together."


def _confidence(drill_path: Sequence[Mapping[str, Any]], operational_cells: Sequence[Mapping[str, Any]], clinical_cells: Sequence[Mapping[str, Any]]) -> str:
    if drill_path and operational_cells and clinical_cells:
        return "High"
    if drill_path and operational_cells:
        return "Medium"
    if operational_cells or drill_path:
        return "Low"
    return "Low"


def _overall_pattern(drill_path: Sequence[Mapping[str, Any]], operational_cells: Sequence[Mapping[str, Any]], offset_cells: Sequence[Mapping[str, Any]], root_trend: Mapping[str, Any]) -> str:
    if _to_float(root_trend.get("baseline_value")) and _to_float(root_trend.get("baseline_value")) < 1000:
        return "Needs validation"
    top_share = 0.0
    if operational_cells:
        top_share = max(top_share, _to_float(_safe_dict(operational_cells[0]).get("share_of_positive_delta")))
    if drill_path:
        lead_segment = _safe_dict(_safe_list(_safe_dict(drill_path[0]).get("top_segments"))[0] if _safe_list(_safe_dict(drill_path[0]).get("top_segments")) else {})
        top_share = max(
            top_share,
            abs(_to_float(lead_segment.get("aligned_contribution_pct_of_aligned_delta")) or _to_float(lead_segment.get("contribution_pct_total"))),
        )
    if top_share >= 0.6 and not offset_cells:
        return "Concentrated"
    if top_share < 0.35:
        return "Broad-based"
    if offset_cells:
        return "Mixed"
    return "Mixed"


def _build_deterministic_recommendations(
    *,
    metric_name: str,
    root_trend: Mapping[str, Any],
    drill_path: Sequence[Mapping[str, Any]],
    explainer_metrics: Mapping[str, Any],
    interaction_matrix: Mapping[str, Any],
    max_recommendations: int,
) -> Dict[str, Any]:
    operational_cells = [item for item in _safe_list(_safe_dict(_safe_dict(interaction_matrix).get("operational")).get("selected_cells")) if isinstance(item, dict)]
    clinical_cells = [item for item in _safe_list(_safe_dict(_safe_dict(interaction_matrix).get("clinical")).get("selected_cells")) if isinstance(item, dict)]
    offset_cells = [item for item in _safe_list(_safe_dict(_safe_dict(interaction_matrix).get("clinical")).get("offset_cells_preview")) if isinstance(item, dict)]
    if not drill_path and not operational_cells:
        return CorrelationRecommendationsSchema(source="empty").model_dump(mode='json')
    metric_lines = _metric_summary_lines(explainer_metrics)
    category = _choose_category(drill_path, interaction_matrix, metric_lines)
    signal_name = _infer_signal(metric_lines, explainer_metrics)
    description = _recommendation_text(str(category.get("name") or ""), signal_name)
    research_consideration = str(
        _safe_list(category.get("research_considerations"))[0]
        if _safe_list(category.get("research_considerations"))
        else "Analyst to review the concentrated variance."
    )
    cost_of_care_suggestion = str(
        _safe_list(category.get("suggestions"))[0]
        if _safe_list(category.get("suggestions"))
        else "Claims processing opportunity."
    )
    why = _build_why(
            metric_name=metric_name,
            root_trend=root_trend,
            drill_path=drill_path,
            operational_cells=operational_cells,
            clinical_cells=clinical_cells,
            offset_cells=offset_cells,
            metric_lines=metric_lines,
        )
    caveat = _root_caveat(root_trend, drill_path, offset_cells)
    item = RecommendedActionItemSchema(
        rank=1,
        priority=_normalize_priority(_confidence(drill_path, operational_cells, clinical_cells)),
        category=_category_to_action_category(str(category.get("name") or "")),
        description=description,
        evidence=_build_evidence_lines(
            metric_name=metric_name,
            root_trend=root_trend,
            drill_path=drill_path,
            operational_cells=operational_cells,
            clinical_cells=clinical_cells,
            offset_cells=offset_cells,
            metric_lines=metric_lines,
            caveat=caveat,
        ),
        story_alignment=_story_alignment_lines(why, research_consideration, cost_of_care_suggestion),
        peer_benchmarking=_build_peer_benchmarking(drill_path),
        citation=[],
    )
    items = [item]
    payload = CorrelationRecommendationsSchema(
        recommended_action=items[:max_recommendations],
        summary=RecommendationSummarySchema(
            overall_pattern=_overall_pattern(drill_path, operational_cells, offset_cells, root_trend),
            primary_next_action=item.description,
            do_not_overgeneralize=caveat,
        ),
        source="deterministic",
    )
    return payload.model_dump(mode='json')


def _llm_recommendations(
    *,
    llm: Any,
    executive_summary_json: Mapping[str, Any],
    interaction_summary_json: Mapping[str, Any],
    interaction_recommendations_json: Mapping[str, Any],
    ehap: Optional[Any] = None,
    llm_reinitializer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RECOMMENDATION_USER_PROMPT_TEMPLATE.replace(
                "{{EXECUTIVE_SUMMARY_JSON}}",
                json.dumps(executive_summary_json, indent=2, ensure_ascii=False, default=_json_default),
            )
            .replace(
                "{{INTERACTION_SUMMARY_JSON}}",
                json.dumps(interaction_summary_json, indent=2, ensure_ascii=False, default=_json_default),
            )
            .replace(
                "{{INTERACTION_RECOMMENDATIONS_JSON}}",
                json.dumps(interaction_recommendations_json, indent=2, ensure_ascii=False, default=_json_default),
            ),
        },
    ]
    try:
        # Use structured_llm_invoke with token retry support if ehap is available
        if ehap is not None and llm_reinitializer is not None:
            from deep_research_utils.ehap_retry import structured_llm_invoke
            
            parsed, _ = structured_llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                schema=CorrelationRecommendationsSchema,
                llm_reinitializer=llm_reinitializer,
            )
            # Token usage tracking not available with retry utility
            input_tokens, output_tokens = 0, 0
        else:
            # Fallback to old behavior for backward compatibility
            try:
                structured_llm = llm.with_structured_output(CorrelationRecommendationsSchema, include_raw=True)
                response = structured_llm.invoke(messages)
                parsed = response.get("parsed") if isinstance(response, dict) else response
                raw_response = response.get("raw") if isinstance(response, dict) else response
            except TypeError:
                structured_llm = llm.with_structured_output(CorrelationRecommendationsSchema)
                response = structured_llm.invoke(messages)
                parsed = response
                raw_response = response
            input_tokens, output_tokens = _extract_token_usage(raw_response)
        if parsed is None:
            return None
        payload = parsed.model_dump(mode='json') if hasattr(parsed, "model_dump") else CorrelationRecommendationsSchema(**_safe_dict(parsed)).model_dump(mode='json')
        payload["source"] = "llm" if payload.get("recommended_action") else "empty"
        payload["llm_tokens"] = correlation_llm_tokens_for_step("recommendations", input_tokens, output_tokens)
        return payload
    except Exception as exc:
        logger.warning("Correlation recommendation LLM call failed.", exc_info=exc)
        return None


def create_correlation_recommendations(
    *,
    metric_name: str,
    root_trend: Mapping[str, Any],
    drill_path: List[Mapping[str, Any]],
    explainer_metrics: Mapping[str, Any],
    interaction_matrix: Mapping[str, Any],
    interaction_summary: Mapping[str, Any],
    prior_recommendations: Mapping[str, Any],
    llm: Any | None,
    max_recommendations: int = 5,
    ehap: Optional[Any] = None,
    llm_reinitializer: Optional[Any] = None,
) -> Dict[str, Any]:
    normalized_prior = normalize_legacy_recommendations(prior_recommendations)
    executive_summary_json = {
        "root_trend": dict(root_trend),
        "drill_path": list(drill_path),
        "positive_drill_segments": [
            {
                "dimension": node.get("dimension"),
                "dimension_label": node.get("dimension_label"),
                "top_segments": _safe_list(node.get("top_segments")),
            }
            for node in drill_path
        ],
        "negative_offset_drill_segments": [
            {
                "dimension": node.get("dimension"),
                "dimension_label": node.get("dimension_label"),
                "bottom_segments": _safe_list(node.get("bottom_segments")),
            }
            for node in drill_path
            if _safe_list(node.get("bottom_segments"))
        ],
        "explainer_metrics": dict(explainer_metrics),
        "interaction_matrix": dict(interaction_matrix),
        "prior_recommendations": normalized_prior,
    }
    if llm is not None:
        llm_payload = _llm_recommendations(
            llm=llm,
            executive_summary_json=executive_summary_json,
            interaction_summary_json=interaction_summary,
            interaction_recommendations_json=normalized_prior,
            ehap=ehap,
            llm_reinitializer=llm_reinitializer,
        )
        if llm_payload is not None:
            limited_items = [item for item in _safe_list(llm_payload.get("recommended_action")) if isinstance(item, dict)][:max_recommendations]
            llm_payload["recommended_action"] = limited_items
            if not limited_items:
                llm_payload["source"] = "empty"
            result = CorrelationRecommendationsSchema(**llm_payload).model_dump(mode='json')
            result["llm_tokens"] = _safe_dict(llm_payload.get("llm_tokens")) or empty_correlation_llm_tokens()
            return result
    result = _build_deterministic_recommendations(
        metric_name=metric_name,
        root_trend=root_trend,
        drill_path=drill_path,
        explainer_metrics=explainer_metrics,
        interaction_matrix=interaction_matrix,
        max_recommendations=max_recommendations,
    )
    result["llm_tokens"] = empty_correlation_llm_tokens()
    return result
