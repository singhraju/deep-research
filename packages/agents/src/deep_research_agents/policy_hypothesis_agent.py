from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from deep_research_utils.app_constant import AppConstants
from deep_research_utils import EHAPBase  # type: ignore
try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover - local/dev fallback
    logger = logging.getLogger(__name__)

try:
    from user_intent import FilterCondition, IntentOutput
except ImportError:  # pragma: no cover - typing-only fallback
    FilterCondition = Dict[str, Any]  # type: ignore
    IntentOutput = Dict[str, Any]  # type: ignore


# ============================================================
# Module: Policy Hypothesis Agent
# ============================================================
#
# This agent consumes:
# - user_intent output (filters, group_by, metric, analysis parameters)
# - correlation agent executive summary + period window
#
# It produces a strict JSON array of policy-change hypotheses grounded in
# Anthem/Elevance policy references. The output is intended for downstream
# ranking and optional human validation.
# ============================================================


class PeriodWindow(TypedDict, total=False):
    time_dimension: str
    rolling_window: str
    start_time: int
    end_time: int
    baseline_start_time: int
    baseline_end_time: int
    comparison_strategy: str
    baseline_months: List[int]
    comparison_months: List[int]


class CorrelationSummary(TypedDict, total=False):
    run_id: str
    executive_summary_path: str
    period_window: PeriodWindow
    root_metric: str
    baseline_value: float
    comparison_value: float
    delta_value: float
    delta_pct: float
    drill_path: List[Dict[str, Any]]
    narrative_summary: str
    narrative_summary_raw: str
    executive_summary: str
    warnings: List[str]


class HypothesisAgentInput(TypedDict, total=False):
    intent: IntentOutput
    correlation_summary: CorrelationSummary
    executive_summary: Optional[str]
    executive_summary_path: Optional[str]
    period_window: Optional[PeriodWindow]
    additional_context: Optional[Dict[str, Any]]


class HypothesisItem(TypedDict):
    hypothesis_title: str
    category_of_hypothesis: List[str]
    source_policy: str
    data_signals_supporting_it: str
    contradicting_signals: str
    evidence_needed: bool
    confidence: str
    reason_for_confidence: str
    notes_assumptions: str


class GraphState(TypedDict, total=False):
    intent: IntentOutput
    correlation_summary: CorrelationSummary
    executive_summary: Optional[str]
    executive_summary_path: Optional[str]
    period_window: Optional[PeriodWindow]
    additional_context: Optional[Dict[str, Any]]
    llm: Any
    result: List[HypothesisItem]


class PeriodWindowModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    time_dimension: Optional[str] = None
    rolling_window: Optional[str] = None
    start_time: Optional[Union[int, str]] = None
    end_time: Optional[Union[int, str]] = None
    baseline_start_time: Optional[Union[int, str]] = None
    baseline_end_time: Optional[Union[int, str]] = None
    comparison_strategy: Optional[str] = None
    baseline_months: List[int] = Field(default_factory=list)
    comparison_months: List[int] = Field(default_factory=list)

    @field_validator("start_time", "end_time", "baseline_start_time", "baseline_end_time", mode="before")
    @classmethod
    def _coerce_time_value(cls, value: Any) -> Optional[Union[int, str]]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return int(stripped) if stripped.isdigit() else stripped
        return value

    @field_validator("baseline_months", "comparison_months", mode="before")
    @classmethod
    def _coerce_int_list(cls, value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            out: List[int] = []
            for item in value:
                if item is None or isinstance(item, bool):
                    continue
                if isinstance(item, (int, float)):
                    out.append(int(item))
                elif isinstance(item, str) and item.strip().isdigit():
                    out.append(int(item.strip()))
            return out
        return []


class CorrelationSummaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: Optional[str] = None
    executive_summary_path: Optional[str] = None
    period_window: Optional[PeriodWindowModel] = None
    root_metric: Optional[str] = None
    baseline_value: Optional[float] = None
    comparison_value: Optional[float] = None
    delta_value: Optional[float] = None
    delta_pct: Optional[float] = None
    drill_path: List[Dict[str, Any]] = Field(default_factory=list)
    narrative_summary: Optional[str] = None
    narrative_summary_raw: Optional[str] = None
    executive_summary: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class IntentContextModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_question: Optional[str] = None
    analysis_mode: Optional[str] = None
    metric_hint: Optional[str] = None
    group_by: List[str] = Field(default_factory=list)
    filters: List[FilterCondition] = Field(default_factory=list)
    analysis_mode_parameters: Optional[Dict[str, Any]] = None
    validation_warnings: List[str] = Field(default_factory=list)


class CorrelationContextModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    period_window_formatted: Optional[Dict[str, Any]] = None
    executive_summary_payload: Optional[Dict[str, Any]] = None


POLICY_CATEGORY_ENUM: Tuple[str, ...] = (
    "UTILIZATION_MANAGEMENT_OR_POLICY",
    "SPECIALTY_DRUG_SPEND",
    "PROVIDER_CONTRACT_OR_REIMBURSEMENT",
    "SITE_OF_CARE_SHIFT",
    "BENEFIT_DESIGN_OR_COVERAGE_CHANGE",
    "VOLUME_SURGE",
    "UNIT_COST_INFLATION",
    "CASE_MIX_OR_SEVERITY_SHIFT",
    "BILLING_CODING_OR_GROUPER_ANOMALY",
    "ONE_TIME_OUTLIER_EVENT",
    "MEMBER_POPULATION_SHIFT",
    "PAYMENT_INTEGRITY_OR_RECOVERY",
)
POLICY_CATEGORY_SET = set(POLICY_CATEGORY_ENUM)
POLICY_SOURCE_TOKENS = ("anthem", "bluecross", "wellpoint", "elevance")

POLICY_AGENT_PARAMS: Dict[str, str] = {
    "qos": "accurate",
    "preview": "false",
    "reasoning": "true",
}


class HypothesisItemModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypothesis_title: str = Field(..., min_length=1)
    category_of_hypothesis: List[str]
    source_policy: str
    data_signals_supporting_it: str = ""
    contradicting_signals: str = ""
    evidence_needed: bool
    confidence: Literal["Low", "Medium", "High"]
    reason_for_confidence: str = Field(..., min_length=1)
    notes_assumptions: str = Field(..., min_length=1)

    @field_validator("hypothesis_title", "reason_for_confidence", "notes_assumptions", mode="before")
    @classmethod
    def _strip_required(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("required field is empty")
        return text

    @field_validator("source_policy", mode="before")
    @classmethod
    def _normalize_source_policy(cls, value: Any) -> str:
        url = "" if value is None else str(value).strip()
        if not url:
            raise ValueError("source_policy is required")
        return url

    @field_validator("source_policy")
    @classmethod
    def _validate_source_policy(cls, value: str) -> str:
        lowered = value.lower()
        if not any(token in lowered for token in POLICY_SOURCE_TOKENS):
            raise ValueError("source_policy must reference Anthem/BlueCross/Wellpoint/Elevance")
        return value

    @field_validator("category_of_hypothesis", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            candidates = []
        normalized = [str(item).strip() for item in candidates if str(item).strip() in POLICY_CATEGORY_SET]
        return normalized

    @field_validator("category_of_hypothesis")
    @classmethod
    def _validate_categories(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("category_of_hypothesis must include a valid enum value")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip().capitalize()
            if cleaned in {"Low", "Medium", "High"}:
                return cleaned
        raise ValueError("confidence must be Low/Medium/High")

    @field_validator("evidence_needed", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError("evidence_needed must be a boolean")

    @field_validator("data_signals_supporting_it", "contradicting_signals", mode="before")
    @classmethod
    def _normalize_bullets(cls, value: Any) -> str:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if item is not None and str(item).strip()]
            return "\n".join(f"- {item}" for item in items)
        if isinstance(value, str):
            lines = [line.strip() for line in value.splitlines() if line.strip()]
            if not lines:
                return ""
            if any(line.startswith("-") for line in lines):
                return "\n".join(lines)
            return "\n".join(f"- {line}" for line in lines)
        return ""


POLICY_SYSTEM_PROMPT = """
You are a claims spike root-cause analyst specializing in payer policy, reimbursement, and coverage changes for Elevance Health / Anthem-branded plans.

Goal:
Given "baseline vs spike vs after" aggregated claim metrics by dimension (e.g., HCPCS/CPT, DRG, Dx, POS, Revenue Code), produce a ranked set of plausible policy or rule changes that could explain the spike.

Key constraints:
- Do NOT invent policy changes. If you cannot find evidence, say "not verified" and propose specific searches or internal follow-ups.
- Treat any dimension-level correlation as a hypothesis, not proof.
- Use the resolved intent context JSON as the primary source of business context (filters, metric, LOB, state, claim type, time window).

Inputs you may receive:
- A list of aggregated tables, each row representing a single dimension value.
- Context about the time windows (baseline/spike/after dates), line of business, state/market, claim type, etc.
- Sometimes additional breakdowns (e.g., provider specialty, network, DRG+RevCode combos).
- A resolved intent context JSON from the intent resolver.

Reasoning process:
1) Identify the biggest contributors to DELTA_ALWD and isolate whether the spike is:
   - "rate" driven (ALWD_PER_CLAIM up) vs
   - "volume/mix" driven (CLAIMS/LINES/MEMBERS up) vs both.
2) Detect consistent signatures:
   - A small number of codes driving most delta (policy/fee schedule likely)
   - Broad uplift across many codes (unit price index, contract, benefit, systemic issue)
   - Facility-only patterns (Rev code/DRG/POS signatures)
3) Propose policy-change hypotheses tied to the observed signature with:
   - Why it fits the data
   - What evidence would confirm/deny it
4) Respond per user instructions

Style:
Direct, concise, and operational. Strict JSON structure. Avoid filler and any other text.
"""


POLICY_USER_PROMPT_TEMPLATE = """Task:
Determine whether a claims spike is plausibly explained by an Anthem/Elevance policy or reimbursement/UM change. Use the aggregated before/spike/after comparison tables to form hypotheses and explain the spike.

Business context (resolved from intent + correlation):
- Claim type: {claim_type}
- Line of business (lob): {lob}
- State/market: {state_market}
- Time window start: {time_window_start}
- Time window end: {time_window_end}

Resolved intent context (verbatim JSON):
{intent_context_json}

Correlation summary context (verbatim JSON):
{correlation_context_json}

Categories of hypothesis:
- UTILIZATION_MANAGEMENT_OR_POLICY
Changes in medical or pharmacy policy (e.g., prior auth, step therapy, coverage rules) that make services newly payable or alter approval behavior.

- SPECIALTY_DRUG_SPEND
Spend driven by high-cost specialty drugs (e.g., J-codes, injectables) with low utilization and very high cost per claim.

- PROVIDER_CONTRACT_OR_REIMBURSEMENT
Cost changes caused by provider contract updates, negotiated rate changes, or out-of-network reimbursement.

- SITE_OF_CARE_SHIFT
Services delivered or billed in a higher-cost setting (e.g., office vs outpatient hospital), increasing per-claim cost without volume growth.

- BENEFIT_DESIGN_OR_COVERAGE_CHANGE
Benefit or formulary changes that move services from non-covered to covered or materially change member liability.

- VOLUME_SURGE
Increased number of claims or services with relatively stable cost per claim.

- UNIT_COST_INFLATION
Increase in allowed amount per claim without clear attribution to provider contract, site of care, or benefit design changes.

- CASE_MIX_OR_SEVERITY_SHIFT
Higher-acuity patients or more complex treatment pathways for the same condition driving higher cost per claim.

- BILLING_CODING_OR_GROUPER_ANOMALY
Data, coding, or grouping issues (e.g., unknown DRG, default revenue codes, mapping errors) that distort allowed amounts.

- ONE_TIME_OUTLIER_EVENT
A very small number of non-recurring, high-cost claims explaining most of the spend change.

- MEMBER_POPULATION_SHIFT
Changes in enrolled population size or composition (e.g., eligibility, demographics) affecting total spend.

- PAYMENT_INTEGRITY_OR_RECOVERY
Spend changes driven by adjustments, recoupments, reprocessing, or recovery activity rather than new utilization.

Output requirements:
Return a JSON array with each element having these fields:
   - hypothesis_title: string (5-10 words)
   - category_of_hypothesis: array of string (enum described above)
   - source_policy: string (url to the state & lob specific policy document/website/blog referred. If this is missing do not give the hypothesis. Do not repeat, put all data for a url in one element. Only provde url specific to anthem/bluecross/wellpoint/elevance health.)
   - data_signals_supporting_it: string (bullet points)
   - contradicting_signals: string (bullet points)
   - evidence_needed: boolean (true/false indicating if more evidence is needed)
   - confidence: string (Low/Medium/High only, do not give a combination of them)
   - reason_for_confidence: string (Reason for confidence)
   - notes_assumptions: string (explicit)

Given the spike summary below, determine the most likely cause of the spike.

{spike_summary}
"""


# =============================
# Generic helpers
# =============================


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def _dedupe_by_source_policy(items: Sequence[HypothesisItem]) -> List[HypothesisItem]:
    seen: set[str] = set()
    deduped: List[HypothesisItem] = []
    for item in items:
        url = item["source_policy"]
        if url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    return deduped


def _format_time_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value_str = str(int(value))
    else:
        value_str = str(value).strip()
    if not value_str:
        return None
    if value_str.isdigit() and len(value_str) == 6:
        return f"{value_str[:4]}-{value_str[4:]}"
    if value_str.isdigit() and len(value_str) == 8:
        return f"{value_str[:4]}-{value_str[4:6]}-{value_str[6:]}"
    return value_str


def _extract_filter_value(filters: Sequence[FilterCondition], keywords: Sequence[str]) -> Optional[str]:
    for flt in filters:
        field = str(flt.get("field", "")).lower()
        if any(keyword in field for keyword in keywords):
            value = flt.get("value")
            if value is not None:
                return str(value)
    return None


def _extract_claim_type(intent: Optional[IntentOutput]) -> str:
    filters = _safe_list(intent.get("filters")) if intent else []
    claim_type = _extract_filter_value(
        filters,
        ("claim_type", "claim_class", "claim_category", "bill_type", "claim_source"),
    )
    return claim_type or "Both"


def _extract_lob(intent: Optional[IntentOutput]) -> str:
    filters = _safe_list(intent.get("filters")) if intent else []
    lob = _extract_filter_value(filters, ("lob", "line_of_business", "product_line"))
    return lob or "Unknown"


def _extract_state_market(intent: Optional[IntentOutput]) -> str:
    filters = _safe_list(intent.get("filters")) if intent else []
    state = _extract_filter_value(filters, ("state", "service_area_state", "service_state"))
    market = _extract_filter_value(filters, ("market", "service_area", "region"))
    if state and market:
        return f"{state} / {market}"
    return state or market or "Unknown"


def _extract_period_from_intent(intent: Optional[IntentOutput]) -> PeriodWindow:
    if not intent:
        return {}
    params = _safe_dict(intent.get("analysis_mode_parameters"))
    period = _safe_dict(params.get("period"))
    model = PeriodWindowModel(
        time_dimension=period.get("rolling_time_dimension"),
        rolling_window=_safe_list(period.get("rolling_window"))[0] if _safe_list(period.get("rolling_window")) else None,
        start_time=period.get("start_time"),
        end_time=period.get("end_time"),
        baseline_start_time=period.get("baseline_start_time"),
        baseline_end_time=period.get("baseline_end_time"),
        comparison_strategy=period.get("comparison_strategy"),
    )
    return model.model_dump(exclude_none=True)  # type: ignore[return-value]


def _resolve_period_window(state: GraphState) -> PeriodWindow:
    summary_period = _safe_dict(state.get("correlation_summary", {})).get("period_window")
    period_window = _safe_dict(state.get("period_window")) or _safe_dict(summary_period)
    if period_window:
        try:
            model = PeriodWindowModel(**period_window)
            return model.model_dump(exclude_none=True)  # type: ignore[return-value]
        except ValidationError as exc:
            logger.warning("Period window validation failed; using raw payload.", exc_info=exc)
            return period_window  # type: ignore[return-value]
    return _extract_period_from_intent(state.get("intent"))


def _build_time_window_labels(period_window: PeriodWindow) -> Tuple[str, str, Dict[str, Any]]:
    start_time = _format_time_value(period_window.get("start_time"))
    end_time = _format_time_value(period_window.get("end_time"))
    baseline_start = _format_time_value(period_window.get("baseline_start_time"))
    baseline_end = _format_time_value(period_window.get("baseline_end_time"))

    summary = {
        "comparison_window": {"start": start_time, "end": end_time},
        "baseline_window": {"start": baseline_start, "end": baseline_end},
        "comparison_strategy": period_window.get("comparison_strategy"),
        "rolling_window": period_window.get("rolling_window"),
        "time_dimension": period_window.get("time_dimension"),
        "baseline_months": period_window.get("baseline_months"),
        "comparison_months": period_window.get("comparison_months"),
    }

    return start_time or "Unknown", end_time or "Unknown", summary


def _load_executive_summary_payload(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        logger.warning("Executive summary path does not exist: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _safe_dict(data)
    except Exception as exc:
        logger.warning("Failed to read executive summary file %s", path, exc_info=exc)
        return {}


def _resolve_spike_summary(state: GraphState, summary_payload: Dict[str, Any]) -> str:
    direct = state.get("executive_summary")
    if direct:
        return str(direct).strip()

    summary = _safe_dict(state.get("correlation_summary"))
    for key in ("executive_summary", "narrative_summary", "narrative_summary_raw"):
        value = summary.get(key)
        if value:
            return str(value).strip()

    for key in ("executive_summary", "narrative_summary", "narrative_summary_raw", "story_stub"):
        value = summary_payload.get(key)
        if value:
            return str(value).strip()

    return "No executive summary was available from the correlation run."


def _build_intent_context(intent: Optional[IntentOutput]) -> Dict[str, Any]:
    if not intent:
        return {}
    try:
        model = IntentContextModel(
            raw_question=intent.get("raw_question"),
            analysis_mode=intent.get("analysis_mode"),
            metric_hint=intent.get("metric_hint"),
            group_by=_safe_list(intent.get("group_by")),
            filters=_safe_list(intent.get("filters")),
            analysis_mode_parameters=_safe_dict(intent.get("analysis_mode_parameters")),
            validation_warnings=_safe_list(intent.get("validation_warnings")),
        )
    except ValidationError as exc:
        logger.warning("Intent context validation failed; using raw intent payload.", exc_info=exc)
        return _safe_dict(intent)
    return model.model_dump(exclude_none=True)


def _build_correlation_context(
    correlation_summary: Mapping[str, Any],
    period_summary: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    try:
        model = CorrelationSummaryModel(**dict(correlation_summary))
        summary.update(model.model_dump(exclude_none=True))
    except ValidationError as exc:
        logger.warning("Correlation summary validation failed; using raw payload.", exc_info=exc)
        summary.update(dict(correlation_summary))

    if period_summary:
        summary["period_window_formatted"] = period_summary
    if summary_payload:
        summary["executive_summary_payload"] = summary_payload

    return summary


def _serialize_for_llm(value: Any, *, max_chars: int = 25_000) -> str:
    if isinstance(value, BaseModel):
        text = value.model_dump_json(exclude_none=True)
        return _truncate_text(text, max_chars)
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))
    return _truncate_text(text, max_chars)


def _parse_hypothesis_response(text: str) -> List[HypothesisItem]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Policy hypothesis response was not valid JSON.")
        return []

    if not isinstance(payload, list):
        logger.warning("Policy hypothesis response was not a JSON array.")
        return []

    normalized: List[HypothesisItem] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            model = HypothesisItemModel(**item)
        except ValidationError as exc:
            logger.warning("Skipping invalid policy hypothesis item.", exc_info=exc)
            continue
        normalized.append(model.model_dump())

    return _dedupe_by_source_policy(normalized)


# =============================
# LLM configuration
# =============================


def build_llm(
    model_name: Optional[str] = None,
    reasoning_effort: str = "high",
    summary_mode: Optional[str] = "auto",
    llm_params: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Build LLM client with automatic token refresh.
    
    Returns a ChatOpenAI client with a fresh token. For long-running processes,
    consider calling this function periodically or using AgentBase which handles
    token refresh automatically.
    
    Args:
        model_name: LLM model name (defaults to DEEP_RESEARCH_LLM_MODEL)
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        summary_mode: Summary mode ("auto", "detailed", or None)
        llm_params: Additional parameters for extra_body
        
    Returns:
        ChatOpenAI client with current token
    """
    ehap = EHAPBase(
        base_url=AppConstants.EHAP_BASE_URL,
        client_id=AppConstants.EHAP_CLIENT_ID,
        client_secret=AppConstants.EHAP_CLIENT_SECRET,
        verify=AppConstants.SSL_CERT_FILE or False,
    )
    token = ehap.get_token()  # Gets fresh token with proactive refresh

    model_name = model_name or AppConstants.DEEP_RESEARCH_LLM_MODEL

    extra_body: Dict[str, Any] = {
        "reasoning_effort": reasoning_effort,
        "summary": summary_mode,
    }
    if llm_params:
        extra_body.update(llm_params)

    return ChatOpenAI(
        base_url=AppConstants.OPENAI_BASE_URL,
        model=model_name,
        api_key=token,
        extra_body=extra_body,
        http_client=AppConstants.http_client_,
        http_async_client=AppConstants.http_async_client_,
    )


# =============================
# Hypothesis generation node
# =============================


def generate_policy_hypotheses(state: GraphState) -> Dict[str, Any]:
    llm = state.get("llm")
    if llm is None:
        logger.warning("Policy hypothesis LLM unavailable; returning empty hypothesis list.")
        return {"result": []}

    intent = state.get("intent")
    raw_summary = _safe_dict(state.get("correlation_summary"))
    try:
        correlation_model = CorrelationSummaryModel(**raw_summary)
        correlation_summary = correlation_model.model_dump(exclude_none=True)
    except ValidationError as exc:
        logger.warning("Correlation summary validation failed; using raw payload.", exc_info=exc)
        correlation_summary = raw_summary
    summary_payload = _load_executive_summary_payload(state.get("executive_summary_path"))
    period_window = _resolve_period_window(state)
    time_window_start, time_window_end, period_summary = _build_time_window_labels(period_window)

    intent_context = _build_intent_context(intent)
    correlation_context = _build_correlation_context(correlation_summary, period_summary, summary_payload)

    claim_type = _extract_claim_type(intent)
    lob = _extract_lob(intent)
    state_market = _extract_state_market(intent)

    spike_summary = _resolve_spike_summary(state, summary_payload)

    user_prompt = POLICY_USER_PROMPT_TEMPLATE.format(
        claim_type=claim_type,
        lob=lob,
        state_market=state_market,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        intent_context_json=_serialize_for_llm(intent_context),
        correlation_context_json=_serialize_for_llm(correlation_context),
        spike_summary=spike_summary,
    )

    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Handle different types of llm parameter (callable, tuple, or direct client)
    try:
        if callable(llm):
            # llm is a factory function
            llm_client = llm()
        elif isinstance(llm, tuple):
            # llm is a tuple (llm_client, ehap_client)
            llm_client = llm[0]
        else:
            # llm is direct LLM client
            llm_client = llm
            
        response = llm_client.invoke(messages)
    except Exception as e:
        logger.error(f"Failed to get LLM client or invoke: {e}")
        return {"result": []}
    content = str(getattr(response, "content", response)).strip()
    hypotheses = _parse_hypothesis_response(content)

    logger.info("Policy hypothesis generation produced %s hypothesis items.", len(hypotheses))
    return {"result": hypotheses}


# =============================
# Graph factory
# =============================


def build_app(
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
    llm_params: Optional[Dict[str, str]] = None,
) -> Callable[..., List[HypothesisItem]]:
    """
    Build a LangGraph app that generates policy hypotheses from intent + correlation context.
    """
    llm_client = llm
    if llm_client is None and llm_builder is not None:
        try:
            llm_client = llm_builder()
        except Exception as exc:
            logger.warning("Failed to build policy hypothesis LLM.", exc_info=exc)
            llm_client = None

    if llm_client is None:
        try:
            llm_client = build_llm(llm_params=llm_params or POLICY_AGENT_PARAMS)
        except Exception as exc:
            logger.warning("Policy hypothesis LLM unavailable; continuing without LLM.", exc_info=exc)
            llm_client = None

    graph = StateGraph(GraphState)
    graph.add_node("generate_policy_hypotheses", generate_policy_hypotheses)
    graph.add_edge(START, "generate_policy_hypotheses")
    graph.add_edge("generate_policy_hypotheses", END)
    app = graph.compile()

    def run(
        intent: Optional[IntentOutput] = None,
        correlation_summary: Optional[Mapping[str, Any]] = None,
        executive_summary: Optional[str] = None,
        executive_summary_path: Optional[str] = None,
        period_window: Optional[Mapping[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> List[HypothesisItem]:
        """
        Generate policy hypotheses.

        Args:
            intent: IntentOutput from user_intent resolver.
            correlation_summary: Correlation agent output payload.
            executive_summary: Optional direct summary override.
            executive_summary_path: Optional path to correlation executive_summary.json.
            period_window: Optional explicit period window override.
            additional_context: Optional extra context for future extensions.

        Returns:
            List of hypothesis items (JSON-serializable).
        """
        state: GraphState = {
            "intent": intent or {},
            "correlation_summary": _safe_dict(correlation_summary),
            "executive_summary": executive_summary,
            "executive_summary_path": executive_summary_path,
            "period_window": _safe_dict(period_window),
            "additional_context": additional_context,
            "llm": llm_client,
        }
        result = app.invoke(state)["result"]
        return result

    return run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _StubLLM:
        def invoke(self, _messages: List[Dict[str, str]]) -> str:
            return json.dumps(
                [
                    {
                        "hypothesis_title": "Sample policy update for H0010",
                        "category_of_hypothesis": ["UTILIZATION_MANAGEMENT_OR_POLICY"],
                        "source_policy": "https://www.anthem.com/provider/medicalpolicies/",
                        "data_signals_supporting_it": "- H0010 is the dominant delta driver",
                        "contradicting_signals": "- No broad uplift across other codes",
                        "evidence_needed": True,
                        "confidence": "Low",
                        "reason_for_confidence": "Stub response for local testing.",
                        "notes_assumptions": "Assumes an Anthem policy update applies to H0010.",
                    }
                ]
            )

    try:
        llm_client = build_llm(llm_params=POLICY_AGENT_PARAMS)
    except Exception as exc:
        logger.warning("LLM unavailable; using stub for local testing only.", exc_info=exc)
        llm_client = _StubLLM()

    run = build_app(llm=llm_client)

    sample_intent: IntentOutput = {
        "analysis_mode": "cost_change_investigation_over_time_window",
        "analysis_mode_parameters": {
            "period": {
                "rolling_window": ["3_months"],
                "rolling_time_dimension": "claims_expense.incurred_month",
                "start_time": 202509,
                "end_time": 202511,
                "baseline_start_time": 202506,
                "baseline_end_time": 202508,
                "comparison_strategy": "rolling",
            },
            "drill_metric": ["claims_expense.total_paid"],
        },
        "filters": [
            {"field": "procedure_code", "operator": "=", "value": "H0010", "source": "dimension_match"},
            {"field": "lob_description", "operator": "=", "value": "Commercial", "source": "dimension_match"},
            {"field": "service_area_state", "operator": "=", "value": "TX", "source": "dimension_match"},
        ],
        "group_by": ["procedure_code"],
        "metric_hint": "claims_expense.total_paid",
        "raw_question": "Find what changed in HCPCS H0010 in Texas.",
        "validation_warnings": [],
    }

    sample_correlation: CorrelationSummary = {
        "executive_summary": "Total paid rose 34% driven by H0010 in TX during the comparison window.",
        "period_window": {
            "start_time": 202509,
            "end_time": 202511,
            "baseline_start_time": 202506,
            "baseline_end_time": 202508,
            "rolling_window": "3_months",
            "time_dimension": "claims_expense.incurred_month",
            "comparison_strategy": "rolling",
        },
        "root_metric": "claims_expense.total_paid",
        "delta_pct": 0.34,
    }

    output = run(intent=sample_intent, correlation_summary=sample_correlation)
    print(json.dumps(output, indent=2))
    # python packages/core/src/deep_research_core/policy_hypothesis_agent.py
