"""
Unit tests for the recommendation validator node added in Draft B.

The validator runs as a node in the LangGraph after `generate_recommendation`
and prunes items that drift from the pattern scope or rest on a source
policy that disclaims the target code (e.g. "referenced only in
policy-history language" — what caused T4 in the original failure).

Drop-only (no regeneration loop), balanced threshold per the plan:
drop on clear scope drift OR clear source-disclaim, keep otherwise.

These tests exercise the node directly with a stub LLM. The agent is built
via `__new__` to bypass Snowflake / mini-LLM init.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pytest

from deep_research_agents import ReimbursementAgent


# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class StubLLM:
    """Returns a canned JSON validator payload and records the prompt sent."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.last_prompt: str = ""
        self.call_count: int = 0

    def invoke(self, messages: List[Dict[str, str]]) -> _StubResponse:
        self.call_count += 1
        self.last_prompt = messages[-1]["content"] if messages else ""
        return _StubResponse(json.dumps(self._payload))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> ReimbursementAgent:
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.reimbursement_validator")
    inst._token_breakdown = {}
    return inst


@pytest.fixture
def pattern() -> Dict[str, Any]:
    return {
        "pattern_title": "ED visit 99284 – KY Commercial gap",
        "evidence_summary": "Kentucky observation rules differ from baseline.",
        "priority_entities": {"states": ["KY"]},
    }


def _citation(cid: str, payer: str, policy_title: str, fact_key: str, fact: str) -> Dict[str, Any]:
    return {
        "id": cid,
        "payer": payer,
        "policy_title": policy_title,
        "policy_url": f"https://example.com/{policy_title.replace(' ', '_')}.pdf",
        "fact_key": fact_key,
        "fact": fact,
    }


@pytest.fixture
def citation_index() -> Dict[str, Dict[str, Any]]:
    return {
        "C1": _citation("C1", "United Health", "ED Facility E&M Policy", "revenue_codes", "0762"),
        "C2": _citation("C2", "United Health", "Preventive Medicine Policy", "required_modifiers", "59"),
    }


@pytest.fixture
def individual_policies() -> List[Dict[str, Any]]:
    """Mirror the per-policy `evidence` summary built in format_output_node —
    note the disclaimer phrasing on Preventive Medicine Policy that the
    validator should treat as a source-disclaim signal."""
    return [
        {
            "payer_name": "United Health",
            "policy_title": "ED Facility E&M Policy",
            "evidence": "Observation services are eligible for reimbursement with revenue code 0762 and HCPCS G0378.",
        },
        {
            "payer_name": "United Health",
            "policy_title": "Preventive Medicine Policy",
            "evidence": "Policy is primarily about modifiers 59 and X{EPSU}; 99284 is referenced only in policy-history language.",
        },
    ]


def _items() -> List[Dict[str, Any]]:
    return [
        {
            "kind": "edit",
            "text": "Require revenue code 0762 when G0378 pairs with 99284.",
            "citations": ["C1"],
            "scope": [],
            "peer_cite": "UHC aligns 0762 with observation services.",
            "exemptions": [],
        },
        {
            "kind": "edit",
            "text": "Require modifier 59/XE/XP/XS/XU when 99284 is distinct.",
            "citations": ["C2"],
            "scope": [],
            "peer_cite": None,
            "exemptions": [],
        },
    ]


def _rec_with_transient(
    items: List[Dict[str, Any]],
    citation_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a recommendation dict shaped like what
    `_generate_policy_recommendations` returns, including the transient
    `_items` / `_citation_index` / `_specific_facts` keys the validator
    node pops."""
    return {
        "rank": 1,
        "priority": "MEDIUM",
        "category": "Policy",
        "description": "Edit 1: ...\nEdit 2: ...",
        "evidence": ["pre-existing"],
        "story_alignment": [],
        "peer_benchmarking": ["UHC aligns 0762 with observation services."],
        "citation": ["pre-existing"],
        "_items": items,
        "_citation_index": citation_index,
        "_specific_facts": {"citations": list(citation_index.values())},
    }


def _state(
    recommendations: List[Dict[str, Any]],
    pattern: Dict[str, Any],
    individual_policies: List[Dict[str, Any]],
    llm: Any,
    query: str = "Compare Elevance KY Commercial policy for CPT 99284 with peers.",
) -> Dict[str, Any]:
    return {
        "recommended_action": recommendations,
        "pattern": pattern,
        "query": query,
        "formatted_output": {
            "individual_policies": individual_policies,
            "summary_table": {"subtitle": "Commercial / KY"},
        },
        "llm": llm,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_drop_decision_filters_one_item_and_rebuilds_chips(
    agent, pattern, citation_index, individual_policies,
):
    items = _items()
    rec = _rec_with_transient(items, citation_index)
    payload = {
        "verdicts": [
            {"item_index": 0, "decision": "keep", "reason": "in scope"},
            {"item_index": 1, "decision": "drop", "reason": "source disclaims target code"},
        ],
        "summary": "dropped 1 item resting on policy-history mention",
    }
    llm = StubLLM(payload)

    result = agent.validate_recommendation_node(
        _state([rec], pattern, individual_policies, llm),
    )

    recs = result["recommended_action"]
    assert len(recs) == 1
    surviving = recs[0]
    # Description re-rendered over kept items only.
    assert "0762" in surviving["description"]
    assert "modifier 59" not in surviving["description"]
    # Evidence chips reflect only the C1 citation.
    assert any("0762" in line for line in surviving["evidence"])
    assert not any("modifier" in line.lower() or '"59"' in line for line in surviving["evidence"])
    # Transient keys are removed by pop() — extract_result must not see them.
    assert "_items" not in surviving
    assert "_citation_index" not in surviving
    assert "_specific_facts" not in surviving

    rv = result["recommendation_validation"]
    assert rv["decision"] == "items_dropped"
    assert rv["dropped_count"] == 1
    assert rv["kept_count"] == 1
    assert len(rv["verdicts"]) == 2


def test_all_drop_suppresses_recommendation(
    agent, pattern, citation_index, individual_policies,
):
    items = _items()
    rec = _rec_with_transient(items, citation_index)
    payload = {
        "verdicts": [
            {"item_index": 0, "decision": "drop", "reason": "scope drift"},
            {"item_index": 1, "decision": "drop", "reason": "scope drift"},
        ],
        "summary": "both items off-pattern",
    }
    llm = StubLLM(payload)

    result = agent.validate_recommendation_node(
        _state([rec], pattern, individual_policies, llm),
    )

    assert result["recommended_action"] == []
    rv = result["recommendation_validation"]
    assert rv["decision"] == "suppressed"
    assert rv["dropped_count"] == 2
    assert rv["kept_count"] == 0


def test_validator_is_noop_when_no_recommendation(
    agent, pattern, individual_policies,
):
    llm = StubLLM({"verdicts": [], "summary": "unused"})
    result = agent.validate_recommendation_node(
        _state([], pattern, individual_policies, llm),
    )
    assert result["recommended_action"] == []
    rv = result["recommendation_validation"]
    assert rv["decision"] == "skipped"
    assert rv["dropped_count"] == 0
    # The LLM must NOT be called when there's nothing to validate.
    assert llm.call_count == 0


def test_validator_default_keeps_when_parse_fails(
    agent, pattern, citation_index, individual_policies,
):
    """If the validator returns garbage, every item stays (we never silently
    suppress a recommendation on a parser hiccup)."""
    items = _items()
    rec = _rec_with_transient(items, citation_index)

    class GarbageLLM:
        def invoke(self, messages):
            return _StubResponse("not json at all {{{")

    result = agent.validate_recommendation_node(
        _state([rec], pattern, individual_policies, GarbageLLM()),
    )
    recs = result["recommended_action"]
    assert len(recs) == 1
    # No verdicts -> drop_indices is empty -> all items kept.
    rv = result["recommendation_validation"]
    assert rv["decision"] == "ok"
    assert rv["dropped_count"] == 0
    assert rv["kept_count"] == 2


def test_prompt_includes_per_policy_disclaimer_signal(
    agent, pattern, citation_index, individual_policies,
):
    """Regression guard for T4: the validator MUST see each cited policy's
    `evidence` summary so it can detect 'referenced only in policy-history
    language' style disclaimers."""
    items = _items()
    rec = _rec_with_transient(items, citation_index)
    llm = StubLLM({"verdicts": [], "summary": ""})

    agent.validate_recommendation_node(
        _state([rec], pattern, individual_policies, llm),
    )

    assert "policy-history" in llm.last_prompt
    # Pattern context must reach the prompt too.
    assert "KY Commercial gap" in llm.last_prompt
    assert "States in scope: KY" in llm.last_prompt


# ---------------------------------------------------------------------------
# extract_result surface
# ---------------------------------------------------------------------------

def test_extract_result_surfaces_validator_check_and_warning(agent):
    """Build a graph_output dict carrying a known recommendation_validation
    and confirm extract_result appends the alignment check + warning."""
    graph_output = {
        "pattern_rank": 1,
        "pattern": {"pattern_title": "ED 99284 KY gap"},
        "result": [{"some": "policy"}],
        "formatted_output": {"summary_table": {}, "individual_policies": []},
        "recommended_action": [{
            "rank": 1, "priority": "MEDIUM", "category": "Policy",
            "description": "Edit 1: kept", "evidence": [], "story_alignment": [],
            "peer_benchmarking": [], "citation": [],
        }],
        "cpt_codes": "99284",
        "drg_codes": [],
        "elevance_executive_summary": None,
        "conversation_id": "demo_001",
        "job_id": "j1",
        "start_time": "2026-06-10T20:00:00+00:00",
        "recommendation_validation": {
            "decision": "items_dropped",
            "summary": "dropped 1 for scope drift",
            "dropped_count": 1,
            "kept_count": 1,
            "verdicts": [{"item_index": 1, "decision": "drop", "reason": "off-pattern"}],
        },
    }

    out = agent.extract_result(graph_output)
    checks = out["validation"]["checks"]
    aligned = [c for c in checks if c["check"] == "recommendation_pattern_aligned"]
    assert len(aligned) == 1
    assert aligned[0]["passed"] is True
    assert "scope drift" in aligned[0]["message"]
    warnings = out["validation"]["warnings"]
    assert any("dropped 1" in w for w in warnings)


def test_extract_result_marks_suppressed_as_failed_check(agent):
    """When the validator suppressed the recommendation, the alignment check
    must report passed=False so consumers see the failure."""
    graph_output = {
        "pattern_rank": 1,
        "pattern": {"pattern_title": "X"},
        "result": [{"some": "policy"}],
        "formatted_output": {"summary_table": {}, "individual_policies": []},
        "recommended_action": [],
        "cpt_codes": "99284",
        "drg_codes": [],
        "elevance_executive_summary": None,
        "conversation_id": "demo_001",
        "job_id": "j1",
        "start_time": None,
        "recommendation_validation": {
            "decision": "suppressed",
            "summary": "all items off-pattern",
            "dropped_count": 2,
            "kept_count": 0,
            "verdicts": [],
        },
    }

    out = agent.extract_result(graph_output)
    aligned = [
        c for c in out["validation"]["checks"]
        if c["check"] == "recommendation_pattern_aligned"
    ]
    assert len(aligned) == 1
    assert aligned[0]["passed"] is False


# ---------------------------------------------------------------------------
# Phase 1b sanitizer
# ---------------------------------------------------------------------------

def test_sanitizer_strips_drg_prompt_leak(agent, caplog):
    """target_codes containing the literal 'DRG Codes:' prompt-header phrase
    must be dropped, even when the entry would otherwise pass the regex.
    Real CPT codes alongside the leak survive."""
    payload = {
        "policy_metadata": {},
        "results": [
            {
                "code": "59510",
                "target_codes": [
                    "DRG Codes: Cesarean Section without Sterilization without CC/MCC",
                    "59510",
                    "59618",
                ],
                "related_codes": [],
                "required_modifiers": [],
                "excluded_modifiers": [],
                "revenue_codes": [],
            }
        ],
    }
    with caplog.at_level(logging.WARNING):
        sanitized = agent._sanitize_extracted_facts(
            "PID-123", "Add-on Coding", payload
        )

    targets = sanitized["results"][0]["target_codes"]
    assert "59510" in targets
    assert "59618" in targets
    assert not any("DRG Codes:" in t for t in targets)
    # Log must name the policy_id and policy_title so reviewers can trace.
    assert any("PID-123" in rec.getMessage() for rec in caplog.records)
    assert any("Add-on Coding" in rec.getMessage() for rec in caplog.records)


def test_sanitizer_strips_non_code_target_codes(agent):
    """Free-text strings in target_codes/related_codes must be dropped
    by the regex whitelist (CPT 5-digit or HCPCS letter+4-digit)."""
    payload = {
        "policy_metadata": {},
        "results": [
            {
                "code": "99284",
                "target_codes": ["99284", "Cesarean Section", "G0378"],
                "related_codes": ["bundled service description", "59510"],
                "required_modifiers": ["25", "the modifier 25"],
                "excluded_modifiers": [],
                "revenue_codes": ["0762", "revenue code 0762"],
            }
        ],
    }
    sanitized = agent._sanitize_extracted_facts("PID-1", "Test Policy", payload)
    rule = sanitized["results"][0]
    assert rule["target_codes"] == ["99284", "G0378"]
    assert rule["related_codes"] == ["59510"]
    assert rule["required_modifiers"] == ["25"]
    assert rule["revenue_codes"] == ["0762"]


# ---------------------------------------------------------------------------
# Phase 2 validate_policies_node + helpers
# ---------------------------------------------------------------------------

def _make_extracted_policy(
    plcy_id: str,
    rules: List[Dict[str, Any]] | None = None,
    policy_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "PLCY_ID": plcy_id,
        "policy_metadata": policy_metadata or {},
        "results": rules or [],
    }


def _make_filtered_policy(
    plcy_id: str, payor: str, title: str
) -> Dict[str, Any]:
    return {
        "policy_id": plcy_id,
        "payor": payor,
        "policy_title": title,
        "external_link": f"https://example.com/{plcy_id}.pdf",
    }


def test_validate_policies_flags_cross_payer_leak(agent):
    """Elevance policy whose evidence says 'Cigna does not typically allow' →
    cross_payer_name_leak flag, severity=quarantine (own payer absent)."""
    extracted = _make_extracted_policy(
        "P1",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": (
                    "For multiple births, Cigna does not typically allow "
                    "additional reimbursement unless modifier 22 is appended."
                ),
                "specific_rule_text": "",
                "target_codes": ["59510"],
            }
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P1", "Elevance Health (external)", "Maternity Services.pdf")
        ],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": [],
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    assert contam is not None
    assert "cross_payer_name_leak" in contam["flags"]
    assert contam["severity"] == "quarantine"
    assert out["policy_contamination_summary"]["quarantined"][0]["policy_id"] == "P1"


def test_validate_policies_flags_topic_irrelevant_via_judge(agent):
    """The LLM relevance judge returns relevant=false with high confidence
    → policy_topic_irrelevant flag, severity=quarantine. Generalizes to
    any specialty because the judge inherits the model's medical knowledge."""
    extracted = _make_extracted_policy(
        "P2",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": (
                    "Global obstetrical codes apply once per pregnancy under "
                    "same TIN; cesarean delivery is bundled."
                ),
                "specific_rule_text": "",
                "target_codes": [],
                "pairing_conditions": ["once per pregnancy", "same TIN"],
                "program_scope": ["Maternity"],
            }
        ],
    )
    judge_llm = StubLLM({
        "relevant": False,
        "confidence": "high",
        "reason": "Readmission policy does not adjudicate obstetric delivery codes.",
    })
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P2", "Molina", "Readmission")
        ],
        "pattern": {"pattern_title": "Cesarean delivery spend"},
        "cpt_codes": "",
        "drg_codes": ["Cesarean Section without Sterilization without CC/MCC"],
        "llm_mini": judge_llm,
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    assert contam is not None
    assert "policy_topic_irrelevant" in contam["flags"]
    assert contam["severity"] == "quarantine"
    assert judge_llm.call_count == 1, "judge should be called once per policy"


def test_validate_policies_judge_medium_confidence_warns_not_quarantines(agent):
    """relevant=false with confidence=medium → warn (not quarantine).
    Regression guard for the 2026-06-12 cesarean run where the judge was
    rating clearly-relevant maternity policies as confidence=high not-
    relevant; demoting medium to warn keeps borderline-rejected policies
    visible in the API output rather than silently dropping them."""
    extracted = _make_extracted_policy(
        "P_MED",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": "Maternity bundling rules apply.",
                "target_codes": ["59510"],
            }
        ],
    )
    judge_llm = StubLLM({
        "relevant": False,
        "confidence": "medium",
        "reason": "Policy describes general OB rules without the pattern's specific drivers.",
    })
    state = {
        "result": [extracted],
        "filtered_policies": [_make_filtered_policy("P_MED", "Cigna", "Global Maternity")],
        "pattern": {"pattern_title": "Cesarean spend"},
        "cpt_codes": "",
        "drg_codes": ["Cesarean"],
        "llm_mini": judge_llm,
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    assert contam is not None
    assert "policy_topic_irrelevant" in contam["flags"]
    assert contam["severity"] == "warn"


def test_validate_policies_judge_low_confidence_warns_not_quarantines(agent):
    """relevant=false with confidence=low → warn (preserves a soft signal
    without blocking the policy from the API output)."""
    extracted = _make_extracted_policy(
        "P_LOW",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": "Some borderline language about modifiers.",
                "target_codes": ["59510"],
            }
        ],
    )
    judge_llm = StubLLM({
        "relevant": False,
        "confidence": "low",
        "reason": "Borderline — not clearly about the requested codes.",
    })
    state = {
        "result": [extracted],
        "filtered_policies": [_make_filtered_policy("P_LOW", "Molina", "Modifiers Policy")],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": ["Cesarean"],
        "llm_mini": judge_llm,
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    assert contam is not None
    assert "policy_topic_irrelevant" in contam["flags"]
    assert contam["severity"] == "warn"


def test_validate_policies_no_llm_mini_skips_judge(agent):
    """When no llm_mini is provided (e.g. unit tests without the judge),
    the deterministic checks still run and the policy passes if clean."""
    extracted = _make_extracted_policy(
        "P_NO_LLM",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": "Maternity codes adjudicated normally.",
                "target_codes": ["59510"],
            }
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [_make_filtered_policy("P_NO_LLM", "Cigna", "Global Maternity")],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": [],
        # No "llm_mini" key — judge must be skipped, no exception raised.
    }
    out = agent.validate_policies_node(state)
    # No judge → no policy_topic_irrelevant flag → clean policy passes.
    assert out["result"][0].get("contamination") is None


def test_validate_policies_anthem_in_elevance_evidence_not_flagged(agent):
    """Anthem and Elevance canonicalize to the same payer key, so an
    Elevance policy that cites legacy 'Anthem' branding must NOT trip
    cross_payer_name_leak. Regression test for the false positive seen
    in the 2026-06-12 cesarean run."""
    extracted = _make_extracted_policy(
        "P_ANTHEM",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": (
                    "Under the Anthem benefit plan, global maternity bundling "
                    "applies per the published Elevance schedule."
                ),
                "target_codes": ["59510"],
            }
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P_ANTHEM", "Elevance Health (external)", "Maternity Services")
        ],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": [],
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    # No cross-payer flag should fire — Anthem == Elevance canonically.
    assert contam is None or "cross_payer_name_leak" not in (contam or {}).get("flags", [])


def test_validate_policies_no_disclaim_flag_for_drg_without_codes(agent):
    """When the pattern only carries DRG codes and the policy text uses
    CPT-level adjudication (no DRG mention, empty target_codes), the
    validator must NOT fire a contamination flag. The deterministic
    `policy_disclaims_codes` check has been removed — the relevance
    judge is now the sole authority on topical relevance, and without
    an llm_mini the policy default-keeps.

    Regression for the 2026-06-12 cesarean run where 8 of 10 OB
    policies were warn-flagged for this reason, starving the
    recommendation pool."""
    extracted = _make_extracted_policy(
        "P_DRG_ONLY",
        rules=[
            {
                "code": "59510",
                "mention_status": "not_mentioned",
                "payor_level_summary": "Global maternity bundling applies via CPT.",
                "target_codes": [],
            },
            {
                "code": "59514",
                "mention_status": "not_mentioned",
                "payor_level_summary": "Global maternity bundling applies via CPT.",
                "target_codes": [],
            },
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P_DRG_ONLY", "Cigna", "Global Maternity/Obstetric Package")
        ],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": ["Cesarean Section"],
        # No llm_mini — judge skipped, deterministic checks are now the
        # only signal and none of them should fire on this clean policy.
    }
    out = agent.validate_policies_node(state)
    assert out["result"][0].get("contamination") is None


def test_sanitizer_keeps_icd10_codes(agent):
    """ICD-10 diagnosis codes (e.g. O30.001 for multiple gestation,
    C50.911 for breast neoplasm, Z51.11 for chemotherapy encounter)
    must survive the sanitizer — they're legitimately referenced in
    obstetric and oncology policies. Regression for the Multiple Births
    policy in the 2026-06-12 run where every ICD-10 code got dropped."""
    payload = {
        "policy_metadata": {},
        "results": [
            {
                "code": "59510",
                "target_codes": ["59510"],
                "related_codes": [
                    "O30.001", "Z37.50-", "Z38.30-", "C50.911", "Z51.11", "59618",
                ],
                "required_modifiers": [],
                "excluded_modifiers": [],
                "revenue_codes": [],
            }
        ],
    }
    sanitized = agent._sanitize_extracted_facts("PID-MB", "Multiple Births", payload)
    related = sanitized["results"][0]["related_codes"]
    # All ICD-10 codes (OB + oncology) plus the CPT survive.
    for code in ("O30.001", "Z37.50-", "Z38.30-", "C50.911", "Z51.11", "59618"):
        assert code in related, f"{code} should survive sanitizer"


def test_extract_concurrency_resolver_per_env(agent, monkeypatch):
    """The resolver returns the per-env default and respects the
    REIMBURSEMENT_EXTRACT_CONCURRENCY env-var override."""
    from deep_research_agents.reimbursement_agent import AppConstants

    monkeypatch.delenv("REIMBURSEMENT_EXTRACT_CONCURRENCY", raising=False)

    monkeypatch.setattr(AppConstants, "ENV", "dev")
    assert agent._resolve_extract_concurrency() == 3
    monkeypatch.setattr(AppConstants, "ENV", "uat")
    assert agent._resolve_extract_concurrency() == 3
    monkeypatch.setattr(AppConstants, "ENV", "prod")
    assert agent._resolve_extract_concurrency() == 8
    monkeypatch.setattr(AppConstants, "ENV", "local")
    assert agent._resolve_extract_concurrency() == 1
    # Unknown env → 3 fallback.
    monkeypatch.setattr(AppConstants, "ENV", "unknown_env")
    assert agent._resolve_extract_concurrency() == 3

    # Env-var override wins regardless of AppConstants.ENV.
    monkeypatch.setenv("REIMBURSEMENT_EXTRACT_CONCURRENCY", "6")
    monkeypatch.setattr(AppConstants, "ENV", "local")
    assert agent._resolve_extract_concurrency() == 6
    # Garbage override falls back to the per-env default (local=1 here).
    monkeypatch.setenv("REIMBURSEMENT_EXTRACT_CONCURRENCY", "not-an-int")
    assert agent._resolve_extract_concurrency() == 1


def test_validate_policies_drg_prompt_leak_defense_in_depth(agent):
    """If a DRG-prompt string survives the sanitizer (shouldn't, but the
    validator must still catch it), severity=quarantine."""
    extracted = _make_extracted_policy(
        "P3",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": "Add-on coding bundles primary surgical allowance.",
                "specific_rule_text": "",
                # Note: bypass sanitize by setting directly.
                "target_codes": ["DRG Codes: Cesarean Section without Sterilization"],
            }
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P3", "Molina", "Add-on Coding")
        ],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": ["Cesarean Section without Sterilization without CC/MCC"],
    }
    out = agent.validate_policies_node(state)
    contam = out["result"][0].get("contamination")
    assert contam is not None
    assert "drg_prompt_leak_detected" in contam["flags"]
    assert contam["severity"] == "quarantine"


def test_validate_policies_clean_policy_gets_no_contamination(agent):
    """A policy whose payer matches its own evidence and whose title topic
    aligns with the content topic must not be flagged."""
    extracted = _make_extracted_policy(
        "P4",
        rules=[
            {
                "code": "59510",
                "payor_level_summary": (
                    "Cigna global maternity bundles antepartum, delivery, "
                    "and postpartum care under the same group practice."
                ),
                "specific_rule_text": "",
                "target_codes": ["59510", "59514"],
                "pairing_conditions": ["same group provides full care"],
                "program_scope": ["Global Maternity"],
            }
        ],
    )
    state = {
        "result": [extracted],
        "filtered_policies": [
            _make_filtered_policy("P4", "Cigna", "Global Maternity/Obstetric Package")
        ],
        "pattern": {},
        "cpt_codes": "",
        "drg_codes": [],
    }
    out = agent.validate_policies_node(state)
    assert out["result"][0].get("contamination") is None


# ---------------------------------------------------------------------------
# Phase 3c extract_result surfacing
# ---------------------------------------------------------------------------

def test_extract_result_surfaces_contamination_warnings(agent):
    """When policy_contamination_summary has quarantined entries,
    extract_result emits warnings naming policy_id:title pairs and
    surfaces cross_payer_contamination_detected + policy_relevance_pass_rate
    checks. Quarantined policies must NOT appear in reimbursement_policies
    (full drop semantics — they don't reach the UI)."""
    graph_output = {
        "pattern_rank": 5,
        "pattern": {"pattern_title": "Cesarean spend"},
        "result": [{"a": 1}, {"b": 2}, {"c": 3}],
        "formatted_output": {
            "summary_table": {"subtitle": "Cesarean spend rising"},
            # format_output_node has already filtered the quarantined
            # entries out — only the clean policy survives into the API
            # output. The aggregate counts ride on
            # policy_contamination_summary below.
            "individual_policies": [
                {"policy_id": "P3", "payer_name": "Cigna", "policy_title": "Global OB",
                 "contamination": None},
            ],
        },
        "recommended_action": [],
        "cpt_codes": "",
        "drg_codes": [],
        "elevance_executive_summary": None,
        "conversation_id": "demo_001",
        "job_id": "j1",
        "start_time": None,
        "policy_contamination_summary": {
            "total_extracted": 3,
            "quarantined": [
                {"policy_id": "P1", "payer": "Elevance", "title": "Maternity",
                 "flags": ["cross_payer_name_leak"], "reasons": []},
                {"policy_id": "P2", "payer": "Molina", "title": "Readmission",
                 "flags": ["title_content_topic_mismatch"], "reasons": []},
            ],
            "warned": [],
        },
    }

    out = agent.extract_result(graph_output)

    # Quarantined policies removed entirely from the UI-visible output.
    surviving_ids = {p["policy_id"] for p in out["output"]["reimbursement_policies"]}
    assert surviving_ids == {"P3"}
    assert "P1" not in surviving_ids
    assert "P2" not in surviving_ids

    checks = out["validation"]["checks"]
    by_name = {c["check"]: c for c in checks}
    assert "cross_payer_contamination_detected" in by_name
    assert by_name["cross_payer_contamination_detected"]["passed"] is False
    assert "policy_relevance_pass_rate" in by_name
    # 2/3 = 66% > 30% threshold → check fails.
    assert by_name["policy_relevance_pass_rate"]["passed"] is False

    warnings = out["validation"]["warnings"]
    # Every quarantined policy must still be named with policy_id:title
    # so reviewers can trace what was dropped even after UI removal.
    assert any("P1:Maternity" in w for w in warnings)
    assert any("P2:Readmission" in w for w in warnings)


def test_format_output_drops_quarantined_from_individual_policies():
    """End-to-end check that format_output_node strips quarantined
    entries from individual_policies entirely, while keeping clean and
    warn-severity entries."""
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.format_output")
    inst._token_breakdown = {}

    # Build minimal extracted results: one quarantined, one warn, one clean.
    results = [
        {
            "PLCY_ID": "Q1",
            "policy_metadata": {"effective_date": "2024-01-01"},
            "results": [{"code": "59510", "payor_level_summary": "OB"}],
            "contamination": {
                "flags": ["cross_payer_name_leak"],
                "severity": "quarantine",
                "reasons": [],
            },
        },
        {
            "PLCY_ID": "W1",
            "policy_metadata": {"effective_date": "2024-02-01"},
            "results": [{"code": "59510", "payor_level_summary": "OB"}],
            "contamination": {
                "flags": ["empty_target_codes_under_specific_pattern"],
                "severity": "warn",
                "reasons": [],
            },
        },
        {
            "PLCY_ID": "C1",
            "policy_metadata": {"effective_date": "2024-03-01"},
            "results": [{"code": "59510", "payor_level_summary": "OB"}],
            "contamination": None,
        },
    ]
    filtered = [
        {"policy_id": "Q1", "payor": "Molina", "policy_title": "Readmission",
         "external_link": "https://example.com/q1.pdf"},
        {"policy_id": "W1", "payor": "Humana", "policy_title": "Assistant",
         "external_link": "https://example.com/w1.pdf"},
        {"policy_id": "C1", "payor": "Cigna", "policy_title": "Global OB",
         "external_link": "https://example.com/c1.pdf"},
    ]

    state = {
        "result": results,
        "filtered_policies": filtered,
        "table_structure": None,  # forces the simple-table branch — no LLM needed
        "pattern": {"pattern_title": "Cesarean spend"},
        "llm": None,
    }

    out = inst.format_output_node(state)
    surviving_ids = {p["policy_id"] for p in out["formatted_output"]["individual_policies"]}
    assert "Q1" not in surviving_ids
    assert surviving_ids == {"W1", "C1"}


def test_format_output_uses_policy_id_lookup_not_positional_index():
    """Regression for the 2026-06-12 bug: Snowflake's GROUP BY returns
    rows in arbitrary order, so `results[i]` and `filtered_policies[i]`
    don't align. The fix replaces positional indexing with a dict
    lookup by PLCY_ID. This test reverses the order of filtered_policies
    relative to results and asserts each policy's payer/title/URL come
    from the correct row, not from `filtered_policies[i]`."""
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.format_output_index")
    inst._token_breakdown = {}

    # results in order [A, B, C]
    results = [
        {
            "PLCY_ID": "A",
            "policy_metadata": {"effective_date": "2024-01-01"},
            "results": [{"code": "1", "payor_level_summary": "evidence-A"}],
            "contamination": None,
        },
        {
            "PLCY_ID": "B",
            "policy_metadata": {"effective_date": "2024-02-01"},
            "results": [{"code": "2", "payor_level_summary": "evidence-B"}],
            "contamination": None,
        },
        {
            "PLCY_ID": "C",
            "policy_metadata": {"effective_date": "2024-03-01"},
            "results": [{"code": "3", "payor_level_summary": "evidence-C"}],
            "contamination": None,
        },
    ]
    # filtered_policies in REVERSED order [C, B, A] — simulates the
    # real-world misalignment where Snowflake returns rows in a
    # different order than the original DataFrame.
    filtered = [
        {"policy_id": "C", "payor": "Cigna",  "policy_title": "Title-C",
         "external_link": "https://example.com/c.pdf"},
        {"policy_id": "B", "payor": "Humana", "policy_title": "Title-B",
         "external_link": "https://example.com/b.pdf"},
        {"policy_id": "A", "payor": "Aetna",  "policy_title": "Title-A",
         "external_link": "https://example.com/a.pdf"},
    ]

    state = {
        "result": results,
        "filtered_policies": filtered,
        "table_structure": None,
        "pattern": {"pattern_title": "test"},
        "llm": None,
    }

    out = inst.format_output_node(state)
    by_id = {
        p["policy_id"]: p
        for p in out["formatted_output"]["individual_policies"]
    }

    # Each policy must carry its OWN payer/title/URL, not whatever was
    # at the same index in filtered_policies. Pre-fix, A would get
    # Cigna/Title-C/c.pdf (filtered[0]) instead of Aetna/Title-A/a.pdf.
    assert by_id["A"]["payer_name"] == "Aetna"
    assert by_id["A"]["policy_title"] == "Title-A"
    assert by_id["A"]["policy_url"] == "https://example.com/a.pdf"

    assert by_id["B"]["payer_name"] == "Humana"
    assert by_id["B"]["policy_title"] == "Title-B"
    assert by_id["B"]["policy_url"] == "https://example.com/b.pdf"

    assert by_id["C"]["payer_name"] == "Cigna"
    assert by_id["C"]["policy_title"] == "Title-C"
    assert by_id["C"]["policy_url"] == "https://example.com/c.pdf"


def test_extract_rules_retry_uses_llm_retry_not_llm_mini():
    """Regression guard: when the first extract call fails, the retry
    must execute against state['llm_retry'] (gpt-5.4-mini), not
    state['llm_mini'] (gpt-5.4-nano). Running the same weak model twice
    is what caused the original contamination to persist."""
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.extract_retry")
    inst._token_breakdown = {}
    inst.max_retries = 1
    inst.retry_delay = 0  # don't sleep in tests

    called_with: List[Any] = []

    def fake_extract(self, policy_text, pattern_details, llm):
        called_with.append(llm)
        if len(called_with) == 1:
            raise RuntimeError("first attempt fails")
        return {
            "policy_metadata": {},
            "results": [{"code": "59510", "target_codes": ["59510"]}],
        }

    inst._extract_policy_rules = fake_extract.__get__(inst, ReimbursementAgent)

    state = {
        "policy_content": [{"PLCY_ID": "P1", "POLICY_TEXT": "any"}],
        "filtered_policies": [
            {"policy_id": "P1", "payor": "Cigna", "policy_title": "Global OB"}
        ],
        "cpt_codes": "59510",
        "drg_codes": [],
        "pattern": {},
        "llm_mini": "NANO",
        "llm_retry": "MINI",
    }
    out = inst.extract_rules_node(state)

    assert called_with == ["NANO", "MINI"]
    # The retried policy must have made it through.
    assert out["result"][0] is not None
    assert out["result"][0]["PLCY_ID"] == "P1"


def test_extract_result_pattern_title_falls_back_to_subtitle(agent):
    """When pattern has no title fields, explanation.pattern_title resolves
    from summary_table.subtitle rather than literal 'Unknown Pattern'."""
    graph_output = {
        "pattern_rank": 5,
        "pattern": {},  # no pattern_title, top_pattern, or title
        "result": [{"a": 1}],
        "formatted_output": {
            "summary_table": {"subtitle": "California cesarean delivery spend is rising"},
            "individual_policies": [],
        },
        "recommended_action": [],
        "cpt_codes": "",
        "drg_codes": [],
        "elevance_executive_summary": None,
        "conversation_id": None,
        "job_id": "j1",
        "start_time": None,
    }
    out = agent.extract_result(graph_output)
    # In direct mode (conversation_id None) extract_result returns the
    # raw formatted output — we want to inspect the explanation that
    # would be on the orchestrator output, so re-build with a fake convo id.
    graph_output["conversation_id"] = "c1"
    out = agent.extract_result(graph_output)
    assert "California cesarean delivery spend" in out["explanation"]["pattern_title"]
    assert "Unknown Pattern" not in out["explanation"]["pattern_title"]


# ---------------------------------------------------------------------------
# Post-fix coverage: warn-severity recommendation pool + warning_rate check
# ---------------------------------------------------------------------------

def test_generate_recommendation_includes_warn_severity_policies():
    """warn-severity policies must still seed the recommendation peer
    benchmarks. Only quarantine-severity is excluded. Regression for the
    2026-06-12 cesarean run where 8 warn-flagged OB policies were
    excluded, leaving only Humana modifier policies to ground a thin
    assistant-at-surgery edit."""
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.recommendation_pool")
    inst._token_breakdown = {}

    captured_facts: List[Dict[str, Any]] = []

    def fake_generate(
        pattern_context,
        pattern_rank,
        summary_table,
        elevance_summary,
        llm,
        user_query=None,
        specific_facts=None,
    ):
        captured_facts.append(specific_facts or {})
        return []

    inst._generate_policy_recommendations = fake_generate
    # No-op elevance summary helper so the test doesn't try to call an LLM.
    inst._generate_elevance_executive_summary = (
        lambda **kwargs: None
    )

    individual_policies = [
        {
            "policy_id": "Q1",
            "payer_name": "Molina",
            "policy_title": "Readmission",
            "policy_url": "https://example.com/q1.pdf",
            "edit_rule_facts": {
                "target_codes": ["99284"],
                "related_codes": [],
                "required_modifiers": [],
                "excluded_modifiers": [],
                "revenue_codes": [],
                "pairing_conditions": [],
                "utilization_limits": [],
                "prior_auth_thresholds": [],
                "discharge_status_conditions": [],
                "program_scope": [],
                "state_specific_rules": [],
                "provider_role_restrictions": [],
                "exemptions": [],
                "action_types": [],
            },
            "contamination": {
                "flags": ["policy_topic_irrelevant"],
                "severity": "quarantine",
                "reasons": [],
            },
        },
        {
            "policy_id": "W1",
            "payer_name": "Elevance Health (external)",
            "policy_title": "Maternity Services",
            "policy_url": "https://example.com/w1.pdf",
            "edit_rule_facts": {
                "target_codes": [],
                "related_codes": [],
                "required_modifiers": [],
                "excluded_modifiers": [],
                "revenue_codes": [],
                "pairing_conditions": ["professional delivery service claims"],
                "utilization_limits": [],
                "prior_auth_thresholds": [],
                "discharge_status_conditions": [],
                "program_scope": ["DRG"],
                "state_specific_rules": ["California"],
                "provider_role_restrictions": [],
                "exemptions": [],
                "action_types": ["deny"],
            },
            "contamination": {
                "flags": ["cross_payer_name_leak"],
                "severity": "warn",
                "reasons": [],
            },
        },
        {
            "policy_id": "C1",
            "payer_name": "Humana",
            "policy_title": "Assistant at Surgery Reimbursement",
            "policy_url": "https://example.com/c1.pdf",
            "edit_rule_facts": {
                "target_codes": ["59514", "59620"],
                "related_codes": [],
                "required_modifiers": [],
                "excluded_modifiers": [],
                "revenue_codes": [],
                "pairing_conditions": ["billing assistance only during a cesarean delivery"],
                "utilization_limits": [],
                "prior_auth_thresholds": [],
                "discharge_status_conditions": [],
                "program_scope": ["Medicare Advantage", "Commercial"],
                "state_specific_rules": [],
                "provider_role_restrictions": ["Assistant at surgery provided by physician or NPP"],
                "exemptions": [],
                "action_types": ["limit"],
            },
            "contamination": None,
        },
    ]

    state = {
        "pattern": {"pattern_title": "Cesarean spend", "pattern_rank": 5},
        "pattern_rank": 5,
        "formatted_output": {
            "summary_table": {"subtitle": "Cesarean spend rising"},
            "individual_policies": individual_policies,
        },
        "llm": StubLLM({}),  # never called — _generate_policy_recommendations stubbed
        "result": [],
        "drg_codes": ["Cesarean"],
        "search_keywords": "cesarean delivery",
        "query": None,
    }

    inst.generate_recommendation_node(state)

    assert len(captured_facts) == 1
    facts = captured_facts[0]
    per_payer = facts.get("per_payer") or {}
    # Quarantine policy (Q1 / Molina Readmission) MUST be excluded.
    assert "Molina" not in per_payer
    # Warn-severity policy (W1 / Elevance Maternity) MUST be included.
    assert "Elevance Health (external)" in per_payer
    elevance_facts = per_payer["Elevance Health (external)"]
    # The Elevance warn policy's structured facts reached the benchmark pool.
    assert "professional delivery service claims" in elevance_facts.get("pairing_conditions", [])
    assert "California" in elevance_facts.get("state_specific_rules", [])
    # Clean policy (C1 / Humana) is also included.
    assert "Humana" in per_payer


def test_validator_keeps_cpt_level_grounding_for_drg_pattern(agent, pattern):
    """For a DRG-driven pattern (cesarean), the validator must NOT drop
    items whose cited source summary says 'does not reference the DRG
    code' as long as the source adjudicates the topic via CPT codes.
    The carve-out preserves OB-bundling and multiple-birth modifier
    recommendations that ground in professional OB policies."""
    citation_index = {
        "C1": _citation(
            "C1", "Cigna",
            "Global Maternity/Obstetric Package",
            "pairing_conditions", "same group provides full care",
        ),
    }
    individual_policies = [
        {
            "payer_name": "Cigna",
            "policy_title": "Global Maternity/Obstetric Package",
            "evidence": (
                "The requested DRG pattern is not explicitly addressed in this "
                "policy. The policy adjudicates the Global Maternity/Obstetric "
                "Package using specific CPT codes and partial-care/complication "
                "diagnosis-coding rules."
            ),
        },
    ]
    items = [
        {
            "kind": "edit",
            "text": (
                "Bundle global OB delivery (CPT 59510/59514) into a single "
                "claim when the same group provides antepartum, delivery, "
                "and postpartum care."
            ),
            "citations": ["C1"],
            "scope": [],
            "peer_cite": None,
            "exemptions": [],
        },
    ]
    rec = _rec_with_transient(items, citation_index)
    # Validator returns keep — the test asserts the criterion change at
    # the prompt level rather than relying on a real LLM call here.
    payload = {
        "verdicts": [
            {"item_index": 0, "decision": "keep",
             "reason": "policy adjudicates topic via CPT — carve-out applies"},
        ],
        "summary": "all items aligned with pattern",
    }
    llm = StubLLM(payload)

    result = agent.validate_recommendation_node(
        _state([rec], pattern, individual_policies, llm),
    )

    recs = result["recommended_action"]
    assert len(recs) == 1
    surviving = recs[0]
    assert "59510" in surviving["description"] or "59514" in surviving["description"]
    rv = result["recommendation_validation"]
    assert rv["decision"] == "ok"
    assert rv["dropped_count"] == 0
    # The prompt must surface the CPT-vs-DRG carve-out so the LLM applies it.
    assert "CPT codes" in llm.last_prompt or "global packages" in llm.last_prompt
    assert "topic disclaim" in llm.last_prompt.lower()


def test_extract_result_policy_warning_rate_triggers_valid_with_warnings(agent):
    """When > 50% of extracted policies carry warn-severity contamination,
    a new policy_warning_rate check fails and is_valid flips to
    'valid_with_warnings' so consumers surface the caveat."""
    warned_entries = [
        {"policy_id": f"W{i}", "payer": "Cigna", "title": f"OB-{i}",
         "flags": ["cross_payer_name_leak"], "reasons": []}
        for i in range(6)
    ]
    graph_output = {
        "pattern_rank": 5,
        "pattern": {"pattern_title": "Cesarean spend"},
        "result": [{"_": i} for i in range(10)],
        "formatted_output": {
            "summary_table": {"subtitle": "Cesarean spend rising"},
            "individual_policies": [
                {"policy_id": f"W{i}", "payer_name": "Cigna",
                 "policy_title": f"OB-{i}", "contamination": {
                     "flags": ["cross_payer_name_leak"], "severity": "warn",
                     "reasons": [],
                 }}
                for i in range(6)
            ] + [
                {"policy_id": f"C{i}", "payer_name": "Humana",
                 "policy_title": f"Clean-{i}", "contamination": None}
                for i in range(4)
            ],
        },
        "recommended_action": [],
        "cpt_codes": "",
        "drg_codes": ["Cesarean"],
        "elevance_executive_summary": None,
        "conversation_id": "demo_001",
        "job_id": "j1",
        "start_time": None,
        "policy_contamination_summary": {
            "total_extracted": 10,
            "quarantined": [],
            "warned": warned_entries,
        },
    }

    out = agent.extract_result(graph_output)

    by_name = {c["check"]: c for c in out["validation"]["checks"]}
    assert "policy_warning_rate" in by_name
    # 6/10 = 60% > 50% threshold → check fails.
    assert by_name["policy_warning_rate"]["passed"] is False
    assert "6/10" in by_name["policy_warning_rate"]["message"]
    # Failed check flips is_valid to the tri-state warning string.
    assert out["validation"]["is_valid"] == "valid_with_warnings"


def test_extract_result_policy_warning_rate_under_threshold_passes(agent):
    """Symmetric test: when ≤ 50% of policies are warn-flagged, the
    check passes and is_valid stays True. Uses policy_topic_irrelevant
    flag (not cross_payer_name_leak) so the existing
    cross_payer_contamination_detected check doesn't independently fail
    the run."""
    warned_entries = [
        {"policy_id": "W1", "payer": "Cigna", "title": "OB-1",
         "flags": ["policy_topic_irrelevant"], "reasons": []},
    ]
    graph_output = {
        "pattern_rank": 5,
        "pattern": {"pattern_title": "Cesarean spend"},
        "result": [{"_": i} for i in range(10)],
        "formatted_output": {
            "summary_table": {"subtitle": "Cesarean spend rising"},
            "individual_policies": [],
        },
        "recommended_action": [],
        "cpt_codes": "",
        "drg_codes": ["Cesarean"],
        "elevance_executive_summary": None,
        "conversation_id": "demo_001",
        "job_id": "j1",
        "start_time": None,
        "policy_contamination_summary": {
            "total_extracted": 10,
            "quarantined": [],
            "warned": warned_entries,
        },
    }

    out = agent.extract_result(graph_output)

    by_name = {c["check"]: c for c in out["validation"]["checks"]}
    assert by_name["policy_warning_rate"]["passed"] is True
    assert out["validation"]["is_valid"] is True
