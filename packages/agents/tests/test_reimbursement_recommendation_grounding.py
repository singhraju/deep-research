"""
Unit tests for citation-grounded recommendation generation in
ReimbursementAgent. These cover the Option B grounding contract:

  1. `_collect_payer_edit_rule_facts` emits a `citations` list binding each
     verbatim fact to its source (payer, policy_title, policy_url).
  2. `_format_facts_for_prompt` returns a citation block tagged with stable
     ids that the LLM is required to cite.
  3. `_generate_policy_recommendations` drops items whose `citations` are
     unknown or absent; if every item drops, the recommendation is
     suppressed entirely (returns []).
  4. Surviving items have `evidence` strings rebuilt from the resolved
     citations — not from `exemptions` / `scope` like the legacy path.

The recommendation method is exercised with a stub LLM so we don't hit any
real model. The agent is instantiated via `__new__` so we skip Snowflake /
mini-LLM init (those are unrelated to the grounding contract under test).
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
        # _record_tokens is defensive — missing usage metadata is logged
        # at debug and silently ignored, so we don't bother populating it.


class StubLLM:
    """Records the prompt it was sent and returns a canned JSON response."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.last_prompt: str = ""

    def invoke(self, messages: List[Dict[str, str]]) -> _StubResponse:
        self.last_prompt = messages[-1]["content"] if messages else ""
        return _StubResponse(json.dumps(self._payload))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> ReimbursementAgent:
    """Bare ReimbursementAgent — bypasses Snowflake/mini-LLM init."""
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.reimbursement_grounding")
    inst._token_breakdown = {}
    return inst


@pytest.fixture
def policies_by_payer() -> Dict[str, List[Dict[str, Any]]]:
    """Two payers, two policies, with overlapping facts so the pivot
    actually produces peer benchmarks and the citation index covers both
    distinct sources for the same fact ("at least 8 hours")."""
    return {
        "United Health": [
            {
                "payer_name": "United Health",
                "policy_title": "Outpatient Hospital Observation Policy, Facility",
                "policy_url": "https://uhc.example.com/obs.pdf",
                "edit_rule_facts": {
                    "revenue_codes": ["0762"],
                    "utilization_limits": ["at least 8 hours"],
                    "state_specific_rules": ["KY: does not require minimum 8 hours"],
                    "action_types": ["allow_with_conditions"],
                },
                "results": [],
            }
        ],
        "Elevance Health": [
            {
                "payer_name": "Elevance Health",
                "policy_title": "Observation Services",
                "policy_url": "https://anthem.example.com/obs.pdf",
                "edit_rule_facts": {
                    "utilization_limits": ["at least 8 hours"],
                    "action_types": ["bundle"],
                },
                "results": [],
            }
        ],
    }


@pytest.fixture
def pattern_context() -> Dict[str, Any]:
    return {
        "top_pattern": "ED visit 99284 – KY Commercial gap",
        "evidence_summary": "Kentucky observation rules differ from baseline.",
    }


@pytest.fixture
def summary_table() -> Dict[str, Any]:
    return {
        "title": "Payer Policy Summary",
        "subtitle": "ED visit 99284",
        "columns": [
            {"id": "payer_org", "label": "Payer", "type": "text"},
        ],
        "rows": [
            {"payer_org": "United Health"},
            {"payer_org": "Elevance Health"},
        ],
    }


# ---------------------------------------------------------------------------
# 1) Citation index
# ---------------------------------------------------------------------------

def test_collect_facts_produces_citation_index(
    agent: ReimbursementAgent,
    policies_by_payer: Dict[str, List[Dict[str, Any]]],
):
    facts = agent._collect_payer_edit_rule_facts(policies_by_payer)

    assert "citations" in facts
    citations = facts["citations"]
    # 4 distinct (payer, policy, key, fact) tuples across the two payers:
    #   UHC: revenue_codes/0762, utilization_limits/at least 8 hours,
    #        state_specific_rules/KY exemption
    #   Elevance: utilization_limits/at least 8 hours
    assert len(citations) == 4

    # Every citation carries the four grounding fields.
    for c in citations:
        assert set(c).issuperset({"id", "payer", "policy_title", "policy_url", "fact_key", "fact"})
        assert c["id"].startswith("C")

    # Peer-benchmark entries gain citation_ids covering BOTH source policies
    # that mention the shared fact (one per payer).
    peers = facts["peer_benchmarks"]
    shared = [p for p in peers if p["fact"] == "at least 8 hours"]
    assert shared, "expected the duplicated fact to appear as a peer benchmark"
    assert len(shared[0]["citation_ids"]) == 2


def test_format_facts_renders_citation_block(
    agent: ReimbursementAgent,
    policies_by_payer: Dict[str, List[Dict[str, Any]]],
):
    facts = agent._collect_payer_edit_rule_facts(policies_by_payer)
    per_payer_block, peer_block, citation_block = agent._format_facts_for_prompt(facts)

    assert citation_block, "citation block should be non-empty when citations exist"
    assert "Available citations" in citation_block
    # Every citation id from the index should appear in the rendered block.
    for c in facts["citations"]:
        assert f"[{c['id']}]" in citation_block
    # Peer block should also include the citation_ids suffix.
    assert "citations:" in peer_block


# ---------------------------------------------------------------------------
# 2) Grounding enforcement in _generate_policy_recommendations
# ---------------------------------------------------------------------------

def _run_recommendation(
    agent: ReimbursementAgent,
    payload: Dict[str, Any],
    pattern_context: Dict[str, Any],
    summary_table: Dict[str, Any],
    specific_facts: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return agent._generate_policy_recommendations(
        pattern_context=pattern_context,
        pattern_rank=1,
        summary_table=summary_table,
        elevance_summary=None,
        llm=StubLLM(payload),
        user_query=None,
        specific_facts=specific_facts,
    )


def test_items_with_valid_citations_are_kept(
    agent: ReimbursementAgent,
    policies_by_payer: Dict[str, List[Dict[str, Any]]],
    pattern_context: Dict[str, Any],
    summary_table: Dict[str, Any],
):
    facts = agent._collect_payer_edit_rule_facts(policies_by_payer)
    # Pick a real citation id from the index so the parse step keeps the item.
    real_cid = facts["citations"][0]["id"]

    payload = {
        "has_recommendation": True,
        "recommendation": {
            "headline": "Adopt observation guardrails for KY Commercial.",
            "items": [
                {
                    "kind": "edit",
                    "text": "When G0378 is billed with 99284, require revenue code 0762.",
                    "citations": [real_cid],
                    "scope": [],
                    "peer_cite": "UHC aligns 0762 with observation services.",
                    "exemptions": [],
                }
            ],
        },
    }

    recs = _run_recommendation(agent, payload, pattern_context, summary_table, facts)

    assert len(recs) == 1
    rec = recs[0]
    # Evidence must be rebuilt from the resolved citation, not from exemptions.
    assert rec["evidence"], "evidence should reflect the resolved citation"
    cited = facts["citations"][0]
    assert any(cited["fact"] in line and cited["payer"] in line for line in rec["evidence"])
    # Citation list carries the source policy + URL for UI rendering.
    assert any(cited["policy_url"] in c for c in rec["citation"])
    # peer_benchmarking still carries the LLM's peer prose verbatim.
    assert "UHC aligns 0762" in rec["peer_benchmarking"][0]


def test_items_with_unknown_citations_are_dropped(
    agent: ReimbursementAgent,
    policies_by_payer: Dict[str, List[Dict[str, Any]]],
    pattern_context: Dict[str, Any],
    summary_table: Dict[str, Any],
):
    facts = agent._collect_payer_edit_rule_facts(policies_by_payer)
    real_cid = facts["citations"][0]["id"]

    payload = {
        "has_recommendation": True,
        "recommendation": {
            "headline": "Tighten observation edits.",
            "items": [
                {
                    "kind": "edit",
                    "text": "Grounded item — references a real citation.",
                    "citations": [real_cid],
                    "scope": [],
                    "peer_cite": None,
                    "exemptions": [],
                },
                {
                    "kind": "edit",
                    "text": "Ungrounded item — invents a citation id.",
                    "citations": ["C999"],
                    "scope": [],
                    "peer_cite": None,
                    "exemptions": [],
                },
                {
                    "kind": "immediate_action",
                    "text": "Also ungrounded — empty citations list.",
                    "citations": [],
                    "scope": [],
                    "peer_cite": None,
                    "exemptions": [],
                },
            ],
        },
    }

    recs = _run_recommendation(agent, payload, pattern_context, summary_table, facts)

    assert len(recs) == 1
    # Only the grounded T1 line survives; ungrounded items must not appear.
    assert "Grounded item" in recs[0]["description"]
    assert "Ungrounded item" not in recs[0]["description"]
    assert "Also ungrounded" not in recs[0]["description"]


def test_all_items_ungrounded_suppresses_recommendation(
    agent: ReimbursementAgent,
    policies_by_payer: Dict[str, List[Dict[str, Any]]],
    pattern_context: Dict[str, Any],
    summary_table: Dict[str, Any],
):
    facts = agent._collect_payer_edit_rule_facts(policies_by_payer)

    payload = {
        "has_recommendation": True,
        "recommendation": {
            "headline": "Should be suppressed.",
            "items": [
                {
                    "kind": "edit",
                    "text": "No real citations.",
                    "citations": ["C42"],
                    "scope": [],
                    "peer_cite": None,
                    "exemptions": [],
                },
                {
                    "kind": "edit",
                    "text": "Also no citations.",
                    "citations": [],
                    "scope": [],
                    "peer_cite": None,
                    "exemptions": [],
                },
            ],
        },
    }

    recs = _run_recommendation(agent, payload, pattern_context, summary_table, facts)

    assert recs == [], (
        "when no items are grounded, the recommendation must be suppressed "
        "rather than emitted with a misleading evidence array"
    )
