"""
Unit tests for the policy-cap stage at the tail of search_policies_node.

Covers `_apply_policy_caps` end-to-end and `_triage_policy_titles` in isolation.
The agent is built via `__new__` to bypass Snowflake / mini-LLM init.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from deep_research_agents import ReimbursementAgent
from deep_research_agents.reimbursement_agent import (
    POLICY_CAP_MAX_PER_PAYOR,
    POLICY_CAP_MAX_TOTAL,
)


# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class StubTriageLLM:
    """Returns a canned triage payload and records prompts."""

    def __init__(self, selected_policy_ids: Optional[List[str]] = None, raise_exc: Optional[Exception] = None) -> None:
        self.selected_policy_ids = selected_policy_ids
        self.raise_exc = raise_exc
        self.invocations: List[List[Dict[str, str]]] = []

    def invoke(self, messages: List[Dict[str, str]]) -> _StubResponse:
        self.invocations.append(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _StubResponse(json.dumps({"selected_policy_ids": self.selected_policy_ids or []}))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> ReimbursementAgent:
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.reimbursement_capping")
    inst._token_breakdown = {}
    return inst


def _make_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with the columns _apply_policy_caps expects."""
    df = pd.DataFrame(rows)
    for col in ("policy_id", "payor", "policy_title"):
        if col not in df.columns:
            df[col] = ""
    return df


def _policies(payor: str, count: int, prefix: str, title_prefix: str = "title") -> List[Dict[str, Any]]:
    return [
        {"policy_id": f"{prefix}_{i:03d}", "payor": payor, "policy_title": f"{title_prefix} {i}"}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Tests — balanced / skewed cases (no Elevance)
# ---------------------------------------------------------------------------


def test_balanced_case_caps_to_total_with_per_payor_limit(agent: ReimbursementAgent) -> None:
    """10 payors × 20 each = 200 rows. Expect total ≤ 30, no payor > 5."""
    rows: List[Dict[str, Any]] = []
    for p in range(10):
        rows.extend(_policies(f"Payor_{p}", 20, prefix=f"P{p}"))
    df = _make_df(rows)

    # Stub triage returns the first 25 ids (the budget after Elevance=0).
    capped_other_pool = df.groupby("payor", sort=False, group_keys=False).head(POLICY_CAP_MAX_PER_PAYOR)
    selected = capped_other_pool["policy_id"].tolist()[: POLICY_CAP_MAX_TOTAL]
    llm = StubTriageLLM(selected_policy_ids=selected)

    out = agent._apply_policy_caps(
        df=df, pattern_rank=1, pattern_details="DRG Codes: 470", llm_mini=llm
    )

    assert len(out) == POLICY_CAP_MAX_TOTAL
    assert out.groupby("payor").size().max() <= POLICY_CAP_MAX_PER_PAYOR
    assert llm.invocations, "Triage LLM should be called when other_df exceeds budget"


def test_skewed_case_caps_dominant_payor(agent: ReimbursementAgent) -> None:
    """One payor has 150 rows, others have 10. After per-payor cap none can exceed 5."""
    rows: List[Dict[str, Any]] = []
    rows.extend(_policies("Dominant", 150, prefix="DOM"))
    for p in range(5):
        rows.extend(_policies(f"Small_{p}", 10, prefix=f"S{p}"))
    df = _make_df(rows)

    capped_other_pool = df.groupby("payor", sort=False, group_keys=False).head(POLICY_CAP_MAX_PER_PAYOR)
    llm = StubTriageLLM(selected_policy_ids=capped_other_pool["policy_id"].tolist())

    out = agent._apply_policy_caps(
        df=df, pattern_rank=2, pattern_details="DRG Codes: 470", llm_mini=llm
    )

    assert out.groupby("payor").size().max() <= POLICY_CAP_MAX_PER_PAYOR
    assert len(out) <= POLICY_CAP_MAX_TOTAL


# ---------------------------------------------------------------------------
# Tests — Elevance pinning
# ---------------------------------------------------------------------------


def test_elevance_policies_always_retained(agent: ReimbursementAgent) -> None:
    """3 Elevance + 100 non-Elevance — all 3 Elevance ids must survive."""
    elevance_rows = _policies("Elevance Health (external)", 3, prefix="ELV")
    other_rows: List[Dict[str, Any]] = []
    for p in range(5):
        other_rows.extend(_policies(f"Other_{p}", 20, prefix=f"O{p}"))
    df = _make_df(elevance_rows + other_rows)

    other_df = df[~df["payor"].str.contains("Elevance", case=False, na=False)]
    capped = other_df.groupby("payor", sort=False, group_keys=False).head(POLICY_CAP_MAX_PER_PAYOR)
    llm = StubTriageLLM(selected_policy_ids=capped["policy_id"].tolist())

    out = agent._apply_policy_caps(
        df=df, pattern_rank=3, pattern_details="ctx", llm_mini=llm
    )

    elevance_ids = {r["policy_id"] for r in elevance_rows}
    assert elevance_ids.issubset(set(out["policy_id"])), "All Elevance policies must be pinned"
    assert len(out) <= POLICY_CAP_MAX_TOTAL


def test_elevance_under_global_pressure_pins_first(agent: ReimbursementAgent) -> None:
    """5 Elevance + 200 others. Elevance keeps its 5; other budget = 25; total = 30."""
    elevance_rows = _policies("Elevance Health (external)", 5, prefix="ELV")
    other_rows: List[Dict[str, Any]] = []
    for p in range(10):
        other_rows.extend(_policies(f"Other_{p}", 20, prefix=f"O{p}"))
    df = _make_df(elevance_rows + other_rows)

    other_df = df[~df["payor"].str.contains("Elevance", case=False, na=False)]
    capped = other_df.groupby("payor", sort=False, group_keys=False).head(POLICY_CAP_MAX_PER_PAYOR)
    llm = StubTriageLLM(selected_policy_ids=capped["policy_id"].tolist())

    out = agent._apply_policy_caps(
        df=df, pattern_rank=4, pattern_details="ctx", llm_mini=llm
    )

    surviving_elevance = out[out["payor"].str.contains("Elevance", case=False, na=False)]
    assert len(surviving_elevance) == 5
    assert len(out) == POLICY_CAP_MAX_TOTAL


# ---------------------------------------------------------------------------
# Tests — triage fallback behavior
# ---------------------------------------------------------------------------


def test_triage_failure_falls_back_to_api_order(agent: ReimbursementAgent) -> None:
    """When the triage LLM raises, the cap still produces a bounded set."""
    rows: List[Dict[str, Any]] = []
    for p in range(10):
        rows.extend(_policies(f"Payor_{p}", 20, prefix=f"P{p}"))
    df = _make_df(rows)

    llm = StubTriageLLM(raise_exc=RuntimeError("transient LLM error"))

    out = agent._apply_policy_caps(
        df=df, pattern_rank=5, pattern_details="ctx", llm_mini=llm
    )

    assert len(out) <= POLICY_CAP_MAX_TOTAL
    assert out.groupby("payor").size().max() <= POLICY_CAP_MAX_PER_PAYOR


def test_triage_malformed_json_falls_back(agent: ReimbursementAgent) -> None:
    """Malformed JSON from triage LLM also lands on the API-order fallback."""

    class BadJSONLLM:
        def invoke(self, messages: List[Dict[str, str]]) -> _StubResponse:
            return _StubResponse("not-json {{")

    rows: List[Dict[str, Any]] = []
    for p in range(10):
        rows.extend(_policies(f"Payor_{p}", 20, prefix=f"P{p}"))
    df = _make_df(rows)

    out = agent._apply_policy_caps(
        df=df, pattern_rank=6, pattern_details="ctx", llm_mini=BadJSONLLM()
    )

    assert len(out) <= POLICY_CAP_MAX_TOTAL


def test_triage_empty_selection_falls_back(agent: ReimbursementAgent) -> None:
    """An empty selected_policy_ids list is treated as failure (don't drop everything)."""
    rows: List[Dict[str, Any]] = []
    for p in range(10):
        rows.extend(_policies(f"Payor_{p}", 20, prefix=f"P{p}"))
    df = _make_df(rows)

    llm = StubTriageLLM(selected_policy_ids=[])
    out = agent._apply_policy_caps(
        df=df, pattern_rank=7, pattern_details="ctx", llm_mini=llm
    )

    assert len(out) > 0
    assert len(out) <= POLICY_CAP_MAX_TOTAL


# ---------------------------------------------------------------------------
# Tests — under-cap no-op
# ---------------------------------------------------------------------------


def test_under_cap_no_triage_call_made(agent: ReimbursementAgent) -> None:
    """15 rows across 4 payors — under both caps; triage must not be invoked."""
    rows: List[Dict[str, Any]] = []
    for p in range(4):
        rows.extend(_policies(f"Payor_{p}", 3, prefix=f"P{p}"))
    rows.extend(_policies("Payor_4", 3, prefix="P4"))  # total = 15
    df = _make_df(rows)

    llm = StubTriageLLM(selected_policy_ids=["should_not_be_used"])
    out = agent._apply_policy_caps(
        df=df, pattern_rank=8, pattern_details="ctx", llm_mini=llm
    )

    assert len(out) == 15
    assert llm.invocations == [], "Triage should not be called when capped set fits the budget"


def test_no_llm_mini_skips_triage_uses_api_order(agent: ReimbursementAgent) -> None:
    """If llm_mini is None, triage is skipped and per-payor + global caps still apply."""
    rows: List[Dict[str, Any]] = []
    for p in range(10):
        rows.extend(_policies(f"Payor_{p}", 20, prefix=f"P{p}"))
    df = _make_df(rows)

    out = agent._apply_policy_caps(
        df=df, pattern_rank=9, pattern_details="ctx", llm_mini=None
    )

    assert len(out) <= POLICY_CAP_MAX_TOTAL
    assert out.groupby("payor").size().max() <= POLICY_CAP_MAX_PER_PAYOR


def test_empty_input_returns_empty(agent: ReimbursementAgent) -> None:
    """An empty DataFrame should pass through untouched (no triage, no error)."""
    df = _make_df([])
    llm = StubTriageLLM(selected_policy_ids=["x"])
    out = agent._apply_policy_caps(
        df=df, pattern_rank=10, pattern_details="ctx", llm_mini=llm
    )
    assert len(out) == 0
    assert llm.invocations == []
