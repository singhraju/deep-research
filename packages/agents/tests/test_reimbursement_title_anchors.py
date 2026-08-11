"""
Unit tests for the title-anchor reduction pipeline in
`ReimbursementPolicyAgent`. Two moving parts under test:

1. `_generate_title_anchors_from_cpt` — LLM helper that returns 2-5
   title-substring anchors for filtering policies by topical relevance
   to the CPT/HCPCS codes.
2. The post-fetch title-regex + (payor, policy_title) dedup step inside
   `search_policies_node`. That's covered by exercising the raw
   dataframe transformation logic against a synthetic payload shaped
   like a real Carelon `sentencelist` response.

Motivation: the Carelon policy_comparison API does exact-string matching
on the full policy TEXT, not just the title, so a keyword like "injection"
returns hundreds of Reimbursement policies that mention "injection" once
in prose. A client-side title-regex + (payor, title) dedup cuts the
result set by 12-100× while preserving all topically-relevant Elevance
policies — measured in scripts/test_policy_api_reduction.py.

Agent instances are built via `__new__` to bypass Snowflake / mini-LLM init
(matches the pattern used by `test_reimbursement_cpt_keywords.py`).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from deep_research_agents import ReimbursementAgent
from deep_research_agents.reimbursement_agent import _extract_anchor_words


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class StubAnchorLLM:
    """Records prompts and returns a canned anchor payload.

    The prompt asks for one anchor per line; the stub emits the raw
    string as-is so tests can exercise the parser edge cases
    (bullets, numbering, quotes, code-fenced blocks, empties).
    """

    def __init__(
        self,
        payload: str = "Ambulance\nTransportation Services\nNon-Emergent Transport",
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.payload = payload
        self.raise_exc = raise_exc
        self.invocations: List[List[Dict[str, str]]] = []

    def invoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> _StubResponse:
        self.invocations.append(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _StubResponse(self.payload)


@pytest.fixture
def agent() -> ReimbursementAgent:
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.reimbursement_title_anchors")
    inst._token_breakdown = {}
    return inst


# ----------------------------------------------------------------------------
# _generate_title_anchors_from_cpt — happy path & parsing
# ----------------------------------------------------------------------------


def test_anchors_happy_path(agent: ReimbursementAgent) -> None:
    """LLM returns 3 anchors, one per line. All 3 must survive."""
    llm = StubAnchorLLM(payload="Ambulance\nTransportation Services\nNon-Emergent Transport")
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == ["Ambulance", "Transportation Services", "Non-Emergent Transport"]
    # LLM must actually be called once with the code in the prompt.
    assert len(llm.invocations) == 1
    prompt = llm.invocations[0][-1]["content"]
    assert "A0427" in prompt


def test_anchors_multi_code_prompt_carries_all_codes(agent: ReimbursementAgent) -> None:
    """When multiple codes are passed, the prompt must include all of them so
    the LLM can pick anchors that cover the union of codes."""
    llm = StubAnchorLLM(payload="Critical Care\nEvaluation and Management")
    anchors = agent._generate_title_anchors_from_cpt("99291,99292", llm)
    assert anchors == ["Critical Care", "Evaluation and Management"]
    prompt = llm.invocations[0][-1]["content"]
    assert "99291" in prompt and "99292" in prompt


def test_anchors_strips_bullets_numbering_and_quotes(agent: ReimbursementAgent) -> None:
    """LLMs sometimes ignore 'no bullets' instructions. Verify we strip:
    - leading dash/asterisk/bullet marks
    - numbered prefixes ('1.', '2)')
    - surrounding quotes and whitespace"""
    payload = "- Ambulance\n* Transportation Services\n1. Non-Emergent Transport\n2) \"Medical Transport\"\n  \n"
    llm = StubAnchorLLM(payload=payload)
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == [
        "Ambulance",
        "Transportation Services",
        "Non-Emergent Transport",
        "Medical Transport",
    ]


def test_anchors_strips_code_fence(agent: ReimbursementAgent) -> None:
    """Some LLMs wrap the answer in ```-fenced blocks. The unwrapping
    must leave the anchor list intact."""
    payload = "```\nAmbulance\nTransportation Services\n```"
    llm = StubAnchorLLM(payload=payload)
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == ["Ambulance", "Transportation Services"]


def test_anchors_deduped_case_insensitively(agent: ReimbursementAgent) -> None:
    """LLM sometimes emits close variants ('Ambulance' + 'ambulance'). Only
    the first must survive; downstream regex is case-insensitive anyway."""
    llm = StubAnchorLLM(payload="Ambulance\nambulance\nAMBULANCE\nTransportation Services")
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == ["Ambulance", "Transportation Services"]


def test_anchors_capped_at_five(agent: ReimbursementAgent) -> None:
    """The prompt asks for 2-5 anchors. If the LLM returns 8, only the first
    5 are kept — more anchors just widen the OR-match into false-positive
    territory (see excluded_patterns_notes in the taxonomy)."""
    payload = "\n".join(f"Anchor {i}" for i in range(8))
    llm = StubAnchorLLM(payload=payload)
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == [f"Anchor {i}" for i in range(5)]


def test_anchors_drops_lines_with_forbidden_chars(agent: ReimbursementAgent) -> None:
    """Guard against LLM emitting JSON/objects/arrays we'd have to parse."""
    payload = "Ambulance\n{invalid: json}\n[list, of, things]\nTransportation Services"
    llm = StubAnchorLLM(payload=payload)
    anchors = agent._generate_title_anchors_from_cpt("A0427", llm)
    assert anchors == ["Ambulance", "Transportation Services"]


# ----------------------------------------------------------------------------
# _generate_title_anchors_from_cpt — failure & degradation
# ----------------------------------------------------------------------------


def test_anchors_empty_input_returns_empty(agent: ReimbursementAgent) -> None:
    """Empty code string → no LLM call, empty result."""
    llm = StubAnchorLLM(payload="anything")
    assert agent._generate_title_anchors_from_cpt("", llm) == []
    assert llm.invocations == []


def test_anchors_none_llm_returns_empty(agent: ReimbursementAgent) -> None:
    """Missing LLM (offline / stubbed tests) → skip anchor step gracefully."""
    assert agent._generate_title_anchors_from_cpt("A0427", None) == []


def test_anchors_llm_error_returns_empty(agent: ReimbursementAgent) -> None:
    """LLM exception → empty list, caller must skip title-regex step but
    dedup still runs. Legacy behavior preserved."""
    llm = StubAnchorLLM(raise_exc=RuntimeError("boom"))
    assert agent._generate_title_anchors_from_cpt("A0427", llm) == []


def test_anchors_empty_llm_response_returns_empty(agent: ReimbursementAgent) -> None:
    """LLM returns only blank/bullet lines → empty list."""
    llm = StubAnchorLLM(payload="\n\n  \n- \n* \n")
    assert agent._generate_title_anchors_from_cpt("A0427", llm) == []


# ----------------------------------------------------------------------------
# Title-regex + dedup pipeline — reproduces the reduction step
# search_policies_node performs after PDF/policy_type/dedup and before
# _apply_policy_caps. This does NOT go through the full node (which needs
# Snowflake / requests / etc), it reproduces the transformation on a
# synthetic dataframe shaped like the sentencelist API response.
# ----------------------------------------------------------------------------


def _synth_row(
    policy_id: str,
    payor: str,
    title: str,
    score: float = 100.0,
) -> Dict[str, Any]:
    return {
        "policy_id": policy_id,
        "payor": payor,
        "policy_title": title,
        "policy_type": "Reimbursement",
        "policy_score": score,
        "external_link": f"https://example.com/{policy_id}.pdf",
        "policy_link": f"https://example.com/{policy_id}.pdf",
    }


def _apply_title_reduction(
    df: pd.DataFrame, title_anchors: List[str]
) -> pd.DataFrame:
    """Exact same transformation the node applies. Kept in the test so the
    node body and this test can be edited independently — if the two
    diverge, the node's behavior wins in production and the test starts
    catching it as a bug."""
    if title_anchors:
        usable = [a.strip() for a in title_anchors if a and a.strip()]
        if usable:
            pattern_str = "|".join(re.escape(a) for a in usable)
            title_re = re.compile(pattern_str, re.IGNORECASE)
            before = len(df)
            titles = df["policy_title"].fillna("")
            filtered = df[titles.apply(lambda t: bool(title_re.search(t)))]
            if len(filtered) == 0:
                # All-zero safety valve: keep pre-filter df.
                pass
            else:
                drop_pct = (before - len(filtered)) / before if before > 0 else 0.0
                if drop_pct > 0.80 and len(filtered) < 30:
                    words = _extract_anchor_words(usable)
                    fallback_applied = False
                    if words:
                        word_pattern = "|".join(rf"\b{re.escape(w)}\b" for w in words)
                        word_re = re.compile(word_pattern, re.IGNORECASE)
                        fallback = df[titles.apply(lambda t: bool(word_re.search(t)))]
                        if len(fallback) > len(filtered):
                            df = fallback
                            fallback_applied = True
                    if not fallback_applied:
                        df = filtered
                else:
                    df = filtered
    if "policy_title" in df.columns and len(df) > 0:
        if "policy_score" in df.columns:
            sort_scores = pd.to_numeric(df["policy_score"], errors="coerce").fillna(0)
            df = df.assign(_sort_score=sort_scores).sort_values(
                by="_sort_score", ascending=False, kind="mergesort"
            ).drop(columns=["_sort_score"])
        df = df.drop_duplicates(subset=["payor", "policy_title"], keep="first")
    return df


def test_pipeline_drops_off_topic_titles() -> None:
    """The `Ambulance` anchor must keep the 3 ambulance rows and drop the
    4 off-topic rows (Modifier Usage, Preadmission, ED Leveling, Incident to)."""
    df = pd.DataFrame([
        _synth_row("elv-1", "Elevance Health (external)", "Transportation Services: Ambulance"),
        _synth_row("elv-2", "Elevance Health (external)", "Ambulance Transportation"),
        _synth_row("elv-3", "Elevance Health (external)", "Ambulance Reimbursement Policy"),
        _synth_row("elv-4", "Elevance Health (external)", "Modifier Usage"),
        _synth_row("elv-5", "Elevance Health (external)", "Preadmission Services for Inpatient Stays"),
        _synth_row("elv-6", "Elevance Health (external)", "Emergency Department Leveling of E&M"),
        _synth_row("uhc-1", "United Health", "Incident to Services and Billing"),
    ])
    out = _apply_title_reduction(df, ["Ambulance", "Transportation Services"])
    titles = set(out["policy_title"])
    # All 3 ambulance titles survive; all 4 off-topic titles are dropped.
    assert titles == {
        "Transportation Services: Ambulance",
        "Ambulance Transportation",
        "Ambulance Reimbursement Policy",
    }
    assert len(out) == 3


def test_pipeline_dedup_collapses_state_fanout() -> None:
    """Elevance publishes identical titles per state — 43 distinct
    policy_ids for one title. Dedup by (payor, title) must collapse
    them to one representative row per (payor, title)."""
    df = pd.DataFrame([
        _synth_row(f"elv-{i}", "Elevance Health (external)",
                   "Transportation Services: Ambulance and Non-Emergent Transport",
                   score=100.0 + i)  # different score per state so keep="first" is meaningful
        for i in range(10)
    ] + [
        _synth_row("uhc-1", "United Health", "Ambulance Transportation"),
    ])
    out = _apply_title_reduction(df, ["Ambulance", "Transportation"])
    assert len(out) == 2
    # Elevance row must be the highest-score state (score=109 → policy_id "elv-9").
    elv_rows = out[out["payor"] == "Elevance Health (external)"]
    assert len(elv_rows) == 1
    assert elv_rows.iloc[0]["policy_id"] == "elv-9"


def test_pipeline_empty_anchors_skips_regex_but_still_dedups() -> None:
    """Empty anchor list = skip title-regex step (legacy behavior). Dedup
    still runs so state fanout still collapses."""
    df = pd.DataFrame([
        _synth_row("elv-1", "Elevance Health (external)", "Title A", score=100.0),
        _synth_row("elv-2", "Elevance Health (external)", "Title A", score=50.0),
        _synth_row("elv-3", "Elevance Health (external)", "Title B", score=75.0),
    ])
    out = _apply_title_reduction(df, [])
    assert len(out) == 2
    # Kept the higher-scored row for Title A (elv-1) via score-desc sort.
    a_rows = out[out["policy_title"] == "Title A"]
    assert a_rows.iloc[0]["policy_id"] == "elv-1"


def test_pipeline_case_insensitive_and_regex_escapes_special_chars() -> None:
    """User-visible category names sometimes contain regex metacharacters
    ('E/M', 'C+ therapy'). re.escape() must neutralize them, and the
    match must be case-insensitive."""
    df = pd.DataFrame([
        _synth_row("elv-1", "Elevance Health (external)", "E/M Services for Emergency Department"),
        _synth_row("elv-2", "Elevance Health (external)", "Preventive Care"),
    ])
    out = _apply_title_reduction(df, ["e/m services"])  # lowercase input, /'/'
    assert set(out["policy_title"]) == {"E/M Services for Emergency Department"}


def test_pipeline_empty_dataframe_returns_empty() -> None:
    """No rows in → no rows out. Must not crash on the sort/dedup path
    when the frame is empty."""
    empty = pd.DataFrame(columns=["policy_id", "payor", "policy_title", "policy_score", "policy_type"])
    out = _apply_title_reduction(empty, ["Ambulance"])
    assert len(out) == 0


def test_pipeline_handles_missing_and_non_numeric_score() -> None:
    """`policy_score` may be missing or come back as a string. Dedup must
    still work — non-numeric scores coerce to 0, so first-occurrence wins."""
    df = pd.DataFrame([
        {**_synth_row("elv-1", "Elevance Health (external)", "Ambulance"), "policy_score": "not-a-number"},
        {**_synth_row("elv-2", "Elevance Health (external)", "Ambulance"), "policy_score": None},
        {**_synth_row("elv-3", "Elevance Health (external)", "Ambulance"), "policy_score": 200.0},
    ])
    out = _apply_title_reduction(df, ["Ambulance"])
    assert len(out) == 1
    # 200.0 beat the non-numeric and None scores.
    assert out.iloc[0]["policy_id"] == "elv-3"


def test_pipeline_zero_match_anchors_reverts_to_prefilter() -> None:
    """Regression: Pattern 3 in the 2026-07-07 UI run generated anchors
    like 'Ambulance Transportation Services' that were LONGER than the
    real 'Ambulance Transportation' policy title, so the substring filter
    dropped all 262 policies to 0 and downstream _get_policy_hashes([])
    crashed on 'WHERE PLCY_ID IN ()'. Fallback: if the filter drops
    everything, revert to the pre-filter df so LLM triage still runs."""
    df = pd.DataFrame([
        _synth_row("elv-1", "Elevance Health (external)", "Ambulance Transportation"),
        _synth_row("elv-2", "Elevance Health (external)", "Air Ambulance"),
        _synth_row("uhc-1", "United Health", "Ambulance Services"),
    ])
    # These anchors are LONGER than any real title above — none will match.
    over_specific = [
        "Ambulance Transportation Services",
        "Emergency Ground Ambulance",
        "Non-Emergent Ambulance",
    ]
    out = _apply_title_reduction(df, over_specific)
    # Must not be empty (downstream would crash) — pre-filter df survives.
    assert len(out) > 0
    # Original 3 rows collapse to 3 unique (payor, title) pairs.
    assert len(out) == 3


def test_get_policy_hashes_empty_list_returns_empty_df() -> None:
    """Regression: _get_policy_hashes([]) previously built
    'WHERE PLCY_ID IN ()' which is invalid SQL and crashes the node.
    Must short-circuit to an empty DataFrame with the right columns."""
    agent = ReimbursementAgent.__new__(ReimbursementAgent)
    agent.logger = logging.getLogger("test.get_policy_hashes_empty")

    class _ShouldNotBeCalled:
        def execute_query_and_return_pandas_df(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            raise AssertionError("snowflake_helper must not be called when policy_ids is empty")

    agent.snowflake_helper = _ShouldNotBeCalled()

    out = agent._get_policy_hashes([])
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert set(out.columns) == {"PLCY_ID", "PDF_HASH_VAL_ID"}


# ----------------------------------------------------------------------------
# _extract_anchor_words — direct unit tests
# ----------------------------------------------------------------------------


def test_extract_anchor_words_dedup_and_stopwords() -> None:
    """Regression: Pattern 2 in the 2026-07-08 UI run generated anchors
    ['Ambulance Transportation', 'Ground Ambulance', 'Non Emergent
    Ambulance', 'Emergency Ambulance']. The fallback must split on
    whitespace/hyphens, drop 'Non' as a stopword, dedup 'Ambulance', and
    keep the rest so a word-level OR-match can pick up 'Ambulance
    Services' and similar titles that the strict multi-word regex missed."""
    result = _extract_anchor_words([
        "Ambulance Transportation",
        "Ground Ambulance",
        "Non-Emergent Ambulance",
        "Emergency Ambulance",
    ])
    # Ambulance appears in 4 anchors → collapsed to 1. 'Non' dropped as
    # stopword. Preserve first-seen casing.
    assert result == ["Ambulance", "Transportation", "Ground", "Emergent", "Emergency"]


def test_extract_anchor_words_case_insensitive_dedup_preserves_first_casing() -> None:
    """LLM might emit 'ambulance' and 'Ambulance' as separate parts. Dedup
    must be case-insensitive but preserve the first-seen casing (used in
    log lines for debuggability)."""
    result = _extract_anchor_words(["Ambulance Transport", "ambulance care"])
    assert result == ["Ambulance", "Transport", "care"]


def test_extract_anchor_words_filters_short_tokens_and_empty_anchors() -> None:
    """Words shorter than 3 chars are dropped (initials like 'A0' or noise
    like 'of', 'a', 'e'). Empty or whitespace-only anchors are skipped."""
    result = _extract_anchor_words(["A0 Ambulance", "   ", "IV Injection Administration"])
    # 'A0' (2 chars) and 'IV' (2 chars) both dropped. Rest kept.
    assert result == ["Ambulance", "Injection", "Administration"]


# ----------------------------------------------------------------------------
# Word-level fallback — end-to-end through _apply_title_reduction
# ----------------------------------------------------------------------------


def test_word_level_fallback_triggers_on_aggressive_drop() -> None:
    """Regression: strict multi-word anchors like 'Ambulance Transportation'
    matched only 3 of 100 candidate policies (97% drop). Fallback splits
    into words and OR-matches on word boundaries, surfacing many more
    ambulance-related titles that the strict regex missed."""
    # 3 rows match the strict anchor "Ambulance Transportation" or
    # "Ground Ambulance". 40 rows contain "Ambulance" alone.
    strict_matches = [
        _synth_row("elv-1", "Elevance Health (external)", "Ambulance Transportation"),
        _synth_row("elv-2", "Elevance Health (external)", "Ground Ambulance Services"),
        _synth_row("uhc-1", "United Health", "Emergency Ambulance"),  # matches "Emergency Ambulance"
    ]
    # These are on-topic ambulance policies with titles the strict regex misses.
    word_matches = [
        _synth_row(f"amb-{i}", f"Payor{i}", f"Ambulance Services Policy {i}")
        for i in range(20)
    ] + [
        _synth_row(f"trans-{i}", f"PayorT{i}", f"Transportation Reimbursement {i}")
        for i in range(20)
    ]
    # Off-topic rows that neither strict nor word-level anchors should match.
    off_topic = [
        _synth_row(f"off-{i}", f"PayorO{i}", f"Modifier Usage Policy {i}")
        for i in range(40)
    ]
    df = pd.DataFrame(strict_matches + word_matches + off_topic)

    anchors = [
        "Ambulance Transportation",
        "Ground Ambulance",
        "Non-Emergent Ambulance",
        "Emergency Ambulance",
    ]
    out = _apply_title_reduction(df, anchors)
    # Fallback must trigger: strict kept 3 (<30, drop >80%). Word-level match
    # on {Ambulance, Transportation, Ground, Emergent, Emergency} picks up
    # all 3 strict rows + 40 word-match rows = 43. After (payor,title) dedup
    # every row has a unique title, so the count survives.
    assert len(out) == 43
    titles = set(out["policy_title"])
    # All strict-match titles present.
    assert "Ambulance Transportation" in titles
    assert "Ground Ambulance Services" in titles
    # Off-topic titles must NOT leak through.
    assert not any("Modifier Usage" in t for t in titles)


def test_word_level_fallback_skipped_when_drop_is_moderate() -> None:
    """Strict filter keeping ≥30 rows must not trigger the fallback, even
    when the drop percentage is >80%. Guards against fallback pulling in
    off-topic titles when the strict regex is already doing its job."""
    # 30 rows match strictly ("Ambulance" is a substring of every title).
    strict_matches = [
        _synth_row(f"strict-{i}", f"Payor{i}", f"Ambulance Title {i}")
        for i in range(30)
    ]
    # 200 unrelated rows (drop_pct = 200/230 = 87% > 80% BUT filtered=30 not <30).
    off_topic = [
        _synth_row(f"off-{i}", f"PayorO{i}", f"Modifier Usage {i}")
        for i in range(200)
    ]
    df = pd.DataFrame(strict_matches + off_topic)
    out = _apply_title_reduction(df, ["Ambulance"])
    # Fallback must NOT trigger (30 rows survived, threshold is <30). Strict
    # result stands: only the 30 Ambulance titles.
    assert len(out) == 30
    assert all("Ambulance" in t for t in out["policy_title"])


def test_word_level_fallback_no_op_when_words_dont_help() -> None:
    """If the strict anchors already contain only unique/high-signal words
    (like 'Electroencephalogram') and no additional titles match the
    word-level regex, keep the strict result."""
    df = pd.DataFrame([
        _synth_row("elv-1", "Elevance Health (external)", "Electroencephalogram Policy"),
    ] + [
        _synth_row(f"off-{i}", f"PayorO{i}", f"Modifier Usage {i}")
        for i in range(50)
    ])
    # Strict keeps 1 out of 51 rows (drop 98%, count <30 → fallback triggered).
    # Word-level split of "Electroencephalogram" is still just
    # ["Electroencephalogram"] — no new matches. Must keep strict.
    out = _apply_title_reduction(df, ["Electroencephalogram"])
    assert len(out) == 1
    assert out.iloc[0]["policy_title"] == "Electroencephalogram Policy"
