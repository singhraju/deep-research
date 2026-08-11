"""
Unit tests for the CPT/HCPCS keyword-generation path in
`ReimbursementPolicyAgent._generate_search_keywords_from_cpt` and its
dispatcher `_generate_search_keywords`.

Motivation: the Carelon policy comparison API does exact-string matching on
document text, so searching with just a CPT/HCPCS code (e.g. `A0427`) drops
narrative payer Reimbursement policies that only describe rules by clinical
topic. The dispatcher must now enrich the raw code with an LLM-generated
category keyword so the API sees `<code>,<category>` — mirroring the DRG
pattern that already existed.

Agent instances are built via `__new__` to bypass Snowflake / mini-LLM init
(matches the pattern used by `test_reimbursement_policy_capping.py`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest

from deep_research_agents import ReimbursementAgent


class _StubResponse:
    """LangChain-style response object with a `.content` string."""

    def __init__(self, content: str) -> None:
        self.content = content


class StubKeywordLLM:
    """Records prompts and returns a canned keyword. Optionally raises."""

    def __init__(
        self,
        keyword: str = "ambulance",
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.keyword = keyword
        self.raise_exc = raise_exc
        self.invocations: List[List[Dict[str, str]]] = []

    def invoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> _StubResponse:
        self.invocations.append(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _StubResponse(self.keyword)


@pytest.fixture
def agent() -> ReimbursementAgent:
    inst = ReimbursementAgent.__new__(ReimbursementAgent)
    inst.logger = logging.getLogger("test.reimbursement_cpt_keywords")
    inst._token_breakdown = {}
    return inst


def test_from_cpt_combines_code_and_llm_keyword(agent: ReimbursementAgent) -> None:
    """Happy path: `<first_code>,<llm_keyword>`."""
    llm = StubKeywordLLM(keyword="ambulance")
    result = agent._generate_search_keywords_from_cpt("A0427", llm)
    assert result == "A0427,ambulance"
    assert len(llm.invocations) == 1
    prompt_text = llm.invocations[0][-1]["content"]
    assert "A0427" in prompt_text
    assert "CATEGORY" in prompt_text


def test_from_cpt_uses_first_code_when_multiple(agent: ReimbursementAgent) -> None:
    """Multi-code input keeps first as the code prefix; LLM sees full list."""
    llm = StubKeywordLLM(keyword="critical care")
    result = agent._generate_search_keywords_from_cpt("99291,99292", llm)
    assert result == "99291,critical care"
    prompt_text = llm.invocations[0][-1]["content"]
    assert "99291" in prompt_text and "99292" in prompt_text


def test_from_cpt_strips_quotes_and_extra_whitespace(agent: ReimbursementAgent) -> None:
    """Some LLMs return quoted strings; we must strip them."""
    llm = StubKeywordLLM(keyword='"behavioral health"\n')
    result = agent._generate_search_keywords_from_cpt("90837", llm)
    assert result == "90837,behavioral health"


def test_from_cpt_takes_only_first_when_llm_returns_comma_list(
    agent: ReimbursementAgent,
) -> None:
    """DRG helper accepts 2 keywords; CPT helper expects exactly 1 category.
    A stray comma from the LLM should not smuggle a second keyword through."""
    llm = StubKeywordLLM(keyword="ambulance, transportation")
    result = agent._generate_search_keywords_from_cpt("A0427", llm)
    # Only the first token becomes the keyword; the trailing comma-joined
    # value is dropped so the API sees exactly two parts (code + category).
    assert result == "A0427,ambulance"


def test_from_cpt_llm_failure_falls_back_to_first_code(
    agent: ReimbursementAgent,
) -> None:
    """On any LLM error, fall back to the raw first code (legacy behavior)."""
    llm = StubKeywordLLM(raise_exc=RuntimeError("boom"))
    result = agent._generate_search_keywords_from_cpt("A0427,A0428", llm)
    assert result == "A0427"


def test_from_cpt_empty_llm_response_falls_back_to_first_code(
    agent: ReimbursementAgent,
) -> None:
    """Empty LLM output shouldn't produce a `<code>,` trailing-comma keyword."""
    llm = StubKeywordLLM(keyword="")
    result = agent._generate_search_keywords_from_cpt("99291", llm)
    assert result == "99291"


def test_from_cpt_empty_input_returns_empty(agent: ReimbursementAgent) -> None:
    """No codes → empty string (no LLM call, no crash)."""
    llm = StubKeywordLLM(keyword="anything")
    result = agent._generate_search_keywords_from_cpt("", llm)
    assert result == ""
    assert llm.invocations == []


def test_dispatcher_routes_cpt_to_llm_helper_when_no_drg(
    agent: ReimbursementAgent,
) -> None:
    """`_generate_search_keywords` must call the new CPT helper when there
    are no DRG codes AND an LLM is available."""
    llm = StubKeywordLLM(keyword="ambulance")
    result = agent._generate_search_keywords(
        drg_codes=[], cpt_codes="A0427", llm=llm
    )
    assert result == "A0427,ambulance"
    assert len(llm.invocations) == 1


def test_dispatcher_preserves_raw_code_fallback_when_no_llm(
    agent: ReimbursementAgent,
) -> None:
    """Legacy behavior for callers that don't provide an LLM (offline/tests):
    still return the raw first code so downstream nodes work."""
    result = agent._generate_search_keywords(
        drg_codes=[], cpt_codes="A0427,A0428", llm=None
    )
    assert result == "A0427"


def test_dispatcher_still_prefers_drg_over_cpt(agent: ReimbursementAgent) -> None:
    """When BOTH DRG codes and CPT codes are present, the DRG helper still
    takes priority — CPT enrichment is a fallback, not a replacement."""
    llm = StubKeywordLLM(keyword="chemotherapy, bowel surgery")
    result = agent._generate_search_keywords(
        drg_codes=["Chemotherapy without malignant leukemia"],
        cpt_codes="A0427",
        llm=llm,
    )
    # DRG helper returns up to 2 comma-joined keywords; CPT helper isn't
    # called at all when DRG codes are present.
    assert result == "chemotherapy, bowel surgery"


def test_dispatcher_returns_none_when_no_codes(agent: ReimbursementAgent) -> None:
    """No DRG and no CPT → None (caller signals insufficient search context)."""
    llm = StubKeywordLLM(keyword="anything")
    result = agent._generate_search_keywords(
        drg_codes=[], cpt_codes="", llm=llm
    )
    assert result is None


def test_dispatcher_prefers_llm_mini_for_cpt_path(agent: ReimbursementAgent) -> None:
    """Cost optimization: CPT keyword generation is a small classification
    task that nano handles reliably. When both `llm` and `llm_mini` are
    provided, the CPT path must route through `llm_mini` and leave the main
    `llm` untouched."""
    main_llm = StubKeywordLLM(keyword="should-not-be-called")
    mini_llm = StubKeywordLLM(keyword="ambulance")
    result = agent._generate_search_keywords(
        drg_codes=[], cpt_codes="A0427", llm=main_llm, llm_mini=mini_llm
    )
    assert result == "A0427,ambulance"
    # The mini LLM did the work; the main LLM was never invoked.
    assert len(mini_llm.invocations) == 1
    assert main_llm.invocations == []


def test_dispatcher_falls_back_to_main_llm_when_mini_absent(
    agent: ReimbursementAgent,
) -> None:
    """When llm_mini isn't wired (e.g. in tests / partial init), the CPT
    path must fall back to the main LLM rather than degrading to raw code."""
    main_llm = StubKeywordLLM(keyword="ambulance")
    result = agent._generate_search_keywords(
        drg_codes=[], cpt_codes="A0427", llm=main_llm, llm_mini=None
    )
    assert result == "A0427,ambulance"
    assert len(main_llm.invocations) == 1


def test_dispatcher_drg_path_still_uses_main_llm_not_mini(
    agent: ReimbursementAgent,
) -> None:
    """DRG keyword generation stays on the main tier — the DRG prompt asks
    for top-2 themes across possibly-long descriptions, which benefits from
    more reasoning than nano provides. This regression-guards the tier split."""
    main_llm = StubKeywordLLM(keyword="chemotherapy, bowel surgery")
    mini_llm = StubKeywordLLM(keyword="should-not-be-called")
    result = agent._generate_search_keywords(
        drg_codes=["Chemotherapy without malignant leukemia"],
        cpt_codes="A0427",
        llm=main_llm,
        llm_mini=mini_llm,
    )
    assert result == "chemotherapy, bowel surgery"
    assert len(main_llm.invocations) == 1
    assert mini_llm.invocations == []
