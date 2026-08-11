import pytest
from typing import Dict, Any

from deep_research_agents import PreferredProviderAgent


def test_preferred_provider_agent_stub_mode():
    agent = PreferredProviderAgent(test_mode=True)

    result: Dict[str, Any] = agent(
        question="Which providers are preferred in CA?",
        context={"state": "CA", "period": "recent"},
        correlation_summary={},
    )

    assert isinstance(result, dict)
    assert "providers_ranked" in result
    assert "executive_summary" in result
    assert "metadata" in result and result["metadata"].get("is_stub") is True
    assert isinstance(result.get("warnings", []), list)

    providers = result.get("providers_ranked", [])
    assert isinstance(providers, list)
    if providers:
        first = providers[0]
        assert {"provider_id", "name", "score", "metrics"}.issubset(first.keys())
