"""Agent implementations for deep-research project.

Each agent is imported defensively so a missing module (e.g. a not-yet-built
agent on a feature branch) does not break the whole package — callers can ask
which agents are actually available via :func:`available_agents` and which
failed to load via :func:`missing_agents`.
"""

import logging
from importlib import import_module
from typing import Dict, List

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


_AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    "ReimbursementPolicyAgent": {
        "module": "deep_research_agents.reimbursement_agent",
        "attr": "ReimbursementPolicyAgent",
    },
    "PatternAgent": {
        "module": "deep_research_agents.pattern_agent",
        "attr": "PatternAgent",
    },
    "PreferredProviderAgent": {
        "module": "deep_research_agents.preferred_provider_agent",
        "attr": "PreferredProviderAgent",
    },
    "ContractAgent": {
        "module": "deep_research_agents.contract_agent",
        "attr": "ContractAgent",
    },
}


_LOADED: Dict[str, object] = {}
_MISSING: Dict[str, str] = {}

for export_name, spec in _AGENT_REGISTRY.items():
    try:
        module = import_module(spec["module"])
        _LOADED[export_name] = getattr(module, spec["attr"])
    except Exception as exc:  # ImportError, AttributeError, etc.
        _MISSING[export_name] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "deep_research_agents: %s not loaded (%s: %s)",
            export_name,
            type(exc).__name__,
            exc,
        )

globals().update(_LOADED)

if "ReimbursementPolicyAgent" in _LOADED:
    ReimbursementAgent = _LOADED["ReimbursementPolicyAgent"]
    _LOADED["ReimbursementAgent"] = ReimbursementAgent


def available_agents() -> List[str]:
    """Return the names of agent classes that successfully imported."""
    return sorted(_LOADED.keys())


def missing_agents() -> Dict[str, str]:
    """Return ``{agent_name: error_message}`` for agents that failed to load."""
    return dict(_MISSING)


__all__ = sorted(set(list(_LOADED.keys()) + ["available_agents", "missing_agents"]))
