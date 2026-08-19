"""Core libraries for deep-research project."""

__version__ = "0.1.0"

# Export main classes
from deep_research_core.base_agent import AgentBase, AgentError, CredentialProvider
from deep_research_core.api_models import (
    AgentResponse,
    AgentErrorResponse,
    AgentInfo,
    AgentsListResponse,
    HealthResponse,
    MetricsResponse,
)

__all__ = [
    "AgentBase",
    "AgentError",
    "CredentialProvider",
    "AgentResponse",
    "AgentErrorResponse",
    "AgentInfo",
    "AgentsListResponse",
    "HealthResponse",
    "MetricsResponse",
]

try:
    from deep_research_core.api_builder import AgentAPIBuilder  # type: ignore
    __all__.append("AgentAPIBuilder")
except ModuleNotFoundError:
    AgentAPIBuilder = None  # type: ignore
