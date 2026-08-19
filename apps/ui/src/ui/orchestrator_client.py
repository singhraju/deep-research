"""Helpers for building and running the deep-research orchestrator in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import os

from deep_research_agents.orchestrator import build_app
from deep_research_agents.user_intent import build_llm
from deep_research_utils.app_constant import AppConstants

try:
    from deep_research_utils.snowflake_helper import SnowparkHelper
except ImportError:  # pragma: no cover - optional dependency
    SnowparkHelper = None  # type: ignore[assignment]

LLM_ENV_VARS = [
    "EHAP_BASE_URL",
    "EHAP_CLIENT_ID",
    "EHAP_CLIENT_SECRET",
    "EHAP_LLM_MODEL",
    "DEEP_RESEARCH_LLM_MODEL",
]
SNOWFLAKE_ENV_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_SECRET",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
]


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for building the orchestrator app."""

    yaml_path: str
    enable_llm: bool
    enable_snowflake: bool
    correlation_output_root: str


def describe_environment_requirements() -> Dict[str, List[str]]:
    """Return environment variables needed for optional runtime integrations."""

    return {
        "llm": list(LLM_ENV_VARS),
        "snowflake": list(SNOWFLAKE_ENV_VARS),
    }


@lru_cache(maxsize=4)
def build_orchestrator(config: OrchestratorConfig) -> Callable[..., Dict[str, Any]]:
    """Build and cache the orchestrator app based on the provided config."""

    output_root = Path(config.correlation_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    llm_builder = build_llm if config.enable_llm else None
    snowflake_helper = _build_snowflake_helper_from_env() if config.enable_snowflake else None

    return build_app(
        yaml_path=config.yaml_path,
        llm_builder=llm_builder,
        snowflake_helper=snowflake_helper,
        correlation_output_root=str(output_root),
        enable_correlation_execution=config.enable_snowflake,
    )


def _build_snowflake_helper_from_env() -> Any:
    if SnowparkHelper is None:
        raise RuntimeError("SnowparkHelper is unavailable; install deep-research-utils to enable Snowflake execution.")

    from deep_research_core.base_agent import CredentialProvider


    # Use CredentialProvider for auto-detection
    creds = CredentialProvider.get_instance()
    snowflake_creds = creds.get_snowflake_credentials()
    
    return SnowparkHelper(
        batch_size=10000,
        max_workers=6,
        enable_metrics=True,
        connection_pool_size=4,
        **snowflake_creds
    )
