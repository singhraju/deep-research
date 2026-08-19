# Core Package - Agent Instructions

## Overview
`deep-research-core` is the shared framework that every agent and API in this project builds on. It defines the agent base class, the auto-API builder, the request/response schemas, and the LLM client.

## Package Structure
```
packages/core/
├── src/
│   └── deep_research_core/
│       ├── __init__.py        # public exports
│       ├── base_agent.py      # AgentBase, AgentError, CredentialProvider
│       ├── api_builder.py     # AgentAPIBuilder (FastAPI auto-generation)
│       ├── api_models.py      # AgentResponse, HealthResponse, MetricsResponse, etc.
│       ├── app_exceptions.py  # Domain-specific exceptions
│       ├── data_types.py      # Shared TypedDicts / dataclasses
│       └── llm.py             # LLM client wrapper used by agents
├── pyproject.toml
├── README.md
└── AGENTS.md (this file)
```

## Key Concepts
- **`AgentBase`** — every agent extends this. Subclasses implement `prepare_state`, `node_function`, `extract_result`, and `build_app`. `AgentAPIBuilder` introspects `prepare_state`'s signature to derive the request schema, so its parameter names and type hints are public API.
- **`AgentAPIBuilder`** — turns an `AgentBase` subclass into a FastAPI app with `/invoke`, `/health`, `/metrics`, and `/agents` endpoints. See `docs/api_builder_guide.md`.
- **`api_models`** — Pydantic models used by both the FastAPI surface and downstream consumers.
- **`llm.py`** — the shared LLM entry point; agents should call through it rather than instantiating clients directly.

## Key Principles
1. **Shared Functionality**: Code here must be reusable across multiple agents and apps
2. **No Domain-Specific Logic**: Healthcare/agent-specific behavior belongs in `packages/agents/`
3. **Stable Public API**: Changes to `AgentBase` or `api_models` are breaking changes — bump versions and update consumers
4. **Type-Hinted**: Public APIs require accurate type hints (the API builder reads them)

## Development Guidelines
- Add new modules under `src/deep_research_core/`
- Export new public symbols from `__init__.py`
- When changing `AgentBase` method signatures or `api_models`, update every agent in `packages/agents/` and the UI client in `apps/ui/src/ui/orchestrator_client.py`
- Update dependencies with `uv add --package deep-research-core <dependency>`
- Tests for shared components live in the root `tests/` directory (e.g., `tests/test_api_builder.py`)

## Common Tasks
- **Add a new exception type**: edit `app_exceptions.py`, export from `__init__.py`
- **Add a new shared response model**: edit `api_models.py`, export from `__init__.py`, regenerate consumers
- **Modify the LLM client**: edit `llm.py`; verify every agent's `node_function` still works
- **Run tests**: `uv run pytest tests/test_api_builder.py` from the project root