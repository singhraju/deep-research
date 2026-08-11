# Healthcare Agents - Root Level Instructions

## Overview
Root-level guidance for AI agents working on the `idiscovery-deep-research` project — a LangGraph-orchestrated healthcare analytics platform (Streamlit UI + FastAPI backend + Snowflake/EHAP data plane).

## Project Structure
- **apps/ui/**: Streamlit UI (`apps/ui/src/app.py`) and orchestrator client
- **apps/ETL/**: Snowpark-based batch pipeline (`apps/ETL/dr_etl_pipeline.py`)
- **packages/agents/**: LangGraph agent implementations + FastAPI service (`packages/agents/src/agent_api.py`, `reimbursement_api.py`)
- **packages/core/**: Shared framework — `AgentBase`, `AgentAPIBuilder`, app exceptions, LLM client
- **packages/utils/**: Helpers — EHAP client, Snowflake helper, semantic view, vault config, logging, cache
- **configs/**: Environment INIs (`dev/local/uat/prod/local_offshore`) and semantic-view YAMLs under `configs/correlation_pattern/`
- **docs/**: `design/` (architecture), `api_builder_guide.md`, `semantic_view_utilities.md`
- **tests/**: Root-level tests for shared components (`test_api_builder.py`, `test_semantic_view.py`). Agent-specific tests live in `packages/agents/tests/`.

## Key Principles
1. **Documentation First**: Review relevant docs in `docs/` and the nearest `AGENTS.md` before making changes
2. **Preserve Functionality**: Never remove existing functionality without explicit permission
3. **Do not create extra helper, Debug & test files**: Do not create any new .md file or any debug or test files unless asked. 
4. **Update Documentation**: Keep docs synchronized with code changes
5. *Do not use Emojis*: Do not use emojis when writing the code.
6. *Do not Delete anything from Redis*: Don't write any command or function to delete the data from Redis cache.

## Verify Changes With Tests
**Every code change must be followed by a test run.** Locate the nearest `tests/` directory to what you changed and run it:

| What you edited | Test command (from project root) |
|---|---|
| `packages/agents/src/...` | `uv run pytest packages/agents/tests/ -v` |
| `packages/core/src/...` | `uv run pytest tests/test_api_builder.py -v` |
| `packages/utils/src/...` | `uv run pytest tests/test_semantic_view.py -v` (plus any package-specific tests) |
| `apps/ui/src/...` | `uv run pytest packages/agents/tests/ -v` (UI consumes agent contracts) |
| `apps/ETL/...` | No automated test suite — run `python apps/ETL/dr_etl_pipeline.py` smoke check if env is configured |
| Shared changes (multiple packages) | `uv run pytest tests/ packages/agents/tests/ -v` |

Rules:
- Run tests **after** the edit, not just before. A passing pre-edit baseline does not prove the change is safe.
- If a test fails because of your change, fix it before reporting completion. If it was already failing on `master`, call that out explicitly.
- If the test suite needs credentials (Snowflake / EHAP) that aren't available in the current environment, say so — don't claim success on the basis of a skipped or import-failing test.
- For standalone demonstration scripts (e.g., `test_reimbursement_complete.py`) that perform live API calls, note that they were not executed when reporting the task; they're not part of the automated suite.

## Before Starting Work
0. Activate the project Python environment: `source .venv/bin/activate` (create it first with `uv venv .venv` if missing). Python `>=3.13` is required (see root `pyproject.toml`).
1. Review the relevant documentation in `docs/`
2. Check the `AGENTS.md` file in the specific package you're working on (`packages/agents/`, `packages/core/`, `packages/utils/`)
3. Understand the architectural design in `docs/design/`

## Running the App
- **Streamlit UI only**: `streamlit run apps/ui/src/app.py` (optionally `--server.headless=true`)
- **FastAPI agent service**: `uvicorn packages.agents.src.agent_api:app --reload --host 0.0.0.0 --port 8000`
- **Both servers together**: `./start_servers.sh` (FastAPI on :8000, Streamlit on :8501)

## Environment Configuration
- Use `--native-tls` with `uv` on the corporate network, or set `UV_NATIVE_TLS=1` in `.env`, to handle corporate TLS certificates.
- See `README.MD` for the full list of required env vars (EHAP, Snowflake, feature flags).

## File Access Restrictions
**DO NOT read or reference the following files:**
- `.env` — contains sensitive credentials and secrets
- Any files with passwords, API keys, or credentials in their names

## Common Tasks
- Agent / API changes: see `packages/agents/AGENTS.md`
- UI changes: see `apps/ui/README.md` (no dedicated AGENTS.md yet)
- ETL pipeline changes: edit `apps/ETL/dr_etl_pipeline.py` and helpers in `apps/ETL/utils/`
- Core framework changes: see `packages/core/AGENTS.md`
- Utility changes: see `packages/utils/AGENTS.md`

## Agents in this Repo
Located in `packages/agents/src/deep_research_agents/`:
- `correlation_agent.py` (plus `correlation_interaction_matrix.py`, `correlation_recommendation.py`)
- `reimbursement_agent.py` (surfaced via `packages/agents/src/reimbursement_api.py`)
- `policy_hypothesis_agent.py`, `mandate_hypothesis_agent.py`
- `pattern_agent.py`, `preferred_provider_agent.py`, `recommendation_agent.py`
- `orchestrator.py` — composes agents via LangGraph

## Agent Change Checklist
When modifying any agent under `packages/agents/src/deep_research_agents/`, also check:
1. **Agent implementation**: `prepare_state`, `node_function`, `extract_result`, `build_app`
2. **API schema**: `AgentAPIBuilder` (in `packages/core/src/deep_research_core/api_builder.py`) uses `prepare_state`'s signature to build request models
3. **Orchestrator integration**: handlers and metadata in `orchestrator.py`
4. **UI consumers**: `apps/ui/src/app.py` and `apps/ui/src/ui/orchestrator_client.py`
5. **Tests**: `packages/agents/tests/` (and root `tests/` for shared components)
6. **Configs/docs**: relevant YAMLs under `configs/correlation_pattern/` and any docs in `docs/`

**Note:** Do not update older example notebooks (including date-based naming conventions) unless explicitly requested.
