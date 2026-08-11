# Agents Package - Agent Instructions

## Overview
This package contains specialized agent implementations for the deep-research project, built on top of the core framework.

## Package Structure
```
packages/agents/
├── src/
│   └── deep_research_agents/
│       ├── __init__.py
│       ├── reimbursement_agent.py
│       ├── correlation_agent.py
│       ├── policy_hypothesis_agent.py
│       ├── mandate_hypothesis_agent.py
│       ├── recommendation_agent.py
│       ├── orchestrator.py
│       └── user_intent.py
├── tests/
│   ├── __init__.py
│   ├── test_correlation_agent_execution.py
│   ├── test_correlation_agent_narrative.py
│   └── test_reimbursement_complete.py
├── pyproject.toml
├── README.md
└── AGENTS.md (this file)
```

## Key Principles
1. **Agent Specialization**: Each agent has a specific domain focus
2. **Core Framework**: All agents extend AgentBase from deep-research-core
3. **Composability**: Agents can be orchestrated together
4. **Well-Tested**: Agent implementations should have comprehensive test coverage

## Development Guidelines
- Add new agents under `src/deep_research_agents/`
- All agents should extend `AgentBase` from `deep-research-core`
- Update dependencies in `pyproject.toml`
- Write tests in the `tests/` directory within this package
- Keep backward compatibility in mind for all changes

## Testing
This package has its own test suite located in the `tests/` directory.

### Running Tests
From the project root:
```bash
# Run all agent tests
pytest packages/agents/tests/

# Run specific test file
pytest packages/agents/tests/test_correlation_agent_execution.py

# Run with verbose output
pytest packages/agents/tests/ -v
```

From the agents package directory:
```bash
cd packages/agents
pytest
```

### Writing Tests
- Place test files in `tests/` directory
- Follow naming convention: `test_*.py`
- Use pytest fixtures and markers as needed
- Mock external dependencies (e.g., Snowflake connections)

### Standalone Test Scripts
Some test files are designed to be run as standalone scripts (not with pytest):
- `test_reimbursement_complete.py` - Comprehensive reimbursement agent demonstration script

Run standalone scripts with:
```bash
python packages/agents/tests/test_reimbursement_complete.py
```

**Note:** These scripts perform live API calls and are intended for manual testing and demonstration purposes.

## Common Tasks
- Adding new agents: Create new Python files under `src/deep_research_agents/`
- Adding dependencies: Use `uv add --package deep-research-agents <dependency>`
- Adding test dependencies: Use `uv add --package deep-research-agents --dev <dependency>`
- Running tests: `pytest packages/agents/tests/` from project root or `pytest` from package directory

## Dependencies
This package depends on:
- `deep-research-core`: Base agent framework and API builder
- `deep-research-utils`: Utility functions and helpers

## Agent Change Checklist
When modifying any agent in this package, also review:
1. **Agent class methods**: `prepare_state`, `node_function`, `extract_result`, `build_app`.
2. **API request/response schema**: `packages/core/src/deep_research_core/api_builder.py` auto-generates request models from `prepare_state`.
3. **Orchestrator wiring**: `packages/agents/src/deep_research_agents/orchestrator.py` handlers, metadata, and correlation integration.
4. **UI consumers**: `apps/ui/src/app.py` and `apps/ui/src/ui/orchestrator_client.py`.
5. **Tests & examples**: `packages/agents/tests/`, notebooks, and `DR_ECAP_agents.postman_collection.json`.
6. **Docs/configs**: `configs/` defaults and any relevant docs.

**Note:** Do not update older example notebooks (including date-based naming conventions) unless explicitly requested.
