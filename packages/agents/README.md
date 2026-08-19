# deep-research-agents

Agent implementations for the deep-research project.

## Overview
This package contains specialized agent implementations built on top of the deep-research-core framework, including:

- **ReimbursementAgent**: Healthcare policy extraction agent
- **CorrelationAgent**: Data correlation and analysis agent
- **PolicyHypothesisAgent**: Policy hypothesis generation agent
- **MandateHypothesisAgent**: Mandate hypothesis generation agent
- **RecommendationAgent**: Recommendation generation agent
- **Orchestrator**: Multi-agent orchestration system
- **UserIntent**: User intent parsing and resolution
- **PatternAgent**: Risk pattern extraction agent for deep-dive reports

## Installation
This package is part of the deep-research monorepo and is managed using uv workspaces.

## Quick Start

### Using the Reimbursement Agent

```python
from deep_research_agents import ReimbursementAgent

agent = ReimbursementAgent(agent_name="reimbursement")
result = agent(question="What are the reimbursement policies for HCPCS 99292?")
```

### Using the Orchestrator

```python
from deep_research_agents.orchestrator import build_app
from deep_research_utils.snowflake_helper import SnowparkHelper

helper = SnowparkHelper(connection_type="programmatic", **snowflake_kwargs)
run = build_app(
    yaml_path="/path/to/semantic_model.yaml",
    snowflake_helper=helper,
    correlation_output_root="correlation_runs",
)
result = run(question="Find what changes in HCPCS 99292?")
```

## Available Agents

### ReimbursementAgent
Extracts and analyzes healthcare reimbursement policies.

### CorrelationAgent
Performs correlation analysis on healthcare data over time windows.

### PolicyHypothesisAgent
Generates hypotheses about policy changes and their impacts.

### MandateHypothesisAgent
Generates hypotheses about healthcare mandates.

### RecommendationAgent
Provides recommendations based on analysis results.

### PatternAgent
Extracts risk-relevant patterns from deep-dive report JSON payloads.

### Orchestrator
Coordinates multiple agents to handle complex queries.

## Development
- Add new agents under `src/deep_research_agents/`
- Update dependencies in `pyproject.toml`
- Write tests in the root `tests/` directory
