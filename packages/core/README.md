# deep-research-core

Core libraries for the deep-research project.

## Overview
This package contains the core framework and shared libraries used across the deep-research project, including:

- **AgentBase**: Base class for creating LangGraph agents
- **AgentAPIBuilder**: Builder for creating REST APIs from agents
- **API Models**: Standard request/response models for agent APIs
- **Credential Provider**: Secure credential management for agents

## Installation
This package is part of the deep-research monorepo and is managed using uv workspaces.

## Quick Start

### Creating a Custom Agent

```python
from deep_research_core import AgentBase

class MyAgent(AgentBase):
    @property
    def node_name(self) -> str:
        return "process"
    
    def node_function(self, state):
        result = self.llm.invoke(state["question"])
        return {"result": result.content}
    
    def prepare_state(self, question: str):
        return {"question": question, "llm": self.llm}
    
    def extract_result(self, graph_output):
        return graph_output["result"]

# Use the agent
agent = MyAgent(agent_name="my_agent")
result = agent(question="What is 2+2?")
```

### Building an API from Agents

#### Auto-Discovery Approach (Recommended)

```python
from deep_research_core import AgentAPIBuilder

# Automatically discover and register all agents in a directory
app = AgentAPIBuilder.create_api(
    agent_directory="packages/agents/src/deep_research_agents",
    title="Healthcare Agents API",
    version="2.0.0",
    debug=True
)

# Run with: uvicorn module:app --reload
```

#### Manual Registration Approach

```python
from deep_research_core import AgentAPIBuilder

# Create agents
agent1 = MyAgent(agent_name="agent1")
agent2 = AnotherAgent(agent_name="agent2")

# Build API
builder = AgentAPIBuilder(title="My Agent API")
builder.add_agent(agent1, path="/process")
builder.add_agent(agent2, path="/analyze")
app = builder.build()

# Run with: uvicorn module:app --reload
```

See [API Builder Guide](../../docs/api_builder_guide.md) for detailed documentation.

## Agent Implementations

For pre-built agent implementations (ReimbursementAgent, CorrelationAgent, Orchestrator, etc.), 
see the `deep-research-agents` package.
