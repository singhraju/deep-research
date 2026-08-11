# Agent API Builder Guide

## Overview

The `AgentAPIBuilder` is a powerful tool that transforms any `AgentBase` subclass into a production-ready REST API with minimal boilerplate. It handles request validation, error handling, logging, metrics, and OpenAPI documentation automatically.

## Quick Start

### Basic Usage

```python
from deep_research_core import AgentBase, AgentAPIBuilder

# 1. Create your custom agent
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

# 2. Build the API
agent = MyAgent(agent_name="my_agent")

builder = AgentAPIBuilder(title="My Agent API")
builder.add_agent(agent)
app = builder.build()

# 3. Run with: uvicorn module:app --reload
```

That's it! Your agent is now available at `POST /agents/my_agent`.

## Features

### 🚀 Automatic Endpoint Generation

Each agent becomes a REST API endpoint with:
- Request validation (auto-generated or custom)
- Response serialization
- Error handling
- Logging and metrics
- OpenAPI documentation

### 🔍 Hybrid Validation

**Option 1: Auto-Generated (Zero Boilerplate)**

The builder automatically generates Pydantic models from your agent's `prepare_state()` signature:

```python
class MyAgent(AgentBase):
    def prepare_state(self, text: str, max_length: int = 100):
        return {"text": text, "max_length": max_length, "llm": self.llm}

# Auto-generates:
# class MyAgentRequest(BaseModel):
#     text: str
#     max_length: int = 100

builder.add_agent(my_agent)  # Uses auto-generated model
```

**Option 2: Custom Models (Full Control)**

Define custom Pydantic models for advanced validation:

```python
from pydantic import BaseModel, Field

class CustomRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=1000)
    max_length: int = Field(100, ge=10, le=500)

builder.add_agent(my_agent, request_model=CustomRequest)
```

### 📊 Built-in Monitoring

**Health Check Endpoint**
```bash
GET /health
```

Returns:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agents": {
    "my_agent": {
      "path": "/agents/my_agent",
      "request_model": "MyAgentRequest"
    }
  },
  "timestamp": "2026-04-27T19:39:00Z"
}
```

**Metrics Endpoint**
```bash
GET /metrics
```

Returns:
```json
{
  "total_requests": 150,
  "total_errors": 3,
  "agents": {
    "my_agent": {
      "total_requests": 150,
      "total_errors": 3,
      "success_rate": 0.98,
      "average_execution_time": 2.34,
      "last_request_time": "2026-04-27T19:39:00Z"
    }
  },
  "uptime_seconds": 3600.0
}
```

## API Reference

### AgentAPIBuilder

```python
class AgentAPIBuilder:
    def __init__(
        self,
        title: str = "Agent API",
        version: str = "1.0.0",
        description: Optional[str] = None,
        debug: bool = False,
    )
```

**Parameters:**
- `title`: API title for documentation
- `version`: API version string
- `description`: API description for OpenAPI docs
- `debug`: Enable debug logging

### add_agent()

```python
def add_agent(
    self,
    agent: AgentBase,
    path: Optional[str] = None,
    methods: List[str] = None,
    request_model: Optional[Type[BaseModel]] = None,
    response_model: Optional[Type[BaseModel]] = None,
    tags: Optional[List[str]] = None,
) -> "AgentAPIBuilder"
```

**Parameters:**
- `agent`: AgentBase instance to expose
- `path`: Custom endpoint path (default: `/agents/{agent_name}`)
- `methods`: HTTP methods (default: `["POST"]`)
- `request_model`: Custom Pydantic request model (default: auto-generated)
- `response_model`: Custom response model (default: `AgentResponse`)
- `tags`: OpenAPI tags for grouping

**Returns:** Self for method chaining

### add_cors()

```python
def add_cors(
    self,
    allow_origins: List[str] = None,
    allow_credentials: bool = True,
    allow_methods: List[str] = None,
    allow_headers: List[str] = None,
) -> "AgentAPIBuilder"
```

Add CORS middleware for cross-origin requests.

### build()

```python
def build(self) -> FastAPI
```

Build and return the configured FastAPI application.

## Response Format

### Success Response

```json
{
  "success": true,
  "result": <agent_result>,
  "execution_time": 2.34,
  "metadata": {
    "agent_name": "my_agent",
    "timestamp": "2026-04-27T19:39:00Z"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error message",
  "error_type": "AgentExecutionError",
  "agent_name": "my_agent",
  "metadata": {
    "execution_time": 1.23
  }
}
```

## Examples

### Example 1: Simple Agent

```python
from deep_research_core import AgentBase, AgentAPIBuilder

class EchoAgent(AgentBase):
    @property
    def node_name(self) -> str:
        return "echo"
    
    def node_function(self, state):
        return {"result": f"Echo: {state['message']}"}
    
    def prepare_state(self, message: str):
        return {"message": message}
    
    def extract_result(self, graph_output):
        return graph_output["result"]

agent = EchoAgent(agent_name="echo")
builder = AgentAPIBuilder()
builder.add_agent(agent)
app = builder.build()
```

**Usage:**
```bash
curl -X POST http://localhost:8000/agents/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World"}'
```

### Example 2: Multiple Agents

```python
builder = AgentAPIBuilder(title="Multi-Agent API")

builder.add_agent(summary_agent, path="/summarize", tags=["text"])
builder.add_agent(analysis_agent, path="/analyze", tags=["analytics"])
builder.add_agent(translation_agent, path="/translate", tags=["text"])

app = builder.build()
```

### Example 3: Custom Validation

```python
from pydantic import BaseModel, Field, validator

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    max_length: int = Field(100, ge=10, le=500)
    style: str = Field("concise", regex="^(concise|detailed|bullet)$")
    
    @validator("text")
    def validate_text(cls, v):
        if "forbidden" in v.lower():
            raise ValueError("Text contains forbidden content")
        return v

builder.add_agent(
    summary_agent,
    path="/summarize",
    request_model=SummaryRequest
)
```

### Example 4: Existing Agent (ReimbursementPolicyAgent)

```python
from deep_research_core import AgentAPIBuilder
from deep_research_core.reimbursement_agent import ReimbursementPolicyAgent
from pydantic import BaseModel, Field

class ReimbursementRequest(BaseModel):
    cpt_codes: str = Field(..., regex=r"^\d{5}(,\d{5})*$")

agent = ReimbursementPolicyAgent(agent_name="reimbursement")

builder = AgentAPIBuilder(title="Reimbursement API")
builder.add_agent(
    agent,
    path="/reimbursement/extract",
    request_model=ReimbursementRequest
)
app = builder.build()
```

## Deployment

### Local Development

```bash
# Install dependencies
uv sync

# Run with auto-reload
uvicorn module:app --reload --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Production

```bash
# Run with Gunicorn + Uvicorn workers
gunicorn module:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

CMD ["uvicorn", "module:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Best Practices

### 1. Use Custom Models for Production APIs

Start with auto-generation for prototyping, but add custom models for production:

```python
# Development: Fast iteration
builder.add_agent(agent)  # Auto-generated

# Production: Explicit validation
builder.add_agent(agent, request_model=CustomRequest)
```

### 2. Add Descriptive Documentation

Use docstrings and Field descriptions:

```python
class MyRequest(BaseModel):
    """Request for my agent."""
    
    text: str = Field(
        ...,
        description="Input text to process",
        example="Sample text here"
    )
```

### 3. Implement Error Handling

Override `handle_execution_error()` in your agent:

```python
class MyAgent(AgentBase):
    def handle_execution_error(self, exc, **kwargs):
        self.logger.error(f"Failed: {exc}")
        return {"error": str(exc), "result": None}
```

### 4. Use Tags for Organization

Group related endpoints:

```python
builder.add_agent(agent1, tags=["text-processing"])
builder.add_agent(agent2, tags=["text-processing"])
builder.add_agent(agent3, tags=["analytics"])
```

### 5. Monitor Metrics

Regularly check `/metrics` endpoint to track:
- Success rates
- Execution times
- Error patterns

## Troubleshooting

### Issue: Auto-generation fails

**Solution:** Ensure `prepare_state()` has type hints:

```python
# Bad
def prepare_state(self, text, max_length=100):
    ...

# Good
def prepare_state(self, text: str, max_length: int = 100):
    ...
```

### Issue: Validation errors

**Solution:** Check request format matches model:

```bash
# Get model schema from /docs
curl http://localhost:8000/openapi.json | jq '.components.schemas'
```

### Issue: Agent errors not caught

**Solution:** Ensure agent raises `AgentError` or subclasses:

```python
from deep_research_core import AgentError

def node_function(self, state):
    if error_condition:
        raise AgentError("Descriptive error message")
```

## Advanced Topics

### Custom Response Models

```python
class CustomResponse(BaseModel):
    data: Any
    metadata: Dict[str, Any]
    warnings: List[str] = []

builder.add_agent(
    agent,
    response_model=CustomResponse
)
```

### Middleware

```python
from fastapi import Request

@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "value"
    return response
```

### Authentication (Future)

```python
# Coming in future iterations
builder.add_authentication(
    type="api_key",
    header="X-API-Key",
    validator=validate_api_key
)
```

## See Also

- [AgentBase Documentation](../packages/core/README.md)
- [Example: Simple Agent API](../examples/simple_agent_api.py)
- [Example: Advanced Agent API](../examples/advanced_agent_api.py)
- [Example: Reimbursement API](../examples/reimbursement_api.py)
