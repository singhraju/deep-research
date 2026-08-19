"""
Tests for AgentAPIBuilder

These tests verify that the API builder correctly creates FastAPI endpoints
from AgentBase instances with proper validation and error handling.
"""

import pytest
from typing import Dict, Any
from pydantic import BaseModel, Field
from fastapi.testclient import TestClient

from deep_research_core import AgentBase, AgentAPIBuilder


# ============================================================================
# Test Agents
# ============================================================================


class SimpleAgent(AgentBase):
    """Simple test agent that echoes input."""
    
    @property
    def node_name(self) -> str:
        return "echo"
    
    def node_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        message = state["message"]
        return {"result": f"Echo: {message}"}
    
    def prepare_state(self, message: str) -> Dict[str, Any]:
        return {"message": message}
    
    def extract_result(self, graph_output: Dict[str, Any]) -> str:
        return graph_output["result"]
    
    def create_stub_llm(self):
        """Stub LLM for testing."""
        class StubLLM:
            def invoke(self, messages):
                class Response:
                    content = "stub response"
                return Response()
        return StubLLM()


class ComplexAgent(AgentBase):
    """Agent with multiple parameters and defaults."""
    
    @property
    def node_name(self) -> str:
        return "process"
    
    def node_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state["text"]
        count = state["count"]
        flag = state["flag"]
        return {"result": {"text": text, "count": count, "flag": flag}}
    
    def prepare_state(
        self,
        text: str,
        count: int = 10,
        flag: bool = False
    ) -> Dict[str, Any]:
        return {"text": text, "count": count, "flag": flag}
    
    def extract_result(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        return graph_output["result"]
    
    def create_stub_llm(self):
        """Stub LLM for testing."""
        class StubLLM:
            def invoke(self, messages):
                class Response:
                    content = "stub response"
                return Response()
        return StubLLM()


# ============================================================================
# Test Custom Request Models
# ============================================================================


class CustomRequest(BaseModel):
    """Custom request model with validation."""
    message: str = Field(..., min_length=5, max_length=100)


# ============================================================================
# Tests
# ============================================================================


class TestAgentAPIBuilder:
    """Test suite for AgentAPIBuilder."""
    
    def test_builder_initialization(self):
        """Test builder can be initialized."""
        builder = AgentAPIBuilder(
            title="Test API",
            version="1.0.0",
            description="Test description"
        )
        
        assert builder.title == "Test API"
        assert builder.version == "1.0.0"
        assert builder.description == "Test description"
        assert len(builder.agents) == 0
    
    def test_add_single_agent(self):
        """Test adding a single agent."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        
        builder.add_agent(agent)
        
        assert "simple" in builder.agents
        assert builder.agents["simple"] == agent
        assert builder.agent_paths["simple"] == "/agents/simple"
    
    def test_add_agent_with_custom_path(self):
        """Test adding agent with custom path."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        
        builder.add_agent(agent, path="/custom/path")
        
        assert builder.agent_paths["simple"] == "/custom/path"
    
    def test_add_multiple_agents(self):
        """Test adding multiple agents."""
        agent1 = SimpleAgent(agent_name="agent1", test_mode=True)
        agent2 = ComplexAgent(agent_name="agent2", test_mode=True)
        
        builder = AgentAPIBuilder()
        builder.add_agent(agent1)
        builder.add_agent(agent2)
        
        assert len(builder.agents) == 2
        assert "agent1" in builder.agents
        assert "agent2" in builder.agents
    
    def test_duplicate_agent_name_raises_error(self):
        """Test that duplicate agent names raise an error."""
        agent1 = SimpleAgent(agent_name="duplicate", test_mode=True)
        agent2 = SimpleAgent(agent_name="duplicate", test_mode=True)
        
        builder = AgentAPIBuilder()
        builder.add_agent(agent1)
        
        with pytest.raises(ValueError, match="already registered"):
            builder.add_agent(agent2)
    
    def test_duplicate_path_raises_error(self):
        """Test that duplicate paths raise an error."""
        agent1 = SimpleAgent(agent_name="agent1", test_mode=True)
        agent2 = SimpleAgent(agent_name="agent2", test_mode=True)
        
        builder = AgentAPIBuilder()
        builder.add_agent(agent1, path="/same/path")
        
        with pytest.raises(ValueError, match="already in use"):
            builder.add_agent(agent2, path="/same/path")
    
    def test_build_creates_app(self):
        """Test that build() creates a FastAPI app."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        
        app = builder.build()
        
        assert app is not None
        assert hasattr(app, "routes")
    
    def test_health_endpoint(self):
        """Test that health endpoint is created."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "simple" in data["agents"]
    
    def test_metrics_endpoint(self):
        """Test that metrics endpoint is created."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "agents" in data
    
    def test_agent_endpoint_with_auto_generated_model(self):
        """Test agent endpoint with auto-generated request model."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        response = client.post(
            "/agents/simple",
            json={"message": "Hello World"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"] == "Echo: Hello World"
        assert "execution_time" in data
    
    def test_agent_endpoint_with_custom_model(self):
        """Test agent endpoint with custom request model."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent, request_model=CustomRequest)
        app = builder.build()
        
        client = TestClient(app)
        
        # Valid request
        response = client.post(
            "/agents/simple",
            json={"message": "Valid message"}
        )
        assert response.status_code == 200
        
        # Invalid request (too short)
        response = client.post(
            "/agents/simple",
            json={"message": "Hi"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_complex_agent_with_defaults(self):
        """Test agent with multiple parameters and defaults."""
        agent = ComplexAgent(agent_name="complex", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        
        # With defaults
        response = client.post(
            "/agents/complex",
            json={"text": "test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["count"] == 10
        assert data["result"]["flag"] is False
        
        # Override defaults
        response = client.post(
            "/agents/complex",
            json={"text": "test", "count": 20, "flag": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["count"] == 20
        assert data["result"]["flag"] is True
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder()
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        
        # Make several requests
        for i in range(3):
            client.post("/agents/simple", json={"message": f"test {i}"})
        
        # Check metrics
        response = client.get("/metrics")
        data = response.json()
        
        assert data["total_requests"] == 3
        assert data["agents"]["simple"]["total_requests"] == 3
        assert data["agents"]["simple"]["success_rate"] == 1.0
    
    def test_method_chaining(self):
        """Test that builder methods support chaining."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        
        builder = AgentAPIBuilder()
        result = builder.add_agent(agent).add_cors()
        
        assert result is builder
    
    def test_openapi_docs(self):
        """Test that OpenAPI documentation is generated."""
        agent = SimpleAgent(agent_name="simple", test_mode=True)
        builder = AgentAPIBuilder(
            title="Test API",
            version="1.0.0",
            description="Test description"
        )
        builder.add_agent(agent)
        app = builder.build()
        
        client = TestClient(app)
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Test API"
        assert data["info"]["version"] == "1.0.0"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
