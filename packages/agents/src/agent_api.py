"""
Example: Auto-Discovery Agent API

This example demonstrates the new auto-discovery approach where the API builder
automatically scans a directory and registers all AgentBase instances.

This is the recommended approach for most use cases as it requires minimal code
and automatically picks up new agents as they are added to the directory.
"""

from deep_research_core import AgentAPIBuilder

# ============================================================================
# Auto-Discovery Approach (Recommended)
# ============================================================================

# Simply point to your agents directory and let the builder do the rest
app = AgentAPIBuilder.create_api(
    agent_directory="packages/agents/src/deep_research_agents",
    title="Healthcare Deep Research API",
    version="2.0.0",
    description="Automated API for all healthcare research agents",
    debug=True,
    path_prefix="/agents",
    exclude_files=["orchestrator.py", "user_intent.py"],  # Exclude non-agent modules
)


# ============================================================================
# Usage Instructions
# ============================================================================

if __name__ == "__main__":
    print("""
    Auto-Discovery Agent API
    ========================
    
    This API automatically discovers and registers all AgentBase instances
    from the specified directory.
    
    Setup:
    ------
    1. Ensure environment variables are configured in .env:
       - SNOWFLAKE_* credentials
       - EHAP_* credentials for LLM access
    
    2. Install dependencies:
       uv sync
    
    3. Run the server:
       uvicorn examples.auto_discovery_api:app --reload --port 8000
    
    Features:
    ---------
    ✓ Automatic agent discovery - no manual registration needed
    ✓ Scans directory for all AgentBase subclasses
    ✓ Auto-generates request models from agent signatures
    ✓ Consistent endpoint naming: /agents/{agent_name}
    ✓ Built-in health checks and metrics
    ✓ OpenAPI/Swagger documentation
    
    Discovered Endpoints:
    --------------------
    The API will automatically create endpoints for all agents found:
    
    - POST /agents/reimbursement
    - POST /agents/correlation  
    - POST /agents/policy_hypothesis
    - POST /agents/mandate_hypothesis
    - POST /agents/recommendation
    
    System Endpoints:
    ----------------
    - GET  /agents   - List all available agents and endpoints
    - GET  /health   - Health check and agent registry
    - GET  /metrics  - Usage metrics and statistics
    - GET  /docs     - Interactive API documentation
    - GET  /redoc    - Alternative API documentation
    
    Example Usage:
    -------------
    # List all available agents
    curl http://localhost:8000/agents
    
    # Health check (see all registered agents)
    curl http://localhost:8000/health
    
    # Execute an agent
    curl -X POST http://localhost:8000/agents/reimbursement \\
      -H "Content-Type: application/json" \\
      -d '{"cpt_codes": "99291,99292"}'
    
    # View metrics
    curl http://localhost:8000/metrics
    
    # Interactive docs
    open http://localhost:8000/docs
    
    Customization:
    -------------
    You can customize the auto-discovery behavior:
    
    - agent_directory: Path to scan for agents
    - path_prefix: Prefix for all endpoints (default: "/agents")
    - exclude_files: List of files to skip (e.g., ["__init__.py", "base.py"])
    - debug: Enable detailed logging
    
    Example with custom settings:
    
    app = AgentAPIBuilder.create_api(
        agent_directory="my_agents/",
        title="My Custom API",
        path_prefix="/api/v1",
        exclude_files=["helper.py", "utils.py"],
        debug=False
    )
    
    Adding New Agents:
    -----------------
    To add a new agent to the API:
    
    1. Create a new Python file in the agents directory
    2. Define a class that extends AgentBase
    3. Implement the required methods
    4. Restart the API server
    
    The new agent will be automatically discovered and registered!
    
    No code changes needed in this file.
    """)
