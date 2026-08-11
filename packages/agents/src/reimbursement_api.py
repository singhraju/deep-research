"""
Example: Reimbursement Policy Agent API

This example demonstrates how to expose the existing ReimbursementPolicyAgent
as a REST API using the AgentAPIBuilder.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
import re

from deep_research_core import AgentAPIBuilder
from deep_research_agents import ReimbursementAgent


# ============================================================================
# Custom Request Model with Validation
# ============================================================================


class ReimbursementRequest(BaseModel):
    """Request model for reimbursement policy extraction."""
    
    cpt_codes: str = Field(
        ...,
        description="Comma-separated CPT/HCPCS codes (5-digit codes)",
        example="99291,99292,99293",
        min_length=5
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking",
        example="conv_123456"
    )
    query: Optional[str] = Field(
        None,
        description="Optional user query for context",
        example="What are the reimbursement rules for critical care codes?"
    )
    
    @validator("cpt_codes")
    def validate_cpt_codes(cls, v):
        """Validate CPT code format."""
        # Remove whitespace
        v = v.strip()
        
        # Check format: comma-separated 5-digit codes
        pattern = r'^\d{5}(,\s*\d{5})*$'
        if not re.match(pattern, v):
            raise ValueError(
                "CPT codes must be comma-separated 5-digit codes (e.g., '99291,99292')"
            )
        
        # Normalize: remove spaces after commas
        v = re.sub(r',\s+', ',', v)
        
        return v


# ============================================================================
# Build API
# ============================================================================


def create_app():
    """Create and configure the FastAPI application."""
    
    # Initialize reimbursement agent
    # Note: In production, you would pass a real snowflake_helper
    # For this example, we'll let it auto-initialize
    reimbursement_agent = ReimbursementAgent(
        # snowflake_helper=your_snowpark_instance,  # Optional
        debug=True,
    )
    
    # Create API builder
    builder = AgentAPIBuilder(
        title="Reimbursement Policy API",
        version="1.0.0",
        description=(
            "API for extracting structured adjudication rules from insurance "
            "reimbursement policies for specific CPT/HCPCS codes"
        ),
        debug=True,
    )
    
    # Add reimbursement agent with custom validation
    builder.add_agent(
        reimbursement_agent,
        path="/reimbursement/extract",
        request_model=ReimbursementRequest,
        tags=["reimbursement", "policy-extraction"]
    )
    
    # Add CORS support
    builder.add_cors(allow_origins=["*"])
    
    # Build and return app
    return builder.build()


# Create app instance
app = create_app()


# ============================================================================
# Usage Instructions
# ============================================================================

if __name__ == "__main__":
    print("""
    Reimbursement Policy Agent API
    ==============================
    
    This API exposes the ReimbursementPolicyAgent for extracting structured
    adjudication rules from insurance reimbursement policies.
    
    Setup:
    ------
    1. Ensure environment variables are configured in .env:
       - SNOWFLAKE_* credentials
       - EHAP_* credentials for LLM access
    
    2. Install dependencies:
       uv sync
    
    3. Run the server (from project root):
       python -m uvicorn reimbursement_api:app --reload --port 8000
       
       Or if deep-research-agents is installed:
       uvicorn deep_research_agents.reimbursement_api:app --reload --port 8000
    
    Usage:
    ------
    
    # Extract policies for CPT codes
    curl -X POST http://localhost:8000/reimbursement/extract \\
      -H "Content-Type: application/json" \\
      -d '{
        "cpt_codes": "99291,99292",
        "conversation_id": "conv_123",
        "query": "What are the reimbursement rules for critical care?"
      }'
    
    # Minimal request (only CPT codes required)
    curl -X POST http://localhost:8000/reimbursement/extract \\
      -H "Content-Type: application/json" \\
      -d '{"cpt_codes": "99291,99292"}'
    
    # Test validation (should fail - invalid format)
    curl -X POST http://localhost:8000/reimbursement/extract \\
      -H "Content-Type: application/json" \\
      -d '{"cpt_codes": "invalid"}'
    
    # Health check
    curl http://localhost:8000/health
    
    # Metrics
    curl http://localhost:8000/metrics
    
    Documentation:
    -------------
    Open http://localhost:8000/docs for interactive API documentation
    
    Response Format:
    ---------------
    {
      "success": true,
      "result": [
        {
          "PLCY_ID": "POL123",
          "PLCY_NM": "Critical Care Policy",
          "results": [
            {
              "code": "99291",
              "denial_conditions": [...],
              "approval_conditions": [...],
              "limitations": [...]
            }
          ]
        }
      ],
      "execution_time": 12.34,
      "metadata": {
        "agent_name": "reimbursement",
        "timestamp": "2026-04-27T19:39:00Z"
      }
    }
    
    Features:
    ---------
    - CPT code format validation (5-digit codes)
    - Automatic code normalization
    - Rich error messages for validation failures
    - Metrics tracking per request
    - OpenAPI/Swagger documentation
    """)
