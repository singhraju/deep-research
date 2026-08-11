"""
API Models for Agent API Builder

This module provides Pydantic models for API request/response handling.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Standard response model for agent execution."""
    
    success: bool = Field(
        ...,
        description="Whether the agent execution was successful"
    )
    result: Any = Field(
        ...,
        description="The result returned by the agent"
    )
    execution_time: float = Field(
        ...,
        description="Execution time in seconds",
        ge=0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the execution"
    )


class AgentErrorResponse(BaseModel):
    """Error response model for failed agent execution."""
    
    success: bool = Field(
        False,
        description="Always false for error responses"
    )
    error: str = Field(
        ...,
        description="Error message"
    )
    error_type: str = Field(
        ...,
        description="Type of error that occurred"
    )
    agent_name: Optional[str] = Field(
        None,
        description="Name of the agent that failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Health status (healthy, degraded, unhealthy)"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Registered agents and their status"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp of health check"
    )
    other_info: Optional[Dict] = {}
    redis_host_name: Optional[str] = None


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    total_requests: int = Field(
        ...,
        description="Total number of requests processed",
        ge=0
    )
    total_errors: int = Field(
        ...,
        description="Total number of errors encountered",
        ge=0
    )
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent metrics"
    )
    uptime_seconds: float = Field(
        ...,
        description="API uptime in seconds",
        ge=0
    )


class AgentInfo(BaseModel):
    """Information about a single agent."""
    
    name: str = Field(
        ...,
        description="Agent name"
    )
    endpoint: str = Field(
        ...,
        description="API endpoint path"
    )
    methods: list = Field(
        default_factory=lambda: ["POST"],
        description="Supported HTTP methods"
    )
    request_model: str = Field(
        ...,
        description="Request model class name"
    )
    description: Optional[str] = Field(
        None,
        description="Agent description"
    )


class AgentsListResponse(BaseModel):
    """Response listing all available agents."""
    
    total_agents: int = Field(
        ...,
        description="Total number of registered agents",
        ge=0
    )
    agents: list = Field(
        default_factory=list,
        description="List of agent information"
    )
    api_version: str = Field(
        ...,
        description="API version"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp"
    )
