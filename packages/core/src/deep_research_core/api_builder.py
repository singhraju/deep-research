"""
Agent API Builder

This module provides a builder class for creating FastAPI applications from AgentBase instances.
Developers can create custom agents and expose them as REST API endpoints with minimal boilerplate.

Quick Start (Auto-Discovery - Recommended):
    >>> from deep_research_core import AgentAPIBuilder
    >>> 
    >>> # Automatically discover and register all agents in a directory
    >>> app = AgentAPIBuilder.create_api(
    ...     agent_directory="packages/agents/src/deep_research_agents",
    ...     title="My Agent API"
    ... )
    >>> 
    >>> # Run with: uvicorn module:app --reload

Quick Start (Manual Registration):
    >>> from deep_research_core.api_builder import AgentAPIBuilder
    >>> from my_agents import MyAgent
    >>> 
    >>> agent = MyAgent(agent_name="my_agent")
    >>> 
    >>> builder = AgentAPIBuilder(title="My Agent API")
    >>> builder.add_agent(agent)
    >>> app = builder.build()
    >>> 
    >>> # Run with: uvicorn module:app --reload

Features:
    - Auto-discovery: Scan directories for agents and register automatically
    - Automatic endpoint generation from agents
    - Hybrid validation (auto-generate or custom Pydantic models)
    - Built-in error handling and logging
    - Health checks and metrics
    - OpenAPI/Swagger documentation
"""

import importlib.util
import inspect
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from deep_research_utils.app_constant import AppConstants
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model
from typing_extensions import get_type_hints

from deep_research_core.base_agent import AgentBase, AgentError, AgentConfigurationError, AgentExecutionError
from deep_research_core.api_models import (
    AgentErrorResponse,
    AgentInfo,
    AgentResponse,
    AgentsListResponse,
    HealthResponse,
    MetricsResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Tracking
# ============================================================================


class AgentMetrics:
    """Track metrics for a single agent."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.total_requests = 0
        self.total_errors = 0
        self.total_execution_time = 0.0
        self.last_request_time: Optional[str] = None
        self.last_error: Optional[str] = None
    
    def record_request(self, execution_time: float, success: bool, error: Optional[str] = None):
        """Record a request execution."""
        self.total_requests += 1
        self.total_execution_time += execution_time
        self.last_request_time = datetime.now(timezone.utc).isoformat()
        
        if not success:
            self.total_errors += 1
            self.last_error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        avg_time = (
            self.total_execution_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "success_rate": (
                (self.total_requests - self.total_errors) / self.total_requests
                if self.total_requests > 0
                else 1.0
            ),
            "average_execution_time": avg_time,
            "last_request_time": self.last_request_time,
            "last_error": self.last_error,
        }


# ============================================================================
# Agent API Builder
# ============================================================================


class AgentAPIBuilder:
    """
    Builder for creating FastAPI applications from AgentBase instances.
    
    This class provides a fluent interface for registering agents as API endpoints
    with automatic request/response handling, validation, and documentation.
    
    Attributes:
        app: FastAPI application instance
        agents: Dictionary of registered agents
        metrics: Dictionary of agent metrics
        start_time: API startup timestamp
    
    Example (Manual Registration):
        >>> builder = AgentAPIBuilder(
        ...     title="Healthcare Agents API",
        ...     version="1.0.0",
        ...     description="API for healthcare analysis agents"
        ... )
        >>> 
        >>> builder.add_agent(summary_agent, path="/summarize")
        >>> builder.add_agent(reimbursement_agent)
        >>> 
        >>> app = builder.build()
    
    Example (Auto-Discovery):
        >>> app = AgentAPIBuilder.create_api(
        ...     agent_directory="packages/agents/src/deep_research_agents",
        ...     title="Healthcare Agents API"
        ... )
    """
    
    def __init__(
        self,
        title: str = "Agent API",
        version: str = "1.0.0",
        description: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the API builder.
        
        Args:
            title: API title for documentation
            version: API version
            description: API description for documentation
            debug: Enable debug logging
        """
        self.title = title
        self.version = version
        self.description = description or f"{title} - Powered by AgentBase"
        self.debug = debug
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=self.title,
            version=self.version,
            description=self.description,
        )
        
        # Agent registry
        self.agents: Dict[str, AgentBase] = {}
        self.agent_paths: Dict[str, str] = {}
        self.agent_request_models: Dict[str, Type[BaseModel]] = {}
        
        # Metrics
        self.metrics: Dict[str, AgentMetrics] = {}
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(f"api_builder.{title}")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        self.logger.info(f"Initializing {title} API Builder v{version}")
    
    def add_agent(
        self,
        agent: AgentBase,
        path: Optional[str] = None,
        methods: List[str] = None,
        request_model: Optional[Type[BaseModel]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        tags: Optional[List[str]] = None,
    ) -> "AgentAPIBuilder":
        """
        Add an agent as an API endpoint.
        
        Args:
            agent: AgentBase instance to expose as API
            path: Custom endpoint path (default: /agents/{agent_name})
            methods: HTTP methods to support (default: ["POST"])
            request_model: Custom Pydantic request model (default: auto-generated)
            response_model: Custom Pydantic response model (default: AgentResponse)
            tags: OpenAPI tags for grouping endpoints
        
        Returns:
            Self for method chaining
        
        Example:
            >>> # Auto-generated request model
            >>> builder.add_agent(simple_agent)
            >>> 
            >>> # Custom request model with validation
            >>> class CustomRequest(BaseModel):
            ...     text: str = Field(..., min_length=1)
            >>> 
            >>> builder.add_agent(
            ...     complex_agent,
            ...     path="/custom",
            ...     request_model=CustomRequest,
            ...     tags=["analysis"]
            ... )
        """
        if not isinstance(agent, AgentBase):
            raise TypeError(f"Agent must be an instance of AgentBase, got {type(agent)}")
        
        # Default values
        agent_name = agent.agent_name
        endpoint_path = path or f"/agents/{agent_name}"
        http_methods = methods or ["POST"]
        endpoint_tags = tags or [agent_name]
        
        # Check for duplicates
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' is already registered")
        
        if endpoint_path in self.agent_paths.values():
            raise ValueError(f"Path '{endpoint_path}' is already in use")
        
        # Generate or use provided request model
        req_model = request_model or self._generate_request_model(agent)
        agent_response_model = getattr(agent, "api_response_model", None)
        resp_model = response_model or agent_response_model or AgentResponse
        
        # Register agent
        self.agents[agent_name] = agent
        self.agent_paths[agent_name] = endpoint_path
        self.agent_request_models[agent_name] = req_model
        self.metrics[agent_name] = AgentMetrics(agent_name)
        
        self.logger.info(
            f"Registered agent '{agent_name}' at {endpoint_path} "
            f"(request_model: {req_model.__name__})"
        )
        
        # Create endpoint
        self._create_endpoint(
            agent=agent,
            path=endpoint_path,
            methods=http_methods,
            request_model=req_model,
            response_model=resp_model,
            tags=endpoint_tags,
        )
        
        return self
    
    def _generate_request_model(self, agent: AgentBase) -> Type[BaseModel]:
        """
        Auto-generate Pydantic request model from agent's prepare_state() signature.
        
        Args:
            agent: AgentBase instance
        
        Returns:
            Generated Pydantic BaseModel class
        """
        try:
            # Get prepare_state signature
            sig = inspect.signature(agent.prepare_state)
            
            # Get type hints (handles forward references)
            try:
                hints = get_type_hints(agent.prepare_state)
            except Exception as e:
                self.logger.warning(
                    f"Could not get type hints for {agent.agent_name}.prepare_state(): {e}. "
                    f"Using inspect.signature annotations instead."
                )
                hints = {}
            
            # Build field definitions
            fields = {}
            for param_name, param in sig.parameters.items():
                # Skip self and **kwargs
                if param_name in ('self', 'kwargs'):
                    continue
                
                # Get type annotation
                if param_name in hints:
                    annotation = hints[param_name]
                elif param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                else:
                    annotation = Any
                
                # Get default value
                if param.default != inspect.Parameter.empty:
                    default = param.default
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue  # Skip **kwargs
                else:
                    default = ...  # Required field
                
                fields[param_name] = (annotation, default)
            
            # Create model
            model_name = f"{agent.agent_name.title().replace('_', '')}Request"
            model = create_model(model_name, **fields)
            
            self.logger.debug(
                f"Auto-generated request model '{model_name}' with fields: {list(fields.keys())}"
            )
            
            return model
            
        except Exception as exc:
            self.logger.error(
                f"Failed to generate request model for {agent.agent_name}: {exc}",
                exc_info=exc
            )
            # Fallback to generic model
            return create_model(
                f"{agent.agent_name}Request",
                **{"data": (Dict[str, Any], ...)}
            )
    
    def _create_endpoint(
        self,
        agent: AgentBase,
        path: str,
        methods: List[str],
        request_model: Type[BaseModel],
        response_model: Type[BaseModel],
        tags: List[str],
    ):
        """
        Create FastAPI endpoint for an agent.
        
        Args:
            agent: AgentBase instance
            path: Endpoint path
            methods: HTTP methods
            request_model: Pydantic request model
            response_model: Pydantic response model
            tags: OpenAPI tags
        """
        agent_name = agent.agent_name
        
        # Create endpoint handler
        async def execute_agent_endpoint(request: request_model):
            """Execute agent with provided parameters."""
            start_time = time.time()
            
            try:
                self.logger.info(f"Executing agent '{agent_name}'")
                
                # Convert request to dict
                request_dict = request.dict()
                
                if self.debug:
                    self.logger.debug(f"Request data: {list(request_dict.keys())}")
                
                # Execute agent
                result = agent.execute(**request_dict)
                
                execution_time = time.time() - start_time
                
                # Record metrics
                self.metrics[agent_name].record_request(
                    execution_time=execution_time,
                    success=True
                )
                
                self.logger.info(
                    f"Agent '{agent_name}' completed successfully in {execution_time:.2f}s"
                )
                
                # Check if result indicates need for clarification (for orchestrator responses)
                if isinstance(result, dict) and result.get("status") == "needs_clarification":
                    clarification_request = result.get("clarification_request", {})
                    self.logger.info(f"Agent '{agent_name}' requires clarification")
                    
                    raise HTTPException(
                        status_code=422,  # Unprocessable Entity - semantically correct for "need more info"
                        detail={
                            "error": "Clarification required before processing can continue",
                            "error_type": "ClarificationRequired",
                            "agent_name": agent_name,
                            "clarification_request": clarification_request,
                            "metadata": {
                                "execution_time": execution_time,
                                "blocking_issues": clarification_request.get("blocking_issues", []),
                                "questions": clarification_request.get("questions", []),
                                "suggested_defaults": clarification_request.get("suggested_defaults", {}),
                            }
                        }
                    )
                
                # Return response
                if response_model is AgentResponse or issubclass(response_model, AgentResponse):
                    return AgentResponse(
                        success=True,
                        result=result,
                        execution_time=execution_time,
                        metadata={
                            "agent_name": agent_name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                return result
                
            except AgentConfigurationError as exc:
                # Configuration error - client can fix this (422)
                execution_time = time.time() - start_time
                error_msg = str(exc)
                
                self.metrics[agent_name].record_request(
                    execution_time=execution_time,
                    success=False,
                    error=error_msg
                )
                
                self.logger.error(
                    f"Agent '{agent_name}' configuration error: {error_msg}",
                    exc_info=exc
                )
                
                raise HTTPException(
                    status_code=422,  # Unprocessable Entity - configuration issue
                    detail=AgentErrorResponse(
                        error=error_msg,
                        error_type=exc.__class__.__name__,
                        agent_name=agent_name,
                        metadata={"execution_time": execution_time}
                    ).dict()
                )
                
            except AgentExecutionError as exc:
                # Execution error - server-side issue (500)
                execution_time = time.time() - start_time
                error_msg = str(exc)
                
                self.metrics[agent_name].record_request(
                    execution_time=execution_time,
                    success=False,
                    error=error_msg
                )
                
                self.logger.error(
                    f"Agent '{agent_name}' execution failed: {error_msg}",
                    exc_info=exc
                )
                
                raise HTTPException(
                    status_code=500,  # Internal Server Error
                    detail=AgentErrorResponse(
                        error=error_msg,
                        error_type=exc.__class__.__name__,
                        agent_name=agent_name,
                        metadata={"execution_time": execution_time}
                    ).dict()
                )
                
            except AgentError as exc:
                # Generic agent error - default to 500
                execution_time = time.time() - start_time
                error_msg = str(exc)
                
                self.metrics[agent_name].record_request(
                    execution_time=execution_time,
                    success=False,
                    error=error_msg
                )
                
                self.logger.error(
                    f"Agent '{agent_name}' failed: {error_msg}",
                    exc_info=exc
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=AgentErrorResponse(
                        error=error_msg,
                        error_type=exc.__class__.__name__,
                        agent_name=agent_name,
                        metadata={"execution_time": execution_time}
                    ).dict()
                )
                
            except Exception as exc:
                # Unexpected error
                execution_time = time.time() - start_time
                error_msg = str(exc)
                
                self.metrics[agent_name].record_request(
                    execution_time=execution_time,
                    success=False,
                    error=error_msg
                )
                
                self.logger.exception(
                    f"Unexpected error in agent '{agent_name}': {error_msg}",
                    exc_info=exc
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=AgentErrorResponse(
                        error=error_msg,
                        error_type=exc.__class__.__name__,
                        agent_name=agent_name,
                        metadata={"execution_time": execution_time}
                    ).dict()
                )
        
        # Set endpoint metadata
        execute_agent_endpoint.__name__ = f"execute_{agent_name}"
        execute_agent_endpoint.__doc__ = (
            agent.__doc__ or f"Execute {agent_name} agent"
        )
        
        # Register endpoint for each method
        for method in methods:
            method_lower = method.lower()
            
            if method_lower == "post":
                self.app.post(
                    path,
                    response_model=response_model,
                    tags=tags,
                    summary=f"Execute {agent_name} agent",
                    description=agent.__doc__ or f"Execute the {agent_name} agent with provided parameters",
                )(execute_agent_endpoint)
            elif method_lower == "get":
                self.app.get(
                    path,
                    response_model=response_model,
                    tags=tags,
                    summary=f"Execute {agent_name} agent",
                )(execute_agent_endpoint)
            else:
                self.logger.warning(f"Unsupported HTTP method: {method}")
    
    def add_cors(
        self,
        allow_origins: List[str] = None,
        allow_credentials: bool = True,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
    ) -> "AgentAPIBuilder":
        """
        Add CORS middleware to the API.
        
        Args:
            allow_origins: List of allowed origins (default: ["*"])
            allow_credentials: Allow credentials
            allow_methods: Allowed HTTP methods (default: ["*"])
            allow_headers: Allowed headers (default: ["*"])
        
        Returns:
            Self for method chaining
        """
        from fastapi.middleware.cors import CORSMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or ["*"],
            allow_credentials=allow_credentials,
            allow_methods=allow_methods or ["*"],
            allow_headers=allow_headers or ["*"],
        )
        
        self.logger.info("CORS middleware added")
        return self
    
    def build(self) -> FastAPI:
        """
        Build and return the FastAPI application.
        
        This method adds health check and metrics endpoints, then returns
        the configured FastAPI app ready to run.
        
        Returns:
            Configured FastAPI application
        
        Example:
            >>> app = builder.build()
            >>> # Run with: uvicorn module:app --reload
        """
        # Add agents list endpoint
        @self.app.get(
            "/agents",
            response_model=AgentsListResponse,
            tags=["system"],
            summary="List all agents",
            description="Get a list of all available agents and their endpoints"
        )
        async def list_agents():
            """List all registered agents."""
            agents_info = []
            
            for name in sorted(self.agents.keys()):
                agent = self.agents[name]
                agent_info = AgentInfo(
                    name=name,
                    endpoint=self.agent_paths[name],
                    methods=["POST"],  # Currently all agents use POST
                    request_model=self.agent_request_models[name].__name__,
                    description=agent.__doc__.strip() if agent.__doc__ else None,
                )
                agents_info.append(agent_info)
            
            return AgentsListResponse(
                total_agents=len(agents_info),
                agents=agents_info,
                api_version=self.version,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        # Add health check endpoint
        @self.app.get(
            "/health",
            response_model=HealthResponse,
            tags=["system"],
            summary="Health check",
            description="Check API health and list registered agents"
        )
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                version=self.version,
                redis_host_name=AppConstants.REDIS_HOST,
                agents={
                    name: {
                        "path": self.agent_paths[name],
                        "request_model": self.agent_request_models[name].__name__,
                    }
                    for name in self.agents.keys()
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        # Add metrics endpoint
        @self.app.get(
            "/metrics",
            response_model=MetricsResponse,
            tags=["system"],
            summary="API metrics",
            description="Get API usage metrics and statistics"
        )
        async def get_metrics():
            """Metrics endpoint."""
            total_requests = sum(m.total_requests for m in self.metrics.values())
            total_errors = sum(m.total_errors for m in self.metrics.values())
            
            return MetricsResponse(
                total_requests=total_requests,
                total_errors=total_errors,
                agents={name: m.to_dict() for name, m in self.metrics.items()},
                uptime_seconds=time.time() - self.start_time,
            )
        
        self.logger.info(
            f"API built successfully with {len(self.agents)} agent(s): "
            f"{list(self.agents.keys())}"
        )
        
        return self.app
    
    @classmethod
    def create_api(
        cls,
        agent_directory: str,
        title: str = "Agent API",
        version: str = "1.0.0",
        description: Optional[str] = None,
        debug: bool = False,
        path_prefix: str = "/agents",
        exclude_files: Optional[List[str]] = None,
    ) -> FastAPI:
        """
        Create a FastAPI application by auto-discovering agents in a directory.
        
        This method scans the specified directory for Python modules, imports them,
        and automatically registers any AgentBase instances found as API endpoints.
        
        Args:
            agent_directory: Path to directory containing agent modules
            title: API title for documentation
            version: API version
            description: API description for documentation
            debug: Enable debug logging
            path_prefix: Prefix for all agent endpoints (default: "/agents")
            exclude_files: List of filenames to exclude (e.g., ["__init__.py", "base.py"])
        
        Returns:
            Configured FastAPI application ready to run
        
        Example:
            >>> app = AgentAPIBuilder.create_api(
            ...     agent_directory="packages/agents/src/deep_research_agents",
            ...     title="Healthcare Agents API",
            ...     version="2.0.0",
            ...     debug=True
            ... )
            >>> # Run with: uvicorn module:app --reload
        
        Note:
            - Only files ending in .py are scanned
            - __init__.py and files in exclude_files are skipped by default
            - Each module is searched for AgentBase instances
            - Agents are registered at {path_prefix}/{agent_name}
        """
        logger_instance = logging.getLogger(f"api_builder.{title}")
        logger_instance.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Configure console handler if not already present
        if not logger_instance.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger_instance.addHandler(console_handler)
        
        # Initialize builder
        builder = cls(
            title=title,
            version=version,
            description=description,
            debug=debug,
        )
        
        # Default exclusions
        exclude = set(exclude_files or [])
        exclude.add("__init__.py")
        
        # Convert to Path object
        agent_dir = Path(agent_directory)
        
        if not agent_dir.exists():
            raise ValueError(f"Agent directory does not exist: {agent_directory}")
        
        if not agent_dir.is_dir():
            raise ValueError(f"Agent directory is not a directory: {agent_directory}")
        
        logger_instance.info(f"Scanning directory for agents: {agent_dir}")
        
        # Discover and load agents
        discovered_agents = cls._discover_agents(
            agent_dir=agent_dir,
            exclude_files=exclude,
            logger=logger_instance,
        )
        
        if not discovered_agents:
            logger_instance.warning(f"No agents found in {agent_directory}")
        
        # Register each discovered agent
        for agent_instance, module_name in discovered_agents:
            agent_name = agent_instance.agent_name
            endpoint_path = f"{path_prefix}/{agent_name}"
            
            try:
                builder.add_agent(
                    agent=agent_instance,
                    path=endpoint_path,
                    tags=["agents", module_name],
                )
                logger_instance.info(
                    f"✓ Registered agent '{agent_name}' from {module_name}.py"
                )
            except Exception as e:
                logger_instance.error(
                    f"✗ Failed to register agent '{agent_name}' from {module_name}.py: {e}"
                )
                if debug:
                    raise
        
        # Build and return the API
        return builder.build()
    
    @staticmethod
    def _discover_agents(
        agent_dir: Path,
        exclude_files: set,
        logger: logging.Logger,
    ) -> List[tuple]:
        """
        Discover AgentBase instances in a directory.
        
        Args:
            agent_dir: Directory to scan
            exclude_files: Set of filenames to exclude
            logger: Logger instance
        
        Returns:
            List of tuples: (agent_instance, module_name)
        """
        discovered = []
        
        # Get all Python files in directory
        python_files = sorted(agent_dir.glob("*.py"))
        
        for py_file in python_files:
            # Skip excluded files
            if py_file.name in exclude_files:
                logger.debug(f"Skipping excluded file: {py_file.name}")
                continue
            
            module_name = py_file.stem
            logger.debug(f"Scanning module: {module_name}")
            
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not load spec for {py_file.name}")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Search for AgentBase instances in module
                for attr_name in dir(module):
                    # Skip private/magic attributes
                    if attr_name.startswith("_"):
                        continue
                    
                    try:
                        attr = getattr(module, attr_name)
                        
                        # Check if it's an AgentBase subclass (not the base class itself)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, AgentBase)
                            and attr is not AgentBase
                        ):
                            logger.debug(
                                f"Found AgentBase subclass: {attr_name} in {module_name}"
                            )
                            
                            # Try to instantiate the agent
                            try:
                                agent_instance = attr()
                                discovered.append((agent_instance, module_name))
                                logger.debug(
                                    f"✓ Instantiated agent: {agent_instance.agent_name}"
                                )
                            except TypeError as e:
                                # Agent requires constructor arguments
                                logger.debug(
                                    f"Cannot auto-instantiate {attr_name}: {e}. "
                                    f"Requires constructor arguments."
                                )
                            except Exception as e:
                                logger.error(
                                    f"✗ Failed to instantiate {attr_name}: {e}",
                                    exc_info=True
                                )
                        
                        # Also check if it's already an instance of AgentBase
                        elif isinstance(attr, AgentBase):
                            logger.debug(
                                f"Found AgentBase instance: {attr_name} in {module_name}"
                            )
                            discovered.append((attr, module_name))
                    
                    except Exception as e:
                        logger.debug(f"Error inspecting {attr_name}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"Failed to import {py_file.name}: {e}")
                continue
        
        return discovered
