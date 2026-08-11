"""
Base Agent Framework for Deep Research

This module provides the foundational AgentBase class that handles all enterprise
complexity (authentication, connections, graph execution, error handling) so that
developers can focus on implementing business logic.

Quick Start:
    >>> from deep_research_core.base_agent import AgentBase
    >>> 
    >>> class MyAgent(AgentBase):
    >>>     @property
    >>>     def node_name(self) -> str:
    >>>         return "process_question"
    >>>     
    >>>     def node_function(self, state):
    >>>         question = state["question"]
    >>>         # Use _invoke_with_token_retry for automatic 401 retry
    >>>         response = self._invoke_with_token_retry([{"role": "user", "content": question}])
    >>>         return {"result": response.content}
    >>>     
    >>>     def prepare_state(self, question: str, **kwargs):
    >>>         return {"question": question, "llm": self.llm, **kwargs}
    >>>     
    >>>     def extract_result(self, graph_output):
    >>>         return graph_output["result"]
    >>> 
    >>> agent = MyAgent(agent_name="my_agent")
    >>> result = agent(question="What is the capital of France?")
"""

import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict
from deep_research_utils.app_constant import AppConstants

from langgraph.graph import END, START, StateGraph


# ============================================================================
# Exceptions
# ============================================================================


class AgentError(Exception):
    """Base exception for all agent errors."""
    pass


class AgentConfigurationError(AgentError):
    """Raised when agent is misconfigured."""
    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


# ============================================================================
# Credential Provider
# ============================================================================


class CredentialProvider:
    """
    Centralized credential management for all agents.

    Handles EHAP authentication, token refresh, and credential retrieval
    using AppConstants (which manages vault and environment variables).
    Uses singleton pattern to share credentials across all agents.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "CredentialProvider":
        """Get singleton instance of credential provider."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_llm_token(self) -> str:
        """
        Get EHAP token for LLM access with automatic refresh using Redis cache.
        
        Delegates to EHAPBase which handles Redis caching and token management.
        
        Returns:
            Valid EHAP token
            
        Raises:
            AgentConfigurationError: If EHAP credentials are not configured
        """
        try:
            from deep_research_utils import EHAPBase
            
            ehap = EHAPBase(
                base_url=AppConstants.EHAP_BASE_URL,
                client_id=AppConstants.EHAP_CLIENT_ID,
                client_secret=AppConstants.EHAP_CLIENT_SECRET,
                verify=AppConstants.SSL_CERT_FILE or False,
            )
            
            # EHAPBase handles Redis caching and automatic refresh
            token = ehap.get_token()
            if not token:
                raise AgentConfigurationError(
                    "EHAP token not available. Ensure EHAP credentials are set in environment variables:\n"
                    "  - EHAP_BASE_URL\n"
                    "  - EHAP_CLIENT_ID\n"
                    "  - EHAP_CLIENT_SECRET"
                )
            
            return token
            
        except Exception as exc:
            raise AgentConfigurationError(
                f"Failed to get EHAP token: {exc}\n"
                f"Check your EHAP credentials in environment variables."
            ) from exc
    
    def get_snowflake_credentials(self) -> Dict[str, str]:
        """
        Get Snowflake credentials using AppConstants.

        Auto-detects connection type:
        - Vault mode: Returns vault configuration for secure_snowflake_connector
        - Programmatic mode: Returns password-based credentials
        
        Returns:
            Dictionary of Snowflake connection parameters
            
        Raises:
            AgentConfigurationError: If required Snowflake credentials are missing
        """

        connection_type = AppConstants.SNOWFLAKE_CONNECTION_TYPE
        print(f"[CredentialProvider] Snowflake connection type: {connection_type}")

        if connection_type == "vault":
            # Vault-based connection - return vault configuration
            credentials = {
                "service_id": AppConstants.SNOWFLAKE_SERVICE_ID,
                "vault_role_name": AppConstants.VAULT_ROLE_NAME,
                "vault_namespace": AppConstants.VAULT_NAMESPACE,
                "vault_path": AppConstants.VAULT_PATH,
                "vault_url": AppConstants.VAULT_URL,
                "verify_ssl": AppConstants.VERIFY_SSL,
                "cert_path": AppConstants.CERT_PATH,
                "warehouse": AppConstants.SNOWFLAKE_WAREHOUSE,
                "database": AppConstants.SNOWFLAKE_DATABASE,
                "schema": AppConstants.SNOWFLAKE_SCHEMA,
                "account": AppConstants.SNOWFLAKE_ACCOUNT,
            }
            
            # Validate vault configuration
            missing = [
                key for key, value in credentials.items() 
                if value is None or (not isinstance(value, bool) and not value)
            ]
            if missing:
                raise AgentConfigurationError(
                    f"Missing required Snowflake vault credentials: {missing}\n"
                    f"Ensure vault environment variables are set:\n"
                    f"  - SNOWFLAKE_SERVICE_ID\n"
                    f"  - VAULT_ROLE_NAME\n"
                    f"  - VAULT_NAMESPACE\n"
                    f"  - VAULT_PATH\n"
                    f"  - VAULT_URL\n"
                    f"  - VERIFY_SSL\n"
                    f"  - CERT_PATH\n"
                    f"  - SNOWFLAKE_ACCOUNT\n"
                    f"  - SNOWFLAKE_WAREHOUSE\n"
                    f"  - SNOWFLAKE_DATABASE\n"
                    f"  - SNOWFLAKE_SCHEMA"
                )
            
            print(f"[CredentialProvider] Using vault credentials for Snowflake")
            print(f"[CredentialProvider] Vault URL: {credentials['vault_url']}")
            print(f"[CredentialProvider] Service ID: {credentials['service_id']}")
            
        else:
            # Programmatic (password-based) connection
            credentials = {
                "account": AppConstants.SNOWFLAKE_ACCOUNT,
                "user": AppConstants.SNOWFLAKE_USER,
                "password": AppConstants.SNOWFLAKE_SECRET,
                "warehouse": AppConstants.SNOWFLAKE_WAREHOUSE,
                "database": AppConstants.SNOWFLAKE_DATABASE,
                "schema": AppConstants.SNOWFLAKE_SCHEMA,
            }
            
            # Validate programmatic configuration
            missing = [key for key, value in credentials.items() if not value]
            if missing:
                raise AgentConfigurationError(
                    f"Missing required Snowflake credentials: {missing}\n"
                    f"Ensure environment variables are set (local dev):\n"
                    f"  - SNOWFLAKE_ACCOUNT\n"
                    f"  - SNOWFLAKE_USER\n"
                    f"  - SNOWFLAKE_SECRET\n"
                    f"  - SNOWFLAKE_WAREHOUSE\n"
                    f"  - SNOWFLAKE_DATABASE\n"
                    f"  - SNOWFLAKE_SCHEMA"
                )
            
            print(f"[CredentialProvider] Using programmatic credentials for Snowflake")
            if credentials.get("password"):
                print(f"[CredentialProvider] Password length: {len(credentials['password'])} characters")
        
        return credentials


# ============================================================================
# Base Agent Class
# ============================================================================


class AgentBase(ABC):
    """
    Base class for all LangGraph agents.
    
    This class handles all enterprise infrastructure concerns:
    - EHAP authentication for LLM access
    - Resource initialization and lifecycle management
    - LangGraph construction and compilation
    - Error handling and logging
    - Execution lifecycle hooks
    
    Developers only need to implement:
    1. node_name property - name of the primary graph node
    2. node_function() - the core agent logic
    3. prepare_state() - map inputs to graph state
    4. extract_result() - extract final output from graph result
    
    Attributes:
        agent_name: Unique name for this agent
        state_class: TypedDict class defining the graph state structure
        llm: Authenticated LLM client (ChatOpenAI)
        logger: Logger instance for this agent
        graph: LangGraph StateGraph instance
        app: Compiled LangGraph application
    
    Example:
        >>> class MyAgent(AgentBase):
        >>>     @property
        >>>     def node_name(self) -> str:
        >>>         return "process"
        >>>     
        >>>     def node_function(self, state):
        >>>         return {"result": self.llm.invoke(state["question"])}
        >>>     
        >>>     def prepare_state(self, question: str, **kwargs):
        >>>         return {"question": question, "llm": self.llm}
        >>>     
        >>>     def extract_result(self, graph_output):
        >>>         return graph_output["result"]
        >>> 
        >>> agent = MyAgent(agent_name="my_agent")
        >>> result = agent(question="Hello!")
    """
    
    # Version information
    BASE_CLASS_VERSION = "1.0.0"
    MIN_PYTHON_VERSION = (3, 9)
    
    # Default state class for simple agents
    DEFAULT_STATE_CLASS = TypedDict(
        "DefaultState",
        {
            "question": str,
            "context": Optional[Dict[str, Any]],
            "llm": Any,
            "result": Any,
        },
        total=False,
    )
    
    def __init__(
        self,
        agent_name: str,
        state_class: Optional[Type[TypedDict]] = None,
        # LLM configuration
        llm: Optional[Any] = None,
        llm_builder: Optional[Callable[[], Any]] = None,
        llm_model: Optional[str] = None,
        llm_reasoning_effort: str = "medium",
        llm_summary_mode: Optional[str] = None,
        llm_timeout: Optional[float] = None,
        # Graph configuration
        checkpointer: Optional[Any] = None,
        # Operational settings
        test_mode: bool = False,
        debug: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize the agent.
        
        Args:
            agent_name: Unique name for this agent
            state_class: TypedDict class defining graph state (uses default if None)
            llm: Pre-configured LLM instance (optional)
            llm_builder: Function to build LLM instance (optional)
            llm_model: LLM model name (defaults to EHAP_LLM_MODEL env var)
            llm_reasoning_effort: Reasoning effort level ("low", "medium", "high")
            llm_summary_mode: Summary mode ("auto", "detailed", or None)
            llm_timeout: Request timeout in seconds for LLM calls. If None, uses AppConstants.LLM_TIMEOUT (default: 300s)
            checkpointer: LangGraph checkpointer for state persistence (optional)
            test_mode: If True, uses stub LLM instead of real one
            debug: If True, enables verbose logging and tracing
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
        Raises:
            AgentConfigurationError: If configuration is invalid
            RuntimeError: If Python version or dependencies are incompatible
        """
        # Validate environment
        self._check_python_version()
        self._check_dependencies()
        
        # Store configuration
        self.agent_name = agent_name
        self.state_class = state_class or self.DEFAULT_STATE_CLASS
        self.checkpointer = checkpointer
        self.test_mode = test_mode
        self.debug = debug
        
        # LLM configuration
        self.llm_model = llm_model or os.environ.get("EHAP_LLM_MODEL", "gpt-5.4")
        self.llm_reasoning_effort = llm_reasoning_effort
        self.llm_summary_mode = llm_summary_mode
        self.llm_timeout = llm_timeout  # Timeout for LLM invocations
        self._llm_builder = llm_builder  # Store for custom LLM builder
        self.ehap = None  # EHAP instance for token management (data viz approach)
        self.llm = None  # Direct LLM attribute (no longer a property)
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if debug:
            # Enable LangChain tracing in debug mode
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            self.logger.debug("Debug mode enabled - LangChain tracing active")
        
        self.logger.info(f"Initializing {agent_name} agent (v{self.BASE_CLASS_VERSION})")
        self.logger.info(f"Configuration: test_mode={test_mode}, debug={debug}, log_level={log_level}")
        
        # Initialize resources
        try:
            self.logger.info("Initializing LLM...")
            
            if llm is not None:
                # Use provided LLM instance (no token management)
                self.llm = llm
                self.ehap = None
                self.logger.info(f"Using provided LLM instance: model={self.llm_model}")
            elif self.test_mode:
                # Test mode - use stub LLM
                self.llm = self.create_stub_llm()
                self.ehap = None
                self.logger.info("Test mode enabled - using stub LLM")
            else:
                # DATA VIZ APPROACH: Initialize EHAP for token management
                from deep_research_utils import EHAPBase
                
                self.ehap = EHAPBase(
                    base_url=AppConstants.EHAP_BASE_URL,
                    client_id=AppConstants.EHAP_CLIENT_ID,
                    client_secret=AppConstants.EHAP_CLIENT_SECRET,
                    verify=AppConstants.SSL_CERT_FILE or False,
                )
                
                # Initialize LLM with EHAP token
                if self._llm_builder is not None:
                    self.llm = self._llm_builder()
                    self.logger.info(f"LLM initialized with custom builder: model={self.llm_model}")
                else:
                    self.llm = self._initialize_llm()
                    self.logger.info(f"LLM initialized with EHAP authentication: model={self.llm_model}")
            
            self.logger.debug("Loading static resources...")
            self._static_resources = self._load_static_resources()
            self.logger.debug(f"Static resources loaded: {list(self._static_resources.keys())}")
            
            # Load agent configuration from .ini file
            self.logger.debug("Loading agent configuration...")
            self._agent_config = self._load_agent_config()
            self.logger.debug(f"Agent config loaded: {len(self._agent_config)} items for environment '{AppConstants.ENV}'")
        except Exception as exc:
            self.logger.error(f"Resource initialization failed: {exc}")
            raise AgentConfigurationError(
                f"Failed to initialize resources for {agent_name}: {exc}"
            ) from exc
        
        # Build and compile graph
        try:
            self.logger.info("Building LangGraph...")
            self.graph = self.build_graph()
            self.logger.debug(f"Graph structure: {len(self.graph.nodes)} nodes")
            
            self.logger.debug("Compiling graph...")
            self.app = self._compile_graph()
            self.logger.info("Graph compiled successfully")
        except Exception as exc:
            self.logger.error(f"Graph building failed: {exc}")
            raise AgentConfigurationError(
                f"Failed to build graph for {agent_name}: {exc}"
            ) from exc
        
        self.logger.info(f"✓ {agent_name} agent initialized successfully")
    
    # ========================================================================
    # Resource Management
    # ========================================================================
    
    def _initialize_llm(self) -> Any:
        """
        Initialize LLM with current EHAP token (DATA VIZ APPROACH).
        
        Returns:
            Initialized ChatOpenAI instance
            
        Raises:
            AgentConfigurationError: If EHAP token cannot be obtained
        """
        from langchain_openai import ChatOpenAI
        
        try:
            token = self.ehap.get_token()
        except Exception as exc:
            raise AgentConfigurationError(
                f"Failed to get EHAP token: {exc}\n"
                f"Check your EHAP credentials in environment variables."
            ) from exc
        
        # Build extra_body parameters
        extra_body = {}
        if self.llm_reasoning_effort:
            extra_body["reasoning_effort"] = self.llm_reasoning_effort
        if self.llm_summary_mode:
            extra_body["summary"] = self.llm_summary_mode
        
        return ChatOpenAI(
            base_url=AppConstants.OPENAI_BASE_URL,
            model=self.llm_model,
            api_key=token,
            extra_body=extra_body if extra_body else None,
            http_client=AppConstants.http_client_,
            http_async_client=AppConstants.http_async_client_,
        )
    
    def _invoke_with_token_retry(self, messages: list, timeout: Optional[float] = None, **invoke_kwargs) -> Any:
        """
        Invoke the LLM with automatic token refresh on 401 AuthenticationError.
        
        This method delegates to the ehap_retry utility which uses tenacity
        for robust retry logic with automatic token refresh.
        
        This method should be used instead of self.llm.invoke() to ensure
        automatic token refresh on authentication errors.
        
        Args:
            messages: Messages to send to LLM (LangChain format)
            timeout: Request timeout in seconds. If None, uses AppConstants.LLM_TIMEOUT (default: 300s)
            **invoke_kwargs: Additional arguments to pass to llm.invoke()
        
        Returns:
            LLM response
            
        Raises:
            AuthenticationError: If retry also fails
            httpx.TimeoutException: If request exceeds timeout duration
            
        Example:
            >>> response = self._invoke_with_token_retry([
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "Hello"}
            ... ])
            >>> # With custom timeout
            >>> response = self._invoke_with_token_retry(
            ...     messages=[{"role": "user", "content": "Complex query"}],
            ...     timeout=600.0  # 10 minutes
            ... )
        """
        from deep_research_utils.ehap_retry import llm_invoke
        
        # Skip token management if using pre-configured LLM or test mode
        if self.ehap is None:
            return self.llm.invoke(messages, **invoke_kwargs)
        
        # Resolve timeout: explicit parameter > agent default > AppConstants
        effective_timeout = timeout if timeout is not None else self.llm_timeout
        
        # Use the tenacity-based retry utility
        # llm_invoke returns (result, updated_llm) to capture token refresh updates
        result, updated_llm = llm_invoke(
            llm=self.llm,
            ehap=self.ehap,
            messages=messages,
            llm_reinitializer=self._initialize_llm,
            timeout=effective_timeout,
            **invoke_kwargs
        )
        
        # Update self.llm to capture any token refresh that occurred
        self.llm = updated_llm
        
        return result
    
    def _load_static_resources(self) -> Dict[str, Any]:
        """
        Load static resources (semantic models, configs, etc.).
        
        Override this method to load resources that should be initialized
        once and reused across all executions (e.g., semantic models,
        configuration files, lookup tables).
        
        Returns:
            Dictionary of resource name -> resource object
        
        Example:
            >>> def _load_static_resources(self):
            >>>     return {
            >>>         "semantic_model": load_semantic_yaml(self.yaml_path),
            >>>         "config": load_config(self.config_path),
            >>>     }
        """
        return {}
    
    def _load_agent_config(self) -> Dict[str, str]:
        """
        Load agent configuration for current environment from .ini file.
        
        Configuration is loaded from environment-specific .ini files in the configs/
        directory. Each agent has its own section with key-value pairs for
        agent-specific settings (tables, paths, parameters, etc.).
        
        Returns:
            Dictionary of config key -> value
            
        Example:
            >>> # For reimbursement_agent in dev environment
            >>> config = self._load_agent_config()
            >>> config["policy_metadata"]
            'D01_COC.COC_DTI.PLCY_MTDTA'
            
            >>> # For correlation agent in dev environment
            >>> config = self._load_agent_config()
            >>> config["semantic_config_path"]
            'configs/correlation_pattern/coc_ecap_ip_auth_sematic_view_with_samples_dev.yaml'
        """
        try:
            config = AppConstants.load_agent_config()
            
            # Get agent-specific section (e.g., "reimbursement_agent", "correlation")
            agent_section = self.agent_name.lower()
            
            if agent_section not in config:
                self.logger.warning(
                    f"No configuration section [{agent_section}] found in "
                    f"{AppConstants.ENV}.ini. Agent will have no configuration."
                )
                return {}
            
            # Convert ConfigParser section to dictionary
            agent_config = dict(config[agent_section])
            self.logger.debug(
                f"Loaded {len(agent_config)} config item(s) from [{agent_section}] section: "
                f"{', '.join(agent_config.keys())}"
            )
            return agent_config
            
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load agent configuration: {e}")
            return {}
    
    def config(self, name: str, default: Optional[str] = None) -> str:
        """
        Get configuration value for the current environment.
        
        This method provides environment-aware configuration access. The same
        configuration key automatically resolves to different values based on
        the deployment environment (dev, uat, prod, local).
        
        Args:
            name: Configuration key (e.g., 'semantic_config_path', 'policy_metadata')
            default: Optional default value if config not found
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If config key not found and no default provided
            
        Examples:
            >>> # Get semantic model path for correlation agent
            >>> self.config('semantic_config_path')
            'configs/correlation_pattern/coc_ecap_ip_auth_sematic_view_with_samples_dev.yaml'
            
            >>> # Get table name for reimbursement agent
            >>> self.config('policy_metadata')
            'D01_COC.COC_DTI.PLCY_MTDTA'
            
            >>> # With default fallback
            >>> self.config('custom_key', default='default_value')
            'default_value'
        """
        if name in self._agent_config:
            return self._agent_config[name]
        
        if default is not None:
            self.logger.warning(
                f"Config '{name}' not found for agent '{self.agent_name}' in environment '{AppConstants.ENV}'. "
                f"Using default: {default}"
            )
            return default
        
        available = ', '.join(sorted(self._agent_config.keys()))
        raise KeyError(
            f"Config '{name}' not configured for agent '{self.agent_name}' in environment '{AppConstants.ENV}'. "
            f"Available config keys: {available}"
        )
    
    def register_config(self, name: str, value: str) -> None:
        """
        Register a custom configuration value for this agent instance.
        
        Useful for runtime configuration or agent-specific overrides.
        
        Args:
            name: Configuration key
            value: Configuration value
            
        Example:
            >>> self.register_config('custom_path', '/path/to/file.yaml')
            >>> self.config('custom_path')
            '/path/to/file.yaml'
        """
        self._agent_config[name] = value
        self.logger.debug(f"Registered config '{name}' -> {value}")
    
    def list_config(self) -> Dict[str, str]:
        """
        Get all configuration values for this agent.
        
        Returns:
            Dictionary of config key -> value
            
        Example:
            >>> config = self.list_config()
            >>> for key, value in config.items():
            >>>     print(f"{key}: {value}")
            semantic_config_path: configs/correlation_pattern/...
            policy_metadata: D01_COC.COC_DTI.PLCY_MTDTA
        """
        return dict(self._agent_config)
    
    
    def build_default_llm(self) -> Any:
        """
        Build default LLM with EHAP authentication.
        
        DEPRECATED: This method is kept for backward compatibility.
        New code should use _initialize_llm() instead.
        
        Override this method to customize LLM initialization for your agent.
        
        Returns:
            Initialized LLM client
            
        Raises:
            AgentConfigurationError: If EHAP credentials are not available
        """
        # For backward compatibility, delegate to _initialize_llm if using EHAP
        if self.ehap is not None:
            return self._initialize_llm()
        
        # Fallback to CredentialProvider for agents that override this method
        from langchain_openai import ChatOpenAI
        
        creds = CredentialProvider.get_instance()
        token = creds.get_llm_token()
        
        extra_body = {}
        if self.llm_reasoning_effort:
            extra_body["reasoning_effort"] = self.llm_reasoning_effort
        if self.llm_summary_mode:
            extra_body["summary"] = self.llm_summary_mode
        
        return ChatOpenAI(
            base_url=AppConstants.OPENAI_BASE_URL,
            model=self.llm_model,
            api_key=token,
            extra_body=extra_body if extra_body else None,
            http_client=AppConstants.http_client_,
            http_async_client=AppConstants.http_async_client_,
        )
    
    def create_stub_llm(self) -> Any:
        """
        Create stub LLM for testing.
        
        Override this method to provide a test stub that returns
        agent-specific mock responses.
        
        Returns:
            Stub LLM instance
            
        Raises:
            NotImplementedError: If test_mode is used but stub not implemented
        """
        raise NotImplementedError(
            f"{self.agent_name} does not implement create_stub_llm(). "
            f"Either provide an LLM instance or implement create_stub_llm() for test mode."
        )
    
    # ========================================================================
    # Graph Construction
    # ========================================================================
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        The default implementation creates a simple single-node graph:
        START -> node_name -> END
        
        Override this method for complex multi-node graphs with conditional
        routing, loops, or parallel execution.
        
        Returns:
            Configured StateGraph instance
        
        Example (custom graph):
            >>> def build_graph(self):
            >>>     graph = StateGraph(self.state_class)
            >>>     graph.add_node("step1", self.step1_function)
            >>>     graph.add_node("step2", self.step2_function)
            >>>     graph.add_edge(START, "step1")
            >>>     graph.add_conditional_edges("step1", self.route_function)
            >>>     graph.add_edge("step2", END)
            >>>     return graph
        """
        graph = StateGraph(self.state_class)
        graph.add_node(self.node_name, self.node_function)
        graph.add_edge(START, self.node_name)
        graph.add_edge(self.node_name, END)
        return graph
    
    def _compile_graph(self):
        """
        Compile the graph with optional checkpointing.
        
        Returns:
            Compiled LangGraph application
        """
        if self.checkpointer:
            self.logger.debug("Compiling graph with checkpointer")
            return self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.logger.debug("Compiling graph without checkpointer")
            return self.graph.compile()
    
    @property
    @abstractmethod
    def node_name(self) -> str:
        """
        Name of the primary graph node.
        
        For simple single-node agents, this is the only node name.
        For complex agents, this should be the main entry node.
        
        Returns:
            Node name string
        
        Example:
            >>> @property
            >>> def node_name(self) -> str:
            >>>     return "process_question"
        """
        pass
    
    @abstractmethod
    def node_function(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary node function implementing agent logic.
        
        This is where your agent's core logic lives. The function receives
        the current state and must return a dictionary of state updates.
        
        Args:
            state: Current graph state (matches state_class structure)
        
        Returns:
            Dictionary of state updates (will be merged into state)
        
        Example:
            >>> def node_function(self, state):
            >>>     question = state["question"]
            >>>     llm = state["llm"]
            >>>     
            >>>     response = llm.invoke([
            >>>         {"role": "user", "content": question}
            >>>     ])
            >>>     
            >>>     return {"result": response.content}
        """
        pass
    
    # ========================================================================
    # Execution Interface
    # ========================================================================
    
    def execute(self, **kwargs) -> Any:
        """
        Execute the agent with provided inputs.
        
        This is the main entry point for running your agent. The base class
        handles all infrastructure concerns:
        - Pre/post execution hooks
        - State preparation
        - Graph execution
        - Result validation
        - Error handling
        
        You just need to implement prepare_state() and extract_result().
        
        Args:
            **kwargs: Flexible keyword arguments passed to prepare_state()
        
        Returns:
            Final result extracted by extract_result()
        
        Raises:
            AgentConfigurationError: If agent is misconfigured
            AgentExecutionError: If execution fails
        
        Example:
            >>> agent = MyAgent(agent_name="my_agent")
            >>> result = agent.execute(question="What is 2+2?")
            >>> print(result)  # "4"
        """
        self.logger.info(f"Executing {self.agent_name} agent")
        
        if self.debug:
            self.logger.debug(f"Input kwargs: {list(kwargs.keys())}")
            self.logger.debug(f"LLM available: {self.llm is not None}")
        
        try:
            # Pre-execution hook
            self.logger.info(f"Starting {self.agent_name} execution")
            self.logger.debug(f"Input parameters: {list(kwargs.keys())}")
            self.on_before_execute(**kwargs)
            
            try:
                # Prepare state
                self.logger.debug("Preparing graph state...")
                state = self.prepare_state(**kwargs)
                
                if self.debug:
                    self.logger.debug(f"Initial state keys: {list(state.keys())}")
                
                # Validate state has required structure
                self._validate_state(state)
                self.logger.debug("State validation passed")
                
                # Execute graph
                self.logger.info("Executing LangGraph pipeline...")
                result = self.app.invoke(state)
                
                if self.debug:
                    self.logger.debug(f"Raw graph output keys: {list(result.keys())}")
                
                # Post-execution hook
                self.on_after_execute(result)
                
                # Validate result
                self.logger.debug("Validating result...")
                warnings_list = self.validate_result(result)
                if warnings_list:
                    self.logger.warning(f"Found {len(warnings_list)} validation warnings")
                    for warning in warnings_list:
                        self.logger.warning(f"  - {warning}")
                else:
                    self.logger.debug("Result validation passed")
                
                # Extract and return final result
                self.logger.debug("Extracting final result...")
                final_result = self.extract_result(result)
                
                self.logger.info(f" {self.agent_name} execution completed successfully")
                return final_result
                
            except Exception as exc:
                self.logger.error(f" {self.agent_name} execution failed: {type(exc).__name__}")
                self.logger.exception(
                    f"Exception details: {exc}",
                    exc_info=exc
                )
                return self.handle_execution_error(exc, **kwargs)
        
        except Exception as exc:
            self.logger.exception(
                f"{self.agent_name} execution failed: {exc}",
                exc_info=exc
            )
            return self.handle_execution_error(exc, **kwargs)
    
    # ... (rest of the class remains the same)
    def prepare_state(self, **kwargs) -> Dict[str, Any]:
        """
        Convert execution inputs to graph state.
        
        This method maps the flexible kwargs passed to execute() into the
        structured state dictionary expected by the graph. Always include
        resources like self.llm and self._static_resources.
        
        Args:
            **kwargs: Flexible keyword arguments from execute()
        
        Returns:
            State dictionary matching state_class structure
        
        Example:
            >>> def prepare_state(self, question: str, context: Optional[Dict] = None):
            >>>     return {
            >>>         "question": question,
            >>>         "context": context,
            >>>         "llm": self.llm,
            >>>         **self._static_resources,
            >>>     }
        """
        pass
    
    @abstractmethod
    def extract_result(self, graph_output: Dict[str, Any]) -> Any:
        """
        Extract final result from graph output.
        
        The graph returns the full state dictionary. This method extracts
        just the final result that should be returned to the user.
        
        Args:
            graph_output: Full state dictionary after graph execution
        
        Returns:
            Final result to return to user
        
        Example:
            >>> def extract_result(self, graph_output):
            >>>     return graph_output["result"]
        """
        pass
    
    def _validate_state(self, state: Dict[str, Any]):
        """
        Validate that state has required structure.
        
        Args:
            state: State dictionary to validate
            
        Raises:
            AgentConfigurationError: If state is invalid
        """
        if not isinstance(state, dict):
            raise AgentConfigurationError(
                f"prepare_state() must return a dictionary, got {type(state)}"
            )
        
        # Check for common mistakes
        if "llm" not in state and self.llm is not None:
            self.logger.warning(
                "State does not include 'llm' key. "
                "Did you forget to add 'llm': self.llm in prepare_state()?"
            )
    
    # ========================================================================
    # Hooks and Customization Points
    # ========================================================================
    
    def on_before_execute(self, **kwargs):
        """
        Hook called before execution starts.
        
        Override this method to add custom pre-execution logic like
        input validation, logging, metrics collection, etc.
        
        Args:
            **kwargs: Input kwargs passed to execute()
        
        Example:
            >>> def on_before_execute(self, **kwargs):
            >>>     if "question" not in kwargs:
            >>>         raise ValueError("question is required")
            >>>     self.metrics.start_timer()
        """
        pass
    
    def on_after_execute(self, result: Dict[str, Any]):
        """
        Hook called after execution completes.
        
        Override this method to add custom post-execution logic like
        result logging, metrics collection, cleanup, etc.
        
        Args:
            result: Full graph output state
        
        Example:
            >>> def on_after_execute(self, result):
            >>>     self.metrics.stop_timer()
            >>>     self.logger.info(f"Execution took {self.metrics.duration}s")
        """
        pass
    
    def validate_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Validate execution result.
        
        Override this method to add custom validation logic. Return a list
        of warning messages (empty list if no warnings).
        
        Args:
            result: Full graph output state
        
        Returns:
            List of warning messages (empty if valid)
        
        Example:
            >>> def validate_result(self, result):
            >>>     warnings = []
            >>>     if "result" not in result:
            >>>         warnings.append("Missing 'result' key in output")
            >>>     if result.get("confidence", 1.0) < 0.5:
            >>>         warnings.append("Low confidence result")
            >>>     return warnings
        """
        return []
    
    def handle_execution_error(self, exc: Exception, **kwargs) -> Any:
        """
        Handle execution errors.
        
        Override this method to provide custom error handling like
        returning default values, retrying, or transforming exceptions.
        
        The default implementation re-raises the exception.
        
        Args:
            exc: The exception that occurred
            **kwargs: Original input kwargs
        
        Returns:
            Fallback result (if not re-raising)
        
        Raises:
            Exception: Re-raises the exception by default
        
        Example:
            >>> def handle_execution_error(self, exc, **kwargs):
            >>>     self.logger.error(f"Execution failed: {exc}")
            >>>     return {"error": str(exc), "result": None}
        """
        raise exc
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def __call__(self, **kwargs) -> Any:
        """
        Allow agent to be called as a function.
        
        This is syntactic sugar for execute().
        
        Args:
            **kwargs: Arguments passed to execute()
        
        Returns:
            Result from execute()
        
        Example:
            >>> agent = MyAgent(agent_name="my_agent")
            >>> result = agent(question="Hello!")  # Same as agent.execute(question="Hello!")
        """
        return self.execute(**kwargs)
    
    # ========================================================================
    # Version and Dependency Checks
    # ========================================================================
    
    def _check_python_version(self):
        """Check Python version compatibility."""
        if sys.version_info < self.MIN_PYTHON_VERSION:
            raise RuntimeError(
                f"AgentBase requires Python {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}+, "
                f"but you have {sys.version_info.major}.{sys.version_info.minor}"
            )
    
    def _check_dependencies(self):
        """Check required dependencies are installed."""
        required = {
            "langgraph": "langgraph",
            "langchain_openai": "langchain-openai",
            "deep_research_utils": "deep-research-utils",
        }
        
        missing = []
        for module_name, package_name in required.items():
            try:
                __import__(module_name)
            except ImportError:
                missing.append(package_name)
        
        if missing:
            raise RuntimeError(
                f"Missing required dependencies: {missing}\n"
                f"Install with: pip install {' '.join(missing)}"
            )
