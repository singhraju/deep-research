"""
Agent API configuration module
Provides environment-specific API base URLs from config.yaml
"""
from utils.config_loader import get_config_loader

# Global variable to store current environment's API base URL
_API_BASE_URL = None
_CURRENT_ENV = None


def initialize_agent_config(env: str):
    """
    Initialize agent configuration for the given environment
    Must be called before using any agent functions
    
    Args:
        env (str): Environment code (dv, ts, pl, pr)
    """
    global _API_BASE_URL, _CURRENT_ENV
    
    config_loader = get_config_loader()
    _API_BASE_URL = config_loader.get_agent_api_url(env)
    _CURRENT_ENV = env
    
    print(f"✅ Agent API configured for {env.upper()} environment")
    print(f"   Base URL: {_API_BASE_URL}")


def get_api_base_url() -> str:
    """
    Get the configured API base URL
    
    Returns:
        str: API base URL
        
    Raises:
        RuntimeError: If agent config not initialized
    """
    if _API_BASE_URL is None:
        raise RuntimeError(
            "Agent configuration not initialized. "
            "Call initialize_agent_config(env) before using agent functions."
        )
    return _API_BASE_URL


def get_current_env() -> str:
    """
    Get the current environment
    
    Returns:
        str: Current environment code
    """
    return _CURRENT_ENV
