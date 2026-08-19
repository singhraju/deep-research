import os
import configparser
from pathlib import Path
from typing import Final, Optional
import logging

import httpx
from dotenv import load_dotenv

from .vault_config import VAULT_CONFIGS

# Load environment variables from .env file
load_dotenv()

# Load secrets from Vault-style file if available
SECRETS_PATH = os.environ.get("SECRETS_PATH", "/vault/secrets/creds")
_secrets = configparser.ConfigParser()
_secrets.read(SECRETS_PATH)

# Set up logger for this module
logger = logging.getLogger(__name__)

# Check which EHAP credentials are available from vault vs environment
_ehap_keys = ["EHAP_BASE_URL", "EHAP_CLIENT_ID", "EHAP_CLIENT_SECRET"]
_vault_found = []
_env_found = []
_missing = []

for key in _ehap_keys:
    has_vault = _secrets.has_option("default", key)
    has_env = key in os.environ
    
    if has_vault:
        _vault_found.append(key)
        logger.info(f"{key} will be read from vault secrets")
    elif has_env:
        _env_found.append(key)
        logger.info(f"{key} not in vault, falling back to environment variable")
    else:
        _missing.append(key)
        logger.warning(f"{key} not found in vault or environment - will be empty string")

if _vault_found:
    logger.info(f"Vault secrets loaded: {', '.join(_vault_found)}")
if _env_found:
    logger.info(f"Environment fallback used: {', '.join(_env_found)}")
if _missing:
    logger.error(f"Missing credentials: {', '.join(_missing)}")

class AppConstants:
    """
    Application constants for centralized environment variable management.

    This class provides a single source of truth for all environment variables
    used throughout the deep-research project. By centralizing these constants,
    we ensure consistent access patterns and easier maintenance.

    Configuration Priority:
    1. Vault secrets file (/vault/secrets/creds) - for sensitive credentials
    2. Environment variables - for configuration and local development
    3. Defaults - for optional settings

    Environment Variables:
    - Required variables (without `.get`) will fail loudly if not set
    - Optional variables (with `.get`) have sensible defaults
    """
    # Application Environment
    ENV: Final[str] = os.environ.get("ENVIRONMENT", "dev").lower()
    
    # EHAP Configuration (Required)
    EHAP_BASE_URL: Final[str] = _secrets.get("default", "EHAP_BASE_URL", fallback=os.environ.get("EHAP_BASE_URL", ""))
    EHAP_CLIENT_ID: Final[str] = _secrets.get("default", "EHAP_CLIENT_ID", fallback=os.environ.get("EHAP_CLIENT_ID", ""))
    EHAP_CLIENT_SECRET: Final[str] = _secrets.get("default", "EHAP_CLIENT_SECRET", fallback=os.environ.get("EHAP_CLIENT_SECRET", ""))
    
    # LLM Configuration (Optional)
    EHAP_LLM_MODEL: Final[str] = os.environ.get("EHAP_LLM_MODEL", "gpt-5.4")
    DEEP_RESEARCH_LLM_MODEL: Final[str] = os.environ.get("DEEP_RESEARCH_LLM_MODEL", "gpt-5.4")
    
    # SSL/TLS Configuration (Optional)
    SSL_CERT_FILE: Final[str] = os.environ.get("SSL_CERT_FILE", "")
    
    _env_config = VAULT_CONFIGS.get(ENV, VAULT_CONFIGS["dev"])

    # Snowflake Configuration
    SNOWFLAKE_ACCOUNT: Final[str] = os.environ.get("SNOWFLAKE_ACCOUNT", _env_config["account"])
    SNOWFLAKE_USER: Final[str] = os.environ.get("SNOWFLAKE_USER", "")
    SNOWFLAKE_SECRET: Final[str] = os.environ.get("SNOWFLAKE_SECRET", "")
    
    SNOWFLAKE_ROLE: Final[str] = os.environ.get("SNOWFLAKE_ROLE", "")
    SNOWFLAKE_WAREHOUSE: Final[str] = os.environ.get("SNOWFLAKE_WAREHOUSE", _env_config["warehouse"])
    SNOWFLAKE_DATABASE: Final[str] = os.environ.get("SNOWFLAKE_DATABASE", _env_config["database"])
    SNOWFLAKE_SCHEMA: Final[str] = os.environ.get("SNOWFLAKE_SCHEMA", _env_config["schema"])
    
    # Snowflake Table Configuration (Optional)
    SNOWFLAKE_TABLE_PREFIX: Final[str] = os.environ.get("SNOWFLAKE_TABLE_PREFIX", "")
    SNOWFLAKE_TABLE_SUFFIX: Final[str] = os.environ.get("SNOWFLAKE_TABLE_SUFFIX", "")

    # Vault-based Snowflake Configuration (environment-aware defaults)
    SNOWFLAKE_SERVICE_ID: Final[str] = os.environ.get("SNOWFLAKE_SERVICE_ID", _env_config["service_id"])
    VAULT_ROLE_NAME: Final[str] = os.environ.get("VAULT_ROLE_NAME", _env_config["vault_role_name"])
    VAULT_NAMESPACE: Final[str] = os.environ.get("VAULT_NAMESPACE", _env_config["vault_namespace"])
    VAULT_PATH: Final[str] = os.environ.get("VAULT_PATH", _env_config["vault_path"])
    VAULT_URL: Final[str] = os.environ.get("VAULT_URL", _env_config["vault_url"])
    VERIFY_SSL: Final[bool] = (os.environ.get("VERIFY_SSL", str(_env_config["verify_ssl"]))).lower() in ("true", "1", "yes")
    CERT_PATH: Final[str] = os.environ.get("CERT_PATH", "root_chain.pem")

    _use_programmatic = bool(SNOWFLAKE_SECRET)

    SNOWFLAKE_CONNECTION_TYPE: Final[str] = "programmatic" if _use_programmatic else "vault"

    # Logging Configuration (Optional)
    DEEP_RESEARCH_ENABLE_CONSOLE_LOGGING: Final[str] = os.environ.get("DEEP_RESEARCH_ENABLE_CONSOLE_LOGGING", "true")
    DEEP_RESEARCH_CONSOLE_LOG_LEVEL: Final[str] = os.environ.get("DEEP_RESEARCH_CONSOLE_LOG_LEVEL", "INFO")

    OPENAI_BASE_URL: Final[str] = os.environ["OPENAI_BASE_URL"]
    
    # Output Configuration (Optional)
    CORRELATION_OUTPUT_ROOT: Final[str] = os.environ.get("CORRELATION_OUTPUT_ROOT", "/tmp/correlation_runs")
    
    # Decision Tree Rules Configuration
    # Compute path relative to project root: packages/utils/src/deep_research_utils/ -> configs/
    DTR_RULES_PATH: Final[str] = str(
        Path(__file__).parent.parent.parent.parent.parent / "configs" / "decision_tree_rules.yaml"
    )

    CACHE_TTL: Final[int] = int(os.environ.get("CACHE_TTL", str(30 * 60)))  # 30 minutes default
    TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", None)

    # Redis Configuration (Optional)
    REDIS_ENABLED: Final[bool] = os.environ.get("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
    REDIS_HOST: Final[str] = os.environ.get("REDIS_HOST", "redis-cache-service")
    REDIS_PORT: Final[int] = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Final[str] = os.environ.get("REDIS_PASSWORD", "")
    REDIS_DB: Final[int] = int(os.environ.get("REDIS_DB", "0"))
    REDIS_SOCKET_CONNECT_TIMEOUT: Final[int] = int(os.environ.get("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    REDIS_SOCKET_TIMEOUT: Final[int] = int(os.environ.get("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_KEY_PREFIX: Final[str] = os.environ.get("REDIS_KEY_PREFIX", "deep_research:")
    REDIS_TOKEN_KEY: Final[str] = os.environ.get("REDIS_TOKEN_KEY", "OPENAI_API_KEY_IDISCOVERY")


    http_client_ = httpx.Client(verify=False, timeout=60.0)
    http_async_client_ = httpx.AsyncClient(verify=False, timeout=60.0)

    @classmethod
    def load_agent_config(cls) -> configparser.ConfigParser:
        """
        Load agent configuration from environment-specific .ini file.
        
        Reads the appropriate .ini file from the configs/ directory based on
        the current environment (dev, uat, prod, local). These files contain
        agent-specific configurations including table names, paths, and settings.
        
        Returns:
            ConfigParser instance with environment-specific agent configurations
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            
        Example:
            >>> config = AppConstants.load_agent_config()
            >>> config['reimbursement_agent']['policy_metadata']
            'D01_COC.COC_DTI.PLCY_MTDTA'  # in dev environment
        """
        from pathlib import Path
        
        # Determine config file path relative to project root
        # app_constant.py is in packages/utils/src/deep_research_utils/
        config_dir = Path(__file__).parent.parent.parent.parent.parent / "configs"
        config_file = config_dir / f"{cls.ENV}.ini"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Expected environment: {cls.ENV}\n"
                f"Available environments: dev, uat, prod, local"
            )
        
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
