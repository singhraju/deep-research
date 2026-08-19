"""Environment-specific vault configuration for Snowflake connections."""

from typing import Dict, Any, Final

# Environment-specific Vault Configuration
VAULT_CONFIGS: Final[Dict[str, Dict[str, Any]]] = {
    "dev": {
        "service_id": "srccocdtidd",
        "vault_role_name": "snowdb-idiscovery-slvr",
        "vault_namespace": "eda-snowflakedb",
        "vault_path": "eda_nonprod/static-creds/aedl_devops-srccocdtidd",
        "vault_url": "https://vault.acr.awsdns.internal.das",
        "verify_ssl": True,
        "warehouse": "D01_COC_DTI_LOAD_WH",
        "schema": "COC_DTI_stg",
        "database": "D01_COC",
        "account": "carelon-eda-nonprod.privatelink",
    },
    "uat": {
        "service_id": "srccocdtidu",
        "vault_role_name": "snowdb-idiscovery-gld",
        "vault_namespace": "eda-snowflakedb",
        "vault_path": "eda_preprod/static-creds/aedl_devops-srccocdtidu",
        "vault_url": "https://vault.acr.awsdns.internal.das",
        "verify_ssl": True,
        "warehouse": "U01_COC_DTI_LOAD_WH",
        "schema": "COC_DTI_stg",
        "database": "U01_COC",
        "account": "carelon-eda-preprod.privatelink",
    },
    "prod": {
        "service_id": "srccocdtidp",
        "vault_role_name": "snowdb-idiscovery-prod",
        "vault_namespace": "eda-snowflakedb",
        "vault_path": "edaprod/static-creds/aedl_devops-srccocdtidp",
        "vault_url": "https://vault.acr.awsdns.internal.das",
        "verify_ssl": True,
        "warehouse": "P01_COC_DTI_LOAD_WH",
        "schema": "COC_DTI_stg",
        "database": "P01_COC",
        "account": "carelon-edaprod1.privatelink",
    },
}


def get_vault_config(environment: str) -> Dict[str, Any]:
    """
    Get vault configuration for the specified environment.
    
    Args:
        environment: Environment name (dev, uat, prod)
        
    Returns:
        Dictionary containing vault configuration for the environment
        
    Raises:
        ValueError: If environment is not recognized
    """
    env = environment.lower()
    if env not in VAULT_CONFIGS:
        raise ValueError(
            f"Unknown environment: {environment}. "
            f"Valid options: {', '.join(VAULT_CONFIGS.keys())}"
        )
    return VAULT_CONFIGS[env]