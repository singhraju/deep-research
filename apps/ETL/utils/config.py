"""
Enterprise-level configuration management for DR ETL pipeline
"""

from typing import Dict, Any

# API Configuration
#API_BASE_URL = "https://idiscovery-deep-research-api.nginx.slvr-dig-sharedservdigital2.awsdns.internal.das"
#API_BASE_URL = "https://idiscovery-deep-research-api-uat.istio.carelon.com"
API_BASE_URL = "http://idiscovery-deep-research-api.nginx.plat-dig-sharedservdigital2.awsdns.internal.das"

def get_api_endpoints(env: str) -> Dict[str, str]:
    """
    Get API endpoints for the given environment
    
    Args:
        env: Environment identifier
        
    Returns:
        Dict[str, str]: API endpoints mapping
    """
    endpoints = {
        'health': f"{API_BASE_URL}/health",
        'deep_research': f"{API_BASE_URL}/api/deep-research",
        'insights': f"{API_BASE_URL}/api/insights",
        'pattern_agent': f"{API_BASE_URL}/agents/pattern_agent",
        'correlation_agents': f"{API_BASE_URL}/agents/correlation",
        'reimbursement_agent': f"{API_BASE_URL}/agents/reimbursement_policy",        
        'recommendation_agent': f"{API_BASE_URL}/agents/recommendation_synthesis"        
    }
    return endpoints


def get_environment_config(env: str, lob: str = "nogbd") -> Dict[str, Any]:
    """
    Get environment-specific configuration with dynamic LOB support
    
    Args:
        env: Environment identifier (dv, ts, pl, pr)
        lob: Line of business (gbd, nogbd)
        
    Returns:
        Dict[str, Any]: Environment configuration
    """
    
    # Base environment configurations
    base_configs = {
        'dv': {
            'schema': 'D01',
            'sf_url': 'carelon-eda-nonprod.privatelink'
        },
        'ts': {
            'schema': 'T01',
            'sf_url': 'carelon-eda-nonprod.privatelink'
        },
        'pl': {
            'schema': 'U01',
            'sf_url': 'carelon-eda-preprod.privatelink'
        },
        'pr': {
            'schema': 'P01',
            'sf_url': 'carelon-edaprod1.privatelink'
        }
    }

    if env not in base_configs:
        raise ValueError(f"Invalid environment '{env}'. Must be one of: {list(base_configs.keys())}")
    
    # Validate LOB parameter
    if lob.lower() not in ["gbd", "nogbd"]:
        raise ValueError(f"Invalid lob '{lob}'. Must be 'gbd' or 'nogbd'")
    
    # Build dynamic configuration
    base_config = base_configs[env]
    schema_prefix = base_config['schema'].lower()
    secret_name = f"eai_aifs_cai_coc_{env}_{lob}_secret_manager_snflk_dti_sdlc"
    
    # Build secret name and target schema based on LOB
    if lob.lower() == "gbd":
        target_schema = "COC_DTI_STG"
    else:  # nogbd
        target_schema = "COC_DTI_STG_NOGBD"
    
    # Return complete configuration
    config = {
        'schema': base_config['schema'],
        'sf_url': base_config['sf_url'],
        'secret_name': secret_name,
        'target_schema': target_schema
    }
    
    return config


# Logging Configuration
LOGGING_CONFIG = {
    'level': 'DEBUG',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs',
    'log_file_prefix': 'dr_etl_pipeline'
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    'timeout': 300,  # 5 minutes
    'max_retries': 3,
    'batch_size': 10,
    'supported_models': ['IP AUTH'],
    'supported_lobs': ['gbd', 'nogbd'],
    'supported_environments': ['dv', 'ts', 'pl', 'pr']
}
