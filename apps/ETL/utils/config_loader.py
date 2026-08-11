"""
Configuration loader utility for Deep Research ETL Pipeline
Handles environment-specific configuration loading from config.yaml
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


# Environment mapping: dv/ts -> dev, pl -> uat, pr -> prod
ENV_MAPPING = {
    'dv': 'dev',
    'ts': 'dev',
    'pl': 'uat',
    'pr': 'prod'
}


class ConfigLoader:
    """Load and manage environment-specific configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path (str): Path to config.yaml file (relative to this module or absolute)
        """
        # If config_path is relative, look for it relative to this module's directory
        if not os.path.isabs(config_path):
            # Get the directory where this module is located (utils/)
            module_dir = Path(__file__).parent
            # Go up one level to the ETL directory where config.yaml is located
            config_path = module_dir.parent / config_path
        
        self.config_path = str(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        print(f"✅ Loaded configuration from {self.config_path}")
    
    def get_env_config(self, env: str) -> Dict[str, Any]:
        """
        Get configuration for specified environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            Dict[str, Any]: Environment-specific configuration
        """
        if env not in ENV_MAPPING:
            raise ValueError(f"Invalid environment '{env}'. Must be one of: {list(ENV_MAPPING.keys())}")
        
        mapped_env = ENV_MAPPING[env]
        
        if mapped_env not in self._config:
            raise ValueError(f"Configuration not found for environment: {mapped_env}")
        
        return self._config[mapped_env]
    
    def get_snowflake_config(self, env: str) -> Dict[str, Any]:
        """
        Get Snowflake configuration for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            Dict[str, Any]: Snowflake configuration
        """
        env_config = self.get_env_config(env)
        return env_config.get('snowflake', {})
    
    def get_s3_config(self, env: str) -> Dict[str, Any]:
        """
        Get S3 configuration for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            Dict[str, Any]: S3 configuration
        """
        env_config = self.get_env_config(env)
        return env_config.get('s3', {})
    
    def get_table_name(self, env: str, lob: str) -> str:
        """
        Get Snowflake table name for environment and LOB
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
            lob (str): LOB type (gbd or nogbd)
        
        Returns:
            str: Full table name
        """
        env_config = self.get_env_config(env)
        table_mapping = env_config.get('table_mapping', {})
        
        if lob not in table_mapping:
            raise ValueError(f"Invalid LOB '{lob}'. Must be 'gbd' or 'nogbd'")
        
        return table_mapping[lob]
    
    def get_agent_api_url(self, env: str) -> str:
        """
        Get agent API base URL for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            str: Agent API base URL
        """
        env_config = self.get_env_config(env)
        agent_api_config = env_config.get('agent_api', {})
        return agent_api_config.get('base_url', '')
    
    def get_agent_api_timeout(self, env: str) -> int:
        """
        Get agent API timeout for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            int: Timeout in seconds
        """
        env_config = self.get_env_config(env)
        agent_api_config = env_config.get('agent_api', {})
        return agent_api_config.get('timeout', 300)
    
    def get_database_config(self, env: str) -> Dict[str, Any]:
        """
        Get database operation configuration for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            Dict[str, Any]: Database configuration with truncate_table and save_to_db flags
        """
        env_config = self.get_env_config(env)
        return env_config.get('database', {})
    
    def should_truncate_table(self, env: str) -> bool:
        """
        Check if table truncation is enabled for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            bool: True if truncate is enabled
        """
        db_config = self.get_database_config(env)
        return db_config.get('truncate_table', False)
    
    def should_save_to_db(self, env: str) -> bool:
        """
        Check if database save is enabled for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            bool: True if save to DB is enabled
        """
        db_config = self.get_database_config(env)
        return db_config.get('save_to_db', True)
    
    def is_job_tracking_enabled(self, env: str) -> bool:
        """
        Check if job tracking system is enabled for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            bool: True if job tracking is enabled
        """
        env_config = self.get_env_config(env)
        job_tracking_config = env_config.get('job_tracking', {})
        return job_tracking_config.get('enabled', True)  # Default to True for backward compatibility
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration
        
        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        return self._config.get('pipeline', {})
    
    def get_s3_bucket(self, env: str) -> str:
        """
        Get S3 bucket name for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            str: S3 bucket name
        """
        s3_config = self.get_s3_config(env)
        return s3_config.get('bucket', '')
    
    def get_s3_base_path(self, env: str) -> str:
        """
        Get S3 base path for environment
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            str: S3 base path
        """
        s3_config = self.get_s3_config(env)
        return s3_config.get('base_path', 'deep_rsrch/dataz/outbound')
    
    def get_vault_config(self, env: str) -> Dict[str, Any]:
        """
        Get Vault configuration for Snowflake credentials
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        
        Returns:
            Dict[str, Any]: Vault configuration
        """
        snowflake_config = self.get_snowflake_config(env)
        return snowflake_config.get('vault', {})
    
    def get_semantic_config_path(self, env: str, statscl_mdl_cd: str) -> str:
        """
        Get semantic YAML config path for correlation and pattern agents
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
            statscl_mdl_cd (str): Statistical model code (e.g., 'OON', 'IP AUTH')
        
        Returns:
            str: Path to semantic YAML configuration file
        """
        pipeline_config = self.get_pipeline_config()
        semantic_config = pipeline_config.get('semantic_config', {})
        
        # Check if local mode is enabled
        is_local = pipeline_config.get('local', False)
        local_lob = pipeline_config.get('local_lob', '')
        
        # Determine model type (OON vs default)
        if statscl_mdl_cd.upper() == 'OON':
            model_key = 'oon'
        elif statscl_mdl_cd.upper() == 'OP OTH BH':
            model_key = 'op_oth_bh'
        else:
            model_key = 'default'
        model_config = semantic_config.get(model_key, {})
        
        # Determine environment key
        if is_local:
            if local_lob.lower() == 'offshore':
                env_key = 'local_offshore'
            else:
                env_key = 'local'
        else:
            # Map environment code to config key
            env_key = ENV_MAPPING.get(env, 'dev')
        
        # Get the path
        semantic_path = model_config.get(env_key, '')
        
        if not semantic_path:
            # Fallback to default if not found
            print(f"⚠️  Warning: Semantic config not found for model={model_key}, env={env_key}. Using default.")
            default_config = semantic_config.get('default', {})
            semantic_path = default_config.get(env_key, 'configs/correlation_pattern/coc_ecap_ip_auth_semantic_view_with_samples_dev.yaml')
        
        return semantic_path
    
    def get_target_table(self, env: str, lob: str = 'gbd') -> str:
        """
        Get target table name from configuration
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
            lob (str): Line of business ('gbd' or 'nogbd')
        
        Returns:
            str: Fully qualified target table name (org or bkp based on use_backup_table flag)
        """
        mapped_env = ENV_MAPPING.get(env, env)
        env_config = self._config.get(mapped_env, {})
        target_table_config = env_config.get('target_table', {})
        
        # Get LOB-specific table config
        lob_config = target_table_config.get(lob, {})
        
        # Check if backup table should be used
        database_config = env_config.get('database', {})
        use_backup_table = database_config.get('use_backup_table', False)
        
        # Select org or bkp table based on flag
        if use_backup_table:
            target_table = lob_config.get('bkp', '')
            if target_table:
                print(f"🔄 Using backup table: {target_table}")
            else:
                print(f"⚠️  Warning: Backup table not found in config for env={env}, lob={lob}")
        else:
            target_table = lob_config.get('org', '')
            if target_table:
                print(f"✅ Using original table: {target_table}")
            else:
                print(f"⚠️  Warning: Original table not found in config for env={env}, lob={lob}")
        
        # Fallback: construct from schema if not found
        if not target_table:
            schema = env_config.get('schema', '').upper()
            if lob == 'nogbd':
                target_table = f"{schema}_COC.{schema}_NOGBD.COC_CMN_DEEP_RSRCH_INSGHT_STG"
            else:
                target_table = f"{schema}_COC.{schema}.COC_CMN_DEEP_RSRCH_INSGHT_STG"
            print(f"⚠️  Using fallback table: {target_table}")
        
        return target_table
    
    def print_config(self, env: str):
        """
        Print configuration for debugging
        
        Args:
            env (str): Environment code (dv, ts, pl, pr)
        """
        mapped_env = ENV_MAPPING.get(env, env)
        print(f"\n{'='*60}")
        print(f"Configuration for {env.upper()} (mapped to {mapped_env.upper()})")
        print(f"{'='*60}")
        
        env_config = self.get_env_config(env)
        
        print(f"\n📊 Snowflake:")
        sf_config = env_config.get('snowflake', {})
        print(f"  Account: {sf_config.get('account')}")
        print(f"  Database: {sf_config.get('database')}")
        print(f"  Schema: {sf_config.get('schema')}")
        print(f"  Warehouse: {sf_config.get('warehouse')}")
        
        print(f"\n☁️  S3:")
        s3_config = env_config.get('s3', {})
        print(f"  Bucket: {s3_config.get('bucket')}")
        print(f"  Region: {s3_config.get('region')}")
        print(f"  Base Path: {s3_config.get('base_path')}")
        
        print(f"\n📋 Tables:")
        table_mapping = env_config.get('table_mapping', {})
        for lob, table in table_mapping.items():
            print(f"  {lob}: {table}")
        
        print(f"\n{'='*60}\n")


# Singleton instance
_config_loader = None


def get_config_loader(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Get singleton ConfigLoader instance
    
    Args:
        config_path (str): Path to config.yaml file
    
    Returns:
        ConfigLoader: Configuration loader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    
    return _config_loader


# Convenience functions
def get_env_config(env: str) -> Dict[str, Any]:
    """Get environment configuration"""
    return get_config_loader().get_env_config(env)


def get_snowflake_config(env: str) -> Dict[str, Any]:
    """Get Snowflake configuration"""
    return get_config_loader().get_snowflake_config(env)


def get_s3_config(env: str) -> Dict[str, Any]:
    """Get S3 configuration"""
    return get_config_loader().get_s3_config(env)


def get_table_name(env: str, lob: str) -> str:
    """Get table name"""
    return get_config_loader().get_table_name(env, lob)


def get_s3_bucket(env: str) -> str:
    """Get S3 bucket"""
    return get_config_loader().get_s3_bucket(env)
