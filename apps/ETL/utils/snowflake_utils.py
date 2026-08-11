"""
Snowflake utility functions for data fetching and processing
"""

import pandas as pd
from snowflake.snowpark import Session
from flexible_snowflake_connector import get_snowflake_connection
from typing import Dict, Any
from utils.config_loader import get_config_loader


def get_table_name(env: str, lob: str = "gbd") -> str:
    """
    Get the appropriate table name based on environment and LOB type
    Reads from config.yaml using config_loader
    
    Args:
        env (str): Environment (dv, ts, pl, pr)
        lob (str): LOB type - "gbd" or "nogbd" (default: "gbd")
        
    Returns:
        str: Full table name from config.yaml
    """
    config_loader = get_config_loader()
    table_name = config_loader.get_table_name(env, lob)
    print(f"Using table: {table_name} for environment: {env}, lob: {lob}")
    return table_name


def fetch_snowflake_data(session: Session, 
                        env: str,
                        statscl_mdl_cd: str = "IP AUTH",
                        lob: str = "gbd",
                        snap_year_mnth_nbr: int = None,
                        trnd_tm_prd_cd: str = None,
                        lob_shrt_desc: str = None) -> pd.DataFrame:
    """
    Fetch data from Snowflake table with specified filters
    
    Args:
        session (Session): Snowpark session object
        env (str): Target environment (dv, ts, pl, pr)
        statscl_mdl_cd (str): STATSCL_MDL_CD filter value (default: "IP AUTH")
        lob (str): LOB type - "gbd" or "nogbd" (default: "gbd")
        snap_year_mnth_nbr (int): Optional SNAP_YEAR_MNTH_NBR filter
        trnd_tm_prd_cd (str): Optional TRND_TM_PRD_CD filter (e.g., 'R3', 'R6', 'R12')
        lob_shrt_desc (str): Optional LOB_SHRT_DESC filter (e.g., 'Commercial_Individual')
        
    Returns:
        pd.DataFrame: DataFrame containing the fetched data
    """
    table_name = get_table_name(env, lob)
    
    query = f"""
    SELECT SNAP_YEAR_MNTH_NBR, TRND_TM_PRD_END_MNTH_NBR, TRND_TM_PRD_CD, 
           LOB_CD, LOB_SHRT_DESC, STATSCL_MDL_CD, INSGHT_TYPE_NM, JSON_TXT
    FROM {table_name}
    WHERE STATSCL_MDL_CD = '{statscl_mdl_cd}'
    """
    
    # Add optional filters
    if snap_year_mnth_nbr:
        query = query + f" AND SNAP_YEAR_MNTH_NBR = {snap_year_mnth_nbr}"
    
    if trnd_tm_prd_cd:
        query = query + f" AND TRND_TM_PRD_CD = '{trnd_tm_prd_cd}'"
    
    if lob_shrt_desc:
        query = query + f" AND LOB_SHRT_DESC = '{lob_shrt_desc}'"
    
    print(query)
    
    try:
        print("Fetching data from Snowflake...")
        filters = [f"STATSCL_MDL_CD = '{statscl_mdl_cd}'"]
        if snap_year_mnth_nbr:
            filters.append(f"SNAP_YEAR_MNTH_NBR = {snap_year_mnth_nbr}")
        if trnd_tm_prd_cd:
            filters.append(f"TRND_TM_PRD_CD = '{trnd_tm_prd_cd}'")
        if lob_shrt_desc:
            filters.append(f"LOB_SHRT_DESC = '{lob_shrt_desc}'")
        
        print(f"Query filters: {', '.join(filters)}")
        print(f"Table: {table_name}")
        df = session.sql(query).to_pandas()
        print(f"Successfully fetched {len(df)} records from Snowflake")
        return df
    except Exception as e:
        print(f"Error fetching data from Snowflake: {str(e)}")
        raise


def create_snowpark_config(env: str) -> Session:
    """
    Create Snowpark configuration for the given environment
    Reads configuration from config.yaml using config_loader
    Environment mapping: dv/ts -> dev, pl -> uat, pr -> prod
    
    Args:
        env (str): Environment code (dv, ts, pl, pr)
        
    Returns:
        Session: Snowpark session object
    """
    config_loader = get_config_loader()
    sf_config = config_loader.get_snowflake_config(env)
    vault_config = config_loader.get_vault_config(env)
    
    print(f"Creating Snowpark session for {env.upper()} environment")
    print(f"  Database: {sf_config['database']}")
    print(f"  Warehouse: {sf_config['warehouse']}")
    print(f"  Vault Role: {vault_config['role_name']}")
    
    session = get_snowflake_connection(
        connector_type="snowpark",
        service_id=vault_config['service_id'],
        vault_role_name=vault_config['role_name'],
        vault_namespace=vault_config['namespace'],
        vault_path=vault_config['path'],
        vault_url=vault_config['url'],
        verify_ssl=vault_config['verify_ssl'],
        cert_path=vault_config['cert_path'],
        snowflake_warehouse=sf_config['warehouse'],
        snowflake_schema=sf_config['schema'],
        sf_database=sf_config['database'],
        sf_account=sf_config['account']
    )
    
    print(f"✅ Snowpark session created successfully for {env.upper()}")
    return session