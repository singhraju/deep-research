from secure_snowflake_connector import get_snowflake_connection
import os

def main():
    """Example Snowflake connection"""
    try:
        session = get_snowflake_connection(
            service_id=os.getenv('SNOWFLAKE_SERVICE_ID', 'srccocdtidd'),
            vault_role_name=os.getenv('VAULT_ROLE_NAME', 'snowdb-idiscovery-slvr'),
            vault_namespace=os.getenv('VAULT_NAMESPACE', 'eda-snowflakedb'),
            vault_path=os.getenv('VAULT_PATH', 'eda_nonprod/static-creds/aedl_devops-srccocdtidd'),
            vault_url=os.getenv('VAULT_URL', 'http://vault.acr.awsdns.internal.das:8200'),
            verify_ssl=True,
            snowflake_warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'D01_COC_APP_WH_L'),
            snowflake_schema=os.getenv('SNOWFLAKE_SCHEMA', 'COC_DTI_stg'),
            sf_database=os.getenv('SNOWFLAKE_DATABASE', 'D01_COC'),
            sf_account=os.getenv('SNOWFLAKE_ACCOUNT', 'carelon-eda_nonprod.privatelink')
        )
        
        print("✅ Connection successful!")
        
        # Test query
        results = session.sql("SELECT CURRENT_USER(), CURRENT_TIMESTAMP()").collect()
        for row in results:
            print(f"Connected as: {row[0]} at {row[1]}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    finally:
        if 'session' in locals() and session:
            session.close()
            print("Session closed")

if __name__ == "__main__":
    main()