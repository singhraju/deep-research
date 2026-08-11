"""
DataLab Snowflake Connection for Job Tracking Tables
Separate connection to NON_CRTFD_AIFS database for storing job tracking data
"""

import logging
from snowflake.snowpark import Session
import snowflake.connector
from typing import Tuple

# DataLab Snowflake configuration (hardcoded - no multi-env)
DATALAB_CONFIG = {
    "account": "carelon-edaprod1.privatelink",
    "user": "AN304146AD",
    "password": "",
    "role": "AN304146AD_PRIVS",
    "warehouse": "DL_AIFS_TMBR_USER_WH_XL",
    "database": "NON_CRTFD_AIFS",
    "schema": "DL_DV_TMBR_STG_NOGBD"
}


def create_datalab_session() -> Session:
    """
    Create Snowpark session for DataLab environment
    Used for job tracking tables
    
    Returns:
        Session: Snowpark session connected to DataLab
    """
    try:
        session = Session.builder.configs(DATALAB_CONFIG).create()
        logging.info("DataLab Snowpark session created successfully")
        print("✅ DataLab Snowpark session created successfully")
        return session
    except Exception as e:
        logging.error(f"Failed to create DataLab session: {str(e)}")
        print(f"❌ Failed to create DataLab session: {str(e)}")
        raise


def create_datalab_connector() -> Tuple[snowflake.connector.SnowflakeConnection, snowflake.connector.cursor.SnowflakeCursor]:
    """
    Create Snowflake connector and cursor for DataLab environment
    Used for direct SQL execution if needed
    
    Returns:
        Tuple: (connection, cursor)
    """
    try:
        connection = snowflake.connector.connect(**DATALAB_CONFIG)
        cursor = connection.cursor()
        logging.info("DataLab Snowflake connector created successfully")
        print("✅ DataLab Snowflake connector created successfully")
        return connection, cursor
    except Exception as e:
        logging.error(f"Failed to create DataLab connector: {str(e)}")
        print(f"❌ Failed to create DataLab connector: {str(e)}")
        raise


def get_datalab_schema() -> str:
    """
    Get the DataLab schema name for job tracking tables
    
    Returns:
        str: Schema name (e.g., 'NON_CRTFD_AIFS.DL_DV_TMBR_STG_NOGBD')
    """
    return f"{DATALAB_CONFIG['database']}.{DATALAB_CONFIG['schema']}"


def test_datalab_connection():
    """
    Test DataLab connection
    """
    try:
        print("Testing DataLab Snowflake connection...")
        session = create_datalab_session()
        
        # Test query
        result = session.sql("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()").collect()
        
        print(f"Connection successful!")
        print(f"User: {result[0][0]}")
        print(f"Role: {result[0][1]}")
        print(f"Database: {result[0][2]}")
        print(f"Schema: {result[0][3]}")
        
        session.close()
        return True
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the connection
    #test_datalab_connection()
    print("test")
