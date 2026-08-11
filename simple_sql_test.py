#!/usr/bin/env python3
"""Simple SQL query executor for Snowflake."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'packages', 'utils', 'src'))

def run_sql():
    try:
        from deep_research_utils.snowflake_helper import SnowparkHelper
        
        # Connect
        sf = SnowparkHelper()
        
        # Execute query
        query = "SELECT DISTINCT LOB_SHRT_DESC FROM P01_COC.COC_DTI_STG_NOGBD.WORK_ELEVATE_COC_CLAIM_DETAIL WHERE LOB_SHRT_DESC IS NOT NULL ORDER BY LOB_SHRT_DESC"
        
        result = sf.session.sql(query).collect()
        
        # Print results
        print(f"Found {len(result)} LOB values:")
        for row in result:
            print(f"  - {row[0]}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_sql()
