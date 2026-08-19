import sys
import json
import builtins
import logging
import os
from snowflake.snowpark import Session
from os.path import abspath
from snowflake.snowpark.functions import *
import snowflake.connector
import numpy as np
import pandas as pd
import boto3
import argparse
import requests
from typing import Dict, Any, List, Tuple
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from datetime import datetime
import time

# Import enterprise-level utility functions
from utils.time_utils import get_ecap_start_month, convert_current_ecap_time_to_previous_year
from utils.agent_config import initialize_agent_config
from utils.agent_utils import (
    generate_agent_common_payload, 
    generate_correlation_agent_common_payload,
    call_correlation_agents,
    call_pattern_agents,
    call_reimbursement_agents,
    call_policy_agent,
    call_recommendation_agents,combine_pattern_reimbursement,
    generate_pattern_final_result, check_api_health
)
from utils.snowflake_utils import fetch_snowflake_data, create_snowpark_config
from utils.config_loader import get_config_loader
from utils.s3_utils import S3Logger
from utils.checkpoint_manager import CheckpointManager, generate_run_id
from utils.job_tracker import JobTracker
from utils.datalab_connection import create_datalab_session, get_datalab_schema, create_datalab_connector

# Disable SSL warnings for internal APIs
urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging for the pipeline"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dr_etl_pipeline_{timestamp}.log")
    
    # Configure logging format
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    # Disable DEBUG logs from external libraries
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
    logging.getLogger("snowflake.snowpark").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger

# Global logger - set to INFO level to avoid DEBUG spam
logger = setup_logging(log_level=logging.INFO)

# Import API configuration from config module


def parse_secrets_manager(secret_name):
    """Parse secrets from AWS Secrets Manager"""
    region_name = "us-east-1"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        response = client.get_secret_value(
            SecretId=secret_name
        )
        if 'SecretString' in response:
            secret = response['SecretString']
        else:
            import base64
            secret = base64.b64decode(response['SecretBinary'])
        return secret

    except Exception as e:
        logger.error(f"Error retrieving secret: {str(e)}")
        raise


def fetch_api_data(api_url: str, headers: Dict[str, str] = None, verify_ssl: bool = False) -> Dict[str, Any]:
    """Fetch data from API endpoint"""
    try:
        print(f"Fetching data from API: {api_url}")
        
        # Default headers for JSON API
        if headers is None:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        
        response = requests.get(
            api_url, 
            headers=headers, 
            verify=verify_ssl,
            timeout=300  # 5 minutes timeout
        )
        
        response.raise_for_status()
        
        data = response.json()
        print(f"Successfully fetched data. Status: {response.status_code}")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {str(e)}")
        raise




def validate_agent_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate agent results and provide summary statistics
    
    Args:
        results (List[Dict[str, Any]]): List of processing results
        
    Returns:
        Dict[str, Any]: Validation summary with statistics
    """
    validation_summary = {
        "total_records": len(results),
        "successful_records": 0,
        "failed_records": 0,
        "agent_results": {
            "pattern_analysis": 0,
            "correlation_states": 0,
            "correlation_providers": 0,
            "correlation_drgs": 0,
            "pattern_summary": 0,
            "reimbursement_policy": 0,
            "final_recommendations": 0
        },
        "errors": []
    }
    
    for result in results:
        if result.get('status') == 'success':
            validation_summary["successful_records"] += 1
        else:
            validation_summary["failed_records"] += 1
            
        # Count agent results
        if 'pattern_analysis' in result and result['pattern_analysis'].get('status') == 'success':
            validation_summary["agent_results"]["pattern_analysis"] += 1
        # Legacy support
        elif 'pattern_agent_result' in result and result['pattern_agent_result'].get('status') == 'success':
            validation_summary["agent_results"]["pattern_analysis"] += 1
            
        if 'correlation_results' in result:
            corr_data = result['correlation_results']
            if 'states' in corr_data:
                validation_summary["agent_results"]["correlation_states"] += 1
            if 'providers' in corr_data:
                validation_summary["agent_results"]["correlation_providers"] += 1
            if 'drgs' in corr_data:
                validation_summary["agent_results"]["correlation_drgs"] += 1
                
        if 'final_summary' in result and result['final_summary'].get('status') == 'success':
            validation_summary["agent_results"]["pattern_summary"] += 1
            
        if 'reimbursement_policy' in result and result['reimbursement_policy'].get('success', False):
            validation_summary["agent_results"]["reimbursement_policy"] += 1
            
        if 'final_recommendations' in result and result['final_recommendations'].get('success', False):
            validation_summary["agent_results"]["final_recommendations"] += 1
    
    return validation_summary

def generate_job_id(snap_year_mnth_nbr, trnd_tm_prd_end_mnth_nbr, trnd_tm_prd_cd, lob_shrt_desc, statscl_mdl_cd):
    """
    Generate unique job ID from combination parameters
    
    Args:
        snap_year_mnth_nbr: Snapshot year-month number
        trnd_tm_prd_end_mnth_nbr: Trend period end month number
        trnd_tm_prd_cd: Trend period code
        lob_shrt_desc: LOB short description
        statscl_mdl_cd: Statistical model code
        
    Returns:
        str: Unique job identifier
    """
    # Sanitize strings for job ID
    safe_lob = str(lob_shrt_desc).replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_model = str(statscl_mdl_cd).replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    return f"{snap_year_mnth_nbr}_{trnd_tm_prd_end_mnth_nbr}_{trnd_tm_prd_cd}_{safe_lob}_{safe_model}"

def store_agent_results_to_snowflake(session: Session, env: str, env_config: Dict[str, Any], 
                                     original_df: pd.DataFrame, results: List[Dict[str, Any]],
                                     lob: str = "gbd", s3_logger=None) -> None:
    """
    Store all agent results back to Snowflake with appropriate INSGHT_TYPE_NM values
    
    Args:
        session (Session): Snowpark session object
        env (str): Environment code (dv, ts, pl, pr)
        env_config (Dict[str, Any]): Environment configuration
        original_df (pd.DataFrame): Original data DataFrame
        results (List[Dict[str, Any]]): API processing results
        lob (str): Line of business ('gbd' or 'nogbd')
        s3_logger (S3Logger): S3 logger instance for saving files to S3
    """
    try:
        # Filter successful results
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            print("No successful results to store in Snowflake")
            return
        
        print(f"Processing {len(successful_results)} successful results for Snowflake storage...")
        
        # Collect all result types across all records
        all_records = []
        
        for result in successful_results:
            # Use data directly from result structure (no record_index needed)
            # Base fields for all records
            base_record = {
                'SNAP_YEAR_MNTH_NBR': result.get('snap_year_mnth_nbr'),
                'TRND_TM_PRD_END_MNTH_NBR': result.get('trnd_tm_prd_end_mnth_nbr'),
                'TRND_TM_PRD_CD': result.get('trnd_tm_prd_cd'),
                'LOB_CD': result.get('lob_cd'),  # LOB_CD should be a code (now available in result)
                'LOB_SHRT_DESC': result.get('lob_shrt_desc'),  # LOB_SHRT_DESC should be description
                'STATSCL_MDL_CD': result.get('statscl_mdl_cd'),
                'EDL_LOB_CD': result.get('lob_cd')
            }
            
            # Store 1. Entire correlation result as JSON text
            """
            if 'correlation_results' in result:
                correlation_record = base_record.copy()
                correlation_record.update({
                    'INSGHT_TYPE_NM': 'Correlation',
                    'JSON_TXT': json.dumps(result['correlation_results']) if isinstance(result['correlation_results'], dict) else str(result['correlation_results'])
                })
                all_records.append(correlation_record)
            """
            
            # Store 2. Pattern result output as JSON text
            # pattern_result is a list of processed patterns (can be empty)
            if 'pattern_result' in result and isinstance(result['pattern_result'], list):
                # Store the entire list of patterns as JSON (even if empty to track zero-pattern cases)
                pattern_record = base_record.copy()
                pattern_record.update({
                    'INSGHT_TYPE_NM': 'Pattern',
                    'JSON_TXT': json.dumps(result['pattern_result'])
                })
                all_records.append(pattern_record)
            
            # Store 3. Recommendation result as JSON text
            if 'recommendation_result' in result:
                recommendation_record = base_record.copy()
                res = result.get("recommendation_result", {}).get("recommendations", [])
                recommendation_record.update({
                    'INSGHT_TYPE_NM': 'Recommendation',
                    'JSON_TXT': json.dumps(res)                    
                })
                all_records.append(recommendation_record)
            
            # Store 4. Policy result as JSON text
            if 'policy_result' in result and result['policy_result']:
                policy_record = base_record.copy()
                policy_record.update({
                    'INSGHT_TYPE_NM': 'Policy',
                    'JSON_TXT': json.dumps(result['policy_result']) if isinstance(result['policy_result'], dict) else str(result['policy_result'])
                })
                all_records.append(policy_record)
        
        if not all_records:
            print("No valid records to store")
            return
        
        # Create DataFrame from all records
        updated_df = pd.DataFrame(all_records)
        updated_df['INSGHT_TYPE_NM'] = updated_df['INSGHT_TYPE_NM'].str.upper()
        #updated_df['JSON_TXT'] = updated_df['JSON_TXT'].apply(lambda x: json.dumps(json.loads(x)) if isinstance(x, str) else json.dumps(x))
        updated_df['JSON_TXT'] = updated_df['JSON_TXT'].apply(json.dumps)
        print(f"Created DataFrame with {len(updated_df)} records for Snowflake storage")
        print(f"INSGHT_TYPE_NM distribution: {updated_df['INSGHT_TYPE_NM'].value_counts().to_dict()}")

        
        # Create Snowflake DataFrame
        snow_df = session.create_dataframe(updated_df)

        snow_df = snow_df.with_column("JSON_TXT", parse_json(col("JSON_TXT")))
        
        # Add audit columns
        final_df_snow_df = (snow_df
                            .with_column('EDL_LOAD_DTM', current_timestamp())
                            .with_column('EDL_RUN_ID', lit('NA'))
                            .with_column('EDL_SOR_CD', lit('NA'))
                            .with_column('KF_TMS', current_timestamp())
                            .with_column('EDL_SCRTY_LVL_CD', lit('NA'))
                            .with_column('EDL_EXTRNL_LOAD_CD', lit('NA'))
                            .with_column('EDL_CREAT_DTM', current_timestamp())
                            .with_column('EDL_INCRMNTL_LOAD_DTM', current_timestamp())
                           )
        
        # Define target columns order
        target_col = [
            "EDL_LOAD_DTM",
            "EDL_RUN_ID", 
            "EDL_SOR_CD",
            "KF_TMS",
            "EDL_SCRTY_LVL_CD",
            "EDL_LOB_CD",
            "EDL_EXTRNL_LOAD_CD",
            "EDL_CREAT_DTM",
            "EDL_INCRMNTL_LOAD_DTM",
            "SNAP_YEAR_MNTH_NBR",
            "TRND_TM_PRD_END_MNTH_NBR",
            "TRND_TM_PRD_CD",
            "LOB_CD",
            "LOB_SHRT_DESC",
            "STATSCL_MDL_CD", 
            "INSGHT_TYPE_NM",
            "JSON_TXT"
        ]
        
        # Reorder columns
        final_df_reordered = final_df_snow_df.select([col(c) for c in target_col])
        
        # Convert Snowpark DataFrame to Pandas for JSON export
        final_df_pandas = final_df_reordered.to_pandas()
        
        # Save Snowflake DataFrame to JSON file for each combination
        for result in successful_results:
            snowflake_filename = f"snowflake_{result.get('snap_year_mnth_nbr')}_{result.get('trnd_tm_prd_end_mnth_nbr')}_{result.get('trnd_tm_prd_cd')}_{result.get('lob_shrt_desc')}_{result.get('statscl_mdl_cd')}.json"
            snowflake_filename = snowflake_filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
            
            # Filter DataFrame for this specific combination
            combination_df = final_df_pandas[
                (final_df_pandas['SNAP_YEAR_MNTH_NBR'] == result.get('snap_year_mnth_nbr')) &
                (final_df_pandas['TRND_TM_PRD_END_MNTH_NBR'] == result.get('trnd_tm_prd_end_mnth_nbr')) &
                (final_df_pandas['TRND_TM_PRD_CD'] == result.get('trnd_tm_prd_cd')) &
                (final_df_pandas['LOB_SHRT_DESC'] == result.get('lob_shrt_desc')) &
                (final_df_pandas['STATSCL_MDL_CD'] == result.get('statscl_mdl_cd'))
            ]
            
            # Convert DataFrame to list of dicts and save as JSON
            combination_records = combination_df.to_dict('records')
            #with open(snowflake_filename, 'w') as f:
            #    json.dump(combination_records, f, indent=2, default=str)
            #print(f"Saved Snowflake DataFrame to: {snowflake_filename}")                        
            # Save to S3 if s3_logger is provided
            if s3_logger:
                s3_logger.save_to_s3(combination_records, snowflake_filename, result.get('statscl_mdl_cd'), 'final_result')
        
        # Get database operation flags and target table from config
        config_loader = get_config_loader()
        should_truncate = config_loader.should_truncate_table(env)
        should_save_db = config_loader.should_save_to_db(env)
        
        # Get target table from config instead of hardcoding
        target_table = config_loader.get_target_table(env, lob)
        print(f"Target table from config: {target_table}")
        
        print(f"Database config - Save to DB: {should_save_db}")
        print(f"Note: Truncate is handled once at run-level, not per combination")
        
        # Conditional save to database based on config
        if should_save_db:
            print(f"Saving {len(updated_df)} records to database...")
            final_df_reordered.write.mode("append").save_as_table(target_table)
            print(f"âœ… Successfully saved {len(updated_df)} records to {target_table}")
        else:
            print(f"Database save disabled - results only saved to S3 as JSON")
            print(f"Prepared {len(updated_df)} records (not saved to DB)")
        
    except Exception as e:
        print(f"Error storing results to Snowflake: {str(e)}")
        raise
    
def main(env: str, use_snowflake: bool = True, csv_file_path: str = "SampleData-dev-deep-research.csv", 
         lob: str = "gbd", statscl_mdl_cd: str = "IP AUTH", snap_year_mnth_nbr: int = None,
         trnd_tm_prd_cd: str = None, lob_shrt_desc: str = None):
    """
    Main ETL pipeline function
    
    Args:
        env: Environment (dv, ts, pl, pr)
        use_snowflake: Use Snowflake or CSV
        csv_file_path: Path to CSV file
        lob: LOB type (gbd or nogbd)
        statscl_mdl_cd: Statistical model code (e.g., 'IP AUTH', 'OON')
        snap_year_mnth_nbr: Optional snapshot year month filter
        trnd_tm_prd_cd: Optional trend period code filter (e.g., 'R3', 'R6', 'R12')
        lob_shrt_desc: Optional LOB short description filter (e.g., 'Commercial_Individual')
    """
    print(f"Starting DR ETL Pipeline for environment: {env}")
    
    # Log filter parameters
    if trnd_tm_prd_cd or lob_shrt_desc:
        print(f"[LOG] Running specific combination filter:")
        if trnd_tm_prd_cd:
            print(f"  - TRND_TM_PRD_CD: {trnd_tm_prd_cd}")
        if lob_shrt_desc:
            print(f"  - LOB_SHRT_DESC: {lob_shrt_desc}")
    else:
        print(f"[LOG] Running all combinations for STATSCL_MDL_CD: {statscl_mdl_cd}")
    
    # Initialize agent API configuration from config.yaml
    initialize_agent_config(env)
    
    # Get environment configuration using config_loader
    config_loader = get_config_loader()
    env_config = config_loader.get_env_config(env)
    print(f"Environment config: {env_config}")
    
    # Get credentials from secrets manager
    #credentials_json = json.loads(parse_secrets_manager(env_config['secret_name']))
    credentials_json = {}

    # Load and process data
    print("Starting complete workflow pipeline...")
    
    if use_snowflake:
        # Create Snowpark session and fetch data
        # Create Snowpark configuration from config.yaml
        session = create_snowpark_config(env)
        print("Snowpark session created successfully")
        
        # Fetch data with optional filters
        print(statscl_mdl_cd)
        df_insights = fetch_snowflake_data(
            session, 
            env=env, 
            statscl_mdl_cd=statscl_mdl_cd, 
            lob=lob, 
            snap_year_mnth_nbr=snap_year_mnth_nbr,
            trnd_tm_prd_cd=trnd_tm_prd_cd,
            lob_shrt_desc=lob_shrt_desc
        )
        
        if df_insights.empty:
            print("No data found in Snowflake table")
            return
    else:
        # Fallback to CSV file
        df_insights = pd.read_csv(csv_file_path)
        df_insights = df_insights[["SNAP_YEAR_MNTH_NBR", "TRND_TM_PRD_END_MNTH_NBR", "TRND_TM_PRD_CD", "LOB_CD", "LOB_SHRT_DESC", "STATSCL_MDL_CD", "INSGHT_TYPE_NM", "JSON_TXT"]]
        print(f"Loaded data from CSV. Shape: {df_insights.shape}")
    
    print(f"Data loaded successfully. Shape: {df_insights.shape}")

    # Initialize S3 Logger for this pipeline run (reads bucket from config.yaml)
    s3_logger = S3Logger(env)
    print(f"[LOG] S3 Logger initialized with timestamp: {s3_logger.timestamp_folder}")    
    
    # Initialize Checkpoint Manager for auto-resume
    config_loader = get_config_loader()
    s3_config = config_loader.get_s3_config(env)
    checkpoint_mgr = CheckpointManager(s3_logger.s3_client, s3_config['bucket'], env)
    
    # Check if should resume from previous failed run
    should_resume, resume_run_id = checkpoint_mgr.should_resume()
    
    # Determine run_id
    if should_resume:
        run_id = resume_run_id
        print(f"\n{'='*60}")
        print(f"ðŸ”„ RESUMING RUN: {run_id}")
        print(f"{'='*60}\n")
        checkpoint_mgr.update_status(run_id, "running")
    else:
        run_id = generate_run_id()
        print(f"\n{'='*60}")
        print(f"🆕 NEW RUN: {run_id}")
        print(f"{'='*60}\n")
        checkpoint_mgr.create_checkpoint(run_id, "running")
    
    # Check if job tracking is enabled
    config_loader = get_config_loader()
    job_tracking_enabled = config_loader.is_job_tracking_enabled(env)
    
    # Initialize Job Tracker for detailed history (using separate DataLab connection)
    if job_tracking_enabled:
        print("Creating DataLab connection for job tracking...")
        datalab_session = create_datalab_session()
        datalab_connector, datalab_cursor = create_datalab_connector()
        datalab_schema = get_datalab_schema()
        job_tracker = JobTracker(datalab_session, run_id, datalab_schema, connector=datalab_connector)
        print(f"✅ Job tracker initialized with DataLab schema: {datalab_schema}")
    else:
        print("⚠️  Job tracking is DISABLED in config.yaml")
        job_tracker = None
        datalab_session = None
        datalab_connector = None
    
    # Get all unique combinations from data
    all_combinations = []
    for snap_year_mnth_nbr in df_insights.SNAP_YEAR_MNTH_NBR.unique():
        for trnd_tm_prd_end_mnth_nbr in df_insights.TRND_TM_PRD_END_MNTH_NBR.unique():
            for trnd_tm_prd_cd in df_insights.TRND_TM_PRD_CD.unique():
                for lob_shrt_desc in df_insights.LOB_SHRT_DESC.unique():
                    for statscl_mdl_cd_val in df_insights.STATSCL_MDL_CD.unique():
                        all_combinations.append({
                            'snap_year_mnth_nbr': snap_year_mnth_nbr,
                            'trnd_tm_prd_end_mnth_nbr': trnd_tm_prd_end_mnth_nbr,
                            'trnd_tm_prd_cd': trnd_tm_prd_cd,
                            'lob_cd': df_insights[df_insights.STATSCL_MDL_CD == statscl_mdl_cd_val].LOB_CD.iloc[0] if not df_insights[df_insights.STATSCL_MDL_CD == statscl_mdl_cd_val].empty else '',
                            'lob_shrt_desc': lob_shrt_desc,
                            'statscl_mdl_cd': statscl_mdl_cd_val
                        })
    
    # Get jobs to process
    if job_tracking_enabled and should_resume:
        # Resume: get pending jobs from database
        pending_jobs = job_tracker.get_pending_jobs()
        print(f"📋 Found {len(pending_jobs)} pending jobs to process")
        print(f"⏭️ Skipping {len(all_combinations) - len(pending_jobs)} already completed jobs\n")
        combinations_to_process = pending_jobs
    else:
        # New run: initialize all jobs and process all
        if job_tracking_enabled:
            job_tracker.initialize_jobs(all_combinations)
        combinations_to_process = all_combinations
        print(f"📋 Initialized {len(all_combinations)} jobs\n")
    
    # Process using the working logic from standalone script
    results = []
    
    # Handle truncate once at run level (not per combination)
    if use_snowflake:
        config_loader = get_config_loader()
        should_truncate = config_loader.should_truncate_table(env)
        
        # Get target table from config instead of hardcoding
        target_table = config_loader.get_target_table(env, lob)
        print(f"[LOG] Target table from config: {target_table}")
        
        if should_truncate and not should_resume:
            # Only truncate for NEW runs, not resumed runs
            print(f"\n⚠️  TRUNCATE enabled for NEW run - clearing table: {target_table}")
            try:
                session.sql(f"TRUNCATE TABLE IF EXISTS {target_table}").collect()
                print(f"✅ Table truncated successfully (run-level, one-time operation)\n")
            except Exception as e:
                print(f"⚠️  Warning: Could not truncate table: {str(e)}\n")
        elif should_truncate and should_resume:
            print(f"\n⚠️  TRUNCATE skipped - resuming existing run (preserving data)\n")
        else:
            print(f"\n✅ TRUNCATE disabled - data will be appended\n")
    
    # Process each unique combination (using the working logic from standalone script)
    print("[LOG] Starting combination processing loop")
    
    # Wrap processing in try-except for checkpoint management
    try:
        for idx, combination in enumerate(combinations_to_process, 1):
            # Extract combination values
            snap_year_mnth_nbr = combination['snap_year_mnth_nbr']
            trnd_tm_prd_end_mnth_nbr = combination['trnd_tm_prd_end_mnth_nbr']
            trnd_tm_prd_cd = combination['trnd_tm_prd_cd']
            lob_shrt_desc = combination['lob_shrt_desc']
            statscl_mdl_cd = combination['statscl_mdl_cd']
            
            # Generate job ID
            job_id = generate_job_id(snap_year_mnth_nbr, trnd_tm_prd_end_mnth_nbr, trnd_tm_prd_cd, lob_shrt_desc, statscl_mdl_cd)
            
            print(f"\n{'='*60}")
            print(f"Processing Job {idx}/{len(combinations_to_process)}: {job_id}")
            print(f"{'='*60}")
            print(f"[LOG] Processing combination: snap_year_mnth_nbr={snap_year_mnth_nbr}, trnd_tm_prd_end_mnth_nbr={trnd_tm_prd_end_mnth_nbr}, trnd_tm_prd_cd={trnd_tm_prd_cd}, lob_shrt_desc={lob_shrt_desc}, statscl_mdl_cd={statscl_mdl_cd}")
            print(f"Processing: {snap_year_mnth_nbr}, {trnd_tm_prd_end_mnth_nbr}, {trnd_tm_prd_cd}, {lob_shrt_desc}, {statscl_mdl_cd}")
            
            # Update job status to IN_PROGRESS
            if job_tracker:
                job_tracker.update_job_status(job_id, "IN_PROGRESS")
            
            try:
                # Data split function (from standalone script)
                first_anomaly = df_insights[(df_insights.SNAP_YEAR_MNTH_NBR == snap_year_mnth_nbr) &
                                              (df_insights.TRND_TM_PRD_END_MNTH_NBR == trnd_tm_prd_end_mnth_nbr) &
                                              (df_insights.TRND_TM_PRD_CD == trnd_tm_prd_cd) &
                                              (df_insights.LOB_SHRT_DESC == lob_shrt_desc) & 
                                              (df_insights.STATSCL_MDL_CD == statscl_mdl_cd) &
                                              (df_insights.INSGHT_TYPE_NM == 'KEY_INSIGHT')]
                
                if first_anomaly.empty:
                    logger.warning(f"No KEY_INSIGHT record found for combination: {snap_year_mnth_nbr}, {trnd_tm_prd_end_mnth_nbr}, {trnd_tm_prd_cd}, {lob_shrt_desc}, {statscl_mdl_cd}")
                    print(f"Warning: No KEY_INSIGHT record found for this combination")
                    continue
                    
                first_deep_dive = df_insights[(df_insights.SNAP_YEAR_MNTH_NBR == snap_year_mnth_nbr) &
                                            (df_insights.TRND_TM_PRD_END_MNTH_NBR == trnd_tm_prd_end_mnth_nbr) &
                                            (df_insights.TRND_TM_PRD_CD == trnd_tm_prd_cd) &
                                            (df_insights.LOB_SHRT_DESC == lob_shrt_desc) & 
                                            (df_insights.STATSCL_MDL_CD == statscl_mdl_cd) &
                                            (df_insights.INSGHT_TYPE_NM == 'DEEP_DIVE')]
                
                if first_deep_dive.empty:
                    logger.warning(f"No DEEP_DIVE record found for combination: {snap_year_mnth_nbr}, {trnd_tm_prd_end_mnth_nbr}, {trnd_tm_prd_cd}, {lob_shrt_desc}, {statscl_mdl_cd}")
                    print(f"Warning: No DEEP_DIVE record found for this combination")
                    continue
                
                # Parse JSON data
                try:
                    anomaly_json = json.loads(json.loads(first_anomaly.JSON_TXT.iloc[0]))
                    deep_dive_json = json.loads(json.loads(first_deep_dive.JSON_TXT.iloc[0]))
                    insights_lob_code = first_deep_dive.LOB_CD.iloc[0]
                except Exception as e:
                    logger.error(f"Error parsing JSON data: {str(e)}")
                    print(f"Error parsing JSON data: {str(e)}")
                    continue
                
                # Generate payload (from standalone script)
                _payload, _id = generate_agent_common_payload(trnd_tm_prd_end_mnth_nbr, trnd_tm_prd_cd, snap_year_mnth_nbr, statscl_mdl_cd, lob_shrt_desc)
                final_payload = generate_correlation_agent_common_payload(_payload, statscl_mdl_cd, env)
                
                # Log semantic config being used
                if 'yaml_path' in final_payload:
                    print(f"[LOG] Using semantic config for correlation: {final_payload['yaml_path']}")

                # Save correlation payload to S3
                model_sanitized = statscl_mdl_cd.replace(" ", "_").replace("/", "_").replace("\\", "_")
                corr_payload_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/payload/correlation_payload_{_id}.json"
                corr_result_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/agents_results/correlation_response_{_id}.json"
                s3_logger.save_to_s3(final_payload, f"correlation_payload_{_id}.json", statscl_mdl_cd, 'payload')
                s3_logger.save_to_s3(anomaly_json, f"correlation_payload_anomaly_{_id}.json", statscl_mdl_cd, 'payload')                        
                
                print(f"[LOG] Starting health check before correlation agents for combination: {snap_year_mnth_nbr}, {trnd_tm_prd_end_mnth_nbr}, {trnd_tm_prd_cd}, {lob_shrt_desc}, {statscl_mdl_cd}")
                # Call correlation agents (from standalone script)
                if not check_api_health(retry_interval=2, timeout=5):
                    print("API health check failed before correlation agents. Skipping this combination.")
                    continue
                print("[LOG] API health check passed for correlation agents")
                print("[LOG] Calling correlation agents...")
                
                # Track correlation agent as sub-job
                if job_tracker:
                    correlation_subjob_id = job_tracker.create_sub_job(job_id, "correlation", s3_payload_path=corr_payload_path)
                
                try:
                    correlation_results = call_correlation_agents(final_payload, anomaly_json)
                    print(f"[LOG] Correlation agents completed. States: {len(correlation_results.get('states', {}))}, Providers: {len(correlation_results.get('providers', {}))}, DRGs: {len(correlation_results.get('drgs', {}))}")
                    
                    # Update sub-job status to SUCCESS
                    if job_tracker:
                        job_tracker.update_sub_job_status(
                            job_id, correlation_subjob_id, "SUCCESS",
                            response_code=200,
                            s3_result_path=corr_result_path
                        )
                except Exception as corr_error:
                    print(f"[LOG] Correlation agents failed: {str(corr_error)}")
                    if job_tracker:
                        job_tracker.update_sub_job_status(
                            job_id, correlation_subjob_id, "FAILED",
                            error_message=str(corr_error),
                            response_code=500
                        )
                    raise

                # Save correlation response to S3
                s3_logger.save_to_s3(correlation_results, f"correlation_response_{_id}.json", statscl_mdl_cd, 'agents_results')                        
                
                print("[LOG] Starting health check before pattern agent")
                # Call pattern agent (from standalone script)
                if not check_api_health(retry_interval=2, timeout=5):
                    print("API health check failed before pattern agent. Skipping this combination.")
                    continue
                print("[LOG] API health check passed for pattern agent")
                print("[LOG] Calling pattern agent...")
                
                # Save pattern payload to S3 (matches actual API request structure)
                pattern_payload_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/payload/pattern_payload_{_id}.json"
                pattern_result_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/agents_results/pattern_response_{_id}.json"
                
                # Get semantic config path for pattern agent
                semantic_config_path = config_loader.get_semantic_config_path(env, statscl_mdl_cd)
                print(f"[LOG] Using semantic config for pattern: {semantic_config_path}")
                
                pattern_payload = {
                    "conversation_id": _id,
                    "query": "Summarize the highest-impact authorization and provider mix themes from the completed correlation analysis.",
                    "semantic_config_path": semantic_config_path,
                    "context": {
                        "anomaly_context": anomaly_json,
                        "deep_dive_report": deep_dive_json,
                        "correlation_results": correlation_results
                    }
                }
                s3_logger.save_to_s3(pattern_payload, f"pattern_payload_{_id}.json", statscl_mdl_cd, 'payload')
                
                # Track pattern agent as sub-job
                if job_tracker:
                    pattern_subjob_id = job_tracker.create_sub_job(job_id, "pattern", s3_payload_path=pattern_payload_path)
                
                try:
                    MAX_RETRIES = 3
                    RETRY_DELAY = 5
                    pattern_success = False
                    final_response_code = None
                    retry_count = 0
                    for attempt in range(MAX_RETRIES):
                        retry_count = attempt
                        pattern_response, pattern_result = call_pattern_agents(_id, anomaly_json, deep_dive_json, correlation_results, semantic_config_path)
                        if pattern_response:
                            final_response_code = pattern_response.status_code
                            if pattern_response.status_code == 200:
                                if pattern_result.get("status") == "success":
                                    pattern_success = True
                                    break
                                else:
                                    print(f"Attempt {attempt+1}: Pattern Agent failed - {pattern_result.get('explanation', {}).get('error')}")
                            else:
                                print(f"Attempt {attempt+1}: Pattern Agent failed - HTTP Error Network Error")
                        else:
                            print(f"Pattern Agent Called Failed ")
                            final_response_code = 500
                        time.sleep(5)
                    
                    print(f"[LOG] Pattern agent completed. Business patterns found: {len(pattern_result['output']['business_patterns']) if pattern_result and 'output' in pattern_result else 0}")
                    
                    # Update sub-job status based on result
                    if job_tracker:
                        if pattern_success:
                            job_tracker.update_sub_job_status(
                                job_id, pattern_subjob_id, "SUCCESS",
                                response_code=final_response_code or 200,
                                retry_count=retry_count,
                                s3_result_path=pattern_result_path
                            )
                        else:
                            job_tracker.update_sub_job_status(
                                job_id, pattern_subjob_id, "FAILED",
                                error_message="Pattern agent failed after retries",
                                response_code=final_response_code or 500,
                                retry_count=retry_count
                            )
                except Exception as pattern_error:
                    print(f"[LOG] Pattern agent failed: {str(pattern_error)}")
                    if job_tracker:
                        job_tracker.update_sub_job_status(
                            job_id, pattern_subjob_id, "FAILED",
                            error_message=str(pattern_error),
                            response_code=500
                        )
                    raise
                
                all_cards = pattern_result['output'].get('cards', [])
                print(f"[LOG] Processing {len(pattern_result['output']['business_patterns'])} business patterns with {len(all_cards)} cards")

                
                # Save pattern response to S3
                s3_logger.save_to_s3(pattern_result, f"pattern_response_{_id}.json", statscl_mdl_cd, 'agents_results')
                processed_pattern_result = []
                full_reim_result = []

                # Handle case when no business patterns are found
                if len(pattern_result['output']['business_patterns']) == 0:
                    print("[LOG] No business patterns found. Using default empty structures for results.")
                    # Set empty structures to maintain pipeline consistency
                    reim_result_output = []
                    recommendation_results = {
                        "success": True,
                        "result": {
                            "metadata": {},
                            "recommendations": [],
                            "skipped_patterns": [],
                            "processing_log": ["No business patterns found - skipped processing"]
                        },
                        "metadata": {
                            "agent_name": "recommendation_synthesis"
                        }
                    }
                else:
                    # Track reimbursement agent as sub-job (only once for all patterns)
                    reim_payload_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/payload/reimbursement_payload_{_id}.json"
                    reim_result_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/agents_results/reimbursement_response_{_id}.json"
                    if job_tracker:
                        reimbursement_subjob_id = job_tracker.create_sub_job(job_id, "reimbursement", s3_payload_path=reim_payload_path)
                    
                    # Process business patterns normally
                    reim_success = True
                    reim_error_msg = None
                    try:
                        for idx, pr in enumerate(pattern_result['output']['business_patterns']):
                            print(f"[LOG] Processing pattern {idx+1}/{len(pattern_result['output']['business_patterns'])}: Rank {pr['pattern_rank']}")
                            _rid = f"{pattern_result['conversation_id']}_{pr['pattern_rank']}"
                            print(f"[LOG] Starting health check before reimbursement agents for pattern {pr['pattern_rank']}")
                            if not check_api_health(retry_interval=2, timeout=5):
                                print("[LOG] API health check failed before reimbursement agents. Using default values.")
                                s_code = 400
                                reimbursement_results = {"detail": {"status": False, "error": "Health check failed"}}
                                reim_success = False
                            else:
                                print("[LOG] API health check passed for reimbursement agents")
                                print(f"[LOG] Calling reimbursement agents for pattern {pr['pattern_rank']}...")

                                # Save reimbursement payload to S3 (matches actual API request structure)
                                reim_payload = {
                                    "context": {
                                        "pattern": pr,
                                        "cards": all_cards
                                    },
                                    "conversation_id": _rid,
                                    "query": f"Analyze reimbursement policies for pattern {pr['pattern_rank']}",
                                    "job_id": f"{_rid}_job"
                                }
                                s3_logger.save_to_s3(reim_payload, f"reimbursement_payload_{_rid}.json", statscl_mdl_cd, 'payload')

                                
                                reimbursement_response, reimbursement_results = call_reimbursement_agents(_rid, pr, all_cards)
                                print(f"[LOG] Reimbursement agents completed for pattern {pr['pattern_rank']}. Status code: {reimbursement_response.status_code if reimbursement_response else None}")
                                # Save reimbursement response to S3
                                s3_logger.save_to_s3(reimbursement_results, f"reimbursement_response_{_rid}.json", statscl_mdl_cd, 'agents_results')
                            if reimbursement_response:
                                s_code = reimbursement_response.status_code
                            else:
                                s_code = 400
                                reim_success = False
                            full_reim_result.append(reimbursement_results)
                            processed_pattern_result.append(generate_pattern_final_result(s_code, pr, reimbursement_results))
                        
                        # Update reimbursement sub-job status
                        if job_tracker:
                            if reim_success:
                                job_tracker.update_sub_job_status(
                                    job_id, reimbursement_subjob_id, "SUCCESS",
                                    response_code=200,
                                    s3_result_path=reim_result_path
                                )
                            else:
                                job_tracker.update_sub_job_status(
                                    job_id, reimbursement_subjob_id, "FAILED",
                                    error_message="One or more reimbursement calls failed",
                                    response_code=400
                                )
                    except Exception as reim_error:
                        print(f"[LOG] Reimbursement agents failed: {str(reim_error)}")
                        if job_tracker:
                            job_tracker.update_sub_job_status(
                                job_id, reimbursement_subjob_id, "FAILED",
                                error_message=str(reim_error),
                                response_code=500
                            )
                        raise 

                    reim_result_output = []
                    for _index in range(len(full_reim_result)):
                        if "detail" in full_reim_result[_index]:
                            reim_result_output.append({
                                "pattern_rank":_index+1,
                                "summary_table":{},
                                "reimbursement_policies":[],
                                "elevance_executive_summary":None,
                                "policies_processed":0,
                                "policies_successful":0,
                                "policies_failed":0
                            })
                            
                        else:
                            reim_result_output.append(full_reim_result[_index]['output'])    
                    s3_logger.save_to_s3(processed_pattern_result, f"pattern_final_{_id}.json", statscl_mdl_cd, 'agents_results')                        
                    
                    # Track recommendation agent as sub-job
                    rec_payload_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/payload/recommendation_payload_{_id}.json"
                    rec_result_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/agents_results/recommendation_response_{_id}.json"
                    if job_tracker:
                        recommendation_subjob_id = job_tracker.create_sub_job(job_id, "recommendation", s3_payload_path=rec_payload_path)
                    
                    print("[LOG] Starting health check before recommendation agents")
                    try:
                        if not check_api_health(retry_interval=2, timeout=5):
                            print("[LOG] API health check failed before recommendation agents. Using default values.")
                            recommendation_results = {"success":True,"result":{"metadata":{},"recommendations":[],"skipped_patterns":[],"processing_log":[]},"metadata":{"agent_name":"recommendation_synthesis"}}
                            recommendation_response = None
                            print("[LOG] Recommendation agents failed (health check)")
                            if job_tracker:
                                job_tracker.update_sub_job_status(
                                    job_id, recommendation_subjob_id, "FAILED",
                                    error_message="Health check failed",
                                    response_code=503
                                )
                        else:
                            print("[LOG] API health check passed for recommendation agents")
                            print("[LOG] Calling recommendation agents...")
                            # Save recommendation payload to S3
                            
                            rec_payload = combine_pattern_reimbursement(pattern_result,reim_result_output)
                            s3_logger.save_to_s3(rec_payload, f"recommendation_payload_{_id}.json", statscl_mdl_cd, 'payload')

                            recommendation_response, recommendation_results = call_recommendation_agents(rec_payload)
                            #print(recommendation_results)
                            # Save recommendation response to S3
                            s3_logger.save_to_s3(recommendation_results, f"recommendation_response_{_id}.json", statscl_mdl_cd, 'agents_results')
                            
                            # Update recommendation sub-job status
                            if job_tracker:
                                if recommendation_results.get("success"):
                                    job_tracker.update_sub_job_status(
                                        job_id, recommendation_subjob_id, "SUCCESS",
                                        response_code=200,
                                        s3_result_path=rec_result_path
                                    )
                                else:
                                    job_tracker.update_sub_job_status(
                                        job_id, recommendation_subjob_id, "FAILED",
                                        error_message="Recommendation agent returned failure",
                                        response_code=400
                                    )
                    except Exception as rec_error:
                        print(f"[LOG] Recommendation agents failed: {str(rec_error)}")
                        if job_tracker:
                            job_tracker.update_sub_job_status(
                                job_id, recommendation_subjob_id, "FAILED",
                                error_message=str(rec_error),
                                response_code=500
                            )
                        raise                                

                
                # Policy agent is independent - call it regardless of business patterns
                print("[LOG] Starting health check before policy agent")
                if not check_api_health(retry_interval=2, timeout=5):
                    print("[LOG] API health check failed before policy agent. Using default values.")
                    policy_results = {}
                else:
                    print("[LOG] API health check passed for policy agent")
                    print("[LOG] Calling policy agent...")
                    policy_results = call_policy_agent()
                    print("[LOG] Policy agent completed")
                
                
                # Store results with better error handling
                try:
                    result_record = {
                        "snap_year_mnth_nbr": int(snap_year_mnth_nbr),
                        "trnd_tm_prd_end_mnth_nbr": int(trnd_tm_prd_end_mnth_nbr),
                        "trnd_tm_prd_cd": trnd_tm_prd_cd,
                        "lob_shrt_desc": lob_shrt_desc,
                        "lob_cd": insights_lob_code,
                        "statscl_mdl_cd": statscl_mdl_cd,
                        "final_payload": final_payload,
                        "anomaly_json": anomaly_json,
                        "deep_dive_json": deep_dive_json,
                        "pattern_result": processed_pattern_result,
                        "recommendation_result": recommendation_results['result'],
                        "policy_result":policy_results,
                        "status": "success",
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Validate result record before saving
                    results.append(result_record)
                    #print(result_record)
                    
                    # Save individual combination result to JSON file
                    combination_filename = f"combination_{snap_year_mnth_nbr}_{trnd_tm_prd_end_mnth_nbr}_{trnd_tm_prd_cd}_{lob_shrt_desc}_{statscl_mdl_cd}.json"
                    combination_filename = combination_filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    
                    # Test JSON serialization before saving
                    try:
                        test_json = json.dumps(result_record, indent=2)
                    except Exception as json_error:
                        # Create minimal record if JSON serialization fails
                        result_record = {
                            "snap_year_mnth_nbr": int(snap_year_mnth_nbr),
                            "trnd_tm_prd_end_mnth_nbr": int(trnd_tm_prd_end_mnth_nbr),
                            "trnd_tm_prd_cd": trnd_tm_prd_cd,
                            "lob_shrt_desc": lob_shrt_desc,
                            "statscl_mdl_cd": statscl_mdl_cd,
                            "status": "json_serialization_error",
                            "error": str(json_error),
                            "processed_at": datetime.now().isoformat()
                        }
                    
                    with open(combination_filename, 'w') as f:
                        json.dump(result_record, f, indent=2)
                    
                    # Store to Snowflake immediately after each combination
                    if use_snowflake and result_record.get('status') == 'success':
                        try:
                            store_agent_results_to_snowflake(session, env, env_config, df_insights, [result_record], lob, s3_logger)
                            print(f"Stored combination to Snowflake successfully")
                        except Exception as e:
                            print(f"Error storing combination to Snowflake: {str(e)}")
                    
                    # Verify file was written correctly
                    with open(combination_filename, 'r') as f:
                        content = f.read()
                        if len(content) < 100:  # Suspiciously short
                            print(f"Warning: File content seems short: {len(content)} characters")
                    
                    print(f"Saved individual result: {combination_filename}")
                    
                    # Mark job as SUCCESS with S3 result path
                    if job_tracker:
                        final_result_s3_path = f"{s3_logger.base_path}/{s3_logger.timestamp_folder}/{model_sanitized}/final_result/snowflake_{_id}.json"
                        job_tracker.update_job_status(job_id, "SUCCESS", s3_path=final_result_s3_path)
                    print(f"✅ Job {job_id} completed successfully")
                    
                except Exception as e:
                    import traceback
                    print(f"Error creating/saving individual result: {str(e)}")
                    
                    # Create error record
                    error_record = {
                        "snap_year_mnth_nbr": int(snap_year_mnth_nbr),
                        "trnd_tm_prd_end_mnth_nbr": int(trnd_tm_prd_end_mnth_nbr),
                        "trnd_tm_prd_cd": trnd_tm_prd_cd,
                        "lob_shrt_desc": lob_shrt_desc,
                        "statscl_mdl_cd": statscl_mdl_cd,
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Try to save error record
                    try:
                        error_filename = f"error_{snap_year_mnth_nbr}_{trnd_tm_prd_end_mnth_nbr}_{trnd_tm_prd_cd}_{lob_shrt_desc}_{statscl_mdl_cd}.json"
                        error_filename = error_filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        with open(error_filename, 'w') as f:
                            json.dump(error_record, f, indent=2)
                    except Exception as save_error:
                        print(f"Failed to save error record: {str(save_error)}")
                    
                    # Mark job as FAILED
                    if job_tracker:
                        job_tracker.update_job_status(job_id, "FAILED", error_message=str(e))
                    print(f"❌ Job {job_id} failed: {str(e)}")
                
                print(f"Successfully processed combination")
            
            # Exception handler for the outer try block (per combination)
            except Exception as combo_error:
                print(f"❌ Error processing combination {job_id}: {str(combo_error)}")
                if job_tracker:
                    job_tracker.update_job_status(job_id, "FAILED", error_message=str(combo_error))
                # Continue to next combination instead of failing entire pipeline
                continue
        
        # All combinations processed successfully
        checkpoint_mgr.mark_success(run_id)
        if job_tracker:
            job_tracker.print_run_summary()
        print(f"\n🎉 Pipeline completed successfully!")
        
    except Exception as pipeline_error:
        # Pipeline-level failure
        print(f"\n❌ Pipeline failed: {str(pipeline_error)}")
        checkpoint_mgr.mark_failed(run_id)
        if job_tracker:
            job_tracker.print_run_summary()
        raise
    
    print(f"\nAPI Processing completed!")
    print(f"Total records processed: {len(results)}")
    
    # Simple validation and summary for your working logic
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') != 'success']
    
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total combinations processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        print(f"\nSuccessful combinations:")
        for result in successful_results:
            print(f"  - {result['snap_year_mnth_nbr']}, {result['trnd_tm_prd_end_mnth_nbr']}, {result['trnd_tm_prd_cd']}, {result['lob_shrt_desc']}, {result['statscl_mdl_cd']}")
            
            # Show correlation results count
            if 'correlation_results' in result:
                corr = result['correlation_results']
                print(f"    Correlation: {len(corr.get('states', {}))} states, {len(corr.get('providers', {}))} providers, {len(corr.get('drgs', {}))} DRGs")
            
            # Show pattern result count
            if 'pattern_result' in result:
                pattern = result['pattern_result']
                if isinstance(pattern, list):
                    print(f"    Pattern Agent: {len(pattern)} patterns found")
                else:
                    print(f"    Pattern Agent: ✓")
    
    # Validate results using existing function for compatibility
    try:
        validation_summary = validate_agent_results(results)
    except Exception as e:
        print(f"Validation function failed (expected with simplified structure): {str(e)}")
    
    # Note: Snowflake storage now happens per combination during processing
    
    # Close sessions if created
    if use_snowflake:
        session.close()
        print("✅ Main Snowflake session closed")
    
    # Close DataLab connections
    if job_tracking_enabled:
        try:
            if datalab_connector:
                datalab_connector.close()
                print("✅ DataLab connector closed")
        except:
            pass  # Connector may not exist if error occurred early
        
        try:
            if datalab_session:
                datalab_session.close()
                print("✅ DataLab session closed")
        except:
            pass  # Session may not exist if error occurred early

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DR Pattern Analysis Pipeline for multiple environments')
    parser.add_argument('--env', required=True, choices=['dv', 'ts', 'pl', 'pr'],
                        help='Target environment (dv, ts, pl, pr)')
    parser.add_argument('--csv-file', default="SampleData-dev-deep-research.csv",
                        help='Path to the CSV file to process (used only with --use-csv)')
    parser.add_argument('--use-csv', action='store_true',
                        help='Use CSV file instead of Snowflake')
    parser.add_argument('--statscl-mdl-cd', default="IP AUTH",
                        help='STATSCL_MDL_CD filter value (default: IP AUTH)')
    parser.add_argument('--insght-type-nm', default="DEEP_DIVE",
                        help='INSGHT_TYPE_NM filter value (default: DEEP_DIVE)')
    parser.add_argument('--lob', choices=['gbd', 'nogbd'], default="gbd",
                        help='LOB type - gbd or nogbd (default: gbd)')
    parser.add_argument('--snap-year-mnth-nbr', type=int, default=None,
                        help='SNAP_YEAR_MNTH_NBR filter value (optional, e.g., 202604)')
    parser.add_argument('--trnd-tm-prd-cd', type=str, default=None,
                        help='TRND_TM_PRD_CD filter value (optional, e.g., R3, R6, R12)')
    parser.add_argument('--lob-shrt-desc', type=str, default=None,
                        help='LOB_SHRT_DESC filter value (optional, e.g., Commercial_Individual)')
    
    args = parser.parse_args()
    
    try:
        use_snowflake = not args.use_csv
        main(args.env, use_snowflake, args.csv_file, args.lob, args.statscl_mdl_cd, args.snap_year_mnth_nbr, args.trnd_tm_prd_cd, args.lob_shrt_desc)
    except Exception as e:
        import traceback
        error_msg = f"Error in Pattern Analysis Pipeline: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        sys.exit(1)
