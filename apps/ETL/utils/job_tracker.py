"""
Job Tracker Utility for Deep Research ETL Pipeline
Manages checkpoint/resume functionality using Snowflake tables
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from snowflake.snowpark import Session
import json


class JobTracker:
    """
    Manages job tracking for checkpoint/resume functionality
    
    Two-level tracking:
    1. Parent Job (Combination level) - COC_CMN_DEEP_RSRCH_JOB_TRACKER
    2. Sub-Job (Agent level) - COC_CMN_DEEP_RSRCH_SUBJOB_TRACKER
    """
    
    def __init__(self, session: Session, run_id: str, datalab_schema: str = "NON_CRTFD_AIFS.DL_DV_TMBR_STG_NOGBD", connector=None):
        """
        Initialize Job Tracker
        
        Args:
            session: Snowpark session (DataLab connection)
            run_id: Unique run identifier (timestamp-based)
            datalab_schema: DataLab schema for job tracking tables (default: NON_CRTFD_AIFS.DL_DV_TMBR_STG_NOGBD)
            connector: Optional snowflake.connector connection for explicit transaction control
        """
        self.session = session
        self.connector = connector
        self.run_id = run_id
        self.schema = datalab_schema
        self.job_table = f"{datalab_schema}.COC_CMN_DEEP_RSRCH_JOB_TRACKER"
        self.subjob_table = f"{datalab_schema}.COC_CMN_DEEP_RSRCH_SUBJOB_TRACKER"
        
        # Ensure tables exist
        self._create_tables_if_not_exist()
    
    def _execute_dml(self, sql: str, silent: bool = False):
        """
        Execute DML statement with explicit commit
        
        Args:
            sql: SQL statement to execute
            silent: If True, suppress SQL logging (default: False)
        """
        try:
            # Log SQL statement for debugging (unless silent mode)
            if not silent:
                print(f"[SQL] Executing DML:\n{sql}")
            
            # Use connector if available for explicit transaction control
            if self.connector:
                cursor = self.connector.cursor()
                cursor.execute(sql)
                self.connector.commit()
                cursor.close()
                if not silent:
                    print(f"[SQL] ✅ DML executed and committed successfully")
            else:
                # Fallback to Snowpark session
                self.session.sql(sql).collect()
                if not silent:
                    print(f"[SQL] ⚠️  DML executed via Snowpark (no explicit commit)")
        except Exception as e:
            print(f"[SQL] ❌ DML execution failed: {str(e)}")
            raise e
    
    def _create_tables_if_not_exist(self):
        """Create tracking tables if they don't exist"""
        
        # Parent job tracker table
        job_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.job_table} (
            RUN_ID VARCHAR(50) NOT NULL,
            JOB_ID VARCHAR(100) NOT NULL,
            SNAP_YEAR_MNTH_NBR INT,
            TRND_TM_PRD_END_MNTH_NBR INT,
            TRND_TM_PRD_CD VARCHAR(10),
            LOB_CD VARCHAR(50),
            LOB_SHRT_DESC VARCHAR(100),
            STATSCL_MDL_CD VARCHAR(50),
            JOB_STATUS VARCHAR(20) DEFAULT 'PENDING',
            JOB_MESSAGE VARCHAR(1000),
            TOTAL_SUB_JOBS INT DEFAULT 0,
            COMPLETED_SUB_JOBS INT DEFAULT 0,
            FAILED_SUB_JOBS INT DEFAULT 0,
            START_TIME TIMESTAMP_NTZ,
            END_TIME TIMESTAMP_NTZ,
            ERROR_MESSAGE VARCHAR(5000),
            S3_RESULT_PATH VARCHAR(500),
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (RUN_ID, JOB_ID)
        )
        """
        
        # Sub-job tracker table
        subjob_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.subjob_table} (
            RUN_ID VARCHAR(50) NOT NULL,
            JOB_ID VARCHAR(100) NOT NULL,
            SUB_JOB_ID VARCHAR(150) NOT NULL,
            AGENT_NAME VARCHAR(50),
            SUB_JOB_STATUS VARCHAR(20) DEFAULT 'PENDING',
            START_TIME TIMESTAMP_NTZ,
            END_TIME TIMESTAMP_NTZ,
            DURATION_SECONDS FLOAT,
            ERROR_MESSAGE VARCHAR(5000),
            API_RESPONSE_CODE INT,
            S3_PAYLOAD_PATH VARCHAR(500),
            S3_RESULT_PATH VARCHAR(500),
            RETRY_COUNT INT DEFAULT 0,
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (RUN_ID, JOB_ID, SUB_JOB_ID)
        )
        """
        
        try:
            self.session.sql(job_table_ddl).collect()
            self.session.sql(subjob_table_ddl).collect()
            print(f"✅ Job tracking tables verified/created in {self.schema}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create tracking tables: {str(e)}")
    
    def generate_job_id(self, combination: Dict[str, Any]) -> str:
        """
        Generate unique job ID from combination
        
        Args:
            combination: Dictionary with combination keys
            
        Returns:
            str: Unique job ID
        """
        # Sanitize strings for job ID - replace spaces and special characters with underscores
        safe_lob = str(combination['lob_shrt_desc']).replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_model = str(combination['statscl_mdl_cd']).replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        return f"{combination['snap_year_mnth_nbr']}_{combination['trnd_tm_prd_end_mnth_nbr']}_{combination['trnd_tm_prd_cd']}_{safe_lob}_{safe_model}"
    
    def initialize_jobs(self, combinations: List[Dict[str, Any]]) -> int:
        """
        Initialize all jobs with PENDING status using batch INSERT
        
        Args:
            combinations: List of combination dictionaries
            
        Returns:
            int: Number of jobs initialized
        """
        if not combinations:
            return 0
        
        # Build batch INSERT with multiple VALUES
        values_list = []
        for combo in combinations:
            job_id = self.generate_job_id(combo)
            values_list.append(f"""(
                '{self.run_id}',
                '{job_id}',
                {combo.get('snap_year_mnth_nbr')},
                {combo.get('trnd_tm_prd_end_mnth_nbr')},
                '{combo.get('trnd_tm_prd_cd')}',
                '{combo.get('lob_cd', '')}',
                '{combo.get('lob_shrt_desc')}',
                '{combo.get('statscl_mdl_cd')}',
                'PENDING',
                4
            )""")
        
        # Single batch INSERT
        batch_insert_sql = f"""
        INSERT INTO {self.job_table} 
        (RUN_ID, JOB_ID, SNAP_YEAR_MNTH_NBR, TRND_TM_PRD_END_MNTH_NBR, 
         TRND_TM_PRD_CD, LOB_CD, LOB_SHRT_DESC, STATSCL_MDL_CD, 
         JOB_STATUS, TOTAL_SUB_JOBS)
        VALUES {','.join(values_list)}
        """
        
        try:
            self._execute_dml(batch_insert_sql, silent=True)
            initialized_count = len(combinations)
            print(f"✅ Initialized {initialized_count} jobs with PENDING status (batch INSERT)")
            return initialized_count
        except Exception as e:
            print(f"❌ Batch INSERT failed: {str(e)}")
            print(f"⚠️  Falling back to individual INSERTs...")
            
            # Fallback to individual INSERTs if batch fails
            initialized_count = 0
            for combo in combinations:
                job_id = self.generate_job_id(combo)
                insert_sql = f"""
                INSERT INTO {self.job_table} 
                (RUN_ID, JOB_ID, SNAP_YEAR_MNTH_NBR, TRND_TM_PRD_END_MNTH_NBR, 
                 TRND_TM_PRD_CD, LOB_CD, LOB_SHRT_DESC, STATSCL_MDL_CD, 
                 JOB_STATUS, TOTAL_SUB_JOBS)
                VALUES (
                    '{self.run_id}',
                    '{job_id}',
                    {combo.get('snap_year_mnth_nbr')},
                    {combo.get('trnd_tm_prd_end_mnth_nbr')},
                    '{combo.get('trnd_tm_prd_cd')}',
                    '{combo.get('lob_cd', '')}',
                    '{combo.get('lob_shrt_desc')}',
                    '{combo.get('statscl_mdl_cd')}',
                    'PENDING',
                    4
                )
                """
                try:
                    self._execute_dml(insert_sql, silent=True)
                    initialized_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Could not initialize job {job_id}: {str(e)}")
            
            print(f"✅ Initialized {initialized_count} jobs with PENDING status (individual INSERTs)")
            return initialized_count
    
    def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all pending and in-progress jobs for current run
        Includes IN_PROGRESS jobs to handle interrupted runs
        
        Returns:
            List of pending job dictionaries
        """
        query = f"""
        SELECT 
            JOB_ID, SNAP_YEAR_MNTH_NBR, TRND_TM_PRD_END_MNTH_NBR,
            TRND_TM_PRD_CD, LOB_CD, LOB_SHRT_DESC, STATSCL_MDL_CD
        FROM {self.job_table}
        WHERE RUN_ID = '{self.run_id}'
        AND JOB_STATUS IN ('PENDING', 'IN_PROGRESS')
        ORDER BY CREATED_AT
        """
        
        result = self.session.sql(query).collect()
        
        pending_jobs = []
        for row in result:
            pending_jobs.append({
                'job_id': row['JOB_ID'],
                'snap_year_mnth_nbr': row['SNAP_YEAR_MNTH_NBR'],
                'trnd_tm_prd_end_mnth_nbr': row['TRND_TM_PRD_END_MNTH_NBR'],
                'trnd_tm_prd_cd': row['TRND_TM_PRD_CD'],
                'lob_cd': row['LOB_CD'],
                'lob_shrt_desc': row['LOB_SHRT_DESC'],
                'statscl_mdl_cd': row['STATSCL_MDL_CD']
            })
        
        return pending_jobs
    
    def update_job_status(self, job_id: str, status: str, 
                         error_message: str = None, 
                         s3_path: str = None):
        """
        Update job status
        
        Args:
            job_id: Job identifier
            status: New status (PENDING, IN_PROGRESS, SUCCESS, FAILED)
            error_message: Optional error message
            s3_path: Optional S3 result path
        """
        update_sql = f"""
        UPDATE {self.job_table}
        SET JOB_STATUS = '{status}',
            UPDATED_AT = CURRENT_TIMESTAMP()
        """
        
        if status == 'IN_PROGRESS':
            update_sql += ", START_TIME = CURRENT_TIMESTAMP()"
        elif status in ['SUCCESS', 'FAILED']:
            update_sql += ", END_TIME = CURRENT_TIMESTAMP()"
        
        if error_message:
            escaped_msg = error_message.replace("'", "''")
            update_sql += f", ERROR_MESSAGE = '{escaped_msg}'"
        
        if s3_path:
            update_sql += f", S3_RESULT_PATH = '{s3_path}'"
        
        update_sql += f" WHERE RUN_ID = '{self.run_id}' AND JOB_ID = '{job_id}'"
        
        try:
            self._execute_dml(update_sql, silent=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not update job status: {str(e)}")
    
    def create_sub_job(self, job_id: str, agent_name: str, s3_payload_path: str = None) -> str:
        """
        Create a sub-job entry for an agent
        
        Args:
            job_id: Parent job ID
            agent_name: Agent name (correlation, pattern, reimbursement, recommendation)
            s3_payload_path: Optional S3 path where payload is saved
            
        Returns:
            str: Sub-job ID
        """
        sub_job_id = f"{job_id}_{agent_name}"
        
        insert_sql = f"""
        INSERT INTO {self.subjob_table}
        (RUN_ID, JOB_ID, SUB_JOB_ID, AGENT_NAME, SUB_JOB_STATUS, START_TIME, RETRY_COUNT
        {', S3_PAYLOAD_PATH' if s3_payload_path else ''})
        VALUES (
            '{self.run_id}',
            '{job_id}',
            '{sub_job_id}',
            '{agent_name}',
            'IN_PROGRESS',
            CURRENT_TIMESTAMP(),
            0
            {f", '{s3_payload_path}'" if s3_payload_path else ''}
        )
        """
        
        try:
            self._execute_dml(insert_sql)
        except Exception as e:
            print(f"⚠️  Warning: Could not create sub-job: {str(e)}")
        
        return sub_job_id
    
    def update_sub_job_status(self, job_id: str, sub_job_id: str, status: str,
                             error_message: str = None,
                             response_code: int = None,
                             retry_count: int = None,
                             s3_payload_path: str = None,
                             s3_result_path: str = None):
        """
        Update sub-job status
        
        Args:
            job_id: Parent job ID
            sub_job_id: Sub-job identifier
            status: New status (PENDING, IN_PROGRESS, SUCCESS, FAILED)
            error_message: Optional error message
            response_code: Optional API response code
            retry_count: Optional retry count
            s3_payload_path: Optional S3 payload path
            s3_result_path: Optional S3 result path
        """
        update_sql = f"""
        UPDATE {self.subjob_table}
        SET SUB_JOB_STATUS = '{status}',
            UPDATED_AT = CURRENT_TIMESTAMP()
        """
        
        if status in ['SUCCESS', 'FAILED']:
            update_sql += ", END_TIME = CURRENT_TIMESTAMP()"
            update_sql += ", DURATION_SECONDS = DATEDIFF('second', START_TIME, CURRENT_TIMESTAMP())"
        
        if error_message:
            escaped_msg = error_message.replace("'", "''")
            update_sql += f", ERROR_MESSAGE = '{escaped_msg}'"
        
        if response_code is not None:
            update_sql += f", API_RESPONSE_CODE = {response_code}"
        
        if retry_count is not None:
            update_sql += f", RETRY_COUNT = {retry_count}"
        
        if s3_payload_path:
            update_sql += f", S3_PAYLOAD_PATH = '{s3_payload_path}'"
        
        if s3_result_path:
            update_sql += f", S3_RESULT_PATH = '{s3_result_path}'"
        
        update_sql += f" WHERE RUN_ID = '{self.run_id}' AND JOB_ID = '{job_id}' AND SUB_JOB_ID = '{sub_job_id}'"
        
        try:
            self._execute_dml(update_sql)
            
            # Update parent job counters
            self._update_parent_job_counters(job_id)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not update sub-job status: {str(e)}")
    
    def _update_parent_job_counters(self, job_id: str):
        """Update parent job completion counters and set job message"""
        
        # Get sub-job details including agent names and statuses
        details_sql = f"""
        SELECT 
            AGENT_NAME,
            SUB_JOB_STATUS,
            ERROR_MESSAGE
        FROM {self.subjob_table}
        WHERE RUN_ID = '{self.run_id}' AND JOB_ID = '{job_id}'
        ORDER BY CREATED_AT
        """
        
        details_result = self.session.sql(details_sql).collect()
        
        completed = 0
        failed = 0
        failed_agents = []
        
        for row in details_result:
            if row['SUB_JOB_STATUS'] == 'SUCCESS':
                completed += 1
            elif row['SUB_JOB_STATUS'] == 'FAILED':
                failed += 1
                agent_info = row['AGENT_NAME']
                if row['ERROR_MESSAGE']:
                    agent_info += f" ({row['ERROR_MESSAGE'][:50]}...)" if len(row['ERROR_MESSAGE']) > 50 else f" ({row['ERROR_MESSAGE']})"
                failed_agents.append(agent_info)
        
        # Build job message
        if completed == 4 and failed == 0:
            job_message = "All 4 sub-jobs completed successfully"
        elif failed > 0:
            job_message = f"{failed} sub-job(s) failed: {', '.join(failed_agents)}"
        elif completed > 0:
            job_message = f"{completed} sub-job(s) completed, {4 - completed - failed} pending"
        else:
            job_message = "No sub-jobs completed yet"
        
        # Update parent job with counters and message
        escaped_message = job_message.replace("'", "''")
        update_sql = f"""
        UPDATE {self.job_table}
        SET COMPLETED_SUB_JOBS = {completed},
            FAILED_SUB_JOBS = {failed},
            JOB_MESSAGE = '{escaped_message}',
            UPDATED_AT = CURRENT_TIMESTAMP()
        WHERE RUN_ID = '{self.run_id}' AND JOB_ID = '{job_id}'
        """
        
        self._execute_dml(update_sql, silent=True)
        
        # If all sub-jobs completed, mark parent as SUCCESS
        if completed == 4 and failed == 0:
            self.update_job_status(job_id, 'SUCCESS')
        elif failed > 0:
            self.update_job_status(job_id, 'FAILED', 
                                 error_message=f"{failed} sub-job(s) failed")
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get summary of current run
        
        Returns:
            Dictionary with run statistics
        """
        summary_sql = f"""
        SELECT 
            COUNT(*) as total_jobs,
            COUNT(CASE WHEN JOB_STATUS = 'PENDING' THEN 1 END) as pending,
            COUNT(CASE WHEN JOB_STATUS = 'IN_PROGRESS' THEN 1 END) as in_progress,
            COUNT(CASE WHEN JOB_STATUS = 'SUCCESS' THEN 1 END) as success,
            COUNT(CASE WHEN JOB_STATUS = 'FAILED' THEN 1 END) as failed
        FROM {self.job_table}
        WHERE RUN_ID = '{self.run_id}'
        """
        
        result = self.session.sql(summary_sql).collect()
        
        if result:
            row = result[0]
            return {
                'run_id': self.run_id,
                'total_jobs': row['TOTAL_JOBS'],
                'pending': row['PENDING'],
                'in_progress': row['IN_PROGRESS'],
                'success': row['SUCCESS'],
                'failed': row['FAILED']
            }
        
        return {}
    
    def print_run_summary(self):
        """Print formatted run summary"""
        summary = self.get_run_summary()
        
        if summary:
            print("\n" + "="*60)
            print(f"📊 RUN SUMMARY - {summary['run_id']}")
            print("="*60)
            print(f"Total Jobs:      {summary['total_jobs']}")
            print(f"✅ Success:      {summary['success']}")
            print(f"⏳ Pending:      {summary['pending']}")
            print(f"🔄 In Progress:  {summary['in_progress']}")
            print(f"❌ Failed:       {summary['failed']}")
            print("="*60 + "\n")
