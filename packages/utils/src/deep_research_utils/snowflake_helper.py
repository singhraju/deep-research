from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import to_varchar
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, FloatType, VariantType
import os
import pandas as pd
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, List, Iterator
from dataclasses import dataclass
from deep_research_utils.logger_config import get_logger
from deep_research_utils.app_constant import AppConstants
from flexible_snowflake_connector import get_snowflake_connection
logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for Snowflake operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    rows_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def rows_per_second(self) -> float:
        duration = self.duration_seconds
        return self.rows_processed / duration if duration > 0 else 0

class SnowparkHelper:
    """
    A helper class for working with Snowpark connections and executing queries.
    Supports programmatic (password-based) and vault-based authentication.
    
    Optimizations included:
    - Connection pooling and session reuse
    - Configurable batch sizes for bulk operations
    - Performance monitoring and metrics
    - Bulk staging operations
    - Thread-safe operations
    """
    
    # Default table name mappings - can be overridden by environment variables
    DEFAULT_TABLE_NAMES = {
        "DOCS": "DOCS",
        "PAGES": "PAGES", 
        "TEXT_BLOCKS": "TEXT_BLOCKS",
        "SECTIONS": "SECTIONS",
        "SECTION_PAGE_REGIONS": "SECTION_PAGE_REGIONS",
        "SECTION_URLS": "SECTION_URLS",
        "SECTION_ATTACHMENTS": "SECTION_ATTACHMENTS",
        "SECTION_TEXTS": "SECTION_TEXTS",
        "TABLES": "TABLES",
        "TABLE_CELLS": "TABLE_CELLS",
        "CHUNKS": "CHUNKS"
    }
    
    def __init__(self, connection_type: str = None, **kwargs):
        """
        Initialize the SnowflakeHelper with connection parameters.
        
        Args:
            connection_type: Type of connection, either "programmatic" or "vault"
            **kwargs: Connection parameters that will override the defaults
                - batch_size: Default batch size for bulk operations (default: 5000)
                - max_workers: Maximum number of parallel workers (default: 4)
                - enable_metrics: Enable performance metrics collection (default: True)
                - connection_pool_size: Number of connections to maintain (default: 3)
        """
        if connection_type is None:
            connection_type = AppConstants.SNOWFLAKE_CONNECTION_TYPE
            logger.info(f"Auto-detected Snowflake connection type: {connection_type}")
        
        self.connection_type = connection_type
        self.connection_params = {}
        self.env_var = {}
        self.session = None
        
        # Performance and optimization settings
        self.batch_size = kwargs.get("batch_size", 5000)
        self.max_workers = kwargs.get("max_workers", 4)
        self.enable_metrics = kwargs.get("enable_metrics", True)
        self.connection_pool_size = kwargs.get("connection_pool_size", 3)
        
        # Performance tracking
        self.metrics: List[PerformanceMetrics] = []
        self._metrics_lock = threading.Lock()
        
        # Connection pooling
        self._session_pool: List[Session] = []
        self._pool_lock = threading.Lock()
        
        # Initialize table names from environment or defaults
        self.table_names = self._load_table_names()
        self.table_prefix = kwargs.get("table_prefix", AppConstants.SNOWFLAKE_TABLE_PREFIX)
        self.table_suffix = kwargs.get("table_suffix", AppConstants.SNOWFLAKE_TABLE_SUFFIX)
        self.database = kwargs.get("database") 
        self.schema = kwargs.get("schema", "PUBLIC")
        
        if connection_type.lower() == "programmatic":
            self._setup_programmatic_connection(**kwargs)
        elif connection_type.lower() == "vault":
            self._setup_vault_connection(**kwargs)
        else:
            raise ValueError("Invalid connection_type. Please use 'programmatic' or 'vault'. Okta authentication is deprecated.")

        # Create the initial session and populate pool
        self._create_session()
        self._initialize_connection_pool()
    
    def _load_table_names(self) -> Dict[str, str]:
        """Load table names from environment variables or use defaults."""
        table_names = {}
        for logical_name, default_name in self.DEFAULT_TABLE_NAMES.items():
            env_var_name = f"SNOWFLAKE_{logical_name}_TABLE"
            table_names[logical_name] = os.getenv(env_var_name, default_name)
        return table_names
    
    def get_table_name(self, logical_name: str) -> str:
        """Get the physical table name for a logical table name."""
        base_name = self.table_names.get(logical_name, logical_name)
        return f"{self.table_prefix}{base_name}{self.table_suffix}"
    
    def get_qualified_table_name(self, logical_name: str) -> str:
        """Get the fully qualified table name (database.schema.table)."""
        table_name = self.get_table_name(logical_name)
        if self.database:
            return f"{self.database}.{self.schema}.{table_name}"
        return f"{self.schema}.{table_name}"
        
    # def _setup_okta_connection(self, **kwargs):
    #     """Set up connection parameters for Okta authentication"""
    #     # Default values for Timber
    #     self.connection_params = {
    #         "account": kwargs.get("account", "carelon-edaprod1.privatelink"),
    #         "authenticator": kwargs.get("authenticator", "https://portalsso.elevancehealth.com/snowflake/okta"),
    #         "user": kwargs.get("user", ""),
    #         "password": kwargs.get("password", ""),
    #         "warehouse": kwargs.get("warehouse", "DL_AIFS_TMBR_USER_WH_L"),
    #         "database": kwargs.get("database", "NON_CRTFD_AIFS"),
    #         "schema": kwargs.get("schema", "DL_DV_TMBR")
    #     }
        
    def _setup_programmatic_connection(self, **kwargs):
        """Set up connection parameters for programmatic access with password authentication."""
        logger.info("🔑 Configuring PROGRAMMATIC connection (password-based authentication)")
        logger.debug(f"Snowflake account: {kwargs.get('account', AppConstants.SNOWFLAKE_ACCOUNT)}")
        logger.debug(f"Snowflake user: {kwargs.get('user', AppConstants.SNOWFLAKE_USER)}")
        
        self.connection_params = {
            "account": kwargs.get("account", AppConstants.SNOWFLAKE_ACCOUNT),
            "user": kwargs.get("user", AppConstants.SNOWFLAKE_USER),
            "password": kwargs.get("password", AppConstants.SNOWFLAKE_SECRET),
            "warehouse": kwargs.get("warehouse", AppConstants.SNOWFLAKE_WAREHOUSE),
            # "role": kwargs.get("role", "SRCCOCDP_PRIVS"),
            "role": kwargs.get("user", AppConstants.SNOWFLAKE_USER) + "_PRIVS",
            # "authenticator": 'https://portalsso.elevancehealth.com/snowflake/okta'
        }
        
        # Add optional parameters if provided
        if "database" in kwargs:
            self.connection_params["database"] = kwargs["database"]
        if "schema" in kwargs:
            self.connection_params["schema"] = kwargs["schema"]
    
    def _setup_vault_connection(self, **kwargs):
        """Set up connection parameters for Vault-based authentication."""
        logger.info("🔐 Configuring VAULT connection (secure vault-based authentication)")
        logger.debug(f"Vault URL: {kwargs.get('vault_url', AppConstants.VAULT_URL)}")
        logger.debug(f"Service ID: {kwargs.get('service_id', AppConstants.SNOWFLAKE_SERVICE_ID)}")
        logger.debug(f"Vault namespace: {kwargs.get('vault_namespace', AppConstants.VAULT_NAMESPACE)}")
    
        # Store vault parameters for use in session creation
        self.vault_params = {
            "service_id": kwargs.get("service_id", AppConstants.SNOWFLAKE_SERVICE_ID),
            "vault_role_name": kwargs.get("vault_role_name", AppConstants.VAULT_ROLE_NAME),
            "vault_namespace": kwargs.get("vault_namespace", AppConstants.VAULT_NAMESPACE),
            "vault_path": kwargs.get("vault_path", AppConstants.VAULT_PATH),
            "vault_url": kwargs.get("vault_url", AppConstants.VAULT_URL),
            "verify_ssl": kwargs.get("verify_ssl", True),
            "cert_path": kwargs.get("cert_path", AppConstants.CERT_PATH),
            "snowflake_warehouse": kwargs.get("warehouse", AppConstants.SNOWFLAKE_WAREHOUSE),
            "snowflake_schema": kwargs.get("schema", AppConstants.SNOWFLAKE_SCHEMA),
            "sf_database": kwargs.get("database", AppConstants.SNOWFLAKE_DATABASE),
            "sf_account": kwargs.get("account", AppConstants.SNOWFLAKE_ACCOUNT),
        }
        # Mark that we're using vault-based connection
        self.use_vault = True

    def _clear_ssl_env_vars(self):
        """Clear SSL environment variables that can interfere with Snowflake connections"""
        ssl_vars = ["CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"]
        for var in ssl_vars:
            if var in os.environ:
                # backup before deleting
                self.env_var[var] = os.environ[var]
                del os.environ[var]

    def _restore_ssl_env_vars(self):
        """Restore SSL environment variables after Snowflake connections"""
        for var, value in self.env_var.items():
            os.environ[var] = value

    def _create_session(self) -> Optional[Session]:
        """Create a new Snowflake session"""
        self._clear_ssl_env_vars()
        try:
            if hasattr(self, 'use_vault') and self.use_vault:
                # Use secure_snowflake_connector for vault-based auth
                logger.info("Creating Snowflake session using Vault credentials")
                self.session = get_snowflake_connection(**self.vault_params)
                
                # Explicitly set warehouse to ensure it's active in the session
                warehouse = self.vault_params.get('snowflake_warehouse')
                if warehouse:
                    try:
                        logger.info(f"Setting active warehouse to: {warehouse}")
                        self.session.sql(f"USE WAREHOUSE {warehouse}").collect()
                        logger.info(f"Warehouse successfully activated: {warehouse}")
                    except Exception as wh_error:
                        logger.warning(f"Could not set warehouse {warehouse}: {wh_error}. Using session default warehouse.")
                else:
                    logger.warning("No warehouse specified in vault parameters")
            else:
                # Use standard Snowpark connection
                self.session = Session.builder.configs(self.connection_params).create()
            
            self._restore_ssl_env_vars()
            logger.info("Snowflake session created successfully")
        except Exception as e:
            logger.error(f"Error creating Snowflake session: {e}")
            self.session = None
            raise
    
    def is_session_active(self) -> bool:
        """Check if the current session is active"""
        if not self.session:
            return False        
        try:
            #  Execute a simple query to check if the session is active
            self.session.sql("SELECT 1").collect()
            return True
        except SnowparkSQLException as e:
            if "Authentication token has expired" in str(e):
                logger.info(f"Authentication token has expired: {e}")
        except Exception as excc:
            logger.error(f"An unexpected error occurred while checking session: {excc}")
            raise
        return False
    
    def _is_auth_expired(self, exc: Exception) -> bool:
        """
        Detect Snowflake auth expiration (error code 390114) robustly.
        """
        msg = str(exc) if exc else ""
        return isinstance(exc, SnowparkSQLException) and ("390114" in msg or "Authentication token has expired" in msg)
    
    def ensure_session(self):
        """Ensure that there's an active session, creating one if needed"""
        if not self.is_session_active():
            self._create_session()
            if not self.session:
                raise RuntimeError("Failed to create Snowflake session")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame with query results
        """
        self.ensure_session()
        try:
            result = self.session.sql(query).collect()
            return result
        except SnowparkSQLException as e:
            if self._is_auth_expired(e):
                logger.info("Session expired while executing query. Refreshing session and retrying once.")
                self._create_session()
                result = self.session.sql(query).collect()
                return result
            logger.error(f"Error executing query: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def execute_query_and_return_pandas_df(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a Pandas DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            result = self.execute_query(query)
            result = pd.DataFrame(result)
            return result
        except Exception as e:
            logger.error(f"Error converting to pandas DataFrame: {e}")
            raise
      
    def save_pandas_df_to_snowflake(self, df: pd.DataFrame, table_name: str, overwrite: bool = True, quote_identifiers: bool = False):
        self.ensure_session()
        try:
            current_schema = self.session.get_current_schema()
            if current_schema == None:
                logger.info("Current schema is None, using `SNOWFLAKE_SCHEMA` environment variable")
                current_schema = AppConstants.SNOWFLAKE_SCHEMA
            if current_schema is None:
                logger.error("No Snowflake schema found, please set `SNOWFLAKE_SCHEMA` environment variable or pass when creating SnowparkHelper")
                raise ValueError("Current schema is None")
            current_schema = current_schema.replace('"', "")
        except Exception as e:
            logger.error(f"Failed to get current schema: {e}")
            raise 
        try:
            current_database = self.session.get_current_database()
            if current_database == None:
                logger.info("Current database is None, using `SNOWFLAKE_DATABASE` environment variable")
                current_database = AppConstants.SNOWFLAKE_DATABASE
            if current_database is None:
                logger.error("No Snowflake database found, please set `SNOWFLAKE_DATABASE` environment variable or pass when creating SnowparkHelper")
                raise ValueError("Current database is None")
            current_database = current_database.replace('"', "")
        except Exception as e:
            logger.error(f"Failed to get current database: {e}")
            raise
        # table_name = self.get_qualified_table_name(table_name)
        logger.info(f"Saving `{table_name}` with `{len(df)}` rows to Snowflake.")
        try:
            snowpark_df = self.session.write_pandas(
                df,
                table_name,
                database=current_database,
                schema=current_schema,
                auto_create_table=True,
                overwrite=overwrite,
                quote_identifiers=quote_identifiers
            )
            logger.info(f"Saved `{snowpark_df.table_name}` with `{snowpark_df.count()}` rows")
        except Exception as e:
            logger.error(f"Failed to save `{table_name}` with `{len(df)}` rows to Snowflake: {e}")
            raise
            
    def close(self):
        """Close all Snowflake sessions and clean up resources"""
        # Close main session
        if self.session:
            try:
                self.session.close()
                logger.info("Main Snowflake session closed")
            except Exception as e:
                logger.info(f"Error closing main Snowflake session: {e}")
            finally:
                self.session = None
        
        # Close pooled sessions
        with self._pool_lock:
            for session in self._session_pool:
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing pooled session: {e}")
            self._session_pool.clear()
        
        # Log performance summary if metrics are enabled
        if self.enable_metrics:
            self._log_performance_summary()
    
    def _initialize_connection_pool(self) -> None:
        """Initialize the connection pool with additional sessions."""
        with self._pool_lock:
            for _ in range(self.connection_pool_size - 1):  # -1 because we already have main session
                try:
                    if hasattr(self, 'use_vault') and self.use_vault:
                        session = get_snowflake_connection(**self.vault_params)
                    else:
                        session = Session.builder.configs(self.connection_params).create()
                    self._session_pool.append(session)
                except Exception as e:
                    logger.warning(f"Failed to create pooled session: {e}")
                    break
        logger.info(f"Initialized connection pool with {len(self._session_pool)} additional sessions")
    
    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a session from the pool or use the main session."""
        session_to_use = None
        acquired_from_pool = False
        
        # Try to get a session from the pool first
        with self._pool_lock:
            if self._session_pool:
                session_to_use = self._session_pool.pop()
                acquired_from_pool = True
        
        # Fall back to main session if pool is empty
        if session_to_use is None:
            self.ensure_session()
            session_to_use = self.session
        
        # Validate that the session is alive; replace if expired
        try:
            session_to_use.sql("SELECT 1").collect()
        except SnowparkSQLException as e:
            if self._is_auth_expired(e):
                logger.info("Pooled session token expired. Creating a fresh session.")
                try:
                    if hasattr(self, 'use_vault') and self.use_vault:
                        session_to_use = get_snowflake_connection(**self.vault_params)
                    else:
                        session_to_use = Session.builder.configs(self.connection_params).create()
                except Exception as ce:
                    logger.warning(f"Failed to create fresh session from pool; falling back to main session: {ce}")
                    self.ensure_session()
                    session_to_use = self.session
            else:
                raise

        try:
            yield session_to_use
        finally:
            # Return session to pool if it was acquired from pool
            if acquired_from_pool and session_to_use:
                with self._pool_lock:
                    self._session_pool.append(session_to_use)
    
    def _track_performance(self, operation_name: str, rows_processed: int = 0) -> PerformanceMetrics:
        """Create a performance metric tracker for an operation."""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            rows_processed=rows_processed
        )
        if self.enable_metrics:
            with self._metrics_lock:
                self.metrics.append(metric)
        return metric
    
    def _finish_performance_tracking(self, metric: PerformanceMetrics, success: bool = True, error_message: Optional[str] = None) -> None:
        """Finish tracking a performance metric."""
        metric.end_time = time.time()
        metric.success = success
        metric.error_message = error_message
        
        if self.enable_metrics:
            logger.info(f"Operation {metric.operation_name}: {metric.duration_seconds:.2f}s, "
                       f"{metric.rows_processed} rows ({metric.rows_per_second:.0f} rows/sec)")
    
    def _log_performance_summary(self) -> None:
        """Log a summary of all performance metrics."""
        if not self.metrics:
            return
        
        total_operations = len(self.metrics)
        successful_operations = sum(1 for m in self.metrics if m.success)
        total_duration = sum(m.duration_seconds for m in self.metrics)
        total_rows = sum(m.rows_processed for m in self.metrics)
        
        logger.info(f"Performance Summary: {successful_operations}/{total_operations} operations successful, "
                   f"{total_duration:.2f}s total, {total_rows} rows processed "
                   f"({total_rows / total_duration:.0f} rows/sec average)")
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get all performance metrics."""
        with self._metrics_lock:
            return self.metrics.copy()
    
    def batch_merge_optimized(self, logical_table_name: str, primary_keys: List[str], 
                             rows: List[Dict[str, Any]], overwrite: bool = False) -> None:
        """Optimized batch merge with configurable batch sizes and performance tracking.
        
        Args:
            logical_table_name: Logical name of the table (e.g., "DOCS", "CHUNKS")
            primary_keys: List of column names that form the primary key
            rows: List of dictionaries containing the data to insert/update
            overwrite: If True, delete existing data before inserting. If False, use MERGE.
        """
        if not rows:
            logger.info(f"No rows to merge for table {logical_table_name}")
            return
        
        # Get optimization settings, with fallbacks for legacy instances
        batch_size = getattr(self, 'batch_size', 1000)  # Default batch size
        max_workers = getattr(self, 'max_workers', 1)   # Default to sequential
        enable_metrics = getattr(self, 'enable_metrics', False)
        
        # Split into batches for optimal performance
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        
        # Track performance if metrics are enabled
        metric = None
        if enable_metrics and hasattr(self, '_track_performance'):
            metric = self._track_performance(f"batch_merge_{logical_table_name}", len(rows))
        
        try:
            # Process batches in parallel if there are multiple batches and parallel processing is enabled
            if len(batches) > 1 and max_workers > 1 and hasattr(self, '_batch_merge_parallel'):
                self._batch_merge_parallel(logical_table_name, primary_keys, batches, overwrite)
            else:
                # Process single batch or sequential processing
                for batch in batches:
                    if hasattr(self, '_batch_merge_single'):
                        self._batch_merge_single(logical_table_name, primary_keys, batch, overwrite)
                    else:
                        # Fallback to legacy batch_merge for compatibility
                        self._execute_legacy_batch_merge(logical_table_name, primary_keys, batch, overwrite)
            
            # Finish performance tracking if enabled
            if metric and hasattr(self, '_finish_performance_tracking'):
                self._finish_performance_tracking(metric, success=True)
            
            logger.info(f"Successfully processed {len(rows)} rows in {len(batches)} batches for {logical_table_name}")
            
        except Exception as e:
            # Finish performance tracking on error if enabled
            if metric and hasattr(self, '_finish_performance_tracking'):
                self._finish_performance_tracking(metric, success=False, error_message=str(e))
            logger.error(f"Failed to merge data into {logical_table_name}: {e}")
            raise
    
    def _batch_merge_parallel(self, logical_table_name: str, primary_keys: List[str], 
                             batches: List[List[Dict[str, Any]]], overwrite: bool) -> None:
        """Process batches in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._batch_merge_single, logical_table_name, primary_keys, batch, overwrite)
                for batch in batches
            ]
            
            # Wait for all batches to complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                    logger.debug(f"Completed batch {i+1}/{len(batches)} for {logical_table_name}")
                except Exception as e:
                    logger.error(f"Batch {i+1} failed for {logical_table_name}: {e}")
                    raise
    
    def _batch_merge_single(self, logical_table_name: str, primary_keys: List[str], 
                           rows: List[Dict[str, Any]], overwrite: bool) -> None:
        """Process a single batch of data."""
        with self.get_session() as session:
            try:
                self._execute_batch_merge(session, logical_table_name, primary_keys, rows, overwrite)
            except SnowparkSQLException as e:
                if self._is_auth_expired(e):
                    logger.info("Session expired during batch merge. Retrying once with a fresh session.")
                    with self.get_session() as retry_session:
                        self._execute_batch_merge(retry_session, logical_table_name, primary_keys, rows, overwrite)
                else:
                    raise
    
    def _execute_batch_merge(self, session: Session, logical_table_name: str, 
                            primary_keys: List[str], rows: List[Dict[str, Any]], 
                            overwrite: bool) -> None:
        """Execute the actual batch merge operation."""
        # Get the physical table name
        table_name = self.get_qualified_table_name(logical_table_name)
        
        logger.debug(f"{'Overwriting' if overwrite else 'Merging'} {len(rows)} rows into {table_name}")
        
        # Create a temporary view with the new data
        temp_view_name = f"TEMP_{logical_table_name}_{hash(str(rows)) % 10000}"
        
        # Convert rows to Snowpark DataFrame with optimized schema inference
        if rows:
            schema = self._infer_optimized_schema(rows[0], logical_table_name)
            df = session.create_dataframe(rows, schema=schema)
            df.create_or_replace_temp_view(temp_view_name)
            
            if overwrite:
                # Use more efficient DELETE+INSERT for overwrite operations
                self._execute_overwrite_operation(session, table_name, temp_view_name, rows[0])
            else:
                # Use optimized MERGE statement
                self._execute_merge_operation(session, table_name, temp_view_name, primary_keys, rows[0])
    
    def _infer_optimized_schema(self, sample_row: Dict[str, Any], logical_table_name: str) -> StructType:
        """Optimized schema inference with caching and known column types."""
        schema_fields = []
        for col_name, value in sample_row.items():
            # Use known VARIANT columns for better performance
            if col_name.upper() in ['CELL_BBOX', 'METADATA', 'PAYLOAD', 'PAGE_SPANS', 'CONTENT']:
                field_type = VariantType()
            elif value is None:
                field_type = StringType()
            elif isinstance(value, bool):
                field_type = StringType()  # Store booleans as strings
            elif isinstance(value, int):
                field_type = IntegerType()
            elif isinstance(value, float):
                field_type = FloatType()
            elif isinstance(value, (dict, list)):
                field_type = VariantType()
            else:
                field_type = StringType()
            schema_fields.append(StructField(col_name, field_type))
        
        return StructType(schema_fields)
    
    def _execute_overwrite_operation(self, session: Session, table_name: str, 
                                   temp_view_name: str, sample_row: Dict[str, Any]) -> None:
        """Execute optimized overwrite operation."""
        session.sql(f"DELETE FROM {table_name}").collect()
        columns = list(sample_row.keys())
        columns_str = ', '.join(columns)
        insert_sql = f"""
            INSERT INTO {table_name} ({columns_str})
            SELECT {columns_str} FROM {temp_view_name}
        """
        session.sql(insert_sql).collect()
    
    def _execute_merge_operation(self, session: Session, table_name: str, 
                               temp_view_name: str, primary_keys: List[str], 
                               sample_row: Dict[str, Any]) -> None:
        """Execute optimized MERGE operation."""
        pk_conditions = [f"target.{pk} = source.{pk}" for pk in primary_keys]
        
        update_assignments = []
        insert_columns = []
        insert_values = []
        
        for col_name in sample_row.keys():
            if col_name not in primary_keys:
                update_assignments.append(f"{col_name} = source.{col_name}")
            insert_columns.append(col_name)
            insert_values.append(f"source.{col_name}")
        
        merge_sql = f"""
            MERGE INTO {table_name} AS target
            USING {temp_view_name} AS source
            ON {' AND '.join(pk_conditions)}
            WHEN MATCHED THEN UPDATE SET {', '.join(update_assignments)}
            WHEN NOT MATCHED THEN INSERT ({', '.join(insert_columns)})
            VALUES ({', '.join(insert_values)})
        """
        
        session.sql(merge_sql).collect()
    
    def batch_merge(self, logical_table_name: str, primary_keys: List[str], 
                   rows: List[Dict[str, Any]], overwrite: bool = False) -> None:
        """Legacy batch merge method - redirects to optimized version.
        
        Args:
            logical_table_name: Logical name of the table (e.g., "DOCS", "CHUNKS")
            primary_keys: List of column names that form the primary key
            rows: List of dictionaries containing the data to insert/update
            overwrite: If True, delete existing data before inserting. If False, use MERGE.
        """
        # Check if we have optimized methods available
        if hasattr(self, 'batch_merge_optimized'):
            return self.batch_merge_optimized(logical_table_name, primary_keys, rows, overwrite)
        else:
            # Fallback to legacy implementation
            return self._execute_legacy_batch_merge(logical_table_name, primary_keys, rows, overwrite)
    
    def _execute_legacy_batch_merge(self, logical_table_name: str, primary_keys: List[str], 
                                   rows: List[Dict[str, Any]], overwrite: bool = False) -> None:
        """Legacy batch merge implementation for backward compatibility."""
        if not rows:
            logger.info(f"No rows to merge for table {logical_table_name}")
            return
            
        self.ensure_session()
        
        # Get the physical table name
        table_name = self.get_qualified_table_name(logical_table_name)
        
        logger.info(f"{'Overwriting' if overwrite else 'Merging'} {len(rows)} rows into {table_name}")
        
        try:
            # Create a temporary view with the new data
            temp_view_name = f"TEMP_{logical_table_name}_{hash(str(rows)) % 10000}"
            
            # Convert rows to Snowpark DataFrame
            if rows:
                # Infer schema from the data
                schema_fields = []
                sample_row = rows[0]
                for col_name, value in sample_row.items():
                    # For known VARIANT columns, always use VariantType regardless of sample value
                    if col_name.upper() in ['CELL_BBOX', 'METADATA', 'PAYLOAD', 'PAGE_SPANS', 'CONTENT']:
                        field_type = VariantType()
                    elif value is None:
                        field_type = StringType()
                    elif isinstance(value, bool):
                        field_type = StringType()  # Store booleans as strings
                    elif isinstance(value, int):
                        field_type = IntegerType()
                    elif isinstance(value, float):
                        field_type = FloatType()
                    elif isinstance(value, (dict, list)):
                        field_type = VariantType()
                    else:
                        field_type = StringType()
                    schema_fields.append(StructField(col_name, field_type))
                
                schema = StructType(schema_fields)
                df = self.session.create_dataframe(rows, schema=schema)
                df.create_or_replace_temp_view(temp_view_name)
                
                if overwrite:
                    # Delete existing data then insert new data
                    self.session.sql(f"DELETE FROM {table_name}").collect()
                    # Specify columns explicitly to handle auto-generated columns
                    columns = list(sample_row.keys())
                    columns_str = ', '.join(columns)
                    insert_sql = f"""
                        INSERT INTO {table_name} ({columns_str})
                        SELECT {columns_str} FROM {temp_view_name}
                    """
                    self.session.sql(insert_sql).collect()
                else:
                    # Use MERGE statement
                    # Build the MERGE statement
                    pk_conditions = []
                    for pk in primary_keys:
                        pk_conditions.append(f"target.{pk} = source.{pk}")
                    
                    update_assignments = []
                    insert_columns = []
                    insert_values = []
                    
                    for col_name in sample_row.keys():
                        if col_name not in primary_keys:
                            update_assignments.append(f"{col_name} = source.{col_name}")
                        insert_columns.append(col_name)
                        insert_values.append(f"source.{col_name}")
                    
                    merge_sql = f"""
                        MERGE INTO {table_name} AS target
                        USING {temp_view_name} AS source
                        ON {' AND '.join(pk_conditions)}
                        WHEN MATCHED THEN UPDATE SET {', '.join(update_assignments)}
                        WHEN NOT MATCHED THEN INSERT ({', '.join(insert_columns)})
                        VALUES ({', '.join(insert_values)})
                    """
                    
                    self.session.sql(merge_sql).collect()
                
                logger.info(f"Successfully {'overwrote' if overwrite else 'merged'} {len(rows)} rows into {table_name}")
                
        except Exception as e:
            logger.error(f"Failed to {'overwrite' if overwrite else 'merge'} data into {table_name}: {e}")
            raise
    
    def create_table_if_not_exists(self, logical_table_name: str, create_sql: str) -> None:
        """Create a table if it doesn't exist using the provided SQL.
        
        Args:
            logical_table_name: Logical name of the table
            create_sql: SQL statement to create the table
        """
        self.ensure_session()
        
        # Replace the table name placeholder in the SQL with the actual qualified name
        qualified_name = self.get_qualified_table_name(logical_table_name)
        final_sql = create_sql.replace("{{TABLE_NAME}}", qualified_name)
        
        logger.info(f"Ensuring table exists: {final_sql[:100]}...")
        
        try:
            self.session.sql(final_sql).collect()
            logger.info(f"Table {qualified_name} is ready")
        except SnowparkSQLException as e:
            if self._is_auth_expired(e):
                logger.info("Session expired while creating table. Refreshing session and retrying once.")
                self._create_session()
                self.session.sql(final_sql).collect()
                logger.info(f"Table {qualified_name} is ready")
            else:
                logger.error(f"Failed to create table {qualified_name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to create table {qualified_name}: {e}")
            raise
    
    def create_transient_table(
        self,
        table_name: str,
        select_query: str,
        cluster_by_column: Optional[str] = None
    ) -> None:
        """
        Create a TRANSIENT table with optional clustering.
        
        Args:
            table_name: Name of table to create (can be qualified DB.SCHEMA.TABLE)
            select_query: SELECT query for CTAS
            cluster_by_column: Optional column to cluster by
        """
        self.ensure_session()
        
        cluster_clause = f"CLUSTER BY ({cluster_by_column})" if cluster_by_column else ""
        
        create_sql = f"""
        CREATE OR REPLACE TRANSIENT TABLE {table_name}
        DATA_RETENTION_TIME_IN_DAYS = 0
        {cluster_clause}
        AS
        {select_query}
        """.strip()
        
        logger.info(f"Creating transient table: {table_name}")
        logger.debug(f"Create SQL: {create_sql[:200]}...")
        
        try:
            self.session.sql(create_sql).collect()
            logger.info(f"Created transient table: {table_name}")
        except SnowparkSQLException as e:
            if self._is_auth_expired(e):
                logger.info("Session expired while creating table. Refreshing and retrying.")
                self._create_session()
                self.session.sql(create_sql).collect()
            else:
                raise
    
    def drop_table_if_exists(self, table_name: str) -> bool:
        """
        Drop a table if it exists.
        
        Args:
            table_name: Name of table to drop (can be qualified)
        
        Returns:
            True if successful, False otherwise
        """
        self.ensure_session()
        
        drop_sql = f"DROP TABLE IF EXISTS {table_name}"
        
        try:
            self.session.sql(drop_sql).collect()
            logger.info(f"Dropped table: {table_name}")
            return True
        except SnowparkSQLException as e:
            if self._is_auth_expired(e):
                logger.info("Session expired while dropping table. Refreshing and retrying.")
                self._create_session()
                self.session.sql(drop_sql).collect()
                return True
            else:
                logger.error(f"Failed to drop table {table_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            return False
    
    def list_tables_by_pattern(
        self,
        pattern: str,
        database: Optional[str] = None,
        schema: Optional[str] = None
    ) -> pd.DataFrame:
        """
        List tables matching a pattern.
        
        Args:
            pattern: SQL LIKE pattern (e.g., 'TEMP_%')
            database: Optional database name (defaults to current)
            schema: Optional schema name (defaults to current)
        
        Returns:
            DataFrame with table information
        """
        self.ensure_session()
        
        # Use current database/schema if not specified
        if database is None:
            database = self.session.get_current_database()
            if database:
                database = database.replace('"', '')
        
        if schema is None:
            schema = self.session.get_current_schema()
            if schema:
                schema = schema.replace('"', '')
        
        # Build SHOW TABLES query
        if database and schema:
            location = f"IN {database}.{schema}"
        elif database:
            location = f"IN DATABASE {database}"
        else:
            location = ""
        
        show_sql = f"SHOW TABLES LIKE '{pattern}' {location}".strip()
        
        try:
            result_df = self.execute_query_and_return_pandas_df(show_sql)
            return result_df
        except Exception as e:
            logger.error(f"Failed to list tables with pattern '{pattern}': {e}")
            return pd.DataFrame()

# Example usage of the SnowflakeHelper class
if __name__ == "__main__":
    # Example 1: Using programmatic access with environment variables
    # Make sure your .env file contains the required variables or they are set in the environment
    logger.info("Example 1: Using programmatic access with environment variables")
    from dotenv import load_dotenv
    load_dotenv(".env")
    snowpark_programmatic_connection_parameters = {
        "account" : AppConstants.SNOWFLAKE_ACCOUNT,
        "user": AppConstants.SNOWFLAKE_USER,
        "password": AppConstants.SNOWFLAKE_SECRET,
        "warehouse": AppConstants.SNOWFLAKE_WAREHOUSE,
    }
    snowpark_programmatic = SnowparkHelper(connection_type="programmatic", **snowpark_programmatic_connection_parameters)
    print(snowpark_programmatic.execute_query("SELECT * FROM P01_EDL.EDL_ALLPHI.HLTH_SRVC limit 1"))
    snowpark_programmatic.close()
    # WIP got error while logging in with okta
    # logger.info("Example 2: Using Okta access")
    # snowpark_okta_connection_parameters = {
    #     "account" : "carelon-edaprod1.privatelink",
    #     "authenticator": 'https://portalsso.elevancehealth.com/snowflake/okta',
    #     "user": "AN285354AD",
    #     "password": "eKdvQc94I%pvTQ%UDY8k",
    #     "warehouse": "DL_AIFS_TMBR_USER_WH_L",
    #     "database": "NON_CRTFD_AIFS",
    #     "schema": "DL_DV_TMBR"
    # }
    # snowpark_okta = SnowparkHelper(connection_type="okta", **snowpark_okta_connection_parameters)
    # print(snowpark_okta.execute_query("SELECT * FROM P01_EDL.EDL_ALLPHI.HLTH_SRVC limit 1"))
    # snowpark_okta.close()
