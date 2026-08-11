"""
Subset Table Manager for Correlation Analysis

This module provides utilities for creating, managing, and cleaning up temporary
subset tables in Snowflake to optimize correlation query performance.

Key features:
- Create TRANSIENT subset tables with filtered data
- Automatic clustering on time dimension
- Context manager support for guaranteed cleanup
- Background cleanup for orphaned tables
"""

import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from deep_research_utils.app_constant import AppConstants
from deep_research_utils.logger_config import get_logger

logger = get_logger(__name__)


class SubsetTableManager:
    """
    Manages temporary subset tables for correlation analysis.
    
    Creates TRANSIENT tables with only required columns and pre-filtered data
    to improve query performance on large tables.
    """
    
    def __init__(self, snowflake_helper: Any, table_prefix: Optional[str] = None):
        """
        Initialize the subset table manager.
        
        Args:
            snowflake_helper: SnowparkHelper instance for executing queries
            table_prefix: Optional prefix for table names (defaults to AppConstants)
        """
        self.snowflake_helper = snowflake_helper
        self.table_prefix = table_prefix or AppConstants.CORRELATION_TEMP_TABLE_PREFIX
        self._validate_prefix()
        self._created_tables: List[str] = []
        self._cached_username: Optional[str] = None
    
    def _validate_prefix(self) -> None:
        """Validate table prefix follows Snowflake naming rules."""
        if len(self.table_prefix) > 50:
            raise ValueError(f"Table prefix too long: {self.table_prefix}")
        
        if not re.match(r'^[A-Z0-9_]*$', self.table_prefix):
            raise ValueError(
                f"Invalid table prefix (use uppercase, numbers, underscore only): {self.table_prefix}"
            )
        
        # Warn if user-specific
        user_patterns = ['AH', 'USER', r'DEV\d', r'USR\d']
        for pattern in user_patterns:
            if re.search(pattern, self.table_prefix):
                logger.warning(
                    f"Table prefix appears user-specific: {self.table_prefix}. "
                    "Consider using environment-based prefix for production."
                )
                break
    
    def _get_current_username(self) -> str:
        """
        Get the current Snowflake username.
        
        Returns:
            Current Snowflake username, or "UNKNOWN" if retrieval fails.
        """
        if self._cached_username is not None:
            return self._cached_username
        
        try:
            result = self.snowflake_helper.execute_query_and_return_pandas_df(
                "SELECT CURRENT_USER() as username"
            )
            if not result.empty:
                self._cached_username = str(result.iloc[0]['USERNAME'])
                logger.info(f"Retrieved Snowflake username: {self._cached_username}")
                return self._cached_username
        except Exception as e:
            logger.warning(f"Failed to retrieve current username: {e}")
        
        self._cached_username = "UNKNOWN"
        return self._cached_username
    
    def _sanitize_username(self, username: str, max_length: int = 20) -> str:
        """
        Sanitize username for use in table names.
        
        Converts to uppercase, replaces special characters with underscores,
        and truncates to max_length to comply with Snowflake naming rules.
        
        Args:
            username: Raw username from Snowflake
            max_length: Maximum length to truncate to (default: 20)
        
        Returns:
            Sanitized uppercase username suitable for table names
        
        Examples:
            >>> _sanitize_username("user@domain.com")
            'USER_DOMAIN_COM'
            >>> _sanitize_username("AH45807")
            'AH45807'
            >>> _sanitize_username("very-long-username-that-exceeds-limit")
            'VERY_LONG_USERNAME_'
        """
        # Convert to uppercase
        sanitized = username.upper()
        
        # Replace special characters with underscores
        sanitized = re.sub(r'[^A-Z0-9_]', '_', sanitized)
        
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Truncate to max_length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Ensure it doesn't end with underscore
        sanitized = sanitized.rstrip('_')
        
        return sanitized
    
    def generate_table_name(self) -> str:
        """
        Generate a unique table name for a subset table.
        
        In non-production environments, includes the username in the table name
        for better traceability and multi-user safety.
        
        Returns:
            Table name in format:
            - Production: {prefix}CORR_SUBSET_{timestamp}_{uuid}
            - Non-production: {prefix}{username}_CORR_SUBSET_{timestamp}_{uuid}
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        unique_id = uuid.uuid4().hex[:8]
        
        if AppConstants.CORRELATION_INCLUDE_USERNAME:
            username = self._get_current_username()
            sanitized_username = self._sanitize_username(username)
            logger.info(f"Creating subset table with username tracking: {sanitized_username}")
            return f"{self.table_prefix}{sanitized_username}_CORR_SUBSET_{timestamp}_{unique_id}"
        else:
            return f"{self.table_prefix}CORR_SUBSET_{timestamp}_{unique_id}"
    
    def create_subset_table(
        self,
        base_table_qualified_name: str,
        required_columns: Optional[List[str]],
        where_clauses: List[str],
        cluster_column: str,
    ) -> str:
        """
        Create a TRANSIENT subset table with filtered data and clustering.
        
        Args:
            base_table_qualified_name: Fully qualified base table (DB.SCHEMA.TABLE)
            required_columns: List of column names to include in subset (use None or empty list for SELECT *)
            where_clauses: List of WHERE clause conditions
            cluster_column: Column to cluster by (typically time dimension)
        
        Returns:
            Name of created subset table
        
        Raises:
            Exception: If table creation fails
        """
        subset_table_name = self.generate_table_name()
        
        # Build column list - use * if no columns specified
        if required_columns:
            columns_sql = ", ".join(required_columns)
        else:
            columns_sql = "*"
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get current database and schema from helper
        try:
            current_database = self.snowflake_helper.session.get_current_database()
            current_schema = self.snowflake_helper.session.get_current_schema()
            
            # Clean up quoted identifiers
            if current_database:
                current_database = current_database.replace('"', '')
            else:
                current_database = AppConstants.SNOWFLAKE_DATABASE
                
            if current_schema:
                current_schema = current_schema.replace('"', '')
            else:
                current_schema = AppConstants.SNOWFLAKE_SCHEMA
                
            qualified_subset_name = f"{current_database}.{current_schema}.{subset_table_name}"
        except Exception as e:
            logger.warning(f"Could not determine current database/schema: {e}")
            qualified_subset_name = subset_table_name
        
        # Build CTAS with TRANSIENT, no Time Travel retention, and CLUSTER BY.
        # DATA_RETENTION_TIME_IN_DAYS = 0 disables Time Travel on this short-lived table,
        # eliminating unnecessary storage cost since the table is explicitly dropped after use.
        comment_clause = ""
        if AppConstants.CORRELATION_INCLUDE_USERNAME:
            username = self._get_current_username()
            creation_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            comment_clause = f"\nCOMMENT = 'Created by {username} at {creation_timestamp} UTC via correlation_agent'"
        
        create_sql = f"""
CREATE OR REPLACE TRANSIENT TABLE {qualified_subset_name}
DATA_RETENTION_TIME_IN_DAYS = 0
CLUSTER BY ({cluster_column}){comment_clause}
AS
SELECT {columns_sql}
FROM {base_table_qualified_name}
WHERE {where_sql}
""".strip()
        
        start_time = time.time()
        logger.info(f"Creating subset table: {qualified_subset_name}")
        logger.info(f"Subset table SQL:\n{create_sql}")
        
        try:
            # Execute CTAS
            self.snowflake_helper.execute_query(create_sql)
            
            # Track created table
            self._created_tables.append(qualified_subset_name)
            
            # Get row count
            count_sql = f"SELECT COUNT(*) as cnt FROM {qualified_subset_name}"
            count_result = self.snowflake_helper.execute_query_and_return_pandas_df(count_sql)
            row_count = int(count_result.iloc[0]['CNT']) if not count_result.empty else 0
            
            elapsed = time.time() - start_time
            logger.info(
                f"✅ Created subset table {qualified_subset_name} with {row_count:,} rows "
                f"(clustered by {cluster_column}) in {elapsed:.2f}s"
            )
            
            return qualified_subset_name
            
        except Exception as e:
            logger.error(f"❌ Failed to create subset table {qualified_subset_name}: {e}")
            raise
    
    def cleanup_table(self, table_name: str, ignore_errors: bool = True) -> bool:
        """
        Drop a specific subset table.
        
        Args:
            table_name: Name of table to drop
            ignore_errors: If True, log errors but don't raise (default: True)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            drop_sql = f"DROP TABLE IF EXISTS {table_name}"
            logger.info(f"Dropping subset table: {table_name}")
            self.snowflake_helper.execute_query(drop_sql)
            
            # Remove from tracking
            if table_name in self._created_tables:
                self._created_tables.remove(table_name)
            
            logger.info(f"✅ Dropped subset table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to drop subset table {table_name}: {e}")
            if not ignore_errors:
                raise
            return False
    
    def cleanup_all_created_tables(self) -> None:
        """Drop all tables created by this manager instance."""
        for table_name in list(self._created_tables):
            self.cleanup_table(table_name, ignore_errors=True)
    
    @contextmanager
    def managed_subset_table(
        self,
        base_table_qualified_name: str,
        required_columns: List[str],
        where_clauses: List[str],
        cluster_column: str,
    ) -> Iterator[str]:
        """
        Context manager for automatic subset table cleanup.
        
        Usage:
            with manager.managed_subset_table(...) as subset_table:
                # Use subset_table for queries
                results = query(subset_table)
            # Table is automatically dropped here
        
        Args:
            base_table_qualified_name: Fully qualified base table
            required_columns: Columns to include
            where_clauses: Filter conditions
            cluster_column: Clustering column
        
        Yields:
            Name of created subset table
        """
        subset_table_name = None
        try:
            subset_table_name = self.create_subset_table(
                base_table_qualified_name,
                required_columns,
                where_clauses,
                cluster_column
            )
            yield subset_table_name
        finally:
            if subset_table_name:
                self.cleanup_table(subset_table_name, ignore_errors=True)
    
    def cleanup_old_tables(self, ttl_hours: Optional[int] = None) -> int:
        """
        Drop subset tables older than TTL (fallback cleanup job).
        
        Args:
            ttl_hours: Time to live in hours (defaults to AppConstants)
        
        Returns:
            Number of tables dropped
        """
        if not AppConstants.CORRELATION_SUBSET_CLEANUP_ENABLED:
            logger.info("Subset table cleanup is disabled")
            return 0
        
        ttl_hours = ttl_hours or AppConstants.CORRELATION_SUBSET_TTL_HOURS
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        
        logger.info(f"Starting cleanup of subset tables older than {ttl_hours} hours")
        
        try:
            # Get current database and schema
            current_database = self.snowflake_helper.session.get_current_database()
            current_schema = self.snowflake_helper.session.get_current_schema()
            
            if current_database:
                current_database = current_database.replace('"', '')
            else:
                current_database = AppConstants.SNOWFLAKE_DATABASE
                
            if current_schema:
                current_schema = current_schema.replace('"', '')
            else:
                current_schema = AppConstants.SNOWFLAKE_SCHEMA
            
            # Query to find matching tables - use pattern that matches both old and new formats
            # Old format: {prefix}CORR_SUBSET_%
            # New format: {prefix}{username}_CORR_SUBSET_%
            pattern = f"{self.table_prefix}%CORR_SUBSET_%"
            show_tables_sql = f"""
            SHOW TABLES LIKE '{pattern}' IN {current_database}.{current_schema}
            """
            
            tables_df = self.snowflake_helper.execute_query_and_return_pandas_df(show_tables_sql)
            
            if tables_df.empty:
                logger.info("No subset tables found for cleanup")
                return 0
            
            dropped_count = 0
            for _, row in tables_df.iterrows():
                table_name = row['name']
                created_on = row.get('created_on')
                
                # Parse timestamp from table name if created_on not available
                if created_on is None:
                    timestamp_match = re.search(r'(\d{8}T\d{6}Z)', table_name)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        try:
                            created_on = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                        except ValueError:
                            logger.warning(f"Could not parse timestamp from table name: {table_name}")
                            continue
                    else:
                        logger.warning(f"No timestamp found in table name: {table_name}")
                        continue
                
                # Check if older than TTL
                if isinstance(created_on, str):
                    created_on = datetime.fromisoformat(created_on.replace('Z', '+00:00'))
                
                if created_on < cutoff_time:
                    qualified_name = f"{current_database}.{current_schema}.{table_name}"
                    if self.cleanup_table(qualified_name, ignore_errors=True):
                        dropped_count += 1
            
            logger.info(f"Cleanup complete: dropped {dropped_count} old subset tables")
            return dropped_count
            
        except Exception as e:
            logger.error(f"Error during cleanup of old subset tables: {e}")
            return 0
