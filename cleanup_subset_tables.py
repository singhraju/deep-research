#!/usr/bin/env python3
"""
Cleanup script for orphaned correlation subset tables in Snowflake.

Lists all transient subset tables created by the correlation agent
(matching the pattern <prefix>CORR_SUBSET_*) and optionally drops them.

In non-production environments, defaults to showing only the current user's tables.

Usage:
    # Dry-run — list current user's tables only (default in non-production)
    python cleanup_subset_tables.py --dry-run

    # List all users' tables
    python cleanup_subset_tables.py --all --dry-run

    # Drop current user's tables (with confirmation prompt)
    python cleanup_subset_tables.py

    # Drop all users' old tables (admin operation)
    python cleanup_subset_tables.py --all --older-than-hours 24

    # Skip the confirmation prompt (e.g. for scheduled jobs)
    python cleanup_subset_tables.py --yes
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root without installing packages
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent
sys.path.append(str(project_root / "packages" / "agents" / "src"))
sys.path.append(str(project_root / "packages" / "core" / "src"))
sys.path.append(str(project_root / "packages" / "utils" / "src"))

from deep_research_utils.app_constant import AppConstants
from deep_research_utils.logger_config import get_logger
from deep_research_utils.snowflake_helper import SnowparkHelper
import re

logger = get_logger(__name__)


def _get_snowflake_helper() -> SnowparkHelper:
    """Initialise and return a connected SnowparkHelper instance."""
    helper = SnowparkHelper()
    helper.ensure_session()
    return helper


def _get_current_username(helper: SnowparkHelper) -> str:
    """
    Get the current Snowflake username.
    
    Args:
        helper: Connected SnowparkHelper instance.
    
    Returns:
        Current Snowflake username, or "UNKNOWN" if retrieval fails.
    """
    try:
        result = helper.execute_query_and_return_pandas_df(
            "SELECT CURRENT_USER() as username"
        )
        if not result.empty:
            username = str(result.iloc[0]['USERNAME'])
            logger.info(f"Retrieved Snowflake username: {username}")
            return username
    except Exception as e:
        logger.warning(f"Failed to retrieve current username: {e}")
    
    return "UNKNOWN"


def _sanitize_username(username: str, max_length: int = 20) -> str:
    """
    Sanitize username for use in table name matching.
    
    Args:
        username: Raw username from Snowflake
        max_length: Maximum length to truncate to
        
    Returns:
        Sanitized uppercase username suitable for table name matching
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


def list_subset_tables(
    helper: SnowparkHelper, 
    prefix: str,
    username_filter: str | None = None
) -> list[dict]:
    """
    Return all CORR_SUBSET tables that match the given prefix.

    Args:
        helper: Connected SnowparkHelper instance.
        prefix: Table name prefix (e.g. 'DR_DEV_TEMP_').
        username_filter: Optional sanitized username to filter tables by.
                        If provided, only returns tables created by that user.

    Returns:
        List of dicts with keys: qualified_name, name, created_on, kind, rows.
    """
    try:
        current_database = helper.session.get_current_database()
        current_schema = helper.session.get_current_schema()

        if current_database:
            current_database = current_database.replace('"', '')
        else:
            current_database = AppConstants.SNOWFLAKE_DATABASE

        if current_schema:
            current_schema = current_schema.replace('"', '')
        else:
            current_schema = AppConstants.SNOWFLAKE_SCHEMA
    except Exception as exc:
        logger.error(f"Could not determine current database/schema: {exc}")
        raise

    # Use pattern that matches both old and new formats
    # Old format: {prefix}CORR_SUBSET_%
    # New format: {prefix}{username}_CORR_SUBSET_%
    pattern = f"{prefix}%CORR_SUBSET_%"
    show_sql = f"SHOW TABLES LIKE '{pattern}' IN {current_database}.{current_schema}"

    logger.info(f"Searching for tables matching: {pattern} in {current_database}.{current_schema}")
    if username_filter:
        logger.info(f"Filtering tables for user: {username_filter}")
    
    tables_df = helper.execute_query_and_return_pandas_df(show_sql)

    if tables_df.empty:
        return []

    results = []
    for _, row in tables_df.iterrows():
        name = row.get("name", "")
        created_on = row.get("created_on")
        kind = row.get("kind", "")
        rows = row.get("rows", "?")

        # Apply username filter if provided
        if username_filter:
            # Keep tables matching: {prefix}{username}_CORR_SUBSET_%
            # Also keep old-format tables: {prefix}CORR_SUBSET_% (no username)
            user_pattern = f"{prefix}{username_filter}_CORR_SUBSET_"
            old_pattern = f"{prefix}CORR_SUBSET_"
            if not (name.startswith(user_pattern) or name.startswith(old_pattern)):
                continue

        # Normalise created_on to a timezone-aware datetime
        if created_on is not None and isinstance(created_on, str):
            try:
                created_on = datetime.fromisoformat(created_on.replace("Z", "+00:00"))
            except ValueError:
                created_on = None

        results.append(
            {
                "qualified_name": f"{current_database}.{current_schema}.{name}",
                "name": name,
                "created_on": created_on,
                "kind": kind,
                "rows": rows,
            }
        )

    return results


def drop_table(helper: SnowparkHelper, qualified_name: str) -> bool:
    """
    Drop a single table by its fully-qualified name.

    Args:
        helper: Connected SnowparkHelper instance.
        qualified_name: Fully-qualified table name (DB.SCHEMA.TABLE).

    Returns:
        True if the drop succeeded, False otherwise.
    """
    try:
        helper.execute_query(f"DROP TABLE IF EXISTS {qualified_name}")
        logger.info(f"Dropped: {qualified_name}")
        return True
    except Exception as exc:
        logger.error(f"Failed to drop {qualified_name}: {exc}")
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and optionally drop correlation subset tables."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching tables without dropping anything.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="Only consider tables created more than HOURS hours ago.",
    )
    parser.add_argument(
        "--prefix",
        default=AppConstants.CORRELATION_TEMP_TABLE_PREFIX,
        help=(
            f"Table name prefix to match "
            f"(default: '{AppConstants.CORRELATION_TEMP_TABLE_PREFIX}'). "
            f"In non-production, tables include username in the name."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show tables from all users (default: current user only in non-production).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt and drop immediately.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    helper = _get_snowflake_helper()
    
    # Determine if username filtering should be active
    use_username_filter = AppConstants.CORRELATION_INCLUDE_USERNAME
    
    # Get current username if filtering is active and --all not specified
    username_filter = None
    if use_username_filter and not args.all:
        current_username = _get_current_username(helper)
        username_filter = _sanitize_username(current_username)
        logger.info(f"Filtering tables for user: {current_username} (sanitized: {username_filter})")
    elif args.all and use_username_filter:
        logger.info("Showing tables for all users (--all flag)")
    
    tables = list_subset_tables(helper, args.prefix, username_filter=username_filter)

    if not tables:
        print("No matching subset tables found.")
        return

    # Apply optional age filter
    if args.older_than_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=args.older_than_hours)
        tables = [
            t for t in tables
            if t["created_on"] is not None and t["created_on"] < cutoff
        ]
        if not tables:
            print(f"No tables older than {args.older_than_hours} hour(s) found.")
            return

    # Print table listing with appropriate message
    if username_filter:
        current_user = _get_current_username(helper)
        print(f"\nFound {len(tables)} subset table(s) for user {current_user}:\n")
    elif use_username_filter and args.all:
        print(f"\nFound {len(tables)} subset table(s) (all users):\n")
    else:
        print(f"\nFound {len(tables)} subset table(s):\n")
    header = f"{'Table Name':<70}  {'Created On':<26}  {'Kind':<12}  {'Rows':>10}"
    print(header)
    print("-" * len(header))
    for t in tables:
        created_str = t["created_on"].isoformat() if t["created_on"] else "unknown"
        print(
            f"{t['qualified_name']:<70}  {created_str:<26}  {t['kind']:<12}  {str(t['rows']):>10}"
        )
    print()

    if args.dry_run:
        print("Dry-run mode — no tables dropped.")
        return

    # Confirmation prompt
    if not args.yes:
        answer = input(f"Drop all {len(tables)} table(s) listed above? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # Drop tables
    dropped = 0
    failed = 0
    for t in tables:
        if drop_table(helper, t["qualified_name"]):
            dropped += 1
        else:
            failed += 1

    print(f"\nDone. Dropped: {dropped}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
