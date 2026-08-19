"""
Semantic View Utilities

This module provides utilities for managing and updating semantic view configurations,
including enriching dimension metadata with sample values from Snowflake.
"""

import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from deep_research_utils.snowflake_helper import SnowparkHelper
from deep_research_utils.logger_config import get_logger

logger = get_logger(__name__)


def update_semantic_view_sample_values(
    yaml_path: str,
    output_path: str,
    connection_type: str = None,
    max_unique_values: int = 10,
    **snowflake_kwargs
) -> None:
    """
    Read a semantic view YAML configuration, connect to Snowflake, and update
    sample_values for dimensions that have no more than max_unique_values unique values.
    
    This function enriches the semantic view configuration by querying Snowflake
    to discover actual dimension values. For dimensions with a manageable number
    of unique values (default: 10 or fewer), it updates the sample_values field
    to reflect the actual values in the database.
    
    Args:
        yaml_path: Path to the input semantic view YAML file
        output_path: Path where the updated YAML file will be saved
        connection_type: Type of Snowflake connection. If None, auto-detects
                        ("vault" in production, "programmatic" locally)
        max_unique_values: Maximum number of unique values to consider for sample_values.
                          Dimensions with more than this many unique values will be skipped.
        **snowflake_kwargs: Additional keyword arguments to pass to SnowparkHelper
                           (e.g., account, user, password, warehouse, database, schema)
    
    Returns:
        None. The updated YAML is written to output_path.
    
    Raises:
        FileNotFoundError: If yaml_path does not exist
        ValueError: If YAML structure is invalid
        Exception: For Snowflake connection or query errors
    
    Example:
        >>> update_semantic_view_sample_values(
        ...     yaml_path="configs/ecap_semantic_view.yaml",
        ...     output_path="configs/ecap_semantic_view_updated.yaml",
        ...     connection_type="programmatic",
        ...     max_unique_values=10
        ... )
    """
    logger.info(f"Reading semantic view configuration from {yaml_path}")
    
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'tables' not in config:
        raise ValueError("Invalid YAML structure: 'tables' key not found")
    
    logger.info("Connecting to Snowflake...")
    snowflake_helper = SnowparkHelper(connection_type=connection_type, **snowflake_kwargs)
    
    try:
        total_dimensions_updated = 0
        total_dimensions_processed = 0
        
        for table in config['tables']:
            table_name = table.get('name')
            base_table = table.get('base_table')
            
            if not base_table:
                logger.warning(f"Table {table_name} has no base_table defined, skipping")
                continue
            
            database = base_table.get('database')
            schema = base_table.get('schema')
            physical_table = base_table.get('table')
            
            if not all([database, schema, physical_table]):
                logger.warning(f"Table {table_name} has incomplete base_table definition, skipping")
                continue
            
            qualified_table = f"{database}.{schema}.{physical_table}"
            logger.info(f"Processing table: {table_name} ({qualified_table})")
            
            dimensions = table.get('dimensions', [])
            
            for dimension in dimensions:
                total_dimensions_processed += 1
                dim_name = dimension.get('name')
                dim_expr = dimension.get('expr')
                
                if not dim_expr:
                    logger.warning(f"Dimension {dim_name} has no expression, skipping")
                    continue
                
                logger.info(f"  Checking dimension: {dim_name} (column: {dim_expr})")
                
                sample_values = _get_dimension_sample_values(
                    snowflake_helper=snowflake_helper,
                    qualified_table=qualified_table,
                    column_expr=dim_expr,
                    max_unique_values=max_unique_values,
                    dimension_name=dim_name
                )
                
                if sample_values is not None:
                    dimension['sample_values'] = sample_values
                    total_dimensions_updated += 1
                    logger.info(f"    Updated sample_values with {len(sample_values)} values")
                else:
                    logger.info(f"    Skipped (too many unique values or error)")
        
        logger.info(f"Updated {total_dimensions_updated} out of {total_dimensions_processed} dimensions")
        
        logger.info(f"Writing updated configuration to {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"Successfully saved updated semantic view to {output_path}")
        
    finally:
        snowflake_helper.close()


def _get_dimension_sample_values(
    snowflake_helper: SnowparkHelper,
    qualified_table: str,
    column_expr: str,
    max_unique_values: int,
    dimension_name: str
) -> Optional[List[str]]:
    """
    Query Snowflake to get unique values for a dimension column.
    
    Args:
        snowflake_helper: SnowparkHelper instance for querying Snowflake
        qualified_table: Fully qualified table name (database.schema.table)
        column_expr: Column expression for the dimension
        max_unique_values: Maximum number of unique values to return
        dimension_name: Name of the dimension (for logging)
    
    Returns:
        List of unique values if count <= max_unique_values, None otherwise
    """
    try:
        count_query = f"""
        SELECT COUNT(DISTINCT {column_expr}) AS unique_count
        FROM {qualified_table}
        WHERE {column_expr} IS NOT NULL
        """
        
        logger.debug(f"Executing count query for {dimension_name}: {count_query}")
        count_result = snowflake_helper.execute_query_and_return_pandas_df(count_query)
        
        if count_result.empty:
            logger.warning(f"No results from count query for {dimension_name}")
            return None
        
        unique_count = int(count_result.iloc[0]['UNIQUE_COUNT'])
        logger.debug(f"Found {unique_count} unique values for {dimension_name}")
        
        if unique_count > max_unique_values:
            logger.debug(f"Skipping {dimension_name}: {unique_count} > {max_unique_values}")
            return None
        
        if unique_count == 0:
            logger.debug(f"Skipping {dimension_name}: no non-null values found")
            return None
        
        values_query = f"""
        SELECT DISTINCT {column_expr} AS value
        FROM {qualified_table}
        WHERE {column_expr} IS NOT NULL
        ORDER BY {column_expr}
        LIMIT {max_unique_values}
        """
        
        logger.debug(f"Executing values query for {dimension_name}: {values_query}")
        values_result = snowflake_helper.execute_query_and_return_pandas_df(values_query)
        
        if values_result.empty:
            return None
        
        sample_values = values_result['VALUE'].astype(str).tolist()
        
        return sample_values
        
    except Exception as e:
        logger.error(f"Error querying sample values for {dimension_name}: {e}")
        return None


def validate_semantic_view_config(yaml_path: str) -> Dict[str, Any]:
    """
    Validate a semantic view YAML configuration file.
    
    Args:
        yaml_path: Path to the semantic view YAML file
    
    Returns:
        Dictionary containing validation results with keys:
        - valid: bool indicating if configuration is valid
        - errors: list of error messages
        - warnings: list of warning messages
        - stats: dictionary with statistics about the configuration
    
    Example:
        >>> result = validate_semantic_view_config("configs/ecap_semantic_view.yaml")
        >>> if result['valid']:
        ...     print(f"Config is valid with {result['stats']['total_dimensions']} dimensions")
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_tables': 0,
            'total_dimensions': 0,
            'dimensions_with_sample_values': 0,
            'total_time_dimensions': 0,
            'total_facts': 0,
            'total_metrics': 0
        }
    }
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to parse YAML: {e}")
        return result
    
    if not isinstance(config, dict):
        result['valid'] = False
        result['errors'].append("YAML root must be a dictionary")
        return result
    
    if 'tables' not in config:
        result['valid'] = False
        result['errors'].append("Missing required 'tables' key")
        return result
    
    tables = config['tables']
    if not isinstance(tables, list):
        result['valid'] = False
        result['errors'].append("'tables' must be a list")
        return result
    
    result['stats']['total_tables'] = len(tables)
    
    for i, table in enumerate(tables):
        if not isinstance(table, dict):
            result['errors'].append(f"Table at index {i} is not a dictionary")
            continue
        
        table_name = table.get('name', f'table_{i}')
        
        if 'base_table' not in table:
            result['warnings'].append(f"Table '{table_name}' missing 'base_table' definition")
        
        dimensions = table.get('dimensions', [])
        result['stats']['total_dimensions'] += len(dimensions)
        
        for dim in dimensions:
            if 'sample_values' in dim:
                result['stats']['dimensions_with_sample_values'] += 1
            if not dim.get('expr'):
                result['warnings'].append(f"Dimension '{dim.get('name')}' in table '{table_name}' missing 'expr'")
        
        result['stats']['total_time_dimensions'] += len(table.get('time_dimensions', []))
        result['stats']['total_facts'] += len(table.get('facts', []))
        result['stats']['total_metrics'] += len(table.get('metrics', []))
    
    if result['errors']:
        result['valid'] = False
    
    return result
