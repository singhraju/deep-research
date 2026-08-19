# Semantic View Utilities

This document describes the semantic view utilities for managing and enriching semantic view configurations with sample dimension values from Snowflake.

## Overview

The semantic view utilities provide functions to:
- Read semantic view YAML configurations
- Query Snowflake to discover actual dimension values
- Update configurations with sample values for enumerable dimensions
- Validate semantic view configurations

## Main Function: `update_semantic_view_sample_values`

### Purpose

This function enriches a semantic view configuration by querying Snowflake to discover actual dimension values. For dimensions with a manageable number of unique values (default: 10 or fewer), it updates the `sample_values` field in the YAML configuration.

### Benefits

1. **Auto-discovery**: Automatically discovers valid dimension values from the database
2. **Configuration enrichment**: Enhances semantic view configs with real sample data
3. **LLM assistance**: Sample values help LLMs better understand dimension cardinality and valid values
4. **Documentation**: Serves as inline documentation for valid dimension values

### Usage

#### Basic Usage

```python
from deep_research_utils import update_semantic_view_sample_values

update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/ecap_semantic_view_with_samples.yaml",
    connection_type="programmatic",
    max_unique_values=10
)
```

#### With Custom Snowflake Connection

```python
update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/ecap_semantic_view_updated.yaml",
    connection_type="programmatic",
    max_unique_values=15,
    database="MY_DATABASE",
    schema="MY_SCHEMA",
    warehouse="MY_WAREHOUSE"
)
```

#### Using Environment Variables

The function uses Snowflake credentials from environment variables when `connection_type="programmatic"`:

```bash
export snowflake_account="your-account"
export snowflake_user="your-username"
export snowflake_secret="your-password"
export snowflake_warehouse="your-warehouse"
```

Then call:

```python
update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/ecap_semantic_view_updated.yaml"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yaml_path` | str | Required | Path to the input semantic view YAML file |
| `output_path` | str | Required | Path where the updated YAML file will be saved |
| `connection_type` | str | `"programmatic"` | Type of Snowflake connection (`"programmatic"` or `"okta"`) |
| `max_unique_values` | int | `10` | Maximum number of unique values for updating sample_values |
| `**snowflake_kwargs` | dict | `{}` | Additional Snowflake connection parameters |

### How It Works

1. **Read Configuration**: Parses the input YAML semantic view configuration
2. **Connect to Snowflake**: Establishes a connection using the specified credentials
3. **Process Each Table**: Iterates through all tables defined in the configuration
4. **Check Dimensions**: For each dimension in each table:
   - Queries the database to count unique values
   - If count ≤ `max_unique_values`, fetches the actual values
   - Updates the dimension's `sample_values` field
5. **Save Updated Config**: Writes the enriched configuration to the output file

### Example Workflow

```python
# Step 1: Update sample values
from deep_research_utils import update_semantic_view_sample_values

update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/ecap_semantic_view_enriched.yaml",
    max_unique_values=10
)

# Step 2: Validate the updated configuration
from deep_research_utils import validate_semantic_view_config

result = validate_semantic_view_config("configs/ecap_semantic_view_enriched.yaml")

if result['valid']:
    print(f"✅ Configuration is valid!")
    print(f"   Total dimensions: {result['stats']['total_dimensions']}")
    print(f"   Dimensions with samples: {result['stats']['dimensions_with_sample_values']}")
else:
    print(f"❌ Configuration has errors:")
    for error in result['errors']:
        print(f"   - {error}")
```

## Validation Function: `validate_semantic_view_config`

### Purpose

Validates the structure and completeness of a semantic view YAML configuration.

### Usage

```python
from deep_research_utils import validate_semantic_view_config

result = validate_semantic_view_config("configs/ecap_semantic_view.yaml")

if result['valid']:
    print("Configuration is valid")
    print(f"Statistics: {result['stats']}")
else:
    print("Errors found:")
    for error in result['errors']:
        print(f"  - {error}")
    
    print("Warnings:")
    for warning in result['warnings']:
        print(f"  - {warning}")
```

### Return Value

Returns a dictionary with:

```python
{
    'valid': bool,  # True if no errors found
    'errors': [str],  # List of error messages
    'warnings': [str],  # List of warning messages
    'stats': {
        'total_tables': int,
        'total_dimensions': int,
        'dimensions_with_sample_values': int,
        'total_time_dimensions': int,
        'total_facts': int,
        'total_metrics': int
    }
}
```

## Command-Line Usage

You can run the example script from the command line:

```bash
# From the project root
python examples/update_semantic_view_samples.py
```

Or create a custom script:

```python
#!/usr/bin/env python3
"""Update semantic view sample values."""

from deep_research_utils import update_semantic_view_sample_values
import sys

if __name__ == "__main__":
    try:
        update_semantic_view_sample_values(
            yaml_path="configs/ecap_semantic_view.yaml",
            output_path="configs/ecap_semantic_view_updated.yaml",
            max_unique_values=10
        )
        print("✅ Successfully updated semantic view!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
```

## Performance Considerations

### Large Tables

For very large tables, the count and distinct queries can be slow. The function:
- Uses `COUNT(DISTINCT column)` to efficiently determine cardinality
- Only fetches actual values if count ≤ `max_unique_values`
- Processes dimensions sequentially to avoid overwhelming Snowflake

### Optimization Tips

1. **Adjust max_unique_values**: Lower values process faster
2. **Filter tables**: Modify the YAML to only include tables you need
3. **Use specific schemas**: Ensure base_table definitions are accurate
4. **Monitor logs**: The function logs detailed progress information

## Error Handling

The function handles various error scenarios:

- **Missing YAML file**: Raises `FileNotFoundError`
- **Invalid YAML structure**: Raises `ValueError`
- **Snowflake connection errors**: Propagates Snowflake exceptions
- **Query errors**: Logs error and skips problematic dimensions
- **Missing columns**: Logs warning and continues with other dimensions

## Logging

The function uses the project's logging configuration. To see detailed progress:

```python
from deep_research_utils import get_logger, LogLevel
import logging

# Set to DEBUG for detailed query logs
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# Now run the update
update_semantic_view_sample_values(...)
```

## Integration with Semantic View Workflow

This utility integrates into the semantic view management workflow:

1. **Initial Creation**: Create semantic view YAML manually or from schema
2. **Enrichment**: Use `update_semantic_view_sample_values()` to add sample data
3. **Validation**: Use `validate_semantic_view_config()` to verify structure
4. **Deployment**: Use the enriched YAML in your LLM applications
5. **Maintenance**: Re-run periodically as dimension values evolve

## Dependencies

- `pyyaml>=6.0`: For YAML parsing and generation
- `pandas>=3.0.2`: For DataFrame operations
- `snowflake-snowpark-python>=1.48.1`: For Snowflake connectivity

These dependencies are automatically installed with the `deep-research-utils` package.

## Related Documentation

- [ECAP Semantic View](../configs/ecap_semantic_view.yaml): Example semantic view configuration
- [Snowflake Helper](../packages/utils/src/deep_research_utils/snowflake_helper.py): Snowflake connection utilities
- [Package README](../packages/utils/README.md): Utils package overview
