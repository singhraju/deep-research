# Semantic View Sample Values Update - Implementation Summary

## Overview

Created a comprehensive solution for automatically updating `sample_values` in the ECAP semantic view YAML configuration by querying Snowflake for actual dimension values.

## What Was Created

### 1. Core Module: `semantic_view.py`
**Location**: `packages/utils/src/deep_research_utils/semantic_view.py`

**Functions**:
- `update_semantic_view_sample_values()`: Main function that reads YAML, queries Snowflake, and updates sample values
- `validate_semantic_view_config()`: Validates semantic view YAML structure
- `_get_dimension_sample_values()`: Helper function for querying dimension values

**Features**:
- ✅ Reads semantic view YAML configuration
- ✅ Connects to Snowflake using existing `SnowparkHelper`
- ✅ Queries each dimension to count unique values
- ✅ Updates `sample_values` for dimensions with ≤ 10 unique values (configurable)
- ✅ Saves updated YAML to a new file
- ✅ Comprehensive error handling and logging
- ✅ Full documentation with docstrings

### 2. Example Script
**Location**: `examples/update_semantic_view_samples.py`

A ready-to-run example demonstrating typical usage.

### 3. Documentation
**Location**: `docs/semantic_view_utilities.md`

Comprehensive documentation including:
- Detailed usage examples
- Parameter descriptions
- Performance considerations
- Integration workflows
- Error handling guide

### 4. Tests
**Location**: `tests/test_semantic_view.py`

Unit tests covering:
- Valid configuration validation
- Invalid YAML handling
- Dimension value querying
- File I/O operations
- Error scenarios

### 5. Package Updates
**Files Modified**:
- `packages/utils/pyproject.toml`: Added PyYAML dependency
- `packages/utils/src/deep_research_utils/__init__.py`: Exported new functions

## Quick Start

### Installation

First, activate your virtual environment and install dependencies:

```bash
source .venv/bin/activate
cd packages/utils
uv pip install -e .
```

### Basic Usage

```python
from deep_research_utils import update_semantic_view_sample_values

# Update sample values from Snowflake
update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/ecap_semantic_view_with_samples.yaml",
    connection_type="programmatic",
    max_unique_values=10
)
```

### Run the Example

```bash
# From project root
python examples/update_semantic_view_samples.py
```

### Prerequisites

Ensure Snowflake credentials are set as environment variables:

```bash
export snowflake_account="your-account"
export snowflake_user="your-username"
export snowflake_secret="your-password"
export snowflake_warehouse="your-warehouse"
```

Or pass them directly:

```python
update_semantic_view_sample_values(
    yaml_path="configs/ecap_semantic_view.yaml",
    output_path="configs/output.yaml",
    connection_type="programmatic",
    account="your-account",
    user="your-user",
    password="your-password",
    warehouse="your-warehouse"
)
```

## How It Works

1. **Parse YAML**: Reads the semantic view configuration file
2. **Connect**: Establishes Snowflake connection using `SnowparkHelper`
3. **Process Tables**: For each table in the configuration:
   - Identifies the base table (database.schema.table)
   - Iterates through all dimensions
4. **Query Dimensions**: For each dimension:
   - Executes `COUNT(DISTINCT column)` to get cardinality
   - If count ≤ `max_unique_values`, fetches actual values
   - Updates the `sample_values` field
5. **Save**: Writes enriched configuration to output file

## Example Query Pattern

For a dimension like `lob_code` with expression `LOB_CD`:

```sql
-- Step 1: Count unique values
SELECT COUNT(DISTINCT LOB_CD) AS unique_count
FROM NON_CRTFD_AIFS.DL_DV_TMBR.AH45807_CHAT_BOT_CLM_EXPNS
WHERE LOB_CD IS NOT NULL;

-- Step 2: If count ≤ 10, fetch values
SELECT DISTINCT LOB_CD AS value
FROM NON_CRTFD_AIFS.DL_DV_TMBR.AH45807_CHAT_BOT_CLM_EXPNS
WHERE LOB_CD IS NOT NULL
ORDER BY LOB_CD
LIMIT 10;
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaml_path` | Required | Input YAML file path |
| `output_path` | Required | Output YAML file path |
| `connection_type` | `"programmatic"` | Snowflake auth type |
| `max_unique_values` | `10` | Max values to include |

## Expected Output

Before:
```yaml
dimensions:
  - name: lob_code
    description: "LINE OF BUSINESS CODE..."
    expr: LOB_CD
    data_type: string
    synonyms: ["lob", "line_of_business"]
```

After:
```yaml
dimensions:
  - name: lob_code
    description: "LINE OF BUSINESS CODE..."
    expr: LOB_CD
    data_type: string
    synonyms: ["lob", "line_of_business"]
    sample_values: ["MCM003", "MCM004", "MCM006"]
```

## Testing

Run the test suite:

```bash
# From project root
pytest tests/test_semantic_view.py -v
```

## Validation

Validate your semantic view configuration:

```python
from deep_research_utils import validate_semantic_view_config

result = validate_semantic_view_config("configs/ecap_semantic_view.yaml")

if result['valid']:
    print(f"✅ Valid! {result['stats']['total_dimensions']} dimensions found")
else:
    print("❌ Errors:", result['errors'])
```

## Performance Notes

- **Large tables**: Count queries can be slow on very large tables
- **Many dimensions**: Processes sequentially to avoid overwhelming Snowflake
- **Configurable threshold**: Adjust `max_unique_values` to control processing time
- **Logging**: Enable DEBUG logging to monitor progress

## Integration Workflow

1. **Create/Update YAML**: Manually create or modify semantic view config
2. **Enrich**: Run `update_semantic_view_sample_values()`
3. **Validate**: Use `validate_semantic_view_config()` to verify
4. **Deploy**: Use enriched YAML in your LLM applications
5. **Maintain**: Re-run periodically as data evolves

## Files Created/Modified

### New Files
- `packages/utils/src/deep_research_utils/semantic_view.py` (core module)
- `examples/update_semantic_view_samples.py` (example script)
- `docs/semantic_view_utilities.md` (documentation)
- `tests/test_semantic_view.py` (test suite)
- `SEMANTIC_VIEW_UPDATE_SUMMARY.md` (this file)

### Modified Files
- `packages/utils/pyproject.toml` (added PyYAML dependency)
- `packages/utils/src/deep_research_utils/__init__.py` (exported new functions)

## Next Steps

1. **Test**: Run the example script to verify it works with your Snowflake connection
2. **Review**: Check the generated output YAML to ensure sample values are correct
3. **Integrate**: Use the enriched YAML in your semantic layer applications
4. **Customize**: Adjust `max_unique_values` based on your needs
5. **Automate**: Consider scheduling periodic updates to keep sample values current

## Troubleshooting

### Connection Issues
- Verify Snowflake credentials are correct
- Check network connectivity to Snowflake
- Ensure warehouse is running

### Query Errors
- Verify table/column names in YAML match Snowflake
- Check user has SELECT permissions on tables
- Review query logs for specific SQL errors

### YAML Issues
- Validate YAML syntax using online validators
- Ensure base_table definitions are complete
- Check that dimension `expr` fields match actual column names

## Support

For detailed documentation, see:
- `docs/semantic_view_utilities.md`: Complete usage guide
- `packages/utils/src/deep_research_utils/semantic_view.py`: Source code with docstrings
- `tests/test_semantic_view.py`: Example usage in tests
