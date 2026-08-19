# deep-research-utils

Utility libraries for the deep-research project.

## Overview
This package contains utility functions and helper modules used across the deep-research project, including:

- **Logger Configuration**: Thread-safe logging with rotating file handlers
- **EHAP API Client**: HTTP client with automatic token management and refresh
- **Snowflake Helper**: Optimized Snowpark connection and query utilities

## Installation
This package is part of the deep-research monorepo and is managed using uv workspaces.

```bash
# Install all project dependencies (includes this package)
uv sync
```

## Usage

### Logger Configuration
```python
from deep_research_utils import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

### EHAP API Client
```python
from deep_research_utils import EHAP

# Uses environment variables: EHAP_BASE_URL, EHAP_CLIENT_ID, EHAP_CLIENT_SECRET
response = EHAP.sendHttpRequest(
    endpoint="/api/v1/endpoint",
    data={"key": "value"}
)
```

### Snowflake Helper
```python
from deep_research_utils import SnowparkHelper

snowflake = SnowparkHelper(connection_type="programmatic")
df = snowflake.execute_query_and_return_pandas_df("SELECT * FROM table LIMIT 10")
snowflake.close()
```

## Documentation
See [USAGE_EXAMPLES.md](../../USAGE_EXAMPLES.md) in the project root for detailed usage examples and configuration.

## Dependencies
- pandas
- python-dotenv
- requests
- snowflake-connector-python
- snowflake-snowpark-python
