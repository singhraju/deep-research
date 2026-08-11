# Deep Research ETL Pipeline

## Overview
The Deep Research (DR) ETL Pipeline is an automated data processing system that analyzes healthcare authorization data to identify business patterns, reimbursement policies, and actionable recommendations. The pipeline integrates multiple AI agents to process anomaly insights and generate comprehensive reports for healthcare decision-makers.

## Pipeline Architecture

### Data Flow
```
Snowflake Data → Correlation Analysis → Pattern Detection → Reimbursement Analysis → Recommendations → Policy Analysis → Snowflake Storage
```

### Key Components

#### 1. **Data Ingestion**
- **Source**: Snowflake database (`COC_CMN_DEEP_RSRCH_INSGHT` table)
- **Data Types**: 
  - `KEY_INSIGHT`: Anomaly detection results with top contributors (states, providers, DRGs)
  - `DEEP_DIVE`: Detailed analysis of authorization trends and patterns
- **Filters**: Processes data by combinations of:
  - `SNAP_YEAR_MNTH_NBR`: Snapshot month
  - `TRND_TM_PRD_END_MNTH_NBR`: Trend period end month
  - `TRND_TM_PRD_CD`: Trend period code (e.g., R3, R6, R12)
  - `LOB_SHRT_DESC`: Line of Business (e.g., Commercial, Medicaid)
  - `STATSCL_MDL_CD`: Statistical model code (e.g., IP AUTH, OP AUTH)

#### 2. **Agent Processing Pipeline**

##### **Correlation Agent**
- **Purpose**: Analyzes relationships between anomalies across states, providers, and DRGs
- **Input**: Anomaly JSON with top contributors
- **Output**: Correlation results mapping relationships between entities
- **Health Check**: Validates API availability before processing

##### **Pattern Agent**
- **Purpose**: Identifies business patterns from correlation results
- **Input**: Anomaly data, deep dive analysis, correlation results
- **Output**: 
  - Business patterns ranked by significance
  - Supporting cards with detailed metrics
  - Pattern metadata (conversation_id, pattern_rank)
- **Special Handling**: 
  - If 0 patterns found, downstream reimbursement and recommendation agents are skipped
  - Empty pattern results are stored to track processed combinations

##### **Reimbursement Agent** (Pattern-Dependent)
- **Purpose**: Analyzes reimbursement policies for each identified pattern
- **Input**: Individual business pattern + supporting cards
- **Output**: 
  - Summary table of reimbursement policies
  - Policy details and recommendations
  - Processing statistics (successful/failed policies)
- **Processing**: Runs once per business pattern
- **Skipped When**: No business patterns are found

##### **Recommendation Agent** (Pattern-Dependent)
- **Purpose**: Synthesizes actionable recommendations from patterns and reimbursement data
- **Input**: All business patterns + reimbursement results
- **Output**: 
  - Prioritized recommendations
  - Skipped patterns (if any)
  - Processing logs
- **Default Structure** (0 patterns):
  ```json
  {
    "success": true,
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
  ```

##### **Policy Agent** (Independent)
- **Purpose**: Retrieves relevant payer policies (e.g., critical care denial policies)
- **Input**: None (independent of patterns)
- **Output**: Policy summary tables with payer-specific rules
- **Processing**: Always runs regardless of pattern count

#### 3. **Output Generation**

##### **JSON Files** (Per Combination)
1. **`combination_{params}.json`**
   - Complete pipeline result
   - Contains all agent outputs
   - Includes metadata (snap month, LOB, model code, etc.)

2. **`snowflake_{params}.json`**
   - Exact DataFrame that will be stored to Snowflake
   - Includes all audit columns
   - Multiple records per combination (Pattern, Recommendation, Policy)

##### **Snowflake Storage**
- **Table**: `{SCHEMA}_COC.{TARGET_SCHEMA}.COC_CMN_DEEP_RSRCH_INSGHT`
- **Record Types** (per combination):
  - `INSGHT_TYPE_NM = 'Pattern'`: Business patterns (list of dicts or empty list)
  - `INSGHT_TYPE_NM = 'Recommendation'`: Recommendations (dict with default structure if 0 patterns)
  - `INSGHT_TYPE_NM = 'Policy'`: Policy results (dict)
- **Audit Columns**:
  - `EDL_LOAD_DTM`: Load timestamp
  - `EDL_CREAT_DTM`: Creation timestamp
  - `EDL_INCRMNTL_LOAD_DTM`: Incremental load timestamp
  - `KF_TMS`: Key field timestamp
  - `EDL_RUN_ID`, `EDL_SOR_CD`, `EDL_SCRTY_LVL_CD`, `EDL_EXTRNL_LOAD_CD`: Metadata fields

## Configuration

### Environment Setup
The pipeline supports multiple environments (dev, qa, prod) configured via:
- `utils/config.py`: API endpoints, environment settings, pipeline configuration
- AWS Secrets Manager: Snowflake credentials

### Required Secrets
```json
{
  "user": "snowflake_username",
  "password": "snowflake_password",
  "account": "snowflake_account",
  "warehouse": "warehouse_name",
  "database": "database_name",
  "schema": "schema_name"
}
```

## Execution

### Command Line
```bash
python dr_etl_pipeline.py --env dev --lob nogbd --use_snowflake
```

### Parameters
- `--env`: Environment (dev/qa/prod)
- `--lob`: Line of Business filter
- `--use_snowflake`: Use Snowflake data source (default: True)
- `--csv_file_path`: CSV fallback path (if not using Snowflake)

## Error Handling

### Health Checks
- **Purpose**: Validate API availability before each agent call
- **Configuration**: 
  - Timeout: 5 seconds
  - Retry interval: 2 seconds
  - Infinite retries until success
- **Failure Handling**: Uses default empty structures if health check fails

### Logging
- **Log Level**: INFO (suppresses DEBUG from external libraries)
- **Log Location**: `logs/dr_etl_pipeline_{timestamp}.log`
- **Custom Logs**: Prefixed with `[LOG]` for pipeline-specific events

### Zero Pattern Handling
When pattern agent returns 0 business patterns:
1. ✅ Pattern result stored as empty list `[]`
2. ✅ Reimbursement agent skipped (pattern-dependent)
3. ✅ Recommendation agent skipped, default structure stored
4. ✅ Policy agent still runs (independent)
5. ✅ All results stored to Snowflake with proper structure

## Dependencies

### Python Packages
- `snowflake-snowpark-python`: Snowflake data processing
- `pandas`: Data manipulation
- `boto3`: AWS Secrets Manager integration
- `requests`: API communication
- `urllib3`: HTTP client

### Utility Modules
- `utils/time_utils.py`: Time period calculations
- `utils/config.py`: Configuration management
- `utils/agent_utilss.py`: Agent communication functions
- `utils/snowflake_utils.py`: Snowflake data operations

## Output Structure

### Combination Result
```json
{
  "snap_year_mnth_nbr": 202604,
  "trnd_tm_prd_end_mnth_nbr": 202601,
  "trnd_tm_prd_cd": "R3",
  "lob_shrt_desc": "Commercial",
  "statscl_mdl_cd": "IP AUTH",
  "final_payload": {...},
  "anomaly_json": {...},
  "deep_dive_json": {...},
  "pattern_result": [...],
  "recommendation_result": {...},
  "policy_result": {...},
  "status": "success",
  "processed_at": "2026-06-10T10:07:59.784732"
}
```

### Snowflake Record
```json
{
  "EDL_LOAD_DTM": "2026-06-10 10:07:59",
  "EDL_RUN_ID": "NA",
  "EDL_SOR_CD": "NA",
  "KF_TMS": "2026-06-10 10:07:59",
  "EDL_SCRTY_LVL_CD": "NA",
  "EDL_LOB_CD": null,
  "EDL_EXTRNL_LOAD_CD": "NA",
  "EDL_CREAT_DTM": "2026-06-10 10:07:59",
  "EDL_INCRMNTL_LOAD_DTM": "2026-06-10 10:07:59",
  "SNAP_YEAR_MNTH_NBR": 202604,
  "TRND_TM_PRD_END_MNTH_NBR": 202601,
  "TRND_TM_PRD_CD": "R3",
  "LOB_CD": null,
  "LOB_SHRT_DESC": "Commercial",
  "STATSCL_MDL_CD": "IP AUTH",
  "INSGHT_TYPE_NM": "Pattern",
  "JSON_TXT": "[...]"
}
```

## Monitoring

### Key Metrics
- Combinations processed
- Patterns identified per combination
- Agent success/failure rates
- Snowflake storage success rate

### Log Messages
- `[LOG] Processing combination: ...` - Combination start
- `[LOG] API health check passed for {agent}` - Health check success
- `[LOG] Calling {agent}...` - Agent invocation
- `[LOG] {agent} completed` - Agent completion
- `[LOG] No business patterns found` - Zero pattern scenario
- `Saved Snowflake DataFrame to: ...` - JSON file creation
- `Successfully prepared N records for {table}` - Snowflake storage success

## Troubleshooting

### Common Issues

**Issue**: No data found in Snowflake
- **Solution**: Verify `STATSCL_MDL_CD` and `LOB` filters match available data

**Issue**: API health check failures
- **Solution**: Check API endpoint configuration and network connectivity

**Issue**: Zero patterns found
- **Solution**: This is expected behavior; pipeline will store empty structures and skip dependent agents

**Issue**: Snowflake storage errors
- **Solution**: Verify credentials in Secrets Manager and table permissions

## Maintenance

### Adding New Agents
1. Create agent function in `utils/agent_utilss.py`
2. Add health check before agent call
3. Handle agent output in result structure
4. Add storage logic in `store_agent_results_to_snowflake()`

### Modifying Output Structure
1. Update result record creation in main loop
2. Update Snowflake storage function
3. Update target column list if adding new fields
4. Test with zero-pattern scenarios

## Contact
For issues or questions, contact the Deep Research team.
