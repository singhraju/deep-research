# OP Oth BH Implementation Documentation

**Statistical Model**: `OP OTH BH` (Outpatient Other Behavioral Health)  
**Implementation Date**: July 2026  
**Status**: Production Ready

---

## Executive Summary

This document details the implementation of OP Oth BH support in the Deep Research platform. OP Oth BH is the third statistical model added to the platform (after IP AUTH and OON), enabling analysis of outpatient behavioral health claims.

**Key Decision**: DRG dimensions are **excluded** for OP Oth BH (not applicable for outpatient care).

---

## Changes Made

### 1. Semantic YAML Configuration (5 files created)

**Location**: `configs/correlation_pattern/`

Created environment-specific YAML files:
- `coc_ecap_op_oth_bh_sematic_view_with_samples_dev.yaml`
- `coc_ecap_op_oth_bh_sematic_view_with_samples_uat.yaml`
- `coc_ecap_op_oth_bh_sematic_view_with_samples_prod.yaml`
- `coc_ecap_op_oth_bh_sematic_view_with_samples_local.yaml`
- `coc_ecap_op_oth_bh_sematic_view_with_samples_local_offshore.yaml`

**Production Config Key Settings** (`prod.yaml`, 714 lines):

```yaml
name: coc_ecap_op_oth_bh_view
description: COC ECAP OP OTH BH semantic view for claims expense and membership analysis
comments: |
  - DRG dimensions disabled for OP OTH BH analysis  # Line 12

base_table:
  database: P01_COC
  schema: COC_DTI_STG
  table: WORK_ELEVATE_COC_CLAIM_DETAIL

# Critical dimension configuration
dimensions:
  - name: hcc_medium
    expr: HCC_CMRCL_MNGMNT_MEDM_SHRT_DESC
    sample_values:
      - OP OTH BH  # Line 122 - Exact value from data
  
  - name: hcc_low
    expr: HCC_CMRCL_MNGMNT_LOW_SHRT_DESC
    sample_values:
      - OP OTH BH  # Line 132

# Drill dimensions (NO DRG included)
drill_dimensions:
  - service_area_state
  - hcc_medium
  - rendering_provider_name
  - facility_type
  - product_description
  - primary_diagnosis_name
  - pa_required_code
  - er_admit_indicator
```

**What's Different from IP AUTH**:
- No `drg_name` or `drg_code` dimensions
- `hcc_medium` and `hcc_low` set to "OP OTH BH"
- Focus on outpatient-specific dimensions (facility_type, procedure codes)

---

### 2. ETL Pipeline Configuration

#### File: `apps/ETL/config.yaml` (Lines 156-162)

**Change**: Added `op_oth_bh` section to semantic configuration

```yaml
semantic_config:
  # Existing: oon, default
  
  # NEW: OP OTH BH model configurations
  op_oth_bh:
    dev: "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_dev.yaml"
    uat: "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_uat.yaml"
    prod: "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_prod.yaml"
    local: "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_local.yaml"
    local_offshore: "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_local_offshore.yaml"
```

#### File: `apps/ETL/utils/config_loader.py` (Lines 267-268)

**Change**: Added model detection logic

```python
def get_semantic_yaml_path(self, env: str, statscl_mdl_cd: str) -> str:
    # ... existing code ...
    
    # Determine model type (OON vs default)
    if statscl_mdl_cd.upper() == 'OON':
        model_key = 'oon'
    elif statscl_mdl_cd.upper() == 'OP OTH BH':  # NEW
        model_key = 'op_oth_bh'                   # NEW
    else:
        model_key = 'default'
```

**Impact**: ETL pipeline now automatically routes OP OTH BH data to correct YAML configuration.

#### File: `apps/ETL/utils/agent_utils.py` (Lines 163-264)

**Change**: Added procedure code processing to correlation agents

```python
def call_correlation_agents(_payload, anomaly_json):
    """Call correlation agents for states, providers, DRGs, and procedures"""
    correlation_results = {
        "states": {},
        "providers": {},
        "drgs": {},
        "procs": {}  # NEW - Added procedure tracking
    }
    
    # ... existing state, provider, DRG processing ...
    
    # NEW: Process Procedures (Lines 243-264)
    if "top_contributors" in anomaly_json and "procedure_trends" in anomaly_json["top_contributors"]:
        for proc in anomaly_json["top_contributors"]["procedure_trends"]:
            correlation_agent_payload = copy.deepcopy(_payload)
            correlation_agent_payload["query"] = f"Where did change happen for procedure {proc['name']}? It {proc['insight'].lower()} by {proc['percentage_change']}"
            correlation_agent_payload["context"]["filters"].append({
                "field": "procedure_name",
                "operator": "=",
                "value": proc["name"],
                "source": "dimension_match"
            })
            # ... API call logic ...
            correlation_results["procs"][proc["name"]] = correlation_result
```

**Impact**: ETL pipeline now processes procedure codes from anomaly detection and calls correlation agent for each top procedure contributor. This is critical for OP Oth BH since outpatient care relies heavily on procedure codes (CPT codes) rather than DRG codes.

---

### 3. Agent Framework Updates

#### File: `packages/agents/src/deep_research_agents/correlation_agent.py` (Lines 482-484)

**Change**: Added HCC dimension folder aliases

```python
DIMENSION_FOLDER_ALIASES: Dict[str, str] = {
    "service_area_state": "state",
    "rendering_provider_name": "provider",
    # ... existing mappings ...
    "hcc_medium": "hcc_medium",  # NEW
    "hcc_high": "hcc_high",      # NEW
    "hcc_low": "hcc_low",        # NEW
}
```

**Impact**: Drill-down results for HCC dimensions are organized correctly in output folders.

#### File: `packages/agents/src/deep_research_agents/correlation_interaction_matrix.py` (Line 110)

**Change**: Added `hcc_medium` to clinical dimensions

```python
DEFAULT_CATEGORIES: Dict[str, CategoryConfig] = {
    "operational": {
        "dimensions": ["pa_required_code", "product_description", "facility_type"],
        "carry_through_dimensions": ["service_area_state", "lob_code"],
    },
    "clinical": {
        "dimensions": ["drg_name", "primary_diagnosis_name", "hcc_medium"],  # hcc_medium added
        "carry_through_dimensions": ["mbu_cls_short_description"],
    },
}
```

**Impact**: Interaction matrix can analyze operational vs. clinical dimensions including HCC categories.

#### Files: `user_intent.py` and `orchestrator.py`

**Change**: Updated examples to include `hcc_medium` filter

```python
# user_intent.py line 684
- "IP BH claims" + sample_values=['IP BH', 'IP Med/Surg', ...] → {field: 'hcc_medium', operator: '=', value: 'IP BH'}

# user_intent.py line 1807
context = {
    "hcc_medium": "IP BH",
    "lob_description": "Commercial",
    "period": "Rolling 3",
}

# orchestrator.py line 1913
context = {"hcc_medium": "IP BH", "lob_description": "Commercial", "period": "Rolling 3"}
```

**Impact**: Documentation and examples now demonstrate HCC filtering.

---

## Testing & Validation

### Test Coverage

**Location**: `tests/OP_Oth_BH/`

**Test Runs**: 4 timestamp directories with 478 total test artifacts

| Timestamp | Items | Coverage |
|-----------|-------|----------|
| 20260705_1248 | 325 | Initial comprehensive testing |
| 20260705_2251 | 109 | YTD period, 5 LOBs |
| 20260705_2344 | 21 | R6 period, Medicaid |
| 20260705_2358 | 23 | R6 period, Medicare Indiv |

### LOBs Tested
- Commercial Individual
- Commercial Local Group
- Medicare GRS
- Medicare Indiv
- Medicaid

### Time Periods Tested
- YTD (Year to Date)
- R3 (Rolling 3 months)
- R6 (Rolling 6 months)
- R12 (Rolling 12 months)

### Agents Validated
- ✅ Correlation Agent
- ✅ Pattern Agent
- ✅ Recommendation Agent
- ✅ Reimbursement Agent

### Automated Validation Tests

**Location**: `packages/agents/tests/test_op_oth_bh_validation.py`

The test artifacts are validated by 18 automated tests that verify:

**Correlation Agent Output**:
- Required fields (job_id, agent, status, output)
- Mathematical consistency (delta = comparison - baseline)
- Drill path structure and level numbering
- Valid status values

**Pattern Agent Output**:
- Required fields and sequential ranking
- Reimbursement policy data inclusion
- Meaningful descriptions (>50 characters)

**Recommendation Agent Output**:
- Metadata consistency (counts match actual data)
- Valid priority levels (HIGH, MEDIUM, LOW) and categories
- Non-empty evidence lists
- Actionable descriptions with action verbs
- Complete processing log

**Snowflake Output Format**:
- Required schema fields
- Correct model code ("OP Oth BH")
- Valid insight types
- Valid JSON in JSON_TXT field

**Running Tests**:
```bash
# Run all OP_Oth_BH validation tests
uv run pytest packages/agents/tests/test_op_oth_bh_validation.py -v

# Run specific test class
uv run pytest packages/agents/tests/test_op_oth_bh_validation.py::TestCorrelationAgentOutput -v
```

See `tests/OP_Oth_BH/README.md` for detailed documentation.

### Sample Test Payload

**File**: `tests/OP_Oth_BH/20260705_2358/payload/correlation_payload_tutorial-OP_Oth_BH-Medicare_Indiv-202606-R6-202603.json`

```json
{
  "conversation_id": "tutorial-OP_Oth_BH-Medicare_Indiv-202606-R6-202603",
  "context": {
    "filters": [
      {"field": "snap_month", "operator": "=", "value": 202606},
      {"field": "lob_description", "operator": "=", "value": "Medicare Indiv"},
      {"field": "hcc_medium", "operator": "=", "value": "OP Oth BH"}
    ]
  },
  "yaml_path": "configs/correlation_pattern/coc_ecap_op_oth_bh_sematic_view_with_samples_prod.yaml"
}
```

---

## Deployment

### Environment Progression

1. **Dev** → Validated with test data
2. **UAT** → Business user validation
3. **Prod** → Production deployment

### Database Targets by Environment

| Environment | Database | Schema | Warehouse |
|-------------|----------|--------|-----------|
| Dev (dv, ts) | D01_COC | COC_DTI_STG | D01_COC_DTI_LOAD_WH |
| UAT (pl) | U01_COC | COC_DTI_STG | U01_COC_DTI_LOAD_WH |
| Prod (pr) | P01_COC | COC_DTI_STG | P01_COC_DTI_LOAD_WH |

---

## Using This as Reference for New HCCs

When implementing a new HCC model (e.g., IP OB, OP Med/Surg), follow this pattern:

### Step 1: Create Semantic YAMLs
- Copy one of the OP Oth BH YAML files as a template
- Update `name`, `description`, and `comments`
- Modify `sample_values` to match your HCC (query Snowflake for actual values)
- Add/remove dimensions based on model requirements (e.g., include DRG for inpatient)
- Create all 5 environment variants

### Step 2: Update ETL Config
- Add new model section to `apps/ETL/config.yaml` under `semantic_config`
- Add detection logic to `apps/ETL/utils/config_loader.py`:
  ```python
  elif statscl_mdl_cd.upper() == '<YOUR_MODEL>':
      model_key = '<your_model_key>'
  ```

### Step 3: Update Agent Framework
- Add dimension aliases to `correlation_agent.py` if needed
- Update interaction matrix dimensions if applicable
- Update examples in `user_intent.py` and `orchestrator.py`

### Step 4: Test Thoroughly
- Create test payloads for multiple LOBs and periods
- Run all agents (correlation, pattern, recommendation, reimbursement)
- Save test results in `tests/<YOUR_MODEL>/`

### Key Considerations

**For Outpatient Models**:
- Exclude DRG dimensions
- Focus on procedure codes, facility types
- Add comment: `- DRG dimensions disabled for <MODEL> analysis`

**For Inpatient Models**:
- Include DRG dimensions
- Include admission-related metrics
- Consider authorization tracking

**Sample Values**:
- Must match exact values from Snowflake (case-sensitive)
- Query: `SELECT DISTINCT <dimension> FROM <table> WHERE <filters> LIMIT 20`

---

## Summary

The OP Oth BH implementation demonstrates the pattern for adding new statistical models:

1. **5 YAML files** define the semantic model
2. **3 ETL changes** enable automatic routing and procedure code processing
3. **4 agent updates** support HCC-specific dimensions
4. **Comprehensive testing** validates across LOBs and periods

This establishes OP Oth BH as a production-ready model alongside IP AUTH and OON, with full support for outpatient behavioral health cost analysis including procedure code correlation.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | July 8, 2026 | Initial documentation of OP Oth BH implementation |
