# Deep Research Streamlit UI

This Streamlit app provides a lightweight interface for the Deep Research orchestrator.

## What it does
- Reads `configs/{ENVIRONMENT}.ini` to pick the active `correlation_pattern/*.yaml` semantic view (`ENVIRONMENT` defaults to `dev`).
- Builds the filter panel dynamically from the YAML: dimensions come from `analysis_modes[*].drill_dimensions`, time filters from `time_dimensions` shared by every table, metrics from `drill_metric` + `explainer_metrics`.
- Pulls live filter values from Snowflake on each session (`SELECT DISTINCT` for enum-like columns, `MIN/MAX` for time columns), cached for one hour. Falls back to the YAML's `sample_values` when Snowflake is unavailable.
- Sends the question and selected filters as context into the orchestrator.
- Shows the high-level step summaries (recent_step_summaries) in a collapsible panel.
- Displays analysis/report/visual/hypothesis contracts returned by the orchestrator.
- Provides a collapsible LangGraph visualization panel on the right.

## Run locally
From the repo root:

```bash
streamlit run apps/ui/src/app.py
```

## Optional integrations
The UI can run without external services. To enable LLM reasoning or Snowflake execution, provide the
following environment variables:

### LLM (EHAP/OpenAI bridge)
- `EHAP_BASE_URL`
- `EHAP_CLIENT_ID`
- `EHAP_CLIENT_SECRET`
- `EHAP_LLM_MODEL` (optional)
- `DEEP_RESEARCH_LLM_MODEL` (optional)

### Snowflake execution
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_SECRET`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`

## Notes
- Filter values map to the orchestrator context so the user_intent step has soft context defaults.
- The HCC dropdown uses values from the provided design screenshots.
- Graph rendering uses LangGraph's Mermaid output with a PNG fallback.
