# Utils Package - Agent Instructions

## Overview
`deep-research-utils` holds the cross-cutting helpers that agents, the UI, and the ETL pipeline all reach for: EHAP/LLM transport, Snowflake helpers, the semantic-view loader, vault config, logging, caching, and retry.

## Package Structure
```
packages/utils/
├── src/
│   └── deep_research_utils/
│       ├── __init__.py          # public exports
│       ├── app_constant.py      # AppConstants
│       ├── logger_config.py     # get_logger, setup_logging, cleanup_old_logs, LogLevel, PolicyExtractorLogger
│       ├── ehap.py              # EHAPBase, EHAP — Elevance Health API Platform client
│       ├── ehap_retry.py        # llm_invoke, structured_llm_invoke, post_req, invoke_with_token_retry
│       ├── snowflake_helper.py  # SnowparkHelper, PerformanceMetrics
│       ├── semantic_view.py     # update_semantic_view_sample_values, validate_semantic_view_config
│       ├── vault_config.py      # secrets / vault loader
│       ├── cache_utils.py       # cache helpers (Redis key handling, etc.)
│       ├── retry_helper.py      # generic retry decorators / utilities
│       └── examples/            # usage examples
├── pyproject.toml
├── README.md
├── SEMANTIC_VIEW_UPDATE_SUMMARY.md
└── AGENTS.md (this file)
```

## What Each Module Does
- **`ehap` / `ehap_retry`** — Token-aware client for the EHAP LLM bridge. Prefer `llm_invoke` / `structured_llm_invoke` from `ehap_retry` over calling `EHAP` directly; they handle token-expiry retry.
- **`snowflake_helper`** — `SnowparkHelper` for query execution with built-in `PerformanceMetrics`. Used by agents and the ETL pipeline.
- **`semantic_view`** — Loads and validates the YAML semantic views under `configs/correlation_pattern/`. See `docs/semantic_view_utilities.md`.
- **`vault_config`** — Reads secrets from vault / `.env`. Never log values from here.
- **`logger_config`** — Project logging setup. Use `get_logger(__name__)` rather than `logging.getLogger`; this routes through the project's file + console handlers and respects `cleanup_old_logs`.
- **`cache_utils` / `retry_helper`** — Generic infrastructure helpers.

## Key Principles
1. **Reusable**: Functions here are generic across agents, UI, and ETL — keep them domain-light
2. **Type-Hinted Public API**: All exported functions need accurate type hints
3. **Minimal Dependencies**: Don't pull in heavy deps for one-off needs
4. **Pure When Possible**: Prefer stateless functions over module-level state

## Development Guidelines
- Add new utility modules under `src/deep_research_utils/`
- Export new public symbols from `__init__.py` (and add them to `__all__`)
- Update dependencies with `uv add --package deep-research-utils <dependency>`
- Document non-obvious behavior with docstrings — these utilities are called from many places

## Common Tasks
- **New semantic-view validation rule**: edit `semantic_view.py`; run agent tests that load YAML configs
- **EHAP transport change**: edit `ehap.py` / `ehap_retry.py`; verify token-retry path still works (`invoke_with_token_retry`)
- **New Snowflake query pattern**: extend `SnowparkHelper` rather than duplicating connection logic in callers
- **New logger / log file**: add to `logger_config.py` and surface through `get_logger`

## Notes
- **Do NOT read `.env`** or any credentials files when working in this package, even though `vault_config.py` references them.
- See `SEMANTIC_VIEW_UPDATE_SUMMARY.md` for the most recent semantic-view schema changes.