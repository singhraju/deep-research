"""Utility libraries for deep-research project."""

__version__ = "0.1.0"

# Export commonly used utilities
from deep_research_utils.app_constant import AppConstants
from deep_research_utils.logger_config import (
    get_logger,
    setup_logging,
    cleanup_old_logs,
    LogLevel,
    PolicyExtractorLogger
)

from deep_research_utils.ehap import EHAPBase, EHAP

from deep_research_utils.ehap_retry import (
    llm_invoke,
    structured_llm_invoke,
    post_req,
    invoke_with_token_retry
)

from deep_research_utils.snowflake_helper import (
    SnowparkHelper,
    PerformanceMetrics
)

from deep_research_utils.subset_table_manager import SubsetTableManager

from deep_research_utils.semantic_view import (
    update_semantic_view_sample_values,
    validate_semantic_view_config
)

from deep_research_utils.metric_role import (
    classify_metric,
    humanize_metric_name,
    find_metric_key_by_role,
    CLAIM_COUNT_METRIC_CANDIDATES,
    ADMISSION_COUNT_METRIC_CANDIDATES,
    PAID_PER_ADMIT_METRIC_CANDIDATES,
    PAID_RATIO_METRIC_CANDIDATES,
    ROLE_LABELS,
)

from deep_research_utils.semantic_role_classifier import (
    ALLOWED_ROLES,
    classify_dimension_semantic_roles,
    companion_json_path,
    load_companion_json,
    source_hash_for_yaml,
)

__all__ = [
    # Logger utilities
    "get_logger",
    "setup_logging",
    "cleanup_old_logs",
    "LogLevel",
    "PolicyExtractorLogger",
    # EHAP utilities
    "EHAPBase",
    "EHAP",
    # EHAP retry utilities
    "llm_invoke",
    "structured_llm_invoke",
    "post_req",
    "invoke_with_token_retry",
    # Snowflake utilities
    "SnowparkHelper",
    "PerformanceMetrics",
    "SubsetTableManager",
    # Semantic view utilities
    "update_semantic_view_sample_values",
    "validate_semantic_view_config",
    # App constants
    "AppConstants",
    # Metric role classifier
    "classify_metric",
    "humanize_metric_name",
    "find_metric_key_by_role",
    "CLAIM_COUNT_METRIC_CANDIDATES",
    "ADMISSION_COUNT_METRIC_CANDIDATES",
    "PAID_PER_ADMIT_METRIC_CANDIDATES",
    "PAID_RATIO_METRIC_CANDIDATES",
    "ROLE_LABELS",
    # Semantic role classifier
    "ALLOWED_ROLES",
    "classify_dimension_semantic_roles",
    "companion_json_path",
    "load_companion_json",
    "source_hash_for_yaml",
]
