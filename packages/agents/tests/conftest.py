"""
Shared pytest fixtures for agent tests.

Includes OON-specific fixtures for testing Out-of-Network use cases.
"""

import json
import pandas as pd
import pytest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# NOTE: Integration tests now load data from JSON fixtures.
# See fixtures/oon/integration_test_data.json for test scenarios.
# Mock data builders from the previous fixtures folder are no longer used.


class StubSnowparkHelperOON:
    """Lightweight SnowparkHelper stub returning OON-specific DataFrames."""

    def __init__(self, dataframes: Iterable[pd.DataFrame]) -> None:
        self._dataframes = list(dataframes)
        self.queries: List[str] = []

    def execute_query_and_return_pandas_df(self, query: str) -> pd.DataFrame:
        """Return the next stub DataFrame for each query."""
        self.queries.append(query)
        if not self._dataframes:
            raise AssertionError("No stubbed DataFrames remaining for query execution.")
        return self._dataframes.pop(0)

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None


@pytest.fixture
def oon_semantic_view_path():
    """Path to local OON semantic view YAML."""
    base_path = Path(__file__).parent.parent.parent.parent.parent
    return base_path / "configs" / "correlation_pattern" / "coc_ecap_oon_semantic_view_with_samples_local.yaml"


# NOTE: The following fixtures used mock data builders from the deleted fixtures folder.
# They are commented out since integration tests now load data directly from CSV files.
# If unit tests need these fixtures, they should be updated to use CSV data or recreated.


@pytest.fixture
def oon_integration_test_data():
    """Load all OON integration test scenarios from JSON."""
    json_path = Path(__file__).parent / "fixtures/oon/integration_test_data.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(params=[
    ("Commercial Individual", "R3"),
    ("Commercial Individual", "R6"),
    ("Commercial Individual", "R12"),
    ("Commercial Individual", "YTD"),
    ("Commercial Local Group", "R3"),
    ("Commercial Local Group", "R6"),
    ("Commercial Local Group", "R12"),
    ("Commercial Local Group", "YTD"),
])
def oon_data_parameterized(request, oon_integration_test_data) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Parameterized fixture providing test data for any LOB/period combination.
    
    Returns: (anomaly_json, deep_dive_json, metadata)
    """
    lob_desc, period_code = request.param
    
    # Find matching scenario
    scenario = next(
        s for s in oon_integration_test_data["test_scenarios"]
        if s["lob_desc"] == lob_desc and s["period_code"] == period_code
    )
    
    # Extract data
    anomaly_json = scenario["key_insight"]
    deep_dive_json = scenario["deep_dive"]
    metadata = {
        "snap_month": scenario["snap_month"],
        "period_end": scenario["period_end"],
        "period_code": scenario["period_code"],
        "lob_code": scenario["lob_code"],
        "lob_desc": scenario["lob_desc"],
        "model_code": scenario["model_code"]
    }
    
    return anomaly_json, deep_dive_json, metadata


@pytest.fixture
def oon_data_r6(oon_integration_test_data) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Specific fixture for Commercial Individual R6 (current test usage).
    
    Returns: (anomaly_json, deep_dive_json, metadata)
    """
    scenario = next(
        s for s in oon_integration_test_data["test_scenarios"]
        if s["lob_desc"] == "Commercial Individual" and s["period_code"] == "R6"
    )
    
    anomaly_json = scenario["key_insight"]
    deep_dive_json = scenario["deep_dive"]
    metadata = {
        "snap_month": scenario["snap_month"],
        "period_end": scenario["period_end"],
        "period_code": scenario["period_code"],
        "lob_code": scenario["lob_code"],
        "lob_desc": scenario["lob_desc"],
        "model_code": scenario["model_code"]
    }
    
    return anomaly_json, deep_dive_json, metadata


@pytest.fixture
def stub_snowpark_helper_oon():
    """Snowpark helper stub returning OON-specific DataFrames."""
    # Create sample DataFrames with OON dimensions
    df1 = pd.DataFrame({
        "in_network_code": ["OUT", "IN", "OUT"],
        "claim_network_category": [
            "Group Network/Contract Exists",
            "PAR_NO_MATCH",
            "Group Network/Contract Exists Within DOS"
        ],
        "idr_type": ["FEDERAL", None, "STATE"],
        "total_paid": [500_000, 300_000, 200_000],
        "claim_count": [100, 80, 50],
    })
    
    df2 = pd.DataFrame({
        "facility_type": ["Hospital Inpatient", "Hospital Outpatient"],
        "product_description": ["Commercial HMO", "Commercial PPO"],
        "delta_value": [800_000, 700_000],
    })
    
    return StubSnowparkHelperOON([df1, df2])


# OON-specific marker registration
def pytest_configure(config):
    """Register custom markers for OON tests."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring live connections"
    )
    config.addinivalue_line(
        "markers", "oon: Tests specific to Out-of-Network use cases"
    )
    config.addinivalue_line(
        "markers", "oon_unit: OON unit tests"
    )
    config.addinivalue_line(
        "markers", "oon_integration: OON integration tests"
    )
    config.addinivalue_line(
        "markers", "oon_correlation: OON correlation agent tests"
    )
    config.addinivalue_line(
        "markers", "oon_pattern: OON pattern agent tests"
    )
    config.addinivalue_line(
        "markers", "oon_reimbursement: OON reimbursement agent tests"
    )
    config.addinivalue_line(
        "markers", "oon_recommendation: OON recommendation agent tests"
    )
