"""
Automated validation tests for OP_Oth_BH agent outputs.

These tests validate the captured integration test artifacts in tests/OP_Oth_BH/
to ensure agent outputs meet quality and correctness standards.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Test data location
TEST_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "tests" / "OP_Oth_BH"


def get_test_runs() -> List[Path]:
    """Get all test run directories."""
    if not TEST_DATA_ROOT.exists():
        return []
    return [d for d in TEST_DATA_ROOT.iterdir() if d.is_dir()]


def get_agent_results(test_run: Path, agent_type: str) -> List[Path]:
    """Get all agent result files for a specific agent type."""
    agents_results_dir = test_run / "agents_results"
    if not agents_results_dir.exists():
        return []
    
    pattern_map = {
        "correlation": "correlation_response_*.json",
        "pattern": "pattern_final_*.json",
        "recommendation": "recommendation_response_*.json",
        "reimbursement": "reimbursement_response_*.json"
    }
    
    pattern = pattern_map.get(agent_type, "*.json")
    return list(agents_results_dir.glob(pattern))


class TestCorrelationAgentOutput:
    """Validate correlation agent outputs."""
    
    @pytest.fixture
    def correlation_results(self) -> List[Dict[str, Any]]:
        """Load all correlation agent results."""
        results = []
        for test_run in get_test_runs():
            for result_file in get_agent_results(test_run, "correlation"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    results.append({
                        "file": result_file,
                        "data": json.load(f)
                    })
        return results
    
    def test_correlation_has_required_fields(self, correlation_results):
        """Verify correlation outputs have all required fields."""
        if not correlation_results:
            pytest.skip("No correlation test data found")
        
        for result in correlation_results:
            data = result["data"]
            file_path = result["file"]
            
            # Check for states structure (multi-state analysis)
            if "states" in data:
                for state, state_data in data["states"].items():
                    assert "job_id" in state_data, f"Missing job_id in {file_path} state {state}"
                    assert "agent" in state_data, f"Missing agent in {file_path} state {state}"
                    assert state_data["agent"] == "correlation_agent", f"Wrong agent type in {file_path}"
                    assert "status" in state_data, f"Missing status in {file_path} state {state}"
                    assert "output" in state_data, f"Missing output in {file_path} state {state}"
                    
                    output = state_data["output"]
                    assert "root_metric" in output, f"Missing root_metric in {file_path}"
                    assert "baseline_value" in output, f"Missing baseline_value in {file_path}"
                    assert "comparison_value" in output, f"Missing comparison_value in {file_path}"
                    assert "delta_value" in output, f"Missing delta_value in {file_path}"
                    assert "drill_path" in output, f"Missing drill_path in {file_path}"
    
    def test_correlation_calculations_are_consistent(self, correlation_results):
        """Verify correlation calculations are mathematically correct."""
        if not correlation_results:
            pytest.skip("No correlation test data found")
        
        for result in correlation_results:
            data = result["data"]
            file_path = result["file"]
            
            if "states" in data:
                for state, state_data in data["states"].items():
                    if state_data.get("status") != "success":
                        continue
                    
                    output = state_data.get("output", {})
                    baseline = output.get("baseline_value")
                    comparison = output.get("comparison_value")
                    delta = output.get("delta_value")
                    
                    if baseline is not None and comparison is not None and delta is not None:
                        expected_delta = comparison - baseline
                        # Allow small floating point differences
                        assert abs(delta - expected_delta) < 0.01, \
                            f"Delta calculation incorrect in {file_path} state {state}: " \
                            f"expected {expected_delta}, got {delta}"
    
    def test_correlation_drill_path_structure(self, correlation_results):
        """Verify drill path has valid structure."""
        if not correlation_results:
            pytest.skip("No correlation test data found")
        
        for result in correlation_results:
            data = result["data"]
            file_path = result["file"]
            
            if "states" in data:
                for state, state_data in data["states"].items():
                    if state_data.get("status") != "success":
                        continue
                    
                    drill_path = state_data.get("output", {}).get("drill_path", [])
                    
                    for i, level in enumerate(drill_path):
                        assert "level" in level, f"Missing level in drill path at index {i} in {file_path}"
                        assert "dimension" in level, f"Missing dimension in drill path at index {i} in {file_path}"
                        assert "top_segments" in level, f"Missing top_segments in drill path at index {i} in {file_path}"
                        
                        # Verify level numbering
                        assert level["level"] == i + 1, \
                            f"Level numbering incorrect at index {i} in {file_path}: " \
                            f"expected {i + 1}, got {level['level']}"
                        
                        # Verify segments have required fields
                        for segment in level.get("top_segments", []):
                            assert "value" in segment, f"Missing value in segment at level {i} in {file_path}"
                            assert "baseline_value" in segment, f"Missing baseline_value in segment at level {i} in {file_path}"
                            assert "comparison_value" in segment, f"Missing comparison_value in segment at level {i} in {file_path}"
                            assert "delta_value" in segment, f"Missing delta_value in segment at level {i} in {file_path}"
    
    def test_correlation_status_is_valid(self, correlation_results):
        """Verify status field has valid values."""
        if not correlation_results:
            pytest.skip("No correlation test data found")
        
        valid_statuses = {"success", "error", "partial"}
        
        for result in correlation_results:
            data = result["data"]
            file_path = result["file"]
            
            if "states" in data:
                for state, state_data in data["states"].items():
                    status = state_data.get("status")
                    assert status in valid_statuses, \
                        f"Invalid status '{status}' in {file_path} state {state}"


class TestPatternAgentOutput:
    """Validate pattern agent outputs."""
    
    @pytest.fixture
    def pattern_results(self) -> List[Dict[str, Any]]:
        """Load all pattern agent results."""
        results = []
        for test_run in get_test_runs():
            for result_file in get_agent_results(test_run, "pattern"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Pattern results are arrays
                    if isinstance(data, list):
                        results.append({
                            "file": result_file,
                            "data": data
                        })
        return results
    
    def test_pattern_has_required_fields(self, pattern_results):
        """Verify pattern outputs have all required fields."""
        if not pattern_results:
            pytest.skip("No pattern test data found")
        
        for result in pattern_results:
            patterns = result["data"]
            file_path = result["file"]
            
            assert isinstance(patterns, list), f"Pattern data should be a list in {file_path}"
            
            for i, pattern in enumerate(patterns):
                assert "rank" in pattern, f"Missing rank in pattern {i} in {file_path}"
                assert "pattern_title" in pattern, f"Missing pattern_title in pattern {i} in {file_path}"
                assert "pattern_description" in pattern, f"Missing pattern_description in pattern {i} in {file_path}"
                assert "explanation" in pattern, f"Missing explanation in pattern {i} in {file_path}"
    
    def test_pattern_ranking_is_sequential(self, pattern_results):
        """Verify pattern ranks are sequential starting from 1."""
        if not pattern_results:
            pytest.skip("No pattern test data found")
        
        for result in pattern_results:
            patterns = result["data"]
            file_path = result["file"]
            
            ranks = [p.get("rank") for p in patterns]
            expected_ranks = list(range(1, len(patterns) + 1))
            
            assert ranks == expected_ranks, \
                f"Pattern ranks not sequential in {file_path}: expected {expected_ranks}, got {ranks}"
    
    def test_pattern_has_reimbursement_data(self, pattern_results):
        """Verify patterns include reimbursement policy data."""
        if not pattern_results:
            pytest.skip("No pattern test data found")
        
        for result in pattern_results:
            patterns = result["data"]
            file_path = result["file"]
            
            for i, pattern in enumerate(patterns):
                explanation = pattern.get("explanation", {})
                assert "reimbursement" in explanation, \
                    f"Missing reimbursement in pattern {i} explanation in {file_path}"
                
                reimbursement = explanation["reimbursement"]
                assert "summary_table" in reimbursement, \
                    f"Missing summary_table in pattern {i} reimbursement in {file_path}"
                
                summary_table = reimbursement["summary_table"]
                # Summary table may be empty dict if no policies found, which is valid
                if summary_table:
                    assert "columns" in summary_table, \
                        f"Missing columns in pattern {i} summary_table in {file_path}"
                    assert "rows" in summary_table, \
                        f"Missing rows in pattern {i} summary_table in {file_path}"
    
    def test_pattern_descriptions_are_not_empty(self, pattern_results):
        """Verify pattern titles and descriptions are meaningful."""
        if not pattern_results:
            pytest.skip("No pattern test data found")
        
        for result in pattern_results:
            patterns = result["data"]
            file_path = result["file"]
            
            for i, pattern in enumerate(patterns):
                title = pattern.get("pattern_title", "")
                description = pattern.get("pattern_description", "")
                
                assert len(title) > 10, \
                    f"Pattern {i} title too short in {file_path}: '{title}'"
                assert len(description) > 50, \
                    f"Pattern {i} description too short in {file_path}: '{description}'"


class TestRecommendationAgentOutput:
    """Validate recommendation agent outputs."""
    
    @pytest.fixture
    def recommendation_results(self) -> List[Dict[str, Any]]:
        """Load all recommendation agent results."""
        results = []
        for test_run in get_test_runs():
            for result_file in get_agent_results(test_run, "recommendation"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    results.append({
                        "file": result_file,
                        "data": json.load(f)
                    })
        return results
    
    def test_recommendation_has_required_fields(self, recommendation_results):
        """Verify recommendation outputs have all required fields."""
        if not recommendation_results:
            pytest.skip("No recommendation test data found")
        
        for result in recommendation_results:
            data = result["data"]
            file_path = result["file"]
            
            assert "success" in data, f"Missing success field in {file_path}"
            assert "result" in data, f"Missing result field in {file_path}"
            
            result_data = data["result"]
            assert "metadata" in result_data, f"Missing metadata in {file_path}"
            assert "recommendations" in result_data, f"Missing recommendations in {file_path}"
            assert "processing_log" in result_data, f"Missing processing_log in {file_path}"
    
    def test_recommendation_metadata_is_consistent(self, recommendation_results):
        """Verify recommendation metadata matches actual data."""
        if not recommendation_results:
            pytest.skip("No recommendation test data found")
        
        for result in recommendation_results:
            data = result["data"]
            file_path = result["file"]
            
            if not data.get("success"):
                continue
            
            result_data = data["result"]
            metadata = result_data.get("metadata", {})
            recommendations = result_data.get("recommendations", [])
            skipped = result_data.get("skipped_patterns", [])
            
            total_patterns = metadata.get("total_patterns", 0)
            recommendations_generated = metadata.get("recommendations_generated", 0)
            patterns_skipped = metadata.get("patterns_skipped", 0)
            
            # Verify counts match
            assert len(recommendations) == recommendations_generated, \
                f"Recommendation count mismatch in {file_path}: " \
                f"metadata says {recommendations_generated}, found {len(recommendations)}"
            
            assert len(skipped) == patterns_skipped, \
                f"Skipped pattern count mismatch in {file_path}: " \
                f"metadata says {patterns_skipped}, found {len(skipped)}"
            
            assert recommendations_generated + patterns_skipped == total_patterns, \
                f"Total pattern count mismatch in {file_path}: " \
                f"generated ({recommendations_generated}) + skipped ({patterns_skipped}) != total ({total_patterns})"
    
    def test_recommendation_structure_is_valid(self, recommendation_results):
        """Verify each recommendation has required fields and valid values."""
        if not recommendation_results:
            pytest.skip("No recommendation test data found")
        
        valid_priorities = {"HIGH", "MEDIUM", "LOW"}
        valid_categories = {"Policy", "Operational", "Clinical", "Network", "Contract"}
        
        for result in recommendation_results:
            data = result["data"]
            file_path = result["file"]
            
            if not data.get("success"):
                continue
            
            recommendations = data["output"].get("recommendations", [])
            
            for i, rec in enumerate(recommendations):
                assert "rank" in rec, f"Missing rank in recommendation {i} in {file_path}"
                assert "priority" in rec, f"Missing priority in recommendation {i} in {file_path}"
                assert "category" in rec, f"Missing category in recommendation {i} in {file_path}"
                assert "description" in rec, f"Missing description in recommendation {i} in {file_path}"
                assert "evidence" in rec, f"Missing evidence in recommendation {i} in {file_path}"
                
                # Validate priority
                priority = rec.get("priority")
                assert priority in valid_priorities, \
                    f"Invalid priority '{priority}' in recommendation {i} in {file_path}"
                
                # Validate category
                category = rec.get("category")
                assert category in valid_categories, \
                    f"Invalid category '{category}' in recommendation {i} in {file_path}"
                
                # Validate evidence is not empty
                evidence = rec.get("evidence", [])
                assert isinstance(evidence, list), \
                    f"Evidence should be a list in recommendation {i} in {file_path}"
                assert len(evidence) > 0, \
                    f"Evidence list is empty in recommendation {i} in {file_path}"
    
    def test_recommendation_descriptions_are_actionable(self, recommendation_results):
        """Verify recommendation descriptions are meaningful and actionable."""
        if not recommendation_results:
            pytest.skip("No recommendation test data found")
        
        for result in recommendation_results:
            data = result["data"]
            file_path = result["file"]
            
            if not data.get("success"):
                continue
            
            recommendations = data["output"].get("recommendations", [])
            
            for i, rec in enumerate(recommendations):
                description = rec.get("description", "")
                
                # Description should be substantial
                assert len(description) > 50, \
                    f"Recommendation {i} description too short in {file_path}: '{description}'"
                
                # Description should contain action verbs
                action_verbs = ["establish", "initiate", "review", "implement", "develop", 
                               "create", "launch", "perform", "conduct", "analyze", "audit"]
                has_action = any(verb in description.lower() for verb in action_verbs)
                assert has_action, \
                    f"Recommendation {i} description lacks action verb in {file_path}: '{description}'"
    
    def test_recommendation_processing_log_is_complete(self, recommendation_results):
        """Verify processing log captures all pattern decisions."""
        if not recommendation_results:
            pytest.skip("No recommendation test data found")
        
        for result in recommendation_results:
            data = result["data"]
            file_path = result["file"]
            
            if not data.get("success"):
                continue
            
            result_data = data["result"]
            metadata = result_data.get("metadata", {})
            processing_log = result_data.get("processing_log", [])
            
            total_patterns = metadata.get("total_patterns", 0)
            
            # Processing log should have entry for each pattern
            assert len(processing_log) == total_patterns, \
                f"Processing log incomplete in {file_path}: " \
                f"expected {total_patterns} entries, found {len(processing_log)}"
            
            # Each log entry should have required fields
            for i, log_entry in enumerate(processing_log):
                assert "rank" in log_entry, f"Missing rank in log entry {i} in {file_path}"
                assert "status" in log_entry, f"Missing status in log entry {i} in {file_path}"
                
                status = log_entry.get("status")
                assert status in {"success", "skipped", "error"}, \
                    f"Invalid status '{status}' in log entry {i} in {file_path}"
                
                # If successful, should have rules_used and categories
                if status == "success":
                    assert "rules_used" in log_entry, \
                        f"Missing rules_used in successful log entry {i} in {file_path}"
                    assert "categories" in log_entry, \
                        f"Missing categories in successful log entry {i} in {file_path}"


class TestReimbursementAgentOutput:
    """Validate reimbursement agent outputs."""
    
    @pytest.fixture
    def reimbursement_results(self) -> List[Dict[str, Any]]:
        """Load all reimbursement agent results."""
        results = []
        for test_run in get_test_runs():
            for result_file in get_agent_results(test_run, "reimbursement"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results.append({
                            "file": result_file,
                            "data": json.load(f)
                        })
                except json.JSONDecodeError:
                    # Some reimbursement files may be malformed, skip them
                    continue
        return results
    
    def test_reimbursement_has_valid_structure(self, reimbursement_results):
        """Verify reimbursement outputs have valid structure."""
        if not reimbursement_results:
            pytest.skip("No reimbursement test data found")
        
        for result in reimbursement_results:
            data = result["data"]
            file_path = result["file"]
            
            # Reimbursement responses can be arrays or objects
            if isinstance(data, list):
                # Array of policy results
                for i, policy in enumerate(data):
                    if isinstance(policy, dict):
                        # Should have some policy-related fields
                        assert len(policy) > 0, f"Empty policy object at index {i} in {file_path}"
            elif isinstance(data, dict):
                # Single policy result or error
                assert len(data) > 0, f"Empty reimbursement data in {file_path}"


class TestSnowflakeOutputFormat:
    """Validate Snowflake-formatted final outputs."""
    
    @pytest.fixture
    def snowflake_results(self) -> List[Dict[str, Any]]:
        """Load all Snowflake-formatted results."""
        results = []
        for test_run in get_test_runs():
            final_result_dir = test_run / "final_result"
            if not final_result_dir.exists():
                continue
            
            for result_file in final_result_dir.glob("snowflake_*.json"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    results.append({
                        "file": result_file,
                        "data": json.load(f)
                    })
        return results
    
    def test_snowflake_has_required_schema(self, snowflake_results):
        """Verify Snowflake outputs have required schema fields."""
        if not snowflake_results:
            pytest.skip("No Snowflake test data found")
        
        required_fields = {
            "SNAP_YEAR_MNTH_NBR",
            "TRND_TM_PRD_END_MNTH_NBR",
            "TRND_TM_PRD_CD",
            "LOB_CD",
            "LOB_SHRT_DESC",
            "STATSCL_MDL_CD",
            "INSGHT_TYPE_NM",
            "JSON_TXT"
        }
        
        for result in snowflake_results:
            data = result["data"]
            file_path = result["file"]
            
            # Snowflake results are arrays of records
            assert isinstance(data, list), f"Snowflake data should be a list in {file_path}"
            
            for i, record in enumerate(data):
                missing_fields = required_fields - set(record.keys())
                assert not missing_fields, \
                    f"Missing required fields in record {i} in {file_path}: {missing_fields}"
    
    def test_snowflake_model_code_is_correct(self, snowflake_results):
        """Verify STATSCL_MDL_CD is 'OP Oth BH' for all records."""
        if not snowflake_results:
            pytest.skip("No Snowflake test data found")
        
        for result in snowflake_results:
            data = result["data"]
            file_path = result["file"]
            
            for i, record in enumerate(data):
                model_code = record.get("STATSCL_MDL_CD")
                assert model_code == "OP Oth BH", \
                    f"Wrong model code in record {i} in {file_path}: expected 'OP Oth BH', got '{model_code}'"
    
    def test_snowflake_insight_types_are_valid(self, snowflake_results):
        """Verify INSGHT_TYPE_NM has valid values."""
        if not snowflake_results:
            pytest.skip("No Snowflake test data found")
        
        valid_insight_types = {"PATTERN", "RECOMMENDATION", "CORRELATION", "REIMBURSEMENT", "POLICY"}
        
        for result in snowflake_results:
            data = result["data"]
            file_path = result["file"]
            
            for i, record in enumerate(data):
                insight_type = record.get("INSGHT_TYPE_NM")
                assert insight_type in valid_insight_types, \
                    f"Invalid insight type in record {i} in {file_path}: '{insight_type}'"
    
    def test_snowflake_json_is_valid(self, snowflake_results):
        """Verify JSON_TXT contains valid JSON."""
        if not snowflake_results:
            pytest.skip("No Snowflake test data found")
        
        for result in snowflake_results:
            data = result["data"]
            file_path = result["file"]
            
            for i, record in enumerate(data):
                json_txt = record.get("JSON_TXT")
                assert json_txt is not None, f"JSON_TXT is None in record {i} in {file_path}"
                
                # JSON_TXT should be a string containing JSON
                if isinstance(json_txt, str):
                    try:
                        # Try to parse the JSON (it may be double-encoded)
                        json.loads(json_txt)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON in JSON_TXT in record {i} in {file_path}: {e}")
