"""
OON Integration Test: Full Agent Pipeline with Live Connections

End-to-end integration test that runs the complete agent pipeline:
correlation → pattern → reimbursement → recommendation

Uses real Snowflake and LLM connections with local_offshore environment.
Test data is loaded from JSON fixtures (fixtures/oon/integration_test_data.json).

Test Markers:
    - @pytest.mark.integration: Requires live connections
    - @pytest.mark.oon_integration: OON-specific integration test

Prerequisites:
    - Snowflake credentials configured (local_offshore.ini)
    - LLM API keys configured
    - EHAP credentials configured
    - Network access to all services

Run with:
    pytest packages/agents/tests/test_oon_integration_pipeline.py -v -s
    pytest -m oon_integration -v -s
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from deep_research_agents.correlation_agent import CorrelationAgent
from deep_research_agents.pattern_agent import PatternAgent
from deep_research_agents import ReimbursementAgent
from deep_research_agents.recommendation_dtr_agent import RecommendationDTRAgent


logger = logging.getLogger(__name__)

# NOTE: oon_data_r6 fixture is now loaded from conftest.py via JSON test data
# See packages/agents/tests/fixtures/oon/integration_test_data.json


@pytest.fixture
def integration_config():
    """Load local_offshore environment configuration."""
    base_path = Path(__file__).parent.parent.parent.parent
    return {
        "semantic_view": str(base_path / "configs/correlation_pattern/coc_ecap_oon_semantic_view_with_samples_local_offshore.yaml"),
        "env_config": str(base_path / "configs/local_offshore.ini"),
        "artifact_dir": Path(__file__).parent / "output/integration/oon_pipeline",
    }


@pytest.fixture
def artifact_dir(integration_config):
    """Create and return artifact directory."""
    artifact_dir = integration_config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def create_base_correlation_context(snap_month: int, lob_desc: str, period_end: int, period_code: str) -> Dict[str, Any]:
    """
    Create base correlation context with dynamic parameters from CSV.
    
    Args:
        snap_month: SNAP_YEAR_MNTH_NBR from CSV (e.g., 202606)
        lob_desc: LOB_SHRT_DESC from CSV (e.g., "Commercial Individual")
        period_end: TRND_TM_PRD_END_MNTH_NBR from CSV (e.g., 202603)
        period_code: TRND_TM_PRD_CD from CSV (e.g., "R6")
    """
    def get_period_start(period_end: int, period_code: str) -> int:
        """Calculate start month for rolling period."""
        from datetime import datetime
        end_date = datetime.strptime(str(period_end), "%Y%m")
        
        if period_code.startswith("R"):
            months_back = int(period_code[1:]) - 1  # R6 = 6 months, so go back 5
            year = end_date.year
            month = end_date.month - months_back
            while month <= 0:
                month += 12
                year -= 1
            return year * 100 + month
        elif period_code == "YTD":
            return end_date.year * 100 + 1  # January of same year
        else:
            raise ValueError(f"Unsupported period code: {period_code}")
    
    def shift_to_previous_year(period: int) -> int:
        """Calculate previous year periods."""
        year = period // 100
        month = period % 100
        return (year - 1) * 100 + month
    
    current_start = get_period_start(period_end, period_code)
    current_end = period_end
    previous_start = shift_to_previous_year(current_start)
    previous_end = shift_to_previous_year(current_end)
    
    return {
        "analysis_mode_parameters": {
            "drill_metric": ["expense_detail.total_paid"],
            "period": {
                "rolling_time_dimension": "expense_detail.incurred_month",
                "current_period": {"start_time": current_start, "end_time": current_end},
                "previous_period": {"start_time": previous_start, "end_time": previous_end}
            }
        },
        "filters": [
            {"field": "snap_month", "operator": "=", "value": snap_month, "source": "dimension_match"},
            {"field": "lob_description", "operator": "=", "value": lob_desc, "source": "dimension_match"}
        ]
    }


def transform_json_to_correlation_intents(
    payload: Dict[str, Any],
    snap_month: int,
    lob_desc: str,
    period_end: int,
    period_code: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform JSON payload into multiple correlation intents - one for each dimension.
    
    This follows the notebook pattern exactly: create separate correlation calls for
    each state, provider, and DRG in the top_contributors.
    
    Args:
        payload: JSON payload with provider/state/DRG trends
        snap_month: Snapshot month from CSV
        lob_desc: LOB description from CSV
        period_end: Period end month from CSV
        period_code: Period code from CSV (R3/R6/R12/YTD)
        
    Returns:
        Dict with keys 'states', 'providers', 'drgs', each containing list of intent dicts
    """
    base_context = create_base_correlation_context(snap_month, lob_desc, period_end, period_code)
    
    intents = {
        "states": [],
        "providers": [],
        "drgs": []
    }
    
    # Create intent for each state
    for state in payload["top_contributors"]["states"]:
        context = copy.deepcopy(base_context)
        context["filters"].append({
            "field": "service_area_state",
            "operator": "=",
            "value": state["name"],
            "source": "dimension_match"
        })
        intents["states"].append({
            "dimension_value": state["name"],
            "query": f"Where did change happen for state {state['name']}? It {state['insight'].lower()} by {state['percentage_change']}",
            "context": context
        })
    
    # Create intent for each provider
    for provider in payload["top_contributors"]["provider_trends"]:
        context = copy.deepcopy(base_context)
        context["filters"].append({
            "field": "rendering_provider_name",
            "operator": "=",
            "value": provider["name"],
            "source": "dimension_match"
        })
        intents["providers"].append({
            "dimension_value": provider["name"],
            "query": f"Where did change happen for provider {provider['name']}? It {provider['insight'].lower()} by {provider['percentage_change']}",
            "context": context
        })
    
    # Create intent for each DRG
    for drg in payload["top_contributors"]["drgs"]:
        context = copy.deepcopy(base_context)
        context["filters"].append({
            "field": "drg_name",
            "operator": "=",
            "value": drg["name"],
            "source": "dimension_match"
        })
        intents["drgs"].append({
            "dimension_value": drg["name"],
            "query": f"Where did change happen for drg {drg['name']}? It {drg['insight'].lower()} by {drg['percentage_change']}",
            "context": context
        })
    
    return intents


def save_agent_output(result: Dict[str, Any], agent_name: str, artifact_dir: Path) -> None:
    """Save agent output to artifact directory with error handling."""
    try:
        output_path = artifact_dir / f"{agent_name}_output.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"✓ Saved {agent_name} output to: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save {agent_name} output: {e}")


def combine_pattern_reimbursement(pattern_result: Dict[str, Any], reimbursement_output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine pattern analysis results with reimbursement policy data.
    Matches production ETL logic in apps/ETL/utils/agent_utils.py:366-507
    
    Args:
        pattern_result: Pattern agent result with business_patterns, cards, groups
        reimbursement_output: List of processed reimbursement outputs (one per pattern)
        
    Returns:
        List of combined pattern data ready for recommendation agent
    """
    # Extract patterns, cards, and groups
    patterns = pattern_result["output"].get("business_patterns", [])
    cards = pattern_result["output"].get("cards", [])
    groups = pattern_result["output"].get("groups", [])
    
    # Create lookup dictionaries for cards and groups by ID
    cards_by_id = {card.get("card_id"): card for card in cards if "card_id" in card}
    groups_by_id = {group.get("group_id"): group for group in groups if "group_id" in group}
    
    # Create lookup by pattern_rank
    reimbursement_by_rank = {
        idx + 1: item for idx, item in enumerate(reimbursement_output)
    }
    
    # Combine data
    combined_patterns = []
    
    for pattern in patterns:
        pattern_rank = pattern.get("pattern_rank")
        
        # Start with pattern data
        combined = dict(pattern)
        
        # Remove priority_entities field (not needed for recommendations)
        combined.pop("priority_entities", None)
        
        # Fetch source cards and groups using IDs
        source_card_ids = pattern.get("source_card_ids", [])
        source_group_ids = pattern.get("source_group_ids", [])
        
        # Fetch full card and group objects
        source_cards = [cards_by_id[card_id] for card_id in source_card_ids if card_id in cards_by_id]
        source_groups = [groups_by_id[group_id] for group_id in source_group_ids if group_id in groups_by_id]
        
        # Add source traceability with full objects
        combined["source_card_ids"] = source_card_ids
        combined["source_group_ids"] = source_group_ids
        combined["source_cards"] = source_cards
        combined["source_groups"] = source_groups
        
        # Add reimbursement data if available
        if pattern_rank in reimbursement_by_rank:
            reimbursement = reimbursement_by_rank[pattern_rank]
            
            # Add reimbursement-specific fields
            combined["reimbursement"] = {
                "summary_table": reimbursement.get("summary_table", {}),
                "reimbursement_policies": reimbursement.get("reimbursement_policies", []),
                "elevance_executive_summary": reimbursement.get("elevance_executive_summary"),
                "policies_processed": reimbursement.get("policies_processed", 0),
                "policies_successful": reimbursement.get("policies_successful", 0),
                "policies_failed": reimbursement.get("policies_failed", 0)
            }
        else:
            combined["reimbursement"] = None
        
        combined_patterns.append(combined)
    
    return combined_patterns


def generate_pipeline_summary(
    results: Dict[str, Dict[str, Any]],
    artifact_dir: Path
) -> Dict[str, Any]:
    """Generate and save pipeline summary report."""
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": "correlation → pattern → reimbursement → recommendation",
        "status": "success" if all(
            r.get("status") in ["success", "partial_success"] 
            for r in results.values()
        ) else "failed",
        "agents": {
            name: {
                "status": result.get("status"),
                "execution_time_ms": result.get("execution", {}).get("duration_ms"),
                "output_keys": list(result.get("output", {}).keys())
            }
            for name, result in results.items()
        },
        "artifacts": {
            "correlation": str(artifact_dir / "correlation_output.json"),
            "pattern": str(artifact_dir / "pattern_output.json"),
            "reimbursement": str(artifact_dir / "reimbursement_output.json"),
            "recommendation": str(artifact_dir / "recommendation_output.json"),
        }
    }
    
    summary_path = artifact_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved pipeline summary to {summary_path}")
    return summary

@pytest.mark.oon
@pytest.mark.integration
@pytest.mark.oon_integration
class TestOONIntegrationPipeline:
    """
    End-to-end integration test for OON agent pipeline.
    
    Requires: Snowflake, LLM, and EHAP credentials configured.
    
    This test executes all four agents in sequence with real connections:
    1. Correlation agent (Snowflake queries)
    2. Pattern agent (LLM synthesis)
    3. Reimbursement agent (EHAP + LLM)
    4. Recommendation DTR agent (LLM + DTR rules)
    """
    
    def test_full_oon_pipeline_with_live_connections(
        self,
        oon_data_r6,
        integration_config,
        artifact_dir
    ):
        """
        Execute complete OON agent pipeline with Commercial Individual R6 data from JSON fixture.
        
        Test Flow:
            1. Transform JSON payload to correlation intent
            2. Execute correlation agent → validate
            3. Execute pattern agent → validate
            4. Execute reimbursement agent → validate
            5. Execute recommendation agent → validate
            6. Generate pipeline summary
        
        Artifacts saved to: packages/agents/tests/output/integration/oon_pipeline/
        """
        # Unpack data from JSON fixture
        anomaly_json, deep_dive_json, metadata = oon_data_r6
        
        # Skip if credentials not configured
        try:
            from deep_research_utils import SnowparkHelper
            SnowparkHelper(env_config_path=integration_config["env_config"])
        except Exception as e:
            pytest.skip(f"Snowflake credentials not configured: {e}")
        
        results = {}
        conversation_id = f"oon_integration_test_{metadata['period_code']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("=" * 80)
        logger.info(f"Starting OON Integration Pipeline Test - {metadata['lob_desc']} {metadata['period_code']}")
        logger.info("=" * 80)
        logger.info(f"Snapshot Month: {metadata['snap_month']}, Period End: {metadata['period_end']}")
        logger.info(f"LOB: {metadata['lob_desc']}, Model: {metadata['model_code']}")
        logger.info(f"Conversation ID: {conversation_id}")
        logger.info(f"Artifacts will be saved to: {artifact_dir}")
        logger.info("=" * 80)
        
        # ========================================
        # Phase 1: Transform JSON to Multiple Intents
        # ========================================
        logger.info("\n[Phase 1] Transforming JSON payload to correlation intents...")
        correlation_intents = transform_json_to_correlation_intents(
            anomaly_json,
            metadata['snap_month'],
            metadata['lob_desc'],
            metadata['period_end'],
            metadata['period_code']
        )
        
        total_intents = (len(correlation_intents["states"]) + 
                        len(correlation_intents["providers"]) + 
                        len(correlation_intents["drgs"]))
        logger.info(f"Generated {total_intents} correlation intents:")
        logger.info(f"  - {len(correlation_intents['states'])} state(s)")
        logger.info(f"  - {len(correlation_intents['providers'])} provider(s)")
        logger.info(f"  - {len(correlation_intents['drgs'])} DRG(s)")
        
        # Save all intents for debugging
        try:
            intents_path = artifact_dir / "correlation_intents_all.json"
            with intents_path.open("w", encoding="utf-8") as f:
                json.dump(correlation_intents, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved all correlation intents to: {intents_path}")
        except Exception as e:
            logger.warning(f"Could not save correlation intents: {e}")
        
        # ========================================
        # Phase 2: Correlation Agent - Multiple Calls
        # ========================================
        logger.info("\n[Phase 2] Executing Correlation Agent for all dimensions...")
        
        correlation_agent = CorrelationAgent(
            yaml_path=integration_config["semantic_view"]
        )
        
        correlation_results = {
            "states": {},
            "providers": {},
            "drgs": {}
        }
        
        # Execute correlation for all states
        logger.info(f"\n  Running correlation for {len(correlation_intents['states'])} state(s)...")
        for intent_data in correlation_intents["states"]:
            dimension_value = intent_data["dimension_value"]
            logger.info(f"    → Analyzing state: {dimension_value}")
            
            result = correlation_agent.execute(
                query=intent_data["query"],
                context=intent_data["context"],
                conversation_id=f"{conversation_id}_state_{dimension_value}",
                job_id=f"{conversation_id}_correlation_state_{dimension_value}"
            )
            correlation_results["states"][dimension_value] = result
            
            if result.get("status") != "success":
                logger.warning(f"    ⚠ Correlation failed for state {dimension_value}: {result.get('status')}")
        
        # Execute correlation for all providers
        logger.info(f"\n  Running correlation for {len(correlation_intents['providers'])} provider(s)...")
        for intent_data in correlation_intents["providers"]:
            dimension_value = intent_data["dimension_value"]
            logger.info(f"    → Analyzing provider: {dimension_value}")
            
            result = correlation_agent.execute(
                query=intent_data["query"],
                context=intent_data["context"],
                conversation_id=f"{conversation_id}_provider_{dimension_value}",
                job_id=f"{conversation_id}_correlation_provider_{dimension_value}"
            )
            correlation_results["providers"][dimension_value] = result
            
            if result.get("status") != "success":
                logger.warning(f"    ⚠ Correlation failed for provider {dimension_value}: {result.get('status')}")
        
        # Execute correlation for all DRGs
        logger.info(f"\n  Running correlation for {len(correlation_intents['drgs'])} DRG(s)...")
        for intent_data in correlation_intents["drgs"]:
            dimension_value = intent_data["dimension_value"]
            logger.info(f"    → Analyzing DRG: {dimension_value}")
            
            result = correlation_agent.execute(
                query=intent_data["query"],
                context=intent_data["context"],
                conversation_id=f"{conversation_id}_drg_{dimension_value}",
                job_id=f"{conversation_id}_correlation_drg_{dimension_value}"
            )
            correlation_results["drgs"][dimension_value] = result
            
            if result.get("status") != "success":
                logger.warning(f"    ⚠ Correlation failed for DRG {dimension_value}: {result.get('status')}")
        
        # Calculate correlation status
        successful_correlations = sum(
            1 for dim_results in correlation_results.values() 
            for result in dim_results.values() 
            if result.get("status") == "success"
        )
        failed_correlations = total_intents - successful_correlations
        
        # Wrap correlation results with status field
        results["correlation"] = {
            "status": "success" if successful_correlations == total_intents else (
                "partial_success" if successful_correlations > 0 else "failed"
            ),
            "output": correlation_results,
            "execution": {
                "total_calls": total_intents,
                "successful": successful_correlations,
                "failed": failed_correlations
            }
        }
        save_agent_output(correlation_results, "correlation", artifact_dir)
        
        logger.info(f"\n✓ Correlation complete: {successful_correlations}/{total_intents} successful")
        
        assert successful_correlations > 0, \
            f"All correlation calls failed - no successful results"
        
        drill_path_len = 0
        logger.info(f"✓ Correlation complete: {drill_path_len} drill level(s)")
        
        # ========================================
        # Phase 3: Pattern Agent
        # ========================================
        logger.info("\n[Phase 3] Executing Pattern Agent with LLM synthesis...")
        
        pattern_agent = PatternAgent()
        
        # Pattern agent expects correlation_results structure matching notebook
        pattern_input = {
            "context": {
                "correlation_results": results["correlation"]["output"],  # Pass full multi-dimensional results
                "anomaly_context": anomaly_json,  # KEY_INSIGHT data from CSV
                "deep_dive_report": deep_dive_json,  # DEEP_DIVE data from CSV
                "semantic_config_path": integration_config["semantic_view"]
            },
            "conversation_id": conversation_id,
            "job_id": f"{conversation_id}_pattern"
        }
        
        pattern_result = pattern_agent.execute(**pattern_input)
        
        results["pattern"] = pattern_result
        save_agent_output(pattern_result, "pattern", artifact_dir)
        
        # Validate pattern output
        assert pattern_result.get("status") == "success", \
            f"Pattern failed: {pattern_result.get('status')}"
        assert "output" in pattern_result, "Pattern result missing output"
        assert "cards" in pattern_result["output"], "Pattern output missing cards"
        
        num_cards = len(pattern_result["output"]["cards"])
        num_groups = len(pattern_result["output"].get("groups", []))
        logger.info(f"✓ Pattern complete: {num_cards} card(s), {num_groups} group(s)")
        
        # Debug: Check why no cards were generated
        if num_cards == 0:
            logger.warning("Pattern agent returned 0 cards!")
            logger.warning(f"Pattern output keys: {list(pattern_result['output'].keys())}")
            logger.warning(f"Groups count: {num_groups}")
            if "patterns" in pattern_result["output"]:
                logger.warning(f"Raw patterns count: {len(pattern_result['output']['patterns'])}")
            # Check correlation drill paths from all dimensions
            total_drill_items = sum(
                len(result.get("output", {}).get("drill_path", []))
                for dim_results in correlation_results.values()
                for result in dim_results.values()
            )
            logger.warning(f"Total correlation drill path items across all dimensions: {total_drill_items}")
            
            # Save detailed debug info
            try:
                debug_info = {
                    "issue": "No pattern cards generated",
                    "pattern_output_keys": list(pattern_result["output"].keys()),
                    "num_groups": num_groups,
                    "num_patterns": len(pattern_result["output"].get("patterns", [])),
                    "total_correlation_drill_items": total_drill_items,
                    "successful_correlations": successful_correlations,
                    "pattern_status": pattern_result.get("status"),
                    "correlation_summary": {
                        dim_type: {dim_val: res.get("status") for dim_val, res in dim_results.items()}
                        for dim_type, dim_results in correlation_results.items()
                    }
                }
                debug_path = artifact_dir / "debug_no_cards.json"
                with debug_path.open("w", encoding="utf-8") as f:
                    json.dump(debug_info, f, indent=2, ensure_ascii=False, default=str)
                logger.warning(f"✓ Saved debug info to: {debug_path}")
            except Exception as e:
                logger.warning(f"Could not save debug info: {e}")
        
        # Verify OON dimensions in cards (only if cards exist)
        has_network_dims = any(
            "network" in card.get("canonical_dimensions", {}) or
            "claim_network_category" in str(card.get("canonical_dimensions", {}))
            for card in pattern_result["output"]["cards"]
        ) if num_cards > 0 else False
        logger.info(f"  Network dimensions in cards: {has_network_dims}")
        
        # ========================================
        # Phase 4: Reimbursement Agent (ALL Patterns)
        # ========================================
        logger.info("\n[Phase 4] Executing Reimbursement Agent for ALL patterns with EHAP + LLM...")
        
        reimbursement_agent = ReimbursementAgent()
        
        # Extract business patterns and cards (matching production ETL pattern)
        business_patterns = pattern_result["output"].get("business_patterns", [])
        all_cards = pattern_result["output"].get("cards", [])
        
        if not business_patterns:
            logger.error("=" * 80)
            logger.error("INTEGRATION TEST ISSUE: No business patterns generated!")
            logger.error("=" * 80)
            logger.error("Possible causes:")
            logger.error("1. Correlation drill path may be empty or insufficient")
            logger.error("2. LLM pattern synthesis found no meaningful patterns")
            logger.error("3. Pattern thresholds filtered out all candidates")
            logger.error("4. Semantic view may not match the correlation output structure")
            logger.error("")
            logger.error("Check the saved artifacts:")
            logger.error(f"  - Correlation Intents: {artifact_dir / 'correlation_intents_all.json'}")
            logger.error(f"  - Correlation Output: {artifact_dir / 'correlation_output.json'}")
            logger.error(f"  - Pattern Output: {artifact_dir / 'pattern_output.json'}")
            logger.error(f"  - Debug Info: {artifact_dir / 'debug_no_cards.json'}")
            logger.error("=" * 80)
            
            # Save partial pipeline summary before failing
            try:
                partial_summary = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "failure_reason": "No business patterns generated",
                    "completed_stages": ["correlation", "pattern"],
                    "failed_stage": "business_patterns_validation",
                    "artifacts": {
                        "correlation_intents": str(artifact_dir / "correlation_intents_all.json"),
                        "correlation_output": str(artifact_dir / "correlation_output.json"),
                        "pattern_output": str(artifact_dir / "pattern_output.json"),
                        "debug_info": str(artifact_dir / "debug_no_cards.json")
                    }
                }
                summary_path = artifact_dir / "pipeline_summary_partial.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(partial_summary, f, indent=2, ensure_ascii=False)
                logger.error(f"✓ Saved partial pipeline summary to: {summary_path}")
            except Exception as e:
                logger.error(f"Could not save partial summary: {e}")
            
            pytest.fail("No business patterns generated - cannot proceed with reimbursement/recommendation")
        
        # Process ALL business patterns (matching production ETL lines 810-846)
        logger.info(f"Processing {len(business_patterns)} business pattern(s) for reimbursement...")
        full_reim_results = []
        
        for idx, business_pattern in enumerate(business_patterns):
            pattern_rank = business_pattern.get('pattern_rank')
            pattern_title = business_pattern.get('top_pattern', 'N/A')
            logger.info(f"  [{idx+1}/{len(business_patterns)}] Pattern {pattern_rank}: {pattern_title}")
            
            try:
                # Call reimbursement with BOTH pattern and cards (matching production ETL)
                reimbursement_result = reimbursement_agent.execute(
                    context={
                        "pattern": business_pattern,  # Business pattern with pattern_rank
                        "cards": all_cards  # All cards for LOB/Product extraction
                    },
                    conversation_id=f"{conversation_id}_{pattern_rank}",
                    job_id=f"{conversation_id}_reimbursement_{pattern_rank}"
                )
                full_reim_results.append(reimbursement_result)
                
                # Log success
                num_policies = len(reimbursement_result.get("output", {}).get("reimbursement_policies", []))
                logger.info(f"    ✓ Success: {num_policies} policy/policies extracted")
                
            except Exception as e:
                # Handle errors gracefully (404 is EXPECTED) - matching production ETL lines 324-326
                logger.warning(f"    ⚠ Reimbursement failed for pattern {pattern_rank}: {type(e).__name__}: {str(e)}")
                logger.warning(f"    → Creating empty structure (404 errors are expected for some patterns)")
                
                # Create error structure matching production ETL
                full_reim_results.append({
                    "detail": {"status": False, "error": str(e)},
                    "error_type": type(e).__name__,
                    "agent_name": "reimbursement_policy"
                })
        
        # Convert to output format - handle errors gracefully (matching production ETL lines 872-886)
        reim_result_output = []
        for idx, result in enumerate(full_reim_results):
            if "detail" in result:  # Error case (404 or other)
                logger.info(f"  Pattern {idx+1}: Using empty reimbursement structure (error during processing)")
                reim_result_output.append({
                    "pattern_rank": idx + 1,
                    "summary_table": {},
                    "reimbursement_policies": [],
                    "elevance_executive_summary": None,
                    "policies_processed": 0,
                    "policies_successful": 0,
                    "policies_failed": 0
                })
            else:  # Success case
                reim_result_output.append(result["output"])
        
        # Calculate reimbursement status
        num_successful = len([r for r in reim_result_output if r.get("reimbursement_policies", [])])
        num_total = len(business_patterns)
        num_failed = num_total - num_successful
        
        # Wrap reimbursement results with status field
        results["reimbursement"] = {
            "status": "success" if num_successful == num_total else (
                "partial_success" if num_successful > 0 else "failed"
            ),
            "output": {
                "full_results": full_reim_results,
                "processed_output": reim_result_output
            },
            "execution": {
                "total_patterns": num_total,
                "successful": num_successful,
                "failed_or_empty": num_failed
            }
        }
        save_agent_output(full_reim_results, "reimbursement_all", artifact_dir)
        save_agent_output(reim_result_output, "reimbursement_processed", artifact_dir)
        
        logger.info(f"\n✓ Reimbursement complete for {num_total} pattern(s):")
        logger.info(f"  - {num_successful} successful (policies extracted)")
        logger.info(f"  - {num_failed} failed/empty (404 or other errors)")
        
        # ========================================
        # Phase 5: Recommendation DTR Agent
        # ========================================
        logger.info("\n[Phase 5] Executing Recommendation DTR Agent on ALL patterns...")
        
        # Combine pattern + reimbursement data (matching production ETL line 913)
        logger.info("Combining pattern and reimbursement data...")
        combined_patterns_data = combine_pattern_reimbursement(
            pattern_result, 
            results["reimbursement"]["output"]["processed_output"]
        )
        
        # Log combined structure
        logger.info(f"Combined {len(combined_patterns_data)} pattern(s) with reimbursement data")
        for idx, cp in enumerate(combined_patterns_data):
            has_policies = len(cp.get("reimbursement", {}).get("reimbursement_policies", [])) if cp.get("reimbursement") else 0
            logger.info(f"  Pattern {idx+1}: {has_policies} policies")
        
        # Save combined payload for debugging
        save_agent_output(combined_patterns_data, "recommendation_payload", artifact_dir)
        
        recommendation_agent = RecommendationDTRAgent()
        
        # Call recommendation with combined data (matching production ETL line 916)
        recommendation_result = recommendation_agent.execute(
            patterns_data=combined_patterns_data,
            conversation_id=conversation_id,
            job_id=f"{conversation_id}_recommendation"
        )
        
        results["recommendation"] = recommendation_result
        save_agent_output(recommendation_result, "recommendation", artifact_dir)
        
        # Validate recommendation output
        assert recommendation_result.get("status") in ["success", "partial_success"], \
            f"Recommendation failed: {recommendation_result.get('status')}"
        assert "output" in recommendation_result, "Recommendation result missing output"
        
        num_recommendations = len(recommendation_result["output"].get("recommendations", []))
        num_skipped = len(recommendation_result["output"].get("skipped_patterns", []))
        logger.info(f"\n✓ Recommendation complete: {num_recommendations} recommendation(s), {num_skipped} skipped")
        
        # ========================================
        # Phase 6: Pipeline Summary
        # ========================================
        logger.info("\n[Phase 6] Generating pipeline summary...")
        
        summary = generate_pipeline_summary(results, artifact_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("OON Integration Pipeline Test Complete")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {summary['status']}")
        logger.info(f"\nArtifacts Directory: {artifact_dir}")
        logger.info("\nSaved Files:")
        logger.info(f"  1. correlation_intents_all.json    - All 12 correlation intents")
        logger.info(f"  2. correlation_output.json         - All correlation results (nested)")
        logger.info(f"  3. pattern_output.json             - Business patterns and cards")
        logger.info(f"  4. reimbursement_all_output.json   - All reimbursement results (raw)")
        logger.info(f"  5. reimbursement_processed_output.json - Processed reimbursement data")
        logger.info(f"  6. recommendation_payload_output.json  - Combined pattern+reimbursement input")
        logger.info(f"  7. recommendation_output.json      - DTR recommendations")
        logger.info(f"  8. pipeline_summary.json           - Overall execution summary")
        
        # Final assertions
        assert summary["status"] in ["success", "partial_success"], \
            "Pipeline did not complete successfully"
        assert len(business_patterns) > 0, "No business patterns generated"
        assert len(results["reimbursement"]["output"]["full_results"]) == len(business_patterns), \
            f"Reimbursement count mismatch: {len(results['reimbursement']['output']['full_results'])} != {len(business_patterns)}"
        assert num_recommendations > 0 or num_skipped > 0, \
            "No recommendations generated or patterns skipped"
        
        logger.info("\n✓ All pipeline stages completed successfully")
        logger.info("=" * 80)
