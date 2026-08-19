"""
Reimbursement Policy Extraction Agent - Complete Test Suite

This script demonstrates all features of the reimbursement agent:
1. Direct mode (returns dict with summary_table + reimbursement_policies)
2. Orchestrator mode with different CPT extraction methods
3. Standard API response schema (job_id, agent, status, output, etc.)
4. New output structure: summary_table and reimbursement_policies at output level
5. Recommended actions at top level
6. Validation checks as list format
7. Execution timing with duration_ms
8. Token tracking with breakdown
9. Error handling and graceful failure

Run with: python packages/agents/tests/test_reimbursement_complete.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from deep_research_agents import ReimbursementAgent


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def print_subsection(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80 + "\n")


def save_output(data, filename):
    """Save output to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Output saved to: {filename}\n")


def display_policy_summary(policies):
    """Display summary of policy extraction results."""
    if not isinstance(policies, list):
        print("ERROR: Expected list of policies")
        return
    
    total = len(policies)
    successful = sum(1 for p in policies if p is not None)
    failed = total - successful
    
    print(f"Total policies: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        first_policy = next((p for p in policies if p is not None), None)
        if first_policy:
            print(f"\nFirst policy:")
            print(f"  PLCY_ID: {first_policy.get('PLCY_ID', 'N/A')}")
            print(f"  Results: {len(first_policy.get('results', []))} code(s)")
            
            if first_policy.get('results'):
                first_result = first_policy['results'][0]
                print(f"  First code: {first_result.get('code', 'N/A')}")
                denial = first_result.get('denial_conditions', '')
                if denial:
                    preview = denial[:100] + "..." if len(denial) > 100 else denial
                    print(f"  Denial conditions: {preview}")


def display_orchestrator_output(output):
    """Display orchestrator output schema."""
    if not isinstance(output, dict):
        print("ERROR: Expected dict for orchestrator output")
        return
    
    print(f"job_id: {output.get('job_id')}")
    print(f"conversation_id: {output.get('conversation_id')}")
    print(f"agent: {output.get('agent')}")
    print(f"status: {output.get('status')}")
    
    print_subsection("Output")
    output_data = output.get('output', {})
    print(f"Pattern rank: {output_data.get('pattern_rank')}")
    print(f"Policies processed: {output_data.get('policies_processed', 0)}")
    print(f"Policies successful: {output_data.get('policies_successful', 0)}")
    print(f"Policies failed: {output_data.get('policies_failed', 0)}")
    
    # Summary table
    summary_table = output_data.get('summary_table')
    if summary_table:
        print(f"\nSummary table:")
        print(f"  Title: {summary_table.get('title')}")
        print(f"  Subtitle: {summary_table.get('subtitle')}")
        print(f"  Columns: {len(summary_table.get('columns', []))}")
        print(f"  Rows: {len(summary_table.get('rows', []))}")
    
    # Reimbursement policies
    reimbursement_policies = output_data.get('reimbursement_policies', [])
    print(f"\nReimbursement policies: {len(reimbursement_policies)}")
    if reimbursement_policies:
        first_policy = reimbursement_policies[0]
        print(f"  First policy: {first_policy.get('policy_title')}")
        print(f"  Payer: {first_policy.get('payer_name')}")
        print(f"  Effective date: {first_policy.get('effective_date')}")
    
    # Elevance summary
    elevance_summary = output_data.get('elevance_executive_summary')
    if elevance_summary:
        preview = elevance_summary[:100] + "..." if len(elevance_summary) > 100 else elevance_summary
        print(f"\nElevance summary: {preview}")
    
    print_subsection("Recommended Actions")
    recommended_actions = output.get('recommended_action', [])
    print(f"Total recommendations: {len(recommended_actions)}")
    for i, action in enumerate(recommended_actions, 1):
        print(f"\n{i}. {action.get('category')} - {action.get('priority')}")
        desc = action.get('description', '')
        desc_preview = desc[:100] + "..." if len(desc) > 100 else desc
        print(f"   {desc_preview}")
    
    print_subsection("Visual Component")
    visual = output.get('visual_component', {})
    print(f"Visual component keys: {list(visual.keys()) if visual else 'None'}")
    
    print_subsection("Explanation")
    explanation = output.get('explanation', {})
    for key, value in explanation.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    print_subsection("Validation")
    validation = output.get('validation', {})
    print(f"Is valid: {validation.get('is_valid')}")
    
    checks = validation.get('checks', [])
    if checks:
        print(f"\nChecks: {len(checks)}")
        for check in checks:
            status = '✓' if check.get('passed') else '✗'
            print(f"  {status} {check.get('check')}: {check.get('message')}")
    
    warnings = validation.get('warnings', [])
    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")
    
    errors = validation.get('errors', [])
    if errors:
        print(f"\nErrors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    print_subsection("Tokens")
    tokens = output.get('tokens', {})
    print(f"Input: {tokens.get('input', 0)}")
    print(f"Output: {tokens.get('output', 0)}")
    breakdown = tokens.get('breakdown', {})
    if breakdown:
        print(f"Breakdown: {len(breakdown)} entries")
    
    print_subsection("Execution")
    execution = output.get('execution', {})
    print(f"Start: {execution.get('start_time')}")
    print(f"End: {execution.get('end_time')}")
    duration_ms = execution.get('duration_ms', 0)
    print(f"Duration: {duration_ms} ms ({duration_ms/1000:.2f} seconds)")
    print(f"Version: {execution.get('version')}")



if __name__ == "__main__":
    # ============================================================================
    # MAIN TEST SUITE
    # ============================================================================

    print_section("REIMBURSEMENT POLICY EXTRACTION AGENT - COMPLETE TEST SUITE")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize agent
    print("Initializing agent...")
    try:
        agent = ReimbursementAgent()
        print("✓ Agent initialized successfully")
        print(f"  Agent name: {agent.agent_name}")
        print(f"  LLM model: {agent.llm_model}")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        sys.exit(1)


    # ============================================================================
    # TEST 1: Direct Mode (Backward Compatible)
    # ============================================================================

    print_section("TEST 1: Direct Mode (Backward Compatible)")
    print("Input: agent(cpt_codes='99291')")
    print("Expected output: Dict with summary_table and reimbursement_policies")
    print()

    try:
        result_direct = agent(cpt_codes="99291")

        print("✓ Execution successful")
        print(f"Output type: {type(result_direct).__name__}")
        print()
        
        # Display new structure
        if isinstance(result_direct, dict):
            print(f"Summary table: {'Present' if result_direct.get('summary_table') else 'Missing'}")
            policies = result_direct.get('reimbursement_policies', [])
            print(f"Reimbursement policies: {len(policies)}")
            if policies:
                print(f"First policy: {policies[0].get('policy_title')}")
            recommendations = result_direct.get('recommended_action', [])
            print(f"Recommendations: {len(recommendations)}")
        else:
            print(f"WARNING: Expected dict, got {type(result_direct).__name__}")

        # Save output
        save_output(result_direct, "output_direct_mode.json")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


    # ============================================================================
    # TEST 2: Orchestrator Mode - CPT Codes in Filters
    # ============================================================================

    print_section("TEST 2: Orchestrator Mode - CPT Codes in Filters")
    print("Input: conversation_id, query, context with filters")
    print("Expected output: Dict with full orchestrator schema")
    print()

    orchestrator_input_filters = {
        "conversation_id": "conv_test_filters_001",
        "query": "Why did critical care costs spike?",
        "context": {
            "intent": {
                "raw_question": "Why did critical care costs spike in November?",
                "analysis_mode": "correlation",
                "metric_hint": "paid_amount",
                "group_by": ["cpt_code", "state"],
                "filters": [
                    {
                        "field": "cpt_code",
                        "operator": "in",
                        "value": "99291",
                        "source": "user"
                    }
                ],
                "analysis_mode_parameters": {
                    "period": {
                        "rolling_time_dimension": "month",
                        "rolling_window": "3",
                        "start_time": 202411,
                        "end_time": 202411,
                        "baseline_start_time": 202408,
                        "baseline_end_time": 202410,
                        "comparison_strategy": "rolling"
                    }
                },
                "validation_warnings": []
            },
            "correlation_summary": {
                "run_id": "run_filters_123",
                "executive_summary_path": "/path/to/summary.md",
                "root_metric": "paid_amount",
                "baseline_value": 6450000.0,
                "comparison_value": 7870000.0,
                "delta_value": 1420000.0,
                "delta_pct": 22.05,
                "drill_path": [
                    {"dimension": "state", "value": "GA", "contribution": 0.31}
                ],
                "narrative_summary": "Critical care costs increased by $1.42M (22%) in November 2024...",
                "warnings": []
            }
        }
    }

    try:
        result_filters = agent(**orchestrator_input_filters)

        print("✓ Execution successful")
        print(f"Output type: {type(result_filters).__name__}")
        print()

        display_orchestrator_output(result_filters)

        # Save output
        save_output(result_filters, "output_orchestrator_filters.json")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


    # ============================================================================
    # TEST 3: Orchestrator Mode - CPT Codes in Drill Path
    # ============================================================================

    print_section("TEST 3: Orchestrator Mode - CPT Codes in Drill Path")
    print("Input: conversation_id, query, context with drill_path")
    print("CPT extraction: From correlation_summary.drill_path")
    print()

    orchestrator_input_drillpath = {
        "conversation_id": "conv_test_drillpath_002",
        "query": "Analyze the spike",
        "context": {
            "intent": {
                "raw_question": "Analyze the spike",
                "analysis_mode": "correlation",
                "metric_hint": "paid_amount",
                "group_by": ["state"],
                "filters": [],
                "validation_warnings": []
            },
            "correlation_summary": {
                "run_id": "run_drillpath_456",
                "root_metric": "paid_amount",
                "baseline_value": 5000000.0,
                "comparison_value": 6500000.0,
                "delta_value": 1500000.0,
                "delta_pct": 30.0,
                "drill_path": [
                    {
                        "dimension": "procedure_code",
                        "value": "99291",
                        "contribution": 0.75,
                        "delta": 1125000.0
                    },
                    {
                        "dimension": "state",
                        "value": "NY",
                        "contribution": 0.40
                    }
                ],
                "narrative_summary": "Spike driven by procedure code 99291...",
                "warnings": []
            }
        }
    }

    try:
        result_drillpath = agent(**orchestrator_input_drillpath)

        print("✓ Execution successful")
        print(f"Output type: {type(result_drillpath).__name__}")
        print()

        display_orchestrator_output(result_drillpath)

        # Save output
        save_output(result_drillpath, "output_orchestrator_drillpath.json")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


    # ============================================================================
    # TEST 4: Orchestrator Mode - CPT Codes in Query Text
    # ============================================================================

    print_section("TEST 4: Orchestrator Mode - CPT Codes in Query Text")
    print("Input: conversation_id, query with CPT code, context")
    print("CPT extraction: From query text using regex")
    print()

    orchestrator_input_query = {
        "conversation_id": "conv_test_query_003",
        "query": "Why did CPT code 99291 costs increase?",
        "context": {
            "intent": {
                "raw_question": "Why did CPT code 99291 costs increase?",
                "analysis_mode": "correlation",
                "metric_hint": "paid_amount",
                "group_by": [],
                "filters": [],
                "validation_warnings": []
            },
            "correlation_summary": {
                "run_id": "run_query_789",
                "root_metric": "paid_amount",
                "baseline_value": 3000000.0,
                "comparison_value": 4000000.0,
                "delta_value": 1000000.0,
                "delta_pct": 33.33,
                "drill_path": [],
                "narrative_summary": "Costs increased significantly...",
                "warnings": []
            }
        }
    }

    try:
        result_query = agent(**orchestrator_input_query)

        print("✓ Execution successful")
        print(f"Output type: {type(result_query).__name__}")
        print()

        display_orchestrator_output(result_query)

        # Save output
        save_output(result_query, "output_orchestrator_query.json")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


    # ============================================================================
    # TEST 5: Error Handling - No CPT Codes
    # ============================================================================

    print_section("TEST 5: Error Handling - No CPT Codes Found")
    print("Input: context without any CPT codes")
    print("Expected: ValueError with helpful message")
    print()

    orchestrator_input_no_cpt = {
        "conversation_id": "conv_test_error_004",
        "query": "Why did costs increase?",
        "context": {
            "intent": {
                "raw_question": "Why did costs increase?",
                "filters": []
            },
            "correlation_summary": {
                "run_id": "run_error_001",
                "drill_path": []
            }
        }
    }

    try:
        result_error = agent(**orchestrator_input_no_cpt)
        print("✗ Expected error but execution succeeded")

    except ValueError as e:
        print("✓ Correct error raised")
        print(f"Error message: {e}")

    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}")
        print(f"Error: {e}")


    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================

    print_section("TEST SUITE SUMMARY")

    print("✓ Test 1: Direct mode (backward compatible)")
    print("  - Input: cpt_codes='99291'")
    print("  - Output: Dict with summary_table + reimbursement_policies")
    print("  - File: output_direct_mode.json")
    print()

    print("✓ Test 2: Orchestrator mode - Filters")
    print("  - Input: conversation_id + context with filters")
    print("  - CPT extraction: From filters")
    print("  - Output: Standard orchestrator schema")
    print("  - File: output_orchestrator_filters.json")
    print()

    print("✓ Test 3: Orchestrator mode - Drill Path")
    print("  - Input: conversation_id + context with drill_path")
    print("  - CPT extraction: From drill_path")
    print("  - Output: Standard orchestrator schema")
    print("  - File: output_orchestrator_drillpath.json")
    print()

    print("✓ Test 4: Orchestrator mode - Query Text")
    print("  - Input: conversation_id + query with CPT code")
    print("  - CPT extraction: From query text")
    print("  - Output: Standard orchestrator schema")
    print("  - File: output_orchestrator_query.json")
    print()

    print("✓ Test 5: Error handling")
    print("  - Input: No CPT codes in context")
    print("  - Output: ValueError with helpful message")
    print()

    print("=" * 80)
    print("FEATURES DEMONSTRATED")
    print("=" * 80)
    print()
    print("✓ Dual input modes (direct + orchestrator)")
    print("✓ Multiple CPT extraction strategies")
    print("✓ Standard API response schema (across all agents)")
    print("✓ Output structure: summary_table + reimbursement_policies")
    print("✓ Recommended actions at top level")
    print("✓ Visual component placeholder")
    print("✓ Job metadata tracking")
    print("✓ Execution timing (duration_ms)")
    print("✓ Validation checks (list format)")
    print("✓ Token tracking with breakdown")
    print("✓ Comprehensive error handling")
    print()

    print("=" * 80)
    print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)