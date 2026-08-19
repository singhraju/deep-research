"""
Enterprise-level agent utility functions for the DR ETL pipeline
"""
import json
import copy
import requests
import logging
from typing import Dict, Any, Tuple
from utils.agent_config import get_api_base_url

# Get the logger from the main pipeline
logger = logging.getLogger(__name__)

# API Configuration
#API_BASE_URL = "https://idiscovery-deep-research-api-uat.istio.carelon.com"
#API_BASE_URL = "http://127.0.0.1:8000"
# API_BASE_URL removed - using lazy load function instead
#API_BASE_URL = "http://127.0.0.1:5821"

def _get_api_url():
    """Lazy load API base URL from config"""
    return get_api_base_url()


def check_api_health(retry_interval=2, timeout=5):
    """
    Check if the API is healthy and ready to accept requests (infinite retry)
    
    Args:
        retry_interval (int): Seconds to wait between retries (default: 2)
        timeout (int): Timeout for each health check request in seconds (default: 5)
        
    Returns:
        bool: True if API is healthy (will retry indefinitely until success)
    """
    import time
    health_url = f"{_get_api_url()}/health"
    
    print(f"Checking API health at: {health_url}")
    
    attempt = 0
    while True:
        try:
            response = requests.get(health_url, verify=False, timeout=timeout)
            if response.status_code == 200:
                print(f"API is healthy (attempt {attempt + 1})")
                return True
            else:
                print(f"API returned status {response.status_code} (attempt {attempt + 1})")
        except requests.exceptions.Timeout:
            print(f"Health check timeout (attempt {attempt + 1})")
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {str(e)} (attempt {attempt + 1})")
        
        print(f"Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
        attempt += 1

def generate_agent_common_payload(trnd_tm_prd_end_mnth_nbr, trnd_tm_prd_cd, snap_year_mnth_nbr, statscl_mdl_cd, lob_shrt_desc):
    """
    Generate common payload for agents
    
    Args:
        trnd_tm_prd_end_mnth_nbr: End month number
        trnd_tm_prd_cd: Time period code
        snap_year_mnth_nbr: Snapshot year month
        statscl_mdl_cd: Statistical model code
        lob_shrt_desc: LOB short description
        
    Returns:
        Tuple[Dict, str]: Payload and conversation ID
    """
    from utils.time_utils import get_ecap_start_month, convert_current_ecap_time_to_previous_year
    
    current_time_period_end = int(trnd_tm_prd_end_mnth_nbr)
    current_time_period_start = get_ecap_start_month(trnd_tm_prd_cd, current_time_period_end)
    previous_period_start, previous_period_end = convert_current_ecap_time_to_previous_year(current_time_period_start, current_time_period_end)
    _id = f"tutorial-{statscl_mdl_cd}-{lob_shrt_desc}-{snap_year_mnth_nbr}-{trnd_tm_prd_cd}-{trnd_tm_prd_end_mnth_nbr}".replace(" ", "_").replace("/", "_").replace("\\", "_")
    _payload = {
        "conversation_id": _id,
        "context": {
            "analysis_mode_parameters": {
                "drill_metric": ["expense_detail.total_paid"],
                "period": {
                    "rolling_time_dimension": "expense_detail.incurred_month",
                    "current_period": {
                        "start_time": current_time_period_start,
                        "end_time": current_time_period_end
                    },
                    "previous_period": {
                        "start_time": previous_period_start,
                        "end_time": previous_period_end
                    }
                }
            },
            "filters": [
                {
                    "field": "snap_month",
                    "operator": "=",
                    "value": int(snap_year_mnth_nbr),
                    "source": "dimension_match"
                },
                {
                    "field": "lob_description",
                    "operator": "=",
                    "value": lob_shrt_desc,
                    "source": "dimension_match"
                }
            ]
        },
    }
    return _payload, _id


def generate_correlation_agent_common_payload(_payload, statscl_mdl_cd, env: str = None):
    """
    Add correlation-specific filters and semantic YAML path to payload
    
    Args:
        _payload: Base payload
        statscl_mdl_cd: Statistical model code
        env: Environment code (dv, ts, pl, pr) - optional
        
    Returns:
        Dict: Modified payload with correlation filters and yaml_path
    """
    if statscl_mdl_cd == "IP AUTH":
        _payload["context"]["filters"].append({
            "field": "hcc_high",
            "operator": "=",
            "value": "IP",
            "source": "dimension_match"
        })
    elif statscl_mdl_cd != "OON":
        _payload["context"]["filters"].append({
            "field": "hcc_medium",
            "operator": "=",
            "value": statscl_mdl_cd,
            "source": "dimension_match"
        })
    
    # Add semantic YAML config path
    if env:
        from utils.config_loader import get_config_loader
        config_loader = get_config_loader()
        semantic_path = config_loader.get_semantic_config_path(env, statscl_mdl_cd)
        _payload["yaml_path"] = semantic_path
    
    return _payload


def call_correlation_agents(_payload, anomaly_json):
    """
    Call correlation agents using the working logic
    
    Args:
        _payload: Correlation payload
        anomaly_json: Anomaly data from KEY_INSIGHT
        
    Returns:
        Dict: Correlation results for states, providers, and DRGs
    """
    correlation_results = {
        "states": {},
        "providers": {},
        "drgs": {},
        "procs": {}
    }
    agent_url = f"{_get_api_url()}/agents/correlation"
    
    # Process states
    if "top_contributors" in anomaly_json and "states" in anomaly_json["top_contributors"]:
        for state in anomaly_json["top_contributors"]["states"]:
            correlation_agent_payload = copy.deepcopy(_payload)
            # Verify yaml_path is preserved
            if "yaml_path" in _payload:
                logger.info(f"Correlation payload for state {state['name']} includes yaml_path: {correlation_agent_payload.get('yaml_path')}")
            correlation_agent_payload["query"] = f"Where did change happen for state {state['name']}? It {state['insight'].lower()} by {state['percentage_change']}"
            correlation_agent_payload["context"]["filters"].append({
                "field": "service_area_state",
                "operator": "=",
                "value": state["name"],
                "source": "dimension_match"
            })            
            try:
                if not check_api_health(retry_interval=2, timeout=5):
                    logger.error(f"Health check failed. Skipping.")
                    correlation_results["states"][state["name"]] = {"status": False, "error": "Health check failed"}
                    continue
                correlation_response = requests.post(agent_url, json=correlation_agent_payload, verify=False)
                correlation_result = correlation_response.json()
                correlation_results["states"][state["name"]] = correlation_result
            except Exception as e:
                logger.error(f"Error calling correlation agent for state {state['name']}: {str(e)}")
                correlation_results["states"][state["name"]] = {"status": False, "error": str(e)}
    
    # Process providers
    if "top_contributors" in anomaly_json and "provider_trends" in anomaly_json["top_contributors"]:
        for provider in anomaly_json["top_contributors"]["provider_trends"]:
            correlation_agent_payload = copy.deepcopy(_payload)
            correlation_agent_payload["query"] = f"Where did change happen for provider {provider['name']}? It {provider['insight'].lower()} by {provider['percentage_change']}"
            correlation_agent_payload["context"]["filters"].append({
                "field": "rendering_provider_name",
                "operator": "=",
                "value": provider["name"],
                "source": "dimension_match"
            })            
            try:
                if not check_api_health(retry_interval=2, timeout=5):
                    logger.error(f"Health check failed. Skipping.")
                    correlation_results["providers"][provider["name"]] = {"status": False, "error": "Health check failed"}
                    continue
                correlation_response = requests.post(agent_url, json=correlation_agent_payload, verify=False)
                correlation_result = correlation_response.json()
                correlation_results["providers"][provider["name"]] = correlation_result
            except Exception as e:
                logger.error(f"Error calling correlation agent for provider {provider['name']}: {str(e)}")
                correlation_results["providers"][provider["name"]] = {"status": False, "error": str(e)}

    # Process DRGs
    if "top_contributors" in anomaly_json and "drgs" in anomaly_json["top_contributors"]:
        for drg in anomaly_json["top_contributors"]["drgs"]:
            correlation_agent_payload = copy.deepcopy(_payload)
            correlation_agent_payload["query"] = f"Where did change happen for drg {drg['name']}? It {drg['insight'].lower()} by {drg['percentage_change']}"
            correlation_agent_payload["context"]["filters"].append({
                "field": "drg_name",
                "operator": "=",
                "value": drg["name"],
                "source": "dimension_match"
            })            
            try:
                if not check_api_health(retry_interval=2, timeout=5):
                    logger.error(f"Health check failed. Skipping.")
                    correlation_results["drgs"][drg["name"]] = {"status": False, "error": "Health check failed"}
                    continue
                correlation_response = requests.post(agent_url, json=correlation_agent_payload, verify=False)
                correlation_result = correlation_response.json()
                correlation_results["drgs"][drg["name"]] = correlation_result
            except Exception as e:
                logger.error(f"Error calling correlation agent for drg {drg['name']}: {str(e)}")
                correlation_results["drgs"][drg["name"]] = {"status": False, "error": str(e)}

    # Process Procedures
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
            try:
                if not check_api_health(retry_interval=2, timeout=5):
                    logger.error(f"Health check failed. Skipping.")
                    correlation_results["procs"][proc["name"]] = {"status": False, "error": "Health check failed"}
                    continue
                correlation_response = requests.post(agent_url, json=correlation_agent_payload, verify=False)
                correlation_result = correlation_response.json()
                correlation_results["procs"][proc["name"]] = correlation_result
            except Exception as e:
                logger.error(f"Error calling correlation agent for procedure {proc['name']}: {str(e)}")
                correlation_results["procs"][proc["name"]] = {"status": False, "error": str(e)}

    return correlation_results


def call_pattern_agents(_id, anomaly_json, deep_dive_json, correlation_results, semantic_config_path: str):
    """
    Call pattern agent using the working logic
    
    Args:
        _id: Conversation ID
        anomaly_json: Anomaly data
        deep_dive_json: Deep dive data
        correlation_results: Results from correlation agents
        semantic_config_path: Path to semantic YAML config file (required)
        
    Returns:
        Tuple: Pattern response and result
    """
    #API_BASE_URL = "http://idiscovery-deep-research-api.nginx.plat-dig-sharedservdigital2.awsdns.internal.das"
    _url = f"{_get_api_url()}/agents/pattern_agent"
    pattern_request = {
        "conversation_id": _id,
        "query": "Summarize the highest-impact authorization and provider mix themes from the completed correlation analysis.",
        "semantic_config_path": semantic_config_path,
        "context": {
            "anomaly_context": anomaly_json,
            "deep_dive_report": deep_dive_json,
            "correlation_results": correlation_results
        }
    }    

    #print(pattern_request)
    #with open("pattern_request_test.json","w") as f:
    #    json.dump(pattern_request,f)
    try:
        pattern_response = requests.post(_url, json=pattern_request, verify=False, timeout=(10, 900))
        pattern_result = pattern_response.json()
        return pattern_response, pattern_result
    except Exception as e:
        logger.error(f"Error calling pattern agent: {str(e)}")
        return None, {"status": False, "error": str(e)}

    
def call_reimbursement_agents(_id, pattern, allcards):
    payload = {
        "context": {
            "pattern": pattern,
            "cards": allcards  # ← CRITICAL for LOB/Product extraction!
        },
        "conversation_id": _id,
        "query": "Analyze reimbursement policies for pattern 5: California cesarean delivery",
        "job_id": f"{_id}_job"
    }  

    _url = f"{_get_api_url()}/agents/reimbursement_policy"
    try:
        reimbursement_response = requests.post(_url, json=payload, verify=False, timeout=(10, 1500))
        reimbursement_results = reimbursement_response.json()
        return reimbursement_response, reimbursement_results
    except Exception as e:
        logger.error(f"Error calling pattern agent: {str(e)}")
        return None, {"detail": {"status": False, "error": str(e)}, 'error_type': 'HTTPError', 'agent_name': 'reimbursement_policy'}


def call_policy_agent():
    return {"summary_table":{"columns":[{"id":"payer_org","label":"Payer Organization","type":"text"},{"id":"critical_care_denial","label":"Critical Care Denial\n(Rev 045X + Home Discharge)","type":"badge"},{"id":"appeals_process","label":"Appeals Process\n(Documented)","type":"badge"},{"id":"policy_effective_date","label":"Policy Effective Date\n(Last Updated)","type":"date"}],"rows":[{"appeals_process":"X","critical_care_denial":"X","payer_org":"Molina Medicaid","policy_effective_date":"10/30/2024"},{"appeals_process":"X","critical_care_denial":"X","payer_org":"Premera Blue Cross","policy_effective_date":"10/30/2024"},{"appeals_process":"X","critical_care_denial":"Bundles Critical Care","payer_org":"BCBS Nebraska","policy_effective_date":"01/01/2022"}],"subtitle":"Critical Care Denial Policies Across Major Payers","title":"Payer Policy Summary"}}


def generate_pattern_final_result(status, pattern, reimbursement):
    if status == 200:
        return {
            "rank": pattern['pattern_rank'],
            "pattern_title": pattern['top_pattern'],
            "pattern_description": pattern['pattern_details'],
            "explanation": {
                "reimbursement":{
                    "summary_table": reimbursement['output']['summary_table'],
                    "individual_policies": reimbursement['output']['reimbursement_policies']
                },
                "claims":{
                    "summary": " ".join(pattern["evidence_summary"])
                }
            }
        }
    else:
        return {
            "rank": pattern['pattern_rank'],
            "pattern_title": pattern['top_pattern'],
            "pattern_description": pattern['pattern_details'],
            "explanation": {
                "reimbursement":{
                    "summary_table": {},
                    "individual_policies": []
                },
                "claims":{
                    "summary": " ".join(pattern["evidence_summary"])
                }
            }
        }
    

def combine_pattern_reimbursement(pattern_data,reimbursement_data) -> Dict[str, Any]:
    """
    Combine pattern analysis results with reimbursement policy data.
   
    This function:
    1. Loads pattern results (from pattern agent)
    2. Loads reimbursement results (from reimbursement policy agent)
    3. Extracts cards and groups from pattern output
    4. Matches patterns by rank
    5. Fetches full card and group objects using source IDs
    6. Combines data for each pattern with full source objects
    7. Removes priority_entities field (not needed for recommendations)
    8. Returns combined data ready for recommendation agent
   
    Args:
        pattern_results_path: Path to pattern agent results JSON
        reimbursement_results_path: Path to reimbursement policy agent results JSON
        output_path: Optional path to save combined results
       
    Returns:
        Combined data dictionary with enriched patterns, including:
        - All pattern fields (pattern_rank, top_pattern, impact_summary, etc.)
        - source_card_ids: List of source card IDs from pattern analysis
        - source_group_ids: List of source group IDs from pattern analysis
        - source_cards: List of full card objects (fetched using source_card_ids)
        - source_groups: List of full group objects (fetched using source_group_ids)
        - reimbursement: Nested object with policy data and source traceability
       
        Note: priority_entities field is excluded from the output
       
    Example:
        >>> combined = combine_pattern_reimbursement(
        ...     "pattern_results.json",
        ...     "reimbursement_results.json"
        ... )
        >>> # Pass to recommendation agent
        >>> from deep_research_agents import RecommendationAgent
        >>> agent = RecommendationAgent()
        >>> recommendations = agent(input_data=combined)
    """
   
    # Extract patterns from wrapper if present
    if isinstance(pattern_data, dict) and "output" in pattern_data:
        patterns = pattern_data["output"].get("business_patterns", [])
        cards = pattern_data["output"].get("cards", [])
        groups = pattern_data["output"].get("groups", [])
        metadata = {
            "job_id": pattern_data.get("job_id"),
            "conversation_id": pattern_data.get("conversation_id"),
            "agent": pattern_data.get("agent"),
            "status": pattern_data.get("status")
        }
    else:
        patterns = pattern_data if isinstance(pattern_data, list) else []
        cards = []
        groups = []
        metadata = {}
   
    # Create lookup dictionaries for cards and groups by ID
    cards_by_id = {card.get("card_id"): card for card in cards if "card_id" in card}
    groups_by_id = {group.get("group_id"): group for group in groups if "group_id" in group}
   
    # Reimbursement data is a list
    reimbursement_patterns = reimbursement_data if isinstance(reimbursement_data, list) else []
   
    # Create lookup by pattern_rank
    reimbursement_by_rank = {
        item["pattern_rank"]: item
        for item in reimbursement_patterns
        if "pattern_rank" in item
    }
   
    print(f"\nFound {len(patterns)} patterns, {len(cards)} cards, {len(groups)} groups, and {len(reimbursement_by_rank)} reimbursement entries")
   
    # Combine data
    combined_patterns = []
   
    for pattern in patterns:
        pattern_rank = pattern.get("pattern_rank")
       
        # Start with pattern data (includes all fields including source_card_ids and source_group_ids)
        combined = dict(pattern)
       
        # Remove priority_entities field
        combined.pop("priority_entities", None)
       
        # Fetch source cards and groups using IDs
        source_card_ids = pattern.get("source_card_ids", [])
        source_group_ids = pattern.get("source_group_ids", [])
       
        # Fetch full card objects
        source_cards = [cards_by_id[card_id] for card_id in source_card_ids if card_id in cards_by_id]
       
        # Fetch full group objects
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
                "pattern_title": reimbursement.get("pattern_title"),
                "pattern_details": reimbursement.get("pattern_details"),
                "evidence_summary": reimbursement.get("evidence_summary", []),
                "individual_policies": reimbursement.get("individual_policies", []),
                "summary_table": reimbursement.get("summary_table", {}),
                "recommendation": reimbursement.get("recommendation"),
                # Include source traceability from reimbursement if available
                "source_card_ids": reimbursement.get("source_card_ids", []),
                "source_group_ids": reimbursement.get("source_group_ids", [])
            }
           
            print(f"  ✓ Pattern {pattern_rank}: Combined with reimbursement data ({len(source_cards)} cards, {len(source_groups)} groups)")
        else:
            print(f"  ⚠ Pattern {pattern_rank}: No reimbursement data found ({len(source_cards)} cards, {len(source_groups)} groups)")
            combined["reimbursement"] = None
       
        combined_patterns.append(combined)
   
    # Build final output structure
    result = {
        "metadata": metadata,
        "patterns_data": combined_patterns,
        "summary": {
            "total_patterns": len(combined_patterns),
            "patterns_with_reimbursement":len([p for p in combined_patterns if p.get("reimbursement")]),
            "patterns_without_reimbursement": len([p for p in combined_patterns if not p.get("reimbursement")])
        }
    }
   
    print(f"\n✓ Combined {result['summary']['total_patterns']} patterns")
    print(f"  - {result['summary']['patterns_with_reimbursement']} with reimbursement data")
    print(f"  - {result['summary']['patterns_without_reimbursement']} without reimbursement data")
   
    return result    
    

def call_recommendation_agents(recommendation_request):

    recommendation_agent_url = f"{_get_api_url()}/agents/recommendation_synthesis"
    #payload = combine_pattern_reimbursement(pattern_result,reim_result_output)
    #recommendation_request = payload
    print("Calling Reimbursment Agent...")
    try:
        recommendation_response = requests.post(recommendation_agent_url, json=recommendation_request, verify=False, timeout=(10,1800))
        recommendation_results = recommendation_response.json()        
        return recommendation_response, recommendation_results
    except Exception as e:
        logger.error(f"Error calling pattern agent: {str(e)}")
        return None, {"success":True,"result":{"metadata":{},"recommendations":[],"skipped_patterns":[],"processing_log":[]},"metadata":{"agent_name":"recommendation_synthesis"}}

    
