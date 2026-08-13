"""
LangGraph orchestration for the DR ETL pipeline.

This module expresses the per-combination agent flow of ``dr_etl_pipeline.py``
as an explicit LangGraph ``StateGraph``. One graph invocation processes one
unique combination (snap month / trend period / LOB / statistical model). The
node flow mirrors the imperative pipeline:

    prepare_payload -> correlation -> pattern -> [conditional]

If the pattern agent returns zero ``business_patterns`` the reimbursement and
recommendation stages are skipped (default structures are substituted), but the
policy agent still runs. If one or more patterns are returned, reimbursement
(once per pattern) and recommendation run first. Both branches converge on the
policy node (policy is pattern-independent and always runs), then an ``assemble``
node builds a per-combination ``result_record`` identical in shape to the one
produced by ``dr_etl_pipeline.py``.

All agent communication reuses the existing helpers in
``utils/agent_utils.py`` -- nothing is reimplemented here. The heavy data-layer
dependencies (pandas / snowflake) are imported lazily inside the driver so the
graph layer can be built and unit-tested with only ``langgraph`` installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

try:  # pragma: no cover - optional dependency, mirrors orchestrator.py
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - older LangGraph versions
    try:
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore
    except ImportError:  # pragma: no cover - allow import without the package
        InMemorySaver = None  # type: ignore

logger = logging.getLogger(__name__)

# Retry / health-check tuning kept identical to dr_etl_pipeline.py
MAX_RETRIES = 3
RETRY_DELAY = 5
HEALTH_RETRY_INTERVAL = 2
HEALTH_TIMEOUT = 5


# =============================
# Graph state
# =============================

class ETLCombinationState(TypedDict, total=False):
    """Per-combination state for a single graph invocation."""

    # Environment / injected context
    env: str
    s3_logger: Optional[Any]

    # Combination identity (inputs)
    snap_year_mnth_nbr: Any
    trnd_tm_prd_end_mnth_nbr: Any
    trnd_tm_prd_cd: str
    lob_shrt_desc: str
    lob_cd: Any
    statscl_mdl_cd: str

    # Parsed source documents (provided by the driver)
    anomaly_json: Dict[str, Any]
    deep_dive_json: Dict[str, Any]
    insights_lob_code: Any

    # Derived payload / ids
    conversation_id: str
    semantic_config_path: str
    final_payload: Dict[str, Any]

    # Agent outputs
    correlation_results: Dict[str, Any]
    pattern_result: Dict[str, Any]
    processed_pattern_result: List[Dict[str, Any]]
    reim_result_output: List[Dict[str, Any]]
    recommendation_results: Dict[str, Any]
    policy_results: Dict[str, Any]

    # Assembled result + control
    result_record: Dict[str, Any]
    status: str
    route: str


# Default recommendation structure used when there are no business patterns.
def _default_recommendation() -> Dict[str, Any]:
    return {
        "success": True,
        "output": {
            "metadata": {},
            "recommendations": [],
            "skipped_patterns": [],
            "processing_log": ["No business patterns found - skipped processing"],
        },
        "metadata": {"agent_name": "recommendation_synthesis"},
    }


# =============================
# Dependency resolution
# =============================

class AgentHelpers:
    """Bundle of the agent-communication helpers used by the graph nodes.

    Defaults are lazily imported from ``utils.agent_utils`` (and
    ``utils.config_loader``) so importing this module does not require the
    request/network stack. Tests inject stubs to exercise routing without any
    real agent calls.
    """

    def __init__(
        self,
        generate_agent_common_payload: Callable[..., Tuple[Dict[str, Any], str]],
        generate_correlation_agent_common_payload: Callable[..., Dict[str, Any]],
        call_correlation_agents: Callable[..., Dict[str, Any]],
        call_pattern_agents: Callable[..., Tuple[Any, Dict[str, Any]]],
        call_reimbursement_agents: Callable[..., Tuple[Any, Dict[str, Any]]],
        combine_pattern_reimbursement: Callable[..., Dict[str, Any]],
        call_recommendation_agents: Callable[..., Tuple[Any, Dict[str, Any]]],
        call_policy_agent: Callable[..., Dict[str, Any]],
        generate_pattern_final_result: Callable[..., Dict[str, Any]],
        check_api_health: Callable[..., bool],
        get_semantic_config_path: Callable[[str, str], str],
    ) -> None:
        self.generate_agent_common_payload = generate_agent_common_payload
        self.generate_correlation_agent_common_payload = generate_correlation_agent_common_payload
        self.call_correlation_agents = call_correlation_agents
        self.call_pattern_agents = call_pattern_agents
        self.call_reimbursement_agents = call_reimbursement_agents
        self.combine_pattern_reimbursement = combine_pattern_reimbursement
        self.call_recommendation_agents = call_recommendation_agents
        self.call_policy_agent = call_policy_agent
        self.generate_pattern_final_result = generate_pattern_final_result
        self.check_api_health = check_api_health
        self.get_semantic_config_path = get_semantic_config_path


def _default_agent_helpers() -> AgentHelpers:
    """Build an :class:`AgentHelpers` backed by the real pipeline utilities."""
    from utils.agent_utils import (
        generate_agent_common_payload,
        generate_correlation_agent_common_payload,
        call_correlation_agents,
        call_pattern_agents,
        call_reimbursement_agents,
        combine_pattern_reimbursement,
        call_recommendation_agents,
        call_policy_agent,
        generate_pattern_final_result,
        check_api_health,
    )
    from utils.config_loader import get_config_loader

    config_loader = get_config_loader()

    return AgentHelpers(
        generate_agent_common_payload=generate_agent_common_payload,
        generate_correlation_agent_common_payload=generate_correlation_agent_common_payload,
        call_correlation_agents=call_correlation_agents,
        call_pattern_agents=call_pattern_agents,
        call_reimbursement_agents=call_reimbursement_agents,
        combine_pattern_reimbursement=combine_pattern_reimbursement,
        call_recommendation_agents=call_recommendation_agents,
        call_policy_agent=call_policy_agent,
        generate_pattern_final_result=generate_pattern_final_result,
        check_api_health=check_api_health,
        get_semantic_config_path=lambda env, mdl: config_loader.get_semantic_config_path(env, mdl),
    )


def _save_to_s3(s3_logger: Optional[Any], data: Any, filename: str, model: str, subfolder: str) -> None:
    """No-op S3 save wrapper -- skips silently when no s3_logger is present."""
    if s3_logger is None:
        return
    try:
        s3_logger.save_to_s3(data, filename, model, subfolder)
    except Exception as exc:  # pragma: no cover - logging side-effect only
        logger.warning("S3 save failed for %s: %s", filename, exc)


def _stub_agent_helpers() -> AgentHelpers:
    """No-op :class:`AgentHelpers` for building the graph without the network
    stack (visualization only -- graph topology is independent of the agents)."""
    noop = lambda *a, **k: {}
    return AgentHelpers(
        generate_agent_common_payload=lambda *a, **k: ({}, "id"),
        generate_correlation_agent_common_payload=lambda *a, **k: {},
        call_correlation_agents=noop,
        call_pattern_agents=lambda *a, **k: (None, {}),
        call_reimbursement_agents=lambda *a, **k: (None, {}),
        combine_pattern_reimbursement=noop,
        call_recommendation_agents=lambda *a, **k: (None, {}),
        call_policy_agent=noop,
        generate_pattern_final_result=noop,
        check_api_health=lambda **k: True,
        get_semantic_config_path=lambda env, mdl: "",
    )


# =============================
# Graph construction
# =============================

def build_etl_graph(
    agents: Optional[AgentHelpers] = None,
    checkpointer: Optional[Any] = None,
):
    """Build and compile the per-combination ETL LangGraph app.

    Args:
        agents: Bundle of agent-communication helpers. Defaults to the real
            helpers from ``utils.agent_utils`` (imported lazily). Inject stubs
            to unit-test routing.
        checkpointer: Optional LangGraph checkpointer. When ``None`` an
            ``InMemorySaver`` is used if available (mirrors orchestrator.py).

    Returns:
        The compiled LangGraph application.
    """
    if agents is None:
        agents = _default_agent_helpers()

    # ---- Nodes -------------------------------------------------------------

    def prepare_payload(state: ETLCombinationState) -> Dict[str, Any]:
        env = state["env"]
        statscl_mdl_cd = state["statscl_mdl_cd"]
        s3_logger = state.get("s3_logger")

        _payload, _id = agents.generate_agent_common_payload(
            state["trnd_tm_prd_end_mnth_nbr"],
            state["trnd_tm_prd_cd"],
            state["snap_year_mnth_nbr"],
            statscl_mdl_cd,
            state["lob_shrt_desc"],
        )
        final_payload = agents.generate_correlation_agent_common_payload(_payload, statscl_mdl_cd, env)

        if isinstance(final_payload, dict) and "yaml_path" in final_payload:
            logger.info("Using semantic config for correlation: %s", final_payload["yaml_path"])

        semantic_config_path = agents.get_semantic_config_path(env, statscl_mdl_cd)

        _save_to_s3(s3_logger, final_payload, f"correlation_payload_{_id}.json", statscl_mdl_cd, "payload")
        _save_to_s3(s3_logger, state.get("anomaly_json"), f"correlation_payload_anomaly_{_id}.json", statscl_mdl_cd, "payload")

        return {
            "conversation_id": _id,
            "final_payload": final_payload,
            "semantic_config_path": semantic_config_path,
        }

    def correlation(state: ETLCombinationState) -> Dict[str, Any]:
        statscl_mdl_cd = state["statscl_mdl_cd"]
        s3_logger = state.get("s3_logger")
        _id = state["conversation_id"]

        logger.info("Health check before correlation agents")
        if not agents.check_api_health(retry_interval=HEALTH_RETRY_INTERVAL, timeout=HEALTH_TIMEOUT):
            logger.warning("API health check failed before correlation agents. Skipping combination.")
            return {"status": "skipped", "route": "skip"}

        correlation_results = agents.call_correlation_agents(state["final_payload"], state.get("anomaly_json"))
        logger.info(
            "Correlation complete. states=%s providers=%s drgs=%s",
            len(correlation_results.get("states", {})),
            len(correlation_results.get("providers", {})),
            len(correlation_results.get("drgs", {})),
        )
        _save_to_s3(s3_logger, correlation_results, f"correlation_response_{_id}.json", statscl_mdl_cd, "agents_results")
        return {"correlation_results": correlation_results, "route": "continue"}

    def pattern(state: ETLCombinationState) -> Dict[str, Any]:
        statscl_mdl_cd = state["statscl_mdl_cd"]
        s3_logger = state.get("s3_logger")
        _id = state["conversation_id"]

        logger.info("Health check before pattern agent")
        if not agents.check_api_health(retry_interval=HEALTH_RETRY_INTERVAL, timeout=HEALTH_TIMEOUT):
            logger.warning("API health check failed before pattern agent. Skipping combination.")
            return {"status": "skipped", "route": "skip"}

        semantic_config_path = state["semantic_config_path"]
        pattern_payload = {
            "conversation_id": _id,
            "query": "Summarize the highest-impact authorization and provider mix themes from the completed correlation analysis.",
            "semantic_config_path": semantic_config_path,
            "context": {
                "anomaly_context": state.get("anomaly_json"),
                "deep_dive_report": state.get("deep_dive_json"),
                "correlation_results": state.get("correlation_results"),
            },
        }
        _save_to_s3(s3_logger, pattern_payload, f"pattern_payload_{_id}.json", statscl_mdl_cd, "payload")

        # Pattern-agent retry loop (identical tuning to dr_etl_pipeline.py)
        pattern_result: Dict[str, Any] = {}
        for attempt in range(MAX_RETRIES):
            pattern_response, pattern_result = agents.call_pattern_agents(
                _id,
                state.get("anomaly_json"),
                state.get("deep_dive_json"),
                state.get("correlation_results"),
                semantic_config_path,
            )
            if pattern_response:
                if getattr(pattern_response, "status_code", None) == 200:
                    if pattern_result.get("status") == "success":
                        break
                    logger.warning(
                        "Attempt %s: Pattern Agent failed - %s",
                        attempt + 1,
                        pattern_result.get("explanation", {}).get("error"),
                    )
                else:
                    logger.warning("Attempt %s: Pattern Agent failed - HTTP Error Network Error", attempt + 1)
            else:
                logger.warning("Pattern Agent Called Failed")
            time.sleep(RETRY_DELAY)

        business_patterns = []
        if pattern_result and "output" in pattern_result:
            business_patterns = pattern_result["output"].get("business_patterns", [])
        logger.info("Pattern agent completed. Business patterns found: %s", len(business_patterns))

        _save_to_s3(s3_logger, pattern_result, f"pattern_response_{_id}.json", statscl_mdl_cd, "agents_results")

        route = "no_pattern" if len(business_patterns) == 0 else "has_pattern"
        return {"pattern_result": pattern_result, "route": route}

    def skip_patterns(state: ETLCombinationState) -> Dict[str, Any]:
        """No business patterns: substitute default structures (policy still runs)."""
        logger.info("No business patterns found. Using default empty structures for results.")
        return {
            "processed_pattern_result": [],
            "reim_result_output": [],
            "recommendation_results": _default_recommendation(),
        }

    def reimbursement(state: ETLCombinationState) -> Dict[str, Any]:
        statscl_mdl_cd = state["statscl_mdl_cd"]
        s3_logger = state.get("s3_logger")
        pattern_result = state["pattern_result"]
        business_patterns = pattern_result["output"]["business_patterns"]
        all_cards = pattern_result["output"].get("cards", [])
        logger.info("Processing %s business patterns with %s cards", len(business_patterns), len(all_cards))

        processed_pattern_result: List[Dict[str, Any]] = []
        full_reim_result: List[Dict[str, Any]] = []

        for idx, pr in enumerate(business_patterns):
            logger.info("Processing pattern %s/%s: Rank %s", idx + 1, len(business_patterns), pr["pattern_rank"])
            _rid = f"{pattern_result['conversation_id']}_{pr['pattern_rank']}"
            reimbursement_response = None
            if not agents.check_api_health(retry_interval=HEALTH_RETRY_INTERVAL, timeout=HEALTH_TIMEOUT):
                logger.warning("Health check failed before reimbursement agents. Using default values.")
                s_code = 400
                reimbursement_results = {"detail": {"status": False, "error": "Health check failed"}}
            else:
                reim_payload = {
                    "context": {"pattern": pr, "cards": all_cards},
                    "conversation_id": _rid,
                    "query": f"Analyze reimbursement policies for pattern {pr['pattern_rank']}",
                    "job_id": f"{_rid}_job",
                }
                _save_to_s3(s3_logger, reim_payload, f"reimbursement_payload_{_rid}.json", statscl_mdl_cd, "payload")

                reimbursement_response, reimbursement_results = agents.call_reimbursement_agents(_rid, pr, all_cards)
                _save_to_s3(s3_logger, reimbursement_results, f"reimbursement_response_{_rid}.json", statscl_mdl_cd, "agents_results")

            s_code = reimbursement_response.status_code if reimbursement_response else 400
            full_reim_result.append(reimbursement_results)
            processed_pattern_result.append(agents.generate_pattern_final_result(s_code, pr, reimbursement_results))

        reim_result_output: List[Dict[str, Any]] = []
        for _index in range(len(full_reim_result)):
            if "detail" in full_reim_result[_index]:
                reim_result_output.append({
                    "pattern_rank": _index + 1,
                    "summary_table": {},
                    "reimbursement_policies": [],
                    "elevance_executive_summary": None,
                    "policies_processed": 0,
                    "policies_successful": 0,
                    "policies_failed": 0,
                })
            else:
                reim_result_output.append(full_reim_result[_index]["output"])

        _save_to_s3(s3_logger, processed_pattern_result, f"pattern_final_{state['conversation_id']}.json", statscl_mdl_cd, "agents_results")

        return {
            "processed_pattern_result": processed_pattern_result,
            "reim_result_output": reim_result_output,
        }

    def recommendation(state: ETLCombinationState) -> Dict[str, Any]:
        statscl_mdl_cd = state["statscl_mdl_cd"]
        s3_logger = state.get("s3_logger")
        _id = state["conversation_id"]

        logger.info("Health check before recommendation agents")
        if not agents.check_api_health(retry_interval=HEALTH_RETRY_INTERVAL, timeout=HEALTH_TIMEOUT):
            logger.warning("API health check failed before recommendation agents. Using default values.")
            return {
                "recommendation_results": {
                    "success": True,
                    "output": {"metadata": {}, "recommendations": [], "skipped_patterns": [], "processing_log": []},
                    "metadata": {"agent_name": "recommendation_synthesis"},
                }
            }

        rec_payload = agents.combine_pattern_reimbursement(state["pattern_result"], state["reim_result_output"])
        _save_to_s3(s3_logger, rec_payload, f"recommendation_payload_{_id}.json", statscl_mdl_cd, "payload")

        _, recommendation_results = agents.call_recommendation_agents(rec_payload)
        _save_to_s3(s3_logger, recommendation_results, f"recommendation_response_{_id}.json", statscl_mdl_cd, "agents_results")
        return {"recommendation_results": recommendation_results}

    def policy(state: ETLCombinationState) -> Dict[str, Any]:
        """Policy agent is pattern-independent and always runs on both branches."""
        logger.info("Health check before policy agent")
        if not agents.check_api_health(retry_interval=HEALTH_RETRY_INTERVAL, timeout=HEALTH_TIMEOUT):
            logger.warning("API health check failed before policy agent. Using default values.")
            return {"policy_results": {}}
        policy_results = agents.call_policy_agent()
        logger.info("Policy agent completed")
        return {"policy_results": policy_results}

    def assemble(state: ETLCombinationState) -> Dict[str, Any]:
        recommendation_results = state.get("recommendation_results") or _default_recommendation()
        result_record = {
            "snap_year_mnth_nbr": int(state["snap_year_mnth_nbr"]),
            "trnd_tm_prd_end_mnth_nbr": int(state["trnd_tm_prd_end_mnth_nbr"]),
            "trnd_tm_prd_cd": state["trnd_tm_prd_cd"],
            "lob_shrt_desc": state["lob_shrt_desc"],
            "lob_cd": state.get("insights_lob_code"),
            "statscl_mdl_cd": state["statscl_mdl_cd"],
            "final_payload": state.get("final_payload"),
            "anomaly_json": state.get("anomaly_json"),
            "deep_dive_json": state.get("deep_dive_json"),
            "pattern_result": state.get("processed_pattern_result", []),
            "recommendation_result": recommendation_results["output"],
            "policy_result": state.get("policy_results", {}),
            "status": "success",
            "processed_at": datetime.now().isoformat(),
        }
        return {"result_record": result_record, "status": "success"}

    # ---- Routing -----------------------------------------------------------

    def route_after_correlation(state: ETLCombinationState) -> str:
        return state.get("route", "continue")

    def route_after_pattern(state: ETLCombinationState) -> str:
        return state.get("route", "no_pattern")

    # ---- Wiring ------------------------------------------------------------

    graph = StateGraph(ETLCombinationState)
    graph.add_node("prepare_payload", prepare_payload)
    graph.add_node("correlation", correlation)
    graph.add_node("pattern", pattern)
    graph.add_node("skip_patterns", skip_patterns)
    graph.add_node("reimbursement", reimbursement)
    graph.add_node("recommendation", recommendation)
    graph.add_node("policy", policy)
    graph.add_node("assemble", assemble)

    graph.add_edge(START, "prepare_payload")
    graph.add_edge("prepare_payload", "correlation")

    # Health-check short-circuit before correlation -> END with status "skipped"
    graph.add_conditional_edges(
        "correlation",
        route_after_correlation,
        {"skip": END, "continue": "pattern"},
    )

    # After the pattern agent: skip to END on health failure, otherwise fan to
    # the no-pattern branch (defaults) or the reimbursement branch.
    graph.add_conditional_edges(
        "pattern",
        route_after_pattern,
        {"skip": END, "no_pattern": "skip_patterns", "has_pattern": "reimbursement"},
    )

    # Both branches converge on the (always-run) policy node.
    graph.add_edge("skip_patterns", "policy")
    graph.add_edge("reimbursement", "recommendation")
    graph.add_edge("recommendation", "policy")
    graph.add_edge("policy", "assemble")
    graph.add_edge("assemble", END)

    if checkpointer is None and InMemorySaver is not None:
        checkpointer = InMemorySaver()

    app = graph.compile(checkpointer=checkpointer)
    return app


# =============================
# Visualization
# =============================

def render_graph(png_path: Optional[str] = None) -> str:
    """Render the execution graph without any network / Snowflake dependency.

    Builds the graph with no-op agents (topology is agent-independent), prints an
    ASCII rendering when possible plus the Mermaid source, and optionally writes a
    PNG. Returns the Mermaid source string.

    Args:
        png_path: If given, also write a PNG to this path (needs internet -- the
            Mermaid PNG renderer is a remote service).
    """
    app = build_etl_graph(agents=_stub_agent_helpers())
    graph = app.get_graph()

    # ASCII (best-effort -- requires the optional ``grandalf`` package)
    try:
        print(graph.draw_ascii())
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[ASCII rendering unavailable: {exc}. Install 'grandalf' for ASCII output.]")

    mermaid = graph.draw_mermaid()
    print("\n--- Mermaid source (paste into any Mermaid renderer) ---\n")
    print(mermaid)

    if png_path:
        try:
            graph.draw_mermaid_png(output_file_path=png_path)
            print(f"\nSaved PNG to: {png_path}")
        except Exception as exc:  # pragma: no cover - needs network
            print(f"\n[PNG rendering failed: {exc}. draw_mermaid_png() needs internet access.]")

    return mermaid


# =============================
# Driver
# =============================

def _iter_unique_combinations(df_insights) -> List[Dict[str, Any]]:
    """Enumerate unique combinations exactly as dr_etl_pipeline.py does."""
    all_combinations: List[Dict[str, Any]] = []
    for snap_year_mnth_nbr in df_insights.SNAP_YEAR_MNTH_NBR.unique():
        for trnd_tm_prd_end_mnth_nbr in df_insights.TRND_TM_PRD_END_MNTH_NBR.unique():
            for trnd_tm_prd_cd in df_insights.TRND_TM_PRD_CD.unique():
                for lob_shrt_desc in df_insights.LOB_SHRT_DESC.unique():
                    for statscl_mdl_cd_val in df_insights.STATSCL_MDL_CD.unique():
                        subset = df_insights[df_insights.STATSCL_MDL_CD == statscl_mdl_cd_val]
                        all_combinations.append({
                            "snap_year_mnth_nbr": snap_year_mnth_nbr,
                            "trnd_tm_prd_end_mnth_nbr": trnd_tm_prd_end_mnth_nbr,
                            "trnd_tm_prd_cd": trnd_tm_prd_cd,
                            "lob_cd": subset.LOB_CD.iloc[0] if not subset.empty else "",
                            "lob_shrt_desc": lob_shrt_desc,
                            "statscl_mdl_cd": statscl_mdl_cd_val,
                        })
    return all_combinations


def run_pipeline_graph(
    env: str,
    use_snowflake: bool = True,
    csv_file_path: str = "SampleData-dev-deep-research.csv",
    lob: str = "gbd",
    statscl_mdl_cd: str = "IP AUTH",
    snap_year_mnth_nbr: Optional[int] = None,
    trnd_tm_prd_cd: Optional[str] = None,
    lob_shrt_desc: Optional[str] = None,
):
    """Thin driver: load data, run the graph once per combination, store results.

    Data loading and Snowflake storage reuse the existing pipeline utilities;
    only the per-combination agent flow is expressed as the LangGraph app.
    """
    import pandas as pd  # noqa: F401  (kept for parity / debugging)

    from utils.agent_config import initialize_agent_config
    from utils.config_loader import get_config_loader
    from utils.s3_utils import S3Logger
    import dr_etl_pipeline

    print(f"Starting DR ETL LangGraph pipeline for environment: {env}")

    # Initialize agent API configuration from config.yaml
    initialize_agent_config(env)

    config_loader = get_config_loader()
    env_config = config_loader.get_env_config(env)
    print(f"Environment config: {env_config}")

    # ---- Load data ----
    if use_snowflake:
        from utils.snowflake_utils import fetch_snowflake_data, create_snowpark_config

        session = create_snowpark_config(env)
        print("Snowpark session created successfully")
        df_insights = fetch_snowflake_data(
            session,
            env=env,
            statscl_mdl_cd=statscl_mdl_cd,
            lob=lob,
            snap_year_mnth_nbr=snap_year_mnth_nbr,
            trnd_tm_prd_cd=trnd_tm_prd_cd,
            lob_shrt_desc=lob_shrt_desc,
        )
        if df_insights.empty:
            print("No data found in Snowflake table")
            return
    else:
        import pandas as pd
        df_insights = pd.read_csv(csv_file_path)
        df_insights = df_insights[[
            "SNAP_YEAR_MNTH_NBR", "TRND_TM_PRD_END_MNTH_NBR", "TRND_TM_PRD_CD",
            "LOB_CD", "LOB_SHRT_DESC", "STATSCL_MDL_CD", "INSGHT_TYPE_NM", "JSON_TXT",
        ]]
        session = None
        print(f"Loaded data from CSV. Shape: {df_insights.shape}")

    print(f"Data loaded successfully. Shape: {df_insights.shape}")

    # Initialize S3 logging (optional -- graph nodes no-op when absent)
    s3_logger = S3Logger(env)
    print(f"[LOG] S3 Logger initialized with timestamp: {s3_logger.timestamp_folder}")

    # Handle run-level truncate once (mirrors dr_etl_pipeline.py)
    if use_snowflake:
        should_truncate = config_loader.should_truncate_table(env)
        target_table = config_loader.get_target_table(env, lob)
        print(f"[LOG] Target table from config: {target_table}")
        if should_truncate:
            print(f"\n⚠️  TRUNCATE enabled - clearing table: {target_table}")
            try:
                session.sql(f"TRUNCATE TABLE IF EXISTS {target_table}").collect()
                print("✅ Table truncated successfully (run-level, one-time operation)\n")
            except Exception as exc:
                print(f"⚠️  Warning: Could not truncate table: {str(exc)}\n")
        else:
            print("\n✅ TRUNCATE disabled - data will be appended\n")

    # Build the compiled graph once and reuse it per combination.
    app = build_etl_graph()

    all_combinations = _iter_unique_combinations(df_insights)
    print(f"📋 Enumerated {len(all_combinations)} combinations")

    results: List[Dict[str, Any]] = []

    for idx, combination in enumerate(all_combinations, 1):
        c_snap = combination["snap_year_mnth_nbr"]
        c_end = combination["trnd_tm_prd_end_mnth_nbr"]
        c_trnd = combination["trnd_tm_prd_cd"]
        c_lob = combination["lob_shrt_desc"]
        c_mdl = combination["statscl_mdl_cd"]

        job_id = dr_etl_pipeline.generate_job_id(c_snap, c_end, c_trnd, c_lob, c_mdl)
        print(f"\n{'='*60}")
        print(f"Processing Job {idx}/{len(all_combinations)}: {job_id}")
        print(f"{'='*60}")

        try:
            # Data split (KEY_INSIGHT + DEEP_DIVE) identical to dr_etl_pipeline.py
            first_anomaly = df_insights[
                (df_insights.SNAP_YEAR_MNTH_NBR == c_snap)
                & (df_insights.TRND_TM_PRD_END_MNTH_NBR == c_end)
                & (df_insights.TRND_TM_PRD_CD == c_trnd)
                & (df_insights.LOB_SHRT_DESC == c_lob)
                & (df_insights.STATSCL_MDL_CD == c_mdl)
                & (df_insights.INSGHT_TYPE_NM == "KEY_INSIGHT")
            ]
            if first_anomaly.empty:
                print("Warning: No KEY_INSIGHT record found for this combination")
                continue

            first_deep_dive = df_insights[
                (df_insights.SNAP_YEAR_MNTH_NBR == c_snap)
                & (df_insights.TRND_TM_PRD_END_MNTH_NBR == c_end)
                & (df_insights.TRND_TM_PRD_CD == c_trnd)
                & (df_insights.LOB_SHRT_DESC == c_lob)
                & (df_insights.STATSCL_MDL_CD == c_mdl)
                & (df_insights.INSGHT_TYPE_NM == "DEEP_DIVE")
            ]
            if first_deep_dive.empty:
                print("Warning: No DEEP_DIVE record found for this combination")
                continue

            try:
                anomaly_json = json.loads(json.loads(first_anomaly.JSON_TXT.iloc[0]))
                deep_dive_json = json.loads(json.loads(first_deep_dive.JSON_TXT.iloc[0]))
                insights_lob_code = first_deep_dive.LOB_CD.iloc[0]
            except Exception as exc:
                print(f"Error parsing JSON data: {str(exc)}")
                continue

            initial_state: ETLCombinationState = {
                "env": env,
                "s3_logger": s3_logger,
                "snap_year_mnth_nbr": c_snap,
                "trnd_tm_prd_end_mnth_nbr": c_end,
                "trnd_tm_prd_cd": c_trnd,
                "lob_shrt_desc": c_lob,
                "lob_cd": combination["lob_cd"],
                "statscl_mdl_cd": c_mdl,
                "anomaly_json": anomaly_json,
                "deep_dive_json": deep_dive_json,
                "insights_lob_code": insights_lob_code,
            }

            final_state = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": job_id}},
            )

            result_record = final_state.get("result_record")
            if result_record and result_record.get("status") == "success":
                results.append(result_record)

                # Persist per-combination result to a local JSON file (parity)
                combination_filename = (
                    f"combination_{c_snap}_{c_end}_{c_trnd}_{c_lob}_{c_mdl}.json"
                ).replace(" ", "_").replace("/", "_").replace("\\", "_")
                try:
                    with open(combination_filename, "w") as f:
                        json.dump(result_record, f, indent=2)
                    print(f"Saved individual result: {combination_filename}")
                except Exception as exc:
                    print(f"Warning: could not save {combination_filename}: {str(exc)}")

                # Store to Snowflake immediately (reuses dr_etl_pipeline helper)
                if use_snowflake:
                    try:
                        dr_etl_pipeline.store_agent_results_to_snowflake(
                            session, env, env_config, df_insights, [result_record], lob, s3_logger
                        )
                        print("Stored combination to Snowflake successfully")
                    except Exception as exc:
                        print(f"Error storing combination to Snowflake: {str(exc)}")
                print(f"✅ Job {job_id} completed successfully")
            else:
                status = final_state.get("status", "unknown")
                print(f"⏭️  Job {job_id} finished with status '{status}' (no result stored)")

        except Exception as combo_error:
            print(f"❌ Error processing combination {job_id}: {str(combo_error)}")
            continue

    print(f"\nAPI Processing completed! Total records processed: {len(results)}")
    successful_results = [r for r in results if r.get("status") == "success"]
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total combinations processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")

    if use_snowflake and session is not None:
        session.close()
        print("✅ Main Snowflake session closed")

    return results


def main(
    env: str,
    use_snowflake: bool = True,
    csv_file_path: str = "SampleData-dev-deep-research.csv",
    lob: str = "gbd",
    statscl_mdl_cd: str = "IP AUTH",
    snap_year_mnth_nbr: Optional[int] = None,
    trnd_tm_prd_cd: Optional[str] = None,
    lob_shrt_desc: Optional[str] = None,
):
    """CLI entry point mirroring dr_etl_pipeline.main()."""
    return run_pipeline_graph(
        env=env,
        use_snowflake=use_snowflake,
        csv_file_path=csv_file_path,
        lob=lob,
        statscl_mdl_cd=statscl_mdl_cd,
        snap_year_mnth_nbr=snap_year_mnth_nbr,
        trnd_tm_prd_cd=trnd_tm_prd_cd,
        lob_shrt_desc=lob_shrt_desc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Pattern Analysis Pipeline (LangGraph orchestration)")
    parser.add_argument("--env", required=False, choices=["dv", "ts", "pl", "pr"],
                        help="Target environment (dv, ts, pl, pr). Required unless --show-graph")
    parser.add_argument("--csv-file", default="SampleData-dev-deep-research.csv",
                        help="Path to the CSV file to process (used only with --use-csv)")
    parser.add_argument("--use-csv", action="store_true",
                        help="Use CSV file instead of Snowflake")
    parser.add_argument("--statscl-mdl-cd", default="IP AUTH",
                        help="STATSCL_MDL_CD filter value (default: IP AUTH)")
    parser.add_argument("--insght-type-nm", default="DEEP_DIVE",
                        help="INSGHT_TYPE_NM filter value (default: DEEP_DIVE)")
    parser.add_argument("--lob", choices=["gbd", "nogbd"], default="gbd",
                        help="LOB type - gbd or nogbd (default: gbd)")
    parser.add_argument("--snap-year-mnth-nbr", type=int, default=None,
                        help="SNAP_YEAR_MNTH_NBR filter value (optional, e.g., 202604)")
    parser.add_argument("--trnd-tm-prd-cd", type=str, default=None,
                        help="TRND_TM_PRD_CD filter value (optional, e.g., R3, R6, R12)")
    parser.add_argument("--lob-shrt-desc", type=str, default=None,
                        help="LOB_SHRT_DESC filter value (optional, e.g., Commercial_Individual)")
    parser.add_argument("--show-graph", action="store_true",
                        help="Print the execution graph (ASCII + Mermaid) and exit; no data/agents needed")
    parser.add_argument("--png", type=str, default=None,
                        help="With --show-graph, also write a PNG to this path (needs internet)")

    args = parser.parse_args()

    # Visualization short-circuit: draw the graph and exit before touching data.
    if args.show_graph:
        render_graph(png_path=args.png)
        sys.exit(0)

    if not args.env:
        parser.error("--env is required (unless using --show-graph)")

    try:
        use_snowflake = not args.use_csv
        main(args.env, use_snowflake, args.csv_file, args.lob, args.statscl_mdl_cd,
             args.snap_year_mnth_nbr, args.trnd_tm_prd_cd, args.lob_shrt_desc)
    except Exception as exc:
        import traceback
        error_msg = f"Error in Pattern Analysis Pipeline (LangGraph): {str(exc)}"
        print(error_msg)
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        sys.exit(1)
