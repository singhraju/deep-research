"""
Reimbursement Policy Extraction Agent

This agent searches for and extracts structured adjudication rules from insurance
reimbursement policies for specific CPT/HCPCS codes. It integrates with the Carelon
policy comparison API and Snowflake policy database to retrieve and analyze policies.

Usage:
    >>> from deep_research_agents import ReimbursementAgent
    >>> 
    >>> agent = ReimbursementPolicyAgent(
    ...     snowflake_helper=snowpark,  # Optional, will auto-build if not provided
    ... )
    >>> 
    >>> results = agent(cpt_codes="99291,99292")
    >>> 
    >>> for policy_result in results:
    ...     if policy_result:
    ...         print(f"Policy {policy_result['PLCY_ID']}")
    ...         for rule in policy_result['results']:
    ...             print(f"  Code {rule['code']}: {rule['denial_conditions']}")
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypedDict

import pandas as pd
import requests
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from deep_research_core.base_agent import AgentBase, CredentialProvider
from deep_research_utils.app_constant import AppConstants
from deep_research_agents.models.reimbursement_models import (
    PolicyExtractionResponse,
    PolicyTriageResponse,
    ColumnLabelsResponse,
    ColumnDefinition,
    ColumnCategoryMap,
    TableSchemaResponse,
    ElevanceSummaryResponse,
    RuleSummary,
    PolicyRecommendationResponse,
    RecommendedAction,
    ReimbursementAgentResponseSchema
)

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover - local/dev fallback
    logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================


class ReimbursementState(TypedDict, total=False):
    """State for reimbursement policy extraction agent (single-pattern processing)."""
    
    # Orchestrator inputs - PATTERN REQUIRED
    conversation_id: str  # Conversation ID from orchestrator
    query: str  # User query from UI
    context: Dict[str, Any]  # Full context from orchestrator (must contain 'pattern')
    job_id: str  # Job ID for this execution
    
    # Pattern data (extracted from context)
    pattern: Dict[str, Any]  # Single pattern structure from pattern agent
    pattern_rank: int  # Pattern identifier
    
    # Extracted codes (from pattern)
    cpt_codes: str  # Comma-separated CPT codes (extracted from pattern)
    drg_codes: List[str]  # DRG codes from pattern
    search_keywords: str  # Generated from DRGs for policy search
    
    # Intermediate state
    raw_policies: List[Dict[str, Any]]  # Raw API response
    filtered_policies: List[Dict[str, Any]]  # After deduplication
    policy_content: List[Dict[str, Any]]  # Full text from Snowflake
    
    # Resources
    snowflake_helper: Any  # SnowparkHelper instance
    llm: Any  # LLM client (GPT-5.4 for all operations except policy processing)
    llm_mini: Any  # LLM client for policy processing (GPT-5.4-nano for speed)
    llm_retry: Any  # LLM client used only on extract retry (GPT-5.4-mini, stronger instruction-following)
    
    # Execution tracking
    start_time: str  # ISO timestamp
    input_tokens: int  # Total input tokens
    output_tokens: int  # Total output tokens
    
    # Output
    result: List[Dict[str, Any]]  # Extracted rules per policy
    policy_contamination_summary: Dict[str, Any]  # validate_policies_node aggregate
    table_structure: Dict[str, Any]  # Column analysis and structure metadata
    column_metadata: List[Dict[str, str]]  # LLM-generated column info
    formatted_output: Dict[str, Any]  # Formatted reimbursement output with summary_table and individual_policies
    elevance_executive_summary: Optional[str]  # Elevance executive summary (from notebook logic)
    recommended_action: Optional[List[Dict[str, Any]]]  # List of structured recommendations
    recommendation_validation: Optional[Dict[str, Any]]  # Validator verdict + dropped-item summary


# ============================================================================
# Prompts
# ============================================================================


SYSTEM_PROMPT = """You are an expert at analyzing healthcare reimbursement policies and extracting key information.
Read the policy text and return valid JSON only.
Follow the exact schema provided.
Do not output markdown or explanatory text.
Use null or [] when information is missing."""

USER_PROMPT_TEMPLATE = """Given the following pattern: {pattern_details}

Policy Text: {policy_text}

Extract adjudication rules for the requested pattern. Return ONLY valid JSON matching this schema:

{{
  "policy_metadata": {{
    "policy_scope": "string",
    "effective_date": "string or null",
    "specialty_specific": "boolean",
    "diagnosis_specific": "boolean"
  }},
  "results": [
    {{
      "code": "string",
      "mention_status": "mentioned | not_mentioned",
      "payor_level_summary": "string",
      "site_of_service": "string",
      "bundling_logic": "string",
      "code_interactions": "string",
      "authorization_requirements": "string",
      "documentation_requirements": "string",
      "limitations": "string",
      "exclusions": "string",
      "specific_rule_text": "string",
      "confidence": "high | medium | low",

      "action_type": "deny | bundle | require_auth | pay_separately | allow_with_conditions | limit | null",
      "target_codes": ["string", ...],
      "related_codes": ["string", ...],
      "required_modifiers": ["string", ...],
      "excluded_modifiers": ["string", ...],
      "revenue_codes": ["string", ...],
      "pairing_conditions": ["string", ...],
      "utilization_limits": ["string", ...],
      "prior_auth_thresholds": ["string", ...],
      "discharge_status_conditions": ["string", ...],
      "program_scope": ["string", ...],
      "state_specific_rules": ["string", ...],
      "provider_role_restrictions": ["string", ...],
      "exemptions": ["string", ...]
    }}
  ]
}}

Prose field guidance:
- code: DRG code or CPT code being analyzed
- mention_status: Whether the code is explicitly mentioned in the policy
- payor_level_summary: Overall aggregated summary for this code
- site_of_service: Allowed, restricted, or excluded settings
- bundling_logic: Whether the code is bundled, inclusive, or separately reimbursed
- code_interactions: Interactions with other codes or procedures
- authorization_requirements: Prior authorization or notification requirements
- documentation_requirements: Required clinical documentation
- limitations: Frequency, quantity, or other limitations
- exclusions: Conditions or scenarios where code is not covered
- specific_rule_text: Exact text from policy supporting the finding
- confidence: Your confidence in the extraction

Structured edit-rule field guidance — these power code-specific recommendations.
Copy code identifiers, modifiers, revenue codes, and numeric limits VERBATIM from
the policy text. Do NOT paraphrase. Keep each array entry ≤8 words. Use empty
arrays (not null) when nothing applies.

- action_type: The dominant action the rule takes (deny / bundle / require_auth /
  pay_separately / allow_with_conditions / limit). null if unclear.
- target_codes: The CPT/HCPCS codes the rule acts on (e.g., ["99291"], ["90837"]).
- related_codes: Other CPT/HCPCS codes referenced in the condition (e.g.,
  ["99281", "99285", "G0378"]).
- required_modifiers / excluded_modifiers: Modifier values verbatim
  (e.g., ["25"], ["59", "XU"]). Preserve the exact modifier format.
- revenue_codes: Revenue codes verbatim, INCLUDING X wildcards. Write "045X"
  not "0450". Examples: ["045X", "0762", "068X"].
- pairing_conditions: Short claim-/timing-level phrases. Examples:
  "same claim line", "same date of service", "day before ED visit",
  "within 24 hours of ED visit".
- utilization_limits: Numeric limits with units. Examples:
  "10 hrs/week BCBA", "max 8 hours per DOS", "30 sessions/year".
- prior_auth_thresholds: PA trigger phrases citing the code + threshold.
  Examples: "PA required after 20 sessions of 90837",
  "auth required for >2 units of G0378".
- discharge_status_conditions: Discharge-disposition restrictions formatted
  "discharge status <code> (<label>)". Example: "discharge status 01 (home)".
- program_scope: Programs the rule applies to. Examples: ["COPPS"], ["DRG"],
  ["APC"], ["OPPS"], ["MS-DRG"], ["Case rate"].
- state_specific_rules: State-scoped rules. Example: "NC: 20 session cap".
- provider_role_restrictions: Role / oversight requirements. Examples:
  "BCBA oversight required", "QP/AP billing only".
- exemptions: Conditions that EXEMPT a claim from the rule (look for "except",
  "unless", "does not apply when"). Cite the exempting code/RC verbatim.
  Examples: "observation RC 0762", "trauma activation RC 068X".

Anti-contamination guardrails (CRITICAL — violations cause downstream filtering):
1. Describe ONLY the payer that authored this policy. Never reference any
   other payer by name (Cigna, Aetna, Humana, UHC, United Health, Elevance,
   Anthem, Molina, BCBS, Blue Cross, Kaiser, WellCare) in payor_level_summary,
   specific_rule_text, or any field. If the policy text mentions another
   payer comparatively, paraphrase without naming them.
2. target_codes, related_codes, required_modifiers, excluded_modifiers, and
   revenue_codes must contain ONLY codes that appear verbatim in the policy
   text. Do NOT copy header phrases from this prompt (e.g., "DRG Codes:",
   "CPT/HCPCS Codes:", "Pattern:") into any field. Codes only.
3. Distinguish "off-topic" from "uses different code system". The pattern
   may reference a DRG/MS-DRG label, a CPT/HCPCS code, an ICD-10 diagnosis,
   or a clinical concept — and the policy may adjudicate the same claims
   using any of those code systems.
   - OFF-TOPIC (policy governs an unrelated clinical area, e.g. an
     anesthesia-only policy when the pattern is about behavioral health,
     or a readmission billing policy when the pattern is about a surgical
     DRG): return empty arrays for the structured edit-rule fields and
     say so in payor_level_summary.
   - ON-TOPIC but uses a different code system than the pattern names
     (e.g. policy cites CPT/HCPCS or modifiers while the pattern names a
     DRG, or policy cites ICD-10 diagnoses while the pattern names a
     procedure): EXTRACT every code/modifier/revenue code that appears
     in the policy and governs the same claims. A bundling rule, a
     modifier requirement, or a coverage limit that applies to the
     pattern's clinical area is in-scope even if the policy never quotes
     the DRG/CPT label from the prompt. Empty arrays are the WRONG
     answer here.
   Do NOT fabricate codes or rules from the prompt headers — every code
   you return must appear verbatim in the policy text.

Return ONLY valid JSON, no explanatory text."""


TRIAGE_SYSTEM_PROMPT = """You rank reimbursement policies by relevance to a billing pattern.
Return ONLY valid JSON: {"selected_policy_ids": ["...", ...]}.
Order the list from most to least relevant.
Pick at most `target_count` policy_ids. Do not invent ids."""

TRIAGE_USER_PROMPT_TEMPLATE = """Pattern context:
{pattern_details}

Candidate policies (policy_id | payor | title):
{policy_lines}

Select up to {target_count} policy_ids whose titles look most relevant to the pattern.
Prefer policies whose titles reference the CPT/DRG codes, the clinical concept in the
pattern, or the LOB/state context. Skip policies whose titles look off-topic.

Return ONLY JSON: {{"selected_policy_ids": ["<id1>", "<id2>", ...]}}"""


# Per-payor + global caps applied at the end of search_policies_node.
# Elevance is pinned (always retained when present in API results) — the
# triage LLM never gets to drop it.
POLICY_CAP_MAX_PER_PAYOR = 5
POLICY_CAP_MAX_TOTAL = 30
ELEVANCE_PAYOR_TOKEN = "Elevance"


# ============================================================================
# Reimbursement Policy Agent
# ============================================================================


class ReimbursementPolicyAgent(AgentBase):
    """
    Agent for extracting structured adjudication rules from reimbursement policies.
    
    This agent:
    1. Searches for policies via Carelon policy comparison API
    2. Filters and deduplicates policies by payor and hash
    3. Retrieves full policy text from Snowflake
    4. Uses LLM to extract structured rules for each CPT code
    
    Args:
        snowflake_helper: Optional SnowparkHelper instance
        snowflake_helper_builder: Optional callable that returns SnowparkHelper
        policy_api_url: Base URL for policy comparison API
        retry_delay: Seconds to wait before retrying failed LLM calls
        max_retries: Maximum number of retries for failed LLM calls
        **kwargs: Additional arguments passed to AgentBase
    """

    api_response_model = ReimbursementAgentResponseSchema
    
    def __init__(
        self,
        snowflake_helper: Optional[Any] = None,
        snowflake_helper_builder: Optional[Callable[[], Any]] = None,
        policy_api_url: str = "https://policy-comparison-api.carelon.com/policy_comparison/search",
        retry_delay: int = 30,
        max_retries: int = 1,
        **kwargs
    ):
        # Configuration (set before base init)
        self.policy_api_url = policy_api_url
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
        # Initialize base agent first (sets up self.logger)
        super().__init__(
            agent_name="reimbursement_policy",
            state_class=ReimbursementState,
            llm_reasoning_effort="medium",  # GPT-5.4 for all operations except policy processing
            **kwargs
        )
        
        # Initialize Snowflake connection (after logger is available)
        self.snowflake_helper = self._init_snowflake(
            snowflake_helper, 
            snowflake_helper_builder
        )
        
        # Build table registry from config (all config items are tables for this agent)
        self._table_registry = self._build_table_registry()
        
        # Initialize mini LLM for policy processing (GPT-5.4-nano for better performance with large sets)
        self.llm_mini = self._init_mini_llm()

        # Initialize the retry LLM (GPT-5.4-mini, one tier up from nano with
        # stronger instruction-following). Used ONLY when the first nano
        # attempt at policy extraction fails — escalating on retry gives
        # the policy a second chance with a model that's better at
        # respecting the anti-leakage guardrails in USER_PROMPT_TEMPLATE.
        self.llm_retry = self._init_retry_llm()

        # Per-invocation token tracking — populated by _record_tokens after every
        # successful llm.invoke() and reset at the start of each prepare_state().
        # The lock protects the read-modify-write inside _record_tokens when
        # extract_rules / the relevance judge fan out across worker threads.
        self._token_breakdown: Dict[str, Dict[str, int]] = {}
        self._token_lock = threading.Lock()

    def _init_mini_llm(self) -> Any:
        """
        Initialize GPT-5.4-nano LLM for policy processing.

        Uses GPT-5.4-nano with low reasoning effort for fast policy extraction.
        This is significantly faster than GPT-5.4 (medium reasoning) when processing large policy sets.

        Returns:
            Configured ChatOpenAI instance with GPT-5.4-nano
        """
        try:
            from langchain_openai import ChatOpenAI
            from deep_research_core.base_agent import CredentialProvider

            creds = CredentialProvider.get_instance()
            token = creds.get_llm_token()

            # Use GPT-5.4-nano with low reasoning effort for speed
            # Supported values: 'low', 'medium', 'high', 'minimal'
            llm_mini = ChatOpenAI(
                model="gpt-5.4-nano",
                api_key=token,
                extra_body={"reasoning_effort": "low"}
            )

            self.logger.info("Initialized mini LLM for policy processing: gpt-5.4-nano with reasoning_effort='low'")
            return llm_mini

        except Exception as e:
            self.logger.warning(f"Failed to initialize mini LLM, will fall back to main LLM: {e}")
            # Fallback to main LLM if mini LLM initialization fails
            return self.llm

    def _init_retry_llm(self) -> Any:
        """
        Initialize the retry LLM for policy extraction (GPT-5.4-mini).

        When the first extraction attempt with the nano model fails, we
        escalate to gpt-5.4-mini with medium reasoning effort. Mini has
        materially better instruction-following than nano-low, so it's
        less prone to the prompt-leakage pattern (DRG header copied into
        target_codes) and cross-payer hallucination the nano model
        sometimes exhibits — exactly the failure modes most likely to
        cause the first attempt to throw on schema validation.

        Returns:
            Configured ChatOpenAI instance with gpt-5.4-mini, or the mini
            LLM as a fallback if init fails.
        """
        try:
            from langchain_openai import ChatOpenAI
            from deep_research_core.base_agent import CredentialProvider

            creds = CredentialProvider.get_instance()
            token = creds.get_llm_token()

            llm_retry = ChatOpenAI(
                model="gpt-5.4-mini",
                api_key=token,
                extra_body={"reasoning_effort": "medium"},
            )

            self.logger.info(
                "Initialized retry LLM for policy processing: "
                "gpt-5.4-mini with reasoning_effort='medium'"
            )
            return llm_retry

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize retry LLM, will fall back to nano on retry: {e}"
            )
            # Fallback to the nano client so the retry still has SOMETHING to run.
            return self.llm_mini

    def _invoke_llm_with_retry(self, llm: Any, messages: list, **invoke_kwargs) -> Any:
        """
        Dispatch an LLM invocation to the right retry path based on which
        of the agent's three LLMs (`self.llm` / `self.llm_mini` /
        `self.llm_retry`) was passed.

        - Main LLM → delegates to the base class `_invoke_with_token_retry`,
          which already wires `self.llm` + `self._initialize_llm` into the
          `ehap_retry.llm_invoke` helper.
        - Mini / retry LLMs → the base method can't be reused (it hardcodes
          `self.llm` and `self._initialize_llm`), so we call `llm_invoke`
          directly with the matching tier-specific reinitializer so a
          refresh rebuilds the SAME tier (nano stays nano, mini stays mini).

        The shared `cache_utils` token cache singleton means a refresh
        triggered by any tier benefits the other two.

        Falls back to plain `llm.invoke(...)` when EHAP is not configured
        (test mode / external LLM injection / agent constructed via
        `__new__`), preserving prior behavior.
        """
        # Guard against attribute-less test instances (`__new__` bypass).
        ehap = getattr(self, "ehap", None)
        if ehap is None:
            return llm.invoke(messages, **invoke_kwargs)

        # Main LLM — reuse the base class helper. It handles token refresh,
        # reinitializer wiring, and `self.llm` reassignment for us.
        if llm is self.llm:
            return self._invoke_with_token_retry(messages, **invoke_kwargs)

        # Mini / retry — base helper is hardcoded to self.llm, so use
        # `ehap_retry.llm_invoke` directly with the matching tier factory.
        if llm is getattr(self, "llm_mini", None):
            reinitializer = self._init_mini_llm
            llm_role = "mini"
        elif llm is getattr(self, "llm_retry", None):
            reinitializer = self._init_retry_llm
            llm_role = "retry"
        else:
            # Unknown LLM (custom inject, partially-built test instance).
            # Safest fallback: invoke without retry wiring rather than
            # silently rebuilding it as the main LLM.
            return llm.invoke(messages, **invoke_kwargs)

        from deep_research_utils.ehap_retry import llm_invoke

        result, updated_llm = llm_invoke(
            llm=llm,
            ehap=ehap,
            messages=messages,
            llm_reinitializer=reinitializer,
            **invoke_kwargs,
        )

        # Replace the stale agent-level reference when a token refresh
        # recreated the client. Parallel threads holding the old closure
        # ref will pick up the fresh client on their next call via the
        # cache_utils singleton check.
        if updated_llm is not llm:
            if llm_role == "mini":
                self.llm_mini = updated_llm
            else:  # "retry"
                self.llm_retry = updated_llm

        return result

    def _build_table_registry(self) -> Dict[str, str]:
        """Build table registry from agent config."""
        # All config items for reimbursement agent are tables
        return dict(self._agent_config)
    
    def table(self, name: str) -> str:
        """
        Get fully qualified table name for the current environment.
        
        Args:
            name: Logical table name (e.g., 'policy_metadata')
            
        Returns:
            Fully qualified table name (e.g., 'D01_COC.COC_DTI.PLCY_MTDTA')
            
        Raises:
            KeyError: If table name not configured
            
        Example:
            >>> self.table('policy_metadata')
            'D01_COC.COC_DTI.PLCY_MTDTA'  # in dev
            'P01_COC.COC_DTI.PLCY_MTDTA'  # in prod
        """
        if name not in self._table_registry:
            available = ', '.join(sorted(self._table_registry.keys()))
            raise KeyError(
                f"Table '{name}' not configured for reimbursement agent in environment '{AppConstants.ENV}'. "
                f"Available tables: {available}"
            )
        return self._table_registry[name]
    
    def _init_snowflake(
        self, 
        helper: Optional[Any], 
        builder: Optional[Callable[[], Any]]
    ) -> Any:
        """
        Initialize Snowflake connection using same pattern as LLM.
        
        Args:
            helper: Pre-configured SnowparkHelper instance
            builder: Callable that returns SnowparkHelper
            
        Returns:
            Configured SnowparkHelper instance
            
        Raises:
            AgentConfigurationError: If Snowflake cannot be initialized
        """
        if helper:
            return helper
        
        if builder:
            try:
                return builder()
            except Exception as e:
                self.logger.error(f"Snowflake builder failed: {e}")
                raise
        
        # Auto-build from credentials
        try:
            from deep_research_utils.snowflake_helper import SnowparkHelper
            
            creds = CredentialProvider.get_instance()
            snowflake_creds = creds.get_snowflake_credentials()
            
            return SnowparkHelper(
                batch_size=10000,
                max_workers=6,
                enable_metrics=True,
                connection_pool_size=4,
                **snowflake_creds
            )
        except Exception as e:
            self.logger.warning(
                f"Snowflake credentials not configured; provide a helper to execute the agent.",
                exc_info=e
            )
            return None
    
    # ========================================================================
    # Graph Construction (Override base class)
    # ========================================================================
    
    def build_graph(self) -> StateGraph:
        """
        Build multi-node graph for policy extraction pipeline with recommendation generation.
        
        Returns:
            Compiled StateGraph
        """
        graph = StateGraph(self.state_class)
        
        # Add nodes for each pipeline stage
        graph.add_node("search_policies", self.search_policies_node)
        graph.add_node("fetch_content", self.fetch_content_node)
        graph.add_node("extract_rules", self.extract_rules_node)
        graph.add_node("validate_policies", self.validate_policies_node)
        graph.add_node("analyze_table_structure", self.analyze_table_structure_node)
        graph.add_node("format_output", self.format_output_node)
        graph.add_node("generate_recommendation", self.generate_recommendation_node)
        graph.add_node("validate_recommendation", self.validate_recommendation_node)

        # Linear flow with:
        #   - validate_policies: deterministic relevance gate that attaches
        #     a `contamination` block to each policy result so downstream
        #     nodes (table summarization, recommendation peer benchmarks)
        #     can skip cross-payer leaks, prompt-leak target_codes, and
        #     title/content topic mismatches.
        #   - validate_recommendation: LLM critique of the final
        #     recommendation items for scope drift / source-disclaim.
        graph.add_edge(START, "search_policies")
        graph.add_edge("search_policies", "fetch_content")
        graph.add_edge("fetch_content", "extract_rules")
        graph.add_edge("extract_rules", "validate_policies")
        graph.add_edge("validate_policies", "analyze_table_structure")
        graph.add_edge("analyze_table_structure", "format_output")
        graph.add_edge("format_output", "generate_recommendation")
        graph.add_edge("generate_recommendation", "validate_recommendation")
        graph.add_edge("validate_recommendation", END)

        return graph
    
    # ========================================================================
    # Graph Nodes
    # ========================================================================
    
    def search_policies_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 1: Search for policies via API using pattern-specific keywords with LOB/State filters.
        
        Uses search_keywords (generated from DRG codes) and applies LOB/State filters
        to significantly reduce the number of irrelevant policies retrieved.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with filtered_policies
        """
        # Use search_keywords if available (new mode), fallback to cpt_codes (legacy mode)
        search_keywords = state.get("search_keywords")
        cpt_codes = state["cpt_codes"]
        pattern = state.get("pattern")
        pattern_rank = pattern.get("pattern_rank") if pattern else None
        
        if search_keywords:
            keyword_for_api = search_keywords
            self.logger.info(f"Pattern {pattern_rank}: Searching with keywords: '{keyword_for_api}'")
        else:
            # Legacy mode: use first CPT code
            keyword_for_api = cpt_codes.split(",")[0].strip()
            self.logger.info(f"Pattern {pattern_rank}: Searching with CPT codes: {cpt_codes}")
        
        # Build POST data with filters (from notebook logic)
        post_data = {
            "policy_type": ["Reimbursement"]
        }
        
        # Add LOB and State filters if pattern is available
        if pattern:
            # Extract LOB, State, and Product using notebook's comprehensive logic
            context = state.get("context", {})
            lob_list, state_list = self._extract_lob_and_state_filters(pattern, context)
            
            # Add LOB filter if available
            if lob_list:
                # modify LOB sub-classes to general categories (Commercial/Medicare)
                for index, lob in enumerate(lob_list):
                    if lob in ["Commercial Individual", "Commercial Local Group"]:
                        lob_list[index] = "Commercial"
                    elif lob in ["Medicare GRS", "Medicare Indiv"]:
                        lob_list[index] = "Medicare"

                post_data["lob"] = lob_list
                
            else:
                self.logger.warning(f"Pattern {pattern_rank}: No LOB filter available (not found in cards)")
            
            # Add State filter if available
            if state_list:
                post_data["state"] = state_list
            else:
                self.logger.warning(f"Pattern {pattern_rank}: No State filter available (not found in pattern or cards)")
        else:
            self.logger.info("Legacy mode: No pattern context, skipping LOB/State filters")
        
        # Log final filter configuration
        active_filters = {k: v for k, v in post_data.items() if k != "policy_type"}
        if active_filters:
            self.logger.info(f"Pattern {pattern_rank}: Active API filters: {active_filters}")
        else:
            self.logger.warning(f"Pattern {pattern_rank}: No LOB/State filters active - may return many irrelevant policies")
        
        # API call with POST body for filters (with retry mechanism)
        url = f"{self.policy_api_url}?keyword={keyword_for_api}&user_type=onshore"
        max_retries = 2
        retry_delays = [1, 2]  # Seconds to wait between retries
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url, 
                    headers={"Content-Type": "application/json"}, 
                    json=post_data, 
                    verify=AppConstants.SSL_CERT_FILE
                )
                response.raise_for_status()
                raw_data = response.json()
                break  # Success, exit retry loop
                
            except (requests.RequestException, requests.HTTPError) as e:
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    self.logger.warning(
                        f"Pattern {pattern_rank}: Policy API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay} second(s)..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Pattern {pattern_rank}: Policy API call failed after {max_retries + 1} attempts: {e}"
                    )
                    raise
        
        self.logger.info(f"Pattern {pattern_rank}: API returned {len(raw_data)} policies (with keyword + filters)")
        
        # Convert to DataFrame for additional filtering
        df = pd.DataFrame(raw_data)
        df = df['sentencelist'].apply(pd.Series)
        
        # Filter by file type
        df["file_type"] = df.policy_link.str.split(".").str[-1]
        df = df[df['file_type'] == 'pdf']
        
        # Policy type should already be filtered by API, but double-check
        df = df[df['policy_type'] == 'Reimbursement']
        
        # Remove duplicates by payor & external_link
        df = df.drop_duplicates(subset=['payor', 'external_link'])
        
        self.logger.info(f"Pattern {pattern_rank}: After PDF/Reimbursement/dedup: {len(df)} policies")
        
        # Get policy hashes from Snowflake
        policy_ids = df.policy_id.unique().tolist()
        df_hash = self._get_policy_hashes(policy_ids)
        
        # Merge and deduplicate by hash
        df = df.merge(df_hash, left_on='policy_id', right_on='PLCY_ID', how='left')
        df = df.drop_duplicates(subset=['payor', 'PDF_HASH_VAL_ID'])
        
        self.logger.info(f"Pattern {pattern_rank}: After hash deduplication: {len(df)} unique policies")
        
        # Remove internal Elevance policies if external ones exist (from notebook logic)
        initial_count = len(df)
        internal_elevance_payors = ['Elevance Health (Internal)', 'Elevance Health (internal)']
        has_external_elevance = (df['payor'] == 'Elevance Health (external)').any()
        
        if has_external_elevance:
            # Remove internal Elevance policies
            df = df[~df['payor'].isin(internal_elevance_payors)]
            removed_count = initial_count - len(df)
            
            if removed_count > 0:
                self.logger.info(
                    f"Pattern {pattern_rank}: Removed {removed_count} internal Elevance policies "
                    f"(external Elevance policies exist)"
                )
                elevance_count = len(df[df['payor'].str.contains('Elevance Health', na=False)])
                external_count = len(df[df['payor'] == 'Elevance Health (external)'])
                self.logger.info(
                    f"Remaining Elevance policies: {elevance_count} "
                    f"({external_count} external only)"
                )
        
        # Cap the policy set with three goals:
        #   1. Always keep Elevance policies when present (pinned, not subject to LLM triage)
        #   2. Cap each payor at POLICY_CAP_MAX_PER_PAYOR so one payor cannot crowd out others
        #   3. Use an LLM title-triage call to pick the most relevant non-Elevance policies
        #      when the per-payor-capped set still exceeds the remaining budget. Triage falls
        #      back to API ordering on any failure so the pattern always makes progress.
        df = self._apply_policy_caps(
            df=df,
            pattern_rank=pattern_rank,
            pattern_details=self._build_pattern_details(state),
            llm_mini=state.get("llm_mini"),
        )

        self.logger.info(f"Pattern {pattern_rank}: ✓ Final policy count: {len(df)} policies ready for extraction")
        
        # Summary of search effectiveness
        if len(df) == 0:
            self.logger.warning(
                f"Pattern {pattern_rank}: No policies found! Check if filters are too restrictive or keyword is too specific."
            )
        
        return {
            "raw_policies": raw_data,
            "filtered_policies": df.to_dict('records')
        }
    
    def fetch_content_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 2: Fetch full policy content from Snowflake.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with policy_content
        """
        # Validate Snowflake connection (following correlation agent pattern)
        snowflake_helper = state.get("snowflake_helper")
        if snowflake_helper is None:
            from deep_research_core.base_agent import AgentExecutionError
            raise AgentExecutionError(
                "Snowflake connection is required but not configured. "
                "Please provide Snowflake credentials via environment variables:\n"
                "  - SNOWFLAKE_ACCOUNT\n"
                "  - SNOWFLAKE_USER\n"
                "  - SNOWFLAKE_SECRET\n"
                "  - SNOWFLAKE_WAREHOUSE\n"
                "  - SNOWFLAKE_DATABASE\n"
                "  - SNOWFLAKE_SCHEMA\n"
                "Or pass a snowflake_helper instance when creating the agent."
            )
        
        policies = state["filtered_policies"]
        policy_ids = [p['policy_id'] for p in policies]
        
        self.logger.info(f"Fetching content for {len(policy_ids)} policies")
        
        # SQL query to get full policy text
        # Uses environment-aware table from registry.
        # ORDER BY PLCY_ID makes the row order deterministic across runs
        # (Snowflake GROUP BY otherwise returns an arbitrary order, which
        # used to misalign format_output_node's positional metadata
        # lookup — the bug is now fixed via dict lookup, but determinism
        # still helps log readability and any future positional consumer).
        sql = f"""
        SELECT PLCY_ID,
               LISTAGG(PAGE_DATA_TXT, '\n') WITHIN GROUP (ORDER BY PAGE_NBR) AS POLICY_TEXT
        FROM {self.table('policy_search_detail')}
        WHERE PLCY_ID IN ({{policy_ids}})
        GROUP BY PLCY_ID
        ORDER BY PLCY_ID
        """
        
        policy_ids_str = ", ".join([f"'{pid}'" for pid in policy_ids])
        df_content = snowflake_helper.execute_query_and_return_pandas_df(
            sql.format(policy_ids=policy_ids_str)
        )
        
        self.logger.info(f"Retrieved content for {len(df_content)} policies")
        
        return {
            "policy_content": df_content.to_dict('records')
        }
    
    def extract_rules_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 3: Extract structured rules from each policy using LLM.
        
        Uses GPT-5.4-nano for policy processing to improve performance with large policy sets.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with result
        """
        policy_content = state["policy_content"]
        filtered_policies = state.get("filtered_policies") or []
        cpt_codes = state["cpt_codes"]
        drg_codes = state.get("drg_codes", [])
        pattern = state.get("pattern", {})
        llm_mini = state["llm_mini"]  # GPT-5.4-nano — first attempt
        # GPT-5.4-mini escalated for the retry path; falls back to nano if
        # the retry LLM wasn't initialized (offline / stub-agent contexts).
        llm_retry = state.get("llm_retry") or llm_mini

        # Build pattern_details string for LLM prompt (from notebook)
        pattern_details_parts = []

        # Add DRG codes if available
        if drg_codes:
            pattern_details_parts.append(f"DRG Codes: {', '.join(drg_codes)}")

        # Add CPT codes if available (fallback)
        if cpt_codes:
            pattern_details_parts.append(f"CPT/HCPCS Codes: {cpt_codes}")

        # Add pattern title if available
        if pattern:
            pattern_title = pattern.get("pattern_title") or pattern.get("top_pattern")
            if pattern_title:
                pattern_details_parts.append(f"Pattern: {pattern_title}")

        pattern_details = "\n".join(pattern_details_parts) if pattern_details_parts else cpt_codes

        # Index filtered_policies by policy_id so we can resolve a stable
        # human-readable title at log time. The API exposes the title via
        # `policy_title` (the doc-name field on the search hit); fall back
        # to `external_link` / `policy_link` filename if no title field.
        title_by_id: Dict[str, str] = {}
        for fp in filtered_policies:
            pid = fp.get("policy_id")
            if not pid:
                continue
            title = (
                fp.get("policy_title")
                or fp.get("document_name")
                or fp.get("external_link")
                or fp.get("policy_link")
                or ""
            )
            title_by_id[pid] = title

        total = len(policy_content)
        concurrency = self._resolve_extract_concurrency()

        self.logger.info(
            f"Extracting rules from {total} policies "
            f"(env={AppConstants.ENV!r}, concurrency={concurrency})"
        )
        self.logger.info(f"Pattern details: {pattern_details}")

        def _process_one(idx: int, policy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Run extract → sanitize for a single policy. Returns the
            sanitized response dict (with PLCY_ID stamped) on success, or
            None on terminal failure. Per-policy retry uses llm_retry."""
            policy_id = policy['PLCY_ID']
            policy_title = title_by_id.get(policy_id, "")
            self.logger.info(
                f"Processing policy {idx+1}/{total}: policy_id={policy_id} title={policy_title!r}"
            )

            try:
                response = self._extract_policy_rules(
                    policy['POLICY_TEXT'],
                    pattern_details,
                    llm_mini
                )
                response = self._sanitize_extracted_facts(policy_id, policy_title, response)
                response["PLCY_ID"] = policy_id
                return response

            except Exception as e:
                self.logger.warning(
                    f"Extract attempt 1 failed for policy_id={policy_id} "
                    f"title={policy_title!r}: {e}; retrying once after {self.retry_delay}s"
                )

                if self.max_retries <= 0:
                    self.logger.error(
                        f"Extract failed and retries disabled for policy_id={policy_id} "
                        f"title={policy_title!r}; dropping policy from this pattern"
                    )
                    return None

                try:
                    time.sleep(self.retry_delay)
                    response = self._extract_policy_rules(
                        policy['POLICY_TEXT'],
                        pattern_details,
                        llm_retry
                    )
                    response = self._sanitize_extracted_facts(policy_id, policy_title, response)
                    response["PLCY_ID"] = policy_id
                    self.logger.info(
                        f"Extract retry succeeded on gpt-5.4-mini for "
                        f"policy_id={policy_id} title={policy_title!r}"
                    )
                    return response

                except Exception as e2:
                    self.logger.error(
                        f"Extract failed after retry for policy_id={policy_id} "
                        f"title={policy_title!r}: {e2}; dropping policy from this pattern"
                    )
                    return None

        # Fan out one task per policy, then reassemble in the original
        # policy_content order so results[i] still corresponds to
        # policy_content[i]. Downstream format_output_node no longer
        # relies on this index, but preserving it keeps logs readable
        # and any future positional consumers correct.
        by_index: Dict[int, Optional[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_process_one, i, policy): i
                for i, policy in enumerate(policy_content)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                by_index[i] = fut.result()

        results: List[Optional[Dict[str, Any]]] = [by_index.get(i) for i in range(total)]
        successful = sum(1 for r in results if r is not None)
        self.logger.info(f"Successfully processed {successful}/{total} policies")

        return {"result": results}

    def validate_policies_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 4: Relevance gate — deterministic checks + LLM relevance judge.

        Sits between extract_rules and analyze_table_structure. For each
        extracted policy, runs:
          - Deterministic identity / cross-payer-leak / prompt-leak /
            mention_status checks (`_check_policy_contamination`). These
            are domain-agnostic — they catch real failures regardless of
            whether the pattern is OB, oncology, cardiology, etc.
          - An LLM relevance judge (`_judge_policy_relevance`) that asks
            "does this policy actually adjudicate this pattern?" The judge
            replaces the previous hardcoded OB topic-keyword heuristic —
            its medical knowledge generalizes to any specialty.

        Judges run in parallel via `_resolve_extract_concurrency()`.
        Each policy gets a `contamination` block attached in place;
        quarantine-severity policies are excluded by downstream nodes.
        Logs every flag with policy_id + payer + title + severity + flags
        + reasons so the operator can trace every drop.
        """
        results = state.get("result") or []
        filtered_policies = state.get("filtered_policies") or []
        pattern = state.get("pattern") or {}
        cpt_codes = (state.get("cpt_codes") or "").strip()
        drg_codes = state.get("drg_codes") or []
        # `llm_mini` powers the judge; tests that build the agent via
        # __new__ may not provide it — in that case the judge is skipped
        # and the remaining deterministic checks (identity, prompt-leak,
        # cross-payer-name-leak) still run.
        llm_mini = state.get("llm_mini")

        meta_by_id: Dict[str, Dict[str, Any]] = {}
        for fp in filtered_policies:
            pid = fp.get("policy_id")
            if pid:
                meta_by_id[pid] = fp

        # Stage 1: build per-policy contexts (no LLM yet). Skipping
        # non-dict results so the rest of the pipeline doesn't get
        # surprises from extract_rules failures.
        policy_contexts: List[Dict[str, Any]] = []
        for policy_result in results:
            if not isinstance(policy_result, dict):
                continue

            policy_id = policy_result.get("PLCY_ID")
            meta = meta_by_id.get(policy_id, {}) if policy_id else {}
            llm_meta = policy_result.get("policy_metadata") or {}

            payer = meta.get("payor") or "Unknown Payer"
            # Prefer the API-supplied title over the LLM-extracted scope
            # for topic inference and the judge prompt — `policy_scope`
            # is often a coverage sentence ("All members in California"),
            # not a document title.
            title = (
                meta.get("policy_title")
                or meta.get("document_name")
                or llm_meta.get("policy_scope")
                or ""
            )
            policy_url = meta.get("external_link") or meta.get("policy_link") or ""

            evidence_chunks: List[str] = []
            for rule in policy_result.get("results") or []:
                if not isinstance(rule, dict):
                    continue
                for key in ("payor_level_summary", "specific_rule_text"):
                    val = rule.get(key)
                    if isinstance(val, str) and val.strip():
                        evidence_chunks.append(val.strip())
            evidence = " ".join(evidence_chunks)

            edit_rule_facts = self._aggregate_edit_rule_facts(
                policy_result.get("results") or []
            )

            policy_contexts.append({
                "policy_result": policy_result,
                "policy_id": policy_id,
                "payer": payer,
                "title": title,
                "evidence": evidence,
                "edit_rule_facts": edit_rule_facts,
                "policy_url": policy_url,
            })

        # Stage 2: fan out LLM relevance judge calls. Judge runs only
        # when an llm_mini is provided AND we have non-empty evidence
        # to score against (empty evidence → mention_status path is the
        # right signal, no need to spend tokens). Verdict default-keeps
        # on any error so judge hiccups never silently quarantine.
        judge_by_id: Dict[Optional[str], Optional[Dict[str, Any]]] = {}
        if llm_mini is not None and policy_contexts:
            pattern_summary = self._build_judge_pattern_summary(
                pattern, drg_codes, cpt_codes
            )
            concurrency = self._resolve_extract_concurrency()
            self.logger.info(
                f"Running relevance judge on {len(policy_contexts)} policies "
                f"(concurrency={concurrency})"
            )

            def _judge_one(ctx: Dict[str, Any]) -> tuple:
                # Skip the LLM call when evidence is empty — judge
                # would have nothing to score and defaults to relevant.
                if not (ctx["evidence"] or "").strip():
                    return ctx["policy_id"], None
                try:
                    verdict = self._judge_policy_relevance(
                        pattern_summary=pattern_summary,
                        policy_title=ctx["title"],
                        evidence=ctx["evidence"],
                        llm=llm_mini,
                    )
                    return ctx["policy_id"], verdict
                except Exception as exc:
                    self.logger.warning(
                        f"Relevance judge failed for "
                        f"policy_id={ctx['policy_id']!r}: {exc}; "
                        f"defaulting to relevant=true"
                    )
                    return ctx["policy_id"], None

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for pid, verdict in pool.map(_judge_one, policy_contexts):
                    judge_by_id[pid] = verdict

        # Stage 3: deterministic checks + apply judge verdict.
        quarantined: List[Dict[str, Any]] = []
        warned: List[Dict[str, Any]] = []

        for ctx in policy_contexts:
            policy_result = ctx["policy_result"]
            contamination = self._check_policy_contamination(
                policy_id=ctx["policy_id"],
                payer=ctx["payer"],
                title=ctx["title"],
                evidence=ctx["evidence"],
                edit_rule_facts=ctx["edit_rule_facts"],
                policy_url=ctx["policy_url"],
                judge_verdict=judge_by_id.get(ctx["policy_id"]),
            )

            if contamination:
                policy_result["contamination"] = contamination
                self.logger.info(
                    f"validate_policies: policy_id={ctx['policy_id']!r} "
                    f"payer={ctx['payer']!r} title={ctx['title']!r} "
                    f"severity={contamination['severity']} "
                    f"flags={contamination['flags']} "
                    f"reasons={contamination['reasons']}"
                )
                summary_entry = {
                    "policy_id": ctx["policy_id"],
                    "payer": ctx["payer"],
                    "title": ctx["title"],
                    "flags": contamination["flags"],
                    "reasons": contamination["reasons"],
                }
                if contamination["severity"] == "quarantine":
                    quarantined.append(summary_entry)
                else:
                    warned.append(summary_entry)

        total_extracted = sum(1 for r in results if isinstance(r, dict))
        self.logger.info(
            f"validate_policies summary: {len(quarantined)} quarantined, "
            f"{len(warned)} warned out of {total_extracted} extracted policies"
        )

        return {
            "result": results,
            "policy_contamination_summary": {
                "total_extracted": total_extracted,
                "quarantined": quarantined,
                "warned": warned,
            },
        }

    # ------------------------------------------------------------------
    # Contamination check helpers
    # ------------------------------------------------------------------

    @classmethod
    def _canonicalize_payer(cls, name: str) -> str:
        """Lowercased payer name with common suffixes stripped — used so
        the cross-payer leak check treats 'Elevance Health (external)' and
        'Elevance' as the same payer when comparing against KNOWN_PAYERS.
        """
        if not name:
            return ""
        lowered = name.lower()
        # Drop parenthetical qualifiers like '(external)' / '(internal)'.
        lowered = re.sub(r"\s*\([^)]*\)\s*", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @classmethod
    def _payer_key_from_name(cls, payer: str) -> Optional[str]:
        """Map a payer_name string to a canonical key (e.g. 'cigna', 'united').
        Returns None when the name doesn't match any known payer token."""
        if not payer:
            return None
        lowered = payer.lower()
        for token, canonical in cls._PAYER_NAME_TO_CANONICAL:
            if token in lowered:
                return canonical
        return None

    @classmethod
    def _payer_key_from_url(cls, policy_url: str) -> Optional[str]:
        """Map a policy_url to a canonical payer key based on its domain.
        Returns None when the URL doesn't match any known domain."""
        if not policy_url:
            return None
        lowered = policy_url.lower()
        for domain, canonical in cls._URL_TO_PAYER:
            if domain in lowered:
                return canonical
        return None

    @classmethod
    def _payer_key_from_policy_id(cls, policy_id: str) -> Optional[str]:
        """Map a policy_id like 'RP_COM_CIG_00021' to a canonical payer key
        using the second-or-third underscore-separated token. Returns None
        when no recognized prefix is present (Elevance often falls here —
        e.g. 'RP_GBD_*' or 'RP_COM_ELV_EXTERNAL_*' — so a None here
        means 'no constraint', not 'mismatch')."""
        if not policy_id:
            return None
        upper = policy_id.upper()
        for token in upper.split("_"):
            if token in cls._POLICY_ID_PREFIX_TO_PAYER:
                return cls._POLICY_ID_PREFIX_TO_PAYER[token]
        return None

    def _check_policy_contamination(
        self,
        policy_id: Optional[str],
        payer: str,
        title: str,
        evidence: str,
        edit_rule_facts: Dict[str, Any],
        policy_url: str = "",
        judge_verdict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run all contamination checks against one policy. Returns None
        when the policy is clean, otherwise a dict
        ``{flags: [...], severity: "quarantine"|"warn", reasons: [...]}``.

        Quarantine wins over warn when both severities trigger on the same
        policy — downstream consumers filter on the worst case.

        All checks here are domain-agnostic:
          - cross_payer_name_leak: canonicalizes both sides so Anthem and
            Elevance are treated as the same payer.
          - drg_prompt_leak_detected: post-sanitizer defense-in-depth.
          - policy_identity_mismatch: payer_name vs URL vs policy_id prefix
            triangulation — catches misattributed records.
          - policy_topic_irrelevant: LLM relevance judge verdict
            (`judge_verdict`). Sole authority on topical relevance —
            replaces the old mention_status-based deterministic disclaim
            check, which fired false-positives on DRG patterns whose
            policies adjudicate via CPT codes.
        """
        flags: List[str] = []
        reasons: List[str] = []
        worst_severity: Optional[str] = None

        def record(flag: str, severity: str, reason: str) -> None:
            nonlocal worst_severity
            if flag not in flags:
                flags.append(flag)
                reasons.append(reason)
            if severity == "quarantine" or worst_severity != "quarantine":
                worst_severity = severity

        # 1. Cross-payer name leakage in evidence text.
        # Canonicalize own payer once so the leak check can compare
        # canonical-vs-canonical (Anthem and Elevance both map to
        # "elevance" — see _PAYER_NAME_TO_CANONICAL — and shouldn't
        # mutually trip the check).
        own_payer_canonical = self._canonicalize_payer(payer)
        own_payer_key = self._payer_key_from_name(payer)
        evidence_lower = evidence.lower()
        own_payer_in_evidence = False
        if own_payer_canonical:
            own_tokens = [t for t in own_payer_canonical.split() if len(t) >= 4]
            own_payer_in_evidence = any(
                token in evidence_lower for token in own_tokens
            )
        leaked_payers: List[str] = []
        for other in self.KNOWN_PAYERS:
            other_canon = self._canonicalize_payer(other)
            if not other_canon:
                continue
            # Skip if the candidate resolves to the same canonical payer
            # as the owner (Anthem == Elevance). This is the fix that
            # eliminates the Anthem-in-Elevance-policy false positive.
            other_key = self._payer_key_from_name(other)
            if other_key and own_payer_key and other_key == own_payer_key:
                continue
            if other_canon in own_payer_canonical or own_payer_canonical in other_canon:
                continue
            if re.search(rf"\b{re.escape(other_canon)}\b", evidence_lower):
                leaked_payers.append(other)
        if leaked_payers:
            severity = "quarantine" if not own_payer_in_evidence else "warn"
            record(
                "cross_payer_name_leak",
                severity,
                f"evidence mentions other payer(s) {leaked_payers}; "
                f"own payer {'absent' if not own_payer_in_evidence else 'also mentioned'}",
            )

        # 2. DRG-prompt leak defense-in-depth (post-sanitize, this should
        # be empty; if anything still trips here, the sanitizer regressed).
        leaked_strings: List[str] = []
        for key in ("target_codes", "related_codes"):
            for v in edit_rule_facts.get(key, []) or []:
                if isinstance(v, str) and self._looks_like_prompt_leak(v):
                    leaked_strings.append(f"{key}={v!r}")
        if leaked_strings:
            record(
                "drg_prompt_leak_detected",
                "quarantine",
                f"prompt-header string(s) survived sanitizer: {leaked_strings}",
            )

        # 3. Hard policy identity check: payer_name, policy_url, and (when
        # recognizable) policy_id prefix must all map to the same canonical
        # payer. A mismatch here means the record is internally inconsistent
        # — the payer label can't be trusted as a source attribution.
        name_payer = self._payer_key_from_name(payer)
        url_payer = self._payer_key_from_url(policy_url)
        id_payer = self._payer_key_from_policy_id(policy_id or "")
        identity_mismatches: List[str] = []
        if name_payer and url_payer and name_payer != url_payer:
            identity_mismatches.append(
                f"payer_name={payer!r} → {name_payer!r} vs "
                f"policy_url={policy_url!r} → {url_payer!r}"
            )
        if name_payer and id_payer and name_payer != id_payer:
            identity_mismatches.append(
                f"payer_name={payer!r} → {name_payer!r} vs "
                f"policy_id={policy_id!r} prefix → {id_payer!r}"
            )
        if url_payer and id_payer and url_payer != id_payer:
            identity_mismatches.append(
                f"policy_url={policy_url!r} → {url_payer!r} vs "
                f"policy_id={policy_id!r} prefix → {id_payer!r}"
            )
        if identity_mismatches:
            record(
                "policy_identity_mismatch",
                "quarantine",
                "; ".join(identity_mismatches),
            )

        # 4. LLM relevance judge verdict. Generalizes to any specialty
        # (OB, oncology, cardiology, etc.) because it inherits the model's
        # medical knowledge.
        # Severity ladder (high-confidence-only quarantine — see below):
        #   relevant=False + confidence == high        → quarantine
        #   relevant=False + confidence in {medium, low, unknown} → warn
        # Only `high` quarantines because the cesarean run on 2026-06-12
        # showed the judge can rate clearly-relevant maternity policies as
        # `confidence=high` not-relevant when the pattern narrative cites
        # specific clinical drivers. Demoting `medium` to warn keeps those
        # borderline cases visible in the API output instead of silently
        # dropping them. On parse failure / missing verdict the policy
        # passes (default keep) so judge hiccups never drop policies.
        if isinstance(judge_verdict, dict) and judge_verdict.get("relevant") is False:
            confidence = (judge_verdict.get("confidence") or "").strip().lower()
            reason = (judge_verdict.get("reason") or "").strip() or "(no reason given)"
            if confidence == "high":
                record(
                    "policy_topic_irrelevant",
                    "quarantine",
                    f"relevance judge: relevant=false confidence={confidence!r}: {reason}",
                )
            else:
                record(
                    "policy_topic_irrelevant",
                    "warn",
                    f"relevance judge: relevant=false confidence={confidence or 'unknown'!r}: {reason}",
                )

        if not flags:
            return None
        return {
            "flags": flags,
            "severity": worst_severity or "warn",
            "reasons": reasons,
        }

    def _build_judge_pattern_summary(
        self,
        pattern: Dict[str, Any],
        drg_codes: List[str],
        cpt_codes: str,
    ) -> str:
        """Build a compact pattern descriptor for the relevance judge prompt.
        Keeps the input the LLM sees small and consistent regardless of
        medical specialty.

        Intentionally omits `pattern_details` / `why_it_matters`. Those fields
        embed the pattern's specific clinical drivers (e.g., hypertensive
        pregnancy, vasa previa for an OB pattern), which historically anchored
        the judge into rejecting topically-relevant policies that didn't
        address those exact drivers. The judge's job is topical screening,
        not driver-level matching — title + codes are sufficient signal.
        """
        parts: List[str] = []
        pattern_title = (
            pattern.get("top_pattern")
            or pattern.get("pattern_title")
            or ""
        )
        if pattern_title:
            parts.append(f"Pattern title: {pattern_title}")
        if drg_codes:
            parts.append(f"DRG codes: {', '.join(drg_codes[:5])}")
        if cpt_codes:
            parts.append(f"CPT/HCPCS codes: {cpt_codes}")
        return "\n".join(parts) if parts else "(no pattern context)"

    def _judge_policy_relevance(
        self,
        pattern_summary: str,
        policy_title: str,
        evidence: str,
        llm: Any,
    ) -> Optional[Dict[str, Any]]:
        """Ask an LLM: could this policy plausibly apply to claims in the
        pattern's clinical area? Returns
        ``{"relevant": bool, "confidence": "high|medium|low", "reason": str}``
        or None on parse failure (caller treats None as default-keep).

        The prompt frames relevance as TOPICAL applicability — a maternity
        bundling policy is relevant to any cesarean DRG pattern, even if the
        policy doesn't address the specific clinical drivers (hypertensive
        pregnancy, prior C-section, etc.) the pattern narrative mentions.
        This matches how reimbursement policies actually adjudicate: at the
        code/procedure level, not at the comorbidity-mix level.

        Mirrors the LLM-as-judge shape used in
        `recommendation_dtr_agent._check_pattern_relevance` — same prompt
        structure, same parse-then-validate flow, same default-keep
        behavior on parser hiccups."""
        # Trim evidence to keep the prompt small — judge mainly needs
        # enough context to recognize the topic, not the full text.
        evidence_excerpt = (evidence or "").strip()
        if len(evidence_excerpt) > 500:
            evidence_excerpt = evidence_excerpt[:500] + "…"

        prompt = (
            "You are screening a healthcare reimbursement policy for TOPICAL "
            "relevance to a clinical pattern. Use your medical knowledge — the "
            "pattern may be from any specialty (OB, oncology, cardiology, etc.).\n\n"
            f"{pattern_summary}\n\n"
            f"Policy title: {policy_title or '(unknown)'}\n"
            f"Policy evidence (first ~500 chars):\n{evidence_excerpt or '(empty)'}\n\n"
            "Question: Could the rules in this policy plausibly apply to claims "
            "for the codes / procedures / clinical area in the pattern?\n\n"
            "Answer relevant=TRUE if the policy governs the same clinical area "
            "(e.g., a 'Global Maternity/Obstetric Package' policy is relevant to "
            "any cesarean delivery DRG; an 'Assistant at Surgery' policy is "
            "relevant if it covers cesarean CPT codes; a 'Multiple Births' "
            "policy is relevant to obstetric delivery patterns). The policy "
            "does NOT need to address every clinical driver, DRG severity "
            "variant, or comorbidity mentioned in the pattern — broad topical "
            "applicability is enough.\n\n"
            "Answer relevant=FALSE only when the policy is clearly about a "
            "different clinical area (e.g., 'Anesthesia Professional Services' "
            "for a cardiology pattern, 'Readmission' billing rules for a "
            "non-readmission pattern, generic 'Add-on Coding' guidance with no "
            "tie to the pattern's procedures).\n\n"
            "Reply with JSON only, no markdown:\n"
            '{"relevant": true|false, "confidence": "high|medium|low", '
            '"reason": "<one-sentence rationale>"}\n\n'
            "Bias toward relevant=true. If unsure, answer relevant=true with "
            "confidence=low — false-positive quarantines are far worse than "
            "letting a borderline policy through."
        )

        messages = [
            {"role": "system", "content": "You are a healthcare policy relevance judge. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("relevance_judge", response)
            content = (response.content or "").strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])
            data = json.loads(content)
            if not isinstance(data, dict):
                return None
            return {
                "relevant": bool(data.get("relevant", True)),
                "confidence": (data.get("confidence") or "low").strip().lower(),
                "reason": (data.get("reason") or "").strip(),
            }
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.debug(f"Relevance judge parse failed: {exc}")
            return None

    def analyze_table_structure_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 4: Analyze pattern context and generate dynamic table structure (from notebook).

        Uses pattern-aware LLM column generation to select the most relevant columns
        for comparing payer behavior on this specific pattern.

        Args:
            state: Current graph state

        Returns:
            Updated state with table_structure (TableSchemaResponse) and column_metadata
        """
        results = state["result"]
        pattern = state.get("pattern", {})
        drg_codes = state.get("drg_codes", [])
        search_keywords = state.get("search_keywords", "")
        llm = state["llm"]

        self.logger.info("Generating dynamic table structure using pattern context")

        # Collect category samples from all policies
        category_samples = self._collect_category_samples(results)

        if not category_samples:
            self.logger.warning("No rule categories found in policies")
            return {
                "table_structure": TableSchemaResponse(columns=[], selected_categories=[]),
                "column_metadata": []
            }

        # Build pattern context for LLM
        pattern_context = self._build_pattern_context(pattern, drg_codes, search_keywords)

        # Generate dynamic columns using LLM (pattern-aware + user-query-aware)
        user_query = state.get("query")
        schema = self._generate_dynamic_columns(pattern_context, category_samples, llm, user_query=user_query)

        # Convert to column_metadata format for backward compatibility
        column_metadata = [col.model_dump() for col in schema.columns]

        self.logger.info(
            f"Generated {len(schema.columns)} dynamic columns: "
            f"{[c.id for c in schema.columns]}"
        )

        return {
            "table_structure": schema,
            "column_metadata": column_metadata
        }

    def format_output_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 5: Format extracted rules using comprehensive notebook logic.

        Uses pattern-aware dynamic columns and per-cell LLM summarization
        to create high-quality payer comparison tables.

        Args:
            state: Current graph state

        Returns:
            Updated state with formatted_output
        """
        from collections import defaultdict

        results = state["result"]
        filtered_policies = state["filtered_policies"]
        table_structure = state.get("table_structure")  # TableSchemaResponse
        pattern = state.get("pattern", {})
        llm = state["llm"]

        self.logger.info("Formatting output with comprehensive payer policy summary logic")

        # Index API metadata by policy_id rather than positional index.
        # `results[i]` corresponds to `policy_content[i]` (Snowflake
        # GROUP BY order), NOT to `filtered_policies[i]` — so the old
        # `filtered_policies[i]` lookup attached the wrong payer / title /
        # URL to every surviving policy. Dict lookup by PLCY_ID matches
        # the pattern already used in `validate_policies_node`.
        meta_by_id: Dict[str, Dict[str, Any]] = {}
        for fp in filtered_policies:
            pid = fp.get("policy_id")
            if pid:
                meta_by_id[pid] = fp

        # Build individual policies list and group by payer
        individual_policies = []
        policies_by_payer = defaultdict(list)

        for policy_result in results:
            if policy_result is None:
                continue

            policy_metadata = meta_by_id.get(policy_result.get("PLCY_ID"), {})

            # Extract metadata from LLM response
            llm_metadata = policy_result.get("policy_metadata", {})
            policy_results = policy_result.get("results", [])

            # Determine payer name
            payer_name = policy_metadata.get("payor", "Unknown Payer")

            # Determine policy title
            policy_title = llm_metadata.get("policy_scope") or policy_metadata.get("policy_title", "Policy Document")

            # Determine tags from payer category
            specialty = llm_metadata.get("specialty_specific")
            diagnosis = llm_metadata.get("diagnosis_specific")
            tags = []
            if specialty:
                tags.append("Specialty-Specific")
            if diagnosis:
                tags.append("Diagnosis-Specific")

            # Get effective date
            effective_date = llm_metadata.get("effective_date", "N/A")

            # Extract evidence from first result
            evidence = "No specific policy rules documented"
            if policy_results:
                first_result = policy_results[0]
                evidence = (first_result.get("payor_level_summary") or
                           first_result.get("specific_rule_text") or
                           first_result.get("authorization_requirements", evidence))

            # Get policy URL
            policy_url = policy_metadata.get("external_link") or policy_metadata.get("policy_link", "")

            # Aggregate structured edit-rule facts across this policy's rules.
            # Surfaces verbatim codes/modifiers/limits in the API output so the
            # UI can render chips alongside the human-readable evidence string.
            edit_rule_facts = self._aggregate_edit_rule_facts(policy_results)

            # Contamination block attached by validate_policies_node
            # (severity = "quarantine" | "warn" | None). Riding it through
            # individual_policies means downstream nodes and the API
            # response see the same quarantine decision.
            contamination = policy_result.get("contamination")

            # Build individual policy entry
            policy_entry = {
                "policy_id": policy_result.get("PLCY_ID"),
                "payer_name": payer_name,
                "policy_title": policy_title,
                "tags": tags,
                "effective_date": effective_date,
                "evidence": evidence,
                "policy_url": policy_url,
                "edit_rule_facts": edit_rule_facts,
                "contamination": contamination,
                "results": policy_results,  # Keep for aggregation
                "metadata": llm_metadata
            }

            # Quarantined policies are removed from the API output
            # entirely — they don't appear in individual_policies, the
            # summary table, or the recommendation. The aggregate counts
            # and policy_id:title pairs survive in
            # `policy_contamination_summary` (graph state) and surface in
            # validation.warnings so reviewers can still see what was
            # caught and why, without polluting the UI.
            if contamination and contamination.get("severity") == "quarantine":
                self.logger.info(
                    f"Output: dropping quarantined "
                    f"policy_id={policy_result.get('PLCY_ID')!r} "
                    f"title={policy_title!r} payer={payer_name!r} "
                    f"flags={contamination.get('flags')} "
                    f"reasons={contamination.get('reasons')}"
                )
                continue

            individual_policies.append(policy_entry)
            policies_by_payer[payer_name].append(policy_entry)

        # Build dynamic summary table using comprehensive notebook logic
        if not table_structure or not table_structure.columns:
            # Fallback if no dynamic columns generated
            self.logger.warning("No dynamic columns generated, using simple table")
            summary_table = {
                "title": "Payer Policy Summary",
                "subtitle": pattern.get("top_pattern") or pattern.get("pattern_title", "Pattern Analysis"),
                "columns": [
                    {"id": "payer_org", "label": "Payer Organization", "type": "text"},
                    {"id": "policy_effective_date", "label": "Policy Effective Date", "type": "date"}
                ],
                "rows": []
            }
        else:
            # Build columns: payer_org + dynamic + appeals + effective_date (notebook order)
            dynamic_columns = [col.model_dump() for col in table_structure.columns]
            BASE_COLUMNS = [
                {"id": "payer_org", "label": "Payer Organization", "type": "text"},
                {"id": "appeals_process", "label": "Appeals Process\n(Documented)", "type": "badge"},
                {"id": "policy_effective_date", "label": "Policy Effective Date\n(Last Updated)", "type": "date"},
            ]
            columns = [BASE_COLUMNS[0]] + dynamic_columns + [BASE_COLUMNS[1], BASE_COLUMNS[2]]

            # Build rows with per-cell LLM summarization (notebook logic)
            summary_rows = []
            for payer_name in sorted(policies_by_payer.keys()):
                payer_policies = policies_by_payer[payer_name]
                row = {"payer_org": payer_name}

                # Fixed badges/dates (notebook helpers)
                row["appeals_process"] = self._appeals_badge(payer_policies)
                row["policy_effective_date"] = self._latest_effective_date(payer_policies)

                # Dynamic columns with per-cell LLM summarization
                for col_mapping in table_structure.selected_categories:
                    col_def = next((c for c in dynamic_columns if c["id"] == col_mapping.id), None)
                    if not col_def:
                        continue

                    # Aggregate rule texts for these categories
                    rule_texts = self._aggregate_rule_texts(payer_policies, col_mapping.categories)

                    # LLM summarization per cell (notebook logic)
                    row[col_mapping.id] = self._summarize_column_value(
                        payer_name,
                        col_def["label"],
                        rule_texts,
                        col_def.get("type", "text"),
                        llm
                    )

                summary_rows.append(row)

            # Build summary table structure
            summary_table = {
                "title": "Payer Policy Summary",
                "subtitle": pattern.get("top_pattern") or pattern.get("pattern_title", "Pattern Analysis"),
                "columns": columns,
                "rows": summary_rows
            }

        # Clean up individual policies (remove temporary fields)
        for policy in individual_policies:
            policy.pop("results", None)
            policy.pop("metadata", None)

        # Build final formatted output
        formatted_output = {
            "summary_table": summary_table,
            "individual_policies": individual_policies
        }

        # Add pattern_rank if in single-pattern mode
        pattern_rank = state.get("pattern_rank")
        if pattern_rank is not None:
            formatted_output["pattern_rank"] = pattern_rank

        self.logger.info(
            f"Formatted {len(individual_policies)} policies into {len(summary_table.get('rows', []))} payer rows "
            f"with {len(summary_table.get('columns', []))} columns (using comprehensive notebook logic)"
        )

        return {"formatted_output": formatted_output}

    def generate_recommendation_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 6: Generate pattern-specific policy recommendations (structured list).

        Uses pattern context + Elevance summary + policy summary table to generate
        executive-facing recommendations via LLM with structured output.
        Uses notebook's logic with rank 1, other fields empty for now.

        Args:
            state: Current graph state

        Returns:
            Updated state with recommended_action as List[Dict]
        """
        pattern = state.get("pattern")
        pattern_rank = state.get("pattern_rank")
        formatted_output = state["formatted_output"]
        llm = state["llm"]

        # Only generate recommendations if in single-pattern mode
        if pattern is None or pattern_rank is None:
            self.logger.info("Skipping recommendation generation (not in single-pattern mode)")
            return {"recommended_action": []}

        # Generate Elevance executive summary first (needed for recommendations).
        # Source from formatted_output.individual_policies, not raw state["result"]:
        # the extractor LLM doesn't return a payer name on its response, so the
        # raw result has no `payer_name`/`payor`/`PAYOR_NM` field for the
        # Elevance filter to match against. Payer attribution is attached during
        # format_output_node via a PLCY_ID join with filtered_policies — so the
        # formatted entries are the first place where `payer_name` is reliably
        # populated. Quarantined policies are already excluded from
        # individual_policies, so no extra filter is needed here.
        drg_codes = state.get("drg_codes", [])
        search_keywords = state.get("search_keywords", "")
        elevance_lookup_policies = formatted_output.get("individual_policies", []) or []
        user_query = state.get("query")

        elevance_summary = self._generate_elevance_executive_summary(
            pattern=pattern,
            pattern_rank=pattern_rank,
            drg_codes=drg_codes,
            keyword=search_keywords,
            pattern_policies=elevance_lookup_policies,
            llm=llm,
            user_query=user_query,
        )

        # Pivot the formatted individual_policies (each already carrying
        # `edit_rule_facts`) by payer, then derive per-payer aggregates and
        # cross-payer benchmarks to feed the recommendation prompt.
        # Only QUARANTINE-severity policies are excluded — those are the
        # ones where the relevance judge said the policy is topically
        # irrelevant or the identity/leak checks tripped at quarantine
        # severity. Warn-severity policies (e.g. cross-payer leak where
        # own payer is also mentioned) still inform peer benchmarks; the
        # warn flag stays attached for reviewer visibility in the API
        # response.
        from collections import defaultdict
        policies_by_payer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        excluded = 0
        for policy in formatted_output.get("individual_policies", []) or []:
            contamination = policy.get("contamination") or {}
            severity = contamination.get("severity")
            if severity == "quarantine":
                excluded += 1
                self.logger.info(
                    f"Recommendation: excluding {severity}-severity "
                    f"policy_id={policy.get('policy_id')!r} "
                    f"title={policy.get('policy_title')!r} "
                    f"payer={policy.get('payer_name')!r} "
                    f"flags={contamination.get('flags')}"
                )
                continue
            payer_name = policy.get("payer_name") or "Unknown Payer"
            policies_by_payer[payer_name].append(policy)
        if excluded:
            self.logger.info(
                f"Recommendation: excluded {excluded} quarantined policy/ies "
                f"from peer benchmarks"
            )
        specific_facts = self._collect_payer_edit_rule_facts(policies_by_payer)

        recommendations = self._generate_policy_recommendations(
            pattern_context=pattern,
            pattern_rank=pattern_rank,
            summary_table=formatted_output.get("summary_table", {}),
            elevance_summary=elevance_summary,
            llm=llm,
            user_query=user_query,
            specific_facts=specific_facts,
        )

        # Attach transient validator inputs to each rec so
        # `validate_recommendation_node` can re-resolve citations and re-render
        # description/evidence/citation after filtering. Popped before
        # `extract_result` builds the API response.
        citation_index = {
            c["id"]: c
            for c in (specific_facts.get("citations") or [])
            if isinstance(c, dict) and isinstance(c.get("id"), str)
        }
        for rec in recommendations:
            rec["_citation_index"] = citation_index
            rec["_specific_facts"] = specific_facts

        self.logger.info(
            f"Generated {len(recommendations)} recommendation(s) for pattern {pattern_rank}"
        )

        # Return both recommendations and elevance summary for extract_result
        return {
            "recommended_action": recommendations,
            "elevance_executive_summary": elevance_summary
        }

    def validate_recommendation_node(self, state: ReimbursementState) -> Dict[str, Any]:
        """
        Node 7: Critique the generated recommendation against the pattern and
        the cited sources, and strip items that fail the balanced
        drop-threshold (scope drift OR source-disclaims-target).

        Always emits `recommendation_validation` so `extract_result` can
        surface a status in the validation block. The recommendation itself
        is either left alone (all items aligned), pruned in place (some
        items dropped — description / evidence / citation re-rendered over
        survivors), or fully suppressed (all items dropped → empty list).
        """
        recommendations = state.get("recommended_action") or []
        if not recommendations:
            self.logger.info("Validator skipped — no recommendation to validate")
            return {
                "recommended_action": [],
                "recommendation_validation": {
                    "decision": "skipped",
                    "summary": "no recommendation to validate",
                    "dropped_count": 0,
                    "kept_count": 0,
                    "verdicts": [],
                }
            }

        llm = state["llm"]
        pattern = state.get("pattern") or {}
        user_query = state.get("query")
        formatted_output = state.get("formatted_output") or {}
        individual_policies = formatted_output.get("individual_policies") or []
        subtitle = (formatted_output.get("summary_table") or {}).get("subtitle", "")

        new_recs: List[Dict[str, Any]] = []
        # Aggregate verdicts across all recs (today there's at most one rec
        # per pattern, but keep the loop honest for future shape changes).
        all_verdicts: List[Dict[str, Any]] = []
        total_dropped = 0
        total_kept = 0
        summary_lines: List[str] = []
        any_items_dropped = False

        for rec_idx, rec in enumerate(recommendations):
            items = rec.pop("_items", None)
            citation_index = rec.pop("_citation_index", None) or {}
            specific_facts = rec.pop("_specific_facts", None)

            if not items:
                # Nothing to validate — keep the rec as-is.
                new_recs.append(rec)
                continue

            verdict_data = self._run_recommendation_validator(
                items=items,
                citation_index=citation_index,
                individual_policies=individual_policies,
                pattern=pattern,
                user_query=user_query,
                summary_table_subtitle=subtitle,
                llm=llm,
            )
            verdicts = verdict_data.get("verdicts") or []
            summary = verdict_data.get("summary") or ""

            # Default-keep when the validator returns no usable verdicts —
            # never silently drop a recommendation on a parser hiccup.
            drop_indices = {
                v["item_index"] for v in verdicts if v["decision"] == "drop"
            }
            kept_items: List[Dict[str, Any]] = []
            for idx, item in enumerate(items):
                if idx in drop_indices:
                    continue
                kept_items.append(item)

            dropped = len(items) - len(kept_items)
            total_dropped += dropped
            total_kept += len(kept_items)
            # Tag verdicts with rec_idx for downstream inspection.
            for v in verdicts:
                v_tagged = dict(v)
                v_tagged["rec_index"] = rec_idx
                all_verdicts.append(v_tagged)
            if summary:
                summary_lines.append(summary)

            if dropped > 0:
                any_items_dropped = True
                self.logger.info(
                    f"Validator dropped {dropped} item(s) from recommendation "
                    f"{rec_idx}: {summary or '(no summary)'}"
                )

            if not kept_items:
                # Full drop — suppress this rec entirely.
                continue

            # Re-render description and rebuild evidence/citation/peer over
            # the survivors so chips don't reference dropped items' sources.
            filtered_rec = {
                "headline": rec.get("headline"),
                "items": kept_items,
            }
            rec["description"] = self._render_recommendation_description(filtered_rec)
            evidence, peer_benchmarking, citation = self._evidence_and_citation_from_items(
                items=kept_items,
                citation_index=citation_index,
                specific_facts=specific_facts,
                has_citations=bool(citation_index),
            )
            rec["evidence"] = evidence
            rec["peer_benchmarking"] = peer_benchmarking
            rec["citation"] = citation
            new_recs.append(rec)

        if not new_recs:
            decision = "suppressed"
        elif any_items_dropped:
            decision = "items_dropped"
        else:
            decision = "ok"

        return {
            "recommended_action": new_recs,
            "recommendation_validation": {
                "decision": decision,
                "summary": " | ".join(summary_lines) if summary_lines else "all items aligned with pattern",
                "dropped_count": total_dropped,
                "kept_count": total_kept,
                "verdicts": all_verdicts,
            },
        }

    # ========================================================================
    # Token Tracking Helpers
    # ========================================================================

    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        """
        Pull input/output token counts from a LangChain LLM response.

        Handles both new-style (`usage_metadata`) and legacy
        (`response_metadata['token_usage']`) shapes. Returns zeros if neither
        is present so callers never need to guard.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return {
                "input": int(usage.get("input_tokens") or 0),
                "output": int(usage.get("output_tokens") or 0),
            }
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or {}
        return {
            "input": int(token_usage.get("prompt_tokens") or 0),
            "output": int(token_usage.get("completion_tokens") or 0),
        }

    def _record_tokens(self, category: str, response: Any) -> None:
        """Record tokens from one LLM call into the per-category breakdown.
        Thread-safe under worker-pool fan-out (extract_rules / relevance
        judge) via self._token_lock."""
        try:
            tokens = self._extract_token_usage(response)
        except Exception as exc:  # defensive — never let token bookkeeping break a node
            self.logger.debug(f"Token extraction failed for {category}: {exc}")
            return
        lock = getattr(self, "_token_lock", None)
        if lock is None:
            # Old test instances built with __new__ may not have the lock;
            # fall back to lockless update — fine in single-threaded tests.
            lock = threading.Lock()
            self._token_lock = lock
        with lock:
            entry = self._token_breakdown.setdefault(
                category, {"input": 0, "output": 0, "calls": 0}
            )
            entry["input"] += tokens["input"]
            entry["output"] += tokens["output"]
            entry["calls"] += 1

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _collect_rule_categories(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Collect all non-empty rule categories across all policies.

        Args:
            results: List of policy extraction results

        Returns:
            Dictionary mapping category names to list of sample values
        """
        category_samples = {
            "site_of_service": [],
            "bundling_logic": [],
            "code_interactions": [],
            "modifier_usage": [],
            "denial_conditions": [],
            "unit_pricing_logic": [],
            "documentation_requirements": [],
            "evidence_summary": []
        }

        for policy_result in results:
            if policy_result is None:
                continue

            policy_rules = policy_result.get("results", [])
            for rule in policy_rules:
                for category in category_samples.keys():
                    value = (rule.get(category) or "").strip()
                    if value and value not in category_samples[category]:
                        category_samples[category].append(value)

        # Filter out empty categories
        return {k: v for k, v in category_samples.items() if v}

    def _select_rule_categories(
        self,
        category_samples: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Select 1-2 most relevant rule categories and combine related ones.

        Args:
            category_samples: Dictionary of category names to sample values

        Returns:
            List of category combinations to use as columns
        """
        # Score categories by variation (more unique values = more informative)
        category_scores = {
            cat: len(values) for cat, values in category_samples.items()
        }

        # Predefined category combinations
        combinations = {
            "denial_summary": ["denial_conditions", "evidence_summary"],
            "bundling_rules": ["bundling_logic", "code_interactions"],
            "service_restrictions": ["site_of_service", "modifier_usage"],
            "documentation_billing": ["documentation_requirements", "unit_pricing_logic"]
        }

        # Score combinations based on constituent categories
        combination_scores = {}
        for combo_name, categories in combinations.items():
            score = sum(category_scores.get(cat, 0) for cat in categories)
            if score > 0:  # Only include if at least one category has data
                combination_scores[combo_name] = {
                    "score": score,
                    "categories": [c for c in categories if c in category_samples]
                }

        # Select top 1-2 combinations
        sorted_combos = sorted(
            combination_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        selected = []
        for combo_name, combo_data in sorted_combos[:2]:  # Max 2 columns
            selected.append({
                "id": combo_name,
                "categories": combo_data["categories"]
            })

        return selected

    def _generate_column_labels(
        self,
        selected_categories: List[Dict[str, Any]],
        category_samples: Dict[str, List[str]],
        llm: Any
    ) -> List[Dict[str, str]]:
        """
        Use LLM to generate concise column labels with Pydantic validation.

        Args:
            selected_categories: Selected category combinations
            category_samples: Sample values for each category
            llm: LLM client

        Returns:
            List of column metadata with id, label, type
        """
        if not selected_categories:
            return []

        # Build samples for LLM
        samples_text = ""
        for cat_combo in selected_categories:
            cat_id = cat_combo["id"]
            categories = cat_combo["categories"]
            samples_text += f"\n{cat_id} (combines: {', '.join(categories)}):\n"
            for cat in categories[:2]:  # Show max 2 samples per category
                if cat in category_samples and category_samples[cat]:
                    samples_text += f"  - {category_samples[cat][0][:80]}...\n"

        prompt = f"""Given these policy rule categories and sample content:
{samples_text}

Generate concise column labels (max 5 words each) for a summary table.
Labels should be clear, professional, and suitable for table headers.

IMPORTANT: Output ONLY valid JSON, no markdown code fences.

Output JSON:
{{
  "columns": [
    {{"id": "category_id", "label": "Column Label (max 5 words)", "type": "text"}}
  ]
}}
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            content = response.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])

            # Parse JSON
            data = json.loads(content)

            # Validate with Pydantic
            validated = ColumnLabelsResponse(**data)

            # Return as list of dicts
            return [col.model_dump() for col in validated.columns]

        except (json.JSONDecodeError, ValidationError) as e:
            if isinstance(e, ValidationError):
                self._handle_validation_error(e, "column label generation")
            else:
                self.logger.warning(f"Failed to parse column labels JSON: {e}")

            # Fallback to simple labels
            self.logger.info("Using fallback column labels")
            return [
                {
                    "id": cat["id"],
                    "label": cat["id"].replace("_", " ").title(),
                    "type": "text"
                }
                for cat in selected_categories
            ]

    def _handle_validation_error(
        self,
        error: ValidationError,
        context: str
    ) -> None:
        """
        Log validation errors with structured information.

        Args:
            error: Pydantic ValidationError
            context: Context description (e.g., "policy extraction")
        """
        self.logger.error(f"Validation error in {context}:")
        for err in error.errors():
            field = " -> ".join(str(loc) for loc in err['loc'])
            self.logger.error(f"  Field: {field}")
            self.logger.error(f"  Error: {err['msg']}")
            self.logger.error(f"  Input: {err.get('input', 'N/A')}")

    def _aggregate_payer_rules(
        self,
        payer_policies: List[Dict[str, Any]],
        category_id: str,
        categories: List[str],
        llm: Any
    ) -> str:
        """
        Aggregate rules for a payer across multiple policies with Pydantic validation.

        Args:
            payer_policies: List of policies for this payer
            category_id: Category combination ID
            categories: List of rule categories to aggregate
            llm: LLM client

        Returns:
            Aggregated rule text (max 15 words)
        """
        # Collect all rule texts for these categories
        rule_texts = []
        for policy in payer_policies:
            if policy is None:
                continue
            policy_results = policy.get("results", [])
            for result in policy_results:
                for cat in categories:
                    text = result.get(cat, "") or ""  # Handle None values
                    text = text.strip() if isinstance(text, str) else ""
                    if text and text not in rule_texts:
                        rule_texts.append(text)

        if not rule_texts:
            return "-"

        if len(rule_texts) == 1:
            return rule_texts[0]

        # Multiple rules - use LLM to summarize
        payer_name = payer_policies[0].get("payer_name", "Unknown")
        rules_list = "\n".join([f"- {text}" for text in rule_texts[:5]])  # Max 5 rules

        prompt = f"""Summarize these policy rules for {payer_name} into ONE concise statement (max 15 words):

{rules_list}

IMPORTANT: Output only the summary text. No JSON, no markdown, just the plain text summary."""

        messages = [
            {"role": "system", "content": "You are a concise policy summarizer. Output only the requested summary text."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            summary_text = response.content.strip()

            # Validate with Pydantic (will auto-truncate if needed)
            validated = RuleSummary(summary=summary_text)

            return validated.summary

        except ValidationError as e:
            self._handle_validation_error(e, f"rule summarization for {payer_name}")
            # Fallback: return first rule (may exceed word limit but at least has content)
            return rule_texts[0]
        except Exception as e:
            self.logger.warning(f"Failed to summarize rules for {payer_name}: {e}")
            # Fallback: return first rule
            return rule_texts[0]

    def _extract_cpt_codes_from_context(
        self,
        context: Dict[str, Any],
        query: Optional[str] = None
    ) -> str:
        """
        Extract CPT codes from orchestrator context.

        Searches in order:
        1. Filters with field="cpt_code" or "procedure_code"
        2. Drill path for CPT-like values
        3. Query text using regex pattern

        Args:
            context: Context from orchestrator
            query: User query (optional)

        Returns:
            Comma-separated CPT codes
        """
        import re

        cpt_codes = set()

        # 1. Check filters in intent
        intent = context.get("intent", {})
        filters = intent.get("filters", [])

        for filter_obj in filters:
            field = filter_obj.get("field", "").lower()
            if "cpt" in field or "procedure" in field or "hcpcs" in field:
                value = str(filter_obj.get("value", ""))
                # Extract numeric codes (CPT codes are typically 5 digits)
                codes = re.findall(r'\b\d{5}\b', value)
                cpt_codes.update(codes)

        # 2. Check drill path in correlation_summary
        correlation = context.get("correlation_summary", {})
        drill_path = correlation.get("drill_path", [])

        for drill_item in drill_path:
            if isinstance(drill_item, dict):
                # Check if dimension field indicates CPT/procedure codes
                dimension = drill_item.get("dimension", "")
                if "cpt" in dimension.lower() or "procedure" in dimension.lower() or "hcpcs" in dimension.lower():
                    segments = drill_item.get("top_segments", []) or []
                    for segment in segments:
                        if not isinstance(segment, dict):
                            continue
                        value = str(segment.get("value", ""))
                        codes = re.findall(r'\b\d{5}\b', value)
                        cpt_codes.update(codes)
                    if not segments:
                        legacy_value = str(drill_item.get("value", ""))
                        codes = re.findall(r'\b\d{5}\b', legacy_value)
                        cpt_codes.update(codes)

                # Also check all key-value pairs for CPT-related content
                for key, value in drill_item.items():
                    if "cpt" in key.lower() or "procedure" in key.lower() or "hcpcs" in key.lower():
                        codes = re.findall(r'\b\d{5}\b', str(value))
                        cpt_codes.update(codes)

        # 3. Check query text if provided
        if query:
            codes = re.findall(r'\b\d{5}\b', query)
            cpt_codes.update(codes)

        # 4. Check raw_question in intent
        raw_question = intent.get("raw_question", "")
        if raw_question:
            codes = re.findall(r'\b\d{5}\b', raw_question)
            cpt_codes.update(codes)

        if not cpt_codes:
            self.logger.warning(
                "No CPT codes found in context. Checked filters, drill_path, and query."
            )
            return ""

        result = ",".join(sorted(cpt_codes))
        self.logger.info(f"Extracted CPT codes from context: {result}")
        return result

    def _extract_drg_codes_from_cards(
        self,
        source_card_ids: List[str],
        cards: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract DRG codes from source cards (from notebook logic).

        Extracts from:
        1. filters where field == 'drg_name'
        2. source_entity.name where source_entity.type == 'drgs'

        Args:
            source_card_ids: List of card IDs to extract from
            cards: Full list of cards

        Returns:
            Sorted list of unique DRG names
        """
        card_lookup = {card.get('card_id'): card for card in cards if card.get('card_id')}
        drgs = set()

        for card_id in source_card_ids:
            card = card_lookup.get(card_id)
            if not card:
                continue

            # Method 1: Check filters for drg_name
            filters = card.get('filters', [])
            for filter_item in filters:
                if isinstance(filter_item, dict) and filter_item.get('field') == 'drg_name':
                    drg_value = filter_item.get('value')
                    if drg_value:
                        drgs.add(drg_value)

            # Method 2: Check source_entity for DRG
            source_entity = card.get('source_entity', {})
            if isinstance(source_entity, dict):
                if source_entity.get('type') == 'drgs':
                    drg_name = source_entity.get('name')
                    if drg_name:
                        drgs.add(drg_name)

        return sorted(list(drgs))

    def _extract_cpt_codes_from_cards(
        self,
        source_card_ids: List[str],
        cards: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract CPT codes from source cards (simplified from notebook pattern).

        Extracts from:
        1. filters where field contains 'cpt', 'procedure', or 'hcpcs'
        2. source_entity.name where source_entity.type == 'procedures'

        Args:
            source_card_ids: List of card IDs to extract from
            cards: Full list of cards

        Returns:
            Sorted list of unique CPT codes
        """
        import re

        card_lookup = {card.get('card_id'): card for card in cards if card.get('card_id')}
        cpt_codes = set()

        for card_id in source_card_ids:
            card = card_lookup.get(card_id)
            if not card:
                continue

            # Method 1: Check filters for CPT/procedure codes
            filters = card.get('filters', [])
            for filter_item in filters:
                if not isinstance(filter_item, dict):
                    continue
                field = filter_item.get('field', '').lower()
                if 'cpt' in field or 'procedure' in field or 'hcpcs' in field:
                    value = str(filter_item.get('value', ''))
                    codes = re.findall(r'\b\d{5}\b', value)
                    cpt_codes.update(codes)

            # Method 2: Check source_entity for procedures
            source_entity = card.get('source_entity', {})
            if isinstance(source_entity, dict):
                if source_entity.get('type') == 'procedures':
                    proc_name = source_entity.get('name', '')
                    codes = re.findall(r'\b\d{5}\b', proc_name)
                    cpt_codes.update(codes)

        return sorted(list(cpt_codes))

    def _extract_cpt_codes_from_pattern(
        self,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Extract CPT codes from pattern using notebook's proven logic.

        Priority:
        1. pattern.cpt_codes (if explicitly provided)
        2. Extract from source cards using _extract_cpt_codes_from_cards()

        Args:
            pattern: Pattern structure from pattern agent
            context: Full context with cards

        Returns:
            Comma-separated CPT codes
        """
        # 1. Check if pattern has explicit CPT codes
        if "cpt_codes" in pattern:
            explicit_codes = pattern["cpt_codes"]
            if isinstance(explicit_codes, list) and explicit_codes:
                result = ",".join(sorted(explicit_codes))
                self.logger.info(f"Using explicit CPT codes from pattern: {result}")
                return result

        # 2. Extract from source cards
        source_card_ids = pattern.get("source_card_ids", [])
        cards = context.get("cards", [])

        if source_card_ids and cards:
            cpt_codes = self._extract_cpt_codes_from_cards(source_card_ids, cards)
            if cpt_codes:
                result = ",".join(cpt_codes)
                self.logger.info(f"Extracted CPT codes from {len(source_card_ids)} cards: {result}")
                return result

        self.logger.warning(
            f"No CPT codes found in pattern {pattern.get('pattern_rank')}. "
            f"Checked pattern.cpt_codes and source cards."
        )
        return ""

    def _extract_lob_and_state_filters(
        self,
        pattern: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """
        Extract LOB and State filters from pattern using notebook's comprehensive logic.

        Extraction strategy (matching notebook):
        1. States: Extract from priority_entities.states
        2. LOB: Extract from source cards (dimensions, canonical_dimensions, filters)
           - dimensions.lob_description
           - canonical_dimensions.line_of_business
           - filters where field == 'lob_description'

        Args:
            pattern: Pattern structure from pattern agent
            context: Full context with cards

        Returns:
            Tuple of (lob_list, state_list) as sorted lists
        """
        lobs = set()
        states = set()

        # Method 1: Extract States from priority_entities
        priority_entities = pattern.get('priority_entities', {})
        if priority_entities:
            pattern_states = priority_entities.get('states', [])
            if pattern_states:
                states.update(pattern_states)

        # Method 2: Extract LOB from source cards
        source_card_ids = pattern.get('source_card_ids', [])
        cards = context.get('cards', [])

        if source_card_ids and cards:
            card_lookup = {card.get('card_id'): card for card in cards if card.get('card_id')}

            for card_id in source_card_ids:
                card = card_lookup.get(card_id)
                if not card:
                    continue

                # Extract from dimensions
                dimensions = card.get('dimensions', {})
                if dimensions:
                    lob_val = dimensions.get('lob_description')
                    if lob_val:
                        lobs.add(lob_val)

                    # Also extract state from dimensions as fallback
                    state_val = dimensions.get('service_area_state')
                    if state_val:
                        states.add(state_val)

                # Extract from canonical_dimensions
                canonical_dims = card.get('canonical_dimensions', {})
                if canonical_dims:
                    canonical_lobs = canonical_dims.get('line_of_business', [])
                    if canonical_lobs:
                        lobs.update(canonical_lobs)

                    # Also extract state from canonical as fallback
                    canonical_states = canonical_dims.get('geography', [])
                    if canonical_states:
                        states.update(canonical_states)

                # Extract from filters
                filters = card.get('filters', [])
                for filter_item in filters:
                    if not isinstance(filter_item, dict):
                        continue

                    field = filter_item.get('field')
                    value = filter_item.get('value')

                    if field == 'lob_description' and value:
                        lobs.add(value)
                    elif field == 'service_area_state' and value:
                        states.add(value)

        lob_list = sorted(list(lobs))
        state_list = sorted(list(states))

        pattern_rank = pattern.get('pattern_rank', '?')
        self.logger.info(
            f"Pattern {pattern_rank}: Extracted LOB={lob_list} from {len(source_card_ids)} cards, "
            f"State={state_list} from priority_entities + cards"
        )

        return lob_list, state_list

    def _generate_search_keywords_from_drg(
        self,
        drg_codes: List[str],
        llm: Any,
        user_query: Optional[str] = None
    ) -> str:
        """
        Generate smart search keywords from DRG codes using LLM (from notebook).

        Uses LLM to identify the top 2 most relevant keywords that best describe
        the DRG codes for searching reimbursement policies. This significantly
        reduces the number of irrelevant policies retrieved.

        Args:
            drg_codes: List of DRG code descriptions
            llm: LLM client for keyword generation
            user_query: Optional user question used only as a tie-breaker signal;
                DRG descriptions remain the primary input.

        Returns:
            Comma-separated keywords (top 2) or first DRG code as fallback
        """
        if not drg_codes:
            return ""

        # Always use LLM for keyword generation (even for single DRG)
        # Raw DRG descriptions can be too long/specific for API search
        import json
        drg_codes_str = json.dumps(drg_codes)

        # Optional user-question framing — supplementary signal only. DRG codes
        # remain primary; the user question is for tie-breaking emphasis.
        cleaned_query = self._clean_user_query(user_query)
        user_question_block = (
            f"\nUser question (use ONLY as a tie-breaker to bias between equally-relevant DRG themes; "
            f"do not derive keywords from this alone):\n{cleaned_query}\n"
            if cleaned_query else ""
        )

        keyword_prompt = f"""Given the following DRG code descriptions: {drg_codes_str}
{user_question_block}
Identify the TOP 2 most relevant keywords that best describe what these codes relate to for searching reimbursement policies.
The keywords should be specific enough to find relevant policies.
Do not use special characters in the answer, it should only be words and phrases.
Rank them in the order of relevance. Avoid generic terms that may lead to irrelevant results, focus on medical terms that are specific to the DRG codes.
Each keyword should be a single word or a short phrase (1-2 words).
Find common theme across the DRG description. Do not repeat a theme.
Avoid generic terms such as "MS DRG", "surgical DRG", "inpatient surgery" which can lead to match with lot of reimbursement policies.

Return ONLY the TOP 2 keywords as a comma-separated list, nothing else. Example: "chemotherapy, bowel surgery" """

        keyword_messages = [
            {"role": "system", "content": "You are a medical coding expert. Generate concise search keywords."},
            {"role": "user", "content": keyword_prompt}
        ]

        try:
            keyword_response = self._invoke_llm_with_retry(llm, keyword_messages)
            self._record_tokens("search_keywords", keyword_response)
            search_keywords_raw = keyword_response.content.strip().strip('"').strip("'")

            # Split by comma and take only first 2
            keywords_list = [k.strip() for k in search_keywords_raw.split(',')]
            search_keywords = ', '.join(keywords_list[:2])

            self.logger.info(f"Generated keywords from {len(drg_codes)} DRG codes: '{search_keywords}' (LLM)")
            return search_keywords

        except Exception as e:
            self.logger.warning(f"LLM keyword generation failed: {e}, extracting keywords from DRG")
            # Fallback: Extract first few meaningful words from first DRG (avoid raw long description)
            fallback_drg = drg_codes[0]
            # Take first 2-3 significant words, remove common medical suffixes
            words = fallback_drg.replace('without', '').replace('with', '').split()
            # Filter out noise words and take first 2 meaningful terms
            meaningful = [w for w in words if len(w) > 2 and w.lower() not in {'the', 'and', 'for'}]
            fallback = ' '.join(meaningful[:2]) if meaningful else drg_codes[0]
            self.logger.info(f"Fallback keywords from DRG: '{fallback}'")
            return fallback

    def _generate_search_keywords(
        self,
        drg_codes: List[str],
        cpt_codes: str,
        llm: Any = None,
        user_query: Optional[str] = None
    ) -> str:
        """
        Generate search keywords from DRG codes (preferred) or CPT codes.

        DRG codes are prioritized because they're more specific for policy search.
        Uses LLM to generate smart keywords from multiple DRG codes.

        Args:
            drg_codes: List of DRG codes from pattern
            cpt_codes: Comma-separated CPT codes
            llm: Optional LLM for smart keyword generation from DRG codes
            user_query: Optional user question forwarded to the DRG keyword
                generator as a tie-breaker signal.

        Returns:
            Search keywords string for policy API
        """
        # Prioritize DRG codes for search with LLM-based keyword generation
        if drg_codes and len(drg_codes) > 0 and llm:
            return self._generate_search_keywords_from_drg(drg_codes, llm, user_query=user_query)
        elif drg_codes and len(drg_codes) > 0:
            # No LLM, extract keywords from first DRG (avoid raw long description)
            primary_drg = drg_codes[0]
            words = primary_drg.replace('without', '').replace('with', '').split()
            meaningful = [w for w in words if len(w) > 2 and w.lower() not in {'the', 'and', 'for'}]
            keywords = ' '.join(meaningful[:2]) if meaningful else primary_drg
            self.logger.info(f"Using extracted keywords from first DRG: {keywords}")
            return keywords

        # Fallback to CPT codes
        if cpt_codes:
            # Use first CPT code
            first_cpt = cpt_codes.split(",")[0].strip()
            self.logger.info(f"Using CPT code as search keyword: {first_cpt}")
            return first_cpt

        self.logger.warning("No DRG or CPT codes available for search keywords")
        return "Unknown"

    def _compact_summary_table(
        self,
        summary_table: Dict[str, Any],
        max_rows: int = 12,
        max_cell_chars: int = 120
    ) -> str:
        """
        Convert summary table to compact text representation for LLM (from notebook).

        Args:
            summary_table: Full summary table structure
            max_rows: Maximum rows to include
            max_cell_chars: Max characters per cell

        Returns:
            Compact JSON string with column labels and truncated values
        """
        columns = summary_table.get("columns", [])
        rows = summary_table.get("rows", [])

        if not columns or not rows:
            return json.dumps({"columns": [], "rows": []})

        # Build column labels mapping
        col_labels = {col["id"]: col.get("label", col["id"]) for col in columns}

        # Build compact rows (limit to first max_rows payers)
        compact_rows = []
        for row in rows[:max_rows]:
            compact_row = {}
            for col in columns:
                col_id = col["id"]
                value = row.get(col_id, "-")
                if value is None or value == "":
                    value = "-"
                if isinstance(value, str) and len(value) > max_cell_chars:
                    value = value[:max_cell_chars] + "..."
                compact_row[col_labels[col_id]] = value
            compact_rows.append(compact_row)

        # Format as JSON for LLM
        compact_data = {
            "columns": [col_labels[c["id"]] for c in columns],
            "rows": compact_rows
        }

        return json.dumps(compact_data, indent=2)

    def _format_facts_for_prompt(
        self,
        specific_facts: Optional[Dict[str, Any]]
    ) -> tuple[str, str, str]:
        """
        Render the structured per-payer facts, peer-benchmark facts, and
        per-fact citation index into three prompt blocks. Returns ('', '', '')
        when nothing to show, so the prompt can include the blocks
        unconditionally without leaving empty headers.

        The citation block is the authoritative grounding list — every
        recommendation item must cite ≥1 id from it (enforced by the parse
        step in `_generate_policy_recommendations`).
        """
        if not specific_facts:
            return "", "", ""

        per_payer = specific_facts.get("per_payer") or {}
        peer_benchmarks = specific_facts.get("peer_benchmarks") or []
        citations = specific_facts.get("citations") or []

        per_payer_lines: List[str] = []
        for payer_name in sorted(per_payer.keys()):
            facts = per_payer[payer_name] or {}
            kept = [(k, v) for k, v in facts.items() if v]
            if not kept:
                continue
            per_payer_lines.append(f"- {payer_name}:")
            for key, values in kept:
                rendered = ", ".join(values[:10])
                per_payer_lines.append(f"    {key}: {rendered}")

        per_payer_block = (
            "Specific facts per payer (cite verbatim — codes, modifiers, "
            "revenue codes, thresholds, limits, exemptions):\n"
            + "\n".join(per_payer_lines)
            if per_payer_lines else ""
        )

        peer_lines: List[str] = []
        for entry in peer_benchmarks[:20]:
            fact = entry.get("fact")
            payers = entry.get("payers") or []
            key = entry.get("fact_key")
            cids = entry.get("citation_ids") or []
            if fact and payers:
                cid_suffix = f" — citations: {', '.join(cids)}" if cids else ""
                peer_lines.append(
                    f"- [{key}] {fact} — payers: {', '.join(payers)}{cid_suffix}"
                )

        peer_block = (
            "Peer benchmark facts (shared by ≥2 payers — useful for 'UHC and "
            "Humana cap at 10 hrs/week' style cites):\n"
            + "\n".join(peer_lines)
            if peer_lines else ""
        )

        citation_lines: List[str] = []
        for entry in citations:
            cid = entry.get("id")
            payer = entry.get("payer")
            policy_title = entry.get("policy_title")
            fact_key = entry.get("fact_key")
            fact = entry.get("fact")
            if not (cid and payer and fact_key and fact):
                continue
            citation_lines.append(
                f"[{cid}] {payer} · \"{policy_title}\" → {fact_key}: \"{fact}\""
            )

        citation_block = (
            "Available citations (grounding for the recommendation — each "
            "items[].citations entry MUST be one of these ids):\n"
            + "\n".join(citation_lines)
            if citation_lines else ""
        )

        return per_payer_block, peer_block, citation_block

    def _render_recommendation_description(
        self,
        recommendation: Dict[str, Any]
    ) -> str:
        """
        Render the structured recommendation object the LLM returns into the
        final multi-line `description` string. Picks Style A (Recommendation <n>: edit-rule)
        or Style B (headline + IA lines) per item.kind.
        """
        headline = (recommendation.get("headline") or "").strip()
        items = recommendation.get("items") or []

        edit_items: List[Dict[str, Any]] = []
        ia_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = (item.get("kind") or "").strip().lower()
            if kind == "edit":
                edit_items.append(item)
            else:
                ia_items.append(item)

        lines: List[str] = []

        # Style A — edit-rule blocks numbered Recommendation 1/Recommendation 2/...
        for idx, item in enumerate(edit_items, start=1):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            line = f"Recommendation {idx}: {text}"
            scope = [s for s in (item.get("scope") or []) if s]
            if scope:
                line += f" Apply to {', '.join(scope)}."
            exemptions = [e for e in (item.get("exemptions") or []) if e]
            if exemptions:
                line += f" Exempt: {', '.join(exemptions)}."
            peer = (item.get("peer_cite") or "").strip()
            if peer:
                line += f" Peer: {peer}."
            lines.append(line)
            lines.append("")

        # Style B — headline + IA bullets
        if ia_items:
            if headline and not lines:
                lines.append(headline)
            elif headline and lines:
                # Mixed output: keep headline before IA section
                lines.append("")
                lines.append(headline)
            for item in ia_items:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                line = f"IA — {text}"
                scope = [s for s in (item.get("scope") or []) if s]
                if scope:
                    line += f" ({', '.join(scope)})"
                peer = (item.get("peer_cite") or "").strip()
                if peer:
                    line += f" {peer}."
                exemptions = [e for e in (item.get("exemptions") or []) if e]
                if exemptions:
                    line += f" Exempt: {', '.join(exemptions)}."
                lines.append(line)

        # Fallback: only headline, no items.
        if not lines and headline:
            lines.append(headline)

        return "\n".join(lines).strip()

    def _evidence_and_citation_from_items(
        self,
        items: List[Dict[str, Any]],
        citation_index: Dict[str, Dict[str, Any]],
        specific_facts: Optional[Dict[str, Any]],
        has_citations: bool,
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Build (evidence, peer_benchmarking, citation) lists from the given
        recommendation items. Used by `_generate_policy_recommendations` and
        re-used by `validate_recommendation_node` after pruning so the chips
        only reflect surviving items' citations.

        Args:
            items: Items that survived grounding / validation.
            citation_index: id -> source-record map from
                `_collect_payer_edit_rule_facts.citations`.
            specific_facts: Original facts payload — used only for the
                no-citation-index fallback (back-compat with legacy callers).
            has_citations: Whether the citation index was populated for this
                recommendation. When False, falls back to the legacy
                exemptions/scope-based evidence aggregation.

        Returns:
            (evidence, peer_benchmarking, citation) — three List[str] surfaces
            for the recommendation dict.
        """
        # Resolve each item's citation ids to the underlying source records,
        # then surface them as the recommendation's `evidence` (grounding)
        # and `citation` (per-policy URL list). `peer_benchmarking` keeps
        # the LLM-provided peer_cite prose for legibility in the UI.
        resolved_citations: List[Dict[str, Any]] = []
        seen_cids: set = set()
        for item in items:
            for cid in (item.get("citations") or []):
                if cid in seen_cids:
                    continue
                src = citation_index.get(cid)
                if not src:
                    continue
                seen_cids.add(cid)
                resolved_citations.append(src)

        evidence: List[str] = []
        for src in resolved_citations:
            line = (
                f'{src["payer"]} · {src["policy_title"]}: '
                f'"{src["fact"]}" [{src["fact_key"]}]'
            )
            if line not in evidence:
                evidence.append(line)

        peer_benchmarking: List[str] = []
        for item in items:
            peer = item.get("peer_cite")
            if isinstance(peer, str) and peer.strip() and peer not in peer_benchmarking:
                peer_benchmarking.append(peer.strip())

        # Citation: one entry per unique source policy, with URL when
        # available so the UI can render clickable references.
        citation: List[str] = []
        seen_policies: set = set()
        for src in resolved_citations:
            key = (src["payer"], src["policy_title"])
            if key in seen_policies:
                continue
            seen_policies.add(key)
            url = src.get("policy_url") or "(no URL)"
            citation.append(
                f'{src["payer"]} · {src["policy_title"]} — {url}'
            )

        # Fallback for the no-citation-index path (defensive — shouldn't
        # happen when has_citations is True). Preserves prior behaviour
        # so legacy callers without specific_facts still get something.
        if not has_citations:
            for item in items:
                for code in (item.get("exemptions") or []):
                    if isinstance(code, str) and code.strip() and code not in evidence:
                        evidence.append(code.strip())
                for scope in (item.get("scope") or []):
                    if isinstance(scope, str) and scope.strip() and scope not in evidence:
                        evidence.append(scope.strip())
            if specific_facts and not citation:
                for entry in (specific_facts.get("peer_benchmarks") or [])[:5]:
                    fact = entry.get("fact")
                    payers = entry.get("payers") or []
                    if fact and payers:
                        citation.append(f"{fact} — {', '.join(payers)}")

        return evidence, peer_benchmarking, citation

    def _generate_policy_recommendations(
        self,
        pattern_context: Dict[str, Any],
        pattern_rank: int,
        summary_table: Dict[str, Any],
        elevance_summary: Optional[str],
        llm: Any,
        user_query: Optional[str] = None,
        specific_facts: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a single, code-specific policy recommendation per pattern.

        The LLM emits a structured object containing a headline and a list of
        items (each kind="edit" for Style A or kind="immediate_action" for
        Style B). The agent renders those items into the multi-line
        `description` string and surfaces the supporting facts in `evidence`
        and `peer_benchmarking`.

        Args:
            pattern_context: Pattern data with title, details, etc.
            pattern_rank: Pattern identifier
            summary_table: Policy summary table with payer comparisons
            elevance_summary: Elevance executive summary (if explainable), or None
            llm: LLM client
            user_query: Optional user question for steering
            specific_facts: Output of `_collect_payer_edit_rule_facts` —
                {per_payer, peer_benchmarks}. When present, the prompt requires
                items[].text to cite at least one fact verbatim.

        Returns:
            List with a single recommendation dict (or [] if none warranted).
        """
        # Extract pattern info (notebook's _pattern_context helper)
        pattern_title = (
            pattern_context.get("top_pattern")
            or pattern_context.get("pattern_title")
            or f"Pattern {pattern_rank}"
        )
        pattern_evidence = (
            pattern_context.get("evidence_summary")
            or pattern_context.get("pattern_details")
            or pattern_context.get("why_it_matters")
            or ""
        )

        # Build compact table for LLM (notebook logic)
        compact_table = self._compact_summary_table(summary_table)

        # Render the structured fact blocks. When non-empty, the prompt will
        # require the LLM to cite from them verbatim AND tag each item with
        # citation ids drawn from the citation block.
        per_payer_block, peer_block, citation_block = self._format_facts_for_prompt(
            specific_facts
        )
        has_specifics = bool(per_payer_block or peer_block)
        has_citations = bool(citation_block)
        citation_index: Dict[str, Dict[str, Any]] = {
            c["id"]: c
            for c in ((specific_facts or {}).get("citations") or [])
            if isinstance(c, dict) and isinstance(c.get("id"), str)
        }

        # Optional user-question framing — included only when the query carries
        # specific intent (generic stubs are filtered out by _clean_user_query).
        cleaned_query = self._clean_user_query(user_query)
        user_question_block = (
            f"User question (frame the recommendation to address this if relevant):\n{cleaned_query}\n\n"
            if cleaned_query else ""
        )

        prompt = f"""You are generating a code-specific reimbursement policy recommendation.

{user_question_block}Pattern:
- Title: {pattern_title}
- Evidence: {pattern_evidence}

Elevance policy summary (if available):
{elevance_summary or "None"}

{per_payer_block}

{peer_block}

{citation_block}

Payer policy summary table (compact):
{compact_table}

Output JSON schema:
{{
  "has_recommendation": true|false,
  "recommendation": {{
    "headline": "1-sentence directive naming the target code/program",
    "items": [
      {{
        "kind": "edit" | "immediate_action",
        "text": "single concrete action — MUST cite ≥1 code/modifier/revenue code/threshold/numeric limit verbatim",
        "citations": ["C3", "C7"],
        "scope": ["COPPS"] or ["NC"] or ["all states"] or [],
        "peer_cite": "UHC and Humana cap at 10 hrs/week" or null,
        "exemptions": ["observation RC 0762"] or []
      }}
    ]
  }} or null
}}

Rules:
1) When the "Specific facts per payer" / "Peer benchmark facts" blocks contain
   data, every items[].text MUST cite at least one VERBATIM code, modifier,
   revenue code, threshold, or numeric limit drawn from those facts.
2) GROUNDING: When an "Available citations" block is present above, every
   items[].citations array MUST be non-empty and every id in it MUST be one
   of the listed citation ids (e.g. "C1", "C2", ...). Cite ONLY ids whose
   fact your items[].text actually relies on. Items without ≥1 valid
   citation id will be DROPPED downstream. If you cannot ground an action
   in ≥1 listed citation, omit that item rather than invent one. If no
   items survive grounding, set has_recommendation=false and
   recommendation=null.
3) Vague phrases like "tighter guardrails", "stricter edits", or "clearer
   documentation" are NOT acceptable when specifics are available. If you
   cannot cite a specific code/modifier/revenue code/threshold from the facts,
   set has_recommendation=false and recommendation=null.
4) Use kind="edit" for deny/bundle/exempt-style rules (renders as
   "Edit 1: Deny 99291 when..."). Use kind="immediate_action" for
   utilization-management or threshold-style actions (renders as
   "IA — Implement PA required for 90837 after 20 sessions").
5) Always preserve X wildcards in revenue codes (write 045X, not 0450) and
   modifier formats verbatim (e.g., "25", "59", "XU").
6) If Elevance evidence is provided, frame the recommendation as aligning
   Elevance to the strongest peer cost-control behavior; otherwise present
   the peer behavior as an adoption opportunity.
7) MERGE OVERLAPPING ITEMS. Two items are overlapping — and MUST be emitted
   as a single item — when they act on the SAME target codes AND the SAME
   required modifiers (or, for IA items, the same code + same threshold
   metric). Express sub-conditions, narrower scopes, or refinements
   (teaching-hospital qualifications, provider-role guardrails, state
   carve-outs, documentation requirements) as additional clauses inside
   that one item's `text`, or via the `exemptions` / `scope` fields — do
   NOT split them into a second `edit` item. Items are genuinely separate
   only when they touch a distinct code set, a distinct modifier set, or
   a distinct action_type (deny vs bundle vs allow_with_conditions vs
   require_auth vs limit). When in doubt, merge.
8) Generate at most 5 items. Prefer fewer, sharper items over many vague ones.
9) Output JSON only, no markdown.
10) If a user question is provided above, tailor the recommendation's framing
    to address it — but never invent claims unsupported by the facts above.
11) PEER CITATION: When the "Peer benchmark facts" block above is non-empty,
    each items[].peer_cite MUST be a non-empty string for items whose cited
    codes / modifiers / revenue codes / thresholds / limits also appear in
    those peer facts. The string MUST name the other payer(s) and summarize
    their guardrail in one sentence (examples — adapt to your domain:
    "UHC and Humana cap at 10 hrs/week", "Cigna requires modifier 22 with a
    documented multiple gestation diagnosis", "Molina applies MPPR bundling
    on the same operative session", "Aetna denies revenue code 0762 outside
    observation"). Use peer_cite=null only when the item's grounding is
    single-payer and no peer benchmark applies.
"""

        messages = [
            {"role": "system", "content": "You are a healthcare reimbursement strategist. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]

        self.logger.debug(
            f"Pattern {pattern_rank} recommendation prompt "
            f"has_specifics={has_specifics} has_citations={has_citations} "
            f"citation_count={len(citation_index)}"
        )

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("policy_recommendations", response)
            content = response.content.strip()

            # Remove markdown fences if present (from notebook logic)
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])

            data = json.loads(content)

            has_recommendation = data.get("has_recommendation", False)
            recommendation_obj = data.get("recommendation")

            # Backward compatibility: tolerate the old shape where
            # recommendation was a plain string.
            if has_recommendation and isinstance(recommendation_obj, str):
                description = recommendation_obj.strip()
                items: List[Dict[str, Any]] = []
            elif has_recommendation and isinstance(recommendation_obj, dict):
                items = [i for i in (recommendation_obj.get("items") or []) if isinstance(i, dict)]
                # When a citation index is available, every item must cite ≥1
                # valid id. Drop ungrounded items; if all drop, the
                # recommendation is suppressed entirely. This is the central
                # mechanism that prevents implementation-level claims from
                # floating without a named policy source.
                if has_citations:
                    grounded_items: List[Dict[str, Any]] = []
                    dropped = 0
                    for item in items:
                        raw_cids = item.get("citations") or []
                        if not isinstance(raw_cids, list):
                            dropped += 1
                            continue
                        valid_cids = [
                            cid for cid in raw_cids
                            if isinstance(cid, str) and cid in citation_index
                        ]
                        if not valid_cids:
                            dropped += 1
                            continue
                        # Normalise to the validated subset so downstream
                        # rendering / evidence aggregation only sees resolvable
                        # ids.
                        item["citations"] = valid_cids
                        grounded_items.append(item)
                    if dropped:
                        self.logger.info(
                            f"Pattern {pattern_rank}: dropped {dropped} ungrounded "
                            f"recommendation item(s) (no valid citation id)"
                        )
                    items = grounded_items
                    if not items:
                        self.logger.info(
                            f"Pattern {pattern_rank}: all recommendation items "
                            f"lacked valid citations — suppressing recommendation"
                        )
                        return []
                # Rebuild the recommendation dict so the description render
                # only reflects items that survived grounding.
                filtered_rec = {
                    "headline": recommendation_obj.get("headline"),
                    "items": items,
                }
                description = self._render_recommendation_description(filtered_rec)
            else:
                self.logger.info(f"No policy-driven recommendation for pattern {pattern_rank}")
                return []

            if not description:
                self.logger.info(
                    f"Pattern {pattern_rank}: LLM returned no renderable recommendation items"
                )
                return []

            evidence, peer_benchmarking, citation = self._evidence_and_citation_from_items(
                items=items,
                citation_index=citation_index,
                specific_facts=specific_facts,
                has_citations=has_citations,
            )

            structured_rec = {
                "rank": 1,
                "priority": "MEDIUM",
                "category": "Policy",
                "description": description,
                "evidence": evidence,
                "story_alignment": [],
                "peer_benchmarking": peer_benchmarking,
                "citation": citation,
                # Transient — consumed by `validate_recommendation_node` and
                # popped before the API response is built. Keys are prefixed
                # with an underscore as a convention; `extract_result` does
                # not look at them.
                "_items": items,
            }

            self.logger.info(f"Generated recommendation for pattern {pattern_rank}")
            return [structured_rec]

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(
                f"Failed to parse recommendation for pattern {pattern_rank}: {e}"
            )
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error generating recommendation for pattern {pattern_rank}: {e}"
            )
            return []

    def _run_recommendation_validator(
        self,
        items: List[Dict[str, Any]],
        citation_index: Dict[str, Dict[str, Any]],
        individual_policies: List[Dict[str, Any]],
        pattern: Dict[str, Any],
        user_query: Optional[str],
        summary_table_subtitle: str,
        llm: Any,
    ) -> Dict[str, Any]:
        """
        Run the pattern/scope critique LLM call. Returns a dict
        `{verdicts: [{item_index, decision, reason}, ...], summary: "..."}`
        — one verdict per input item, in the same order.

        Two drop criteria (balanced threshold):
          1. The item's scenario is outside the pattern's scope.
          2. The cited source policy's own evidence summary indicates the
             policy is off-topic for the pattern (e.g. an anesthesia
             policy cited for an OB bundling rule). Note: a policy that
             adjudicates the topic via CPT codes / global packages while
             not naming a specific DRG label is NOT a topic disclaim —
             see prompt for the carve-out.

        Other items must be `keep` — the validator must not drop items
        merely because it would word them differently.

        Returns an empty-verdicts dict on parse failure so the caller can
        fall back to the unfiltered recommendation rather than swallow it.
        """
        if not items:
            return {"verdicts": [], "summary": "no items to validate"}

        cleaned_query = self._clean_user_query(user_query)
        user_question_block = (
            f"User question:\n{cleaned_query}\n\n" if cleaned_query else ""
        )

        # Build the pattern context block.
        pattern_title = (
            pattern.get("top_pattern")
            or pattern.get("pattern_title")
            or f"Pattern {pattern.get('pattern_rank', '')}"
        )
        pattern_evidence = (
            pattern.get("evidence_summary")
            or pattern.get("pattern_details")
            or pattern.get("why_it_matters")
            or ""
        )
        priority_entities = pattern.get("priority_entities") or {}
        states = priority_entities.get("states") or []
        states_block = ", ".join(states) if states else "(none specified)"
        pattern_block = (
            f"Pattern:\n"
            f"- Title: {pattern_title}\n"
            f"- States in scope: {states_block}\n"
            f"- LOB / Subtitle: {summary_table_subtitle or '(none)'}\n"
            f"- Pattern evidence: {pattern_evidence}\n"
        )

        # Look up each cited policy's per-policy evidence summary so the
        # validator sees disclaimer phrases like "referenced only in
        # policy-history language" — the T4-catching signal.
        policy_summary_by_key: Dict[tuple, str] = {}
        for p in individual_policies or []:
            key = (p.get("payer_name") or "", p.get("policy_title") or "")
            summary = p.get("evidence") or ""
            if summary:
                policy_summary_by_key[key] = summary

        items_block_lines: List[str] = []
        for idx, item in enumerate(items):
            text = (item.get("text") or "").strip()
            kind = (item.get("kind") or "").strip()
            scope = item.get("scope") or []
            peer = item.get("peer_cite") or ""
            cids = item.get("citations") or []
            items_block_lines.append(f"Item {idx}:")
            items_block_lines.append(f"  kind: {kind}")
            items_block_lines.append(f"  text: {text}")
            items_block_lines.append(
                f"  scope: {', '.join(scope) if scope else '(none)'}"
            )
            items_block_lines.append(f"  peer_cite: {peer or '(none)'}")
            items_block_lines.append("  citations:")
            for cid in cids:
                src = citation_index.get(cid)
                if not src:
                    items_block_lines.append(f"    [{cid}] (unknown id)")
                    continue
                policy_summary = policy_summary_by_key.get(
                    (src["payer"], src["policy_title"]),
                    "(no policy-level summary available)",
                )
                items_block_lines.append(
                    f"    [{cid}] {src['payer']} · \"{src['policy_title']}\" → "
                    f"{src['fact_key']}: \"{src['fact']}\""
                )
                items_block_lines.append(
                    f"          source policy summary: {policy_summary}"
                )
        items_block = "\n".join(items_block_lines)

        prompt = f"""You are reviewing draft reimbursement-policy recommendation items for fit with the user's pattern and the cited sources.

{user_question_block}{pattern_block}
Items to review:
{items_block}

Decide `keep` or `drop` for each item by item_index. Mark an item `drop` ONLY when one of the following is clearly true:

  1. SCOPE DRIFT — The item's scenario is outside the pattern's scope.
     Example: an item about preventive-medicine pairing in a pattern that
     is about an ED visit gap. The cited fact may be real but the
     scenario doesn't apply to this pattern.

  2. SOURCE DISCLAIMS TOPIC — At least one cited source's "source policy
     summary" indicates the policy does NOT adjudicate the clinical area
     at all. Examples: an Anesthesia Professional Services policy cited
     for an OB bundling rule, a generic "Add-on Coding" guide cited for a
     modifier-22 documentation rule, a Readmission policy cited for an ED
     denial rule.

     IMPORTANT carve-out: a policy that adjudicates the topic via CPT
     codes / global packages / modifiers — while not explicitly naming
     the pattern's DRG label — is NOT a topic disclaim. For DRG-driven
     patterns (e.g. cesarean delivery), professional OB policies using
     CPT-based global maternity logic, multiple-birth modifier rules,
     and assistant-at-surgery rules are LEGITIMATE grounding even if
     their summary says "does not reference the DRG code." Keep those
     items.

Otherwise keep the item. Do NOT drop items merely because they are
aggressive, you would word them differently, or you cannot verify them
without external lookup. Default to `keep` when uncertain.

Return JSON only, no markdown:
{{
  "verdicts": [
    {{"item_index": 0, "decision": "keep" | "drop", "reason": "1 short sentence"}},
    ...
  ],
  "summary": "1-sentence overview of what was dropped and why, or 'all items aligned with pattern'"
}}
"""

        messages = [
            {"role": "system", "content": "You are a healthcare reimbursement policy reviewer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("validate_recommendation", response)
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])
            data = json.loads(content)
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            self.logger.warning(f"Validator response parse failed: {e}")
            return {"verdicts": [], "summary": f"validator parse failure: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected validator error: {e}")
            return {"verdicts": [], "summary": f"validator error: {e}"}

        verdicts_raw = data.get("verdicts") or []
        verdicts: List[Dict[str, Any]] = []
        for v in verdicts_raw:
            if not isinstance(v, dict):
                continue
            idx = v.get("item_index")
            decision = (v.get("decision") or "").strip().lower()
            if not isinstance(idx, int) or decision not in ("keep", "drop"):
                continue
            if idx < 0 or idx >= len(items):
                continue
            verdicts.append({
                "item_index": idx,
                "decision": decision,
                "reason": (v.get("reason") or "").strip(),
            })

        return {"verdicts": verdicts, "summary": (data.get("summary") or "").strip()}

    def _get_policy_hashes(self, policy_ids: List[str]) -> pd.DataFrame:
        """
        Get policy hashes from Snowflake for deduplication.

        Args:
            policy_ids: List of policy IDs

        Returns:
            DataFrame with PLCY_ID and PDF_HASH_VAL_ID columns
        """
        # Uses environment-aware table from registry
        sql = f"""
        SELECT PLCY_ID, PDF_HASH_VAL_ID
        FROM {self.table('policy_metadata')}
        WHERE PLCY_ID IN ({{policy_ids}})
          AND ACTV_STTS_NM = 'Active'
          AND STTS_NM NOT ILIKE '%delete%'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY PLCY_ID ORDER BY RCRD_DT DESC) = 1
        """

        policy_ids_str = ", ".join([f"'{pid}'" for pid in policy_ids])
        return self.snowflake_helper.execute_query_and_return_pandas_df(
            sql.format(policy_ids=policy_ids_str)
        )

    def _build_pattern_details(self, state: ReimbursementState) -> str:
        """Build the pattern_details string used to give the triage LLM (and any
        future relevance-aware step) the same context that extract_rules_node
        uses. Mirrors the shape assembled at the top of extract_rules_node."""
        parts: List[str] = []

        drg_codes = state.get("drg_codes") or []
        if drg_codes:
            parts.append(f"DRG Codes: {', '.join(drg_codes)}")

        cpt_codes = state.get("cpt_codes")
        if cpt_codes:
            parts.append(f"CPT/HCPCS Codes: {cpt_codes}")

        pattern = state.get("pattern") or {}
        pattern_title = pattern.get("pattern_title") or pattern.get("top_pattern")
        if pattern_title:
            parts.append(f"Pattern: {pattern_title}")

        return "\n".join(parts) if parts else (cpt_codes or "")

    def _apply_policy_caps(
        self,
        df: pd.DataFrame,
        pattern_rank: Optional[int],
        pattern_details: str,
        llm_mini: Any,
    ) -> pd.DataFrame:
        """Apply Elevance pinning + per-payor cap + LLM relevance triage + global cap.

        Returns a DataFrame of length <= POLICY_CAP_MAX_TOTAL. Elevance policies
        present in the input are guaranteed in the output (subject to the
        per-payor cap). The triage LLM is consulted only for non-Elevance rows,
        and only when the per-payor-capped non-Elevance set still exceeds the
        remaining budget. Any triage failure silently falls back to API order.
        """
        before = len(df)
        if before == 0:
            return df

        is_elevance = df["payor"].str.contains(ELEVANCE_PAYOR_TOKEN, case=False, na=False)
        elevance_df = (
            df[is_elevance]
            .groupby("payor", sort=False, group_keys=False)
            .head(POLICY_CAP_MAX_PER_PAYOR)
        )
        other_df = (
            df[~is_elevance]
            .groupby("payor", sort=False, group_keys=False)
            .head(POLICY_CAP_MAX_PER_PAYOR)
        )

        other_budget = max(0, POLICY_CAP_MAX_TOTAL - len(elevance_df))
        triage_ran = False

        if len(other_df) > other_budget and other_budget > 0 and llm_mini is not None:
            selected_ids = self._triage_policy_titles(
                other_df=other_df,
                pattern_details=pattern_details,
                target_count=other_budget,
                llm=llm_mini,
                pattern_rank=pattern_rank,
            )
            if selected_ids:
                available = set(other_df["policy_id"])
                ordered = [pid for pid in selected_ids if pid in available]
                if ordered:
                    other_df = (
                        other_df.set_index("policy_id")
                        .loc[ordered]
                        .reset_index()
                    )
                    triage_ran = True

        other_df = other_df.head(other_budget)
        result = pd.concat([elevance_df, other_df], ignore_index=True)

        if len(result) < before:
            self.logger.warning(
                f"Pattern {pattern_rank}: Capped {before} → {len(result)} policies "
                f"(≤{POLICY_CAP_MAX_PER_PAYOR}/payor, ≤{POLICY_CAP_MAX_TOTAL} total, "
                f"Elevance pinned, triage_used={triage_ran}). "
                f"Payor distribution: {result['payor'].value_counts().to_dict()}"
            )

        return result

    def _triage_policy_titles(
        self,
        other_df: pd.DataFrame,
        pattern_details: str,
        target_count: int,
        llm: Any,
        pattern_rank: Optional[int],
    ) -> Optional[List[str]]:
        """Ask the mini LLM to rank non-Elevance policy_ids by title relevance.

        Returns the LLM's selected_policy_ids list (most relevant first), or
        None on any failure (LLM error, malformed JSON, empty selection,
        validation error). Callers must treat None as "fall back to API order".
        """
        records = other_df[["policy_id", "payor", "policy_title"]].fillna("").to_dict("records") \
            if "policy_title" in other_df.columns else \
            [
                {"policy_id": r.get("policy_id", ""), "payor": r.get("payor", ""), "policy_title": ""}
                for r in other_df.to_dict("records")
            ]

        policy_lines = "\n".join(
            f"{r['policy_id']} | {r['payor']} | {r['policy_title']}"
            for r in records
        )

        user_prompt = TRIAGE_USER_PROMPT_TEMPLATE.format(
            pattern_details=pattern_details or "(no pattern context)",
            policy_lines=policy_lines,
            target_count=target_count,
        )
        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("triage_policies", response)

            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])

            data = json.loads(content)
            validated = PolicyTriageResponse(**data)
            if not validated.selected_policy_ids:
                self.logger.warning(
                    f"Pattern {pattern_rank}: Triage returned empty selection; falling back to API order"
                )
                return None
            return validated.selected_policy_ids

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.warning(
                f"Pattern {pattern_rank}: Triage LLM returned malformed payload ({type(e).__name__}); "
                f"falling back to API order"
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"Pattern {pattern_rank}: Triage LLM call failed ({type(e).__name__}: {e}); "
                f"falling back to API order"
            )
            return None

    def _extract_policy_rules(
        self,
        policy_text: str,
        pattern_details: str,
        llm: Any
    ) -> Dict[str, Any]:
        """
        Extract structured rules from a single policy using LLM with Pydantic validation.

        Uses pattern details (DRG codes, CPT codes, pattern context) from notebook approach.

        Args:
            policy_text: Full policy text
            pattern_details: Pattern context including DRG/CPT codes and pattern title
            llm: LLM client

        Returns:
            Dictionary with 'policy_metadata' and 'results' keys

        Raises:
            json.JSONDecodeError: If LLM response is not valid JSON
            ValidationError: If response doesn't match Pydantic schema
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            policy_text=policy_text,
            pattern_details=pattern_details
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self._invoke_llm_with_retry(llm, messages)
        self._record_tokens("extract_rules", response)

        # Parse and validate with Pydantic
        try:
            content = response.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first and last line (code fences)
                if len(lines) > 2:
                    # Also remove language identifier if present (e.g., ```json)
                    content = "\n".join(lines[1:-1])
                else:
                    content = content

            # Parse JSON
            data = json.loads(content)

            # Validate with Pydantic
            validated = PolicyExtractionResponse(**data)

            # Return as dictionary
            return validated.model_dump()

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in policy extraction: {e}")
            self.logger.error(f"Raw response: {response.content[:500]}...")
            raise
        except ValidationError as e:
            self._handle_validation_error(e, "policy extraction")
            raise

    # ========================================================================
    # Comprehensive Payer Policy Summary Helpers (from notebook)
    # ========================================================================

    # Constants for comprehensive summary logic
    RULE_CATEGORIES = [
        "site_of_service",
        "bundling_logic",
        "code_interactions",
        "authorization_requirements",
        "documentation_requirements",
        "limitations",
        "exclusions",
        "mention_status",
        "payor_level_summary",
        "specific_rule_text",
    ]

    # Known payer names used by the cross-payer contamination check.
    # Match is case-insensitive and word-boundary-anchored; any of these
    # names appearing in another payer's evidence is a leak signal.
    KNOWN_PAYERS = [
        "Cigna",
        "Aetna",
        "Humana",
        "United Health",
        "UnitedHealth",
        "UHC",
        "United Healthcare",
        "Elevance",
        "Anthem",
        "Molina",
        "BCBS",
        "Blue Cross",
        "BlueCross",
        "Kaiser",
        "WellCare",
    ]

    # URL-substring → canonical payer key. Used by the policy identity
    # check to confirm policy_url matches payer_name. The first matching
    # substring wins, so order more-specific domains before generic ones.
    _URL_TO_PAYER: List[tuple] = [
        ("uhcprovider.com", "united"),
        ("unitedhealthcare.com", "united"),
        ("uhc.com", "united"),
        ("elevancehealth.com", "elevance"),
        ("anthem.com", "elevance"),
        ("molinahealthcare.com", "molina"),
        ("molina.com", "molina"),
        ("cigna.com", "cigna"),
        ("humana.com", "humana"),
        ("aetna.com", "aetna"),
        ("kaiserpermanente.org", "kaiser"),
        ("kaiser.com", "kaiser"),
        ("wellcare.com", "wellcare"),
        ("bcbs.com", "bcbs"),
    ]

    # Payer-name token → canonical payer key. Substring match against the
    # lowercased payer name from the search API. Order more-specific tokens
    # first so "blue cross" wins over "blue".
    _PAYER_NAME_TO_CANONICAL: List[tuple] = [
        ("blue cross", "bcbs"),
        ("bluecross", "bcbs"),
        ("united health", "united"),
        ("unitedhealth", "united"),
        ("united healthcare", "united"),
        ("elevance", "elevance"),
        ("anthem", "elevance"),
        ("cigna", "cigna"),
        ("humana", "humana"),
        ("molina", "molina"),
        ("aetna", "aetna"),
        ("kaiser", "kaiser"),
        ("wellcare", "wellcare"),
        ("bcbs", "bcbs"),
        ("uhc", "united"),
    ]

    # policy_id prefix → canonical payer key. Used as a soft signal — we
    # only flag a mismatch when the prefix is recognized AND it disagrees
    # with the URL-derived canonical, because Elevance issues policies
    # under many prefix conventions (RP_GBD_*, RP_COM_ELV_EXTERNAL_*).
    _POLICY_ID_PREFIX_TO_PAYER: Dict[str, str] = {
        "CIG": "cigna",
        "UHC": "united",
        "HUM": "humana",
        "ELV": "elevance",
        "MOH": "molina",
    }

    # Prompt-header phrases that occasionally leak into structured fields
    # when the LLM has no real codes to extract. Used by the sanitizer and
    # by the defense-in-depth `drg_prompt_leak_detected` check.
    PROMPT_LEAK_MARKERS = (
        "drg codes:",
        "cpt/hcpcs codes:",
        "cpt codes:",
        "hcpcs codes:",
        "pattern:",
    )

    # Per-environment concurrency caps for LLM-heavy nodes (extract_rules
    # + the relevance judge). Sized against ~12K tokens/call on
    # gpt-5.4-nano against each env's tier limits:
    #   prod=8  → 3M TPM / 100 RPM tier: ~80 RPM, ~1M TPM (well under both)
    #   dev=3   → shared-quota safe
    #   uat=3   → shared-quota safe
    #   local=1 → 30 RPM / 100K TPM tier: TPM is binding (12K/call × 8.3 =
    #             ~100K TPM ceiling), so a single worker is the only safe
    #             setting. Going higher will trip 429s.
    # Override at runtime with REIMBURSEMENT_EXTRACT_CONCURRENCY=<int>.
    _EXTRACT_CONCURRENCY_BY_ENV: Dict[str, int] = {
        "dev": 3, "uat": 3, "prod": 8, "local": 8,
    }

    @classmethod
    def _resolve_extract_concurrency(cls) -> int:
        """Read REIMBURSEMENT_EXTRACT_CONCURRENCY env var if set, else fall
        back to the per-env default keyed off AppConstants.ENV. Always
        returns >= 1 so callers can pass it straight to ThreadPoolExecutor."""
        override = os.environ.get("REIMBURSEMENT_EXTRACT_CONCURRENCY")
        if override:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return cls._EXTRACT_CONCURRENCY_BY_ENV.get(AppConstants.ENV, 3)

    # Field-level regex whitelists for sanitization.
    # CPT: 5 digits, optional trailing letter (e.g. 99284, 99284T).
    # HCPCS: 1 letter + 4 digits (e.g. G0378). Also accept 4-digit revenue
    # codes when seen in target/related (rare but legitimate).
    # ICD-10 diagnosis: letter + 2 digits + optional ".dddd" + optional
    # trailing "-" (e.g. O30.001, Z37.50-, C50.911). Universally clinical
    # (OB uses O/Z codes, oncology uses C/D, cardiology uses I, etc.)
    # so widening here unblocks every medical domain at once.
    _CODE_RE = re.compile(
        r"^\d{4,5}[A-Z]?$"             # CPT (5-digit with optional alpha suffix)
        r"|^[A-Z]\d{4}$"               # HCPCS (e.g. G0378, J1234)
        r"|^[A-Z]\d{2}(\.\d{1,4})?-?$" # ICD-10 (e.g. O30.001, Z37.50-, C50)
    )
    # Modifiers: 2 chars (digits or letters), occasionally 3 (e.g. XEPSU
    # variants written as 3-char shorthands). Accept 1-3 alnum chars.
    _MODIFIER_RE = re.compile(r"^[A-Z0-9]{1,3}$")
    # Revenue codes: 3-4 digits with optional X wildcard (e.g. 0762, 045X, 068X).
    _REVENUE_CODE_RE = re.compile(r"^\d{3,4}[X0-9]?$|^\d{2,3}X$")

    # Structured edit-rule fact keys (List[str] in PolicyRuleResult). These
    # power the specific-recommendation path: aggregated per-policy into
    # `edit_rule_facts` on each reimbursement_policies entry, and pivoted
    # across payers into peer benchmarks for the recommendation prompt.
    EDIT_RULE_FACT_KEYS = [
        "target_codes",
        "related_codes",
        "required_modifiers",
        "excluded_modifiers",
        "revenue_codes",
        "pairing_conditions",
        "utilization_limits",
        "prior_auth_thresholds",
        "discharge_status_conditions",
        "program_scope",
        "state_specific_rules",
        "provider_role_restrictions",
        "exemptions",
    ]

    DATE_FORMATS = (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
    )

    def _parse_date(self, date_str: str) -> tuple[str, Optional[datetime]]:
        """Parse date string into datetime; return original string and parsed datetime if possible."""
        if not date_str:
            return "N/A", None
        for fmt in self.DATE_FORMATS:
            try:
                return date_str, datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return date_str, None

    def _latest_effective_date(self, payer_policies: List[Dict[str, Any]]) -> str:
        """Pick the most recent effective date across payer policies."""
        dates: List[tuple[str, Optional[datetime]]] = []
        for policy in payer_policies:
            if policy is None:
                continue
            effective_date = policy.get("policy_metadata", {}).get("effective_date")
            raw, parsed = self._parse_date(effective_date or "")
            if raw != "N/A":
                dates.append((raw, parsed))
        if not dates:
            return "N/A"
        parsed_dates = [d for d in dates if d[1] is not None]
        if parsed_dates:
            return max(parsed_dates, key=lambda x: x[1])[0]
        return dates[0][0]

    def _appeals_badge(self, payer_policies: List[Dict[str, Any]]) -> str:
        """Return X if any policy documents appeals, else '-'."""
        documented = any(
            policy.get("policy_metadata", {}).get("appeals_process_documented") or
            policy.get("policy_metadata", {}).get("specialty_specific")
            for policy in payer_policies
            if policy is not None
        )
        return "X" if documented else "-"

    def _collect_payer_edit_rule_facts(
        self,
        policies_by_payer: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Build the per-payer aggregate + cross-payer peer benchmarks + per-fact
        citation index the recommendation prompt consumes.

        Args:
            policies_by_payer: Mapping of payer name -> list of policy_entry dicts
                (each already carrying `edit_rule_facts` from format_output_node,
                with `results` still attached for safe fallback aggregation).

        Returns:
            {
              "per_payer": {<payer>: {<key>: [...], ...}, ...},
              "peer_benchmarks": [
                {"fact_key": "utilization_limits", "fact": "10 hrs/week BCBA",
                 "payers": ["UHC", "Humana"], "citation_ids": ["C7", "C12"]},
                ...
              ],
              "citations": [
                {"id": "C1", "payer": "United Health",
                 "policy_title": "Outpatient Hospital Observation Policy, Facility",
                 "policy_url": "https://...",
                 "fact_key": "revenue_codes", "fact": "0762"},
                ...
              ]
            }

            peer_benchmarks contains only facts that appear under ≥2 payers,
            sorted by payer-count descending so the LLM sees the strongest
            cross-payer signals first.

            `citations` is the authoritative grounding list: every entry binds
            a verbatim fact to its source payer + policy_title + policy_url.
            The recommendation prompt requires each item to cite ≥1 id from
            this list, so implementation-level rules cannot float without a
            named source.
        """
        per_payer: Dict[str, Dict[str, List[str]]] = {}
        citations: List[Dict[str, Any]] = []
        # (payer, policy_title, fact_key, fact) -> citation id, used so the
        # same fact from the same source is cited under one stable id.
        citation_by_source: Dict[tuple, str] = {}
        # (fact_key, fact) -> list of citation ids, for peer_benchmarks lookup.
        citations_by_fact: Dict[tuple, List[str]] = {}

        for payer_name, policies in (policies_by_payer or {}).items():
            payer_facts: Dict[str, List[str]] = {key: [] for key in self.EDIT_RULE_FACT_KEYS}
            payer_actions: List[str] = []
            for policy in policies or []:
                if policy is None:
                    continue
                policy_title = policy.get("policy_title") or "Unknown Policy"
                policy_url = policy.get("policy_url") or ""
                # Prefer the pre-aggregated edit_rule_facts on the policy entry;
                # fall back to recomputing from `results` if missing (defensive).
                facts = policy.get("edit_rule_facts")
                if not facts:
                    facts = self._aggregate_edit_rule_facts(policy.get("results", []))
                for key in self.EDIT_RULE_FACT_KEYS:
                    for value in facts.get(key, []) or []:
                        if not isinstance(value, str):
                            continue
                        cleaned = value.strip()
                        if not cleaned:
                            continue
                        if cleaned not in payer_facts[key]:
                            payer_facts[key].append(cleaned)
                        source_key = (payer_name, policy_title, key, cleaned)
                        if source_key not in citation_by_source:
                            cid = f"C{len(citations) + 1}"
                            citation_by_source[source_key] = cid
                            citations.append({
                                "id": cid,
                                "payer": payer_name,
                                "policy_title": policy_title,
                                "policy_url": policy_url,
                                "fact_key": key,
                                "fact": cleaned,
                            })
                            citations_by_fact.setdefault((key, cleaned), []).append(cid)
                for action in facts.get("action_types", []) or []:
                    if action and action not in payer_actions:
                        payer_actions.append(action)
            payer_facts["action_types"] = payer_actions
            per_payer[payer_name] = payer_facts

        # Pivot to peer benchmarks: each distinct (fact_key, fact) -> list of payers.
        index: Dict[tuple, List[str]] = {}
        for payer_name, facts in per_payer.items():
            for key in self.EDIT_RULE_FACT_KEYS:
                for value in facts.get(key, []):
                    bucket = index.setdefault((key, value), [])
                    if payer_name not in bucket:
                        bucket.append(payer_name)

        peer_benchmarks: List[Dict[str, Any]] = []
        for (fact_key, fact), payers in index.items():
            if len(payers) >= 2:
                peer_benchmarks.append({
                    "fact_key": fact_key,
                    "fact": fact,
                    "payers": sorted(payers),
                    "citation_ids": list(citations_by_fact.get((fact_key, fact), [])),
                })
        peer_benchmarks.sort(key=lambda b: (-len(b["payers"]), b["fact_key"], b["fact"]))

        return {
            "per_payer": per_payer,
            "peer_benchmarks": peer_benchmarks,
            "citations": citations,
        }

    def _aggregate_edit_rule_facts(
        self,
        policy_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate one policy's rule results into a flat de-duplicated dict
        of edit-rule fact arrays. Preserves first-seen order for stable output.

        Args:
            policy_results: The `results` list from one PolicyExtractionResponse
                (each item ~ PolicyRuleResult.model_dump()).

        Returns:
            Dict with one List[str] per EDIT_RULE_FACT_KEYS key (always present,
            possibly empty) plus an `action_types` aggregate for visibility.
        """
        facts: Dict[str, List[str]] = {key: [] for key in self.EDIT_RULE_FACT_KEYS}
        action_types: List[str] = []

        for rule in policy_results or []:
            if not isinstance(rule, dict):
                continue
            for key in self.EDIT_RULE_FACT_KEYS:
                values = rule.get(key) or []
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, str):
                        continue
                    cleaned = value.strip()
                    if cleaned and cleaned not in facts[key]:
                        facts[key].append(cleaned)
            action = rule.get("action_type")
            if isinstance(action, str):
                cleaned = action.strip()
                if cleaned and cleaned not in action_types:
                    action_types.append(cleaned)

        facts["action_types"] = action_types
        return facts

    # ========================================================================
    # Phase 1b: Post-extraction sanitization
    # ========================================================================
    #
    # The extraction LLM (gpt-5.4-nano with reasoning_effort=low) is known
    # to copy the prompt's "DRG Codes:" header back into target_codes when
    # the policy text has no real codes to extract — a classic prompt-leak
    # pattern. It also sometimes substitutes a competing payer's name in
    # the prose evidence. The new prompt guardrails attack this upstream
    # (see USER_PROMPT_TEMPLATE), but we still sanitize the output as a
    # belt-and-suspenders pass so any miss is logged and stripped before
    # the data reaches the validation node or downstream consumers.
    #
    # Sanitizer drops are logged at WARNING with policy_id + policy_title +
    # field + dropped value + reason, so reviewers can trace every change.

    @staticmethod
    def _looks_like_prompt_leak(value: str) -> bool:
        """True iff the value contains one of the known prompt-header phrases."""
        lowered = value.lower()
        return any(marker in lowered for marker in ReimbursementPolicyAgent.PROMPT_LEAK_MARKERS)

    def _sanitize_field_values(
        self,
        policy_id: Optional[str],
        policy_title: Optional[str],
        field: str,
        values: List[Any],
        pattern: "re.Pattern[str]",
    ) -> List[str]:
        """Filter one structured list field against a regex whitelist,
        dropping prompt-leak strings unconditionally. Logs every drop."""
        kept: List[str] = []
        for raw in values or []:
            if not isinstance(raw, str):
                self.logger.warning(
                    f"Sanitizer dropped {field}={raw!r} from "
                    f"policy_id={policy_id} title={policy_title!r}: not a string"
                )
                continue
            cleaned = raw.strip()
            if not cleaned:
                continue
            if self._looks_like_prompt_leak(cleaned):
                self.logger.warning(
                    f"Sanitizer dropped {field}={cleaned!r} from "
                    f"policy_id={policy_id} title={policy_title!r}: prompt-header leak"
                )
                continue
            if not pattern.match(cleaned):
                self.logger.warning(
                    f"Sanitizer dropped {field}={cleaned!r} from "
                    f"policy_id={policy_id} title={policy_title!r}: failed {field} format check"
                )
                continue
            kept.append(cleaned)
        return kept

    def _sanitize_extracted_facts(
        self,
        policy_id: Optional[str],
        policy_title: Optional[str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Belt-and-suspenders cleanup of LLM-extracted facts. Mutates and
        returns `payload`. For every rule in payload["results"]:

          - target_codes / related_codes are filtered against _CODE_RE
            (CPT 5-digit or HCPCS letter+4-digit) and stripped of any
            prompt-header strings (e.g. "DRG Codes: Cesarean Section ...").
          - required_modifiers / excluded_modifiers are filtered against
            _MODIFIER_RE (1-3 alphanumeric chars).
          - revenue_codes are filtered against _REVENUE_CODE_RE.

        Every drop is logged at WARNING with policy_id + title for traceability.
        Returns payload unchanged if it isn't a dict (defensive — extract_rules
        catches LLM failures upstream).
        """
        if not isinstance(payload, dict):
            return payload
        rules = payload.get("results")
        if not isinstance(rules, list):
            return payload

        field_patterns: List[tuple] = [
            ("target_codes", self._CODE_RE),
            ("related_codes", self._CODE_RE),
            ("required_modifiers", self._MODIFIER_RE),
            ("excluded_modifiers", self._MODIFIER_RE),
            ("revenue_codes", self._REVENUE_CODE_RE),
        ]

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            for field, pattern in field_patterns:
                values = rule.get(field)
                if not isinstance(values, list):
                    continue
                rule[field] = self._sanitize_field_values(
                    policy_id, policy_title, field, values, pattern
                )

        return payload

    def _aggregate_rule_texts(
        self,
        payer_policies: List[Dict[str, Any]],
        categories: List[str]
    ) -> List[str]:
        """Collect unique rule texts for the payer across selected categories."""
        texts: List[str] = []
        for policy in payer_policies:
            if policy is None:
                continue
            for rule in policy.get("results", []) or []:
                for cat in categories:
                    value = (rule.get(cat) or "").strip()
                    if value and value not in texts:
                        texts.append(value)
        return texts

    def _summarize_column_value(
        self,
        payer_name: str,
        column_label: str,
        rule_texts: List[str],
        column_type: str,
        llm: Any
    ) -> str:
        """
        Summarize payer rule texts into a compact cell value using LLM (from notebook).
        """
        if not rule_texts:
            return "-"

        sample_text = "\n".join(f"- {text[:300]}" for text in rule_texts[:5])

        if column_type == "badge":
            prompt = f"""Summarize payer behavior into a short badge (<=3 words) for column: {column_label}.
If the payer clearly has a rule, use a concise label (e.g., "Bundles", "Denies", "Requires Auth").
If unclear, return "-".
Payer: {payer_name}
Rules:
{sample_text}
Output ONLY the badge text."""
        else:
            prompt = f"""Summarize payer behavior into ONE concise sentence (<=15 words) for column: {column_label}.
Payer: {payer_name}
Rules:
{sample_text}
Output ONLY the summary text."""

        messages = [
            {"role": "system", "content": "You are a concise policy summarizer. Output only the requested text."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("cell_summarization", response)
            return response.content.strip() or "-"
        except Exception as exc:
            self.logger.warning(f"Summarization failed for {payer_name}/{column_label}: {exc}")
            fallback = rule_texts[0][:120]
            return fallback + ("..." if len(rule_texts[0]) > 120 else "")

    def _collect_category_samples(
        self,
        policies: List[Dict[str, Any]],
        max_samples: int = 3
    ) -> Dict[str, List[str]]:
        """Collect distinct sample snippets for each rule category."""
        samples = {category: [] for category in self.RULE_CATEGORIES}
        for policy in policies:
            # Skip None policies (failed extractions)
            if policy is None:
                continue
            for rule in policy.get("results", []) or []:
                for category in self.RULE_CATEGORIES:
                    value = (rule.get(category) or "").strip()
                    if value and value not in samples[category]:
                        samples[category].append(value)
        return {k: v[:max_samples] for k, v in samples.items() if v}

    # Generic user queries that carry no useful intent signal. When the query
    # matches one of these (case-insensitive, after stripping), helpers treat it
    # as if no query were provided and fall back to pattern/DRG context alone.
    _GENERIC_USER_QUERIES = frozenset({
        "",
        "analyze the spike",
        "why did costs increase?",
        "why did costs increase",
        "analyze this",
        "what happened?",
        "what happened",
    })

    def _clean_user_query(self, user_query: Optional[str]) -> Optional[str]:
        """
        Normalize a user query for use as an LLM steering signal.

        Returns the stripped query when it carries specific intent, or None when
        it's empty, whitespace, or a known generic stub (e.g. "Analyze the spike").
        Callers should fall back to existing behavior when this returns None.
        """
        if not user_query or not isinstance(user_query, str):
            return None
        cleaned = user_query.strip()
        if not cleaned:
            return None
        if cleaned.lower() in self._GENERIC_USER_QUERIES:
            return None
        return cleaned

    def _build_pattern_context(
        self,
        pattern: Dict[str, Any],
        drg_codes: List[str],
        keyword: str
    ) -> str:
        """Build a concise pattern context string for the LLM."""
        title = pattern.get("top_pattern") or pattern.get("pattern_title") or f"Pattern {pattern.get('pattern_rank', '')}"
        details = pattern.get("pattern_details", "")
        drg_preview = json.dumps(drg_codes[:5]) + ("..." if len(drg_codes) > 5 else "")
        return (
            f"Pattern title: {title}\n"
            f"Keywords: {keyword or 'N/A'}\n"
            f"DRG codes: {drg_preview}\n"
            f"Details: {details}"
        )

    def _generate_dynamic_columns(
        self,
        pattern_context: str,
        category_samples: Dict[str, List[str]],
        llm: Any,
        user_query: Optional[str] = None
    ) -> TableSchemaResponse:
        """
        Ask the LLM to pick 1-2 dynamic columns relevant to the pattern (from notebook).
        """
        if not category_samples:
            return TableSchemaResponse(columns=[], selected_categories=[])

        samples_text = "\n".join(
            f"- {cat}: {(' | '.join(vals))[:350]}"
            for cat, vals in category_samples.items()
        )

        # Optional user-question framing — included only when the query carries
        # specific intent (generic stubs are filtered out by _clean_user_query).
        cleaned_query = self._clean_user_query(user_query)
        user_question_block = (
            f"\nUser question (lean toward categories that help answer this, if supporting samples exist):\n{cleaned_query}\n"
            if cleaned_query else ""
        )

        prompt = f"""You are designing a payer policy summary table for a specific pattern.

Pattern context:
{pattern_context}
{user_question_block}
Available rule categories with samples:
{samples_text}

Task:
Select the 1-2 most informative columns to compare payer behavior for this pattern.
Each column should map to 1-2 underlying categories from the list above.

Output JSON ONLY with this schema:
{{
  "columns": [
    {{"id": "snake_case_id", "label": "Short Label (<=6 words)", "type": "text|badge"}}
  ],
  "selected_categories": [
    {{"id": "snake_case_id", "categories": ["category1", "category2"]}}
  ]
}}

Rules:
- Use "badge" when the output is a short flag/value (<=3 words, e.g., "Bundles", "Denies", "Requires Auth").
- Use "text" when a brief sentence is needed.
- Do NOT include payer or effective date columns; those are added separately.
- If a user question is provided above and emphasizes a specific category (e.g., denials, bundling, authorization, modifiers), prefer columns covering that category — but only when the available samples support it.
"""

        messages = [
            {"role": "system", "content": "You are a healthcare reimbursement analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("dynamic_columns", response)
            content = response.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])

            data = json.loads(content)
            return TableSchemaResponse(**data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self.logger.warning(f"Column generation failed, using fallback. Error: {exc}")
            return TableSchemaResponse(columns=[], selected_categories=[])

    # ========================================================================
    # Elevance Executive Summary Helpers (from notebook)
    # ========================================================================

    def _extract_elevance_policies(self, pattern_policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter to policies from Elevance (internal or external).

        Skips both failed extractions (None) and quarantined policies so
        the Elevance executive summary cannot be built from contaminated
        evidence (e.g. an Elevance policy whose extracted text actually
        names Cigna).
        """
        elevance_policies = []
        for policy in pattern_policies:
            # Skip None policies (failed extractions)
            if policy is None:
                continue
            # Skip quarantined policies — same rationale as everywhere
            # else: contaminated extractions must not seed downstream
            # outputs visible to the user.
            contamination = policy.get("contamination") if isinstance(policy, dict) else None
            if contamination and contamination.get("severity") == "quarantine":
                continue
            payer = (policy.get("PAYOR_NM") or policy.get("payer_name") or policy.get("payor", "")).lower()
            if "elevance" in payer:
                elevance_policies.append(policy)
        return elevance_policies

    def _build_elevance_evidence(self, policies: List[Dict[str, Any]], max_rules: int = 12) -> str:
        """Extract key rule evidence from Elevance policies."""
        fields = [
            "bundling_logic",
            "denial_conditions",
            "code_interactions",
            "site_of_service",
            "authorization_requirements",
            "documentation_requirements",
            "limitations",
            "exclusions",
            "payor_level_summary",
            "specific_rule_text",
        ]
        lines: List[str] = []
        for policy in policies:
            # Skip None policies (failed extractions)
            if policy is None:
                continue
            # Accept either the raw extract dict (key="policy_metadata") or the
            # formatted individual_policies dict (key="metadata"). The two
            # callers pass different shapes; this normalization lets the same
            # function work for both without per-caller branching.
            metadata = policy.get("metadata") or policy.get("policy_metadata") or {}
            policy_title = (
                policy.get("policy_title")
                or metadata.get("policy_title")
                or metadata.get("policy_scope")
                or policy.get("policy_id")
                or policy.get("PLCY_ID", "Policy")
            )
            eff_date = (
                policy.get("effective_date")
                or metadata.get("effective_date")
                or "N/A"
            )
            lines.append(f"Policy: {policy_title} (Effective: {eff_date})")

            for rule in policy.get("results", []) or []:
                for field in fields:
                    value = (rule.get(field) or "").strip()
                    if value:
                        lines.append(f"- {field}: {value}")

        evidence_text = "\n".join(lines[:max_rules])
        # Trim to max 4000 chars for prompt safety
        return evidence_text if len(evidence_text) <= 4000 else evidence_text[:4000] + "..."

    def _generate_elevance_executive_summary(
        self,
        pattern: Dict[str, Any],
        pattern_rank: int,
        drg_codes: List[str],
        keyword: str,
        pattern_policies: List[Dict[str, Any]],
        llm: Any,
        user_query: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a short executive summary ONLY when Elevance policy evidence plausibly explains the pattern.
        Returns None if not explainable or no Elevance policy evidence is available.

        From notebook logic.
        """
        elevance_policies = self._extract_elevance_policies(pattern_policies)
        if not elevance_policies:
            self.logger.info(f"Pattern {pattern_rank}: No Elevance policies found")
            return None

        evidence = self._build_elevance_evidence(elevance_policies)
        if not evidence.strip():
            self.logger.info(f"Pattern {pattern_rank}: No relevant Elevance evidence found")
            return None

        pattern_context = self._build_pattern_context(pattern, drg_codes, keyword)

        # Optional user-question framing — included only when the query carries
        # specific intent (generic stubs are filtered out by _clean_user_query).
        cleaned_query = self._clean_user_query(user_query)
        user_question_block = (
            f"\nUser question (frame the summary to address this if relevant):\n{cleaned_query}\n"
            if cleaned_query else ""
        )

        prompt = f"""You are writing an executive summary explaining whether an Elevance reimbursement policy plausibly explains a cost anomaly pattern.

Pattern context:
{pattern_context}
{user_question_block}
Elevance policy evidence:
{evidence}

STRICT RULES:
1) Only mark "explainable" = true if the evidence supports a reimbursement rule or restriction that could potentially explain the pattern.
2) If explainable = false, summary MUST be null.
3) Summary must be 1-3 sentences, executive tone, no speculation, no uncertainty language.
4) If a user question is provided above, orient the summary toward it — but never assert anything not supported by the Elevance policy evidence.

Output JSON ONLY:
{{"explainable": true|false, "summary": "..." or null}}
"""

        messages = [
            {"role": "system", "content": "You are a reimbursement policy analyst. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._invoke_llm_with_retry(llm, messages)
            self._record_tokens("elevance_summary", response)
            content = response.content.strip()

            # Remove markdown code fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])

            data = json.loads(content)
            validated = ElevanceSummaryResponse(**data)

            self.logger.info(f"Pattern {pattern_rank}: Elevance explainable={validated.explainable}")
            if validated.explainable and validated.summary:
                self.logger.info(f"Pattern {pattern_rank}: {validated.summary}")
                return validated.summary.strip()
            return None
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self.logger.warning(f"Elevance summary parse failed for pattern {pattern_rank}: {exc}")
            return None

    # ========================================================================
    # Required Base Class Methods
    # ========================================================================

    @property
    def node_name(self) -> str:
        """Entry node name (not used since we override build_graph)."""
        return "search_policies"

    def node_function(self, state: ReimbursementState) -> Dict[str, Any]:
        """Node function (not used since we override build_graph)."""
        pass

    def prepare_state(
        self,
        cpt_codes: Optional[str] = None,
        conversation_id: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare initial state for single-pattern processing.

        BREAKING CHANGE: Pattern context is now REQUIRED in context parameter for
        orchestrator mode. For backward compatibility, direct cpt_codes mode still works.

        Args:
            cpt_codes: Comma-separated CPT codes (backward compatibility - direct mode)
            conversation_id: Conversation ID from orchestrator
            query: User query from UI
            context: REQUIRED for orchestrator mode - must contain 'pattern' with pattern data
            job_id: Job ID for this execution (auto-generated if not provided)
            **kwargs: Additional state values

        Returns:
            Initial state dictionary

        Raises:
            ValueError: If pattern context is missing in orchestrator mode
        """
        # Note: Snowflake validation moved to fetch_content_node (where it's actually used)
        # This follows the correlation agent pattern and allows API to start without credentials

        # Reset per-invocation token bookkeeping. Must happen BEFORE any LLM call
        # in this method (e.g., _generate_search_keywords_from_drg).
        self._token_breakdown = {}

        # Check if pattern context is provided (new single-pattern mode)
        pattern = None
        pattern_rank = None
        drg_codes = []
        search_keywords = None

        if context is not None and "pattern" in context:
            # NEW MODE: Single-pattern processing
            pattern = context["pattern"]
            if not pattern:
                raise ValueError(
                    "Pattern data is required in context['pattern']. "
                    "Expected structure: {pattern_rank, pattern_title, drg_codes, ...}"
                )

            pattern_rank = pattern.get("pattern_rank")
            if pattern_rank is None:
                raise ValueError("pattern_rank is required in pattern context")

            # Extract CPT codes from pattern using notebook logic
            cpt_codes = self._extract_cpt_codes_from_pattern(pattern, context)

            # Extract DRG codes: priority to explicit pattern field, then extract from cards
            drg_codes = pattern.get("drg_codes", [])
            if not drg_codes:
                source_card_ids = pattern.get("source_card_ids", [])
                cards = context.get("cards", [])
                if source_card_ids and cards:
                    drg_codes = self._extract_drg_codes_from_cards(source_card_ids, cards)
                    if drg_codes:
                        self.logger.info(f"Extracted {len(drg_codes)} DRG codes from cards")

            # Generate search keywords from DRG codes using LLM (or use CPT if no DRGs).
            # Forward the user query as a tie-breaker signal only — DRG codes remain primary.
            search_keywords = self._generate_search_keywords(
                drg_codes, cpt_codes, llm=self.llm, user_query=query
            )

            self.logger.info(
                f"Processing pattern {pattern_rank}: '{pattern.get('pattern_title')}'"
            )
            self.logger.info(f"Search keywords: {search_keywords}")
            self.logger.info(f"CPT codes: {cpt_codes}")
            self.logger.info(f"DRG codes: {drg_codes}")

            # Generate job_id with pattern identifier
            if job_id is None:
                job_id = f"{uuid.uuid4().hex}_{pattern_rank}"
        else:
            # BACKWARD COMPATIBILITY: Old direct mode or legacy context mode
            if cpt_codes is None and context is not None:
                cpt_codes = self._extract_cpt_codes_from_context(context, query)

            if not cpt_codes:
                raise ValueError(
                    "CPT codes must be provided either:\n"
                    "  1. Via 'cpt_codes' parameter (direct mode), OR\n"
                    "  2. Via 'context' with pattern data (new single-pattern mode), OR\n"
                    "  3. Via 'context' with filters/drill_path (legacy mode)"
                )

            # Use CPT codes as search keywords in legacy mode
            search_keywords = cpt_codes

            # Generate job_id
            if job_id is None:
                job_id = uuid.uuid4().hex

        return {
            "conversation_id": conversation_id,
            "query": query,
            "context": context,
            "job_id": job_id,
            "pattern": pattern,
            "pattern_rank": pattern_rank,
            "cpt_codes": cpt_codes,
            "drg_codes": drg_codes,
            "search_keywords": search_keywords,
            "snowflake_helper": self.snowflake_helper,
            "llm": self.llm,  # GPT-5.4 for most operations
            "llm_mini": self.llm_mini,  # GPT-5.4-nano for policy processing
            "llm_retry": self.llm_retry,  # GPT-5.4-mini used only on extract retry
            "start_time": datetime.now(timezone.utc).isoformat(),
            "input_tokens": 0,
            "output_tokens": 0,
            **kwargs
        }

    def extract_result(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract final result for single pattern with structured recommendations.

        Returns orchestrator-compliant output with pattern identification
        and structured recommendations at top level.

        Args:
            graph_output: Final graph state

        Returns:
            Orchestrator-compliant output with reimbursement data and recommendations
        """
        # Extract pattern information.
        # Resolution order: pattern.pattern_title → pattern.top_pattern →
        # pattern.title → summary_table.subtitle. Only fall back to
        # "Unknown Pattern" when no source provides a real title; the
        # subtitle fallback closes a regression where the table read
        # "California cesarean delivery spend..." but the explanation
        # block still said "Unknown Pattern".
        pattern_rank = graph_output.get("pattern_rank")
        pattern = graph_output.get("pattern", {}) or {}
        formatted_output = graph_output.get("formatted_output", {}) or {}
        summary_table = formatted_output.get("summary_table")
        subtitle_fallback = (summary_table or {}).get("subtitle") if isinstance(summary_table, dict) else None
        pattern_title = (
            pattern.get("pattern_title")
            or pattern.get("top_pattern")
            or pattern.get("title")
            or subtitle_fallback
            or "Unknown Pattern"
        )

        # Extract results
        policies = graph_output.get("result", [])
        reimbursement_policies = formatted_output.get("individual_policies", [])
        recommended_action = graph_output.get("recommended_action", [])  # List of structured recs

        # Calculate statistics
        total_policies = len(policies)
        successful = sum(1 for p in policies if p is not None)
        failed = total_policies - successful

        # Extract CPT/DRG codes processed
        cpt_codes = graph_output.get("cpt_codes", "")
        drg_codes = graph_output.get("drg_codes", [])
        codes_list = [c.strip() for c in cpt_codes.split(",") if c.strip()] if cpt_codes else []

        # Extract Elevance Executive Summary (generated in recommendation node)
        # Note: This is generated during generate_recommendation_node, not here
        elevance_executive_summary = graph_output.get("elevance_executive_summary")

        # Build explanation
        explanation = {
            "summary": f"Processed reimbursement policies for pattern {pattern_rank}: {pattern_title}",
            "pattern_rank": pattern_rank,
            "pattern_title": pattern_title,
            "cpt_codes_requested": codes_list,
            "drg_codes_requested": drg_codes,
            "total_policies_found": total_policies,
            "successful_extractions": successful,
            "failed_extractions": failed,
            "recommendations_generated": len(recommended_action) if recommended_action else 0,
            "extraction_method": "LLM-based structured extraction from policy text",
            "data_sources": [
                "Carelon Policy Comparison API",
                "Snowflake Policy Database (environment-aware table registry)"
            ]
        }

        # Build validation (standard format: checks as list, not dict)
        checks = []
        if total_policies > 0:
            checks.append({"check": "policies_found", "passed": True, "message": f"Found {total_policies} policies"})
        else:
            checks.append({"check": "policies_found", "passed": False, "message": "No policies found"})

        if successful > 0:
            checks.append({"check": "extractions_successful", "passed": True, "message": f"{successful} successful extractions"})
        else:
            checks.append({"check": "extractions_successful", "passed": False, "message": "All extractions failed"})

        if pattern_rank is not None:
            checks.append({"check": "pattern_identified", "passed": True, "message": f"Pattern {pattern_rank} identified"})
        else:
            checks.append({"check": "pattern_identified", "passed": False, "message": "Pattern rank not identified"})

        # Surface the recommendation validator verdict in the existing
        # validation block. `validate_recommendation_node` always emits a
        # decision; `skipped` means no recommendation was produced upstream
        # and we don't add a noisy check for it.
        rec_validation = graph_output.get("recommendation_validation") or {}
        rec_decision = rec_validation.get("decision")
        if rec_decision in ("ok", "items_dropped", "suppressed"):
            checks.append({
                "check": "recommendation_pattern_aligned",
                "passed": rec_decision != "suppressed",
                "message": rec_validation.get("summary", ""),
            })

        # Surface policy-level contamination from validate_policies_node.
        # The aggregate `policy_contamination_summary` carries the per-policy
        # severity decisions; per-policy `contamination` blocks ride along
        # on reimbursement_policies entries automatically.
        contamination_summary = graph_output.get("policy_contamination_summary") or {}
        quarantined_list = contamination_summary.get("quarantined") or []
        warned_list = contamination_summary.get("warned") or []
        contamination_total = contamination_summary.get("total_extracted") or successful

        cross_payer_hits = [
            q for q in (quarantined_list + warned_list)
            if "cross_payer_name_leak" in (q.get("flags") or [])
        ]
        checks.append({
            "check": "cross_payer_contamination_detected",
            "passed": not cross_payer_hits,
            "message": (
                f"{len(cross_payer_hits)} policy/ies leak another payer's name in evidence"
                if cross_payer_hits else "no cross-payer name leakage detected"
            ),
        })

        if contamination_total > 0:
            quarantine_rate = len(quarantined_list) / contamination_total
            relevance_passed = quarantine_rate < 0.3
            checks.append({
                "check": "policy_relevance_pass_rate",
                "passed": relevance_passed,
                "message": (
                    f"{len(quarantined_list)}/{contamination_total} policies quarantined "
                    f"({quarantine_rate:.0%}); threshold 30%"
                ),
            })

            # When more than half the extracted policies carry warn-level
            # contamination flags, the run is technically successful but
            # the source pool is suspect — flip is_valid to
            # "valid_with_warnings" so consumers surface the caveat.
            warning_rate = len(warned_list) / contamination_total
            warning_passed = warning_rate < 0.5
            checks.append({
                "check": "policy_warning_rate",
                "passed": warning_passed,
                "message": (
                    f"{len(warned_list)}/{contamination_total} policies warning-flagged "
                    f"({warning_rate:.0%}); threshold 50%"
                ),
            })

        # is_valid carries three states:
        #   - False (bool): no policies extracted successfully — the agent
        #     produced nothing trustworthy.
        #   - "valid_with_warnings" (str): extractions succeeded but at
        #     least one critical check failed (cross-payer leak, low
        #     relevance pass rate, recommendation suppressed, etc.).
        #     Consumers should surface the failed checks before acting.
        #   - True (bool): every check passed.
        if successful == 0:
            is_valid: Any = False
        elif any(not c["passed"] for c in checks):
            is_valid = "valid_with_warnings"
        else:
            is_valid = True

        validation = {
            "is_valid": is_valid,
            "checks": checks,
            "warnings": [],
            "errors": []
        }

        def _format_policy_pairs(entries: List[Dict[str, Any]]) -> str:
            pairs = []
            for e in entries:
                pid = e.get("policy_id") or "?"
                title = e.get("title") or "(no title)"
                pairs.append(f"{pid}:{title}")
            return ", ".join(pairs)

        if quarantined_list:
            validation["warnings"].append(
                f"{len(quarantined_list)} policy/ies quarantined for contamination "
                f"(excluded from peer benchmarks and summary-table summarization): "
                f"[{_format_policy_pairs(quarantined_list)}]. See per-policy "
                f"`contamination` field on reimbursement_policies for flags + reasons."
            )
        if warned_list:
            validation["warnings"].append(
                f"{len(warned_list)} policy/ies flagged with contamination warnings "
                f"(kept in output): [{_format_policy_pairs(warned_list)}]."
            )

        rec_dropped = rec_validation.get("dropped_count", 0)
        if rec_dropped:
            validation["warnings"].append(
                f"Validator dropped {rec_dropped} recommendation item(s) for "
                f"scope drift or source-disclaimed citations. "
                f"See recommendation_validation in graph state for per-item reasons."
            )

        if failed > 0:
            validation["warnings"].append(
                f"{failed} out of {total_policies} policies failed extraction. "
                f"Check logs for details."
            )

        if successful == 0:
            validation["errors"].append(
                "No policies were successfully extracted. All LLM extractions failed."
            )

        if pattern_rank is None:
            validation["warnings"].append(
                "Pattern rank not identified. This may indicate missing pattern context."
            )

        # Check if conversation_id was provided (orchestrator mode)
        conversation_id = graph_output.get("conversation_id")
        job_id = graph_output.get("job_id", uuid.uuid4().hex)

        # Calculate execution time
        start_time = graph_output.get("start_time")
        end_time = datetime.now(timezone.utc).isoformat()

        # Build tokens info from per-LLM-call bookkeeping accumulated during this run.
        # Each category corresponds to one of the agent's LLM call sites:
        #   - search_keywords:        keyword generation from DRG codes (prepare_state)
        #   - extract_rules:          per-policy rule extraction (extract_rules_node, gpt-5.4-nano)
        #   - dynamic_columns:        pattern-aware column selection (analyze_table_structure_node)
        #   - cell_summarization:     per-cell payer summary (format_output_node)
        #   - elevance_summary:       Elevance executive summary (generate_recommendation_node)
        #   - policy_recommendations: final policy recommendation (generate_recommendation_node)
        breakdown = self._token_breakdown
        total_input = sum(entry.get("input", 0) for entry in breakdown.values())
        total_output = sum(entry.get("output", 0) for entry in breakdown.values())
        tokens = {
            "input": total_input,
            "output": total_output,
            "breakdown": breakdown
        }

        # Build execution info (standard format: duration_ms in milliseconds)
        duration_seconds = self._calculate_duration(start_time, end_time)
        execution = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": int(duration_seconds * 1000),  # Convert to milliseconds
            "version": "2.0.0"  # Updated for single-pattern processing
        }

        # Determine status using standard values: success | partial_success | failed
        if successful == total_policies and total_policies > 0:
            status = "success"
        elif successful > 0:
            status = "partial_success"
        else:
            status = "failed"

        # Build orchestrator-compliant output (standard format across all agents)
        orchestrator_output = {
            "job_id": job_id,
            "conversation_id": conversation_id,
            "agent": "reimbursement_policy",
            "status": status,
            "output": {
                "pattern_rank": pattern_rank,
                "summary_table": summary_table,
                "reimbursement_policies": reimbursement_policies,
                "elevance_executive_summary": elevance_executive_summary,
                "policies_processed": total_policies,
                "policies_successful": successful,
                "policies_failed": failed
            },
            "recommended_action": recommended_action,
            "visual_component": {},  # Reserved for future visualizations
            "explanation": explanation,
            "validation": validation,
            "tokens": tokens,
            "execution": execution
        }

        # If conversation_id is None, we're in direct mode - return formatted output with recommendations
        if conversation_id is None:
            return {
                "summary_table": summary_table,
                "reimbursement_policies": reimbursement_policies,
                "elevance_executive_summary": elevance_executive_summary,
                "recommended_action": recommended_action
            }

        return orchestrator_output

    def _calculate_duration(self, start_time: Optional[str], end_time: str) -> float:
        """
        Calculate duration in seconds between start and end times.

        Args:
            start_time: ISO timestamp string
            end_time: ISO timestamp string

        Returns:
            Duration in seconds
        """
        if not start_time:
            return 0.0

        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds()
        except Exception:
            return 0.0


# ============================================================================
# Main Entry Point
# ============================================================================


# ============================================================================
# Pattern Extraction Utilities (from notebook)
# ============================================================================

def extract_drg_codes_from_pattern_output(pattern_results: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract DRG codes per pattern from pattern analysis output.

    For each pattern:
    1. Get source_card_ids
    2. Find matching cards
    3. Extract drg_name from:
       a) filters where field == 'drg_name'
       b) source_entity.name where source_entity.type == 'drgs'

    Args:
        pattern_results: Pattern agent output JSON with structure:
            {
                "output": {
                    "business_patterns": [...],
                    "cards": [...]
                }
            }

    Returns:
        Dictionary mapping pattern_rank to list of unique DRG names for that pattern
        Example: {
            1: ["DRG 765", "DRG 767"],
            2: ["DRG 834"],
            ...
        }
    """
    if not pattern_results or 'output' not in pattern_results:
        return {}

    output = pattern_results['output']
    patterns = output.get('business_patterns', [])
    cards = output.get('cards', [])

    # Create a lookup dictionary for cards by card_id
    card_lookup = {card['card_id']: card for card in cards if 'card_id' in card}

    # Dictionary to store DRG codes per pattern
    pattern_drg_map = {}

    # Iterate through all patterns
    for pattern in patterns:
        pattern_rank = pattern.get('pattern_rank')
        source_card_ids = pattern.get('source_card_ids', [])

        # Set to collect unique DRG codes for this pattern
        pattern_drgs = set()

        # For each source card ID
        for card_id in source_card_ids:
            card = card_lookup.get(card_id)
            if not card:
                continue

            # Method 1: Check filters for drg_name
            filters = card.get('filters', [])
            for filter_item in filters:
                if isinstance(filter_item, dict) and filter_item.get('field') == 'drg_name':
                    drg_value = filter_item.get('value')
                    if drg_value:
                        pattern_drgs.add(drg_value)

            # Method 2: Check source_entity for DRG
            source_entity = card.get('source_entity', {})
            if isinstance(source_entity, dict):
                if source_entity.get('type') == 'drgs':
                    drg_name = source_entity.get('name')
                    if drg_name:
                        pattern_drgs.add(drg_name)

        # Store sorted list of DRGs for this pattern
        if pattern_rank is not None:
            pattern_drg_map[pattern_rank] = sorted(list(pattern_drgs))

    return pattern_drg_map


def extract_lob_state_product_from_pattern_output(pattern_results: Dict[str, Any]) -> Dict[int, Dict[str, List[str]]]:
    """
    Extract LOB, state, and product per pattern from pattern analysis output.

    For each pattern:
    1. Extract from priority_entities (states and products)
    2. Get source_card_ids and find matching cards
    3. Extract from card dimensions and filters:
       - lob_description / line_of_business
       - service_area_state / geography
       - product_description / product

    Args:
        pattern_results: Pattern agent output JSON with structure:
            {
                "output": {
                    "business_patterns": [...],
                    "cards": [...]
                }
            }

    Returns:
        Dictionary mapping pattern_rank to dict of unique values for that pattern
        Example: {
            1: {
                "lob": ["Commercial"],
                "state": ["CO", "ME"],
                "product": ["HMO", "PPO"]
            },
            2: {
                "lob": ["Commercial"],
                "state": ["GA"],
                "product": ["HMO"]
            },
            ...
        }
    """
    if not pattern_results or 'output' not in pattern_results:
        return {}

    output = pattern_results['output']
    patterns = output.get('business_patterns', [])
    cards = output.get('cards', [])

    # Create a lookup dictionary for cards by card_id
    card_lookup = {card['card_id']: card for card in cards if 'card_id' in card}

    # Dictionary to store lob/state/product per pattern
    pattern_entities_map = {}

    # Iterate through all patterns
    for pattern in patterns:
        pattern_rank = pattern.get('pattern_rank')

        # Sets to collect unique values for this pattern
        lobs = set()
        states = set()
        products = set()

        # Method 1: Extract from priority_entities (if available)
        priority_entities = pattern.get('priority_entities', {})
        if priority_entities:
            # States from priority_entities
            pattern_states = priority_entities.get('states', [])
            if pattern_states:
                states.update(pattern_states)

            # Products from priority_entities
            pattern_products = priority_entities.get('products', [])
            if pattern_products:
                products.update(pattern_products)

        # Method 2: Extract from source cards
        source_card_ids = pattern.get('source_card_ids', [])
        for card_id in source_card_ids:
            card = card_lookup.get(card_id)
            if not card:
                continue

            # Check dimensions
            dimensions = card.get('dimensions', {})
            if dimensions:
                lob_val = dimensions.get('lob_description')
                if lob_val:
                    lobs.add(lob_val)

                state_val = dimensions.get('service_area_state')
                if state_val:
                    states.add(state_val)

                product_val = dimensions.get('product_description')
                if product_val:
                    products.add(product_val)

            # Check canonical_dimensions
            canonical_dims = card.get('canonical_dimensions', {})
            if canonical_dims:
                canonical_lobs = canonical_dims.get('line_of_business', [])
                if canonical_lobs:
                    lobs.update(canonical_lobs)

                canonical_states = canonical_dims.get('geography', [])
                if canonical_states:
                    states.update(canonical_states)

                canonical_products = canonical_dims.get('product', [])
                if canonical_products:
                    products.update(canonical_products)

            # Check filters
            filters = card.get('filters', [])
            for filter_item in filters:
                if not isinstance(filter_item, dict):
                    continue

                field = filter_item.get('field')
                value = filter_item.get('value')

                if field == 'lob_description' and value:
                    lobs.add(value)
                elif field == 'service_area_state' and value:
                    states.add(value)
                elif field == 'product_description' and value:
                    products.add(value)

        # Store results for this pattern
        if pattern_rank is not None:
            pattern_entities_map[pattern_rank] = {
                'lob': sorted(list(lobs)),
                'state': sorted(list(states)),
                'product': sorted(list(products))
            }

    return pattern_entities_map


# ============================================================================
# Main Agent Builder
# ============================================================================

def build_app(
    snowflake_helper: Optional[Any] = None,
    snowflake_helper_builder: Optional[Callable[[], Any]] = None,
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
    **kwargs
) -> Callable[[str], List[Dict[str, Any]]]:
    """
    Build reimbursement policy extraction agent.

    Args:
        snowflake_helper: Optional SnowparkHelper instance
        snowflake_helper_builder: Optional callable that returns SnowparkHelper
        llm: Optional pre-configured LLM client
        llm_builder: Optional callable that returns LLM client
        **kwargs: Additional arguments passed to agent

    Returns:
        Callable that takes cpt_codes and returns extracted rules

    Example:
        >>> agent = build_app(snowflake_helper=snowpark)
        >>> results = agent(cpt_codes="99291,99292")
    """
    agent = ReimbursementPolicyAgent(
        snowflake_helper=snowflake_helper,
        snowflake_helper_builder=snowflake_helper_builder,
        llm=llm,
        llm_builder=llm_builder,
        **kwargs
    )

    return agent


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Build agent (will auto-configure from environment)
    agent = build_app()

    # Extract rules for critical care codes
    results = agent(cpt_codes="99291,99292")

    # Display results
    print(f"\nProcessed {len(results)} policies\n")

    for policy_result in results:
        if policy_result:
            print(f"Policy: {policy_result['PLCY_ID']}")
            for rule in policy_result.get('results', []):
                print(f"  Code {rule['code']}:")
                print(f"    Denial conditions: {rule.get('denial_conditions', 'N/A')}")
                print(f"    Documentation: {rule.get('documentation_requirements', 'N/A')}")
            print()
