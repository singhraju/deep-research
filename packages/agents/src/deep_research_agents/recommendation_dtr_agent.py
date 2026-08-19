"""
Recommendation DTR Agent with LLM-Based Relevance Determination

This agent generates policy recommendations from combined pattern+reimbursement data
with strict business decision tree rule (DTR) alignment. Unlike keyword-based matching,
this agent uses the LLM itself to determine pattern relevance to DTR rules.

Key Features:
- LLM determines if each pattern is relevant to any DTR rules
- LLM identifies which specific rules or categories apply
- Only generates recommendations when LLM confirms relevance
- Disregards patterns that LLM deems irrelevant to DTR scope
- Processes patterns individually (not in batch)

Workflow:
1. Load business decision tree rules from YAML
2. For each pattern:
   a. LLM evaluates relevance to DTR rules
   b. If relevant: LLM identifies applicable rules
   c. Generate recommendation using identified rules
   d. If not relevant: skip pattern
3. Export recommendations with processing log

Usage:
    >>> from deep_research_agents import RecommendationDTRAgent
    >>> 
    >>> agent = RecommendationDTRAgent()
    >>> 
    >>> # DTR rules path auto-configured from AppConstants
    >>> results = agent(patterns_data=combined_patterns)
    >>> 
    >>> print(f"Generated {len(results['recommendations'])} recommendations")
    >>> print(f"Skipped {len(results['skipped_patterns'])} patterns")
    >>> 
    >>> # Optional: Override DTR rules path if needed
    >>> results = agent(
    ...     patterns_data=combined_patterns,
    ...     dtr_rules_path="custom/path/rules.yaml"
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from deep_research_core.base_agent import AgentBase
from deep_research_agents.decision_tree_rules import DecisionTreeRuleEngine
from deep_research_utils.app_constant import AppConstants
from deep_research_utils.ehap_retry import structured_llm_invoke_with_tokens

try:
    from deep_research_utils.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class RecommendationDTRState(TypedDict, total=False):
    """State for DTR-based recommendation agent."""
    
    patterns_data: List[Dict[str, Any]]
    dtr_rules_path: str
    all_rules: List[Dict[str, Any]]
    
    all_recommendations: List[Dict[str, Any]]
    skipped_patterns: List[Dict[str, Any]]
    processing_log: List[Dict[str, Any]]
    
    llm: Any
    require_llm_relevance: bool
    debug_mode: bool
    
    # Token tracking
    start_time: str  # ISO timestamp
    token_breakdown_per_pattern: Dict[str, Dict[str, Dict[str, int]]]  # Per-pattern detailed
    token_breakdown_aggregated: Dict[str, Dict[str, int]]  # Aggregated by operation


class PatternRelevanceSchema(BaseModel):
    """Schema for LLM pattern relevance determination."""
    is_relevant: bool = False
    relevance_reason: str = ""
    applicable_categories: List[str] = Field(default_factory=list)
    applicable_rule_ids: List[str] = Field(default_factory=list)


class RecommendationItemSchema(BaseModel):
    """Schema for a single recommendation."""
    rank: int
    priority: str
    category: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    story_alignment: List[str] = Field(default_factory=list)
    peer_benchmarking: List[str] = Field(default_factory=list)
    citation: List[str] = Field(default_factory=list)


class RecommendationResponseSchema(BaseModel):
    """Schema for LLM recommendation generation response."""
    recommendations: List[RecommendationItemSchema] = Field(default_factory=list)


# ============================================================================
# PROMPTS FOR LLM-BASED RELEVANCE DETERMINATION
# ============================================================================

RELEVANCE_SYSTEM_PROMPT = """You are an expert healthcare analyst evaluating pattern relevance to business decision tree rules.

Your task:
1. Determine if the healthcare pattern is relevant to ANY of the business decision tree rules
2. If relevant, identify which rules apply (up to 5 most relevant)
3. If not relevant, explicitly state no relevance

Output MUST be valid JSON only.
"""

RELEVANCE_USER_PROMPT = """Evaluate if this healthcare pattern is relevant to the business decision tree rules.

PATTERN:
{pattern_summary}

BUSINESS DECISION TREE RULES (Sample - {total_rules} total rules across {total_categories} categories):
{rules_sample}

INSTRUCTIONS:
1. Analyze the pattern's type, clinical area, and business impact
2. Determine if ANY business rules are relevant to this pattern
3. If relevant, identify which rule categories and specific trends apply
4. If NOT relevant, return empty arrays

OUTPUT SCHEMA:
{{
  "is_relevant": true/false,
  "relevance_reason": "<brief explanation of why relevant or not relevant>",
  "applicable_categories": ["<category names if relevant>"],
  "applicable_rule_ids": ["<trend_id values if relevant>"]
}}

Return ONLY valid JSON.
"""

RECOMMENDATION_SYSTEM_PROMPT = """You are an expert healthcare policy analyst that generates recommendations STRICTLY based on business decision tree rules.

CRITICAL RULES:
1. You MUST use ONLY the business decision tree rules provided
2. If the provided rules don't match the pattern, return an empty recommendations array
3. DO NOT make up or infer rules that weren't provided
4. DO NOT create recommendations without explicit rule support
5. Use ONLY information from the input pattern and reimbursement data
6. The relevance of these rules has been pre-determined by LLM analysis

Your output MUST be valid JSON only, no markdown or explanatory text.
"""

RECOMMENDATION_USER_PROMPT = """Generate a policy recommendation for this healthcare pattern using the LLM-identified relevant business rules.

PATTERN DATA:
{pattern_json}

LLM-IDENTIFIED RELEVANT BUSINESS RULES:
{matched_rules}

RELEVANCE ASSESSMENT:
The LLM has determined these rules are relevant because: {relevance_reason}

INSTRUCTIONS:
1. Review the pattern data and LLM-identified relevant rules
2. Create ONE recommendation based on these rules
3. The recommendation MUST:
   - Be a specific action (not a summary of pattern data)
   - Come from hints in the business decision tree rules
   - Use actual data from the pattern (amounts, codes, names)
   - Extract evidence directly from pattern data
   - Include policy URLs from reimbursement data if available
   - Do not repeat numbers, statistics and details from pattern data, should only consist of a recommendation
   - Do not try to explain the recommendation

OUTPUT SCHEMA:
{{
  "recommendations": [
    {{
      "rank": <pattern_rank>,
      "priority": "HIGH|MEDIUM|LOW",
      "category": "Policy",
      "description": "<specific action with identifiers>",
      "evidence": ["<specific evidence from pattern data, max 15 words each>"],
      "story_alignment": ["Business Rule <Category> - <Research Consideration>: <how it applies to this pattern, max 30 words>"],
      "peer_benchmarking": ["<peer payor practice from reimbursement data, max 25 words>"],
      "citation": ["Policy: <PayerName> - <URL from reimbursement data>"]
    }}
  ]
}}

RETURN ONLY VALID JSON.
"""


class RecommendationDTRAgent(AgentBase):
    """
    Agent for generating recommendations with LLM-based DTR relevance determination.
    
    This agent processes patterns individually and uses the LLM to determine which
    patterns are relevant to business decision tree rules. Only relevant patterns
    result in recommendations.
    
    Args:
        **kwargs: Additional arguments passed to AgentBase
    """
    
    # Concurrency configuration by environment
    _CONCURRENCY_BY_ENV = {
        "local": 3,
        "dev": 4,
        "uat": 5,
        "prod": 5,
        "local_offshore": 3
    }
    
    # Retry configuration
    _RETRY_DELAY = 2  # seconds
    _MAX_RETRIES = 1  # retry once
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="recommendation_synthesis",
            state_class=RecommendationDTRState,
            llm_reasoning_effort="high",
            llm_summary_mode="auto",
            **kwargs
        )
    
    def build_graph(self) -> StateGraph:
        """
        Build simple 2-node graph for DTR-based recommendation pipeline.
        
        Returns:
            Configured StateGraph
        """
        graph = StateGraph(self.state_class)
        
        # Add nodes
        graph.add_node("load_rules", self.load_rules_node)
        graph.add_node("process_all_patterns", self.process_all_patterns_node)
        
        # Define flow
        graph.add_edge(START, "load_rules")
        graph.add_edge("load_rules", "process_all_patterns")
        graph.add_edge("process_all_patterns", END)
        
        return graph
    
    def load_rules_node(self, state: RecommendationDTRState) -> Dict[str, Any]:
        """
        Node 1: Load decision tree rules from YAML.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with rule engine and flattened rules
        """
        dtr_rules_path = state["dtr_rules_path"]
        
        self.logger.info("[SETUP] Loading business decision tree rules")
        self.logger.debug(f"DTR rules path: {dtr_rules_path}")
        
        # Initialize rule engine
        rule_engine = DecisionTreeRuleEngine(str(dtr_rules_path))
        
        # Get all categories and flatten rules with category tags
        all_categories = rule_engine.get_category_names()
        all_rules = []
        
        for category in all_categories:
            rules = rule_engine.get_rules_by_category(category)
            for rule in rules:
                rule_copy = dict(rule)
                rule_copy["_category"] = category
                all_rules.append(rule_copy)
        
        rule_count = len(all_rules)
        category_count = len(all_categories)
        
        self.logger.info(f"✓ Loaded {rule_count} rules from {category_count} categories")
        self.logger.debug(f"Categories: {', '.join(all_categories)}")
        
        return {
            "all_rules": all_rules
        }
    
    def _process_single_pattern(
        self,
        pattern: Dict[str, Any],
        all_rules: List[Dict[str, Any]],
        pattern_idx: int,
        total_patterns: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single pattern: relevance check → fetch rules → generate recommendation.
        
        Args:
            pattern: Pattern data dictionary
            all_rules: All available DTR rules
            pattern_idx: Zero-based index of pattern in list
            total_patterns: Total number of patterns being processed
            
        Returns:
            Dict with:
            - recommendation: The recommendation object (or None if skipped)
            - skip_info: Skip information if pattern was skipped
            - log_entry: Processing log entry for this pattern
            
            Returns None only on catastrophic failure (should not happen)
        """
        pattern_rank = pattern.get("pattern_rank", pattern_idx + 1)
        top_pattern = pattern.get("top_pattern", "Unknown")
        
        # Track tokens for this pattern
        pattern_tokens = {
            "relevance": {"input": 0, "output": 0},
            "recommendation": {"input": 0, "output": 0}
        }
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[Pattern {pattern_rank}] [{pattern_idx + 1}/{total_patterns}] {top_pattern[:60]}...")
        self.logger.info(f"{'='*80}")
        
        # Step 1: LLM determines relevance to DTR rules - NOW RETURNS TOKENS
        self.logger.info(f"[Pattern {pattern_rank}] [1/3] LLM checking relevance...")
        relevance_result, relevance_tokens = self._check_pattern_relevance(pattern, all_rules)
        pattern_tokens["relevance"] = relevance_tokens
        
        is_relevant = relevance_result.get("is_relevant", False)
        relevance_reason = relevance_result.get("relevance_reason", "")
        applicable_categories = relevance_result.get("applicable_categories", [])
        applicable_rule_ids = relevance_result.get("applicable_rule_ids", [])
        
        self.logger.info(f"[Pattern {pattern_rank}]   Relevant: {is_relevant}")
        self.logger.debug(f"[Pattern {pattern_rank}]   Reason: {relevance_reason[:150]}..." if len(relevance_reason) > 150 else f"[Pattern {pattern_rank}]   Reason: {relevance_reason}")
        
        if not is_relevant:
            self.logger.info(f"[Pattern {pattern_rank}]   ⊗ Not relevant to DTR - skipping")
            return {
                "recommendation": None,
                "skip_info": {
                    "rank": pattern_rank,
                    "pattern": top_pattern,
                    "reason": f"LLM determined not relevant: {relevance_reason}"
                },
                "log_entry": {
                    "rank": pattern_rank,
                    "status": "skipped",
                    "reason": "llm_not_relevant",
                    "relevance_reason": relevance_reason
                },
                "tokens": pattern_tokens
            }
        
        # Step 2: Get relevant rules based on LLM determination
        self.logger.info(f"[Pattern {pattern_rank}] [2/3] Fetching relevant rules...")
        
        # First try specific rules by IDs
        relevant_rules = self._get_rules_by_ids(applicable_rule_ids, all_rules)
        
        # If no specific rules, get rules from categories
        if not relevant_rules and applicable_categories:
            relevant_rules = self._get_rules_by_categories(applicable_categories, all_rules)
        
        if not relevant_rules:
            self.logger.warning(f"[Pattern {pattern_rank}]   ⚠️  Could not fetch rules")
            return {
                "recommendation": None,
                "skip_info": {
                    "rank": pattern_rank,
                    "pattern": top_pattern,
                    "reason": "Could not fetch relevant rules"
                },
                "log_entry": {
                    "rank": pattern_rank,
                    "status": "skipped",
                    "reason": "rules_fetch_failed"
                },
                "tokens": pattern_tokens
            }
        
        self.logger.info(f"[Pattern {pattern_rank}]   ✓ Fetched {len(relevant_rules)} rules")
        
        # Step 3: Generate recommendation using LLM-identified rules - NOW RETURNS TOKENS
        self.logger.info(f"[Pattern {pattern_rank}] [3/3] Generating recommendation...")
        recommendation, rec_tokens = self._generate_recommendation(
            pattern,
            relevant_rules,
            relevance_reason
        )
        pattern_tokens["recommendation"] = rec_tokens
        
        if recommendation:
            self.logger.info(f"[Pattern {pattern_rank}]   ✓ Recommendation generated")
            self.logger.debug(f"[Pattern {pattern_rank}]     Priority: {recommendation.get('priority', 'N/A')}")
            self.logger.debug(f"[Pattern {pattern_rank}]     Description: {recommendation.get('description', '')[:80]}...")
            
            return {
                "recommendation": recommendation,
                "skip_info": None,
                "log_entry": {
                    "rank": pattern_rank,
                    "status": "success",
                    "rules_used": len(relevant_rules),
                    "categories": applicable_categories,
                    "relevance_reason": relevance_reason
                },
                "tokens": pattern_tokens
            }
        else:
            self.logger.warning(f"[Pattern {pattern_rank}]   ⚠️  No recommendation generated")
            return {
                "recommendation": None,
                "skip_info": {
                    "rank": pattern_rank,
                    "pattern": top_pattern,
                    "reason": "Failed to generate recommendation"
                },
                "log_entry": {
                    "rank": pattern_rank,
                    "status": "skipped",
                    "reason": "generation_failed",
                    "rules_available": len(relevant_rules)
                },
                "tokens": pattern_tokens
            }
    
    def process_all_patterns_node(self, state: RecommendationDTRState) -> Dict[str, Any]:
        """
        Node 2: Process all patterns using parallel execution.
        
        Uses ThreadPoolExecutor to process multiple patterns simultaneously.
        Each pattern runs its full workflow (relevance → rules → recommendation)
        in a separate thread. Results are collected and reassembled in original
        pattern_rank order.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with all recommendations and processing log
        """
        patterns_data = state["patterns_data"]
        all_rules = state["all_rules"]
        
        total_patterns = len(patterns_data)
        concurrency = self._resolve_concurrency()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("PROCESSING PATTERNS WITH PARALLEL LLM-BASED DTR RELEVANCE")
        self.logger.info(f"Total patterns: {total_patterns}")
        self.logger.info(f"Concurrency: {concurrency} workers")
        self.logger.info(f"{'='*80}")
        
        # Fan out parallel pattern processing (NO RETRY WRAPPER)
        by_index: Dict[int, Optional[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(self._process_single_pattern, pattern, all_rules, i, total_patterns): i
                for i, pattern in enumerate(patterns_data)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    by_index[i] = fut.result()
                except Exception as e:
                    # Log error but continue processing other patterns
                    self.logger.error(f"Pattern {i+1} failed: {e}")
                    by_index[i] = None
        
        # Reassemble results in original order
        results = [by_index.get(i) for i in range(total_patterns)]
        
        # Extract recommendations, skipped patterns, logs, and tokens
        all_recommendations = []
        skipped_patterns = []
        processing_log = []
        
        # Initialize token tracking structures
        token_breakdown_per_pattern = {}
        token_breakdown_aggregated = {
            "relevance_checks": {"input": 0, "output": 0, "calls": 0},
            "recommendations": {"input": 0, "output": 0, "calls": 0}
        }
        
        for result in results:
            if result is None:
                continue
            
            if result["recommendation"]:
                all_recommendations.append(result["recommendation"])
            
            if result["skip_info"]:
                skipped_patterns.append(result["skip_info"])
            
            processing_log.append(result["log_entry"])
            
            # Extract and aggregate tokens
            if "tokens" in result:
                pattern_rank = result["log_entry"]["rank"]
                pattern_key = f"pattern_{pattern_rank}"
                
                # Store per-pattern breakdown
                token_breakdown_per_pattern[pattern_key] = result["tokens"]
                
                # Aggregate by operation type
                rel_tokens = result["tokens"]["relevance"]
                token_breakdown_aggregated["relevance_checks"]["input"] += rel_tokens["input"]
                token_breakdown_aggregated["relevance_checks"]["output"] += rel_tokens["output"]
                token_breakdown_aggregated["relevance_checks"]["calls"] += 1
                
                rec_tokens = result["tokens"]["recommendation"]
                token_breakdown_aggregated["recommendations"]["input"] += rec_tokens["input"]
                token_breakdown_aggregated["recommendations"]["output"] += rec_tokens["output"]
                if result["recommendation"]:  # Only count successful recommendations
                    token_breakdown_aggregated["recommendations"]["calls"] += 1
        
        # Summary logging
        successful = len(all_recommendations)
        skipped = len(skipped_patterns)
        self.logger.info(f"\n{'='*80}")
        self.logger.info("PATTERN PROCESSING COMPLETE")
        self.logger.info(f"Total patterns: {total_patterns}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Skipped: {skipped}")
        self.logger.info(f"{'='*80}")
        
        return {
            "all_recommendations": all_recommendations,
            "skipped_patterns": skipped_patterns,
            "processing_log": processing_log,
            "token_breakdown_per_pattern": token_breakdown_per_pattern,
            "token_breakdown_aggregated": token_breakdown_aggregated
        }
    
    @classmethod
    def _resolve_concurrency(cls) -> int:
        """
        Read RECOMMENDATION_CONCURRENCY env var if set, else fall
        back to the per-env default keyed off AppConstants.ENV.
        Always returns >= 1 for ThreadPoolExecutor.
        
        Returns:
            Number of worker threads for parallel pattern processing
        """        
        override = os.environ.get("RECOMMENDATION_CONCURRENCY")
        if override:
            try:
                return max(1, min(5, int(override)))
            except ValueError:
                pass
        return cls._CONCURRENCY_BY_ENV.get(AppConstants.ENV, 3)
    
    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract input/output token counts from LLM response.
        
        Handles both new-style (usage_metadata) and legacy (response_metadata)
        token usage formats.
        
        Args:
            response: LangChain LLM response object
            
        Returns:
            Dict with 'input' and 'output' token counts
        """
        # Check new-style usage_metadata
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return {
                "input": int(usage.get("input_tokens") or 0),
                "output": int(usage.get("output_tokens") or 0),
            }
        
        # Check legacy response_metadata.token_usage
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or {}
        return {
            "input": int(token_usage.get("prompt_tokens") or 0),
            "output": int(token_usage.get("completion_tokens") or 0),
        }
    
    # ========================================================================
    # HELPER METHODS - LLM-BASED RELEVANCE
    # ========================================================================
    
    def _check_pattern_relevance(
        self,
        pattern: Dict[str, Any],
        all_rules: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, int]]:
        """
        Use LLM to determine if pattern is relevant to DTR rules.
        
        Args:
            pattern: Pattern data
            all_rules: All available rules with category tags
            
        Returns:
            Tuple of (relevance_result, token_usage)
        """
        # Create pattern summary
        pattern_summary = {
            "pattern_rank": pattern.get("pattern_rank"),
            "top_pattern": pattern.get("top_pattern"),
            "pattern_type": pattern.get("pattern_type"),
            "what_is_impacting": pattern.get("what_is_impacting"),
            "impact_summary": pattern.get("impact_summary"),
            "pattern_details": pattern.get("pattern_details", "")[:200] + "..." if pattern.get("pattern_details") else ""
        }
        
        pattern_summary_json = json.dumps(pattern_summary, indent=2)
        
        # Get rules sample
        rules_sample = self._format_rules_sample(all_rules)
        
        # Get unique categories
        categories = set(rule.get("_category", "Unknown") for rule in all_rules)
        
        # Build prompt
        user_prompt = RELEVANCE_USER_PROMPT.format(
            pattern_summary=pattern_summary_json,
            rules_sample=rules_sample,
            total_rules=len(all_rules),
            total_categories=len(categories)
        )
        
        messages = [
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Invoke LLM with automatic token refresh and token tracking
        try:
            result_schema, _, raw_response = structured_llm_invoke_with_tokens(
                llm=self.llm,
                ehap=self.ehap,
                messages=messages,
                schema=PatternRelevanceSchema,
                llm_reinitializer=self._initialize_llm,
            )
            
            # Extract tokens
            tokens = self._extract_token_usage(raw_response)
            
            result = {
                "is_relevant": result_schema.is_relevant,
                "relevance_reason": result_schema.relevance_reason,
                "applicable_categories": result_schema.applicable_categories,
                "applicable_rule_ids": result_schema.applicable_rule_ids
            }
            
            return result, tokens
        
        except Exception as e:
            self.logger.error(f"  ❌ LLM relevance check failed: {e}")
            return {
                "is_relevant": False,
                "relevance_reason": f"LLM error: {str(e)}",
                "applicable_categories": [],
                "applicable_rule_ids": []
            }, {"input": 0, "output": 0}
    
    def _generate_recommendation(
        self,
        pattern: Dict[str, Any],
        relevant_rules: List[Dict[str, Any]],
        relevance_reason: str
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, int]]:
        """
        Generate recommendation using LLM-identified relevant rules.
        
        Args:
            pattern: Pattern data
            relevant_rules: Rules identified as relevant by LLM
            relevance_reason: LLM's explanation of relevance
            
        Returns:
            Tuple of (recommendation_dict, token_usage)
        """
        # Format pattern data
        pattern_json = json.dumps(pattern, indent=2, ensure_ascii=False)
        
        # Format matched rules
        rules_text = self._format_rules_for_prompt(relevant_rules)
        
        # Build prompt
        user_prompt = RECOMMENDATION_USER_PROMPT.format(
            pattern_json=pattern_json,
            matched_rules=rules_text,
            relevance_reason=relevance_reason
        )
        
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Invoke LLM with automatic token refresh and token tracking
        try:
            result_schema, _, raw_response = structured_llm_invoke_with_tokens(
                llm=self.llm,
                ehap=self.ehap,
                messages=messages,
                schema=RecommendationResponseSchema,
                llm_reinitializer=self._initialize_llm,
            )
            
            # Extract tokens
            tokens = self._extract_token_usage(raw_response)
            
            if result_schema.recommendations:
                return result_schema.recommendations[0].model_dump(), tokens
            else:
                return None, tokens
        
        except Exception as e:
            self.logger.error(f"  ❌ LLM recommendation generation failed: {e}")
            return None, {"input": 0, "output": 0}
    
    def _format_rules_sample(
        self,
        all_rules: List[Dict[str, Any]],
        max_sample: int = 50
    ) -> str:
        """
        Format a sample of rules for LLM review.
        
        Args:
            all_rules: All available rules
            max_sample: Maximum number of rules to sample
            
        Returns:
            Formatted text representation
        """
        # Take a diverse sample across categories
        categories = {}
        for rule in all_rules:
            cat = rule.get("_category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rule)
        
        # Sample from each category
        sample = []
        rules_per_cat = max(1, max_sample // len(categories))
        
        for cat, rules in categories.items():
            sample.extend(rules[:rules_per_cat])
        
        # Format sample
        formatted = []
        for i, rule in enumerate(sample[:max_sample], 1):
            category = rule.get("_category", "Unknown")
            trend_id = rule.get("trend_id", "N/A")
            why = rule.get("why", "")[:100]
            
            formatted.append(f"{i}. [{category}] Trend {trend_id}: {why}...")
        
        return "\n".join(formatted)
    
    def _format_rules_for_prompt(self, rules: List[Dict[str, Any]]) -> str:
        """
        Format rules for LLM prompt.
        
        Args:
            rules: Rules to format
            
        Returns:
            Formatted text representation
        """
        if not rules:
            return "No rules provided."
        
        formatted = []
        for i, rule in enumerate(rules, 1):
            category = rule.get("_category", "Unknown")
            trend_id = rule.get("trend_id", "N/A")
            why = rule.get("why", "")
            suggestions = rule.get("cost_of_care_suggestions") or rule.get("coding_accuracy_suggestions") or ""
            
            rule_text = f"{i}. Category: {category}\n"
            rule_text += f"   Trend ID: {trend_id}\n"
            rule_text += f"   Why: {why}\n"
            if suggestions:
                rule_text += f"   Suggestion: {suggestions}\n"
            
            formatted.append(rule_text)
        
        return "\n".join(formatted)
    
    def _get_rules_by_ids(
        self,
        rule_ids: List[str],
        all_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get specific rules by their trend IDs.
        
        Args:
            rule_ids: List of trend IDs to fetch
            all_rules: All available rules
            
        Returns:
            Matched rules
        """
        matched = []
        for rule in all_rules:
            if rule.get("trend_id") in rule_ids:
                matched.append(rule)
        return matched
    
    def _get_rules_by_categories(
        self,
        categories: List[str],
        all_rules: List[Dict[str, Any]],
        max_per_category: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get rules from specific categories.
        
        Args:
            categories: Category names to fetch
            all_rules: All available rules
            max_per_category: Maximum rules per category
            
        Returns:
            Matched rules
        """
        matched = []
        for cat in categories:
            cat_rules = [r for r in all_rules if r.get("_category") == cat]
            matched.extend(cat_rules[:max_per_category])
        return matched
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response, handling markdown code fences.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed JSON or None if parsing failed
        """
        cleaned = response_text.strip()
        
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2:
                cleaned = "\n".join(lines[1:-1])
        
        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            self.logger.debug(f"Failed to parse JSON: {e}")
            return None
    
    # ========================================================================
    # AGENT BASE INTERFACE
    # ========================================================================
    
    @property
    def node_name(self) -> str:
        """Entry node name (not used since we override build_graph)."""
        return "process_all_patterns"
    
    def node_function(self, state: RecommendationDTRState) -> Dict[str, Any]:
        """Node function (not used since we override build_graph)."""
        pass
    
    def prepare_state(
        self,
        patterns_data: List[Dict[str, Any]],
        dtr_rules_path: Optional[str] = None,
        require_llm_relevance: bool = True,
        debug_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare initial state for graph execution.
        
        Args:
            patterns_data: List of combined pattern+reimbursement data
            dtr_rules_path: Optional path to decision tree rules YAML file.
                          If None, uses AppConstants.DTR_RULES_PATH (configs/decision_tree_rules.yaml)
            require_llm_relevance: If True, skip patterns deemed irrelevant by LLM
            debug_mode: Enable verbose logging
            **kwargs: Additional state values
            
        Returns:
            Initial state dictionary
        """
        # Use default DTR rules path from AppConstants if not provided
        if dtr_rules_path is None:
            from deep_research_utils.app_constant import AppConstants
            dtr_rules_path = AppConstants.DTR_RULES_PATH
        
        from datetime import datetime, timezone
        
        return {
            "patterns_data": patterns_data,
            "dtr_rules_path": dtr_rules_path,
            "require_llm_relevance": require_llm_relevance,
            "debug_mode": debug_mode,
            "llm": self.llm,
            "start_time": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    
    def extract_result(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract final result with token tracking and execution metrics.
        
        Args:
            graph_output: Final graph state
            
        Returns:
            Results dictionary with metadata, recommendations, tokens, and execution info
        """
        from datetime import datetime, timezone
        
        patterns_data = graph_output.get("patterns_data", [])
        all_recommendations = graph_output.get("all_recommendations", [])
        skipped_patterns = graph_output.get("skipped_patterns", [])
        processing_log = graph_output.get("processing_log", [])
        dtr_rules_path = graph_output.get("dtr_rules_path", "")
        
        # Extract token breakdowns
        breakdown_per_pattern = graph_output.get("token_breakdown_per_pattern", {})
        breakdown_aggregated = graph_output.get("token_breakdown_aggregated", {})
        
        # Calculate total tokens from aggregated breakdown
        total_input = sum(entry.get("input", 0) for entry in breakdown_aggregated.values())
        total_output = sum(entry.get("output", 0) for entry in breakdown_aggregated.values())
        
        # Build token structure (matching reimbursement agent format)
        tokens = {
            "input": total_input,
            "output": total_output,
            "breakdown": {
                "per_pattern": breakdown_per_pattern,
                "aggregated": breakdown_aggregated
            }
        }
        
        # Calculate execution duration
        start_time = graph_output.get("start_time", "")
        end_time = datetime.now(timezone.utc).isoformat()
        
        duration_ms = 0
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                duration_seconds = (end_dt - start_dt).total_seconds()
                duration_ms = int(duration_seconds * 1000)
            except Exception:
                pass
        
        execution = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "version": "1.0.0"
        }
        
        # Remove story_alignment from all recommendations
        for rec in all_recommendations:
            rec.pop("story_alignment", None)
            
        return {
            "metadata": {
                "total_patterns": len(patterns_data),
                "recommendations_generated": len(all_recommendations),
                "patterns_skipped": len(skipped_patterns),
                "approach": "llm_based_dtr_relevance",
                "dtr_rules_path": dtr_rules_path
            },
            "recommendations": all_recommendations,
            "skipped_patterns": skipped_patterns,
            "processing_log": processing_log,
            "tokens": tokens,
            "execution": execution
        }


def build_app(
    llm: Optional[Any] = None,
    llm_builder: Optional[Callable[[], Any]] = None,
    **kwargs
) -> Callable[[List[Dict[str, Any]], Optional[str]], Dict[str, Any]]:
    """
    Build DTR-based recommendation agent with LLM relevance determination.
    
    Args:
        llm: Optional pre-configured LLM client
        llm_builder: Optional callable that returns LLM client
        **kwargs: Additional arguments passed to agent
        
    Returns:
        Callable that takes patterns_data (and optional dtr_rules_path), returns results
        
    Example:
        >>> agent = build_app()
        >>> # DTR rules path auto-configured from AppConstants
        >>> results = agent(patterns_data=combined_patterns)
        >>> print(f"Generated {len(results['recommendations'])} recommendations")
        >>> 
        >>> # Or override the DTR rules path if needed
        >>> results = agent(
        ...     patterns_data=combined_patterns,
        ...     dtr_rules_path="custom/path/rules.yaml"
        ... )
    """
    agent = RecommendationDTRAgent(
        llm=llm,
        llm_builder=llm_builder,
        **kwargs
    )
    
    return agent


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Example usage
    agent = build_app(debug=True)
    
    # Sample pattern data (in practice, load from combined_pattern_reimbursement_results.json)
    sample_patterns = [
        {
            "pattern_rank": 1,
            "top_pattern": "Maine alcohol-related IP BH growth",
            "pattern_type": "Inpatient",
            "what_is_impacting": "Behavioral Health",
            "impact_summary": "Significant cost increase in Maine BH admissions",
            "pattern_details": "Maine alcohol-related inpatient behavioral health admissions increased by $2.3M...",
            "reimbursement": {
                "pattern_title": "Observation Stay Policies",
                "individual_policies": [
                    {
                        "payer_name": "Point32Health",
                        "policy_url": "https://www.point32health.org/provider/observation-stay"
                    }
                ]
            }
        }
    ]
    
    # DTR rules path is auto-configured from AppConstants.DTR_RULES_PATH
    # No need to pass dtr_rules_path parameter!
    try:
        results = agent(patterns_data=sample_patterns)
        
        print(json.dumps(results, indent=2))
    except FileNotFoundError as e:
        print(f"Note: DTR rules file not found: {e}")
        print("Make sure configs/decision_tree_rules.yaml exists in the project root.")
