# """
# Recommendation Agent

# This agent transforms pattern analysis output (with varying input structures) into 
# standardized actionable recommendations with evidence, story alignment, and peer benchmarking.

# The agent is schema-agnostic and can handle different input JSON structures from various
# clients while always producing a consistent output format.

# Usage:
#     >>> from deep_research_agents import RecommendationAgent
#     >>> 
#     >>> agent = RecommendationAgent()
#     >>> 
#     >>> # Input can be any valid JSON structure
#     >>> results = agent(input_data=pattern_data)
#     >>> 
#     >>> # Output is always standardized
#     >>> for rec in results['recommendations']:
#     ...     print(f"{rec['rank']}. [{rec['priority']}] {rec['description']}")
# """

# from __future__ import annotations

# import json
# import logging
# from typing import Any, Callable, Dict, List, Optional, TypedDict

# from langgraph.graph import END, START, StateGraph

# from deep_research_core.base_agent import AgentBase

# try:
#     from deep_research_utils.logger_config import get_logger

#     logger = get_logger(__name__)
# except ImportError:
#     logger = logging.getLogger(__name__)


# class RecommendationState(TypedDict, total=False):
#     """State for recommendation synthesis agent."""
    
#     input_data: Any
#     input_json_str: str
#     llm: Any
#     result: Dict[str, List[Dict[str, Any]]]


# SYSTEM_PROMPT = """You are an expert healthcare policy analyst specializing in healthcare policy recommendations across multiple domains including reimbursement policies, provider contracts, utilization management, and network optimization.

# Your role is to:
# - Analyze healthcare data and identify actionable policy opportunities
# - Generate evidence-based recommendations grounded in the provided data
# - Synthesize peer practices and industry benchmarks from the data
# - Prioritize recommendations by financial impact and scope

# Critical rules:
# - Use ONLY information from the input data provided
# - Do not hallucinate or add external knowledge
# - Extract evidence directly from the data
# - Output valid JSON only, no markdown or explanatory text
# - All recommendations must be grounded in the input data
# """

# USER_PROMPT_TEMPLATE = """You are analyzing healthcare data to generate policy recommendations.

# INPUT DATA (structure may vary):
# {input_json}

# INSTRUCTIONS:
# 1. Analyze the input data and identify:
#    - Patterns, findings, or issues (regardless of how they're labeled in the JSON)
#    - Supporting evidence from claims/data (look for dollar amounts, percentages, volumes, growth rates)
#    - Healthcare policies, rules, or contractual terms (from any section of the input)
#    - Peer practices, industry benchmarks, or comparative information (if present)

# 2. Generate actionable policy recommendations:
#    - Consolidate related patterns into single recommendations when they support the same action
#    - Determine priority (HIGH/MEDIUM/LOW) based on:
#      * Financial impact (dollar amounts, percentage of spend)
#      * Scope (number of states, providers, members affected)
#      * Urgency (policy gaps, peer adoption trends)
#    - Rank recommendations by priority and impact (most impactful first)
#    - Extract evidence bullets directly from the input data
#    - Synthesize peer benchmarking or industry practices from the input
#    - Write clear, concise description containing ONLY the recommended action (do not include next steps or peer support in description)

# 3. Use ONLY information from the input data - no external knowledge or assumptions

# OUTPUT SCHEMA (REQUIRED):
# {{
#   "recommendations": [
#     {{
#       "rank": <number starting from 1>,
#       "priority": "HIGH|MEDIUM|LOW",
#       "category": "Policy",
#       "description": "<clear, concise action statement describing what should be done>",
#       "evidence": ["<bullet point from input data>", "<another bullet>", ...],
#       "story_alignment": ["<how the pattern/data supports this recommendation>", ...],
#       "peer_benchmarking": ["<peer payer name>: <their practice from input data>", ...]
#     }}
#   ]
# }}

# IMPORTANT:
# - Every field in evidence, story_alignment, and peer_benchmarking must come from the input data
# - Do not invent or assume information not present in the input
# - If peer benchmarking data is not in the input, use empty array []
# - Category must always be "Policy"
# - Rank recommendations by priority (HIGH first, then MEDIUM, then LOW)
# - Description field should contain ONLY the recommended action, not next steps or peer support

# Return ONLY valid JSON matching this schema. No markdown, no explanations.
# """


# class RecommendationAgent(AgentBase):
#     """
#     Agent for generating standardized policy recommendations from varying input structures.
    
#     This agent:
#     1. Accepts any valid JSON structure as input
#     2. Uses LLM to semantically extract patterns, evidence, and policies
#     3. Consolidates related patterns into recommendations
#     4. Produces standardized output with priority, evidence, and peer benchmarking
    
#     Args:
#         **kwargs: Additional arguments passed to AgentBase
#     """
    
#     def __init__(self, **kwargs):
#         super().__init__(
#             agent_name="recommendation_synthesis_old",
#             state_class=RecommendationState,
#             llm_reasoning_effort="high",
#             llm_summary_mode="auto",
#             **kwargs
#         )
    
#     def build_graph(self) -> StateGraph:
#         """
#         Build multi-node graph for recommendation synthesis pipeline.
        
#         Returns:
#             Configured StateGraph
#         """
#         graph = StateGraph(self.state_class)
        
#         graph.add_node("parse_input", self.parse_input_node)
#         graph.add_node("synthesize_recommendations", self.synthesize_recommendations_node)
#         graph.add_node("validate_output", self.validate_output_node)
        
#         graph.add_edge(START, "parse_input")
#         graph.add_edge("parse_input", "synthesize_recommendations")
#         graph.add_edge("synthesize_recommendations", "validate_output")
#         graph.add_edge("validate_output", END)
        
#         return graph
    
#     def parse_input_node(self, state: RecommendationState) -> Dict[str, Any]:
#         """
#         Node 1: Parse and validate input (schema-agnostic).
        
#         Args:
#             state: Current graph state
            
#         Returns:
#             Updated state with serialized input
#         """
#         input_data = state["input_data"]
        
#         self.logger.info("[1/3] Parsing input data (schema-agnostic)")
#         self.logger.debug(f"Input data type: {type(input_data).__name__}")
        
#         if not self._validate_json_input(input_data):
#             self.logger.error("Input validation failed: not valid JSON")
#             raise ValueError("Input must be valid JSON (dict, list, or JSON string)")
        
#         self.logger.debug("Input validation passed")
#         input_json_str = self._serialize_input_for_llm(input_data)
        
#         self.logger.info(f"✓ Input serialized: {len(input_json_str)} characters")
#         if self.debug:
#             self.logger.debug(f"Serialized preview: {input_json_str[:200]}...")
        
#         return {
#             "input_json_str": input_json_str
#         }
    
#     def synthesize_recommendations_node(self, state: RecommendationState) -> Dict[str, Any]:
#         """
#         Node 2: Synthesize recommendations using LLM.
        
#         Args:
#             state: Current graph state
            
#         Returns:
#             Updated state with recommendations
#         """
#         input_json_str = state["input_json_str"]
#         llm = state["llm"]
        
#         self.logger.info("[2/3] Synthesizing recommendations with LLM")
#         self.logger.debug(f"Prompt size: {len(input_json_str)} characters")
        
#         user_prompt = USER_PROMPT_TEMPLATE.format(input_json=input_json_str)
        
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_prompt}
#         ]
        
#         self.logger.debug("Invoking LLM...")
#         response = llm.invoke(messages)
#         self.logger.debug(f"LLM response received: {len(response.content)} characters")
        
#         self.logger.debug("Parsing LLM response...")
#         recommendations = self._parse_recommendation_response(response.content)
        
#         rec_count = len(recommendations.get('recommendations', []))
#         self.logger.info(f"✓ Generated {rec_count} recommendations")
        
#         if rec_count > 0 and self.debug:
#             priorities = {}
#             for rec in recommendations['recommendations']:
#                 priority = rec.get('priority', 'UNKNOWN')
#                 priorities[priority] = priorities.get(priority, 0) + 1
#             self.logger.debug(f"Priority breakdown: {priorities}")
        
#         return {
#             "result": recommendations
#         }
    
#     def validate_output_node(self, state: RecommendationState) -> Dict[str, Any]:
#         """
#         Node 3: Validate output schema.
        
#         Args:
#             state: Current graph state
            
#         Returns:
#             Updated state (validation errors logged)
#         """
#         result = state["result"]
        
#         self.logger.info("[3/3] Validating output schema")
        
#         validation_errors = self._validate_output_schema(result)
        
#         if validation_errors:
#             self.logger.warning(f"Found {len(validation_errors)} validation issues:")
#             for error in validation_errors:
#                 self.logger.warning(f"  - {error}")
#         else:
#             self.logger.info("✓ Output schema validation passed")
        
#         return {}
    
#     def _validate_json_input(self, input_data: Any) -> bool:
#         """
#         Validate that input is valid JSON.
        
#         Args:
#             input_data: Input to validate
            
#         Returns:
#             True if valid, False otherwise
#         """
#         if isinstance(input_data, (dict, list)):
#             return True
        
#         if isinstance(input_data, str):
#             try:
#                 json.loads(input_data)
#                 return True
#             except json.JSONDecodeError:
#                 return False
        
#         return False
    
#     def _serialize_input_for_llm(self, input_data: Any) -> str:
#         """
#         Convert input to JSON string for LLM processing.
        
#         Args:
#             input_data: Input data (dict, list, or JSON string)
            
#         Returns:
#             JSON string
#         """
#         if isinstance(input_data, str):
#             parsed = json.loads(input_data)
#             return json.dumps(parsed, indent=2, ensure_ascii=False)
        
#         return json.dumps(input_data, indent=2, ensure_ascii=False)
    
#     def _parse_recommendation_response(self, response_text: str) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Parse LLM response into recommendations structure.
        
#         Args:
#             response_text: Raw LLM response
            
#         Returns:
#             Parsed recommendations dictionary
#         """
#         cleaned = response_text.strip()
        
#         # Remove markdown code fences if present
#         if cleaned.startswith("```"):
#             self.logger.debug("Removing markdown code fences from LLM response")
#             lines = cleaned.split("\n")
#             if len(lines) > 2:
#                 if lines[0].startswith("```json"):
#                     cleaned = "\n".join(lines[1:-1])
#                 else:
#                     cleaned = "\n".join(lines[1:-1])
        
#         try:
#             result = json.loads(cleaned)
            
#             if not isinstance(result, dict):
#                 self.logger.warning("LLM response is not a dictionary, wrapping in recommendations key")
#                 return {"recommendations": []}
            
#             if "recommendations" not in result:
#                 self.logger.warning("LLM response missing 'recommendations' key")
#                 return {"recommendations": []}
            
#             self.logger.debug("LLM response parsed successfully")
#             return result
            
#         except json.JSONDecodeError as e:
#             self.logger.error(f"Failed to parse LLM response as JSON: {e}")
#             if self.debug:
#                 self.logger.debug(f"Response text preview: {cleaned[:500]}")
#             return {"recommendations": []}
    
#     def _validate_output_schema(self, result: Dict[str, Any]) -> List[str]:
#         """
#         Validate output matches expected schema.
        
#         Args:
#             result: Result dictionary to validate
            
#         Returns:
#             List of validation error messages (empty if valid)
#         """
#         errors = []
        
#         if not isinstance(result, dict):
#             errors.append("Result is not a dictionary")
#             return errors
        
#         if "recommendations" not in result:
#             errors.append("Missing 'recommendations' key")
#             return errors
        
#         recommendations = result["recommendations"]
        
#         if not isinstance(recommendations, list):
#             errors.append("'recommendations' is not a list")
#             return errors
        
#         required_fields = ["rank", "priority", "category", "description", "evidence", "story_alignment", "peer_benchmarking"]
        
#         for i, rec in enumerate(recommendations):
#             if not isinstance(rec, dict):
#                 errors.append(f"Recommendation {i} is not a dictionary")
#                 continue
            
#             for field in required_fields:
#                 if field not in rec:
#                     errors.append(f"Recommendation {i} missing required field: {field}")
            
#             if "priority" in rec and rec["priority"] not in ["HIGH", "MEDIUM", "LOW"]:
#                 errors.append(f"Recommendation {i} has invalid priority: {rec['priority']}")
            
#             if "category" in rec and rec["category"] != "Policy":
#                 errors.append(f"Recommendation {i} has non-Policy category: {rec['category']}")
            
#             for list_field in ["evidence", "story_alignment", "peer_benchmarking"]:
#                 if list_field in rec and not isinstance(rec[list_field], list):
#                     errors.append(f"Recommendation {i} field '{list_field}' is not a list")
        
#         return errors
    
#     @property
#     def node_name(self) -> str:
#         """Entry node name (not used since we override build_graph)."""
#         return "synthesize_recommendations"
    
#     def node_function(self, state: RecommendationState) -> Dict[str, Any]:
#         """Node function (not used since we override build_graph)."""
#         pass
    
#     def prepare_state(
#         self,
#         input_data: Any,
#         **kwargs
#     ) -> Dict[str, Any]:
#         """
#         Prepare initial state for graph execution.
        
#         Args:
#             input_data: Input data (any valid JSON structure)
#             **kwargs: Additional state values
            
#         Returns:
#             Initial state dictionary
#         """
#         return {
#             "input_data": input_data,
#             "llm": self.llm,
#             **kwargs
#         }
    
#     def extract_result(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Extract final result from graph output.
        
#         Args:
#             graph_output: Final graph state
            
#         Returns:
#             Recommendations dictionary
#         """
#         return graph_output.get("result", {"recommendations": []})


# def build_app(
#     llm: Optional[Any] = None,
#     llm_builder: Optional[Callable[[], Any]] = None,
#     **kwargs
# ) -> Callable[[Any], Dict[str, Any]]:
#     """
#     Build recommendation synthesis agent.
    
#     Args:
#         llm: Optional pre-configured LLM client
#         llm_builder: Optional callable that returns LLM client
#         **kwargs: Additional arguments passed to agent
        
#     Returns:
#         Callable that takes input_data and returns recommendations
        
#     Example:
#         >>> agent = build_app()
#         >>> results = agent(input_data=pattern_data)
#         >>> print(results['recommendations'])
#     """
#     agent = RecommendationAgent(
#         llm=llm,
#         llm_builder=llm_builder,
#         **kwargs
#     )
    
#     return agent


# if __name__ == "__main__":
#     import os
#     from dotenv import load_dotenv
    
#     load_dotenv()
    
#     agent = build_app()
    
#     sample_input = [
#         {
#             "rank": 1,
#             "pattern_title": "High-Volume Critical Care Billing",
#             "pattern_description": "High paid volume for 99291 with home discharge",
#             "explanation": {
#                 "reimbursement": {
#                     "individual_policies": [
#                         {
#                             "payer_name": "Molina Medicaid",
#                             "evidence": "Deny critical care in ED when patient discharged home"
#                         }
#                     ]
#                 },
#                 "claim_evidence": {
#                     "summary": "≈$9.3M FI, ≈$25.8M ASO in total exposure"
#                 }
#             }
#         }
#     ]
    
#     results = agent(input_data=sample_input)
    
#     print(json.dumps(results, indent=2))
