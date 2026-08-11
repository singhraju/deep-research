"""Pydantic models for reimbursement policy agent LLM responses."""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Literal, Dict, Any, Union
from datetime import datetime


class PolicyRuleResult(BaseModel):
    """Individual CPT code rule extraction result."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="CPT/HCPCS code")

    # Prose fields (free-text rule descriptions)
    site_of_service: Optional[str] = Field(None, description="Where the code is allowed or restricted")
    bundling_logic: Optional[str] = Field(None, description="Whether code is bundled with other services")
    code_interactions: Optional[str] = Field(None, description="How this code interacts with other codes")
    modifier_usage: Optional[str] = Field(None, description="Required, allowed, or disallowed modifiers")
    denial_conditions: Optional[str] = Field(None, description="Common reasons this code would be denied")
    unit_pricing_logic: Optional[str] = Field(None, description="Time/unit rules and limits")
    documentation_requirements: Optional[str] = Field(None, description="Key documentation needed")
    evidence_summary: Optional[str] = Field(None, description="Brief summary of key denial or restriction")
    # Prose fields the user prompt already asks for — formerly dropped by the schema
    authorization_requirements: Optional[str] = Field(None, description="Prior auth / notification requirements")
    limitations: Optional[str] = Field(None, description="Frequency, quantity, or other limitations")
    exclusions: Optional[str] = Field(None, description="Scenarios where the code is not covered")
    specific_rule_text: Optional[str] = Field(None, description="Verbatim text supporting the finding")
    mention_status: Optional[str] = Field(None, description="mentioned | not_mentioned")
    payor_level_summary: Optional[str] = Field(None, description="Overall aggregated summary for this code")
    confidence: Optional[str] = Field(None, description="Extraction confidence: high | medium | low")

    # Structured edit-rule / utilization-management facts — power the
    # specific recommendation generator. All optional / default []
    # for backward compatibility.
    action_type: Optional[str] = Field(
        None,
        description="deny | bundle | require_auth | pay_separately | allow_with_conditions | limit",
    )
    target_codes: List[str] = Field(default_factory=list, description="Codes the rule acts on")
    related_codes: List[str] = Field(default_factory=list, description="HCPCS/CPT codes referenced in the condition")
    required_modifiers: List[str] = Field(default_factory=list, description="Modifiers that must be present")
    excluded_modifiers: List[str] = Field(default_factory=list, description="Modifiers that trigger the action")
    revenue_codes: List[str] = Field(
        default_factory=list,
        description="Revenue codes verbatim, preserving X wildcards (e.g., 045X)",
    )
    pairing_conditions: List[str] = Field(
        default_factory=list,
        description="Short pairing/timing phrases (e.g., 'same claim line', 'day before ED visit')",
    )
    utilization_limits: List[str] = Field(
        default_factory=list,
        description="Numeric limits with units (e.g., '10 hrs/week BCBA', 'max 8 hours per DOS')",
    )
    prior_auth_thresholds: List[str] = Field(
        default_factory=list,
        description="PA trigger phrases (e.g., 'PA required after 20 sessions of 90837')",
    )
    discharge_status_conditions: List[str] = Field(
        default_factory=list,
        description="Discharge-status restrictions formatted 'discharge status <code> (<label>)'",
    )
    program_scope: List[str] = Field(
        default_factory=list,
        description="Programs the rule applies to (e.g., COPPS, DRG, APC, OPPS, MS-DRG)",
    )
    state_specific_rules: List[str] = Field(
        default_factory=list,
        description="State-scoped rules (e.g., 'NC: 20 session cap')",
    )
    provider_role_restrictions: List[str] = Field(
        default_factory=list,
        description="Provider/role oversight requirements (e.g., 'BCBA oversight required')",
    )
    exemptions: List[str] = Field(
        default_factory=list,
        description="Conditions that exempt a claim (e.g., 'observation RC 0762')",
    )


class PolicyMetadata(BaseModel):
    """Policy document metadata."""
    
    policy_title: Optional[str] = Field(None, description="Full title of the policy document")
    effective_date: Optional[str] = Field(None, description="Policy effective date in MM/DD/YYYY format")
    payer_category: Optional[str] = Field(None, description="Payer type (Medicaid, Commercial, Medicare Advantage, etc.)")
    appeals_process_documented: bool = Field(False, description="Whether policy documents an appeals process")


class PolicyExtractionResponse(BaseModel):
    """Complete policy extraction response from LLM."""
    
    policy_metadata: PolicyMetadata = Field(..., description="Policy metadata")
    results: List[PolicyRuleResult] = Field(default_factory=list, description="List of rule extractions per CPT code")


class ColumnDefinition(BaseModel):
    """Table column definition."""
    
    id: str = Field(..., description="Column identifier")
    label: str = Field(..., description="Column label")
    type: Literal["text", "badge", "date"] = Field("text", description="Column type")


class ColumnLabelsResponse(BaseModel):
    """Column labels generation response from LLM."""
    
    columns: List[ColumnDefinition] = Field(default_factory=list, description="List of column definitions")


class RuleSummary(BaseModel):
    """Rule summary for a payer."""
    
    summary: str = Field(..., description="Summary text (max 15 words)")


class RecommendedAction(BaseModel):
    """Structured recommendation action item."""
    
    rank: int = Field(..., description="Recommendation rank/priority order")
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(..., description="Priority level")
    category: Literal["Policy", "Operational", "Clinical", "Financial"] = Field(..., description="Recommendation category")
    description: str = Field(..., description="Action description (2-3 sentences max)")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence from policy table")
    story_alignment: List[str] = Field(default_factory=list, description="How recommendation aligns with pattern findings")
    peer_benchmarking: List[str] = Field(default_factory=list, description="Peer payer comparison data")
    citation: List[str] = Field(default_factory=list, description="Policy citations and sources")


class ColumnCategoryMap(BaseModel):
    """Maps a column id to policy rule categories that should populate it."""
    
    id: str = Field(..., description="Column id used in rows")
    categories: List[str] = Field(..., description="Rule categories to aggregate")


class TableSchemaResponse(BaseModel):
    """LLM response for dynamic table schema (from notebook)."""
    
    columns: List[ColumnDefinition] = Field(default_factory=list, description="Dynamic columns for this pattern")
    selected_categories: List[ColumnCategoryMap] = Field(default_factory=list, description="Category mappings for columns")


class ElevanceSummaryResponse(BaseModel):
    """LLM response for Elevance executive summary (from notebook)."""
    
    explainable: bool = Field(..., description="True if the pattern is plausibly explained by Elevance policy evidence")
    summary: Optional[str] = Field(None, description="1-3 sentence executive summary when explainable is True")


class PolicyRecommendationResponse(BaseModel):
    """LLM response for policy recommendations."""
    
    has_recommendation: bool = Field(..., description="True if a policy-driven recommendation is justified")
    recommendations: List[RecommendedAction] = Field(
        default_factory=list, 
        description="List of structured recommendations (1-3 items)"
    )


class ValidationCheck(BaseModel):
    """Validation check result for API responses."""

    check: str = Field(..., description="Validation check identifier")
    passed: bool = Field(..., description="True when the check passed")
    message: Optional[str] = Field(None, description="Optional check details")


class ReimbursementValidationSchema(BaseModel):
    """Standard validation payload for reimbursement API responses."""

    model_config = ConfigDict(extra="ignore")

    is_valid: Union[bool, Literal["valid_with_warnings"]] = Field(
        True,
        description=(
            "True when all validation checks passed. False when no policies "
            "were successfully extracted (truly invalid). "
            "'valid_with_warnings' when extractions succeeded but one or more "
            "critical contamination/relevance checks failed."
        ),
    )
    checks: List[ValidationCheck] = Field(default_factory=list, description="Validation checks performed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal validation warnings")
    errors: List[str] = Field(default_factory=list, description="Validation errors that indicate failure")


class ReimbursementTokensSchema(BaseModel):
    """Standard token usage payload for reimbursement API responses."""

    model_config = ConfigDict(extra="ignore")

    input: int = Field(0, description="Input token count")
    output: int = Field(0, description="Output token count")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Optional token breakdown details")


class ReimbursementExecutionSchema(BaseModel):
    """Standard execution metadata payload for reimbursement API responses."""

    model_config = ConfigDict(extra="ignore")

    start_time: Optional[str] = Field(None, description="Execution start time (UTC ISO format)")
    end_time: Optional[str] = Field(None, description="Execution end time (UTC ISO format)")
    duration_ms: int = Field(0, description="Execution duration in milliseconds")
    version: Optional[str] = Field(None, description="Agent version")


class ReimbursementAgentResponseSchema(BaseModel):
    """Standard API response schema for reimbursement agent output."""

    model_config = ConfigDict(extra="ignore")

    job_id: Optional[str] = Field(None, description="Job identifier for the execution")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier from orchestrator")
    agent: str = Field(..., description="Agent name")
    status: str = Field(..., description="Execution status: success | partial_success | failed")
    output: Dict[str, Any] = Field(default_factory=dict, description="Agent output payload")
    recommended_action: List[RecommendedAction] = Field(
        default_factory=list,
        description="Structured recommendations generated by the agent",
    )
    visual_component: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional visualization payload",
    )
    explanation: Dict[str, Any] = Field(default_factory=dict, description="Execution explanation metadata")
    validation: ReimbursementValidationSchema = Field(
        default_factory=ReimbursementValidationSchema,
        description="Validation payload",
    )
    tokens: ReimbursementTokensSchema = Field(
        default_factory=ReimbursementTokensSchema,
        description="Token usage details",
    )
    execution: ReimbursementExecutionSchema = Field(
        default_factory=ReimbursementExecutionSchema,
        description="Execution timing and version metadata",
    )


class PolicyRelevanceScore(BaseModel):
    """Relevancy score for a single policy."""
    
    model_config = ConfigDict(extra="ignore")
    
    policy_id: str = Field(
        description="The policy ID being scored"
    )
    relevancy_score: int = Field(
        ge=0,
        le=100,
        description="Relevancy score from 0-100, where 100 is most relevant"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the score"
    )


class PolicyTriageResponse(BaseModel):
    """LLM verdict on which non-Elevance policy_ids to keep when search returns too many."""

    model_config = ConfigDict(extra="ignore")

    selected_policy_ids: List[str] = Field(
        default_factory=list,
        description="policy_id values to keep, ordered by relevance (most relevant first)",
    )
    
    policy_scores: Optional[List[PolicyRelevanceScore]] = Field(
        default=None,
        description="Detailed relevancy scores for each policy"
    )
