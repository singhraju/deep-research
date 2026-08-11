"""
Preferred Provider Agent (Scaffold)

This agent identifies and summarizes preferred providers by combining:
- Optional inputs from ReimbursementAgent (policy/hints)
- Snowflake-derived provider metrics
- LLM-generated executive summary

The initial implementation is a runnable stub wired to the common framework.
Replace the placeholder Snowflake query and ranking logic with your notebook
logic and update output fields accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from deep_research_core.base_agent import AgentBase, CredentialProvider
from deep_research_utils.app_constant import AppConstants

try:
    from deep_research_utils.logger_config import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - dev fallback
    logger = logging.getLogger(__name__)


class CorrelationAgentResponse(TypedDict, total=False):
    """Correlation agent output structure."""
    job_id: str
    conversation_id: str
    agent: str
    status: str
    recommended_action: List[Dict[str, Any]]
    visual_component: Dict[str, Any]
    output: Dict[str, Any]
    explanation: Dict[str, Any]
    validation: Dict[str, Any]
    tokens: Dict[str, int]
    execution: Dict[str, Any]


class PreferredProviderState(TypedDict, total=False):
    """State for PreferredProviderAgent."""

    # Inputs
    question: str
    context: Dict[str, Any]
    correlation_summary: CorrelationAgentResponse
    run_id: str

    # Resources
    snowflake_helper: Any
    llm: Any

    # Outputs
    result: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    warnings: List[str]


class PreferredProviderAgent(AgentBase):
    """
    Preferred Provider Agent

    Responsibilities (intended):
    1) Optionally consume correlation agent results as hints/constraints
    2) Query Snowflake for provider metrics and attributes
    3) Rank/select preferred providers using a transparent score
    4) Produce an executive summary via LLM
    5) Persist artifacts (CSV/manifest) under CORRELATION_OUTPUT_ROOT/run_id

    Current implementation is a scaffold with a safe, runnable stub.
    """

    def __init__(
        self,
        *,
        snowflake_helper: Optional[Any] = None,
        snowflake_helper_builder: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize base (logging, LLM, graph)
        agent_name = kwargs.pop("agent_name", "preferred_provider")
        super().__init__(
            agent_name=agent_name,
            state_class=PreferredProviderState,
            **kwargs,
        )
        # Initialize Snowflake
        self.snowflake_helper = self._init_snowflake(snowflake_helper, snowflake_helper_builder)

    @property
    def node_name(self) -> str:
        return "preferred_provider"

    def _init_snowflake(
        self,
        helper: Optional[Any],
        builder: Optional[Callable[[], Any]],
    ) -> Any:
        if helper is not None:
            return helper
        if builder is not None:
            try:
                return builder()
            except Exception as exc:  # pragma: no cover - pass-through
                self.logger.error(f"Snowflake builder failed: {exc}")
                raise
        # Auto-build from environment if available
        try:
            from deep_research_utils.snowflake_helper import SnowparkHelper

            creds = CredentialProvider.get_instance()
            snowflake_creds = creds.get_snowflake_credentials()
            return SnowparkHelper(
                batch_size=10000,
                max_workers=6,
                enable_metrics=True,
                connection_pool_size=4,
                **snowflake_creds,
            )
        except Exception as exc:
            # Optional dependency; agent can still operate in stub mode
            self.logger.warning(
                "Snowflake helper unavailable; running in stub mode. Provide a SnowparkHelper to enable queries.",
                exc_info=exc,
            )
            return None

    def prepare_state(
        self,
        *,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_summary: Optional[CorrelationAgentResponse] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "context": context or {},
            "correlation_summary": correlation_summary or {},
            "run_id": run_id or "",
            "llm": self.llm,
            "snowflake_helper": self.snowflake_helper,
        }

    def node_function(self, state: PreferredProviderState) -> Dict[str, Any]:
        ctx = state.get("context", {})
        correlation = state.get("correlation_summary", {})
        helper = state.get("snowflake_helper")
        llm = state.get("llm")

        warnings: List[str] = []

        # 1) Extract provider names from correlation summary
        correlation_hint = bool(correlation)
        provider_names: List[str] = []
        
        if correlation and correlation.get("output"):
            drill_path = correlation["output"].get("drill_path", [])
            # Find nodes with dimension "rendering_provider_name" and extract provider names
            for node in drill_path:
                if node.get("dimension") == "rendering_provider_name":
                    top_segments = node.get("top_segments", [])
                    provider_names.extend([seg.get("value", "") for seg in top_segments if seg.get("value")])
        
        # Remove duplicates while preserving order
        seen = set()
        provider_names = [x for x in provider_names if not (x in seen or seen.add(x))]
        
        self.logger.info(f"Extracted {len(provider_names)} provider names from correlation output")

        # 2) Execute Snowflake query with provider name filtering
        providers_ranked: List[Dict[str, Any]] = []
        if helper is None:
            warnings.append(
                "Snowflake helper not configured; returning stubbed provider list."
            )
            # Minimal stubbed data structure to prove contract
            providers_ranked = [
                {"provider_id": "P0001", "name": "Example Health", "score": 0.0, "metrics": {}},
            ]
        else:
            if not provider_names:
                warnings.append("No provider names found in correlation summary; returning empty results.")
            else:
                try:
                    # Build WHERE clause for provider name filtering
                    provider_names_quoted = [f"'{name.replace(chr(39), chr(39) + chr(39))}'" for name in provider_names]
                    provider_filter = f"ddc_cd_prvdr_nme IN ({', '.join(provider_names_quoted)})"
                    
                    # Modified SQL query with provider name filtering
                    sql_query = f"""
                    SELECT 
                        CLMP.ddc_cd_dcn, 
                        ddc_cd_prvdr_nme, 
                        ddc_cd_prvdr_tax_id, 
                        ddc_cd_bha_prvdr_ind, 
                        ddc_cd_prvdr_spclty_cde, 
                        ddc_cd_prvdr_on_aud,
                        ddc_cd_prvdr_type_cde, 
                        CASE 
                            WHEN CLMP.DDC_CD_LMTD_RELTD_IND = 'P' THEN 'Par'
                            WHEN CLMP.DDC_CD_LMTD_RELTD_IND = 'N' THEN 'Non-Par'
                            WHEN ((SUBSTR(CLMP.DDC_CD_DCN,6,2) IN ('48','87','08','47','49','1A')) OR
                                    (SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'MA' AND 'MZ') or
                                    (SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'M1' AND 'M9') or
                                    (SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'NA' AND 'NZ') or
                                    (SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'N1' AND 'N9'))
                                    AND TRIM(CLMP.DDC_CD_ITS_ORIG_SCCF_NBR_NEW) <> ' ' AND CLMP.DDC_CD_ITS_HOST_PRVDR_IND IN ('Y','P')) then 'Par'
                            WHEN ((SUBSTR(CLMP.DDC_CD_DCN,6,2) IN ('48','87','08','47','49','1A') or
                                    SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'MA' AND 'MZ' or
                                    SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'M1' AND 'M9' or
                                    SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'NA' AND 'NZ' or
                                    SUBSTR(CLMP.DDC_CD_DCN,6,2) BETWEEN 'N1' AND 'N9') 
                                    AND TRIM(CLMP.DDC_CD_ITS_ORIG_SCCF_NBR_NEW) <> ' ' AND CLMP.DDC_CD_ITS_HOST_PRVDR_IND IN ('N')) then 'Non-Par'
                            WHEN CLMP.DDC_CD_PAR_KEYED_IND IN ('P','Y') then 'Par'
                            WHEN CLMP.DDC_CD_PAR_KEYED_IND IN ('N') then 'Non-Par'
                            WHEN CLMP.DDC_CD_MX_PAR_IND IN ('Y', 'E', 'X', 'T', 'F', 'U', '2', '1', '4', '7', 'I') then 'Par'
                            WHEN CLMP.DDC_CD_MX_PAR_IND IN ('N','D','M','A','Q','+','C','3','K','Z') then 'Non-Par'
                                ELSE 'Non-Par' 
                        END AS REND_PROV_PAR_DESC
                    from P01_EDL.EDL_RAWZ_CMPCT_ALLPHI.CLM_WGS_GNCCLMP_CMPCT clmp 
                    where edl_load_dtm > current_date()-366
                    AND {provider_filter}
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY ddc_cd_prvdr_nme ORDER BY edl_load_dtm DESC) <= 5
                    """
                    
                    self.logger.info(f"Executing Snowflake query for {len(provider_names)} providers")
                    df = helper.execute_query_and_return_pandas_df(sql_query)
                    
                    # Convert DataFrame to list of dictionaries
                    providers_ranked = df.to_dict('records')
                    self.logger.info(f"Retrieved {len(providers_ranked)} provider records from Snowflake")
                    
                except Exception as exc:
                    self.logger.error(f"Snowflake query failed: {exc}", exc_info=exc)
                    warnings.append(f"Snowflake query failed: {str(exc)}")
                    providers_ranked = []

        # 3) LLM executive summary (optional)
        executive_summary = ""
        if llm is not None:
            try:
                prompt = (
                    "Summarize the preferred provider findings in 3-4 concise bullet points. "
                    "Focus on why the highlighted providers are preferred and any key caveats."
                )
                response = self._invoke_with_token_retry([
                    {"role": "system", "content": "You are a healthcare analytics assistant."},
                    {"role": "user", "content": prompt},
                ])
                executive_summary = getattr(response, "content", "") or ""
            except Exception as exc:  # pragma: no cover - safety net
                self.logger.warning(f"LLM summary failed: {exc}")
                warnings.append("LLM summary failed; returning without narrative.")
        else:
            executive_summary = "Preferred provider analysis (stub)."

        # 4) Persist artifacts (optional; leave empty in scaffold)
        artifacts: List[Dict[str, Any]] = []

        result: Dict[str, Any] = {
            "providers_ranked": providers_ranked,
            "supporting_evidence": {
                "correlation_hint_used": correlation_hint,
                "filters": ctx,
            },
            "executive_summary": executive_summary,
            "artifacts": artifacts,
            "warnings": warnings,
            "metadata": {"is_stub": True},
        }
        return {"result": result}

    def extract_result(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        return graph_output.get("result", {})

    # Provide a stub LLM for test_mode=True
    def create_stub_llm(self) -> Any:  # pragma: no cover - simple stub
        class _Stub:
            def invoke(self, messages: List[Dict[str, str]]):
                class _Resp:
                    content = "Stub: preferred provider summary"
                return _Resp()
        return _Stub()


# Convenience factory for simple usage and testing
@dataclass(frozen=True)
class PreferredProviderConfig:
    llm_builder: Optional[Callable[[], Any]] = None
    snowflake_helper: Optional[Any] = None
    snowflake_helper_builder: Optional[Callable[[], Any]] = None


def build_app(
    *,
    llm_builder: Optional[Callable[[], Any]] = None,
    snowflake_helper: Optional[Any] = None,
    snowflake_helper_builder: Optional[Callable[[], Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Return a callable(question, context, correlation_summary, run_id) -> result dict."""
    agent = PreferredProviderAgent(
        llm_builder=llm_builder,
        snowflake_helper=snowflake_helper,
        snowflake_helper_builder=snowflake_helper_builder,
    )

    def _run(
        *,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_summary: Optional[CorrelationAgentResponse] = None,
        run_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        return agent.execute(
            question=question,
            context=context,
            correlation_summary=correlation_summary,
            run_id=run_id,
        )

    # Expose compiled graph for streaming UIs if needed
    _run.graph = getattr(agent, "app", None)  # type: ignore[attr-defined]
    return _run
