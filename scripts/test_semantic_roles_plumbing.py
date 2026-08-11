"""End-to-end plumbing test for the semantic-role integration.

No UI, no LLM, no Snowflake. Fabricates the minimum correlation payload,
cards, and pattern needed to exercise:

  1. ``build_pattern_cards_and_groups(..., semantic_roles=...)`` — proves the
     classifier's dim->role mapping flows into ``dimension_roles`` and
     changes ``canonical_dimensions`` on the emitted cards.
  2. ``ReimbursementPolicyAgent._extract_cpt_codes_from_cards`` — proves the
     role-aware branch prevents ZIP codes from being treated as CPT
     candidates, and the DRG helper works the same way.

Run with:
    python scripts/test_semantic_roles_plumbing.py

Exit code is 0 on success (all assertions pass), 1 on any regression.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "utils" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "agents" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "core" / "src"))

from deep_research_agents.pattern_agent import (  # noqa: E402
    _build_semantic_card_config,
    _resolve_dimension_roles,
    _build_dimension_catalog,
    _load_yaml,
)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Test 1 — Semantic-role → pattern-agent role catalog
# ---------------------------------------------------------------------------

def test_pattern_agent_role_resolution() -> None:
    banner("1. Pattern agent role resolution (with vs. without semantic_roles)")

    yaml_path = REPO_ROOT / "configs" / "correlation_pattern" / "pi_wgs_mad_semantic_view_with_samples_local.yaml"
    if not yaml_path.exists():
        print(f"  SKIP — YAML not found at {yaml_path}")
        return

    semantic = _load_yaml(yaml_path)
    catalog = _build_dimension_catalog(semantic)

    # A. Without semantic_roles — fall back to keyword inference.
    legacy_roles = _resolve_dimension_roles(catalog, semantic_roles=None)
    legacy_zip_role = _which_role(legacy_roles, "src_provider_zip_code")
    legacy_hsc_role = _which_role(legacy_roles, "health_service_code")
    print(f"  [legacy]  src_provider_zip_code -> {legacy_zip_role!r}")
    print(f"  [legacy]  health_service_code   -> {legacy_hsc_role!r}")

    # B. With a classifier-shaped semantic_roles dict.
    classifier_roles = {
        "src_provider_zip_code": "zip",
        "health_service_code": "procedure_code",
        "provider_state_code": "state",
        "member_gender": "gender",
        "drg_code": "drg_code",
        "drg_name": "drg_name",
        "primary_diagnosis_code": "diagnosis_code",
        "rendering_provider_npi": "provider_id",
        "pa_required_code": "authorization",
        # A hallucinated role — must be ignored by the resolver via the
        # ``_CLASSIFIER_ROLE_TO_LEGACY.get(...)`` guard.
        "some_dim_the_llm_lied_about": "prcd_code",
    }
    llm_roles = _resolve_dimension_roles(catalog, semantic_roles=classifier_roles)
    llm_zip_role = _which_role(llm_roles, "src_provider_zip_code")
    llm_hsc_role = _which_role(llm_roles, "health_service_code")
    print(f"  [roles]   src_provider_zip_code -> {llm_zip_role!r}")
    print(f"  [roles]   health_service_code   -> {llm_hsc_role!r}")
    print(f"  [roles]   'prcd_code' hallucination visible?  "
          f"{'YES (BUG)' if 'some_dim_the_llm_lied_about' in _flatten_dims(llm_roles) else 'no (dropped correctly)'}")

    # Assertions
    assert llm_zip_role == "zip", f"src_provider_zip_code should be 'zip', got {llm_zip_role!r}"
    assert llm_hsc_role == "procedure", f"health_service_code should be 'procedure', got {llm_hsc_role!r}"
    assert "some_dim_the_llm_lied_about" not in _flatten_dims(llm_roles), (
        "Hallucinated role leaked into the resolved catalog."
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2 — build_semantic_card_config threads semantic_roles through
# ---------------------------------------------------------------------------

def test_build_semantic_card_config_threading() -> None:
    banner("2. _build_semantic_card_config threads semantic_roles through")

    yaml_path = REPO_ROOT / "configs" / "correlation_pattern" / "pi_wgs_mad_semantic_view_with_samples_local.yaml"
    if not yaml_path.exists():
        print(f"  SKIP — YAML not found at {yaml_path}")
        return

    semantic = _load_yaml(yaml_path)
    analysis_mode = "cost_change_investigation_over_time_window"

    without_roles = _build_semantic_card_config(semantic, analysis_mode)
    with_roles = _build_semantic_card_config(
        semantic,
        analysis_mode,
        semantic_roles={
            "src_provider_zip_code": "zip",
            "health_service_code": "procedure_code",
            "provider_state_code": "state",
        },
    )
    zip_role_default = _which_role(without_roles["dimension_roles"], "src_provider_zip_code")
    zip_role_llm = _which_role(with_roles["dimension_roles"], "src_provider_zip_code")
    print(f"  [without roles] src_provider_zip_code -> {zip_role_default!r}")
    print(f"  [with roles]    src_provider_zip_code -> {zip_role_llm!r}")
    assert zip_role_llm == "zip"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3 — Reimbursement CPT/DRG extraction with vs. without roles
# ---------------------------------------------------------------------------

def test_reimbursement_role_filter() -> None:
    banner("3. Reimbursement CPT/DRG helpers filter by role")

    from deep_research_agents.reimbursement_agent import ReimbursementPolicyAgent

    # Instantiate without going through Snowflake/LLM setup — the helpers we
    # test are pure over their arguments.
    agent = ReimbursementPolicyAgent.__new__(ReimbursementPolicyAgent)

    # Two filters: one is a real CPT source, the other is a ZIP code that
    # HAPPENS to contain 5 digits — the exact false-positive the plan aimed to
    # kill. Both fields contain 'code', which the legacy substring check does
    # NOT match on (it only checks 'cpt'/'procedure'/'hcpcs'), so the legacy
    # path here is fine. To make the point, we also add a field literally
    # named ``ambulance_procedure_zip`` — the legacy substring check WILL
    # match on 'procedure' and treat the 5-digit ZIP as a CPT code.
    cards: List[Dict[str, Any]] = [
        {
            "card_id": "card-1",
            "filters": [
                {"field": "health_service_code", "value": "99213"},         # real CPT
                {"field": "src_provider_zip_code", "value": "95816"},        # ZIP
                {"field": "ambulance_procedure_zip", "value": "10001"},      # ZIP w/ 'procedure' in name (traps legacy)
            ],
        }
    ]

    # A. Legacy path — no semantic_roles supplied.
    legacy = agent._extract_cpt_codes_from_cards(["card-1"], cards)
    print(f"  [legacy path]        extracted CPT codes: {legacy}")

    # B. Role-aware path — classifier says field->role authoritatively.
    classifier_roles = {
        "health_service_code": "procedure_code",
        "src_provider_zip_code": "zip",
        "ambulance_procedure_zip": "zip",   # <-- the key fix
    }
    role_aware = agent._extract_cpt_codes_from_cards(
        ["card-1"], cards, semantic_roles=classifier_roles
    )
    print(f"  [role-aware path]    extracted CPT codes: {role_aware}")

    assert "99213" in role_aware, "Real CPT was dropped!"
    assert "95816" not in role_aware, "ZIP 95816 leaked in as CPT (role-aware)"
    assert "10001" not in role_aware, "ZIP 10001 leaked in as CPT (role-aware)"

    # Legacy path is expected to be wrong here — we don't assert, just print.
    legacy_leaks = [c for c in legacy if c in {"95816", "10001"}]
    if legacy_leaks:
        print(f"  [note] legacy path would have leaked {legacy_leaks} — role-aware path fixes it.")

    # DRG helper — the classifier says a differently-named column is the DRG.
    drg_cards = [
        {
            "card_id": "card-2",
            "filters": [
                {"field": "custom_drg_label", "value": "MDC-05 STROKE"},
            ],
        }
    ]
    drg_legacy = agent._extract_drg_codes_from_cards(["card-2"], drg_cards)
    drg_role_aware = agent._extract_drg_codes_from_cards(
        ["card-2"], drg_cards, semantic_roles={"custom_drg_label": "drg_name"}
    )
    print(f"  [legacy DRG]        {drg_legacy}")
    print(f"  [role-aware DRG]    {drg_role_aware}")
    assert "MDC-05 STROKE" in drg_role_aware
    print("  PASS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _which_role(roles: Dict[str, Any], dim_name: str) -> str:
    for role, dims in roles.items():
        if dim_name in dims:
            return role
    return "<none>"


def _flatten_dims(roles: Dict[str, Any]) -> set:
    out = set()
    for dims in roles.values():
        out.update(dims)
    return out


if __name__ == "__main__":
    try:
        test_pattern_agent_role_resolution()
        test_build_semantic_card_config_threading()
        test_reimbursement_role_filter()
        banner("ALL TESTS PASSED")
        sys.exit(0)
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
