"""Tests for the LLM-driven semantic role classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from deep_research_utils.semantic_role_classifier import (
    ALLOWED_ROLES,
    _build_dimension_catalog,
    _compute_source_hash,
    _drill_dimensions_for_mode,
    _parse_llm_json,
    _partition_valid_invalid,
    classify_dimension_semantic_roles,
    companion_json_path,
    load_companion_json,
    source_hash_for_yaml,
)


class StubLLM:
    """Minimal LangChain-compatible stub.

    ``responses`` is a list of dicts or strings, popped in FIFO order per
    invoke() call — lets tests script the retry path.
    """

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.invocations: List[List[Dict[str, str]]] = []
        self.model_name = "stub-model-v1"

    def invoke(self, messages, **_kwargs):
        self.invocations.append(messages)
        if not self._responses:
            raise AssertionError("StubLLM ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def sample_yaml_doc() -> Dict[str, Any]:
    return {
        "name": "test_view",
        "description": "Test semantic view",
        "tables": [
            {
                "name": "claims",
                "base_table": {"database": "DB", "schema": "S", "table": "T"},
                "dimensions": [
                    {
                        "name": "provider_state_code",
                        "description": "Two-letter state code of the servicing provider",
                        "expr": "PROV_STATE",
                        "data_type": "string",
                        "synonyms": ["state"],
                        "sample_values": ["CA", "TX", "NY"],
                    },
                    {
                        "name": "src_provider_zip_code",
                        "description": "5-digit ZIP code of the servicing provider",
                        "expr": "PROV_ZIP",
                        "data_type": "string",
                        "sample_values": ["95816", "10001"],
                    },
                    {
                        "name": "health_service_code",
                        "description": "CPT/HCPCS procedure code billed on the claim line",
                        "expr": "HLTH_SVC_CD",
                        "data_type": "string",
                        "sample_values": ["99213", "A0427"],
                    },
                    {
                        "name": "member_gender",
                        "description": "Member gender code",
                        "expr": "MBR_GNDR",
                        "data_type": "string",
                        "sample_values": ["M", "F"],
                    },
                    {
                        "name": "unlisted_dim",
                        "description": "Not referenced by any analysis mode",
                        "expr": "UNLISTED",
                        "data_type": "string",
                    },
                ],
            }
        ],
        "analysis_modes": [
            {
                "name": "cost_change_investigation",
                "drill_dimensions": [
                    "provider_state_code",
                    "src_provider_zip_code",
                    "health_service_code",
                    "member_gender",
                ],
            },
            {
                "name": "auth_investigation",
                "drill_dimensions": [
                    "provider_state_code",
                    "src_provider_zip_code",
                ],
            },
        ],
    }


@pytest.fixture
def sample_yaml_path(tmp_path: Path, sample_yaml_doc: Dict[str, Any]) -> Path:
    p = tmp_path / "view.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(sample_yaml_doc, f)
    return p


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def test_drill_dimensions_scoped_to_mode(sample_yaml_doc: Dict[str, Any]) -> None:
    dims = _drill_dimensions_for_mode(sample_yaml_doc, "cost_change_investigation")
    assert dims == [
        "provider_state_code",
        "src_provider_zip_code",
        "health_service_code",
        "member_gender",
    ]
    # ``unlisted_dim`` is in the YAML but not the drill_dimensions list.
    assert "unlisted_dim" not in dims


def test_drill_dimensions_union_across_modes(sample_yaml_doc: Dict[str, Any]) -> None:
    dims = _drill_dimensions_for_mode(sample_yaml_doc, None)
    assert set(dims) == {
        "provider_state_code",
        "src_provider_zip_code",
        "health_service_code",
        "member_gender",
    }


def test_drill_dimensions_unknown_mode_raises(sample_yaml_doc: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Analysis mode not found"):
        _drill_dimensions_for_mode(sample_yaml_doc, "does_not_exist")


def test_source_hash_is_deterministic(sample_yaml_doc: Dict[str, Any]) -> None:
    catalog = _build_dimension_catalog(sample_yaml_doc)
    dims = _drill_dimensions_for_mode(sample_yaml_doc, "cost_change_investigation")
    h1 = _compute_source_hash(dims, catalog)
    # Reordering the input list must not change the hash.
    h2 = _compute_source_hash(list(reversed(dims)), catalog)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_source_hash_changes_on_dim_edit(
    sample_yaml_doc: Dict[str, Any], tmp_path: Path
) -> None:
    catalog = _build_dimension_catalog(sample_yaml_doc)
    dims = _drill_dimensions_for_mode(sample_yaml_doc, "cost_change_investigation")
    h_before = _compute_source_hash(dims, catalog)

    edited_doc = json.loads(json.dumps(sample_yaml_doc))  # deep copy
    edited_doc["tables"][0]["dimensions"][0]["description"] = "new description"
    catalog_after = _build_dimension_catalog(edited_doc)
    h_after = _compute_source_hash(dims, catalog_after)
    assert h_before != h_after


def test_source_hash_for_yaml_matches_internal(sample_yaml_path: Path) -> None:
    computed = source_hash_for_yaml(sample_yaml_path, "cost_change_investigation")
    assert computed.startswith("sha256:")


# ---------------------------------------------------------------------------
# LLM output parsing + taxonomy validation
# ---------------------------------------------------------------------------


def test_parse_llm_json_from_string() -> None:
    raw = '{"a": "state", "b": "zip"}'
    assert _parse_llm_json(raw) == {"a": "state", "b": "zip"}


def test_parse_llm_json_strips_fenced_block() -> None:
    raw = '```json\n{"a": "state"}\n```'
    assert _parse_llm_json(raw) == {"a": "state"}


def test_parse_llm_json_from_message_object() -> None:
    class _Msg:
        content = '{"a": "state"}'

    assert _parse_llm_json(_Msg()) == {"a": "state"}


def test_partition_valid_invalid_fuzz() -> None:
    mapping = {
        "d1": "state",
        "d2": "zip",
        "d3": "PROCEDURE",           # wrong case — invalid
        "d4": "made_up_role",        # not in taxonomy
        "d5": "code",                # generic, not in taxonomy
        "d6": "",                    # empty
        "d7": "provider_specialty",  # valid
    }
    valid, invalid = _partition_valid_invalid(mapping)
    assert valid == {"d1": "state", "d2": "zip", "d7": "provider_specialty"}
    assert set(invalid.keys()) == {"d3", "d4", "d5", "d6"}


def test_allowed_roles_covers_taxonomy() -> None:
    for role in ("procedure_code", "drg_code", "state", "zip", "member_id", "other"):
        assert role in ALLOWED_ROLES


# ---------------------------------------------------------------------------
# classify_dimension_semantic_roles end-to-end
# ---------------------------------------------------------------------------


def test_classify_happy_path_writes_companion_json(sample_yaml_path: Path) -> None:
    llm_response = {
        "provider_state_code": "state",
        "src_provider_zip_code": "zip",
        "health_service_code": "procedure_code",
        "member_gender": "gender",
    }
    llm = StubLLM(responses=[llm_response])

    payload = classify_dimension_semantic_roles(
        sample_yaml_path,
        scope_to_analysis_mode="cost_change_investigation",
        llm=llm,
    )

    assert payload["dimension_roles"] == llm_response
    assert payload["view_name"] == "test_view"
    assert payload["analysis_mode"] == "cost_change_investigation"
    assert payload["source_hash"].startswith("sha256:")
    assert payload["llm_model"] == "stub-model-v1"

    expected_path = companion_json_path(sample_yaml_path, "cost_change_investigation")
    assert expected_path.exists()
    on_disk = load_companion_json(expected_path)
    assert on_disk is not None
    assert on_disk["dimension_roles"] == llm_response
    assert on_disk["source_hash"] == payload["source_hash"]

    # Regression assertion — ZIP must not become procedure_code.
    assert payload["dimension_roles"]["src_provider_zip_code"] == "zip"
    assert payload["dimension_roles"]["health_service_code"] == "procedure_code"

    assert len(llm.invocations) == 1  # No retry needed on a clean response.


def test_classify_retries_then_coerces_invalid_labels(sample_yaml_path: Path) -> None:
    first_response = {
        "provider_state_code": "state",
        "src_provider_zip_code": "geographical_thing",   # invalid
        "health_service_code": "procedure",              # invalid
        "member_gender": "gender",
    }
    retry_response = {
        "src_provider_zip_code": "zip",                  # fixed on retry
        "health_service_code": "totally_bogus",          # still invalid
    }
    llm = StubLLM(responses=[first_response, retry_response])

    payload = classify_dimension_semantic_roles(
        sample_yaml_path,
        scope_to_analysis_mode="cost_change_investigation",
        llm=llm,
        write_output=False,
    )

    assert len(llm.invocations) == 2  # invocation + retry
    roles = payload["dimension_roles"]
    assert roles["provider_state_code"] == "state"       # kept from first pass
    assert roles["src_provider_zip_code"] == "zip"       # fixed by retry
    assert roles["health_service_code"] == "other"       # coerced after retry
    assert roles["member_gender"] == "gender"            # kept from first pass


def test_classify_scopes_to_analysis_mode(sample_yaml_path: Path) -> None:
    llm_response = {
        "provider_state_code": "state",
        "src_provider_zip_code": "zip",
    }
    llm = StubLLM(responses=[llm_response])

    payload = classify_dimension_semantic_roles(
        sample_yaml_path,
        scope_to_analysis_mode="auth_investigation",
        llm=llm,
        write_output=False,
    )

    assert set(payload["dimension_roles"]) == {
        "provider_state_code",
        "src_provider_zip_code",
    }
    # Prompt payload sent to the LLM must not include out-of-scope dims.
    user_msg = llm.invocations[0][-1]["content"]
    assert "health_service_code" not in user_msg
    assert "member_gender" not in user_msg


def test_classify_omitted_dim_coerced_to_other(sample_yaml_path: Path) -> None:
    partial_response = {
        "provider_state_code": "state",
        "src_provider_zip_code": "zip",
        # health_service_code + member_gender omitted entirely
    }
    llm = StubLLM(responses=[partial_response])

    payload = classify_dimension_semantic_roles(
        sample_yaml_path,
        scope_to_analysis_mode="cost_change_investigation",
        llm=llm,
        write_output=False,
    )

    assert payload["dimension_roles"]["health_service_code"] == "other"
    assert payload["dimension_roles"]["member_gender"] == "other"


def test_classify_requires_llm(sample_yaml_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an `llm`"):
        classify_dimension_semantic_roles(sample_yaml_path, llm=None)


def test_companion_json_path_naming(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pi_wgs_mad.yaml"
    scoped = companion_json_path(yaml_path, "cost_change_investigation")
    assert scoped.name == "pi_wgs_mad_cost_change_investigation_semantic_roles.json"

    unscoped = companion_json_path(yaml_path, None)
    assert unscoped.name == "pi_wgs_mad_semantic_roles.json"


def test_dimension_catalog_captures_metadata(sample_yaml_doc: Dict[str, Any]) -> None:
    catalog = _build_dimension_catalog(sample_yaml_doc)
    zip_entry = catalog["src_provider_zip_code"]
    assert zip_entry["description"].startswith("5-digit ZIP")
    assert "95816" in zip_entry["sample_values"]
    assert zip_entry["data_type"] == "string"
