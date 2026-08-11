"""LLM-driven semantic role classifier for YAML dimensions.

Classifies each ``drill_dimensions`` entry from a semantic-view YAML into one
normalized semantic role (``procedure_code``, ``state``, ``drg_name``, ...) so
downstream agents can identify columns by semantic role instead of brittle
keyword/regex heuristics.

Design notes
------------
* Model-agnostic — accepts any LangChain-compatible ``llm`` and any ``ehap``
  instance. The classifier itself does not hardcode model IDs; the caller
  routes through whichever EHAP endpoint they have provisioned.
* Companion JSON file — the mapping is written next to the YAML with a
  ``source_hash`` (sha256 of the sorted drill_dimensions metadata block) so
  the UI can detect YAML drift and short-circuit the LLM call on cache hits.
* Taxonomy-enforcing validator — LLM output is checked against the allowed
  set; invalid labels retry once, then get coerced to ``other`` with a
  warning. No silent hallucinations.
* CLI entry point — companion JSONs can be pre-generated in CI or dev
  without booting the UI (see ``python -m ...semantic_role_classifier``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from deep_research_utils.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allowed taxonomy (kept in sync with the prompt below)
# ---------------------------------------------------------------------------

ALLOWED_ROLES: Tuple[str, ...] = (
    # Geography
    "state", "zip", "county", "city", "region", "geography",
    # Demographics
    "age", "gender", "race", "ethnicity", "language", "demographic",
    # Business / operational
    "line_of_business", "product", "facility", "network", "authorization",
    "claim", "utilization", "financial", "plan", "contract", "status",
    # Clinical
    "clinical_category", "drg_code", "drg_name", "diagnosis_code",
    "diagnosis_name", "procedure_code", "procedure_name", "revenue_code",
    "place_of_service", "service_category",
    # Provider
    "provider", "provider_name", "provider_type", "provider_specialty",
    "provider_id",
    # Member / patient
    "member", "member_id", "patient", "patient_id",
    # Time
    "time", "date", "month", "quarter", "year",
    # Fallback
    "other",
)
_ALLOWED_ROLES_SET = frozenset(ALLOWED_ROLES)


SYSTEM_PROMPT = """You are a healthcare data schema classification expert.

Your task is to classify each column/dimension into one normalized semantic category based on the column name, description, synonyms, and sample values when available.

This classification is intended for healthcare financial analytics, including claims, utilization, authorization, provider, member, product, geography, demographics, time, and clinical reporting.

Allowed output categories:

Geography: state, zip, county, city, region, geography
Demographics: age, gender, race, ethnicity, language, demographic
Business/Operational: line_of_business, product, facility, network, authorization, claim, utilization, financial, plan, contract, status
Clinical: clinical_category, drg_code, drg_name, diagnosis_code, diagnosis_name, procedure_code, procedure_name, revenue_code, place_of_service, service_category
Provider: provider, provider_name, provider_type, provider_specialty, provider_id
Member/Patient: member, member_id, patient, patient_id
Time: time, date, month, quarter, year
Other: other

Classification rules:

1. Use the most specific category available.
   * "billing_provider_npi" -> "provider_id" (not "provider").
   * "diag_cd" -> "diagnosis_code" (not "clinical_category").
   * "member_age_band" -> "age" (not "demographic").

2. Geography fields describe physical, service, provider, member, or market location.
   * "provider_state" -> "state". "member_zip_code" -> "zip". "service_county" -> "county". "market_region" -> "region".

3. Demographic fields describe characteristics of a member, patient, subscriber, or population.
   * "member_age" -> "age". "member_gender" -> "gender". "race_code" -> "race". "ethnicity_description" -> "ethnicity". "preferred_language" -> "language".

4. Do NOT classify member or patient location fields as demographics.
   * "member_state" -> "state" (not "demographic"). "patient_zip" -> "zip".

5. Prefer healthcare-domain meaning over generic wording.
   * "LOB", "business segment", "market segment" -> "line_of_business".
   * "plan", "benefit plan", "insurance product" -> "product" or "plan".
   * "auth", "authorization", "prior authorization", "precert" -> "authorization".

6. Distinguish codes from names/descriptions when the allowed category supports it.
   * Columns ending in "_code", "_cd", "id", "num", or containing coded sample values are usually code/id categories.
   * Columns ending in "_name", "_desc", "_description", or containing readable labels are usually name/description categories.
   * If no separate code/name category exists, use the normalized concept category.
   * "race_cd" -> "race". "gender_desc" -> "gender".

7. Financial fields include amounts, allowed amounts, paid amounts, cost, expense, revenue, premium, savings, PMPM, and variance-related dimensions.
   * "allowed_amt_bucket" -> "financial". "paid_amount_range" -> "financial".

8. Time fields include service date, admission date, discharge date, claim paid date, year, month, quarter, week, and reporting period.
   * "svc_yr_mo" -> "month". "claim_paid_date" -> "date".

9. Use "other" only when no allowed category reasonably fits.

10. Do not invent categories outside the allowed list.

11. Return only a valid JSON object. Do not include explanations, markdown, comments, or extra text.

Input format:
You will receive a JSON array of column metadata. Each item may include:
* dimension_name, description, synonyms, sample_values, data_type

Output format:
Return a JSON object mapping each input dimension_name to exactly one allowed category.

Example:
{"service_area_state": "state", "member_zip_code": "zip", "health_service_code": "procedure_code", "billing_provider_npi": "provider_id", "claim_paid_month": "month", "allowed_amount_band": "financial"}
"""


USER_PROMPT_TEMPLATE = """Classify these dimensions:

{dimension_metadata_json}

Return ONLY the JSON mapping."""


STRICTER_RETRY_SUFFIX = """

STRICT NOTE: your previous response contained one or more labels outside the allowed set. Every value MUST be one of: {allowed_list}.
Do not invent new labels. Do not include markdown, comments, or explanations. Return only the JSON object."""


# ---------------------------------------------------------------------------
# YAML metadata extraction
# ---------------------------------------------------------------------------


def _load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_dimension_catalog(yaml_doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for table in yaml_doc.get("tables", []) or []:
        for section in ("dimensions", "time_dimensions"):
            for dim in table.get(section, []) or []:
                name = dim.get("name")
                if not name or name in catalog:
                    continue
                catalog[str(name)] = {
                    "dimension_name": str(name),
                    "description": str(dim.get("description") or "").strip(),
                    "synonyms": [str(s) for s in (dim.get("synonyms") or []) if s],
                    "sample_values": [str(v) for v in (dim.get("sample_values") or []) if v is not None],
                    "data_type": str(dim.get("data_type") or "").strip(),
                }
    return catalog


def _drill_dimensions_for_mode(
    yaml_doc: Mapping[str, Any],
    analysis_mode: Optional[str],
) -> List[str]:
    """Return drill_dimensions for a given mode, or the union across modes."""

    modes = yaml_doc.get("analysis_modes", []) or []
    if analysis_mode is None:
        seen: Dict[str, None] = {}
        for mode in modes:
            for name in mode.get("drill_dimensions", []) or []:
                if name:
                    seen[str(name)] = None
        return list(seen.keys())

    for mode in modes:
        if mode.get("name") == analysis_mode:
            return [str(n) for n in (mode.get("drill_dimensions", []) or []) if n]

    available = [mode.get("name") for mode in modes]
    raise ValueError(
        f"Analysis mode not found: {analysis_mode}. Available: {available}"
    )


def _build_llm_input(
    dim_names: Sequence[str],
    catalog: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the JSON payload sent to the LLM. Missing dims get name-only entries."""

    payload: List[Dict[str, Any]] = []
    for name in dim_names:
        entry = catalog.get(name)
        if entry is None:
            payload.append({"dimension_name": name})
            continue
        # Trim sample_values to keep prompts small — 5 is plenty for typing.
        entry_copy = dict(entry)
        samples = entry_copy.get("sample_values") or []
        if len(samples) > 5:
            entry_copy["sample_values"] = list(samples[:5])
        payload.append(entry_copy)
    return payload


def _compute_source_hash(
    dim_names: Sequence[str],
    catalog: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash of the sorted drill_dimensions metadata block.

    Two runs over the same YAML must produce identical hashes; edits to any
    included dimension's metadata must change it. Sorted key order + sorted
    lists = deterministic.
    """

    normalized: List[Dict[str, Any]] = []
    for name in sorted(dim_names):
        entry = catalog.get(name) or {"dimension_name": name}
        normalized.append(
            {
                "dimension_name": entry.get("dimension_name", name),
                "description": entry.get("description", ""),
                "data_type": entry.get("data_type", ""),
                "synonyms": sorted(entry.get("synonyms") or []),
                "sample_values": sorted(entry.get("sample_values") or []),
            }
        )
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# LLM invocation + validation
# ---------------------------------------------------------------------------


def _resolve_llm_model_id(llm: Any) -> str:
    for attr in ("model_name", "model", "deployment_name"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return llm.__class__.__name__ if llm is not None else "unknown"


def _parse_llm_json(raw: Any) -> Dict[str, str]:
    """Best-effort parse of LLM output into a dim -> role dict."""

    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}

    if hasattr(raw, "content"):
        raw = raw.content

    if isinstance(raw, list):
        raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw)

    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()
    # LLMs occasionally wrap JSON in fenced blocks despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        # Drop optional language tag on first line.
        first_newline = text.find("\n")
        if first_newline != -1:
            first_line = text[:first_newline].strip().lower()
            if first_line in {"json", "javascript", ""}:
                text = text[first_newline + 1 :]
    # Find outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM output did not contain a JSON object: {raw!r}")
    return {str(k): str(v) for k, v in json.loads(text[start : end + 1]).items()}


def _partition_valid_invalid(
    mapping: Mapping[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    valid: Dict[str, str] = {}
    invalid: Dict[str, str] = {}
    for dim, role in mapping.items():
        if role in _ALLOWED_ROLES_SET:
            valid[dim] = role
        else:
            invalid[dim] = role
    return valid, invalid


def _invoke_llm(llm: Any, messages: List[Dict[str, str]]) -> Any:
    """Invoke the LLM with plain messages (no structured output wrapper).

    We can't use ``with_structured_output`` because our schema is a dict with
    dynamic string keys, not a fixed Pydantic model. LangChain-compatible LLMs
    all accept the same message-list interface.
    """

    if hasattr(llm, "invoke"):
        return llm.invoke(messages)
    if callable(llm):
        return llm(messages)
    raise TypeError(f"LLM instance {type(llm).__name__} has no invoke() and is not callable")


def _classify_via_llm(
    llm: Any,
    dim_names: Sequence[str],
    catalog: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """Send one LLM call. Retry once on taxonomy violations; coerce leftovers."""

    payload = _build_llm_input(dim_names, catalog)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        dimension_metadata_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = _invoke_llm(llm, messages)
    parsed = _parse_llm_json(raw)
    valid, invalid = _partition_valid_invalid(parsed)

    if invalid:
        logger.warning(
            "LLM returned %d out-of-taxonomy labels on first pass: %s. Retrying with stricter prompt.",
            len(invalid),
            {k: v for k, v in list(invalid.items())[:10]},
        )
        retry_messages = list(messages)
        retry_messages[-1] = {
            "role": "user",
            "content": user_prompt + STRICTER_RETRY_SUFFIX.format(
                allowed_list=", ".join(ALLOWED_ROLES)
            ),
        }
        raw_retry = _invoke_llm(llm, retry_messages)
        parsed_retry = _parse_llm_json(raw_retry)
        valid_retry, invalid_retry = _partition_valid_invalid(parsed_retry)
        valid.update(valid_retry)
        if invalid_retry:
            logger.warning(
                "LLM retry still returned %d out-of-taxonomy labels; coercing to 'other': %s",
                len(invalid_retry),
                invalid_retry,
            )
            for dim in invalid_retry:
                valid[dim] = "other"

    # Any dim the LLM omitted altogether -> "other".
    missing = [name for name in dim_names if name not in valid]
    if missing:
        logger.warning(
            "LLM omitted %d dimensions; coercing to 'other': %s",
            len(missing),
            missing,
        )
        for name in missing:
            valid[name] = "other"

    # Drop dims the LLM invented that we didn't ask about.
    requested = set(dim_names)
    return {dim: role for dim, role in valid.items() if dim in requested}


# ---------------------------------------------------------------------------
# Companion JSON I/O
# ---------------------------------------------------------------------------


def companion_json_path(yaml_path: str | Path, analysis_mode: Optional[str]) -> Path:
    """Compute the companion JSON path for a given YAML + analysis mode.

    Convention: ``{yaml_stem}_{analysis_mode}_semantic_roles.json`` alongside
    the YAML. When analysis_mode is None, the suffix is just ``_semantic_roles``.
    """

    yaml_p = Path(yaml_path)
    if analysis_mode:
        stem = f"{yaml_p.stem}_{analysis_mode}_semantic_roles"
    else:
        stem = f"{yaml_p.stem}_semantic_roles"
    return yaml_p.with_name(f"{stem}.json")


def load_companion_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read companion JSON at %s: %s", p, exc)
        return None


def _write_companion_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_dimension_semantic_roles(
    yaml_path: str | Path,
    *,
    scope_to_analysis_mode: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    write_output: bool = True,
    llm: Optional[Any] = None,
    ehap: Optional[Any] = None,
) -> Dict[str, Any]:
    """Classify drill_dimensions into normalized semantic roles via LLM.

    Args:
        yaml_path: Path to a semantic-view YAML.
        scope_to_analysis_mode: Filter to a single analysis mode's
            drill_dimensions. When None, the union across all modes is used.
        output_path: Explicit companion JSON path. Defaults to the alongside
            path produced by :func:`companion_json_path`.
        write_output: When True (default), write the companion JSON to disk.
        llm: LangChain-compatible LLM. Required — this function does not
            resolve a default endpoint.
        ehap: Reserved for future EHAP token integration; unused today.

    Returns:
        The full companion JSON payload (dict), including ``dimension_roles``.
    """

    if llm is None:
        raise ValueError(
            "classify_dimension_semantic_roles requires an `llm` — pass the "
            "LangChain-compatible LLM you configured for the classifier endpoint."
        )
    del ehap  # accepted for future compatibility; unused

    yaml_doc = _load_yaml(yaml_path)
    catalog = _build_dimension_catalog(yaml_doc)
    dim_names = _drill_dimensions_for_mode(yaml_doc, scope_to_analysis_mode)

    if not dim_names:
        logger.warning(
            "No drill_dimensions found for mode=%s in %s",
            scope_to_analysis_mode,
            yaml_path,
        )
        dimension_roles: Dict[str, str] = {}
    else:
        dimension_roles = _classify_via_llm(llm, dim_names, catalog)

    source_hash = _compute_source_hash(dim_names, catalog)

    payload: Dict[str, Any] = {
        "view_name": yaml_doc.get("name"),
        "analysis_mode": scope_to_analysis_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": _resolve_llm_model_id(llm),
        "source_hash": source_hash,
        "dimension_roles": dimension_roles,
    }

    if write_output:
        target = Path(output_path) if output_path else companion_json_path(
            yaml_path, scope_to_analysis_mode
        )
        _write_companion_json(target, payload)
        logger.info("Wrote semantic roles companion JSON to %s", target)

    return payload


def source_hash_for_yaml(
    yaml_path: str | Path,
    analysis_mode: Optional[str],
) -> str:
    """Compute the current source_hash for a YAML + analysis mode.

    Used by the UI loader to compare against a cached companion JSON's
    stored hash without re-running the LLM.
    """

    yaml_doc = _load_yaml(yaml_path)
    catalog = _build_dimension_catalog(yaml_doc)
    dim_names = _drill_dimensions_for_mode(yaml_doc, analysis_mode)
    return _compute_source_hash(dim_names, catalog)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_cli_llm() -> Any:
    """Build the same LLM the UI uses.

    Kept in a lazy helper so importing this module does not require every
    LangChain/EHAP dependency to be installed at import time (matters for
    unit tests that inject their own stub LLM).
    """

    from deep_research_utils.ehap import EHAPBase  # noqa: F401 — validate import
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

    ehap = EHAPBase()
    token = ehap.get_token()
    return ChatOpenAI(api_key=token, temperature=0)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a semantic-role companion JSON for a semantic-view YAML.",
    )
    parser.add_argument("yaml_path", type=Path, help="Path to the semantic-view YAML.")
    parser.add_argument(
        "--analysis-mode",
        default=None,
        help="Restrict to one analysis mode's drill_dimensions. If omitted, "
        "uses the union across all modes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output JSON path. Defaults to alongside the YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the mapping and print to stdout without writing a file.",
    )
    args = parser.parse_args(argv)

    llm = _default_cli_llm()
    payload = classify_dimension_semantic_roles(
        args.yaml_path,
        scope_to_analysis_mode=args.analysis_mode,
        output_path=args.output,
        write_output=not args.dry_run,
        llm=llm,
    )
    if args.dry_run:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
