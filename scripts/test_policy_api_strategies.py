#!/usr/bin/env python3
"""
Test harness for the Carelon policy comparison API.

Goal: determine the best `keyword=` strategy for finding Reimbursement policies
(and specifically Elevance Reimbursement policies) for CPT/HCPCS procedure codes.

Motivation: searching with just a procedure code like `a0427` returns fee schedules
and provider manuals but drops Elevance Reimbursement policies. Searching with the
clinical description (e.g. `ambulance`) recovers them. This script measures how
many Reimbursement policies and Elevance Reimbursement policies each keyword
strategy returns across ~8 procedure codes and ~6 strategies each.

Usage:
    export SSL_CERT_FILE=/Users/AH45807/project/idiscovery-deep-research/cacert.pem
    python scripts/test_policy_api_strategies.py

Output:
    - scripts/policy_api_test_results.csv    : row per test with counts + latency
    - scripts/policy_api_test_details.json   : full per-test payor breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests


API_URL = "https://policy-comparison-api.carelon.com/policy_comparison/search"

# 8 procedure codes spanning multiple specialties. Each entry pairs the code
# with a narrow (specific clinical noun) and broad (higher-level category)
# description. These are the two keyword variants we combine with the code.
CODES: List[Dict[str, str]] = [
    {
        "code": "a0427",
        "narrow": "ambulance",
        "broad": "transportation",
        "notes": "ALS Level 1 emergency ambulance (HCPCS)",
    },
    {
        "code": "99291",
        "narrow": "critical care",
        "broad": "evaluation and management",
        "notes": "Critical care first 30-74 min (CPT)",
    },
    {
        "code": "90837",
        "narrow": "psychotherapy",
        "broad": "behavioral health",
        "notes": "Psychotherapy 60 min (CPT)",
    },
    {
        "code": "22551",
        "narrow": "spine fusion",
        "broad": "spine surgery",
        "notes": "Anterior cervical fusion (CPT)",
    },
    {
        "code": "71260",
        "narrow": "chest CT",
        "broad": "diagnostic imaging",
        "notes": "CT chest with contrast (CPT)",
    },
    {
        "code": "G0463",
        "narrow": "outpatient clinic visit",
        "broad": "hospital outpatient",
        "notes": "Hospital outpatient clinic visit (HCPCS)",
    },
    {
        "code": "J3490",
        "narrow": "unclassified drug",
        "broad": "injection",
        "notes": "Unclassified drug (HCPCS)",
    },
    {
        "code": "43239",
        "narrow": "upper endoscopy",
        "broad": "endoscopy",
        "notes": "Upper GI endoscopy with biopsy (CPT)",
    },
]


def build_scenarios() -> List[Dict[str, str]]:
    """Cross-join codes with 6 keyword strategies each.

    Strategies:
      code_only         : keyword=<code>                (current behavior)
      narrow_only       : keyword=<narrow>              (description only)
      broad_only        : keyword=<broad>               (broader description)
      code_and_narrow   : keyword=<code>,<narrow>       (proposed default)
      narrow_and_code   : keyword=<narrow>,<code>       (order test)
      code_and_broad    : keyword=<code>,<broad>        (broader combo)
    """
    scenarios: List[Dict[str, str]] = []
    for entry in CODES:
        code = entry["code"]
        narrow = entry["narrow"]
        broad = entry["broad"]
        notes = entry["notes"]
        scenarios.extend([
            {"code": code, "strategy": "code_only", "keyword": code, "notes": notes},
            {"code": code, "strategy": "narrow_only", "keyword": narrow, "notes": notes},
            {"code": code, "strategy": "broad_only", "keyword": broad, "notes": notes},
            {"code": code, "strategy": "code_and_narrow", "keyword": f"{code},{narrow}", "notes": notes},
            {"code": code, "strategy": "narrow_and_code", "keyword": f"{narrow},{code}", "notes": notes},
            {"code": code, "strategy": "code_and_broad", "keyword": f"{code},{broad}", "notes": notes},
        ])
    return scenarios


def call_api(keyword: str, ssl_cert: Optional[str], timeout: int = 60) -> Dict[str, Any]:
    """POST to the policy comparison API with policy_type=Reimbursement filter.

    Returns dict with:
      - ok: bool
      - status: HTTP status
      - latency_s: float
      - sentencelist: list (empty on error)
      - error: str | None
    """
    url = f"{API_URL}?keyword={keyword}&user_type=onshore"
    post_body = {"policy_type": ["Reimbursement"]}
    verify = ssl_cert if ssl_cert else True

    t0 = time.time()
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=post_body,
            verify=verify,
            timeout=timeout,
        )
        latency = time.time() - t0
        if resp.status_code >= 400:
            return {
                "ok": False,
                "status": resp.status_code,
                "latency_s": latency,
                "sentencelist": [],
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        payload = resp.json()
        sentencelist = payload.get("sentencelist") or []
        return {
            "ok": True,
            "status": resp.status_code,
            "latency_s": latency,
            "sentencelist": sentencelist,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": 0,
            "latency_s": time.time() - t0,
            "sentencelist": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def analyze_response(sentencelist: List[Dict[str, Any]], target_code: str) -> Dict[str, Any]:
    """Summarize a sentencelist: counts, payor distribution, code-mention rate.

    - total_hits: raw sentence count
    - unique_policies: dedup by policy_id
    - reimbursement_policies: sentences with policy_type == 'Reimbursement'
    - reimbursement_unique_policies: dedup Reimbursement by policy_id
    - elevance_reimbursement_policies: subset of Reimbursement where payor
      contains 'Elevance'
    - unique_reimbursement_payors: count of distinct payors in Reimbursement
    - top_reimbursement_payors: top 10 payors by policy count
    - code_mentions_in_reimbursement: how many Reimbursement sentences
      literally contain the target_code in context.sentence (case-insensitive)
    """
    total = len(sentencelist)
    unique_ids = {s.get("policy_id") for s in sentencelist if s.get("policy_id")}
    reimb = [s for s in sentencelist if s.get("policy_type") == "Reimbursement"]
    reimb_ids = {s.get("policy_id") for s in reimb if s.get("policy_id")}

    payor_counter: Counter = Counter()
    elv_ids = set()
    code_mentions = 0
    for s in reimb:
        payor = s.get("payor") or "Unknown"
        pid = s.get("policy_id") or ""
        # dedup payor counter by policy_id
        if pid not in payor_counter:
            payor_counter[payor] += 1
        if "Elevance" in payor or "ELV" in payor.upper():
            elv_ids.add(pid)
        # scan context.sentence for the target code
        for ctx in s.get("context") or []:
            sent = (ctx.get("sentence") or "").lower()
            if target_code.lower() in sent:
                code_mentions += 1
                break

    return {
        "total_hits": total,
        "unique_policies": len(unique_ids),
        "reimbursement_hits": len(reimb),
        "reimbursement_unique_policies": len(reimb_ids),
        "elevance_reimbursement_unique_policies": len(elv_ids),
        "unique_reimbursement_payors": len({s.get("payor") for s in reimb}),
        "top_reimbursement_payors": payor_counter.most_common(10),
        "reimbursement_sentences_mentioning_code": code_mentions,
    }


def run_scenario(scenario: Dict[str, str], ssl_cert: Optional[str]) -> Dict[str, Any]:
    """Execute one keyword strategy and analyze the result."""
    call = call_api(scenario["keyword"], ssl_cert=ssl_cert)
    summary = analyze_response(call["sentencelist"], scenario["code"])
    row = {
        "code": scenario["code"],
        "strategy": scenario["strategy"],
        "keyword": scenario["keyword"],
        "notes": scenario["notes"],
        "ok": call["ok"],
        "status": call["status"],
        "latency_s": round(call["latency_s"], 2),
        "error": call["error"] or "",
        **summary,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssl-cert",
        default=os.environ.get("SSL_CERT_FILE", ""),
        help="Path to SSL CA bundle (falls back to SSL_CERT_FILE env)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent API calls (default 4 — the API is single-region)",
    )
    parser.add_argument(
        "--out-csv", default="scripts/policy_api_test_results.csv",
    )
    parser.add_argument(
        "--out-json", default="scripts/policy_api_test_details.json",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only run the first N scenarios (0 = all)",
    )
    args = parser.parse_args()

    ssl_cert = args.ssl_cert or None
    if ssl_cert and not os.path.exists(ssl_cert):
        print(f"[warn] SSL cert file not found: {ssl_cert} — falling back to system CAs", file=sys.stderr)
        ssl_cert = None

    scenarios = build_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"Running {len(scenarios)} scenarios against {API_URL} "
          f"(concurrency={args.concurrency}, ssl_cert={ssl_cert or 'system'})")

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_scenario, sc, ssl_cert): sc for sc in scenarios
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sc = futures[fut]
            row = fut.result()
            rows.append(row)
            print(
                f"[{i}/{len(scenarios)}] code={row['code']:8s} "
                f"strategy={row['strategy']:20s} kw={row['keyword']!r:40s} "
                f"→ reimb={row['reimbursement_unique_policies']:4d} "
                f"elv={row['elevance_reimbursement_unique_policies']:3d} "
                f"payors={row['unique_reimbursement_payors']:3d} "
                f"({row['latency_s']}s)"
                + (f" ERR={row['error']}" if not row['ok'] else "")
            )

    # Sort rows by (code, strategy) for a stable output
    rows.sort(key=lambda r: (r["code"], r["strategy"]))

    # Write CSV (numeric fields only)
    csv_cols = [
        "code", "strategy", "keyword", "notes", "ok", "status", "latency_s",
        "total_hits", "unique_policies",
        "reimbursement_hits", "reimbursement_unique_policies",
        "elevance_reimbursement_unique_policies",
        "unique_reimbursement_payors",
        "reimbursement_sentences_mentioning_code",
        "error",
    ]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nCSV written to {args.out_csv}")

    # Write full JSON with top_reimbursement_payors preserved
    with open(args.out_json, "w") as fp:
        json.dump(rows, fp, indent=2, default=str)
    print(f"JSON written to {args.out_json}")

    # Print per-code summary table
    print("\n" + "=" * 100)
    print(f"{'code':10s} {'strategy':22s} {'reimb':>6s} {'elv_reimb':>10s} {'payors':>7s}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['code']:10s} {row['strategy']:22s} "
            f"{row['reimbursement_unique_policies']:6d} "
            f"{row['elevance_reimbursement_unique_policies']:10d} "
            f"{row['unique_reimbursement_payors']:7d}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
