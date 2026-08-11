"""Build a compact JSON taxonomy of Elevance Reimbursement policy titles for a
handful of representative CPT/HCPCS procedure codes.

Reads:
    /Users/AH45807/Downloads/distinct_policy_titles.csv (1462 titles)
    /Users/AH45807/project/idiscovery-deep-research/docs/op_*.json (for context only)

Writes:
    /Users/AH45807/project/idiscovery-deep-research/scripts/elevance_title_taxonomy.json

The taxonomy maps procedure code -> clinical summary + a set of
title-substring patterns (case-insensitive regex) with the exact number of
titles they match in the CSV. Downstream a post-fetch filter can OR the
patterns together to keep only the ~5-20 most-relevant Elevance policies.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Sequence

CSV_PATH = Path("/Users/AH45807/Downloads/distinct_policy_titles.csv")
OUT_PATH = Path(
    "/Users/AH45807/project/idiscovery-deep-research/scripts/elevance_title_taxonomy.json"
)


def load_titles(csv_path: Path) -> list[str]:
    titles: list[str] = []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header "PLCY_TTL_TXT"
        for row in reader:
            if row and row[0].strip():
                titles.append(row[0].strip())
    return titles


def count(titles: Sequence[str], pattern: str) -> tuple[int, list[str]]:
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [t for t in titles if rx.search(t)]
    return len(hits), hits


# ---------------------------------------------------------------------------
# Per-code pattern definitions. `notes` calls out FP risk / rationale.
# Patterns are ordered highest-signal first, so a downstream filter can take
# the top-K if it wants to be strict.
# ---------------------------------------------------------------------------

CODES: dict[str, dict] = {
    "a0427": {
        "clinical_summary": (
            "Ambulance service, Advanced Life Support, non-emergency transport, "
            "Level 1 (ALS1). HCPCS ground transportation code."
        ),
        "recommended_search_keyword": "ambulance",
        "patterns": [
            ("Ambulance", "canonical Elevance ambulance stem; hits all ambulance policy titles"),
            (r"Transportation Services?\b", "Elevance's canonical umbrella title 'Transportation Services: Ambulance and Non-Emergent Transport'"),
            (r"Non[- ]?Emergent Transport", "second half of the canonical title; catches 'Ambulance and Non-Emergent Transport'"),
            (r"Medical Transport", "catches 'Ambulance and Medical Transport Services'"),
            (r"Ambulance Reimbursement", "explicit reimbursement variant"),
            (r"Air Ambulance", "air-transport specific policy"),
        ],
        "excluded_patterns_notes": [
            "'Transportation and Lodging Related to Transplants' is transport-adjacent but NOT ambulance -- avoid bare 'Transportation'.",
        ],
        "recommended_title_regex": r"(?i)ambulance|transportation service|non[- ]?emergent transport|medical transport",
    },
    "99291": {
        "clinical_summary": (
            "Critical care evaluation and management, first 30-74 minutes. E/M CPT code."
        ),
        "recommended_search_keyword": "critical care",
        "patterns": [
            (r"Critical Care", "direct match; canonical 'Critical Care to Home' stem"),
            (r"Evaluation and Management", "canonical E/M title stem; used for E/M leveling, documentation, modifier 25/57 etc."),
            (r"\bEM Services\b|\bE/M Services\b|\bE and M\b", "abbreviated E/M variants"),
            (r"Emergency Department[: ].{0,40}(Evaluation|Level)", "ED leveling policies (highly cross-referenced with 99291)"),
            (r"After[- ]?Hours.*E(/|M| and )M", "after-hours E/M services"),
            (r"Modifiers? 25 and 57", "E/M-with-global modifier policies applicable to 99291"),
            (r"Modifier 24", "unrelated E/M in postoperative period"),
            (r"Consultations?", "E/M consultation policy"),
            (r"Preventive Medicine.*Same Day|Preventive Meds Sick", "preventive vs sick-visit E/M same-day"),
        ],
        "excluded_patterns_notes": [
            "Do NOT match on bare 'Emergency' -- pulls in ambulance emergency titles.",
        ],
        "recommended_title_regex": r"(?i)critical care|evaluation and management|\bE(M|/M) Services\b|Modifiers? 2[45]|Modifier 57|Consultations?",
    },
    "90837": {
        "clinical_summary": (
            "Psychotherapy, 60 minutes with patient. Outpatient behavioral-health CPT code."
        ),
        "recommended_search_keyword": "psychotherapy",
        "patterns": [
            (r"Psychotherapy", "direct match; 'Documentation Guidelines for Psychotherapy Services'"),
            (r"Behavioral Health", "Elevance uses 'Behavioral Health - ...' as the canonical BH policy stem"),
            (r"Psychiatric", "psychiatric IOP / partial hosp / TMS titles"),
            (r"Mental Health", "occasional variant stem"),
            (r"Substance Use Disorder", "co-located under behavioral health policies"),
            (r"Opioid (Treatment|Use Disorder)", "OTP/OUD-related BH policies"),
            (r"Partial Hospitalization|Intensive Outpatient Program", "PHP/IOP policies (BH level-of-care)"),
        ],
        "excluded_patterns_notes": [
            "'Behavioral Health V Codes' is BH-adjacent (ICD V-code billing), keep it in.",
            "'Transcranial Magnetic Stimulation as a Treatment of Depression' matches Behavioral Health prefix -- correctly kept.",
        ],
        "recommended_title_regex": r"(?i)psychotherapy|behavioral health|psychiatric|mental health|opioid.*(treatment|use disorder)|partial hospitalization|intensive outpatient program",
    },
    "22551": {
        "clinical_summary": (
            "Arthrodesis (fusion) of cervical spine, anterior interbody, C2 and below, "
            "with discectomy. Spine-surgery CPT code."
        ),
        "recommended_search_keyword": "spine",
        "patterns": [
            (r"Spinal|Spine\b", "canonical spine noun; catches 'Spinal Stenosis', 'Non-Spine Management', 'Spinal Manipulation'"),
            (r"Vertebr|Discect", "vertebral / discectomy stems"),
            (r"Interspinous|Interlaminar", "spinal spacer / fusion device policies"),
            (r"Orthopedic", "orthopedic implants / procedures cover spine surgery"),
            (r"Multiple Surgery|Multiple and Bilateral Surgery", "multi-procedure reimbursement rules apply to any surgery incl. spine"),
            (r"Global Surg(ery|ical Package)", "global-period rules for spine surgery"),
            (r"Assistant[- ]?at[- ]?Surgery|Co[- ]?Surgeon|Modifier 62", "assistant/co-surgeon rules commonly reduced on spine claims"),
            (r"Modifiers? 80.*81.*82", "assistant-surgeon modifier policies (spine surgery frequently reports these)"),
            (r"Modifiers? 50 and 51", "bilateral/multiple modifier policies"),
        ],
        "excluded_patterns_notes": [
            "'Non-Spine Management' is a headache injection policy -- it correctly still surfaces because spine reviewers reference it.",
            "'Endoscopic Discectomy' correctly appears under both spine and endoscopy searches -- disambiguate by clinical_summary in caller.",
        ],
        "recommended_title_regex": r"(?i)spine|spinal|vertebr|discect|interspinous|interlaminar|orthopedic|multiple (and bilateral )?surgery|global surg|assistant.at.surgery|co.surgeon|modifier 62",
    },
    "71260": {
        "clinical_summary": (
            "CT thorax with contrast. Diagnostic-imaging chest CT CPT code."
        ),
        "recommended_search_keyword": "diagnostic imaging",
        "patterns": [
            (r"Radiology", "canonical Elevance stem 'Radiology - ...'"),
            (r"Diagnostic Imaging", "canonical multi-procedure reduction policy stem"),
            (r"Multiple (Diagnostic Imaging|Radiology)", "MPPR titles that directly change 71260 reimbursement"),
            (r"Diagnostic Radiopharmaceuticals|Contrast Material", "contrast/agent policies that apply to CT with contrast"),
            (r"3D Radiology|Three[- ]?Dimensional", "3D reconstruction add-on codes commonly billed with CT"),
            (r"Computed Tomography", "explicit CT match; catches 'Whole Body Computed Tomography Scan'"),
            (r"Portable.*(Radiology|Imaging)|Mobile.*Radiology", "portable/mobile imaging"),
            (r"Imaging (for|of|in)|Magnetic Resonance", "MRI/imaging modality policies (relevant for reduction guidance)"),
        ],
        "excluded_patterns_notes": [
            "Bare 'Imaging' pulls in Beta-Amyloid, Monoclonal-Antibody, Myocardial Sympathetic Innervation -- these are correctly imaging policies but only tangentially related to chest CT. The recommended regex leaves them in; a stricter filter can drop them.",
        ],
        "recommended_title_regex": r"(?i)radiology|diagnostic imaging|multiple (diagnostic imaging|radiology)|diagnostic radiopharm|contrast material|3D radiology|three.dimensional|computed tomography|portable.*(radiology|imaging)",
    },
    "g0463": {
        "clinical_summary": (
            "Hospital outpatient clinic visit for assessment and management. "
            "HCPCS code used by OPPS for facility E/M billing."
        ),
        "recommended_search_keyword": "outpatient",
        "patterns": [
            (r"Clinic Charge", "direct match; 'Clinic Charges - Facility' is the canonical G0463 policy"),
            (r"Outpatient Facility Revenue Code", "revenue-code billing rules for hospital outpatient facility claims"),
            (r"HCPCS[- ]?CPT Code Requirements", "revenue-code-to-HCPCS mapping (G0463 sits here)"),
            (r"Revenue Codes? Requiring Procedure Codes", "sibling of the above"),
            (r"Treatment Rooms?", "treatment-room-with-office-E/M policies"),
            (r"Place of Service", "POS billing rules -- outpatient hospital POS 22"),
            (r"Site of Service", "site-of-service payment differentials for facility vs office"),
            (r"Observation (Services|Room|, Facility)", "OP observation billing adjacent to G0463"),
            (r"Outpatient Code Editor|OCE", "OPPS OCE edits directly apply to G0463 lines"),
            (r"Emergency Department[: ].{0,40}Level", "ED leveling -- sister facility E/M policy"),
        ],
        "excluded_patterns_notes": [
            "'Outpatient Drug Screen Testing' and 'Behavioral Health - Intensive Outpatient Program' are outpatient in name only; excluded by not matching on bare 'Outpatient'.",
        ],
        "recommended_title_regex": r"(?i)clinic charge|outpatient facility revenue code|HCPCS.CPT code requirements|revenue codes? requiring procedure|treatment rooms?|place of service|site of service|outpatient code editor|OCE edits|observation (services|room|, facility)",
    },
    "j3490": {
        "clinical_summary": (
            "Unclassified drugs (NOC). Used to bill any injectable drug that lacks a "
            "specific J-code. Requires NDC and clinical documentation."
        ),
        "recommended_search_keyword": "unlisted",
        "patterns": [
            (r"Drugs? and Biologicals?", "canonical NOC-drug policy stem 'Reimbursement - Drugs and Biologicals'"),
            (r"Unlisted (or |and )?Miscellaneous Codes?|Unlisted Misc Codes", "unlisted/miscellaneous HCPCS reimbursement policy"),
            (r"NDC Requirement", "NDC reporting is required for J3490 -- direct policy hit"),
            (r"Drugs? and Injectable Limits?", "drug-and-injectable frequency limits"),
            (r"Unit Freq(uency)? Max for Drugs", "unit-frequency max policy for drugs/biologicals"),
            (r"Injection and Infusion Administration", "administration-code companion policy"),
            (r"Specialty Drugs? in the Outpatient Setting", "specialty-drug policy (many J3490 claims are specialty drugs)"),
            (r"Injectable Substances", "'Injectable Substances with Injection Services'"),
            (r"Not Otherwise Classified|\bNOC\b", "generic NOC anchor (rarely used in Elevance titles but included for safety)"),
        ],
        "excluded_patterns_notes": [
            "Do NOT match bare 'Injection' or 'Drug' -- pulls in 'Injection Therapy for Headache', 'Drug Screen Testing' etc. Use 'Drugs and Biologicals' / 'Unlisted' anchors.",
            "'Radiology - Monoclonal Antibody Imaging' matched an earlier 'NOC'-style regex only by coincidence; excluded here.",
        ],
        "recommended_title_regex": r"(?i)drugs? and biologicals?|unlisted (or |and )?miscellaneous code|unlisted misc code|NDC requirement|drugs? and injectable limit|unit freq(uency)? max for drugs|injection and infusion administration|specialty drugs? in the outpatient|injectable substances|not otherwise classified|\bNOC\b",
    },
    "43239": {
        "clinical_summary": (
            "Esophagogastroduodenoscopy (EGD) with biopsy, single or multiple. "
            "Upper GI endoscopy CPT code."
        ),
        "recommended_search_keyword": "endoscopy",
        "patterns": [
            (r"Endoscop", "canonical endoscopy stem (catches Endoscopy, Endoscopic, Chromoendoscopy, Peroral Endoscopic Myotomy, Capsule Endoscopy, Transanal Endoscopic Microsurgery)"),
            (r"Multiple Endoscopy", "'Reimbursement - Multiple Endoscopy Services' -- the multiple-endoscopy reduction rule directly hits 43239 when billed with sibling codes"),
            (r"Colonoscopy", "sibling endoscopy code family"),
            (r"Esophag|Gastroesophageal|Gastro", "upper-GI anatomy stems (Esophageal pH Monitoring, GERD, etc.)"),
            (r"Moderate .{0,10}Sedation", "moderate/conscious sedation policies -- EGD is routinely billed with sedation"),
            (r"Global Surg(ery|ical Package)", "global-period rules apply to endoscopic surgery"),
            (r"Assistant[- ]?at[- ]?Surgery", "assistant-at-surgery rules for endoscopy"),
            (r"Modifiers? 5[09]", "distinct-procedural-service modifiers commonly appended to EGD lines"),
        ],
        "excluded_patterns_notes": [
            "'Surgery - Automated Percutaneous and Endoscopic Discectomy' correctly matches 'Endoscop' but is a SPINE procedure, not GI. Downstream filters should disambiguate by CPT range or by also requiring an upper-GI anatomy noun.",
            "'Electrogastrography, Cutaneous' matches 'Gastro' but is a rarely-relevant diagnostic test.",
        ],
        "recommended_title_regex": r"(?i)endoscop|multiple endoscopy|colonoscop|esophag|gastroesophageal|moderate.{0,10}sedation|global surg|assistant.at.surgery|modifiers? 5[09]",
    },
}


def build_taxonomy(titles: Sequence[str]) -> dict:
    codes_out: dict[str, dict] = {}
    for code, spec in CODES.items():
        expected: list[dict] = []
        for pattern, notes in spec["patterns"]:
            n, _hits = count(titles, pattern)
            expected.append({"pattern": pattern, "match_count": n, "notes": notes})

        # Also compute the count for the OR-combined recommended regex.
        combined_n, combined_hits = count(titles, spec["recommended_title_regex"])
        codes_out[code] = {
            "clinical_summary": spec["clinical_summary"],
            "expected_title_patterns": expected,
            "excluded_patterns_notes": spec.get("excluded_patterns_notes", []),
            "recommended_search_keyword": spec["recommended_search_keyword"],
            "recommended_title_regex": spec["recommended_title_regex"],
            "recommended_regex_match_count": combined_n,
            "recommended_regex_example_matches": combined_hits[:10],
        }

    return {
        "codes": codes_out,
        "meta": {
            "total_titles_scanned": len(titles),
            "source_csv": str(CSV_PATH),
            "generated_at": _dt.date.today().isoformat(),
            "usage": (
                "For a given procedure code, apply recommended_title_regex to each "
                "policy title returned by the Elevance fetch. Keep only matches. "
                "Expect the filtered set to be 5-20 policies for a single code."
            ),
        },
    }


def main() -> None:
    titles = load_titles(CSV_PATH)
    taxonomy = build_taxonomy(titles)
    OUT_PATH.write_text(json.dumps(taxonomy, indent=2))
    print(f"Wrote {OUT_PATH} covering {len(taxonomy['codes'])} codes over {len(titles)} titles.")

    # Print a short human-readable summary per code so we can eyeball it.
    for code, block in taxonomy["codes"].items():
        n = block["recommended_regex_match_count"]
        print(f"\n[{code}] recommended regex -> {n} matches")
        for m in block["recommended_regex_example_matches"][:5]:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
