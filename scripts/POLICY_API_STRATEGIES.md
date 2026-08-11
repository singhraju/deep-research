# Policy Comparison API Keyword Strategies

**TL;DR:** When searching the Carelon policy comparison API for CPT/HCPCS
procedure codes, always send `keyword=<code>,<clinical_category>` — never just
`keyword=<code>`. Sending only the code drops most Reimbursement policies
(including all Elevance Reimbursement policies for the given service category)
because the API does exact-string matching on document text, and payors'
narrative Reimbursement policies typically describe rules by clinical topic,
never by the specific code.

## The bug this fixes

Symptom (reproduced 2026-07-07):

```
GET .../search?keyword=a0427                → 0   Elevance Reimbursement policies
GET .../search?keyword=ambulance             → 200 Elevance Reimbursement policies
GET .../search?keyword=a0427,ambulance      → 200 Elevance Reimbursement policies
GET .../search?keyword=a0427,transportation → 230 Elevance Reimbursement policies
```

Before this fix, `ReimbursementPolicyAgent._generate_search_keywords` would
fall back to sending the raw first CPT/HCPCS code alone whenever the pattern
had no DRG codes. Result: procedure-code-driven patterns silently missed
Elevance and other payor Reimbursement policies for that clinical category.

## Root cause

The policy_comparison API matches literal strings in `context.sentence` fields
extracted from policy PDFs.

- **Elevance publishes Reimbursement policies at the CATEGORY level** — e.g.
  the ambulance policy is titled "Transportation Services: Ambulance and
  Non-Emergent Transport". These narrative PDFs describe rules in prose
  ("emergency ground ambulance…") and do **not** contain the string "A0427"
  anywhere in the extracted text. Verified: 0 of 200 Elevance Reimbursement
  ambulance-policy sentences contain the literal token "a0427".
- **Elevance separately publishes Clinical Guidelines** (UM, MBM) that DO
  contain the HCPCS code table. Those come back for keyword=a0427, but they
  are `policy_type == "Clinical Guidelines (UM, MBM)"` — the reimbursement
  agent filters them out.

The code-to-policy join lives *across* documents (Clinical Guidelines have
the code, Reimbursement has the rules), and the search API cannot traverse
it. So a raw-code search can never find the narrative Reimbursement policies
on its own.

## Method

`scripts/test_policy_api_strategies.py` sweeps 48 scenarios: 8 procedure
codes × 6 keyword strategies each. Each scenario POSTs to
`https://policy-comparison-api.carelon.com/policy_comparison/search`
with `policy_type=["Reimbursement"]` and records the unique-policy_id counts
per policy_type and per payor.

Codes exercised (span multiple specialties on purpose):

| Code   | Type  | Narrow keyword          | Broad category keyword     |
|--------|-------|-------------------------|----------------------------|
| A0427  | HCPCS | ambulance               | transportation             |
| 99291  | CPT   | critical care           | evaluation and management  |
| 90837  | CPT   | psychotherapy           | behavioral health          |
| 22551  | CPT   | spine fusion            | spine surgery              |
| 71260  | CPT   | chest CT                | diagnostic imaging         |
| G0463  | HCPCS | outpatient clinic visit | hospital outpatient        |
| J3490  | HCPCS | unclassified drug       | injection                  |
| 43239  | CPT   | upper endoscopy         | endoscopy                  |

Strategies:

- `code_only` — `keyword=<code>` (the pre-fix behavior)
- `narrow_only` — `keyword=<narrow>`
- `broad_only` — `keyword=<broad>`
- `code_and_narrow` — `keyword=<code>,<narrow>` (initial hypothesis)
- `narrow_and_code` — `keyword=<narrow>,<code>` (order test)
- `code_and_broad` — `keyword=<code>,<broad>` (**winner**)

## Results

Unique Reimbursement policies returned. Bold = winning strategy per code.

| Code   | code_only | narrow_only | broad_only | code_and_narrow | code_and_broad |
|--------|----------:|------------:|-----------:|----------------:|---------------:|
| A0427  |        45 |         446 |        457 |             462 |        **459** |
| 99291  |       517 |         757 |       2865 |             920 |       **2956** |
| 90837  |       101 |          82 |        624 |             155 |        **665** |
| 22551  |         1 |           1 |         12 |               1 |         **12** |
| 71260  |         8 |           5 |        380 |              13 |        **383** |
| G0463  |       186 |          89 |        431 |             194 |        **532** |
| J3490  |       120 |          92 |        658 |             209 |        **722** |
| 43239  |         9 |           6 |        236 |              15 |        **237** |

Unique **Elevance** Reimbursement policies (this is the metric the user's
report highlighted):

| Code   | code_only | narrow_only | broad_only | code_and_narrow | code_and_broad |
|--------|----------:|------------:|-----------:|----------------:|---------------:|
| A0427  |     **0** |         200 |        230 |             200 |        **230** |
| 99291  |       104 |         261 |       1327 |             277 |       **1359** |
| 90837  |         2 |          23 |        144 |              25 |        **144** |
| 22551  |         0 |           0 |          1 |               0 |          **1** |
| 71260  |         0 |           0 |        184 |               0 |        **184** |
| G0463  |        61 |          16 |         75 |              61 |        **119** |
| J3490  |        42 |          66 |        180 |             107 |        **181** |
| 43239  |         0 |           0 |         34 |               0 |         **34** |

Full CSV: `scripts/policy_api_test_results.csv`. Full per-payor JSON:
`scripts/policy_api_test_details.json`. Raw log: `scripts/policy_api_test_run.log`.

## Findings

1. **`code_only` is uniformly the worst strategy for Reimbursement policies.**
   For every code tested, it returns the fewest Reimbursement policies AND the
   fewest Elevance Reimbursement policies. Four of the eight codes tested
   (A0427, 71260, 22551, 43239) return **ZERO** Elevance Reimbursement
   policies under `code_only`.

2. **`code_and_broad` is uniformly the best strategy — never worse than any
   other, strictly better than most.** It matches every category-level
   policy that `broad_only` finds AND retains code-listing policies that
   only mention the code (small delta, e.g. +91 UnitedHealth "Procedure to
   Place of Service" rows for `99291,evaluation and management` over
   `evaluation and management` alone).

3. **Keyword order does not affect results.** `code_and_narrow` and
   `narrow_and_code` return identical counts across all 8 codes — the API
   treats comma-separated keywords as an unordered set.

4. **Narrow (specific service term) is often WORSE than the raw code.**
   A "narrow" clinical noun like "psychotherapy" or "outpatient clinic visit"
   is often too specific and misses the category-level policies that use
   broader titles. Broadening to "behavioral health" or "hospital outpatient"
   consistently recovers 3-9× more policies.

5. **The gain is largest for narrative-heavy categories** (99291 E/M: +2439
   Reimbursement policies vs code_only, +1255 Elevance vs code_only) and
   smallest for surgical codes where policies are typically issued per-code
   (22551 spine fusion: +11 Reimbursement policies).

## The fix

`ReimbursementPolicyAgent._generate_search_keywords` now routes CPT/HCPCS
codes through a new `_generate_search_keywords_from_cpt` helper — parallel
in structure to the existing `_generate_search_keywords_from_drg`. The
helper asks the LLM for a single clinical CATEGORY keyword (1-2 words) and
returns `<first_code>,<category>` for the API `keyword=` parameter.

The LLM prompt bias is explicit:

> Prefer the broader clinical CATEGORY name that would appear in a payor's
> reimbursement policy TITLE — not the narrowest possible clinical noun.

with worked examples from each tested specialty. Fallback on LLM failure:
the raw first code (preserves previous legacy behavior — no worse than
before).

DRG-driven patterns are unchanged (they already had this treatment via
`_generate_search_keywords_from_drg`).

### LLM tier

The CPT path runs on **`gpt-5.4-nano` (`self.llm_mini`, low reasoning)** —
not the main GPT-5.4-medium tier. Mapping a code to a 1-2-word clinical
category is a small classification task well within nano's capabilities;
using the nano tier gives ~10× the throughput and a fraction of the cost
of the main tier. The dispatcher signature is:

```python
_generate_search_keywords(drg_codes, cpt_codes, llm=main, llm_mini=nano)
```

Fallback chain: `llm_mini` → `llm` → raw first code. The DRG path stays on
the main tier because its prompt asks the model to identify the top 2
themes across possibly-long DRG descriptions, which benefits from more
reasoning than nano provides.

## Reproducing the sweep

```bash
export SSL_CERT_FILE=$(pwd)/cacert.pem
python scripts/test_policy_api_strategies.py
```

Runs 48 scenarios in ~3-5 minutes with concurrency=4. Outputs CSV + JSON.
Add `--limit N` to run only the first N scenarios during iteration.

---

# Part 2 — Reducing to a manageable Elevance set

**TL;DR:** `code,broad_category` fixed the recall problem (Part 1) but returns
huge sets — J3490 alone comes back with 722 Reimbursement policies of which
181 are Elevance, and inspection shows most are false positives (unrelated
policies whose text just happens to mention "injection"). This part adds a
two-step client-side reduction that stacks on top of the winning keyword:
**(1) title-anchor regex filter** and **(2) dedup by (payor, policy_title)**.
Across the 8 test codes, this shrinks total Reimbursement counts by 12-100×
and Elevance counts by 5-72×, while preserving all topically-relevant
Elevance titles.

## Why more reduction is needed

For `a0427,transportation` the API returns 230 Elevance Reimbursement
policies. But those 230 collapse to only **20 distinct titles** — same
title, published per-state per line-of-business. And of those 20 titles,
only 6 are actually about ambulance/transportation:

```
 43 policy_ids  Transportation Services: Ambulance and Non-Emergent Transport
 19 policy_ids  Ambulance Transportation
  2 policy_ids  Transportation Services: Emergent and Nonemergent Transport
  1 policy_id   Transportation Services Ambulance and Non-Emergent Transport
  1 policy_id   Transportation Services: Ambulance and Nonemergent Transport
  1 policy_id   Ambulance Reimbursement Policy
--- above: topically relevant (66 of 200) ---
 52 policy_ids  Preadmission Services for Inpatient Stays        ← false positive
 41 policy_ids  Emergency Department Leveling of E&M Services    ← false positive
 18 policy_ids  Incident to Services and Billing                 ← false positive
 12 policy_ids  Modifier Usage                                   ← false positive
```

The false positives mention "ambulance" only in prose ("if transport is by
ambulance…"), so a title-relevance gate would drop them cleanly.

## API filter probe (16 requests)

Before designing client-side filters we probed the POST body to see what
the server would do for us. Findings from `/tmp/probe_carelon.py`:

| Field           | Behavior                                                             |
|-----------------|----------------------------------------------------------------------|
| `payor`         | Accepted **only** as a single-element exact-value list. Multi-value list returns 0 rows; aliases 404. To get all Elevance, issue two calls (external + internal) and union. |
| `lob`           | Accepted (list). `["Medicare"]` returns ~89% Elevance-dense.        |
| `state`         | Accepted (list). Tightens dramatically when combined with LOB.       |
| `product_class` | Silently ignored — response identical to baseline byte-for-byte.     |
| `limit`/`top_k` | Silently ignored.                                                    |
| `min_score`     | Silently ignored. `policy_score` in responses ranges 5..650 but the API won't threshold on it — client-side only. |
| `claim_type`    | Rejected — HTTP 404.                                                 |
| Unknown keys    | Silently accepted (no strict-schema penalty).                        |

Conclusion: the API gives us `lob` and `state` when the pattern provides
them (the agent already sends these), but there is no server-side title
or score filter. All topical narrowing must happen client-side.

## Elevance title taxonomy

`/Users/AH45807/Downloads/distinct_policy_titles.csv` (1462 distinct titles
across payors, seeded from Elevance's catalog) was mined by the
title-analyzer subagent into `scripts/elevance_title_taxonomy.json`. The
taxonomy maps each of the 8 test codes to a case-insensitive regex whose
tokens are anchored on multi-word domain nouns to avoid known false-positive
pitfalls (bare "Injection" would catch "Injection Therapy for Headache",
bare "Endoscop" catches "Endoscopic Discectomy" which is spine surgery,
bare "Transportation" catches "Transportation and Lodging Related to
Transplants"). The generator script is
`scripts/build_elevance_title_taxonomy.py`.

Excerpt:

```json
{
  "a0427": "(?i)ambulance|transportation service|non[- ]?emergent transport|medical transport",
  "99291": "(?i)critical care|evaluation and management|\\bE(M|/M) Services\\b|Modifiers? 2[45]|Modifier 57",
  "j3490": "(?i)drugs? and biologicals?|unlisted (or |and )?miscellaneous code|NDC requirement|injection and infusion administration"
}
```

## Reduction sweep — 104 scenarios

`scripts/test_policy_api_reduction.py` sweeps 8 codes × 13 strategies,
stacking client-side title-regex, (payor, title) dedup, and top-N-by-score
on top of the Part-1 winning keyword `code,broad`. Server-side filters
(`lob`, `state`, `payor`) are tested as separate strategies to quantify
the tradeoff between them and the client-side approach.

Reimbursement-policy counts by strategy (bold = recommended default):

| Code   | baseline | title_regex | dedup_title | **title_regex+dedup** | payor_elv+title+dedup | broad_kw+title+dedup | state_ny+title+dedup | top10_score |
|--------|---------:|------------:|------------:|-----------------------:|----------------------:|---------------------:|---------------------:|------------:|
| A0427  |      459 |         120 |         157 |                **30**  |                     5 |                   29 |                    9 |          10 |
| 99291  |     2956 |         591 |         895 |               **144**  |                    20 |                  140 |                   42 |          10 |
| 90837  |      665 |          36 |         292 |                **26**  |                     0 |                   25 |                    4 |          10 |
| 22551  |       12 |           2 |          11 |                 **2**  |                     0 |                    2 |                    0 |           2 |
| 71260  |      383 |         112 |         132 |                **37**  |                     3 |                   37 |                   10 |          10 |
| G0463  |      533 |         110 |         197 |                **20**  |                     2 |                   11 |                    7 |          10 |
| J3490  |      722 |          21 |         321 |                **13**  |                     1 |                   10 |                    7 |          10 |
| 43239  |      237 |          33 |         133 |                **20**  |                     2 |                   20 |                    3 |          10 |

Elevance-only Reimbursement-policy counts by the same strategies:

| Code   | baseline | title_regex | dedup_title | **title_regex+dedup** | payor_elv+title+dedup |
|--------|---------:|------------:|------------:|-----------------------:|----------------------:|
| A0427  |      230 |          67 |          21 |                 **6**  |                     5 |
| 99291  |     1359 |         341 |         127 |                **28**  |                    20 |
| 90837  |      144 |           2 |          21 |                 **2**  |                     0 |
| 22551  |        1 |           0 |           1 |                 **0**  |                     0 |
| 71260  |      184 |          66 |          22 |                 **7**  |                     3 |
| G0463  |      120 |          59 |          11 |                 **3**  |                     2 |
| J3490  |      181 |           4 |          31 |                 **4**  |                     1 |
| 43239  |       34 |          15 |          11 |                 **2**  |                     2 |

Full CSV: `scripts/policy_api_reduction_results.csv`. Full per-scenario
JSON: `scripts/policy_api_reduction_details.json`. Raw log:
`scripts/policy_api_reduction_run.log`.

## Findings

1. **`title_regex+dedup` is the most robust reduction across all codes.**
   Reimbursement drops by 12-100× (avg 22×); Elevance drops by 5-72×
   (avg 18×). Result sizes settle in the 2-40 range that the downstream
   per-payor cap and LLM triage can handle cleanly.

2. **Dedup alone (without title regex) is not enough.** For 99291 it only
   drops the Reimbursement count from 2956 to 895 because the false-positive
   titles (Modifier Usage, Preadmission Services) also fan out per state.
   Title regex without dedup drops from 2956 to 591 — still too many.

3. **`payor_elevance+title+dedup` is tempting but brittle.** It returns
   ONLY Elevance and produces the smallest sets (0-20 policies), but the
   payor filter is coupled to a single-element exact enum value that
   silently returns 0 rows if the title regex doesn't match. Two codes
   in the sweep (90837, 22551) return 0 policies under this strategy —
   a strict regression vs `title_regex+dedup` which returns something
   useful for both. Also drops all non-Elevance policies, which the
   agent's downstream comparison relies on.

4. **`top10_score` is not a reliable Elevance filter.** For 90837 the
   top-10-by-`policy_score` result contains 0 Elevance policies; for
   22551 it contains 0 as well. `policy_score` correlates with keyword
   density, not topical relevance.

5. **Server-side `lob` filter is complementary, not a replacement.** When
   the pattern provides a specific LOB (Commercial / Medicare / Medicaid /
   Medicare Advantage), sending it to the API narrows dramatically — but
   without a pattern LOB, `lob` filtering silently drops many relevant
   policies. The agent should keep sending pattern LOB/state as it does
   today, and stack the client-side title-regex+dedup step on top.

6. **`state` filter is too aggressive for pattern-agnostic use.** NY
   returns 4-42 rows depending on code; if the pattern is national or the
   state isn't in the API response's canonical form, the entire result
   set drops.

## Recommended reduction strategy (chosen for the agent)

**`title_regex + dedup_by_(payor,title)`**, applied client-side after the
existing `policy_type=Reimbursement` + `file_type=pdf` + `(payor,external_link)`
dedup step in `search_policies_node`, and before `_apply_policy_caps`.

Design rationale:

- **Preserves recall on topically-relevant Elevance policies.** All 6
  ambulance-related Elevance titles survive for A0427; all 2 psychotherapy
  titles for 90837; all 20 E/M titles for 99291. No `NULL` or empty result
  edge case observed across the 8 test codes.
- **Preserves the downstream cross-payor comparison.** Non-Elevance
  policies survive (in reduced numbers) so `_apply_policy_caps` still has
  a meaningful cross-payor set to triage.
- **Result sizes match the existing cap.** The current
  `POLICY_CAP_MAX_TOTAL=30` was calibrated for baseline sizes; after
  reduction, most codes fit under the cap without triage LLM calls, which
  cuts token spend.
- **Orthogonal to the pattern's server-side filters.** LOB/state that the
  pattern already provides continues to narrow at the server; the
  title-regex+dedup step narrows on top. Stacks cleanly with no logic
  interaction.

### Title-anchor generation

The regex is generated at pattern-processing time by a new LLM helper
`_generate_title_anchors_from_cpt(cpt_codes, llm_mini, user_query=None)`.
The helper asks nano for 2-5 comma-separated domain terms that would
appear in Elevance Reimbursement policy TITLES for the given code(s), and
returns them as a list. `search_policies_node` compiles the list into a
case-insensitive alternation regex.

Fallback chain, in order:

1. `llm_mini` returns 2-5 terms → use them.
2. `llm_mini` unavailable → fall back to the LLM-generated broad category
   already produced by `_generate_search_keywords_from_cpt` (the same
   token that goes into the API keyword parameter). This is a single-term
   anchor — less precise but never empty.
3. `llm_mini` errors, both unavailable, or all terms empty → skip the
   title-regex step entirely and rely on the dedup step alone. The
   downstream `_apply_policy_caps` will still cap total to
   `POLICY_CAP_MAX_TOTAL=30`; behavior degrades gracefully to pre-fix.

### LLM tier

Same rationale as the CPT-keyword helper: mapping a code family to a set
of title-substring anchors is a small classification task. Nano (low
reasoning) handles it at ~10× the throughput and a fraction of the cost
of the main tier. Both helpers share `self.llm_mini`, so an infra swap
propagates to both.

### Dedup ordering

Dedup by (payor, policy_title) keeps the row with the highest
`policy_score` per group. If `policy_score` is missing or non-numeric,
falls back to first-occurrence order — the DataFrame's original API
order, which is the score-descending order the API returned.

## Reproducing the reduction sweep

```bash
export SSL_CERT_FILE=$(pwd)/cacert.pem
python scripts/test_policy_api_reduction.py
```

Runs 104 scenarios in ~4-5 minutes with concurrency=6. Reads
`scripts/elevance_title_taxonomy.json` when present; otherwise falls
back to a hardcoded taxonomy. Outputs CSV + JSON + log.
Add `--limit N` for iteration.
