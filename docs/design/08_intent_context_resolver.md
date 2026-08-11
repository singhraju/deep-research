# Intent & Context Resolver

## Purpose

The Resolver should understand the ask without taking on planning responsibilities.
It should provide enough structured meaning for the Orchestrator to choose a playbook seed, request semantic bindings, and compile the execution graph.

## The Resolver should answer six questions

1. What is the user trying to understand?
2. Which entities and filters are explicit or implicit?
3. Which deliverables does the user expect?
4. Which investigation agent best seeds the plan?
5. Which hypothesis family or business lens is the user asking about?
6. Which constraints, missing context, or follow-up state already exist?

## Recommended resolver output

```json
{
  "goal_category": "EXPLAIN_VARIANCE",
  "deliverables": [
    "answer",
    "evidence_bundle"
  ],
  "entities": {
    "market": "California",
    "metric": "allowed_pmpm",
    "time_scope": "recent"
  },
  "investigation_agent": "PROVIDER",
  "playbook_seed": "focused_variance_explainability",
  "hypothesis_families": [
    "PROVIDER_OUTLIER_COST_PER_UNIT"
  ],
  "routing_hints": [
    "needs_focus_window_selection",
    "needs_semantic_binding",
    "prefer_executive_narrative"
  ],
  "constraints": {
    "answer_style": "executive",
    "time_budget_sec": 60
  },
  "missing_context": [],
  "confidence": 0.95
}
```

## What the Resolver should not do

It should not:
- output a full step list,
- define branch dependencies,
- encode DAG topology,
- decide retries or timeouts,
- cancel branches,
- mutate a running plan.

Those belong to the Orchestrator and Executor.

## Top-level goal categories

| Goal category | Use when | Typical deliverables |
|---|---|---|
| `UNDERSTAND_CHANGE` | user wants trend, anomaly, increase/decrease, or drivers | `trend_summary`, `driver_explanation`, `chart` |
| `EXPLAIN_VARIANCE` | user asks whether a specific factor explains a variance | `answer`, `evidence_bundle`, `caveats` |
| `VALIDATE_CONTROL` | user asks whether a control or policy logic is working | `control_findings`, `exceptions`, `evidence_bundle` |
| `COMPARE_SEGMENTS` | user wants market/provider/member/service comparison | `comparison_summary`, `ranked_deltas`, `chart` |
| `FORECAST_OR_SCENARIO` | user asks what may happen next or under a scenario | `forecast`, `scenario_table`, `assumptions` |
| `BENCHMARK` | user wants peer, historical, or target comparison | `benchmark_summary`, `gaps`, `recommendations` |
| `EXPORT_PRESENTATION` | user wants packaging of already-known findings | `report`, `slides`, `pdf` |

## Investigation Agents

| Investigation Agent | Use when |
|---|---|
| `CROSS_DOMAIN` | user asks for generic drivers or broad explanation |
| `PROVIDER` | provider mix, provider concentration, provider outlier behavior |
| `MEMBER` | case mix, risk shift, cohort, age, program, eligibility |
| `DIAGNOSIS` | ICD trend, diagnosis prevalence, diagnosis-driven cost |
| `SERVICE` | HCPCS/CPT/service line/utilization changes |
| `REIMBURSEMENT` | fee schedule, payment methodology, rate, paid per unit |
| `AUTH_CLAIM` | auth-to-claim, approvals, denials, mismatch, leakage |
| `CLAIM_LIFECYCLE` | paid/denied/reversed/pended/resubmitted processing questions |
| `ADMISSION_DISCHARGE` | admit/discharge/readmit, LOS, care setting transitions |
| `UM_PAV` | utilization management and prior-auth variation |
| `READMISSION` | readmission-specific workflows |
| `COMPETITOR` | external program or market response comparison |

## Hypothesis family taxonomy

These are semantic categories that help the Orchestrator choose the right diagnostic tests.

| Hypothesis family | Typical user wording | Usually maps to |
|---|---|---|
| `CONTRACT_CHANGE_OR_CONCESSION` | contract changes, concessions, rate exceptions, limited ELV management | contract and pricing tests |
| `FACILITY_COST_SHIFT` | higher-cost facilities, site-of-care shift | provider and facility tests |
| `UNIT_COST_SHIFT` | cost per unit, UPTPM change, price increase | reimbursement tests |
| `OON_DISTRIBUTION_SHIFT` | out-of-network mix, leakage, OON share | provider and network tests |
| `CAH_CONTROL_LIMITED_SHIFT` | critical access hospitals, limited controls | facility-control tests |
| `CAPITATION_LEAKAGE` | should be covered by capitation, delegated risk | contract and risk tests |
| `PROVIDER_OUTLIER_COST_PER_UNIT` | subset of providers, outlier cost per unit | provider-outlier tests |
| `SERVICE_MIX_SHIFT` | procedure mix, level of care, DRG change | service tests |
| `MEMBER_CASE_MIX_SHIFT` | risk mix, cohort shift, diagnosis burden | member tests |

## Example outputs

### Example A
User question:
> Show me the recent trend in California.

Resolver output:
```json
{
  "goal_category": "UNDERSTAND_CHANGE",
  "deliverables": ["trend_summary", "chart"],
  "playbook_seed": "trend_only"
}
```

### Example B
User question:
> What was the recent trend in California and explain the drivers?

Resolver output:
```json
{
  "goal_category": "UNDERSTAND_CHANGE",
  "deliverables": ["trend_summary", "driver_explanation", "chart"],
  "investigation_agent": "CROSS_DOMAIN",
  "playbook_seed": "trend_driver",
  "routing_hints": [
    "needs_outlier_detection",
    "needs_focus_window_selection",
    "needs_multi_branch_correlation",
    "needs_semantic_binding"
  ]
}
```

### Example C
User question:
> Is variance explainable by increased utilization at higher cost facilities?

Resolver output:
```json
{
  "goal_category": "EXPLAIN_VARIANCE",
  "deliverables": ["answer", "evidence_bundle"],
  "investigation_agent": "PROVIDER",
  "playbook_seed": "focused_variance_explainability",
  "hypothesis_families": ["FACILITY_COST_SHIFT"],
  "routing_hints": [
    "needs_focus_window_selection",
    "needs_semantic_binding",
    "prefer_executive_narrative"
  ]
}
```

### Example D
User question:
> Are we paying for claims that should be covered by a capitated arrangement?

Resolver output:
```json
{
  "goal_category": "VALIDATE_CONTROL",
  "deliverables": ["control_findings", "exceptions", "evidence_bundle"],
  "investigation_agent": "PROVIDER",
  "playbook_seed": "focused_variance_explainability",
  "hypothesis_families": ["CAPITATION_LEAKAGE"],
  "routing_hints": [
    "needs_semantic_binding",
    "needs_contract_and_risk_domains",
    "prefer_analyst_detail"
  ]
}
```

## Routing hints

Routing hints help the Orchestrator choose or patch a template, but they are not steps.

Recommended hint vocabulary:
- `needs_outlier_detection`
- `needs_focus_window_selection`
- `needs_multi_branch_correlation`
- `needs_semantic_binding`
- `needs_contract_and_risk_domains`
- `needs_provider_breakout`
- `needs_member_segmentation`
- `needs_reimbursement_review`
- `needs_auth_claim_crosswalk`
- `needs_benchmark_context`
- `prefer_executive_narrative`
- `prefer_analyst_detail`
- `exportable`

## Minimal production contract

```json
{
  "goal_category": "...",
  "deliverables": ["..."],
  "entities": {"...": "..."},
  "investigation_agent": "...",
  "playbook_seed": "...",
  "hypothesis_families": ["..."],
  "routing_hints": ["..."],
  "missing_context": ["..."],
  "confidence": 0.0
}
```

## Recommendation

Keep the Resolver semantically rich but execution-light:
- the Resolver understands the ask,
- the semantic core resolves business meaning,
- the Orchestrator builds the run,
- the Executor runs the graph,
- the agents and tests remain generic and reusable.
