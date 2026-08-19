# Deep Research Architecture Overview

## Goal

This describes a production-oriented agent framework for a healthcare analytics investigation system that can answer trend, root-cause, control-validation, and reporting questions.

The design centers on five ideas:
- a thin UI that renders status events and artifacts,
- a semantically rich but execution-light Intent & Context Resolver,
- an Orchestrator that compiles and supervises a graph,
- generic capability agents that consume typed inputs and emit typed outputs,
- evidence-gated synthesis so executive answers stay grounded,
- recommend actions with human validation.


## Pack contents

- `01_architecture.md` — production architecture and system shape
- `08_intent_context_resolver.md` — resolver taxonomy, output contract, and examples
- `10_sample_design_row_crosswalk.md` — row-by-row mapping from workbook questions to playbooks and domains

```
- `02_agent_catalog.md` — control-plane modules, capability agents, and responsibilities
- `03_contracts_artifacts_state.md` — request/result contracts, graph objects, artifacts, and run state
- `04_execution_patterns.md` — reliability, gating, privacy, and testing patterns
- `05_workflows_and_ux.md` — how user-facing workflows map to backend execution
- `06_delivery_roadmap.md` — implementation phases, repo shape, and release posture
- `07_sample_design_table_coverage.md` — coverage assessment for the sample design table workbook

- `09_investigation_playbook_catalog.md` — reusable investigation playbooks by business family

- `11_dag_orchestration_architecture.md` — graph compilation, supervision, and patching model
- `12_generic_node_and_agent_contracts.md` — location-independent node and agent contracts
- `13_dag_executor_and_replanning.md` — executor responsibilities, retries, cancellation, and patch handling
- `14_example_trend_and_driver_dag.md` — worked example for “recent trend in California and explain the drivers”
```

## Reading paths

### Product and design path
Read these first:
1. `00_README.md`
2. `01_target_architecture.md`
3. `05_workflows_and_ux.md`
4. `07_sample_design_table_coverage.md`
5. `08_intent_context_resolver.md`
6. `11_dag_orchestration_architecture.md`
7. `14_example_trend_and_driver_dag.md`

### Engineering path
Read these first:
1. `02_agent_catalog.md`
2. `03_contracts_artifacts_state.md`
3. `04_execution_patterns.md`
4. `08_intent_context_resolver.md`
5. `11_dag_orchestration_architecture.md`
6. `12_generic_node_and_agent_contracts.md`
7. `13_dag_executor_and_replanning.md`
8. `06_delivery_roadmap.md`