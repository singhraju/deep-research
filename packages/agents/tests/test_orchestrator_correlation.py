from __future__ import annotations

from deep_research_agents.orchestrator import _build_correlation_artifacts, _build_correlation_findings


def test_orchestrator_surfaces_interaction_findings_and_artifacts() -> None:
    result = {
        "root_metric": "expense_detail.total_paid",
        "baseline_value": 100.0,
        "comparison_value": 140.0,
        "delta_value": 40.0,
        "delta_pct": 0.4,
        "drill_path": [{"dimension": "hcc_medium", "top_segments": [{"value": "IP OB"}]}],
        "narrative_summary": "Narrative summary.",
        "interaction_summary": {
            "text": "Operational interactions concentrated in one cell.",
            "source": "llm",
        },
        "recommendations": {
            "items": [
                {
                    "text": '{"priority":1,"action":"Review the top operational cell.","rationale":"Legacy rationale.","cell_ids":["op_001"],"review_area":"unit_cost"}',
                    "cell_ids": ["op_001"],
                }
            ],
            "source": "llm",
        },
        "warnings": ["sample warning"],
        "run_dir": "/tmp/run_dir",
        "manifest_path": "/tmp/run_dir/manifest.json",
        "executive_summary_path": "/tmp/run_dir/summary/executive_summary.json",
        "interaction_matrix": {
            "operational": {
                "artifact_paths": {
                    "sql": "queries/interaction_matrix/operational.sql",
                    "delta": "aggregates/interaction_matrix/operational_delta.parquet",
                    "full_matrix": "aggregates/interaction_matrix/operational_full_matrix.parquet",
                }
            },
            "clinical": {
                "artifact_paths": {
                    "sql": "queries/interaction_matrix/clinical.sql",
                    "delta": "aggregates/interaction_matrix/clinical_delta.parquet",
                    "full_matrix": "aggregates/interaction_matrix/clinical_full_matrix.parquet",
                }
            },
        },
    }

    findings = _build_correlation_findings(result)
    artifacts = _build_correlation_artifacts(result)

    finding_ids = {item["id"] for item in findings}
    artifact_types = {item["type"] for item in artifacts}

    assert "interaction_summary" in finding_ids
    assert "interaction_recommendations" in finding_ids
    assert "warnings" in finding_ids
    recommendation_finding = next(item for item in findings if item["id"] == "interaction_recommendations")
    assert recommendation_finding["items"][0]["description"] == "Review the top operational cell."
    assert recommendation_finding["items"][0]["evidence"] == ["Interaction cell reference: op_001"]
    assert "interaction_operational_sql" in artifact_types
    assert "interaction_operational_delta" in artifact_types
    assert "interaction_operational_full_matrix" in artifact_types
    assert "interaction_clinical_sql" in artifact_types
    assert "interaction_clinical_delta" in artifact_types
    assert "interaction_clinical_full_matrix" in artifact_types


def test_orchestrator_ignores_empty_summary_findings_but_keeps_executive_artifact() -> None:
    result = {
        "root_metric": "expense_detail.total_paid",
        "baseline_value": 100.0,
        "comparison_value": 140.0,
        "delta_value": 40.0,
        "delta_pct": 0.4,
        "drill_path": [{"dimension": "hcc_medium", "top_segments": [{"value": "IP OB"}]}],
        "narrative_summary": "",
        "interaction_summary": {
            "text": "",
            "source": "disabled",
        },
        "recommended_action": [],
        "warnings": [],
        "run_dir": "/tmp/run_dir",
        "manifest_path": "/tmp/run_dir/manifest.json",
        "executive_summary_path": "/tmp/run_dir/summary/executive_summary.json",
        "interaction_matrix": {
            "operational": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}},
            "clinical": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}},
        },
    }

    findings = _build_correlation_findings(result)
    artifacts = _build_correlation_artifacts(result)

    finding_ids = {item["id"] for item in findings}
    artifact_types = {item["type"] for item in artifacts}

    assert "narrative_summary" not in finding_ids
    assert "interaction_summary" not in finding_ids
    assert "interaction_recommendations" not in finding_ids
    assert "executive_summary" in artifact_types


def test_orchestrator_surfaces_recommendations_without_interaction_summary() -> None:
    result = {
        "root_metric": "expense_detail.total_paid",
        "baseline_value": 100.0,
        "comparison_value": 140.0,
        "delta_value": 40.0,
        "delta_pct": 0.4,
        "drill_path": [{"dimension": "hcc_medium", "top_segments": [{"value": "IP OB"}]}],
        "narrative_summary": "",
        "interaction_summary": {
            "text": "",
            "source": "disabled",
        },
        "recommended_action": [
            {
                "rank": 1,
                "priority": "HIGH",
                "category": "Operational",
                "description": "Review the concentrated operational cell.",
                "evidence": [
                    "Drill segment: hcc_medium=IP OB",
                    "Interaction cell reference: op_001",
                    "Metric movement: delta +40",
                    "Caveat: Validate before broad action.",
                ],
                "story_alignment": [
                    "Why: One operational cell explains most of the delta.",
                    "research_consideration: Analyst to review claim drivers.",
                    "cost_of_care_suggestion: Claims processing opportunity.",
                ],
                "peer_benchmarking": [],
                "citation": [],
            }
        ],
        "warnings": [],
        "run_dir": "/tmp/run_dir",
        "manifest_path": "/tmp/run_dir/manifest.json",
        "executive_summary_path": "/tmp/run_dir/summary/executive_summary.json",
        "interaction_matrix": {
            "operational": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}},
            "clinical": {"artifact_paths": {"sql": "", "delta": "", "full_matrix": ""}},
        },
    }

    findings = _build_correlation_findings(result)

    finding_ids = {item["id"] for item in findings}

    assert "interaction_summary" not in finding_ids
    assert "interaction_recommendations" in finding_ids
