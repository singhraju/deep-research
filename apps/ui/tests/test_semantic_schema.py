"""Unit tests for ui.semantic_schema — YAML → SemanticSchema parsing."""

import textwrap

import pytest

from ui.semantic_schema import load_semantic_schema


@pytest.fixture
def mini_yaml(tmp_path):
    yaml_text = textwrap.dedent(
        """
        name: mini_view
        description: tiny view for testing
        tables:
          - name: facts
            base_table:
              database: DB1
              schema: SCH1
              table: FACT_TABLE
            dimensions:
              - name: state
                description: state code
                expr: STATE_CD
                data_type: string
                is_enum: true
                sample_values: [CA, NY, TX]
              - name: product
                description: product code
                expr: PROD_CD
                data_type: string
                sample_values: [PM01, PM02]
              - name: provider_id
                description: free-text provider
                expr: PROV_ID
                data_type: string
            time_dimensions:
              - name: incurred_month
                description: incurred month yyyymm
                expr: INCRD_MNTH
                data_type: number
              - name: snap_month
                expr: SNAP_MNTH
                data_type: number
            metrics:
              - name: facts.total_paid
                description: total paid
                expr: SUM(PAID_AMT)
              - name: facts.claim_count
                description: count of claims
                expr: COUNT(*)
          - name: members
            base_table:
              database: DB1
              schema: SCH1
              table: MBR_TABLE
            dimensions:
              - name: state
                description: state code
                expr: STATE_CD
                sample_values: [CA, NY, FL]
            time_dimensions:
              - name: incurred_month
                description: incurred month yyyymm
                expr: INCRD_MNTH
                data_type: number
        metrics:
          - name: pmpm
            description: per member per month
            expr: SUM(PAID_AMT)/COUNT(MBR)
        analysis_modes:
          - name: cost_change
            drill_metric: [facts.total_paid]
            explainer_metrics: [facts.claim_count, pmpm]
            drill_dimensions: [state, product, provider_id]
            period:
              rolling_time_dimension: facts.incurred_month
        """
    ).strip()
    path = tmp_path / "mini.yaml"
    path.write_text(yaml_text)
    return path


def test_load_basic_shape(mini_yaml):
    s = load_semantic_schema(mini_yaml)
    assert s.view_name == "mini_view"
    assert s.default_drill_metric == "facts.total_paid"
    assert s.principal_time_dimension == "incurred_month"


def test_dimensions_match_drill_dimensions(mini_yaml):
    s = load_semantic_schema(mini_yaml)
    names = [d.name for d in s.dimensions]
    assert names == ["state", "product", "provider_id"]


def test_dimension_dedupe_and_union(mini_yaml):
    s = load_semantic_schema(mini_yaml)
    state = next(d for d in s.dimensions if d.name == "state")
    assert state.sample_values == ("CA", "NY", "TX", "FL")
    assert state.is_enum is True
    assert len(state.source_tables) == 2
    assert {t.table_name for t in state.source_tables} == {"facts", "members"}


def test_time_dimensions_union_across_tables(mini_yaml):
    """The parser used to require time dims in EVERY table (intersection),
    which dropped fields like ``snap_month`` that only one table declares as a
    time dim. The UI needs ``snap_month`` available as a Research-time filter,
    so the parser now unions across tables.
    """
    s = load_semantic_schema(mini_yaml)
    names = {t.name for t in s.time_dimensions}
    assert names == {"incurred_month", "snap_month"}


def test_metrics_curated_with_descriptions(mini_yaml):
    s = load_semantic_schema(mini_yaml)
    by_name = {m.name: m for m in s.metrics}
    assert "facts.total_paid" in by_name and by_name["facts.total_paid"].is_drill_metric
    assert by_name["facts.claim_count"].is_drill_metric is False
    assert by_name["pmpm"].description == "per member per month"


def test_free_text_dimension_has_empty_sample_values(mini_yaml):
    s = load_semantic_schema(mini_yaml)
    prov = next(d for d in s.dimensions if d.name == "provider_id")
    assert prov.sample_values == ()
    assert prov.is_enum is False


def test_load_real_local_yaml():
    """Smoke test against the actual repo YAML to catch real-world drift."""
    from pathlib import Path

    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "correlation_pattern"
        / "coc_ecap_ip_auth_sematic_view_with_samples_local.yaml"
    )
    if not yaml_path.exists():
        pytest.skip(f"YAML not present at {yaml_path}")
    s = load_semantic_schema(yaml_path)
    dim_names = [d.name for d in s.dimensions]
    assert "service_area_state" in dim_names
    assert "hcc_medium" in dim_names
    # Real YAML lists snap_month under expense_detail.time_dimensions only;
    # the new union semantics surface it so the UI can render its widget.
    time_dim_names = {t.name for t in s.time_dimensions}
    assert "incurred_month" in time_dim_names
    assert "snap_month" in time_dim_names
    assert s.default_drill_metric == "expense_detail.total_paid"
    assert s.principal_time_dimension == "incurred_month"
    assert s.principal_time_dimension_qualified == "expense_detail.incurred_month"
