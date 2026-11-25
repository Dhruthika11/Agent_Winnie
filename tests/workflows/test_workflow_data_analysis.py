"""
Tests for Workflow 3: Data Analysis.
"""
from __future__ import annotations

import pytest

from src.workflows.base import create_initial_workflow_state
from src.workflows.data_analysis.graph import data_analysis_graph


def test_data_analysis_workflow_with_statistical_trap_detection():
    """
    Scenario: User asks: "Analyze this CSV and tell me whether the two groups are statistically different
    and what causes the difference."
    Hermes loads the dataset, computes Welch's t-test, detects statistical traps
    (correlation not causation, small sample power), and produces formatted comparison charts.
    """
    sample_csv = (
        "group,score,user_id\n"
        "treatment,85.2,u1\n"
        "treatment,89.1,u2\n"
        "treatment,92.0,u3\n"
        "treatment,86.5,u4\n"
        "treatment,90.4,u5\n"
        "control,73.1,u6\n"
        "control,74.5,u7\n"
        "control,71.2,u8\n"
        "control,75.8,u9\n"
        "control,72.4,u10\n"
    )

    initial_state = create_initial_workflow_state(
        objective="Analyze this CSV and tell me whether treatment causes higher scores and whether groups are statistically different",
        workflow_type="data_analysis",
        workflow_data={"raw_csv": sample_csv},
    )

    final_state = data_analysis_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"
    assert final_state["verification_status"] == "verified"

    wf_data = final_state["workflow_data"]

    # 1. Dataset summary
    summary = wf_data["dataset_summary"]
    assert summary["row_count"] == 10
    assert "group" in summary["columns"]
    assert "score" in summary["columns"]

    # 2. Statistical trap detection (correlation vs causation flagged!)
    traps = wf_data["statistical_traps"]
    assert len(traps) >= 1
    trap_types = [t["trap_type"] for t in traps]
    assert "correlation_not_causation" in trap_types
    assert "low_statistical_power" in trap_types

    # 3. Statistical test results
    test_res = wf_data["statistical_result"]
    assert test_res["group_a_mean"] > test_res["group_b_mean"]
    assert test_res["p_value"] < 0.05
    assert test_res["is_significant_at_alpha_05"] is True
    assert test_res["effect_size_cohens_d"] > 0.8  # Large effect

    # 4. Final output
    final_out = final_state["final_output"]
    assert final_out["is_statistically_significant"] is True
    report = final_out["analysis_report"]
    assert "Distribution Comparison Chart" in report["visualization_markdown"]
    assert len(report["substantive_findings"]) >= 1
    assert any("correlation" in lim.lower() for lim in report["limitations"])
