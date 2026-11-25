"""
Tests for Workflow 6: Autonomous Multi-Step Tasks.
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from src.workflows.autonomous.graph import autonomous_graph
from src.workflows.autonomous.planner import compute_topological_batches, decompose_to_dag, validate_dag
from src.workflows.base import create_initial_workflow_state


def test_autonomous_dag_planner_and_cycle_validator():
    """Verify dynamic DAG decomposition, cycle checking, and topological batching."""
    objective = "Research three open-source vector databases, compare their performance and generate a report"
    dag = decompose_to_dag(objective)

    assert len(dag.tasks) == 5
    assert "task_research_qdrant" in dag.tasks
    assert "task_research_milvus" in dag.tasks
    assert "task_research_chroma" in dag.tasks
    assert "task_compare_performance" in dag.tasks
    assert "task_generate_final_report" in dag.tasks

    # Verify dependency structure
    comp_task = dag.tasks["task_compare_performance"]
    assert "task_research_qdrant" in comp_task.dependencies
    assert "task_research_milvus" in comp_task.dependencies
    assert "task_research_chroma" in comp_task.dependencies

    # Verify topological batching
    assert len(dag.execution_batches) == 3
    # Batch 1: the 3 independent research tasks can run in parallel
    assert len(dag.execution_batches[0]) == 3
    # Batch 2: the comparison task
    assert dag.execution_batches[1] == ["task_compare_performance"]
    # Batch 3: final report synthesis
    assert dag.execution_batches[2] == ["task_generate_final_report"]

    is_valid, reason = validate_dag(dag)
    assert is_valid is True


def test_autonomous_workflow_execution_and_artifact_assembly():
    """Verify autonomous workflow full execution through LangGraph."""
    objective = "Research three open-source vector databases, compare their performance and generate a report"
    initial_state = create_initial_workflow_state(
        objective=objective,
        workflow_type="autonomous",
    )

    with patch("src.workflows.autonomous.nodes.tool_execution_engine.execute") as mock_exec:
        mock_exec.return_value.output = "Search benchmark data retrieved."
        final_state = autonomous_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"
    assert final_state["verification_status"] == "verified"

    final_out = final_state["final_output"]
    assert final_out["is_satisfied"] is True
    artifact = final_out["artifact"]
    assert artifact["total_tasks"] == 5
    assert artifact["successful_tasks"] == 5
    assert artifact["objective_satisfied"] is True

    outputs = artifact["combined_outputs"]
    assert "task_compare_performance" in outputs
    assert "comparison_matrix" in outputs["task_compare_performance"]
