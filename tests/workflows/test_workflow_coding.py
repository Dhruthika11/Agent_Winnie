"""
Tests for Workflow 2: Code Execution & Debugging.
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from src.workflows.base import create_initial_workflow_state
from src.workflows.coding.graph import coding_graph


def test_coding_workflow_clean_execution():
    """Test standard clean code generation, sandbox execution, and test reporting."""
    initial_state = create_initial_workflow_state(
        objective="Write a Python function to compute the arithmetic mean of a dataset and verify output",
        workflow_type="coding",
    )

    final_state = coding_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"
    assert final_state["verification_status"] == "verified"

    final_out = final_state["final_output"]
    assert final_out["is_verified"] is True
    assert "def solve():" in final_out["final_code"]
    assert "Computed result: 30.0" in final_out["execution_output"]


def test_coding_workflow_automated_debugging_loop_on_error():
    """
    Scenario: Initial code has a ZeroDivisionError bug.
    Workflow executes -> fails test -> triggers debug_fix node -> patches code -> re-executes -> passes!
    """
    buggy_code = (
        "def compute_average(items):\n"
        "    data = []\n"
        "    return sum(data) / len(data)\n"
        "compute_average([])\n"
    )

    initial_state = create_initial_workflow_state(
        objective="Write a Python script to compute average without zero division errors",
        workflow_type="coding",
        workflow_data={"generated_code": buggy_code},
    )

    final_state = coding_graph.invoke(initial_state)

    assert final_state["retry_count"] >= 1
    wf_data = final_state["workflow_data"]
    assert len(wf_data["debug_attempts"]) >= 1

    # Verify that the patch was applied
    attempt = wf_data["debug_attempts"][0]
    assert "ZeroDivisionError" in attempt["error_message"] or "division by zero" in attempt["error_message"]
    assert "max(1, len(data))" in wf_data["generated_code"]
