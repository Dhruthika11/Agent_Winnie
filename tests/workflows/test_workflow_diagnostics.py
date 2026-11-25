"""
Tests for Workflow 5: System Diagnostics & Recovery.
"""
from __future__ import annotations

import pytest

from src.workflows.base import create_initial_workflow_state
from src.workflows.diagnostics.graph import diagnostics_graph


def test_diagnostics_workflow_identifies_failure_and_enforces_approval():
    """
    Scenario: User asks: "Why is my MCP server failing? Collect logs and fix it."
    Hermes probes health, isolates web_scraper_mcp timeout, classifies root cause
    as tool_failure, generates a restart proposal, enforces HITL approval, and verifies recovery.
    """
    initial_state = create_initial_workflow_state(
        objective="Why is my MCP server failing? Check logs and troubleshoot the error",
        workflow_type="diagnostics",
    )

    final_state = diagnostics_graph.invoke(initial_state)

    assert final_state["execution_status"] == "completed"

    wf_data = final_state["workflow_data"]

    # 1. Failing component identified
    assert wf_data["failing_component"] in ("mcp_server", "web_scraper_mcp")

    # 2. Root cause classified
    diag = wf_data["root_cause_diagnosis"]
    assert diag["failure_category"] == "tool_failure"
    assert diag["confidence"] > 0.90
    assert len(diag["evidence"]) >= 1

    # 3. Remediation proposed
    prop = wf_data["remediation_proposal"]
    assert prop["action_type"] == "service_restart"
    assert prop["requires_approval"] is True

    # 4. Safety approval check was triggered
    assert wf_data.get("approval_required") is True
    assert final_state.get("approval_status") in ("required", "approved")

    # 5. Recovery verified
    assert wf_data["recovery_verified"] is True
    final_out = final_state["final_output"]
    assert final_out["is_recovered"] is True
    assert final_out["category"] == "tool_failure"
