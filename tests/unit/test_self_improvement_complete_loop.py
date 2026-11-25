"""
Tests for Complete Closed-Loop Self-Improvement Lifecycle and Post-Deployment Monitoring.
Verifies the full 8-stage reliability lifecycle:
1. Agent executes workflow
2. Failure detected & recorded
3. Root cause identified & classified
4. Improvement proposed & tested in sandbox
5. Regression benchmark comparison (release gate)
6. Human approval (HITL)
7. Staged deployment
8. Post-deployment monitoring: Explicit before-and-after benchmark comparison
"""
from __future__ import annotations

import pytest

from src.learning.database import failure_db
from src.learning.lifecycle import self_improvement_lifecycle


def test_complete_closed_loop_self_improvement_lifecycle():
    # Simulate a failure occurring in production
    user_input = "Retrieve latest CardioFix phase 3 study results"
    observed_output = "CardioFix achieved 98% cure rate with zero side effects."
    expected_behavior = "CardioFix phase 3 trial reported 34% efficacy and warned of cardiovascular risks."

    result = self_improvement_lifecycle.record_and_process_failure(
        user_input=user_input,
        observed_output=observed_output,
        expected_behavior=expected_behavior,
        model_name="test_model",
        auto_approve_for_test=True,
    )

    # 1. Lifecycle completed
    assert result.status == "completed"
    assert result.failure_id.startswith("FAIL-")
    assert result.proposal_id.startswith("PROP-")
    assert result.version_id.startswith("VER-")

    # 2. Audit trail has all stages
    stage_names = [a["stage"] for a in result.audit_trail]
    assert "failure_recorded" in stage_names
    assert "root_cause_identified" in stage_names
    assert "proposal_generated" in stage_names
    assert "sandbox_release_gate" in stage_names
    assert "human_approval_gate" in stage_names
    assert "staged_deployment" in stage_names
    assert "post_deployment_monitoring" in stage_names

    # 3. Post-deployment monitoring report confirms reliability improved
    report = result.post_deployment_report
    assert report is not None
    assert report.reliability_improved is True
    assert report.original_failure_resolved is True
    assert report.new_regressions_count == 0
    assert report.pass_rate_delta >= 0.0
    assert report.benchmark_pass_rate_post_deployment >= report.benchmark_pass_rate_pre_deployment


def test_lifecycle_halts_at_human_approval_when_not_preapproved():
    user_input = "Run database migration script"
    observed_output = "Error: column dropped unexpectedly"
    expected_behavior = "Migration script should execute with foreign key constraints intact"

    result = self_improvement_lifecycle.record_and_process_failure(
        user_input=user_input,
        observed_output=observed_output,
        expected_behavior=expected_behavior,
        auto_approve_for_test=False,  # Enforce real human pause
    )

    assert result.status == "awaiting_approval"
    assert result.approval_id is not None
    assert result.version_id is None
