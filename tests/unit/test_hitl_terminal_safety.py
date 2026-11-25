"""
Comprehensive Regression Test Suite for Hermes HITL Terminal Safety & Destructive Command Control.

Verifies all 12 user-specified test scenarios:
1. `ls ./logs` executes without approval (READ_ONLY, ALLOW).
2. `cat ./logs/example.log` follows read policy (READ_ONLY, ALLOW).
3. `rm ./logs/example.log` requests approval (DESTRUCTIVE, REQUIRE_APPROVAL).
4. `rm -rf ./logs/*.log` requests approval (DESTRUCTIVE, REQUIRE_APPROVAL).
5. Recursive deletion outside the permitted workspace is blocked (BLOCKED, DENY).
6. User rejects deletion: no execution occurs (file is preserved).
7. Approval expires: no execution occurs (file is preserved).
8. Modified command after approval: fresh approval required (mutation rejected).
9. Another tenant cannot approve the request (cross-tenant rejected).
10. Duplicate approval submissions do not execute twice (single-use replay protection).
11. Dangerous prohibited commands remain blocked (BLOCKED, DENY).
12. Successful approval results in execution and post-execution verification (dummy file deleted).
"""
import os
import shutil
import sys
import time
from pathlib import Path
import pytest

from src.agent.tool_policy import (
    tool_policy,
    PolicyDecision,
    RiskLevel,
    ActionCategory,
    validate_command_structure,
    WORKSPACE_DIR,
)
from src.tools import hitl
from src.tools.core import terminal


@pytest.fixture(autouse=True)
def clean_memory_store():
    """Ensure in-memory approval store is clean before and after each test."""
    hitl._memory_store.clear()
    yield
    hitl._memory_store.clear()


@pytest.fixture
def dummy_logs_dir():
    """Create a temporary logs directory in the workspace for testing."""
    logs_dir = WORKSPACE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    example_file = logs_dir / "example.log"
    example_file.write_text("2026-11-25 INFO test log message\n")
    yield logs_dir, example_file
    # Cleanup
    if logs_dir.exists():
        shutil.rmtree(logs_dir, ignore_errors=True)


def test_1_ls_logs_executes_without_approval(dummy_logs_dir):
    """Scenario 1: `ls ./logs` executes without approval (READ_ONLY, ALLOW)."""
    logs_dir, _ = dummy_logs_dir
    cmd = "ls ./logs"

    # Policy evaluation check
    eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
    assert eval_res.decision == PolicyDecision.ALLOW
    assert eval_res.risk_level == RiskLevel.LOW
    assert eval_res.action_category == ActionCategory.READ_ONLY

    # Structural check
    dec, risk, reason, cat = validate_command_structure(cmd)
    assert dec == PolicyDecision.ALLOW
    assert risk == RiskLevel.LOW
    assert cat == ActionCategory.READ_ONLY

    # Execution check without approval
    res = terminal.run_command(cmd, approved=False, approval_id=None)
    assert res["success"] is True
    assert "example.log" in res["data"]["stdout"] or res["data"]["returncode"] == 0


def test_2_cat_logs_example_log_follows_read_policy(dummy_logs_dir):
    """Scenario 2: `cat ./logs/example.log` follows read policy (READ_ONLY, ALLOW)."""
    _, example_file = dummy_logs_dir
    cmd = "cat ./logs/example.log"

    # Terminal command check
    eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
    assert eval_res.decision == PolicyDecision.ALLOW
    assert eval_res.risk_level == RiskLevel.LOW
    assert eval_res.action_category == ActionCategory.READ_ONLY

    # Filesystem tool equivalent check
    fs_eval = tool_policy.evaluate("filesystem.read_file", {"path": "logs/example.log"})
    assert fs_eval.decision == PolicyDecision.ALLOW
    assert fs_eval.risk_level == RiskLevel.LOW
    assert fs_eval.action_category == ActionCategory.READ_ONLY

    # Execution check without approval
    res = terminal.run_command(cmd, approved=False, approval_id=None)
    assert res["success"] is True
    assert "2026-11-25 INFO test log message" in res["data"]["stdout"]


def test_3_rm_logs_example_log_requests_approval(dummy_logs_dir):
    """Scenario 3: `rm ./logs/example.log` requests approval (DESTRUCTIVE, REQUIRE_APPROVAL)."""
    _, example_file = dummy_logs_dir
    cmd = "rm ./logs/example.log"

    # Policy evaluation check
    eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
    assert eval_res.decision == PolicyDecision.REQUIRE_APPROVAL
    assert eval_res.risk_level == RiskLevel.HIGH
    assert eval_res.action_category == ActionCategory.DESTRUCTIVE

    # Structural check
    dec, risk, reason, cat = validate_command_structure(cmd)
    assert dec == PolicyDecision.REQUIRE_APPROVAL
    assert risk == RiskLevel.HIGH
    assert cat == ActionCategory.DESTRUCTIVE

    # Execution without approval must be blocked
    res = terminal.run_command(cmd, approved=False, approval_id=None)
    assert res["success"] is False
    assert res["error"]["code"] in ("APPROVAL_REQUIRED", "PERMISSION_DENIED")
    assert "approval" in res["error"]["message"].lower()
    # File must not be deleted
    assert example_file.exists() is True


def test_4_rm_rf_logs_wildcard_requests_approval(dummy_logs_dir):
    """Scenario 4: `rm -rf ./logs/*.log` requests approval (DESTRUCTIVE, REQUIRE_APPROVAL)."""
    _, example_file = dummy_logs_dir
    cmd = "rm -rf ./logs/*.log"

    # Policy evaluation check
    eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
    assert eval_res.decision == PolicyDecision.REQUIRE_APPROVAL
    assert eval_res.risk_level == RiskLevel.HIGH
    assert eval_res.action_category == ActionCategory.DESTRUCTIVE

    # Structural check
    dec, risk, reason, cat = validate_command_structure(cmd)
    assert dec == PolicyDecision.REQUIRE_APPROVAL
    assert risk == RiskLevel.HIGH
    assert cat == ActionCategory.DESTRUCTIVE

    # Execution without approval must be blocked
    res = terminal.run_command(cmd, approved=False, approval_id=None)
    assert res["success"] is False
    assert res["error"]["code"] in ("APPROVAL_REQUIRED", "PERMISSION_DENIED")
    assert "approval" in res["error"]["message"].lower()
    # File must not be deleted
    assert example_file.exists() is True


def test_5_recursive_deletion_outside_workspace_is_blocked():
    """Scenario 5: Recursive deletion outside the permitted workspace is blocked (BLOCKED, DENY)."""
    prohibited_escapes = [
        "rm -rf /",
        "rm -rf ../../../etc",
        "rm -rf ~",
        "del ..\\..\\Windows\\System32",
        "rm -rf ../../",
    ]

    for cmd in prohibited_escapes:
        eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
        assert eval_res.decision == PolicyDecision.DENY, f"Expected DENY for '{cmd}', got {eval_res.decision}"
        assert eval_res.risk_level == RiskLevel.CRITICAL
        assert eval_res.action_category == ActionCategory.BLOCKED

        dec, risk, reason, cat = validate_command_structure(cmd)
        assert dec == PolicyDecision.DENY
        assert risk == RiskLevel.CRITICAL
        assert cat == ActionCategory.BLOCKED

        # Even if someone attempts to pass approved=True, gateway rejects DENY commands
        res = terminal.run_command(cmd, approved=True, approval_id="fake_id")
        assert res["success"] is False
        assert res["error"]["code"] in ("POLICY_DENIED", "PERMISSION_DENIED")


def test_6_user_rejects_deletion_no_execution_occurs():
    """Scenario 6: User rejects deletion: no execution occurs and file is preserved."""
    dummy_file = WORKSPACE_DIR / "dummy_reject_scenario.log"
    dummy_file.write_text("important data to preserve\n")

    cmd = f"rm {dummy_file.name}"
    approval_id = hitl.request_approval(
        user_id="analyst_1",
        tenant_id="tenant_prod",
        session_id="sess_reject",
        tool_name="terminal.run",
        command=cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
    )

    # Human rejects the request
    rec = hitl.resolve_approval(
        approval_id=approval_id,
        approved=False,
        tenant_id="tenant_prod",
        user_id="analyst_1",
        decided_by="admin_reviewer",
    )
    assert rec["status"] == "rejected"
    assert hitl.is_approved(approval_id) is False

    # Attempt execution with rejected approval token (either approved=False or approved=True)
    res = terminal.run_command(
        command=cmd,
        approved=False,
        approval_id=approval_id,
        user_id="analyst_1",
        tenant_id="tenant_prod",
    )
    assert res["success"] is False
    assert res["error"]["code"] in ("APPROVAL_REQUIRED", "PERMISSION_DENIED")

    res_bypass = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="analyst_1",
        tenant_id="tenant_prod",
    )
    assert res_bypass["success"] is False
    assert res_bypass["error"]["code"] == "PERMISSION_DENIED"
    assert "rejected" in res_bypass["error"]["message"].lower()

    # Post-condition verification: File MUST be preserved
    assert dummy_file.exists() is True
    assert dummy_file.read_text() == "important data to preserve\n"

    # Cleanup
    dummy_file.unlink(missing_ok=True)


def test_7_approval_expires_no_execution_occurs():
    """Scenario 7: Approval expires: no execution occurs and file is preserved."""
    dummy_file = WORKSPACE_DIR / "dummy_expire_scenario.log"
    dummy_file.write_text("sensitive log to preserve on expiration\n")

    cmd = f"rm {dummy_file.name}"
    # Request approval with 0.05 second expiration
    approval_id = hitl.request_approval(
        user_id="analyst_2",
        tenant_id="tenant_prod",
        session_id="sess_expire",
        tool_name="terminal.run",
        command=cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
        timeout_seconds=0.05,
    )

    # Wait for expiration
    time.sleep(0.08)

    # Resolution attempt after expiration must fail
    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(
            approval_id=approval_id,
            approved=True,
            tenant_id="tenant_prod",
        )
    assert "expired" in str(exc_info.value).lower()

    # Execution attempt must be rejected
    res = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="analyst_2",
        tenant_id="tenant_prod",
    )
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "expired" in res["error"]["message"].lower()

    # Post-condition verification: File MUST still exist
    assert dummy_file.exists() is True

    # Cleanup
    dummy_file.unlink(missing_ok=True)


def test_8_modified_command_after_approval_requires_fresh_approval():
    """Scenario 8: Modified command after approval: fresh approval required."""
    dummy_orig = WORKSPACE_DIR / "dummy_orig.log"
    dummy_mutated = WORKSPACE_DIR / "dummy_mutated.log"
    dummy_orig.write_text("original file\n")
    dummy_mutated.write_text("mutated target file\n")

    orig_cmd = f"rm {dummy_orig.name}"
    mutated_cmd = f"rm {dummy_mutated.name}"

    approval_id = hitl.request_approval(
        user_id="engineer_bob",
        tenant_id="tenant_security",
        session_id="sess_mutation",
        tool_name="terminal.run",
        command=orig_cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
    )

    # Approved for orig_cmd
    hitl.resolve_approval(
        approval_id=approval_id,
        approved=True,
        tenant_id="tenant_security",
        user_id="engineer_bob",
    )
    assert hitl.is_approved(approval_id) is True

    # Attempt to execute mutated_cmd using the approval token for orig_cmd
    res = terminal.run_command(
        command=mutated_cmd,
        approved=True,
        approval_id=approval_id,
        user_id="engineer_bob",
        tenant_id="tenant_security",
    )
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "mutated" in res["error"]["message"].lower() or "fresh approval" in res["error"]["message"].lower()

    # Mutated file must NOT have been touched
    assert dummy_mutated.exists() is True
    assert dummy_orig.exists() is True

    # Cleanup
    dummy_orig.unlink(missing_ok=True)
    dummy_mutated.unlink(missing_ok=True)


def test_9_cross_tenant_cannot_approve_request():
    """Scenario 9: Another tenant cannot approve the request."""
    cmd = "rm test_tenant_isolation.log"
    approval_id = hitl.request_approval(
        user_id="legit_user",
        tenant_id="tenant_alpha",
        session_id="sess_cross_tenant",
        tool_name="terminal.run",
        command=cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
    )

    # Attacker from tenant_beta attempts to approve tenant_alpha's request
    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(
            approval_id=approval_id,
            approved=True,
            tenant_id="tenant_beta",
            user_id="attacker_user",
        )
    assert "cross-tenant" in str(exc_info.value).lower()

    # Status must remain pending
    record = hitl.get_approval(approval_id)
    assert record["status"] == "pending"

    # Execution while still pending must be rejected
    res_pending = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="attacker_user",
        tenant_id="tenant_beta",
    )
    assert res_pending["success"] is False
    assert res_pending["error"]["code"] == "PERMISSION_DENIED"
    assert "pending" in res_pending["error"]["message"].lower()

    # Legit operator from tenant_alpha approves it
    hitl.resolve_approval(
        approval_id=approval_id,
        approved=True,
        tenant_id="tenant_alpha",
        user_id="legit_user",
    )
    assert hitl.is_approved(approval_id) is True

    # Now attacker from tenant_beta attempts cross-tenant execution of approved token
    res_cross = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="attacker_user",
        tenant_id="tenant_beta",
    )
    assert res_cross["success"] is False
    assert res_cross["error"]["code"] == "PERMISSION_DENIED"
    assert "cross-tenant" in res_cross["error"]["message"].lower()


def test_10_duplicate_approval_submissions_do_not_execute_twice():
    """Scenario 10: Duplicate approval submissions do not execute twice."""
    dummy_file = WORKSPACE_DIR / "dummy_duplicate_execution.log"
    dummy_file.write_text("single execution only\n")

    cmd = f"rm {dummy_file.name}"
    approval_id = hitl.request_approval(
        user_id="user_prod",
        tenant_id="tenant_core",
        session_id="sess_replay",
        tool_name="terminal.run",
        command=cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
    )

    # Approve
    hitl.resolve_approval(
        approval_id=approval_id,
        approved=True,
        tenant_id="tenant_core",
        user_id="user_prod",
    )

    # First execution -> Must succeed
    res1 = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="user_prod",
        tenant_id="tenant_core",
    )
    assert res1["success"] is True
    assert dummy_file.exists() is False

    # Second execution using identical token -> Must be rejected as already executed
    res2 = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="user_prod",
        tenant_id="tenant_core",
    )
    assert res2["success"] is False
    assert res2["error"]["code"] == "PERMISSION_DENIED"
    assert "already been executed" in res2["error"]["message"].lower() or "executed" in res2["error"]["message"].lower()


def test_11_dangerous_prohibited_commands_remain_blocked():
    """Scenario 11: Dangerous prohibited commands remain blocked regardless of approval."""
    dangerous_commands = [
        "mkfs.ext4 /dev/sda1",
        ":(){ :|:& };:",
        "cat .env",
        "dd if=/dev/zero of=/dev/sda",
        "cat ~/.ssh/id_rsa",
        "cat secrets.key",
    ]

    for cmd in dangerous_commands:
        eval_res = tool_policy.evaluate("terminal.run", {"command": cmd})
        assert eval_res.decision == PolicyDecision.DENY, f"Expected DENY for '{cmd}'"
        assert eval_res.risk_level == RiskLevel.CRITICAL
        assert eval_res.action_category == ActionCategory.BLOCKED

        dec, risk, reason, cat = validate_command_structure(cmd)
        assert dec == PolicyDecision.DENY
        assert risk == RiskLevel.CRITICAL
        assert cat == ActionCategory.BLOCKED

        # Execution gateway blocks DENY commands
        res = terminal.run_command(cmd, approved=True, approval_id="bypass_attempt")
        assert res["success"] is False
        assert res["error"]["code"] in ("POLICY_DENIED", "PERMISSION_DENIED")


def test_12_successful_approval_executes_and_verifies_post_condition():
    """Scenario 12: Successful approval results in execution and post-execution verification."""
    dummy_file = WORKSPACE_DIR / "dummy_success_delete.log"
    dummy_file.write_text("delete this file upon approved HITL authorization\n")
    assert dummy_file.exists() is True

    cmd = f"rm {dummy_file.name}"
    approval_id = hitl.request_approval(
        user_id="lead_dev",
        tenant_id="tenant_main",
        session_id="sess_lifecycle",
        tool_name="terminal.run",
        command=cmd,
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
    )

    # Human reviews and approves
    rec = hitl.resolve_approval(
        approval_id=approval_id,
        approved=True,
        tenant_id="tenant_main",
        user_id="lead_dev",
        decided_by="admin_lead",
    )
    assert rec["status"] == "approved"
    assert hitl.is_approved(approval_id) is True

    # Execute approved action through terminal execution gateway
    res = terminal.run_command(
        command=cmd,
        approved=True,
        approval_id=approval_id,
        user_id="lead_dev",
        tenant_id="tenant_main",
    )
    assert res["success"] is True

    # Post-execution verification:
    # 1. Dummy file has been permanently deleted
    assert dummy_file.exists() is False

    # 2. Approval status is transitioned to 'executed'
    final_rec = hitl.get_approval(approval_id)
    assert final_rec["status"] == "executed"

    # 3. Cannot be re-approved or reused
    assert hitl.is_approved(approval_id) is False
