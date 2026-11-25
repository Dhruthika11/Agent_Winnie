"""
Comprehensive test suite for Strict Human-In-The-Loop (HITL) Binding,
Cross-Tenant / Cross-User Replay Prevention, and Execution Gateway Verification.
"""
import os
import time
import pytest

from src.tools import hitl
from src.tools.core import terminal
from src.tools.core.filesystem import WORKSPACE_DIR


@pytest.fixture(autouse=True)
def clean_memory_store():
    """Ensure in-memory store is clean before each test."""
    hitl._memory_store.clear()
    yield
    hitl._memory_store.clear()


def test_strict_approval_binding_fields():
    """Requirement 1: Verify all required cryptographic and operational fields are strictly bound."""
    approval_id = hitl.request_approval(
        user_id="user_prod_1",
        tenant_id="tenant_alpha",
        session_id="sess_123",
        tool_name="terminal.run",
        command="rm -rf /workspace/tmp/data.log",
        arguments={"command": "rm -rf /workspace/tmp/data.log"},
        working_directory=str(WORKSPACE_DIR),
        risk_level="high",
        timeout_seconds=60.0,
    )

    record = hitl.get_approval(approval_id)
    assert record is not None
    assert record["approval_id"] == approval_id
    assert record["user_id"] == "user_prod_1"
    assert record["tenant_id"] == "tenant_alpha"
    assert record["session_id"] == "sess_123"
    assert record["tool_name"] == "terminal.run"
    assert record["command"] == "rm -rf /workspace/tmp/data.log"
    assert record["arguments"] == {"command": "rm -rf /workspace/tmp/data.log"}
    assert record["exact_command_and_arguments"]["command"] == "rm -rf /workspace/tmp/data.log"
    assert os.path.normpath(record["working_directory"]) == os.path.normpath(str(WORKSPACE_DIR))
    assert record["risk_level"] == "high"
    assert "/workspace/tmp/data.log" in record["affected_files"]
    assert record["status"] == "pending"
    assert record["created_at"] <= time.time()
    assert record["expires_at"] == pytest.approx(record["created_at"] + 60.0, rel=1e-2)
    assert "binding_hash" in record
    assert hitl.verify_binding(record) is True


def test_cryptographic_binding_tamper_detection():
    """Requirement 1: Any tampering with bound fields invalidates cryptographic verification."""
    approval_id = hitl.request_approval(
        user_id="alice",
        tenant_id="tenant_1",
        session_id="s1",
        tool_name="terminal.run",
        command="cat secret.txt",
    )
    record = hitl.get_approval(approval_id)
    assert hitl.verify_binding(record) is True

    # Tamper with command
    tampered_record = dict(record)
    tampered_record["command"] = "cat other_secret.txt"
    assert hitl.verify_binding(tampered_record) is False

    # Tamper with tenant
    tampered_record_tenant = dict(record)
    tampered_record_tenant["tenant_id"] = "tenant_attacker"
    assert hitl.verify_binding(tampered_record_tenant) is False

    # Tamper with user
    tampered_record_user = dict(record)
    tampered_record_user["user_id"] = "mallory"
    assert hitl.verify_binding(tampered_record_user) is False


def test_cross_tenant_resolution_rejected():
    """Requirement 2: Tenant mismatch in resolve_approval must raise PermissionError."""
    approval_id = hitl.request_approval(
        user_id="user_a",
        tenant_id="tenant_legit",
        session_id="s_alpha",
        tool_name="terminal.run",
        command="ls -la",
    )

    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(
            approval_id=approval_id,
            approved=True,
            tenant_id="tenant_attacker",
        )
    assert "Cross-tenant access denied" in str(exc_info.value)

    # Ensure record remains pending
    rec = hitl.get_approval(approval_id)
    assert rec["status"] == "pending"


def test_cross_user_resolution_rejected():
    """Requirement 2: User mismatch in resolve_approval must raise PermissionError."""
    approval_id = hitl.request_approval(
        user_id="alice",
        tenant_id="tenant_1",
        session_id="s_alpha",
        tool_name="terminal.run",
        command="ls -la",
    )

    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(
            approval_id=approval_id,
            approved=True,
            user_id="eve",  # Different user
            tenant_id="tenant_1",
        )
    assert "Cross-user access denied" in str(exc_info.value)

    rec = hitl.get_approval(approval_id)
    assert rec["status"] == "pending"


def test_expired_approval_rejected():
    """Requirement 2: Expired approvals must set status='expired' and reject execution."""
    approval_id = hitl.request_approval(
        user_id="bob",
        tenant_id="tenant_corp",
        session_id="sess_exp",
        tool_name="terminal.run",
        command="echo expired",
        timeout_seconds=0.1,  # Short timeout
    )

    # Wait for expiration
    time.sleep(0.15)

    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(approval_id=approval_id, approved=True, tenant_id="tenant_corp")
    assert "expired" in str(exc_info.value).lower()

    record = hitl.get_approval(approval_id)
    assert record["status"] == "expired"
    assert hitl.is_approved(approval_id) is False


def test_duplicate_resolution_and_execution_prevention():
    """Requirement 2: Once resolved or executed, cannot be resolved or executed twice."""
    approval_id = hitl.request_approval(
        user_id="alice",
        tenant_id="tenant_corp",
        session_id="sess_dup",
        tool_name="terminal.run",
        command="echo once",
    )

    # First resolution: OK
    res = hitl.resolve_approval(approval_id=approval_id, approved=True, tenant_id="tenant_corp")
    assert res["status"] == "approved"

    # Second resolution: Duplicate resolution prohibited
    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(approval_id=approval_id, approved=True, tenant_id="tenant_corp")
    assert "already been resolved" in str(exc_info.value)

    # Mark executed
    hitl.mark_executed(approval_id)
    rec = hitl.get_approval(approval_id)
    assert rec["status"] == "executed"
    assert hitl.is_approved(approval_id) is False

    # Duplicate execution attempt via mark_executed
    with pytest.raises(PermissionError) as exc_info:
        hitl.mark_executed(approval_id)
    assert "already been executed" in str(exc_info.value)

    # Duplicate resolution after execution attempt
    with pytest.raises(PermissionError) as exc_info:
        hitl.resolve_approval(approval_id=approval_id, approved=True, tenant_id="tenant_corp")
    assert "already been executed" in str(exc_info.value)


def test_execution_gateway_plain_boolean_untrusted():
    """Requirement 3: Never trust plain boolean approved=True without approval token."""
    res = terminal.run_command("rm -rf data", approved=True, approval_id=None)
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "approval_id token is required" in res["error"]["message"]


def test_execution_gateway_unapproved_token_rejected():
    """Requirement 3: An approval token with status='pending' or 'rejected' is rejected."""
    # Pending token
    app_id = hitl.request_approval(
        user_id="user1",
        tenant_id="t1",
        session_id="s1",
        tool_name="terminal.run",
        command="echo 'unapproved test'",
    )
    res_pending = terminal.run_command("echo 'unapproved test'", approved=True, approval_id=app_id)
    assert res_pending["success"] is False
    assert "not been approved" in res_pending["error"]["message"]

    # Rejected token
    hitl.resolve_approval(app_id, approved=False, tenant_id="t1")
    res_rejected = terminal.run_command("echo 'unapproved test'", approved=True, approval_id=app_id)
    assert res_rejected["success"] is False
    assert "was explicitly rejected" in res_rejected["error"]["message"]


def test_execution_gateway_command_mutation_rejected():
    """Requirement 2 & 3: If command changes even slightly after approval, require fresh approval."""
    app_id = hitl.request_approval(
        user_id="user1",
        tenant_id="t1",
        session_id="s1",
        tool_name="terminal.run",
        command="echo safe",
    )
    hitl.resolve_approval(app_id, approved=True, tenant_id="t1")

    # Command altered slightly
    res = terminal.run_command("echo safe --extra", approved=True, approval_id=app_id)
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "Command mismatch" in res["error"]["message"]
    assert "Fresh approval required" in res["error"]["message"]


def test_execution_gateway_workspace_mutation_rejected():
    """Requirement 3: Approval must be valid for the exact workspace directory."""
    app_id = hitl.request_approval(
        user_id="user1",
        tenant_id="t1",
        session_id="s1",
        tool_name="terminal.run",
        command="echo test_workspace",
        working_directory="/tmp/fake_foreign_workspace",
    )
    hitl.resolve_approval(app_id, approved=True, tenant_id="t1")

    # Attempt execution in real WORKSPACE_DIR
    res = terminal.run_command("echo test_workspace", approved=True, approval_id=app_id)
    assert res["success"] is False
    assert res["error"]["code"] == "PERMISSION_DENIED"
    assert "Workspace directory mismatch" in res["error"]["message"]


def test_execution_gateway_successful_run_and_replay_protection():
    """Requirement 2 & 3: Approved execution succeeds and prevents second execution (replay)."""
    cmd = "echo Executed_safely"
    app_id = hitl.request_approval(
        user_id="operator_1",
        tenant_id="tenant_prod",
        session_id="sess_valid",
        tool_name="terminal.run",
        command=cmd,
    )
    hitl.resolve_approval(app_id, approved=True, tenant_id="tenant_prod")

    # First execution: Must succeed
    res1 = terminal.run_command(cmd, approved=True, approval_id=app_id)
    assert res1["success"] is True
    assert "Executed_safely" in res1["data"]["stdout"]

    # Verify status changed to 'executed'
    rec = hitl.get_approval(app_id)
    assert rec["status"] == "executed"
    assert "executed_at" in rec

    # Second execution (Replay attack): Must be rejected
    res2 = terminal.run_command(cmd, approved=True, approval_id=app_id)
    assert res2["success"] is False
    assert res2["error"]["code"] == "PERMISSION_DENIED"
    assert "already been executed" in res2["error"]["message"]


def test_execution_gateway_cross_tenant_run_rejected():
    """Requirement 2: Cross-tenant execution at gateway rejected."""
    cmd = "echo tenant_check"
    app_id = hitl.request_approval(
        user_id="user1",
        tenant_id="tenant_a",
        session_id="sess_1",
        tool_name="terminal.run",
        command=cmd,
    )
    hitl.resolve_approval(app_id, approved=True, tenant_id="tenant_a")

    # Tenant B tries to execute Tenant A's approval
    res = terminal.run_command(cmd, approved=True, approval_id=app_id, tenant_id="tenant_b")
    assert res["success"] is False
    assert "Cross-tenant access denied" in res["error"]["message"]


def test_affected_files_extraction():
    """Requirement 1: Verify extraction of affected files from arguments and shell command."""
    files_arg = hitl.extract_affected_files(
        "terminal.delete_file",
        {"path": "data/sample.json", "paths": ["logs/a.txt", "logs/b.txt"]},
    )
    assert "data/sample.json" in files_arg
    assert "logs/a.txt" in files_arg
    assert "logs/b.txt" in files_arg

    files_cmd = hitl.extract_affected_files(
        "terminal.run",
        command="cat input.txt > /tmp/output.log && rm cleanup.tmp",
    )
    assert "/tmp/output.log" in files_cmd
    assert "input.txt" in files_cmd
    assert "cleanup.tmp" in files_cmd
