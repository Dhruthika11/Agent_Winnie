"""
Unit tests for Approval API Endpoints and UI Integration in Hermes Agent Platform.
Verifies GET /approvals/{approval_id}, POST /approvals/{approval_id}, tenant isolation,
double-approval prevention, expired approval prevention, and UI requirements.
"""
import os
import time
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.security.auth import create_access_token
from src.tools import hitl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def clean_hitl_store():
    """Ensure in-memory and Redis approval store is clean before and after each test."""
    hitl._memory_store.clear()
    if hasattr(hitl, "_store") and getattr(hitl._store, "_redis", None):
        try:
            for k in hitl._store._redis.scan_iter("approval:*"):
                hitl._store._redis.delete(k)
        except Exception:
            pass
    yield
    hitl._memory_store.clear()
    if hasattr(hitl, "_store") and getattr(hitl._store, "_redis", None):
        try:
            for k in hitl._store._redis.scan_iter("approval:*"):
                hitl._store._redis.delete(k)
        except Exception:
            pass


@pytest.fixture
def test_client():
    """Create FastAPI test client with suppressed external telemetry latency."""
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    return TestClient(app)


def test_get_approval_detailed_structured_record(test_client):
    """
    Requirement 1: GET /approvals/{approval_id}
    Returns detailed structured approval record containing all required fields:
    approval_id, tool_name, command, arguments, working_directory, affected_files,
    risk_level, reason, status, tenant_id, user_id, created_at, expires_at.
    """
    tenant_id = "tenant_alpha"
    user_id = "user_alpha_1"
    token = create_access_token(user_id=user_id, role="operator", tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    approval_id = hitl.request_approval(
        user_id=user_id,
        tenant_id=tenant_id,
        session_id="sess_100",
        tool_name="terminal.run",
        command="rm -rf /workspace/test_cache",
        arguments={"command": "rm -rf /workspace/test_cache", "cwd": "/workspace"},
        working_directory="/workspace",
        affected_files=["/workspace/test_cache"],
        risk_level="high",
        reason="Destructive directory removal requires administrative signoff.",
        timeout_seconds=300.0,
    )

    resp = test_client.get(f"/approvals/{approval_id}", headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Verify all 13 canonical structured fields
    assert data["approval_id"] == approval_id
    assert data["tool_name"] == "terminal.run"
    assert data["command"] == "rm -rf /workspace/test_cache"
    assert data["arguments"] == {"command": "rm -rf /workspace/test_cache", "cwd": "/workspace"}
    assert data["working_directory"] == "/workspace"
    assert "/workspace/test_cache" in data["affected_files"]
    assert data["risk_level"] == "high"
    assert "Destructive directory removal" in data["reason"]
    assert data["status"] == "pending"
    assert data["tenant_id"] == tenant_id
    assert data["user_id"] == user_id
    assert isinstance(data["created_at"], float)
    assert data["expires_at"] == pytest.approx(data["created_at"] + 300.0, rel=1e-2)


def test_get_approval_cross_tenant_isolation(test_client):
    """
    Enforces that the requesting user's tenant must match the approval record's tenant.
    Cross-tenant access must return HTTP 403 Forbidden.
    Admin users are exempt.
    """
    approval_id = hitl.request_approval(
        user_id="user_alpha",
        tenant_id="tenant_alpha",
        tool_name="filesystem.write_file",
        command="write_file /etc/hosts",
        arguments={"path": "/etc/hosts", "content": "127.0.0.1 attacker"},
        risk_level="critical",
    )

    # User from a different tenant
    token_bravo = create_access_token(user_id="user_bravo", role="operator", tenant_id="tenant_bravo")
    resp_bravo = test_client.get(f"/approvals/{approval_id}", headers={"Authorization": f"Bearer {token_bravo}"})
    assert resp_bravo.status_code == 403
    assert "does not match approval tenant" in resp_bravo.json()["detail"]

    # Admin user can access
    token_admin = create_access_token(user_id="admin_user", role="admin", tenant_id="tenant_gamma")
    resp_admin = test_client.get(f"/approvals/{approval_id}", headers={"Authorization": f"Bearer {token_admin}"})
    assert resp_admin.status_code == 200
    assert resp_admin.json()["approval_id"] == approval_id


def test_get_approval_not_found(test_client):
    """Missing approval record returns HTTP 404."""
    token = create_access_token(user_id="user1", role="operator", tenant_id="default")
    resp = test_client.get("/approvals/non_existent_approval_id", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_post_approval_approve_flow(test_client):
    """
    Requirement 1: POST /approvals/{approval_id}
    Accepts approval, validates tenant, transitions state cleanly.
    """
    tenant_id = "tenant_finance"
    user_id = "analyst_1"
    token = create_access_token(user_id=user_id, role="operator", tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    approval_id = hitl.request_approval(
        user_id=user_id,
        tenant_id=tenant_id,
        tool_name="terminal.run",
        command="echo 'Approval approved test'",
        arguments={"command": "echo 'Approval approved test'"},
        risk_level="medium",
    )

    post_resp = test_client.post(
        f"/approvals/{approval_id}",
        headers=headers,
        json={"approve": True, "decided_by": user_id},
    )
    assert post_resp.status_code == 200, post_resp.text
    data = post_resp.json()
    assert data["stop_reason"] in ("approved", "executed")

    # Verify status in store
    rec = hitl.get_approval(approval_id)
    assert rec["status"] in ("approved", "executed")


def test_post_approval_reject_flow(test_client):
    """
    Requirement 1: POST /approvals/{approval_id}
    Accepts rejection, transitions state cleanly to 'rejected'.
    """
    tenant_id = "tenant_ops"
    user_id = "dev_1"
    token = create_access_token(user_id=user_id, role="operator", tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    approval_id = hitl.request_approval(
        user_id=user_id,
        tenant_id=tenant_id,
        tool_name="filesystem.delete_file",
        command="rm /tmp/sample.txt",
        arguments={"path": "/tmp/sample.txt"},
        risk_level="high",
    )

    post_resp = test_client.post(
        f"/approvals/{approval_id}",
        headers=headers,
        json={"approve": False, "decided_by": user_id},
    )
    assert post_resp.status_code == 200, post_resp.text
    data = post_resp.json()
    assert data["stop_reason"] == "rejected"

    # Verify status in store
    rec = hitl.get_approval(approval_id)
    assert rec["status"] == "rejected"


def test_post_approval_cross_tenant_rejected(test_client):
    """
    Validates tenant context on POST /approvals/{approval_id}.
    User from another tenant cannot approve or reject.
    """
    approval_id = hitl.request_approval(
        user_id="legit_user",
        tenant_id="tenant_secure",
        tool_name="terminal.run",
        command="reboot",
        arguments={"command": "reboot"},
        risk_level="critical",
    )

    token_intruder = create_access_token(user_id="intruder", role="operator", tenant_id="tenant_other")
    resp = test_client.post(
        f"/approvals/{approval_id}",
        headers={"Authorization": f"Bearer {token_intruder}"},
        json={"approve": True},
    )
    assert resp.status_code == 403
    assert "does not match approval tenant" in resp.json()["detail"]


def test_post_approval_double_approval_prevention(test_client):
    """
    Prevents double-approval or re-decision on an already decided approval.
    Attempting to approve or reject again returns HTTP 400 Bad Request.
    """
    tenant_id = "tenant_corp"
    user_id = "manager_1"
    token = create_access_token(user_id=user_id, role="operator", tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    approval_id = hitl.request_approval(
        user_id=user_id,
        tenant_id=tenant_id,
        tool_name="terminal.run",
        command="ls -l",
        arguments={"command": "ls -l"},
        risk_level="low",
    )

    # First decision (Reject)
    r1 = test_client.post(
        f"/approvals/{approval_id}",
        headers=headers,
        json={"approve": False, "decided_by": user_id},
    )
    assert r1.status_code == 200

    # Second decision attempt must be blocked
    r2 = test_client.post(
        f"/approvals/{approval_id}",
        headers=headers,
        json={"approve": True, "decided_by": user_id},
    )
    assert r2.status_code == 400
    assert "already" in r2.json()["detail"].lower()


def test_post_approval_expired_prevention(test_client):
    """
    Prevents execution of expired approvals.
    Approvals past their expiration timestamp return HTTP 400 Bad Request.
    """
    tenant_id = "tenant_corp"
    user_id = "manager_2"
    token = create_access_token(user_id=user_id, role="operator", tenant_id=tenant_id)
    headers = {"Authorization": f"Bearer {token}"}

    approval_id = hitl.request_approval(
        user_id=user_id,
        tenant_id=tenant_id,
        tool_name="terminal.run",
        command="echo 'will expire'",
        arguments={"command": "echo 'will expire'"},
        risk_level="medium",
        timeout_seconds=0.01,
    )

    # Wait for expiration
    time.sleep(0.05)

    resp = test_client.post(
        f"/approvals/{approval_id}",
        headers=headers,
        json={"approve": True, "decided_by": user_id},
    )
    assert resp.status_code == 400
    assert "expired" in resp.json()["detail"].lower()


def test_list_approvals_tenant_filtering(test_client):
    """
    GET /api/approvals returns list filtered to user's tenant.
    """
    tenant_x = f"tenant_X_{uuid.uuid4().hex[:6]}"
    tenant_y = f"tenant_Y_{uuid.uuid4().hex[:6]}"

    # Create records in two tenants
    hitl.request_approval(
        user_id="u1",
        tenant_id=tenant_x,
        tool_name="tool_1",
        command="cmd1",
    )
    hitl.request_approval(
        user_id="u2",
        tenant_id=tenant_y,
        tool_name="tool_2",
        command="cmd2",
    )

    token_x = create_access_token(user_id="u1", role="operator", tenant_id=tenant_x)
    resp_x = test_client.get("/api/approvals", headers={"Authorization": f"Bearer {token_x}"})
    assert resp_x.status_code == 200
    data_x = resp_x.json()["approvals"]
    assert len(data_x) == 1
    assert data_x[0]["tenant_id"] == tenant_x

    # Admin sees records across all tenants
    token_admin = create_access_token(user_id="admin", role="admin", tenant_id=tenant_x)
    resp_admin = test_client.get("/api/approvals", headers={"Authorization": f"Bearer {token_admin}"})
    assert resp_admin.status_code == 200
    admin_approvals = resp_admin.json()["approvals"]
    admin_tenant_ids = {a["tenant_id"] for a in admin_approvals}
    assert tenant_x in admin_tenant_ids
    assert tenant_y in admin_tenant_ids


def test_ui_html_approval_specifications():
    """
    Requirement 2: Verify UI HTML files include all required components and states:
    - Requested Operation
    - Exact Command
    - Working Directory
    - Affected Files / Scope
    - Risk Level & Explanation banner
    - Interactive Approve & Reject buttons
    - Clear 7 status states: Awaiting approval, Approved, Rejected, Expired, Executing, Completed, Failed.
    """
    index_path = PROJECT_ROOT / "src" / "web" / "index.html"
    chat_path = PROJECT_ROOT / "src" / "web" / "chat.html"

    assert index_path.exists(), f"index.html missing at {index_path}"
    assert chat_path.exists(), f"chat.html missing at {chat_path}"

    with open(index_path, "r", encoding="utf-8") as f:
        index_content = f.read()

    with open(chat_path, "r", encoding="utf-8") as f:
        chat_content = f.read()

    # Verify required UI labels & fields in index.html
    assert "Requested Operation" in index_content
    assert "Exact Command" in index_content
    assert "Working Directory" in index_content
    assert "Affected Files / Scope" in index_content
    assert "risk-banner" in index_content
    assert "handleModalAction(true)" in index_content or "handleCardApproval" in index_content

    # Verify 7 distinct status states in index.html
    for status_name in [
        "status-awaiting",
        "status-approved",
        "status-rejected",
        "status-expired",
        "status-executing",
        "status-completed",
        "status-failed",
    ]:
        assert status_name in index_content, f"Status class '{status_name}' missing from index.html"

    # Verify required UI elements in chat.html
    assert "hitl-box" in chat_content
    assert "Exact Command" in chat_content
    assert "Working Directory" in chat_content or "hitl-wd-" in chat_content
    assert "Affected Files" in chat_content or "Scope:" in chat_content
    assert "chat-approval-modal" in chat_content
    assert "openChatApprovalModal" in chat_content

    # Verify 7 distinct status states in chat.html
    for status_name in [
        "status-awaiting",
        "status-approved",
        "status-rejected",
        "status-expired",
        "status-executing",
        "status-completed",
        "status-failed",
    ]:
        assert status_name in chat_content, f"Status class '{status_name}' missing from chat.html"
