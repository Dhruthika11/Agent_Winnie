"""
Comprehensive Unit Test Suite for Hermes Architectural Pillars:
1. Centralized Tool Execution Engine (Idempotency, RBAC, Policy, HITL, Circuit Breaker, Validators)
2. Durable Execution & Crash Recovery (States, Checkpointing, Stale Reconciliation, Triage)
3. Multi-Tenant Isolation & Role-Based Access Control (RBAC Engine, Resource Isolation)
"""
from __future__ import annotations

import datetime
import uuid
import pytest
from unittest.mock import MagicMock, patch

from src.agent.durable import (
    DurableExecutionRecord,
    DurableExecutionStore,
    ExecutionLifecycleState,
    RecoveryDisposition,
    durable_store,
)
from src.agent.recovery import recovery_manager
from src.security.rbac import (
    Permission,
    Role,
    ROLE_PERMISSIONS,
    rbac_engine,
)
from src.tools.circuit_breaker import tool_circuit_breaker
from src.tools.execution_engine import (
    ToolExecutionEngine,
    ToolExecutionRequest,
    ToolExecutionResult,
    tool_execution_engine,
)
from src.tools import hitl
from src.db.chat_history import chat_history_db


# =========================================================================== #
# Pillar A: Centralized Tool Execution Engine
# =========================================================================== #
class TestCentralizedToolExecutionEngine:
    def test_idempotency_key_replay(self):
        """Calling with identical idempotency key returns cached result without re-execution."""
        engine = ToolExecutionEngine()
        key = f"idem_{uuid.uuid4().hex[:8]}"

        valid_search_output = {
            "success": True,
            "data": [
                {
                    "title": "Test Title",
                    "url": "https://example.com/test",
                    "snippet": "Test search snippet for test query",
                }
            ],
        }

        with patch("src.tools.mcp_client.mcp_registry.call_tool", return_value=valid_search_output) as mock_tool:
            req1 = ToolExecutionRequest(
                tool_name="web_scraper.search",
                arguments={"query": "test query"},
                user_role="viewer",
                idempotency_key=key,
            )
            res1 = engine.execute(req1)
            assert res1.is_success is True
            assert res1.idempotent_replay is False
            assert mock_tool.call_count == 1

            # Second call with the same idempotency key
            req2 = ToolExecutionRequest(
                tool_name="web_scraper.search",
                arguments={"query": "test query"},
                user_role="viewer",
                idempotency_key=key,
            )
            res2 = engine.execute(req2)
            assert res2.is_success is True
            assert res2.idempotent_replay is True
            # Underlying tool should NOT be called a second time
            assert mock_tool.call_count == 1

    def test_universal_security_policy_denial(self):
        """Dangerous shell exploit (fork bomb) is blocked by Security Policy universally."""
        engine = ToolExecutionEngine()
        req = ToolExecutionRequest(
            tool_name="terminal.run",
            arguments={"command": ":(){ :|:& };:"},
            user_role="admin",  # Even admin cannot run a fork bomb
        )
        res = engine.execute(req)
        assert res.status == "denied"
        assert "Action Denied by Security Policy" in res.error
        assert res.validation_status == "failed"

    def test_rbac_authorization_denial_for_viewer(self):
        """Viewer role attempting to execute sandboxed code is blocked by RBAC."""
        engine = ToolExecutionEngine()
        req = ToolExecutionRequest(
            tool_name="run_sandboxed_code",
            arguments={"code": "import os; print(os.name)", "language": "python"},
            user_role="viewer",  # Viewer lacks tool:sandbox_execute
        )
        res = engine.execute(req)
        assert res.status == "denied"
        assert "Action Denied by RBAC Policy" in res.error
        assert "viewer" in res.error
        assert res.validation_status == "failed"

    def test_rbac_authorization_allowed_for_operator(self):
        """Operator role is authorized to execute sandboxed code."""
        engine = ToolExecutionEngine()
        with patch("src.tools.sandbox.sandbox.execute", return_value="output: ok"):
            req = ToolExecutionRequest(
                tool_name="run_sandboxed_code",
                arguments={"code": "print(42)", "language": "python"},
                user_role="operator",
            )
            res = engine.execute(req)
            assert res.status == "success"
            assert "output: ok" in str(res.output)

    def test_hitl_gating_for_high_risk_actions(self):
        """High-risk action without prior approval returns hitl_required and creates approval request."""
        engine = ToolExecutionEngine()
        # High risk action requiring approval
        req = ToolExecutionRequest(
            tool_name="web_scraper.read_url",
            arguments={"url": "https://internal-admin.local/sensitive"},
            user_role="operator",
            risk_level="high",
        )
        res = engine.execute(req)
        assert res.status == "hitl_required"
        assert res.approval_id is not None
        assert "requires human approval" in str(res.output)

    def test_circuit_breaker_trip_and_fast_fail(self):
        """Tripped circuit breaker returns circuit_broken status immediately."""
        engine = ToolExecutionEngine()
        server_key = "failing_mcp"
        tool_circuit_breaker.trip(server_key, reason="Test trip")

        try:
            req = ToolExecutionRequest(
                tool_name=f"{server_key}.action",
                arguments={},
                user_role="admin",
            )
            res = engine.execute(req)
            assert res.status == "circuit_broken"
            assert res.circuit_tripped is True
            assert "is OPEN" in res.error
        finally:
            tool_circuit_breaker.reset(server_key)

    def test_bounded_retries_on_transient_failure(self):
        """Tool execution retries up to max_retries on transient failure."""
        engine = ToolExecutionEngine()
        valid_recovered_output = {
            "success": True,
            "data": [
                {
                    "title": "Recovered Title",
                    "url": "https://example.com/recovered",
                    "snippet": "Recovered search snippet",
                }
            ],
        }
        with patch(
            "src.tools.mcp_client.mcp_registry.call_tool",
            side_effect=[RuntimeError("Temporary drop"), valid_recovered_output],
        ) as mock_tool:
            req = ToolExecutionRequest(
                tool_name="web_scraper.search",
                arguments={"query": "test retry"},
                user_role="viewer",
                max_retries=2,
            )
            res = engine.execute(req)
            assert res.status == "success"
            assert res.output == valid_recovered_output
            assert mock_tool.call_count == 2


# =========================================================================== #
# Pillar B: Durable Execution and Crash Recovery
# =========================================================================== #
class TestDurableExecutionAndCrashRecovery:
    def test_lifecycle_state_transitions(self):
        """DurableExecutionStore correctly records state transitions and updates record."""
        store = DurableExecutionStore()
        exec_id = f"exec_test_{uuid.uuid4().hex[:8]}"

        # 1. REQUEST_RECEIVED
        chk1 = store.record_transition(
            execution_id=exec_id,
            state_from=None,
            state_to=ExecutionLifecycleState.REQUEST_RECEIVED.value,
            state_payload={"query": "hello world"},
            user_id="user_1",
            session_id="session_1",
            tenant_id="tenant_1",
        )
        assert chk1.state_to == ExecutionLifecycleState.REQUEST_RECEIVED.value
        rec1 = store.get_execution(exec_id)
        assert rec1.status == ExecutionLifecycleState.REQUEST_RECEIVED
        assert rec1.current_step == 1

        # 2. AGENT_RUNNING
        chk2 = store.record_transition(
            execution_id=exec_id,
            state_from=ExecutionLifecycleState.REQUEST_RECEIVED.value,
            state_to=ExecutionLifecycleState.AGENT_RUNNING.value,
            state_payload={"iteration": 1},
        )
        assert chk2.state_to == ExecutionLifecycleState.AGENT_RUNNING.value
        rec2 = store.get_execution(exec_id)
        assert rec2.status == ExecutionLifecycleState.AGENT_RUNNING
        assert rec2.current_step == 2

        # 3. WAITING_FOR_TOOL
        chk3 = store.record_transition(
            execution_id=exec_id,
            state_from=ExecutionLifecycleState.AGENT_RUNNING.value,
            state_to=ExecutionLifecycleState.WAITING_FOR_TOOL.value,
            state_payload={"pending_tools": ["search"]},
        )
        assert chk3.state_to == ExecutionLifecycleState.WAITING_FOR_TOOL.value
        rec3 = store.get_execution(exec_id)
        assert rec3.status == ExecutionLifecycleState.WAITING_FOR_TOOL

        # 4. TOOL_EXECUTED
        chk4 = store.record_transition(
            execution_id=exec_id,
            state_from=ExecutionLifecycleState.WAITING_FOR_TOOL.value,
            state_to=ExecutionLifecycleState.TOOL_EXECUTED.value,
            state_payload={"result": "found info"},
        )
        assert chk4.state_to == ExecutionLifecycleState.TOOL_EXECUTED.value
        rec4 = store.get_execution(exec_id)
        assert rec4.status == ExecutionLifecycleState.TOOL_EXECUTED

        # 5. COMPLETED
        chk5 = store.record_transition(
            execution_id=exec_id,
            state_from=ExecutionLifecycleState.TOOL_EXECUTED.value,
            state_to=ExecutionLifecycleState.COMPLETED.value,
            state_payload={"final_reply": "Here is your answer."},
        )
        assert chk5.state_to == ExecutionLifecycleState.COMPLETED.value
        rec5 = store.get_execution(exec_id)
        assert rec5.status == ExecutionLifecycleState.COMPLETED
        assert rec5.current_step == 5

        # Checkpoints
        checkpoints = store.get_checkpoints(exec_id)
        assert len(checkpoints) == 5
        assert checkpoints[0].state_to == ExecutionLifecycleState.REQUEST_RECEIVED.value
        assert checkpoints[-1].state_to == ExecutionLifecycleState.COMPLETED.value

    def test_recovery_disposition_triage(self):
        """Durable execution correctly triages interrupted states into RETRY_FAILED, RECOVER_UNKNOWN, RESUME_COMPLETED."""
        store = DurableExecutionStore()

        # Case 1: Definitively failed state -> RETRY_FAILED
        rec_failed = DurableExecutionRecord(
            execution_id="exec_f",
            session_id="s",
            user_id="u",
            status=ExecutionLifecycleState.FAILED,
            error="Connection timeout",
        )
        assert store.determine_recovery_disposition(rec_failed) == RecoveryDisposition.RETRY_FAILED

        # Case 2: Interrupted during WAITING_FOR_TOOL -> RECOVER_UNKNOWN
        rec_tool = DurableExecutionRecord(
            execution_id="exec_t",
            session_id="s",
            user_id="u",
            status=ExecutionLifecycleState.WAITING_FOR_TOOL,
        )
        assert store.determine_recovery_disposition(rec_tool) == RecoveryDisposition.RECOVER_UNKNOWN

        # Case 3: Interrupted after TOOL_EXECUTED -> RESUME_COMPLETED
        rec_done_tool = DurableExecutionRecord(
            execution_id="exec_dt",
            session_id="s",
            user_id="u",
            status=ExecutionLifecycleState.TOOL_EXECUTED,
        )
        assert store.determine_recovery_disposition(rec_done_tool) == RecoveryDisposition.RESUME_COMPLETED

    def test_stale_execution_reconciliation(self):
        """RecoveryManager identifies stale executions past heartbeat threshold and triages them."""
        exec_id = f"exec_stale_{uuid.uuid4().hex[:8]}"

        # Record an active execution with an artificially old heartbeat
        rec = durable_store.record_transition(
            execution_id=exec_id,
            state_from=None,
            state_to=ExecutionLifecycleState.AGENT_RUNNING.value,
            state_payload={"user_msg": "stalled task"},
            user_id="user_stale",
            session_id="session_stale",
            tenant_id="tenant_test",
        )

        # Set last_heartbeat to 120 seconds in the past
        past_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
        with durable_store._lock:
            durable_store._mem_executions[exec_id]["last_heartbeat"] = past_time.isoformat()

        # Run reconciliation with a 60-second threshold
        stale_list = recovery_manager.reconcile_stale_executions(
            stale_threshold_seconds=60.0,
            tenant_id="tenant_test",
        )

        assert any(item["execution_id"] == exec_id for item in stale_list)
        reconciled_rec = durable_store.get_execution(exec_id)
        assert reconciled_rec.status == ExecutionLifecycleState.INTERRUPTED
        assert "Stale execution detected" in reconciled_rec.error

    def test_cancel_execution(self):
        """RecoveryManager allows operator to cancel an execution."""
        exec_id = f"exec_cancel_{uuid.uuid4().hex[:8]}"
        durable_store.record_transition(
            execution_id=exec_id,
            state_from=None,
            state_to=ExecutionLifecycleState.AGENT_RUNNING.value,
            state_payload={},
        )
        success = recovery_manager.cancel_execution(exec_id, reason="Admin manually stopped")
        assert success is True
        rec = durable_store.get_execution(exec_id)
        assert rec.status == ExecutionLifecycleState.FAILED
        assert rec.error == "Admin manually stopped"


# =========================================================================== #
# Pillar C: Multi-Tenant Security and RBAC
# =========================================================================== #
class TestMultiTenantSecurityAndRBAC:
    def test_role_permissions_hierarchy(self):
        """Viewer, Operator, and Admin follow strictly enforced least-privilege matrix."""
        assert rbac_engine.has_permission("viewer", Permission.TOOL_SEARCH) is True
        assert rbac_engine.has_permission("viewer", Permission.TOOL_SANDBOX_EXECUTE) is False
        assert rbac_engine.has_permission("viewer", Permission.RECOVERY_MANAGE) is False

        assert rbac_engine.has_permission("operator", Permission.TOOL_SEARCH) is True
        assert rbac_engine.has_permission("operator", Permission.TOOL_SANDBOX_EXECUTE) is True
        assert rbac_engine.has_permission("operator", Permission.RECOVERY_MANAGE) is False

        assert rbac_engine.has_permission("admin", Permission.TOOL_SANDBOX_EXECUTE) is True
        assert rbac_engine.has_permission("admin", Permission.RECOVERY_MANAGE) is True
        assert rbac_engine.has_permission("admin", Permission.TENANT_AUDIT_READ) is True

    def test_tenant_cross_access_blocked(self):
        """Cross-tenant resource access is strictly forbidden, even for user owning matching user_id."""
        # Tenant A user trying to access Tenant B resource
        allowed = rbac_engine.authorize_resource_access(
            role="operator",
            resource_owner_id="user_123",
            requesting_user_id="user_123",
            resource_tenant_id="tenant_alpha",
            requesting_tenant_id="tenant_beta",
        )
        assert allowed is False

    def test_same_tenant_ownership_access_allowed(self):
        """Same tenant and matching user_id allows access."""
        allowed = rbac_engine.authorize_resource_access(
            role="viewer",
            resource_owner_id="user_123",
            requesting_user_id="user_123",
            resource_tenant_id="tenant_alpha",
            requesting_tenant_id="tenant_alpha",
        )
        assert allowed is True

    def test_same_tenant_unauthorized_user_access_blocked(self):
        """Same tenant but different user_id is blocked for non-admin."""
        allowed = rbac_engine.authorize_resource_access(
            role="operator",
            resource_owner_id="user_123",
            requesting_user_id="user_456",
            resource_tenant_id="tenant_alpha",
            requesting_tenant_id="tenant_alpha",
        )
        assert allowed is False

    def test_admin_intra_tenant_access_allowed(self):
        """Admin role can access resources within their tenant regardless of ownership."""
        allowed = rbac_engine.authorize_resource_access(
            role="admin",
            resource_owner_id="user_123",
            requesting_user_id="admin_user",
            resource_tenant_id="tenant_alpha",
            requesting_tenant_id="tenant_alpha",
        )
        assert allowed is True

    def test_chat_history_tenant_scoping(self):
        """Chat history list_conversations respects tenant scoping."""
        user_id = f"user_{uuid.uuid4().hex[:6]}"
        s1 = f"sess_alpha_{uuid.uuid4().hex[:6]}"
        s2 = f"sess_beta_{uuid.uuid4().hex[:6]}"

        chat_history_db.get_or_create_conversation(s1, user_id=user_id, tenant_id="tenant_alpha", title="Alpha Conv")
        chat_history_db.get_or_create_conversation(s2, user_id=user_id, tenant_id="tenant_beta", title="Beta Conv")

        # List scoped to tenant_alpha
        alpha_convs = chat_history_db.list_conversations(user_id=user_id, tenant_id="tenant_alpha")
        assert len(alpha_convs) == 1
        assert alpha_convs[0]["session_id"] == s1
        assert alpha_convs[0]["tenant_id"] == "tenant_alpha"

        # List scoped to tenant_beta
        beta_convs = chat_history_db.list_conversations(user_id=user_id, tenant_id="tenant_beta")
        assert len(beta_convs) == 1
        assert beta_convs[0]["session_id"] == s2
        assert beta_convs[0]["tenant_id"] == "tenant_beta"
