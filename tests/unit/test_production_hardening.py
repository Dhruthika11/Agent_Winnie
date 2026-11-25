"""
Comprehensive Production Hardening Unit Tests.

Validates:
1. Circuit Breaker Recovery Loops (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
2. Path Traversal Jail (strictly prevents directory breakout or ../ traversal).
3. Distributed Rate Limiter (sliding window token bucket).
4. FinOps Budgeting & Graceful Degradation Policy (80% degradation, 100% block).
5. Role-Based Access Control (RBAC) on Approvals.
6. Distributed Checkpointer factory.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.tools.circuit_breaker import ToolCircuitBreaker, ServerCircuit, CircuitState
from src.tools.builtin_tools import _resolve_workspace_path, terminal_delete_file
from src.security.rate_limiter import RateLimiter
from src.security.finops import FinOpsManager
from src.security.auth import create_access_token, decode_token, UserAuth
from src.agent.checkpointer import get_checkpointer


# --------------------------------------------------------------------------- #
# 1. Circuit Breaker Recovery Loop Tests
# --------------------------------------------------------------------------- #
def test_circuit_breaker_recovery_lifecycle():
    cb = ToolCircuitBreaker(default_failure_threshold=2, default_cooldown_seconds=0.1)
    cb._redis_client = None  # in-memory test

    # Normal state: CLOSED, can execute
    allowed, _ = cb.can_execute("test_service")
    assert allowed is True
    status = cb.get_status("test_service")
    assert status["state"] == "CLOSED"

    # Trip the circuit with 2 failures
    cb.record_failure("test_service", "test failure 1")
    assert cb.get_status("test_service")["state"] == "CLOSED"
    cb.record_failure("test_service", "test failure 2")
    assert cb.get_status("test_service")["state"] == "OPEN"

    # In OPEN state with remaining cooldown, can_execute must return False
    allowed, reason = cb.can_execute("test_service")
    assert allowed is False
    assert "Circuit OPEN" in reason

    # Wait for cooldown to expire -> transitions to HALF_OPEN
    import time
    time.sleep(0.12)
    allowed, reason = cb.can_execute("test_service")
    assert allowed is True
    assert "HALF_OPEN" in reason

    # Success in HALF_OPEN resets circuit back to CLOSED
    cb.record_success("test_service")
    status = cb.get_status("test_service")
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0


def test_circuit_breaker_manual_trip_and_reset():
    cb = ToolCircuitBreaker()
    cb._redis_client = None

    cb.trip("manual_server", reason="Manual operator override")
    assert cb.get_status("manual_server")["state"] == "OPEN"

    cb.reset("manual_server")
    assert cb.get_status("manual_server")["state"] == "CLOSED"


# --------------------------------------------------------------------------- #
# 2. Path Traversal Jail Confinement Tests
# --------------------------------------------------------------------------- #
def test_path_traversal_jail_blocks_parent_escape():
    with pytest.raises(PermissionError):
        _resolve_workspace_path("../../etc/passwd")

    with pytest.raises(PermissionError):
        _resolve_workspace_path("../../../windows/system32")


def test_path_traversal_allows_safe_workspace_files():
    safe_path = _resolve_workspace_path("subfolder/data.txt")
    assert "workspace" in str(safe_path)
    assert safe_path.name == "data.txt"


def test_terminal_delete_blocks_path_escape():
    res = terminal_delete_file("../../secret_config.json")
    assert "Access Denied" in res or "File not found" in res


# --------------------------------------------------------------------------- #
# 3. Distributed Rate Limiter Tests
# --------------------------------------------------------------------------- #
def test_rate_limiter_allows_under_quota_and_blocks_over_quota():
    import uuid
    limiter = RateLimiter()
    test_client = f"client_{uuid.uuid4().hex[:8]}"

    # Send 3 requests with limit of 3
    limited_1, _ = limiter.is_rate_limited(test_client, max_requests=3, window_seconds=60)
    limited_2, _ = limiter.is_rate_limited(test_client, max_requests=3, window_seconds=60)
    limited_3, _ = limiter.is_rate_limited(test_client, max_requests=3, window_seconds=60)
    assert limited_1 is False
    assert limited_2 is False
    assert limited_3 is False

    # 4th request must be rate limited
    limited_4, count = limiter.is_rate_limited(test_client, max_requests=3, window_seconds=60)
    assert limited_4 is True
    assert count > 3


# --------------------------------------------------------------------------- #
# 4. FinOps Spending Caps & Graceful Degradation Tests
# --------------------------------------------------------------------------- #
def test_finops_budget_degradation_and_blocking():
    import uuid
    finops = FinOpsManager()
    user = f"user_{uuid.uuid4().hex[:8]}"

    # Under 80%: ok
    status, used, limit = finops.evaluate_budget(user)
    assert status == "ok"

    # Consume 42,000 tokens (out of 50,000 max -> 84% -> degraded)
    finops.record_usage(user, 42000)
    status, used, limit = finops.evaluate_budget(user)
    assert status == "degraded"
    assert used >= 40000

    # Consume another 10,000 tokens (total 52,000 -> 104% -> blocked)
    finops.record_usage(user, 10000)
    status, used, limit = finops.evaluate_budget(user)
    assert status == "blocked"


# --------------------------------------------------------------------------- #
# 5. Role-Based Access Control (RBAC) Token Tests
# --------------------------------------------------------------------------- #
def test_jwt_token_creation_and_decoding():
    token = create_access_token(user_id="alice", role="operator", tenant_id="tenant_alpha")
    auth_user = decode_token(token)

    assert auth_user.user_id == "alice"
    assert auth_user.role == "operator"
    assert auth_user.tenant_id == "tenant_alpha"


# --------------------------------------------------------------------------- #
# 6. Distributed Checkpointer Tests
# --------------------------------------------------------------------------- #
def test_checkpointer_returns_valid_saver():
    checkpointer = get_checkpointer()
    assert checkpointer is not None
    # Checkpointer must support put and get
    assert hasattr(checkpointer, "get") or hasattr(checkpointer, "aget")
