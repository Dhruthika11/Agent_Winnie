"""
Ensures destructive / high-stakes tool calls are always routed through
human approval, and that unapproved high-risk calls are blocked at
execution time rather than silently allowed.
"""
import pytest

from src.agent.graph import check_safety, execute_tools
from src.tools import hitl


def _state_with_calls(calls):
    return {
        "user_id": "u1",
        "session_id": "s1",
        "pending_tool_calls": calls,
        "awaiting_approval": False,
        "approval_id": None,
    }


@pytest.mark.parametrize(
    "tool_name",
    ["terminal.delete_file", "payments.wire_transfer", "db.drop_table"],
)
def test_high_risk_tools_require_approval(tool_name):
    calls = [{"id": "1", "name": tool_name, "arguments": {}, "risk_level": hitl.classify_risk(tool_name, {})}]
    state = _state_with_calls(calls)
    result = check_safety(state)
    assert result["awaiting_approval"] is True
    assert result["approval_id"] is not None


def test_low_risk_tools_do_not_require_approval():
    calls = [{"id": "1", "name": "web_scraper.search", "arguments": {}, "risk_level": "low"}]
    state = _state_with_calls(calls)
    result = check_safety(state)
    assert result["awaiting_approval"] is False


def test_unapproved_high_risk_call_is_blocked_at_execution():
    calls = [{"id": "1", "name": "terminal.delete_file", "arguments": {"path": "/x"}, "risk_level": "high"}]
    state = _state_with_calls(calls)
    state["approval_id"] = "nonexistent-approval"
    result = execute_tools(state)
    tool_msg = result["messages"][0]
    assert "requires human approval" in tool_msg["content"]
