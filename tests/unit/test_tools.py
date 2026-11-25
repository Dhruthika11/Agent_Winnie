"""Unit tests for HITL risk classification and the MCP registry."""
from src.tools import hitl
from src.tools.mcp_client import MCPRegistry


def test_low_risk_read_tool():
    assert hitl.classify_risk("web_scraper.search", {"query": "weather"}) == "low"


def test_high_risk_delete_tool():
    assert hitl.classify_risk("terminal.delete_file", {"path": "/data/a.txt"}) == "high"


def test_high_risk_financial_tool():
    assert hitl.classify_risk("payments.wire_transfer", {"amount": 500}) == "high"


def test_medium_risk_write_tool():
    assert hitl.classify_risk("crm.update_contact", {"id": 1}) == "medium"


def test_heuristic_flags_destructive_shell_argument():
    risk = hitl.classify_risk("terminal.run", {"command": "rm -rf /data"})
    assert risk == "high"


def test_approval_lifecycle_memory_backend():
    approval_id = hitl.request_approval(
        user_id="u1", session_id="s1", tool_calls=[{"id": "1", "name": "terminal.delete_file"}]
    )
    assert hitl.is_approved(approval_id) is False
    hitl.decide(approval_id, approve=True, decided_by="alice")
    assert hitl.is_approved(approval_id) is True


def test_mcp_registry_always_exposes_sandbox_tool(tmp_path):
    registry = MCPRegistry(config_dir=tmp_path)  # empty dir
    names = [t["function"]["name"] for t in registry.tool_schemas()]
    assert "run_sandboxed_code" in names
