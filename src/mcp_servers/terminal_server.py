"""
Anthropic MCP Server for Terminal & File Deletion Operations.
Thin MCP adapter over src.tools.core.terminal with structural security policy enforcement.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from src.agent.tool_policy import tool_policy, PolicyDecision
from src.tools.core import terminal
from src.tools import hitl

app = MCPServer("terminal")


@app.tool(name="terminal.run")
def run_command(command: str, approval_id: str | None = None) -> str:
    """Run a shell command inside the isolated workspace directory."""
    eval_res = tool_policy.evaluate("terminal.run", {"command": command})
    if eval_res.decision == PolicyDecision.DENY:
        raise PermissionError(f"Security Policy DENIED command: {eval_res.reason}")
    if eval_res.decision == PolicyDecision.REQUIRE_APPROVAL:
        if not (approval_id and hitl.is_approved(approval_id)):
            raise PermissionError(f"Human-in-the-Loop approval required: {eval_res.reason}")

    is_appr = bool(approval_id and hitl.is_approved(approval_id))
    res = terminal.run_command(command, approved=is_appr, approval_id=approval_id)
    data = res.get("data") or {}
    if not res.get("success"):
        err = res.get("error") or {}
        return f"Error ({err.get('code', 'EXECUTION_FAILED')}): {err.get('message', 'Failed')}"
    return f"Return Code: {data.get('returncode', -1)}\nSTDOUT:\n{data.get('stdout', '')}\nSTDERR:\n{data.get('stderr', '')}"


@app.tool(name="terminal.delete_file")
def delete_file(path: str, approval_id: str | None = None) -> str:
    """Delete a file permanently from the workspace."""
    eval_res = tool_policy.evaluate("terminal.delete_file", {"path": path})
    if eval_res.decision == PolicyDecision.DENY:
        raise PermissionError(f"Security Policy DENIED file deletion: {eval_res.reason}")
    if eval_res.decision == PolicyDecision.REQUIRE_APPROVAL:
        if not (approval_id and hitl.is_approved(approval_id)):
            raise PermissionError(f"Human-in-the-Loop approval required: {eval_res.reason}")

    is_appr = bool(approval_id and hitl.is_approved(approval_id))
    res = terminal.delete_file(path, approved=is_appr, approval_id=approval_id)
    if not res["success"]:
        raise RuntimeError(res["error"]["message"])
    return str(res["data"])


if __name__ == "__main__":
    app.run(transport="stdio")
