"""
Anthropic MCP Server for GitHub API Operations.
Thin MCP adapter over src.tools.core.github.
"""
from __future__ import annotations

from typing import Any
from mcp.server.mcpserver import MCPServer
from src.tools.core import github

app = MCPServer("github")


@app.tool(name="github.search_repositories")
def search_repositories(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search for public repositories on GitHub."""
    res = github.search_repositories(query, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


@app.tool(name="github.get_repository")
def get_repository(owner: str = "", repo: str = "") -> dict[str, Any]:
    """Retrieve details for a specific GitHub repository."""
    res = github.get_repository(owner, repo)
    if not res["success"]:
        err_dict = {"error": res["error"]["message"]}
        if res.get("data"):
            err_dict.update(res["data"])
        return err_dict
    return res["data"]


@app.tool(name="github.list_issues")
def list_issues(owner: str = "", repo: str = "", state: str = "open", limit: int = 5) -> list[dict[str, Any]]:
    """List issues for a given repository."""
    res = github.list_issues(owner, repo, state=state, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


@app.tool(name="github.list_user_repositories")
def list_user_repositories(username: str = "", limit: int = 10) -> list[dict[str, Any]]:
    """List public repositories for a specific user."""
    res = github.list_user_repositories(username, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


if __name__ == "__main__":
    app.run(transport="stdio")
