"""
Anthropic MCP Server for Filesystem Operations.
Thin MCP adapter over src.tools.core.filesystem.
"""
from __future__ import annotations

from typing import Any
from mcp.server.mcpserver import MCPServer
from src.tools.core import filesystem

app = MCPServer("filesystem")


@app.tool(name="filesystem.read_file")
def read_file(path: str) -> str:
    """Read the complete UTF-8 content of a file within the workspace."""
    res = filesystem.read_file(path)
    if not res["success"]:
        raise RuntimeError(res["error"]["message"])
    return str(res["data"])


@app.tool(name="filesystem.write_file")
def write_file(path: str, content: str) -> str:
    """Write text content to a file in the workspace."""
    res = filesystem.write_file(path, content)
    if not res["success"]:
        raise RuntimeError(res["error"]["message"])
    return str(res["data"])


@app.tool(name="filesystem.list_directory")
def list_directory(path: str = ".") -> list[dict[str, Any]]:
    """List contents of a directory in the workspace with metadata."""
    res = filesystem.list_directory(path)
    if not res["success"]:
        raise RuntimeError(res["error"]["message"])
    return res["data"]


if __name__ == "__main__":
    app.run(transport="stdio")
