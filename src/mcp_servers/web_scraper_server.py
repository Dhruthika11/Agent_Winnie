"""
Anthropic MCP Server for Web Scraping & Search.
Thin MCP adapter over src.tools.core.web.
"""
from __future__ import annotations

from typing import Any
from mcp.server.mcpserver import MCPServer
from src.tools.core import web

app = MCPServer("web_scraper")


@app.tool(name="web_scraper.search")
def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search DuckDuckGo and return top result snippets with URLs."""
    res = web.search(query, max_results=max_results)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


@app.tool(name="web_scraper.read_url")
def read_url(url: str) -> str:
    """Fetch and extract the main text content of a given URL."""
    res = web.read_url(url)
    if not res["success"]:
        return f"[Failed to read webpage {url}: {res['error']['message']}]"
    return str(res["data"])


if __name__ == "__main__":
    app.run(transport="stdio")
