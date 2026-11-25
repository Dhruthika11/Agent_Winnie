"""
Built-in local tool adapter for Hermes Agent Platform.
Delegates directly to canonical core implementations in src.tools.core.
Maintains full backward compatibility for fallbacks and unit test harnesses.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.tools.core import filesystem, github, terminal, web

logger = logging.getLogger("hermes.tools.builtin")


def _resolve_workspace_path(rel_path: str) -> Path:
    """Workspace resolution jail delegate."""
    return filesystem.resolve_workspace_path(rel_path)


def web_scraper_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo delegate."""
    res = web.search(query, max_results=max_results)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return [
        {
            "title": r.get("title", ""),
            "href": r.get("url", ""),
            "body": r.get("snippet", ""),
        }
        for r in res["data"]
    ]


def web_scraper_read_url(url: str) -> str:
    """Fetch URL text delegate."""
    res = web.read_url(url)
    if not res["success"]:
        return f"Failed to fetch {url}: {res['error']['message']}"
    return str(res["data"])


def terminal_delete_file(path: str) -> str:
    """Delete file delegate."""
    res = terminal.delete_file(path)
    if not res["success"]:
        code = res["error"]["code"]
        if code == "PERMISSION_DENIED":
            return f"Access Denied: {res['error']['message']}"
        elif code == "NOT_FOUND":
            return f"File not found: {path}"
        return f"Failed to delete {path}: {res['error']['message']}"
    return str(res["data"])


def filesystem_read_file(path: str) -> str:
    """Read file delegate."""
    res = filesystem.read_file(path)
    if not res["success"]:
        return f"Error: {res['error']['message']}"
    return str(res["data"])


def filesystem_write_file(path: str, content: str) -> str:
    """Write file delegate."""
    res = filesystem.write_file(path, content)
    if not res["success"]:
        return f"Error: {res['error']['message']}"
    return str(res["data"])


def filesystem_list_directory(path: str = ".") -> list[dict[str, Any]]:
    """List directory delegate."""
    res = filesystem.list_directory(path)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


def github_search_repositories(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Search GitHub repositories delegate."""
    res = github.search_repositories(query, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


def github_get_repository(owner: str, repo: str) -> dict[str, Any]:
    """Get GitHub repository delegate."""
    res = github.get_repository(owner, repo)
    if not res["success"]:
        err_dict = {"error": res["error"]["message"]}
        if res.get("data"):
            err_dict.update(res["data"])
        return err_dict
    return res["data"]


def github_list_issues(owner: str, repo: str, state: str = "open", limit: int = 5) -> list[dict[str, Any]]:
    """List repository issues delegate."""
    res = github.list_issues(owner, repo, state=state, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]


def github_list_user_repositories(username: str, limit: int = 10) -> list[dict[str, Any]]:
    """List user repositories delegate."""
    res = github.list_user_repositories(username, limit=limit)
    if not res["success"]:
        return [{"error": res["error"]["message"]}]
    return res["data"]
