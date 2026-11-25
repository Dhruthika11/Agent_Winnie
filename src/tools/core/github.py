"""
Canonical Core Logic for GitHub API Operations.
"""
from __future__ import annotations

import os
from typing import Any
import httpx

from src.tools.core.contract import error_result, success_result

GITHUB_BASE_URL = "https://api.github.com"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "hermes-agent-platform-mcp/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"token {token.strip()}"
    return headers


def parse_repo_identifier(owner: str, repo: str) -> tuple[str, str]:
    """Parse and normalize owner and repo from various user/model input formats."""
    # If URL was passed in repo or owner
    raw = repo or owner
    if raw and ("github.com/" in raw or raw.startswith("http")):
        parts = [p for p in raw.split("github.com/")[-1].split("?")[0].strip("/").split("/") if p]
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()

    if "/" in repo and not owner:
        p0, p1 = repo.split("/", 1)
        return p0.strip(), p1.strip()
    if "/" in owner and not repo:
        p0, p1 = owner.split("/", 1)
        return p0.strip(), p1.strip()
    if not owner and " " in repo:
        parts = repo.split()
        if len(parts) == 2:
            return parts[1].strip(), parts[0].strip()

    return owner.strip(), repo.strip()


def search_repositories(query: str, limit: int = 5) -> dict[str, Any]:
    """Search public repositories on GitHub."""
    tool_name = "github.search_repositories"
    if not query or not query.strip():
        return error_result("INVALID_ARGUMENT", "Search query cannot be empty", tool_name)

    limit = min(max(1, limit), 10)
    url = f"{GITHUB_BASE_URL}/search/repositories"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params={"q": query, "per_page": limit}, headers=_headers())
            if resp.status_code == 403:
                return error_result(
                    "RATE_LIMITED",
                    "GitHub API rate limit (403) reached for unauthenticated requests.",
                    tool_name,
                    retryable=True,
                    data=[],
                )
            resp.raise_for_status()
            data = resp.json()
            items = [
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "stargazers_count": item.get("stargazers_count"),
                    "language": item.get("language"),
                }
                for item in data.get("items", [])
            ]
            return success_result(items, tool_name, query=query, count=len(items))
    except Exception as exc:
        return error_result("EXECUTION_ERROR", f"GitHub search error: {exc}", tool_name)


def get_repository(owner: str = "", repo: str = "") -> dict[str, Any]:
    """Retrieve details for a specific GitHub repository."""
    tool_name = "github.get_repository"
    clean_owner, clean_repo = parse_repo_identifier(owner, repo)

    if not clean_owner or not clean_repo:
        return error_result(
            "INVALID_ARGUMENT",
            f"Both repository owner and name are required. Provided owner='{owner}', repo='{repo}'.",
            tool_name,
        )

    repo_url = f"https://github.com/{clean_owner}/{clean_repo}"
    url = f"{GITHUB_BASE_URL}/repos/{clean_owner}/{clean_repo}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers=_headers())
            if resp.status_code == 403:
                # Rate limited: return structured data indicating rate limit with known URL
                return error_result(
                    "RATE_LIMITED",
                    f"GitHub API rate limit (403) reached for unauthenticated requests. Repository URL is {repo_url}.",
                    tool_name,
                    retryable=True,
                    data={
                        "name": clean_repo,
                        "full_name": f"{clean_owner}/{clean_repo}",
                        "html_url": repo_url,
                        "status": "rate_limited",
                    },
                )
            if resp.status_code == 404:
                return error_result(
                    "NOT_FOUND",
                    f"Repository '{clean_owner}/{clean_repo}' not found on GitHub.",
                    tool_name,
                    retryable=False,
                    data={"html_url": repo_url},
                )
            resp.raise_for_status()
            data = resp.json()
            repo_data = {
                "name": data.get("name"),
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "html_url": data.get("html_url") or repo_url,
                "stargazers_count": data.get("stargazers_count"),
                "forks_count": data.get("forks_count"),
                "open_issues_count": data.get("open_issues_count"),
                "default_branch": data.get("default_branch"),
                "language": data.get("language"),
            }
            return success_result(repo_data, tool_name)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", f"Failed retrieving repository: {exc}", tool_name)


def list_issues(owner: str, repo: str, state: str = "open", limit: int = 5) -> dict[str, Any]:
    """List issues for a repository."""
    tool_name = "github.list_issues"
    clean_owner, clean_repo = parse_repo_identifier(owner, repo)
    if not clean_owner or not clean_repo:
        return error_result("INVALID_ARGUMENT", "Owner and repo are required", tool_name)

    limit = min(max(1, limit), 20)
    url = f"{GITHUB_BASE_URL}/repos/{clean_owner}/{clean_repo}/issues"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params={"state": state, "per_page": limit}, headers=_headers())
            if resp.status_code == 403:
                return error_result("RATE_LIMITED", "GitHub API rate limit (403) reached.", tool_name, retryable=True, data=[])
            if resp.status_code == 404:
                return error_result("NOT_FOUND", f"Repository '{clean_owner}/{clean_repo}' not found.", tool_name, data=[])
            resp.raise_for_status()
            data = resp.json()
            issues = [
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "html_url": item.get("html_url"),
                    "created_at": item.get("created_at"),
                }
                for item in data
                if "pull_request" not in item  # Exclude pull requests
            ]
            return success_result(issues, tool_name, count=len(issues))
    except Exception as exc:
        return error_result("EXECUTION_ERROR", f"Error listing issues: {exc}", tool_name)


def list_user_repositories(username: str, limit: int = 10) -> dict[str, Any]:
    """List public repositories for a specific user."""
    tool_name = "github.list_user_repositories"
    clean_user = username.strip().lstrip("@")
    if not clean_user:
        return error_result("INVALID_ARGUMENT", "Username cannot be empty", tool_name)

    limit = min(max(1, limit), 20)
    url = f"{GITHUB_BASE_URL}/users/{clean_user}/repos"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params={"sort": "updated", "per_page": limit}, headers=_headers())
            if resp.status_code == 403:
                return error_result(
                    "RATE_LIMITED",
                    f"GitHub API rate limit (403) reached for user '{clean_user}'.",
                    tool_name,
                    retryable=True,
                    data=[],
                )
            if resp.status_code == 404:
                return error_result("NOT_FOUND", f"User '{clean_user}' not found on GitHub.", tool_name, data=[])
            resp.raise_for_status()
            data = resp.json()
            repos = [
                {
                    "name": item.get("name"),
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "stargazers_count": item.get("stargazers_count"),
                    "language": item.get("language"),
                }
                for item in data
            ]
            return success_result(repos, tool_name, username=clean_user, count=len(repos))
    except Exception as exc:
        return error_result("EXECUTION_ERROR", f"Error listing user repositories: {exc}", tool_name)
