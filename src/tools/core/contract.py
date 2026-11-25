"""
Standard Tool Contract for Hermes Agent Platform.
Provides uniform result representations across stdio MCP servers and local fallbacks.
"""
from __future__ import annotations

import json
from typing import Any, Optional, TypedDict


class ToolError(TypedDict, total=False):
    code: str  # e.g., "RATE_LIMITED", "NOT_FOUND", "PERMISSION_DENIED", "EXECUTION_ERROR"
    message: str
    retryable: bool


class ToolResult(TypedDict):
    success: bool
    data: Any
    error: Optional[ToolError]
    metadata: dict[str, Any]


def success_result(data: Any, tool_name: str, **extra_meta: Any) -> dict[str, Any]:
    """Create a standardized successful ToolResult."""
    return {
        "success": True,
        "data": data,
        "error": None,
        "metadata": {
            "tool_name": tool_name,
            **extra_meta,
        },
    }


def error_result(
    code: str,
    message: str,
    tool_name: str,
    retryable: bool = False,
    data: Any = None,
    **extra_meta: Any,
) -> dict[str, Any]:
    """Create a standardized error ToolResult."""
    return {
        "success": False,
        "data": data,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
        "metadata": {
            "tool_name": tool_name,
            **extra_meta,
        },
    }


def format_for_llm(res: dict[str, Any]) -> str:
    """Format a ToolResult into a compact, human/LLM readable string."""
    if res.get("success"):
        data = res.get("data")
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        return str(data)
    else:
        err = res.get("error") or {}
        code = err.get("code", "ERROR")
        msg = err.get("message", "Tool execution failed")
        tool = res.get("metadata", {}).get("tool_name", "tool")
        retryable_str = " (retryable)" if err.get("retryable") else ""
        return f"[{tool} failed: {code}{retryable_str}] {msg}"
