"""
Hermes Canonical Tool Core.
Single source of truth for all tool operations, business logic, and security jails.
"""
from src.tools.core.contract import (
    ToolResult,
    ToolError,
    success_result,
    error_result,
    format_for_llm,
)
from src.tools.core import filesystem, github, terminal, web

__all__ = [
    "ToolResult",
    "ToolError",
    "success_result",
    "error_result",
    "format_for_llm",
    "filesystem",
    "github",
    "terminal",
    "web",
]
