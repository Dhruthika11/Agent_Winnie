"""
Canonical Core Logic for Filesystem Operations.
Enforces strict workspace directory confinement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.tools.core.contract import error_result, success_result

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", Path.cwd() / "workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(rel_path: str) -> Path:
    """Resolve a relative path and strictly enforce workspace confinement."""
    clean = str(rel_path).lstrip("/\\").replace("workspace/", "").replace("workspace\\", "")
    target = (WORKSPACE_DIR / clean).resolve()
    if not target.is_relative_to(WORKSPACE_DIR):
        raise PermissionError(f"Access denied: path '{rel_path}' escapes the workspace directory.")
    return target


def read_file(path: str) -> dict[str, Any]:
    """Read file content within the workspace."""
    tool_name = "filesystem.read_file"
    try:
        target = resolve_workspace_path(path)
        if not target.exists():
            return error_result("NOT_FOUND", f"File '{path}' does not exist in workspace.", tool_name)
        if not target.is_file():
            return error_result("IS_DIRECTORY", f"Path '{path}' is a directory, not a file.", tool_name)
        content = target.read_text(encoding="utf-8", errors="replace")[:8000]
        return success_result(content, tool_name, path=str(path))
    except PermissionError as exc:
        return error_result("PERMISSION_DENIED", str(exc), tool_name)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", str(exc), tool_name)


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write text content to a file in the workspace."""
    tool_name = "filesystem.write_file"
    try:
        target = resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return success_result(f"Successfully wrote {len(content)} characters to '{path}'.", tool_name, path=str(path))
    except PermissionError as exc:
        return error_result("PERMISSION_DENIED", str(exc), tool_name)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", str(exc), tool_name)


def list_directory(path: str = ".") -> dict[str, Any]:
    """List directory contents within the workspace."""
    tool_name = "filesystem.list_directory"
    try:
        target = resolve_workspace_path(path)
        if not target.exists():
            return error_result("NOT_FOUND", f"Directory '{path}' does not exist in workspace.", tool_name)
        if not target.is_dir():
            return error_result("NOT_A_DIRECTORY", f"Path '{path}' is a file, not a directory.", tool_name)

        entries = []
        for item in sorted(target.iterdir()):
            entries.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else 0,
            })
        return success_result(entries, tool_name, path=str(path))
    except PermissionError as exc:
        return error_result("PERMISSION_DENIED", str(exc), tool_name)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", str(exc), tool_name)
