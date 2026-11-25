"""
Canonical Core Logic for Terminal and File Deletion Operations.
Strictly confined to workspace directory with structural policy enforcement.
Enforces strict Human-in-the-Loop approval binding and execution gateway verification.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Optional

from src.agent.tool_policy import tool_policy, PolicyDecision
from src.tools.core.contract import error_result, success_result
from src.tools.core.filesystem import resolve_workspace_path, WORKSPACE_DIR
from src.tools import hitl


def run_command(
    command: str,
    timeout: float = 30.0,
    approved: bool = False,
    approval_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute a shell command inside the workspace.
    Enforces Component 3 structural policy evaluation, strict Human-in-the-Loop
    approval token verification, and invokes programs with argument arrays (shell=False)
    where possible to avoid shell evasion.
    """
    tool_name = "terminal.run"
    
    # 1. Structural Command Security Policy Check
    eval_res = tool_policy.evaluate(tool_name, {"command": command})
    if eval_res.decision == PolicyDecision.DENY:
        return error_result(
            "POLICY_DENIED",
            f"Security Policy DENIED command: {eval_res.reason}",
            tool_name,
        )

    needs_approval = (eval_res.decision == PolicyDecision.REQUIRE_APPROVAL)

    # 2. Human-In-The-Loop Approval Gateway Verification
    if approved or needs_approval:
        if not approved:
            return error_result(
                "APPROVAL_REQUIRED",
                f"Human-in-the-Loop approval required: {eval_res.reason}",
                tool_name,
            )

        # Never trust a plain boolean without validating the approval token
        if not approval_id:
            return error_result(
                "PERMISSION_DENIED",
                "Execution rejected: approval_id token is required when approved=True; plain boolean approval is untrusted.",
                tool_name,
            )

        # Verify that approval was explicitly granted and matches exact command and workspace
        is_valid, err_msg = hitl.validate_approval_for_command(
            approval_id=approval_id,
            command=command,
            working_directory=str(WORKSPACE_DIR),
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if not is_valid:
            return error_result(
                "PERMISSION_DENIED",
                f"Execution rejected: {err_msg}",
                tool_name,
            )

        # Mark approval as executed so the exact same approval cannot be triggered twice
        try:
            hitl.mark_executed(approval_id)
        except Exception as exc:
            return error_result(
                "PERMISSION_DENIED",
                f"Execution rejected: failed to mark approval token as executed ({str(exc)})",
                tool_name,
            )

    try:
        # Avoid shell interpretation entirely for standard commands without piping
        has_shell_meta = any(char in command for char in (";", "&&", "||", "|", ">", "<", "*", "?", "["))
        use_shell = has_shell_meta
        
        if not use_shell:
            try:
                cmd_args = shlex.split(command, posix=(sys.platform != "win32"))
                if sys.platform == "win32" and isinstance(cmd_args, list) and cmd_args:
                    windows_builtins = {"echo", "dir", "del", "type", "copy", "move", "mkdir", "rmdir", "set", "cd", "cls"}
                    if cmd_args[0].lower() in windows_builtins:
                        use_shell = True
                        cmd_args = command
                    elif not shutil.which(cmd_args[0]):
                        # Commands like ls, cat, rm, pwd are built-in aliases in PowerShell
                        cmd_args = ["powershell", "-NoProfile", "-Command", command]
                        use_shell = False
            except Exception:
                cmd_args = command
                use_shell = True
        else:
            cmd_args = command

        proc = subprocess.run(
            cmd_args,
            shell=use_shell,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        data = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode != 0:
            return error_result(
                "NON_ZERO_EXIT",
                f"Command exited with return code {proc.returncode}: {proc.stderr.strip()}",
                tool_name,
                data=data,
            )
        return success_result(data, tool_name)
    except subprocess.TimeoutExpired:
        return error_result("TIMEOUT", f"Command timed out after {timeout} seconds", tool_name, retryable=True)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", str(exc), tool_name)


def delete_file(path: str) -> dict[str, Any]:
    """Delete a file permanently from the workspace."""
    tool_name = "terminal.delete_file"
    try:
        target = resolve_workspace_path(path)
        if not target.exists():
            return error_result("NOT_FOUND", f"File '{path}' does not exist in workspace.", tool_name)
        if target.is_dir():
            return error_result("IS_DIRECTORY", f"Cannot delete directory using delete_file: '{path}'", tool_name)
        target.unlink()
        return success_result(f"File '{path}' successfully deleted from workspace.", tool_name, path=str(path))
    except PermissionError as exc:
        return error_result("PERMISSION_DENIED", str(exc), tool_name)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", str(exc), tool_name)
