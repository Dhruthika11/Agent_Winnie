"""
Human-in-the-Loop (HITL) gateway.

Classifies tool calls by risk and, for anything high-risk, parks the call
in a pending-approvals store until a human explicitly approves or rejects
it via the /approvals API in src/main.py. Backed by Redis so approvals
survive process restarts and work across multiple agent workers; falls
back to an in-memory dict if Redis is unreachable (e.g. local dev/tests).

Enforces strict cryptographic binding of:
- user_id, tenant_id, session_id
- tool_name, exact command and arguments
- working_directory (WORKSPACE_DIR)
- risk_level, affected_files
- approval_id, status ("pending" | "approved" | "rejected" | "expired" | "executed")
- created_at, expires_at (respecting settings.hitl_approval_timeout_seconds)
"""
from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import logging
import os
import re
import shlex
import sys
import time
import uuid
from typing import Any, Optional

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None

from src.config import settings
from src.tools.core.filesystem import WORKSPACE_DIR

logger = logging.getLogger("hermes.tools.hitl")

# Glob patterns matched against tool name -> risk level.
# Anything not matched defaults to "low".
RISK_REGISTRY: dict[str, str] = {
    "*.delete_*": "high",
    "*.drop_*": "high",
    "*wire_transfer*": "high",
    "*send_payment*": "high",
    "*execute_trade*": "high",
    "*rm_*": "high",
    "*terminal.run*": "medium",
    "*.write_*": "medium",
    "*.update_*": "medium",
    "*.read_*": "low",
    "*.search*": "low",
    "*.get_*": "low",
    "run_sandboxed_code": "medium",
}

_memory_store: dict[str, dict[str, Any]] = {}


def classify_risk(tool_name: str, arguments: dict[str, Any]) -> str:
    """Classify tool risk level based on name and argument patterns."""
    danger_markers = ("--force", "rm -rf", "drop table", "truncate")
    try:
        flat_args = json.dumps(arguments, default=str).lower()
    except Exception:
        flat_args = str(arguments).lower()

    if any(marker in flat_args for marker in danger_markers):
        return "high"

    for pattern, level in RISK_REGISTRY.items():
        if fnmatch.fnmatch(tool_name, pattern):
            return level

    return "low"


def extract_affected_files(
    tool_name: str,
    arguments: Optional[dict[str, Any]] = None,
    command: Optional[str] = None,
) -> list[str]:
    """Extract affected file paths or patterns from tool arguments and commands."""
    files: set[str] = set()
    arguments = arguments or {}

    # 1. From arguments dict
    for key in ("path", "file_path", "target_path", "file", "dest", "src", "destination", "source"):
        val = arguments.get(key)
        if val and isinstance(val, str) and val.strip():
            files.add(val.strip())

    for key in ("paths", "files", "affected_files"):
        val = arguments.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    files.add(item.strip())

    # 2. From shell command string
    cmd_str = command or arguments.get("command")
    if cmd_str and isinstance(cmd_str, str):
        redirect_matches = re.findall(r"(?:>>|>|<)\s*([^\s;|&]+)", cmd_str)
        for m in redirect_matches:
            if m.strip():
                files.add(m.strip())

        try:
            tokens = shlex.split(cmd_str, posix=(sys.platform != "win32"))
            for token in tokens:
                token_clean = token.strip().strip("'\"")
                if (
                    "/" in token_clean
                    or "\\" in token_clean
                    or (len(token_clean) > 2 and "." in token_clean)
                    or "*" in token_clean
                ) and not token_clean.startswith("-"):
                    files.add(token_clean)
        except Exception:
            pass

    return sorted(list(files))


def compute_binding_hash(
    approval_id: str,
    user_id: str,
    tenant_id: str,
    session_id: str,
    tool_name: str,
    command: Optional[str],
    arguments: Any,
    working_directory: str,
    risk_level: str,
    affected_files: list[str],
) -> str:
    """Compute a cryptographic HMAC-SHA256 signature binding the approval payload."""
    canonical_payload = {
        "approval_id": str(approval_id),
        "user_id": str(user_id),
        "tenant_id": str(tenant_id),
        "session_id": str(session_id),
        "tool_name": str(tool_name),
        "command": str(command) if command is not None else "",
        "arguments": arguments if arguments is not None else {},
        "working_directory": os.path.normpath(str(working_directory)),
        "risk_level": str(risk_level),
        "affected_files": sorted(list(affected_files or [])),
    }
    serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), default=str)
    secret_key = (getattr(settings, "jwt_secret_key", None) or "hermes-hitl-insecure-key").encode("utf-8")
    return hmac.new(secret_key, serialized.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_binding(record: dict[str, Any]) -> bool:
    """Verify cryptographic binding integrity of an approval record."""
    if not isinstance(record, dict):
        return False
    expected_hash = record.get("binding_hash")
    if not expected_hash:
        return False

    actual_hash = compute_binding_hash(
        approval_id=record.get("approval_id", ""),
        user_id=record.get("user_id", ""),
        tenant_id=record.get("tenant_id", "default"),
        session_id=record.get("session_id", ""),
        tool_name=record.get("tool_name", ""),
        command=record.get("command"),
        arguments=record.get("arguments", {}),
        working_directory=record.get("working_directory", str(WORKSPACE_DIR)),
        risk_level=record.get("risk_level", "low"),
        affected_files=record.get("affected_files", []),
    )
    return hmac.compare_digest(str(expected_hash), str(actual_hash))


class ApprovalStore:
    def __init__(self):
        self._redis = None
        if redis_lib is not None:
            try:
                self._redis = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
                self._redis.ping()
            except Exception:  # noqa: BLE001
                logger.warning("hitl_redis_unavailable_using_memory_store")
                self._redis = None

    def put(self, approval_id: str, record: dict) -> None:
        if self._redis:
            # Keep key in Redis long enough to allow viewing expired / executed status
            ttl = int(record.get("expires_at", time.time() + settings.hitl_approval_timeout_seconds) - time.time())
            redis_ttl = max(ttl + 86400, 86400)
            try:
                self._redis.set(
                    f"approval:{approval_id}",
                    json.dumps(record, default=str),
                    ex=redis_ttl,
                )
            except Exception as exc:
                logger.error("hitl_redis_put_failed_fallback_to_memory", extra={"error": str(exc)})
                _memory_store[approval_id] = record
        else:
            _memory_store[approval_id] = record

    def get(self, approval_id: str) -> Optional[dict]:
        record: Optional[dict] = None
        if self._redis:
            try:
                raw = self._redis.get(f"approval:{approval_id}")
                record = json.loads(raw) if raw else None
            except Exception as exc:
                logger.error("hitl_redis_get_failed_fallback_to_memory", extra={"error": str(exc)})
                record = _memory_store.get(approval_id)
        else:
            record = _memory_store.get(approval_id)

        if record:
            # Check if pending record has expired on access
            if record.get("status") == "pending" and time.time() > record.get("expires_at", float("inf")):
                record["status"] = "expired"
                self.put(approval_id, record)

        return record

    def list_all(self) -> list[dict]:
        records: list[dict] = []
        if self._redis:
            try:
                for key in self._redis.scan_iter("approval:*"):
                    raw = self._redis.get(key)
                    if raw:
                        records.append(json.loads(raw))
                return records
            except Exception:
                pass
        return list(_memory_store.values())

    def list_pending(self, tenant_id: Optional[str] = None) -> list[dict]:
        pending: list[dict] = []
        now = time.time()
        for rec in self.list_all():
            if rec.get("status") == "pending":
                if now > rec.get("expires_at", float("inf")):
                    rec["status"] = "expired"
                    self.put(rec["approval_id"], rec)
                else:
                    if tenant_id and rec.get("tenant_id") and rec.get("tenant_id") != tenant_id:
                        continue
                    pending.append(format_approval_record(rec))
        return pending

    def list_approvals(self, tenant_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        result: list[dict] = []
        now = time.time()
        for rec in self.list_all():
            if rec.get("status") == "pending" and now > rec.get("expires_at", float("inf")):
                rec["status"] = "expired"
                self.put(rec["approval_id"], rec)
            if tenant_id and rec.get("tenant_id") and rec.get("tenant_id") != tenant_id:
                continue
            formatted = format_approval_record(rec)
            if status and formatted.get("status") != status:
                continue
            result.append(formatted)
        result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return result


def format_approval_record(record: dict) -> dict:
    """Format and enrich an approval record into the canonical structured contract."""
    now = time.time()
    created_at = float(record.get("created_at", now))
    timeout = float(record.get("timeout_seconds") or getattr(settings, "hitl_approval_timeout_seconds", 300))
    expires_at = float(record.get("expires_at", created_at + timeout))

    status = record.get("status", "pending")
    if status == "pending" and now > expires_at:
        status = "expired"
        record["status"] = "expired"

    tool_name = record.get("tool_name") or "unknown_tool"
    arguments = record.get("arguments") or {}
    command = record.get("command")
    if not command and tool_name != "unknown_tool":
        command = f"{tool_name}({json.dumps(arguments)})" if arguments else f"{tool_name}()"

    working_directory = record.get("working_directory") or str(WORKSPACE_DIR)
    affected_files = record.get("affected_files") or []
    risk_level = record.get("risk_level") or "high"
    reason = record.get("reason") or f"Tool '{tool_name}' classified as {risk_level} risk requires manual operator authorization."
    tenant_id = record.get("tenant_id") or "default"
    user_id = record.get("user_id") or "default"

    return {
        "approval_id": record.get("approval_id"),
        "tool_name": tool_name,
        "command": command,
        "arguments": arguments,
        "working_directory": working_directory,
        "affected_files": affected_files,
        "risk_level": risk_level,
        "reason": reason,
        "status": status,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "created_at": created_at,
        "expires_at": expires_at,
        "session_id": record.get("session_id"),
        "decided_by": record.get("decided_by"),
        "decided_at": record.get("decided_at"),
        "executed_at": record.get("executed_at"),
    }


_store = ApprovalStore()


def request_approval(
    user_id: str = "default_user",
    session_id: str = "default_session",
    tool_calls: Optional[list[dict]] = None,
    state: Optional[dict] = None,
    approval_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    command: Optional[str] = None,
    arguments: Optional[dict[str, Any]] = None,
    working_directory: Optional[str] = None,
    risk_level: Optional[str] = None,
    affected_files: Optional[list[str]] = None,
    timeout_seconds: Optional[float] = None,
    reason: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Register a strictly bound approval request in the store.
    Binds user_id, tenant_id, session_id, tool_name, exact command and arguments,
    working_directory, risk_level, affected_files, status, created_at, and expires_at.
    """
    approval_id = approval_id or str(uuid.uuid4())
    now = time.time()
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else getattr(settings, "hitl_approval_timeout_seconds", 300)
    )
    expires_at = now + timeout

    # Context extraction from state if present
    if state and isinstance(state, dict):
        user_id = user_id or state.get("user_id") or "default_user"
        tenant_id = tenant_id or state.get("tenant_id") or "default"
        session_id = session_id or state.get("session_id") or "default_session"

    tenant_id = tenant_id or "default"
    user_id = user_id or "default_user"
    session_id = session_id or "default_session"

    # Context extraction from tool_calls if passed
    if tool_calls and len(tool_calls) > 0:
        first_call = tool_calls[0]
        if not tool_name:
            tool_name = (
                first_call.get("name")
                or first_call.get("tool")
                or (
                    first_call.get("function", {}).get("name")
                    if isinstance(first_call.get("function"), dict)
                    else None
                )
            )
        if arguments is None:
            raw_args = first_call.get("arguments") or (
                first_call.get("function", {}).get("arguments")
                if isinstance(first_call.get("function"), dict)
                else None
            )
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {"raw": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args
    elif tool_name and tool_calls is None:
        tool_calls = [{"name": tool_name, "arguments": arguments or {}}]
    else:
        tool_calls = tool_calls or []

    tool_name = tool_name or (tool_calls[0].get("name") if tool_calls else "unknown_tool")
    arguments = arguments or {}

    if command is None and isinstance(arguments, dict):
        command = arguments.get("command")

    working_directory = str(
        working_directory
        or (arguments.get("cwd") if isinstance(arguments, dict) else None)
        or (arguments.get("working_directory") if isinstance(arguments, dict) else None)
        or (arguments.get("workdir") if isinstance(arguments, dict) else None)
        or (arguments.get("dir") if isinstance(arguments, dict) else None)
        or WORKSPACE_DIR
    )
    risk_level = risk_level or classify_risk(tool_name, arguments)
    if affected_files is None:
        affected_files = extract_affected_files(tool_name, arguments, command)

    if reason is None:
        reason = f"Tool '{tool_name}' classified as {risk_level} risk requires manual operator authorization."

    binding_hash = compute_binding_hash(
        approval_id=approval_id,
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id,
        tool_name=tool_name,
        command=command,
        arguments=arguments,
        working_directory=working_directory,
        risk_level=risk_level,
        affected_files=affected_files,
    )

    record = {
        "approval_id": approval_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "tool_name": tool_name,
        "command": command,
        "arguments": arguments,
        "exact_command_and_arguments": {
            "command": command,
            "arguments": arguments,
        },
        "working_directory": working_directory,
        "risk_level": risk_level,
        "reason": reason,
        "affected_files": affected_files,
        "status": "pending",
        "created_at": now,
        "expires_at": expires_at,
        "binding_hash": binding_hash,
        "tool_calls": tool_calls,
        "state": state,
    }

    _store.put(approval_id, record)
    logger.info(
        "hitl_approval_requested",
        extra={
            "approval_id": approval_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "tool": tool_name,
            "risk": risk_level,
            "expires_at": expires_at,
        },
    )
    return approval_id


def resolve_approval(
    approval_id: str,
    approved: bool,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    decided_by: Optional[str] = None,
) -> dict[str, Any]:
    """
    Resolve an approval request (approve or reject).
    Enforces:
    - Expiration verification (rejects with status='expired' and PermissionError if expired).
    - Tenant and User isolation (rejects cross-tenant and cross-user replays with PermissionError).
    - Prevention of duplicate resolution / execution.
    """
    record = _store.get(approval_id)
    if record is None:
        raise KeyError(f"Approval request '{approval_id}' not found.")

    # 1. Tenant validation
    if tenant_id is not None and record.get("tenant_id") is not None:
        if str(tenant_id) != str(record.get("tenant_id")):
            raise PermissionError(
                f"Cross-tenant access denied: tenant '{tenant_id}' does not match approval record tenant '{record.get('tenant_id')}'."
            )

    # 2. User validation (prevents cross-user replay & reuse)
    if user_id is not None and record.get("user_id") is not None:
        if str(user_id) != str(record.get("user_id")):
            raise PermissionError(
                f"Cross-user access denied: user '{user_id}' does not match approval record user '{record.get('user_id')}'."
            )

    # 3. Duplicate execution & resolution protection
    current_status = record.get("status")
    if current_status == "executed":
        raise PermissionError(
            f"Approval request '{approval_id}' has already been executed. Duplicate execution prohibited."
        )
    if current_status in ("approved", "rejected"):
        raise PermissionError(
            f"Approval request '{approval_id}' has already been resolved with status '{current_status}'."
        )
    if current_status == "expired":
        raise PermissionError(f"Approval request '{approval_id}' has expired.")

    # 4. Expiration check
    now = time.time()
    expires_at = record.get("expires_at", float("inf"))
    if now > expires_at:
        record["status"] = "expired"
        _store.put(approval_id, record)
        raise PermissionError(
            f"Approval request '{approval_id}' has expired at {expires_at} (current time {now}). Execution rejected."
        )

    # 5. Apply resolution
    actor = decided_by or user_id or "unknown"
    record["status"] = "approved" if approved else "rejected"
    record["decided_by"] = actor
    record["decided_at"] = now
    _store.put(approval_id, record)

    logger.info(
        "hitl_approval_resolved",
        extra={
            "approval_id": approval_id,
            "status": record["status"],
            "user_id": user_id,
            "tenant_id": tenant_id,
            "decided_by": actor,
        },
    )
    return record


def mark_executed(approval_id: str) -> dict[str, Any]:
    """
    Mark an approved record as 'executed' to protect against duplicate execution / replay attacks.
    """
    record = _store.get(approval_id)
    if record is None:
        raise KeyError(f"Approval request '{approval_id}' not found.")
    if record.get("status") == "executed":
        raise PermissionError(
            f"Approval request '{approval_id}' has already been executed. Duplicate execution prohibited."
        )
    if record.get("status") != "approved":
        raise PermissionError(
            f"Cannot execute approval '{approval_id}' with status '{record.get('status')}'. Explicit approval required."
        )

    record["status"] = "executed"
    record["executed_at"] = time.time()
    _store.put(approval_id, record)
    logger.info("hitl_approval_marked_executed", extra={"approval_id": approval_id})
    return record


def validate_approval_for_command(
    approval_id: Optional[str],
    command: str,
    working_directory: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate that approval_id is present, approved, unexpired, unexecuted,
    and strictly matches the exact command, arguments, and workspace directory.
    Returns (is_valid, reason_if_invalid).
    """
    if not approval_id:
        return False, "Approval token (approval_id) is missing."

    record = _store.get(approval_id)
    if record is None:
        return False, f"Approval request '{approval_id}' not found in approval store."

    status = record.get("status")
    if status == "executed":
        return False, f"Approval token '{approval_id}' has already been executed. Duplicate execution prohibited."
    if status == "rejected":
        return False, f"Approval token '{approval_id}' was explicitly rejected."
    if status == "expired" or time.time() > record.get("expires_at", float("inf")):
        record["status"] = "expired"
        _store.put(approval_id, record)
        return False, f"Approval token '{approval_id}' has expired."
    if status != "approved":
        return False, f"Approval token '{approval_id}' has not been approved (current status: '{status}')."

    # Cross-tenant and cross-user check at gateway
    if tenant_id is not None and record.get("tenant_id") is not None:
        if str(tenant_id) != str(record.get("tenant_id")):
            return (
                False,
                f"Cross-tenant access denied: tenant '{tenant_id}' does not match approval record tenant '{record.get('tenant_id')}'.",
            )
    if user_id is not None and record.get("user_id") is not None:
        if str(user_id) != str(record.get("user_id")):
            return (
                False,
                f"Cross-user access denied: user '{user_id}' does not match approval record user '{record.get('user_id')}'.",
            )

    # Validate exact command
    record_cmd = record.get("command")
    if record_cmd is not None and record_cmd != command:
        return (
            False,
            f"Command mismatch: command changed after approval was requested. "
            f"Expected '{record_cmd}', got '{command}'. Fresh approval required.",
        )

    # Validate arguments command if stored in dict
    record_args = record.get("arguments")
    if isinstance(record_args, dict) and "command" in record_args:
        if record_args["command"] != command:
            return (
                False,
                f"Command arguments mismatch: expected '{record_args['command']}', got '{command}'. Fresh approval required.",
            )

    # Validate workspace directory
    if working_directory:
        record_wd = record.get("working_directory")
        if record_wd:
            norm_rec = os.path.abspath(os.path.normpath(str(record_wd)))
            norm_req = os.path.abspath(os.path.normpath(str(working_directory)))
            if norm_rec != norm_req:
                return (
                    False,
                    f"Workspace directory mismatch: working directory changed after approval was requested. "
                    f"Expected '{norm_rec}', got '{norm_req}'. Fresh approval required.",
                )

    # Verify cryptographic binding hash
    if record.get("binding_hash"):
        if not verify_binding(record):
            return False, "Approval token binding integrity check failed (cryptographic hash mismatch)."

    return True, ""


def decide(approval_id: str, approve: bool, decided_by: str = "unknown") -> Optional[dict]:
    """
    Backwards-compatible wrapper over resolve_approval for API endpoints.
    Returns the updated record or None if approval is not found / invalid / expired.
    """
    try:
        return resolve_approval(
            approval_id=approval_id,
            approved=approve,
            user_id=None,
            tenant_id=None,
            decided_by=decided_by,
        )
    except (KeyError, PermissionError) as exc:
        logger.warning("hitl_decide_failed", extra={"approval_id": approval_id, "error": str(exc)})
        return None


def is_approved(approval_id: Optional[str]) -> bool:
    """Check if an approval_id is valid, currently approved, and unexpired."""
    if not approval_id:
        return False
    record = _store.get(approval_id)
    if not record:
        return False
    if time.time() > record.get("expires_at", float("inf")):
        if record.get("status") != "expired":
            record["status"] = "expired"
            _store.put(approval_id, record)
        return False
    return bool(record.get("status") == "approved")


def mark_expired(approval_id: str) -> Optional[dict]:
    """Mark an approval request as expired."""
    record = _store.get(approval_id)
    if record is None:
        return None
    record["status"] = "expired"
    _store.put(approval_id, record)
    return record


def get_approval(approval_id: str, structured: bool = False) -> Optional[dict]:
    """Retrieve an approval record by approval_id."""
    record = _store.get(approval_id)
    if record is None:
        return None
    if record.get("status") == "pending" and time.time() > record.get("expires_at", float("inf")):
        record["status"] = "expired"
        _store.put(approval_id, record)
    if structured:
        return format_approval_record(record)
    return record


def list_pending_approvals(tenant_id: Optional[str] = None) -> list[dict]:
    """List all currently pending and unexpired approval requests."""
    return _store.list_pending(tenant_id=tenant_id)


def list_approvals(tenant_id: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
    """List approvals optionally filtered by tenant and status."""
    return _store.list_approvals(tenant_id=tenant_id, status=status)
