"""
Central Security Audit Logger for Hermes Agent Platform.

Implements Phase 0 & Phase 3 Foundations:
1. Structured append-only JSON Lines logging (`logs/security_audit.jsonl`).
2. Records every tool invocation: tool_name, arguments, policy decision, risk level,
   reason, session_id, user_id, and timestamps.
3. Automatically redacts all PII and known secrets prior to persistence.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from src.security.output_guards import redact_pii

logger = logging.getLogger("hermes.security.audit")

AUDIT_LOG_DIR = Path("logs")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "security_audit.jsonl"


def _sanitize_record(data: Any) -> Any:
    """Recursively sanitize data structures, redacting PII and sensitive values."""
    if isinstance(data, str):
        return redact_pii(data)
    elif isinstance(data, dict):
        return {k: _sanitize_record(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_record(item) for item in data]
    return data


class SecurityAuditLogger:
    """Append-only structured security audit logger."""

    def __init__(self, log_path: Path = AUDIT_LOG_FILE):
        self.log_path = log_path
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        decision: str,
        risk_level: str,
        reason: str,
        session_id: str = "default",
        user_id: str = "default",
        action_category: Optional[str] = None,
        is_tainted: bool = False,
        execution_result: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Record an immutable audit entry for a tool call.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_args = _sanitize_record(arguments)
        clean_result = _sanitize_record(execution_result) if execution_result else None

        entry = {
            "timestamp": now_iso,
            "session_id": session_id,
            "user_id": user_id,
            "tool_name": tool_name,
            "action_category": action_category or "generic",
            "decision": decision,
            "risk_level": risk_level,
            "reason": reason,
            "is_tainted": is_tainted,
            "arguments": clean_args,
            "execution_result_preview": str(clean_result)[:500] if clean_result else None,
        }

        try:
            line = json.dumps(entry, default=str) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.error("security_audit_log_write_failed", exc_info=exc)

        return entry

    def record_search_decision(
        self,
        prompt: str,
        decision: dict[str, Any],
        session_id: str = "default",
        user_id: str = "default",
    ) -> dict[str, Any]:
        """
        Record an auditable entry for a Search Decision Router routing choice.
        Enables offline evaluation of unnecessary searches, missed searches, and search effectiveness.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_prompt = redact_pii(prompt)

        entry = {
            "timestamp": now_iso,
            "event_type": "search_decision",
            "session_id": session_id,
            "user_id": user_id,
            "prompt": clean_prompt[:500],
            "requires_search": decision.get("requires_search", False),
            "search_type": decision.get("search_type", "none"),
            "reason": decision.get("reason", ""),
            "confidence": decision.get("confidence", 0.0),
            "generated_queries": decision.get("queries", []),
            "signals": decision.get("signals", {}),
        }

        try:
            line = json.dumps(entry, default=str) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.error("search_decision_audit_log_write_failed", exc_info=exc)

        return entry


audit_logger = SecurityAuditLogger()
