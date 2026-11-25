"""
Centralized Tool Execution Engine for Hermes Agent Platform.

Unified execution choke-point between the agent and all external tools (Web MCP, Sandbox MCP, core tools):
1. Idempotency management to prevent duplicate side effects on retries.
2. Structural argument sanitization and schema validation.
3. Permission and policy verification (RBAC & Tool Policy Engine).
4. Human-in-the-Loop (HITL) approval gating for high-risk operations.
5. Timeout enforcement, bounded retries, and circuit breaker protection.
6. Result validation, evidence extraction, and append-only audit logging.
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tool_policy import RiskLevel, tool_policy
from src.security.audit import SecurityAuditLogger
from src.security.rbac import rbac_engine
from src.tools import hitl
from src.tools.circuit_breaker import circuit_breaker_registry
from src.tools.mcp_client import mcp_registry
from src.tools.sandbox import sandbox
from src.tools.validators import validate_tool_output

logger = logging.getLogger("hermes.tools.execution_engine")


class ToolExecutionRequest(BaseModel):
    """Canonical request contract for invoking any tool in the Hermes platform."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_id: str = "default"
    session_id: str = "default"
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}")
    tenant_id: str = "default"
    risk_level: str = "low"
    user_role: str = "viewer"
    idempotency_key: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    approval_id: Optional[str] = None
    wrap_untrusted: bool = True


class ToolExecutionResult(BaseModel):
    """Canonical result contract for all tool execution outputs, errors, and metadata."""
    execution_id: str
    status: str  # "success" | "denied" | "error" | "hitl_required" | "circuit_broken" | "timeout"
    output: Any | None = None
    error: str | None = None
    latency_ms: float = 0.0
    idempotent_replay: bool = False
    validation_status: str = "verified"  # "verified" | "flagged" | "failed"
    validation_details: dict[str, Any] = Field(default_factory=dict)
    evidence_item: Optional[dict[str, Any]] = None
    approval_id: Optional[str] = None
    circuit_tripped: bool = False

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ToolExecutionEngine:
    """Centralized execution layer guaranteeing identical policy, safety, and logging."""

    def __init__(self):
        self._lock = threading.Lock()
        self._idempotency_store: dict[str, ToolExecutionResult] = {}
        self._audit_logger = SecurityAuditLogger()

    def execute(self, req: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool through the full 6-phase centralized pipeline."""
        start_time = time.perf_counter()
        exec_id = f"exec_{uuid.uuid4().hex[:12]}"
        clean_args = req.arguments if isinstance(req.arguments, dict) else {}

        # ----------------------------------------------------------------------
        # 1. Idempotency Check: prevent duplicate side-effects on retries
        # ----------------------------------------------------------------------
        if req.idempotency_key:
            with self._lock:
                cached = self._idempotency_store.get(req.idempotency_key)
                if cached:
                    logger.info(
                        "idempotent_replay_hit",
                        extra={"idempotency_key": req.idempotency_key, "tool": req.tool_name},
                    )
                    cached_dict = cached.model_dump()
                    cached_dict["idempotent_replay"] = True
                    return ToolExecutionResult(**cached_dict)

        # ----------------------------------------------------------------------
        # 2. Universal Tool Policy & Evasion Neutralization
        # ----------------------------------------------------------------------
        from src.security.untrusted import taint_tracker, wrap_untrusted_content, scan_for_indirect_injection
        is_tainted = taint_tracker.is_tainted(req.session_id)
        eval_res = tool_policy.evaluate(
            tool_name=req.tool_name,
            arguments=clean_args,
            user_id=req.user_id,
            user_role=req.user_role,
            session_id=req.session_id,
            is_tainted=is_tainted,
        )
        if eval_res.is_denied:
            self._audit_logger.record_tool_call(
                tool_name=req.tool_name,
                arguments=clean_args,
                decision="deny",
                risk_level=eval_res.risk_level.value if eval_res.risk_level else req.risk_level,
                reason=eval_res.reason,
                session_id=req.session_id,
                user_id=req.user_id,
                action_category=eval_res.action_category.value if eval_res.action_category else "security_policy",
            )
            return ToolExecutionResult(
                execution_id=exec_id,
                status="denied",
                error=f"Action Denied by Security Policy: {eval_res.reason}. This operation is strictly prohibited.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                validation_status="failed",
                validation_details={"is_valid": False, "issues": [f"Policy Denial: {eval_res.reason}"]},
            )

        # ----------------------------------------------------------------------
        # 3. Role-Based Access Control (RBAC) Check (Enforced outside the LLM)
        # ----------------------------------------------------------------------
        is_auth, auth_reason = rbac_engine.authorize_tool_call(
            role=req.user_role,
            tool_name=req.tool_name,
            arguments=clean_args,
        )
        if not is_auth:
            self._audit_logger.record_tool_call(
                tool_name=req.tool_name,
                arguments=clean_args,
                decision="deny",
                risk_level=req.risk_level,
                reason=auth_reason,
                session_id=req.session_id,
                user_id=req.user_id,
                action_category="rbac_denial",
            )
            return ToolExecutionResult(
                execution_id=exec_id,
                status="denied",
                error=f"Action Denied by RBAC Policy: {auth_reason}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                validation_status="failed",
                validation_details={"is_valid": False, "issues": [auth_reason]},
            )

        # ----------------------------------------------------------------------
        # 4. Human-In-The-Loop (HITL) Gating
        # ----------------------------------------------------------------------
        # Determine highest risk level between policy and request
        policy_risk = eval_res.risk_level.value if eval_res.risk_level else "low"
        req_risk = req.risk_level or "low"
        risk_precedence = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        eff_risk = policy_risk if risk_precedence.get(policy_risk, 1) >= risk_precedence.get(req_risk, 1) else req_risk
        needs_hitl = eval_res.requires_approval or eff_risk in ("high", "critical")

        if needs_hitl:
            if not req.approval_id or not hitl.is_approved(req.approval_id):
                # Trigger/link pending approval request with strict context binding
                approval_id = req.approval_id or hitl.request_approval(
                    user_id=req.user_id,
                    tenant_id=req.tenant_id,
                    session_id=req.session_id,
                    tool_name=req.tool_name,
                    command=clean_args.get("command") if isinstance(clean_args, dict) else None,
                    arguments=clean_args,
                    risk_level=eff_risk,
                    tool_calls=[{"name": req.tool_name, "arguments": clean_args}],
                )
                self._audit_logger.record_tool_call(
                    tool_name=req.tool_name,
                    arguments=clean_args,
                    decision="require_approval",
                    risk_level=eff_risk,
                    reason="Operation requires human approval",
                    session_id=req.session_id,
                    user_id=req.user_id,
                )
                return ToolExecutionResult(
                    execution_id=exec_id,
                    status="hitl_required",
                    output=f"Blocked: '{req.tool_name}' requires human approval before proceeding ({eval_res.reason}).",
                    approval_id=approval_id,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    validation_status="flagged",
                    validation_details={"requires_approval": True, "reason": eval_res.reason},
                )

        # ----------------------------------------------------------------------
        # 5. Circuit Breaker, Timeout, and Bounded Retries
        # ----------------------------------------------------------------------
        server_key = req.tool_name.split(".")[0] if "." in req.tool_name else req.tool_name
        cb = circuit_breaker_registry.get(server_key)

        if not cb.allow_request():
            logger.warning("tool_execution_tripped_circuit", extra={"tool": req.tool_name, "server": server_key})
            return ToolExecutionResult(
                execution_id=exec_id,
                status="circuit_broken",
                error=f"Circuit breaker for service '{server_key}' is OPEN. Requests blocked to prevent cascading failure.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                circuit_tripped=True,
                validation_status="failed",
            )

        raw_out: Any = None
        exec_error: Optional[str] = None
        status = "success"

        for attempt in range(max(1, req.max_retries)):
            try:
                if req.tool_name == "run_sandboxed_code":
                    raw_out = sandbox.execute(
                        code=clean_args.get("code", ""),
                        language=clean_args.get("language", "python"),
                        timeout_seconds=req.timeout_seconds,
                    )
                elif req.tool_name in ("terminal.run", "terminal_run"):
                    from src.tools.core import terminal
                    is_appr = bool(req.approval_id and hitl.is_approved(req.approval_id))
                    term_res = terminal.run_command(
                        command=clean_args.get("command", ""),
                        timeout=req.timeout_seconds,
                        approved=is_appr,
                        approval_id=req.approval_id,
                        user_id=req.user_id,
                        tenant_id=req.tenant_id,
                    )
                    if not term_res.get("success"):
                        err = term_res.get("error") or {}
                        raw_out = f"Error ({err.get('code', 'EXECUTION_FAILED')}): {err.get('message', 'Failed')}"
                    else:
                        data = term_res.get("data") or {}
                        raw_out = f"Return Code: {data.get('returncode', 0)}\nSTDOUT:\n{data.get('stdout', '')}\nSTDERR:\n{data.get('stderr', '')}"
                elif req.tool_name in ("terminal.delete_file", "terminal_delete_file"):
                    from src.tools.core import terminal
                    is_appr = bool(req.approval_id and hitl.is_approved(req.approval_id))
                    del_res = terminal.delete_file(
                        path=clean_args.get("path", ""),
                        approved=is_appr,
                        approval_id=req.approval_id,
                        user_id=req.user_id,
                        tenant_id=req.tenant_id,
                    )
                    if not del_res.get("success"):
                        err = del_res.get("error") or {}
                        raw_out = f"Error ({err.get('code', 'DELETION_FAILED')}): {err.get('message', 'Failed')}"
                    else:
                        raw_out = str(del_res.get("data"))
                else:
                    raw_out = mcp_registry.call_tool(req.tool_name, clean_args)
                cb.record_success()
                exec_error = None
                break
            except Exception as exc:
                exec_error = str(exc)
                logger.warning(
                    "tool_execution_attempt_failed",
                    extra={"tool": req.tool_name, "attempt": attempt + 1, "error": exec_error},
                )
                cb.record_failure(exec_error)
                if attempt == req.max_retries - 1:
                    status = "error"
                    raw_out = f"Tool execution failed after {req.max_retries} attempts: {exec_error}"

        if status == "success" and req.approval_id and hitl.is_approved(req.approval_id):
            try:
                hitl.mark_executed(req.approval_id)
            except Exception:
                pass

        latency = (time.perf_counter() - start_time) * 1000

        # ----------------------------------------------------------------------
        # 6. Result Validation, Evidence Extraction, and Audit Logging
        # ----------------------------------------------------------------------
        val_res, evidence_item = validate_tool_output(req.tool_name, clean_args, raw_out)
        val_status = val_res.status.value
        if not val_res.is_valid and status == "success":
            status = val_status

        if evidence_item:
            try:
                from src.security.evidence_store import evidence_store
                evidence_store.record_evidence(
                    extracted_text=str(evidence_item.get("content") or raw_out),
                    source_url=str(evidence_item.get("url") or ""),
                    source_title=str(evidence_item.get("source") or req.tool_name),
                    tool_execution_id=req.request_id or exec_id,
                    provenance_type="tool",
                    user_id=req.user_id,
                    tenant_id=req.tenant_id,
                    metadata={"tool_name": req.tool_name},
                )
            except Exception:
                pass

        self._audit_logger.record_tool_call(
            tool_name=req.tool_name,
            arguments=clean_args,
            decision="allow" if status == "success" else "error",
            risk_level=eff_risk,
            reason="Execution completed" if status == "success" else (exec_error or "Error"),
            session_id=req.session_id,
            user_id=req.user_id,
            action_category=eval_res.action_category.value,
        )

        # Check for untrusted external retrieved content, scan for injection, and wrap defensively
        final_out = raw_out
        is_external_data_tool = any(
            k in req.tool_name.lower() for k in ["web_scraper", "read_url", "scrape"]
        )
        if isinstance(raw_out, str) and raw_out.strip():
            injections = scan_for_indirect_injection(raw_out)
            if injections:
                taint_tracker.mark_tainted(req.session_id)
                logger.warning(
                    "indirect_prompt_injection_detected",
                    extra={"tool": req.tool_name, "injections": injections, "session_id": req.session_id},
                )
            if is_external_data_tool and req.wrap_untrusted:
                src_lbl = str(clean_args.get("url") or clean_args.get("query") or req.tool_name)
                final_out = wrap_untrusted_content(
                    content=raw_out,
                    source=src_lbl,
                    label="retrieved_tool_data",
                )

        result = ToolExecutionResult(
            execution_id=exec_id,
            status=status,
            output=final_out,
            error=exec_error,
            latency_ms=round(latency, 2),
            validation_status=val_status,
            validation_details=val_res.to_details_dict(),
            evidence_item=evidence_item,
            approval_id=req.approval_id,
        )

        # Store in idempotency store if requested
        if req.idempotency_key and status == "success":
            with self._lock:
                self._idempotency_store[req.idempotency_key] = result

        return result


tool_execution_engine = ToolExecutionEngine()
