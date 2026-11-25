"""
Shared pipeline utilities for Hermes Agent execution.
Ensures identical tool execution, token budgeting, metadata serialization,
and terminal synthesis rules across LangGraph and SSE streaming.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from src.agent.tool_parser import strip_unwanted_tool_xml
from src.tools.mcp_client import mcp_registry
from src.tools.sandbox import sandbox

logger = logging.getLogger("hermes.agent.pipeline_common")

TOKEN_BUDGET_CHAR_LIMIT = 2500


def create_tool_call_record(
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
    thought: str | None = None,
    preview: str | None = None,
    status: str = "success",
    validation_status: str = "verified",
    validation_details: dict[str, Any] | None = None,
    evidence_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized versioned tool call record for database persistence and telemetry."""
    if preview is None:
        preview = output[:300] + ("..." if len(output) > 300 else "")
    if thought is None:
        thought = f"Invoked '{tool_name}'"

    return {
        "schema_version": 1,
        "id": call_id,
        "tool": tool_name,
        "arguments": arguments if isinstance(arguments, dict) else {},
        "thought": thought,
        "preview": preview,
        "output": output,
        "status": status,
        "validation_status": validation_status,
        "validation_details": validation_details or {},
        "evidence_item": evidence_item,
    }


def execute_tool_with_guard(
    tool_name: str,
    tool_args: dict[str, Any],
    call_id: str = "",
    user_id: str = "default",
    session_id: str = "default",
    tenant_id: str = "default",
    user_role: str = "viewer",
    idempotency_key: str | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute a tool call safely through the Centralized Tool Execution Engine,
    applying idempotency, RBAC, safety policy, HITL, retries, circuit breakers,
    and output validation.
    Single source of truth for tool execution in both graph.py and stream_engine.py.
    """
    clean_args = tool_args if isinstance(tool_args, dict) else {}
    from src.tools.execution_engine import ToolExecutionRequest, tool_execution_engine

    req = ToolExecutionRequest(
        tool_name=tool_name,
        arguments=clean_args,
        request_id=call_id or f"req_{uuid.uuid4().hex[:10]}",
        user_id=user_id,
        session_id=session_id,
        tenant_id=tenant_id,
        user_role=user_role,
        idempotency_key=idempotency_key,
        approval_id=approval_id,
    )

    res = tool_execution_engine.execute(req)

    # Determine status and output representation
    if res.status == "denied":
        out_str = str(res.error or "Action Denied by Security Policy")
        thought = f"Attempted to invoke '{tool_name}' but blocked by security policy/RBAC."
        status = "denied"
    elif res.status == "hitl_required":
        out_str = str(res.output or f"Blocked: '{tool_name}' requires human approval before proceeding.")
        thought = f"Operation '{tool_name}' requires human approval."
        status = "hitl_required"
    elif res.status == "circuit_broken":
        out_str = str(res.error or "Circuit breaker for service is OPEN.")
        thought = f"Tool '{tool_name}' was blocked by open circuit breaker."
        status = "error"
    elif res.status == "error":
        out_str = str(res.output or res.error or "Tool execution error")
        thought = f"Tool '{tool_name}' failed execution."
        status = "error"
    else:
        out_str = str(res.output)
        thought = f"Decided to invoke '{tool_name}' with parameters: {json.dumps(clean_args)}."
        status = "success"

    # Enforce token budget limit
    if len(out_str) > TOKEN_BUDGET_CHAR_LIMIT:
        out_str = out_str[:TOKEN_BUDGET_CHAR_LIMIT] + "\n... [Output truncated to stay within token budget]"

    preview = out_str[:300] + ("..." if len(out_str) > 300 else "")

    return create_tool_call_record(
        call_id=call_id or res.execution_id,
        tool_name=tool_name,
        arguments=clean_args,
        output=out_str,
        thought=thought,
        preview=preview,
        status=status,
        validation_status=res.validation_status,
        validation_details=res.validation_details,
        evidence_item=res.evidence_item,
    )


def clean_final_reply(reply: str, executed_steps: list[dict[str, Any]] | None = None) -> str:
    """
    Sanitize final assistant reply, ensuring no raw XML tags leak,
    and generating clean fallback messages if the reply is empty.
    """
    cleaned = strip_unwanted_tool_xml(reply).strip()
    if not cleaned:
        if executed_steps:
            tools_summary = ", ".join(f"`{s['tool']}`" for s in executed_steps)
            return f"I completed running the requested tools ({tools_summary}), but did not generate a synthesized summary."
        return "I'm sorry, I could not generate a response. Please try again."
    return cleaned
