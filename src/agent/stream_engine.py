"""
Streaming Agent Adapter for Real-Time Server-Sent Events (SSE).
Delegates all agent orchestration, safety gating, tool execution, and synthesis
solely to LangGraph (agent_graph).
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
import uuid
from typing import Any, AsyncGenerator

from src.agent.graph import agent_graph
from src.agent.router import agent_router
from src.agent.state import AgentState
from src.db.chat_history import chat_history_db
from src.security.output_guards import StreamingOutputBuffer
from src.utils.telemetry import get_langfuse_client

logger = logging.getLogger("hermes.agent.stream")


def _format_sse(event_type: str, data: dict[str, Any]) -> str:
    """Format an event according to the Server-Sent Events (SSE) standard."""
    payload = json.dumps({"type": event_type, **data})
    return f"event: {event_type}\ndata: {payload}\n\n"


async def stream_agent_workflow(
    message: str,
    user_id: str = "web_visitor",
    session_id: str | None = None,
    tenant_id: str = "default",
    max_iterations: int = 8,
    images: list[str] | None = None,
    request: Any | None = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming adapter for SSE: executes the single LangGraph agent runtime in a background worker,
    streaming tokens, status updates, tool events, and safety gates in real-time.
    """
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

    workflow_start = time.perf_counter()
    first_token_time: float | None = None

    # Telemetry setup
    lf = get_langfuse_client()
    trace_id = None
    root_span = None
    if lf:
        try:
            trace = lf.trace(
                name="chat_turn_stream",
                session_id=session_id,
                user_id=user_id,
                input=message,
            )
            trace_id = trace.id
            root_span = lf.span(
                trace_id=trace.id,
                name="agent_stream_workflow",
                input={"user_id": user_id, "session_id": session_id, "message": message, "has_images": bool(images)},
            )
        except Exception:
            pass

    yield _format_sse("session_init", {
        "session_id": session_id,
        "user_id": user_id,
        "trace_id": trace_id,
        "has_images": bool(images),
    })

    # Persist user prompt to chat history
    try:
        chat_history_db.append_message(
            session_id=session_id,
            role="user",
            content=message,
            user_id=user_id,
            tenant_id=tenant_id,
        )
    except Exception:
        logger.warning("failed_to_persist_user_message_to_history", exc_info=True)

    # Construct user message (multimodal payload if images are attached)
    if images:
        user_content: Any = [{"type": "text", "text": message}]
        for img in images:
            user_content.append({"type": "image_url", "image_url": {"url": img}})
        yield _format_sse("status", {"message": f"Processing {len(images)} image(s) with multimodal vision..."})
    else:
        user_content = message

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_content}],
        "user_id": user_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "trace_id": trace_id,
        "parent_span_id": root_span.id if root_span else None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "memory_context": "",
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "detected_intent": None,
        "evidence_urls": [],
        "evidence_items": [],
        "evidence_verification": None,
        "denied_tool_calls": [],
        "grounding_retries": 0,
        "done": False,
        "stop_reason": None,
    }

    config = {
        "configurable": {
            "thread_id": session_id,
            "event_queue": event_queue,
        }
    }

    # Execute LangGraph in background thread
    def _run_graph():
        try:
            return agent_graph.invoke(initial_state, config=config)
        finally:
            event_queue.put(None)  # Sentinel to unblock consumer

    graph_task = asyncio.to_thread(_run_graph)
    graph_future = asyncio.create_task(graph_task)

    full_reply = ""
    executed_tool_steps: list[dict[str, Any]] = []
    # Buffered sliding-window stream validator prevents PII and credentials from leaking mid-token
    stream_buffer = StreamingOutputBuffer(buffer_size=40)

    # Stream events from LangGraph execution in real-time
    while True:
        if request and await request.is_disconnected():
            logger.info("Client disconnected from SSE stream.")
            break

        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            if graph_future.done():
                # Drain any remaining events before exiting
                while not event_queue.empty():
                    ev = event_queue.get_nowait()
                    if ev is not None:
                        if ev.get("type") == "token":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            delta = ev.get("delta", "")
                            full_reply += delta
                            safe_chunk = stream_buffer.push(delta)
                            if safe_chunk:
                                yield _format_sse("token", {"type": "token", "delta": safe_chunk})
                        elif ev.get("type") == "tool_end":
                            executed_tool_steps.append(ev)
                            yield _format_sse(ev.get("type", "message"), ev)
                        else:
                            yield _format_sse(ev.get("type", "message"), ev)
                break
            await asyncio.sleep(0.01)
            continue

        if event is None:
            break

        ev_type = event.get("type", "message")
        if ev_type == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter()
            delta = event.get("delta", "")
            full_reply += delta
            safe_chunk = stream_buffer.push(delta)
            if safe_chunk:
                yield _format_sse("token", {"type": "token", "delta": safe_chunk})
            continue
        elif ev_type == "tool_end":
            executed_tool_steps.append(event)

        yield _format_sse(ev_type, event)

        if ev_type in ("hitl_required", "error"):
            return

    # Flush any remaining validated content from the sliding buffer
    flushed_tail = stream_buffer.flush()
    if flushed_tail:
        yield _format_sse("token", {"type": "token", "delta": flushed_tail})

    # Await graph completion
    graph_result = await graph_future

    # Extract final assistant reply from LangGraph state
    final_reply = ""
    for msg in reversed(graph_result.get("messages", [])):
        r = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if r in ("assistant", "ai"):
            final_reply = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            break

    from src.agent.pipeline_common import clean_final_reply
    from src.agent.tool_parser import strip_unwanted_tool_xml
    from src.security.output_guards import sanitize_output

    final_reply = strip_unwanted_tool_xml(final_reply or full_reply).strip()
    if not final_reply and executed_tool_steps:
        final_reply = clean_final_reply("", executed_tool_steps)
    if not final_reply:
        final_reply = "I completed the requested action."

    # Phase 3 & 5: Enforce strict URL evidence grounding, claim verification, and input-output grounding on final stream output
    evidence_urls = graph_result.get("evidence_urls", []) if isinstance(graph_result, dict) else []
    evidence_items = graph_result.get("evidence_items", []) if isinstance(graph_result, dict) else []
    detected_intent = graph_result.get("detected_intent") if isinstance(graph_result, dict) else None
    evidence_verification = graph_result.get("evidence_verification") if isinstance(graph_result, dict) else None
    resolved_tenant_id = tenant_id if tenant_id and tenant_id != "default" else (graph_result.get("tenant_id") if isinstance(graph_result, dict) else "default") or "default"

    try:
        final_reply = sanitize_output(
            final_reply,
            evidence_urls=evidence_urls,
            evidence_items=evidence_items,
            prompt=message,
            intent_category=detected_intent,
        )
    except Exception as sant_err:
        logger.warning("output_sanitization_failed_using_raw: %s", sant_err, exc_info=True)

    if root_span:
        try:
            root_span.end(output=final_reply)
        except Exception:
            pass

    total_latency_ms = round((time.perf_counter() - workflow_start) * 1000, 1)
    ttft_ms = round((first_token_time - workflow_start) * 1000, 1) if first_token_time else None
    model_name = agent_router.last_dispatched_model or "groq/qwen/qwen3.8-27b"
    task_type = agent_router.last_detected_task or "general"

    # Persist final turn with telemetry and executed tools to PostgreSQL
    metadata = {
        "model": model_name,
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "task_type": task_type,
        "detected_intent": detected_intent,
        "evidence_verification": evidence_verification,
    }

    # FinOps Subsystem Metrics Recording (isolated to prevent interrupting user stream)
    try:
        from src.security.finops import FinOpsMetricRecord, finops
        prompt_tokens = max(1, len(message) // 4)
        completion_tokens = max(1, len(final_reply) // 4)
        total_tokens = prompt_tokens + completion_tokens
        tool_count = len(executed_tool_steps) if executed_tool_steps else 0
        grounding_retries_count = graph_result.get("grounding_retries", 0) if isinstance(graph_result, dict) else 0
        unverified_count = 1 if ("[Unverified link `" in final_reply or "[Warning: Contradicted" in final_reply) else 0

        finops_rec = FinOpsMetricRecord(
            request_id=f"req_stream_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            user_id=user_id,
            tenant_id=resolved_tenant_id,
            model_name=model_name,
            ttft_ms=ttft_ms,
            end_to_end_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=tool_count,
            grounding_retries=grounding_retries_count,
            cache_hit=False,
            grounding_failures_count=unverified_count,
            total_claims_count=max(1, len(final_reply.split(". "))),
            task_completed=True,
        )
        finops.record_request_metrics(finops_rec)
        metadata["cost_usd"] = finops_rec.cost_usd
    except Exception as finops_err:
        logger.warning("finops_metric_recording_failed: %s", finops_err, exc_info=True)

    try:
        chat_history_db.append_message(
            session_id=session_id,
            role="assistant",
            content=final_reply,
            tool_calls=executed_tool_steps if executed_tool_steps else None,
            metadata=metadata,
            user_id=user_id,
            tenant_id=resolved_tenant_id,
        )
    except Exception as hist_err:
        logger.warning("failed_to_persist_assistant_message_to_history: %s", hist_err, exc_info=True)

    yield _format_sse("done", {
        "reply": final_reply,
        "iterations": graph_result.get("iteration", 1),
        "session_id": session_id,
        "model": model_name,
        "task_type": task_type,
        "detected_intent": detected_intent,
        "evidence_verification": evidence_verification,
        "latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
    })
