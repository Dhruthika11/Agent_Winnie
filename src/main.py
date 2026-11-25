"""
FastAPI entrypoint for the Hermes-like agent service.

Endpoints:
    POST /chat                          -> run one turn of the agent loop
    GET  /health                        -> liveness/readiness probe
    GET  /memories/{user_id}            -> inspect stored long-term memories
    DELETE /memories/{user_id}/{mem_id} -> delete a stored memory
    GET  /approvals/{approval_id}       -> check a pending HITL approval
    POST /approvals/{approval_id}       -> approve/reject and resume the turn
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import time
from pathlib import Path
from typing import Any, Optional
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.agent.graph import agent_graph, resume_approval
from src.agent.pipeline_common import create_tool_call_record
from src.agent.router import agent_router
from src.agent.task_queue import task_queue
from src.db.chat_history import chat_history_db
from src.memory.manager import memory_manager
from src.security.auth import UserAuth, create_access_token, get_current_user, require_role
from src.security.finops import finops, FinOpsMetricRecord
from src.security.rate_limiter import rate_limit_dependency
from src.security.secrets import secrets_manager
from src.tools import hitl
from src.tools.mcp_client import mcp_registry
from src.config import settings
from src.utils.logger import configure_logging
from src.utils.telemetry import get_langfuse_client, setup_telemetry

configure_logging()
logger = logging.getLogger("hermes.main")

from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    secrets_manager.load_secrets()
    setup_telemetry(app)
    mcp_registry.warmup()
    try:
        chat_history_db.init_db()
    except Exception as exc:
        logger.warning(f"Chat history DB init failed: {exc}")
    try:
        from src.learning.database import failure_db
        failure_db.init_db()
    except Exception as exc:
        logger.warning(f"Failure DB init failed: {exc}")
    try:
        from src.agent.durable import durable_store
        durable_store.init_db()
    except Exception as exc:
        logger.warning(f"Durable store DB init failed: {exc}")
    logger.info("startup_complete")
    yield
    try:
        chat_history_db.close()
    except Exception:
        pass
    try:
        from src.learning.database import failure_db
        failure_db.close()
    except Exception:
        pass
    try:
        from src.agent.durable import durable_store
        durable_store.close()
    except Exception:
        pass
    mcp_registry.shutdown()
    logger.info("shutdown_complete")


app = FastAPI(title="Hermes Agent Platform", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    tenant_id: str | None = "default"
    message: str
    max_iterations: int | None = None
    images: list[str] = []  # Image URLs or base64 data URIs for Multimodal / Vision analysis


class RegisterUserRequest(BaseModel):
    username: str
    password: str
    email: str | None = None


class LoginUserRequest(BaseModel):
    username: str
    password: str


class UpdateConversationRequest(BaseModel):
    title: str


class TokenRequest(BaseModel):
    user_id: str
    role: str = "viewer"
    tenant_id: str = "default"


class SubmitTaskRequest(BaseModel):
    task_type: str = "chat"
    payload: dict[str, Any] = {}


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    reply: str
    stop_reason: str | None
    awaiting_approval: bool
    approval_id: str | None
    model: str | None = None
    task_type: str | None = None
    latency_ms: float | None = None


class ApprovalDecision(BaseModel):
    approve: bool
    decided_by: str = "unknown"


class ApprovalRecord(BaseModel):
    approval_id: str
    tool_name: str
    command: Optional[str] = None
    arguments: dict[str, Any] = {}
    working_directory: Optional[str] = None
    affected_files: list[str] = []
    risk_level: str = "high"
    reason: str = "Operation requires human authorization"
    status: str = "pending"  # "pending" | "approved" | "rejected" | "expired" | "executed"
    tenant_id: str = "default"
    user_id: str = "default"
    created_at: float
    expires_at: float
    session_id: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: Optional[float] = None
    executed_at: Optional[float] = None


class ExecuteToolRequest(BaseModel):
    tool: str
    arguments: dict[str, Any] = {}
    idempotency_key: str | None = None
    user_id: str = "default"
    session_id: str = "default"
    tenant_id: str = "default"
    user_role: str = "viewer"
    approval_id: str | None = None


class ReconcileRequest(BaseModel):
    stale_threshold_seconds: float = 60.0
    tenant_id: Optional[str] = None


class CancelExecutionRequest(BaseModel):
    reason: str = "Operator cancelled"


class RegisterToolServerRequest(BaseModel):
    name: str
    transport: str = "sse"
    command: str = "python"
    args: list[str] = []
    env: dict[str, str] | None = None
    base_url: str = ""
    risk_level: str = "low"
    tools: list[dict] = []
    timeout_seconds: float = 20.0


class TripCircuitRequest(BaseModel):
    reason: str = "Manual operator override"


@app.get("/", response_class=HTMLResponse)
@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui_page():
    html_file = WEB_DIR / "chat.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Hermes Chatbot - Web UI not found</h1>", status_code=404)


@app.get("/mcp", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def mcp_hub_page():
    html_file = WEB_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Hermes Agent Platform - MCP Hub not found</h1>", status_code=404)


@app.get("/assets/{filename}")
def get_asset(filename: str):
    file_path = WEB_DIR / "assets" / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/api/mcp/servers")
def get_mcp_servers():
    return {"servers": mcp_registry.get_servers_info()}


@app.get("/api/mcp/ping/{server_name}")
def ping_mcp_server(server_name: str):
    try:
        return mcp_registry.ping_server(server_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/mcp/execute")
def execute_mcp_tool(req: ExecuteToolRequest):
    start = time.perf_counter()
    try:
        res = mcp_registry.call_tool(req.tool, req.arguments)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "success", "result": res, "duration_ms": duration_ms}
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("mcp_web_execution_failed", extra={"tool": req.tool})
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {str(exc)}")


# --- Centralized Tool Execution Engine API ---

@app.post("/api/tools/execute")
def execute_tool_endpoint(
    req: ExecuteToolRequest,
    current_user: UserAuth = Depends(get_current_user),
):
    """
    Execute an arbitrary tool through the centralized Tool Execution Engine.
    Enforces idempotency, input validation, RBAC, tool policy, HITL, circuit breaker, and audit logging.
    """
    from src.tools.execution_engine import ToolExecutionRequest, tool_execution_engine

    role = req.user_role if (not settings.auth_enabled and req.user_role) else (current_user.role if current_user else req.user_role)
    tenant = current_user.tenant_id if current_user else req.tenant_id
    user_id = current_user.user_id if current_user else req.user_id

    engine_req = ToolExecutionRequest(
        tool_name=req.tool,
        arguments=req.arguments,
        user_id=user_id,
        session_id=req.session_id,
        tenant_id=tenant,
        user_role=role,
        idempotency_key=req.idempotency_key,
        approval_id=req.approval_id,
    )
    result = tool_execution_engine.execute(engine_req)
    return result.to_dict()


# --- Durable Execution & Crash Recovery API ---

@app.get("/api/recovery/executions")
def list_durable_executions(
    tenant_id: Optional[str] = None,
    limit: int = 50,
    current_user: UserAuth = Depends(get_current_user),
):
    """List durable agent execution records scoped by tenant."""
    from src.agent.durable import durable_store
    effective_tenant = current_user.tenant_id if (current_user and current_user.role != "admin") else tenant_id
    records = durable_store.list_executions(tenant_id=effective_tenant, limit=limit)
    return {"executions": [r.to_dict() for r in records]}


@app.get("/api/recovery/executions/{execution_id}")
def get_durable_execution(
    execution_id: str,
    current_user: UserAuth = Depends(get_current_user),
):
    """Retrieve full execution lifecycle state."""
    from src.agent.durable import durable_store
    rec = durable_store.get_execution(execution_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    if current_user and current_user.role != "admin" and rec.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access forbidden")
    return rec.to_dict()


@app.get("/api/recovery/executions/{execution_id}/checkpoints")
def get_execution_checkpoints(
    execution_id: str,
    current_user: UserAuth = Depends(get_current_user),
):
    """Retrieve all incremental step checkpoints for an execution."""
    from src.agent.durable import durable_store
    rec = durable_store.get_execution(execution_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    if current_user and current_user.role != "admin" and rec.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access forbidden")
    checkpoints = durable_store.get_checkpoints(execution_id)
    return {"execution_id": execution_id, "checkpoints": [c.to_dict() for c in checkpoints]}


@app.post("/api/recovery/reconcile")
def reconcile_stale_executions_endpoint(
    req: ReconcileRequest = ReconcileRequest(),
    current_user: UserAuth = Depends(require_role("operator", "admin")),
):
    """Scan and triage interrupted or stale executions across workers."""
    from src.agent.recovery import recovery_manager
    effective_tenant = current_user.tenant_id if (current_user and current_user.role != "admin") else req.tenant_id
    stale = recovery_manager.reconcile_stale_executions(
        stale_threshold_seconds=req.stale_threshold_seconds,
        tenant_id=effective_tenant,
    )
    return {"status": "reconciliation_complete", "stale_count": len(stale), "stale_records": stale}


@app.post("/api/recovery/resume/{execution_id}")
def resume_execution_endpoint(
    execution_id: str,
    current_user: UserAuth = Depends(require_role("operator", "admin")),
):
    """Resume an interrupted execution from its latest persistent checkpoint."""
    from src.agent.durable import durable_store
    rec = durable_store.get_execution(execution_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    if current_user and current_user.role != "admin" and rec.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access forbidden")

    from src.agent.recovery import recovery_manager
    result = recovery_manager.resume_execution(execution_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Resume failed"))
    return result


@app.post("/api/recovery/cancel/{execution_id}")
def cancel_execution_endpoint(
    execution_id: str,
    req: CancelExecutionRequest = CancelExecutionRequest(),
    current_user: UserAuth = Depends(require_role("operator", "admin")),
):
    """Cancel an interrupted or running execution."""
    from src.agent.durable import durable_store
    rec = durable_store.get_execution(execution_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    if current_user and current_user.role != "admin" and rec.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access forbidden")

    from src.agent.recovery import recovery_manager
    ok = recovery_manager.cancel_execution(execution_id, reason=req.reason)
    return {"status": "cancelled", "execution_id": execution_id, "success": ok}


# --- RBAC Permissions API ---

@app.get("/api/rbac/permissions")
def get_user_permissions(current_user: UserAuth = Depends(get_current_user)):
    """Return effective permissions and role for authenticated user."""
    from src.security.rbac import ROLE_PERMISSIONS, rbac_engine
    norm_role = rbac_engine.normalize_role(current_user.role)
    perms = [p.value for p in ROLE_PERMISSIONS.get(norm_role, set())]
    return {
        "user_id": current_user.user_id,
        "role": norm_role,
        "tenant_id": current_user.tenant_id,
        "permissions": perms,
    }


# --- Distributed Tool Registry & Circuit Breaker API ---

@app.get("/api/tools/circuits")
def get_all_circuits():
    """Return circuit-breaking health status across all registered tool servers."""
    return mcp_registry.get_circuit_status()


@app.get("/api/tools/circuits/{server_name}")
def get_server_circuit(server_name: str):
    """Return circuit breaker state for a specific server."""
    status = mcp_registry.get_circuit_status(server_name)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@app.post("/api/tools/circuits/{server_name}/reset")
def reset_server_circuit(server_name: str):
    """Manually reset a tripped circuit breaker back to CLOSED."""
    ok = mcp_registry.reset_circuit(server_name)
    return {"status": "reset", "server": server_name, "success": ok}


@app.post("/api/tools/circuits/{server_name}/trip")
def trip_server_circuit(server_name: str, req: TripCircuitRequest = TripCircuitRequest()):
    """Manually trip a server's circuit breaker to OPEN (for maintenance or testing)."""
    ok = mcp_registry.trip_circuit(server_name, reason=req.reason)
    return {"status": "tripped", "server": server_name, "reason": req.reason, "success": ok}


@app.post("/api/tools/register")
def register_tool_server(req: RegisterToolServerRequest):
    """Dynamically register a new distributed tool microservice into the registry."""
    cfg = mcp_registry.register_server(req.model_dump())
    return {
        "status": "registered",
        "server": cfg.name,
        "transport": cfg.transport,
        "tools_count": len(cfg.tools),
    }


@app.delete("/api/tools/{server_name}")
def deregister_tool_server(server_name: str):
    """Dynamically deregister a tool microservice from the registry."""
    ok = mcp_registry.deregister_server(server_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    return {"status": "deregistered", "server": server_name}


# --- FinOps & Prometheus Metrics ---
_metrics = {
    "chat_requests_total": 0,
    "chat_stream_requests_total": 0,
    "tool_executions_total": 0,
    "tokens_consumed_total": 0,
}


@app.get("/metrics")
def get_prometheus_metrics():
    """Prometheus metrics exposition endpoint for FinOps and observability."""
    import datetime
    today = datetime.date.today().isoformat()
    circuit_data = mcp_registry.get_circuit_status().get("circuits", {})
    total_trips = sum(c.get("total_trips", 0) for c in circuit_data.values())

    lines = [
        "# HELP hermes_chat_requests_total Cumulative /chat invocation count",
        "# TYPE hermes_chat_requests_total counter",
        f"hermes_chat_requests_total {_metrics['chat_requests_total']}",
        "# HELP hermes_chat_stream_requests_total Cumulative /chat/stream invocation count",
        "# TYPE hermes_chat_stream_requests_total counter",
        f"hermes_chat_stream_requests_total {_metrics['chat_stream_requests_total']}",
        "# HELP hermes_tool_executions_total Total tool calls dispatched",
        "# TYPE hermes_tool_executions_total counter",
        f"hermes_tool_executions_total {_metrics['tool_executions_total']}",
        "# HELP hermes_tokens_consumed_total Total tokens tracked across LLM requests",
        "# TYPE hermes_tokens_consumed_total counter",
        f"hermes_tokens_consumed_total {_metrics['tokens_consumed_total']}",
        "# HELP hermes_circuit_trips_total Total circuit breaker trips across all tool servers",
        "# TYPE hermes_circuit_trips_total counter",
        f"hermes_circuit_trips_total {total_trips}",
    ]
    for s_name, s_data in circuit_data.items():
        state_val = 1 if s_data.get("state") == "CLOSED" else 0
        lines.append(f'hermes_circuit_healthy{{server="{s_name}"}} {state_val}')

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# --- Kubernetes Liveness and Readiness Probes ---

@app.get("/health")
@app.get("/health/live")
def liveness():
    """Kubernetes Liveness Probe: returns 200 if Uvicorn event loop is responsive."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready")
def readiness():
    """Kubernetes Readiness Probe: deep healthcheck of Redis, Postgres, Qdrant, and MCP tool pool."""
    checks = {}
    errors = []

    # 1. Redis Check
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url, socket_timeout=1.0)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as exc:
        checks["redis"] = f"unhealthy ({exc.__class__.__name__})"
        errors.append(f"Redis: {exc}")

    # 2. Vector Store / Qdrant Check
    try:
        import httpx
        with httpx.Client(timeout=1.0) as client:
            q_url = settings.mem0_vector_store_url.rstrip("/")
            resp = client.get(f"{q_url}/readyz")
            checks["qdrant"] = "healthy" if resp.status_code == 200 else f"code_{resp.status_code}"
    except Exception as exc:
        checks["qdrant"] = f"unhealthy ({exc.__class__.__name__})"
        # Qdrant optional warning, not fatal if mock or offline

    # 3. MCP Tools & Circuit Breaker Status
    try:
        circuits = mcp_registry.get_circuit_status().get("circuits", {})
        open_circuits = [s for s, c in circuits.items() if c.get("state") == "OPEN"]
        if open_circuits:
            checks["mcp_tools"] = f"degraded (circuits open: {open_circuits})"
        else:
            checks["mcp_tools"] = "healthy"
    except Exception as exc:
        checks["mcp_tools"] = f"unhealthy ({exc.__class__.__name__})"

    if errors:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks, "errors": errors},
        )
    return {"status": "ready", "checks": checks}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    auth_user: UserAuth = Depends(get_current_user),
    _rl=Depends(rate_limit_dependency()),
):
    import datetime
    import time
    chat_start = time.perf_counter()
    _metrics["chat_requests_total"] += 1

    effective_user_id = auth_user.user_id if auth_user.user_id != "dev_user" else req.user_id
    effective_tenant_id = auth_user.tenant_id if auth_user.tenant_id != "default" else (req.tenant_id or "default")

    # --- 1. Idempotency Key Handling ---
    idempotency_key = request.headers.get("X-Idempotency-Key")
    redis_client = None
    if getattr(settings, "redis_url", ""):
        try:
            import redis
            redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=1.0)
            if idempotency_key:
                cached_resp = redis_client.get(f"hermes:idempotency:{effective_tenant_id}:{idempotency_key}")
                if cached_resp:
                    logger.info(f"[IDEMPOTENCY HIT] Returning cached response for key '{idempotency_key}'")
                    return ChatResponse.model_validate_json(cached_resp)
        except Exception:
            redis_client = None

    # --- 2. FinOps Spending Cap & Graceful Degradation Policy ---
    budget_status, used_tokens, limit_tokens = finops.evaluate_budget(effective_user_id, effective_tenant_id)
    if budget_status == "blocked":
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily token budget exceeded ({used_tokens}/{limit_tokens} tokens). "
                "Limit resets at midnight UTC."
            ),
        )
    elif budget_status == "degraded":
        logger.info(
            f"[FinOps Degradation] User '{effective_user_id}' has reached 80% daily budget. "
            "Routing to cost-efficient models."
        )

    session_id = req.session_id or str(uuid.uuid4())

    lf = get_langfuse_client()
    trace = None
    trace_id = None
    if lf and hasattr(lf, "trace"):
        try:
            trace = lf.trace(
                name="chat_turn",
                session_id=session_id,
                user_id=req.user_id,
                input=req.message,
            )
            trace_id = trace.id
        except Exception:
            logger.warning("langfuse_trace_creation_failed", exc_info=True)

    root_span = None
    if trace and lf:
        try:
            root_span = lf.span(
                trace_id=trace.id,
                name="agent_workflow",
                input={"user_id": req.user_id, "session_id": session_id, "message": req.message},
            )
        except Exception:
            pass

    exec_id = f"exec_{uuid.uuid4().hex[:12]}"
    initial_state = {
        "messages": [{"role": "user", "content": req.message}],
        "user_id": effective_user_id,
        "session_id": session_id,
        "tenant_id": effective_tenant_id,
        "user_role": getattr(auth_user, "role", "viewer") if "auth_user" in locals() else "viewer",
        "execution_id": exec_id,
        "trace_id": trace_id,
        "parent_span_id": root_span.id if root_span else None,
        "iteration": 0,
        "max_iterations": req.max_iterations or 8,
        "memory_context": "",
        "pending_tool_calls": [],
        "awaiting_approval": False,
        "approval_id": None,
        "grounding_retries": 0,
        "done": False,
        "stop_reason": None,
    }

    try:
        from src.agent.durable import ExecutionLifecycleState, durable_store
        durable_store.record_transition(
            execution_id=exec_id,
            state_from=None,
            state_to=ExecutionLifecycleState.REQUEST_RECEIVED.value,
            state_payload={"messages": initial_state["messages"]},
            session_id=session_id,
            user_id=effective_user_id,
            tenant_id=effective_tenant_id,
        )
    except Exception:
        pass

    config = {"configurable": {"thread_id": session_id}}
    try:
        result = agent_graph.invoke(initial_state, config=config)
    except Exception as exc:
        if root_span:
            try:
                root_span.end(level="ERROR", status_message=str(exc))
            except Exception:
                pass
        logger.exception("chat_turn_failed", extra={"error": str(exc), "user_id": effective_user_id})
        raise HTTPException(
            status_code=500,
            detail=f"Agent loop error: {exc}",
        )

    reply = ""
    for msg in reversed(result["messages"]):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("assistant", "ai"):
            reply = msg.get("content") if isinstance(msg, dict) else msg.content
            break

    from src.agent.pipeline_common import clean_final_reply
    from src.agent.tool_parser import strip_unwanted_tool_xml

    reply = strip_unwanted_tool_xml(reply).strip()
    if not reply:
        reply = "I completed the requested action."

    if root_span:
        try:
            root_span.end(output=reply)
        except Exception:
            pass

    if trace and lf:
        try:
            trace.update(output=reply)
            lf.flush()
        except Exception:
            logger.warning("langfuse_trace_flush_failed", exc_info=True)

    # FinOps Token Accounting & 9-Dimension Request Telemetry
    estimated_tokens = max(1, (len(req.message) + len(reply)) // 4)
    _metrics["tokens_consumed_total"] += estimated_tokens

    try:
        prompt_tokens = max(1, len(req.message) // 4)
        completion_tokens = max(1, len(reply) // 4)
        total_tokens = prompt_tokens + completion_tokens
        lat_ms = round((time.perf_counter() - chat_start) * 1000, 1)
        m_name = agent_router.last_dispatched_model or "groq/qwen/qwen3.8-27b"
        g_retries = result.get("grounding_retries", 0)
        unverified_cnt = 1 if ("[Unverified link `" in reply or "[Warning: Contradicted" in reply) else 0

        finops_rec = FinOpsMetricRecord(
            request_id=f"req_sync_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            user_id=effective_user_id,
            tenant_id=effective_tenant_id,
            model_name=m_name,
            ttft_ms=round(lat_ms * 0.45, 1),
            end_to_end_latency_ms=lat_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=len(tool_steps),
            grounding_retries=g_retries,
            cache_hit=False,
            grounding_failures_count=unverified_cnt,
            total_claims_count=max(1, len(reply.split(". "))),
            task_completed=True,
        )
        finops.record_request_metrics(finops_rec)
    except Exception:
        finops.record_usage(effective_user_id, estimated_tokens, effective_tenant_id)

    # Save conversation turn to persistent PostgreSQL chat history using standard versioned schema
    tool_steps = []
    for msg in result.get("messages", []):
        r = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if r in ("tool", "tool_message"):
            c_id = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
            t_name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "tool")
            t_out = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            tool_steps.append(
                create_tool_call_record(
                    call_id=c_id,
                    tool_name=t_name,
                    arguments={},
                    output=t_out,
                    preview=t_out[:300] + ("..." if len(t_out) > 300 else ""),
                    status="error" if "error" in t_out.lower() else "success",
                )
            )

    metadata = {
        "model": agent_router.last_dispatched_model or "groq/qwen/qwen3.8-27b",
        "latency_ms": round((time.perf_counter() - chat_start) * 1000, 1),
        "task_type": agent_router.last_detected_task or "general",
    }

    try:
        chat_history_db.append_message(
            session_id=session_id,
            role="user",
            content=req.message,
            user_id=effective_user_id,
            tenant_id=effective_tenant_id,
        )
        chat_history_db.append_message(
            session_id=session_id,
            role="assistant",
            content=reply,
            tool_calls=tool_steps if tool_steps else None,
            metadata=metadata,
            user_id=effective_user_id,
            tenant_id=effective_tenant_id,
        )
    except Exception as exc:
        logger.warning(f"Failed to persist chat turn to database: {exc}")

    response_obj = ChatResponse(
        user_id=effective_user_id,
        session_id=session_id,
        reply=reply,
        stop_reason=result.get("stop_reason"),
        awaiting_approval=result.get("awaiting_approval", False),
        approval_id=result.get("approval_id"),
        model=agent_router.last_dispatched_model,
        task_type=agent_router.last_detected_task,
        latency_ms=round((time.perf_counter() - chat_start) * 1000, 1),
    )

    # Save to Redis for Idempotency
    if redis_client and idempotency_key:
        try:
            redis_client.setex(f"hermes:idempotency:{idempotency_key}", 3600, response_obj.model_dump_json())
        except Exception:
            pass

    return response_obj


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    auth_user: UserAuth = Depends(get_current_user),
):
    """
    Real-Time Server-Sent Events (SSE) streaming endpoint.
    Streams token-by-token text deltas and tool execution updates.
    Supports multimodal queries (text + images) and client disconnect handling.
    """
    from src.agent.stream_engine import stream_agent_workflow

    session_id = req.session_id or str(uuid.uuid4())
    effective_user_id = auth_user.user_id if auth_user.user_id != "dev_user" else (req.user_id or "web_visitor")
    effective_tenant_id = auth_user.tenant_id if getattr(auth_user, "tenant_id", None) and auth_user.tenant_id != "default" else (req.tenant_id or "default")
    generator = stream_agent_workflow(
        message=req.message,
        user_id=effective_user_id,
        session_id=session_id,
        tenant_id=effective_tenant_id,
        max_iterations=req.max_iterations or 8,
        images=req.images,
        request=request,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/multimodal", response_model=ChatResponse)
def chat_multimodal(req: ChatRequest):
    """
    Dedicated Multimodal API endpoint powered by Google Gemini Vision.
    Processes mixed input: text prompts plus images/screenshots.
    Includes active integration placeholder when GOOGLE_API_KEY is not yet set.
    """
    from src.agent.router import agent_router

    session_id = req.session_id or str(uuid.uuid4())

    # Build multimodal content payload for LiteLLM / Gemini
    content: Any = req.message
    if req.images:
        content = [{"type": "text", "text": req.message}]
        for img in req.images:
            content.append({"type": "image_url", "image_url": {"url": img}})

    if agent_router.is_multimodal_configured:
        try:
            logger.info("Executing Google Gemini multimodal completion", extra={"images_count": len(req.images)})
            resp = agent_router.completion(
                model="winnie-multimodal-model",
                messages=[{"role": "user", "content": content}],
                max_tokens=3000,
            )
            reply = resp.choices[0].message.content or ""
            return ChatResponse(
                user_id=req.user_id,
                session_id=session_id,
                reply=reply,
                stop_reason="multimodal_complete",
                awaiting_approval=False,
                approval_id=None,
            )
        except Exception as exc:
            logger.exception("multimodal_gemini_call_failed", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail=f"Google Gemini Multimodal API error: {exc}")
    else:
        # Informative placeholder guiding the user to activate their Google Gemini Key
        placeholder_reply = (
            "✨ **Google Gemini Multimodal Vision API (Placeholder)**\n\n"
            f"Winnie received your multimodal request with **{len(req.images)} attached image(s)** and prompt:\n"
            f"> *\"{req.message}\"*\n\n"
            "To activate live vision and audio multimodal inference, add your free Google AI Studio API key to `.env`:\n"
            "```dotenv\n"
            "GOOGLE_API_KEY=AIzaSy...\n"
            "GEMINI_MULTIMODAL_MODEL=gemini/gemini-1.5-flash\n"
            "```\n"
            "👉 Get your free key at: [Google AI Studio](https://aistudio.google.com/)."
        )
        return ChatResponse(
            user_id=req.user_id,
            session_id=session_id,
            reply=placeholder_reply,
            stop_reason="multimodal_key_placeholder",
            awaiting_approval=False,
            approval_id=None,
        )




@app.get("/memories/{user_id}")
def list_memories(user_id: str):
    return {"memories": [m.model_dump() for m in memory_manager.get_all(user_id)]}


@app.delete("/memories/{user_id}/{memory_id}")
def delete_memory(user_id: str, memory_id: str):
    ok = memory_manager.delete(user_id, memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found or backend unavailable")
    return {"status": "deleted", "memory_id": memory_id}


# --------------------------------------------------------------------------- #
# Memory Governance & Lifecycle Management APIs
# --------------------------------------------------------------------------- #
class GovernedMemoryAddRequest(BaseModel):
    user_id: str
    content: str
    tenant_id: str = "default"
    category: Optional[str] = None  # "preference" | "fact" | "project" | "temporary_context"
    confidence: float = 1.0
    ttl_days: Optional[int] = None
    source: str = "user"
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class GovernedMemoryUpdateRequest(BaseModel):
    user_id: str
    new_text: str
    tenant_id: str = "default"
    category: Optional[str] = None
    confidence: Optional[float] = None


@app.get("/api/memory")
def get_governed_memories(
    user_id: str,
    tenant_id: str = "default",
    category: Optional[str] = None,
    include_expired: bool = False,
    include_inactive: bool = False,
):
    """Retrieve memories scoped strictly by user_id and tenant_id with governance metadata."""
    items = memory_manager.get_all(
        user_id=user_id,
        tenant_id=tenant_id,
        category=category,
        include_expired=include_expired,
        include_inactive=include_inactive,
    )
    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "count": len(items),
        "memories": [item.to_dict() for item in items],
    }


@app.post("/api/memory")
def add_governed_memory(req: GovernedMemoryAddRequest):
    """Store memory through the governed lifecycle (classification, contradiction resolution, deduplication)."""
    results = memory_manager.add(
        user_id=req.user_id,
        messages=[{"role": "user", "content": req.content}],
        tenant_id=req.tenant_id,
        source=req.source,
        category=req.category,
        ttl_days=req.ttl_days,
        confidence=req.confidence,
        session_id=req.session_id,
        metadata=req.metadata,
    )
    return {"results": [r.model_dump() for r in results]}


@app.put("/api/memory/{memory_id}")
def update_governed_memory(memory_id: str, req: GovernedMemoryUpdateRequest):
    """User-controlled correction of an existing memory item."""
    updated = memory_manager.update(
        memory_id=memory_id,
        new_text=req.new_text,
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        category=req.category,
        confidence=req.confidence,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found or not owned by user/tenant")
    return {"status": "updated", "memory": updated.to_dict()}


@app.delete("/api/memory/{memory_id}")
def delete_governed_memory(memory_id: str, user_id: str, tenant_id: str = "default"):
    """Delete a memory item with strict tenant boundary enforcement."""
    ok = memory_manager.delete(user_id=user_id, memory_id=memory_id, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found or not owned by user/tenant")
    return {"status": "deleted", "memory_id": memory_id, "tenant_id": tenant_id}


# --------------------------------------------------------------------------- #
# Evidence and Provenance Grounding APIs
# --------------------------------------------------------------------------- #
class EvidenceVerifyRequest(BaseModel):
    text: str
    evidence_items: Optional[list[dict[str, Any]]] = None


@app.get("/api/evidence")
def list_evidence(
    provenance_type: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 50,
):
    """List stored evidence items with content hashes, URLs, and provenance."""
    from src.security.evidence_store import evidence_store
    items = evidence_store.list_evidence(provenance_type=provenance_type, tenant_id=tenant_id, limit=limit)
    return {"count": len(items), "evidence": [e.to_dict() for e in items]}


@app.delete("/api/evidence")
def clear_evidence_store():
    """Clear all stored evidence items and caches for fresh testing."""
    from src.security.evidence_store import evidence_store
    evidence_store.clear()
    try:
        from src.config import settings
        import redis
        r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        keys = r.keys("hermes:evidence:*") + r.keys("hermes:search:*") + r.keys("hermes:idempotency:*")
        if keys:
            r.delete(*keys)
    except Exception:
        pass
    return {"status": "cleared", "message": "Evidence store and search caches successfully cleared."}


@app.get("/api/evidence/{evidence_id}")
def get_evidence_item(evidence_id: str):
    """Fetch an individual evidence item by ID."""
    from src.security.evidence_store import evidence_store
    item = evidence_store.get_evidence(evidence_id)
    if not item:
        raise HTTPException(status_code=404, detail="Evidence item not found")
    return item.to_dict()


@app.post("/api/evidence/verify")
def verify_evidence_grounding(req: EvidenceVerifyRequest):
    """Execute the full Grounding Pipeline on response text against evidence store.

    Extracts atomic claims and classifies each as SUPPORTED, CONTRADICTED, or UNKNOWN.
    """
    from src.security.evidence_store import verify_response_grounding
    result = verify_response_grounding(req.text, evidence_items=req.evidence_items)
    return result.to_dict()


# --------------------------------------------------------------------------- #
# FinOps & Performance Optimization APIs (9 Core Metrics)
# --------------------------------------------------------------------------- #
@app.get("/api/finops/analytics")
def get_finops_analytics(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 500,
):
    """Aggregated performance and cost analytics across all 9 core dimensions:

    TTFT, latency, tokens, cost, tool calls, retry rate, cache hit rate,
    grounding failure rate, and task completion rate.
    """
    from src.security.finops import finops
    analytics = finops.get_performance_and_cost_analytics(user_id=user_id, tenant_id=tenant_id, limit=limit)
    return analytics


@app.get("/api/finops/records")
def list_finops_records(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 50,
):
    """List recent per-request FinOps telemetry traces."""
    from src.security.finops import finops
    records = finops.list_records(user_id=user_id, tenant_id=tenant_id, limit=limit)
    return {
        "count": len(records),
        "records": [r.to_dict() for r in records],
    }


@app.get("/api/finops/budget")
def get_finops_budget(user_id: str, tenant_id: str = "default"):
    """Evaluate spending caps and graceful degradation status for user/tenant."""
    from src.security.finops import finops
    status, used, limit = finops.evaluate_budget(user_id, tenant_id)
    return {
        "status": status,
        "tokens_used": used,
        "tokens_limit": limit,
        "percent_used": round((used / max(1, limit)) * 100, 2),
    }


@app.get("/approvals/pending")
def list_pending(auth_user: UserAuth = Depends(get_current_user)):
    tenant = auth_user.tenant_id if auth_user.role != "admin" else None
    return {"pending_approvals": hitl.list_pending_approvals(tenant_id=tenant)}


@app.get("/api/approvals")
def list_all_approvals(
    status: Optional[str] = None,
    auth_user: UserAuth = Depends(get_current_user),
):
    tenant = auth_user.tenant_id if auth_user.role != "admin" else None
    return {"approvals": hitl.list_approvals(tenant_id=tenant, status=status)}


@app.get("/approvals/{approval_id}", response_model=ApprovalRecord)
def get_approval(
    approval_id: str,
    auth_user: UserAuth = Depends(get_current_user),
):
    raw_record = hitl.get_approval(approval_id)
    if raw_record is None:
        raise HTTPException(status_code=404, detail="Approval not found")

    record = hitl.format_approval_record(raw_record)

    approval_tenant = record.get("tenant_id", "default")
    if auth_user.role != "admin" and auth_user.tenant_id != approval_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access forbidden: User tenant '{auth_user.tenant_id}' does not match approval tenant '{approval_tenant}'",
        )

    return ApprovalRecord(**record)


@app.post("/approvals/{approval_id}", response_model=ChatResponse)
def decide_approval(
    approval_id: str,
    decision: ApprovalDecision,
    auth_user: UserAuth = Depends(require_role("operator", "admin")),
):
    raw_record = hitl.get_approval(approval_id)
    if raw_record is None:
        raise HTTPException(status_code=404, detail="Approval not found")

    approval_tenant = raw_record.get("tenant_id", "default")
    if auth_user.role != "admin" and auth_user.tenant_id != approval_tenant:
        raise HTTPException(
            status_code=403,
            detail=f"Access forbidden: User tenant '{auth_user.tenant_id}' does not match approval tenant '{approval_tenant}'",
        )

    # Check expiration
    created_at = float(raw_record.get("created_at", 0))
    timeout = float(raw_record.get("timeout_seconds") or settings.hitl_approval_timeout_seconds)
    expires_at = float(raw_record.get("expires_at", created_at + timeout))
    now = time.time()

    if now > expires_at or raw_record.get("status") == "expired":
        hitl.mark_expired(approval_id)
        raise HTTPException(
            status_code=400,
            detail=f"Approval '{approval_id}' has expired and can no longer be decided or executed.",
        )

    # Prevent double-approval or re-decision
    current_status = raw_record.get("status")
    if current_status in ("approved", "rejected", "executed"):
        raise HTTPException(
            status_code=400,
            detail=f"Approval '{approval_id}' has already been {current_status}.",
        )

    decided_by = auth_user.user_id if auth_user.user_id != "dev_user" else decision.decided_by
    record = hitl.decide(approval_id, approve=decision.approve, decided_by=decided_by)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval not found")

    reply = ""
    stop_reason = "approved" if decision.approve else "rejected"
    awaiting_approval = False
    next_approval_id = None

    if record.get("state"):
        if decision.approve and auth_user.role in ("operator", "admin"):
            record["state"]["user_role"] = auth_user.role
        result = resume_approval(record, approve=decision.approve)
        for msg in reversed(result.get("messages", [])):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
            if role in ("assistant", "ai"):
                reply = msg.get("content") if isinstance(msg, dict) else msg.content
                break
        stop_reason = result.get("stop_reason") or stop_reason
        awaiting_approval = result.get("awaiting_approval", False)
        next_approval_id = result.get("approval_id")
    else:
        # Standalone tool execution without LangGraph state
        if decision.approve:
            tool_calls = record.get("tool_calls", [])
            if tool_calls:
                from src.tools.execution_engine import ToolExecutionRequest, tool_execution_engine
                tcall = tool_calls[0]
                tname = tcall.get("name") or tcall.get("tool", "unknown")
                targs = tcall.get("arguments", {})
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except Exception:
                        targs = {}
                engine_req = ToolExecutionRequest(
                    tool_name=tname,
                    arguments=targs,
                    user_id=record.get("user_id", auth_user.user_id),
                    session_id=record.get("session_id", "default"),
                    tenant_id=approval_tenant,
                    user_role=auth_user.role,
                    approval_id=approval_id,
                )
                exec_res = tool_execution_engine.execute(engine_req)
                if exec_res.is_success:
                    reply = f"Tool '{tname}' was approved and executed successfully.\n\nOutput:\n{exec_res.output}"
                    stop_reason = "executed"
                else:
                    err_msg = getattr(exec_res, "error", None) or getattr(exec_res, "error_message", None) or str(exec_res.output)
                    reply = f"Tool '{tname}' was approved, but execution failed: {err_msg}"
                    stop_reason = "execution_failed"
            else:
                reply = f"The requested action has been approved by {decided_by}."
                stop_reason = "approved"
        else:
            reply = f"The requested action was rejected by {decided_by}."
            stop_reason = "rejected"

    # If approved and finished execution, mark status = 'executed'
    if decision.approve and not awaiting_approval and stop_reason != "execution_failed":
        try:
            hitl.mark_executed(approval_id)
        except Exception:
            pass

    return ChatResponse(
        user_id=record.get("user_id", "unknown"),
        session_id=record.get("session_id", "default"),
        reply=reply,
        stop_reason=stop_reason,
        awaiting_approval=awaiting_approval,
        approval_id=next_approval_id,
    )


# --- Authentication & Token Minting ---

@app.post("/api/auth/token")
def mint_access_token(req: TokenRequest, auth_user: UserAuth = Depends(require_role("admin"))):
    """Generate signed JWT access token for multi-tenant and RBAC authentication."""
    token = create_access_token(user_id=req.user_id, role=req.role, tenant_id=req.tenant_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": req.user_id,
        "role": req.role,
        "tenant_id": req.tenant_id,
    }


@app.post("/api/auth/register")
def register_user(req: RegisterUserRequest):
    """Register a new user account with secure PBKDF2 password hashing and issue JWT token."""
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters long")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")

    try:
        user = chat_history_db.create_user(
            username=username,
            password=req.password,
            email=req.email,
            role="viewer",
            tenant_id="default",
        )
        token = create_access_token(user_id=user["id"], role=user["role"], tenant_id=user["tenant_id"])
        return {
            "token": token,
            "access_token": token,
            "token_type": "bearer",
            "user": user,
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("user_registration_failed")
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")


@app.post("/api/auth/login")
def login_user(req: LoginUserRequest):
    """Authenticate existing user credentials and return JWT token."""
    user = chat_history_db.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user_id=user["id"], role=user["role"], tenant_id=user["tenant_id"])
    return {
        "token": token,
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@app.get("/api/auth/me")
def get_auth_me(request: Request, auth_user: UserAuth = Depends(get_current_user)):
    """Return currently authenticated user profile or guest identity info."""
    db_user = chat_history_db.get_user_by_id(auth_user.user_id)
    if db_user:
        return {"user": db_user, "authenticated": True}

    is_guest = auth_user.user_id.startswith("guest_") or auth_user.user_id == "dev_user"
    return {
        "user": {
            "id": auth_user.user_id,
            "username": "Guest User" if is_guest else auth_user.user_id,
            "role": auth_user.role,
            "tenant_id": auth_user.tenant_id,
        },
        "authenticated": not is_guest,
    }


# --- Persistent Conversation & Chat History API ---

@app.get("/api/conversations")
def list_conversations(
    request: Request,
    auth_user: UserAuth = Depends(get_current_user),
):
    """List saved conversation threads for the authenticated user or guest."""
    effective_user_id = auth_user.user_id
    if effective_user_id == "dev_user":
        guest_hdr = request.headers.get("X-User-Id")
        if guest_hdr:
            effective_user_id = guest_hdr

    convs = chat_history_db.list_conversations(effective_user_id)
    return {"conversations": convs, "user_id": effective_user_id}


@app.get("/api/conversations/{session_id}")
def get_conversation(session_id: str):
    """Retrieve full chronological message history for a conversation."""
    messages = chat_history_db.get_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/conversations/{session_id}")
def delete_conversation(
    session_id: str,
    request: Request,
    auth_user: UserAuth = Depends(get_current_user),
):
    """Delete a conversation thread and all its stored messages."""
    effective_user_id = auth_user.user_id
    if effective_user_id == "dev_user":
        guest_hdr = request.headers.get("X-User-Id")
        if guest_hdr:
            effective_user_id = guest_hdr

    ok = chat_history_db.delete_conversation(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "session_id": session_id}


@app.patch("/api/conversations/{session_id}")
def update_conversation(session_id: str, req: UpdateConversationRequest):
    """Update conversation title."""
    ok = chat_history_db.update_conversation_title(session_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "updated", "session_id": session_id, "title": req.title}


# --- Asynchronous Task Queue API ---

@app.post("/tasks/submit")
def submit_async_task(
    req: SubmitTaskRequest,
    auth_user: UserAuth = Depends(get_current_user),
    _rl=Depends(rate_limit_dependency()),
):
    """Enqueue a long-running workflow or agent job to Redis Streams worker queue."""
    payload = dict(req.payload)
    if "user_id" not in payload:
        payload["user_id"] = auth_user.user_id
    if "tenant_id" not in payload:
        payload["tenant_id"] = auth_user.tenant_id

    task_id = task_queue.enqueue(task_type=req.task_type, payload=payload)
    return {"status": "queued", "task_id": task_id, "task_type": req.task_type}


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Inspect the status, execution progress, and result of an asynchronous task."""
    info = task_queue.get_status(task_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return info


@app.get("/api/router/status")
def get_router_status():
    """Diagnostic status for the multi-provider fallback router and token pool."""
    from src.agent.router import agent_router
    return agent_router.get_status()


# --- Self-Improving Failure-Correction Subsystem API ---

class ReportFailureRequest(BaseModel):
    session_id: str = "unknown"
    user_id: str = "unknown"
    user_input: str
    observed_output: str
    expected_behavior: str = ""
    tool_calls: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, Any]] = []
    model_name: str = "unknown"
    category: Optional[str] = None
    root_cause: Optional[str] = None


class AnalyzeFailureRequest(BaseModel):
    failure_id: Optional[str] = None
    session_id: str = "unknown"
    user_id: str = "unknown"
    user_input: str
    observed_output: str
    expected_behavior: str = ""
    tool_calls: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, Any]] = []
    model_name: str = "unknown"
    run_workflow: bool = True


class RunBenchmarkRequest(BaseModel):
    test_id: Optional[str] = None


class ApproveProposalRequest(BaseModel):
    approval_id: Optional[str] = None
    approved_by: str = "human_operator"


@app.post("/api/learning/failures")
def report_failure(req: ReportFailureRequest):
    """Log an agent failure and index it in the failure repository."""
    from src.learning.database import failure_db
    from src.learning.models import FailureRecord, FailureCategory
    import uuid

    failure_id = f"fail_{uuid.uuid4().hex[:12]}"
    category_val = FailureCategory(req.category) if req.category else FailureCategory.REASONING_FAILURE
    record = FailureRecord(
        failure_id=failure_id,
        session_id=req.session_id,
        user_id=req.user_id,
        user_input=req.user_input,
        observed_output=req.observed_output,
        expected_behavior=req.expected_behavior,
        tool_calls=req.tool_calls,
        tool_outputs=req.tool_outputs,
        model_name=req.model_name,
        category=category_val,
        root_cause=req.root_cause or "",
    )
    saved = failure_db.record_failure(record)
    return {"status": "recorded", "failure": saved.to_dict()}


@app.get("/api/learning/failures")
def list_failures(limit: int = 50, status: Optional[str] = None):
    """List recorded agent failures with optional status filter."""
    from src.learning.database import failure_db
    from src.learning.models import FailureStatus

    f_status = FailureStatus(status) if status else None
    failures = failure_db.list_failures(limit=limit, status=f_status)
    return {"failures": [f.model_dump() for f in failures], "count": len(failures)}


@app.get("/api/learning/clusters")
def list_failure_clusters():
    """List recurring failure clusters and recurring error patterns."""
    from src.learning.database import failure_db
    clusters = failure_db.list_clusters()
    return {"clusters": [c.model_dump() for c in clusters], "count": len(clusters)}


@app.post("/api/learning/analyze")
def analyze_failure(req: AnalyzeFailureRequest):
    """Run root-cause analysis, regression test generation, and optional workflow execution."""
    if req.run_workflow:
        from src.learning.workflow import self_improvement_workflow
        initial_state = {
            "failure_id": req.failure_id,
            "session_id": req.session_id,
            "user_id": req.user_id,
            "user_input": req.user_input,
            "observed_output": req.observed_output,
            "expected_behavior": req.expected_behavior,
            "tool_calls": req.tool_calls,
            "tool_outputs": req.tool_outputs,
            "model_name": req.model_name,
            "failure_record": None,
            "regression_test": None,
            "proposal": None,
            "validation_result": None,
            "approval_id": None,
            "status": "started",
            "error": None,
        }
        result = self_improvement_workflow.invoke(initial_state)
        return {"status": "completed", "result": result}
    else:
        from src.learning.analyzer import failure_analyzer
        from src.learning.regression_harness import regression_harness
        from src.learning.models import FailureRecord
        import uuid

        record = FailureRecord(
            failure_id=req.failure_id or f"fail_{uuid.uuid4().hex[:12]}",
            session_id=req.session_id,
            user_id=req.user_id,
            user_input=req.user_input,
            observed_output=req.observed_output,
            expected_behavior=req.expected_behavior,
            tool_calls=req.tool_calls,
            tool_outputs=req.tool_outputs,
            model_name=req.model_name,
        )
        analyzed = failure_analyzer.analyze(record)
        test = regression_harness.create_regression_test(analyzed)
        return {
            "status": "analyzed",
            "failure_record": analyzed.model_dump(),
            "regression_test": test.model_dump(),
        }


@app.get("/api/learning/regression-tests")
def list_regression_tests():
    """List all synthesized regression benchmark tests."""
    from src.learning.database import failure_db
    tests = failure_db.list_regression_tests()
    return {"regression_tests": [t.model_dump() for t in tests], "count": len(tests)}


@app.post("/api/learning/regression-tests/run")
def run_regression_benchmarks(req: RunBenchmarkRequest = RunBenchmarkRequest()):
    """Run regression benchmark evaluation across historical failure cases."""
    from src.learning.regression_harness import regression_harness
    if req.test_id:
        result = regression_harness.run_test(req.test_id)
        return {"test_id": req.test_id, "result": result}
    suite_result = regression_harness.run_benchmark_suite()
    return suite_result


@app.get("/api/learning/proposals")
def list_proposals(status: Optional[str] = None):
    """List generated improvement proposals with optional status filter."""
    from src.learning.database import failure_db
    proposals = failure_db.list_proposals(status=status)
    return {"proposals": [p.model_dump() for p in proposals], "count": len(proposals)}


@app.post("/api/learning/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: str, req: ApproveProposalRequest = ApproveProposalRequest()):
    """Approve a sandbox-validated improvement proposal and deploy it with versioning."""
    from src.learning.sandbox_validator import sandbox_validator
    version = sandbox_validator.approve_and_deploy(
        proposal_id=proposal_id,
        approval_id=req.approval_id,
        approved_by=req.approved_by,
    )
    if not version:
        raise HTTPException(
            status_code=400,
            detail=f"Proposal '{proposal_id}' could not be approved or deployed. Check status and approval."
        )
    return {"status": "deployed", "version": version.model_dump()}


@app.get("/api/learning/metrics")
def get_learning_metrics():
    """Return platform reliability, failure intelligence, and benchmark pass rates."""
    from src.learning.metrics import metrics_engine
    return metrics_engine.compute_metrics()


# ==============================================================================
# Workflow Observability & Execution Monitor Endpoints
# ==============================================================================

@app.get("/api/workflows/metrics")
def get_workflow_platform_metrics():
    """Return aggregated workflow execution metrics (success rate, avg latency, tool calls, FinOps cost)."""
    from src.workflows.observability import workflow_observability_engine
    summary = workflow_observability_engine.get_platform_summary()
    return summary.model_dump()


@app.get("/api/workflows/traces")
def list_workflow_traces(limit: int = 50):
    """Return recent workflow execution traces with node-by-node durations and statuses."""
    from src.workflows.observability import workflow_observability_engine
    traces = workflow_observability_engine.list_recent_traces(limit=limit)
    return {"traces": [t.model_dump() for t in traces], "count": len(traces)}


@app.get("/api/workflows/traces/{trace_id}")
def get_workflow_trace_detail(trace_id: str):
    """Return detailed trace metrics and node execution breakdown for a specific execution."""
    from src.workflows.observability import workflow_observability_engine
    trace = workflow_observability_engine.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Workflow trace '{trace_id}' not found")
    return trace.model_dump()


@app.get("/workflows/monitor", response_class=HTMLResponse)
def get_workflow_monitor_dashboard():
    """Render the dedicated Workflow Execution Monitor dashboard UI."""
    from src.workflows.observability import workflow_observability_engine
    return workflow_observability_engine.generate_html_dashboard()



