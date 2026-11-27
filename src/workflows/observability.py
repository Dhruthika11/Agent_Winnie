"""
Comprehensive Live Observability, Trace Explorer, and Evaluation System for Hermes.

Connects directly to authoritative PostgreSQL traces, observations, and checkpointer stores:
- Real P50/P95/P99 latency calculation
- True provider fallback frequency and candidate attempts
- Interactive node-by-node LangGraph execution timelines
- Tool execution parameters and outputs with sensitive data redaction
- Integrated execution evaluation runner and historical regression comparison
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.telemetry import sanitize_payload

logger = logging.getLogger("hermes.workflows.observability")


class NodeExecutionTrace(BaseModel):
    node_name: str
    status: str  # "completed", "in_progress", "waiting", "failed"
    duration_ms: float = 0.0
    summary: str = ""
    timestamp: Optional[str] = None


class WorkflowTraceRecord(BaseModel):
    trace_id: str
    workflow_id: str
    workflow_type: str
    objective: str
    status: str
    total_duration_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    nodes: list[NodeExecutionTrace] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class ObservabilityPlatformSummary(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    success_rate_pct: float = 100.0
    p50_latency_sec: float = 0.0
    p95_latency_sec: float = 0.0
    p99_latency_sec: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    fallback_rate_pct: float = 0.0
    total_tool_calls: int = 0
    active_workflows_count: int = 0
    postgres_status: str = "healthy"
    redis_status: str = "healthy"
    litellm_status: str = "healthy"


class WorkflowObservabilityEngine:
    """Authoritative observability engine connected directly to PostgreSQL telemetry tables."""

    def __init__(self):
        self._lock = threading.Lock()
        self._memory_traces: dict[str, dict[str, Any]] = {}

    def _get_db_connection(self):
        """Acquire a direct connection to the authoritative PostgreSQL database."""
        try:
            import psycopg
            db_url = getattr(settings, "postgres_url", None)
            if not db_url:
                db_url = getattr(settings, "database_url", "postgresql://hermes:hermes@localhost:5432/hermes_agent")
            return psycopg.connect(db_url, connect_timeout=3)
        except Exception as exc:
            logger.debug(f"Direct PostgreSQL connection unavailable: {exc}")
            return None

    def get_summary_metrics(self) -> ObservabilityPlatformSummary:
        """Calculate real-time operational and FinOps metrics directly from PostgreSQL."""
        conn = self._get_db_connection()
        if not conn:
            # Fallback for offline unit tests
            return ObservabilityPlatformSummary(
                total_requests=len(self._memory_traces),
                postgres_status="offline_or_mocked",
            )

        try:
            with conn.cursor() as cur:
                # 1. Total traces & success counts
                cur.execute("""
                    SELECT 
                        count(*) as total,
                        count(CASE WHEN output IS NOT NULL AND NOT (output::text ILIKE '%%error%%' OR output::text ILIKE '%%exception%%') THEN 1 END) as succ,
                        count(CASE WHEN output::text ILIKE '%%error%%' OR output::text ILIKE '%%exception%%' THEN 1 END) as err
                    FROM traces;
                """)
                row = cur.fetchone() or (0, 0, 0)
                tot_req = int(row[0] or 0)
                succ_req = int(row[1] or 0)
                err_req = int(row[2] or 0)
                succ_pct = round((succ_req / max(1, tot_req)) * 100, 1) if tot_req > 0 else 100.0

                # 2. Token consumption, costs, and tool calls from observations
                cur.execute("""
                    SELECT 
                        coalesce(sum(total_tokens), 0) as tokens,
                        coalesce(sum(total_cost), 0.0) as cost,
                        count(CASE WHEN name LIKE 'tool:%%' THEN 1 END) as tools,
                        count(CASE WHEN metadata::text LIKE '%%fallback_occurred%%true%%' OR name LIKE 'llm_candidate:%%' THEN 1 END) as fallbacks
                    FROM observations;
                """)
                obs_row = cur.fetchone() or (0, 0.0, 0, 0)
                tokens = int(obs_row[0] or 0)
                cost = round(float(obs_row[1] or 0.0), 4)
                tools = int(obs_row[2] or 0)
                fallbacks = int(obs_row[3] or 0)
                fb_pct = round((fallbacks / max(1, tot_req)) * 100, 1) if tot_req > 0 else 0.0

                # 3. Latency percentiles (P50, P95, P99)
                cur.execute("""
                    SELECT 
                        EXTRACT(EPOCH FROM (max(end_time) - min(start_time))) as dur
                    FROM observations
                    WHERE start_time IS NOT NULL AND end_time IS NOT NULL
                    GROUP BY trace_id
                    HAVING EXTRACT(EPOCH FROM (max(end_time) - min(start_time))) > 0;
                """)
                dur_rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                dur_rows.sort()

                if dur_rows:
                    n = len(dur_rows)
                    p50 = round(dur_rows[int(n * 0.50)], 2)
                    p95 = round(dur_rows[min(int(n * 0.95), n - 1)], 2)
                    p99 = round(dur_rows[min(int(n * 0.99), n - 1)], 2)
                else:
                    p50, p95, p99 = 0.0, 0.0, 0.0

                return ObservabilityPlatformSummary(
                    total_requests=tot_req,
                    successful_requests=succ_req,
                    error_requests=err_req,
                    success_rate_pct=succ_pct,
                    p50_latency_sec=p50,
                    p95_latency_sec=p95,
                    p99_latency_sec=p99,
                    total_tokens=tokens,
                    total_cost_usd=cost,
                    fallback_rate_pct=fb_pct,
                    total_tool_calls=tools,
                    active_workflows_count=0,
                    postgres_status="healthy",
                )
        except Exception as exc:
            logger.warning(f"Error calculating platform summary: {exc}")
            return ObservabilityPlatformSummary(postgres_status=f"error: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def list_recent_traces(self, limit: int = 50, search: Optional[str] = None) -> list[dict[str, Any]]:
        """Fetch list of actual workflow execution traces from PostgreSQL."""
        conn = self._get_db_connection()
        if not conn:
            return list(self._memory_traces.values())[:limit]

        try:
            with conn.cursor() as cur:
                query = """
                    SELECT 
                        t.id,
                        t.name,
                        t.timestamp,
                        t.session_id,
                        t.user_id,
                        t.input,
                        t.output,
                        t.metadata,
                        coalesce(EXTRACT(EPOCH FROM (max(o.end_time) - min(o.start_time))) * 1000.0, 0.0) as duration_ms,
                        count(o.id) as observation_count,
                        count(CASE WHEN o.type = 'GENERATION' THEN 1 END) as llm_calls,
                        count(CASE WHEN o.name LIKE 'tool:%%' THEN 1 END) as tool_calls,
                        coalesce(sum(o.total_cost), 0.0) as cost_usd
                    FROM traces t
                    LEFT JOIN observations o ON t.id = o.trace_id
                """
                params: list[Any] = []
                if search:
                    query += " WHERE t.id ILIKE %s OR t.session_id ILIKE %s OR t.name ILIKE %s OR t.user_id ILIKE %s"
                    s_wild = f"%{search}%"
                    params.extend([s_wild, s_wild, s_wild, s_wild])

                query += """
                    GROUP BY t.id, t.name, t.timestamp, t.session_id, t.user_id, t.input, t.output, t.metadata
                    ORDER BY t.timestamp DESC
                    LIMIT %s;
                """
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

                records = []
                for r in rows:
                    t_id = str(r[0])
                    name = str(r[1] or "chat_turn")
                    ts = r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2])
                    sess_id = str(r[3] or "default")
                    u_id = str(r[4] or "unknown")
                    inp = r[5]
                    out = r[6]
                    meta = r[7] or {}
                    dur_ms = round(float(r[8] or 0.0), 1)
                    obs_count = int(r[9] or 0)
                    llm_calls = int(r[10] or 0)
                    tool_calls = int(r[11] or 0)
                    cost = round(float(r[12] or 0.0), 4)

                    # Determine status
                    status = "completed"
                    if out and ("error" in str(out).lower() or "exception" in str(out).lower()):
                        status = "error"
                    elif meta.get("awaiting_approval") or "awaiting_approval" in str(out).lower():
                        status = "waiting_approval"

                    clean_input = sanitize_payload(inp)
                    clean_output = sanitize_payload(out)

                    records.append({
                        "trace_id": t_id,
                        "name": name,
                        "session_id": sess_id,
                        "user_id": u_id,
                        "timestamp": ts,
                        "status": status,
                        "duration_ms": dur_ms,
                        "observation_count": obs_count,
                        "llm_calls": llm_calls,
                        "tool_calls": tool_calls,
                        "cost_usd": cost,
                        "input": clean_input,
                        "output": clean_output,
                        "metadata": sanitize_payload(meta),
                    })
                return records
        except Exception as exc:
            logger.warning(f"Error listing traces: {exc}")
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_trace_detail(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full trace details with complete nested observation spans from PostgreSQL."""
        conn = self._get_db_connection()
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                # 1. Fetch trace row
                cur.execute("""
                    SELECT id, name, timestamp, session_id, user_id, input, output, metadata
                    FROM traces
                    WHERE id = %s;
                """, (trace_id,))
                t_row = cur.fetchone()
                if not t_row:
                    return None

                trace_record = {
                    "trace_id": str(t_row[0]),
                    "name": str(t_row[1] or "chat_turn"),
                    "timestamp": t_row[2].isoformat() if hasattr(t_row[2], "isoformat") else str(t_row[2]),
                    "session_id": str(t_row[3] or "default"),
                    "user_id": str(t_row[4] or "unknown"),
                    "input": sanitize_payload(t_row[5]),
                    "output": sanitize_payload(t_row[6]),
                    "metadata": sanitize_payload(t_row[7] or {}),
                }

                # 2. Fetch observations
                cur.execute("""
                    SELECT 
                        id, name, type, parent_observation_id, start_time, end_time,
                        model, input, output, metadata, level, status_message,
                        total_tokens, prompt_tokens, completion_tokens, total_cost
                    FROM observations
                    WHERE trace_id = %s
                    ORDER BY start_time ASC;
                """, (trace_id,))
                obs_rows = cur.fetchall()

                observations = []
                tool_executions = []
                routing_events = []
                timeline_nodes = []

                for o in obs_rows:
                    o_id = str(o[0])
                    o_name = str(o[1])
                    o_type = str(o[2])
                    parent_id = str(o[3]) if o[3] else None
                    st = o[4].isoformat() if hasattr(o[4], "isoformat") else str(o[4])
                    et = o[5].isoformat() if hasattr(o[5], "isoformat") else str(o[5]) if o[5] else None

                    dur_ms = 0.0
                    if o[4] and o[5]:
                        dur_ms = round((o[5] - o[4]).total_seconds() * 1000.0, 1)

                    model = str(o[6]) if o[6] else None
                    inp = sanitize_payload(o[7])
                    out = sanitize_payload(o[8])
                    meta = sanitize_payload(o[9] or {})
                    level = str(o[10])
                    stat_msg = str(o[11]) if o[11] else None

                    obs_item = {
                        "id": o_id,
                        "name": o_name,
                        "type": o_type,
                        "parent_id": parent_id,
                        "start_time": st,
                        "end_time": et,
                        "duration_ms": dur_ms,
                        "model": model,
                        "input": inp,
                        "output": out,
                        "metadata": meta,
                        "level": level,
                        "status_message": stat_msg,
                        "total_tokens": int(o[12] or 0),
                        "total_cost": round(float(o[15] or 0.0), 5),
                    }
                    observations.append(obs_item)

                    # Timeline node representation
                    clean_node_name = o_name.replace("tool:", "").replace("_", " ").title()
                    node_status = "failed" if level == "ERROR" else "completed"
                    timeline_nodes.append({
                        "name": clean_node_name,
                        "type": o_type,
                        "duration_ms": dur_ms,
                        "status": node_status,
                        "summary": stat_msg or (f"Model: {model}" if model else f"Executed {clean_node_name}"),
                    })

                    # Extract tools
                    if o_name.startswith("tool:"):
                        tool_executions.append({
                            "tool_name": o_name[5:],
                            "arguments": inp,
                            "output_preview": str(out)[:300],
                            "duration_ms": dur_ms,
                            "status": node_status,
                        })

                    # Extract candidate routing
                    if o_name.startswith("llm_candidate:") or meta.get("routing_events"):
                        r_evts = meta.get("routing_events")
                        if isinstance(r_evts, list):
                            routing_events.extend(r_evts)
                        elif o_name.startswith("llm_candidate:"):
                            routing_events.append({
                                "candidate": o_name[14:],
                                "duration_ms": dur_ms,
                                "status": "failed" if level == "ERROR" else "success",
                                "message": stat_msg,
                            })

                trace_record["observations"] = observations
                trace_record["timeline_nodes"] = timeline_nodes
                trace_record["tool_executions"] = tool_executions
                trace_record["routing_events"] = routing_events
                trace_record["total_duration_ms"] = sum(o["duration_ms"] for o in observations if o["parent_id"] is None) or sum(o["duration_ms"] for o in observations)
                return trace_record
        except Exception as exc:
            logger.warning(f"Error fetching trace detail for {trace_id}: {exc}")
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_evaluation_history(self) -> list[dict[str, Any]]:
        """Read historical evaluation runs from workspace/evaluation_history.json."""
        hist_file = os.path.join(os.getcwd(), "workspace", "evaluation_history.json")
        if os.path.exists(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def run_live_evaluation(self, base_url: str = "http://localhost:8080") -> dict[str, Any]:
        """Trigger an independent live evaluation suite run and return the summary."""
        from src.evaluation.live_runner import LiveEvaluationRunner
        runner = LiveEvaluationRunner(base_url=base_url)
        summary = runner.run_all()
        return summary.model_dump()

    def generate_html_dashboard(self) -> str:
        """Render the complete modern responsive HTML live observability dashboard."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hermes - Live Observability & Evaluation Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #090d16;
            --bg-surface: #101726;
            --bg-card: #151f33;
            --bg-hover: #1b2842;
            --border: #22314d;
            --border-subtle: #19253d;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.2);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.12);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.12);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.12);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --font-sans: 'Plus Jakarta Sans', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans);
            background-color: var(--bg-base);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-badge {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .header-title { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
        .header-subtitle { color: var(--text-muted); font-size: 13px; }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }
        .btn:hover { background: var(--bg-hover); border-color: var(--accent); }
        .btn-primary {
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
        }
        .btn-primary:hover { background: #2563eb; }

        /* KPI Banner */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
        }
        .kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
        .kpi-value { font-size: 24px; font-weight: 700; margin: 4px 0; color: #fff; }
        .kpi-sub { font-size: 12px; color: var(--text-dim); }

        /* Navigation Tabs */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn:hover { color: var(--text-main); background: var(--bg-surface); }
        .tab-btn.active { color: #fff; background: var(--border); }

        /* Main Workspace Split */
        .main-workspace {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 20px;
            min-height: 650px;
        }

        /* Left Column: Trace List */
        .trace-list-panel {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .list-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .search-box {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-main);
            padding: 8px 12px;
            font-size: 13px;
        }
        .search-box:focus { outline: none; border-color: var(--accent); }
        .trace-scroll-area {
            flex: 1;
            overflow-y: auto;
            max-height: 700px;
        }
        .trace-item {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            cursor: pointer;
            transition: all 0.15s;
        }
        .trace-item:hover { background: var(--bg-card); }
        .trace-item.active { background: var(--bg-card); border-left: 3px solid var(--accent); }
        .trace-item-title { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 4px; }
        .trace-item-meta { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-success { background: var(--success-bg); color: var(--success); }
        .badge-warning { background: var(--warning-bg); color: var(--warning); }
        .badge-danger { background: var(--danger-bg); color: var(--danger); }

        /* Right Column: Trace Inspector */
        .trace-detail-panel {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            overflow-y: auto;
            max-height: 780px;
        }
        .detail-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 18px;
        }
        .detail-card-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #fff;
        }

        /* Timeline Nodes */
        .timeline { list-style: none; position: relative; padding-left: 20px; }
        .timeline::before {
            content: '';
            position: absolute;
            left: 6px;
            top: 6px;
            bottom: 6px;
            width: 2px;
            background: var(--border);
        }
        .timeline-step { position: relative; margin-bottom: 16px; }
        .timeline-bullet {
            position: absolute;
            left: -20px;
            top: 4px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--accent);
            border: 3px solid var(--bg-card);
        }
        .timeline-bullet.failed { background: var(--danger); }
        .timeline-body {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 8px;
        }
        .timeline-header { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; }
        .timeline-desc { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

        /* Code/JSON Pre */
        pre {
            font-family: var(--font-mono);
            font-size: 12px;
            background: #090e17;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
            color: #cbd5e1;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }

        /* Evaluation Table */
        .eval-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .eval-table th {
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
        }
        .eval-table td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .eval-table tr:hover td { background: var(--bg-card); }
        .trace-link {
            color: var(--accent);
            cursor: pointer;
            text-decoration: underline;
            font-family: var(--font-mono);
            font-size: 12px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-badge">Live System</span>
            <div>
                <h1 class="header-title">Hermes Autonomous Agent Observability & Evaluation</h1>
                <p class="header-subtitle">Real-time distributed tracing, evidence verification, and independent reliability testing</p>
            </div>
        </div>
        <div class="header-actions">
            <button class="btn" onclick="fetchSummary(); fetchTraces();">🔄 Refresh</button>
            <button class="btn btn-primary" onclick="triggerLiveEvaluation()">▶ Run Live Evaluation</button>
        </div>
    </header>

    <!-- Top KPI Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Request Success Rate</div>
            <div class="kpi-value" id="kpi-succ-rate">--%</div>
            <div class="kpi-sub" id="kpi-req-count">0 total requests</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">P50 Latency</div>
            <div class="kpi-value" id="kpi-p50">--s</div>
            <div class="kpi-sub">Median execution duration</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">P95 / P99 Latency</div>
            <div class="kpi-value" id="kpi-p95">--s</div>
            <div class="kpi-sub" id="kpi-p99">P99: --s</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Provider Fallbacks</div>
            <div class="kpi-value" id="kpi-fallback">--%</div>
            <div class="kpi-sub">Multi-provider failover rate</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Tokens & Total Cost</div>
            <div class="kpi-value" id="kpi-cost">$0.00</div>
            <div class="kpi-sub" id="kpi-tokens">0 tokens consumed</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
        <button class="tab-btn active" id="tab-btn-traces" onclick="switchTab('traces')">Workflows & Distributed Traces</button>
        <button class="tab-btn" id="tab-btn-evals" onclick="switchTab('evals')">Independent Evaluation & Tests</button>
        <button class="tab-btn" id="tab-btn-history" onclick="switchTab('history')">Historical Regression Reports</button>
    </div>

    <!-- Tab 1: Live Traces & Explorer -->
    <div id="view-traces" class="main-workspace">
        <!-- Left Panel: Trace List -->
        <div class="trace-list-panel">
            <div class="list-header">
                <input type="text" class="search-box" id="trace-search" placeholder="Filter by trace ID, user, or session..." oninput="debounceSearch()">
            </div>
            <div class="trace-scroll-area" id="trace-list-container">
                <div style="padding: 24px; text-align: center; color: var(--text-dim);">Loading live traces...</div>
            </div>
        </div>

        <!-- Right Panel: Trace Inspector -->
        <div class="trace-detail-panel" id="trace-detail-container">
            <div style="padding: 40px; text-align: center; color: var(--text-dim);">
                Select a workflow trace on the left to inspect its live execution timeline, model routing, and tool calls.
            </div>
        </div>
    </div>

    <!-- Tab 2: Independent Evaluation Suite -->
    <div id="view-evals" style="display: none;">
        <div class="detail-card">
            <div class="detail-card-title">
                <span>🛡️ Automated Functional, Security, and Governance Evaluation Suite</span>
                <span id="eval-status-badge" class="badge badge-success">Ready</span>
            </div>
            <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
                All test scenarios interact directly with running HTTP API endpoints without mocks. Each status links to actual trace and execution evidence.
            </p>
            <table class="eval-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Test ID</th>
                        <th>Category</th>
                        <th>Scenario Title</th>
                        <th>Observed Outcome & Evidence</th>
                        <th>Duration</th>
                        <th>Trace Evidence</th>
                    </tr>
                </thead>
                <tbody id="eval-table-body">
                    <tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 24px;">Click 'Run Live Evaluation' to execute test cases against the live API.</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Tab 3: Historical Regression Reports -->
    <div id="view-history" style="display: none;">
        <div class="detail-card">
            <div class="detail-card-title">📈 Historical Application Evaluation Runs & Regression Tracking</div>
            <div id="history-container">
                <div style="padding: 24px; text-align: center; color: var(--text-dim);">Loading historical evaluation runs...</div>
            </div>
        </div>
    </div>

    <script>
        let currentTraceId = null;
        let searchTimeout = null;

        async function fetchSummary() {
            try {
                const res = await fetch('/api/observability/summary');
                const data = await res.json();
                document.getElementById('kpi-succ-rate').innerText = data.success_rate_pct + '%';
                document.getElementById('kpi-req-count').innerText = data.total_requests + ' total requests';
                document.getElementById('kpi-p50').innerText = data.p50_latency_sec + 's';
                document.getElementById('kpi-p95').innerText = data.p95_latency_sec + 's';
                document.getElementById('kpi-p99').innerText = 'P99: ' + data.p99_latency_sec + 's';
                document.getElementById('kpi-fallback').innerText = data.fallback_rate_pct + '%';
                document.getElementById('kpi-cost').innerText = '$' + data.total_cost_usd.toFixed(4);
                document.getElementById('kpi-tokens').innerText = data.total_tokens.toLocaleString() + ' tokens consumed';
            } catch (err) {
                console.error("Failed to fetch summary:", err);
            }
        }

        async function fetchTraces(search = '') {
            try {
                const url = '/api/observability/traces' + (search ? '?search=' + encodeURIComponent(search) : '');
                const res = await fetch(url);
                const data = await res.json();
                const traces = data.traces || [];
                const container = document.getElementById('trace-list-container');

                if (traces.length === 0) {
                    container.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-dim);">No traces found.</div>';
                    return;
                }

                let html = '';
                traces.forEach(t => {
                    const isActive = t.trace_id === currentTraceId ? 'active' : '';
                    const badgeClass = t.status === 'completed' ? 'badge-success' : (t.status === 'error' ? 'badge-danger' : 'badge-warning');
                    html += `
                    <div class="trace-item ${isActive}" onclick="loadTraceDetail('${t.trace_id}')">
                        <div class="trace-item-title">${escapeHtml(t.name)}</div>
                        <div style="display: flex; gap: 6px; margin: 4px 0;">
                            <span class="badge ${badgeClass}">${t.status}</span>
                            <span style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted);">${(t.duration_ms/1000).toFixed(2)}s</span>
                            <span style="font-size: 11px; color: var(--text-dim);">${t.tool_calls} tools</span>
                        </div>
                        <div class="trace-item-meta">
                            <span>ID: ${t.trace_id.substring(0, 12)}...</span>
                            <span>${new Date(t.timestamp).toLocaleTimeString()}</span>
                        </div>
                    </div>`;
                });
                container.innerHTML = html;

                if (!currentTraceId && traces.length > 0) {
                    loadTraceDetail(traces[0].trace_id);
                }
            } catch (err) {
                console.error("Failed to fetch traces:", err);
            }
        }

        async function loadTraceDetail(traceId) {
            currentTraceId = traceId;
            document.querySelectorAll('.trace-item').forEach(el => el.classList.remove('active'));
            const container = document.getElementById('trace-detail-container');
            container.innerHTML = '<div style="padding: 40px; text-align: center; color: var(--text-dim);">Loading trace details...</div>';

            try {
                const res = await fetch('/api/observability/traces/' + encodeURIComponent(traceId));
                const t = await res.json();

                let timelineHtml = '';
                (t.timeline_nodes || []).forEach(node => {
                    const bulletClass = node.status === 'failed' ? 'failed' : '';
                    timelineHtml += `
                    <li class="timeline-step">
                        <div class="timeline-bullet ${bulletClass}"></div>
                        <div class="timeline-body">
                            <div class="timeline-header">
                                <span>${escapeHtml(node.name)}</span>
                                <span style="font-family: var(--font-mono);">${node.duration_ms} ms</span>
                            </div>
                            <div class="timeline-desc">${escapeHtml(node.summary || '')}</div>
                        </div>
                    </li>`;
                });

                let toolsHtml = '';
                if ((t.tool_executions || []).length > 0) {
                    t.tool_executions.forEach(tool => {
                        toolsHtml += `
                        <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 13px;">
                                <span>🔧 ${escapeHtml(tool.tool_name)}</span>
                                <span class="badge ${tool.status === 'completed' ? 'badge-success' : 'badge-danger'}">${tool.status}</span>
                            </div>
                            <div style="margin-top: 6px; font-size: 12px; color: var(--text-muted);">
                                <strong>Arguments:</strong> <code>${escapeHtml(JSON.stringify(tool.arguments))}</code>
                            </div>
                            <div style="margin-top: 4px; font-size: 12px; color: var(--text-dim);">
                                <strong>Output Preview:</strong> ${escapeHtml(tool.output_preview)}
                            </div>
                        </div>`;
                    });
                } else {
                    toolsHtml = '<p style="font-size: 13px; color: var(--text-dim);">No external tools invoked in this turn.</p>';
                }

                let routingHtml = '';
                if ((t.routing_events || []).length > 0) {
                    t.routing_events.forEach(evt => {
                        const statusBadge = evt.status === 'success' ? 'badge-success' : 'badge-danger';
                        routingHtml += `
                        <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-bottom: 8px; font-size: 12px;">
                            <div style="display: flex; justify-content: space-between; font-weight: 600;">
                                <span>Model: ${escapeHtml(evt.model || evt.candidate || 'default')}</span>
                                <span class="badge ${statusBadge}">${evt.status}</span>
                            </div>
                            <div style="color: var(--text-dim); margin-top: 4px;">Provider: ${escapeHtml(evt.provider || 'primary')} | Latency: ${evt.latency_ms || 0}ms</div>
                            ${evt.error_message ? `<div style="color: var(--danger); margin-top: 4px;">Error: ${escapeHtml(evt.error_message)}</div>` : ''}
                        </div>`;
                    });
                } else {
                    routingHtml = '<p style="font-size: 13px; color: var(--text-dim);">Dispatched via primary model deployment without fallback.</p>';
                }

                container.innerHTML = `
                <div>
                    <!-- Header -->
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                        <div>
                            <span class="badge badge-success">${escapeHtml(t.name)}</span>
                            <h2 style="font-size: 18px; margin-top: 6px; color: #fff;">Trace ID: ${t.trace_id}</h2>
                            <p style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); margin-top: 2px;">Session: ${t.session_id} | User: ${t.user_id}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 20px; font-weight: 700; color: #fff;">${(t.total_duration_ms/1000).toFixed(2)}s</div>
                            <div style="font-size: 11px; color: var(--text-dim);">${new Date(t.timestamp).toLocaleString()}</div>
                        </div>
                    </div>

                    <!-- Visual Timeline -->
                    <div class="detail-card">
                        <div class="detail-card-title">⚡ LangGraph Node-by-Node Execution Timeline</div>
                        <ul class="timeline">
                            ${timelineHtml}
                        </ul>
                    </div>

                    <!-- Model Provider & Fallbacks -->
                    <div class="detail-card">
                        <div class="detail-card-title">🔀 LLM Provider Routing & Fallback Verification</div>
                        ${routingHtml}
                    </div>

                    <!-- Tool Calls -->
                    <div class="detail-card">
                        <div class="detail-card-title">🛠️ Tool Invocations & Verified Results</div>
                        ${toolsHtml}
                    </div>

                    <!-- Payloads with Redaction -->
                    <div class="detail-card">
                        <div class="detail-card-title">🔒 Request & Response Payloads (Sensitive Tokens Redacted)</div>
                        <div style="margin-bottom: 12px;">
                            <strong style="font-size: 12px; color: var(--text-muted);">User Input:</strong>
                            <pre>${escapeHtml(typeof t.input === 'string' ? t.input : JSON.stringify(t.input, null, 2))}</pre>
                        </div>
                        <div>
                            <strong style="font-size: 12px; color: var(--text-muted);">Assistant Output:</strong>
                            <pre>${escapeHtml(typeof t.output === 'string' ? t.output : JSON.stringify(t.output, null, 2))}</pre>
                        </div>
                    </div>
                </div>`;
            } catch (err) {
                container.innerHTML = '<div style="color: var(--danger); padding: 24px;">Failed to load trace details.</div>';
            }
        }

        async function triggerLiveEvaluation() {
            const btn = document.querySelector('.btn-primary');
            const badge = document.getElementById('eval-status-badge');
            btn.disabled = true;
            btn.innerText = '⏳ Running Evaluation Suite...';
            badge.innerText = 'Executing';
            badge.className = 'badge badge-warning';

            switchTab('evals');

            try {
                const res = await fetch('/api/observability/evaluations/run', { method: 'POST' });
                const report = await res.json();
                renderEvaluationResults(report);
                badge.innerText = `${report.pass_rate_pct}% Passed`;
                badge.className = report.pass_rate_pct >= 80 ? 'badge badge-success' : 'badge badge-danger';
                fetchSummary();
            } catch (err) {
                alert("Evaluation run failed: " + err);
                badge.innerText = 'Error';
                badge.className = 'badge badge-danger';
            } finally {
                btn.disabled = false;
                btn.innerText = '▶ Run Live Evaluation';
            }
        }

        function renderEvaluationResults(report) {
            const tbody = document.getElementById('eval-table-body');
            let html = '';
            (report.results || []).forEach(r => {
                const badgeClass = r.status === 'PASS' ? 'badge-success' : (r.status === 'FAIL' ? 'badge-danger' : 'badge-warning');
                const traceLink = r.trace_id ? `<span class="trace-link" onclick="viewTraceFromEval('${r.trace_id}')">${r.trace_id.substring(0, 10)}...</span>` : 'N/A';
                html += `
                <tr>
                    <td><span class="badge ${badgeClass}">${r.status}</span></td>
                    <td style="font-family: var(--font-mono); font-weight: 600;">${r.test_id}</td>
                    <td>${escapeHtml(r.category)}</td>
                    <td style="font-weight: 500;">${escapeHtml(r.title)}</td>
                    <td style="color: var(--text-muted); font-size: 12px;">${escapeHtml(r.observed_behavior)}</td>
                    <td style="font-family: var(--font-mono);">${r.duration_ms.toFixed(1)}ms</td>
                    <td>${traceLink}</td>
                </tr>`;
            });
            tbody.innerHTML = html;
        }

        async function loadEvaluationHistory() {
            try {
                const res = await fetch('/api/observability/evaluations');
                const history = await res.json();
                const container = document.getElementById('history-container');

                if (!history || history.length === 0) {
                    container.innerHTML = '<p style="color: var(--text-dim); text-align: center; padding: 24px;">No historical evaluation runs recorded yet.</p>';
                    return;
                }

                let html = '<div style="display: flex; flex-direction: column; gap: 12px;">';
                history.forEach(run => {
                    const badgeClass = run.pass_rate_pct >= 80 ? 'badge-success' : 'badge-danger';
                    html += `
                    <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 600; font-size: 13px;">Commit: <code style="color: var(--accent);">${run.git_commit}</code> | Run: ${run.run_id}</div>
                            <div style="font-size: 12px; color: var(--text-dim); margin-top: 4px;">${new Date(run.timestamp).toLocaleString()} | Duration: ${(run.total_duration_ms/1000).toFixed(2)}s</div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span class="badge ${badgeClass}">${run.pass_rate_pct}% Pass</span>
                            <span style="font-size: 12px; color: var(--text-muted);">${run.passed_cases}/${run.total_cases} passed</span>
                        </div>
                    </div>`;
                });
                html += '</div>';
                container.innerHTML = html;
            } catch (err) {
                console.error("Failed to load history:", err);
            }
        }

        function viewTraceFromEval(traceId) {
            switchTab('traces');
            loadTraceDetail(traceId);
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('view-traces').style.display = tabId === 'traces' ? 'grid' : 'none';
            document.getElementById('view-evals').style.display = tabId === 'evals' ? 'block' : 'none';
            document.getElementById('view-history').style.display = tabId === 'history' ? 'block' : 'none';
            document.getElementById('tab-btn-' + tabId).classList.add('active');

            if (tabId === 'history') {
                loadEvaluationHistory();
            }
        }

        function debounceSearch() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const query = document.getElementById('trace-search').value.trim();
                fetchTraces(query);
            }, 300);
        }

        function escapeHtml(text) {
            if (!text) return '';
            return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        // Initialize on load
        fetchSummary();
        fetchTraces();
        setInterval(fetchSummary, 10000);
    </script>
</body>
</html>"""


workflow_observability_engine = WorkflowObservabilityEngine()
