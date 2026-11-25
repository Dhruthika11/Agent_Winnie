"""
FinOps & Performance Optimization Subsystem for Hermes Agent Platform.
Tracks 9 core performance, cost, and reliability metrics for every request:

1. TTFT (Time To First Token) - Initial response speed
2. End-to-end latency - Overall responsiveness
3. Tokens per request - Model efficiency (prompt, completion, total)
4. Cost per request - Financial efficiency in USD
5. Tool calls per request - Planning efficiency
6. Retry rate - Reliability (tool retries, grounding re-searches, inference retries)
7. Cache hit rate - Cost reduction (idempotency, prompt, and semantic cache)
8. Grounding failure rate - Answer quality (unverified claims, hallucinations)
9. Task completion rate - Actual usefulness (successful fulfillment vs failure/timeout)

Also enforces daily token spending caps and graceful model degradation.
"""
from __future__ import annotations

import datetime
import logging
import threading
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger("hermes.security.finops")

BudgetStatus = Literal["ok", "degraded", "blocked"]

# Pricing matrix per 1,000,000 tokens in USD for all project-implemented models
MODEL_PRICING_RATES: dict[str, dict[str, float]] = {
    # --- Project Groq LLM Routing Pool ---
    "groq/openai/gpt-oss-120b": {"prompt": 0.89, "completion": 1.19},  # Tier 1 Ultra-Large Reasoning
    "groq/qwen/qwen3.8-27b": {"prompt": 0.20, "completion": 0.20},     # Tier 2 Specialized Coding
    "groq/openai/gpt-oss-20b": {"prompt": 0.15, "completion": 0.60},   # Tier 3 Fast Synthesis / Quick Chat
    "groq/llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},  # Tier 4 Flagship Fallback
    "groq/llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},     # Tier 5 Fast Fallback
    "groq/llama3-70b-8192": {"prompt": 0.59, "completion": 0.79},          # Tier 6 Legacy 70B
    "groq/mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},       # Tier 7 MoE Fallback

    # --- Project Google Gemini Multimodal Pool ---
    "gemini/gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},       # Fast Multimodal
    "gemini/gemini-2.0-flash-exp": {"prompt": 0.00, "completion": 0.00},   # Experimental Free
    "gemini/gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},      # Vision & Audio OCR
    "gemini/gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},         # Deep Long-Context Multimodal

    # --- Standard External Models (LiteLLM Fallbacks) ---
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28},

    # --- Fallback Default Rate ---
    "default": {"prompt": 0.20, "completion": 0.40},
}


def calculate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated financial cost in USD for a given token usage across implemented models."""
    clean_model = (model_name or "").lower().strip()
    rates = MODEL_PRICING_RATES["default"]

    # 1. Exact match check
    if clean_model in MODEL_PRICING_RATES:
        rates = MODEL_PRICING_RATES[clean_model]
    else:
        # 2. Substring match (longest key first to avoid partial prefix collisions)
        for key in sorted(MODEL_PRICING_RATES.keys(), key=len, reverse=True):
            if key != "default" and key in clean_model:
                rates = MODEL_PRICING_RATES[key]
                break

    prompt_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
    return round(prompt_cost + completion_cost, 6)


class FinOpsMetricRecord(BaseModel):
    """Detailed telemetry record tracking all 9 performance and cost metrics for a request."""
    request_id: str
    session_id: str = "default"
    user_id: str = "default"
    tenant_id: str = "default"
    model_name: str = "default"

    # 1. TTFT
    ttft_ms: Optional[float] = None

    # 2. End-to-end latency
    end_to_end_latency_ms: float = 0.0

    # 3. Tokens per request
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # 4. Cost per request
    cost_usd: float = 0.0

    # 5. Tool calls per request
    tool_calls_count: int = 0

    # 6. Retry rate components
    tool_retries: int = 0
    grounding_retries: int = 0
    inference_retries: int = 0
    total_retries: int = 0
    retry_rate: float = 0.0

    # 7. Cache hit rate
    cache_hit: bool = False

    # 8. Grounding failure rate components
    grounding_failures_count: int = 0
    total_claims_count: int = 0
    grounding_failure_rate: float = 0.0

    # 9. Task completion rate
    task_completed: bool = True
    stop_reason: Optional[str] = None

    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def __init__(self, **data: Any):
        # Auto-compute total tokens
        p_tok = data.get("prompt_tokens", 0)
        c_tok = data.get("completion_tokens", 0)
        if "total_tokens" not in data or not data["total_tokens"]:
            data["total_tokens"] = p_tok + c_tok

        # Auto-compute cost in USD if not explicitly supplied
        if not data.get("cost_usd"):
            m_name = data.get("model_name", "default")
            data["cost_usd"] = calculate_cost_usd(m_name, p_tok, c_tok)

        # Auto-compute retries and retry rate
        t_ret = data.get("tool_retries", 0)
        g_ret = data.get("grounding_retries", 0)
        i_ret = data.get("inference_retries", 0)
        tot_ret = data.get("total_retries", t_ret + g_ret + i_ret)
        data["total_retries"] = tot_ret

        # Retry rate = retries / (initial execution + retries)
        if "retry_rate" not in data or data["retry_rate"] == 0.0:
            data["retry_rate"] = round(tot_ret / max(1, 1 + tot_ret), 3)

        # Auto-compute grounding failure rate
        g_fails = data.get("grounding_failures_count", 0)
        t_claims = data.get("total_claims_count", 0)
        if "grounding_failure_rate" not in data or data["grounding_failure_rate"] == 0.0:
            data["grounding_failure_rate"] = round(g_fails / max(1, t_claims), 3) if t_claims > 0 else 0.0

        super().__init__(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "model_name": self.model_name,
            "ttft_ms": self.ttft_ms,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "tool_calls_count": self.tool_calls_count,
            "tool_retries": self.tool_retries,
            "grounding_retries": self.grounding_retries,
            "inference_retries": self.inference_retries,
            "total_retries": self.total_retries,
            "retry_rate": self.retry_rate,
            "cache_hit": self.cache_hit,
            "grounding_failures_count": self.grounding_failures_count,
            "total_claims_count": self.total_claims_count,
            "grounding_failure_rate": self.grounding_failure_rate,
            "task_completed": self.task_completed,
            "stop_reason": self.stop_reason,
            "timestamp": self.timestamp,
        }


class FinOpsManager:
    """Enterprise FinOps & Performance Optimization Manager.

    Tracks and aggregates the 9 request metrics, enforces spending budgets,
    and analyzes financial and operational efficiency.
    """

    def __init__(self):
        self._redis = None
        self._lock = threading.Lock()
        self._in_memory_token_usage: dict[str, int] = {}
        self._request_records: list[FinOpsMetricRecord] = []

    @property
    def redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                )
            except Exception:
                self._redis = None
        return self._redis

    def _get_key(self, user_id: str, tenant_id: str = "default") -> str:
        today = datetime.date.today().isoformat()
        return f"hermes:tokens:{tenant_id}:{user_id}:{today}"

    def get_token_usage(self, user_id: str, tenant_id: str = "default") -> int:
        """Get cumulative daily token consumption for a user/tenant."""
        key = self._get_key(user_id, tenant_id)
        r = self.redis
        if r:
            try:
                val = r.get(key)
                return int(val) if val else 0
            except Exception:
                pass
        with self._lock:
            return self._in_memory_token_usage.get(key, 0)

    def record_usage(self, user_id: str, tokens: int, tenant_id: str = "default") -> int:
        """Record daily token consumption."""
        key = self._get_key(user_id, tenant_id)
        r = self.redis
        if r:
            try:
                pipe = r.pipeline()
                pipe.incrby(key, tokens)
                pipe.expire(key, 86400 * 2)
                res = pipe.execute()
                return int(res[0])
            except Exception:
                pass

        with self._lock:
            self._in_memory_token_usage[key] = self._in_memory_token_usage.get(key, 0) + tokens
            return self._in_memory_token_usage[key]

    def evaluate_budget(self, user_id: str, tenant_id: str = "default") -> tuple[BudgetStatus, int, int]:
        """Evaluate spending limit.
        Returns:
            (status: "ok" | "degraded" | "blocked", tokens_used, max_tokens)
        """
        max_limit = settings.max_daily_tokens_per_user
        used = self.get_token_usage(user_id, tenant_id)

        if used >= max_limit * settings.finops_budget_block_pct:
            logger.warning(
                "finops_budget_blocked",
                extra={"user_id": user_id, "used": used, "limit": max_limit},
            )
            return "blocked", used, max_limit
        elif used >= max_limit * settings.finops_budget_degradation_pct:
            logger.info(
                "finops_budget_degraded",
                extra={"user_id": user_id, "used": used, "limit": max_limit},
            )
            return "degraded", used, max_limit
        return "ok", used, max_limit

    # --------------------------------------------------------------------------- #
    # FinOps Subsystem: 9 Core Metrics Recording & Analytics
    # --------------------------------------------------------------------------- #
    def record_request_metrics(self, record: FinOpsMetricRecord | dict[str, Any]) -> FinOpsMetricRecord:
        """Record request telemetry across all 9 performance, cost, and reliability metrics."""
        rec = record if isinstance(record, FinOpsMetricRecord) else FinOpsMetricRecord(**record)

        with self._lock:
            self._request_records.append(rec)
            # Retain up to 2000 recent records in active memory
            if len(self._request_records) > 2000:
                self._request_records = self._request_records[-2000:]

        # Also update token usage accounting
        if rec.total_tokens > 0:
            self.record_usage(rec.user_id, rec.total_tokens, rec.tenant_id)

        logger.info(
            "finops_request_metrics_recorded",
            extra={
                "request_id": rec.request_id,
                "ttft_ms": rec.ttft_ms,
                "latency_ms": rec.end_to_end_latency_ms,
                "tokens": rec.total_tokens,
                "cost_usd": rec.cost_usd,
                "tool_calls": rec.tool_calls_count,
                "retry_rate": rec.retry_rate,
                "cache_hit": rec.cache_hit,
                "grounding_failure_rate": rec.grounding_failure_rate,
                "task_completed": rec.task_completed,
            },
        )
        return rec

    def list_records(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[FinOpsMetricRecord]:
        """Fetch individual request metric traces with optional user/tenant filtering."""
        with self._lock:
            items = list(self._request_records)

        if tenant_id:
            items = [r for r in items if r.tenant_id == tenant_id]
        if user_id:
            items = [r for r in items if r.user_id == user_id]

        items.sort(key=lambda r: r.timestamp, reverse=True)
        return items[:limit]

    def get_performance_and_cost_analytics(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Compute aggregated statistics for the 9 core performance and cost metrics."""
        records = self.list_records(user_id=user_id, tenant_id=tenant_id, limit=limit)
        total_reqs = len(records)

        if total_reqs == 0:
            return {
                "total_requests": 0,
                "ttft": {"avg_ms": 0.0, "p95_ms": 0.0, "count": 0},
                "latency": {"avg_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0},
                "tokens": {"avg_per_request": 0, "total": 0, "prompt_total": 0, "completion_total": 0},
                "cost": {"total_usd": 0.0, "avg_per_request_usd": 0.0},
                "tool_calls": {"avg_per_request": 0.0, "total": 0},
                "reliability": {"overall_retry_rate_pct": 0.0, "total_retries": 0},
                "cache": {"hit_rate_pct": 0.0, "total_hits": 0},
                "grounding": {"failure_rate_pct": 0.0, "total_failures": 0},
                "task_completion": {"completion_rate_pct": 100.0, "completed": 0, "total": 0},
            }

        # 1. TTFT
        ttfts = [r.ttft_ms for r in records if r.ttft_ms is not None]
        avg_ttft = round(sum(ttfts) / len(ttfts), 2) if ttfts else 0.0
        p95_ttft = round(sorted(ttfts)[int(len(ttfts) * 0.95)], 2) if ttfts else avg_ttft

        # 2. End-to-end latency
        latencies = sorted([r.end_to_end_latency_ms for r in records])
        avg_latency = round(sum(latencies) / total_reqs, 2)
        p95_latency = round(latencies[int(total_reqs * 0.95)] if total_reqs > 1 else latencies[0], 2)
        min_latency = round(latencies[0], 2)
        max_latency = round(latencies[-1], 2)

        # 3. Tokens per request
        total_prompt_tok = sum(r.prompt_tokens for r in records)
        total_comp_tok = sum(r.completion_tokens for r in records)
        total_tok = sum(r.total_tokens for r in records)
        avg_tok = round(total_tok / total_reqs, 1)

        # 4. Cost per request
        total_cost = round(sum(r.cost_usd for r in records), 6)
        avg_cost = round(total_cost / total_reqs, 6)

        # 5. Tool calls per request
        total_tools = sum(r.tool_calls_count for r in records)
        avg_tools = round(total_tools / total_reqs, 2)

        # 6. Retry rate
        total_retries = sum(r.total_retries for r in records)
        overall_retry_rate = round((total_retries / max(1, total_reqs + total_retries)) * 100, 2)

        # 7. Cache hit rate
        cache_hits = sum(1 for r in records if r.cache_hit)
        cache_hit_rate = round((cache_hits / total_reqs) * 100, 2)

        # 8. Grounding failure rate
        total_grounding_failures = sum(r.grounding_failures_count for r in records)
        total_claims = sum(r.total_claims_count for r in records)
        grounding_failure_rate = (
            round((total_grounding_failures / total_claims) * 100, 2) if total_claims > 0 else 0.0
        )

        # 9. Task completion rate
        completed_tasks = sum(1 for r in records if r.task_completed)
        task_completion_rate = round((completed_tasks / total_reqs) * 100, 2)

        return {
            "total_requests": total_reqs,
            "ttft": {
                "avg_ms": avg_ttft,
                "p95_ms": p95_ttft,
                "count": len(ttfts),
            },
            "latency": {
                "avg_ms": avg_latency,
                "p95_ms": p95_latency,
                "min_ms": min_latency,
                "max_ms": max_latency,
            },
            "tokens": {
                "avg_per_request": avg_tok,
                "total": total_tok,
                "prompt_total": total_prompt_tok,
                "completion_total": total_comp_tok,
            },
            "cost": {
                "total_usd": total_cost,
                "avg_per_request_usd": avg_cost,
            },
            "tool_calls": {
                "avg_per_request": avg_tools,
                "total": total_tools,
            },
            "reliability": {
                "overall_retry_rate_pct": overall_retry_rate,
                "total_retries": total_retries,
            },
            "cache": {
                "hit_rate_pct": cache_hit_rate,
                "total_hits": cache_hits,
            },
            "grounding": {
                "failure_rate_pct": grounding_failure_rate,
                "total_failures": total_grounding_failures,
                "total_claims": total_claims,
            },
            "task_completion": {
                "completion_rate_pct": task_completion_rate,
                "completed": completed_tasks,
                "total": total_reqs,
            },
        }

    def clear(self) -> None:
        """Reset in-memory records and token usage for test isolation."""
        with self._lock:
            self._request_records.clear()
            self._in_memory_token_usage.clear()


finops = FinOpsManager()
