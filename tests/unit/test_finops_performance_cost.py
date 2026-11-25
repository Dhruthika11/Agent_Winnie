"""
Comprehensive Unit Tests for the FinOps & Performance Optimization Subsystem.

Tests all 9 core metrics:
1. TTFT (Time To First Token) - Initial response speed
2. End-to-end latency - Overall responsiveness
3. Tokens per request - Model efficiency
4. Cost per request - Financial efficiency (USD cost computation)
5. Tool calls per request - Planning efficiency
6. Retry rate - Reliability (tool, grounding, inference retries)
7. Cache hit rate - Cost reduction (cache hits vs total)
8. Grounding failure rate - Answer quality
9. Task completion rate - Actual usefulness

Also tests spending caps (80% degradation, 100% block) and REST API endpoints.
"""
import pytest
from starlette.testclient import TestClient

from src.main import app
from src.security.finops import (
    FinOpsManager,
    FinOpsMetricRecord,
    calculate_cost_usd,
    finops,
)


@pytest.fixture(autouse=True)
def clean_finops():
    """Ensure clean FinOps state before each test."""
    finops.clear()
    yield


client = TestClient(app)


def test_finops_metric_record_initialization_and_cost_calculation():
    """Verify FinOpsMetricRecord initializes all 9 metrics and calculates USD cost."""
    record = FinOpsMetricRecord(
        request_id="req_finops_1",
        session_id="sess_123",
        user_id="user_alice",
        tenant_id="tenant_prod",
        model_name="groq/qwen/qwen3.8-27b",
        ttft_ms=145.2,
        end_to_end_latency_ms=850.5,
        prompt_tokens=1000,
        completion_tokens=500,
        tool_calls_count=2,
        tool_retries=1,
        grounding_retries=0,
        cache_hit=False,
        grounding_failures_count=0,
        total_claims_count=3,
        task_completed=True,
    )

    assert record.total_tokens == 1500
    assert record.ttft_ms == 145.2
    assert record.end_to_end_latency_ms == 850.5
    assert record.cost_usd > 0.0
    assert record.tool_calls_count == 2
    assert record.total_retries == 1
    assert record.retry_rate > 0.0
    assert record.grounding_failure_rate == 0.0
    assert record.task_completed is True

    d = record.to_dict()
    assert d["request_id"] == "req_finops_1"
    assert d["model_name"] == "groq/qwen/qwen3.8-27b"
    assert "cost_usd" in d


def test_model_cost_calculation_matrix():
    """Verify financial cost calculation across different model price points."""
    # 1. Groq Qwen: $0.20 / 1M prompt, $0.20 / 1M completion
    cost_qwen = calculate_cost_usd("groq/qwen/qwen3.8-27b", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_qwen == 0.40

    # 2. GPT-4o: $2.50 / 1M prompt, $10.00 / 1M completion
    cost_gpt4o = calculate_cost_usd("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_gpt4o == 12.50

    # 3. DeepSeek: $0.14 / 1M prompt, $0.28 / 1M completion
    cost_deepseek = calculate_cost_usd("deepseek-chat", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_deepseek == 0.42

    # 4. Implemented Project Models: Groq Flagships & Fast Models
    cost_120b = calculate_cost_usd("groq/openai/gpt-oss-120b", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_120b == round(0.89 + 1.19, 2)

    cost_llama70b = calculate_cost_usd("groq/llama-3.3-70b-versatile", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_llama70b == round(0.59 + 0.79, 2)

    cost_llama8b = calculate_cost_usd("groq/llama-3.1-8b-instant", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_llama8b == round(0.05 + 0.08, 2)

    # 5. Implemented Project Models: Google Gemini Multimodal Pool
    cost_gemini_flash = calculate_cost_usd("gemini/gemini-1.5-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_gemini_flash == round(0.075 + 0.30, 3)

    cost_gemini_pro = calculate_cost_usd("gemini/gemini-1.5-pro", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_gemini_pro == round(1.25 + 5.00, 2)

    cost_gemini2_flash = calculate_cost_usd("gemini/gemini-2.0-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost_gemini2_flash == round(0.10 + 0.40, 2)


def test_finops_aggregated_analytics_all_9_metrics():
    """Verify aggregated statistics computation for all 9 metrics across requests."""
    # Request 1: Fast, clean, cached
    finops.record_request_metrics(FinOpsMetricRecord(
        request_id="req_1",
        user_id="user_bob",
        tenant_id="tenant_a",
        model_name="groq/qwen/qwen3.8-27b",
        ttft_ms=100.0,
        end_to_end_latency_ms=500.0,
        prompt_tokens=400,
        completion_tokens=200,
        tool_calls_count=1,
        total_retries=0,
        cache_hit=True,
        grounding_failures_count=0,
        total_claims_count=2,
        task_completed=True,
    ))

    # Request 2: Slower, had tool retries, grounding contradiction
    finops.record_request_metrics(FinOpsMetricRecord(
        request_id="req_2",
        user_id="user_bob",
        tenant_id="tenant_a",
        model_name="groq/qwen/qwen3.8-27b",
        ttft_ms=200.0,
        end_to_end_latency_ms=1500.0,
        prompt_tokens=600,
        completion_tokens=400,
        tool_calls_count=3,
        tool_retries=1,
        grounding_retries=1,
        cache_hit=False,
        grounding_failures_count=1,
        total_claims_count=2,
        task_completed=True,
    ))

    analytics = finops.get_performance_and_cost_analytics(tenant_id="tenant_a")

    assert analytics["total_requests"] == 2

    # 1. TTFT
    assert analytics["ttft"]["avg_ms"] == 150.0
    assert analytics["ttft"]["count"] == 2

    # 2. End-to-end latency
    assert analytics["latency"]["avg_ms"] == 1000.0
    assert analytics["latency"]["min_ms"] == 500.0
    assert analytics["latency"]["max_ms"] == 1500.0

    # 3. Tokens per request
    assert analytics["tokens"]["total"] == 1600  # (400+200) + (600+400)
    assert analytics["tokens"]["avg_per_request"] == 800.0

    # 4. Cost per request
    assert analytics["cost"]["total_usd"] > 0.0
    assert analytics["cost"]["avg_per_request_usd"] > 0.0

    # 5. Tool calls per request
    assert analytics["tool_calls"]["total"] == 4
    assert analytics["tool_calls"]["avg_per_request"] == 2.0

    # 6. Retry rate
    assert analytics["reliability"]["total_retries"] == 2
    assert analytics["reliability"]["overall_retry_rate_pct"] > 0.0

    # 7. Cache hit rate
    assert analytics["cache"]["total_hits"] == 1
    assert analytics["cache"]["hit_rate_pct"] == 50.0

    # 8. Grounding failure rate
    assert analytics["grounding"]["total_failures"] == 1
    assert analytics["grounding"]["failure_rate_pct"] == 25.0  # 1 / 4 claims = 25%

    # 9. Task completion rate
    assert analytics["task_completion"]["completed"] == 2
    assert analytics["task_completion"]["completion_rate_pct"] == 100.0


def test_finops_tenant_scoping_and_isolation():
    """Verify FinOps metrics can be scoped strictly to individual tenants."""
    finops.record_request_metrics(FinOpsMetricRecord(
        request_id="req_alpha",
        user_id="user_1",
        tenant_id="tenant_alpha",
        end_to_end_latency_ms=300.0,
        prompt_tokens=500,
        completion_tokens=250,
    ))

    finops.record_request_metrics(FinOpsMetricRecord(
        request_id="req_beta",
        user_id="user_2",
        tenant_id="tenant_beta",
        end_to_end_latency_ms=900.0,
        prompt_tokens=1000,
        completion_tokens=500,
    ))

    alpha_stats = finops.get_performance_and_cost_analytics(tenant_id="tenant_alpha")
    beta_stats = finops.get_performance_and_cost_analytics(tenant_id="tenant_beta")

    assert alpha_stats["total_requests"] == 1
    assert alpha_stats["tokens"]["total"] == 750

    assert beta_stats["total_requests"] == 1
    assert beta_stats["tokens"]["total"] == 1500


def test_finops_spending_cap_evaluation():
    """Verify daily token budgeting, degradation at 80%, and blocking at 100%."""
    manager = FinOpsManager()
    user = "budget_user_test"

    status, used, limit = manager.evaluate_budget(user)
    assert status == "ok"
    assert used == 0

    # Record 42,000 tokens (limit is 50,000 -> 84% > 80% degradation threshold)
    manager.record_usage(user, 42000)
    status, used, limit = manager.evaluate_budget(user)
    assert status == "degraded"
    assert used == 42000

    # Record another 10,000 tokens (52,000 > 50,000 -> hard block)
    manager.record_usage(user, 10000)
    status, used, limit = manager.evaluate_budget(user)
    assert status == "blocked"
    assert used == 52000


def test_finops_rest_api_endpoints():
    """Verify REST API endpoints for FinOps analytics, request records, and budget."""
    # Seed telemetry
    finops.record_request_metrics(FinOpsMetricRecord(
        request_id="req_api_finops",
        user_id="api_user",
        tenant_id="tenant_api",
        ttft_ms=120.0,
        end_to_end_latency_ms=600.0,
        prompt_tokens=300,
        completion_tokens=150,
        tool_calls_count=1,
        cache_hit=True,
    ))

    # 1. GET /api/finops/analytics
    res_analytics = client.get("/api/finops/analytics?tenant_id=tenant_api")
    assert res_analytics.status_code == 200
    data = res_analytics.json()
    assert data["total_requests"] == 1
    assert data["ttft"]["avg_ms"] == 120.0
    assert data["cache"]["hit_rate_pct"] == 100.0

    # 2. GET /api/finops/records
    res_records = client.get("/api/finops/records?tenant_id=tenant_api")
    assert res_records.status_code == 200
    r_data = res_records.json()
    assert r_data["count"] == 1
    assert r_data["records"][0]["request_id"] == "req_api_finops"

    # 3. GET /api/finops/budget
    res_budget = client.get("/api/finops/budget?user_id=api_user&tenant_id=tenant_api")
    assert res_budget.status_code == 200
    b_data = res_budget.json()
    assert b_data["status"] in ("ok", "degraded", "blocked")
    assert "tokens_used" in b_data
