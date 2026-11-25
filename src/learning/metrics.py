"""
Failure Intelligence and Reliability Metrics Engine for Hermes.
Tracks system-level reliability, hallucination rates, tool health, and benchmark progression.
"""
from __future__ import annotations

import logging
from typing import Any

from src.learning.database import failure_db
from src.learning.regression_harness import regression_harness

logger = logging.getLogger("hermes.learning.metrics")


class ReliabilityMetricsEngine:
    """Computes failure analytics and benchmark trends over time."""

    def compute_dashboard_metrics(self) -> dict[str, Any]:
        """Aggregate comprehensive reliability and learning loop telemetry."""
        failures = failure_db.list_failures(limit=500)
        clusters = failure_db.list_clusters()
        proposals = failure_db.list_proposals()
        versions = failure_db.list_versions()
        benchmark = regression_harness.run_benchmark()

        total_failures = len(failures)
        category_breakdown: dict[str, int] = {}
        for f in failures:
            cat = f.category.value if hasattr(f.category, "value") else str(f.category)
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
            # Also record backwards-compatible aliases in category breakdown
            if cat == "grounding_failure":
                category_breakdown["context_grounding"] = category_breakdown.get("context_grounding", 0) + 1
            elif cat == "tool_failure":
                category_breakdown["system_failure"] = category_breakdown.get("system_failure", 0) + 1
            elif cat == "retrieval_failure":
                category_breakdown["tool_orchestration"] = category_breakdown.get("tool_orchestration", 0) + 1

        recurring_failures = sum(1 for f in failures if f.is_recurring)
        grounding_failures = category_breakdown.get("grounding_failure", 0) or category_breakdown.get("context_grounding", 0)
        system_failures = category_breakdown.get("tool_failure", 0) or category_breakdown.get("system_failure", 0)

        # Estimate grounding accuracy and hallucination rate based on recorded failures
        hallucination_rate = round((grounding_failures / max(total_failures, 1)) * 100, 1) if total_failures > 0 else 0.0

        return {
            "total_failures_recorded": total_failures,
            "total_failures": total_failures,
            "recurring_failures_count": recurring_failures,
            "active_failure_clusters": len(clusters),
            "category_breakdown": category_breakdown,
            "failures_by_category": category_breakdown,
            "proposals_generated": len(proposals),
            "proposals_approved": sum(1 for p in proposals if p.approval_status == "approved"),
            "improvements_deployed": len(versions),
            "benchmark_tests_count": benchmark["total_tests"],
            "benchmark_pass_rate_percent": benchmark["pass_rate"],
            "benchmark_summary": benchmark,
            "estimated_hallucination_rate_percent": hallucination_rate,
            "grounding_failure_rate_pct": hallucination_rate,
            "learning_maturity": (
                "Advancing: benchmark regression coverage established"
                if benchmark["total_tests"] > 0
                else "Initializing: observing baseline interaction telemetry"
            ),
        }

    compute_metrics = compute_dashboard_metrics


metrics_engine = ReliabilityMetricsEngine()
