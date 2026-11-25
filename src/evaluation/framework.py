"""
Agent Evaluation Framework Runner for Hermes Agent Platform.
Executes test cases from the fixed benchmark suite against Hermes workflows,
evaluates across the 5 dimensions, and aggregates quantitative telemetry and scores.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from src.evaluation.benchmark_data import load_benchmark_dataset
from src.evaluation.evaluators import (
    GroundingEvaluator,
    ReasoningPlanningEvaluator,
    TaskCompletionEvaluator,
    ToolExecutionEvaluator,
    WorkflowReliabilityEvaluator,
)
from src.evaluation.schemas import (
    BenchmarkTestCase,
    CategoryScore,
    EvaluationCategory,
    EvaluationReport,
    EvaluationResult,
)
from src.workflows.executor import workflow_executor
from src.workflows.schemas import WorkflowState

logger = logging.getLogger("hermes.evaluation.framework")


class AgentEvaluationFramework:
    """Coordinates benchmark evaluation runs across all 5 categories."""

    def __init__(self):
        self.task_evaluator = TaskCompletionEvaluator()
        self.reasoning_evaluator = ReasoningPlanningEvaluator()
        self.tool_evaluator = ToolExecutionEvaluator()
        self.grounding_evaluator = GroundingEvaluator()
        self.reliability_evaluator = WorkflowReliabilityEvaluator()

    def evaluate_state(self, case: BenchmarkTestCase, state: WorkflowState) -> EvaluationResult:
        """Score an executed WorkflowState across all 5 evaluation categories."""
        start_t = time.perf_counter()

        cat_scores: dict[str, CategoryScore] = {
            EvaluationCategory.TASK_COMPLETION.value: self.task_evaluator.evaluate(case, state),
            EvaluationCategory.REASONING_PLANNING.value: self.reasoning_evaluator.evaluate(case, state),
            EvaluationCategory.TOOL_EXECUTION.value: self.tool_evaluator.evaluate(case, state),
            EvaluationCategory.GROUNDING.value: self.grounding_evaluator.evaluate(case, state),
            EvaluationCategory.WORKFLOW_RELIABILITY.value: self.reliability_evaluator.evaluate(case, state),
        }

        # Overall score is the mean across the 5 dimensions
        avg_score = sum(cs.score for cs in cat_scores.values()) / len(cat_scores)
        overall_score = round(avg_score, 3)
        passed = overall_score >= 0.70 and all(cs.score >= 0.50 for cs in cat_scores.values())

        metrics = state.get("metrics", {})
        res = EvaluationResult(
            test_id=case.test_id,
            input=case.input,
            expected_workflow=case.expected_workflow,
            actual_workflow=state.get("workflow_type"),
            passed=passed,
            overall_score=overall_score,
            category_scores=cat_scores,
            execution_time_ms=metrics.get("duration_ms", (time.perf_counter() - start_t) * 1000),
            llm_calls_count=metrics.get("llm_calls", 0),
            tool_calls_count=metrics.get("tool_calls", 0),
            cost_usd=metrics.get("cost_usd", 0.0),
            error=state.get("error"),
        )
        return res

    def run_case(
        self,
        case: BenchmarkTestCase,
        custom_workflow_data: Optional[dict[str, Any]] = None,
        custom_runner: Optional[Callable[[BenchmarkTestCase], WorkflowState]] = None,
    ) -> EvaluationResult:
        """Run a single benchmark test case through Hermes and score it."""
        try:
            if custom_runner is not None:
                state = custom_runner(case)
            else:
                wf_data = custom_workflow_data or {}
                # Provide mock search results for research test cases if running headless
                if case.expected_workflow == "research" and not wf_data.get("raw_search_results"):
                    wf_data["raw_search_results"] = [
                        {
                            "subquestion_id": "sub_1",
                            "query": case.input[:40],
                            "status": "success",
                            "output": f"Empirical findings regarding {case.input[:30]} with 99.4% precision.",
                            "evidence_item": {
                                "url": f"https://arxiv.org/abs/{case.test_id.lower()}",
                                "source": f"Arxiv {case.test_id}",
                                "content": f"Empirical findings regarding {case.input[:30]} with 99.4% precision.",
                            },
                        }
                    ]

                state = workflow_executor.run(
                    user_request=case.input,
                    workflow_type=case.expected_workflow,
                    workflow_data=wf_data,
                )

            return self.evaluate_state(case, state)
        except Exception as exc:
            logger.exception("case_execution_failed", extra={"test_id": case.test_id})
            return EvaluationResult(
                test_id=case.test_id,
                input=case.input,
                expected_workflow=case.expected_workflow,
                actual_workflow=None,
                passed=False,
                overall_score=0.0,
                error=str(exc),
            )

    def run_benchmark(
        self,
        cases: Optional[list[BenchmarkTestCase]] = None,
        workflow_filter: Optional[str] = None,
        limit: Optional[int] = None,
        custom_workflow_data: Optional[dict[str, Any]] = None,
        custom_runner: Optional[Callable[[BenchmarkTestCase], WorkflowState]] = None,
    ) -> EvaluationReport:
        """Run the evaluation benchmark suite across cases and generate an aggregated report."""
        all_cases = cases or load_benchmark_dataset()

        if workflow_filter:
            all_cases = [c for c in all_cases if c.expected_workflow == workflow_filter]
        if limit and limit > 0:
            all_cases = all_cases[:limit]

        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        results: list[EvaluationResult] = []

        total_exec_time = 0.0
        total_cost = 0.0

        for case in all_cases:
            res = self.run_case(case, custom_workflow_data=custom_workflow_data, custom_runner=custom_runner)
            results.append(res)
            total_exec_time += res.execution_time_ms
            total_cost += res.cost_usd

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = round((passed_count / max(total_count, 1)) * 100, 2)
        avg_score = round(sum(r.overall_score for r in results) / max(total_count, 1), 3)

        # Compute category averages
        cat_sums: dict[str, float] = {}
        cat_counts: dict[str, int] = {}
        for r in results:
            for cat, c_score in r.category_scores.items():
                cat_sums[cat] = cat_sums.get(cat, 0.0) + c_score.score
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_avgs = {
            cat: round(cat_sums[cat] / max(cat_counts[cat], 1), 3)
            for cat in cat_sums
        }

        # Breakdown by workflow
        wf_breakdown: dict[str, dict[str, Any]] = {}
        for r in results:
            wf = r.expected_workflow
            if wf not in wf_breakdown:
                wf_breakdown[wf] = {"total": 0, "passed": 0, "scores": []}
            wf_breakdown[wf]["total"] += 1
            if r.passed:
                wf_breakdown[wf]["passed"] += 1
            wf_breakdown[wf]["scores"].append(r.overall_score)

        for wf, data in wf_breakdown.items():
            tot = data["total"]
            pas = data["passed"]
            scs = data["scores"]
            wf_breakdown[wf] = {
                "total": tot,
                "passed": pas,
                "pass_rate": round((pas / max(tot, 1)) * 100, 2),
                "avg_score": round(sum(scs) / max(len(scs), 1), 3),
            }

        return EvaluationReport(
            run_id=run_id,
            total_cases=total_count,
            passed_cases=passed_count,
            pass_rate=pass_rate,
            overall_avg_score=avg_score,
            category_averages=cat_avgs,
            workflow_breakdown=wf_breakdown,
            total_execution_time_ms=round(total_exec_time, 2),
            total_cost_usd=round(total_cost, 4),
            case_results=results,
        )


agent_evaluation_framework = AgentEvaluationFramework()
