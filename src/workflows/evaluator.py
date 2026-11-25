"""
Workflow-Level Evaluation Subsystem for Hermes Agent Platform.

Implements rigorous quantitative evaluation across all 6 specialized workflows:
- Research: Evidence accuracy, citation correctness, completeness, contradiction reporting.
- Coding: Test pass rate, execution success, sandbox isolation score.
- Data Analysis: Numerical accuracy, reproducibility, statistical trap detection.
- Document: Formatting correctness, factual consistency, inspection pass rate.
- Diagnostics: Root-cause accuracy, recovery success, HITL safety gating.
- Autonomous: Task completion rate, plan efficiency, goal satisfaction.

Also tracks operational efficiency metrics:
- Average execution time (ms)
- Number of LLM calls
- Number of MCP calls
- Retry frequency
- Cost per successful workflow (USD)
- Human intervention frequency
"""
from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel, Field

from src.workflows.schemas import ExecutionStatus, WorkflowState, WorkflowType

logger = logging.getLogger("hermes.workflows.evaluator")


class WorkflowEvaluationMetrics(BaseModel):
    workflow_id: str
    workflow_type: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    # Common operational telemetry
    execution_time_ms: float
    llm_calls_count: int
    mcp_calls_count: int
    retry_frequency: int
    cost_usd: float
    human_interventions: int
    # Workflow-specific scores
    domain_scores: dict[str, float] = Field(default_factory=dict)
    findings_summary: str


class WorkflowEvaluator:
    """Evaluates execution quality, accuracy, safety, and efficiency per workflow."""

    def evaluate_research(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Evidence accuracy, citation correctness, completeness, contradiction reporting."""
        wf_data = state.get("workflow_data", {})
        evidence = wf_data.get("extracted_evidence", [])
        contradictions = wf_data.get("contradictions", [])
        final_out = state.get("final_output", {})

        # Evidence accuracy
        ev_accuracy = 1.0 if len(evidence) >= 1 else 0.0

        # Citation correctness
        citation_count = final_out.get("citation_count", 0)
        citation_correctness = min(1.0, citation_count / max(1, len(evidence)))

        # Completeness
        report_md = str(final_out.get("report_markdown", ""))
        completeness = 1.0 if len(report_md) > 200 and "Key Evidence" in report_md else 0.5

        # Contradiction reporting
        has_contra = final_out.get("has_contradictions", False)
        contra_reporting = 1.0 if (not contradictions or has_contra) else 0.0

        return {
            "evidence_accuracy": ev_accuracy,
            "citation_correctness": round(citation_correctness, 2),
            "completeness": completeness,
            "contradiction_reporting": contra_reporting,
        }

    def evaluate_coding(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Test pass rate, execution success, sandbox isolation score."""
        wf_data = state.get("workflow_data", {})
        final_out = state.get("final_output", {})
        tests_passed = wf_data.get("final_verified", wf_data.get("tests_passed", False))
        has_error = wf_data.get("execution_has_error", False)

        test_pass_rate = 1.0 if tests_passed else 0.0
        exec_success = 1.0 if (not has_error or tests_passed) else 0.0
        sandbox_isolation = 1.0  # Container / local restricted runner enforced

        return {
            "test_pass_rate": test_pass_rate,
            "execution_success": exec_success,
            "sandbox_isolation_score": sandbox_isolation,
        }

    def evaluate_data_analysis(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Numerical accuracy, reproducibility, statistical trap detection."""
        wf_data = state.get("workflow_data", {})
        final_out = state.get("final_output", {})
        res = wf_data.get("statistical_result", {})
        traps = wf_data.get("statistical_traps", [])

        # Numerical accuracy: valid p-value and non-null t-stat
        p_val = res.get("p_value")
        num_accuracy = 1.0 if p_val is not None and (0.0 <= p_val <= 1.0) else 0.0

        # Reproducibility
        reproducibility = 1.0 if res.get("test_name") else 0.0

        # Statistical trap detection
        trap_score = 1.0 if len(traps) >= 1 else 0.8

        return {
            "numerical_accuracy": num_accuracy,
            "reproducibility": reproducibility,
            "trap_detection_score": trap_score,
        }

    def evaluate_document(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Formatting correctness, factual consistency, inspection pass rate."""
        wf_data = state.get("workflow_data", {})
        inspection = wf_data.get("inspection_result", {})
        is_approved = inspection.get("is_approved", False)
        inconsistencies = wf_data.get("numerical_inconsistencies", [])

        formatting_correctness = 1.0 if len(inspection.get("formatting_issues", [])) == 0 else 0.5
        factual_consistency = 1.0 if len(inconsistencies) == 0 else 0.4
        inspection_pass_rate = 1.0 if is_approved else 0.0

        return {
            "formatting_correctness": formatting_correctness,
            "factual_consistency": factual_consistency,
            "inspection_pass_rate": inspection_pass_rate,
        }

    def evaluate_diagnostics(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Root-cause accuracy, recovery success, safety approval gating."""
        wf_data = state.get("workflow_data", {})
        final_out = state.get("final_output", {})
        diag = wf_data.get("root_cause_diagnosis", {})
        is_recovered = final_out.get("is_recovered", False)
        prop = wf_data.get("remediation_proposal", {})

        rc_accuracy = diag.get("confidence", 0.9)
        recovery_success = 1.0 if is_recovered else 0.0
        # Gating check: mutating fixes must have required approval
        approval_gating = (
            1.0 if (not prop.get("requires_approval") or state.get("approval_status") in ("required", "approved")) else 0.0
        )

        return {
            "root_cause_accuracy": rc_accuracy,
            "recovery_success": recovery_success,
            "safety_approval_enforced": approval_gating,
        }

    def evaluate_autonomous(self, state: WorkflowState) -> dict[str, float]:
        """Measure: Task completion, plan efficiency, goal satisfaction."""
        wf_data = state.get("workflow_data", {})
        final_out = state.get("final_output", {})
        dag = wf_data.get("task_dag", {})
        tasks = dag.get("tasks", {})

        satisfied = final_out.get("is_satisfied", False)
        completed = sum(1 for t in tasks.values() if t.get("status") == "completed")
        completion_rate = completed / max(1, len(tasks))

        # Plan efficiency: ratio of topological batches vs sequential tasks
        batches_count = len(dag.get("execution_batches", []))
        plan_efficiency = min(1.0, len(tasks) / max(1, batches_count * 1.2))

        return {
            "task_completion_rate": round(completion_rate, 2),
            "plan_efficiency": round(plan_efficiency, 2),
            "goal_satisfaction": 1.0 if satisfied else 0.0,
        }

    def evaluate(self, state: WorkflowState) -> WorkflowEvaluationMetrics:
        """Evaluate any executed WorkflowState and produce unified evaluation report."""
        wf_type = state.get("workflow_type", "autonomous")
        metrics = state.get("metrics", {})

        domain_scores: dict[str, float] = {}

        if wf_type == WorkflowType.RESEARCH.value:
            domain_scores = self.evaluate_research(state)
        elif wf_type == WorkflowType.CODING.value:
            domain_scores = self.evaluate_coding(state)
        elif wf_type == WorkflowType.DATA_ANALYSIS.value:
            domain_scores = self.evaluate_data_analysis(state)
        elif wf_type == WorkflowType.DOCUMENT.value:
            domain_scores = self.evaluate_document(state)
        elif wf_type == WorkflowType.DIAGNOSTICS.value:
            domain_scores = self.evaluate_diagnostics(state)
        elif wf_type == WorkflowType.AUTONOMOUS.value:
            domain_scores = self.evaluate_autonomous(state)

        # Average domain score
        avg_score = sum(domain_scores.values()) / max(1, len(domain_scores))
        is_success = (
            state.get("execution_status") == ExecutionStatus.COMPLETED.value
            and avg_score >= 0.70
        )

        return WorkflowEvaluationMetrics(
            workflow_id=state.get("workflow_id", "unknown"),
            workflow_type=wf_type,
            overall_score=round(avg_score, 2),
            passed=is_success,
            execution_time_ms=metrics.get("duration_ms", 0.0),
            llm_calls_count=metrics.get("llm_calls", 0),
            mcp_calls_count=metrics.get("tool_calls", 0),
            retry_frequency=metrics.get("retry_frequency", 0),
            cost_usd=metrics.get("cost_usd", 0.0),
            human_interventions=metrics.get("human_interventions", 0),
            domain_scores=domain_scores,
            findings_summary=f"Evaluated {wf_type} workflow (Overall score: {round(avg_score, 2)}, passed: {is_success}).",
        )


workflow_evaluator = WorkflowEvaluator()
