"""
Adaptive Model Routing and Workflow Selection Optimization for Hermes.
Enforces cost-effective tiered model dispatching using models present in the platform:
- Tier 1 (Fast & Cheap: groq/openai/gpt-oss-20b or groq/llama-3.1-8b-instant):
  Intent classification, workflow selection, simple extraction, query rewriting.
- Tier 2 (Advanced Reasoning: groq/openai/gpt-oss-120b for architecture/logic, groq/qwen/qwen3.8-27b for coding):
  Complex reasoning, DAG planning, difficult code generation, cross-source synthesis.
- Tier 3 (Independent Verifier: gemini/gemini-1.5-flash / groq/llama-3.3-70b-versatile):
  Independent cross-provider grounding verification, citation checking, document inspection.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.evaluation.benchmark_data import load_benchmark_dataset
from src.evaluation.schemas import BenchmarkTestCase
from src.workflows.registry import classify_workflow

logger = logging.getLogger("hermes.workflows.optimization")


class ModelTier(str, Enum):
    FAST_CHEAP = "fast_cheap"                      # ~$0.15 / 1M tokens (e.g. groq/openai/gpt-oss-20b)
    ADVANCED_REASONING = "advanced_reasoning"      # ~$0.89 / 1M tokens (e.g. groq/openai/gpt-oss-120b, qwen3.8-27b)
    INDEPENDENT_VERIFIER = "independent_verifier"  # Cross-provider verification (e.g. gemini/gemini-1.5-flash)


class TaskType(str, Enum):
    # Tier 1 Tasks
    INTENT_CLASSIFICATION = "intent_classification"
    WORKFLOW_SELECTION = "workflow_selection"
    SIMPLE_EXTRACTION = "simple_extraction"
    QUERY_REWRITING = "query_rewriting"

    # Tier 2 Tasks
    COMPLEX_REASONING = "complex_reasoning"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    FINAL_SYNTHESIS = "final_synthesis"

    # Tier 3 Tasks
    GROUNDING_VERIFICATION = "grounding_verification"
    DOCUMENT_INSPECTION = "document_inspection"


class ModelSelection(BaseModel):
    task_type: TaskType
    model_tier: ModelTier
    model_name: str
    estimated_cost_per_1k_tokens: float
    reason: str


class AdaptiveModelRouter:
    """Selects optimal model tier for each pipeline node using models configured in the platform."""

    def __init__(
        self,
        tier1_model: Optional[str] = None,
        tier2_reasoning_model: Optional[str] = None,
        tier2_coding_model: Optional[str] = None,
        tier3_verifier_model: Optional[str] = None,
    ):
        from src.config import settings
        from src.security.finops import MODEL_PRICING_RATES

        self._tier1_model = (
            tier1_model
            or getattr(settings, "tier1_model", "")
            or "groq/openai/gpt-oss-20b"
        )
        self._tier2_reasoning_model = (
            tier2_reasoning_model
            or getattr(settings, "tier2_reasoning_model", "")
            or "groq/openai/gpt-oss-120b"
        )
        self._tier2_coding_model = (
            tier2_coding_model
            or getattr(settings, "tier2_coding_model", "")
            or "groq/qwen/qwen3.8-27b"
        )
        self._tier3_verifier_model = (
            tier3_verifier_model
            or getattr(settings, "tier3_verifier_model", "")
            or "gemini/gemini-1.5-flash"
        )

        self._tier_models = {
            ModelTier.FAST_CHEAP: self._tier1_model,
            ModelTier.ADVANCED_REASONING: self._tier2_reasoning_model,
            ModelTier.INDEPENDENT_VERIFIER: self._tier3_verifier_model,
        }
        self._pricing = MODEL_PRICING_RATES

    def _get_rate_per_1k(self, model_name: str) -> float:
        rates = self._pricing.get(model_name, self._pricing.get("default", {"prompt": 0.20}))
        return round(rates.get("prompt", 0.20) / 1000.0, 6)

    def route_task(self, task_type: TaskType | str) -> ModelSelection:
        """Route specific subtask to optimal model tier."""
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                task_type = TaskType.COMPLEX_REASONING

        # Tier 1 assignments (fast, lightweight query & routing)
        if task_type in (
            TaskType.INTENT_CLASSIFICATION,
            TaskType.WORKFLOW_SELECTION,
            TaskType.SIMPLE_EXTRACTION,
            TaskType.QUERY_REWRITING,
        ):
            tier = ModelTier.FAST_CHEAP
            model = self._tier1_model
            reason = f"Fast, lightweight model ({model}) optimal for sub-200ms classification and query reformulations."

        # Tier 3 assignments (independent cross-provider or cross-model verifier)
        elif task_type in (
            TaskType.GROUNDING_VERIFICATION,
            TaskType.DOCUMENT_INSPECTION,
        ):
            tier = ModelTier.INDEPENDENT_VERIFIER
            model = self._tier3_verifier_model
            reason = f"Independent cross-provider verifier ({model}) prevents common-mode hallucinations during audit gates."

        # Tier 2 coding assignments (syntax accuracy and sandbox unit testing)
        elif task_type == TaskType.CODE_GENERATION:
            tier = ModelTier.ADVANCED_REASONING
            model = self._tier2_coding_model
            reason = f"Specialized coding model ({model}) required for syntax precision, tool JSON schemas, and debugging."

        # Tier 2 complex reasoning assignments (architecture, DAG planning, and synthesis)
        else:
            tier = ModelTier.ADVANCED_REASONING
            model = self._tier2_reasoning_model
            reason = f"Flagship high-capacity reasoning model ({model}) required for multi-step planning and cross-source synthesis."

        rate = self._get_rate_per_1k(model)

        return ModelSelection(
            task_type=task_type,
            model_tier=tier,
            model_name=model,
            estimated_cost_per_1k_tokens=rate,
            reason=reason,
        )


class WorkflowSelectionEvaluationReport(BaseModel):
    total_samples: int
    correct_selections: int
    classification_accuracy: float
    per_workflow_accuracy: dict[str, dict[str, Any]] = Field(default_factory=dict)
    misclassifications: list[dict[str, Any]] = Field(default_factory=list)


class WorkflowSelectionEvaluator:
    """Evaluates the router's workflow selection accuracy against the benchmark dataset."""

    def evaluate_benchmark(
        self,
        cases: Optional[list[BenchmarkTestCase]] = None,
    ) -> WorkflowSelectionEvaluationReport:
        all_cases = cases or load_benchmark_dataset()

        correct = 0
        total = len(all_cases)
        per_wf: dict[str, dict[str, int]] = {}
        misclassified: list[dict[str, Any]] = []

        for case in all_cases:
            exp = case.expected_workflow
            if exp not in per_wf:
                per_wf[exp] = {"total": 0, "correct": 0}
            per_wf[exp]["total"] += 1

            decision = classify_workflow(case.input)
            act = decision.workflow

            if act == exp:
                correct += 1
                per_wf[exp]["correct"] += 1
            else:
                misclassified.append({
                    "test_id": case.test_id,
                    "input": case.input[:80],
                    "expected": exp,
                    "actual": act,
                    "confidence": decision.confidence,
                })

        accuracy = round((correct / max(total, 1)) * 100, 2)
        summary_per_wf: dict[str, dict[str, Any]] = {}
        for wf, counts in per_wf.items():
            tot = counts["total"]
            corr = counts["correct"]
            summary_per_wf[wf] = {
                "total": tot,
                "correct": corr,
                "accuracy": round((corr / max(tot, 1)) * 100, 2),
            }

        return WorkflowSelectionEvaluationReport(
            total_samples=total,
            correct_selections=correct,
            classification_accuracy=accuracy,
            per_workflow_accuracy=summary_per_wf,
            misclassifications=misclassified,
        )


adaptive_model_router = AdaptiveModelRouter()
workflow_selection_evaluator = WorkflowSelectionEvaluator()
