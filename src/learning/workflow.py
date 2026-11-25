"""
LangGraph Self-Improvement Workflow for Hermes Agent Platform.
Orchestrates the complete closed-loop reliability lifecycle:
1. Failure Detection & Telemetry
2. Confidence-Aware Root Cause Analysis (with human triage branch for low confidence)
3. Regression Test Synthesis
4. Architectural Proposal Generation
5. Sandbox Benchmark Comparison & Release Gate Validation
6. Human-in-the-Loop (HITL) Staged Deployment
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict
from langgraph.graph import END, StateGraph

from src.learning.analyzer import failure_analyzer
from src.learning.database import failure_db
from src.learning.generator import proposal_generator
from src.learning.models import FailureRecord, RegressionTest
from src.learning.regression_harness import regression_harness
from src.learning.sandbox_validator import sandbox_validator
from src.tools import hitl

logger = logging.getLogger("hermes.learning.workflow")


class LearningState(TypedDict):
    failure_id: Optional[str]
    session_id: str
    user_id: str
    user_input: str
    observed_output: str
    expected_behavior: str
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    model_name: str
    failure_record: Optional[dict[str, Any]]
    regression_test: Optional[dict[str, Any]]
    proposal: Optional[dict[str, Any]]
    validation_result: Optional[dict[str, Any]]
    approval_id: Optional[str]
    status: str
    error: Optional[str]


def analyze_failure_node(state: LearningState) -> dict[str, Any]:
    """Analyze the failure to identify taxonomy category, responsible layer, confidence, and root cause."""
    try:
        record = failure_analyzer.analyze(
            user_input=state["user_input"],
            observed_output=state["observed_output"],
            expected_behavior=state.get("expected_behavior", ""),
            tool_calls=state.get("tool_calls", []),
            tool_outputs=state.get("tool_outputs", []),
            model_name=state.get("model_name", "default"),
            session_id=state.get("session_id", "unknown"),
            user_id=state.get("user_id", "unknown"),
        )
        status = "requires_human_triage" if record.requires_human_triage else "analyzed"
        return {
            "failure_id": record.failure_id,
            "failure_record": record.to_dict(),
            "status": status,
        }
    except Exception as exc:
        logger.exception("analyze_failure_node_failed")
        return {"error": str(exc), "status": "failed"}


def triage_confidence_router(state: LearningState) -> str:
    """Route low-confidence or ambiguous classifications to human triage before fix generation."""
    if state.get("status") == "failed" or state.get("error"):
        return END

    f_dict = state.get("failure_record") or {}
    confidence = f_dict.get("confidence", 1.0)
    requires_triage = f_dict.get("requires_human_triage", False)

    if confidence < 0.70 or requires_triage:
        return "await_human_triage"
    return "generate_regression_test"


def await_human_triage_node(state: LearningState) -> dict[str, Any]:
    """Request Human-in-the-Loop review when failure diagnosis confidence is low (< 0.70)."""
    f_dict = state.get("failure_record") or {}
    failure_id = state.get("failure_id") or f_dict.get("failure_id", "unknown")

    tool_calls = [{
        "name": "learning.human_triage_review",
        "arguments": {
            "failure_id": failure_id,
            "confidence": f_dict.get("confidence", 0.5),
            "evidence": f_dict.get("evidence", []),
            "alternative_causes": f_dict.get("alternative_causes", []),
            "user_input": state.get("user_input", "")[:100],
            "observed_output": state.get("observed_output", "")[:100],
        },
    }]

    approval_id = hitl.request_approval(
        user_id=state.get("user_id", "unknown"),
        session_id=state.get("session_id", "learning_triage"),
        tool_calls=tool_calls,
    )

    logger.info("low_confidence_failure_routed_to_human_triage", extra={"failure_id": failure_id, "approval_id": approval_id})
    return {
        "approval_id": approval_id,
        "status": "awaiting_human_triage",
    }


def generate_regression_test_node(state: LearningState) -> dict[str, Any]:
    """Convert confirmed failure into an automated regression test."""
    f_dict = state.get("failure_record")
    if not f_dict:
        return {"error": "Missing failure_record", "status": "failed"}

    try:
        record = FailureRecord(**f_dict)
        test = regression_harness.create_test_from_failure(record)
        return {
            "regression_test": test.to_dict(),
            "status": "test_generated",
        }
    except Exception as exc:
        logger.exception("generate_regression_test_node_failed")
        return {"error": str(exc), "status": "failed"}


def generate_proposal_node(state: LearningState) -> dict[str, Any]:
    """Generate minimal targeted improvement proposal mapped to the responsible architectural layer."""
    f_dict = state.get("failure_record")
    if not f_dict:
        return {"error": "Missing failure_record", "status": "failed"}

    try:
        record = FailureRecord(**f_dict)
        proposal = proposal_generator.generate_proposal(record)
        return {
            "proposal": proposal.to_dict(),
            "status": "proposal_generated",
        }
    except Exception as exc:
        logger.exception("generate_proposal_node_failed")
        return {"error": str(exc), "status": "failed"}


def sandbox_validate_node(state: LearningState) -> dict[str, Any]:
    """Validate proposal against regression benchmarks with a release gate and request HITL approval."""
    p_dict = state.get("proposal")
    t_dict = state.get("regression_test")
    if not p_dict:
        return {"error": "Missing proposal for validation", "status": "failed"}

    try:
        from src.learning.models import ImprovementProposal
        proposal = ImprovementProposal(**p_dict)
        test = RegressionTest(**t_dict) if t_dict else None
        res = sandbox_validator.validate_proposal(proposal, test=test)

        status = "awaiting_approval" if res.get("passed_gate") and res.get("approval_id") else (
            "gate_rejected" if not res.get("passed_gate") else "validation_complete"
        )

        return {
            "validation_result": res,
            "approval_id": res.get("approval_id"),
            "status": status,
        }
    except Exception as exc:
        logger.exception("sandbox_validate_node_failed")
        return {"error": str(exc), "status": "failed"}


def build_self_improvement_graph():
    """Build the LangGraph state machine for the Hermes closed-loop self-improving system."""
    graph = StateGraph(LearningState)

    graph.add_node("analyze_failure", analyze_failure_node)
    graph.add_node("await_human_triage", await_human_triage_node)
    graph.add_node("generate_regression_test", generate_regression_test_node)
    graph.add_node("generate_proposal", generate_proposal_node)
    graph.add_node("sandbox_validate", sandbox_validate_node)

    graph.set_entry_point("analyze_failure")

    # Confidence-aware branching:
    # High confidence (>= 0.70) -> proceed autonomously to regression test & proposal
    # Low confidence (< 0.70) -> route to human triage before generating patches
    graph.add_conditional_edges(
        "analyze_failure",
        triage_confidence_router,
        {
            "await_human_triage": "await_human_triage",
            "generate_regression_test": "generate_regression_test",
            END: END,
        },
    )

    graph.add_edge("await_human_triage", END)
    graph.add_edge("generate_regression_test", "generate_proposal")
    graph.add_edge("generate_proposal", "sandbox_validate")
    graph.add_edge("sandbox_validate", END)

    return graph.compile()


self_improvement_workflow = build_self_improvement_graph()
