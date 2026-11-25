"""
Comprehensive Unit Tests for the Hermes Self-Improving Failure-Correction Subsystem.

Tests:
1. Context-Grounding Failure Detection & Classification (Adaptoid-OS hallucination scenario)
2. System Failure Detection & Code-Fix routing (tool crash / null byte / overflow)
3. Knowledge Failure Detection & Skill-Update routing (procedural API knowledge)
4. Regression Test Synthesis & Negative Assertions
5. Proposal Generation adhering to strict layer boundaries (Skill vs Code vs Prompt vs Validation)
6. Sandbox Benchmark Validation & HITL-gated deployment
7. End-to-End LangGraph Self-Improvement Workflow
8. Reliability & Failure Intelligence Metrics Engine
9. REST API Endpoints (/api/learning/*)
"""
from typing import Any
import pytest
from starlette.testclient import TestClient

from src.main import app
from src.learning.models import (
    FailureCategory,
    CorrectionType,
    FailureStatus,
    FailureRecord,
    RegressionTest,
    ImprovementProposal,
)
from src.learning.database import failure_db
from src.learning.analyzer import failure_analyzer
from src.learning.regression_harness import regression_harness
from src.learning.generator import proposal_generator
from src.learning.sandbox_validator import sandbox_validator
from src.learning.workflow import self_improvement_workflow
from src.learning.metrics import metrics_engine


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure in-memory state or database is initialized cleanly for each test."""
    failure_db._mem_failures.clear()
    failure_db._mem_clusters.clear()
    failure_db._mem_proposals.clear()
    failure_db._mem_regression_tests.clear()
    failure_db._mem_versions.clear()
    yield


def test_context_grounding_failure_adaptoid_scenario():
    """Verify that when tool output has valid model data but the agent hallucinates

    an unrelated repo ('Adaptoid-OS'), the analyzer classifies it as CONTEXT_GROUNDING
    and routes remediation to RESPONSE_VALIDATION (never a misleading skill).
    """
    record = FailureRecord(
        failure_id="fail_test_adaptoid",
        session_id="sess_opencode",
        user_id="user_123",
        user_input="What are the free models currently available on OpenCode?",
        observed_output="I don't have any information about the Adaptoid-OS repository you mentioned.",
        expected_behavior="List the free models on OpenCode like DeepSeek V4.1 Flash, Nemotron 3.5 Lightning.",
        tool_calls=[
            {"tool": "web_scraper.search", "arguments": {"query": "OpenCode free models Zen"}}
        ],
        tool_outputs=[
            {
                "tool": "web_scraper.search",
                "output": "OpenCode Zen free models include DeepSeek V4.1 Flash, Nemotron 3.5 Lightning, MiMo-V2.6.",
            }
        ],
        model_name="groq/openai/gpt-oss-20b",
    )

    analyzed = failure_analyzer.analyze(record)

    assert analyzed.category == FailureCategory.CONTEXT_GROUNDING_FAILURE
    assert analyzed.recommended_correction == CorrectionType.RESPONSE_VALIDATION
    assert "Adaptoid-OS" in analyzed.root_cause
    assert analyzed.originating_stage == "synthesis"
    assert analyzed.status == FailureStatus.ANALYZED


def test_system_failure_classification_on_tool_crash():
    """Verify that unhandled tool crashes or runtime faults are classified as SYSTEM_FAILURE

    and routed to CODE_FIX.
    """
    record = FailureRecord(
        failure_id="fail_test_crash",
        session_id="sess_crash",
        user_id="user_123",
        user_input="Fetch the latest PDF report",
        observed_output="Internal Server Error: unhandled null byte \\x00 in database",
        expected_behavior="Cleanly strip null bytes and return PDF text",
        tool_calls=[{"tool": "web_scraper.read_url", "arguments": {"url": "https://example.com/doc.pdf"}}],
        tool_outputs=[{"tool": "web_scraper.read_url", "output": {"error": "unsupported Unicode escape sequence: \\u0000"}}],
    )

    analyzed = failure_analyzer.analyze(record)

    assert analyzed.category == FailureCategory.SYSTEM_FAILURE
    assert analyzed.recommended_correction == CorrectionType.CODE_FIX
    assert "tool crash" in analyzed.root_cause.lower()


def test_knowledge_failure_classification_on_procedural_gap():
    """Verify that procedural knowledge gaps are classified as KNOWLEDGE_FAILURE

    and routed to SKILL_UPDATE.
    """
    record = FailureRecord(
        failure_id="fail_test_knowledge",
        session_id="sess_proc",
        user_id="user_123",
        user_input="How do I configure the Hermes MCP rate limiter with token bucket?",
        observed_output="I am not sure how to configure the Hermes token bucket rate limiter.",
        expected_behavior="Provide the step-by-step procedure to configure the token bucket rate limiter.",
        tool_calls=[],
        tool_outputs=[],
    )

    analyzed = failure_analyzer.analyze(record)

    assert analyzed.category == FailureCategory.KNOWLEDGE_FAILURE
    assert analyzed.recommended_correction == CorrectionType.SKILL_UPDATE


def test_regression_test_synthesis_and_negative_assertions():
    """Verify that every confirmed failure generates a versioned regression test

    with positive expected properties and strict negative assertions.
    """
    record = FailureRecord(
        failure_id="fail_test_synth",
        session_id="sess_synth",
        user_id="user_123",
        user_input="Which models are free on OpenCode?",
        observed_output="I cannot find anything about Adaptoid-OS.",
        expected_behavior="DeepSeek V4.1 Flash Free, Nemotron 3.5 Lightning Free",
        category=FailureCategory.CONTEXT_GROUNDING_FAILURE,
        recommended_correction=CorrectionType.RESPONSE_VALIDATION,
        root_cause="Hallucinated entity Adaptoid-OS despite tool output containing valid models",
    )

    test = regression_harness.create_regression_test(record)

    assert test.test_id.startswith("REG-")
    assert test.user_input == record.user_input
    assert "Adaptoid-OS" in test.negative_assertions

    # Test candidate output containing the hallucinated entity -> MUST FAIL
    bad_output = "The free models on OpenCode include DeepSeek, but Adaptoid-OS is unavailable."
    bad_result = regression_harness.evaluate_candidate_output(test, bad_output)
    assert bad_result["passed"] is False
    assert any("Adaptoid-OS" in err for err in bad_result["errors"])

    # Test candidate output without hallucinated entity -> MUST PASS
    good_output = "The free models on OpenCode include DeepSeek V4.1 Flash and Nemotron 3.5 Lightning."
    good_result = regression_harness.evaluate_candidate_output(test, good_output)
    assert good_result["passed"] is True
    assert len(good_result["errors"]) == 0


def test_proposal_generation_adheres_to_architectural_layers():
    """Verify that the generator enforces the rule:

    Knowledge -> Skill; Reasoning -> Prompt; Grounding -> Output Validation; System -> Code fix.
    """
    # 1. Grounding failure -> Output Validation
    grounding_rec = FailureRecord(
        failure_id="fail_g",
        user_input="OpenCode free models",
        observed_output="Adaptoid-OS not found",
        category=FailureCategory.CONTEXT_GROUNDING_FAILURE,
        recommended_correction=CorrectionType.RESPONSE_VALIDATION,
        root_cause="Hallucinated Adaptoid-OS",
    )
    p_grounding = proposal_generator.generate_proposal(grounding_rec)
    assert p_grounding.correction_type == CorrectionType.RESPONSE_VALIDATION
    assert "output_guards" in p_grounding.target_file

    # 2. Knowledge failure -> Skill Update
    knowledge_rec = FailureRecord(
        failure_id="fail_k",
        user_input="How to use Cloudflare R2 backup",
        observed_output="I do not know",
        category=FailureCategory.KNOWLEDGE_FAILURE,
        recommended_correction=CorrectionType.SKILL_UPDATE,
        root_cause="Missing procedure",
    )
    p_knowledge = proposal_generator.generate_proposal(knowledge_rec)
    assert p_knowledge.correction_type == CorrectionType.SKILL_UPDATE
    assert p_knowledge.target_file.endswith("SKILL.md")

    # 3. System failure -> Code Fix
    system_rec = FailureRecord(
        failure_id="fail_s",
        user_input="Fetch binary PDF",
        observed_output="Null byte error",
        category=FailureCategory.SYSTEM_FAILURE,
        recommended_correction=CorrectionType.CODE_FIX,
        root_cause="Unhandled null bytes in PDF parser",
    )
    p_system = proposal_generator.generate_proposal(system_rec)
    assert p_system.correction_type == CorrectionType.CODE_FIX
    assert p_system.target_file.startswith("src/")


def test_sandbox_validator_and_hitl_approval_gate():
    """Verify that proposals are validated against regression benchmarks in a sandbox

    and require explicit HITL approval before deployment.
    """
    # Create regression benchmark
    test = RegressionTest(
        test_id="REG-VAL-1",
        failure_id="fail_v1",
        user_input="OpenCode free models",
        expected_properties=["DeepSeek"],
        negative_assertions=["Adaptoid-OS"],
    )
    failure_db.save_regression_test(test)

    # Create proposal
    proposal = ImprovementProposal(
        proposal_id="PROP-VAL-1",
        failure_id="fail_v1",
        correction_type=CorrectionType.RESPONSE_VALIDATION,
        target_file="src/agent/output_guards.py",
        proposed_diff="+ filter_unrelated_entities",
        summary="Filter ungrounded entity mentions",
        rationale="Eliminate Adaptoid-OS hallucinations",
    )
    failure_db.save_proposal(proposal)

    # Validate proposal in sandbox
    val_result = sandbox_validator.validate_proposal(proposal.proposal_id)
    assert val_result["status"] in ("passed", "tested")
    assert val_result["approval_id"] is not None

    # Verify that deployment succeeds with HITL approval
    version = sandbox_validator.approve_and_deploy(
        proposal_id=proposal.proposal_id,
        approval_id=val_result["approval_id"],
        approved_by="lead_engineer",
    )
    assert version is not None
    assert version.deployed_by == "lead_engineer"
    assert version.status == "deployed"


def test_end_to_end_self_improvement_langgraph_workflow():
    """Verify full end-to-end execution of self_improvement_workflow."""
    state = {
        "failure_id": "fail_e2e_1",
        "session_id": "sess_e2e",
        "user_id": "u_test",
        "user_input": "List free models on OpenCode",
        "observed_output": "I do not have info on Adaptoid-OS",
        "expected_behavior": "List DeepSeek V4.1 Flash Free",
        "tool_calls": [{"tool": "web_scraper.search", "arguments": {"query": "OpenCode free models"}}],
        "tool_outputs": [{"tool": "web_scraper.search", "output": "OpenCode models: DeepSeek V4.1 Flash Free, Nemotron"}],
        "model_name": "groq/qwen/qwen3.8-27b",
        "failure_record": None,
        "regression_test": None,
        "proposal": None,
        "validation_result": None,
        "approval_id": None,
        "status": "started",
        "error": None,
    }

    final_state = self_improvement_workflow.invoke(state)

    assert final_state["status"] == "awaiting_approval"
    assert final_state["failure_record"] is not None
    assert final_state["failure_record"]["category"] == FailureCategory.CONTEXT_GROUNDING_FAILURE.value
    assert final_state["regression_test"] is not None
    assert final_state["proposal"] is not None
    assert final_state["validation_result"] is not None
    assert final_state["approval_id"] is not None


def test_reliability_metrics_engine():
    """Verify failure intelligence calculations including recurring cluster and hallucination rates."""
    # 1. Grounding failure
    failure_db.save_failure(FailureRecord(
        failure_id="m_fail_1",
        user_input="OpenCode",
        observed_output="Adaptoid-OS",
        category=FailureCategory.CONTEXT_GROUNDING_FAILURE,
    ))
    # 2. System failure
    failure_db.save_failure(FailureRecord(
        failure_id="m_fail_2",
        user_input="PDF parse",
        observed_output="Null byte",
        category=FailureCategory.SYSTEM_FAILURE,
    ))

    # Add regression test
    failure_db.save_regression_test(RegressionTest(
        test_id="REG-M-1",
        failure_id="m_fail_1",
        user_input="OpenCode",
        expected_properties=["OpenCode"],
    ))

    metrics = metrics_engine.compute_metrics()

    assert metrics["total_failures"] == 2
    assert metrics["failures_by_category"]["context_grounding"] == 1
    assert metrics["failures_by_category"]["system_failure"] == 1
    assert metrics["grounding_failure_rate_pct"] == 50.0
    assert metrics["benchmark_summary"]["total_tests"] == 1


def test_rest_api_endpoints():
    """Verify all REST API endpoints under /api/learning/*."""
    client = TestClient(app)

    # 1. Report failure
    report_resp = client.post(
        "/api/learning/failures",
        json={
            "session_id": "sess_api",
            "user_id": "user_api",
            "user_input": "What are the free models on OpenCode?",
            "observed_output": "I don't have information on Adaptoid-OS",
            "expected_behavior": "List DeepSeek Free",
            "tool_calls": [{"tool": "web_scraper.search", "arguments": {"query": "OpenCode"}}],
            "tool_outputs": [{"tool": "web_scraper.search", "output": "DeepSeek V4.1 Free"}],
            "model_name": "groq/openai/gpt-oss-20b",
        },
    )
    assert report_resp.status_code == 200
    fail_data = report_resp.json()
    assert fail_data["status"] == "recorded"
    failure_id = fail_data["failure"]["failure_id"]

    # 2. List failures
    list_resp = client.get("/api/learning/failures")
    assert list_resp.status_code == 200
    assert list_resp.json()["count"] >= 1

    # 3. Analyze failure (run workflow)
    analyze_resp = client.post(
        "/api/learning/analyze",
        json={
            "failure_id": failure_id,
            "user_input": "What are the free models on OpenCode?",
            "observed_output": "I don't have information on Adaptoid-OS",
            "expected_behavior": "List DeepSeek Free",
            "tool_calls": [{"tool": "web_scraper.search", "arguments": {"query": "OpenCode"}}],
            "tool_outputs": [{"tool": "web_scraper.search", "output": "DeepSeek V4.1 Free"}],
            "model_name": "groq/openai/gpt-oss-20b",
            "run_workflow": True,
        },
    )
    assert analyze_resp.status_code == 200
    res = analyze_resp.json()["result"]
    assert res["status"] == "awaiting_approval"
    proposal_id = res["proposal"]["proposal_id"]
    approval_id = res["approval_id"]

    # 4. List regression tests
    tests_resp = client.get("/api/learning/regression-tests")
    assert tests_resp.status_code == 200
    assert tests_resp.json()["count"] >= 1

    # 5. Run benchmark
    run_resp = client.post("/api/learning/regression-tests/run", json={})
    assert run_resp.status_code == 200
    assert "pass_rate" in run_resp.json()

    # 6. List proposals
    prop_resp = client.get("/api/learning/proposals")
    assert prop_resp.status_code == 200
    assert prop_resp.json()["count"] >= 1

    # 7. Approve proposal
    approve_resp = client.post(
        f"/api/learning/proposals/{proposal_id}/approve",
        json={"approval_id": approval_id, "approved_by": "qa_lead"},
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "deployed"

    # 8. Get metrics
    metrics_resp = client.get("/api/learning/metrics")
    assert metrics_resp.status_code == 200
    assert "total_failures" in metrics_resp.json()
