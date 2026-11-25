"""
Comprehensive Unit Tests for the Hermes Self-Learning Failure Taxonomy,
Confidence-Aware Triage, and Evaluation Release Gate.

Tests:
1. Expanded 7-Category Failure Taxonomy & Responsible Layer Mapping:
   - Retrieval failure -> search routing
   - Grounding failure -> evidence verification
   - Reasoning failure -> reasoning strategy
   - Tool failure -> execution layer
   - Memory failure -> memory retrieval
   - Planning failure -> planning policy
   - Security failure -> authorization policy
2. Confidence-Aware Failure Classification & Evidence Trails
3. Low-Confidence Triage Gating (< 0.70 routes to Human Review)
4. Proposal Generation across all Architectural Boundaries
5. Formal Evaluation Release Gate & Benchmark Comparison:
   - Pass rate before / after
   - Original failure resolution
   - New regressions detection and rejection
   - Latency and token cost deltas
6. Staged Deployment Pipeline: staging -> canary -> production
7. Closed-Loop LangGraph Workflow with Confidence Branching
"""
import pytest
from src.learning.models import (
    BenchmarkComparison,
    CATEGORY_CORRECTIVE_ACTION_MAP,
    CATEGORY_LAYER_MAP,
    CorrectionType,
    FailureCategory,
    FailureRecord,
    FailureStatus,
    ImprovementProposal,
    ImprovementVersion,
    RegressionTest,
)
from src.learning.database import failure_db
from src.learning.analyzer import failure_analyzer
from src.learning.generator import proposal_generator
from src.learning.regression_harness import regression_harness
from src.learning.sandbox_validator import sandbox_validator
from src.learning.workflow import self_improvement_workflow
from src.tools import hitl


@pytest.fixture(autouse=True)
def clean_database():
    """Reset failure and proposal stores for isolated testing."""
    failure_db._mem_failures.clear()
    failure_db._mem_clusters.clear()
    failure_db._mem_proposals.clear()
    failure_db._mem_regression_tests.clear()
    failure_db._mem_versions.clear()
    hitl._memory_store.clear()
    yield


# =========================================================================== #
# Pillar A: Failure Taxonomy & Architectural Layer Mapping
# =========================================================================== #

def test_failure_taxonomy_layer_and_corrective_action_mappings():
    """Verify that all 7 failure categories cleanly map to their responsible

    architectural layer and canonical corrective actions.
    """
    expected_mappings = {
        FailureCategory.RETRIEVAL_FAILURE: ("retrieval_and_search_routing", "Improve search routing"),
        FailureCategory.GROUNDING_FAILURE: ("evidence_verification_and_guardrails", "Improve evidence verification"),
        FailureCategory.REASONING_FAILURE: ("synthesis_and_reasoning", "Improve reasoning strategy"),
        FailureCategory.TOOL_FAILURE: ("tool_execution_engine", "Fix execution layer"),
        FailureCategory.MEMORY_FAILURE: ("memory_and_context_store", "Improve memory retrieval"),
        FailureCategory.PLANNING_FAILURE: ("agent_planning_policy", "Improve planning policy"),
        FailureCategory.SECURITY_FAILURE: ("security_and_authorization_gate", "Fix authorization policy"),
    }

    for cat, (expected_layer, expected_action) in expected_mappings.items():
        assert CATEGORY_LAYER_MAP[cat] == expected_layer
        assert CATEGORY_CORRECTIVE_ACTION_MAP[cat] == expected_action

        # Check FailureRecord auto-initialization
        rec = FailureRecord(
            failure_id=f"f_{cat.value}",
            user_input="test query",
            observed_output="test output",
            category=cat,
        )
        assert rec.responsible_layer == expected_layer
        assert rec.corrective_action == expected_action


def test_retrieval_failure_classification():
    """Verify that queries returning zero verified search results are diagnosed as RETRIEVAL_FAILURE."""
    record = FailureRecord(
        failure_id="fail_retrieval_1",
        user_input="what is the box office collection of Aakasam lo oka thara?",
        observed_output="I searched the web, but no box-office information was found in the available evidence.",
        tool_calls=[{"tool": "web_scraper.search", "arguments": {"query": "Aakasam lo oka thara box office"}}],
        tool_outputs=[{"tool": "web_scraper.search", "output": "Generic results only. No verified box office numbers."}],
    )

    analyzed = failure_analyzer.analyze(record)
    assert analyzed.category == FailureCategory.RETRIEVAL_FAILURE
    assert analyzed.responsible_layer == "retrieval_and_search_routing"
    assert analyzed.corrective_action == "Improve search routing"
    assert analyzed.correction_type == CorrectionType.ROUTING_CHANGE
    assert analyzed.confidence >= 0.70
    assert any("Relevant webpage or document not retrieved" in e for e in analyzed.evidence)


def test_security_failure_classification():
    """Verify unauthorized tool invocations or RBAC violations are diagnosed as SECURITY_FAILURE."""
    record = FailureRecord(
        failure_id="fail_sec_1",
        user_input="Delete production user table",
        observed_output="Action blocked by security gate: Unauthorized tool execution attempted under role 'viewer'.",
        tool_calls=[{"tool": "database.execute_sql", "arguments": {"query": "DROP TABLE users;"}}],
        tool_outputs=[{"error": "Permission denied: unauthorized action blocked by security gate RBAC policy."}],
    )

    analyzed = failure_analyzer.analyze(record)
    assert analyzed.category == FailureCategory.SECURITY_FAILURE
    assert analyzed.responsible_layer == "security_and_authorization_gate"
    assert analyzed.corrective_action == "Fix authorization policy"
    assert analyzed.correction_type == CorrectionType.POLICY_FIX
    assert analyzed.confidence >= 0.90
    assert any("Unauthorized" in e for e in analyzed.evidence)


def test_memory_failure_classification():
    """Verify forgotten user preferences or memory recall errors are diagnosed as MEMORY_FAILURE."""
    record = FailureRecord(
        failure_id="fail_mem_1",
        user_input="Summarize this in French as I told you earlier in my preferred settings",
        observed_output="I do not recall or remember that you requested French, here it is in English.",
        tool_calls=[],
        tool_outputs=[],
    )

    analyzed = failure_analyzer.analyze(record)
    assert analyzed.category == FailureCategory.MEMORY_FAILURE
    assert analyzed.responsible_layer == "memory_and_context_store"
    assert analyzed.corrective_action == "Improve memory retrieval"
    assert analyzed.correction_type == CorrectionType.MEMORY_UPDATE
    assert analyzed.confidence >= 0.75


def test_planning_failure_classification_on_redundant_loops():
    """Verify cyclic redundant tool calls without progress are diagnosed as PLANNING_FAILURE."""
    repeated_calls = [
        {"tool": "web_scraper.search", "arguments": {"q": "Hermes architecture"}},
        {"tool": "web_scraper.search", "arguments": {"q": "Hermes architecture"}},
        {"tool": "web_scraper.search", "arguments": {"q": "Hermes architecture"}},
    ]
    record = FailureRecord(
        failure_id="fail_plan_1",
        user_input="Explain Hermes platform architecture",
        observed_output="Execution timed out after repeated loop steps.",
        tool_calls=repeated_calls,
        tool_outputs=[{"output": "ok"}, {"output": "ok"}, {"output": "ok"}],
    )

    analyzed = failure_analyzer.analyze(record)
    assert analyzed.category == FailureCategory.PLANNING_FAILURE
    assert analyzed.responsible_layer == "agent_planning_policy"
    assert analyzed.corrective_action == "Improve planning policy"
    assert analyzed.correction_type == CorrectionType.PLANNING_POLICY
    assert analyzed.confidence >= 0.75


# =========================================================================== #
# Pillar C: Confidence-Aware Failure Classification & Human Review Gating
# =========================================================================== #

def test_confidence_aware_classification_structure():
    """Verify that failure classification outputs confidence, evidence, and alternative causes."""
    record = FailureRecord(
        failure_id="fail_grounding_conf",
        user_input="List free models on OpenCode",
        observed_output="I don't have any info on Adaptoid-OS repository.",
        tool_calls=[{"tool": "search", "arguments": {"q": "OpenCode free models"}}],
        tool_outputs=[{"output": "OpenCode free models include DeepSeek V4.1 and Nemotron 3.5 Lightning"}],
    )

    analyzed = failure_analyzer.analyze(record)
    assert 0.0 <= analyzed.confidence <= 1.0
    assert len(analyzed.evidence) > 0
    assert len(analyzed.alternative_causes) > 0
    assert analyzed.requires_human_triage is False

    d = analyzed.to_dict()
    assert "confidence" in d
    assert "evidence" in d
    assert "alternative_causes" in d
    assert "requires_human_triage" in d


def test_low_confidence_ambiguous_failure_flags_human_triage():
    """Verify that ambiguous failure symptoms with low confidence (< 0.70) trigger requires_human_triage."""
    record = FailureRecord(
        failure_id="fail_ambiguous_1",
        user_input="Hello there",
        observed_output="I did something unexpected without tool telemetry.",
        tool_calls=[],
        tool_outputs=[],
    )

    analyzed = failure_analyzer.analyze(record)
    assert analyzed.confidence < 0.70
    assert analyzed.requires_human_triage is True
    assert analyzed.status == FailureStatus.REQUIRES_HUMAN_TRIAGE
    assert "Ambiguous" in analyzed.root_cause


# =========================================================================== #
# Pillar B: Evaluation & Release Gate with Benchmark Comparison
# =========================================================================== #

def test_benchmark_comparison_pass_and_release_gate_success():
    """Verify that a candidate fix resolving the original failure and introducing zero regressions

    passes the Release Gate and requests HITL approval.
    """
    # 1. Existing benchmark test that passes
    t1 = RegressionTest(
        test_id="REG-GATE-1",
        failure_id="f_prev",
        user_input="DeepSeek specs",
        expected_properties=["DeepSeek"],
        negative_assertions=["Error"],
        last_run_status="PASS",
    )
    failure_db.save_regression_test(t1)

    # 2. Target failure test
    t2 = RegressionTest(
        test_id="REG-GATE-2",
        failure_id="f_target",
        user_input="OpenCode models",
        expected_properties=["Nemotron"],
        negative_assertions=["Adaptoid-OS"],
        last_run_status="FAIL",
    )
    failure_db.save_regression_test(t2)

    proposal = ImprovementProposal(
        proposal_id="PROP-GATE-PASS",
        failure_id="f_target",
        correction_type=CorrectionType.RESPONSE_VALIDATION,
        target_file="src/security/output_guards.py",
        proposed_diff="+ validate_entity_grounding",
        summary="Validate grounding to prevent Adaptoid hallucinations",
        rationale="Eliminates hallucinated entity references",
    )
    failure_db.save_proposal(proposal)

    # Validate proposal
    val_res = sandbox_validator.validate_proposal(proposal, test=t2)

    assert val_res["status"] == "passed"
    assert val_res["passed_gate"] is True
    assert val_res["gate_decision"] == "requires_human_approval"
    assert val_res["original_failure_resolved"] is True
    assert val_res["new_failures_introduced"] == 0
    assert val_res["requires_hitl_approval"] is True
    assert val_res["approval_id"] is not None

    # Verify BenchmarkComparison metrics
    comp = val_res["benchmark_comparison"]
    assert "pass_rate_before" in comp
    assert "pass_rate_after" in comp
    assert comp["new_failures_introduced"] == 0
    assert comp["original_failure_resolved"] is True
    assert comp["latency_delta_ms"] >= 0.0


def test_release_gate_rejects_proposal_introducing_regressions():
    """Verify that a candidate proposal introducing regressions to existing working tests

    is strictly REJECTED by the Evaluation Release Gate.
    """
    # Test 1 previously PASSED (baseline)
    t1 = RegressionTest(
        test_id="REG-REGRESS-1",
        failure_id="f_baseline",
        user_input="What is Hermes platform?",
        expected_properties=["Hermes"],
        negative_assertions=["BrokenSystem"],
        last_run_status="PASS",
    )
    failure_db.save_regression_test(t1)

    # Test 2 is the target failure being repaired
    t2 = RegressionTest(
        test_id="REG-REGRESS-2",
        failure_id="f_target_bug",
        user_input="How to query agents?",
        expected_properties=["agents"],
        negative_assertions=["BrokenSystem"],
        last_run_status="FAIL",
    )
    failure_db.save_regression_test(t2)

    # Proposal that claims to fix t2 but breaks t1
    proposal = ImprovementProposal(
        proposal_id="PROP-REGRESS-FAIL",
        failure_id="f_target_bug",
        correction_type=CorrectionType.CODE_FIX,
        target_file="src/core.py",
        proposed_diff="+ break_everything()",
        summary="Buggy patch",
        rationale="Introduces a regression in baseline suite",
    )
    failure_db.save_proposal(proposal)

    # Candidate runner that causes existing t1 to fail
    def buggy_runner(prompt, evidence):
        if "Hermes" in prompt:
            return "BrokenSystem failure message"  # breaks previously passing test t1!
        return "Clean output satisfying agents query"

    val_res = sandbox_validator.validate_proposal(proposal, test=t2, candidate_runner=buggy_runner)

    assert val_res["status"] == "failed"
    assert val_res["passed_gate"] is False
    assert val_res["gate_decision"] == "rejected"
    assert val_res["new_failures_introduced"] > 0
    assert any("regressions" in r.lower() and "introduced" in r.lower() for r in val_res["rejection_reasons"])
    assert val_res["requires_hitl_approval"] is False


def test_staged_deployment_pipeline():
    """Verify staged deployment progression: staging -> canary -> production."""
    proposal = ImprovementProposal(
        proposal_id="PROP-STAGE-1",
        failure_id="f_stage",
        correction_type=CorrectionType.ROUTING_CHANGE,
        target_file="src/agent/search_router.py",
        proposed_diff="+ search_rules",
        summary="Search router improvement",
        rationale="Fixes search routing",
    )
    failure_db.save_proposal(proposal)

    # 1. Sandbox validation passes and issues HITL approval
    val_res = sandbox_validator.validate_proposal(proposal)
    approval_id = val_res["approval_id"]

    # 2. Deploy to staging
    deploy_staging = sandbox_validator.deploy_approved_proposal(
        proposal_id=proposal.proposal_id,
        approval_id=approval_id,
        decided_by="qa_engineer",
        target_stage="staging",
    )
    assert deploy_staging["success"] is True
    assert deploy_staging["stage"] == "staging"
    version_id = deploy_staging["version_id"]

    # 3. Promote to canary
    promo_canary = sandbox_validator.promote_stage(version_id, target_stage="canary", promoted_by="release_lead")
    assert promo_canary["success"] is True
    assert promo_canary["current_stage"] == "canary"

    # 4. Promote to production
    promo_prod = sandbox_validator.promote_stage(version_id, target_stage="production", promoted_by="eng_director")
    assert promo_prod["success"] is True
    assert promo_prod["current_stage"] == "production"

    # Verify version audit log
    versions = failure_db.list_versions()
    v = [ver for ver in versions if ver.version_id == version_id][0]
    assert v.deployment_stage == "production"
    assert v.can_rollback is True


# =========================================================================== #
# End-to-End Workflow with Confidence Triage Branching
# =========================================================================== #

def test_workflow_low_confidence_routes_to_human_triage():
    """Verify that the self-improving LangGraph workflow diverts low-confidence failures

    to human triage rather than autonomously generating unverified fixes.
    """
    initial_state = {
        "failure_id": None,
        "session_id": "sess_ambig",
        "user_id": "user_low_conf",
        "user_input": "ambiguous request with no clear signal",
        "observed_output": "unclear failure with no tool trace",
        "expected_behavior": "correct response",
        "tool_calls": [],
        "tool_outputs": [],
        "model_name": "default",
        "failure_record": None,
        "regression_test": None,
        "proposal": None,
        "validation_result": None,
        "approval_id": None,
        "status": "new",
        "error": None,
    }

    final_state = self_improvement_workflow.invoke(initial_state)

    assert final_state["status"] == "awaiting_human_triage"
    assert final_state["approval_id"] is not None
    # No automated proposal should have been generated
    assert final_state.get("proposal") is None


def test_workflow_high_confidence_proceeds_to_sandbox_validation():
    """Verify that high-confidence failures cleanly flow through regression test generation,

    proposal generation, sandbox benchmark comparison, and HITL release gating.
    """
    initial_state = {
        "failure_id": None,
        "session_id": "sess_high_conf",
        "user_id": "user_high",
        "user_input": "What are the free models on OpenCode?",
        "observed_output": "I don't have information about Adaptoid-OS.",
        "expected_behavior": "List free models like DeepSeek V4.1 Flash",
        "tool_calls": [{"tool": "web_scraper.search", "arguments": {"query": "OpenCode free models"}}],
        "tool_outputs": [{"tool": "web_scraper.search", "output": "OpenCode free models include DeepSeek V4.1"}],
        "model_name": "default",
        "failure_record": None,
        "regression_test": None,
        "proposal": None,
        "validation_result": None,
        "approval_id": None,
        "status": "new",
        "error": None,
    }

    final_state = self_improvement_workflow.invoke(initial_state)

    assert final_state["status"] == "awaiting_approval"
    assert final_state["failure_record"] is not None
    assert final_state["failure_record"]["category"] == FailureCategory.GROUNDING_FAILURE.value
    assert final_state["regression_test"] is not None
    assert final_state["proposal"] is not None
    assert final_state["validation_result"]["passed_gate"] is True
    assert final_state["approval_id"] is not None
