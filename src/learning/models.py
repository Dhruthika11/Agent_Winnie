"""
Data models for the Hermes Self-Improving Failure-Correction Subsystem.
Defines failure taxonomy, improvement proposals, regression benchmarks, versioning,
and release gate benchmark comparisons.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Categorization of agent failures to determine the correct architectural layer for repair.
    
    Expanded 7-category taxonomy with measurable outcomes plus backwards-compatible aliases.
    """
    # Canonical 7 Taxonomy Categories
    RETRIEVAL_FAILURE = "retrieval_failure"              # Relevant webpage/doc not retrieved -> Search routing
    GROUNDING_FAILURE = "grounding_failure"              # Unsupported claim/hallucination in response -> Evidence verification
    REASONING_FAILURE = "reasoning_failure"              # Incorrect logical conclusion or self-contradiction -> Reasoning strategy
    TOOL_FAILURE = "tool_failure"                        # Tool timeout, crash, or malformed output -> Execution layer
    MEMORY_FAILURE = "memory_failure"                    # Incorrect user preference or history recalled -> Memory retrieval
    PLANNING_FAILURE = "planning_failure"                # Agent chooses unnecessary steps or loops -> Planning policy
    SECURITY_FAILURE = "security_failure"                # Unauthorized tool invocation or policy violation -> Authorization policy

    # Backwards-compatible aliases (same values as canonical or resolved via _missing_)
    KNOWLEDGE_FAILURE = "knowledge_failure"              # Procedural/domain gap -> Skill update
    SYSTEM_FAILURE = "tool_failure"                      # Legacy alias for tool/system error
    CONTEXT_GROUNDING_FAILURE = "grounding_failure"      # Legacy alias for grounding failure
    TOOL_ORCHESTRATION_FAILURE = "retrieval_failure"    # Legacy alias for search/retrieval routing error

    @classmethod
    def _missing_(cls, value: object) -> Any:
        """Resolve legacy enum strings for backwards compatibility."""
        val_str = str(value).lower()
        if val_str in ("context_grounding", "context_grounding_failure"):
            return cls.GROUNDING_FAILURE
        if val_str in ("system_failure", "system"):
            return cls.TOOL_FAILURE
        if val_str in ("tool_orchestration", "tool_orchestration_failure"):
            return cls.RETRIEVAL_FAILURE
        return super()._missing_(value)


# Mapping each FailureCategory to its responsible architectural layer
CATEGORY_LAYER_MAP: dict[FailureCategory | str, str] = {
    FailureCategory.RETRIEVAL_FAILURE: "retrieval_and_search_routing",
    FailureCategory.GROUNDING_FAILURE: "evidence_verification_and_guardrails",
    FailureCategory.REASONING_FAILURE: "synthesis_and_reasoning",
    FailureCategory.TOOL_FAILURE: "tool_execution_engine",
    FailureCategory.MEMORY_FAILURE: "memory_and_context_store",
    FailureCategory.PLANNING_FAILURE: "agent_planning_policy",
    FailureCategory.SECURITY_FAILURE: "security_and_authorization_gate",
    # Aliases
    FailureCategory.KNOWLEDGE_FAILURE: "procedural_knowledge_base",
    FailureCategory.SYSTEM_FAILURE: "tool_execution_engine",
    FailureCategory.CONTEXT_GROUNDING_FAILURE: "evidence_verification_and_guardrails",
    FailureCategory.TOOL_ORCHESTRATION_FAILURE: "retrieval_and_search_routing",
    "retrieval_failure": "retrieval_and_search_routing",
    "grounding_failure": "evidence_verification_and_guardrails",
    "reasoning_failure": "synthesis_and_reasoning",
    "tool_failure": "tool_execution_engine",
    "memory_failure": "memory_and_context_store",
    "planning_failure": "agent_planning_policy",
    "security_failure": "security_and_authorization_gate",
    "knowledge_failure": "procedural_knowledge_base",
    "system_failure": "tool_execution_engine",
    "context_grounding": "evidence_verification_and_guardrails",
    "tool_orchestration": "retrieval_and_search_routing",
}

# Mapping each FailureCategory to its canonical corrective action
CATEGORY_CORRECTIVE_ACTION_MAP: dict[FailureCategory | str, str] = {
    FailureCategory.RETRIEVAL_FAILURE: "Improve search routing",
    FailureCategory.GROUNDING_FAILURE: "Improve evidence verification",
    FailureCategory.REASONING_FAILURE: "Improve reasoning strategy",
    FailureCategory.TOOL_FAILURE: "Fix execution layer",
    FailureCategory.MEMORY_FAILURE: "Improve memory retrieval",
    FailureCategory.PLANNING_FAILURE: "Improve planning policy",
    FailureCategory.SECURITY_FAILURE: "Fix authorization policy",
    # Aliases
    FailureCategory.KNOWLEDGE_FAILURE: "Update procedural skill",
    FailureCategory.SYSTEM_FAILURE: "Fix execution layer",
    FailureCategory.CONTEXT_GROUNDING_FAILURE: "Improve evidence verification",
    FailureCategory.TOOL_ORCHESTRATION_FAILURE: "Improve search routing",
    "retrieval_failure": "Improve search routing",
    "grounding_failure": "Improve evidence verification",
    "reasoning_failure": "Improve reasoning strategy",
    "tool_failure": "Fix execution layer",
    "memory_failure": "Improve memory retrieval",
    "planning_failure": "Improve planning policy",
    "security_failure": "Fix authorization policy",
    "knowledge_failure": "Update procedural skill",
    "system_failure": "Fix execution layer",
    "context_grounding": "Improve evidence verification",
    "tool_orchestration": "Improve search routing",
}


class CorrectionType(str, Enum):
    """The appropriate remediation mechanism for a failure."""
    SKILL_UPDATE = "skill_update"                    # Procedural knowledge bundle
    PROMPT_CHANGE = "prompt_change"                  # System or synthesis prompt improvement
    CODE_FIX = "code_fix"                            # Source code patch
    ROUTING_CHANGE = "routing_change"                # Model/domain router adjustment
    RESPONSE_VALIDATION = "response_validation"      # Output guardrail or grounding filter
    EVALUATION_CASE = "evaluation_case"              # Regression test addition only
    POLICY_FIX = "policy_fix"                        # Security or authorization policy patch
    MEMORY_UPDATE = "memory_update"                  # Memory indexing or retrieval update
    PLANNING_POLICY = "planning_policy"              # Agent planner rule update


class FailureStatus(str, Enum):
    """Lifecycle status of a recorded failure."""
    NEW = "new"
    ANALYZED = "analyzed"
    REQUIRES_HUMAN_TRIAGE = "requires_human_triage"
    PROPOSAL_GENERATED = "proposal_generated"
    TESTING = "testing"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class FailureRecord(BaseModel):
    """Structured record of a single detected or reported agent failure."""
    failure_id: str
    session_id: str = "unknown"
    user_id: str = "unknown"
    user_input: str
    observed_output: str
    expected_behavior: str = ""
    category: FailureCategory = FailureCategory.REASONING_FAILURE
    correction_type: CorrectionType = CorrectionType.RESPONSE_VALIDATION
    root_cause: str = ""
    originating_stage: str = "synthesis"  # "input_routing", "retrieval", "tool_execution", "context_assembly", "planning", "synthesis", "output_guard"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str = "default"
    is_recurring: bool = False
    cluster_id: Optional[str] = None
    status: FailureStatus = FailureStatus.NEW
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    # Architecture layer and confidence-aware triage fields
    responsible_layer: str = ""
    corrective_action: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    evidence: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    requires_human_triage: bool = False

    def __init__(self, **data: Any):
        if "recommended_correction" in data and "correction_type" not in data:
            data["correction_type"] = data["recommended_correction"]
        if data.get("category") is None:
            data["category"] = FailureCategory.REASONING_FAILURE
        if data.get("root_cause") is None:
            data["root_cause"] = ""

        # Auto-populate responsible_layer and corrective_action if not specified
        cat = data.get("category")
        if not data.get("responsible_layer"):
            data["responsible_layer"] = CATEGORY_LAYER_MAP.get(cat, "synthesis_and_reasoning")
        if not data.get("corrective_action"):
            data["corrective_action"] = CATEGORY_CORRECTIVE_ACTION_MAP.get(cat, "Improve reasoning strategy")

        # Low confidence default triage flag
        conf = data.get("confidence", 1.0)
        if conf is not None and conf < 0.70 and "requires_human_triage" not in data:
            data["requires_human_triage"] = True

        super().__init__(**data)

    @property
    def recommended_correction(self) -> CorrectionType:
        return self.correction_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_input": self.user_input,
            "observed_output": self.observed_output,
            "expected_behavior": self.expected_behavior,
            "category": self.category.value if isinstance(self.category, FailureCategory) else str(self.category),
            "correction_type": self.correction_type.value if isinstance(self.correction_type, CorrectionType) else str(self.correction_type),
            "root_cause": self.root_cause,
            "originating_stage": self.originating_stage,
            "tool_calls": self.tool_calls,
            "tool_outputs": self.tool_outputs,
            "model_name": self.model_name,
            "is_recurring": self.is_recurring,
            "cluster_id": self.cluster_id,
            "status": self.status.value if isinstance(self.status, FailureStatus) else str(self.status),
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
            "responsible_layer": self.responsible_layer,
            "corrective_action": self.corrective_action,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "alternative_causes": self.alternative_causes,
            "requires_human_triage": self.requires_human_triage,
        }


class FailureCluster(BaseModel):
    """Cluster grouping recurring failures sharing the same root cause signature."""
    cluster_id: str
    category: FailureCategory
    signature: str
    failure_count: int = 1
    representative_failure_id: str
    description: str = ""
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "category": self.category.value if isinstance(self.category, FailureCategory) else str(self.category),
            "signature": self.signature,
            "failure_count": self.failure_count,
            "representative_failure_id": self.representative_failure_id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RegressionTest(BaseModel):
    """Automated benchmark regression test derived from a confirmed failure."""
    test_id: str
    failure_id: str = ""
    name: str = ""
    input_prompt: str = ""
    available_evidence: list[dict[str, Any]] = Field(default_factory=list)
    expected_properties: list[str] = Field(default_factory=list)  # Concepts/facts that MUST be present
    negative_assertions: list[str] = Field(default_factory=list)  # Hallucinated entities/terms that MUST NOT appear
    last_run_status: str = "UNTESTED"  # "PASS", "FAIL", "UNTESTED"
    last_run_error: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def __init__(self, **data: Any):
        if "user_input" in data and "input_prompt" not in data:
            data["input_prompt"] = data["user_input"]
        if "name" not in data or not data["name"]:
            data["name"] = f"Regression Test for {data.get('test_id', 'unknown')}"
        super().__init__(**data)

    @property
    def user_input(self) -> str:
        return self.input_prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "failure_id": self.failure_id,
            "name": self.name,
            "input_prompt": self.input_prompt,
            "available_evidence": self.available_evidence,
            "expected_properties": self.expected_properties,
            "negative_assertions": self.negative_assertions,
            "last_run_status": self.last_run_status,
            "last_run_error": self.last_run_error,
            "created_at": self.created_at.isoformat(),
        }


class BenchmarkComparison(BaseModel):
    """Rigorous evaluation comparing candidate proposal against previous stable version.
    
    Tracks regression pass rates, original resolution, new regression count, latency, and cost impact.
    """
    pass_rate_before: float
    pass_rate_after: float
    original_failure_resolved: bool
    new_failures_introduced: int = 0
    latency_delta_ms: float = 0.0
    token_cost_delta: float = 0.0
    gate_decision: str = "pending"  # "approved", "rejected", "requires_human_approval"
    rejection_reasons: list[str] = Field(default_factory=list)

    @property
    def passed_gate(self) -> bool:
        """Formal Release Gate logic: Must resolve original failure, introduce zero regressions,

        and maintain or improve pass rate.
        """
        return (
            self.original_failure_resolved
            and self.new_failures_introduced == 0
            and self.pass_rate_after >= self.pass_rate_before
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_rate_before": self.pass_rate_before,
            "pass_rate_after": self.pass_rate_after,
            "original_failure_resolved": self.original_failure_resolved,
            "new_failures_introduced": self.new_failures_introduced,
            "latency_delta_ms": self.latency_delta_ms,
            "token_cost_delta": self.token_cost_delta,
            "gate_decision": self.gate_decision,
            "passed_gate": self.passed_gate,
            "rejection_reasons": self.rejection_reasons,
        }


class ImprovementProposal(BaseModel):
    """Formal remediation proposal (code patch, prompt revision, or skill spec)."""
    proposal_id: str
    failure_id: str
    cluster_id: Optional[str] = None
    correction_type: CorrectionType
    summary: str
    proposed_diff: str
    target_file: Optional[str] = None
    rationale: str
    validation_status: str = "pending"  # "pending", "passed", "failed"
    approval_status: str = "pending"    # "pending", "approved", "rejected"
    approval_id: Optional[str] = None
    deployment_stage: str = "staging"   # "staging", "canary", "production"
    benchmark_comparison: Optional[BenchmarkComparison] = None
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "failure_id": self.failure_id,
            "cluster_id": self.cluster_id,
            "correction_type": self.correction_type.value if isinstance(self.correction_type, CorrectionType) else str(self.correction_type),
            "summary": self.summary,
            "proposed_diff": self.proposed_diff,
            "target_file": self.target_file,
            "rationale": self.rationale,
            "validation_status": self.validation_status,
            "approval_status": self.approval_status,
            "approval_id": self.approval_id,
            "deployment_stage": self.deployment_stage,
            "benchmark_comparison": self.benchmark_comparison.to_dict() if self.benchmark_comparison else None,
            "created_at": self.created_at.isoformat(),
        }


class ImprovementVersion(BaseModel):
    """Auditable log of deployed self-improvement patches."""
    version_id: str
    proposal_id: str
    deployed_by: str = "system"
    diff_applied: str
    benchmark_pass_rate_before: float
    benchmark_pass_rate_after: float
    deployment_stage: str = "staging"  # "staging", "canary", "production"
    original_failure_resolved: bool = True
    new_failures_introduced: int = 0
    latency_delta_ms: float = 0.0
    token_cost_delta: float = 0.0
    benchmark_comparison: Optional[dict[str, Any]] = None
    deployed_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    can_rollback: bool = True
    status: str = "deployed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "proposal_id": self.proposal_id,
            "deployed_by": self.deployed_by,
            "diff_applied": self.diff_applied,
            "benchmark_pass_rate_before": self.benchmark_pass_rate_before,
            "benchmark_pass_rate_after": self.benchmark_pass_rate_after,
            "deployment_stage": self.deployment_stage,
            "original_failure_resolved": self.original_failure_resolved,
            "new_failures_introduced": self.new_failures_introduced,
            "latency_delta_ms": self.latency_delta_ms,
            "token_cost_delta": self.token_cost_delta,
            "benchmark_comparison": self.benchmark_comparison,
            "deployed_at": self.deployed_at.isoformat(),
            "can_rollback": self.can_rollback,
            "status": self.status,
        }
