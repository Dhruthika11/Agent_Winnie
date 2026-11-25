"""
Hermes Self-Improving Failure-Correction Subsystem.

A continuous self-improvement and reliability loop that:
1. Detects agent failures across the 7-category taxonomy with confidence scoring.
2. Identifies root causes and maps them to responsible architectural layers.
3. Routes low-confidence failures to human review.
4. Synthesizes regression evaluation benchmarks for every confirmed failure.
5. Generates targeted improvement proposals adhering to strict architectural boundaries.
6. Enforces an Evaluation Release Gate comparing candidate fixes against stable benchmarks.
7. Deploys approved improvements safely through staged deployment (staging -> canary -> production).
"""

from src.learning.models import (
    FailureCategory,
    CATEGORY_LAYER_MAP,
    CATEGORY_CORRECTIVE_ACTION_MAP,
    CorrectionType,
    FailureStatus,
    FailureRecord,
    FailureCluster,
    RegressionTest,
    BenchmarkComparison,
    ImprovementProposal,
    ImprovementVersion,
)
from src.learning.database import FailureDatabase, failure_db
from src.learning.analyzer import FailureAnalyzer, failure_analyzer
from src.learning.regression_harness import RegressionHarness, regression_harness
from src.learning.generator import ProposalGenerator, proposal_generator
from src.learning.sandbox_validator import SandboxValidator, sandbox_validator
from src.learning.workflow import LearningState, self_improvement_workflow
from src.learning.metrics import ReliabilityMetricsEngine, metrics_engine
from src.learning.lifecycle import (
    PostDeploymentReliabilityReport,
    SelfImprovementLifecycleEngine,
    SelfImprovementLifecycleResult,
    self_improvement_lifecycle,
)

__all__ = [
    "FailureCategory",
    "CATEGORY_LAYER_MAP",
    "CATEGORY_CORRECTIVE_ACTION_MAP",
    "CorrectionType",
    "FailureStatus",
    "FailureRecord",
    "FailureCluster",
    "RegressionTest",
    "BenchmarkComparison",
    "ImprovementProposal",
    "ImprovementVersion",
    "FailureDatabase",
    "failure_db",
    "FailureAnalyzer",
    "failure_analyzer",
    "RegressionHarness",
    "regression_harness",
    "ProposalGenerator",
    "proposal_generator",
    "SandboxValidator",
    "sandbox_validator",
    "LearningState",
    "self_improvement_workflow",
    "ReliabilityMetricsEngine",
    "metrics_engine",
]
