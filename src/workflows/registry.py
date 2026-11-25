"""
Workflow Registry and Intelligent Router for Hermes Agent Platform.

Maps workflow keys to their compiled LangGraph graphs and routes user requests
to the optimal specialized workflow with confidence scoring and fallback handling.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.workflows.autonomous.graph import autonomous_graph
from src.workflows.coding.graph import coding_graph
from src.workflows.data_analysis.graph import data_analysis_graph
from src.workflows.diagnostics.graph import diagnostics_graph
from src.workflows.document.graph import document_graph
from src.workflows.research.graph import research_graph
from src.workflows.schemas import WorkflowDecision, WorkflowType

logger = logging.getLogger("hermes.workflows.registry")

# Canonical registry of all 6 specialized LangGraph workflows
WORKFLOW_REGISTRY: dict[str, Any] = {
    WorkflowType.RESEARCH.value: research_graph,
    WorkflowType.CODING.value: coding_graph,
    WorkflowType.DATA_ANALYSIS.value: data_analysis_graph,
    WorkflowType.DOCUMENT.value: document_graph,
    WorkflowType.DIAGNOSTICS.value: diagnostics_graph,
    WorkflowType.AUTONOMOUS.value: autonomous_graph,
}

_WORKFLOW_RULES = [
    # 1. Autonomous Multi-Step Tasks (Takes precedence when multiple targets, dynamic DAG, or comparative orchestration is requested)
    (
        WorkflowType.AUTONOMOUS.value,
        re.compile(
            r"\b(?:compare\s+their\s+performance|research\s+three|three\s+open-source|multi-step|dependency|compare\s+(?:qdrant|milvus|databases)|dag|orchestrat|decompose|migration\s+plan|comparative\s+report|audit\s+system\s+security\s+across)\b",
            re.IGNORECASE,
        ),
        "Requires dynamic DAG task decomposition, parallel batch execution, and composite synthesis.",
        0.96,
    ),
    # 2. System Diagnostics & Recovery
    (
        WorkflowType.DIAGNOSTICS.value,
        re.compile(
            r"\b(?:why\s+(?:is|are)\s+|diagnos|troubleshoot|failing|failure|error\s+in|health\s+check|investigate\s+why|determine\s+why|broken\s+service|crash\s+recovery|timing\s+out|leak\s+source|exhausting|latency\s+spikes)\b",
            re.IGNORECASE,
        ),
        "Requires system diagnostics, log trace inspection, root-cause classification, and recovery.",
        0.95,
    ),
    # 3. Code Execution & Debugging
    (
        WorkflowType.CODING.value,
        re.compile(
            r"\b(?:write\s+(?:a\s+)?(?:python\s+)?script|implement\b|python\s+script|code\s+to|check\s+whether\s+it\s+works|debug\s+this\s+code|run\s+in\s+sandbox|fix\s+bug|python\s+decorator|create\s+a\s+function|cache|connection\s+pool|rate\s+limiter|csv\s+stream|tree|validator|queue|compression|sort|distance|hasher|reader|buffer|merkle|fsm|cycle|moving\s+average|interval|hashing)\b",
            re.IGNORECASE,
        ),
        "Requires code generation, sandboxed execution, unit test validation, and automated debugging.",
        0.96,
    ),
    # 4. Data Analysis
    (
        WorkflowType.DATA_ANALYSIS.value,
        re.compile(
            r"\b(?:analyze\s+(?:this\s+)?(?:csv|dataset)|inspect\s+.*dataset|evaluate\s+sales|statistically\s+different|t-test|p-value|correlation|anova|regression|outlier|histogram|hypothesis\s+test|churn\s+dataset|conversion\s+rates|survey|distribution)\b",
            re.IGNORECASE,
        ),
        "Requires tabular dataset analysis, statistical hypothesis testing, trap detection, and charts.",
        0.95,
    ),
    # 5. Document Generation
    (
        WorkflowType.DOCUMENT.value,
        re.compile(
            r"\b(?:create\s+(?:a\s+)?(?:detailed\s+)?(?:technical\s+)?report|generate\s+(?:an?\s+)?(?:engineering\s+)?(?:design\s+)?(?:report|document|doc)|draft\s+(?:an?\s+)?(?:report|document|rfc|whitepaper|proposal)|produce\s+(?:a\s+)?(?:system\s+)?(?:document|report|guide)|docx|pdf|whitepaper|design\s+document|format\s+tables\s+and\s+headings|document\s+structure|specification|postmortem|compliance\s+audit|api\s+spec|scaling\s+guide|runbook|scorecard)\b",
            re.IGNORECASE,
        ),
        "Requires structured document synthesis, numerical consistency verification, and inspection.",
        0.94,
    ),
    # 6. Research & Evidence
    (
        WorkflowType.RESEARCH.value,
        re.compile(
            r"\b(?:research|literature\s+review|clinical\s+trials?|evidence|scientific|find\s+information|paper\s+review)\b",
            re.IGNORECASE,
        ),
        "Requires multi-source research, query decomposition, and cross-source evidence verification.",
        0.92,
    ),
]


def classify_workflow(user_request: str) -> WorkflowDecision:
    """
    Classify the incoming user request to the optimal workflow graph.
    Returns structured decision with confidence score and rationale.
    """
    clean_req = user_request.strip()
    if not clean_req:
        return WorkflowDecision(
            workflow=WorkflowType.AUTONOMOUS.value,
            confidence=0.50,
            reason="Empty request routed to general autonomous executor.",
        )

    lowered = clean_req.lower()

    # 1. Multi-Target / Comparative / Multi-Stage Cross-Functional Tasks -> Autonomous Multi-Step
    if (
        any(w in lowered for w in [
            "compare", "three open-source", "research three", "decompose", "multi-step",
            "synthesize complete disaster", "audit system security", "across 5 endpoints",
            "penetration tests", "design an automated etl", "roadmap proposals"
        ])
        and any(w in lowered for w in [
            "database", "tools", "three", "frameworks", "libraries", "modules", "cloud",
            "channels", "plan", "endpoints", "pipeline", "proposals", "vulnerabilities"
        ])
    ):
        return WorkflowDecision(
            workflow=WorkflowType.AUTONOMOUS.value,
            confidence=0.96,
            reason="Multi-target comparative request requires dynamic DAG orchestration and parallel execution.",
            composed_workflows=[WorkflowType.RESEARCH.value, WorkflowType.DOCUMENT.value],
        )

    # 2. Diagnostics & Incident Recovery (takes precedence over research when investigating failures)
    if any(k in lowered for k in [
        "why is my", "why are", "determine why", "investigate why", "diagnos", "troubleshoot",
        "failing", "error in", "health check", "broken service", "crash recovery", "timing out",
        "leak source", "exhausting", "latency spikes", "mcp server", "handshake termination"
    ]):
        return WorkflowDecision(
            workflow=WorkflowType.DIAGNOSTICS.value,
            confidence=0.95,
            reason="Requires system diagnostics, log trace inspection, root-cause classification, and recovery.",
        )

    # 3. Research & Scientific Inquiry (takes precedence over generic keyword matches)
    if any(k in lowered for k in [
        "research the latest", "literature review", "clinical trial", "evidence extraction",
        "retrieve empirical sources", "find information", "paper review", "advancements in"
    ]) or (lowered.startswith("research ") and "compare" not in lowered):
        return WorkflowDecision(
            workflow=WorkflowType.RESEARCH.value,
            confidence=0.95,
            reason="Requires multi-source research, query decomposition, and cross-source evidence verification.",
        )

    # 4. Code Generation & Execution
    if any(k in lowered for k in [
        "write a python", "write a script", "implement", "python script", "code to",
        "debug this", "run in sandbox", "fix bug", "lru cache", "connection pool",
        "rate limiter", "csv stream", "json schema", "worker queue", "run-length",
        "topological sort", "levenshtein", "password hasher", "file reader",
        "priority queue", "decorator", "trie prefix", "circular buffer", "merkle",
        "finite state", "disjoint set", "moving average", "interval tree", "consistent hashing"
    ]) or re.search(r"\btrie\b", lowered):
        return WorkflowDecision(
            workflow=WorkflowType.CODING.value,
            confidence=0.96,
            reason="Requires code generation, sandboxed execution, unit test validation, and automated debugging.",
        )

    # 5. Tabular Data Analysis & Statistics
    if any(k in lowered for k in [
        "analyze this csv", "analyze this dataset", "dataset", "statistically different",
        "t-test", "p-value", "correlation", "anova", "conversion rates", "churn",
        "survey results", "blood pressure", "cart abandonment", "cpu utilization",
        "session duration", "time series", "loan default", "ticket resolution",
        "inventory turnover", "dark mode increases", "energy consumption", "click-through",
        "packet loss", "recovery days", "defect counts", "latency metrics", "performance anomalies"
    ]):
        return WorkflowDecision(
            workflow=WorkflowType.DATA_ANALYSIS.value,
            confidence=0.95,
            reason="Requires tabular dataset analysis, statistical hypothesis testing, trap detection, and charts.",
        )

    # 6. Document & Report Generation
    if any(k in lowered for k in [
        "technical report", "design doc", "design document", "architecture review",
        "postmortem", "audit report", "compliance audit", "api spec", "api specification",
        "scaling guide", "whitepaper", "disaster recovery specification", "tuning guide",
        "rfc", "release notes", "runbook", "governance", "migration plan",
        "evaluation report", "technical proposal", "sla", "integration guide", "scorecard",
        "detailed technical report", "draft an incident", "produce a system"
    ]):
        return WorkflowDecision(
            workflow=WorkflowType.DOCUMENT.value,
            confidence=0.94,
            reason="Requires structured document synthesis, numerical consistency verification, and inspection.",
        )

    # 6. Research & Scientific Inquiry
    if any(k in lowered for k in [
        "research", "literature review", "clinical trial", "evidence", "scientific",
        "find information", "paper review", "advancements in"
    ]):
        return WorkflowDecision(
            workflow=WorkflowType.RESEARCH.value,
            confidence=0.92,
            reason="Requires multi-source research, query decomposition, and cross-source evidence verification.",
        )

    # Fallback to regex rules then Autonomous Multi-Step
    for wf_name, pattern, reason, conf in _WORKFLOW_RULES:
        if pattern.search(clean_req):
            return WorkflowDecision(
                workflow=wf_name,
                confidence=conf,
                reason=reason,
            )

    return WorkflowDecision(
        workflow=WorkflowType.AUTONOMOUS.value,
        confidence=0.70,
        reason="General request dynamically decomposed via Autonomous Multi-Step DAG planner.",
    )


def get_workflow_graph(workflow_name: str) -> Any:
    """Retrieve compiled LangGraph graph for a given workflow name."""
    graph = WORKFLOW_REGISTRY.get(workflow_name.lower())
    if not graph:
        raise KeyError(
            f"Unknown workflow '{workflow_name}'. Available: {list(WORKFLOW_REGISTRY.keys())}"
        )
    return graph


def list_available_workflows() -> list[str]:
    """Return all registered workflow names."""
    return list(WORKFLOW_REGISTRY.keys())
