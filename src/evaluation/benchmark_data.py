"""
Benchmark Dataset Generator and Loader for Hermes Agent Platform.
Provides 120 curated benchmark test cases:
- 25 Research tasks (RES-001 - RES-025)
- 25 Coding tasks (COD-001 - COD-025)
- 20 Data Analysis tasks (DAT-001 - DAT-020)
- 20 Document Generation tasks (DOC-001 - DOC-020)
- 20 System Diagnostics tasks (DIA-001 - DIA-020)
- 10 Complex Multi-Step tasks (AUT-001 - AUT-010)
"""
from __future__ import annotations

import json
from pathlib import Path
from src.evaluation.schemas import BenchmarkTestCase

DATASET_FILE = Path(__file__).resolve().parent / "benchmark_dataset.json"


def _generate_default_dataset() -> list[BenchmarkTestCase]:
    cases: list[BenchmarkTestCase] = []

    # 1. 25 Research Tasks (RES-001 to RES-025)
    research_topics = [
        ("quantum error correction surface codes", "physics", ["qubits", "thresholds"]),
        ("transformer linear attention state-space models", "ai", ["mamba", "complexity"]),
        ("crispr cas13 rna editing off-target effects", "biotech", ["specificity", "delivery"]),
        ("zero knowledge rollup validity proofs ethereum", "crypto", ["starks", "snarks"]),
        ("solid-state battery silicon anode degradation", "materials", ["capacity", "electrolyte"]),
        ("mrna vaccine thermostability lipid nanoparticles", "medicine", ["storage", "formulation"]),
        ("optical computing photonic matrix multiplication", "hardware", ["energy", "latency"]),
        ("neuromorphic spike-timing dependent plasticity", "neuroscience", ["synaptic", "memristors"]),
        ("direct air capture metal-organic frameworks", "climate", ["sorption", "desorption"]),
        ("post-quantum cryptography lattice-based kyber", "security", ["nist", "keysize"]),
        ("fusion energy high-temperature superconducting magnets", "energy", ["tokamak", "confinement"]),
        ("synthetic biology metabolic engineering yeast terpenes", "bioengineering", ["pathway", "yield"]),
        ("federated learning differential privacy gradient leakage", "privacy", ["epsilon", "clipping"]),
        ("perovskite tandem solar cell durability halide migration", "renewables", ["efficiency", "degradation"]),
        ("graph neural networks over-smoothing long-range", "ml", ["laplacian", "rewiring"]),
        ("autonomous driving end-to-end vision-language-action", "robotics", ["imitation", "teleoperation"]),
        ("organoid intelligence biological neural networks computing", "bio-computing", ["electrophysiology"]),
        ("carbon nanotube field-effect transistors sub-nanometer", "semiconductors", ["ballistic", "gate"]),
        ("synthetic aperture radar satellite earth observation", "aerospace", ["interferometry", "polarimetry"]),
        ("microbiome gut-brain axis neuroinflammation", "microbiology", ["vagus", "short-chain"]),
        ("homomorphic encryption secure multi-party computation", "cryptography", ["bootstrapping", "cipher"]),
        ("deep sea mineral mining environmental hydrothermal", "oceanography", ["benthic", "plume"]),
        ("nanomaterial thermoelectric power generation waste heat", "thermodynamics", ["seebek", "figure-of-merit"]),
        ("liquid biopsy circulating tumor dna early detection", "oncology", ["sensitivity", "methylation"]),
        ("high-entropy alloys radiation damage resistance nuclear", "metallurgy", ["dislocation", "swelling"]),
    ]
    for idx, (topic, domain, keys) in enumerate(research_topics, start=1):
        t_id = f"RES-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=f"Research the latest developments in {topic}. Retrieve empirical sources and verify claim grounding.",
                expected_workflow="research",
                required_capabilities=["web_search", "evidence_extraction", "grounding_verification", "contradiction_detection"],
                success_criteria=[
                    "Relevant sources retrieved",
                    "Claims supported by empirical evidence",
                    "Sources cited correctly with provenance",
                    "Contradictions between sources explicitly flagged",
                ],
                metadata={"domain": domain, "keywords": keys},
            )
        )

    # 2. 25 Coding Tasks (COD-001 to COD-025)
    coding_tasks = [
        ("Write a Python script to compute average dataset values and handle division by zero", "python", "arithmetic"),
        ("Implement an LRU cache with O(1) get and put in Python", "python", "data_structures"),
        ("Write a script to validate and parse SemVer semantic version strings", "python", "parsers"),
        ("Implement a thread-safe connection pool with lease timeouts", "python", "concurrency"),
        ("Write a rate limiter using the sliding window log algorithm", "python", "networking"),
        ("Create a function to parse CSV stream without loading entire file in memory", "python", "io"),
        ("Implement a red-black tree rotation and balance checking routine", "python", "algorithms"),
        ("Write a JSON schema validator for nested user profiles", "python", "validation"),
        ("Implement an async worker queue with exponential backoff retries", "python", "async"),
        ("Write a Python script to compress data using Run-Length Encoding", "python", "compression"),
        ("Implement a topological sort on a directed acyclic task graph", "python", "graphs"),
        ("Write a script to compute Levenshtein distance between strings with matrix memoization", "python", "strings"),
        ("Implement a secure PBKDF2 password hasher with salt generation", "python", "security"),
        ("Write a memory-mapped file reader for huge binary datasets", "python", "systems"),
        ("Implement a priority queue using a min-heap structure", "python", "heaps"),
        ("Write a Python decorator for memoization with TTL expiration", "python", "meta"),
        ("Implement a Trie prefix tree with autocomplete prefix matching", "python", "trees"),
        ("Write a script to evaluate mathematical expressions using Dijkstra's Shunting-yard", "python", "parsers"),
        ("Implement a circular buffer with overwriting on overflow", "python", "buffers"),
        ("Write a Python script to compute SHA-256 Merkle root of a list of transactions", "python", "crypto"),
        ("Implement a basic finite state machine with transition guard conditions", "python", "fsm"),
        ("Write a script to detect cycles in an undirected graph using Disjoint Set Union", "python", "graphs"),
        ("Implement a streaming moving average with window size k", "python", "stream"),
        ("Write an interval tree for finding overlapping calendar events", "python", "intervals"),
        ("Implement a consistent hashing ring with virtual nodes", "python", "distributed"),
    ]
    for idx, (prompt, lang, category) in enumerate(coding_tasks, start=1):
        t_id = f"COD-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=prompt,
                expected_workflow="coding",
                required_capabilities=["sandbox_execution", "test_generation", "automated_debugging", "unit_testing"],
                success_criteria=[
                    "Code executes cleanly inside sandbox",
                    "All generated unit tests pass",
                    "Exceptions diagnosed and patched if errors occur",
                    "Final implementation verified against edge cases",
                ],
                metadata={"language": lang, "category": category},
            )
        )

    # 3. 20 Data Analysis Tasks (DAT-001 to DAT-020)
    data_tasks = [
        ("Analyze this CSV and tell me whether treatment and control groups are statistically different", "ab_testing"),
        ("Inspect customer churn dataset, check missing values, and test correlation with tenure", "churn"),
        ("Evaluate sales conversion rates across 3 marketing channels and test variance significance", "marketing"),
        ("Analyze latency metrics between v1 and v2 microservices and detect performance anomalies", "infra"),
        ("Assess employee satisfaction survey results and test for response bias", "survey"),
        ("Check clinical trial patient blood pressure dataset before and after medication", "clinical"),
        ("Analyze ecommerce cart abandonment rates by device type and test hypothesis", "ecommerce"),
        ("Examine server CPU utilization distribution and detect peak hour anomalies", "telemetry"),
        ("Test whether educational test scores differ significantly by classroom teaching method", "education"),
        ("Analyze app user session duration dataset and check for skewness and power law fit", "product"),
        ("Inspect sensor vibration time series and detect outlier frequency components", "iot"),
        ("Evaluate financial loan default rates across risk categories with contingency table", "finance"),
        ("Analyze customer support ticket resolution times and test whether tier-2 takes longer", "support"),
        ("Examine warehouse inventory turnover rates and identify slow-moving stock clusters", "logistics"),
        ("Test whether website dark mode increases reading time statistically significantly", "ux"),
        ("Analyze energy consumption data across summer months and test variance homogeneity", "energy"),
        ("Inspect ad click-through rates across demographics and flag low statistical power samples", "ads"),
        ("Analyze network packet loss percentages under high vs low load conditions", "networking"),
        ("Evaluate patient recovery days across two surgical techniques with effect sizes", "healthcare"),
        ("Analyze manufacturing defect counts per batch and test Poisson distribution fit", "manufacturing"),
    ]
    for idx, (prompt, domain) in enumerate(data_tasks, start=1):
        t_id = f"DAT-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=prompt,
                expected_workflow="data_analysis",
                required_capabilities=["schema_inspection", "trap_detection", "hypothesis_testing", "visualization"],
                success_criteria=[
                    "Schema and missing values inspected",
                    "Statistical traps detected (e.g. correlation vs causation, small sample size)",
                    "Appropriate statistical method selected (e.g. Welch's t-test)",
                    "Visual summary and effect sizes reported",
                ],
                metadata={"domain": domain},
            )
        )

    # 4. 20 Document Generation Tasks (DOC-001 to DOC-020)
    doc_tasks = [
        ("Create a detailed technical report on Project Orion architecture and performance benchmarks", "tech_report"),
        ("Generate an engineering design document for the new event-driven message bus", "design_doc"),
        ("Produce a system architecture review document with SLA and latency guarantees", "architecture"),
        ("Draft an incident postmortem report for the recent Redis cluster failover outage", "postmortem"),
        ("Create a security compliance audit report detailing RBAC and credential rotation policies", "security_audit"),
        ("Generate an API specification document for external partner integration endpoints", "api_spec"),
        ("Write an infrastructure scaling guide for high-throughput container deployments", "infra_guide"),
        ("Draft a technical whitepaper on zero-trust microservice communication architecture", "whitepaper"),
        ("Create a disaster recovery specification with RTO and RPO benchmarks", "dr_spec"),
        ("Produce a performance tuning guide for PostgreSQL database indexing and query plans", "db_guide"),
        ("Draft an engineering RFC proposing adoption of distributed tracing across services", "rfc"),
        ("Generate a release notes document summarizing major features, bug fixes, and breaking changes", "release_notes"),
        ("Create an operational runbook for diagnosing and mitigating high memory leaks", "runbook"),
        ("Produce a data governance and privacy policy document for multi-tenant storage", "governance"),
        ("Draft a microservices migration plan document with phase gates and rollback criteria", "migration"),
        ("Generate an evaluation report comparing LLM inference latency and cost profiles", "eval_report"),
        ("Write a formal technical proposal for implementing vector database indexing", "proposal"),
        ("Draft a service level agreement (SLA) specification for enterprise tier clients", "sla_spec"),
        ("Create an integration guide for developers embedding our webhook subscriptions", "integration_guide"),
        ("Produce an engineering quality scorecard report covering unit and integration pass rates", "scorecard"),
    ]
    for idx, (prompt, doc_type) in enumerate(doc_tasks, start=1):
        t_id = f"DOC-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=prompt,
                expected_workflow="document",
                required_capabilities=["fact_extraction", "outline_generation", "consistency_checking", "inspection_gate"],
                success_criteria=[
                    "Numerical facts extracted from sources",
                    "Multi-section structured outline created",
                    "Numerical consistency verified across sections",
                    "Dedicated document inspection passed (no broken tables or unclosed code fences)",
                ],
                metadata={"doc_type": doc_type},
            )
        )

    # 5. 20 System Diagnostics Tasks (DIA-001 to DIA-020)
    diag_tasks = [
        ("Why is my MCP server failing? Check logs, isolate the root cause, and propose a fix", "mcp_failure"),
        ("Investigate why vector search queries are timing out under concurrent user load", "vector_timeout"),
        ("Diagnose high memory consumption in the Python worker pool and identify leak sources", "memory_leak"),
        ("Determine why PostgreSQL connections are exhausting connection pool limits", "db_pool"),
        ("Troubleshoot Redis queue latency spikes during peak scheduled automation runs", "redis_latency"),
        ("Investigate why authentication tokens are unexpectedly expiring prematurely", "auth_expiry"),
        ("Diagnose sandbox container execution failures with non-zero exit code 137", "container_oom"),
        ("Identify why grounding verification is intermittently rejecting valid source URLs", "grounding_rejection"),
        ("Troubleshoot rate limiter blocking legitimate user requests during burst traffic", "rate_limiter"),
        ("Investigate why Kafka consumer lag is steadily growing on the audit stream topic", "consumer_lag"),
        ("Diagnose cross-tenant data leak warnings triggered by the security audit logger", "security_leak"),
        ("Troubleshoot SSL certificate handshake termination errors on the ingress gateway", "gateway_ssl"),
        ("Identify why the regression test synthesizer is failing on empty diff inputs", "synthesizer_fail"),
        ("Diagnose circuit breaker remaining in OPEN state after external API recovered", "circuit_breaker"),
        ("Investigate high CPU usage in the full-text search indexing background worker", "cpu_spike"),
        ("Troubleshoot file permission denied errors when saving artifacts to storage", "permission_denied"),
        ("Diagnose why WebSocket streaming disconnects after 60 seconds of client inactivity", "websocket_drop"),
        ("Investigate why Mem0 vector retrieval returns stale facts for updated profiles", "stale_memory"),
        ("Troubleshoot unexpected HTTP 502 Bad Gateway responses from downstream LLM proxy", "llm_proxy_502"),
        ("Diagnose failure in the HITL approval webhook notification dispatch queue", "hitl_webhook"),
    ]
    for idx, (prompt, issue_type) in enumerate(diag_tasks, start=1):
        t_id = f"DIA-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=prompt,
                expected_workflow="diagnostics",
                required_capabilities=["log_collection", "root_cause_taxonomy", "hitl_approval_gate", "recovery_verification"],
                success_criteria=[
                    "System health checks and log traces collected",
                    "Failing component isolated and classified into reliability taxonomy",
                    "Mutating remediations require explicit HITL approval",
                    "Recovery verified post-remediation",
                ],
                metadata={"issue_type": issue_type},
            )
        )

    # 6. 10 Complex Multi-Step Tasks (AUT-001 to AUT-010)
    autonomous_tasks = [
        ("Research three open-source vector databases (Qdrant, Milvus, Chroma), benchmark their performance, and generate a comparative report", "vector_db_comparison"),
        ("Decompose our monolithic auth module, plan migration to OAuth2/OIDC, write tests, and generate transition guide", "auth_migration"),
        ("Evaluate three cloud deployment providers for LLM hosting, calculate cost per million tokens, and compile recommendations", "cloud_evaluation"),
        ("Audit system security across 5 endpoints, run penetration tests, patch vulnerabilities, and compile audit report", "security_remediation"),
        ("Analyze customer feedback from three channels, perform sentiment clustering, and draft product roadmap proposals", "product_roadmap"),
        ("Design an automated ETL pipeline, implement Python extraction workers, validate schema outputs, and export documentation", "etl_pipeline"),
        ("Investigate top 3 LLM prompt injection techniques, generate defensive guardrail unit tests, and publish defense whitepaper", "guardrail_defense"),
        ("Perform architectural review of microservices, identify circular dependencies, propose decoupled event buses, and write RFC", "arch_refactor"),
        ("Benchmark SQLite vs PostgreSQL for session caching under high write loads, analyze statistical differences, and draft guide", "db_benchmark"),
        ("Synthesize complete disaster recovery plan: analyze failure modes, build automated recovery scripts, and create drill runbook", "dr_plan"),
    ]
    for idx, (prompt, goal) in enumerate(autonomous_tasks, start=1):
        t_id = f"AUT-{idx:03d}"
        cases.append(
            BenchmarkTestCase(
                test_id=t_id,
                input=prompt,
                expected_workflow="autonomous",
                required_capabilities=["dag_planning", "cycle_validation", "topological_batching", "composite_artifact_assembly"],
                success_criteria=[
                    "Multi-step objective decomposed into acyclic Task DAG",
                    "Independent subtasks executed concurrently in topological batches",
                    "Subtask outputs verified and aggregated",
                    "Comprehensive final composite artifact assembled",
                ],
                metadata={"goal": goal},
            )
        )

    return cases


def save_benchmark_dataset(cases: list[BenchmarkTestCase], path: Path = DATASET_FILE) -> None:
    """Save benchmark cases to JSON."""
    raw = [c.model_dump() for c in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def load_benchmark_dataset(path: Path = DATASET_FILE) -> list[BenchmarkTestCase]:
    """Load benchmark test cases from JSON file, generating defaults if absent."""
    if not path.exists():
        cases = _generate_default_dataset()
        save_benchmark_dataset(cases, path)
        return cases
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BenchmarkTestCase(**item) for item in data]
    except Exception:
        cases = _generate_default_dataset()
        save_benchmark_dataset(cases, path)
        return cases


# Initialize default dataset file on import if missing
if not DATASET_FILE.exists():
    _default_cases = _generate_default_dataset()
    save_benchmark_dataset(_default_cases, DATASET_FILE)
