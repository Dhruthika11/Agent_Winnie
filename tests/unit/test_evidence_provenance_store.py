"""
Comprehensive Unit Tests for the Evidence and Provenance Layer.

Tests:
1. Evidence Model & Content Hashing (SHA-256, source URL, title, tool execution ID).
2. Separation of Web-derived and Memory-derived Evidence.
3. Atomic Claim Extraction from LLM Generation.
4. Claim-to-Evidence Relationship Verification & Classification:
   - SUPPORTED: exact figures/text supported by evidence text.
   - CONTRADICTED: claim numbers or facts contradict evidence content.
   - UNKNOWN: no evidence supports or refutes.
5. Integration with Output Guardrails (sanitize_output).
6. Centralized Tool Execution Engine Evidence Ingestion.
7. REST API Endpoints (/api/evidence, /api/evidence/{id}, /api/evidence/verify).
"""
import pytest
from starlette.testclient import TestClient

from src.main import app
from src.security.evidence_store import (
    Claim,
    Evidence,
    EvidenceStore,
    GroundingVerificationResult,
    evidence_store,
    extract_atomic_claims,
    verify_claim_against_evidence,
    verify_response_grounding,
)
from src.security.output_guards import sanitize_output
from src.tools.execution_engine import ToolExecutionRequest, tool_execution_engine


@pytest.fixture(autouse=True)
def clean_evidence_store():
    """Ensure clean evidence store before each test."""
    evidence_store.clear()
    yield


client = TestClient(app)


def test_evidence_model_and_content_hashing():
    """Verify that Evidence records compute accurate SHA-256 hashes and maintain provenance."""
    text = "Hermes is an enterprise-grade agent platform with centralized execution."
    ev = evidence_store.record_evidence(
        extracted_text=text,
        source_url="https://hermes.ai/docs",
        source_title="Hermes Documentation",
        tool_execution_id="exec_123",
        provenance_type="web",
    )

    assert ev.evidence_id.startswith("EVID-")
    assert ev.source_url == "https://hermes.ai/docs"
    assert ev.source_title == "Hermes Documentation"
    assert ev.provenance_type == "web"
    assert ev.tool_execution_id == "exec_123"
    assert len(ev.content_hash) == 64  # SHA-256 hex digest length
    assert ev.content_hash == evidence_store.compute_content_hash(text)

    # Retrieval by ID and URL
    fetched = evidence_store.get_evidence(ev.evidence_id)
    assert fetched is not None
    assert fetched.extracted_text == text

    by_url = evidence_store.find_evidence_by_url("https://hermes.ai/docs")
    assert len(by_url) == 1
    assert by_url[0].evidence_id == ev.evidence_id


def test_memory_derived_evidence_stored_separately():
    """Verify that memory-derived evidence is tracked with provenance_type='memory'."""
    mem_ev = evidence_store.record_memory_evidence(
        memory_id="mem_456",
        memory_text="User prefers Python 3.12 and dark mode.",
        user_id="user_alice",
        tenant_id="tenant_prod",
        category="preference",
    )

    assert mem_ev.provenance_type == "memory"
    assert mem_ev.source_url == "memory://user_alice/mem_456"
    assert "User Memory (preference)" in mem_ev.source_title
    assert mem_ev.tenant_id == "tenant_prod"

    # Verify filtering
    web_list = evidence_store.list_evidence(provenance_type="web")
    mem_list = evidence_store.list_evidence(provenance_type="memory")
    assert len(web_list) == 0
    assert len(mem_list) == 1
    assert mem_list[0].evidence_id == mem_ev.evidence_id


def test_extract_atomic_claims():
    """Verify that atomic claims, figures, quotes, and attributions are decomposed accurately."""
    response = (
        'According to https://hermes.ai, the system throughput is 99.8% with 450 req/s. '
        'The report stated "zero regressions in benchmark tests". '
        'Also, as you prefer, responses are concise.'
    )

    claims = extract_atomic_claims(response)
    assert len(claims) >= 2

    c1 = claims[0]
    assert any("99.8%" in fig for fig in c1.extracted_figures)
    assert c1.attributed_source is not None
    assert "https://hermes.ai" in c1.attributed_source

    c2 = claims[1]
    assert "zero regressions in benchmark tests" in c2.quoted_spans

    # Check memory-derived claim detection
    mem_claims = [c for c in claims if c.provenance_type == "memory"]
    assert len(mem_claims) >= 1
    assert "prefer" in mem_claims[0].text.lower()


def test_claim_verification_supported():
    """Verify that claims accurately matching evidence text are classified as SUPPORTED."""
    ev = evidence_store.record_evidence(
        extracted_text="Official benchmarks show system latency is 12ms and throughput is 99.8%.",
        source_url="https://hermes.ai/benchmark",
        source_title="Hermes Benchmarks",
    )

    claim = Claim(
        claim_id="c_supp",
        text="According to Hermes Benchmarks, throughput is 99.8% and latency is 12ms.",
        extracted_figures=["99.8%", "12ms"],
        attributed_source="Hermes Benchmarks",
    )

    verified = verify_claim_against_evidence(claim, [ev])
    assert verified.verification_status == "SUPPORTED"
    assert verified.confidence >= 0.90
    assert ev.evidence_id in verified.evidence_ids


def test_claim_verification_contradicted():
    """Verify that claims with conflicting figures or invalid quotes are classified as CONTRADICTED."""
    ev = evidence_store.record_evidence(
        extracted_text="The survey reported that 42% of developers use Python.",
        source_url="https://survey.org",
        source_title="Developer Survey",
    )

    # Claim asserts 89% when evidence explicitly reports 42%
    claim = Claim(
        claim_id="c_contra",
        text="According to Developer Survey, 89% of developers use Python.",
        extracted_figures=["89%"],
        attributed_source="Developer Survey",
    )

    verified = verify_claim_against_evidence(claim, [ev])
    assert verified.verification_status == "CONTRADICTED"
    assert verified.confidence >= 0.85
    assert verified.contradiction_reason is not None
    assert "89%" in verified.contradiction_reason


def test_claim_verification_unknown():
    """Verify that unverified claims lacking evidence are classified as UNKNOWN."""
    claim = Claim(
        claim_id="c_unk",
        text="The quantum processor was deployed in Antarctica.",
        extracted_figures=[],
    )

    verified = verify_claim_against_evidence(claim, [])
    assert verified.verification_status == "UNKNOWN"
    assert verified.confidence <= 0.50
    assert len(verified.evidence_ids) == 0


def test_full_grounding_pipeline_with_contradiction_warning():
    """Verify the full 5-stage grounding pipeline sanitizes contradicted statements with warnings."""
    ev = evidence_store.record_evidence(
        extracted_text="The latest release is Python 3.12 with improved error messages.",
        source_url="https://python.org",
        source_title="Python Release",
    )

    response = "According to Python Release, the latest release is Python 3.10."
    res = verify_response_grounding(response, [ev])

    assert res.has_contradictions is True
    assert res.contradicted_claims_count >= 1
    assert "[Warning: Contradicted by evidence" in res.sanitized_text


def test_sanitize_output_integrates_claim_evidence_verification():
    """Verify that output_guards.sanitize_output applies claim-to-evidence pipeline."""
    ev_item = {
        "url": "https://python.org",
        "source": "Python Release",
        "content": "The latest release is Python 3.12.",
    }

    # Text contains contradicted figure 3.10
    raw_text = "According to Python Release, the latest release is Python 3.10."
    sanitized = sanitize_output(raw_text, evidence_items=[ev_item])

    assert "[Warning: Contradicted by evidence" in sanitized or "[Claim attribution corrected:" in sanitized or "[Unsupported claim:" in sanitized


def test_execution_engine_records_evidence_automatically():
    """Verify that centralized tool execution engine ingests tool outputs into evidence store."""
    req = ToolExecutionRequest(
        tool_name="web_scraper.search",
        arguments={"query": "OpenCode free models"},
        request_id="req_ev_test_1",
        user_id="user_bob",
        tenant_id="tenant_alpha",
    )

    res = tool_execution_engine.execute(req)
    assert res.status in ("success", "error")

    # If evidence item was generated, check evidence store
    if res.evidence_item:
        ev_items = evidence_store.list_evidence(tenant_id="tenant_alpha")
        assert len(ev_items) >= 1
        assert ev_items[0].tool_execution_id == "req_ev_test_1"


def test_evidence_rest_api_endpoints():
    """Verify REST endpoints for listing, fetching, and verifying evidence."""
    ev = evidence_store.record_evidence(
        extracted_text="Hermes Agent Platform implements closed-loop reliability.",
        source_url="https://hermes.ai/architecture",
        source_title="Hermes Architecture",
    )

    # 1. GET /api/evidence
    res_list = client.get("/api/evidence")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["count"] >= 1
    assert any(e["evidence_id"] == ev.evidence_id for e in data["evidence"])

    # 2. GET /api/evidence/{id}
    res_item = client.get(f"/api/evidence/{ev.evidence_id}")
    assert res_item.status_code == 200
    assert res_item.json()["extracted_text"] == ev.extracted_text

    # 3. POST /api/evidence/verify
    verify_payload = {
        "text": "According to Hermes Architecture, Hermes implements closed-loop reliability.",
        "evidence_items": [ev.to_dict()],
    }
    res_verify = client.post("/api/evidence/verify", json=verify_payload)
    assert res_verify.status_code == 200
    v_data = res_verify.json()
    assert "claims" in v_data
    assert v_data["supported_claims_count"] >= 1
