"""
Security subsystem for Hermes Agent Platform.
Provides input/output guardrails, secret redaction, RBAC, untrusted content sanitization,
the Evidence and Provenance Grounding Layer, and the User-Facing Grounding Presentation Layer.
"""
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
from src.security.grounding_presentation import (
    GroundingPresentationFormatter,
    InternalClaimVerification,
    VerificationCategory,
    format_clickable_source_link,
    grounding_presentation_formatter,
)
from src.security.output_guards import (
    StreamingOutputBuffer,
    filter_exfiltration_channels,
    redact_pii,
    redact_secrets,
    sanitize_html_and_scripts,
    sanitize_output,
    sanitize_user_facing_output,
    validate_evidence_grounding,
    verify_claims_against_evidence,
    verify_input_output_grounding,
)
from src.security.finops import (
    FinOpsManager,
    FinOpsMetricRecord,
    MODEL_PRICING_RATES,
    calculate_cost_usd,
    finops,
)
from src.security.rbac import (
    Permission,
    RBACEngine,
    ROLE_PERMISSIONS,
    Role,
    rbac_engine,
    rbac_manager,
)

__all__ = [
    "Evidence",
    "Claim",
    "EvidenceStore",
    "evidence_store",
    "GroundingVerificationResult",
    "extract_atomic_claims",
    "verify_claim_against_evidence",
    "verify_response_grounding",
    "VerificationCategory",
    "InternalClaimVerification",
    "GroundingPresentationFormatter",
    "grounding_presentation_formatter",
    "format_clickable_source_link",
    "redact_secrets",
    "redact_pii",
    "filter_exfiltration_channels",
    "validate_evidence_grounding",
    "verify_claims_against_evidence",
    "sanitize_html_and_scripts",
    "sanitize_output",
    "sanitize_user_facing_output",
    "verify_input_output_grounding",
    "StreamingOutputBuffer",
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "RBACEngine",
    "rbac_engine",
    "rbac_manager",
    "FinOpsManager",
    "finops",
    "FinOpsMetricRecord",
    "calculate_cost_usd",
    "MODEL_PRICING_RATES",
]
