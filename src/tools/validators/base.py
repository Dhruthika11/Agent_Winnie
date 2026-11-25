"""
Base classes and schemas for tool output validators and normalized evidence.
"""
from __future__ import annotations

import datetime
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict


class ValidationStatus(str, enum.Enum):
    VERIFIED = "verified"      # Output is valid, grounded, and meets quality standards
    FLAGGED = "flagged"        # Output has warnings (partial match, stale, potential discrepancy)
    FAILED = "failed"          # Output represents execution error, empty result, or invalid schema
    UNVERIFIED = "unverified"  # Default when validation is inconclusive


class EvidenceType(str, enum.Enum):
    WEB_SEARCH = "web_search"
    WEB_READER = "web_reader"
    CODE_EXECUTION = "code_execution"
    FILESYSTEM = "filesystem"
    TERMINAL = "terminal"
    MATH = "math"
    GITHUB = "github"
    API = "api"
    GENERIC = "generic"


class EvidenceItem(TypedDict):
    """Normalized evidence record stored in central evidence store."""
    id: str
    tool_name: str
    type: str
    source: str
    content: str
    timestamp: str
    validation_status: str
    validation_details: dict[str, Any]
    raw_output: Any


@dataclass
class ValidationResult:
    """Outcome of validating a single tool execution."""
    status: ValidationStatus
    is_valid: bool
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    sanitized_content: str = ""

    def to_details_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "confidence": round(self.confidence, 3),
            "issues": self.issues,
            "metrics": self.metrics,
        }


class BaseToolValidator:
    """Abstract base validator for individual tools."""
    tool_type: EvidenceType = EvidenceType.GENERIC

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        """Validate tool execution output. Must be implemented by concrete validators."""
        raise NotImplementedError

    def create_evidence_item(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
        validation: ValidationResult,
        source: Optional[str] = None,
    ) -> EvidenceItem:
        """Convert tool execution and validation result into a normalized EvidenceItem."""
        ev_id = f"ev_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Determine source identifier
        resolved_source = source
        if not resolved_source:
            resolved_source = (
                arguments.get("url")
                or arguments.get("path")
                or arguments.get("query")
                or arguments.get("command")
                or arguments.get("owner", "") + "/" + arguments.get("repo", "")
                or tool_name
            )

        content = validation.sanitized_content or str(raw_output)

        return {
            "id": ev_id,
            "tool_name": tool_name,
            "type": self.tool_type.value,
            "source": str(resolved_source),
            "content": content,
            "timestamp": now_iso,
            "validation_status": validation.status.value,
            "validation_details": validation.to_details_dict(),
            "raw_output": raw_output,
        }

