"""
Terminal and Shell Execution Output Validator.
Validates process exit codes, error streams, and command output integrity.
"""
from __future__ import annotations

import re
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)


class TerminalValidator(BaseToolValidator):
    tool_type = EvidenceType.TERMINAL

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        raw_text = str(raw_output)
        command = str(arguments.get("command", "") or arguments.get("path", "")).strip()
        issues: list[str] = []
        metrics: dict[str, Any] = {"command": command}

        lower_text = raw_text.lower()

        # 1. Non-zero exit code check
        exit_match = re.search(r"exited\s+non-zero\s*\((\d+)\)", raw_text, re.IGNORECASE)
        if exit_match:
            code = int(exit_match.group(1))
            metrics["exit_code"] = code
            return ValidationResult(
                status=ValidationStatus.FAILED,
                is_valid=False,
                confidence=0.0,
                issues=[f"Shell command failed with exit code {code}"],
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # 2. Command not found or permission denied
        error_indicators = (
            "command not found",
            "no such file or directory",
            "permission denied",
            "access denied",
            "is not recognized as an internal or external command",
        )
        for ind in error_indicators:
            if ind in lower_text:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=[f"Shell execution error: {ind}"],
                    metrics=metrics,
                    sanitized_content=raw_text,
                )

        # 3. File deletion check
        if "delete_file" in tool_name:
            if "failed" in lower_text or "not found" in lower_text:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=["File deletion failed"],
                    metrics=metrics,
                    sanitized_content=raw_text,
                )
            return ValidationResult(
                status=ValidationStatus.VERIFIED,
                is_valid=True,
                confidence=1.0,
                issues=[],
                metrics=metrics,
                sanitized_content=raw_text,
            )

        return ValidationResult(
            status=ValidationStatus.VERIFIED,
            is_valid=True,
            confidence=0.95,
            issues=[],
            metrics=metrics,
            sanitized_content=raw_text,
        )

