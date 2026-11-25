"""
Filesystem and Document Reader Output Validator.
Validates file reads, writes, and directory listings for readability and errors.
"""
from __future__ import annotations

import json
from typing import Any

from src.tools.validators.base import (
    BaseToolValidator,
    EvidenceType,
    ValidationResult,
    ValidationStatus,
)


class FilesystemValidator(BaseToolValidator):
    tool_type = EvidenceType.FILESYSTEM

    def validate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_output: Any,
    ) -> ValidationResult:
        raw_text = str(raw_output)
        path = str(arguments.get("path", "")).strip()
        issues: list[str] = []
        metrics: dict[str, Any] = {"path": path}

        lower_text = raw_text.lower()

        # 1. Error checks
        error_indicators = (
            "no such file or directory",
            "file not found",
            "permission denied",
            "is a directory",
            "not a directory",
            "error: ",
            "access denied",
        )
        for ind in error_indicators:
            if ind in lower_text:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    is_valid=False,
                    confidence=0.0,
                    issues=[f"Filesystem error: {ind}"],
                    metrics=metrics,
                    sanitized_content=raw_text,
                )

        # 2. Directory listing check
        if "list_directory" in tool_name:
            entries = []
            if isinstance(raw_output, list):
                entries = raw_output
            else:
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, list):
                        entries = parsed
                except Exception:
                    pass
            metrics["entry_count"] = len(entries)
            return ValidationResult(
                status=ValidationStatus.VERIFIED,
                is_valid=True,
                confidence=1.0,
                issues=[],
                metrics=metrics,
                sanitized_content=raw_text,
            )

        # 3. Read file text length check
        if "read_file" in tool_name:
            metrics["char_count"] = len(raw_text)
            if not raw_text.strip():
                issues.append(f"File at '{path}' is empty (0 bytes)")
                return ValidationResult(
                    status=ValidationStatus.FLAGGED,
                    is_valid=True,
                    confidence=0.7,
                    issues=issues,
                    metrics=metrics,
                    sanitized_content="[Empty file]",
                )

        # 4. Write file check
        if "write_file" in tool_name:
            if "success" in lower_text or "wrote" in lower_text or "bytes" in lower_text:
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
            issues=issues,
            metrics=metrics,
            sanitized_content=raw_text,
        )

