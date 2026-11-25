"""
Schemas for Workflow 4: Document Generation.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    title: str
    content: str
    word_count: int
    tables: list[str] = Field(default_factory=list)
    verified: bool = True


class DocumentOutline(BaseModel):
    title: str
    target_audience: str
    sections: list[str] = Field(default_factory=list)


class DocumentInspectionResult(BaseModel):
    is_approved: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    formatting_issues: list[str] = Field(default_factory=list)
    numerical_consistency_issues: list[str] = Field(default_factory=list)
    ungrounded_claims: list[str] = Field(default_factory=list)


class FinalDocumentArtifact(BaseModel):
    title: str
    format: str = "markdown"  # "markdown" | "html" | "pdf_text"
    content: str
    section_count: int
    word_count: int
    inspection_passed: bool
