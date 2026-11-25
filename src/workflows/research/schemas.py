"""
Schemas for Workflow 1: Research & Evidence.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str
    question: str
    search_query: str
    rationale: str
    status: str = "pending"  # "pending" | "searched" | "answered"


class SourceEvidence(BaseModel):
    evidence_id: str
    source_url: str
    title: str
    extracted_text: str
    content_hash: str
    relevance_score: float = 1.0


class SourceContradiction(BaseModel):
    topic: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    disagreement_analysis: str


class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    contradictions_and_disagreements: list[SourceContradiction] = Field(default_factory=list)
    citations: list[dict[str, str]] = Field(default_factory=list)
    confidence_score: float = 1.0
    grounding_status: str = "verified"
