"""
Central Evidence Store for Winnie Agent.
Maintains normalized, validated evidence items gathered throughout agent turns.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from src.tools.validators.base import EvidenceItem, ValidationStatus

logger = logging.getLogger("hermes.agent.evidence_store")


class EvidenceStore:
    """Central normalized evidence store for a conversation turn or session."""

    def __init__(self, initial_items: Iterable[EvidenceItem] | None = None):
        self._items: list[EvidenceItem] = []
        if initial_items:
            for it in initial_items:
                self.add(it)

    def add(self, item: EvidenceItem) -> None:
        """Add a validated evidence item to the store."""
        if not item or not isinstance(item, dict):
            return
        # Avoid exact duplicate content and source
        for existing in self._items:
            if existing.get("id") == item.get("id"):
                return
            if (
                existing.get("source") == item.get("source")
                and existing.get("content") == item.get("content")
                and existing.get("tool_name") == item.get("tool_name")
            ):
                return
        self._items.append(item)

    def get_all(self) -> list[EvidenceItem]:
        """Return all gathered evidence items."""
        return list(self._items)

    def get_verified(self) -> list[EvidenceItem]:
        """Return only verified evidence items."""
        return [
            item for item in self._items
            if item.get("validation_status") == ValidationStatus.VERIFIED.value
        ]

    def get_flagged(self) -> list[EvidenceItem]:
        """Return flagged evidence items requiring user awareness."""
        return [
            item for item in self._items
            if item.get("validation_status") == ValidationStatus.FLAGGED.value
        ]

    def get_failed(self) -> list[EvidenceItem]:
        """Return failed or empty tool evidence items."""
        return [
            item for item in self._items
            if item.get("validation_status") == ValidationStatus.FAILED.value
        ]

    def get_urls(self) -> list[str]:
        """Extract all verified URLs from web search and web reader evidence."""
        urls = []
        for item in self.get_verified():
            src = item.get("source", "")
            if src.startswith("http://") or src.startswith("https://"):
                if src not in urls:
                    urls.append(src)
            # Also check raw data if search returned multiple URLs
            raw = item.get("raw_output")
            if isinstance(raw, dict) and isinstance(raw.get("data"), list):
                for sub in raw["data"]:
                    u = sub.get("url") or sub.get("href")
                    if u and (u.startswith("http://") or u.startswith("https://")) and u not in urls:
                        urls.append(u)
            elif isinstance(raw, list):
                for sub in raw:
                    if isinstance(sub, dict):
                        u = sub.get("url") or sub.get("href")
                        if u and (u.startswith("http://") or u.startswith("https://")) and u not in urls:
                            urls.append(u)
        return urls

    def format_for_context(self) -> str:
        """Format verified and flagged evidence into structured markdown for model synthesis."""
        if not self._items:
            return "(No tool evidence gathered yet)"

        lines = []
        for idx, item in enumerate(self._items, start=1):
            status = item.get("validation_status", "unverified").upper()
            t_type = item.get("type", "generic").replace("_", " ").title()
            source = item.get("source", "unknown")
            content = str(item.get("content", "")).strip()
            details = item.get("validation_details", {})
            confidence = details.get("confidence", 1.0)
            issues = details.get("issues", [])

            status_badge = "✅ VERIFIED" if status == "VERIFIED" else "⚠️ FLAGGED" if status == "FLAGGED" else "❌ FAILED"
            lines.append(f"### [Evidence #{idx}] {t_type} — {status_badge} (Confidence: {confidence:.2f})")
            lines.append(f"- **Source:** `{source}`")
            lines.append(f"- **Tool:** `{item.get('tool_name')}`")
            if issues:
                lines.append(f"- **Validation Notes:** {'; '.join(issues)}")
            lines.append(f"```text\n{content}\n```\n")

        return "\n".join(lines)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return list(self._items)

    @classmethod
    def from_dict_list(cls, items: list[dict[str, Any]] | None) -> EvidenceStore:
        return cls(items or [])

