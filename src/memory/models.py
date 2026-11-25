"""
Pydantic schemas for governed memory entities with provenance, classification,
lifecycle expiration, confidence, and tenant isolation.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    """Canonical 4-class memory governance taxonomy plus legacy aliases."""
    PREFERENCE = "preference"              # User tastes, style preferences, output formatting constraints
    FACT = "fact"                          # Technical environment, personal facts, infrastructure details
    PROJECT = "project"                    # Architecture decisions, repo links, sprint/task context
    TEMPORARY_CONTEXT = "temporary_context"  # Ephemeral session context, current debugging focus (short TTL)

    # Legacy category aliases
    USER_PREFERENCE = "preference"
    TECHNICAL_ENVIRONMENT = "fact"
    PROJECT_CONTEXT = "project"
    GENERAL_FACT = "fact"
    EXTERNAL_REFERENCE = "fact"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        val_str = str(value).lower()
        if "pref" in val_str:
            return cls.PREFERENCE
        if "proj" in val_str:
            return cls.PROJECT
        if "temp" in val_str or "ephemeral" in val_str:
            return cls.TEMPORARY_CONTEXT
        return cls.FACT


# Standard Time-To-Live (TTL) in days based on memory classification
CATEGORY_DEFAULT_TTL_DAYS: dict[str, Optional[int]] = {
    MemoryCategory.TEMPORARY_CONTEXT.value: 2,   # 48 hours for transient session context
    MemoryCategory.PROJECT.value: 90,            # 90 days for project lifecycle context
    MemoryCategory.FACT.value: 180,              # 180 days for technical environment facts
    MemoryCategory.PREFERENCE.value: 365,        # 1 year / durable user preference
}


class MemoryItem(BaseModel):
    """Structured memory entity with provenance, lifecycle governance, confidence, and tenant boundary."""
    id: str
    user_id: str
    tenant_id: str = "default"
    text: str
    score: Optional[float] = None
    category: str = "preference"  # "preference" | "fact" | "project" | "temporary_context"
    confidence: float = 1.0       # 0.0 to 1.0 confidence in extracted fact
    source: str = "user"          # "user" | "system" | "web_search" | "agent_inference" | "github" | "tool_output"
    trust_level: str = "user_confirmed"  # "user_confirmed" | "user_stated" | "external_untrusted" | "system"
    verification_status: str = "user_confirmed"  # "user_confirmed" | "system_inferred" | "unverified"
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    expires_at: Optional[str] = None
    is_active: bool = True
    superseded_by: Optional[str] = None
    superseded_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        # Auto-resolve category to canonical enum
        cat_raw = data.get("category", "preference")
        resolved_cat = (
            cat_raw.value if isinstance(cat_raw, MemoryCategory)
            else (MemoryCategory(cat_raw).value if hasattr(MemoryCategory, str(cat_raw).upper())
                  else MemoryCategory._missing_(cat_raw).value)
        )
        data["category"] = resolved_cat

        # Calculate default expiration if not explicitly provided
        if "expires_at" not in data or data["expires_at"] is None:
            ttl_days = CATEGORY_DEFAULT_TTL_DAYS.get(resolved_cat)
            if ttl_days:
                now = datetime.datetime.now(datetime.timezone.utc)
                data["expires_at"] = (now + datetime.timedelta(days=ttl_days)).isoformat()

        super().__init__(**data)

    def is_expired(self, now: Optional[datetime.datetime] = None) -> bool:
        """Check if memory has passed its expiration timestamp."""
        if not self.expires_at:
            return False
        current_time = now or datetime.datetime.now(datetime.timezone.utc)
        try:
            exp = datetime.datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=datetime.timezone.utc)
            return current_time > exp
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "text": self.text,
            "score": self.score,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "trust_level": self.trust_level,
            "verification_status": self.verification_status,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "metadata": self.metadata,
        }


class MemoryWriteResult(BaseModel):
    """Result of memory lifecycle write operation (validation, deduplication, superseding)."""
    id: str
    status: str  # "created" | "updated" | "skipped" | "superseded" | "rejected_poisoning" | "rejected_transient"
    category: str = "preference"
    confidence: float = 1.0
    tenant_id: str = "default"
    superseded_memory_id: Optional[str] = None
    reason: Optional[str] = None
