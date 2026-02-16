from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class Insight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "behavioral_pattern", "actionable_alert", "relationship_intelligence"
    summary: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    category: str = ""
    entity: Optional[str] = None
    source_key: str = ""  # Source weight key for feedback routing (e.g. "email.marketing")
    staleness_ttl_hours: int = 168  # 7 days
    dedup_key: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feedback: Optional[str] = None

    def compute_dedup_key(self) -> str:
        raw = f"{self.type}:{self.category}:{self.entity or ''}"
        self.dedup_key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.dedup_key
