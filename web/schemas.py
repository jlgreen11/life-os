"""
Life OS — Web API Request/Response Schemas
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class CommandRequest(BaseModel):
    text: str
    context: Optional[dict] = None


class TaskCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    domain: Optional[str] = None
    priority: str = "normal"
    due_date: Optional[str] = None


class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None
    title: Optional[str] = None


class RuleCreateRequest(BaseModel):
    name: str
    trigger_event: str
    conditions: list[dict] = []
    actions: list[dict] = []


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[dict] = None


class DraftRequest(BaseModel):
    contact_id: Optional[str] = None
    channel: str = "email"
    incoming_message: str = ""
    context: Optional[str] = None


class FeedbackRequest(BaseModel):
    message: str


class PreferenceUpdate(BaseModel):
    key: str
    value: Any


# ---------------------------------------------------------------------------
# Context Events (from iOS app / mobile devices)
# ---------------------------------------------------------------------------

class ContextPayload(BaseModel):
    """Payload for contextual data from mobile devices."""
    # Location fields
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    horizontal_accuracy: Optional[float] = None
    speed: Optional[float] = None
    place_name: Optional[str] = None
    place_type: Optional[str] = None

    # Device discovery fields
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    signal_strength: Optional[int] = None
    is_connected: Optional[bool] = None

    # Time context fields
    local_time: Optional[str] = None
    timezone: Optional[str] = None
    day_of_week: Optional[str] = None
    is_weekend: Optional[bool] = None

    # Activity fields
    activity: Optional[str] = None
    confidence: Optional[float] = None


class ContextMetadata(BaseModel):
    """Device metadata from mobile client."""
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    battery_level: Optional[float] = None
    network_type: Optional[str] = None
    app_state: Optional[str] = None


class ContextEventRequest(BaseModel):
    """A single context event from the mobile app."""
    type: str
    source: str = "ios_app"
    timestamp: Optional[str] = None
    payload: ContextPayload
    metadata: Optional[ContextMetadata] = None


class ContextBatchRequest(BaseModel):
    """A batch of context events from the mobile app."""
    events: list[ContextEventRequest]
