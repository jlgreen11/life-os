"""
Life OS — Web API Request/Response Schemas

Pydantic models that define the shape of incoming JSON request bodies for each
API endpoint.  FastAPI automatically validates requests against these schemas
and returns 422 errors for malformed input.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# --- POST /api/command ---
# The unified command bar input.  ``text`` is the raw user command string;
# ``context`` carries optional client-side state (e.g. current view).
class CommandRequest(BaseModel):
    text: str
    context: Optional[dict] = None


# --- POST /api/tasks ---
# Create a new task.  Only ``title`` is required; all other fields have
# sensible defaults so that quick-capture ("task Buy groceries") works.
class TaskCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    domain: Optional[str] = None
    priority: str = "normal"
    due_date: Optional[str] = None


# --- PATCH /api/tasks/{task_id} ---
# Partial update — all fields are optional; only supplied fields are changed.
class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None
    title: Optional[str] = None


# --- POST /api/rules ---
# Create an automation rule.  ``trigger_event`` specifies the event type that
# activates the rule; ``conditions`` and ``actions`` are lists of JSON objects
# interpreted by the rules engine.
class RuleCreateRequest(BaseModel):
    name: str
    trigger_event: str
    conditions: list[dict] = []
    actions: list[dict] = []


# --- POST /api/search ---
# Semantic search request.  ``filters`` allows narrowing results by metadata
# fields (e.g. {"source": "email"}).
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[dict] = None


# --- POST /api/draft ---
# Request an AI-generated message draft.  ``incoming_message`` provides the
# message being replied to; ``contact_id`` and ``channel`` guide style matching.
class DraftRequest(BaseModel):
    contact_id: Optional[str] = None
    channel: str = "email"
    incoming_message: str = ""
    context: Optional[str] = None


# --- POST /api/messages/send ---
# Send a direct message via the appropriate messaging connector (iMessage or
# Signal).  ``channel`` hints which connector to prefer; ``recipient`` is the
# destination address (phone number, Apple ID, or Signal number).
class SendMessageRequest(BaseModel):
    recipient: str
    message: str
    channel: str = "message"  # "imessage", "signal", or generic "message"


# --- POST /api/feedback ---
# Explicit user feedback (free-text).  Processed by the feedback collector
# to update the learning loop.
class FeedbackRequest(BaseModel):
    message: str


# --- PATCH /api/user-model/facts/{key} ---
# Correct a semantic fact.  When the user identifies an incorrect fact, this
# endpoint marks it as corrected and reduces its confidence to discourage
# future use.  Optionally provides a corrected value.
class FactCorrectionRequest(BaseModel):
    corrected_value: Optional[Any] = None
    reason: Optional[str] = None


# --- POST /api/user-model/facts/{key}/confirm ---
# Confirm an inferred semantic fact is correct.  This is the positive
# counterpart to FactCorrectionRequest: it bumps confidence by +0.05
# (matching the architectural rule in CLAUDE.md) and increments
# times_confirmed.
class FactConfirmationRequest(BaseModel):
    reason: Optional[str] = None


# --- PUT /api/preferences ---
# Update a single user preference (key-value pair).  ``value`` is typed as
# ``Any`` to support strings, numbers, booleans, and nested objects.
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


class ConnectorConfigRequest(BaseModel):
    config: dict[str, Any]


class SetupSubmitRequest(BaseModel):
    step_id: str
    value: Any


# ---------------------------------------------------------------------------
# Source Weights (tunable insight engine)
# ---------------------------------------------------------------------------

class SourceWeightUpdate(BaseModel):
    """Update the user-controlled weight for a data source."""
    weight: float  # 0.0 = ignore, 1.0 = max influence


class SourceWeightCreate(BaseModel):
    """Create a custom source weight entry."""
    source_key: str
    category: str
    label: str
    description: str = ""
    weight: float = 0.5
