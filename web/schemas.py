"""
Life OS — Web API Request/Response Schemas

Pydantic models that define the shape of incoming JSON request bodies for each
API endpoint.  FastAPI automatically validates requests against these schemas
and returns 422 errors for malformed input.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# --- POST /api/command ---
# The unified command bar input.  ``text`` is the raw user command string;
# ``context`` carries optional client-side state (e.g. current view).
class CommandRequest(BaseModel):
    text: str
    context: dict | None = None


# --- POST /api/tasks ---
# Create a new task.  Only ``title`` is required; all other fields have
# sensible defaults so that quick-capture ("task Buy groceries") works.
class TaskCreateRequest(BaseModel):
    title: str
    description: str | None = None
    domain: str | None = None
    priority: str = "normal"
    due_date: str | None = None


# --- PATCH /api/tasks/{task_id} ---
# Partial update — all fields are optional; only supplied fields are changed.
class TaskUpdateRequest(BaseModel):
    status: str | None = None
    priority: str | None = None
    due_date: str | None = None
    title: str | None = None


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
    filters: dict | None = None


# --- POST /api/draft ---
# Request an AI-generated message draft.  ``incoming_message`` provides the
# message being replied to; ``contact_id`` and ``channel`` guide style matching.
class DraftRequest(BaseModel):
    contact_id: str | None = None
    channel: str = "email"
    incoming_message: str = ""
    context: str | None = None


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
    corrected_value: Any | None = None
    reason: str | None = None


# --- POST /api/user-model/facts/{key}/confirm ---
# Confirm an inferred semantic fact is correct.  This is the positive
# counterpart to FactCorrectionRequest: it bumps confidence by +0.05
# (matching the architectural rule in CLAUDE.md) and increments
# times_confirmed.
class FactConfirmationRequest(BaseModel):
    reason: str | None = None


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
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    horizontal_accuracy: float | None = None
    speed: float | None = None
    place_name: str | None = None
    place_type: str | None = None

    # Device discovery fields
    device_name: str | None = None
    device_type: str | None = None
    signal_strength: int | None = None
    is_connected: bool | None = None

    # Time context fields
    local_time: str | None = None
    timezone: str | None = None
    day_of_week: str | None = None
    is_weekend: bool | None = None

    # Activity fields
    activity: str | None = None
    confidence: float | None = None


class ContextMetadata(BaseModel):
    """Device metadata from mobile client."""

    device_model: str | None = None
    os_version: str | None = None
    battery_level: float | None = None
    network_type: str | None = None
    app_state: str | None = None


class ContextEventRequest(BaseModel):
    """A single context event from the mobile app."""

    type: str
    source: str = "ios_app"
    timestamp: str | None = None
    payload: ContextPayload
    metadata: ContextMetadata | None = None


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


class BackupRestoreRequest(BaseModel):
    """Request body for restoring a database from a backup file."""

    backup_path: str
    db_name: str = "user_model"


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


# --- PATCH /api/user-model/templates/{template_id} ---
# Partial update for a communication template.  All fields are optional;
# only supplied fields are changed.  Structural fields (id, context,
# contact_id, channel) are not exposed here — they are immutable.
class BadgeCountsResponse(BaseModel):
    """Response model for GET /api/dashboard/badges.

    Returns per-topic badge counts so the frontend can update tab badges
    with a single lightweight request instead of 5 separate feed calls.
    """

    badges: dict[str, int]


class TemplateUpdateRequest(BaseModel):
    greeting: str | None = None
    closing: str | None = None
    formality: float | None = None
    typical_length: float | None = None
    uses_emoji: bool | None = None
    common_phrases: list[str] | None = None
    avoids_phrases: list[str] | None = None
    tone_notes: list[str] | None = None
