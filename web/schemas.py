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


# --- POST /api/feedback ---
# Explicit user feedback (free-text).  Processed by the feedback collector
# to update the learning loop.
class FeedbackRequest(BaseModel):
    message: str


# --- PUT /api/preferences ---
# Update a single user preference (key-value pair).  ``value`` is typed as
# ``Any`` to support strings, numbers, booleans, and nested objects.
class PreferenceUpdate(BaseModel):
    key: str
    value: Any
