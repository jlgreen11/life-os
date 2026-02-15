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
