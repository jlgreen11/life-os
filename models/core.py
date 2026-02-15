"""
Life OS — Core Data Models

These are the foundational types that flow through the entire system.
Every external service, every AI inference, and every user interaction
is represented as one of these types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Top-level event categories. Every event in the system has one."""
    # Communication
    EMAIL_RECEIVED = "email.received"
    EMAIL_SENT = "email.sent"
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_SENT = "message.sent"
    CALL_RECEIVED = "call.received"
    CALL_MISSED = "call.missed"

    # Calendar
    CALENDAR_EVENT_CREATED = "calendar.event.created"
    CALENDAR_EVENT_UPDATED = "calendar.event.updated"
    CALENDAR_EVENT_DELETED = "calendar.event.deleted"
    CALENDAR_EVENT_REMINDER = "calendar.event.reminder"
    CALENDAR_CONFLICT_DETECTED = "calendar.conflict.detected"

    # Tasks
    TASK_CREATED = "task.created"
    TASK_COMPLETED = "task.completed"
    TASK_OVERDUE = "task.overdue"
    TASK_UPDATED = "task.updated"

    # Finance
    TRANSACTION_NEW = "finance.transaction.new"
    BALANCE_CHANGE = "finance.balance.change"
    SUBSCRIPTION_DETECTED = "finance.subscription.detected"
    SPENDING_ANOMALY = "finance.spending.anomaly"

    # Health
    HEALTH_METRIC_UPDATED = "health.metric.updated"
    SLEEP_RECORDED = "health.sleep.recorded"
    EXERCISE_RECORDED = "health.exercise.recorded"

    # Location
    LOCATION_CHANGED = "location.changed"
    LOCATION_ARRIVED = "location.arrived"
    LOCATION_DEPARTED = "location.departed"

    # Smart Home
    DEVICE_STATE_CHANGED = "home.device.state_changed"
    HOME_ARRIVED = "home.arrived"
    HOME_DEPARTED = "home.departed"

    # System
    CONNECTOR_SYNC_COMPLETE = "system.connector.sync_complete"
    CONNECTOR_ERROR = "system.connector.error"
    USER_COMMAND = "system.user.command"
    AI_SUGGESTION = "system.ai.suggestion"
    AI_ACTION_TAKEN = "system.ai.action_taken"
    RULE_TRIGGERED = "system.rule.triggered"

    # Notifications
    NOTIFICATION_CREATED = "notification.created"
    NOTIFICATION_DELIVERED = "notification.delivered"
    NOTIFICATION_DISMISSED = "notification.dismissed"
    NOTIFICATION_ACTED_ON = "notification.acted_on"


class Priority(str, Enum):
    CRITICAL = "critical"       # Wake them up
    HIGH = "high"               # Surface immediately
    NORMAL = "normal"           # Include in next batch
    LOW = "low"                 # Background / daily digest
    SILENT = "silent"           # Log but never notify


class SourceType(str, Enum):
    """Where this event originated."""
    PROTON_MAIL = "proton_mail"
    PROTON_CALENDAR = "proton_calendar"
    PROTON_DRIVE = "proton_drive"
    SIGNAL = "signal"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    CALDAV = "caldav"
    PLAID = "plaid"
    HOME_ASSISTANT = "home_assistant"
    APPLE_HEALTH = "apple_health"
    GARMIN = "garmin"
    RSS = "rss"
    USER_INPUT = "user_input"
    AI_ENGINE = "ai_engine"
    RULES_ENGINE = "rules_engine"
    SYSTEM = "system"


class Tone(str, Enum):
    WARM = "warm"
    CASUAL = "casual"
    PROFESSIONAL = "professional"


class Verbosity(str, Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    DETAILED = "detailed"


class AutonomyLevel(str, Enum):
    SUPERVISED = "supervised"     # Ask about everything
    MODERATE = "moderate"         # Handle obvious stuff, ask about judgment calls
    HIGH = "high"                 # Handle most things, ask about irreversible actions


class NotificationMode(str, Enum):
    MINIMAL = "minimal"           # Only urgent
    BATCHED = "batched"           # 2-3 digest windows per day
    FREQUENT = "frequent"         # Real-time for most things


class BoundaryMode(str, Enum):
    STRICT_SEPARATION = "strict_separation"
    SOFT_SEPARATION = "soft_separation"
    UNIFIED = "unified"


class MoodDimension(str, Enum):
    ENERGY = "energy"
    STRESS = "stress"
    SOCIAL_BATTERY = "social_battery"
    COGNITIVE_LOAD = "cognitive_load"
    EMOTIONAL_VALENCE = "emotional_valence"


class FeedbackType(str, Enum):
    ENGAGED = "engaged"           # Tapped, read, acted on
    IGNORED = "ignored"           # Saw but didn't interact
    DISMISSED = "dismissed"       # Actively swiped away
    OVERRIDDEN = "overridden"     # Changed the AI's suggestion
    DELAYED = "delayed"           # Acted on it much later
    EXPLICIT_POSITIVE = "positive" # "That was great"
    EXPLICIT_NEGATIVE = "negative" # "Don't do that"


class ConfidenceGate(str, Enum):
    """How the AI should behave at different confidence levels."""
    OBSERVE = "observe"           # < 0.3: Watch silently
    SUGGEST = "suggest"           # 0.3-0.6: "Would you like me to..."
    DEFAULT = "default"           # 0.6-0.8: Do it, but let them undo
    AUTONOMOUS = "autonomous"     # > 0.8: Just handle it


# ---------------------------------------------------------------------------
# Core Event Model
# ---------------------------------------------------------------------------

class Event(BaseModel):
    """
    The fundamental unit of the system. Every single thing that happens —
    an email arriving, a location change, an AI inference — is an Event
    flowing through the NATS event bus.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    source: SourceType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: Priority = Priority.NORMAL
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: EventMetadata = Field(default_factory=lambda: EventMetadata())

    # Set after embedding pipeline processes this event
    embedding_id: Optional[str] = None


class EventMetadata(BaseModel):
    """Contextual metadata attached to every event."""
    # Who/what is this about
    related_contacts: list[str] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)

    # Where/when context
    location: Optional[str] = None
    timezone: Optional[str] = None

    # Which life domain does this belong to
    domain: Optional[str] = None            # "work", "family", "hobby", etc.
    vault: Optional[str] = None             # If in a private vault

    # Processing flags
    processed_by: list[str] = Field(default_factory=list)
    requires_action: bool = False
    is_sensitive: bool = False


# ---------------------------------------------------------------------------
# Entity Models (People, Places, Things)
# ---------------------------------------------------------------------------

class Contact(BaseModel):
    """A person in the user's life."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    aliases: list[str] = Field(default_factory=list)         # Nicknames, variations
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    channels: dict[str, str] = Field(default_factory=dict)   # {"signal": "+1...", "slack": "U123"}

    relationship: Optional[str] = None                        # "spouse", "boss", "friend"
    domains: list[str] = Field(default_factory=list)          # ["work", "personal"]
    is_priority: bool = False
    preferred_channel: Optional[str] = None                   # How they prefer to be reached
    always_surface: bool = False                              # Always show their messages

    # Learned over time
    typical_response_time: Optional[float] = None             # seconds
    communication_style: Optional[str] = None                 # "formal", "casual", "terse"
    last_contact: Optional[datetime] = None
    contact_frequency_days: Optional[float] = None            # avg days between contacts
    notes: list[str] = Field(default_factory=list)            # AI-extracted observations

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Place(BaseModel):
    """A meaningful location."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                                                  # "Home", "Office", "Mom's house"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    wifi_ssid: Optional[str] = None                            # For indoor detection
    place_type: Optional[str] = None                           # "home", "work", "gym", "restaurant"
    domain: Optional[str] = None                               # Which life domain this belongs to
    visit_count: int = 0
    avg_duration_minutes: Optional[float] = None
    associated_behaviors: dict[str, Any] = Field(default_factory=dict)


class Subscription(BaseModel):
    """A recurring service or payment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    amount: float
    currency: str = "USD"
    frequency: str = "monthly"                                 # "monthly", "yearly", "weekly"
    last_charge: Optional[datetime] = None
    next_charge: Optional[datetime] = None
    category: Optional[str] = None
    last_used: Optional[datetime] = None                       # When the user last opened/used this
    usage_frequency: Optional[str] = None                      # "daily", "weekly", "rarely", "never"
    cancel_url: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Communication Models
# ---------------------------------------------------------------------------

class MessagePayload(BaseModel):
    """Payload for any message event (email, chat, SMS)."""
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    channel: str                                               # "proton_mail", "signal", "slack"
    direction: str                                             # "inbound" or "outbound"

    from_contact: Optional[str] = None                         # Contact ID or raw address
    from_address: Optional[str] = None                         # Raw email/phone
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)

    subject: Optional[str] = None
    body: str = ""
    body_plain: str = ""                                       # Stripped text version
    snippet: str = ""                                          # First ~100 chars

    has_attachments: bool = False
    attachment_names: list[str] = Field(default_factory=list)
    is_reply: bool = False
    is_forwarded: bool = False
    in_reply_to: Optional[str] = None

    # AI-extracted
    sentiment: Optional[float] = None                          # -1 to 1
    urgency: Optional[float] = None                            # 0 to 1
    action_items: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    requires_response: Optional[bool] = None


class CalendarEventPayload(BaseModel):
    """Payload for calendar events."""
    event_id: Optional[str] = None
    calendar_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    start_time: datetime
    end_time: datetime
    is_all_day: bool = False
    attendees: list[str] = Field(default_factory=list)
    organizer: Optional[str] = None
    status: str = "confirmed"                                  # "confirmed", "tentative", "cancelled"
    recurrence: Optional[str] = None
    reminders: list[int] = Field(default_factory=list)         # Minutes before


class TransactionPayload(BaseModel):
    """Payload for financial transactions."""
    transaction_id: Optional[str] = None
    account_id: Optional[str] = None
    account_name: Optional[str] = None
    amount: float
    currency: str = "USD"
    merchant: Optional[str] = None
    category: Optional[str] = None
    date: datetime
    is_pending: bool = False
    is_recurring: Optional[bool] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Task Model
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """A tracked action item — created by user, AI, or rules."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None

    # Origin
    source: str = "user"                                       # "user", "ai_extracted", "rule"
    source_event_id: Optional[str] = None                      # Event that generated this task
    source_context: Optional[str] = None                       # "Boss said in Slack: ..."

    # Classification
    domain: Optional[str] = None                               # "work", "personal", etc.
    priority: Priority = Priority.NORMAL
    tags: list[str] = Field(default_factory=list)

    # Timing
    due_date: Optional[datetime] = None
    reminder_at: Optional[datetime] = None
    estimated_minutes: Optional[int] = None

    # Relations
    related_contacts: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    related_events: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)        # Other task IDs

    # State
    status: str = "pending"                                    # "pending", "in_progress", "completed", "cancelled"
    completed_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
