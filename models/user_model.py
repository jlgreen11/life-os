"""
Life OS — User Model

This is the heart of the system. The User Model is a continuously-evolving
representation of who the user is — their communication style, emotional
patterns, decision-making habits, relationships, and rhythms of life.

The model is built entirely from passive observation. The user never fills
out a form or takes a quiz. They just live their life, and the model learns.

Architecture:
    Layer 1: Episodic Memory    — Individual events with full context
    Layer 2: Semantic Memory    — Distilled facts, preferences, relationships
    Layer 3: Procedural Memory  — Learned workflows and habits
    Layer 4: Predictive Models  — Forward-looking intelligence
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from models.core import (
    AutonomyLevel,
    BoundaryMode,
    ConfidenceGate,
    NotificationMode,
    Priority,
    Tone,
    Verbosity,
)


# ===========================================================================
# SIGNAL PROFILES — Raw behavioral signals extracted from interactions
# ===========================================================================

class LinguisticProfile(BaseModel):
    """
    How the user communicates. Built from every message they send,
    every draft they edit, every voice command they speak.
    """
    # Complexity & Style
    vocabulary_complexity: float = 0.5          # 0=simple, 1=complex (Flesch-Kincaid derived)
    avg_sentence_length: float = 12.0           # words
    formality_spectrum: float = 0.5             # 0="hey whats up" → 1="I hope this finds you well"

    # Personality Markers
    hedge_frequency: float = 0.0                # "maybe", "I think", "sort of" — per message
    assertion_frequency: float = 0.0            # "We need to", "This must" — per message
    question_frequency: float = 0.0             # How often they ask vs state
    humor_markers: list[str] = Field(default_factory=list)  # Detected humor types

    # Punctuation & Formatting
    exclamation_rate: float = 0.0               # Per message
    emoji_usage: dict[str, float] = Field(default_factory=dict)  # Emoji → frequency
    uses_oxford_comma: Optional[bool] = None
    capitalization_style: str = "standard"       # "standard", "all_lower", "mixed"
    ellipsis_frequency: float = 0.0

    # Response Patterns — how they say common things
    affirmative_patterns: list[str] = Field(default_factory=list)   # ["sure", "yep", "sounds good"]
    negative_patterns: list[str] = Field(default_factory=list)      # ["nah", "I don't think so"]
    gratitude_patterns: list[str] = Field(default_factory=list)     # ["thanks!", "appreciate it"]
    greeting_patterns: list[str] = Field(default_factory=list)      # ["hey", "hi there"]
    closing_patterns: list[str] = Field(default_factory=list)       # ["cheers", "talk soon"]

    # Per-Contact Style Variations — enables per-contact voice matching for AI-drafted messages
    style_by_contact: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # {"contact_id": {"formality": 0.8, "emoji_rate": 0.1, "avg_length": 45}}

    # Profanity
    profanity_comfort: float = 0.0              # 0=never, 1=frequently
    profanity_contexts: list[str] = Field(default_factory=list)  # When they swear

    samples_analyzed: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CadenceProfile(BaseModel):
    """
    When and how quickly the user communicates. Reveals who matters,
    what's being avoided, and natural rhythms.
    """
    # Response times (seconds) — who they prioritize
    avg_response_time_by_contact: dict[str, float] = Field(default_factory=dict)
    avg_response_time_by_channel: dict[str, float] = Field(default_factory=dict)
    avg_response_time_by_domain: dict[str, float] = Field(default_factory=dict)

    # Activity windows
    hourly_activity: dict[int, float] = Field(default_factory=dict)     # Hour → message count avg
    daily_activity: dict[str, float] = Field(default_factory=dict)      # "monday" → activity level
    peak_hours: list[int] = Field(default_factory=list)                  # Most active hours
    quiet_hours_observed: list[tuple[int, int]] = Field(default_factory=list)  # Naturally quiet periods

    # Conversation patterns
    initiates_ratio_by_contact: dict[str, float] = Field(default_factory=dict)  # Who starts convos
    thread_completion_rate: float = 0.5          # Do they follow through on threads?
    avg_thread_length: float = 3.0               # Messages per conversation

    # Avoidance signals — read_not_replied is a key avoidance detection signal
    read_not_replied: list[dict[str, Any]] = Field(default_factory=list)  # Messages seen but ignored
    avg_delay_for_difficult_topics: Optional[float] = None  # Seconds longer for hard conversations

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TemporalProfile(BaseModel):
    """
    The user's relationship with time — energy rhythms, weekly patterns,
    seasonal trends, and how they handle deadlines.
    """
    # Chronotype
    chronotype: str = "unknown"                  # "early_bird", "night_owl", "variable"
    typical_wake_time: Optional[time] = None
    typical_sleep_time: Optional[time] = None

    # Energy curve (0-1 energy level by hour)
    energy_by_hour: dict[int, float] = Field(default_factory=dict)

    # Weekly rhythm
    productive_days: list[str] = Field(default_factory=list)     # ["monday", "tuesday", "wednesday"]
    social_days: list[str] = Field(default_factory=list)          # ["thursday", "friday"]
    recharge_days: list[str] = Field(default_factory=list)        # ["saturday"]

    # Deadline behavior
    procrastination_score: float = 0.5           # 0=always early, 1=always last minute
    advance_planning_horizon_days: dict[str, float] = Field(default_factory=dict)
    # {"travel": 30, "social": 1, "work_deadlines": 2}

    # Pre/post event patterns
    pre_event_anxiety_signals: list[str] = Field(default_factory=list)
    post_social_recovery_hours: Optional[float] = None  # Introvert/extrovert signal

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionProfile(BaseModel):
    """
    How the user makes decisions — speed, research depth, risk tolerance,
    and who they defer to.
    """
    # Speed by domain (seconds to decide)
    decision_speed_by_domain: dict[str, float] = Field(default_factory=dict)
    # {"food": 5, "purchases_under_50": 10, "purchases_over_500": 86400}

    # Research depth (0=gut feel, 1=exhaustive research)
    research_depth_by_domain: dict[str, float] = Field(default_factory=dict)

    # Risk tolerance (0=very conservative, 1=very adventurous)
    risk_tolerance_by_domain: dict[str, float] = Field(default_factory=dict)
    # {"restaurants": 0.8, "finance": 0.2, "career": 0.4}

    # Delegation
    delegation_comfort: float = 0.5              # 0=micromanage everything, 1=fully delegated
    delegation_by_domain: dict[str, float] = Field(default_factory=dict)

    # Social influence
    defers_to: dict[str, list[str]] = Field(default_factory=dict)
    # {"restaurants": ["partner", "friend_mike"], "tech": ["self"]}

    # Decision fatigue
    fatigue_time_of_day: Optional[int] = None    # Hour when "whatever, you pick" starts
    fatigue_trigger_count: Optional[int] = None  # Decisions per day before fatigue

    # Reversal rate
    mind_change_frequency: float = 0.1           # How often they reverse decisions

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpatialProfile(BaseModel):
    """
    How behavior changes based on location. Powers context-switching
    when the user moves between home, work, gym, etc.
    """
    # Known places and their behavioral signatures
    place_behaviors: dict[str, PlaceBehavior] = Field(default_factory=dict)

    # Transition patterns
    typical_transitions: list[dict[str, Any]] = Field(default_factory=list)
    # [{"from": "home", "to": "work", "typical_time": "08:30", "duration_min": 25}]

    # Current location awareness
    current_place_id: Optional[str] = None
    last_transition: Optional[datetime] = None

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlaceBehavior(BaseModel):
    """Behavioral patterns associated with a specific place."""
    place_id: str
    notification_preference: str = "normal"      # "minimal", "normal", "frequent"
    dominant_domain: str = "personal"             # "work", "personal", "social"
    typical_duration_minutes: float = 60.0
    typical_activities: list[str] = Field(default_factory=list)
    tone_shift: Optional[str] = None             # "more_formal", "more_casual"


# ===========================================================================
# MOOD & EMOTIONAL STATE — Inferred, never asked
# ===========================================================================

class MoodState(BaseModel):
    """
    The current inferred emotional/cognitive state of the user.
    Built from a composite of all available signals.
    
    CRITICAL: This is NEVER shared with anyone. It's used solely to 
    adjust the AI's behavior (tone, timing, proactivity).
    """
    # Core dimensions — all use a 0.0-1.0 scale convention across every dimension
    energy_level: float = 0.5
    stress_level: float = 0.3
    social_battery: float = 0.5
    cognitive_load: float = 0.3
    emotional_valence: float = 0.5               # 0=very negative, 1=very positive

    # How confident are we in this reading
    confidence: float = 0.0                       # 0=guessing, 1=very confident

    # What signals contributed to this inference
    contributing_signals: list[MoodSignal] = Field(default_factory=list)

    # Trend
    trend: str = "stable"                         # "improving", "declining", "stable", "volatile"

    inferred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MoodSignal(BaseModel):
    """A single signal that contributed to the mood inference."""
    signal_type: str                              # "typing_speed", "sleep_quality", "music_choice"
    value: float                                  # The raw signal value
    delta_from_baseline: float                    # How far from their normal
    weight: float                                 # How much this matters for mood inference
    source: str                                   # Which connector/system provided this


# ===========================================================================
# LAYER 1: EPISODIC MEMORY — Individual events with full context
# ===========================================================================

class Episode(BaseModel):
    """
    A single remembered interaction. Not just what happened, but the full
    context: where the user was, how they were feeling, what they did with it.
    
    Episodes are the raw material from which all other memory layers are built.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    event_id: str                                 # Reference to the source Event

    # Context at time of episode
    location: Optional[str] = None
    inferred_mood: Optional[MoodState] = None
    active_domain: Optional[str] = None           # "work", "personal", etc.
    energy_level: Optional[float] = None

    # Content
    interaction_type: str                         # "voice_command", "text_input", "passive", "draft_edit"
    content_summary: str                          # Brief description
    content_full: Optional[str] = None            # Full text (for search)

    # Entities involved
    contacts_involved: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)  # Places, things, concepts

    # What happened as a result
    outcome: Optional[str] = None                 # "user_acted", "user_ignored", "ai_handled"
    user_satisfaction: Optional[float] = None      # 0-1 based on implicit signals

    # For vector search
    embedding_id: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ===========================================================================
# LAYER 2: SEMANTIC MEMORY — Distilled knowledge about the user
# ===========================================================================

class SemanticMemory(BaseModel):
    """
    Long-term knowledge about the user, distilled from episodes.
    These are facts and preferences that persist across time.
    """
    # Hard facts
    facts: dict[str, SemanticFact] = Field(default_factory=dict)
    # {"shellfish_allergy": SemanticFact(value="allergic to shellfish", ...)}

    # Explicit preferences (user stated)
    explicit_preferences: dict[str, SemanticFact] = Field(default_factory=dict)
    # {"seat_preference": SemanticFact(value="window seat", ...)}

    # Implicit preferences (observed but never stated)
    implicit_preferences: dict[str, SemanticFact] = Field(default_factory=dict)
    # {"prefers_text_over_calls": SemanticFact(value=True, confidence=0.85, ...)}

    # Anti-preferences (things they dislike)
    anti_preferences: dict[str, SemanticFact] = Field(default_factory=dict)
    # {"reply_all": SemanticFact(value="hates reply-all emails", ...)}

    # Expertise map
    expertise: dict[str, float] = Field(default_factory=dict)
    # {"python": 0.9, "cooking": 0.4, "tax_law": 0.1}

    # Values (what they prioritize in life)
    values: dict[str, float] = Field(default_factory=dict)
    # {"privacy": 0.95, "family_time": 0.9, "career_growth": 0.7, "fitness": 0.5}

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticFact(BaseModel):
    """A single piece of knowledge with provenance."""
    key: str
    value: Any
    confidence: float = 0.5                       # 0-1
    source_episodes: list[str] = Field(default_factory=list)  # Episode IDs that support this
    first_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    times_confirmed: int = 1
    is_user_corrected: bool = False               # User explicitly set/corrected this


# ===========================================================================
# LAYER 3: PROCEDURAL MEMORY — How the user does things
# ===========================================================================

class ProceduralMemory(BaseModel):
    """
    Learned sequences and workflows. How the user does recurring things.
    """
    routines: dict[str, Routine] = Field(default_factory=dict)
    workflows: dict[str, Workflow] = Field(default_factory=dict)
    communication_templates: dict[str, CommunicationTemplate] = Field(default_factory=dict)

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Routine(BaseModel):
    """A recurring behavioral pattern (morning routine, end of day, etc.)."""
    name: str
    trigger: str                                  # "wake_up", "arrive_work", "friday_evening"
    steps: list[RoutineStep] = Field(default_factory=list)
    typical_duration_minutes: float = 30.0
    consistency_score: float = 0.5                # How reliably they follow this pattern
    times_observed: int = 0
    variations: list[str] = Field(default_factory=list)  # Known departures from the pattern


class RoutineStep(BaseModel):
    """A single step in a routine."""
    order: int
    action: str                                   # "check_email", "review_calendar", "make_coffee"
    typical_duration_minutes: float = 5.0
    skip_rate: float = 0.0                        # How often this step is skipped


class Workflow(BaseModel):
    """A multi-step process for accomplishing a specific type of task."""
    name: str                                     # "responding_to_boss", "planning_trip", "weekly_review"
    trigger_conditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    typical_duration_minutes: Optional[float] = None
    tools_used: list[str] = Field(default_factory=list)
    success_rate: float = 0.5
    times_observed: int = 0


class CommunicationTemplate(BaseModel):
    """
    How the user writes to specific people or in specific contexts.
    Used by the AI to draft messages in the user's voice.
    """
    context: str                                  # "email_to_boss", "text_to_partner", "slack_team"
    contact_id: Optional[str] = None
    channel: Optional[str] = None

    # Style parameters for this specific context
    greeting: Optional[str] = None                # "Hey", "Hi [Name]", none
    closing: Optional[str] = None                 # "Best", "Talk soon", none
    formality: float = 0.5
    typical_length_words: float = 50.0
    uses_emoji: bool = False
    common_phrases: list[str] = Field(default_factory=list)
    avoids_phrases: list[str] = Field(default_factory=list)
    tone_notes: list[str] = Field(default_factory=list)  # "always leads with conclusion", "uses bullet points"

    # Example messages (for few-shot prompting)
    example_message_ids: list[str] = Field(default_factory=list)

    samples_analyzed: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ===========================================================================
# LAYER 4: PREDICTIVE MODELS — Forward-looking intelligence
# ===========================================================================

class Prediction(BaseModel):
    """A single prediction about what the user will need."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_type: str                          # "need", "conflict", "opportunity", "risk", "reminder", "routine_deviation"
    description: str                              # Human-readable description
    confidence: float                             # 0-1
    confidence_gate: ConfidenceGate               # How the AI should act on this
    time_horizon: str                             # "15_minutes", "2_hours", "24_hours", "this_week"

    # What should happen
    suggested_action: Optional[str] = None
    suggested_message: Optional[str] = None

    # Context
    supporting_signals: dict[str, Any] = Field(default_factory=dict)  # Structured metadata for behavioral inference
    relevant_episodes: list[str] = Field(default_factory=list)
    relevant_contacts: list[str] = Field(default_factory=list)

    # Evaluation
    was_surfaced: bool = False
    user_response: Optional[str] = None           # FeedbackType
    was_accurate: Optional[bool] = None
    filter_reason: Optional[str] = None           # Why prediction was filtered (e.g., "reaction_score:-0.2", "confidence:0.15")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[str] = None             # ISO timestamp when prediction was resolved


class ReactionPrediction(BaseModel):
    """
    Before surfacing anything, predict how the user will react.
    This is the gatekeeper that prevents annoying suggestions.
    """
    proposed_action: str
    predicted_reaction: str                       # "helpful", "annoying", "intrusive", "neutral"
    confidence: float
    reasoning: str

    # Context that informed this prediction
    current_mood: Optional[MoodState] = None
    current_location: Optional[str] = None
    current_time: Optional[datetime] = None
    recent_dismissals: int = 0                    # How many things they've dismissed recently


# ===========================================================================
# USER PREFERENCES — Set during onboarding, refined over time
# ===========================================================================

class UserPreferences(BaseModel):
    """
    The explicit preferences captured during onboarding and refined
    through interaction. These are the user's stated wishes.
    """
    # Communication
    verbosity: Verbosity = Verbosity.BALANCED
    tone: Tone = Tone.CASUAL
    proactivity: str = "moderate"                 # "low", "moderate", "high"

    # Autonomy
    autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE
    draft_replies: bool = True
    auto_actions: list[str] = Field(default_factory=list)  # Actions AI can take without asking

    # Life domains
    life_domains: list[LifeDomain] = Field(default_factory=list)

    # Priority contacts
    priority_contacts: list[str] = Field(default_factory=list)  # Contact IDs

    # Vaults (private compartments)
    vaults: list[Vault] = Field(default_factory=list)

    # Screen privacy
    screen_privacy: ScreenPrivacy = Field(default_factory=lambda: ScreenPrivacy())

    # Notifications
    notification_mode: NotificationMode = NotificationMode.BATCHED
    quiet_hours: list[QuietHours] = Field(default_factory=list)

    # Updated by the system as it learns
    effective_preferences: dict[str, Any] = Field(default_factory=dict)

    onboarding_completed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LifeDomain(BaseModel):
    """A compartment of the user's life."""
    name: str                                     # "work", "family", "hobby:guitar"
    context: Optional[str] = None                 # "consulting firm", "spouse + 2 kids"
    boundary: BoundaryMode = BoundaryMode.SOFT_SEPARATION
    notification_priority: Priority = Priority.NORMAL
    active_hours: Optional[tuple[int, int]] = None  # (9, 17) for work


class Vault(BaseModel):
    """A private compartment with separate access control."""
    name: str
    auth_method: str = "biometric"                # "biometric", "pin", "passphrase"
    excluded_from: list[str] = Field(default_factory=lambda: ["search", "briefing", "unified_inbox"])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScreenPrivacy(BaseModel):
    """What to hide on screen by default."""
    finances_redacted: bool = False
    blur_images: bool = False
    hide_message_previews: bool = False
    sensitive_domains: list[str] = Field(default_factory=list)


class QuietHours(BaseModel):
    """Time periods when only emergencies get through."""
    start: time
    end: time
    days: list[str] = Field(default_factory=lambda: [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday"
    ])
    exceptions: list[str] = Field(default_factory=lambda: ["emergency", "priority_contacts"])


# ===========================================================================
# COMPOSITE: THE COMPLETE USER MODEL
# ===========================================================================

# Composite object that aggregates all four memory layers (episodic, semantic,
# procedural, predictive) along with signal profiles and user preferences
class UserModel(BaseModel):
    """
    The complete model of the user. This is the single source of truth
    that every service in the system reads from.

    It's stored locally, encrypted at rest, and never transmitted.
    """
    # Explicit preferences (from onboarding + manual updates)
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    # Signal profiles (passively built)
    linguistic: LinguisticProfile = Field(default_factory=LinguisticProfile)
    cadence: CadenceProfile = Field(default_factory=CadenceProfile)
    temporal: TemporalProfile = Field(default_factory=TemporalProfile)
    decisions: DecisionProfile = Field(default_factory=DecisionProfile)
    spatial: SpatialProfile = Field(default_factory=SpatialProfile)

    # Current state
    current_mood: MoodState = Field(default_factory=MoodState)

    # Memory layers
    semantic_memory: SemanticMemory = Field(default_factory=SemanticMemory)
    procedural_memory: ProceduralMemory = Field(default_factory=ProceduralMemory)

    # Metadata
    model_version: str = "0.1.0"
    total_events_processed: int = 0
    total_episodes_stored: int = 0
    model_quality_score: float = 0.0              # 0-1: how well do we know this user
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Need this import at the top for Episode
import uuid
