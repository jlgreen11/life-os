"""Pydantic request/response schemas for the v2 API surface.

Single source of truth for the JSON wire format of the 14 REST endpoints
listed in ``docs/plans/2026-04-21-v2-rewrite-plan.md`` § "Locked REST
endpoints". The schemas deliberately mirror the shape of
:class:`core.moment.types.Moment` and the engineering-plan payload
contracts, not a 1:1 ORM projection — the API is the public contract
and the repo is free to evolve its columns underneath.

Pydantic v2 is required (see :mod:`requirements.txt`). All schemas are
frozen via ``model_config = ConfigDict(extra="forbid")`` so that
unexpected fields fail validation loudly — clients cannot silently
feed us forward-compatible junk.

Schemas locked by this module (per NEXT_TASKS.md Week 8 skeleton):

- :class:`MomentOut` / :class:`MomentListOut` / :class:`MomentActionIn`
- :class:`YouOut` / :class:`PeopleListOut` / :class:`ContactDossierOut`
- :class:`ConnectorOut` / :class:`ConnectorConfigIn`
- :class:`HealthOut` / :class:`MetricsOut`

Route logic lands in Week 8 onward (``api/routes/*``); this module
intentionally has no side effects and no service imports.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core.moment.types import ActionKind, InsightType, MomentState

# Forbid unknown fields on every request/response: unexpected keys mean a
# drifted client or a schema bug, both of which should fail loudly.
_STRICT = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


class ActionOut(BaseModel):
    """A proposed side effect attached to a Moment.

    Mirrors :class:`core.moment.types.Action`. The per-``kind`` schema of
    ``params`` is validated by the outbox dispatcher (Week 11), not here —
    letting the field stay open avoids churning the schema every time a
    new action variant lands.
    """

    model_config = _STRICT

    kind: ActionKind
    params: dict = Field(default_factory=dict)


class StateHistoryOut(BaseModel):
    """One audit-log row of a Moment state transition.

    Mirrors :class:`core.moment.types.StateHistoryEntry`. Included on
    :class:`MomentOut` so a single ``GET /api/moments/{id}`` payload
    carries the full audit trail without a second round-trip.
    """

    model_config = _STRICT

    from_state: MomentState | None = None
    to_state: MomentState
    ts: int
    annotation: str | None = None


class MomentOut(BaseModel):
    """A Moment rendered for the API.

    Every field comes straight from :class:`core.moment.types.Moment`
    (Unix epoch timestamps preserved as integers so clients don't need
    to parse ISO strings). ``evidence`` is always a list (never ``None``)
    so clients never need to null-check.
    """

    model_config = _STRICT

    id: str
    created_at: int
    expires_at: int
    insight: str
    evidence: list[str] = Field(default_factory=list)
    evidence_hash: str
    proposed_action: ActionOut
    source_insight_type: InsightType
    state: MomentState = MomentState.SUGGESTED
    scheduled_for: int | None = None
    context_trigger: str | None = None
    snooze_until: int | None = None
    confidence: float = 0.0
    feedback_weight: float = 1.0
    state_history: list[StateHistoryOut] = Field(default_factory=list)


class MomentListOut(BaseModel):
    """The Now tab payload.

    Shape locked by engineering plan § "GET /api/now":
    ``{pending, scheduled, done}``. Limits 20/10/10 are enforced in the
    route, not the schema — a smaller list is valid, a larger one is
    not forbidden here (routes clamp).
    """

    model_config = _STRICT

    pending: list[MomentOut] = Field(default_factory=list)
    scheduled: list[MomentOut] = Field(default_factory=list)
    done: list[MomentOut] = Field(default_factory=list)


class MomentActionIn(BaseModel):
    """Request body for the four moment-action endpoints.

    A single shared schema with all fields optional lets the four
    endpoints (accept / dismiss / snooze / edit) share one type
    declaration. Per-endpoint validation (e.g. snooze requires
    ``snooze_until``) lives in the route handler so malformed requests
    return 422 with a clear message instead of silently ignoring the
    field.
    """

    model_config = _STRICT

    snooze_until: int | None = None
    action_params: dict | None = None
    annotation: str | None = None


# ---------------------------------------------------------------------------
# You tab + People tab
# ---------------------------------------------------------------------------


class RoutineOut(BaseModel):
    """One detected routine from the routine producer.

    Empty-state is carried as an explicit ``detected=False`` row so the
    You tab can render "No routine detected yet" without falling back to
    a null check on the parent list.
    """

    model_config = _STRICT

    name: str
    detected: bool = True
    description: str | None = None
    confidence: float = 0.0
    sample_size: int = 0


class PersonaStyleOut(BaseModel):
    """Per-audience writing summary for the You tab § "HOW YOU WRITE"."""

    model_config = _STRICT

    audience: str
    tone: str
    formality: float = 0.0
    sample_size: int = 0


class DriftingContactOut(BaseModel):
    """A contact whose cadence has slipped past the usual gap.

    Shape used by both the You tab ("DRIFTING" section) and the People
    tab ("NEEDS ATTENTION" section) — engineering plan § "Now / You /
    People payloads".
    """

    model_config = _STRICT

    contact_id: str
    name: str
    days_since_last: int
    usual_cadence_days: int


class YouOut(BaseModel):
    """Self-portrait payload for the You tab.

    Sections per DESIGN.md § "You tab":
    - ``when_at_best`` — top 3 patterns (strings)
    - ``how_you_write`` — per-audience style summaries
    - ``your_routines`` — detected routines (plus empty-state rows)
    - ``drifting`` — contacts with slipped cadence

    Header numbers (observed months, interactions, confidence) come
    from the insight/accuracy loops (Week 13 instrumentation); they
    default to 0 so empty installs still round-trip.
    """

    model_config = _STRICT

    observed_months: int = 0
    interactions_count: int = 0
    confidence_pct: int = 0
    when_at_best: list[str] = Field(default_factory=list)
    how_you_write: list[PersonaStyleOut] = Field(default_factory=list)
    your_routines: list[RoutineOut] = Field(default_factory=list)
    drifting: list[DriftingContactOut] = Field(default_factory=list)


class ContactSummaryOut(BaseModel):
    """A row of the People tab contact list.

    Right-aligned monospace stats in the UI (DESIGN.md § "People tab")
    correspond to ``last_contact_ts`` and ``cadence_deviation_days``.
    """

    model_config = _STRICT

    contact_id: str
    name: str
    last_contact_ts: int | None = None
    cadence_deviation_days: int | None = None
    needs_attention: bool = False


class PeopleListOut(BaseModel):
    """The People tab payload.

    ``you`` is the self-portrait pinned to the top of the list per
    DESIGN.md § "Always starts with YOU". ``needs_attention`` and
    ``active_this_week`` are the two sub-lists rendered below the
    pinned row.
    """

    model_config = _STRICT

    you: YouOut
    needs_attention: list[ContactSummaryOut] = Field(default_factory=list)
    active_this_week: list[ContactSummaryOut] = Field(default_factory=list)
    total: int = 0
    query: str | None = None


class ContactDossierOut(BaseModel):
    """Per-contact dossier payload (GET /api/people/{contact_id}).

    Structure matches DESIGN.md § "per-contact wireframe":
    - ``comm_template`` — the communication template the user typically
      uses with this contact (populated by the comm_template producer)
    - ``cadence_sparkline`` — per-day contact counts for the last N days
    - ``recent_topics`` — topic strings from recent episodes
    - ``predicted_next`` — forward-looking reminder text (from the
      relationship producer's next-contact prediction)
    """

    model_config = _STRICT

    contact_id: str
    name: str
    last_contact_ts: int | None = None
    usual_cadence_days: int | None = None
    comm_template: str | None = None
    cadence_sparkline: list[int] = Field(default_factory=list)
    recent_topics: list[str] = Field(default_factory=list)
    predicted_next: str | None = None


# ---------------------------------------------------------------------------
# Connectors (Settings tab)
# ---------------------------------------------------------------------------


class ConnectorOut(BaseModel):
    """A connector row for the settings page.

    Fernet-encrypted credentials are NEVER serialized here — the
    ``PATCH`` handler round-trips secrets via the preferences table
    and only returns masked values. Only status-level fields are
    exposed.
    """

    model_config = _STRICT

    id: str
    kind: str
    enabled: bool = False
    status: str = "unknown"
    last_sync_at: int | None = None
    last_error: str | None = None


class ConnectorConfigIn(BaseModel):
    """Request body for PATCH /api/connectors/{id}.

    Fields are intentionally permissive: per-connector config shape
    lives in ``connectors/registry.py``; the API only validates that
    the update is a dict with the right top-level keys. Secret fields
    are passed as plaintext on the wire (Tailscale-only in Phase 1)
    and re-encrypted server-side before storage.
    """

    model_config = _STRICT

    enabled: bool | None = None
    config: dict | None = None
    secrets: dict | None = None


# ---------------------------------------------------------------------------
# Health + Metrics
# ---------------------------------------------------------------------------


class HealthOut(BaseModel):
    """Deep-health response (GET /api/health).

    Multi-key by design — a single overall status hides the failure
    modes the engineering plan wants visible at a glance (connectors,
    DB last-write recency, scheduler heartbeat, producer activity per
    type, pending-moment count). A top-level ``ok`` summary is the
    AND of the component checks.
    """

    model_config = _STRICT

    ok: bool
    ts: int
    connectors: dict[str, str] = Field(default_factory=dict)
    db_last_write_ts: int | None = None
    scheduler_heartbeat_ts: int | None = None
    producer_activity: dict[str, int] = Field(default_factory=dict)
    pending_moments: int = 0
    notes: list[str] = Field(default_factory=list)


class MetricsOut(BaseModel):
    """Structured metrics sibling to the Prometheus text endpoint.

    The actual ``GET /metrics`` endpoint returns plain-text Prometheus
    exposition format; this schema is the typed form used by the daily
    JSONL dump to ``./data/metrics/metrics-YYYYMMDD.jsonl`` that the
    ``lifeos-report`` CLI ingests (see engineering plan § "metrics
    persistence").
    """

    model_config = _STRICT

    ts: int
    counters: dict[str, float] = Field(default_factory=dict)
    gauges: dict[str, float] = Field(default_factory=dict)
    histograms: dict[str, dict[str, float]] = Field(default_factory=dict)
