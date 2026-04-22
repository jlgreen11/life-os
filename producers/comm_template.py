"""Comm-template producer — drafts a reply when a known contact pings.

The fifth of the six Phase 1 producers. Where :mod:`producers.cadence`
and :mod:`producers.relationship` watch *people-shaped* signals,
:mod:`producers.temporal` and :mod:`producers.spatial` watch
*context-shaped* signals, the comm-template producer watches
*conversational* signals: an inbound message from a contact the system
has a known communication style for, and proposes a tentative draft so
the user can accept-and-edit instead of reply-from-blank.

Trigger
-------
Every ``email.received`` and ``message.received`` event wakes the
producer. Unlike cadence/relationship — which scan **all** profiles on
every inbound to detect time-shaped drift — this producer cares about
the *specific* contact who just sent something. It looks them up by
``payload.contact_id`` and decides on the spot.

That makes the gating cheap: one keyed read per inbound event, no
table scan. The cost is that the connector layer must populate
``contact_id`` on the payload (the v2 normalisation contract). Events
without a resolvable ``contact_id`` short-circuit to ``[]`` rather
than raise — fail-open per the v1 convention preserved in v2.

Profile shape
-------------
Profiles live in ``signal_profiles`` under
``producer='comm_template'`` with one row per known contact keyed by
``contact_id``. A reasonable profile:

.. code-block:: json

    {
        "contact_name": "Alice",
        "channel": "email",
        "template_style": "casual",
        "last_event_ids": ["evt-a", "evt-b", "evt-c"]
    }

- ``contact_name`` is the display name used in microcopy and in the
  stub draft greeting.
- ``channel`` (``email`` / ``imessage`` / ``signal`` / ...) is echoed
  on the proposed action so the dispatcher can route the eventual
  outbound through the right connector.
- ``template_style`` is *unused by the Week 5 stub*. It is recorded
  here as the hand-off surface to the Week 7 AI engine, which will
  read it to bias :meth:`ai.engine.AIEngine.draft_reply` toward the
  user's per-contact tone.
- ``last_event_ids`` is the most-recent inbound event ids (newest
  first) used for the UI evidence list.

A future comm-template signal extractor owns *writing* this row; the
producer only reads it. Missing fields silently skip rather than
raise.

Stub draft generation
---------------------
Per the Week 5 task body: *"scaffold + stub draft generation
(deterministic ``Hi {name},``); mark NOTE: AI engine integration
deferred to Week 7"*. The draft is literally ``"Hi {contact_name},"``
with no body — Week 7 wires real generation via
:meth:`ai.engine.AIEngine.draft_reply`. The stub is intentionally
plain so reviewers can tell at a glance that the AI integration has
not yet landed.

The microcopy presented on the Moment card is
``"Reply to {Name}? Draft ready."`` — the draft itself rides on
``proposed_action.params.draft`` so the UI can render it in the
recessed-bg block called out in DESIGN.md.

Confidence is a fixed ``STUB_CONFIDENCE = 0.5`` for the Week 5 stub.
The Week 7 AI engine will compute a real confidence (e.g. perplexity
floor on the generated draft); for now the fixed mid-band value lets
the Week 6 feedback-weight EWMA either promote or demote the producer
based on user accept/dismiss without the producer pretending to know.

Idempotency per inbound event
-----------------------------
One Moment per (inbound event, contact) pair. The ``evidence_hash``
basis is ``[event_id, contact_id]``:

- A retry of the same inbound event for the same contact yields the
  same hash; the ``UNIQUE (source_insight_type, evidence_hash)``
  constraint on ``moments`` collapses the second insert to the first
  id.
- A fresh inbound event for the same contact yields a fresh hash, so
  every new ping gets its own draft Moment.

The ``evidence`` list stored on the Moment is the profile's
``last_event_ids`` (up to 3) so the UI surfaces the recent
conversational thread. If ``last_event_ids`` is missing, the inbound
event id alone is the evidence.

References
----------
- CEO plan § "The Moment Primitive (producers)":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 5: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- Sibling producers :mod:`producers.cadence`,
  :mod:`producers.relationship` for the inbound-event trigger and
  fail-open conventions.

.. note::

    AI engine integration deferred to Week 7. The
    ``STUB_CONFIDENCE`` constant and ``_render_stub_draft`` helper
    here are placeholders that Week 7's
    :class:`ai.engine.AIEngine` will replace with real
    ``draft_reply()`` calls. Tests assert the stub explicitly so
    Week 7 has a checklist of what to swap out.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import Any

from core.moment.producer import Event, Producer, register
from core.moment.types import Action, ActionKind, InsightType, Moment

logger = logging.getLogger(__name__)


COMM_TEMPLATE_PRODUCER_KEY = "comm_template"
"""``signal_profiles.producer`` value owned by the comm-template producer."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

INBOUND_EVENT_TYPES: frozenset[str] = frozenset({"email.received", "message.received"})
"""Event types this producer cares about. All others short-circuit to ``[]``."""

STUB_CONFIDENCE = 0.5
"""Fixed confidence for the Week 5 stub. Week 7 AI engine will replace this."""


def _render_stub_draft(contact_name: str) -> str:
    """Return the Week 5 deterministic stub draft.

    Literally ``"Hi {contact_name},"``. Kept as a named helper so the
    Week 7 AI engine integration has a single, greppable swap point —
    delete this function, route through
    :meth:`ai.engine.AIEngine.draft_reply` instead.
    """
    return f"Hi {contact_name},"


@register
class CommTemplateProducer(Producer):
    """Emits a draft-reply Moment when a known contact sends an inbound.

    Constructor takes a ``sqlite3.Connection`` that points at the v2
    ``lifeos.db``. The producer never writes — the (future)
    comm-template signal extractor owns profile maintenance — so the
    connection's transaction settings are left alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.COMM_TEMPLATE

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so
        column access in :meth:`_read_profile` is by name. This matches
        the sibling producers' convention; sharing a connection across
        producers is safe because none of them write.
        """
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._id_fn: Callable[[], str] = id_fn or (lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Producer contract
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> list[Moment]:
        """Return zero or one Moments for an inbound from a known contact.

        Non-inbound events short-circuit before any DB access. Inbound
        events without a resolvable ``contact_id`` or ``id`` also
        skip; an unknown contact (no profile row) skips silently.
        """
        if event.get("type") not in INBOUND_EVENT_TYPES:
            return []
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            return []
        contact_id = payload.get("contact_id")
        if not isinstance(contact_id, str) or not contact_id:
            return []
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id:
            return []
        profile = self._read_profile(contact_id)
        if profile is None:
            return []
        moment = self._build_moment(event_id, contact_id, profile)
        return [moment] if moment is not None else []

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _read_profile(self, contact_id: str) -> dict[str, Any] | None:
        """Return the profile dict for ``contact_id`` or ``None``.

        A missing row returns ``None`` silently — that is the
        "unknown contact" path the task body calls out. Malformed
        JSON or a non-object top-level also returns ``None`` with a
        logged warning, matching sibling producers' fail-open posture.
        """
        row = self._conn.execute(
            "SELECT profile FROM signal_profiles WHERE producer=? AND key=?",
            (COMM_TEMPLATE_PRODUCER_KEY, contact_id),
        ).fetchone()
        if row is None:
            return None
        try:
            profile = json.loads(row["profile"])
        except (TypeError, ValueError):
            logger.warning(
                "comm_template profile for key=%r is not valid JSON; skipping",
                contact_id,
            )
            return None
        if not isinstance(profile, dict):
            logger.warning(
                "comm_template profile for key=%r is not a JSON object; skipping",
                contact_id,
            )
            return None
        return profile

    def _build_moment(
        self,
        event_id: str,
        contact_id: str,
        profile: dict[str, Any],
    ) -> Moment | None:
        """Return a Moment proposing a stub draft reply.

        Returns ``None`` only if the contact name resolves to an empty
        string after fallback — defensive, since profile-write paths
        should always populate one of ``contact_name`` or the key.
        """
        contact_name = str(profile.get("contact_name") or contact_id).strip()
        if not contact_name:
            return None

        channel = str(profile.get("channel") or "")
        evidence = self._build_evidence(profile, event_id)
        draft = _render_stub_draft(contact_name)

        now = int(self._now_fn())
        insight = f"Reply to {contact_name}? Draft ready."

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            # Hash basis is (event_id, contact_id) so a retry of the
            # same inbound for the same contact collapses; a new
            # inbound yields a new hash. See module docstring
            # "Idempotency per inbound event".
            evidence_hash=self.evidence_hash([event_id, contact_id]),
            proposed_action=Action(
                kind=ActionKind.DRAFT_MESSAGE,
                params={
                    "contact_id": contact_id,
                    "channel": channel,
                    "draft": draft,
                    "in_reply_to_event_id": event_id,
                },
            ),
            source_insight_type=InsightType.COMM_TEMPLATE,
            evidence=evidence,
            confidence=STUB_CONFIDENCE,
        )

    @staticmethod
    def _build_evidence(profile: dict[str, Any], event_id: str) -> list[str]:
        """Return up to 3 prior event ids, else just the trigger event id.

        The trigger event id is always a valid evidence pointer, so an
        empty/missing ``last_event_ids`` falls back to a single-element
        list rather than the empty-evidence sentinel.
        """
        raw = profile.get("last_event_ids")
        if isinstance(raw, list):
            evidence = [str(e) for e in raw[:3] if e]
            if evidence:
                return evidence
        return [event_id]


__all__ = [
    "COMM_TEMPLATE_PRODUCER_KEY",
    "DEFAULT_EXPIRY_SECONDS",
    "INBOUND_EVENT_TYPES",
    "STUB_CONFIDENCE",
    "CommTemplateProducer",
]
