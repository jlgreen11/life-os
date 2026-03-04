"""
Life OS — Signal Extractor Pipeline

The main pipeline that routes events through all extractors.
Subscribes to the NATS event bus and processes every event.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from models.user_model import MoodState
from storage.database import DatabaseManager, UserModelStore

logger = logging.getLogger(__name__)

# Each extractor is responsible for one behavioral dimension.  The pipeline
# instantiates all of them and fans every incoming event out to each one that
# declares interest via its `can_process` method.
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor
from services.signal_extractor.temporal import TemporalExtractor
from services.signal_extractor.spatial import SpatialExtractor
from services.signal_extractor.decision import DecisionExtractor
from services.signal_extractor.base import BaseExtractor

# Maps each signal profile name to the event types its extractor actually
# processes (derived from each extractor's can_process() method).  Used by
# rebuild_profiles_from_events() to narrow the SQL query to only the event
# types that are relevant to the missing profiles, avoiding wasted budget
# on system.rule.triggered and other internal events that no extractor handles.
PROFILE_EVENT_TYPES: dict[str, list[str]] = {
    "linguistic": ["email.sent", "message.sent", "system.user.command"],
    "linguistic_inbound": ["email.received", "message.received"],
    "cadence": ["email.sent", "message.sent", "email.received", "message.received"],
    "mood_signals": [
        "email.received", "email.sent", "message.received", "message.sent",
        "health.metric.updated", "health.sleep.recorded",
        "calendar.event.created", "finance.transaction.new",
        "location.changed", "system.user.command",
    ],
    "relationships": ["email.received", "email.sent", "message.received", "message.sent"],
    "temporal": [
        "email.sent", "message.sent",
        "calendar.event.created", "calendar.event.updated",
        "task.created", "task.completed", "task.updated",
        "system.user.command",
    ],
    "topics": ["email.received", "email.sent", "message.received", "message.sent", "system.user.command"],
    "spatial": ["calendar.event.created", "ios.context.update", "system.user.location_update"],
    "decision": ["task.completed", "task.created", "email.sent", "message.sent", "calendar.event.created"],
}


class SignalExtractorPipeline:
    """
    The main pipeline that routes events through all extractors.
    Subscribes to the NATS event bus and processes every event.
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        self.db = db
        self.ums = user_model_store

        # Register every extractor that participates in the pipeline.  Each one
        # receives the shared database handle and user-model store so it can
        # both read historical context and persist the signals it produces.
        self.extractors: list[BaseExtractor] = [
            LinguisticExtractor(db, user_model_store),
            CadenceExtractor(db, user_model_store),
            MoodInferenceEngine(db, user_model_store),
            RelationshipExtractor(db, user_model_store),
            TopicExtractor(db, user_model_store),
            TemporalExtractor(db, user_model_store),
            SpatialExtractor(db, user_model_store),
            DecisionExtractor(db, user_model_store),
        ]

        # Keep a direct reference to the mood engine already in the extractor
        # list so we can call compute_current_mood() on demand without iterating.
        # Using the SAME instance (not a new one) ensures that any in-memory
        # state updated during extract() is visible to get_current_mood().
        self.mood_engine = next(e for e in self.extractors if isinstance(e, MoodInferenceEngine))

    async def process_event(self, event: dict) -> list[dict]:
        """
        Process an event through all applicable extractors.
        Returns all extracted signals.

        This is the main entry point called by the NATS event-bus subscriber.
        Each incoming event is offered to every registered extractor; the
        extractor's `can_process` gate decides whether it should run.  Signals
        produced by all interested extractors are collected and returned as a
        flat list so callers can forward them downstream (e.g., to the user
        model aggregator or an analytics sink).
        """
        all_signals = []

        for extractor in self.extractors:
            # Route: let each extractor decide if this event type is relevant.
            if extractor.can_process(event):
                try:
                    # extract() both returns signals AND persists them into the
                    # extractor's own profile store as a side-effect.
                    signals = extractor.extract(event)
                    all_signals.extend(signals)
                except Exception as e:
                    # Fail-open: signal extraction must never block or crash the
                    # main event processing loop.  Log and continue.
                    logger.error("Extractor %s error: %s", type(extractor).__name__, e, exc_info=True)

        return all_signals

    def check_and_rebuild_missing_profiles(self) -> dict:
        """Check for missing signal profiles and rebuild them from historical events if needed.

        Defines the full set of expected profile types and queries each one.
        If any are missing AND events exist in events.db, triggers a full
        rebuild via rebuild_profiles_from_events().  Returns immediately if
        all profiles are present — no unnecessary event replay.

        This method is designed to run once at startup as a self-healing
        mechanism after data loss, schema migrations, or extractor additions.
        The entire method is wrapped in try/except so a failure never blocks
        startup.

        Returns:
            A dict with keys:
            - ``missing_before``: list of profile types that were absent
            - ``rebuilt``: list of profile types that were successfully rebuilt
            - ``skipped``: bool, True if rebuild was skipped (no events or no missing profiles)
        """
        expected_profiles = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]

        try:
            # Determine which profiles are missing.
            missing = []
            for profile_type in expected_profiles:
                profile = self.ums.get_signal_profile(profile_type)
                if profile is None:
                    missing.append(profile_type)

            if not missing:
                logger.info("check_and_rebuild_missing_profiles: all expected signal profiles present")
                return {"missing_before": [], "rebuilt": [], "skipped": True}

            # Check whether there are events to replay.
            with self.db.get_connection("events") as conn:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
                event_count = row["cnt"] if row else 0

            if event_count == 0:
                logger.info(
                    "check_and_rebuild_missing_profiles: %d profiles missing (%s) but no events in events.db — skipping rebuild",
                    len(missing), ", ".join(missing),
                )
                return {"missing_before": missing, "rebuilt": [], "skipped": True}

            logger.info(
                "check_and_rebuild_missing_profiles: %d profiles missing (%s), %d events available — starting rebuild",
                len(missing), ", ".join(missing), event_count,
            )

            # Replay up to 10,000 recent events through all extractors,
            # filtering to only the event types the missing profiles need.
            rebuild_result = self.rebuild_profiles_from_events(
                event_limit=10000, missing_profiles=missing,
            )

            # Determine which previously-missing profiles now exist.
            rebuilt = []
            still_missing = []
            for profile_type in missing:
                profile = self.ums.get_signal_profile(profile_type)
                if profile is not None:
                    rebuilt.append(profile_type)
                else:
                    still_missing.append(profile_type)

            logger.info(
                "check_and_rebuild_missing_profiles: rebuilt %d profiles (%s), %d still missing (%s), %d events processed, %d errors",
                len(rebuilt), ", ".join(rebuilt) or "none",
                len(still_missing), ", ".join(still_missing) or "none",
                rebuild_result.get("events_processed", 0),
                len(rebuild_result.get("errors", [])),
            )

            return {"missing_before": missing, "rebuilt": rebuilt, "skipped": False}

        except Exception as e:
            # Fail-open: never block startup due to profile rebuild failures.
            logger.warning("check_and_rebuild_missing_profiles: failed (non-fatal): %s", e, exc_info=True)
            return {"missing_before": [], "rebuilt": [], "skipped": True}

    def rebuild_profiles_from_events(
        self,
        event_limit: int = 5000,
        missing_profiles: list[str] | None = None,
    ) -> dict:
        """Rebuild signal profiles by replaying historical events from events.db.

        After database corruption and repair, signal profiles may be permanently
        lost because the normal pipeline only processes NEW events from NATS.
        This method replays stored events through the extractors to reconstruct
        all signal profiles (linguistic, cadence, mood, relationships, topics,
        temporal, spatial, decision).

        When ``missing_profiles`` is provided, the SQL query is narrowed to only
        the event types that the corresponding extractors can process.  This
        avoids wasting the event budget on types like ``system.rule.triggered``
        (57 % of all events) that no extractor handles, ensuring the replay
        window reaches further back into relevant history.

        Events are loaded in reverse-chronological order from events.db and then
        processed in chronological order so that profiles accumulate correctly
        (e.g., response-time calculations need earlier events first).

        Each extractor call is wrapped in try/except following the fail-open
        pattern: a failure in one extractor does not block others from processing
        the same event.

        Args:
            event_limit: Maximum number of recent events to replay.  Defaults to
                5000 which covers roughly 2-3 days of typical activity.
            missing_profiles: Optional list of profile names (e.g. ``["cadence",
                "spatial"]``) whose required event types are used to filter the
                query.  When ``None`` or empty, all event types are fetched
                (original behaviour).

        Returns:
            A stats dict with keys ``events_processed``, ``profiles_rebuilt``
            (list of extractor names that successfully processed at least one
            event), and ``errors`` (list of error description strings).
        """
        logger.info("rebuild_profiles_from_events: loading up to %d events from events.db", event_limit)

        # When missing_profiles is specified, compute the union of event types
        # needed by those profiles and filter the query accordingly.
        type_filter_values: list[str] = []
        if missing_profiles:
            type_set: set[str] = set()
            for profile_name in missing_profiles:
                type_set.update(PROFILE_EVENT_TYPES.get(profile_name, []))
            type_filter_values = sorted(type_set)

        if type_filter_values:
            logger.info(
                "rebuild_profiles_from_events: filtering to %d event types for %d missing profiles",
                len(type_filter_values), len(missing_profiles or []),
            )

        # Load recent events from events.db (DESC so we get the most recent first).
        with self.db.get_connection("events") as conn:
            if type_filter_values:
                placeholders = ", ".join("?" for _ in type_filter_values)
                rows = conn.execute(
                    "SELECT id, type, source, timestamp, priority, payload, metadata "
                    f"FROM events WHERE type IN ({placeholders}) "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (*type_filter_values, event_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, type, source, timestamp, priority, payload, metadata "
                    "FROM events ORDER BY timestamp DESC LIMIT ?",
                    (event_limit,),
                ).fetchall()

        if not rows:
            logger.info("rebuild_profiles_from_events: no events found in events.db")
            return {"events_processed": 0, "profiles_rebuilt": [], "errors": []}

        # Reverse to chronological order so profiles accumulate correctly.
        rows = list(reversed(rows))

        logger.info("rebuild_profiles_from_events: replaying %d events through %d extractors",
                     len(rows), len(self.extractors))

        events_processed = 0
        # Track which extractors successfully processed at least one event.
        extractor_hits: dict[str, int] = {}
        errors: list[str] = []

        for i, row in enumerate(rows):
            # Deserialize the row into the event dict format extractors expect.
            try:
                event = {
                    "id": row["id"],
                    "type": row["type"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "priority": row["priority"],
                    "payload": json.loads(row["payload"]) if row["payload"] else {},
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
            except (json.JSONDecodeError, KeyError) as e:
                errors.append(f"Event row deserialization error at index {i}: {e}")
                continue

            # Fan out to every extractor, just like process_event() does.
            for extractor in self.extractors:
                if extractor.can_process(event):
                    try:
                        extractor.extract(event)
                        name = type(extractor).__name__
                        extractor_hits[name] = extractor_hits.get(name, 0) + 1
                    except Exception as e:
                        # Fail-open: log and continue with next extractor.
                        errors.append(f"{type(extractor).__name__} error on event {event.get('id', '?')}: {e}")

            events_processed += 1

            # Progress logging every 500 events.
            if (i + 1) % 500 == 0:
                logger.info("rebuild_profiles_from_events: processed %d / %d events", i + 1, len(rows))

        profiles_rebuilt = sorted(extractor_hits.keys())
        logger.info(
            "rebuild_profiles_from_events: done — %d events processed, %d profiles rebuilt (%s), %d errors",
            events_processed, len(profiles_rebuilt), ", ".join(profiles_rebuilt) or "none", len(errors),
        )

        return {
            "events_processed": events_processed,
            "profiles_rebuilt": profiles_rebuilt,
            "errors": errors,
        }

    def get_current_mood(self) -> MoodState:
        """Get the current mood estimate and persist it to the mood history.

        Delegates to the dedicated MoodInferenceEngine which aggregates recent
        mood signals (sleep, language sentiment, calendar density, etc.) into a
        multi-dimensional MoodState.  Called on-demand by the orchestrator or
        periodically (every ~15 minutes) to keep the mood snapshot fresh.

        **Persistence side-effect:** After computing the mood, this method
        stores the snapshot in the ``mood_history`` time-series table so that
        ``MoodInferenceEngine._compute_trend()`` has historical data to compare
        against on the next call.  Without this write, trend detection always
        falls back to "stable" because the history table remains empty.

        This is the only place that writes to ``mood_history`` during normal
        operation.  The write is best-effort: failures are logged and silently
        ignored so that mood history never crashes the event loop.
        """
        mood = self.mood_engine.compute_current_mood()

        # Persist the snapshot to mood_history so trend computation has data.
        # Only write when there are actual signals (confidence > 0) to avoid
        # flooding the history table with identical "no data" neutral entries
        # that would obscure real trend signals.
        if mood.confidence > 0:
            try:
                self.ums.store_mood({
                    "energy_level": mood.energy_level,
                    "stress_level": mood.stress_level,
                    "social_battery": mood.social_battery,
                    "cognitive_load": mood.cognitive_load,
                    "emotional_valence": mood.emotional_valence,
                    "confidence": mood.confidence,
                    "trend": mood.trend,
                    "contributing_signals": [
                        {
                            "signal_type": s.signal_type,
                            "value": s.value,
                            "weight": s.weight,
                        }
                        for s in (mood.contributing_signals or [])
                    ],
                })
            except Exception as e:
                # Fail-open: mood history persistence must never crash the
                # event loop.  The mood state is still returned to the caller.
                logger.warning("Failed to persist mood snapshot to history: %s", e)

        return mood

    def get_user_summary(self) -> dict:
        """Get a summary of what we know about the user.

        Collects metadata from every signal profile (linguistic, cadence, mood,
        relationships, topics) and merges it with high-confidence semantic facts
        from the user-model store.  The result is a lightweight snapshot
        suitable for the orchestrator's system prompt or for a debugging UI.

        Returns a degraded response with empty data if user_model.db is
        corrupted or unavailable, rather than propagating the exception to
        callers (briefing generation, web API routes).
        """
        try:
            # Gather per-dimension profile summaries (sample counts and freshness).
            profiles = {}
            for profile_type in ["linguistic", "cadence", "mood_signals", "relationships", "topics", "temporal", "spatial", "decision"]:
                profile = self.ums.get_signal_profile(profile_type)
                if profile:
                    profiles[profile_type] = {
                        "samples_count": profile["samples_count"],
                        "last_updated": profile["updated_at"],
                    }

            # Pull semantic facts that have accumulated enough confidence (>= 0.3)
            # to be worth mentioning, then surface only the high-confidence ones
            # (>= 0.7) in the summary payload so the orchestrator can rely on them.
            facts = self.ums.get_semantic_facts(min_confidence=0.3)

            return {
                "profiles": profiles,
                "semantic_facts_count": len(facts),
                "high_confidence_facts": [
                    {"key": f["key"], "value": f["value"], "confidence": f["confidence"]}
                    for f in facts
                    if f["confidence"] >= 0.7
                ],
            }
        except (sqlite3.DatabaseError, Exception) as e:
            # Fail-open: user_model.db corruption must never crash briefing
            # generation or web API routes.  Return an empty but valid summary
            # with a degraded flag so callers can detect the fallback.
            logger.warning(
                "get_user_summary: user_model.db unavailable (%s), returning empty summary", e
            )
            return {
                "profiles": {},
                "semantic_facts_count": 0,
                "high_confidence_facts": [],
                "degraded": True,
            }
