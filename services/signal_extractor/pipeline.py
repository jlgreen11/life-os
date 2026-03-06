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


# Maps profile names to the extractor class names that write them.
# Used by write verification to correlate extractor_hits with profile persistence.
_PROFILE_TO_EXTRACTOR: dict[str, list[str]] = {
    "linguistic": ["LinguisticExtractor"],
    "linguistic_inbound": ["LinguisticExtractor"],
    "cadence": ["CadenceExtractor"],
    "mood_signals": ["MoodInferenceEngine"],
    "relationships": ["RelationshipExtractor"],
    "topics": ["TopicExtractor"],
    "temporal": ["TemporalExtractor"],
    "spatial": ["SpatialExtractor"],
    "decision": ["DecisionExtractor"],
}


def _profile_extractor_hits(profile_name: str, extractor_hits: dict[str, int]) -> int:
    """Sum extractor hit counts for extractors that write the given profile.

    Args:
        profile_name: The signal profile name (e.g. 'linguistic').
        extractor_hits: Dict mapping extractor class names to hit counts.

    Returns:
        Total number of extractor hits relevant to this profile.
    """
    extractors = _PROFILE_TO_EXTRACTOR.get(profile_name, [])
    return sum(extractor_hits.get(ext, 0) for ext in extractors)


def _is_profile_stale(profile: dict) -> bool:
    """Determine whether an existing signal profile row has no useful data.

    A profile is considered stale when:
    - Its ``data`` field is falsy (None, empty dict)
    - Its ``data`` has no keys beyond common metadata (e.g., only ``updated_at``)
    - Its ``samples_count`` is below the minimum threshold (< 5)

    This catches rows that were created by a single event or schema migration
    but don't contain enough accumulated signal data to be useful.

    Args:
        profile: A dict as returned by ``UserModelStore.get_signal_profile()``,
            containing at minimum ``data`` and ``samples_count`` keys.

    Returns:
        True if the profile should be treated as missing and rebuilt.
    """
    # Metadata-only keys that don't count as real signal data.
    _METADATA_KEYS = {"updated_at", "created_at", "profile_type"}
    _MIN_SAMPLES = 5

    data = profile.get("data")
    samples = profile.get("samples_count", 0)

    # No data at all — stale.
    if not data:
        return True

    # Data exists but has no meaningful keys (only metadata or empty nested dicts).
    meaningful_keys = {k for k in data if k not in _METADATA_KEYS}
    if not meaningful_keys:
        return True

    # Data keys exist but all values are empty containers (e.g., {"averages": {}}).
    if all(isinstance(data[k], dict) and not data[k] for k in meaningful_keys):
        return True

    # Too few samples to be reliable.
    if samples < _MIN_SAMPLES:
        return True

    return False


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

        # Cached results from the last rebuild and profile health check,
        # accessible by diagnostics endpoints without re-querying.
        self._last_profile_health: dict | None = None
        self._last_rebuild_result: dict | None = None

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

    def get_profile_health(self) -> dict:
        """Return the health status of each of the 9 expected signal profiles.

        Queries every expected profile from the user model store and classifies
        it as 'ok', 'stale', or 'missing'. The result is cached on
        ``self._last_profile_health`` for later access by diagnostics endpoints.

        Returns:
            A dict mapping profile type names to dicts with keys:
            - ``status``: 'ok', 'stale', or 'missing'
            - ``samples``: sample count (0 if missing)
            - ``data_keys``: first 5 keys from the profile data dict
        """
        expected = [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
        result = {}
        for ptype in expected:
            try:
                profile = self.ums.get_signal_profile(ptype)
                if profile is None:
                    result[ptype] = {"status": "missing", "samples": 0, "data_keys": []}
                elif _is_profile_stale(profile):
                    result[ptype] = {
                        "status": "stale",
                        "samples": profile.get("samples_count", 0),
                        "data_keys": list((profile.get("data") or {}).keys())[:5],
                    }
                else:
                    result[ptype] = {
                        "status": "ok",
                        "samples": profile.get("samples_count", 0),
                        "data_keys": list((profile.get("data") or {}).keys())[:5],
                    }
            except Exception as e:
                # Fail-open: a single profile query failure doesn't block the rest.
                result[ptype] = {"status": "error", "samples": 0, "data_keys": [], "error": str(e)}

        self._last_profile_health = result
        return result

    def get_rebuild_diagnostics(self) -> dict:
        """Return the result of the last profile rebuild, including write failures.

        Returns the cached ``_last_rebuild_result`` dict from the most recent
        call to ``rebuild_profiles_from_events()``.  This includes:
        - ``events_processed``: number of events replayed
        - ``profiles_rebuilt``: list of extractor names that processed events
        - ``errors``: list of error description strings
        - ``write_failures``: dict of profiles where extractors ran but
          persistence failed (the key diagnostic for silent write failures)

        Returns:
            The last rebuild result dict, or a dict indicating no rebuild
            has been performed yet.
        """
        if self._last_rebuild_result is None:
            return {"status": "no_rebuild_performed"}
        return self._last_rebuild_result

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
            # Determine which profiles are missing or stale (exist but have
            # no useful data).  Stale profiles are treated identically to
            # missing ones — they need a full rebuild from historical events.
            missing = []
            for profile_type in expected_profiles:
                profile = self.ums.get_signal_profile(profile_type)
                if profile is None:
                    missing.append(profile_type)
                elif _is_profile_stale(profile):
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

            # Replay up to 50,000 recent events through all extractors,
            # filtering to only the event types the missing profiles need.
            # With 57K+ total events where ~55% are system.rule.triggered
            # (filtered out by type), 50K ensures we reach far enough back
            # to capture sufficient email/message/calendar events for all profiles.
            rebuild_result = self.rebuild_profiles_from_events(
                event_limit=50000, missing_profiles=missing,
            )

            # Determine which previously-missing/stale profiles now have
            # useful data.  Use the same stale check so a rebuild that only
            # produced a single sample isn't counted as success.
            rebuilt = []
            still_missing = []
            for profile_type in missing:
                profile = self.ums.get_signal_profile(profile_type)
                if profile is not None and not _is_profile_stale(profile):
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

            # Check for silent write failures: profiles where extractors ran
            # but update_signal_profile() silently failed to persist data.
            write_failures = rebuild_result.get("write_failures", {})
            if write_failures:
                logger.critical(
                    "check_and_rebuild_missing_profiles: %d profiles had extractor hits but FAILED "
                    "to persist — this indicates update_signal_profile() is silently failing. "
                    "Profiles: %s",
                    len(write_failures),
                    ", ".join(
                        f"{name} ({info['extractor_hits']} hits)"
                        for name, info in write_failures.items()
                    ),
                )

            # Log detailed per-profile health after rebuild for operator visibility.
            health = self.get_profile_health()
            for ptype, info in health.items():
                logger.info(
                    "  profile %-20s status=%-7s samples=%d keys=%s",
                    ptype, info["status"], info["samples"], info["data_keys"],
                )

            return {"missing_before": missing, "rebuilt": rebuilt, "skipped": False}

        except Exception as e:
            # Fail-open: never block startup due to profile rebuild failures.
            logger.warning("check_and_rebuild_missing_profiles: failed (non-fatal): %s", e, exc_info=True)
            return {"missing_before": [], "rebuilt": [], "skipped": True}

    def rebuild_profiles_from_events(
        self,
        event_limit: int = 50000,
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

        logger.info(
            "rebuild_profiles_from_events: fetched %d events (limit was %d)%s",
            len(rows), event_limit,
            " — limit reached, older relevant events may exist" if len(rows) >= event_limit else "",
        )

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
        # Count errors per extractor class to surface patterns without flooding logs.
        extractor_error_counts: dict[str, int] = {}

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
                        ename = type(extractor).__name__
                        extractor_error_counts[ename] = extractor_error_counts.get(ename, 0) + 1
                        # Cap detailed error messages to avoid 12K+ entry lists.
                        if len(errors) < 20:
                            errors.append(f"{ename} error on event {event.get('id', '?')}: {e}")

            events_processed += 1

            # Progress logging every 500 events.
            if (i + 1) % 500 == 0:
                logger.info("rebuild_profiles_from_events: processed %d / %d events", i + 1, len(rows))

        profiles_rebuilt = sorted(extractor_hits.keys())
        logger.info(
            "rebuild_profiles_from_events: done — %d events processed, %d profiles rebuilt (%s), %d errors",
            events_processed, len(profiles_rebuilt), ", ".join(profiles_rebuilt) or "none", len(errors),
        )

        # Log per-extractor error counts for operators diagnosing rebuild failures.
        if extractor_error_counts:
            logger.info(
                "rebuild_profiles_from_events: error counts by extractor: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(extractor_error_counts.items())),
            )

        # -- Write verification phase --
        # Check that profiles which had extractor hits were actually persisted.
        # update_signal_profile() wraps its INSERT in try/except and silently
        # logs a warning on failure, so extractor_hits can show thousands of
        # "processed" events while zero data actually reached the DB.
        write_failures: dict[str, dict] = {}
        for profile_name in PROFILE_EVENT_TYPES:
            if missing_profiles and profile_name not in missing_profiles:
                continue
            # Check if any extractor that writes this profile had hits.
            # Use the extractor→profile mapping to correlate.
            profile_extractor_hits = _profile_extractor_hits(profile_name, extractor_hits)
            if profile_extractor_hits == 0:
                continue
            # Verify the profile was actually written to the DB.
            try:
                profile = self.ums.get_signal_profile(profile_name)
                profile_exists = profile is not None
                samples_count = profile.get("samples_count", 0) if profile else 0
                if not profile_exists or samples_count == 0:
                    write_failures[profile_name] = {
                        "extractor_hits": profile_extractor_hits,
                        "profile_exists": profile_exists,
                        "samples_count": samples_count,
                    }
            except Exception as e:
                write_failures[profile_name] = {
                    "extractor_hits": profile_extractor_hits,
                    "profile_exists": False,
                    "samples_count": 0,
                    "error": str(e),
                }

        if write_failures:
            logger.critical(
                "rebuild_profiles_from_events: WRITE FAILURES DETECTED — %d profiles had extractor hits "
                "but no data persisted: %s",
                len(write_failures),
                ", ".join(
                    f"{name} ({info['extractor_hits']} hits, exists={info['profile_exists']})"
                    for name, info in write_failures.items()
                ),
            )

        result = {
            "events_processed": events_processed,
            "profiles_rebuilt": profiles_rebuilt,
            "errors": errors,
            "extractor_error_counts": extractor_error_counts,
            "write_failures": write_failures,
        }
        self._last_rebuild_result = result
        return result

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

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the signal extraction pipeline.

        Reports per-profile status, available event types for rebuild,
        extractor readiness, and overall health so operators can understand
        why profiles are or aren't being populated.

        Each section is queried independently with try/except so that a single
        DB failure doesn't prevent the rest of the diagnostics from returning.
        """
        result: dict = {}

        # 1. Profile status — which profiles exist and their sample counts
        try:
            expected = [
                "linguistic", "linguistic_inbound", "cadence", "mood_signals",
                "relationships", "topics", "temporal", "spatial", "decision",
            ]
            profiles = {}
            for pt in expected:
                profile = self.ums.get_signal_profile(pt)
                if profile is None:
                    profiles[pt] = {"status": "missing"}
                elif _is_profile_stale(profile):
                    profiles[pt] = {
                        "status": "stale",
                        "samples_count": profile.get("samples_count", 0),
                        "updated_at": profile.get("updated_at"),
                    }
                else:
                    profiles[pt] = {
                        "status": "ok",
                        "samples_count": profile.get("samples_count", 0),
                        "updated_at": profile.get("updated_at"),
                    }
            result["profiles"] = profiles
            result["profiles_present"] = sum(1 for v in profiles.values() if v["status"] == "ok")
            result["profiles_missing"] = sum(
                1 for v in profiles.values() if v["status"] in ("missing", "stale")
            )
        except Exception as e:
            result["profiles"] = {"error": str(e)}

        # 2. Available event types for rebuild — what's in events.db.
        # Cross-reference with PROFILE_EVENT_TYPES to show which missing
        # profiles COULD be rebuilt from available event data.
        try:
            with self.db.get_connection("events") as conn:
                rows = conn.execute(
                    "SELECT type, COUNT(*) as cnt FROM events "
                    "WHERE type NOT LIKE 'test%' AND type NOT LIKE 'system.rule%' "
                    "GROUP BY type ORDER BY cnt DESC LIMIT 30"
                ).fetchall()
            event_counts = {r["type"]: r["cnt"] for r in rows}
            result["available_event_types"] = event_counts

            # For each missing profile, check if qualifying events exist.
            rebuild_feasibility: dict = {}
            for pt, status in result.get("profiles", {}).items():
                if isinstance(status, dict) and status.get("status") in ("missing", "stale"):
                    needed_types = PROFILE_EVENT_TYPES.get(pt, [])
                    available = sum(event_counts.get(t, 0) for t in needed_types)
                    rebuild_feasibility[pt] = {
                        "needed_event_types": needed_types,
                        "available_events": available,
                        "can_rebuild": available > 0,
                    }
            result["rebuild_feasibility"] = rebuild_feasibility
        except Exception as e:
            result["available_event_types"] = {"error": str(e)}

        # 3. Extractor registration — list all registered extractors
        try:
            result["extractors"] = [
                type(ext).__name__ for ext in self.extractors
            ]
            result["extractor_count"] = len(self.extractors)
        except Exception as e:
            result["extractors"] = {"error": str(e)}

        # 4. Overall health
        try:
            present = result.get("profiles_present", 0)
            missing = result.get("profiles_missing", 0)
            if missing == 0:
                health = "ok"
            elif present >= 2:
                health = "partial"
            else:
                health = "degraded"
            result["health"] = health
        except Exception:
            result["health"] = "unknown"

        return result
