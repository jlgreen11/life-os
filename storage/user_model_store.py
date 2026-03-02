"""
Life OS — User Model Store

High-level operations on the user model database:
episodes, semantic facts, signal profiles, mood, predictions,
and communication templates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


class UserModelStore:
    """High-level operations on the user model database.

    Provides CRUD helpers for the three memory layers (episodic, semantic,
    procedural) plus mood tracking, signal profiles, and predictions.
    """

    def __init__(self, db: DatabaseManager, event_bus: Any = None, event_store: Any = None):
        """Initialize UserModelStore with database and optional event bus/store.

        Args:
            db: DatabaseManager for user_model.db access
            event_bus: Optional EventBus for real-time telemetry (requires NATS)
            event_store: Optional EventStore for persisting telemetry events directly
                        to events.db when event bus is unavailable
        """
        self.db = db
        self._event_bus = event_bus
        self._event_store = event_store

    def _emit_telemetry(self, event_type: str, payload: dict):
        """Fire-and-forget telemetry event publication.

        CRITICAL FIX (iteration 143):
            The previous implementation had 100% telemetry loss when NATS was not
            running. This broke ALL observability: 340K+ predictions generated but
            NO telemetry events published.

            Root causes:
            1. Telemetry required event bus to be connected (is_connected check)
            2. When NATS isn't running, is_connected=False and all telemetry was
               silently skipped
            3. The try/except RuntimeError: pass pattern masked async context issues

            Fix: Dual-path telemetry with automatic fallback:
            - PRIMARY: Publish to event bus (real-time, requires NATS)
            - FALLBACK: Write directly to event store (events.db, always works)

            This ensures telemetry ALWAYS succeeds whether or not NATS is running,
            enabling observability in all deployment modes (Docker, local dev, tests).
        """
        # If we have an event bus AND it's connected, try publishing through it
        # for real-time event distribution. If it fails or isn't available, we'll
        # fall back to direct event store writes.
        published_via_bus = False

        if self._event_bus and self._event_bus.is_connected:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._event_bus.publish(event_type, payload, source="user_model_store")
                )
                published_via_bus = True
            except RuntimeError:
                # No running event loop in this context. Fall through to event
                # store fallback below.
                pass
            except Exception:
                # Any other error (connection lost, etc). Fall through to fallback.
                pass

        # FALLBACK: If event bus publish failed or wasn't available, write the
        # telemetry event directly to events.db so it's still captured for
        # analytics and observability.
        if not published_via_bus and self._event_store:
            try:
                # Create a proper event envelope matching the Event model format
                from datetime import datetime, timezone
                import uuid

                event = {
                    "id": str(uuid.uuid4()),
                    "type": event_type,
                    "source": "user_model_store",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "priority": "normal",
                    "payload": payload,
                    "metadata": {"telemetry": True},
                }
                self._event_store.store_event(event)
            except Exception as e:
                # Last resort: if both event bus AND event store fail, log the
                # error so we know telemetry is completely broken.
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to publish telemetry event {event_type} via both "
                    f"event bus and event store: {e}"
                )

    def store_episode(self, episode: dict):
        """Store an episodic memory.

        Uses INSERT OR REPLACE so that re-processing the same event (same ``id``)
        overwrites the previous episode rather than raising a uniqueness error.
        This makes the operation idempotent — safe to retry on failure.

        Multi-value fields (contacts_involved, topics, entities) and structured
        fields (inferred_mood) are serialized to JSON for storage.

        CRITICAL FIX (iteration 150):
        Fixed inferred_mood serialization bug where dict.get("inferred_mood", {})
        would return None instead of {} when the key exists with None value,
        causing json.dumps(None) to store the string "null" instead of null.
        This broke ALL episode mood tracking (31K+ episodes with null mood
        despite 29K+ mood signals available). Now explicitly checks for None
        and uses {} as fallback, ensuring proper JSON serialization.
        """
        with self.db.get_connection("user_model") as conn:
            # INSERT OR REPLACE: if a row with the same PRIMARY KEY (id) already
            # exists, SQLite deletes it and inserts the new row.  This is the
            # idempotent upsert strategy used throughout the episodic layer.

            # CRITICAL: Handle None values explicitly for JSON serialization.
            # dict.get(key, default) returns the actual value (even if None)
            # when the key exists, NOT the default. So we must check explicitly.
            inferred_mood = episode.get("inferred_mood") or {}
            contacts_involved = episode.get("contacts_involved") or []
            topics = episode.get("topics") or []
            entities = episode.get("entities") or []

            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, timestamp, event_id, location, inferred_mood, active_domain,
                    energy_level, interaction_type, content_summary, content_full,
                    contacts_involved, topics, entities, outcome, user_satisfaction, embedding_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode["id"],
                    episode["timestamp"],
                    episode["event_id"],
                    episode.get("location"),
                    json.dumps(inferred_mood),
                    episode.get("active_domain"),
                    episode.get("energy_level"),
                    episode.get("interaction_type", "unknown"),
                    episode.get("content_summary", ""),
                    episode.get("content_full"),
                    json.dumps(contacts_involved),
                    json.dumps(topics),
                    json.dumps(entities),
                    episode.get("outcome"),
                    episode.get("user_satisfaction"),
                    episode.get("embedding_id"),
                ),
            )

        self._emit_telemetry("usermodel.episode.stored", {
            "episode_id": episode["id"],
            "event_id": episode["event_id"],
            "interaction_type": episode.get("interaction_type", "unknown"),
            "active_domain": episode.get("active_domain"),
            # Use the normalized list values (not raw episode dict) to avoid
            # len(None) error when contacts/topics are None
            "contacts_count": len(contacts_involved),
            "topics_count": len(topics),
            "stored_at": episode["timestamp"],
        })

    def update_semantic_fact(self, key: str, category: str, value: Any,
                            confidence: float, episode_id: Optional[str] = None):
        """Update or create a semantic memory fact.

        Confidence increment logic:
        - If the fact already exists, its confidence is bumped by +0.05 (capped
          at 1.0) each time it is re-confirmed.  This gradual increase reflects
          the principle that repeatedly observed facts should be trusted more.
        - If the fact is new, it is inserted with the caller-supplied initial
          confidence (typically 0.5 for inferred facts).
        - ``source_episodes`` accumulates the list of episode IDs that support
          this fact, providing an audit trail back to raw observations.

        User-correction protection:
        - If the existing fact has ``is_user_corrected = 1``, this method skips
          the update entirely and returns without modifying the fact.  This
          guarantees that inference never silently overwrites an explicit user
          correction.  The PATCH /api/user-model/facts/{key} endpoint writes
          ``is_user_corrected`` directly via SQL (bypassing this method) so it
          can still update corrected facts when the user changes their mind.
        """
        is_new = False
        existing = None
        with self.db.get_connection("user_model") as conn:
            # Check whether this fact already exists in semantic memory.
            existing = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

            if existing:
                # --- Guard: never overwrite a user-corrected fact ---
                # When is_user_corrected = 1, the user has explicitly told the
                # system that the inferred value is wrong.  Honoring that flag
                # is critical: without this guard, every SemanticFactInferrer
                # run would silently restore the incorrect inferred value,
                # making user corrections completely ineffective.
                if existing["is_user_corrected"]:
                    logger.debug(
                        "Skipping update for user-corrected fact '%s' "
                        "(inference cannot overwrite explicit user corrections)",
                        key,
                    )
                    return

                # --- Existing fact: increment confidence and append the source episode ---
                episodes = json.loads(existing["source_episodes"])
                if episode_id and episode_id not in episodes:
                    episodes.append(episode_id)

                # Confidence grows by 0.05 per re-confirmation, but never exceeds 1.0.
                # This slow ramp prevents a single burst of duplicate signals from
                # immediately maxing out confidence.
                new_confidence = min(1.0, existing["confidence"] + 0.05)

                conn.execute(
                    """UPDATE semantic_facts
                       SET value = ?, confidence = ?, source_episodes = ?,
                           last_confirmed = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                           times_confirmed = times_confirmed + 1
                       WHERE key = ?""",
                    (json.dumps(value), new_confidence, json.dumps(episodes), key),
                )
            else:
                # --- New fact: insert with the initial confidence from the caller ---
                is_new = True
                episodes = [episode_id] if episode_id else []
                conn.execute(
                    """INSERT INTO semantic_facts (key, category, value, confidence, source_episodes)
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, category, json.dumps(value), confidence, json.dumps(episodes)),
                )

        self._emit_telemetry("usermodel.fact.learned", {
            "key": key,
            "category": category,
            "confidence": confidence if is_new else min(1.0, (existing["confidence"] if existing else confidence) + 0.05),
            "is_new": is_new,
            "episode_id": episode_id,
            "learned_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_semantic_fact(self, key: str) -> Optional[dict]:
        """
        Retrieve a single semantic memory fact by key.

        Returns the fact dict with deserialized value and source_episodes,
        or None if the fact does not exist.

        Args:
            key: Unique identifier for the semantic fact

        Returns:
            Fact dictionary or None if not found
        """
        with self.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

            if row:
                fact = dict(row)
                # Deserialize JSON fields
                fact["value"] = json.loads(fact["value"])
                fact["source_episodes"] = json.loads(fact["source_episodes"])
                return fact
            return None

    def get_semantic_facts(self, category: Optional[str] = None,
                          min_confidence: float = 0.0) -> list[dict]:
        """Retrieve semantic memory facts.

        Deserializes the ``value`` and ``source_episodes`` fields from JSON
        so they are returned as native Python objects.
        """
        query = "SELECT * FROM semantic_facts WHERE confidence >= ?"
        params: list[Any] = [min_confidence]

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY confidence DESC"

        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(query, params).fetchall()
            facts = []
            for row in rows:
                fact = dict(row)
                # Deserialize JSON fields
                fact["value"] = json.loads(fact["value"])
                fact["source_episodes"] = json.loads(fact["source_episodes"])
                facts.append(fact)
            return facts

    def update_signal_profile(self, profile_type: str, data: dict):
        """Store or update a signal profile (linguistic, cadence, etc.).

        Upsert with sample counting:
        The SQL uses INSERT OR REPLACE combined with a COALESCE sub-select to
        atomically read the current ``samples_count`` before the old row is
        deleted by the REPLACE, then increment it by 1 in the new row.
        This ensures the sample counter survives the replace and accurately
        reflects how many data points have been incorporated into the profile.
        """
        with self.db.get_connection("user_model") as conn:
            # INSERT OR REPLACE deletes any existing row with the same PRIMARY KEY
            # and inserts the new one.  The COALESCE sub-select reads the old
            # samples_count *before* the delete occurs (within the same statement),
            # so we can carry it forward incremented by 1.
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?,
                           COALESCE((SELECT samples_count FROM signal_profiles WHERE profile_type = ?), 0) + 1,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                (profile_type, json.dumps(data), profile_type),
            )

        self._emit_telemetry("usermodel.signal_profile.updated", {
            "profile_type": profile_type,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_signal_profile(self, profile_type: str) -> Optional[dict]:
        """Retrieve a signal profile.

        The ``data`` column is stored as a JSON string in SQLite, so we
        deserialize it back to a Python dict before returning.
        """
        with self.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM signal_profiles WHERE profile_type = ?",
                (profile_type,),
            ).fetchone()
            if row:
                result = dict(row)
                # Deserialize the JSON blob back into a native Python dict.
                result["data"] = json.loads(result["data"])
                return result
            return None

    def store_mood(self, mood: dict):
        """Log a mood state reading.

        Appends a new row to the mood_history time-series.  Default values are
        provided for all dimensions so that partial mood readings (e.g. only
        energy and stress) can still be recorded without raising errors.
        """
        timestamp = mood.get("timestamp", datetime.now(timezone.utc).isoformat())
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO mood_history
                   (timestamp, energy_level, stress_level, social_battery,
                    cognitive_load, emotional_valence, confidence, contributing_signals, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    mood.get("energy_level", 0.5),
                    mood.get("stress_level", 0.3),
                    mood.get("social_battery", 0.5),
                    mood.get("cognitive_load", 0.3),
                    mood.get("emotional_valence", 0.5),
                    mood.get("confidence", 0.0),
                    json.dumps(mood.get("contributing_signals", [])),
                    mood.get("trend", "stable"),
                ),
            )

        self._emit_telemetry("usermodel.mood.recorded", {
            "energy_level": mood.get("energy_level", 0.5),
            "stress_level": mood.get("stress_level", 0.3),
            "social_battery": mood.get("social_battery", 0.5),
            "cognitive_load": mood.get("cognitive_load", 0.3),
            "emotional_valence": mood.get("emotional_valence", 0.5),
            "confidence": mood.get("confidence", 0.0),
            "trend": mood.get("trend", "stable"),
            "signals_count": len(mood.get("contributing_signals", [])),
            "recorded_at": timestamp,
        })

    def store_prediction(self, prediction: dict):
        """Store a prediction for later accuracy evaluation.

        Predictions are logged when generated and later resolved (was_accurate
        is updated) to create a feedback loop — the system can measure its own
        prediction quality over time and adjust confidence thresholds.

        Predictions that are immediately filtered (not surfaced) are stored with
        resolved_at and user_response='filtered' to prevent database bloat.

        DEDUPLICATION: Before storing a new prediction, check if an identical
        recent prediction already exists (same type + same description).
        If a duplicate exists, skip storage to prevent:
        - Database bloat (340K+ duplicate predictions)
        - Event bus spam (redundant telemetry)
        - Processing waste (regenerating same predictions every 15 min)

        This is critical because the prediction engine runs every 15 minutes
        and would otherwise recreate ALL predictions (even unresolved ones)
        on every cycle, causing exponential growth.

        The deduplication window is 24 hours because:
        - Unresolved (surfaced) predictions: User hasn't interacted yet, don't duplicate
        - Filtered predictions: Same conditions likely still apply (confidence/reaction
          gates), don't regenerate within 24h
        - After 24h: Conditions may have changed (relationship gaps widening, events
          approaching, stress levels shifting), allow regeneration
        """
        with self.db.get_connection("user_model") as conn:
            # Check for existing prediction with same type + description within 24h
            # This catches both:
            # 1. Unresolved predictions (resolved_at IS NULL) that user hasn't acted on
            # 2. Recently filtered predictions (resolved_at within 24h) that failed gates
            #
            # CRITICAL: Use datetime(resolved_at) to parse ISO timestamps correctly.
            # Direct string comparison fails because ISO format ('2026-02-15T16:00:00+00:00')
            # sorts differently than SQLite format ('2026-02-15 18:00:00').
            existing = conn.execute(
                """SELECT id FROM predictions
                   WHERE prediction_type = ?
                   AND description = ?
                   AND (resolved_at IS NULL OR datetime(resolved_at) > datetime('now', '-24 hours'))
                   LIMIT 1""",
                (prediction["prediction_type"], prediction["description"]),
            ).fetchone()

            # If duplicate exists (either unresolved or recently filtered), skip storage
            if existing:
                # Telemetry for observability: track that deduplication occurred
                self._emit_telemetry("usermodel.prediction.deduplicated", {
                    "existing_prediction_id": existing["id"],
                    "attempted_prediction_type": prediction["prediction_type"],
                    "attempted_description": prediction["description"][:100],  # Truncate for telemetry
                    "deduplicated_at": datetime.now(timezone.utc).isoformat(),
                })
                return  # Skip storage

            # No duplicate found, store the prediction
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, supporting_signals, was_surfaced,
                    user_response, resolved_at, filter_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction["id"],
                    prediction["prediction_type"],
                    prediction["description"],
                    prediction["confidence"],
                    prediction["confidence_gate"],
                    prediction.get("time_horizon"),
                    prediction.get("suggested_action"),
                    json.dumps(prediction.get("supporting_signals", {})),
                    prediction.get("was_surfaced", False),
                    prediction.get("user_response"),
                    prediction.get("resolved_at"),
                    prediction.get("filter_reason"),
                ),
            )

        # Emit telemetry for successfully stored (non-duplicate) prediction
        self._emit_telemetry("usermodel.prediction.generated", {
            "prediction_id": prediction["id"],
            "prediction_type": prediction["prediction_type"],
            "confidence": prediction["confidence"],
            "confidence_gate": prediction["confidence_gate"],
            "time_horizon": prediction.get("time_horizon"),
            "was_surfaced": prediction.get("was_surfaced", False),
            "signals_count": len(prediction.get("supporting_signals", [])),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    def store_communication_template(self, template: dict):
        """Store or update a communication template.

        Communication templates capture the user's writing style for a specific
        context + contact + channel combination (e.g. "professional email to
        manager" or "casual Slack to teammate").  INSERT OR REPLACE ensures that
        re-analyzing the same context/contact pair simply overwrites the old
        template with updated style signals.

        List-type style fields (common_phrases, avoids_phrases, tone_notes,
        example_message_ids) are serialized to JSON for flexible storage.
        ``samples_analyzed`` tracks how many real messages were used to derive
        the template, giving the AI engine a sense of statistical reliability.
        """
        with self.db.get_connection("user_model") as conn:
            # INSERT OR REPLACE: idempotent upsert keyed on template ``id``.
            conn.execute(
                """INSERT OR REPLACE INTO communication_templates
                   (id, context, contact_id, channel, greeting, closing, formality,
                    typical_length, uses_emoji, common_phrases, avoids_phrases,
                    tone_notes, example_message_ids, samples_analyzed, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                (
                    template["id"],
                    template["context"],
                    template.get("contact_id"),
                    template.get("channel"),
                    template.get("greeting"),
                    template.get("closing"),
                    template.get("formality", 0.5),
                    template.get("typical_length", 50.0),
                    int(template.get("uses_emoji", False)),
                    json.dumps(template.get("common_phrases", [])),
                    json.dumps(template.get("avoids_phrases", [])),
                    json.dumps(template.get("tone_notes", [])),
                    json.dumps(template.get("example_message_ids", [])),
                    template.get("samples_analyzed", 0),
                ),
            )

        self._emit_telemetry("usermodel.template.updated", {
            "template_id": template["id"],
            "contact_id": template.get("contact_id"),
            "channel": template.get("channel"),
            "formality": template.get("formality", 0.5),
            "samples_analyzed": template.get("samples_analyzed", 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def _deserialize_template_row(self, row) -> dict:
        """Convert a communication_templates row to a dict with deserialized JSON fields.

        The common_phrases, avoids_phrases, tone_notes, and example_message_ids
        columns are stored as JSON strings in SQLite. This helper deserializes
        them back to native Python lists so callers get ready-to-use dicts.
        """
        template = dict(row)
        for field in ("common_phrases", "avoids_phrases", "tone_notes", "example_message_ids"):
            raw = template.get(field)
            if raw is not None:
                try:
                    template[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    template[field] = []
            else:
                template[field] = []
        return template

    def get_communication_template(
        self, contact_id: str | None = None, channel: str | None = None
    ) -> Optional[dict]:
        """Retrieve the best-matching communication template.

        Looks up a template by contact_id and/or channel, returning the one
        with the highest ``samples_analyzed`` (most statistically reliable).
        Uses an OR query so that a contact-specific template or a channel-wide
        default can both match.

        Args:
            contact_id: Optional contact identifier to match
            channel: Optional channel name to match (e.g. "email", "slack")

        Returns:
            Deserialized template dict, or None if no match found
        """
        with self.db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT * FROM communication_templates
                   WHERE contact_id = ? OR channel = ?
                   ORDER BY samples_analyzed DESC LIMIT 1""",
                (contact_id, channel),
            ).fetchone()

            if row:
                return self._deserialize_template_row(row)
            return None

    def get_communication_templates(
        self,
        contact_id: str | None = None,
        channel: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Retrieve communication templates with optional filtering.

        Builds the WHERE clause dynamically based on which filters are provided:
        - contact_id only → WHERE contact_id = ?
        - channel only → WHERE channel = ?
        - both → WHERE contact_id = ? OR channel = ?
        - neither → no WHERE clause (returns all templates)

        Results are ordered by samples_analyzed DESC so the most data-rich
        templates appear first.

        Args:
            contact_id: Optional contact identifier filter
            channel: Optional channel name filter
            limit: Maximum number of templates to return (default 20)

        Returns:
            List of deserialized template dicts
        """
        query = "SELECT * FROM communication_templates"
        params: list[Any] = []

        if contact_id and channel:
            query += " WHERE contact_id = ? OR channel = ?"
            params.extend([contact_id, channel])
        elif contact_id:
            query += " WHERE contact_id = ?"
            params.append(contact_id)
        elif channel:
            query += " WHERE channel = ?"
            params.append(channel)

        query += " ORDER BY samples_analyzed DESC LIMIT ?"
        params.append(limit)

        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._deserialize_template_row(row) for row in rows]

    def resolve_prediction(self, prediction_id: str, was_accurate: bool,
                          user_response: str = None, resolution_reason: str = None):
        """Mark a prediction as resolved with user feedback.

        Records whether the prediction was accurate, any optional user response
        text, and an optional machine-readable resolution_reason. This feedback
        drives the accuracy-based confidence adjustment in the prediction engine.

        Args:
            prediction_id: UUID of the prediction to resolve
            was_accurate: True if the prediction was helpful/correct,
                         False if it was unhelpful/incorrect
            user_response: Optional free-text or coded response ('inferred',
                          'filtered', or direct user feedback text)
            resolution_reason: Optional machine-readable reason for resolution.
                Callers should use well-known values so the accuracy multiplier
                can selectively exclude non-behavioral resolutions:
                  'automated_sender_fast_path' — resolved immediately because
                      the contact is a marketing/automated sender; this is a
                      historical bug cleanup, not a real user-behavior signal
                  'timeout_no_action' — inference window elapsed with no action
                  NULL — user explicit feedback or pre-v4 resolution (no reason)

        The resolved_at timestamp is set automatically to track when the
        prediction was resolved. This enables queries like "predictions
        resolved in the last 30 days" for computing rolling accuracy rates.
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """UPDATE predictions
                   SET was_accurate = ?,
                       user_response = ?,
                       resolution_reason = ?,
                       resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                   WHERE id = ?""",
                (int(was_accurate), user_response, resolution_reason, prediction_id),
            )

        self._emit_telemetry("usermodel.prediction.resolved", {
            "prediction_id": prediction_id,
            "was_accurate": was_accurate,
            "has_response": bool(user_response),
            "resolution_reason": resolution_reason,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })

    def store_routine(self, routine: dict):
        """Store or update a detected routine (Layer 3: Procedural Memory).

        Routines are recurring behavioral patterns discovered by analyzing
        episodic memory. Examples: morning routine (check email → review
        calendar → coffee), arrive-at-work routine, post-meeting workflow.

        Uses INSERT OR REPLACE keyed on routine name. If a routine with the
        same name already exists, this overwrites it with updated statistics
        (consistency_score, times_observed, typical_duration). The steps list
        is completely replaced on each update.

        Steps are stored as JSON to preserve order and all metadata (action,
        duration, skip_rate). Variations list tracks known departures from
        the canonical pattern (e.g., "skips coffee on Mondays").

        Args:
            routine: Dictionary with keys:
                - name (str): Human-readable routine name
                - trigger (str): What initiates this routine ("morning", "arrive_home")
                - steps (list[dict]): Ordered list of actions with timing metadata
                - typical_duration_minutes (float): Normal total duration
                - consistency_score (float): 0-1, how reliably user follows pattern
                - times_observed (int): Number of instances detected
                - variations (list[str]): Known pattern deviations
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO routines
                   (name, trigger_condition, steps, typical_duration, consistency_score,
                    times_observed, variations, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                (
                    routine["name"],
                    routine["trigger"],
                    json.dumps(routine.get("steps", [])),
                    routine.get("typical_duration_minutes", 30.0),
                    routine.get("consistency_score", 0.5),
                    routine.get("times_observed", 0),
                    json.dumps(routine.get("variations", [])),
                ),
            )

        self._emit_telemetry("usermodel.routine.updated", {
            "routine_name": routine["name"],
            "trigger": routine["trigger"],
            "steps_count": len(routine.get("steps", [])),
            "consistency_score": routine.get("consistency_score", 0.5),
            "times_observed": routine.get("times_observed", 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_routines(self, trigger: Optional[str] = None) -> list[dict]:
        """Retrieve stored routines, optionally filtered by trigger.

        Args:
            trigger: Optional trigger filter (e.g., "morning", "arrive_home").
                    If None, returns all routines.

        Returns:
            List of routine dictionaries with steps deserialized from JSON
        """
        with self.db.get_connection("user_model") as conn:
            if trigger:
                cursor = conn.execute(
                    """SELECT name, trigger_condition, steps, typical_duration,
                              consistency_score, times_observed, variations, updated_at
                       FROM routines
                       WHERE trigger_condition = ?
                       ORDER BY consistency_score DESC, times_observed DESC""",
                    (trigger,)
                )
            else:
                cursor = conn.execute(
                    """SELECT name, trigger_condition, steps, typical_duration,
                              consistency_score, times_observed, variations, updated_at
                       FROM routines
                       ORDER BY consistency_score DESC, times_observed DESC"""
                )

            routines = []
            for row in cursor.fetchall():
                routines.append({
                    "name": row[0],
                    "trigger": row[1],
                    "steps": json.loads(row[2]) if row[2] else [],
                    "typical_duration_minutes": row[3],
                    "consistency_score": row[4],
                    "times_observed": row[5],
                    "variations": json.loads(row[6]) if row[6] else [],
                    "updated_at": row[7],
                })

            return routines

    def store_workflow(self, workflow: dict):
        """Store or update a detected workflow (Layer 3: Procedural Memory).

        Workflows are multi-step processes for accomplishing specific types of
        tasks. Unlike routines (time/location triggered), workflows are goal-driven
        and can be initiated in various contexts.

        Examples:
        - "responding_to_boss": read email → draft response → check tone → send
        - "planning_trip": search flights → book hotel → add to calendar → notify
        - "weekly_review": check completed tasks → review calendar → plan next week

        Uses INSERT OR REPLACE keyed on workflow name. If a workflow with the
        same name already exists, this overwrites it with updated statistics
        (success_rate, times_observed, typical_duration). The steps list and
        tools_used list are completely replaced on each update.

        Args:
            workflow: Dictionary with keys:
                - name (str): Human-readable workflow name
                - trigger_conditions (list[str]): Conditions that initiate this workflow
                - steps (list[str]): Ordered list of actions
                - typical_duration_minutes (float, optional): Normal total duration
                - tools_used (list[str]): Tools/apps involved in workflow
                - success_rate (float): 0-1, how often workflow completes successfully
                - times_observed (int): Number of instances detected
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO workflows
                   (name, trigger_conditions, steps, typical_duration, tools_used,
                    success_rate, times_observed, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                (
                    workflow["name"],
                    json.dumps(workflow.get("trigger_conditions", [])),
                    json.dumps(workflow.get("steps", [])),
                    workflow.get("typical_duration_minutes"),
                    json.dumps(workflow.get("tools_used", [])),
                    workflow.get("success_rate", 0.5),
                    workflow.get("times_observed", 0),
                ),
            )

        self._emit_telemetry("usermodel.workflow.updated", {
            "workflow_name": workflow["name"],
            "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
            "steps_count": len(workflow.get("steps", [])),
            "tools_count": len(workflow.get("tools_used", [])),
            "success_rate": workflow.get("success_rate", 0.5),
            "times_observed": workflow.get("times_observed", 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_workflows(self, name_filter: Optional[str] = None) -> list[dict]:
        """Retrieve stored workflows, optionally filtered by name pattern.

        Args:
            name_filter: Optional name filter (SQL LIKE pattern, e.g., "%email%").
                        If None, returns all workflows.

        Returns:
            List of workflow dictionaries with arrays deserialized from JSON
        """
        with self.db.get_connection("user_model") as conn:
            if name_filter:
                cursor = conn.execute(
                    """SELECT name, trigger_conditions, steps, typical_duration,
                              tools_used, success_rate, times_observed, updated_at
                       FROM workflows
                       WHERE name LIKE ?
                       ORDER BY success_rate DESC, times_observed DESC""",
                    (name_filter,)
                )
            else:
                cursor = conn.execute(
                    """SELECT name, trigger_conditions, steps, typical_duration,
                              tools_used, success_rate, times_observed, updated_at
                       FROM workflows
                       ORDER BY success_rate DESC, times_observed DESC"""
                )

            workflows = []
            for row in cursor.fetchall():
                workflows.append({
                    "name": row[0],
                    "trigger_conditions": json.loads(row[1]) if row[1] else [],
                    "steps": json.loads(row[2]) if row[2] else [],
                    "typical_duration_minutes": row[3],
                    "tools_used": json.loads(row[4]) if row[4] else [],
                    "success_rate": row[5],
                    "times_observed": row[6],
                    "updated_at": row[7],
                })

            return workflows
