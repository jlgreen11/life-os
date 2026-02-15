"""
Life OS — User Model Store

High-level operations on the user model database:
episodes, semantic facts, signal profiles, mood, predictions,
and communication templates.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional

from storage.manager import DatabaseManager


class UserModelStore:
    """High-level operations on the user model database.

    Provides CRUD helpers for the three memory layers (episodic, semantic,
    procedural) plus mood tracking, signal profiles, and predictions.
    """

    def __init__(self, db: DatabaseManager, event_bus: Any = None):
        self.db = db
        self._event_bus = event_bus

    def _emit_telemetry(self, event_type: str, payload: dict):
        """Fire-and-forget telemetry event publication.

        Since UserModelStore methods are synchronous but called from async
        contexts, this uses asyncio.create_task for non-blocking publishing.
        """
        if not self._event_bus or not self._event_bus.is_connected:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._event_bus.publish(event_type, payload, source="user_model_store")
            )
        except RuntimeError:
            pass  # No running event loop — skip telemetry

    def store_episode(self, episode: dict):
        """Store an episodic memory.

        Uses INSERT OR REPLACE so that re-processing the same event (same ``id``)
        overwrites the previous episode rather than raising a uniqueness error.
        This makes the operation idempotent — safe to retry on failure.

        Multi-value fields (contacts_involved, topics, entities) and structured
        fields (inferred_mood) are serialized to JSON for storage.
        """
        with self.db.get_connection("user_model") as conn:
            # INSERT OR REPLACE: if a row with the same PRIMARY KEY (id) already
            # exists, SQLite deletes it and inserts the new row.  This is the
            # idempotent upsert strategy used throughout the episodic layer.
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
                    json.dumps(episode.get("inferred_mood", {})),
                    episode.get("active_domain"),
                    episode.get("energy_level"),
                    episode["interaction_type"],
                    episode["content_summary"],
                    episode.get("content_full"),
                    json.dumps(episode.get("contacts_involved", [])),
                    json.dumps(episode.get("topics", [])),
                    json.dumps(episode.get("entities", [])),
                    episode.get("outcome"),
                    episode.get("user_satisfaction"),
                    episode.get("embedding_id"),
                ),
            )

        self._emit_telemetry("usermodel.episode.stored", {
            "episode_id": episode["id"],
            "event_id": episode["event_id"],
            "interaction_type": episode["interaction_type"],
            "active_domain": episode.get("active_domain"),
            "contacts_count": len(episode.get("contacts_involved", [])),
            "topics_count": len(episode.get("topics", [])),
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
        """
        is_new = False
        with self.db.get_connection("user_model") as conn:
            # Check whether this fact already exists in semantic memory.
            existing = conn.execute(
                "SELECT * FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

            if existing:
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

    def get_semantic_facts(self, category: Optional[str] = None,
                          min_confidence: float = 0.0) -> list[dict]:
        """Retrieve semantic memory facts."""
        query = "SELECT * FROM semantic_facts WHERE confidence >= ?"
        params: list[Any] = [min_confidence]

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY confidence DESC"

        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

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
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, supporting_signals, was_surfaced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction["id"],
                    prediction["prediction_type"],
                    prediction["description"],
                    prediction["confidence"],
                    prediction["confidence_gate"],
                    prediction.get("time_horizon"),
                    prediction.get("suggested_action"),
                    json.dumps(prediction.get("supporting_signals", [])),
                    prediction.get("was_surfaced", False),
                ),
            )

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

    def resolve_prediction(self, prediction_id: str, was_accurate: bool,
                          user_response: str = None):
        """Mark a prediction as resolved with user feedback.

        Records whether the prediction was accurate and any optional user
        response text. This feedback drives the accuracy-based confidence
        adjustment system in the prediction engine.

        Args:
            prediction_id: UUID of the prediction to resolve
            was_accurate: True if the prediction was helpful/correct,
                         False if it was unhelpful/incorrect
            user_response: Optional free-text feedback from the user

        The resolved_at timestamp is set automatically to track when the
        user provided feedback. This enables queries like "predictions
        resolved in the last 30 days" for computing rolling accuracy rates.
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """UPDATE predictions
                   SET was_accurate = ?,
                       user_response = ?,
                       resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                   WHERE id = ?""",
                (int(was_accurate), user_response, prediction_id),
            )

        self._emit_telemetry("usermodel.prediction.resolved", {
            "prediction_id": prediction_id,
            "was_accurate": was_accurate,
            "has_response": bool(user_response),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
