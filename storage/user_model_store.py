"""
Life OS — User Model Store

High-level operations on the user model database:
episodes, semantic facts, signal profiles, mood, predictions,
and communication templates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from storage.manager import DatabaseManager


class UserModelStore:
    """High-level operations on the user model database.

    Provides CRUD helpers for the three memory layers (episodic, semantic,
    procedural) plus mood tracking, signal profiles, and predictions.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

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
                episodes = [episode_id] if episode_id else []
                conn.execute(
                    """INSERT INTO semantic_facts (key, category, value, confidence, source_episodes)
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, category, json.dumps(value), confidence, json.dumps(episodes)),
                )

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
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO mood_history 
                   (timestamp, energy_level, stress_level, social_battery,
                    cognitive_load, emotional_valence, confidence, contributing_signals, trend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mood.get("timestamp", datetime.now(timezone.utc).isoformat()),
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
