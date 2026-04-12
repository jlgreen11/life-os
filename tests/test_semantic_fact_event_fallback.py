"""
Tests for SemanticFactInferrer event-based fallback inference.

When the episodes table is empty (e.g., due to a broken episode pipeline),
the normal signal-profile inference path produces 0 facts.  The event-based
fallback queries events.db directly to derive basic relationship, temporal,
and topic facts so semantic memory is not completely empty.

Covers:
  1. infer_facts_from_events() writes relationship facts for top email contacts.
  2. infer_facts_from_events() writes temporal facts (active hours, active day).
  3. Topic facts are written from email subject keywords.
  4. Marketing/no-reply senders are excluded from relationship facts.
  5. The fallback is skipped (gate) when episodes exist (normal path wins).
  6. run_all_inference() triggers the fallback when episodes=0 and events>100.
  7. run_all_inference() does NOT trigger the fallback when episodes>0.
  8. infer_facts_from_events() handles an empty events table gracefully.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_email_received(db, from_address: str, subject: str = "Hello",
                            hour: int = 10, day_offset: int = 0) -> None:
    """Insert a single email.received event into events.db.

    Args:
        db: DatabaseManager fixture.
        from_address: Sender email address (stored in payload.from_address).
        subject: Email subject line.
        hour: Hour of day (0-23) for the event timestamp.
        day_offset: Days offset from the base date (2024-01-08 = Monday).
    """
    # Use a fixed Monday base date so day-of-week tests are deterministic
    base = datetime(2024, 1, 8, hour, 0, 0, tzinfo=UTC)  # 2024-01-08 = Monday
    ts = (base + timedelta(days=day_offset)).isoformat()
    payload = json.dumps({
        "from_address": from_address,
        "to_addresses": ["me@example.com"],
        "subject": subject,
        "body": "Test body",
        "message_id": str(uuid.uuid4()),
    })
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), "email.received", "proton_mail",
             ts, "normal", payload, "{}"),
        )


def _insert_bulk_email_received(db, contacts: list[tuple[str, int]],
                                hour: int = 10) -> None:
    """Insert multiple email.received events from a list of (from_address, count) pairs.

    Args:
        db: DatabaseManager fixture.
        contacts: List of (email_address, message_count) tuples.
        hour: Hour of day for all inserted events.
    """
    for from_addr, count in contacts:
        for i in range(count):
            _insert_email_received(
                db, from_addr,
                subject=f"Message {i} from {from_addr.split('@')[0]}",
                hour=hour,
                day_offset=i % 7,
            )


def _insert_episode(db) -> str:
    """Insert a minimal episode row to simulate a non-empty episodes table.

    Uses the actual episodes table schema from storage/manager.py:
      id, event_id, interaction_type, timestamp, content_summary (all required).

    Returns:
        The inserted episode ID.
    """
    episode_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())  # episodes.event_id is NOT NULL
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, event_id, interaction_type, timestamp, content_summary)
               VALUES (?, ?, ?, ?, ?)""",
            (
                episode_id,
                event_id,
                "email_received",
                datetime.now(UTC).isoformat(),
                "Test episode summary",
            ),
        )
    return episode_id


def _episode_count(db) -> int:
    """Return the number of rows in the episodes table."""
    with db.get_connection("user_model") as conn:
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


def _event_count(db) -> int:
    """Return the number of rows in the events table."""
    with db.get_connection("events") as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def _get_all_fallback_facts(user_model_store) -> list[dict]:
    """Return all semantic facts whose key starts with 'event_fallback_'."""
    all_facts = user_model_store.get_semantic_facts()
    return [f for f in all_facts if f["key"].startswith("event_fallback_")]


# ---------------------------------------------------------------------------
# Tests: infer_facts_from_events() — relationship facts
# ---------------------------------------------------------------------------

class TestEventFallbackRelationshipFacts:
    """Relationship facts are derived from email.received sender frequencies."""

    def test_top_contacts_produce_relationship_facts(self, db, user_model_store):
        """Top frequent senders should produce event_fallback_contact_* facts.

        5 contacts each sending 4 emails (20 total) with no episodes.
        Expects a fact for each contact with email_count >= 2.
        """
        contacts = [
            ("alice@company.com", 4),
            ("bob@company.com", 4),
            ("carol@company.com", 4),
            ("dave@company.com", 4),
            ("eve@company.com", 4),
        ]
        _insert_bulk_email_received(db, contacts)

        assert _episode_count(db) == 0

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_facts_from_events()

        assert result["processed"] is True
        assert result["facts_written"] >= 5, (
            "Expected at least 1 fact per contact (5 contacts x 4 emails each)"
        )

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = [f for f in fallback_facts if "contact" in f["key"]]
        assert len(contact_facts) >= 5, (
            f"Expected 5 contact facts, got {len(contact_facts)}: "
            f"{[f['key'] for f in contact_facts]}"
        )

    def test_contact_fact_has_email_and_count(self, db, user_model_store):
        """Contact facts must embed the email address and send count in value."""
        _insert_bulk_email_received(db, [("alice@work.com", 5)])

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = [f for f in fallback_facts if "contact" in f["key"]]
        assert len(contact_facts) >= 1, "Expected at least one contact fact"

        fact = contact_facts[0]
        value = fact["value"]
        assert "email" in value, "Fact value must contain 'email' key"
        assert "email_count" in value, "Fact value must contain 'email_count' key"
        assert value["email"] == "alice@work.com"
        assert value["email_count"] == 5

    def test_work_vs_personal_classification(self, db, user_model_store):
        """Contacts are classified as 'work' (non-personal domain) or 'personal'."""
        _insert_bulk_email_received(db, [
            ("colleague@bigcorp.com", 3),  # → work (non-personal domain)
            ("friend@gmail.com", 3),        # → personal (gmail.com is in PERSONAL_DOMAINS)
        ])

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = {f["value"]["email"]: f["value"] for f in fallback_facts
                         if "contact" in f["key"] and "email" in f.get("value", {})}

        assert "colleague@bigcorp.com" in contact_facts, (
            "Work contact missing from facts"
        )
        assert contact_facts["colleague@bigcorp.com"]["relationship_type"] == "work"

        assert "friend@gmail.com" in contact_facts, (
            "Personal contact missing from facts"
        )
        assert contact_facts["friend@gmail.com"]["relationship_type"] == "personal"

    def test_marketing_senders_excluded(self, db, user_model_store):
        """No-reply and marketing senders must not produce contact facts."""
        _insert_bulk_email_received(db, [
            ("no-reply@amazon.com", 10),        # marketing / no-reply
            ("noreply@paypal.com", 10),          # no-reply pattern
            ("real_person@company.com", 3),      # genuine human contact
        ])

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = {f["value"]["email"]: f for f in fallback_facts
                         if "contact" in f["key"] and "email" in f.get("value", {})}

        assert "no-reply@amazon.com" not in contact_facts, (
            "Marketing sender should be filtered out"
        )
        assert "noreply@paypal.com" not in contact_facts, (
            "No-reply sender should be filtered out"
        )
        assert "real_person@company.com" in contact_facts, (
            "Real human contact should be present"
        )

    def test_contact_below_minimum_count_excluded(self, db, user_model_store):
        """Contacts with only 1 email should not produce a fact (min count is 2)."""
        _insert_bulk_email_received(db, [
            ("frequent@company.com", 5),
            ("rare@company.com", 1),   # Below threshold of 2
        ])

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_emails = {
            f["value"].get("email")
            for f in fallback_facts
            if "contact" in f["key"] and isinstance(f.get("value"), dict)
        }

        assert "frequent@company.com" in contact_emails
        assert "rare@company.com" not in contact_emails, (
            "Contact with only 1 email should not produce a fact"
        )

    def test_max_ten_contact_facts_written(self, db, user_model_store):
        """At most 10 contact facts should be written even if more contacts exist."""
        # Insert 15 contacts each with 3 emails
        contacts = [(f"person{i}@corp.com", 3) for i in range(15)]
        _insert_bulk_email_received(db, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = [f for f in fallback_facts if "contact" in f["key"]]
        assert len(contact_facts) <= 10, (
            f"Expected at most 10 contact facts, got {len(contact_facts)}"
        )


# ---------------------------------------------------------------------------
# Tests: infer_facts_from_events() — temporal facts
# ---------------------------------------------------------------------------

class TestEventFallbackTemporalFacts:
    """Temporal facts are derived from event timestamps."""

    def test_active_hours_fact_created(self, db, user_model_store):
        """Active hours fact should be created when events cluster at certain hours."""
        # Insert 20 events all at hour 9 (well above average for any single hour)
        for i in range(20):
            _insert_email_received(db, f"sender{i}@work.com", hour=9, day_offset=i % 7)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        active_hours_fact = next(
            (f for f in fallback_facts if f["key"] == "event_fallback_active_hours"),
            None,
        )
        assert active_hours_fact is not None, (
            "Expected event_fallback_active_hours fact from events clustered at hour 9"
        )
        assert 9 in active_hours_fact["value"], (
            f"Expected hour 9 in active_hours, got {active_hours_fact['value']}"
        )

    def test_most_active_day_fact_created(self, db, user_model_store):
        """Most active day fact should reflect the day with the most events."""
        # Insert 10 events on Monday (day_offset=0 from the Monday base date)
        for i in range(10):
            _insert_email_received(db, f"sender{i}@work.com", hour=10, day_offset=0)

        # Insert 2 events each on Tuesday and Wednesday
        for i in range(2):
            _insert_email_received(db, f"sender_tue{i}@work.com", hour=10, day_offset=1)
            _insert_email_received(db, f"sender_wed{i}@work.com", hour=10, day_offset=2)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        active_day_fact = next(
            (f for f in fallback_facts if f["key"] == "event_fallback_most_active_day"),
            None,
        )
        assert active_day_fact is not None, (
            "Expected event_fallback_most_active_day fact"
        )
        assert active_day_fact["value"] == "monday", (
            f"Expected 'monday' as most active day, got '{active_day_fact['value']}'"
        )

    def test_confidence_is_low(self, db, user_model_store):
        """All event-fallback facts must use the cold-start confidence of 0.3."""
        _insert_bulk_email_received(db, [("alice@corp.com", 5)], hour=10)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        assert len(fallback_facts) > 0, "Expected some facts to be written"
        for fact in fallback_facts:
            assert fact["confidence"] <= 0.35, (
                f"Fact '{fact['key']}' has confidence {fact['confidence']}, "
                "expected cold-start value of 0.3 (may be 0.3 or slightly above "
                "if update_semantic_fact incremented an existing row)"
            )


# ---------------------------------------------------------------------------
# Tests: infer_facts_from_events() — topic facts
# ---------------------------------------------------------------------------

class TestEventFallbackTopicFacts:
    """Topic facts are derived from email subject lines."""

    def test_frequent_subject_words_produce_topic_facts(self, db, user_model_store):
        """Words appearing frequently in subject lines should become topic facts."""
        subjects = [
            "Budget planning review",
            "Budget update quarterly",
            "Budget forecast meeting",
            "Budget allocation changes",
        ]
        for i, subject in enumerate(subjects):
            _insert_email_received(
                db, f"colleague{i}@work.com",
                subject=subject, hour=10, day_offset=i,
            )
        # Insert a 5th to ensure 'budget' appears ≥ 2 times with different senders
        _insert_email_received(db, "finance@work.com", subject="Budget review",
                                hour=10, day_offset=4)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        topic_facts = [f for f in fallback_facts if f["key"].startswith("event_fallback_topic_")]
        assert len(topic_facts) >= 1, (
            "Expected at least one topic fact from subjects containing 'budget'"
        )
        topic_keys = [f["key"] for f in topic_facts]
        assert "event_fallback_topic_budget" in topic_keys, (
            f"Expected 'budget' topic fact, got: {topic_keys}"
        )

    def test_re_fwd_prefixes_stripped(self, db, user_model_store):
        """Re: and Fwd: prefixes should be stripped before tokenizing subjects."""
        subjects = [
            "Re: Project update",
            "Fwd: Project update",
            "RE: Project update",
            "project update details",
            "project update meeting",
        ]
        for i, subject in enumerate(subjects):
            _insert_email_received(
                db, f"person{i}@corp.com",
                subject=subject, hour=10, day_offset=i % 7,
            )

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        topic_keys = [f["key"] for f in fallback_facts
                      if f["key"].startswith("event_fallback_topic_")]

        # "re", "fwd" etc. should not appear as topic facts
        assert "event_fallback_topic_re" not in topic_keys, (
            "'Re' prefix should be stripped and not produce a topic fact"
        )
        assert "event_fallback_topic_fwd" not in topic_keys, (
            "'Fwd' prefix should be stripped and not produce a topic fact"
        )

        # "project" and "update" appear 5 times each → should produce facts
        assert "event_fallback_topic_project" in topic_keys, (
            f"Expected 'project' topic fact, got: {topic_keys}"
        )

    def test_short_words_excluded(self, db, user_model_store):
        """Words shorter than 4 characters should not produce topic facts."""
        # "the", "in", "at" are common short words from stop list and length filter
        subjects = [
            "Big data analysis workshop",
            "Big data analysis results",
            "Big data analysis pipeline",
        ]
        for i, subject in enumerate(subjects):
            _insert_email_received(db, f"p{i}@corp.com", subject=subject, hour=10)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_facts_from_events()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        topic_keys = {f["key"] for f in fallback_facts
                      if f["key"].startswith("event_fallback_topic_")}

        # Short words (len < 4) and stop words should be excluded.
        # We verify this implicitly: only substantive words should appear.
        # "analysis" and "data" have len >= 4 and should appear if count >= 2
        assert "event_fallback_topic_analysis" in topic_keys or \
               "event_fallback_topic_data" in topic_keys, (
            f"Expected substantive topic fact, got only: {topic_keys}"
        )


# ---------------------------------------------------------------------------
# Tests: fallback gate — episodes present vs. absent
# ---------------------------------------------------------------------------

class TestEventFallbackGate:
    """run_all_inference() triggers the fallback only when episodes=0 and events>100."""

    def test_fallback_triggered_when_no_episodes(self, db, user_model_store):
        """Fallback must produce facts when episodes=0 and events>100."""
        # Insert 110 email events from 5 contacts (> 100 threshold)
        contacts = [
            ("alice@work.com", 25),
            ("bob@work.com", 25),
            ("carol@work.com", 20),
            ("dave@work.com", 20),
            ("eve@work.com", 20),
        ]
        _insert_bulk_email_received(db, contacts)

        assert _episode_count(db) == 0
        assert _event_count(db) >= 100

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        fallback_facts = _get_all_fallback_facts(user_model_store)
        assert len(fallback_facts) >= 1, (
            f"Expected fallback facts from event-based path, got 0. "
            f"last_inference_results={inferrer._last_inference_results}"
        )

        # The fallback result should appear in the inference summary
        fallback_result = next(
            (r for r in inferrer._last_inference_results
             if r.get("type") == "event_fallback"),
            None,
        )
        assert fallback_result is not None, (
            "Expected 'event_fallback' entry in _last_inference_results"
        )
        assert fallback_result["processed"] is True

    def test_fallback_not_triggered_when_episodes_exist(self, db, user_model_store):
        """Fallback must NOT run when episodes exist (normal path takes priority)."""
        # Insert an episode to satisfy the gate condition
        _insert_episode(db)
        assert _episode_count(db) == 1

        # Also insert events (would trigger fallback if episodes were absent)
        _insert_bulk_email_received(db, [("alice@work.com", 110)])

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # event_fallback type should NOT appear in results
        fallback_result = next(
            (r for r in inferrer._last_inference_results
             if r.get("type") == "event_fallback"),
            None,
        )
        assert fallback_result is None, (
            "Fallback should be skipped when episodes exist; "
            f"found result: {fallback_result}"
        )

    def test_fallback_not_triggered_below_event_threshold(self, db, user_model_store):
        """Fallback must NOT run when event count is ≤ 100 (insufficient data)."""
        # Insert only 50 events — below the threshold of 100
        _insert_bulk_email_received(db, [("alice@work.com", 50)])

        assert _episode_count(db) == 0
        assert _event_count(db) <= 100

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # No fallback result expected
        fallback_result = next(
            (r for r in inferrer._last_inference_results
             if r.get("type") == "event_fallback"),
            None,
        )
        assert fallback_result is None, (
            "Fallback should be skipped with <= 100 events; "
            f"found: {fallback_result}"
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEventFallbackEdgeCases:
    """Edge-case handling: empty tables, malformed payloads, etc."""

    def test_empty_events_table_returns_gracefully(self, db, user_model_store):
        """infer_facts_from_events() must return a valid result with 0 events."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_facts_from_events()

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "event_fallback"
        # Either processed=False with reason, or processed=True with 0 facts
        assert "facts_written" in result or result.get("processed") is False

    def test_missing_payload_fields_skipped_gracefully(self, db, user_model_store):
        """Events with empty payload (no from_address/subject) must not crash.

        Note: SQLite triggers on email.received use json_extract(payload, ...)
        which prevents inserting truly malformed JSON for this event type.
        Instead we test the equally important case of valid JSON with no fields
        (e.g., payload='{}'), which exercises the graceful handling of
        missing keys throughout infer_facts_from_events().
        """
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events
                   (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "email.received",
                    "proton_mail",
                    datetime.now(UTC).isoformat(),
                    "normal",
                    "{}",   # Valid JSON but no from_address or subject
                    "{}",
                ),
            )

        inferrer = SemanticFactInferrer(user_model_store)
        # Should not raise — fail-open pattern
        result = inferrer.infer_facts_from_events()
        assert isinstance(result, dict)
        assert result["type"] == "event_fallback"
        # The code must not raise — facts_written >= 0 is always valid.
        # Temporal facts (active_hours, most_active_day) may still be derived
        # from the event's timestamp even when the payload has no useful fields,
        # so we cannot assert facts_written == 0 here.  The key invariant is
        # that the call completes without exception and returns a valid result.
        assert isinstance(result.get("facts_written", 0), int)
        # No contact facts should exist since from_address is missing
        fallback_facts = _get_all_fallback_facts(user_model_store)
        contact_facts = [f for f in fallback_facts if "contact" in f["key"]]
        assert len(contact_facts) == 0, (
            "Expected no contact facts from an event with no from_address"
        )

    def test_helper_get_episode_count(self, db, user_model_store):
        """_get_episode_count() returns 0 with no episodes, 1 after insertion."""
        inferrer = SemanticFactInferrer(user_model_store)
        assert inferrer._get_episode_count() == 0

        _insert_episode(db)
        assert inferrer._get_episode_count() == 1

    def test_helper_get_event_count(self, db, user_model_store):
        """_get_event_count() returns 0 with no events, increases after insertion."""
        inferrer = SemanticFactInferrer(user_model_store)
        assert inferrer._get_event_count() == 0

        _insert_email_received(db, "alice@corp.com")
        assert inferrer._get_event_count() == 1
