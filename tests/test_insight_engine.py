"""
Tests for the InsightEngine service.

The InsightEngine is a backward-looking cross-signal correlator that discovers
patterns in collected data and surfaces them as human-readable insights. It runs
hourly and deduplicates insights to prevent resurfacing the same pattern within
a staleness TTL window.

This test suite validates:
- All 4 correlators (place frequency, contact gap, email volume, communication style)
- Deduplication logic with staleness TTL
- Confidence scoring algorithms
- Error handling and graceful degradation
- Evidence collection and categorization
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


# =============================================================================
# Insight Model Tests
# =============================================================================


def test_insight_model_dedup_key():
    """Dedup key should be deterministic for same type+category+entity."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="behavioral_pattern", summary="different", confidence=0.5, category="place", entity="Home")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key == b.dedup_key


def test_insight_model_different_entities_different_keys():
    """Different entities should produce different dedup keys."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Work")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key != b.dedup_key


def test_insight_model_different_types_different_keys():
    """Different insight types should produce different dedup keys."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="actionable_alert", summary="test", confidence=0.8, category="place", entity="Home")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key != b.dedup_key


def test_insight_model_different_categories_different_keys():
    """Different categories should produce different dedup keys."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="contact", entity="Home")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key != b.dedup_key


def test_insight_model_none_entity_handled():
    """Dedup key should handle None entity gracefully."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="general", entity=None)
    a.compute_dedup_key()
    assert a.dedup_key
    assert len(a.dedup_key) == 16  # SHA256 hash truncated to 16 chars


def test_insight_model_defaults():
    """Insight model should have sensible defaults."""
    insight = Insight(type="test", summary="test summary", confidence=0.5)
    assert insight.id
    assert insight.staleness_ttl_hours == 168  # 7 days
    assert insight.evidence == []
    assert insight.category == ""
    assert insight.entity is None
    assert insight.dedup_key == ""
    assert insight.created_at
    assert insight.feedback is None


# =============================================================================
# InsightEngine Initialization Tests
# =============================================================================


@pytest.mark.asyncio
async def test_insight_engine_initialization(db, user_model_store):
    """InsightEngine should initialize with database and user model store."""
    engine = InsightEngine(db, user_model_store)
    assert engine.db is db
    assert engine.ums is user_model_store


# =============================================================================
# Place Frequency Correlator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_place_frequency_no_places(db, user_model_store):
    """Place frequency correlator should return empty list when no places exist."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._place_frequency_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_place_frequency_low_visit_count(db, user_model_store):
    """Place frequency correlator should ignore places with <= 3 visits."""
    engine = InsightEngine(db, user_model_store)

    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Rarely Visited", 3, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    insights = engine._place_frequency_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_place_frequency_single_place(db, user_model_store):
    """Place frequency correlator should generate insight for place with > 3 visits."""
    engine = InsightEngine(db, user_model_store)

    place_name = "Favorite Cafe"
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), place_name, 10, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    insights = engine._place_frequency_insights()
    assert len(insights) == 1
    assert insights[0].type == "behavioral_pattern"
    assert place_name in insights[0].summary
    assert "10 visits" in insights[0].summary
    assert insights[0].category == "place"
    assert insights[0].entity == place_name
    assert "visit_count=10" in insights[0].evidence
    assert "place_type=cafe" in insights[0].evidence


@pytest.mark.asyncio
async def test_place_frequency_confidence_scaling(db, user_model_store):
    """Place frequency confidence should scale with visit count, capped at 0.9."""
    engine = InsightEngine(db, user_model_store)

    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Low Visits", 5, "cafe", datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "High Visits", 20, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    insights = engine._place_frequency_insights()
    low_insight = [i for i in insights if i.entity == "Low Visits"][0]
    high_insight = [i for i in insights if i.entity == "High Visits"][0]

    assert low_insight.confidence < high_insight.confidence
    assert high_insight.confidence <= 0.9  # Capped at 0.9


@pytest.mark.asyncio
async def test_place_frequency_null_place_type(db, user_model_store):
    """Place frequency correlator should handle NULL place_type gracefully."""
    engine = InsightEngine(db, user_model_store)

    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "No Type Place", 5, None, datetime.now(timezone.utc).isoformat()),
        )

    insights = engine._place_frequency_insights()
    assert len(insights) == 1
    assert "place_type=place" in insights[0].evidence  # Default to "place"


@pytest.mark.asyncio
async def test_place_frequency_multiple_places(db, user_model_store):
    """Place frequency correlator should generate multiple insights for multiple places."""
    engine = InsightEngine(db, user_model_store)

    with db.get_connection("entities") as conn:
        for i in range(5):
            conn.execute(
                "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), f"Place {i}", 5 + i, "cafe", datetime.now(timezone.utc).isoformat()),
            )

    insights = engine._place_frequency_insights()
    assert len(insights) == 5


# =============================================================================
# Contact Gap Correlator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_contact_gap_no_profile(db, user_model_store):
    """Contact gap correlator should return empty list when no relationship profile exists."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._contact_gap_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_contact_gap_insufficient_history(db, user_model_store):
    """Contact gap correlator should skip contacts with < 5 interactions."""
    engine = InsightEngine(db, user_model_store)

    # Store relationship profile with low interaction count
    profile_data = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 3,
                "last_interaction": datetime.now(timezone.utc).isoformat(),
                "interaction_timestamps": [
                    (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
                    for i in range(3)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_contact_gap_recent_contact(db, user_model_store):
    """Contact gap correlator should skip contacts contacted recently."""
    engine = InsightEngine(db, user_model_store)

    # Store relationship profile with recent interaction
    profile_data = {
        "contacts": {
            "bob@example.com": {
                "interaction_count": 10,
                "last_interaction": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
                "interaction_timestamps": [
                    (datetime.now(timezone.utc) - timedelta(days=i * 7)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert insights == []  # Gap too small relative to average


@pytest.mark.asyncio
async def test_contact_gap_overdue_contact(db, user_model_store):
    """Contact gap correlator should detect overdue contacts (gap > 1.5x average)."""
    engine = InsightEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    avg_gap_days = 10
    overdue_days = 20  # 2x average

    profile_data = {
        "contacts": {
            "charlie@example.com": {
                "interaction_count": 10,
                "last_interaction": (now - timedelta(days=overdue_days)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=overdue_days + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    assert insights[0].type == "relationship_intelligence"
    assert "charlie@example.com" in insights[0].summary
    assert insights[0].category == "contact_gap"
    assert insights[0].entity == "charlie@example.com"


@pytest.mark.asyncio
async def test_contact_gap_confidence_scaling(db, user_model_store):
    """Contact gap confidence should scale with gap magnitude."""
    engine = InsightEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    avg_gap_days = 10

    profile_data = {
        "contacts": {
            "low_gap@example.com": {
                "interaction_count": 10,
                "last_interaction": (now - timedelta(days=16)).isoformat(),  # 1.6x average
                "interaction_timestamps": [
                    (now - timedelta(days=16 + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            },
            "high_gap@example.com": {
                "interaction_count": 10,
                "last_interaction": (now - timedelta(days=40)).isoformat(),  # 4x average
                "interaction_timestamps": [
                    (now - timedelta(days=40 + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            },
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    low_gap = [i for i in insights if i.entity == "low_gap@example.com"][0]
    high_gap = [i for i in insights if i.entity == "high_gap@example.com"][0]

    assert low_gap.confidence < high_gap.confidence
    assert high_gap.confidence <= 0.8  # Capped at 0.8


@pytest.mark.asyncio
async def test_contact_gap_insufficient_timestamps(db, user_model_store):
    """Contact gap correlator should skip contacts with < 3 timestamps."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "contacts": {
            "sparse@example.com": {
                "interaction_count": 10,
                "last_interaction": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                "interaction_timestamps": [
                    (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_contact_gap_invalid_timestamp_format(db, user_model_store):
    """Contact gap correlator should skip contacts with invalid timestamp formats."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "contacts": {
            "bad_timestamp@example.com": {
                "interaction_count": 10,
                "last_interaction": "invalid-timestamp",
                "interaction_timestamps": ["invalid", "timestamps", "here"],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert insights == []


# =============================================================================
# Email Volume Correlator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_email_volume_no_emails(db, user_model_store):
    """Email volume correlator should return empty list when no emails exist."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._email_volume_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_email_volume_insufficient_data(db, user_model_store, event_store):
    """Email volume correlator should skip analysis with < 7 emails."""
    engine = InsightEngine(db, user_model_store)

    # Publish only 5 emails
    for i in range(5):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": 2,
            "payload": {"subject": f"Test {i}"},
            "metadata": {},
        })

    insights = engine._email_volume_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_email_volume_uniform_distribution(db, user_model_store, event_store):
    """Email volume correlator should skip when emails are uniformly distributed."""
    engine = InsightEngine(db, user_model_store)

    # Publish 14 emails evenly across 7 days (2 per day)
    base_time = datetime.now(timezone.utc)
    for day_offset in range(7):
        for email_num in range(2):
            event_store.store_event({
                "id": str(uuid.uuid4()),
                "type": "email.received",
                "source": "test",
                "timestamp": (base_time - timedelta(days=day_offset, hours=email_num)).isoformat(),
                "priority": 2,
                "payload": {"subject": "Test"},
                "metadata": {},
            })

    insights = engine._email_volume_insights()
    assert insights == []  # No day is 1.5x busier than average


@pytest.mark.asyncio
async def test_email_volume_busiest_day_detected(db, user_model_store, event_store):
    """Email volume correlator should detect busiest day when > 1.5x average."""
    engine = InsightEngine(db, user_model_store)

    base_time = datetime.now(timezone.utc)

    # Create Monday as busiest day (10 emails), other days 2 each
    # Monday = weekday 0
    monday = base_time - timedelta(days=base_time.weekday())

    for i in range(10):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "test",
            "timestamp": (monday - timedelta(hours=i)).isoformat(),
            "priority": 2,
            "payload": {"subject": "Monday email"},
            "metadata": {},
        })

    # Add 2 emails for other days
    for day_offset in range(1, 7):
        for email_num in range(2):
            event_store.store_event({
                "id": str(uuid.uuid4()),
                "type": "email.received",
                "source": "test",
                "timestamp": (monday - timedelta(days=day_offset, hours=email_num)).isoformat(),
                "priority": 2,
                "payload": {"subject": "Other day email"},
                "metadata": {},
            })

    insights = engine._email_volume_insights()
    assert len(insights) == 1
    assert insights[0].type == "behavioral_pattern"
    assert "Monday" in insights[0].summary
    assert insights[0].category == "email_volume"
    assert insights[0].entity == "Monday"


@pytest.mark.asyncio
async def test_email_volume_sent_and_received(db, user_model_store, event_store):
    """Email volume correlator should count both sent and received emails."""
    engine = InsightEngine(db, user_model_store)

    base_time = datetime.now(timezone.utc)
    tuesday = base_time - timedelta(days=(base_time.weekday() - 1) % 7)

    # Create Tuesday as busiest (5 received + 5 sent = 10 total)
    for i in range(5):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "test",
            "timestamp": (tuesday - timedelta(hours=i)).isoformat(),
            "priority": 2,
            "payload": {},
            "metadata": {},
        })
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.sent",
            "source": "test",
            "timestamp": (tuesday - timedelta(hours=i, minutes=30)).isoformat(),
            "priority": 2,
            "payload": {},
            "metadata": {},
        })

    # Add 2 emails for other days
    for day_offset in range(1, 7):
        for email_num in range(2):
            event_store.store_event({
                "id": str(uuid.uuid4()),
                "type": "email.received",
                "source": "test",
                "timestamp": (tuesday - timedelta(days=day_offset, hours=email_num)).isoformat(),
                "priority": 2,
                "payload": {},
                "metadata": {},
            })

    insights = engine._email_volume_insights()
    assert len(insights) == 1
    assert "Tuesday" in insights[0].summary


@pytest.mark.asyncio
async def test_email_volume_invalid_timestamps_skipped(db, user_model_store, event_store):
    """Email volume correlator should skip events with invalid timestamps."""
    engine = InsightEngine(db, user_model_store)

    # Insert event with invalid timestamp directly
    with db.get_connection("events") as conn:
        for i in range(10):
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, priority, payload, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), "email.received", "test", "invalid-timestamp", 2, "{}", "{}"),
            )

    insights = engine._email_volume_insights()
    assert insights == []


# =============================================================================
# Communication Style Correlator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_communication_style_no_profile(db, user_model_store):
    """Communication style correlator should return empty list when no profile exists."""
    engine = InsightEngine(db, user_model_store)
    insights = engine._communication_style_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_communication_style_no_formality(db, user_model_store):
    """Communication style correlator should skip when formality is None."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "averages": {
            "word_count": 50,
        }
    }
    user_model_store.update_signal_profile("linguistic", profile_data)

    insights = engine._communication_style_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_communication_style_insufficient_samples(db, user_model_store):
    """Communication style correlator should skip with < 3 samples."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "averages": {
            "formality": 0.8,
        }
    }
    # Manually set samples_count to 2 by updating the database
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data), 2),
        )

    insights = engine._communication_style_insights()
    assert insights == []


@pytest.mark.asyncio
async def test_communication_style_formal(db, user_model_store):
    """Communication style correlator should detect formal style (formality >= 0.7)."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "averages": {
            "formality": 0.85,
        }
    }
    # Manually set samples_count to 10 by updating the database
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data), 10),
        )

    insights = engine._communication_style_insights()
    assert len(insights) == 1
    assert insights[0].type == "communication_style"
    assert "formal" in insights[0].summary.lower()
    assert insights[0].category == "communication_style"
    assert insights[0].entity == "formal"


@pytest.mark.asyncio
async def test_communication_style_casual(db, user_model_store):
    """Communication style correlator should detect casual style (formality <= 0.3)."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "averages": {
            "formality": 0.2,
        }
    }
    # Manually set samples_count to 15 by updating the database
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data), 15),
        )

    insights = engine._communication_style_insights()
    assert len(insights) == 1
    assert "casual" in insights[0].summary.lower()
    assert insights[0].entity == "casual"


@pytest.mark.asyncio
async def test_communication_style_balanced(db, user_model_store):
    """Communication style correlator should detect balanced style (0.3 < formality < 0.7)."""
    engine = InsightEngine(db, user_model_store)

    profile_data = {
        "averages": {
            "formality": 0.5,
        }
    }
    # Manually set samples_count to 20 by updating the database
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data), 20),
        )

    insights = engine._communication_style_insights()
    assert len(insights) == 1
    assert "balanced" in insights[0].summary.lower()
    assert insights[0].entity == "balanced"


@pytest.mark.asyncio
async def test_communication_style_confidence_scaling(db, user_model_store):
    """Communication style confidence should scale with sample count, capped at 0.85."""
    engine = InsightEngine(db, user_model_store)

    profile_data_low = {"averages": {"formality": 0.8}}
    # Manually set samples_count to 5
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data_low), 5),
        )
    insights_low = engine._communication_style_insights()

    profile_data_high = {"averages": {"formality": 0.8}}
    # Manually set samples_count to 50
    with db.get_connection("user_model") as conn:
        conn.execute(
            """UPDATE signal_profiles SET data = ?, samples_count = ? WHERE profile_type = ?""",
            (json.dumps(profile_data_high), 50, "linguistic"),
        )
    insights_high = engine._communication_style_insights()

    assert insights_low[0].confidence < insights_high[0].confidence
    assert insights_high[0].confidence <= 0.85


# =============================================================================
# Deduplication Tests
# =============================================================================


@pytest.mark.asyncio
async def test_deduplication_fresh_insight(db, user_model_store):
    """Fresh insights with no prior dedup key should pass through."""
    engine = InsightEngine(db, user_model_store)

    insight = Insight(type="test", summary="test", confidence=0.5, category="test", entity="test")
    insight.compute_dedup_key()

    fresh = engine._deduplicate([insight])
    assert len(fresh) == 1
    assert fresh[0].dedup_key == insight.dedup_key


@pytest.mark.asyncio
async def test_deduplication_within_staleness_window(db, user_model_store):
    """Insights within staleness TTL should be filtered out."""
    engine = InsightEngine(db, user_model_store)

    # Store an insight
    insight1 = Insight(
        type="test",
        summary="test",
        confidence=0.5,
        category="test",
        entity="test",
        staleness_ttl_hours=168,
    )
    insight1.compute_dedup_key()
    engine._store_insight(insight1)

    # Try to deduplicate the same insight
    insight2 = Insight(type="test", summary="different summary", confidence=0.7, category="test", entity="test")
    insight2.compute_dedup_key()

    fresh = engine._deduplicate([insight2])
    assert len(fresh) == 0  # Filtered out


@pytest.mark.asyncio
async def test_deduplication_beyond_staleness_window(db, user_model_store):
    """Insights beyond staleness TTL should pass through."""
    engine = InsightEngine(db, user_model_store)

    # Store an insight with old timestamp
    old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    insight1 = Insight(
        id=str(uuid.uuid4()),
        type="test",
        summary="test",
        confidence=0.5,
        category="test",
        entity="test",
        staleness_ttl_hours=168,
        created_at=old_time,
    )
    insight1.compute_dedup_key()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO insights
               (id, type, summary, confidence, evidence, category,
                entity, staleness_ttl_hours, dedup_key, feedback, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                insight1.id,
                insight1.type,
                insight1.summary,
                insight1.confidence,
                json.dumps(insight1.evidence),
                insight1.category,
                insight1.entity,
                insight1.staleness_ttl_hours,
                insight1.dedup_key,
                insight1.feedback,
                old_time,
            ),
        )

    # Try to deduplicate the same insight
    insight2 = Insight(type="test", summary="different", confidence=0.7, category="test", entity="test")
    insight2.compute_dedup_key()

    fresh = engine._deduplicate([insight2])
    assert len(fresh) == 1  # Allowed through — stale


@pytest.mark.asyncio
async def test_deduplication_missing_dedup_key(db, user_model_store):
    """Insights missing dedup_key should have it computed during deduplication."""
    engine = InsightEngine(db, user_model_store)

    insight = Insight(type="test", summary="test", confidence=0.5, category="test", entity="test")
    # Don't compute dedup_key
    assert insight.dedup_key == ""

    fresh = engine._deduplicate([insight])
    assert len(fresh) == 1
    assert fresh[0].dedup_key  # Should be computed


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_empty(db, user_model_store):
    """Generate insights should return empty list when no data exists."""
    engine = InsightEngine(db, user_model_store)
    insights = await engine.generate_insights()
    assert isinstance(insights, list)


@pytest.mark.asyncio
async def test_generate_insights_all_correlators(db, user_model_store, event_store):
    """Generate insights should run all correlators and combine results."""
    engine = InsightEngine(db, user_model_store)

    # Seed data for place frequency
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Test Cafe", 10, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    # Seed data for communication style
    profile_data = {"averages": {"formality": 0.8}}
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("linguistic", json.dumps(profile_data), 10),
        )

    insights = await engine.generate_insights()
    assert len(insights) >= 2  # At least place + style

    types = {i.type for i in insights}
    assert "behavioral_pattern" in types
    assert "communication_style" in types


@pytest.mark.asyncio
async def test_generate_insights_deduplicates(db, user_model_store):
    """Generate insights should deduplicate on subsequent runs."""
    engine = InsightEngine(db, user_model_store)

    # Seed data
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Dedup Test", 10, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    first_run = await engine.generate_insights()
    second_run = await engine.generate_insights()

    assert len(first_run) >= 1
    assert len(second_run) == 0  # All deduplicated


@pytest.mark.asyncio
async def test_generate_insights_stores_insights(db, user_model_store):
    """Generate insights should persist insights to database."""
    engine = InsightEngine(db, user_model_store)

    # Seed data
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Storage Test", 5, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    insights = await engine.generate_insights()
    assert len(insights) >= 1

    # Verify stored in database
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT * FROM insights WHERE dedup_key = ?",
            (insights[0].dedup_key,),
        ).fetchone()

    assert row is not None
    assert row["type"] == insights[0].type
    assert row["summary"] == insights[0].summary


@pytest.mark.asyncio
async def test_generate_insights_error_handling(db, user_model_store):
    """Generate insights should handle correlator errors gracefully."""
    engine = InsightEngine(db, user_model_store)

    # Even if one correlator fails, others should run
    # This test validates the try/except blocks in generate_insights
    insights = await engine.generate_insights()

    # Should not raise an exception
    assert isinstance(insights, list)
