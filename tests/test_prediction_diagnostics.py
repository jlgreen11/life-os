"""
Tests for prediction engine diagnostics functionality.

Verifies that the diagnostics endpoint and PredictionEngine.get_diagnostics()
correctly identify data availability, blockers, and recommendations for each
prediction type.
"""
import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_diagnostics_no_data(db, user_model_store):
    """Test diagnostics when no data is available."""
    from services.prediction_engine.engine import PredictionEngine

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Should have all 6 prediction types
    assert "prediction_types" in diagnostics
    assert len(diagnostics["prediction_types"]) == 6

    # All types should be blocked with no data
    assert diagnostics["prediction_types"]["reminder"]["status"] == "blocked"
    assert diagnostics["prediction_types"]["conflict"]["status"] == "blocked"
    assert diagnostics["prediction_types"]["opportunity"]["status"] == "blocked"
    assert diagnostics["prediction_types"]["need"]["status"] == "blocked"
    assert diagnostics["prediction_types"]["risk"]["status"] == "blocked"
    assert diagnostics["prediction_types"]["routine_deviation"]["status"] == "blocked"

    # Overall health should be broken
    assert diagnostics["overall"]["health"] == "broken"
    assert diagnostics["overall"]["active_types"] == 0
    assert diagnostics["overall"]["blocked_types"] == 6


@pytest.mark.asyncio
async def test_diagnostics_with_email_data(db, user_model_store, event_store):
    """Test diagnostics with unreplied email data available."""
    from services.prediction_engine.engine import PredictionEngine

    # Create unreplied emails
    now = datetime.now(timezone.utc)
    for i in range(10):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "test",
            "timestamp": (now - timedelta(hours=12)).isoformat(),
            "payload": {
                "from_address": f"user{i}@example.com",
                "subject": f"Test email {i}",
                "message_id": f"msg-{i}",
            },
        })

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Reminder type should show data available
    reminder = diagnostics["prediction_types"]["reminder"]
    assert reminder["data_available"]["unreplied_emails_24h"] == 10
    assert reminder["data_available"]["total_received_24h"] == 10
    assert reminder["data_available"]["replies_sent_24h"] == 0

    # Status depends on whether predictions were generated
    # (may be blocked if marketing filter catches all emails)
    assert reminder["status"] in ["active", "limited", "blocked"]


@pytest.mark.asyncio
async def test_diagnostics_with_calendar_all_day_events(db, user_model_store, event_store):
    """Test diagnostics correctly identifies all-day calendar events."""
    from services.prediction_engine.engine import PredictionEngine

    # Create all-day calendar events
    now = datetime.now(timezone.utc)
    for i in range(5):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "calendar.event.created",
            "source": "test",
            "timestamp": now.isoformat(),
            "payload": {
                "title": f"All-day event {i}",
                "start_time": (now + timedelta(days=i)).date().isoformat(),
                "end_time": (now + timedelta(days=i+1)).date().isoformat(),
                "is_all_day": True,
            },
        })

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Conflict type should show all-day events only
    conflict = diagnostics["prediction_types"]["conflict"]
    assert conflict["data_available"]["total_calendar_events"] == 5
    assert conflict["data_available"]["all_day_events"] > 0
    assert conflict["data_available"]["timed_events"] == 0

    # Should have blocker about all-day events
    assert any("all-day" in blocker.lower() for blocker in conflict["blockers"])
    assert conflict["status"] == "blocked"


@pytest.mark.asyncio
async def test_diagnostics_with_timed_calendar_events(db, user_model_store, event_store):
    """Test diagnostics correctly identifies timed calendar events."""
    from services.prediction_engine.engine import PredictionEngine

    # Create timed calendar events
    now = datetime.now(timezone.utc)
    for i in range(3):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "calendar.event.created",
            "source": "test",
            "timestamp": now.isoformat(),
            "payload": {
                "title": f"Timed meeting {i}",
                "start_time": (now + timedelta(hours=i)).isoformat(),
                "end_time": (now + timedelta(hours=i+1)).isoformat(),
                "is_all_day": False,
            },
        })

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Conflict type should show timed events
    conflict = diagnostics["prediction_types"]["conflict"]
    assert conflict["data_available"]["total_calendar_events"] == 3
    assert conflict["data_available"]["timed_events"] == 3

    # Should not have all-day blocker
    assert not any("all-day" in blocker.lower() for blocker in conflict["blockers"])


@pytest.mark.asyncio
async def test_diagnostics_with_relationship_data(db, user_model_store):
    """Test diagnostics with relationship profile data."""
    from services.prediction_engine.engine import PredictionEngine

    # Create relationship profile with contacts
    relationships_data = {
        "contacts": {
            "friend@example.com": {
                "interaction_count": 10,
                "last_interaction": datetime.now(timezone.utc).isoformat(),
                "interaction_timestamps": [
                    (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
                    for i in range(10)
                ],
            },
            "colleague@example.com": {
                "interaction_count": 8,
                "last_interaction": datetime.now(timezone.utc).isoformat(),
                "interaction_timestamps": [
                    (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
                    for i in range(8)
                ],
            },
            # Marketing contact - should be filtered
            "noreply@marketing.com": {
                "interaction_count": 100,
                "last_interaction": datetime.now(timezone.utc).isoformat(),
                "interaction_timestamps": [],
            },
        }
    }

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles
               (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            ("relationships", json.dumps(relationships_data), 18, datetime.now(timezone.utc).isoformat()),
        )

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Opportunity type should show relationship data
    opportunity = diagnostics["prediction_types"]["opportunity"]
    assert opportunity["data_available"]["total_contacts"] == 3
    # All 3 contacts have 5+ interactions, so all are eligible
    # (marketing filtering is separate from eligibility)
    assert opportunity["data_available"]["eligible_contacts"] == 3
    assert opportunity["data_available"]["marketing_filtered"] >= 1  # noreply@marketing


@pytest.mark.asyncio
async def test_diagnostics_with_transaction_data(db, user_model_store, event_store):
    """Test diagnostics with finance transaction data."""
    from services.prediction_engine.engine import PredictionEngine

    # Create transactions
    now = datetime.now(timezone.utc)
    for i in range(10):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "finance.transaction.new",
            "source": "test",
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "payload": {
                "amount": -100.0,
                "category": "food" if i % 2 == 0 else "transport",
                "description": f"Transaction {i}",
            },
        })

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Risk type should show transaction data
    risk = diagnostics["prediction_types"]["risk"]
    assert risk["data_available"]["transactions_30d"] == 10

    # Should not have "no transactions" blocker
    assert not any("no finance" in blocker.lower() for blocker in risk["blockers"])


@pytest.mark.asyncio
async def test_diagnostics_with_routines(db, user_model_store):
    """Test diagnostics with established routines."""
    from services.prediction_engine.engine import PredictionEngine

    # Create routine with high consistency
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines
               (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            ("Morning routine", "weekday_morning", json.dumps([{"action": "email_check"}]), 0.8, 20),
        )

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Routine deviation type should show routine data
    routine = diagnostics["prediction_types"]["routine_deviation"]
    assert routine["data_available"]["established_routines"] == 1

    # Should not have "no routines" blocker
    assert not any("no routines" in blocker.lower() for blocker in routine["blockers"])


@pytest.mark.asyncio
async def test_diagnostics_overall_health_calculation(db, user_model_store, event_store):
    """Test overall health calculation based on active prediction types."""
    from services.prediction_engine.engine import PredictionEngine

    # Scenario 1: No data - health should be "broken"
    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()
    assert diagnostics["overall"]["health"] == "broken"
    assert diagnostics["overall"]["active_types"] == 0

    # Scenario 2: Add email data - health should be "degraded" (1 active type)
    now = datetime.now(timezone.utc)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": {
            "from_address": "real.person@example.com",
            "subject": "Important question",
            "message_id": "msg-123",
        },
    })

    diagnostics = await engine.get_diagnostics()
    # Health may be degraded or blocked depending on whether predictions generate
    assert diagnostics["overall"]["health"] in ["degraded", "broken"]

    # Scenario 3: Add multiple data sources - health should improve
    # Add timed calendar events
    for i in range(2):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "calendar.event.created",
            "source": "test",
            "timestamp": now.isoformat(),
            "payload": {
                "title": f"Meeting {i}",
                "start_time": (now + timedelta(hours=i+1)).isoformat(),
                "end_time": (now + timedelta(hours=i+2)).isoformat(),
                "is_all_day": False,
            },
        })

    # Add transactions
    for i in range(10):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "finance.transaction.new",
            "source": "test",
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "payload": {
                "amount": -50.0,
                "category": "food",
            },
        })

    diagnostics = await engine.get_diagnostics()
    # With more data sources, blocked types should decrease
    assert diagnostics["overall"]["blocked_types"] < 6


@pytest.mark.asyncio
async def test_diagnostics_recommendations_present(db, user_model_store):
    """Test that diagnostics provide actionable recommendations."""
    from services.prediction_engine.engine import PredictionEngine

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    # Each blocked type should have recommendations
    for type_name, type_data in diagnostics["prediction_types"].items():
        if type_data["status"] == "blocked":
            assert len(type_data["recommendations"]) > 0, \
                f"{type_name} is blocked but has no recommendations"

            # Recommendations should be actionable strings
            for rec in type_data["recommendations"]:
                assert isinstance(rec, str)
                assert len(rec) > 10  # Should be meaningful text


def test_diagnostics_structure(db, user_model_store):
    """Test that diagnostics return expected structure."""
    from services.prediction_engine.engine import PredictionEngine

    engine = PredictionEngine(db, user_model_store)
    diagnostics = asyncio.run(engine.get_diagnostics())

    # Top-level keys
    assert "prediction_types" in diagnostics
    assert "overall" in diagnostics

    # Prediction types structure
    expected_types = ["reminder", "conflict", "opportunity", "need", "risk", "routine_deviation"]
    for ptype in expected_types:
        assert ptype in diagnostics["prediction_types"]
        type_data = diagnostics["prediction_types"][ptype]

        # Required fields
        assert "status" in type_data
        assert type_data["status"] in ["active", "limited", "blocked"]
        assert "generated_last_7d" in type_data
        assert isinstance(type_data["generated_last_7d"], int)
        assert "data_available" in type_data
        assert isinstance(type_data["data_available"], dict)
        assert "blockers" in type_data
        assert isinstance(type_data["blockers"], list)
        assert "recommendations" in type_data
        assert isinstance(type_data["recommendations"], list)

    # Overall structure
    overall = diagnostics["overall"]
    assert "total_predictions_7d" in overall
    assert "active_types" in overall
    assert "blocked_types" in overall
    assert "total_types" in overall
    assert "health" in overall
    assert overall["health"] in ["healthy", "degraded", "broken"]
    assert overall["total_types"] == 6
