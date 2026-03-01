"""
Tests for automatic relationship profile cleanup on startup.

This feature was added in iteration 147 to solve the problem where marketing
contacts from historical data (before the filter existed) were polluting the
relationships profile and preventing relationship maintenance predictions.
"""

import json
import pytest
from datetime import datetime, timezone


def test_cleanup_runs_on_startup_when_needed(db):
    """Verify cleanup runs automatically on startup when profile contains marketing contacts."""
    from main import LifeOS
    from scripts.clean_relationship_profile_marketing import is_marketing_or_noreply

    # Pre-populate the relationships profile with both human and marketing contacts
    marketing_contacts = {
        "no-reply@spireenergy.com": {
            "interaction_count": 229,
            "inbound_count": 229,
            "outbound_count": 0,
            "last_interaction": "2026-02-16T15:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 5000,
            "interaction_timestamps": ["2026-02-16T15:00:00+00:00"],
        },
        "newsletter@example.com": {
            "interaction_count": 150,
            "inbound_count": 150,
            "outbound_count": 0,
            "last_interaction": "2026-02-16T14:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 3000,
            "interaction_timestamps": ["2026-02-16T14:00:00+00:00"],
        },
        "email@email.tivo.com": {
            "interaction_count": 5,
            "inbound_count": 5,
            "outbound_count": 0,
            "last_interaction": "2026-02-16T13:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 2000,
            "interaction_timestamps": ["2026-02-16T13:00:00+00:00"],
        },
    }

    human_contacts = {
        "alice@example.com": {
            "interaction_count": 47,
            "inbound_count": 25,
            "outbound_count": 22,
            "last_interaction": "2026-02-16T12:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 1200,
            "interaction_timestamps": ["2026-02-16T12:00:00+00:00"],
        },
        "bob.smith@company.com": {
            "interaction_count": 92,
            "inbound_count": 48,
            "outbound_count": 44,
            "last_interaction": "2026-02-16T11:00:00+00:00",
            "channels_used": ["google", "slack"],
            "avg_message_length": 800,
            "interaction_timestamps": ["2026-02-16T11:00:00+00:00"],
        },
    }

    # Combine into full profile
    all_contacts = {**marketing_contacts, **human_contacts}
    profile_data = {"contacts": all_contacts}

    # Store the polluted profile
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                "relationships",
                json.dumps(profile_data),
                len(all_contacts),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Verify pre-state: profile contains 5 contacts (3 marketing + 2 human)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    contacts_before = json.loads(row["data"]).get("contacts", {})
    assert len(contacts_before) == 5, "Should start with 5 contacts"
    assert "no-reply@spireenergy.com" in contacts_before
    assert "alice@example.com" in contacts_before

    # Create LifeOS instance with test database
    import asyncio
    from storage.user_model_store import UserModelStore
    ums = UserModelStore(db)
    lifeos = LifeOS(db=db, user_model_store=ums)

    # Run just the cleanup method (simulating what happens during startup)
    asyncio.run(lifeos._clean_relationship_profile_if_needed())

    # Verify post-state: marketing contacts removed, human contacts preserved
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    contacts_after = json.loads(row["data"]).get("contacts", {})

    # Should have only 2 contacts (the human ones)
    assert len(contacts_after) == 2, f"Should have 2 contacts after cleanup, got {len(contacts_after)}"

    # Human contacts should be preserved
    assert "alice@example.com" in contacts_after
    assert "bob.smith@company.com" in contacts_after

    # Marketing contacts should be removed
    assert "no-reply@spireenergy.com" not in contacts_after
    assert "newsletter@example.com" not in contacts_after
    assert "email@email.tivo.com" not in contacts_after


def test_cleanup_skips_when_profile_already_clean(db):
    """Verify cleanup is skipped when < 10% of contacts are marketing."""
    from main import LifeOS

    # Create a clean profile with mostly human contacts
    contacts = {
        f"user{i}@example.com": {
            "interaction_count": 10 + i,
            "inbound_count": 5,
            "outbound_count": 5 + i,
            "last_interaction": "2026-02-16T12:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 1000,
            "interaction_timestamps": ["2026-02-16T12:00:00+00:00"],
        }
        for i in range(20)  # 20 human contacts
    }

    # Add just 1 marketing contact (5% of total)
    contacts["newsletter@example.com"] = {
        "interaction_count": 50,
        "inbound_count": 50,
        "outbound_count": 0,
        "last_interaction": "2026-02-16T11:00:00+00:00",
        "channels_used": ["google"],
        "avg_message_length": 3000,
        "interaction_timestamps": ["2026-02-16T11:00:00+00:00"],
    }

    profile_data = {"contacts": contacts}

    # Store the clean profile
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                "relationships",
                json.dumps(profile_data),
                len(contacts),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Run the cleanup
    import asyncio
    from storage.user_model_store import UserModelStore
    ums = UserModelStore(db)
    lifeos = LifeOS(db=db, user_model_store=ums)
    asyncio.run(lifeos._clean_relationship_profile_if_needed())

    # Verify cleanup was skipped (all contacts still present)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    contacts_after = json.loads(row["data"]).get("contacts", {})

    # Should still have all 21 contacts (cleanup threshold not met)
    assert len(contacts_after) == 21


def test_cleanup_skips_when_profile_empty(db):
    """Verify cleanup handles empty profile gracefully."""
    from main import LifeOS
    from storage.user_model_store import UserModelStore

    # Create empty profile
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                "relationships",
                json.dumps({"contacts": {}}),
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Run the cleanup using the test-scoped db fixture so we don't touch
    # the real (potentially corrupted) production database.
    import asyncio
    ums = UserModelStore(db)
    lifeos = LifeOS(db=db, user_model_store=ums)
    asyncio.run(lifeos._clean_relationship_profile_if_needed())

    # Verify profile is still empty
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    contacts = json.loads(row["data"]).get("contacts", {})
    assert len(contacts) == 0


def test_cleanup_skips_when_profile_missing(db):
    """Verify cleanup handles missing profile gracefully."""
    from main import LifeOS
    from storage.user_model_store import UserModelStore

    # Don't create any profile
    # Run the cleanup using the test-scoped db fixture so we don't touch
    # the real (potentially corrupted) production database.
    import asyncio
    ums = UserModelStore(db)
    lifeos = LifeOS(db=db, user_model_store=ums)
    asyncio.run(lifeos._clean_relationship_profile_if_needed())

    # Verify no profile was created
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    assert row[0] == 0


def test_cleanup_preserves_human_contact_data(db):
    """Verify cleanup preserves all fields for human contacts."""
    from main import LifeOS

    # Create profile with detailed human contact
    contacts = {
        "alice@example.com": {
            "interaction_count": 127,
            "inbound_count": 64,
            "outbound_count": 63,
            "last_interaction": "2026-02-16T12:00:00+00:00",
            "last_inbound_timestamp": "2026-02-16T11:30:00+00:00",
            "channels_used": ["google", "slack"],
            "avg_message_length": 1234,
            "interaction_timestamps": [
                "2026-02-15T10:00:00+00:00",
                "2026-02-16T08:00:00+00:00",
                "2026-02-16T12:00:00+00:00",
            ],
            "response_times_seconds": [3600, 7200, 1800],
            "avg_response_time_seconds": 4200,
        },
        "no-reply@marketing.com": {
            "interaction_count": 500,
            "inbound_count": 500,
            "outbound_count": 0,
            "last_interaction": "2026-02-16T10:00:00+00:00",
            "channels_used": ["google"],
            "avg_message_length": 5000,
            "interaction_timestamps": ["2026-02-16T10:00:00+00:00"],
        },
    }

    profile_data = {"contacts": contacts}

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (
                "relationships",
                json.dumps(profile_data),
                len(contacts),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Run the cleanup
    import asyncio
    from storage.user_model_store import UserModelStore
    ums = UserModelStore(db)
    lifeos = LifeOS(db=db, user_model_store=ums)
    asyncio.run(lifeos._clean_relationship_profile_if_needed())

    # Verify human contact data is fully preserved
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
        ).fetchone()

    contacts_after = json.loads(row["data"]).get("contacts", {})
    alice = contacts_after["alice@example.com"]

    # All fields should be preserved exactly
    assert alice["interaction_count"] == 127
    assert alice["inbound_count"] == 64
    assert alice["outbound_count"] == 63
    assert alice["channels_used"] == ["google", "slack"]
    assert alice["avg_message_length"] == 1234
    assert len(alice["interaction_timestamps"]) == 3
    assert alice["response_times_seconds"] == [3600, 7200, 1800]
    assert alice["avg_response_time_seconds"] == 4200
