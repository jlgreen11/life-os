"""
Tests for relationship profile cleanup script.

Verifies that marketing contacts are correctly identified and removed from
the relationships signal profile, while preserving human contacts.
"""

import json
from datetime import datetime, timezone

import pytest

from scripts.clean_relationship_profile_marketing import (
    clean_relationship_profile,
    is_marketing_or_noreply,
)
from storage.manager import DatabaseManager


class TestMarketingFilter:
    """Test the marketing email detection logic."""

    def test_noreply_addresses(self):
        """No-reply addresses should be detected as marketing."""
        assert is_marketing_or_noreply("noreply@example.com")
        assert is_marketing_or_noreply("no-reply@example.com")
        assert is_marketing_or_noreply("donotreply@example.com")
        assert is_marketing_or_noreply("do-not-reply@example.com")
        assert is_marketing_or_noreply("mailer-daemon@example.com")
        assert is_marketing_or_noreply("postmaster@example.com")

    def test_bulk_sender_localparts(self):
        """Bulk sender patterns should be detected as marketing."""
        assert is_marketing_or_noreply("newsletter@example.com")
        assert is_marketing_or_noreply("notifications@example.com")
        assert is_marketing_or_noreply("updates@example.com")
        assert is_marketing_or_noreply("promo@example.com")
        assert is_marketing_or_noreply("marketing@example.com")
        assert is_marketing_or_noreply("email@example.com")
        assert is_marketing_or_noreply("reply@example.com")
        assert is_marketing_or_noreply("orders@example.com")
        assert is_marketing_or_noreply("rewards@example.com")

    def test_embedded_notification_patterns(self):
        """Embedded notification patterns should be detected."""
        assert is_marketing_or_noreply("system-notifications@example.com")
        assert is_marketing_or_noreply("user-alerts@example.com")
        assert is_marketing_or_noreply("account-updates@example.com")
        assert is_marketing_or_noreply("HOA-Notifications@example.com")

    def test_marketing_domains(self):
        """Marketing domain patterns should be detected."""
        assert is_marketing_or_noreply("test@news-us.example.com")
        assert is_marketing_or_noreply("test@email.example.com")
        assert is_marketing_or_noreply("test@reply.example.com")
        assert is_marketing_or_noreply("test@engage.example.com")
        assert is_marketing_or_noreply("test@comms.example.com")  # Critical case

    def test_marketing_service_providers(self):
        """Marketing service provider domains should be detected."""
        assert is_marketing_or_noreply("sender@company.e2ma.net")
        assert is_marketing_or_noreply("sender@company.sendgrid.net")
        assert is_marketing_or_noreply("sender@company.mailchimp.com")
        assert is_marketing_or_noreply("sender@company.hubspot.com")

    def test_real_world_marketing_addresses(self):
        """Real marketing addresses from production should be detected."""
        # These are actual addresses from the polluted relationships profile
        assert is_marketing_or_noreply("callofduty@comms.activision.com")
        assert is_marketing_or_noreply("rei_email@email.rei.com")
        assert is_marketing_or_noreply("newsletter@company.com")
        assert is_marketing_or_noreply("notifications@service.com")

    def test_human_addresses_not_flagged(self):
        """Real human email addresses should NOT be flagged as marketing."""
        assert not is_marketing_or_noreply("john.doe@example.com")
        assert not is_marketing_or_noreply("sarah.smith@company.com")
        assert not is_marketing_or_noreply("alice@startup.io")
        assert not is_marketing_or_noreply("bob_jones@university.edu")
        assert not is_marketing_or_noreply("testuser@example.com")

    def test_edge_cases(self):
        """Edge cases should be handled correctly."""
        # Address with "email" in domain but not as subdomain (should NOT flag)
        assert not is_marketing_or_noreply("user@emailcorp.com")

        # Address with "reply" in username but not at start (should NOT flag)
        assert not is_marketing_or_noreply("sarah.reply@startup.io")

        # Address with "notification" but not in hyphenated form (should NOT flag)
        assert not is_marketing_or_noreply("notificationuser@example.com")


class TestRelationshipProfileCleanup:
    """Test the relationship profile cleanup functionality."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db = DatabaseManager(str(tmp_path))
        # Initialize all database schemas
        db.initialize_all()
        return db

    def test_clean_empty_profile(self, db):
        """Cleanup should handle missing relationships profile gracefully."""
        stats = clean_relationship_profile(db, dry_run=False)

        assert stats["total"] == 0
        assert stats["removed"] == 0
        assert stats["remaining"] == 0

    def test_clean_profile_dry_run(self, db):
        """Dry run should report changes without modifying database."""
        # Create a relationships profile with mixed contacts
        profile_data = {
            "contacts": {
                "john@example.com": {
                    "interaction_count": 10,
                    "last_interaction": datetime.now(timezone.utc).isoformat(),
                },
                "newsletter@marketing.com": {
                    "interaction_count": 50,
                    "last_interaction": datetime.now(timezone.utc).isoformat(),
                },
                "noreply@service.com": {
                    "interaction_count": 30,
                    "last_interaction": datetime.now(timezone.utc).isoformat(),
                },
            }
        }

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles
                   (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    "relationships",
                    json.dumps(profile_data),
                    60,  # Total interaction count
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        # Run cleanup in dry-run mode
        stats = clean_relationship_profile(db, dry_run=True)

        assert stats["total"] == 3
        assert stats["removed"] == 2  # newsletter@ and noreply@
        assert stats["remaining"] == 1  # john@

        # Verify database was NOT modified
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
            ).fetchone()
            data = json.loads(row["data"])
            assert len(data["contacts"]) == 3  # All contacts still present

    def test_clean_profile_removes_marketing_contacts(self, db):
        """Cleanup should remove marketing contacts and preserve human contacts."""
        # Create a polluted relationships profile similar to production
        profile_data = {
            "contacts": {
                # Human contacts (should be preserved)
                "john.doe@example.com": {
                    "interaction_count": 15,
                    "inbound_count": 8,
                    "outbound_count": 7,
                    "last_interaction": "2026-02-15T10:00:00+00:00",
                    "interaction_timestamps": [
                        "2026-02-01T10:00:00+00:00",
                        "2026-02-15T10:00:00+00:00",
                    ],
                },
                "sarah.smith@company.com": {
                    "interaction_count": 20,
                    "inbound_count": 10,
                    "outbound_count": 10,
                    "last_interaction": "2026-02-14T15:30:00+00:00",
                    "interaction_timestamps": [
                        "2026-02-01T08:00:00+00:00",
                        "2026-02-14T15:30:00+00:00",
                    ],
                },
                # Marketing contacts (should be removed)
                "callofduty@comms.activision.com": {
                    "interaction_count": 268,
                    "inbound_count": 268,
                    "outbound_count": 0,
                    "last_interaction": "2026-02-16T11:43:31+00:00",
                },
                "rei_email@email.rei.com": {
                    "interaction_count": 1101,
                    "inbound_count": 1101,
                    "outbound_count": 0,
                    "last_interaction": "2026-02-16T11:43:38+00:00",
                },
                "newsletter@marketing.com": {
                    "interaction_count": 50,
                    "inbound_count": 50,
                    "outbound_count": 0,
                    "last_interaction": "2026-02-10T09:00:00+00:00",
                },
                "noreply@service.com": {
                    "interaction_count": 30,
                    "inbound_count": 30,
                    "outbound_count": 0,
                    "last_interaction": "2026-02-12T14:00:00+00:00",
                },
            }
        }

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles
                   (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    "relationships",
                    json.dumps(profile_data),
                    1484,  # Total interactions
                    "2026-02-16T11:43:38+00:00",
                ),
            )

        # Run cleanup
        stats = clean_relationship_profile(db, dry_run=False)

        assert stats["total"] == 6
        assert stats["removed"] == 4  # All marketing contacts
        assert stats["remaining"] == 2  # Only human contacts

        # Verify the database was updated correctly
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
            ).fetchone()
            data = json.loads(row["data"])
            contacts = data["contacts"]

            # Human contacts should remain
            assert "john.doe@example.com" in contacts
            assert "sarah.smith@company.com" in contacts
            assert contacts["john.doe@example.com"]["interaction_count"] == 15
            assert contacts["sarah.smith@company.com"]["interaction_count"] == 20

            # Marketing contacts should be removed
            assert "callofduty@comms.activision.com" not in contacts
            assert "rei_email@email.rei.com" not in contacts
            assert "newsletter@marketing.com" not in contacts
            assert "noreply@service.com" not in contacts

    def test_clean_all_marketing_profile(self, db):
        """Profile with only marketing contacts should be cleaned to empty."""
        profile_data = {
            "contacts": {
                "newsletter@example.com": {"interaction_count": 50},
                "noreply@service.com": {"interaction_count": 30},
                "callofduty@comms.activision.com": {"interaction_count": 268},
            }
        }

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles
                   (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    "relationships",
                    json.dumps(profile_data),
                    348,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        stats = clean_relationship_profile(db, dry_run=False)

        assert stats["total"] == 3
        assert stats["removed"] == 3
        assert stats["remaining"] == 0

        # Verify profile is now empty
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
            ).fetchone()
            data = json.loads(row["data"])
            assert len(data["contacts"]) == 0

    def test_clean_preserves_all_human_contacts(self, db):
        """Profile with only human contacts should remain unchanged."""
        profile_data = {
            "contacts": {
                "john@example.com": {"interaction_count": 15},
                "sarah@company.com": {"interaction_count": 20},
                "alice@startup.io": {"interaction_count": 10},
            }
        }

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles
                   (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    "relationships",
                    json.dumps(profile_data),
                    45,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        stats = clean_relationship_profile(db, dry_run=False)

        assert stats["total"] == 3
        assert stats["removed"] == 0
        assert stats["remaining"] == 3

        # Verify all contacts are still present
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT data FROM signal_profiles WHERE profile_type = 'relationships'"
            ).fetchone()
            data = json.loads(row["data"])
            assert len(data["contacts"]) == 3
            assert "john@example.com" in data["contacts"]
            assert "sarah@company.com" in data["contacts"]
            assert "alice@startup.io" in data["contacts"]

    def test_cleanup_updates_timestamp(self, db):
        """Cleanup should update the profile's updated_at timestamp."""
        profile_data = {
            "contacts": {
                "john@example.com": {"interaction_count": 10},
                "newsletter@marketing.com": {"interaction_count": 50},
            }
        }

        old_timestamp = "2026-02-15T10:00:00+00:00"

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles
                   (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                ("relationships", json.dumps(profile_data), 60, old_timestamp),
            )

        clean_relationship_profile(db, dry_run=False)

        # Verify timestamp was updated
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT updated_at FROM signal_profiles WHERE profile_type = 'relationships'"
            ).fetchone()
            new_timestamp = row["updated_at"]

            # New timestamp should be different (and more recent)
            assert new_timestamp != old_timestamp
            # Timestamp should be valid ISO format
            datetime.fromisoformat(new_timestamp.replace("Z", "+00:00"))
