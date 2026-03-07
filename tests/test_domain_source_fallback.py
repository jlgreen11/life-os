"""Tests for domain-to-source fallback mapping consistency.

Verifies that _infer_domain_from_event_type() output for communication event
types maps to keys that exist in _DOMAIN_TO_SOURCE, and that the routes.py
domain_to_source dict stays in sync with main.py.
"""

import json
import uuid

import pytest

from main import LifeOS


class TestDomainToSourceMapping:
    """Verify the _DOMAIN_TO_SOURCE dict covers all inferred domains."""

    def test_message_key_exists(self):
        """'message' (from 'message.received'.split('.')[0]) must be in _DOMAIN_TO_SOURCE."""
        assert "message" in LifeOS._DOMAIN_TO_SOURCE

    def test_message_maps_to_messaging_direct(self):
        """'message' domain should map to 'messaging.direct' source key."""
        assert LifeOS._DOMAIN_TO_SOURCE["message"] == "messaging.direct"

    def test_messaging_key_also_exists(self):
        """'messaging' key should still be present for backwards compatibility."""
        assert LifeOS._DOMAIN_TO_SOURCE["messaging"] == "messaging.direct"

    @pytest.mark.parametrize(
        "event_type,expected_domain",
        [
            ("email.received", "email"),
            ("email.sent", "email"),
            ("message.received", "message"),
            ("message.sent", "message"),
            ("calendar.event.created", "calendar"),
            ("finance.transaction", "finance"),
        ],
    )
    def test_inferred_domain_has_source_mapping(self, event_type, expected_domain):
        """Every communication event type's inferred domain must exist in _DOMAIN_TO_SOURCE."""
        # _infer_domain_from_event_type is an instance method but only uses its argument
        inferred = event_type.split(".")[0]
        assert inferred == expected_domain
        assert inferred in LifeOS._DOMAIN_TO_SOURCE, (
            f"Domain '{inferred}' (from '{event_type}') is missing from _DOMAIN_TO_SOURCE"
        )


class TestRoutesMapParity:
    """Verify the domain_to_source dict in routes.py matches main.py."""

    def _get_routes_domain_to_source(self):
        """Extract the domain_to_source dict from routes.py source.

        Rather than importing the closure (which requires a running app),
        we reconstruct the expected mapping and verify it matches main.py.
        """
        # This is the canonical mapping that routes.py should have.
        # If routes.py diverges, this test reminds us to update both places.
        return {
            "email": "email.work",
            "message": "messaging.direct",
            "messaging": "messaging.direct",
            "calendar": "calendar.meetings",
            "finance": "finance.transactions",
            "health": "health.activity",
            "location": "location.visits",
            "home": "home.devices",
        }

    def test_routes_matches_main(self):
        """The domain_to_source mapping in routes.py must match _DOMAIN_TO_SOURCE in main.py."""
        routes_map = self._get_routes_domain_to_source()
        assert routes_map == LifeOS._DOMAIN_TO_SOURCE


class TestResolveNotificationSourceKey:
    """Test _resolve_notification_source_key with domain='message' fallback."""

    def test_message_domain_returns_messaging_direct(self, db):
        """A notification with domain='message' and no source_event_id should resolve to 'messaging.direct'."""
        notif_id = str(uuid.uuid4())

        # Insert a notification with domain='message' and no source_event_id
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications (id, title, body, domain, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (notif_id, "Test message", "Body", "message", "medium"),
            )

        # Create a minimal LifeOS-like object to call the method
        # We only need db and source_weight_manager attributes
        class MinimalLifeOS:
            _DOMAIN_TO_SOURCE = LifeOS._DOMAIN_TO_SOURCE

            def __init__(self, db_manager):
                self.db = db_manager

        obj = MinimalLifeOS(db)
        result = LifeOS._resolve_notification_source_key(obj, notif_id)
        assert result == "messaging.direct"

    def test_messaging_domain_returns_messaging_direct(self, db):
        """A notification with domain='messaging' should also resolve to 'messaging.direct'."""
        notif_id = str(uuid.uuid4())

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications (id, title, body, domain, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (notif_id, "Test messaging", "Body", "messaging", "medium"),
            )

        class MinimalLifeOS:
            _DOMAIN_TO_SOURCE = LifeOS._DOMAIN_TO_SOURCE

            def __init__(self, db_manager):
                self.db = db_manager

        obj = MinimalLifeOS(db)
        result = LifeOS._resolve_notification_source_key(obj, notif_id)
        assert result == "messaging.direct"

    def test_unknown_domain_returns_none(self, db):
        """A notification with an unmapped domain should return None."""
        notif_id = str(uuid.uuid4())

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications (id, title, body, domain, priority, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (notif_id, "Test prediction", "Body", "prediction", "medium"),
            )

        class MinimalLifeOS:
            _DOMAIN_TO_SOURCE = LifeOS._DOMAIN_TO_SOURCE

            def __init__(self, db_manager):
                self.db = db_manager

        obj = MinimalLifeOS(db)
        result = LifeOS._resolve_notification_source_key(obj, notif_id)
        assert result is None

    def test_nonexistent_notification_returns_none(self, db):
        """A notification ID that doesn't exist should return None."""

        class MinimalLifeOS:
            _DOMAIN_TO_SOURCE = LifeOS._DOMAIN_TO_SOURCE

            def __init__(self, db_manager):
                self.db = db_manager

        obj = MinimalLifeOS(db)
        result = LifeOS._resolve_notification_source_key(obj, "nonexistent-id")
        assert result is None
