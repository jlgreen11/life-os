"""
Tests for the avoids_phrases cross-contact comparative analysis feature.

The avoids_phrases field identifies words/phrases that are commonly used in
the user's outbound templates to OTHER contacts but are conspicuously absent
from the current contact's template — revealing per-contact communication
style differences.
"""

import json

import pytest

from services.signal_extractor.relationship import RelationshipExtractor
from storage.user_model_store import UserModelStore


@pytest.fixture()
def ums(db):
    """UserModelStore without event bus (simpler for unit tests)."""
    return UserModelStore(db)


@pytest.fixture()
def extractor(db, ums):
    """RelationshipExtractor wired to temp DB and user model store."""
    return RelationshipExtractor(db, ums)


def _insert_template(db, contact_id: str, common_phrases: list[str], context: str = "user_to_contact"):
    """Helper to insert a communication template directly into the DB.

    Bypasses the full extraction pipeline so we can set up controlled
    test scenarios for the cross-contact avoids_phrases analysis.
    """
    import hashlib

    template_id = hashlib.sha256(f"{contact_id}:email:out".encode()).hexdigest()[:16]
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO communication_templates
               (id, context, contact_id, channel, formality, typical_length,
                uses_emoji, common_phrases, avoids_phrases, tone_notes,
                example_message_ids, samples_analyzed)
               VALUES (?, ?, ?, 'email', 0.5, 100.0, 0, ?, '[]', '[]', '[]', 10)""",
            (template_id, context, contact_id, json.dumps(common_phrases)),
        )


class TestColdStartGuard:
    """avoids_phrases should return empty when insufficient templates exist."""

    def test_no_other_templates(self, extractor):
        """With zero other templates, returns empty list."""
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello", "thanks"])
        assert result == []

    def test_one_other_template(self, db, extractor):
        """With only 1 other template, returns empty (need 3+ for comparison)."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "meeting"])
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello"])
        assert result == []

    def test_two_other_templates(self, db, extractor):
        """With only 2 other templates, returns empty (need 3+ for comparison)."""
        _insert_template(db, "bob@example.com", ["hello", "thanks"])
        _insert_template(db, "carol@example.com", ["hello", "thanks"])
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello"])
        assert result == []

    def test_three_other_templates_triggers(self, db, extractor):
        """With 3 other templates, cross-contact analysis kicks in."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "meeting"])
        _insert_template(db, "carol@example.com", ["hello", "thanks", "update"])
        _insert_template(db, "dave@example.com", ["hello", "thanks", "review"])
        # "thanks" appears in 3/3 templates (100%) but not in current contact's phrases
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello"])
        assert "thanks" in result


class TestThresholdLogic:
    """Phrases must appear in 40%+ of other templates to count as avoided."""

    def test_phrase_above_threshold(self, db, extractor):
        """Phrase in 3/5 other templates (60%) should appear in avoids."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "meeting"])
        _insert_template(db, "carol@example.com", ["hello", "thanks", "update"])
        _insert_template(db, "dave@example.com", ["hello", "thanks", "review"])
        _insert_template(db, "eve@example.com", ["hello", "project", "deadline"])
        _insert_template(db, "frank@example.com", ["hello", "budget", "quarterly"])

        # "thanks" in 3/5 = 60% >= 40% threshold and absent from current
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello", "report"])
        assert "thanks" in result

    def test_phrase_below_threshold(self, db, extractor):
        """Phrase in 1/5 other templates (20%) should NOT appear in avoids."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "meeting"])
        _insert_template(db, "carol@example.com", ["hello", "update", "review"])
        _insert_template(db, "dave@example.com", ["hello", "project", "review"])
        _insert_template(db, "eve@example.com", ["hello", "project", "deadline"])
        _insert_template(db, "frank@example.com", ["hello", "budget", "quarterly"])

        # "thanks" only in 1/5 = 20% < 40% threshold
        result = extractor._compute_avoids_phrases("alice@example.com", [])
        assert "thanks" not in result

    def test_minimum_two_templates_threshold(self, db, extractor):
        """Even at 40%, the minimum is 2 templates (max(2, count*0.4))."""
        # With 3 templates, 40% = 1.2 → max(2, 1.2) = 2, so need 2+ templates
        _insert_template(db, "bob@example.com", ["hello", "thanks"])
        _insert_template(db, "carol@example.com", ["hello", "update"])
        _insert_template(db, "dave@example.com", ["hello", "review"])

        # "thanks" in 1/3 (33%) but also below the min-2 threshold
        result = extractor._compute_avoids_phrases("alice@example.com", [])
        assert "thanks" not in result


class TestExcludesCurrentPhrases:
    """Phrases already in the current contact's common_phrases are not 'avoided'."""

    def test_phrase_in_current_not_avoided(self, db, extractor):
        """Phrase present in current contact's common_phrases should not be listed."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "meeting"])
        _insert_template(db, "carol@example.com", ["hello", "thanks", "update"])
        _insert_template(db, "dave@example.com", ["hello", "thanks", "review"])

        # "thanks" is in 3/3 (100%) but also in current contact's phrases
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello", "thanks"])
        assert "thanks" not in result

    def test_some_phrases_excluded_others_not(self, db, extractor):
        """Only phrases absent from current contact are returned."""
        _insert_template(db, "bob@example.com", ["hello", "thanks", "cheers"])
        _insert_template(db, "carol@example.com", ["hello", "thanks", "cheers"])
        _insert_template(db, "dave@example.com", ["hello", "thanks", "cheers"])

        # Current contact has "thanks" but not "cheers"
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello", "thanks"])
        assert "thanks" not in result
        assert "cheers" in result


class TestCapAtTen:
    """avoids_phrases is capped at 10 entries."""

    def test_cap_at_ten(self, db, extractor):
        """When more than 10 phrases qualify, only top 10 by frequency are returned."""
        # Create 5 templates each with 12 unique phrases that the current contact lacks
        shared_phrases = [f"phrase_{i}" for i in range(12)]
        for i, name in enumerate(["bob", "carol", "dave", "eve", "frank"]):
            _insert_template(db, f"{name}@example.com", shared_phrases)

        # Current contact has none of these phrases → all 12 qualify
        result = extractor._compute_avoids_phrases("alice@example.com", [])
        assert len(result) <= 10


class TestEmptyPhrases:
    """Handles templates with empty or null common_phrases gracefully."""

    def test_empty_phrases_in_other_templates(self, db, extractor):
        """Other templates with empty common_phrases don't break the computation."""
        _insert_template(db, "bob@example.com", [])
        _insert_template(db, "carol@example.com", [])
        _insert_template(db, "dave@example.com", [])

        result = extractor._compute_avoids_phrases("alice@example.com", ["hello"])
        assert result == []

    def test_mix_of_empty_and_populated(self, db, extractor):
        """Empty templates are counted but contribute no phrases."""
        _insert_template(db, "bob@example.com", ["hello", "thanks"])
        _insert_template(db, "carol@example.com", ["hello", "thanks"])
        _insert_template(db, "dave@example.com", [])
        _insert_template(db, "eve@example.com", [])

        # "thanks" in 2/4 = 50% >= 40% and meets min-2 threshold
        result = extractor._compute_avoids_phrases("alice@example.com", [])
        assert "thanks" in result


class TestExcludesCurrentContact:
    """The current contact's own template is excluded from the comparison set."""

    def test_own_template_not_counted(self, db, extractor):
        """Current contact's template should not influence avoids_phrases."""
        _insert_template(db, "alice@example.com", ["hello", "thanks", "meeting"])
        _insert_template(db, "bob@example.com", ["hello", "thanks"])
        _insert_template(db, "carol@example.com", ["hello", "thanks"])
        _insert_template(db, "dave@example.com", ["hello", "thanks"])

        # alice's own template exists but should be excluded from comparison.
        # 3 other templates all have "thanks" → should be in avoids
        result = extractor._compute_avoids_phrases("alice@example.com", ["hello"])
        assert "thanks" in result


class TestOnlyOutboundTemplates:
    """avoids_phrases only considers user_to_contact templates."""

    def test_inbound_templates_ignored(self, db, extractor):
        """Inbound (contact_to_user) templates should not be considered."""
        # Insert inbound templates — should be ignored
        _insert_template(db, "bob@example.com", ["hello", "thanks"], context="contact_to_user")
        _insert_template(db, "carol@example.com", ["hello", "thanks"], context="contact_to_user")
        _insert_template(db, "dave@example.com", ["hello", "thanks"], context="contact_to_user")

        result = extractor._compute_avoids_phrases("alice@example.com", [])
        # No user_to_contact templates → cold-start guard → empty
        assert result == []


class TestIntegrationWithTemplateExtraction:
    """End-to-end test: avoids_phrases populated during template extraction."""

    def test_avoids_populated_after_enough_samples(self, db, extractor):
        """After 5+ samples, avoids_phrases is computed from cross-contact data."""
        # Seed 3 other outbound templates with shared phrases
        _insert_template(db, "bob@example.com", ["meeting", "thanks", "update"])
        _insert_template(db, "carol@example.com", ["meeting", "thanks", "review"])
        _insert_template(db, "dave@example.com", ["meeting", "thanks", "project"])

        # Seed an existing template for alice with 4 samples (below threshold)
        import hashlib
        alice_template_id = hashlib.sha256("alice@example.com:email:out".encode()).hexdigest()[:16]
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO communication_templates
                   (id, context, contact_id, channel, formality, typical_length,
                    uses_emoji, common_phrases, avoids_phrases, tone_notes,
                    example_message_ids, samples_analyzed)
                   VALUES (?, 'user_to_contact', 'alice@example.com', 'email',
                           0.5, 100.0, 0, '["hello", "report"]', '[]', '[]', '[]', 4)""",
                (alice_template_id,),
            )

        # Process an outbound email to alice (this is the 5th sample)
        event = {
            "id": "evt-test-001",
            "type": "email.sent",
            "source": "email",
            "timestamp": "2026-03-01T10:00:00Z",
            "payload": {
                "to_addresses": ["alice@example.com"],
                "body": "Hello Alice, here is the report you requested. Let me know if you have questions.",
                "channel": "email",
            },
        }
        extractor._extract_communication_templates(event, ["alice@example.com"], is_outbound=True)

        # Read the updated template
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT avoids_phrases, samples_analyzed FROM communication_templates WHERE id = ?",
                (alice_template_id,),
            ).fetchone()

        assert row is not None
        avoids = json.loads(row["avoids_phrases"])
        assert row["samples_analyzed"] == 5
        # "thanks" appears in 3/3 other templates but not in alice's phrases
        assert "thanks" in avoids
        # "meeting" also appears in 3/3 and is absent from alice's phrases
        assert "meeting" in avoids

    def test_avoids_not_computed_for_inbound(self, db, extractor):
        """Inbound messages should not trigger avoids_phrases computation."""
        # Seed 3 other outbound templates
        _insert_template(db, "bob@example.com", ["meeting", "thanks"])
        _insert_template(db, "carol@example.com", ["meeting", "thanks"])
        _insert_template(db, "dave@example.com", ["meeting", "thanks"])

        # Seed an existing inbound template for alice with 10 samples
        import hashlib
        alice_in_id = hashlib.sha256("alice@example.com:email:in".encode()).hexdigest()[:16]
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO communication_templates
                   (id, context, contact_id, channel, formality, typical_length,
                    uses_emoji, common_phrases, avoids_phrases, tone_notes,
                    example_message_ids, samples_analyzed)
                   VALUES (?, 'contact_to_user', 'alice@example.com', 'email',
                           0.5, 100.0, 0, '["hello"]', '[]', '[]', '[]', 10)""",
                (alice_in_id,),
            )

        # Process an inbound email from alice
        event = {
            "id": "evt-test-002",
            "type": "email.received",
            "source": "email",
            "timestamp": "2026-03-01T10:00:00Z",
            "payload": {
                "from_address": "alice@example.com",
                "body": "Hey there, just checking in on the status of things. Hope all is well.",
                "channel": "email",
            },
        }
        extractor._extract_communication_templates(event, ["alice@example.com"], is_outbound=False)

        # Read the updated template — avoids_phrases should remain empty
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT avoids_phrases FROM communication_templates WHERE id = ?",
                (alice_in_id,),
            ).fetchone()

        assert row is not None
        avoids = json.loads(row["avoids_phrases"])
        assert avoids == []
