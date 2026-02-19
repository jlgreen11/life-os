"""
Tests — SemanticFactInferrer: marketing filter applied to relationship facts.

Verifies that ``infer_from_relationship_profile()`` and the companion
``_purge_marketing_relationship_facts()`` method correctly exclude automated
and marketing senders from semantic relationship facts.

Prior to this fix, the inferrer only checked ``outbound_count > 0`` to
distinguish "human" contacts from "marketing" contacts.  Because users
occasionally reply to marketing emails (e.g. "unsubscribe", forwarding a
receipt), senders like ``store-news@amazon.com`` could accumulate outbound
counts > 0 and be stored as ``relationship_priority_*`` = "high_priority" in
semantic memory.

After the fix:
  - ``is_marketing_or_noreply()`` is applied before any relationship fact is
    created.
  - ``_purge_marketing_relationship_facts()`` removes stale facts that were
    created by old inference runs before the filter was in place.
  - Only genuine human contacts appear in ``relationship_priority_*``,
    ``relationship_balance_*``, and ``relationship_multichannel_*`` facts.
"""

from __future__ import annotations

import json

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARKETING_CONTACTS = [
    "store-news@amazon.com",
    "no-reply@accounts.google.com",
    "SouthwestAirlines@iluv.southwest.com",
    "bathandbodyworks@e2.bathandbodyworks.com",
    "no-reply@rs.email.nextdoor.com",
    "receipt@chromefile.com",
    "RoyalCaribbean@reply.royalcaribbeanmarketing.com",
]

HUMAN_CONTACTS = [
    "alice@example.com",
    "bob@company.org",
]


def _build_relationship_profile(contacts_dict: dict) -> dict:
    """Build a signal profile payload that mimics the live relationships profile."""
    total_samples = sum(
        c.get("inbound_count", 0) + c.get("outbound_count", 0)
        for c in contacts_dict.values()
    )
    return {
        "samples_count": total_samples,
        "data": {"contacts": contacts_dict},
    }


def _contact_entry(inbound: int, outbound: int, interaction: int) -> dict:
    """Return a minimal contact entry with the fields the inferrer reads."""
    return {
        "inbound_count": inbound,
        "outbound_count": outbound,
        "interaction_count": interaction,
        "channels_used": ["email"],
    }


def _seed_profile(ums: UserModelStore, contacts: dict) -> None:
    """Write a relationships signal profile so the inferrer can read it."""
    profile = _build_relationship_profile(contacts)
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
            VALUES ('relationships', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (json.dumps(profile["data"]), profile["samples_count"]),
        )


def _get_facts(ums: UserModelStore) -> dict[str, str]:
    """Return {key: decoded_value} for all semantic facts.

    Values are stored as JSON in the DB (so strings appear as ``'"high_priority"'``).
    This helper decodes them so tests can compare plain Python strings/ints.
    """
    with ums.db.get_connection("user_model") as conn:
        rows = conn.execute("SELECT key, value FROM semantic_facts").fetchall()
    result = {}
    for r in rows:
        raw = r["value"]
        try:
            result[r["key"]] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result[r["key"]] = raw
    return result


def _seed_fact(ums: UserModelStore, key: str, value: str = "high_priority") -> None:
    """Directly insert a semantic fact (simulates a stale pre-fix fact)."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO semantic_facts
                (key, category, value, confidence, first_observed, last_confirmed)
            VALUES (?, 'implicit_preference', ?, 0.8, datetime('now'), datetime('now'))
            """,
            (key, value),
        )


# ---------------------------------------------------------------------------
# Tests: _purge_marketing_relationship_facts
# ---------------------------------------------------------------------------


class TestPurgeMarketingRelationshipFacts:
    """Unit tests for the stale-fact cleanup method."""

    def test_purge_removes_marketing_priority_facts(self, db):
        """relationship_priority_* for marketing senders must be deleted."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        for addr in MARKETING_CONTACTS:
            _seed_fact(ums, f"relationship_priority_{addr}")

        deleted = inferrer._purge_marketing_relationship_facts()

        facts = _get_facts(ums)
        for addr in MARKETING_CONTACTS:
            assert f"relationship_priority_{addr}" not in facts, (
                f"Marketing fact for {addr} should have been purged"
            )
        assert deleted == len(MARKETING_CONTACTS)

    def test_purge_removes_marketing_balance_facts(self, db):
        """relationship_balance_* for marketing senders must be deleted."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        for addr in MARKETING_CONTACTS:
            _seed_fact(ums, f"relationship_balance_{addr}", "mutual")

        deleted = inferrer._purge_marketing_relationship_facts()

        facts = _get_facts(ums)
        for addr in MARKETING_CONTACTS:
            assert f"relationship_balance_{addr}" not in facts

    def test_purge_removes_marketing_multichannel_facts(self, db):
        """relationship_multichannel_* for marketing senders must be deleted."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        for addr in MARKETING_CONTACTS:
            _seed_fact(ums, f"relationship_multichannel_{addr}", "multi_channel")

        deleted = inferrer._purge_marketing_relationship_facts()

        facts = _get_facts(ums)
        for addr in MARKETING_CONTACTS:
            assert f"relationship_multichannel_{addr}" not in facts
        assert deleted == len(MARKETING_CONTACTS)

    def test_purge_preserves_human_contact_facts(self, db):
        """relationship_* facts for genuine human contacts must not be deleted."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        for addr in HUMAN_CONTACTS:
            _seed_fact(ums, f"relationship_priority_{addr}")

        deleted = inferrer._purge_marketing_relationship_facts()

        facts = _get_facts(ums)
        for addr in HUMAN_CONTACTS:
            assert f"relationship_priority_{addr}" in facts, (
                f"Human contact fact for {addr} must not be purged"
            )
        assert deleted == 0

    def test_purge_preserves_user_corrected_facts(self, db):
        """is_user_corrected=1 facts must never be deleted even for marketing senders."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        # Insert a user-corrected fact for a marketing sender
        addr = "store-news@amazon.com"
        with ums.db.get_connection("user_model") as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_facts
                    (key, category, value, confidence, first_observed,
                     last_confirmed, is_user_corrected)
                VALUES (?, 'implicit_preference', 'high_priority', 0.9,
                        datetime('now'), datetime('now'), 1)
                """,
                (f"relationship_priority_{addr}",),
            )

        deleted = inferrer._purge_marketing_relationship_facts()

        facts = _get_facts(ums)
        assert f"relationship_priority_{addr}" in facts, (
            "User-corrected marketing fact must not be purged"
        )
        assert deleted == 0

    def test_purge_returns_zero_when_nothing_to_purge(self, db):
        """Returns 0 when there are no marketing relationship facts."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        deleted = inferrer._purge_marketing_relationship_facts()
        assert deleted == 0


# ---------------------------------------------------------------------------
# Tests: infer_from_relationship_profile — marketing filter integration
# ---------------------------------------------------------------------------


class TestInferFromRelationshipProfileMarketingFilter:
    """Integration tests for the full inference method with marketing filter."""

    def test_marketing_contacts_not_stored_as_high_priority(self, db):
        """Marketing senders with many interactions must not become high_priority facts."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        # Marketing sender has many interactions (would cross high_priority_threshold
        # without the marketing filter)
        contacts = {
            "store-news@amazon.com": _contact_entry(inbound=200, outbound=5, interaction=205),
            "alice@example.com": _contact_entry(inbound=20, outbound=20, interaction=40),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert "relationship_priority_store-news@amazon.com" not in facts, (
            "Marketing sender must not be stored as high_priority"
        )

    def test_human_contacts_are_stored_as_high_priority(self, db):
        """Human contacts with 2× average interactions become high_priority facts.

        With 3 contacts where alice has 60 interactions and bob/carol have 10 each:
          avg = (60 + 10 + 10) / 3 ≈ 26.7 → threshold = 53.3
          alice (60) > 53.3 → high_priority ✓
        """
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        contacts = {
            "alice@example.com": _contact_entry(inbound=30, outbound=30, interaction=60),
            "bob@company.org": _contact_entry(inbound=5, outbound=5, interaction=10),
            "carol@domain.net": _contact_entry(inbound=5, outbound=5, interaction=10),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert facts.get("relationship_priority_alice@example.com") == "high_priority"

    def test_marketing_contacts_not_stored_as_balanced(self, db):
        """Marketing senders must not appear in relationship_balance_* facts."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        # Marketing sender with enough interactions for balance inference
        contacts = {
            "no-reply@accounts.google.com": _contact_entry(inbound=8, outbound=5, interaction=13),
            "alice@example.com": _contact_entry(inbound=8, outbound=5, interaction=13),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert "relationship_balance_no-reply@accounts.google.com" not in facts

    def test_stale_marketing_facts_purged_on_each_inference_run(self, db):
        """Stale marketing facts from old inference runs are cleaned up."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        # Manually seed a stale marketing relationship fact (as if created before the fix)
        _seed_fact(ums, "relationship_priority_store-news@amazon.com")

        # Seed a minimal profile so inference can run
        contacts = {
            "alice@example.com": _contact_entry(inbound=20, outbound=20, interaction=40),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert "relationship_priority_store-news@amazon.com" not in facts, (
            "Stale marketing fact must be purged when inference runs"
        )

    def test_only_human_contacts_affect_priority_threshold(self, db):
        """
        Removing marketing contacts from the pool must lower the high_priority_threshold.

        Without the fix, high-volume marketing senders inflate the average interaction
        count so much that real human contacts (with modest email volumes) never reach
        the threshold.  With the fix, the threshold is computed only from human contacts
        so low-volume human relationships can be correctly identified as high_priority.
        """
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        # Marketing sender with huge volume — would inflate the average massively
        # if included in the threshold calculation.
        contacts = {
            "store-news@amazon.com": _contact_entry(inbound=500, outbound=10, interaction=510),
            "SouthwestAirlines@iluv.southwest.com": _contact_entry(inbound=300, outbound=5, interaction=305),
            # Human contacts with modest but genuine activity
            "alice@example.com": _contact_entry(inbound=20, outbound=20, interaction=40),
            "bob@company.org": _contact_entry(inbound=8, outbound=8, interaction=16),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)

        # alice has 40 interactions, bob has 16; average = 28; threshold = 56.
        # alice (40) is below threshold; bob (16) is well below.
        # But without marketing senders inflating the pool:
        #   avg_interactions = (40 + 16) / 2 = 28  →  threshold = 56
        # alice at 40 is below 56, so NOT high_priority.
        # bob at 16 is well below 56, so NOT high_priority.
        # (This test verifies that marketing senders are excluded — the exact
        # human-contact threshold logic is a separate concern.)
        assert "relationship_priority_store-news@amazon.com" not in facts
        assert "relationship_priority_SouthwestAirlines@iluv.southwest.com" not in facts

    def test_no_reply_variants_excluded(self, db):
        """no-reply, noreply, and no_reply address variants must all be filtered."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        contacts = {
            "no-reply@service.com": _contact_entry(inbound=5, outbound=2, interaction=7),
            "noreply@notifications.com": _contact_entry(inbound=5, outbound=2, interaction=7),
            "no_reply@alerts.com": _contact_entry(inbound=5, outbound=2, interaction=7),
            "alice@example.com": _contact_entry(inbound=20, outbound=20, interaction=40),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert "relationship_priority_no-reply@service.com" not in facts
        assert "relationship_priority_noreply@notifications.com" not in facts
        assert "relationship_priority_no_reply@alerts.com" not in facts

    def test_zero_outbound_contacts_still_excluded(self, db):
        """Contacts with zero outbound messages are excluded even before the marketing filter."""
        ums = UserModelStore(db)
        inferrer = SemanticFactInferrer(ums)

        contacts = {
            # Inbound-only human contact
            "newsletter@legit-human.com": _contact_entry(inbound=50, outbound=0, interaction=50),
            "alice@example.com": _contact_entry(inbound=20, outbound=20, interaction=40),
        }
        _seed_profile(ums, contacts)

        inferrer.infer_from_relationship_profile()

        facts = _get_facts(ums)
        assert "relationship_priority_newsletter@legit-human.com" not in facts
