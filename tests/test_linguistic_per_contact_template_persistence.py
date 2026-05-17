"""Tests for the per-contact communication template write hardening.

``LinguisticExtractor._store_per_contact_templates`` materializes the
"outbound" side of the linguistic profile as one CommunicationTemplate row
per contact.  Previously the write was wrapped in a broad try/except that
silently swallowed any error, with no pre-write serialization check and no
post-write verification.  In production this surfaced as a near-empty
communication_templates table (2 rows) despite 394k+ inbound linguistic
samples — writes were being dropped without operator visibility.

These tests cover the three new guards:

  1. Happy path — the write succeeds, the read-back confirms persistence,
     ``_template_writes_verified`` increments, ``_template_write_failures``
     does not, and no CRITICAL log is emitted.
  2. Silent drop — the write returns without raising but the read-back
     returns an empty list (simulating WAL corruption / lock contention).
     A CRITICAL log MUST be emitted and ``_template_write_failures`` MUST
     increment so a future diagnostic surface can report the drop.
  3. Non-serializable payload — a template containing a value that
     ``json.dumps`` rejects (e.g. a ``set``) triggers a pre-write CRITICAL
     log naming the offending key and ``store_communication_template`` is
     NEVER called.  ``_template_write_failures`` increments.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from services.signal_extractor.linguistic import LinguisticExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor() -> LinguisticExtractor:
    """Build a LinguisticExtractor with a mocked UserModelStore.

    Bypasses ``__init__`` because ``BaseExtractor.__init__`` requires a
    fully-instantiated ``DatabaseManager``.  The tests only exercise
    ``_store_per_contact_templates`` and ``_get_existing_phrase_data``
    (mocked via the ``ums`` proxy), so a minimal extractor suffices.
    """
    ums = MagicMock()
    ums.get_communication_templates.return_value = []
    extractor = LinguisticExtractor.__new__(LinguisticExtractor)
    extractor.ums = ums
    # Reset class-level counters to instance-level to avoid cross-test bleed.
    extractor._template_write_failures = 0
    extractor._template_writes_verified = 0
    return extractor


def _samples(n: int = 6, channel: str = "email") -> list[dict]:
    """Return ``n`` synthetic linguistic samples sufficient to cross the
    ``_MIN_TEMPLATE_SAMPLES`` threshold (default 5).

    Each sample carries the minimum fields the method touches: word count,
    channel, and optional greeting/closing detections.
    """
    return [
        {
            "word_count": 12 + i,
            "channel": channel,
            "greeting_detected": "hey",
            "closing_detected": "thanks",
        }
        for i in range(n)
    ]


def _avgs(samples_count: int = 6) -> dict:
    """Per-contact averages dict with enough samples to trigger a write."""
    return {
        "samples_count": samples_count,
        "formality": 0.55,
        "emoji_rate": 0.0,
        "hedge_rate": 0.05,
        "assertion_rate": 0.05,
        "question_rate": 0.05,
        "exclamation_rate": 0.05,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerContactTemplatePersistence:
    """Verify the three new guards on the per-contact template write path."""

    def test_happy_path_verifies_and_increments_verified_counter(self, caplog):
        """Successful write + successful read-back: counter increments, no CRITICAL."""
        extractor = _make_extractor()
        contact_id = "alice@example.com"
        per_contact_samples = {contact_id: _samples()}
        per_contact_avgs = {contact_id: _avgs()}

        # ``get_communication_templates`` is called twice per contact:
        #   1. by ``_get_existing_phrase_data`` before building the template
        #      (limit=10) — must return [] so phrase data starts clean.
        #   2. by the post-write verification (limit=1) — must include a row
        #      whose id matches the just-stored template.
        # We capture the stored template and echo it on subsequent calls.
        stored: dict = {}

        def _capture(template):
            stored["template"] = template

        def _readback(contact_id=None, limit=20, **kwargs):
            tpl = stored.get("template")
            return [tpl] if tpl else []

        extractor.ums.store_communication_template.side_effect = _capture
        extractor.ums.get_communication_templates.side_effect = _readback

        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.linguistic"):
            extractor._store_per_contact_templates(per_contact_samples, per_contact_avgs)

        assert extractor.ums.store_communication_template.call_count == 1, (
            "happy path must call store_communication_template exactly once"
        )
        assert extractor._template_writes_verified == 1
        assert extractor._template_write_failures == 0
        critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical_records == [], (
            f"happy path must not emit CRITICAL logs; got: {critical_records}"
        )

    def test_silent_drop_logs_critical_and_increments_failure_counter(self, caplog):
        """Write succeeds but read-back is empty → CRITICAL + failure counter."""
        extractor = _make_extractor()
        contact_id = "bob@example.com"
        per_contact_samples = {contact_id: _samples()}
        per_contact_avgs = {contact_id: _avgs()}

        # store succeeds (no exception), but read-back returns empty —
        # simulating a silent WAL/lock failure where the row never lands.
        # ``get_communication_templates`` is called twice (phrase lookup +
        # post-write verification); both return [].
        extractor.ums.store_communication_template.return_value = None
        extractor.ums.get_communication_templates.return_value = []

        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.linguistic"):
            extractor._store_per_contact_templates(per_contact_samples, per_contact_avgs)

        assert extractor.ums.store_communication_template.call_count == 1
        assert extractor._template_write_failures == 1
        assert extractor._template_writes_verified == 0

        critical_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL
        ]
        assert any("FAILED to persist" in m for m in critical_messages), (
            f"missing post-write CRITICAL log; got: {critical_messages}"
        )
        # Operator-actionable diagnostics must include contact_id and samples count.
        assert any(contact_id in m for m in critical_messages), (
            "CRITICAL log must include the contact_id for operator visibility"
        )

    def test_non_serializable_value_logs_critical_and_skips_store(self, caplog):
        """A non-JSON-serializable value (set) must abort the write pre-call."""
        extractor = _make_extractor()
        contact_id = "carol@example.com"
        per_contact_samples = {contact_id: _samples()}
        per_contact_avgs = {contact_id: _avgs()}

        # Inject a set via _get_existing_phrase_data so the template ends up
        # with a non-serializable value in ``common_phrases``.  This bypasses
        # the need to mutate _store_per_contact_templates and exercises the
        # real serialization guard.  ``_get_existing_phrase_data`` calls
        # ``get_communication_templates`` (plural) and picks the first
        # non-empty ``common_phrases`` value it sees.
        extractor.ums.get_communication_templates.return_value = [
            {
                "id": "preexisting",
                "common_phrases": {"a-set-is-not-json-serializable"},
                "avoids_phrases": [],
            }
        ]

        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.linguistic"):
            extractor._store_per_contact_templates(per_contact_samples, per_contact_avgs)

        # Critical: the store must NEVER be called when serialization fails.
        assert extractor.ums.store_communication_template.call_count == 0, (
            "pre-write serialization guard must prevent the store call"
        )
        assert extractor._template_write_failures == 1
        assert extractor._template_writes_verified == 0

        critical_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL
        ]
        assert any("FAILED to serialize" in m for m in critical_messages), (
            f"missing pre-write CRITICAL log; got: {critical_messages}"
        )
        # The offending key must be named so operators know which field broke.
        assert any("common_phrases" in m for m in critical_messages), (
            "CRITICAL log must name the offending key (common_phrases)"
        )
