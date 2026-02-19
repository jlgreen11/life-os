"""
Tests for the inbound linguistic profile averages fix.

Previously, ``_update_inbound_profile`` stored only 6 of the 10 metrics
computed per signal in ``per_contact_averages``.  The three metrics that
were computed but silently discarded were:

  - ``question_rate``   (read by ContextAssembler.assemble_draft_context)
  - ``ellipsis_rate``   (informality / trailing-thought signal)
  - ``unique_word_ratio`` (vocabulary richness)

As a result, ``assemble_draft_context`` always received 0.0 for
``question_rate`` regardless of how many questions a contact asked, breaking
style-matching for inquisitive contacts.

This test suite confirms that all three previously missing metrics are now
present in ``per_contact_averages`` after processing inbound signals.
"""

from __future__ import annotations

import statistics
from unittest.mock import MagicMock

import pytest

from services.signal_extractor.linguistic import LinguisticExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor() -> LinguisticExtractor:
    """Build a LinguisticExtractor with a mocked UserModelStore.

    The mock captures the last value passed to ``update_signal_profile`` so
    tests can inspect what was actually stored.
    """
    ums = MagicMock()
    # Simulate an empty profile on first call (bootstrap path).
    ums.get_signal_profile.return_value = None
    extractor = LinguisticExtractor.__new__(LinguisticExtractor)
    extractor.ums = ums
    return extractor


def _make_inbound_event(
    body: str,
    from_address: str = "alice@example.com",
    source: str = "email",
) -> dict:
    """Create a minimal inbound email event dict."""
    return {
        "type": "email.received",
        "source": source,
        "timestamp": "2026-02-19T10:00:00Z",
        "payload": {
            "from_address": from_address,
            "to_addresses": ["me@example.com"],
            "body": body,
        },
    }


def _last_inbound_data(extractor: LinguisticExtractor) -> dict:
    """Return the data dict last passed to update_signal_profile('linguistic_inbound')."""
    for call in reversed(extractor.ums.update_signal_profile.call_args_list):
        args, _ = call
        if args[0] == "linguistic_inbound":
            return args[1]
    raise AssertionError("update_signal_profile was never called with 'linguistic_inbound'")


# ---------------------------------------------------------------------------
# Tests: presence of previously missing metrics
# ---------------------------------------------------------------------------

class TestInboundAveragesMissingMetrics:
    """Confirm that per_contact_averages now includes the 3 previously missing metrics."""

    def test_question_rate_present_in_inbound_averages(self):
        """question_rate must appear in per_contact_averages after processing inbound messages."""
        extractor = _make_extractor()
        # A message with explicit question marks to make question_rate nonzero.
        event = _make_inbound_event(
            "Hi! Can we reschedule? Does Thursday work for you? What time suits you?",
            from_address="alice@example.com",
        )
        extractor._update_inbound_profile(extractor.extract(event)[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["alice@example.com"]
        assert "question_rate" in avg, (
            "question_rate must be stored in per_contact_averages "
            "(draft context reads it to detect inquisitive contacts)"
        )

    def test_ellipsis_rate_present_in_inbound_averages(self):
        """ellipsis_rate must appear in per_contact_averages after processing inbound messages."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "I was thinking... maybe we could try a different approach... not sure though...",
            from_address="bob@example.com",
        )
        extractor._update_inbound_profile(extractor.extract(event)[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["bob@example.com"]
        assert "ellipsis_rate" in avg, (
            "ellipsis_rate must be stored in per_contact_averages "
            "(signals casual/hesitant writing style)"
        )

    def test_unique_word_ratio_present_in_inbound_averages(self):
        """unique_word_ratio must appear in per_contact_averages after processing inbound messages."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "The quarterly revenue projections indicate a significant upswing in enterprise adoption.",
            from_address="carol@example.com",
        )
        extractor._update_inbound_profile(extractor.extract(event)[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["carol@example.com"]
        assert "unique_word_ratio" in avg, (
            "unique_word_ratio must be stored in per_contact_averages "
            "(vocabulary richness signal for style matching)"
        )


# ---------------------------------------------------------------------------
# Tests: correct values
# ---------------------------------------------------------------------------

class TestInboundAverageValues:
    """Verify that the newly added metrics contain numerically correct values."""

    def test_question_rate_value_nonzero_for_questioning_contact(self):
        """A contact who asks many questions should have question_rate > 0 in their averages."""
        extractor = _make_extractor()
        # Three question-heavy messages
        for body in [
            "Do you want to meet tomorrow? Can we discuss the proposal?",
            "What did you think of the report? Should we revise it?",
            "Are you free this afternoon? Would that work for you?",
        ]:
            signals = extractor.extract(_make_inbound_event(body, "dave@example.com"))
            if signals:
                extractor._update_inbound_profile(signals[0])

        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["dave@example.com"]
        assert avg["question_rate"] > 0, (
            f"Expected question_rate > 0 for a questioning contact, got {avg['question_rate']}"
        )

    def test_question_rate_zero_for_non_questioning_contact(self):
        """A contact with no question marks should have question_rate = 0."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "Thanks for the update. I will review the document shortly.",
            from_address="eve@example.com",
        )
        signals = extractor.extract(event)
        assert signals
        extractor._update_inbound_profile(signals[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["eve@example.com"]
        assert avg["question_rate"] == 0.0, (
            f"Expected question_rate=0 for contact without questions, got {avg['question_rate']}"
        )

    def test_ellipsis_rate_value_reflects_actual_ellipsis_usage(self):
        """ellipsis_rate should be positive for contacts who use trailing dots."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "I was thinking... maybe not... we could try later... or not.",
            from_address="frank@example.com",
        )
        signals = extractor.extract(event)
        assert signals
        extractor._update_inbound_profile(signals[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["frank@example.com"]
        assert avg["ellipsis_rate"] > 0, (
            f"Expected ellipsis_rate > 0 for contact who uses '...', got {avg['ellipsis_rate']}"
        )

    def test_unique_word_ratio_between_zero_and_one(self):
        """unique_word_ratio must always be in [0, 1] (it is a type-token ratio)."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "The project is moving along nicely and the team is making great progress.",
            from_address="grace@example.com",
        )
        signals = extractor.extract(event)
        assert signals
        extractor._update_inbound_profile(signals[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["grace@example.com"]
        ratio = avg["unique_word_ratio"]
        assert 0.0 <= ratio <= 1.0, (
            f"unique_word_ratio must be in [0, 1], got {ratio}"
        )


# ---------------------------------------------------------------------------
# Tests: multi-message averaging
# ---------------------------------------------------------------------------

class TestInboundAveragesAreActualMeans:
    """Verify that multiple messages produce proper statistical means, not last-value overwrites."""

    def test_question_rate_is_mean_across_messages(self):
        """question_rate in averages must be the mean across all messages from the contact.

        Uses extract() for each message (which internally calls _update_inbound_profile),
        feeding accumulated data back to the mock between iterations so the profile
        grows incrementally as it does in production.
        """
        extractor = _make_extractor()
        contact = "hank@example.com"

        individual_rates = []
        messages = [
            "Are you available? Can we talk?",       # 2 questions → rate > 0
            "I just wanted to check in with you.",    # 0 questions → rate 0
            "What time works? Is morning okay?",      # 2 questions → rate > 0
        ]
        for body in messages:
            # Feed accumulated profile data back before each extract() call so
            # the profile grows the same way as in production.
            try:
                existing_data = _last_inbound_data(extractor)
                extractor.ums.get_signal_profile.return_value = {
                    "data": existing_data
                }
            except AssertionError:
                pass  # First iteration: profile is empty (mock returns None)

            signals = extractor.extract(_make_inbound_event(body, contact))
            # extract() internally calls _update_inbound_profile — no explicit call needed.
            if signals:
                individual_rates.append(signals[0]["metrics"]["question_rate"])

        data = _last_inbound_data(extractor)
        stored_avg = data["per_contact_averages"][contact]["question_rate"]
        expected_avg = statistics.mean(individual_rates)
        assert abs(stored_avg - expected_avg) < 1e-9, (
            f"stored question_rate={stored_avg:.6f} does not match "
            f"expected mean={expected_avg:.6f} of {individual_rates}"
        )

    def test_all_ten_metrics_present_in_inbound_averages(self):
        """All metrics that appear in the signal dict must be present in per_contact_averages."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "Hi! How are things going? I wanted to follow up on the proposal... "
            "definitely makes sense to revisit. Let me know what you think!",
            from_address="irene@example.com",
        )
        signals = extractor.extract(event)
        assert signals, "extract() must return signals for a sufficiently long message"
        extractor._update_inbound_profile(signals[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["irene@example.com"]

        required_keys = {
            "avg_sentence_length",
            "formality",
            "hedge_rate",
            "assertion_rate",
            "exclamation_rate",
            "question_rate",
            "ellipsis_rate",
            "unique_word_ratio",
            "emoji_rate",
            "samples_count",
        }
        missing = required_keys - avg.keys()
        assert not missing, (
            f"per_contact_averages is missing metrics: {missing}. "
            "These must be present so downstream consumers (draft context, "
            "insight engine) can read them without silent 0.0 fallbacks."
        )


# ---------------------------------------------------------------------------
# Tests: existing metrics unaffected
# ---------------------------------------------------------------------------

class TestExistingMetricsUnchanged:
    """Confirm that the 6 pre-existing metrics still work correctly after the fix."""

    def test_existing_formality_metric_still_present(self):
        """Adding new metrics must not break the pre-existing formality metric."""
        extractor = _make_extractor()
        event = _make_inbound_event(
            "Regarding the proposal, furthermore we must consider the impact accordingly.",
            from_address="jack@example.com",
        )
        signals = extractor.extract(event)
        assert signals
        extractor._update_inbound_profile(signals[0])
        data = _last_inbound_data(extractor)
        avg = data["per_contact_averages"]["jack@example.com"]
        assert "formality" in avg
        assert 0.0 <= avg["formality"] <= 1.0

    def test_samples_count_increments_correctly(self):
        """samples_count must equal the number of messages processed for the contact.

        Uses extract() exclusively (which internally calls _update_inbound_profile),
        feeding accumulated data back between iterations to simulate production
        behaviour without double-calling the update method.
        """
        extractor = _make_extractor()
        contact = "kate@example.com"
        messages_to_send = 3

        for i in range(messages_to_send):
            # Feed accumulated profile back before each extract call.
            try:
                existing_data = _last_inbound_data(extractor)
                extractor.ums.get_signal_profile.return_value = {
                    "data": existing_data
                }
            except AssertionError:
                pass  # First iteration: profile is empty (mock returns None)

            extractor.extract(
                _make_inbound_event(
                    f"This is message number {i + 1} from Kate with enough words for analysis.",
                    contact,
                )
            )
            # extract() calls _update_inbound_profile internally — no explicit call.

        data = _last_inbound_data(extractor)
        count = data["per_contact_averages"][contact]["samples_count"]
        assert count == messages_to_send, (
            f"Expected samples_count={messages_to_send}, got {count}"
        )
