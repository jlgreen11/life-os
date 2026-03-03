"""
Tests for exclamation_rate field mapping in LinguisticExtractor.

Verifies that exclamation_rate is correctly mapped from the internal averages
dict to a top-level key in the stored profile data, matching the canonical
LinguisticProfile field name (models/user_model.py).

Previously, exclamation_rate was computed in averages but never promoted to a
top-level key — unlike hedge_frequency, assertion_frequency, question_frequency,
and ellipsis_frequency which all had explicit mappings.  This left
LinguisticProfile.exclamation_rate permanently at 0.0 regardless of actual
exclamation usage.
"""

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


@pytest.fixture
def extractor(db, user_model_store):
    """Create a LinguisticExtractor instance with test databases."""
    return LinguisticExtractor(db=db, user_model_store=user_model_store)


def _email_sent(body: str, to: str = "recipient@example.com") -> dict:
    """Return a minimal email.sent event with the given body text."""
    return {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "body": body,
            "to_addresses": [to],
            "from_address": "user@example.com",
        },
    }


# ---------------------------------------------------------------------------
# Test 1: exclamation_rate mapped to top-level profile key
# ---------------------------------------------------------------------------


def test_exclamation_rate_mapped_to_profile(extractor, user_model_store):
    """exclamation_rate should exist as a top-level key in stored profile data.

    After processing a message with exclamation marks, the extractor should
    map averages['exclamation_rate'] to data['exclamation_rate'] — matching
    the pattern used for hedge_frequency, assertion_frequency, etc.
    """
    extractor.extract(_email_sent("Great news! We got the deal! Amazing!"))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    # exclamation_rate must exist as a top-level key (not just inside averages)
    assert "exclamation_rate" in data, (
        "exclamation_rate should be a top-level key in profile data, "
        "mapped from averages['exclamation_rate']"
    )
    # It should also still be in averages (the source of the mapping)
    assert "exclamation_rate" in data["averages"]


# ---------------------------------------------------------------------------
# Test 2: exclamation_rate nonzero for emphatic text
# ---------------------------------------------------------------------------


def test_exclamation_rate_nonzero_for_emphatic_text(extractor, user_model_store):
    """Processing emphatic messages should produce a positive exclamation_rate.

    Multiple messages with heavy exclamation usage should yield a nonzero
    top-level exclamation_rate in the profile.
    """
    emphatic_messages = [
        "This is incredible! I can't believe it! Wow!",
        "Amazing work! The team really delivered! Outstanding results!",
        "Yes! We did it! Congratulations to everyone involved!",
    ]
    for body in emphatic_messages:
        extractor.extract(_email_sent(body))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    assert data["exclamation_rate"] > 0, (
        "exclamation_rate should be positive for messages with heavy exclamation usage"
    )


# ---------------------------------------------------------------------------
# Test 3: exclamation_rate zero for calm text
# ---------------------------------------------------------------------------


def test_exclamation_rate_zero_for_calm_text(extractor, user_model_store):
    """Processing calm messages with no exclamation marks should yield zero.

    Messages without any exclamation marks should produce
    exclamation_rate == 0.0 in the profile.
    """
    calm_messages = [
        "Please review the attached quarterly report at your convenience.",
        "The meeting has been rescheduled to Thursday afternoon.",
        "I wanted to follow up on our previous discussion about the budget.",
    ]
    for body in calm_messages:
        extractor.extract(_email_sent(body))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    assert data["exclamation_rate"] == 0.0, (
        "exclamation_rate should be 0.0 for messages without exclamation marks"
    )


# ---------------------------------------------------------------------------
# Test 4: All 5 rate fields mapped as top-level keys (regression guard)
# ---------------------------------------------------------------------------


def test_all_rate_fields_mapped(extractor, user_model_store):
    """All 5 rate-to-frequency field mappings must exist as top-level keys.

    This is a regression guard ensuring that every rate field computed in
    averages is also mapped to the canonical LinguisticProfile field name
    at the top level of the profile data dict.  If a future change adds a
    rate field but forgets the mapping, this test will catch it.
    """
    # Process enough messages to build a profile
    messages = [
        "Maybe we should try this approach. I think it might work.",
        "We need to act now. This has to happen. Definitely the right call.",
        "What do you think? Is this the right direction? Any concerns?",
    ]
    for body in messages:
        extractor.extract(_email_sent(body))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]

    # All 5 rate/frequency fields must exist as top-level keys
    expected_fields = [
        "hedge_frequency",
        "assertion_frequency",
        "question_frequency",
        "ellipsis_frequency",
        "exclamation_rate",
    ]
    for field in expected_fields:
        assert field in data, (
            f"'{field}' should be a top-level key in profile data, "
            f"mapped from averages"
        )
        assert isinstance(data[field], float), (
            f"'{field}' should be a float, got {type(data[field])}"
        )


# ---------------------------------------------------------------------------
# Test 5: Per-contact exclamation_rate computed
# ---------------------------------------------------------------------------


def test_per_contact_exclamation_rate_computed(extractor, user_model_store):
    """Per-contact averages should include exclamation_rate for contacts with enough samples.

    After processing enough messages to/from the same contact (meeting the
    _MIN_PER_CONTACT_SAMPLES threshold), the per_contact_averages entry
    should contain an exclamation_rate field.
    """
    contact = "alice@example.com"
    # Send 5 messages to the same contact (well above _MIN_PER_CONTACT_SAMPLES=3)
    messages = [
        "Great to hear from you! Looking forward to our meeting!",
        "That sounds wonderful! I really appreciate the update!",
        "Fantastic work on the presentation! The team loved it!",
        "Exciting news about the project! Can't wait to see the results!",
        "Wonderful progress! Keep up the amazing work on this!",
    ]
    for body in messages:
        extractor.extract(_email_sent(body, to=contact))

    profile = user_model_store.get_signal_profile("linguistic")
    per_contact_avgs = profile["data"].get("per_contact_averages", {})

    assert contact in per_contact_avgs, (
        f"Contact '{contact}' should have per-contact averages after 5 messages"
    )
    assert "exclamation_rate" in per_contact_avgs[contact], (
        "per_contact_averages should include exclamation_rate"
    )
    assert per_contact_avgs[contact]["exclamation_rate"] >= 0, (
        "exclamation_rate should be non-negative"
    )
    # These messages all have exclamations, so rate should be positive
    assert per_contact_avgs[contact]["exclamation_rate"] > 0, (
        "exclamation_rate should be positive for emphatic messages"
    )
