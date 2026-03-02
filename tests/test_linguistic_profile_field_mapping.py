"""
Tests for LinguisticProfile canonical field name mapping.

The LinguisticExtractor stores computed averages under internal keys like
``hedge_rate``, but the typed ``LinguisticProfile`` model expects canonical
field names like ``hedge_frequency``.  A mapping block in ``_update_profile()``
bridges this gap by copying values to both key variants.

These tests verify that:
  - All 7 canonical LinguisticProfile field names are populated after a flush
  - The original internal _rate keys still exist (backward compatibility)
  - Canonical values match the corresponding internal values
"""

from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


def _make_outbound_event(body: str, to: str = "test@example.com") -> dict:
    """Build a minimal outbound email event for testing."""
    return {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "email",
        "payload": {
            "body": body,
            "to_addresses": [to],
        },
    }


# Sample messages designed to produce non-zero values for the mapped fields.
# They include hedge words, assertions, questions, ellipses, greetings, and
# closings so that the canonical fields are populated with real data.
_SAMPLE_MESSAGES = [
    "Hey there! Maybe we should discuss this further? I think it could work...",
    "Hello! We need to finalize the proposal. Definitely the right approach. Thanks for your help!",
    "Hi! I guess perhaps we might consider an alternative? Not sure about the timeline... Cheers!",
    "Hey! Clearly this is the way forward. We must act now. Best regards.",
    "Hello team! I think maybe we could try something different? Perhaps... Talk soon!",
]


def test_canonical_fields_populated_after_flush(db, user_model_store):
    """After processing enough messages, all 7 canonical LinguisticProfile fields
    should be present in the stored profile data with non-default values."""
    extractor = LinguisticExtractor(db=db, user_model_store=user_model_store)

    for body in _SAMPLE_MESSAGES:
        extractor.extract(_make_outbound_event(body))

    profile = user_model_store.get_signal_profile("linguistic")
    assert profile is not None
    data = profile["data"]

    # Canonical field: hedge_frequency (mapped from averages.hedge_rate)
    assert "hedge_frequency" in data
    assert data["hedge_frequency"] > 0.0, "hedge_frequency should be non-zero with hedge words in samples"

    # Canonical field: assertion_frequency (mapped from averages.assertion_rate)
    assert "assertion_frequency" in data
    assert data["assertion_frequency"] > 0.0, "assertion_frequency should be non-zero with assertion words in samples"

    # Canonical field: question_frequency (mapped from averages.question_rate)
    assert "question_frequency" in data
    assert data["question_frequency"] > 0.0, "question_frequency should be non-zero with question marks in samples"

    # Canonical field: ellipsis_frequency (mapped from averages.ellipsis_rate)
    assert "ellipsis_frequency" in data
    assert data["ellipsis_frequency"] > 0.0, "ellipsis_frequency should be non-zero with ellipses in samples"

    # Canonical field: greeting_patterns (mapped from common_greetings)
    assert "greeting_patterns" in data
    assert isinstance(data["greeting_patterns"], list)
    assert len(data["greeting_patterns"]) > 0, "greeting_patterns should be populated from common_greetings"

    # Canonical field: closing_patterns (mapped from common_closings)
    assert "closing_patterns" in data
    assert isinstance(data["closing_patterns"], list)
    assert len(data["closing_patterns"]) > 0, "closing_patterns should be populated from common_closings"

    # Canonical field: formality_spectrum (mapped from averages.formality)
    assert "formality_spectrum" in data
    assert isinstance(data["formality_spectrum"], float)
    assert 0.0 <= data["formality_spectrum"] <= 1.0


def test_original_rate_keys_still_present(db, user_model_store):
    """The original internal _rate keys must still exist for backward compatibility.
    Other code (per-contact averages, inbound profile, context assembler) reads them."""
    extractor = LinguisticExtractor(db=db, user_model_store=user_model_store)

    for body in _SAMPLE_MESSAGES:
        extractor.extract(_make_outbound_event(body))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]
    averages = data["averages"]

    # Original rate keys in averages dict
    assert "hedge_rate" in averages
    assert "assertion_rate" in averages
    assert "question_rate" in averages
    assert "ellipsis_rate" in averages
    assert "formality" in averages

    # Original list keys at top level
    assert "common_greetings" in data
    assert "common_closings" in data


def test_canonical_values_match_internal_values(db, user_model_store):
    """Canonical field values must exactly match their internal counterparts."""
    extractor = LinguisticExtractor(db=db, user_model_store=user_model_store)

    for body in _SAMPLE_MESSAGES:
        extractor.extract(_make_outbound_event(body))

    profile = user_model_store.get_signal_profile("linguistic")
    data = profile["data"]
    averages = data["averages"]

    # Rate → frequency mappings should be identical
    assert data["hedge_frequency"] == averages["hedge_rate"]
    assert data["assertion_frequency"] == averages["assertion_rate"]
    assert data["question_frequency"] == averages["question_rate"]
    assert data["ellipsis_frequency"] == averages["ellipsis_rate"]
    assert data["formality_spectrum"] == averages["formality"]

    # List mappings should be identical references
    assert data["greeting_patterns"] == data["common_greetings"]
    assert data["closing_patterns"] == data["common_closings"]


def test_canonical_fields_update_incrementally(db, user_model_store):
    """Canonical fields should update as new messages arrive, not stay stale."""
    extractor = LinguisticExtractor(db=db, user_model_store=user_model_store)

    # First batch: mostly hedging, no assertions
    hedge_messages = [
        "Maybe I think perhaps we might try something different here soon.",
        "I guess possibly we could sort of consider this not sure though maybe.",
    ]
    for body in hedge_messages:
        extractor.extract(_make_outbound_event(body))

    profile_after_hedges = user_model_store.get_signal_profile("linguistic")
    hedge_freq_v1 = profile_after_hedges["data"]["hedge_frequency"]
    assert hedge_freq_v1 > 0.0

    # Second batch: lots of assertions
    assertion_messages = [
        "We need to act now. Definitely must do this. Clearly the right path. Obviously correct.",
        "Without a doubt we need to proceed. Absolutely the best approach. We must deliver.",
        "Definitely clearly obviously we must do this. We need to ship it now.",
    ]
    for body in assertion_messages:
        extractor.extract(_make_outbound_event(body))

    profile_after_assertions = user_model_store.get_signal_profile("linguistic")
    assertion_freq = profile_after_assertions["data"]["assertion_frequency"]
    assert assertion_freq > 0.0, "assertion_frequency should be non-zero after assertion-heavy messages"

    # The hedge frequency should have diluted (more messages, same hedge count)
    hedge_freq_v2 = profile_after_assertions["data"]["hedge_frequency"]
    assert hedge_freq_v2 < hedge_freq_v1, "hedge_frequency should decrease as non-hedge messages dilute the average"
