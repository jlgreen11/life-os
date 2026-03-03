"""
Tests for the three new CadenceProfile derived metrics:

  - avg_response_time_by_contact — per-contact reply latency averages
  - avg_response_time_by_channel — per-channel reply latency averages
  - initiates_ratio_by_contact   — who starts conversations per contact

These metrics fill CadenceProfile fields (models/user_model.py) that were
previously left empty despite the raw data already being collected.

Test strategy:
  - Unit tests inject pre-built data dicts into _compute_* methods directly.
  - End-to-end tests exercise the full extract() → _update_profile() path
    to verify initiation tracking works through the signal pipeline.
"""

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor(db, user_model_store):
    """CadenceExtractor wired to test databases."""
    return CadenceExtractor(db=db, user_model_store=user_model_store)


def _base_data(**overrides) -> dict:
    """Build a minimal cadence data dict with sensible defaults.

    Accepts keyword overrides for any top-level key (e.g.,
    per_contact_response_times, per_contact_initiations).
    """
    data = {
        "response_times": [],
        "hourly_activity": {},
        "daily_activity": {},
        "per_contact_response_times": {},
        "per_channel_response_times": {},
        "per_contact_initiations": {},
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# _compute_contact_response_times unit tests
# ---------------------------------------------------------------------------


class TestAvgResponseTimeByContact:
    """Tests for avg_response_time_by_contact derived metric."""

    def test_avg_response_time_by_contact_computed(self, extractor):
        """Contacts with 3+ samples should have their average computed."""
        data = _base_data(per_contact_response_times={
            "alice@gmail.com": [300.0, 600.0, 900.0],  # avg = 600
            "bob@work.com": [1200.0, 2400.0, 3600.0],  # avg = 2400
        })

        extractor._compute_contact_response_times(data)

        assert "avg_response_time_by_contact" in data
        avgs = data["avg_response_time_by_contact"]
        assert abs(avgs["alice@gmail.com"] - 600.0) < 0.01
        assert abs(avgs["bob@work.com"] - 2400.0) < 0.01

    def test_min_sample_threshold_contact(self, extractor):
        """Contacts with fewer than 3 samples should be excluded."""
        data = _base_data(per_contact_response_times={
            "alice@gmail.com": [300.0, 600.0],  # Only 2 — excluded
            "bob@work.com": [100.0, 200.0, 300.0],  # 3 — included
        })

        extractor._compute_contact_response_times(data)

        avgs = data.get("avg_response_time_by_contact", {})
        assert "alice@gmail.com" not in avgs
        assert "bob@work.com" in avgs

    def test_empty_contact_data_is_noop(self, extractor):
        """Empty per_contact_response_times should not set the derived field."""
        data = _base_data(per_contact_response_times={})

        extractor._compute_contact_response_times(data)

        assert "avg_response_time_by_contact" not in data

    def test_all_contacts_below_threshold(self, extractor):
        """When every contact has <3 samples, the derived field is not set."""
        data = _base_data(per_contact_response_times={
            "alice@gmail.com": [300.0],
            "bob@work.com": [100.0, 200.0],
        })

        extractor._compute_contact_response_times(data)

        assert "avg_response_time_by_contact" not in data

    def test_phone_number_contacts_included(self, extractor):
        """Phone number contacts should still get averages (unlike domain grouping)."""
        data = _base_data(per_contact_response_times={
            "+15551234567": [60.0, 90.0, 120.0],  # avg = 90
        })

        extractor._compute_contact_response_times(data)

        avgs = data.get("avg_response_time_by_contact", {})
        assert "+15551234567" in avgs
        assert abs(avgs["+15551234567"] - 90.0) < 0.01


# ---------------------------------------------------------------------------
# _compute_channel_response_times unit tests
# ---------------------------------------------------------------------------


class TestAvgResponseTimeByChannel:
    """Tests for avg_response_time_by_channel derived metric."""

    def test_avg_response_time_by_channel_computed(self, extractor):
        """Channels with 3+ samples should have their average computed."""
        data = _base_data(per_channel_response_times={
            "email": [3600.0, 7200.0, 1800.0],  # avg = 4200
            "signal": [60.0, 120.0, 180.0],     # avg = 120
        })

        extractor._compute_channel_response_times(data)

        assert "avg_response_time_by_channel" in data
        avgs = data["avg_response_time_by_channel"]
        assert abs(avgs["email"] - 4200.0) < 0.01
        assert abs(avgs["signal"] - 120.0) < 0.01

    def test_min_sample_threshold_channel(self, extractor):
        """Channels with fewer than 3 samples should be excluded."""
        data = _base_data(per_channel_response_times={
            "email": [3600.0, 7200.0],           # 2 — excluded
            "signal": [60.0, 120.0, 180.0],      # 3 — included
        })

        extractor._compute_channel_response_times(data)

        avgs = data.get("avg_response_time_by_channel", {})
        assert "email" not in avgs
        assert "signal" in avgs

    def test_empty_channel_data_is_noop(self, extractor):
        """Empty per_channel_response_times should not set the derived field."""
        data = _base_data(per_channel_response_times={})

        extractor._compute_channel_response_times(data)

        assert "avg_response_time_by_channel" not in data


# ---------------------------------------------------------------------------
# _compute_initiates_ratio unit tests
# ---------------------------------------------------------------------------


class TestInitiatesRatioByContact:
    """Tests for initiates_ratio_by_contact derived metric."""

    def test_initiates_ratio_computation(self, extractor):
        """Ratio should be user_count / (user_count + contact_count)."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 3, "contact": 1},  # ratio = 0.75
            "bob@work.com": {"user": 1, "contact": 3},     # ratio = 0.25
        })

        extractor._compute_initiates_ratio(data)

        assert "initiates_ratio_by_contact" in data
        ratios = data["initiates_ratio_by_contact"]
        assert abs(ratios["alice@gmail.com"] - 0.75) < 0.01
        assert abs(ratios["bob@work.com"] - 0.25) < 0.01

    def test_initiates_ratio_min_threshold(self, extractor):
        """Contacts with fewer than 3 total initiations should be excluded."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 1, "contact": 1},  # total 2 — excluded
            "bob@work.com": {"user": 2, "contact": 1},     # total 3 — included
        })

        extractor._compute_initiates_ratio(data)

        ratios = data.get("initiates_ratio_by_contact", {})
        assert "alice@gmail.com" not in ratios
        assert "bob@work.com" in ratios

    def test_empty_initiations_is_noop(self, extractor):
        """Empty per_contact_initiations should not set the derived field."""
        data = _base_data(per_contact_initiations={})

        extractor._compute_initiates_ratio(data)

        assert "initiates_ratio_by_contact" not in data

    def test_all_user_initiated(self, extractor):
        """If the user always initiates, ratio should be 1.0."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 5, "contact": 0},
        })

        extractor._compute_initiates_ratio(data)

        ratios = data["initiates_ratio_by_contact"]
        assert abs(ratios["alice@gmail.com"] - 1.0) < 0.01

    def test_all_contact_initiated(self, extractor):
        """If the contact always initiates, ratio should be 0.0."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 0, "contact": 4},
        })

        extractor._compute_initiates_ratio(data)

        ratios = data["initiates_ratio_by_contact"]
        assert abs(ratios["alice@gmail.com"] - 0.0) < 0.01

    def test_all_contacts_below_threshold(self, extractor):
        """When every contact has <3 total initiations, field is not set."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 1, "contact": 0},
            "bob@work.com": {"user": 0, "contact": 1},
        })

        extractor._compute_initiates_ratio(data)

        assert "initiates_ratio_by_contact" not in data


# ---------------------------------------------------------------------------
# Initiation tracking in extract() — integration tests
# ---------------------------------------------------------------------------


class TestInitiationTrackingExtract:
    """Tests for conversation initiation signal emission from extract()."""

    def test_outbound_non_reply_emits_user_initiation(self, extractor):
        """An outbound message that is not a reply should emit a user initiation signal."""
        event = {
            "id": "init-out-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "to_addresses": ["alice@gmail.com"],
                "body": "Hey, want to grab lunch?",
            },
            "metadata": {},
        }

        signals = extractor.extract(event)

        initiation_signals = [s for s in signals if s["type"] == "cadence_initiation"]
        assert len(initiation_signals) == 1
        assert initiation_signals[0]["contact_id"] == "alice@gmail.com"
        assert initiation_signals[0]["initiator"] == "user"

    def test_inbound_non_reply_emits_contact_initiation(self, extractor):
        """An inbound message that is not a reply should emit a contact initiation signal."""
        event = {
            "id": "init-in-1",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "sender": "bob@work.com",
                "body": "Quick question about the project",
            },
            "metadata": {},
        }

        signals = extractor.extract(event)

        initiation_signals = [s for s in signals if s["type"] == "cadence_initiation"]
        assert len(initiation_signals) == 1
        assert initiation_signals[0]["contact_id"] == "bob@work.com"
        assert initiation_signals[0]["initiator"] == "contact"

    def test_reply_does_not_emit_initiation(self, extractor):
        """A reply message should not emit an initiation signal."""
        event = {
            "id": "reply-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": "original-msg-123",
                "to_addresses": ["alice@gmail.com"],
                "body": "Sounds good!",
            },
            "metadata": {},
        }

        signals = extractor.extract(event)

        initiation_signals = [s for s in signals if s["type"] == "cadence_initiation"]
        assert len(initiation_signals) == 0

    def test_inbound_with_from_address_fallback(self, extractor):
        """Inbound initiation should fall back to from_address if sender is absent."""
        event = {
            "id": "init-in-2",
            "type": EventType.MESSAGE_RECEIVED.value,
            "source": "signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "from_address": "+15551234567",
                "body": "Hey there!",
            },
            "metadata": {},
        }

        signals = extractor.extract(event)

        initiation_signals = [s for s in signals if s["type"] == "cadence_initiation"]
        assert len(initiation_signals) == 1
        assert initiation_signals[0]["contact_id"] == "+15551234567"
        assert initiation_signals[0]["initiator"] == "contact"

    def test_initiations_persisted_to_profile(self, extractor, user_model_store):
        """Initiation counts should accumulate in the stored profile."""
        # Send 3 user-initiated messages to alice
        for i in range(3):
            extractor.extract({
                "id": f"user-init-{i}",
                "type": EventType.EMAIL_SENT.value,
                "source": "email",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "to_addresses": ["alice@gmail.com"],
                    "body": f"Starting conversation {i}",
                },
                "metadata": {},
            })

        # Receive 2 contact-initiated messages from alice
        for i in range(2):
            extractor.extract({
                "id": f"contact-init-{i}",
                "type": EventType.EMAIL_RECEIVED.value,
                "source": "email",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "sender": "alice@gmail.com",
                    "body": f"Alice starts conversation {i}",
                },
                "metadata": {},
            })

        profile = user_model_store.get_signal_profile("cadence")
        initiations = profile["data"]["per_contact_initiations"]
        assert initiations["alice@gmail.com"]["user"] == 3
        assert initiations["alice@gmail.com"]["contact"] == 2

        # With 5 total (>= 3), the ratio should be computed.
        ratios = profile["data"].get("initiates_ratio_by_contact", {})
        assert "alice@gmail.com" in ratios
        assert abs(ratios["alice@gmail.com"] - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Edge cases — empty data handling
# ---------------------------------------------------------------------------


class TestEmptyDataEdgeCases:
    """Verify that empty data dicts produce empty results without errors."""

    def test_all_empty_dicts(self, extractor):
        """Completely empty data should not raise and should not set any derived fields."""
        data = _base_data()

        # Should not raise any exceptions.
        extractor._compute_derived_metrics(data)

        assert "avg_response_time_by_contact" not in data
        assert "avg_response_time_by_channel" not in data
        assert "initiates_ratio_by_contact" not in data

    def test_missing_keys_handled_gracefully(self, extractor):
        """Even if raw data keys are completely absent, derived computation should not crash."""
        data = {
            "response_times": [],
            "hourly_activity": {},
            "daily_activity": {},
        }

        # Should not raise KeyError or any other exception.
        extractor._compute_derived_metrics(data)

        assert "avg_response_time_by_contact" not in data
        assert "avg_response_time_by_channel" not in data
        assert "initiates_ratio_by_contact" not in data

    def test_initiations_with_zero_counts(self, extractor):
        """Contacts with zero counts should be handled gracefully."""
        data = _base_data(per_contact_initiations={
            "alice@gmail.com": {"user": 0, "contact": 0},
        })

        extractor._compute_initiates_ratio(data)

        # Total is 0 (< 3), so should be excluded.
        assert "initiates_ratio_by_contact" not in data
