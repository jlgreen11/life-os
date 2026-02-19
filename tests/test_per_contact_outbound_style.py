"""
Tests for per-contact outbound style averages in the linguistic profile.

PR #281 adds ``per_contact_averages`` to the ``linguistic`` signal profile
so that ``ContextAssembler.assemble_draft_context()`` can show the LLM how
the user specifically writes to a given contact, rather than their global
average style.

Coverage:
  - LinguisticExtractor._update_profile() computes per_contact_averages after
    enough samples (>= _MIN_PER_CONTACT_SAMPLES) are collected per contact.
  - Contacts with fewer than _MIN_PER_CONTACT_SAMPLES samples are excluded.
  - Per-contact averages correctly mirror the inbound-profile format so
    assemble_draft_context() can read them with the same field names.
  - assemble_draft_context() uses per-contact averages (not global) when
    available and labels them distinctly.
  - assemble_draft_context() falls back to global averages when no per-contact
    data exists.
  - A formality-delta note is appended when the per-contact formality differs
    from the global baseline by more than 0.15.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from services.signal_extractor.linguistic import LinguisticExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extractor(ums: MagicMock) -> LinguisticExtractor:
    """Build a LinguisticExtractor wired to a mock UserModelStore."""
    extractor = LinguisticExtractor.__new__(LinguisticExtractor)
    extractor.ums = ums
    extractor.db = MagicMock()
    return extractor


def _profile_data(samples, per_contact):
    """Build a minimal existing-profile dict."""
    return {"data": {"samples": samples, "per_contact": per_contact}}


def _make_signal(contact_id: str, metrics: dict) -> dict:
    """Build a minimal outbound linguistic signal dict."""
    return {
        "type": "linguistic",
        "timestamp": "2026-01-01T10:00:00Z",
        "contact_id": contact_id,
        "metrics": metrics,
    }


def _flat_metrics(
    formality: float = 0.5,
    avg_sentence_length: float = 10.0,
    word_count: int = 20,
    hedge_rate: float = 0.1,
    assertion_rate: float = 0.1,
    exclamation_rate: float = 0.0,
    question_rate: float = 0.2,
    ellipsis_rate: float = 0.0,
    unique_word_ratio: float = 0.6,
    emoji_count: int = 0,
    profanity_count: int = 0,
    greeting_detected: str | None = None,
    closing_detected: str | None = None,
) -> dict:
    """Return a complete metrics snapshot that mirrors what extract() produces."""
    return {
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "unique_word_ratio": unique_word_ratio,
        "formality": formality,
        "hedge_rate": hedge_rate,
        "assertion_rate": assertion_rate,
        "exclamation_rate": exclamation_rate,
        "question_rate": question_rate,
        "ellipsis_rate": ellipsis_rate,
        "emoji_count": emoji_count,
        "emojis_used": [],
        "profanity_count": profanity_count,
        "greeting_detected": greeting_detected,
        "closing_detected": closing_detected,
    }


# ---------------------------------------------------------------------------
# Tests: per_contact_averages computed in _update_profile
# ---------------------------------------------------------------------------

class TestPerContactAveragesComputed:
    """LinguisticExtractor._update_profile() populates per_contact_averages."""

    def test_per_contact_averages_computed_after_min_samples(self, user_model_store):
        """Averages are computed for a contact once MIN_PER_CONTACT_SAMPLES
        messages have been collected."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES

        # Feed exactly min_samples messages to contact alice@example.com.
        for _ in range(min_samples):
            signal = _make_signal("alice@example.com", _flat_metrics(formality=0.8))
            extractor._update_profile(signal)

        profile = user_model_store.get_signal_profile("linguistic")
        assert profile is not None, "profile must exist after samples"
        pca = profile["data"].get("per_contact_averages", {})
        assert "alice@example.com" in pca, (
            "alice should appear in per_contact_averages after "
            f"{min_samples} messages"
        )

    def test_per_contact_averages_absent_below_min_samples(self, user_model_store):
        """Contacts with fewer than MIN_PER_CONTACT_SAMPLES samples are excluded
        from per_contact_averages to avoid single-message noise."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES

        # Feed fewer than the minimum.
        for _ in range(min_samples - 1):
            signal = _make_signal("bob@example.com", _flat_metrics(formality=0.3))
            extractor._update_profile(signal)

        profile = user_model_store.get_signal_profile("linguistic")
        pca = profile["data"].get("per_contact_averages", {})
        assert "bob@example.com" not in pca, (
            "bob should NOT appear in per_contact_averages — "
            f"only {min_samples - 1} messages (below threshold)"
        )

    def test_per_contact_averages_correct_values(self, user_model_store):
        """Computed formality average reflects the actual sample values."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES

        formalities = [0.6, 0.8, 0.7]
        assert len(formalities) >= min_samples, "test data must meet minimum"
        for f in formalities:
            signal = _make_signal("carol@example.com", _flat_metrics(formality=f))
            extractor._update_profile(signal)

        profile = user_model_store.get_signal_profile("linguistic")
        pca = profile["data"]["per_contact_averages"]
        avg_formality = pca["carol@example.com"]["formality"]
        expected = sum(formalities) / len(formalities)
        assert abs(avg_formality - expected) < 0.01, (
            f"formality avg {avg_formality:.3f} should equal {expected:.3f}"
        )

    def test_per_contact_averages_includes_samples_count(self, user_model_store):
        """samples_count field reflects the number of messages analysed."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES
        n = min_samples + 2  # More than min to confirm count is tracked

        for _ in range(n):
            signal = _make_signal("dave@example.com", _flat_metrics())
            extractor._update_profile(signal)

        profile = user_model_store.get_signal_profile("linguistic")
        count = profile["data"]["per_contact_averages"]["dave@example.com"]["samples_count"]
        assert count == n, f"samples_count should be {n}, got {count}"

    def test_per_contact_averages_all_metric_keys_present(self, user_model_store):
        """All expected metric keys are present in the per-contact averages dict."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES

        for _ in range(min_samples):
            signal = _make_signal("eve@example.com", _flat_metrics())
            extractor._update_profile(signal)

        profile = user_model_store.get_signal_profile("linguistic")
        contact_avg = profile["data"]["per_contact_averages"]["eve@example.com"]

        required_keys = {
            "avg_sentence_length", "formality", "hedge_rate", "assertion_rate",
            "exclamation_rate", "question_rate", "ellipsis_rate",
            "unique_word_ratio", "emoji_rate", "samples_count",
        }
        missing = required_keys - set(contact_avg.keys())
        assert not missing, f"Missing keys in per-contact averages: {missing}"

    def test_multiple_contacts_tracked_independently(self, user_model_store):
        """Two contacts accumulate separate per-contact averages."""
        extractor = _make_extractor(user_model_store)
        min_samples = LinguisticExtractor._MIN_PER_CONTACT_SAMPLES

        for _ in range(min_samples):
            extractor._update_profile(
                _make_signal("formal@example.com", _flat_metrics(formality=0.9))
            )
        for _ in range(min_samples):
            extractor._update_profile(
                _make_signal("casual@example.com", _flat_metrics(formality=0.1))
            )

        profile = user_model_store.get_signal_profile("linguistic")
        pca = profile["data"]["per_contact_averages"]

        assert pca["formal@example.com"]["formality"] > 0.8
        assert pca["casual@example.com"]["formality"] < 0.2


# ---------------------------------------------------------------------------
# Tests: assemble_draft_context uses per-contact averages
# ---------------------------------------------------------------------------

class TestDraftContextUsesPerContactAverages:
    """assemble_draft_context() prefers per-contact averages over global ones."""

    def _build_context_assembler(self, db, ums):
        """Instantiate a ContextAssembler wired to real db and ums."""
        from services.ai_engine.context import ContextAssembler
        ca = ContextAssembler.__new__(ContextAssembler)
        ca.db = db
        ca.ums = ums
        return ca

    def _store_linguistic_profile(self, ums, global_formality: float,
                                   contact_id: str | None = None,
                                   contact_formality: float | None = None,
                                   contact_samples: int = 5):
        """Persist a linguistic profile with optional per-contact averages."""
        data: dict = {
            "samples": [_flat_metrics(formality=global_formality)],
            "per_contact": {},
            "averages": {
                "avg_sentence_length": 10.0,
                "formality": global_formality,
                "hedge_rate": 0.05,
                "assertion_rate": 0.05,
                "exclamation_rate": 0.0,
                "question_rate": 0.1,
                "ellipsis_rate": 0.0,
                "unique_word_ratio": 0.55,
                "emoji_rate": 0.0,
                "profanity_rate": 0.0,
            },
            "common_greetings": [],
            "common_closings": [],
            "per_contact_averages": {},
        }
        if contact_id and contact_formality is not None:
            data["per_contact_averages"][contact_id] = {
                "avg_sentence_length": 12.0,
                "formality": contact_formality,
                "hedge_rate": 0.02,
                "assertion_rate": 0.08,
                "exclamation_rate": 0.0,
                "question_rate": 0.15,
                "ellipsis_rate": 0.0,
                "unique_word_ratio": 0.60,
                "emoji_rate": 0.0,
                "samples_count": contact_samples,
            }
        ums.update_signal_profile("linguistic", data)

    def test_uses_per_contact_label_when_available(self, db, user_model_store):
        """Draft context uses 'style with this contact' label when per-contact
        data is available."""
        ca = self._build_context_assembler(db, user_model_store)
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.4,
            contact_id="alice@example.com",
            contact_formality=0.8,
            contact_samples=10,
        )

        result = ca.assemble_draft_context(
            contact_id="alice@example.com",
            channel="email",
            incoming_message="Hey, can we chat about the project?",
        )

        assert "style with this contact" in result, (
            "draft context should use per-contact label when data exists"
        )
        assert "10 msgs" in result, "sample count should appear in label"

    def test_falls_back_to_global_when_no_per_contact_data(self, db, user_model_store):
        """Draft context falls back to global averages for new/rare contacts."""
        ca = self._build_context_assembler(db, user_model_store)
        # Store profile with no per-contact data for this contact.
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.5,
        )

        result = ca.assemble_draft_context(
            contact_id="unknown@example.com",
            channel="email",
            incoming_message="Hello!",
        )

        assert "general style" in result, (
            "draft context should use 'general style' label when no per-contact data"
        )

    def test_per_contact_formality_value_used(self, db, user_model_store):
        """The formality value in the context reflects the per-contact average,
        not the global average."""
        ca = self._build_context_assembler(db, user_model_store)
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.3,   # casual globally
            contact_id="formal@example.com",
            contact_formality=0.92,  # formal with this contact
            contact_samples=8,
        )

        result = ca.assemble_draft_context(
            contact_id="formal@example.com",
            channel="email",
            incoming_message="Please review the attached document.",
        )

        # The per-contact formality (0.92) should appear, not global (0.3).
        assert "formality=0.92" in result, (
            "per-contact formality value should appear in draft context"
        )
        assert "formality=0.30" not in result, (
            "global formality should not replace per-contact value"
        )

    def test_formality_delta_note_shown_when_large(self, db, user_model_store):
        """A note about register difference is appended when per-contact formality
        deviates from the global baseline by more than 0.15."""
        ca = self._build_context_assembler(db, user_model_store)
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.4,   # balanced global
            contact_id="manager@example.com",
            contact_formality=0.85,  # notably more formal with manager (Δ0.45)
            contact_samples=6,
        )

        result = ca.assemble_draft_context(
            contact_id="manager@example.com",
            channel="email",
            incoming_message="Status update please.",
        )

        assert "more formal" in result, (
            "delta note should indicate 'more formal' when per-contact formality "
            "is substantially higher than global average"
        )

    def test_formality_delta_note_absent_when_small(self, db, user_model_store):
        """No delta note is appended when per-contact formality is close to global."""
        ca = self._build_context_assembler(db, user_model_store)
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.5,
            contact_id="peer@example.com",
            contact_formality=0.55,  # tiny delta (0.05 < 0.15 threshold)
            contact_samples=4,
        )

        result = ca.assemble_draft_context(
            contact_id="peer@example.com",
            channel="email",
            incoming_message="Quick question.",
        )

        assert "more formal" not in result
        assert "more casual" not in result

    def test_per_contact_below_min_samples_falls_back_to_global(
        self, db, user_model_store
    ):
        """Per-contact averages with < MIN_PER_CONTACT_SAMPLES are treated as if
        absent — the draft context falls back to global averages."""
        ca = self._build_context_assembler(db, user_model_store)
        self._store_linguistic_profile(
            user_model_store,
            global_formality=0.5,
            contact_id="new@example.com",
            contact_formality=0.9,
            contact_samples=1,  # below the minimum threshold
        )

        result = ca.assemble_draft_context(
            contact_id="new@example.com",
            channel="email",
            incoming_message="Hi there.",
        )

        assert "general style" in result, (
            "should fall back to global style when per-contact samples < min"
        )
        # Global formality 0.5 should appear, not per-contact 0.9.
        assert "formality=0.90" not in result
