"""
Tests for Semantic Fact Inference — Inbound Linguistic Profile

Verifies that the SemanticFactInferrer correctly derives facts about the user's
communication environment from the 'linguistic_inbound' signal profile (messages
received by the user from contacts).
"""

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestInboundLinguisticInference:
    """Tests for infer_from_inbound_linguistic_profile."""

    def _setup_inbound_profile(self, ums, per_contact_averages, samples_count=50):
        """Helper to create a linguistic_inbound profile with given per-contact averages."""
        profile_data = {
            "per_contact": {},
            "per_contact_averages": per_contact_averages,
        }
        ums.update_signal_profile("linguistic_inbound", profile_data)
        _set_samples(ums, "linguistic_inbound", samples_count)

    # -------------------------------------------------------------------
    # Formality environment inference
    # -------------------------------------------------------------------

    def test_infer_formal_professional_environment(self, user_model_store):
        """High average inbound formality (>0.7) infers formal professional environment."""
        self._setup_inbound_profile(user_model_store, {
            "alice@work.com": {
                "formality": 0.85,
                "question_rate": 0.1,
                "hedge_rate": 0.05,
                "samples_count": 30,
            },
            "bob@work.com": {
                "formality": 0.75,
                "question_rate": 0.2,
                "hedge_rate": 0.03,
                "samples_count": 20,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["type"] == "inbound_linguistic"
        assert result["processed"] is True

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is not None
        assert env_fact["value"] == "formal_professional_environment"
        assert env_fact["confidence"] > 0.0

    def test_infer_casual_informal_environment(self, user_model_store):
        """Low average inbound formality (<0.3) infers casual informal environment."""
        self._setup_inbound_profile(user_model_store, {
            "friend1@example.com": {
                "formality": 0.15,
                "question_rate": 0.1,
                "hedge_rate": 0.05,
                "samples_count": 25,
            },
            "friend2@example.com": {
                "formality": 0.2,
                "question_rate": 0.15,
                "hedge_rate": 0.08,
                "samples_count": 25,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is not None
        assert env_fact["value"] == "casual_informal_environment"

    # -------------------------------------------------------------------
    # Question intensity inference
    # -------------------------------------------------------------------

    def test_infer_frequently_asked_questions(self, user_model_store):
        """High inbound question rate (>0.5) infers user is a go-to expert."""
        self._setup_inbound_profile(user_model_store, {
            "colleague@work.com": {
                "formality": 0.5,
                "question_rate": 0.7,
                "hedge_rate": 0.05,
                "samples_count": 40,
            },
            "intern@work.com": {
                "formality": 0.5,
                "question_rate": 0.6,
                "hedge_rate": 0.1,
                "samples_count": 10,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        question_fact = next((f for f in facts if f["key"] == "inbound_question_intensity"), None)
        assert question_fact is not None
        assert question_fact["value"] == "frequently_asked_questions"

    def test_no_question_fact_when_rate_is_low(self, user_model_store):
        """Question rate below 0.5 does not produce question intensity fact."""
        self._setup_inbound_profile(user_model_store, {
            "contact@example.com": {
                "formality": 0.5,
                "question_rate": 0.2,
                "hedge_rate": 0.05,
                "samples_count": 50,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        question_fact = next((f for f in facts if f["key"] == "inbound_question_intensity"), None)
        assert question_fact is None

    # -------------------------------------------------------------------
    # Hedge rate / cautious senders inference
    # -------------------------------------------------------------------

    def test_infer_cautious_senders(self, user_model_store):
        """High inbound hedge rate (>0.2) infers cautious senders."""
        self._setup_inbound_profile(user_model_store, {
            "cautious@example.com": {
                "formality": 0.5,
                "question_rate": 0.1,
                "hedge_rate": 0.35,
                "samples_count": 50,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        style_fact = next((f for f in facts if f["key"] == "inbound_communication_style"), None)
        assert style_fact is not None
        assert style_fact["value"] == "cautious_senders"

    # -------------------------------------------------------------------
    # Insufficient samples
    # -------------------------------------------------------------------

    def test_skip_inference_with_insufficient_samples(self, user_model_store):
        """Inference is skipped when samples_count < 10."""
        self._setup_inbound_profile(
            user_model_store,
            {"contact@example.com": {"formality": 0.9, "question_rate": 0.8, "hedge_rate": 0.3, "samples_count": 5}},
            samples_count=5,
        )

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is False
        assert "insufficient" in result["reason"]

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_skip_inference_with_no_profile(self, user_model_store):
        """Inference is skipped when the inbound linguistic profile does not exist."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is False
        assert "insufficient" in result["reason"]

    # -------------------------------------------------------------------
    # Per-contact formality distribution
    # -------------------------------------------------------------------

    def test_per_contact_formality_reinforcement(self, user_model_store):
        """When >70% of contacts are formal, the formal environment fact is reinforced."""
        # 4 out of 5 contacts (80%) have formality > 0.7
        self._setup_inbound_profile(user_model_store, {
            "formal1@work.com": {"formality": 0.85, "question_rate": 0.1, "hedge_rate": 0.02, "samples_count": 10},
            "formal2@work.com": {"formality": 0.75, "question_rate": 0.1, "hedge_rate": 0.02, "samples_count": 10},
            "formal3@work.com": {"formality": 0.80, "question_rate": 0.1, "hedge_rate": 0.02, "samples_count": 10},
            "formal4@work.com": {"formality": 0.72, "question_rate": 0.1, "hedge_rate": 0.02, "samples_count": 10},
            "casual@friend.com": {"formality": 0.2, "question_rate": 0.1, "hedge_rate": 0.02, "samples_count": 10},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is not None
        assert env_fact["value"] == "formal_professional_environment"

    # -------------------------------------------------------------------
    # Integration with run_all_inference
    # -------------------------------------------------------------------

    def test_run_all_inference_includes_inbound_linguistic(self, user_model_store):
        """run_all_inference processes the inbound linguistic profile."""
        self._setup_inbound_profile(user_model_store, {
            "contact@work.com": {
                "formality": 0.85,
                "question_rate": 0.1,
                "hedge_rate": 0.05,
                "samples_count": 30,
            },
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is not None
        assert env_fact["value"] == "formal_professional_environment"

    # -------------------------------------------------------------------
    # Confidence scaling
    # -------------------------------------------------------------------

    def test_confidence_scales_with_sample_count(self, user_model_store):
        """Lower sample counts produce lower confidence via _early_inference_confidence."""
        # 15 samples — below the 50-sample threshold for full confidence
        self._setup_inbound_profile(
            user_model_store,
            {"contact@work.com": {"formality": 0.85, "question_rate": 0.1, "hedge_rate": 0.05, "samples_count": 15}},
            samples_count=15,
        )

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is not None
        # With 15 samples / 50 threshold, base confidence is scaled down from 0.5
        # The exact value depends on _early_inference_confidence scaling
        assert env_fact["confidence"] > 0.0
        assert env_fact["confidence"] <= 0.95

    def test_no_per_contact_data_skips_inference(self, user_model_store):
        """Inference is skipped when profile exists but has no per-contact averages."""
        profile_data = {
            "per_contact": {},
            "per_contact_averages": {},
        }
        user_model_store.update_signal_profile("linguistic_inbound", profile_data)
        _set_samples(user_model_store, "linguistic_inbound", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is False
        assert "no per-contact data" in result["reason"]

    # -------------------------------------------------------------------
    # Mixed environment (neutral formality)
    # -------------------------------------------------------------------

    def test_neutral_formality_creates_no_environment_fact(self, user_model_store):
        """Formality between 0.3 and 0.7 does not create an environment fact."""
        self._setup_inbound_profile(user_model_store, {
            "contact1@example.com": {"formality": 0.5, "question_rate": 0.1, "hedge_rate": 0.05, "samples_count": 25},
            "contact2@example.com": {"formality": 0.45, "question_rate": 0.1, "hedge_rate": 0.05, "samples_count": 25},
        })

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next((f for f in facts if f["key"] == "inbound_communication_environment"), None)
        assert env_fact is None
