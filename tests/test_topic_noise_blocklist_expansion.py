"""
Tests for the expanded topic noise blocklist in semantic fact inference.

Verifies that the expanded blocklist correctly filters:
  1. Generic English stopwords (more, here, please, free, valid, view)
  2. Generic email/marketing vocabulary (email, message, update, offer, shop)
  3. Additional CSS/font artifacts (lspace, rspace, sans, serif, line)

Also tests the _purge_noise_topic_facts() method that cleans up stale garbage
facts left behind from earlier inference cycles (before the blocklist was expanded).

Background:
  The topic profile accumulates ~96K samples that are dominated by marketing
  email content. Without filtering generic English and email words, the inferrer
  produces facts like "expertise_more", "expertise_view", "interest_please" — which
  carry no signal about the user's actual expertise or interests.
"""

import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile type."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


def _seed_stale_fact(ums, key, category, value, is_user_corrected=0):
    """Insert a stale fact directly into the database, bypassing is_user_corrected guard."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO semantic_facts
               (key, category, value, confidence, is_user_corrected)
               VALUES (?, ?, ?, 0.9, ?)""",
            (key, category, f'"{value}"', is_user_corrected),
        )


class TestGenericStopwordFiltering:
    """Verify generic English stopwords are filtered from expertise/interest facts."""

    def test_filters_generic_english_stopwords(self, user_model_store):
        """Common English words like 'more', 'here', 'please' must not become expertise facts.

        These words appear in nearly every marketing email and carry zero signal
        about user expertise. Without filtering, they crowd out real expertise
        words by dominating frequency counts.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with generic English words at high frequency
        # (>10% threshold that would trigger 'expertise' classification)
        # Use 1000 total samples so topic thresholds are reachable.
        # interest threshold: >= 5 occurrences AND > 5% frequency (>= 50/1000).
        # expertise threshold: >= 10 occurrences AND > 10% frequency (>= 100/1000).
        topic_data = {
            "topic_counts": {
                "more": 200,     # Generic word — must be filtered (20%)
                "here": 60,      # Generic word — must be filtered (6%)
                "please": 65,    # Generic word — must be filtered (6.5%)
                "free": 60,      # Generic word — must be filtered (6%)
                "valid": 55,     # Generic word — must be filtered (5.5%)
                "view": 150,     # Generic word — must be filtered (15%)
                "just": 55,      # Generic stopword — must be filtered (5.5%)
                "python": 120,   # Legitimate expertise — must pass through (12%)
                "kubernetes": 60, # Legitimate interest — must pass through (6%)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 1000)

        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        all_keys = {f["key"] for f in facts}

        # Generic English words must not create any facts
        assert "expertise_more" not in all_keys, "Generic word 'more' became expertise fact"
        assert "interest_here" not in all_keys, "Generic word 'here' became interest fact"
        assert "interest_please" not in all_keys, "Generic word 'please' became interest fact"
        assert "interest_free" not in all_keys, "Generic word 'free' became interest fact"
        assert "interest_valid" not in all_keys, "Generic word 'valid' became interest fact"
        assert "expertise_view" not in all_keys, "Generic word 'view' became expertise fact"
        assert "interest_just" not in all_keys, "Generic word 'just' became interest fact"

        # Legitimate topics must still be preserved
        assert "expertise_python" in all_keys, "Legitimate topic 'python' was incorrectly filtered"
        assert "interest_kubernetes" in all_keys, "Legitimate topic 'kubernetes' was incorrectly filtered"

    def test_filters_generic_email_vocabulary(self, user_model_store):
        """Generic email/communication words must not become expertise/interest facts.

        Words like 'email', 'message', 'update', 'offer', 'shop' dominate topic
        profiles because they appear in almost every marketing email body. They
        provide no signal about user expertise or genuine interests.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Use 1000 total samples so thresholds are reachable:
        # interest: >= 3 count AND > 3% (>30); expertise: >= 5 count AND > 8% (>80).
        topic_data = {
            "topic_counts": {
                "email": 150,    # Generic email word — must be filtered (15%)
                "message": 100,  # Generic email word — must be filtered (10%)
                "update": 95,    # Generic email word — must be filtered (9.5%)
                "offer": 70,     # Generic marketing word — must be filtered (7%)
                "shop": 60,      # Generic marketing word — must be filtered (6%)
                "account": 55,   # Generic email word — must be filtered (5.5%)
                "click": 55,     # Generic marketing word — must be filtered (5.5%)
                "chatgpt": 90,   # Legitimate topic — must pass through (9% → expertise)
                "task": 90,      # Legitimate topic — must pass through (9% → expertise)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 1000)

        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        all_keys = {f["key"] for f in facts}

        assert "expertise_email" not in all_keys, "'email' became expertise fact"
        assert "interest_message" not in all_keys, "'message' became interest fact"
        assert "interest_update" not in all_keys, "'update' became interest fact"
        assert "interest_offer" not in all_keys, "'offer' became interest fact"
        assert "interest_shop" not in all_keys, "'shop' became interest fact"
        assert "interest_account" not in all_keys, "'account' became interest fact"
        assert "interest_click" not in all_keys, "'click' became interest fact"

        # Legitimate topics must pass through (now expertise-level with lowered thresholds)
        assert "expertise_chatgpt" in all_keys, "'chatgpt' was incorrectly filtered"
        assert "expertise_task" in all_keys, "'task' was incorrectly filtered"

    def test_filters_css_font_whitespace_artifacts(self, user_model_store):
        """CSS font/whitespace tokens seen in HTML email templates must be filtered.

        Tokens like 'lspace', 'rspace', 'sans', 'serif' come from font declarations
        in HTML email templates and are CSS artifacts, not real topics.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Use 1000 total samples so thresholds are reachable.
        topic_data = {
            "topic_counts": {
                "lspace": 100,   # CSS whitespace artifact — must be filtered (10%)
                "rspace": 90,    # CSS whitespace artifact — must be filtered (9%)
                "sans": 80,      # Font family fragment — must be filtered (8%)
                "serif": 60,     # Font family fragment — must be filtered (6%)
                "line": 90,      # Generic CSS/layout word — must be filtered (9%)
                "normal": 75,    # CSS keyword — must be filtered (7.5%)
                "bold": 70,      # CSS keyword — must be filtered (7%)
                "machine-learning": 110,  # Legitimate expertise — must pass through (11%)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 1000)

        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        all_keys = {f["key"] for f in facts}

        assert "interest_lspace" not in all_keys, "'lspace' became interest fact"
        assert "interest_rspace" not in all_keys, "'rspace' became interest fact"
        assert "interest_sans" not in all_keys, "'sans' became interest fact"
        assert "interest_serif" not in all_keys, "'serif' became interest fact"
        assert "interest_line" not in all_keys, "'line' became interest fact"
        assert "interest_normal" not in all_keys, "'normal' became interest fact"
        assert "interest_bold" not in all_keys, "'bold' became interest fact"

        # Legitimate expertise must still surface (110/1000 = 11% → expertise tier)
        assert "expertise_machine-learning" in all_keys, "'machine-learning' was incorrectly filtered"


class TestPurgeNoiseFacts:
    """Verify _purge_noise_topic_facts() removes stale garbage facts."""

    def test_purges_stale_expertise_noise_facts(self, user_model_store):
        """Stale expertise_more / expertise_view facts left from old inference are removed.

        When the noise blocklist is expanded, facts stored in earlier inference
        cycles (e.g., 'expertise_more', 'expertise_view') must be deleted so the
        semantic memory reflects the updated blocklist.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Seed stale noise facts that would have been created before the expanded blocklist
        _seed_stale_fact(user_model_store, "expertise_more", "expertise", "more")
        _seed_stale_fact(user_model_store, "expertise_view", "expertise", "view")
        _seed_stale_fact(user_model_store, "interest_please", "implicit_preference", "please")
        _seed_stale_fact(user_model_store, "interest_free", "implicit_preference", "free")
        # Seed a legitimate fact that must NOT be removed
        _seed_stale_fact(user_model_store, "expertise_python", "expertise", "python")

        # Run the purge with a noise set that includes the stale words
        noise_set = {"more", "view", "please", "free", "email", "message"}
        deleted = inferrer._purge_noise_topic_facts(noise_set)

        # 4 stale facts should have been deleted
        assert deleted == 4, f"Expected 4 stale facts purged, got {deleted}"

        facts = user_model_store.get_semantic_facts()
        remaining_keys = {f["key"] for f in facts}

        # Noise facts must be gone
        assert "expertise_more" not in remaining_keys
        assert "expertise_view" not in remaining_keys
        assert "interest_please" not in remaining_keys
        assert "interest_free" not in remaining_keys

        # Legitimate fact must survive
        assert "expertise_python" in remaining_keys

    def test_purge_preserves_user_corrected_facts(self, user_model_store):
        """User-corrected facts must never be deleted, even if their key is in the noise set.

        A user might have manually corrected a fact like 'interest_free' to mean
        something specific. We must never overwrite user corrections.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Seed a user-corrected fact
        _seed_stale_fact(user_model_store, "interest_free", "implicit_preference", "free_software", is_user_corrected=1)
        # Seed a non-user-corrected noise fact
        _seed_stale_fact(user_model_store, "expertise_more", "expertise", "more", is_user_corrected=0)

        noise_set = {"more", "free", "email"}
        deleted = inferrer._purge_noise_topic_facts(noise_set)

        # Only the non-user-corrected fact should be deleted
        assert deleted == 1, f"Expected 1 fact purged (not user-corrected), got {deleted}"

        facts = user_model_store.get_semantic_facts()
        remaining_keys = {f["key"] for f in facts}

        # User-corrected fact must survive
        assert "interest_free" in remaining_keys, "User-corrected fact was incorrectly deleted"
        # Non-user-corrected noise fact must be gone
        assert "expertise_more" not in remaining_keys

    def test_purge_only_affects_expertise_and_interest_keys(self, user_model_store):
        """Purge must only delete expertise_* and interest_* keys, not other facts.

        Facts like 'chronotype', 'work_life_boundaries', 'stress_baseline' must
        not be touched even if a word like 'more' appears in their value.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Seed non-expertise/interest facts that happen to contain noise words
        _seed_stale_fact(user_model_store, "chronotype", "implicit_preference", "morning_person")
        _seed_stale_fact(user_model_store, "work_life_boundaries", "values", "flexible_boundaries")
        _seed_stale_fact(user_model_store, "stress_baseline", "implicit_preference", "low_stress")
        # Seed an expertise noise fact that SHOULD be purged
        _seed_stale_fact(user_model_store, "expertise_email", "expertise", "email")

        noise_set = {"email", "more", "free", "message"}
        deleted = inferrer._purge_noise_topic_facts(noise_set)

        # Only expertise_email should be deleted
        assert deleted == 1

        facts = user_model_store.get_semantic_facts()
        remaining_keys = {f["key"] for f in facts}

        # Non-expertise/interest facts must survive
        assert "chronotype" in remaining_keys
        assert "work_life_boundaries" in remaining_keys
        assert "stress_baseline" in remaining_keys
        # Noise expertise fact must be gone
        assert "expertise_email" not in remaining_keys

    def test_purge_empty_noise_set_is_safe(self, user_model_store):
        """Calling purge with an empty noise set must not delete anything."""
        inferrer = SemanticFactInferrer(user_model_store)

        _seed_stale_fact(user_model_store, "expertise_python", "expertise", "python")

        deleted = inferrer._purge_noise_topic_facts(set())
        assert deleted == 0

        facts = user_model_store.get_semantic_facts()
        assert any(f["key"] == "expertise_python" for f in facts)

    def test_purge_returns_zero_when_nothing_to_delete(self, user_model_store):
        """Purge on a clean database with no noise facts returns 0."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Only seed a legitimate fact
        _seed_stale_fact(user_model_store, "expertise_machine_learning", "expertise", "machine_learning")

        noise_set = {"more", "free", "email", "view", "please"}
        deleted = inferrer._purge_noise_topic_facts(noise_set)

        assert deleted == 0

    def test_infer_from_topic_profile_triggers_purge(self, user_model_store):
        """infer_from_topic_profile() automatically purges stale noise facts.

        The purge is called at the start of each inference run so that expanding
        the blocklist takes effect immediately on the next inference cycle, without
        requiring a manual cleanup step.
        """
        inferrer = SemanticFactInferrer(user_model_store)

        # Seed stale noise facts from a hypothetical previous cycle
        _seed_stale_fact(user_model_store, "expertise_more", "expertise", "more")
        _seed_stale_fact(user_model_store, "interest_please", "implicit_preference", "please")

        # Run inference with a topic profile that only has legitimate topics
        topic_data = {
            "topic_counts": {
                "more": 15000,    # Still in profile (will be blocked, not re-added)
                "python": 8000,   # Legitimate — will create new fact (8.3% → expertise)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 96000)

        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        all_keys = {f["key"] for f in facts}

        # Stale noise facts must be gone (purged by inference)
        assert "expertise_more" not in all_keys
        assert "interest_please" not in all_keys

        # Legitimate new fact must be present (expertise with lowered thresholds)
        assert "expertise_python" in all_keys


class TestLogMessageUpdated:
    """Verify log messages use the updated 'noise tokens' terminology."""

    def test_log_message_says_noise_tokens(self, user_model_store, caplog):
        """Log message must say 'noise tokens' not 'HTML/CSS tokens'."""
        import logging
        caplog.set_level(logging.INFO)

        inferrer = SemanticFactInferrer(user_model_store)

        topic_data = {
            "topic_counts": {
                "nbsp": 150,    # HTML entity
                "more": 140,    # Generic English stopword (new blocklist)
                "email": 130,   # Generic email word (new blocklist)
                "python": 120,  # Legitimate
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 400)

        inferrer.infer_from_topic_profile()

        # Log must say "noise tokens" (not the old "HTML/CSS tokens")
        assert any(
            "noise tokens" in record.message for record in caplog.records
        ), "Log message should say 'noise tokens'"
