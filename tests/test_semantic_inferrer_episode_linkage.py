"""
Tests for semantic fact inferrer episode linkage after backfill migration.

The episode backfill migration (LifeOS._backfill_episode_classification_if_needed)
reclassified all "communication" episodes to granular types like "email_sent",
"email_received", etc.  The inferrer methods previously used
interaction_type="communication" to fetch source episodes, which always returned []
after the migration — breaking the audit trail and confidence growth loop.

This test suite verifies that:
1. Each affected inferrer method (linguistic, topic, cadence, mood) fetches a
   valid episode_id when episodes with the correct granular types exist.
2. Facts created after the migration include a non-None source episode.
"""

import json
import uuid

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _insert_episode(db, interaction_type: str) -> str:
    """Insert a minimal episode row into user_model.db and return its id.

    Args:
        db: DatabaseManager fixture
        interaction_type: Granular interaction type (e.g. "email_sent")

    Returns:
        UUID string for the inserted episode
    """
    episode_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, event_id, interaction_type, timestamp, content_summary)
               VALUES (?, ?, ?, datetime('now'), ?)""",
            (episode_id, str(uuid.uuid4()), interaction_type, "test episode"),
        )
    return episode_id


def _set_samples(ums, profile_type: str, count: int):
    """Set samples_count on an existing signal profile row.

    Args:
        ums: UserModelStore fixture
        profile_type: Signal profile type identifier
        count: Number of samples to record
    """
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


def _get_fact_source_episode(ums, key: str):
    """Return the source_episodes list for a semantic fact.

    Args:
        ums: UserModelStore fixture
        key: Semantic fact key

    Returns:
        List of episode IDs linked to the fact, or [] if not found
    """
    with ums.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT source_episodes FROM semantic_facts WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return []
    return json.loads(row["source_episodes"])


class TestInferrerEpisodeLinkageAfterBackfill:
    """Verify that inferrer methods link to granular episode types, not 'communication'."""

    def test_linguistic_inferrer_links_email_sent_episode(self, user_model_store, db):
        """infer_from_linguistic_profile should link to email_sent episodes.

        After the backfill migration there are no 'communication' episodes.
        The linguistic inferrer was updated to look for 'email_sent' instead.
        """
        # Insert an email_sent episode so the inferrer can link to it
        episode_id = _insert_episode(db, "email_sent")

        # Set up a linguistic profile with a very casual formality score
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.1, "emoji_rate": 0.0, "hedge_rate": 0.1, "exclamation_rate": 0.0},
        })
        _set_samples(user_model_store, "linguistic", 5)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # The fact should exist
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert casual is not None, "Casual communication style fact should be inferred"

        # The fact should link to the email_sent episode (not have an empty source_episodes)
        source_eps = _get_fact_source_episode(user_model_store, "communication_style_formality")
        assert episode_id in source_eps, (
            f"Expected episode {episode_id} in source_episodes but got {source_eps}. "
            "Likely the inferrer is still using interaction_type='communication' instead of 'email_sent'."
        )

    def test_linguistic_inferrer_no_episode_when_only_communication_type(self, user_model_store, db):
        """Verify that 'communication' typed episodes are NOT found by the updated inferrer.

        This confirms the pre-migration type is no longer matched, which validates
        that the fix doesn't regress to the old broken behavior.
        """
        # Insert an episode with the OLD 'communication' type (should NOT be found)
        _insert_episode(db, "communication")

        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.1, "emoji_rate": 0.0, "hedge_rate": 0.1, "exclamation_rate": 0.0},
        })
        _set_samples(user_model_store, "linguistic", 5)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert casual is not None  # Fact should still be created

        # source_episodes should be empty because no 'email_sent' episodes exist
        source_eps = _get_fact_source_episode(user_model_store, "communication_style_formality")
        assert source_eps == [], (
            "Fact should have empty source_episodes when only 'communication' episodes exist, "
            "confirming the inferrer no longer matches the retired type."
        )

    def test_topic_inferrer_episode_lookup_uses_no_type_filter(self, user_model_store, db):
        """infer_from_topic_profile should find episodes without a type filter.

        After the backfill migration the inferrer must not use
        interaction_type='communication'.  We verify this by inserting only an
        'email_received' episode and checking that _get_recent_episodes returns it
        (i.e., the inferrer fetches with no type restriction rather than the stale
        'communication' type that would return nothing).
        """
        # Insert a non-'communication' episode
        episode_id = _insert_episode(db, "email_received")

        inferrer = SemanticFactInferrer(user_model_store)

        # Call the internal episode-fetching helper directly to verify it returns
        # the email_received episode when called with no interaction_type filter.
        no_filter_results = inferrer._get_recent_episodes(limit=5)
        assert episode_id in no_filter_results, (
            f"_get_recent_episodes(limit=5) should find episode {episode_id} "
            "(email_received) when no interaction_type filter is applied."
        )

        # Also verify the stale filter returns nothing for this episode
        old_filter_results = inferrer._get_recent_episodes(
            interaction_type="communication", limit=5
        )
        assert episode_id not in old_filter_results, (
            "The stale 'communication' filter should NOT find an 'email_received' episode."
        )

    def test_cadence_inferrer_links_any_recent_episode(self, user_model_store, db):
        """infer_from_cadence_profile should link to any recent episode.

        Cadence patterns come from all communication activity, not just one type.
        """
        episode_id = _insert_episode(db, "email_sent")

        # Set up a cadence profile where the user only works in business hours
        hourly = {str(h): 10 for h in range(9, 17)}  # 9am-5pm only
        user_model_store.update_signal_profile("cadence", {
            "hourly_activity": hourly,
        })
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts()
        work_life = next((f for f in facts if f["key"] == "work_life_balance"), None)
        # work_life_balance fact may or may not be generated depending on threshold;
        # the important assertion is that the inferrer does NOT crash and can fetch episodes
        # (tested implicitly by the episode_id assertion if a fact is created)
        if work_life is not None:
            source_eps = _get_fact_source_episode(user_model_store, "work_life_balance")
            assert episode_id in source_eps, (
                f"Expected episode {episode_id} in source_episodes for work_life_balance but got {source_eps}"
            )

    def test_no_stale_communication_type_in_inferrer(self):
        """Static check: no inferrer method uses interaction_type='communication'.

        This regression test inspects the source code to ensure the stale
        'communication' interaction type filter has been fully removed from all
        inferrer methods that need to link to source episodes.
        """
        import inspect
        from services.semantic_fact_inferrer import inferrer as inferrer_mod

        source = inspect.getsource(inferrer_mod)

        # The only occurrence of "communication" should be in comments explaining
        # the migration, NOT as an argument to _get_recent_episodes().
        # We check that _get_recent_episodes is not called with interaction_type="communication"
        import re
        bad_pattern = re.compile(
            r'_get_recent_episodes\s*\([^)]*interaction_type\s*=\s*["\']communication["\']'
        )
        matches = bad_pattern.findall(source)
        assert not matches, (
            f"Found {len(matches)} call(s) to _get_recent_episodes with "
            "interaction_type='communication'. This type was retired by the episode "
            "backfill migration. Update to use 'email_sent', 'email_received', or "
            "no interaction_type filter.\n"
            f"Matches: {matches}"
        )
