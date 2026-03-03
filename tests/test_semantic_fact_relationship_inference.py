"""
Tests for SemanticFactInferrer — relationship profile inference edge cases.

Covers the bare-return bug fix at inferrer.py:300, defensive handling of
corrupted contact data, and end-to-end verification that run_all_inference()
never produces None results.
"""

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestRelationshipInferenceEdgeCases:
    """Edge-case tests for infer_from_relationship_profile."""

    def test_relationship_inference_empty_contacts(self, user_model_store):
        """Return a proper dict (not None) when the contact list is empty."""
        profile_data = {"contacts": {}}
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert isinstance(result, dict)
        assert result["type"] == "relationship"

    def test_relationship_inference_only_marketing_contacts(self, user_model_store):
        """Return expected dict when all contacts are marketing/noreply senders.

        The marketing filter at line 284-288 strips these, leaving
        human_contacts empty — which hits the guard at line 290.
        """
        profile_data = {
            "contacts": {
                "noreply@example.com": {
                    "interaction_count": 50,
                    "outbound_count": 2,
                },
                "newsletter@store.com": {
                    "interaction_count": 30,
                    "outbound_count": 1,
                },
                "marketing@bigcorp.com": {
                    "interaction_count": 20,
                    "outbound_count": 1,
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert isinstance(result, dict)
        assert result["type"] == "relationship"
        assert "processed" in result

    def test_relationship_inference_contacts_with_zero_interactions(self, user_model_store):
        """Contacts with interaction_count: 0 (below the < 3 threshold) are skipped gracefully."""
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 0,
                    "outbound_count": 1,
                },
                "bob@example.com": {
                    "interaction_count": 0,
                    "outbound_count": 1,
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert isinstance(result, dict)
        assert result["type"] == "relationship"
        assert result["processed"] is True

    def test_relationship_inference_corrupted_contact_data(self, user_model_store):
        """Handle non-dict contact values (e.g., None from corrupted data) without crashing.

        Before the fix, human_contacts.values() containing None would cause
        AttributeError on c.get('interaction_count', 0) in the list comprehension.
        The defensive fix skips non-dict entries.
        """
        profile_data = {
            "contacts": {
                # Normal human contact
                "alice@example.com": {
                    "interaction_count": 5,
                    "outbound_count": 2,
                },
                # Corrupted entries — None instead of dict
                "corrupted1@example.com": None,
                "corrupted2@example.com": None,
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        # Must not crash and must return the expected dict shape
        assert result is not None
        assert isinstance(result, dict)
        assert result["type"] == "relationship"
        assert result["processed"] is True

    def test_relationship_inference_high_priority_contact(self, user_model_store):
        """A contact with interaction_count >= 2x average gets a relationship_priority fact.

        Setup: alice (10), bob (1), carol (1).
          avg = 12/3 = 4; threshold = 8. alice (10) >= 8 -> HIGH PRIORITY.
        """
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 10,
                    "avg_response_time_seconds": 1800,
                    "outbound_count": 5,
                },
                "bob@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,
                },
                "carol@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert result["type"] == "relationship"
        assert result["processed"] is True

        # alice (10) >= threshold (8) -> should be labelled high_priority
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_fact = next(
            (f for f in facts if f["key"] == "relationship_priority_alice@example.com"),
            None,
        )
        assert priority_fact is not None
        assert priority_fact["value"] == "high_priority"

    def test_relationship_inference_result_in_run_all(self, user_model_store):
        """run_all_inference() results list must contain no None entries.

        A None result from any infer_from_* method causes AttributeError
        in _log_inference_summary() when it calls r.get('processed') on
        each result.
        """
        # Seed relationship profile with minimal valid data
        user_model_store.update_signal_profile("relationships", {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 5,
                    "outbound_count": 2,
                },
            }
        })
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)

        # run_all_inference must complete without AttributeError from
        # _log_inference_summary encountering a None result
        inferrer.run_all_inference()

        # Verify no crash — if we got here, _log_inference_summary handled
        # all results correctly (no None entries in the results list)
