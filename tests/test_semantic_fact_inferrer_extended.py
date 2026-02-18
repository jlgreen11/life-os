"""
Tests for Semantic Fact Inference — Extended Profiles

Verifies that the SemanticFactInferrer correctly derives semantic facts
from the three newer signal profiles: temporal, spatial, and decision.
"""

import json
import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to set sample count in a profile's data."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestTemporalInference:
    """Test suite for temporal profile semantic fact inference."""

    def test_morning_person_chronotype(self, user_model_store):
        """Verify morning person chronotype inference from morning-heavy activity."""
        # Setup: temporal profile with 60% morning activity (6-10am)
        profile_data = {
            "activity_by_hour": {
                "6": 20, "7": 30, "8": 40, "9": 35, "10": 25,  # 150 morning
                "14": 20, "15": 15, "16": 10, "20": 10, "21": 5,  # 60 other
            },
            "activity_by_day": {
                "monday": 30, "tuesday": 35, "wednesday": 40,
                "thursday": 30, "friday": 25, "saturday": 5, "sunday": 5,
            },
        }
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 75)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_temporal_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        chronotype_fact = next((f for f in facts if f["key"] == "chronotype"), None)

        assert chronotype_fact is not None, "Should infer chronotype from temporal profile"
        assert chronotype_fact["value"] == "morning_person"
        assert chronotype_fact["confidence"] > 0.6

    def test_night_owl_chronotype(self, user_model_store):
        """Verify night owl chronotype inference from evening-heavy activity."""
        # Setup: temporal profile with 50% evening activity (8pm-11pm)
        profile_data = {
            "activity_by_hour": {
                "9": 10, "10": 8, "14": 12,  # 30 daytime
                "20": 25, "21": 30, "22": 20, "23": 15,  # 90 evening
            },
            "activity_by_day": {
                "monday": 15, "tuesday": 18, "wednesday": 20,
                "thursday": 18, "friday": 20, "saturday": 12, "sunday": 10,
            },
        }
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 60)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_temporal_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        chronotype_fact = next((f for f in facts if f["key"] == "chronotype"), None)

        assert chronotype_fact is not None
        assert chronotype_fact["value"] == "night_owl"
        assert chronotype_fact["confidence"] > 0.6

    def test_peak_productivity_hour(self, user_model_store):
        """Verify peak productivity hour inference from activity spikes."""
        # Setup: clear peak at 2pm (hour 14)
        profile_data = {
            "activity_by_hour": {
                "9": 10, "10": 12, "11": 15,
                "14": 60,  # Clear peak
                "15": 20, "16": 18,
            },
            "activity_by_day": {"monday": 20, "tuesday": 25},
        }
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_temporal_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_hour_fact = next((f for f in facts if f["key"] == "peak_productivity_hour"), None)

        assert peak_hour_fact is not None
        assert peak_hour_fact["value"] == 14
        assert peak_hour_fact["confidence"] > 0.5

    def test_weekday_only_work_pattern(self, user_model_store):
        """Verify weekday-only work boundary inference."""
        # Setup: 95% weekday activity, 5% weekend
        profile_data = {
            "activity_by_hour": {"9": 10, "10": 15, "14": 20},
            "activity_by_day": {
                "monday": 20, "tuesday": 22, "wednesday": 25,
                "thursday": 20, "friday": 18,
                "saturday": 2, "sunday": 1,  # Minimal weekend activity
            },
        }
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 55)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_temporal_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "temporal_work_boundaries"), None)

        assert boundary_fact is not None
        assert boundary_fact["value"] == "weekday_only_work"
        assert boundary_fact["confidence"] > 0.6

    def test_insufficient_temporal_samples(self, user_model_store):
        """Verify no inference with <50 samples."""
        profile_data = {"activity_by_hour": {"9": 5, "10": 3}, "activity_by_day": {"monday": 8}}
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 30)  # Below 50-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_temporal_profile()

        facts = user_model_store.get_semantic_facts()
        temporal_facts = [f for f in facts if "chronotype" in f["key"] or "peak_productivity" in f["key"]]
        assert len(temporal_facts) == 0, "Should not infer from insufficient temporal samples"


class TestSpatialInference:
    """Test suite for spatial profile semantic fact inference."""

    def test_primary_work_location(self, user_model_store):
        """Verify primary work location inference from high-visit work place."""
        # Setup: spatial profile with clear primary work location
        place_behaviors = {
            "office_building_downtown": {
                "place_id": "office_building_downtown",
                "visit_count": 45,
                "dominant_domain": "work",
                "average_duration_minutes": 480,
            },
            "home": {
                "place_id": "home",
                "visit_count": 120,
                "dominant_domain": "personal",
                "average_duration_minutes": 960,
            },
        }
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 165)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        work_location_fact = next((f for f in facts if f["key"] == "primary_work_location"), None)

        assert work_location_fact is not None
        assert work_location_fact["value"] == "office_building_downtown"
        assert work_location_fact["confidence"] > 0.5

    def test_home_office_detection(self, user_model_store):
        """Verify home office work location type inference."""
        # Setup: primary work location is a residence
        place_behaviors = {
            "residence_inn_by_marriott": {
                "place_id": "residence_inn_by_marriott",
                "visit_count": 50,
                "dominant_domain": "work",
                "average_duration_minutes": 480,
            },
        }
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        work_type_fact = next((f for f in facts if f["key"] == "work_location_type"), None)

        assert work_type_fact is not None
        assert work_type_fact["value"] == "home_office"
        assert work_type_fact["confidence"] > 0.6

    def test_external_office_detection(self, user_model_store):
        """Verify external office work location type inference."""
        # Setup: primary work location is an office building
        place_behaviors = {
            "corporate_headquarters_building_a": {
                "place_id": "corporate_headquarters_building_a",
                "visit_count": 60,
                "dominant_domain": "work",
                "average_duration_minutes": 500,
            },
        }
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 60)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        work_type_fact = next((f for f in facts if f["key"] == "work_location_type"), None)

        assert work_type_fact is not None
        assert work_type_fact["value"] == "external_office"
        assert work_type_fact["confidence"] > 0.6

    def test_frequent_location_tracking(self, user_model_store):
        """Verify frequent location tracking (10+ visits)."""
        # Setup: multiple places with varying visit counts
        place_behaviors = {
            "favorite_coffee_shop": {
                "place_id": "favorite_coffee_shop",
                "visit_count": 25,
                "dominant_domain": "personal",
                "average_duration_minutes": 45,
            },
            "gym": {
                "place_id": "gym",
                "visit_count": 15,
                "dominant_domain": "personal",
                "average_duration_minutes": 90,
            },
            "one_time_restaurant": {
                "place_id": "one_time_restaurant",
                "visit_count": 2,
                "dominant_domain": "personal",
                "average_duration_minutes": 60,
            },
        }
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 42)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        frequent_location_facts = [f for f in facts if "frequent_location_" in f["key"]]

        # Should track coffee shop and gym (10+ visits), not restaurant (2 visits)
        assert len(frequent_location_facts) == 2
        frequent_places = {f["value"] for f in frequent_location_facts}
        assert "favorite_coffee_shop" in frequent_places
        assert "gym" in frequent_places
        assert "one_time_restaurant" not in frequent_places

    def test_location_domain_tracking(self, user_model_store):
        """Verify dominant domain tracking per location."""
        place_behaviors = {
            "office": {
                "place_id": "office",
                "visit_count": 30,
                "dominant_domain": "work",
            },
            "home": {
                "place_id": "home",
                "visit_count": 50,
                "dominant_domain": "personal",
            },
        }
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 80)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")

        office_domain = next((f for f in facts if f["key"] == "location_domain_office"), None)
        home_domain = next((f for f in facts if f["key"] == "location_domain_home"), None)

        assert office_domain is not None
        assert office_domain["value"] == "work"
        assert home_domain is not None
        assert home_domain["value"] == "personal"

    def test_insufficient_spatial_samples(self, user_model_store):
        """Verify no inference with <10 samples."""
        place_behaviors = {"place_a": {"place_id": "place_a", "visit_count": 3}}
        profile_data = {"place_behaviors": json.dumps(place_behaviors)}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 5)  # Below 10-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_spatial_profile()

        facts = user_model_store.get_semantic_facts()
        spatial_facts = [f for f in facts if "location" in f["key"].lower()]
        assert len(spatial_facts) == 0, "Should not infer from insufficient spatial samples"


class TestDecisionInference:
    """Test suite for decision profile semantic fact inference."""

    def test_quick_decision_speed(self, user_model_store):
        """Verify quick decision speed inference (<60 seconds)."""
        # Setup: decision profile with fast decisions in food domain
        profile_data = {
            "decision_speed_by_domain": {
                "food": 15,  # 15 seconds — very fast
                "purchases_under_50": 30,  # 30 seconds — quick
            },
            "research_depth_by_domain": {
                "food": 0.1,
                "purchases_under_50": 0.2,
            },
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        food_speed_fact = next((f for f in facts if f["key"] == "decision_speed_food"), None)

        assert food_speed_fact is not None
        assert food_speed_fact["value"] == "quick_decision"
        assert food_speed_fact["confidence"] > 0.6

    def test_deliberate_decision_speed(self, user_model_store):
        """Verify deliberate decision speed inference (>1 day)."""
        # Setup: decision profile with slow decisions in major purchases
        profile_data = {
            "decision_speed_by_domain": {
                "purchases_over_500": 172800,  # 2 days — deliberate
                "career_moves": 604800,  # 7 days — very deliberate
            },
            "research_depth_by_domain": {
                "purchases_over_500": 0.8,
                "career_moves": 0.9,
            },
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 30)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        purchase_speed_fact = next((f for f in facts if f["key"] == "decision_speed_purchases_over_500"), None)

        assert purchase_speed_fact is not None
        assert purchase_speed_fact["value"] == "deliberate_decision"
        assert purchase_speed_fact["confidence"] > 0.5

    def test_data_driven_research_preference(self, user_model_store):
        """Verify data-driven research preference inference (depth > 0.7)."""
        profile_data = {
            "decision_speed_by_domain": {"investments": 259200},  # 3 days
            "research_depth_by_domain": {"investments": 0.9},  # Exhaustive research
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 20)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        research_fact = next((f for f in facts if f["key"] == "research_preference_investments"), None)

        assert research_fact is not None
        assert research_fact["value"] == "data_driven"
        assert research_fact["confidence"] > 0.5

    def test_gut_feel_research_preference(self, user_model_store):
        """Verify gut-feel research preference inference (depth < 0.3)."""
        profile_data = {
            "decision_speed_by_domain": {"clothing": 120},  # 2 minutes
            "research_depth_by_domain": {"clothing": 0.15},  # Minimal research
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 22)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        research_fact = next((f for f in facts if f["key"] == "research_preference_clothing"), None)

        assert research_fact is not None
        assert research_fact["value"] == "gut_feel"
        assert research_fact["confidence"] > 0.5

    def test_high_risk_tolerance(self, user_model_store):
        """Verify high risk tolerance inference (fast + low research across domains)."""
        profile_data = {
            "decision_speed_by_domain": {
                "food": 10,
                "purchases_under_100": 45,
                "social_plans": 30,
            },
            "research_depth_by_domain": {
                "food": 0.1,
                "purchases_under_100": 0.2,
                "social_plans": 0.15,
            },
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        risk_fact = next((f for f in facts if f["key"] == "risk_tolerance"), None)

        assert risk_fact is not None
        assert risk_fact["value"] == "high_risk_tolerance"
        assert risk_fact["confidence"] > 0.5

    def test_risk_averse(self, user_model_store):
        """Verify risk-averse inference (slow + high research across domains)."""
        profile_data = {
            "decision_speed_by_domain": {
                "purchases_over_500": 86400,  # 1 day
                "travel": 172800,  # 2 days
                "contracts": 604800,  # 7 days
            },
            "research_depth_by_domain": {
                "purchases_over_500": 0.85,
                "travel": 0.9,
                "contracts": 0.95,
            },
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 30)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        risk_fact = next((f for f in facts if f["key"] == "risk_tolerance"), None)

        assert risk_fact is not None
        assert risk_fact["value"] == "risk_averse"
        assert risk_fact["confidence"] > 0.5

    def test_insufficient_decision_samples(self, user_model_store):
        """Verify no inference with <20 samples."""
        profile_data = {
            "decision_speed_by_domain": {"food": 20},
            "research_depth_by_domain": {"food": 0.2},
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 10)  # Below 20-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_decision_profile()

        facts = user_model_store.get_semantic_facts()
        decision_facts = [f for f in facts if "decision" in f["key"] or "research" in f["key"]]
        assert len(decision_facts) == 0, "Should not infer from insufficient decision samples"


class TestRunAllInferenceExtended:
    """Test that run_all_inference includes the new profile types."""

    def test_run_all_includes_extended_profiles(self, user_model_store):
        """Verify run_all_inference processes temporal, spatial, and decision profiles."""
        # Setup all three new profiles with valid data
        temporal_data = {
            "activity_by_hour": {"9": 20, "10": 25, "14": 30, "20": 15},
            "activity_by_day": {"monday": 30, "tuesday": 35},
        }
        user_model_store.update_signal_profile("temporal", temporal_data)
        _set_samples(user_model_store, "temporal", 60)

        spatial_data = {
            "place_behaviors": json.dumps({
                "office": {
                    "place_id": "office",
                    "visit_count": 40,
                    "dominant_domain": "work",
                },
            })
        }
        user_model_store.update_signal_profile("spatial", spatial_data)
        _set_samples(user_model_store, "spatial", 40)

        decision_data = {
            "decision_speed_by_domain": {"food": 20},
            "research_depth_by_domain": {"food": 0.2},
        }
        user_model_store.update_signal_profile("decision", decision_data)
        _set_samples(user_model_store, "decision", 25)

        # Run full inference
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # Verify facts were generated from all three new profiles
        facts = user_model_store.get_semantic_facts()

        temporal_facts = [f for f in facts if "peak_productivity" in f["key"] or "chronotype" in f["key"]]
        spatial_facts = [f for f in facts if "location" in f["key"]]
        decision_facts = [f for f in facts if "decision_speed" in f["key"] or "research_preference" in f["key"]]

        assert len(temporal_facts) > 0, "run_all_inference should process temporal profile"
        assert len(spatial_facts) > 0, "run_all_inference should process spatial profile"
        assert len(decision_facts) > 0, "run_all_inference should process decision profile"
