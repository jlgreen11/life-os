"""
Validate that PROFILE_EVENT_TYPES stays in sync with each extractor's can_process().

If an extractor accepts an event type via can_process(), that type MUST be
listed in the corresponding PROFILE_EVENT_TYPES entry.  Drift between these
two causes the profile rebuild mechanism to miss qualifying events, leaving
signal profiles empty or incomplete.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models.core import EventType
from services.signal_extractor.pipeline import PROFILE_EVENT_TYPES, _PROFILE_TO_EXTRACTOR
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.decision import DecisionExtractor
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.spatial import SpatialExtractor
from services.signal_extractor.temporal import TemporalExtractor
from services.signal_extractor.topic import TopicExtractor


# Map extractor class names to their classes.
_EXTRACTOR_CLASSES: dict[str, type] = {
    "LinguisticExtractor": LinguisticExtractor,
    "CadenceExtractor": CadenceExtractor,
    "MoodInferenceEngine": MoodInferenceEngine,
    "RelationshipExtractor": RelationshipExtractor,
    "TopicExtractor": TopicExtractor,
    "TemporalExtractor": TemporalExtractor,
    "SpatialExtractor": SpatialExtractor,
    "DecisionExtractor": DecisionExtractor,
}


def _make_event(event_type: str) -> dict:
    """Create a synthetic event with a rich payload so conditional can_process()
    methods (like SpatialExtractor's location check) return True when the event
    type is genuinely supported."""
    return {
        "id": "test-event-1",
        "type": event_type,
        "source": "test",
        "timestamp": "2026-01-15T10:00:00Z",
        "priority": "normal",
        "payload": {
            # Communication fields
            "from": "alice@example.com",
            "to": "user@example.com",
            "subject": "Test",
            "body": "Hello world, this is a test message with enough content.",
            "sender": "alice",
            "recipient": "user",
            # Calendar fields
            "title": "Meeting",
            "location": "Conference Room A",
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T11:00:00Z",
            # Finance fields
            "amount": 42.00,
            "merchant": "Test Store",
            "category": "shopping",
            # Health fields
            "metric": "heart_rate",
            "value": 72,
            # Location fields
            "timezone": "America/New_York",
            "sender_timezone": "America/New_York",
            # Task fields
            "task_title": "Do the thing",
            "description": "A test task",
            # iOS context fields
            "device_proximity": {"iphone": True},
        },
        "metadata": {},
    }


def _get_accepted_types(extractor) -> set[str]:
    """Return the set of EventType values that an extractor's can_process() accepts."""
    accepted = set()
    for et in EventType:
        event = _make_event(et.value)
        if extractor.can_process(event):
            accepted.add(et.value)
    return accepted


class TestProfileEventTypesSync:
    """Ensure PROFILE_EVENT_TYPES matches what each extractor actually accepts."""

    def _make_extractor(self, cls):
        """Instantiate an extractor with mock dependencies."""
        db = MagicMock()
        ums = MagicMock()
        return cls(db=db, user_model_store=ums)

    @pytest.mark.parametrize("profile_name", list(PROFILE_EVENT_TYPES.keys()))
    def test_profile_event_types_covers_all_accepted_types(self, profile_name):
        """Every event type accepted by can_process() must appear in PROFILE_EVENT_TYPES."""
        extractor_names = _PROFILE_TO_EXTRACTOR.get(profile_name, [])
        assert extractor_names, f"No extractor mapped for profile '{profile_name}'"

        registered_types = set(PROFILE_EVENT_TYPES[profile_name])

        for ext_name in extractor_names:
            cls = _EXTRACTOR_CLASSES[ext_name]
            extractor = self._make_extractor(cls)
            accepted = _get_accepted_types(extractor)

            # For profiles that are a subset of an extractor's scope (e.g.
            # linguistic vs linguistic_inbound), we only check that the
            # registered types for THIS profile are a subset of what the
            # extractor accepts OR that accepted types are covered.
            # The key invariant: accepted types must be present in at least
            # one of the extractor's PROFILE_EVENT_TYPES entries.
            all_profile_types_for_extractor = set()
            for pname, enames in _PROFILE_TO_EXTRACTOR.items():
                if ext_name in enames:
                    all_profile_types_for_extractor.update(PROFILE_EVENT_TYPES[pname])

            missing = accepted - all_profile_types_for_extractor
            assert not missing, (
                f"Extractor {ext_name} accepts event types {sorted(missing)} "
                f"via can_process() but they are not listed in any "
                f"PROFILE_EVENT_TYPES entry for this extractor. "
                f"Add them to PROFILE_EVENT_TYPES['{profile_name}'] in pipeline.py."
            )

    def test_all_profiles_have_extractor_mapping(self):
        """Every profile in PROFILE_EVENT_TYPES must have a _PROFILE_TO_EXTRACTOR entry."""
        for profile_name in PROFILE_EVENT_TYPES:
            assert profile_name in _PROFILE_TO_EXTRACTOR, (
                f"Profile '{profile_name}' is in PROFILE_EVENT_TYPES but not in "
                f"_PROFILE_TO_EXTRACTOR — add the mapping."
            )

    def test_all_extractor_mappings_have_profile(self):
        """Every profile in _PROFILE_TO_EXTRACTOR must have a PROFILE_EVENT_TYPES entry."""
        for profile_name in _PROFILE_TO_EXTRACTOR:
            assert profile_name in PROFILE_EVENT_TYPES, (
                f"Profile '{profile_name}' is in _PROFILE_TO_EXTRACTOR but not in "
                f"PROFILE_EVENT_TYPES — add the event type list."
            )
