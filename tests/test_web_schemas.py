"""
Comprehensive test suite for web/schemas.py

Tests all Pydantic request/response schemas for API validation, including
optional fields, type coercion, validation errors, and serialization behavior.

Coverage: 15 schema classes, 151 LOC
"""

import pytest
from pydantic import ValidationError

from web.schemas import (
    CommandRequest,
    ConnectorConfigRequest,
    ContextBatchRequest,
    ContextEventRequest,
    ContextMetadata,
    ContextPayload,
    DraftRequest,
    FeedbackRequest,
    PreferenceUpdate,
    RuleCreateRequest,
    SearchRequest,
    SetupSubmitRequest,
    TaskCreateRequest,
    TaskUpdateRequest,
)


# ---------------------------------------------------------------------------
# CommandRequest
# ---------------------------------------------------------------------------


def test_command_request_valid():
    """Test CommandRequest with valid data."""
    req = CommandRequest(text="search meetings", context={"view": "inbox"})
    assert req.text == "search meetings"
    assert req.context == {"view": "inbox"}


def test_command_request_no_context():
    """Test CommandRequest with optional context omitted."""
    req = CommandRequest(text="briefing")
    assert req.text == "briefing"
    assert req.context is None


def test_command_request_missing_text():
    """Test CommandRequest rejects missing required text field."""
    with pytest.raises(ValidationError):
        CommandRequest()


# ---------------------------------------------------------------------------
# TaskCreateRequest
# ---------------------------------------------------------------------------


def test_task_create_request_minimal():
    """Test TaskCreateRequest with only required fields."""
    req = TaskCreateRequest(title="Buy groceries")
    assert req.title == "Buy groceries"
    assert req.description is None
    assert req.domain is None
    assert req.priority == "normal"
    assert req.due_date is None


def test_task_create_request_full():
    """Test TaskCreateRequest with all fields populated."""
    req = TaskCreateRequest(
        title="Complete project",
        description="Finish the quarterly report",
        domain="work",
        priority="high",
        due_date="2026-02-20T17:00:00Z"
    )
    assert req.title == "Complete project"
    assert req.description == "Finish the quarterly report"
    assert req.domain == "work"
    assert req.priority == "high"
    assert req.due_date == "2026-02-20T17:00:00Z"


def test_task_create_request_missing_title():
    """Test TaskCreateRequest rejects missing title."""
    with pytest.raises(ValidationError):
        TaskCreateRequest(priority="high")


# ---------------------------------------------------------------------------
# TaskUpdateRequest
# ---------------------------------------------------------------------------


def test_task_update_request_partial():
    """Test TaskUpdateRequest with partial updates."""
    req = TaskUpdateRequest(status="in_progress", priority="high")
    assert req.status == "in_progress"
    assert req.priority == "high"
    assert req.due_date is None
    assert req.title is None


def test_task_update_request_empty():
    """Test TaskUpdateRequest allows empty updates."""
    req = TaskUpdateRequest()
    assert req.status is None
    assert req.priority is None
    assert req.due_date is None
    assert req.title is None


def test_task_update_request_title_only():
    """Test TaskUpdateRequest can update just the title."""
    req = TaskUpdateRequest(title="Updated title")
    assert req.title == "Updated title"


# ---------------------------------------------------------------------------
# RuleCreateRequest
# ---------------------------------------------------------------------------


def test_rule_create_request_minimal():
    """Test RuleCreateRequest with minimal fields."""
    req = RuleCreateRequest(name="Test rule", trigger_event="email.received")
    assert req.name == "Test rule"
    assert req.trigger_event == "email.received"
    assert req.conditions == []
    assert req.actions == []


def test_rule_create_request_full():
    """Test RuleCreateRequest with conditions and actions."""
    req = RuleCreateRequest(
        name="Auto-tag work emails",
        trigger_event="email.received",
        conditions=[{"field": "sender", "op": "contains", "value": "@work.com"}],
        actions=[{"type": "tag", "value": "work"}]
    )
    assert req.name == "Auto-tag work emails"
    assert len(req.conditions) == 1
    assert len(req.actions) == 1


def test_rule_create_request_missing_name():
    """Test RuleCreateRequest rejects missing name."""
    with pytest.raises(ValidationError):
        RuleCreateRequest(trigger_event="email.received")


# ---------------------------------------------------------------------------
# SearchRequest
# ---------------------------------------------------------------------------


def test_search_request_minimal():
    """Test SearchRequest with just query."""
    req = SearchRequest(query="meetings")
    assert req.query == "meetings"
    assert req.limit == 10
    assert req.filters is None


def test_search_request_with_limit():
    """Test SearchRequest with custom limit."""
    req = SearchRequest(query="test", limit=50)
    assert req.limit == 50


def test_search_request_with_filters():
    """Test SearchRequest with metadata filters."""
    req = SearchRequest(query="emails", filters={"source": "email", "priority": "high"})
    assert req.filters == {"source": "email", "priority": "high"}


def test_search_request_missing_query():
    """Test SearchRequest rejects missing query."""
    with pytest.raises(ValidationError):
        SearchRequest(limit=10)


# ---------------------------------------------------------------------------
# DraftRequest
# ---------------------------------------------------------------------------


def test_draft_request_minimal():
    """Test DraftRequest with defaults."""
    req = DraftRequest()
    assert req.contact_id is None
    assert req.channel == "email"
    assert req.incoming_message == ""
    assert req.context is None


def test_draft_request_full():
    """Test DraftRequest with all fields."""
    req = DraftRequest(
        contact_id="c123",
        channel="signal",
        incoming_message="Can we meet tomorrow?",
        context="Replying to project discussion"
    )
    assert req.contact_id == "c123"
    assert req.channel == "signal"
    assert req.incoming_message == "Can we meet tomorrow?"
    assert req.context == "Replying to project discussion"


# ---------------------------------------------------------------------------
# FeedbackRequest
# ---------------------------------------------------------------------------


def test_feedback_request_valid():
    """Test FeedbackRequest with message."""
    req = FeedbackRequest(message="Great feature!")
    assert req.message == "Great feature!"


def test_feedback_request_missing_message():
    """Test FeedbackRequest rejects missing message."""
    with pytest.raises(ValidationError):
        FeedbackRequest()


# ---------------------------------------------------------------------------
# PreferenceUpdate
# ---------------------------------------------------------------------------


def test_preference_update_string_value():
    """Test PreferenceUpdate with string value."""
    req = PreferenceUpdate(key="theme", value="dark")
    assert req.key == "theme"
    assert req.value == "dark"


def test_preference_update_number_value():
    """Test PreferenceUpdate with numeric value."""
    req = PreferenceUpdate(key="timeout", value=30)
    assert req.key == "timeout"
    assert req.value == 30


def test_preference_update_boolean_value():
    """Test PreferenceUpdate with boolean value."""
    req = PreferenceUpdate(key="notifications_enabled", value=True)
    assert req.value is True


def test_preference_update_dict_value():
    """Test PreferenceUpdate with nested dict value."""
    req = PreferenceUpdate(key="quiet_hours", value={"start": "22:00", "end": "07:00"})
    assert req.value == {"start": "22:00", "end": "07:00"}


def test_preference_update_missing_key():
    """Test PreferenceUpdate rejects missing key."""
    with pytest.raises(ValidationError):
        PreferenceUpdate(value="test")


# ---------------------------------------------------------------------------
# ContextPayload
# ---------------------------------------------------------------------------


def test_context_payload_location():
    """Test ContextPayload with location data."""
    payload = ContextPayload(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=50.0,
        horizontal_accuracy=10.0,
        place_name="San Francisco",
        place_type="city"
    )
    assert payload.latitude == 37.7749
    assert payload.longitude == -122.4194
    assert payload.place_name == "San Francisco"


def test_context_payload_device():
    """Test ContextPayload with device discovery data."""
    payload = ContextPayload(
        device_name="iPhone 15",
        device_type="smartphone",
        signal_strength=-45,
        is_connected=True
    )
    assert payload.device_name == "iPhone 15"
    assert payload.signal_strength == -45
    assert payload.is_connected is True


def test_context_payload_time():
    """Test ContextPayload with time context data."""
    payload = ContextPayload(
        local_time="14:30:00",
        timezone="America/Los_Angeles",
        day_of_week="Saturday",
        is_weekend=True
    )
    assert payload.local_time == "14:30:00"
    assert payload.is_weekend is True


def test_context_payload_activity():
    """Test ContextPayload with activity data."""
    payload = ContextPayload(activity="walking", confidence=0.92)
    assert payload.activity == "walking"
    assert payload.confidence == 0.92


def test_context_payload_all_optional():
    """Test ContextPayload with all fields optional."""
    payload = ContextPayload()
    assert payload.latitude is None
    assert payload.device_name is None
    assert payload.local_time is None


# ---------------------------------------------------------------------------
# ContextMetadata
# ---------------------------------------------------------------------------


def test_context_metadata_full():
    """Test ContextMetadata with all fields."""
    metadata = ContextMetadata(
        device_model="iPhone 15 Pro",
        os_version="17.3",
        battery_level=0.75,
        network_type="5G",
        app_state="foreground"
    )
    assert metadata.device_model == "iPhone 15 Pro"
    assert metadata.battery_level == 0.75


def test_context_metadata_empty():
    """Test ContextMetadata with all fields optional."""
    metadata = ContextMetadata()
    assert metadata.device_model is None
    assert metadata.battery_level is None


# ---------------------------------------------------------------------------
# ContextEventRequest
# ---------------------------------------------------------------------------


def test_context_event_request_minimal():
    """Test ContextEventRequest with minimal required fields."""
    req = ContextEventRequest(
        type="context.location",
        payload=ContextPayload(latitude=37.7749, longitude=-122.4194)
    )
    assert req.type == "context.location"
    assert req.source == "ios_app"
    assert req.timestamp is None
    assert req.metadata is None


def test_context_event_request_full():
    """Test ContextEventRequest with all fields."""
    req = ContextEventRequest(
        type="context.device_nearby",
        source="ios_app_v2",
        timestamp="2026-02-15T12:00:00Z",
        payload=ContextPayload(device_name="iPhone", signal_strength=-50),
        metadata=ContextMetadata(device_model="iPhone 15", battery_level=0.8)
    )
    assert req.type == "context.device_nearby"
    assert req.source == "ios_app_v2"
    assert req.timestamp == "2026-02-15T12:00:00Z"
    assert req.metadata.battery_level == 0.8


def test_context_event_request_missing_type():
    """Test ContextEventRequest rejects missing type."""
    with pytest.raises(ValidationError):
        ContextEventRequest(payload=ContextPayload())


def test_context_event_request_missing_payload():
    """Test ContextEventRequest rejects missing payload."""
    with pytest.raises(ValidationError):
        ContextEventRequest(type="context.location")


# ---------------------------------------------------------------------------
# ContextBatchRequest
# ---------------------------------------------------------------------------


def test_context_batch_request_single_event():
    """Test ContextBatchRequest with one event."""
    event = ContextEventRequest(
        type="context.location",
        payload=ContextPayload(latitude=37.7749, longitude=-122.4194)
    )
    req = ContextBatchRequest(events=[event])
    assert len(req.events) == 1


def test_context_batch_request_multiple_events():
    """Test ContextBatchRequest with multiple events."""
    events = [
        ContextEventRequest(
            type="context.location",
            payload=ContextPayload(latitude=37.7749, longitude=-122.4194)
        ),
        ContextEventRequest(
            type="context.device_nearby",
            payload=ContextPayload(device_name="iPhone")
        )
    ]
    req = ContextBatchRequest(events=events)
    assert len(req.events) == 2


def test_context_batch_request_empty_events():
    """Test ContextBatchRequest with empty events list."""
    req = ContextBatchRequest(events=[])
    assert req.events == []


def test_context_batch_request_missing_events():
    """Test ContextBatchRequest rejects missing events field."""
    with pytest.raises(ValidationError):
        ContextBatchRequest()


# ---------------------------------------------------------------------------
# ConnectorConfigRequest
# ---------------------------------------------------------------------------


def test_connector_config_request_valid():
    """Test ConnectorConfigRequest with config dict."""
    req = ConnectorConfigRequest(config={"api_key": "secret", "domain": "example.com"})
    assert req.config == {"api_key": "secret", "domain": "example.com"}


def test_connector_config_request_empty_config():
    """Test ConnectorConfigRequest with empty config."""
    req = ConnectorConfigRequest(config={})
    assert req.config == {}


def test_connector_config_request_missing_config():
    """Test ConnectorConfigRequest rejects missing config."""
    with pytest.raises(ValidationError):
        ConnectorConfigRequest()


# ---------------------------------------------------------------------------
# SetupSubmitRequest
# ---------------------------------------------------------------------------


def test_setup_submit_request_string_value():
    """Test SetupSubmitRequest with string value."""
    req = SetupSubmitRequest(step_id="name", value="John Doe")
    assert req.step_id == "name"
    assert req.value == "John Doe"


def test_setup_submit_request_list_value():
    """Test SetupSubmitRequest with list value."""
    req = SetupSubmitRequest(step_id="contacts", value=["john@example.com", "jane@example.com"])
    assert req.value == ["john@example.com", "jane@example.com"]


def test_setup_submit_request_dict_value():
    """Test SetupSubmitRequest with dict value."""
    req = SetupSubmitRequest(step_id="preferences", value={"theme": "dark", "locale": "en-US"})
    assert req.value == {"theme": "dark", "locale": "en-US"}


def test_setup_submit_request_missing_step_id():
    """Test SetupSubmitRequest rejects missing step_id."""
    with pytest.raises(ValidationError):
        SetupSubmitRequest(value="test")


# ---------------------------------------------------------------------------
# Serialization and validation edge cases
# ---------------------------------------------------------------------------


def test_command_request_extra_fields_ignored():
    """Test that extra fields in CommandRequest are ignored."""
    req = CommandRequest(text="test", extra_field="ignored")
    assert req.text == "test"
    assert not hasattr(req, "extra_field")


def test_task_create_request_dict_export():
    """Test that TaskCreateRequest can be exported to dict."""
    req = TaskCreateRequest(title="Test", priority="high")
    data = req.dict()
    assert data["title"] == "Test"
    assert data["priority"] == "high"


def test_task_create_request_dict_exclude_none():
    """Test TaskCreateRequest dict export excludes None values."""
    req = TaskCreateRequest(title="Test")
    data = req.dict(exclude_none=True)
    assert "title" in data
    assert "description" not in data  # Was None


def test_search_request_json_serialization():
    """Test SearchRequest can be serialized to JSON."""
    req = SearchRequest(query="test", limit=5, filters={"source": "email"})
    json_str = req.json()
    assert "test" in json_str
    assert "email" in json_str


def test_context_payload_numeric_coercion():
    """Test ContextPayload coerces string numbers to floats."""
    payload = ContextPayload(latitude="37.7749", longitude="-122.4194")
    assert isinstance(payload.latitude, float)
    assert payload.latitude == 37.7749


def test_preference_update_none_value():
    """Test PreferenceUpdate allows None as value."""
    req = PreferenceUpdate(key="optional_setting", value=None)
    assert req.key == "optional_setting"
    assert req.value is None


def test_rule_create_request_complex_conditions():
    """Test RuleCreateRequest with complex nested conditions."""
    req = RuleCreateRequest(
        name="Complex rule",
        trigger_event="email.received",
        conditions=[
            {"field": "sender", "op": "contains", "value": "@work.com"},
            {"field": "priority", "op": "gte", "value": 0.7},
            {"field": "tags", "op": "in", "value": ["urgent", "critical"]}
        ],
        actions=[
            {"type": "notify", "channel": "push"},
            {"type": "tag", "value": "important"}
        ]
    )
    assert len(req.conditions) == 3
    assert len(req.actions) == 2


def test_context_metadata_partial():
    """Test ContextMetadata with only some fields populated."""
    metadata = ContextMetadata(device_model="iPhone 15", battery_level=0.5)
    assert metadata.device_model == "iPhone 15"
    assert metadata.os_version is None
    assert metadata.network_type is None


def test_context_event_request_default_source():
    """Test ContextEventRequest defaults source to 'ios_app'."""
    req = ContextEventRequest(
        type="context.location",
        payload=ContextPayload(latitude=1.0, longitude=1.0)
    )
    assert req.source == "ios_app"


def test_draft_request_default_channel():
    """Test DraftRequest defaults channel to 'email'."""
    req = DraftRequest(incoming_message="Test")
    assert req.channel == "email"


def test_task_create_request_default_priority():
    """Test TaskCreateRequest defaults priority to 'normal'."""
    req = TaskCreateRequest(title="Test")
    assert req.priority == "normal"


def test_search_request_default_limit():
    """Test SearchRequest defaults limit to 10."""
    req = SearchRequest(query="test")
    assert req.limit == 10


# ---------------------------------------------------------------------------
# Type validation edge cases
# ---------------------------------------------------------------------------


def test_task_update_request_invalid_type():
    """Test TaskUpdateRequest rejects invalid types for typed fields."""
    # Pydantic V2 validates types strictly and raises errors on type mismatches
    with pytest.raises(ValidationError):
        TaskUpdateRequest(priority=123)  # Integer not allowed for string field


def test_context_payload_invalid_latitude():
    """Test ContextPayload rejects non-numeric latitude."""
    with pytest.raises(ValidationError):
        ContextPayload(latitude="invalid")


def test_search_request_negative_limit():
    """Test SearchRequest allows negative limit (validation not enforced)."""
    # Pydantic doesn't enforce positive integers by default
    req = SearchRequest(query="test", limit=-1)
    assert req.limit == -1  # Passes validation, but would be caught at runtime


def test_connector_config_request_nested_config():
    """Test ConnectorConfigRequest with deeply nested config."""
    req = ConnectorConfigRequest(config={
        "api": {
            "key": "secret",
            "endpoints": {
                "base": "https://api.example.com",
                "version": "v2"
            }
        },
        "retries": 3
    })
    assert req.config["api"]["endpoints"]["base"] == "https://api.example.com"


def test_feedback_request_long_message():
    """Test FeedbackRequest handles very long messages."""
    long_message = "x" * 10000
    req = FeedbackRequest(message=long_message)
    assert len(req.message) == 10000
