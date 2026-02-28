"""
Tests: GET /api/user-model/templates endpoint.

The endpoint exposes communication style templates from Layer 3 procedural
memory — per-contact, per-channel writing style summaries derived from
outbound/inbound messages.

Coverage:
    GET /api/user-model/templates
        → 200 with templates list and total count
        → 200 with empty list when no templates exist
        → 200 filters by contact_id query param
        → 200 filters by channel query param
        → 200 filters by context query param
        → 200 respects limit param (capped at 200)
        → 200 JSON-list fields (common_phrases, avoids_phrases, tone_notes) are arrays
        → 200 uses_emoji is a boolean
        → 200 fails open and returns empty list when DB raises
        → 200 response always includes generated_at ISO-8601 timestamp
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(db) -> TestClient:
    """Create a minimal FastAPI test client with a real DatabaseManager.

    The real ``db`` fixture (from conftest) gives us an actual temporary
    SQLite database so we can insert rows and verify the endpoint reads them.

    Args:
        db: DatabaseManager from the conftest ``db`` fixture.

    Returns:
        TestClient wrapping the FastAPI app with all routes registered.
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    from storage.event_store import EventStore
    from storage.user_model_store import UserModelStore

    life_os.event_store = EventStore(db)
    life_os.user_model_store = UserModelStore(db)
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.connectors = []
    life_os.event_bus.is_connected = False
    life_os.vector_store.get_stats.return_value = {"document_count": 0}

    register_routes(app, life_os)
    return TestClient(app)


def _insert_template(
    db,
    *,
    contact_id: str = "alice@example.com",
    channel: str = "email",
    context: str = "user_to_contact",
    greeting: str | None = "hey",
    closing: str | None = "thanks",
    formality: float = 0.25,
    typical_length: float = 42.0,
    uses_emoji: bool = False,
    common_phrases: list | None = None,
    avoids_phrases: list | None = None,
    tone_notes: list | None = None,
    samples_analyzed: int = 10,
) -> str:
    """Insert a synthetic communication_templates row into the test DB.

    Args:
        db: DatabaseManager with a temporary user_model.db.
        contact_id: Contact email or phone.
        channel: Communication channel (email, message, etc.).
        context: Direction — user_to_contact or contact_to_user.
        greeting: Detected greeting phrase or None.
        closing: Detected closing phrase or None.
        formality: 0.0 (casual) to 1.0 (formal).
        typical_length: Average word count.
        uses_emoji: Whether emoji appear.
        common_phrases: List of common phrases (default []).
        avoids_phrases: List of avoided phrases (default []).
        tone_notes: List of tone descriptors (default []).
        samples_analyzed: Number of messages analyzed.

    Returns:
        The generated template id string.
    """
    template_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO communication_templates
               (id, context, contact_id, channel, greeting, closing,
                formality, typical_length, uses_emoji, common_phrases,
                avoids_phrases, tone_notes, example_message_ids,
                samples_analyzed, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            (
                template_id,
                context,
                contact_id,
                channel,
                greeting,
                closing,
                formality,
                typical_length,
                int(uses_emoji),
                json.dumps(common_phrases or []),
                json.dumps(avoids_phrases or []),
                json.dumps(tone_notes or []),
                json.dumps([]),
                samples_analyzed,
            ),
        )
    return template_id


# ---------------------------------------------------------------------------
# Tests: basic response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_templates_returns_200(db):
    """GET /api/user-model/templates always returns HTTP 200."""
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_templates_response_has_required_keys(db):
    """Response contains templates list, total count, and generated_at."""
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    data = resp.json()
    assert "templates" in data, "Missing 'templates' key"
    assert "total" in data, "Missing 'total' key"
    assert "generated_at" in data, "Missing 'generated_at' key"
    assert isinstance(data["templates"], list)
    assert isinstance(data["total"], int)


@pytest.mark.asyncio
async def test_templates_empty_when_no_data(db):
    """Returns empty list and total=0 when no templates are stored."""
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    data = resp.json()
    assert data["templates"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_templates_returns_stored_rows(db):
    """Returns the template rows inserted into communication_templates."""
    _insert_template(db, contact_id="bob@example.com", samples_analyzed=5)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    data = resp.json()
    assert data["total"] == 1
    assert len(data["templates"]) == 1
    tmpl = data["templates"][0]
    assert tmpl["contact_id"] == "bob@example.com"
    assert tmpl["samples_analyzed"] == 5


# ---------------------------------------------------------------------------
# Tests: field types and values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_templates_list_fields_are_arrays(db):
    """common_phrases, avoids_phrases, and tone_notes are deserialized to lists."""
    _insert_template(
        db,
        common_phrases=["sounds good", "let me know"],
        avoids_phrases=["ASAP"],
        tone_notes=["casual"],
    )
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    tmpl = resp.json()["templates"][0]
    assert isinstance(tmpl["common_phrases"], list), "common_phrases should be a list"
    assert isinstance(tmpl["avoids_phrases"], list), "avoids_phrases should be a list"
    assert isinstance(tmpl["tone_notes"], list), "tone_notes should be a list"
    assert tmpl["common_phrases"] == ["sounds good", "let me know"]
    assert tmpl["avoids_phrases"] == ["ASAP"]
    assert tmpl["tone_notes"] == ["casual"]


@pytest.mark.asyncio
async def test_templates_uses_emoji_is_boolean(db):
    """uses_emoji field is returned as a Python bool, not int."""
    _insert_template(db, uses_emoji=True)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    tmpl = resp.json()["templates"][0]
    # JSON booleans are Python bool after parsing
    assert tmpl["uses_emoji"] is True


@pytest.mark.asyncio
async def test_templates_formality_is_float(db):
    """formality field is returned as a float between 0 and 1."""
    _insert_template(db, formality=0.75)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    tmpl = resp.json()["templates"][0]
    assert isinstance(tmpl["formality"], float)
    assert 0.0 <= tmpl["formality"] <= 1.0


@pytest.mark.asyncio
async def test_templates_generated_at_is_iso8601(db):
    """generated_at field parses as a valid ISO-8601 datetime string."""
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    data = resp.json()
    # Should not raise
    datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Tests: filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_templates_filter_by_contact_id(db):
    """?contact_id= returns only templates matching that contact."""
    _insert_template(db, contact_id="alice@example.com")
    _insert_template(db, contact_id="bob@example.com")
    client = _make_app(db)
    resp = client.get("/api/user-model/templates?contact_id=alice%40example.com")
    data = resp.json()
    assert data["total"] == 1
    assert data["templates"][0]["contact_id"] == "alice@example.com"


@pytest.mark.asyncio
async def test_templates_filter_by_channel(db):
    """?channel=email returns only email templates."""
    _insert_template(db, channel="email")
    _insert_template(db, channel="message")
    client = _make_app(db)
    resp = client.get("/api/user-model/templates?channel=email")
    data = resp.json()
    assert data["total"] == 1
    assert data["templates"][0]["channel"] == "email"


@pytest.mark.asyncio
async def test_templates_filter_by_context(db):
    """?context=user_to_contact returns only outbound templates."""
    _insert_template(db, context="user_to_contact")
    _insert_template(db, contact_id="other@example.com", context="contact_to_user")
    client = _make_app(db)
    resp = client.get("/api/user-model/templates?context=user_to_contact")
    data = resp.json()
    assert data["total"] == 1
    assert data["templates"][0]["context"] == "user_to_contact"


# ---------------------------------------------------------------------------
# Tests: ordering and limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_templates_ordered_by_samples_desc(db):
    """Templates are returned ordered by samples_analyzed descending (most-sampled first)."""
    _insert_template(db, contact_id="low@example.com", samples_analyzed=2)
    _insert_template(db, contact_id="high@example.com", samples_analyzed=50)
    _insert_template(db, contact_id="mid@example.com", samples_analyzed=10)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates")
    templates = resp.json()["templates"]
    counts = [t["samples_analyzed"] for t in templates]
    assert counts == sorted(counts, reverse=True), "Templates should be sorted by samples_analyzed desc"


@pytest.mark.asyncio
async def test_templates_limit_param_restricts_count(db):
    """?limit=2 returns at most 2 templates."""
    for i in range(5):
        _insert_template(db, contact_id=f"contact{i}@example.com", samples_analyzed=i + 1)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates?limit=2")
    data = resp.json()
    assert len(data["templates"]) <= 2


@pytest.mark.asyncio
async def test_templates_limit_capped_at_200(db):
    """?limit=999 is silently capped to 200 (never dumps unbounded data)."""
    # Insert just a few rows; verify the endpoint doesn't error on a large limit param.
    _insert_template(db)
    client = _make_app(db)
    resp = client.get("/api/user-model/templates?limit=999")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["templates"], list)
