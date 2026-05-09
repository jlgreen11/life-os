"""Tests for :mod:`api.routes.settings` — connector listing, patch, dry-run,
plus the three Week-10 HTML routes (``GET /settings``,
``GET /settings/connectors/{id}/edit``, ``POST /settings/connectors/{id}/test``).

Tests use a stub ``connector_repo`` rather than a live SQL repository —
the storage-backed repo lands in a later task and will be swapped in
here without touching the route handlers.

Coverage:

- ``GET /api/connectors`` returns the repo's list in
  :class:`~api.schemas.ConnectorOut` shape.
- ``PATCH /api/connectors/{id}`` forwards every ``ConnectorConfigIn``
  field to the repo, and the response body NEVER contains a plaintext
  secret the request carried (Fernet invariant).
- ``PATCH`` with an unknown id returns 404.
- ``PATCH`` with an unexpected field returns 422 (``extra='forbid'``).
- ``POST /api/connectors/{id}/test`` returns the repo's status dict.
- ``POST`` with an unknown id returns 404.
- Every JSON endpoint returns 503 when ``connector_repo`` is not wired.
- ``GET /settings`` renders the full HTML page with the connectors list
  and the four page-render preferences from ``life_os.db``.
- ``GET /settings/connectors/{id}/edit`` renders the edit form partial
  and NEVER leaks Fernet secrets (invariant re-checked on the partial).
- ``POST /settings/connectors/{id}/test`` renders the HTML result
  partial and surfaces success / failure symbols.
- Preferences persist across requests via the /api/preferences shim;
  the second ``GET /settings`` reflects the stored value.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi.testclient import TestClient

from api.app import create_app
from storage import schema


class StubConnectorRepo:
    """Duck-typed connector repo used by the three Settings routes."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.tests: list[str] = []
        self._rows: dict[str, dict[str, Any]] = {
            "proton_mail": {
                "id": "proton_mail",
                "kind": "proton_mail",
                "enabled": True,
                "status": "ok",
                "last_sync_at": 1_777_204_700,
                "last_error": None,
            },
            "signal": {
                "id": "signal",
                "kind": "signal",
                "enabled": False,
                "status": "disabled",
                "last_sync_at": None,
                "last_error": None,
            },
        }

    def list(self) -> list[dict[str, Any]]:
        return list(self._rows.values())

    def update(
        self,
        connector_id: str,
        *,
        enabled: bool | None,
        config: dict | None,
        secrets: dict | None,
    ) -> dict[str, Any] | None:
        row = self._rows.get(connector_id)
        if row is None:
            return None
        self.updates.append(
            {
                "id": connector_id,
                "enabled": enabled,
                "config": config,
                "secrets": secrets,
            }
        )
        updated = dict(row)
        if enabled is not None:
            updated["enabled"] = enabled
        # Simulate the repo returning ONLY status-level fields — no secrets.
        return updated

    def test(self, connector_id: str) -> dict[str, Any] | None:
        if connector_id not in self._rows:
            return None
        self.tests.append(connector_id)
        return {
            "ok": True,
            "message": "dry-run sync complete",
            "details": {"probed": connector_id},
        }


class StubConnectorRepoWithEditView(StubConnectorRepo):
    """Extends the base stub with the optional ``edit_view`` method.

    Used to cover the HTML edit-form path — the storage-backed repo
    will implement this hook and the route picks it up via duck-type
    dispatch. The view surfaces ``config`` + ``secret_keys`` but never
    the secret values themselves (Fernet invariant).
    """

    def edit_view(self, connector_id: str) -> dict[str, Any] | None:
        row = self._rows.get(connector_id)
        if row is None:
            return None
        return {
            **row,
            "config": {"username": "alice@example.com"},
            "secret_keys": ["password"],
        }


class DummyLifeOS:
    def __init__(
        self,
        connector_repo: Any = None,
        db: sqlite3.Connection | None = None,
    ) -> None:
        self.config: dict = {}
        self.connector_repo = connector_repo
        self.db = db


def _client(life_os: Any) -> TestClient:
    return TestClient(create_app(life_os))


# ---------------------------------------------------------------------------
# GET /api/connectors
# ---------------------------------------------------------------------------


def test_list_connectors_returns_registry_rows():
    repo = StubConnectorRepo()
    resp = _client(DummyLifeOS(repo)).get("/api/connectors")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    ids = {row["id"] for row in body}
    assert ids == {"proton_mail", "signal"}
    # Schema-forbidden extras — no stray fields leak through ConnectorOut.
    assert set(body[0].keys()) == {
        "id",
        "kind",
        "enabled",
        "status",
        "last_sync_at",
        "last_error",
    }


def test_list_connectors_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).get("/api/connectors")
    assert resp.status_code == 503
    assert "connector_repo" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# PATCH /api/connectors/{id}
# ---------------------------------------------------------------------------


def test_patch_connector_forwards_body_to_repo():
    repo = StubConnectorRepo()
    resp = _client(DummyLifeOS(repo)).patch(
        "/api/connectors/proton_mail",
        json={
            "enabled": False,
            "config": {"username": "alice@example.com"},
            "secrets": {"password": "SUPER-SECRET-42"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "proton_mail"
    assert body["enabled"] is False
    # The repo saw every field verbatim.
    assert repo.updates == [
        {
            "id": "proton_mail",
            "enabled": False,
            "config": {"username": "alice@example.com"},
            "secrets": {"password": "SUPER-SECRET-42"},
        }
    ]


def test_patch_connector_never_returns_raw_secrets():
    """Fernet invariant — the plaintext secret on the wire never comes back.

    This is the single most important security invariant of the Settings
    surface. A regression here means a readable GET-after-PATCH leaks
    credentials to any client that can hit the API.
    """
    repo = StubConnectorRepo()
    resp = _client(DummyLifeOS(repo)).patch(
        "/api/connectors/proton_mail",
        json={"secrets": {"password": "LEAK-CANARY-STRING"}},
    )
    assert resp.status_code == 200
    # Scan the serialized response body for the plaintext canary.
    assert "LEAK-CANARY-STRING" not in resp.text
    # And the schema doesn't have a "secrets" or "password" key at all.
    body = resp.json()
    assert "secrets" not in body
    assert "password" not in body


def test_patch_connector_404_when_unknown():
    resp = _client(DummyLifeOS(StubConnectorRepo())).patch(
        "/api/connectors/does_not_exist",
        json={"enabled": True},
    )
    assert resp.status_code == 404


def test_patch_connector_rejects_extra_fields():
    resp = _client(DummyLifeOS(StubConnectorRepo())).patch(
        "/api/connectors/proton_mail",
        json={"unexpected": "value"},
    )
    assert resp.status_code == 422


def test_patch_connector_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).patch(
        "/api/connectors/proton_mail",
        json={"enabled": True},
    )
    assert resp.status_code == 503


def test_patch_connector_accepts_empty_body():
    """A PATCH with all-null fields is a legal no-op — client just touches the row."""
    repo = StubConnectorRepo()
    resp = _client(DummyLifeOS(repo)).patch("/api/connectors/proton_mail", json={})
    assert resp.status_code == 200
    assert repo.updates[0] == {
        "id": "proton_mail",
        "enabled": None,
        "config": None,
        "secrets": None,
    }


# ---------------------------------------------------------------------------
# POST /api/connectors/{id}/test
# ---------------------------------------------------------------------------


def test_test_connector_returns_repo_status_dict():
    repo = StubConnectorRepo()
    resp = _client(DummyLifeOS(repo)).post("/api/connectors/proton_mail/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "ok": True,
        "message": "dry-run sync complete",
        "details": {"probed": "proton_mail"},
    }
    assert repo.tests == ["proton_mail"]


def test_test_connector_404_when_unknown():
    resp = _client(DummyLifeOS(StubConnectorRepo())).post("/api/connectors/missing/test")
    assert resp.status_code == 404


def test_test_connector_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).post("/api/connectors/proton_mail/test")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /settings (HTML page)
# ---------------------------------------------------------------------------


def _db_with_schema() -> sqlite3.Connection:
    """Spin up a fresh in-memory db with the full v2 schema.

    Exposes a :class:`sqlite3.Connection` shaped like ``life_os.db`` so
    the settings page can read its preference rows and the preferences
    shim can write to them.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()
    return conn


def test_get_settings_page_renders_connectors_and_defaults():
    repo = StubConnectorRepo()
    conn = _db_with_schema()
    client = _client(DummyLifeOS(repo, db=conn))
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # The Settings nav tab is current.
    assert 'data-active-tab="settings"' in body
    # Both connector rows render.
    assert 'data-connector-id="proton_mail"' in body
    assert 'data-connector-id="signal"' in body
    # Preferences fall back to schema defaults on a fresh install.
    assert 'value="22:00"' in body
    assert 'value="07:00"' in body
    # 0.5 is the default autonomy AND proactivity.
    assert body.count('value="0.5"') >= 2


def test_get_settings_page_reflects_persisted_preferences():
    """POST /api/preferences must round-trip through GET /settings.

    This is the task's "preferences persist" acceptance check — writing
    a value via the shim must show up on the next page render.
    """
    repo = StubConnectorRepo()
    conn = _db_with_schema()
    client = _client(DummyLifeOS(repo, db=conn))
    # Persist all four keys via the existing shim (iOS-compat POST).
    for key, value in [
        ("quiet_hours_start", "21:30"),
        ("quiet_hours_end", "06:45"),
        ("autonomy_level", 0.7),
        ("proactivity", 0.25),
    ]:
        resp = client.post("/api/preferences", json={"key": key, "value": value})
        assert resp.status_code == 200
    # Re-render the page — every stored value is visible.
    body = client.get("/settings").text
    assert 'value="21:30"' in body
    assert 'value="06:45"' in body
    assert 'value="0.7"' in body
    assert 'value="0.25"' in body


def test_get_settings_page_empty_connectors_renders_empty_state():
    repo = StubConnectorRepo()
    # Drop all rows to simulate a fresh install.
    repo._rows.clear()
    client = _client(DummyLifeOS(repo))
    body = client.get("/settings").text
    assert 'data-empty="connectors"' in body
    assert "No connectors configured yet." in body


def test_get_settings_page_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).get("/settings")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /settings/connectors/{id}/edit  (HTML edit-form partial)
# ---------------------------------------------------------------------------


def test_edit_connector_renders_form_partial_via_edit_view():
    repo = StubConnectorRepoWithEditView()
    client = _client(DummyLifeOS(repo))
    resp = client.get("/settings/connectors/proton_mail/edit")
    assert resp.status_code == 200
    body = resp.text
    assert 'data-slot="edit-form"' in body
    assert 'hx-patch="/api/connectors/proton_mail"' in body
    assert "alice@example.com" in body
    assert 'name="secrets.password"' in body


def test_edit_connector_never_leaks_secret_values_on_any_path():
    """Fernet invariant — the edit form never carries a secret value.

    Fed a repo that deliberately tries to stuff a plaintext secret into
    its ``edit_view`` payload, the route MUST still render the form
    with an empty password input.
    """

    class LeakyRepo(StubConnectorRepoWithEditView):
        def edit_view(self, connector_id: str) -> dict[str, Any] | None:
            view = super().edit_view(connector_id)
            if view is None:
                return None
            # Even if a buggy repo leaked the secret, the TEMPLATE must
            # drop it. The edit form never iterates secret VALUES — it
            # only renders the key names from ``secret_keys``.
            view["secrets"] = {"password": "LEAK-CANARY-STRING"}
            return view

    client = _client(DummyLifeOS(LeakyRepo()))
    body = client.get("/settings/connectors/proton_mail/edit").text
    assert "LEAK-CANARY-STRING" not in body


def test_edit_connector_falls_back_to_list_when_edit_view_missing():
    """Stub repos without ``edit_view`` still render a minimal form."""
    repo = StubConnectorRepo()  # no edit_view
    client = _client(DummyLifeOS(repo))
    resp = client.get("/settings/connectors/proton_mail/edit")
    assert resp.status_code == 200
    body = resp.text
    assert 'data-slot="edit-form"' in body
    # No config fields / secret fields because the stub doesn't surface them.
    assert 'data-slot="config-field"' not in body
    assert 'data-slot="secret-field"' not in body


def test_edit_connector_404_when_unknown():
    client = _client(DummyLifeOS(StubConnectorRepoWithEditView()))
    resp = client.get("/settings/connectors/does_not_exist/edit")
    assert resp.status_code == 404


def test_edit_connector_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).get("/settings/connectors/proton_mail/edit")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /settings/connectors/{id}/test (HTML test-result partial)
# ---------------------------------------------------------------------------


def test_post_test_connector_html_returns_success_partial():
    client = _client(DummyLifeOS(StubConnectorRepo()))
    resp = client.post("/settings/connectors/proton_mail/test")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert 'data-slot="test-ok"' in body
    assert "dry-run sync complete" in body


def test_post_test_connector_html_returns_failure_partial():
    class FailingRepo(StubConnectorRepo):
        def test(self, connector_id: str) -> dict[str, Any] | None:
            if connector_id not in self._rows:
                return None
            return {"ok": False, "message": "auth failed"}

    client = _client(DummyLifeOS(FailingRepo()))
    body = client.post("/settings/connectors/proton_mail/test").text
    assert 'data-slot="test-fail"' in body
    assert "auth failed" in body


def test_post_test_connector_html_404_when_unknown():
    resp = _client(DummyLifeOS(StubConnectorRepo())).post("/settings/connectors/does_not_exist/test")
    assert resp.status_code == 404


def test_post_test_connector_html_503_when_repo_not_wired():
    resp = _client(DummyLifeOS(connector_repo=None)).post("/settings/connectors/proton_mail/test")
    assert resp.status_code == 503


def test_settings_detail_empty_returns_placeholder():
    """Cancel button on the edit form hits this endpoint to reset the pane."""
    client = _client(DummyLifeOS(StubConnectorRepo()))
    resp = client.get("/settings/detail-empty")
    assert resp.status_code == 200
    assert 'data-empty="detail-pane"' in resp.text
    assert "Select a connector to edit." in resp.text
