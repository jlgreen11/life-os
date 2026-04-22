"""Tests for :mod:`api.routes.settings` — connector listing, patch, dry-run.

Three REST routes + the 503 fail-soft path + the Fernet-never-leaks
invariant. Tests use a stub ``connector_repo`` rather than a live SQL
repository because the Week 8 task only locks the route surface — the
storage-backed repo lands in a later task (it will be swapped in here
without touching the route handlers).

Coverage (per NEXT_TASKS.md Week 8 acceptance):

- ``GET /api/connectors`` returns the repo's list in
  :class:`~api.schemas.ConnectorOut` shape.
- ``PATCH /api/connectors/{id}`` forwards every ``ConnectorConfigIn``
  field to the repo, and the response body NEVER contains a plaintext
  secret the request carried (Fernet invariant).
- ``PATCH`` with an unknown id returns 404.
- ``PATCH`` with an unexpected field returns 422 (``extra='forbid'``).
- ``POST /api/connectors/{id}/test`` returns the repo's status dict.
- ``POST`` with an unknown id returns 404.
- Every endpoint returns 503 when ``connector_repo`` is not wired.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from api.app import create_app


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


class DummyLifeOS:
    def __init__(self, connector_repo: Any = None) -> None:
        self.config: dict = {}
        self.connector_repo = connector_repo


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
