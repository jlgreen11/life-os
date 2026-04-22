"""Tests for :mod:`api.routes.now` — Now tab + 4 moment-action endpoints.

Every test spins up a fresh in-memory SQLite with the full v2 schema,
wires a real :class:`~storage.repos.moments.MomentRepository` + a real
:class:`~core.moment.feedback_weight.FeedbackWeightStore` onto a dummy
``LifeOS`` double, and hits the app via :class:`fastapi.testclient.TestClient`.
No mocks of the storage layer — same pattern the v1 ``tests/conftest.py``
established.

Coverage (per NEXT_TASKS.md Week 8 acceptance):

- ``GET /api/now`` returns ``{pending, scheduled, done}`` in the schema
  shape, each capped at 20/10/10.
- Each action endpoint round-trips through the state machine and the
  feedback-weight EWMA.
- Invalid transitions (e.g. snooze a DISMISSED Moment) return 409.
- Unknown ``moment_id`` returns 404 on every mutation endpoint.
- ``snooze`` without ``snooze_until`` returns 422.
- ``edit`` updates ``proposed_action.params`` without moving state.
- 503 when ``moment_repo`` is not wired onto life_os.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.moment.feedback_weight import ALPHA, DEFAULT_WEIGHT, FeedbackWeightStore
from core.moment.types import (
    Action,
    ActionKind,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.moments import MomentRepository

# Fixed reference epoch — 2026-04-22T12:00:00Z — so date-sensitive tests
# (``list_done_today``) give stable results.
REF_NOW = 1_777_204_800


class Clock:
    """Mutable ``time.time`` stand-in used by the repo and weight store."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class DummyLifeOS:
    """Minimal life_os double the route layer dereferences.

    Attributes match what :mod:`api.routes.now` looks up:
    ``moment_repo`` and ``feedback_weight_store``. Tests that want to
    exercise the 503 fail-soft path leave one of them as ``None`` or
    skip the attribute entirely.
    """

    def __init__(self, moment_repo=None, feedback_weight_store=None) -> None:
        self.config: dict = {}
        self.moment_repo = moment_repo
        self.feedback_weight_store = feedback_weight_store


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the full v2 schema and FKs on."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def repo(conn, clock):
    return MomentRepository(conn, now_fn=clock)


@pytest.fixture
def feedback(conn, clock):
    return FeedbackWeightStore(conn, now_fn=clock)


@pytest.fixture
def client(repo, feedback):
    life_os = DummyLifeOS(moment_repo=repo, feedback_weight_store=feedback)
    app = create_app(life_os)
    return TestClient(app)


def _make_moment(
    *,
    insight_type: InsightType = InsightType.CADENCE,
    evidence_hash: str | None = None,
    state: MomentState = MomentState.SUGGESTED,
    scheduled_for: int | None = None,
    confidence: float = 0.5,
    insight: str = "ping your sister",
    draft: str = "Hey — been a minute. How are you?",
    moment_id: str | None = None,
    created_at: int = REF_NOW,
    expires_at: int = REF_NOW + 3 * 24 * 3600,
) -> Moment:
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=created_at,
        expires_at=expires_at,
        insight=insight,
        evidence_hash=evidence_hash or f"hash-{uuid.uuid4().hex[:8]}",
        proposed_action=Action(kind=ActionKind.DRAFT_MESSAGE, params={"body": draft}),
        source_insight_type=insight_type,
        scheduled_for=scheduled_for,
        evidence=["evt-1", "evt-2"],
        state=state,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# GET /api/now
# ---------------------------------------------------------------------------


def test_get_now_empty_state_returns_empty_lists(client: TestClient) -> None:
    """Empty repo → empty lists (never ``None``)."""
    resp = client.get("/api/now")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"pending": [], "scheduled": [], "done": []}


def test_get_now_schema_round_trip(client: TestClient, repo) -> None:
    """A SUGGESTED moment shows up under ``pending`` with every field."""
    mom = _make_moment(confidence=0.9, insight="water the plants")
    repo.create(mom)

    resp = client.get("/api/now")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pending"]) == 1
    assert data["scheduled"] == []
    assert data["done"] == []

    out = data["pending"][0]
    # State-independent fields.
    assert out["id"] == mom.id
    assert out["insight"] == "water the plants"
    assert out["evidence_hash"] == mom.evidence_hash
    assert out["state"] == MomentState.SUGGESTED.value
    assert out["source_insight_type"] == InsightType.CADENCE.value
    assert out["proposed_action"]["kind"] == ActionKind.DRAFT_MESSAGE.value
    assert out["proposed_action"]["params"]["body"].startswith("Hey")
    # Creation appends exactly one history row.
    assert len(out["state_history"]) == 1
    assert out["state_history"][0]["to_state"] == MomentState.SUGGESTED.value


def test_get_now_orders_pending_by_confidence_desc(client: TestClient, repo) -> None:
    low = _make_moment(confidence=0.3, evidence_hash="low", insight="low")
    high = _make_moment(confidence=0.9, evidence_hash="high", insight="high")
    repo.create(low)
    repo.create(high)

    data = client.get("/api/now").json()
    assert [m["insight"] for m in data["pending"]] == ["high", "low"]


def test_get_now_respects_pending_limit(client: TestClient, repo) -> None:
    """The 20-moment cap is enforced by the route, not just the schema."""
    from api.routes.now import PENDING_LIMIT

    for i in range(PENDING_LIMIT + 5):
        repo.create(_make_moment(evidence_hash=f"hash-{i}", confidence=0.5))

    data = client.get("/api/now").json()
    assert len(data["pending"]) == PENDING_LIMIT


def test_get_now_returns_503_when_moment_repo_missing() -> None:
    """Half-constructed life_os → 503 instead of an AttributeError trace."""
    app = create_app(DummyLifeOS(moment_repo=None, feedback_weight_store=None))
    c = TestClient(app)
    resp = c.get("/api/now")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/moments/{id}/accept
# ---------------------------------------------------------------------------


def test_accept_moves_suggested_to_accepted(client: TestClient, repo, feedback) -> None:
    mom = _make_moment(insight_type=InsightType.CADENCE, confidence=0.9)
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/accept")
    assert resp.status_code == 200
    out = resp.json()
    assert out["state"] == MomentState.ACCEPTED.value
    # History has create + transition rows.
    states = [h["to_state"] for h in out["state_history"]]
    assert states == [MomentState.SUGGESTED.value, MomentState.ACCEPTED.value]
    # Feedback weight shifted upward toward 1.0.
    weight, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 1
    expected = ALPHA * 1.0 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight == pytest.approx(expected)


def test_accept_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.post("/api/moments/does-not-exist/accept")
    assert resp.status_code == 404


def test_accept_from_terminal_state_returns_409(client: TestClient, repo) -> None:
    mom = _make_moment(confidence=0.9)
    repo.create(mom)
    repo.transition(mom.id, MomentState.DISMISSED, annotation="user dismissed")

    resp = client.post(f"/api/moments/{mom.id}/accept")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/moments/{id}/dismiss
# ---------------------------------------------------------------------------


def test_dismiss_moves_to_dismissed_and_weights_drop(client: TestClient, repo, feedback) -> None:
    mom = _make_moment(insight_type=InsightType.RELATIONSHIP, confidence=0.9)
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/dismiss")
    assert resp.status_code == 200
    out = resp.json()
    assert out["state"] == MomentState.DISMISSED.value

    weight, decisions = feedback.get(InsightType.RELATIONSHIP.value)
    assert decisions == 1
    expected = ALPHA * 0.0 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight == pytest.approx(expected)


def test_dismiss_invalid_transition_returns_409(client: TestClient, repo) -> None:
    mom = _make_moment(confidence=0.9)
    repo.create(mom)
    repo.transition(mom.id, MomentState.ACCEPTED)

    resp = client.post(f"/api/moments/{mom.id}/dismiss")
    assert resp.status_code == 409


def test_dismiss_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.post("/api/moments/missing/dismiss")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/moments/{id}/snooze
# ---------------------------------------------------------------------------


def test_snooze_requires_snooze_until(client: TestClient, repo) -> None:
    mom = _make_moment()
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/snooze", json={})
    assert resp.status_code == 422


def test_snooze_records_snooze_until_and_weight(client: TestClient, repo, feedback) -> None:
    mom = _make_moment(insight_type=InsightType.ROUTINE, confidence=0.9)
    repo.create(mom)
    target = REF_NOW + 3600  # 1h snooze

    resp = client.post(f"/api/moments/{mom.id}/snooze", json={"snooze_until": target})
    assert resp.status_code == 200
    out = resp.json()
    assert out["state"] == MomentState.SNOOZED.value
    assert out["snooze_until"] == target

    weight, decisions = feedback.get(InsightType.ROUTINE.value)
    assert decisions == 1
    expected = ALPHA * 0.5 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight == pytest.approx(expected)


def test_snooze_past_expires_at_coerces_to_expired(client: TestClient, repo, feedback) -> None:
    """Per eng plan § 'Snooze semantics'. Still a 200 response."""
    mom = _make_moment(expires_at=REF_NOW + 60)
    repo.create(mom)

    target = REF_NOW + 3600  # past expires_at
    resp = client.post(f"/api/moments/{mom.id}/snooze", json={"snooze_until": target})
    assert resp.status_code == 200
    assert resp.json()["state"] == MomentState.EXPIRED.value


def test_snooze_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.post("/api/moments/missing/snooze", json={"snooze_until": REF_NOW + 60})
    assert resp.status_code == 404


def test_snooze_from_accepted_returns_409(client: TestClient, repo) -> None:
    mom = _make_moment()
    repo.create(mom)
    repo.transition(mom.id, MomentState.ACCEPTED)

    resp = client.post(f"/api/moments/{mom.id}/snooze", json={"snooze_until": REF_NOW + 60})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/moments/{id}/edit
# ---------------------------------------------------------------------------


def test_edit_updates_action_params_in_place(client: TestClient, repo, feedback) -> None:
    mom = _make_moment()
    repo.create(mom)

    new_params = {"body": "Rewritten draft", "priority": "high"}
    resp = client.post(
        f"/api/moments/{mom.id}/edit",
        json={"action_params": new_params},
    )
    assert resp.status_code == 200
    out = resp.json()
    # State is unchanged — edit is not a transition.
    assert out["state"] == MomentState.SUGGESTED.value
    assert out["proposed_action"]["params"] == new_params
    # And no feedback decision was recorded.
    _, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 0


def test_edit_requires_action_params(client: TestClient, repo) -> None:
    mom = _make_moment()
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/edit", json={})
    assert resp.status_code == 422


def test_edit_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/api/moments/missing/edit",
        json={"action_params": {"body": "x"}},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# forbid-extra on inputs
# ---------------------------------------------------------------------------


def test_snooze_rejects_unknown_fields(client: TestClient, repo) -> None:
    """``extra='forbid'`` on MomentActionIn means drifted clients 422."""
    mom = _make_moment()
    repo.create(mom)

    resp = client.post(
        f"/api/moments/{mom.id}/snooze",
        json={"snooze_until": REF_NOW + 60, "unknown": "drop-me"},
    )
    assert resp.status_code == 422
