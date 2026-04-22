"""Tests for ``POST /api/moments/{id}/undo`` (api.routes.now.undo_moment).

The undo endpoint bounces a Moment back to ``SUGGESTED`` within a 3 s
grace window after the user's accept / dismiss / snooze. Per design note
``docs/plans/2026-04-22-undo-grace.md`` § "Decision 1" + "Test plan".

Branches under test (per design note § "Test plan → Undo route"):

- **200** — accept then undo within 3 s → state back to SUGGESTED,
  ``state_history`` ends in an ``annotation='undo'`` row, feedback
  EWMA is compensated with the inverse signal.
- **410 Gone** — undo after the 3 s grace window has elapsed; Moment
  stays ACCEPTED, no extra history row appended.
- **409 Conflict** — undo on a Moment that has only the creation row
  (``to_state == SUGGESTED``); also when the current top transition is
  not in the undoable set (e.g. EXPIRED, DONE).
- **404 Not Found** — undo on an unknown id.
- Dismiss → undo path works the same way (no outbox row to cancel —
  outbox cancellation is the next NEXT_TASKS task).
- Idempotency: undo twice within the grace window — first 200, second
  409 (already SUGGESTED).
- HTMX path (``HX-Request: true``) returns the OOB swap partial that
  re-injects the bounced card at the top of ``#now-list``.

The repo's clock is exposed as ``repo._now_fn`` (an injected callable);
tests advance it directly to simulate elapsed time. No ``freezegun`` —
matches the eng-review stdlib-only constraint.
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

REF_NOW = 1_777_204_800


class Clock:
    """Mutable ``time.time`` stand-in shared by repo + feedback store."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class DummyLifeOS:
    def __init__(self, moment_repo=None, feedback_weight_store=None) -> None:
        self.config: dict = {}
        self.moment_repo = moment_repo
        self.feedback_weight_store = feedback_weight_store


@pytest.fixture
def conn():
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
    return TestClient(create_app(life_os))


def _make_moment(
    *,
    insight_type: InsightType = InsightType.CADENCE,
    confidence: float = 0.9,
    moment_id: str | None = None,
    evidence_hash: str | None = None,
    insight: str = "ping your sister",
) -> Moment:
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=REF_NOW,
        expires_at=REF_NOW + 3 * 24 * 3600,
        insight=insight,
        evidence_hash=evidence_hash or f"hash-{uuid.uuid4().hex[:8]}",
        proposed_action=Action(kind=ActionKind.DRAFT_MESSAGE, params={"body": "Hey"}),
        source_insight_type=insight_type,
        evidence=["evt-1"],
        state=MomentState.SUGGESTED,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 200 — happy path within the grace window
# ---------------------------------------------------------------------------


def test_undo_within_grace_returns_moment_to_suggested(
    client: TestClient,
    repo,
    feedback,
    clock,
) -> None:
    """Accept then undo within 3 s → state back to SUGGESTED with annotation='undo'."""
    mom = _make_moment(insight_type=InsightType.CADENCE)
    repo.create(mom)

    # User accepts at t=REF_NOW.
    accept_resp = client.post(f"/api/moments/{mom.id}/accept")
    assert accept_resp.status_code == 200
    weight_after_accept, _ = feedback.get(InsightType.CADENCE.value)
    expected_after_accept = ALPHA * 1.0 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight_after_accept == pytest.approx(expected_after_accept)

    # User clicks Undo 2s later — well within the 3s grace window.
    clock.t = REF_NOW + 2

    undo_resp = client.post(f"/api/moments/{mom.id}/undo")
    assert undo_resp.status_code == 200
    out = undo_resp.json()

    # State is back to SUGGESTED.
    assert out["state"] == MomentState.SUGGESTED.value
    # History has create + ACCEPTED + undo-back-to-SUGGESTED.
    states = [(h["from_state"], h["to_state"]) for h in out["state_history"]]
    assert states == [
        (None, MomentState.SUGGESTED.value),
        (MomentState.SUGGESTED.value, MomentState.ACCEPTED.value),
        (MomentState.ACCEPTED.value, MomentState.SUGGESTED.value),
    ]
    annotations = [h["annotation"] for h in out["state_history"]]
    assert annotations[-1] == "undo"

    # Feedback compensation: inverse of ACCEPTED is DISMISSED (signal=0.0).
    weight_after_undo, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 2
    expected_after_undo = ALPHA * 0.0 + (1 - ALPHA) * weight_after_accept
    assert weight_after_undo == pytest.approx(expected_after_undo)


def test_undo_dismiss_within_grace_succeeds(
    client: TestClient,
    repo,
    feedback,
    clock,
) -> None:
    """Dismiss then undo: same SUGGESTED return, inverse feedback (1.0 — accept-equivalent)."""
    mom = _make_moment(insight_type=InsightType.RELATIONSHIP)
    repo.create(mom)

    client.post(f"/api/moments/{mom.id}/dismiss")
    weight_after_dismiss, _ = feedback.get(InsightType.RELATIONSHIP.value)
    expected_after_dismiss = ALPHA * 0.0 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight_after_dismiss == pytest.approx(expected_after_dismiss)

    clock.t = REF_NOW + 1
    undo_resp = client.post(f"/api/moments/{mom.id}/undo")
    assert undo_resp.status_code == 200
    assert undo_resp.json()["state"] == MomentState.SUGGESTED.value

    weight_after_undo, decisions = feedback.get(InsightType.RELATIONSHIP.value)
    assert decisions == 2
    # Inverse of DISMISSED is ACCEPTED (signal=1.0).
    expected_after_undo = ALPHA * 1.0 + (1 - ALPHA) * weight_after_dismiss
    assert weight_after_undo == pytest.approx(expected_after_undo)


def test_undo_snooze_within_grace_succeeds(
    client: TestClient,
    repo,
    feedback,
    clock,
) -> None:
    """Snooze then undo: state returns to SUGGESTED; SNOOZED→SUGGESTED was already a legal edge.

    The compensation EWMA records SNOOZED again (signal=0.5 — same as the
    original snooze), which is intentional per design note: snooze's
    inverse is itself, so the running mean is unchanged.
    """
    mom = _make_moment(insight_type=InsightType.ROUTINE)
    repo.create(mom)

    client.post(
        f"/api/moments/{mom.id}/snooze",
        json={"snooze_until": REF_NOW + 3600},
    )

    clock.t = REF_NOW + 1
    undo_resp = client.post(f"/api/moments/{mom.id}/undo")
    assert undo_resp.status_code == 200
    assert undo_resp.json()["state"] == MomentState.SUGGESTED.value
    # decision_count grew on both the snooze AND the undo.
    _, decisions = feedback.get(InsightType.ROUTINE.value)
    assert decisions == 2


# ---------------------------------------------------------------------------
# 410 — grace window expired
# ---------------------------------------------------------------------------


def test_undo_after_grace_returns_410_and_leaves_state(
    client: TestClient,
    repo,
    feedback,
    clock,
) -> None:
    """Past the 3s grace window the endpoint returns 410 Gone."""
    mom = _make_moment(insight_type=InsightType.CADENCE)
    repo.create(mom)
    client.post(f"/api/moments/{mom.id}/accept")

    # 5 s past the accept — outside the 3 s window.
    clock.t = REF_NOW + 5
    resp = client.post(f"/api/moments/{mom.id}/undo")
    assert resp.status_code == 410

    # The Moment stays ACCEPTED — no rollback happened.
    fresh = repo.get(mom.id)
    assert fresh is not None
    assert fresh.state == MomentState.ACCEPTED
    # And no extra history row was appended.
    assert len(fresh.state_history) == 2  # create + ACCEPTED
    # Feedback decision_count stayed at 1 (no compensation was recorded).
    _, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 1


def test_undo_at_exact_grace_boundary_succeeds(
    client: TestClient,
    repo,
    clock,
) -> None:
    """Boundary check: ``now_ts - last.ts == 3`` is **inside** the window."""
    mom = _make_moment()
    repo.create(mom)
    client.post(f"/api/moments/{mom.id}/accept")

    clock.t = REF_NOW + 3  # exactly at the boundary
    resp = client.post(f"/api/moments/{mom.id}/undo")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 409 — nothing to undo / state not undoable
# ---------------------------------------------------------------------------


def test_undo_when_only_creation_row_returns_409(
    client: TestClient,
    repo,
) -> None:
    """A pristine SUGGESTED Moment (creation row only) → 409 nothing to undo."""
    mom = _make_moment()
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/undo")
    assert resp.status_code == 409
    assert "nothing to undo" in resp.json()["detail"].lower()


def test_undo_after_done_returns_409(
    client: TestClient,
    repo,
    clock,
) -> None:
    """ACCEPTED → DONE is not undoable (DONE is post-action; no grace concept)."""
    mom = _make_moment()
    repo.create(mom)
    repo.transition(mom.id, MomentState.ACCEPTED)
    repo.transition(mom.id, MomentState.DONE)
    # Even immediately within the "grace window," DONE is not in
    # _UNDOABLE_STATES.
    clock.t = REF_NOW + 1

    resp = client.post(f"/api/moments/{mom.id}/undo")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 404 — unknown moment id
# ---------------------------------------------------------------------------


def test_undo_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.post("/api/moments/does-not-exist/undo")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Idempotency: double-undo
# ---------------------------------------------------------------------------


def test_undo_twice_within_grace_first_200_second_409(
    client: TestClient,
    repo,
    clock,
) -> None:
    """Undo is not idempotent — the second call sees SUGGESTED and 409s."""
    mom = _make_moment()
    repo.create(mom)
    client.post(f"/api/moments/{mom.id}/accept")

    clock.t = REF_NOW + 1
    first = client.post(f"/api/moments/{mom.id}/undo")
    assert first.status_code == 200

    second = client.post(f"/api/moments/{mom.id}/undo")
    assert second.status_code == 409


# ---------------------------------------------------------------------------
# HTMX path — OOB swap partial
# ---------------------------------------------------------------------------


HX_HEADERS = {"HX-Request": "true"}


def test_undo_htmx_returns_oob_swap_for_now_list(
    client: TestClient,
    repo,
    clock,
) -> None:
    """HTMX caller gets the OOB partial that re-injects the card."""
    mom = _make_moment(insight="ping mom")
    repo.create(mom)
    client.post(f"/api/moments/{mom.id}/accept", headers=HX_HEADERS)

    clock.t = REF_NOW + 1
    resp = client.post(f"/api/moments/{mom.id}/undo", headers=HX_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # OOB target is the Now-tab pending list with afterbegin swap.
    assert 'id="now-list"' in body
    assert 'hx-swap-oob="afterbegin"' in body
    # The bounced card carries the moment_id and the data-moment-undone marker.
    assert f'data-moment-id="{mom.id}"' in body
    assert "data-moment-undone" in body
    # The card itself is the moment_card primitive — has the .moment-card class.
    assert "moment-card" in body
    # No JSON shape leaked through.
    assert "state_history" not in body


def test_undo_non_htmx_returns_json_moment_out(
    client: TestClient,
    repo,
    clock,
) -> None:
    """Curl / iOS clients get the typed MomentOut payload."""
    mom = _make_moment()
    repo.create(mom)
    client.post(f"/api/moments/{mom.id}/accept")

    clock.t = REF_NOW + 1
    resp = client.post(f"/api/moments/{mom.id}/undo")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    out = resp.json()
    assert out["state"] == MomentState.SUGGESTED.value
    assert "state_history" in out


# ---------------------------------------------------------------------------
# Toast button wiring (template-level smoke)
# ---------------------------------------------------------------------------


def test_undo_toast_button_wires_htmx_post(client: TestClient) -> None:
    """The base template's Undo toast handler sets hx-post on the button."""
    body = client.get("/").text
    # The vanilla-JS handler builds the URL via `/api/moments/.../undo`.
    assert "/undo" in body
    # And uses hx-swap=none plus htmx.process so the dynamically-created
    # button picks up its hx-* attributes.
    assert "hx-swap" in body
    assert "htmx.process" in body
