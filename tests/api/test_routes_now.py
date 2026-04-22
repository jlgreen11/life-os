"""Tests for :mod:`api.routes.now` — Now tab + 4 moment-action endpoints.

Every test spins up a fresh in-memory SQLite with the full v2 schema,
wires a real :class:`~storage.repos.moments.MomentRepository` + a real
:class:`~core.moment.feedback_weight.FeedbackWeightStore` onto a dummy
``LifeOS`` double, and hits the app via :class:`fastapi.testclient.TestClient`.
No mocks of the storage layer — same pattern the v1 ``tests/conftest.py``
established.

Coverage (per NEXT_TASKS.md Week 8 + Week 9 acceptance):

- ``GET /api/now`` returns ``{pending, scheduled, done}`` in the schema
  shape, each capped at 20/10/10.
- Each action endpoint round-trips through the state machine and the
  feedback-weight EWMA.
- Invalid transitions (e.g. snooze a DISMISSED Moment) return 409.
- Unknown ``moment_id`` returns 404 on every mutation endpoint.
- ``snooze`` without ``snooze_until`` returns 422.
- ``edit`` updates ``proposed_action.params`` without moving state.
- 503 when ``moment_repo`` is not wired onto life_os.
- HTMX (``HX-Request: true``) callers get a Moment-card swap partial
  with an ``HX-Trigger`` header for the Undo toast; non-HTMX callers
  keep getting the legacy JSON ``MomentOut`` shape.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# GET /api/moments/{id}/evidence — HTMX reveal partial
# ---------------------------------------------------------------------------


def test_get_evidence_renders_partial_html(client: TestClient, repo) -> None:
    """The endpoint returns the evidence-list partial as text/html."""
    mom = _make_moment()
    repo.create(mom)

    resp = client.get(f"/api/moments/{mom.id}/evidence")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert 'data-slot="evidence-list"' in body
    # The default fixture moment carries two evidence refs.
    assert "evt-1" in body
    assert "evt-2" in body


def test_get_evidence_empty_state(client: TestClient, repo) -> None:
    """A moment with no evidence still 200s and shows the empty row."""
    mom = _make_moment()
    # Strip evidence after construction; the repo writes whatever we hand it.
    mom.evidence.clear()
    repo.create(mom)

    resp = client.get(f"/api/moments/{mom.id}/evidence")
    assert resp.status_code == 200
    assert 'data-slot="evidence-empty"' in resp.text
    assert "No source events recorded." in resp.text


def test_get_evidence_unknown_id_returns_404(client: TestClient) -> None:
    resp = client.get("/api/moments/missing/evidence")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET / — Now-tab page
# ---------------------------------------------------------------------------


def test_now_page_renders_html(client: TestClient) -> None:
    """The home route returns full HTML with all three named sections."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "<!DOCTYPE html>" in body
    assert 'data-section="now"' in body
    assert 'data-section="up-next"' in body
    assert 'data-section="done-today"' in body


def test_now_page_includes_pending_moment_card(client: TestClient, repo) -> None:
    """A pending moment shows up as a Moment card in the page."""
    mom = _make_moment(insight="water the plants", confidence=0.9)
    repo.create(mom)

    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "moment-card" in body
    assert "water the plants" in body
    assert f'data-moment-id="{mom.id}"' in body
    # And the page sits under the Now tab in the nav.
    assert 'data-active-tab="now"' in body


def test_now_page_503_when_repo_missing() -> None:
    """Half-constructed life_os surfaces the same 503 the JSON route does."""
    app = create_app(DummyLifeOS(moment_repo=None, feedback_weight_store=None))
    c = TestClient(app)
    resp = c.get("/")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# HTMX wiring (Week 9 task 3): accept / dismiss / snooze swap response
# ---------------------------------------------------------------------------


HX_HEADERS = {"HX-Request": "true"}


def _hx_trigger_payload(resp) -> dict:
    """Return the parsed ``HX-Trigger`` JSON payload, or fail with context."""
    assert "hx-trigger" in {k.lower() for k in resp.headers.keys()}, resp.headers
    raw = resp.headers["hx-trigger"]
    return json.loads(raw)


def test_accept_htmx_returns_html_swap_with_next_pending(client: TestClient, repo) -> None:
    """HTMX accept replaces the card with the next-highest-confidence pending."""
    high = _make_moment(confidence=0.9, evidence_hash="h-high", insight="ping mom")
    mid = _make_moment(confidence=0.5, evidence_hash="h-mid", insight="water plants")
    repo.create(high)
    repo.create(mid)

    resp = client.post(f"/api/moments/{high.id}/accept", headers=HX_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # The swap target is the next pending Moment card (the one we did
    # NOT just accept).
    assert "moment-card" in body
    assert f'data-moment-id="{mid.id}"' in body
    assert f'data-moment-id="{high.id}"' not in body
    # And no JSON body shape leaked through.
    assert "state_history" not in body


def test_accept_htmx_empty_queue_returns_placeholder(client: TestClient, repo) -> None:
    """No more pending → swap returns the empty-card placeholder."""
    only = _make_moment(confidence=0.9, evidence_hash="h-only")
    repo.create(only)

    resp = client.post(f"/api/moments/{only.id}/accept", headers=HX_HEADERS)
    assert resp.status_code == 200
    body = resp.text
    assert 'data-moment-empty="true"' in body
    assert "moment-card--empty" in body
    # The placeholder must NOT carry the acted-on id (the parent <li>
    # handles the slot; the card itself is gone).
    assert f'data-moment-id="{only.id}"' not in body


def test_accept_htmx_sets_undo_trigger_header(client: TestClient, repo) -> None:
    """HX-Trigger fires the Undo toast with moment_id + previous state."""
    mom = _make_moment(confidence=0.9)
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/accept", headers=HX_HEADERS)
    payload = _hx_trigger_payload(resp)
    assert "lifeos:moment-acted" in payload
    detail = payload["lifeos:moment-acted"]
    assert detail["momentId"] == mom.id
    assert detail["previousState"] == MomentState.SUGGESTED.value
    assert detail["newState"] == MomentState.ACCEPTED.value
    assert detail["action"] == "accepted"


def test_dismiss_htmx_returns_html_with_trigger(client: TestClient, repo) -> None:
    high = _make_moment(confidence=0.9, evidence_hash="h-high", insight="ping mom")
    low = _make_moment(confidence=0.3, evidence_hash="h-low", insight="archive that")
    repo.create(high)
    repo.create(low)

    resp = client.post(f"/api/moments/{high.id}/dismiss", headers=HX_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert f'data-moment-id="{low.id}"' in resp.text
    detail = _hx_trigger_payload(resp)["lifeos:moment-acted"]
    assert detail["action"] == "dismissed"


def test_snooze_htmx_returns_html_with_trigger(client: TestClient, repo) -> None:
    mom = _make_moment(confidence=0.9)
    repo.create(mom)

    resp = client.post(
        f"/api/moments/{mom.id}/snooze",
        json={"snooze_until": REF_NOW + 3600},
        headers=HX_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    detail = _hx_trigger_payload(resp)["lifeos:moment-acted"]
    assert detail["momentId"] == mom.id
    assert detail["action"] == "snoozed"
    assert detail["newState"] == MomentState.SNOOZED.value


def test_non_htmx_accept_still_returns_json_moment_out(client: TestClient, repo) -> None:
    """Curl / iOS clients without HX-Request still get the typed JSON shape."""
    mom = _make_moment(confidence=0.9)
    repo.create(mom)

    resp = client.post(f"/api/moments/{mom.id}/accept")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    out = resp.json()
    # MomentOut shape is preserved.
    assert out["id"] == mom.id
    assert out["state"] == MomentState.ACCEPTED.value
    assert "state_history" in out
    # No HX-Trigger leaks into the JSON path.
    assert "hx-trigger" not in {k.lower() for k in resp.headers.keys()}


def test_htmx_action_records_feedback_same_as_json_path(client: TestClient, repo, feedback) -> None:
    """HTMX path must not skip the EWMA update."""
    mom = _make_moment(insight_type=InsightType.CADENCE, confidence=0.9)
    repo.create(mom)

    client.post(f"/api/moments/{mom.id}/accept", headers=HX_HEADERS)

    weight, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 1
    expected = ALPHA * 1.0 + (1 - ALPHA) * DEFAULT_WEIGHT
    assert weight == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Snooze popover + Undo toast — template-level wiring
# ---------------------------------------------------------------------------


def test_moment_card_renders_snooze_popover_with_chip_row(client: TestClient, repo) -> None:
    """Each rendered Moment card carries the [1h][3h][Tonight][Tomorrow][3d][Custom]
    chip popover wired with HTMX + json-enc."""
    mom = _make_moment(insight="ping mom", confidence=0.9)
    repo.create(mom)

    body = client.get("/").text
    # Popover container present and hidden by default.
    assert f'id="snooze-popover-{mom.id}"' in body
    assert "data-snooze-popover" in body
    # All five preset chips + the Custom escape hatch.
    for preset in ("1h", "3h", "tonight", "tomorrow", "3d", "custom"):
        assert f'data-snooze-preset="{preset}"' in body
    # Chips POST via HTMX with the json-enc extension and the swap
    # target is the parent card.
    assert 'hx-ext="json-enc"' in body
    assert f'hx-post="/api/moments/{mom.id}/snooze"' in body
    assert "lifeos.snoozeChipUntil" in body
    assert 'hx-target="closest .moment-card"' in body


def test_base_template_includes_undo_toast_handler(client: TestClient) -> None:
    """The vanilla-JS Undo toast handler must be wired in base.html so HTMX
    swaps trigger it without per-page boilerplate."""
    body = client.get("/").text
    # json-enc extension is loaded so chip JSON encoding works.
    assert "htmx-ext-json-enc" in body
    # Toast region from the original task is still present...
    assert 'id="toast-region"' in body
    # ...and the handler that renders Undo toasts is wired to the
    # `lifeos:moment-acted` event (fired via HX-Trigger from each
    # action endpoint).
    assert "lifeos:moment-acted" in body
    assert "showUndoToast" in body
    # 3-second auto-dismiss (per DESIGN.md and task spec).
    assert "3000" in body


# ---------------------------------------------------------------------------
# Inline draft editor (Week 11 task 1) — render-time DOM + edit→accept chain
# ---------------------------------------------------------------------------


def test_moment_card_renders_draft_trigger_and_hidden_editor(client: TestClient, repo) -> None:
    """The draft block is promoted to a clickable trigger and a sibling
    <textarea data-slot="draft-editor" hidden> lives alongside it. The
    textarea is hidden by default; clicking the trigger swaps them in JS."""
    mom = _make_moment(insight="ping mom", draft="hey — been a minute")
    repo.create(mom)

    body = client.get("/").text

    # Trigger: role=button + tabindex + onclick hooks into lifeos JS.
    assert 'data-slot="draft"' in body
    assert 'role="button"' in body
    assert 'onclick="lifeos.activateDraftEditor(this)"' in body
    # Enter / Space keyboard activation wired on the trigger.
    assert "onkeydown=" in body
    assert "event.key===&#39;Enter&#39;" in body or "event.key==='Enter'" in body

    # Textarea exists, carries the moment id, starts hidden.
    assert 'data-slot="draft-editor"' in body
    assert f'data-moment-id="{mom.id}"' in body
    # Must be present with the hidden attribute so tab order is stable
    # (not `display:none`, not absent — we toggle `hidden` from JS).
    assert "hidden>" in body
    # Keyboard & input handlers for Esc cancel + Cmd/Ctrl+Enter commit.
    assert "lifeos.handleDraftEditorKey(event)" in body
    assert "lifeos.autosizeDraftEditor(this)" in body


def test_moment_card_draft_editor_preserves_body_text(client: TestClient, repo) -> None:
    """The textarea's initial content is the draft body so users start
    editing from the existing text, not a blank box."""
    draft = "Hey, been a while — want to grab coffee?"
    mom = _make_moment(draft=draft)
    repo.create(mom)

    body = client.get("/").text
    # Body text appears both in the clickable <div> and inside the textarea.
    assert draft in body
    # Textarea block contains the draft between its tags.
    import re

    match = re.search(
        r'<textarea[^>]*data-slot="draft-editor"[^>]*>([^<]*)</textarea>',
        body,
    )
    assert match is not None
    assert match.group(1) == draft


def test_edit_button_wires_into_draft_editor_activator(client: TestClient, repo) -> None:
    """The existing Edit ghost button now calls the same activator so
    mouse users who click Edit get the same textarea as draft-clickers."""
    mom = _make_moment()
    repo.create(mom)

    body = client.get("/").text
    # The Edit button still exists (tab navigation preserved) and now
    # triggers the inline editor rather than being a dead surface.
    assert 'data-action="edit"' in body
    # There is an onclick on the Edit button that calls activateDraftEditor.
    import re

    edit_btn = re.search(
        r'<button[^>]*data-action="edit"[^>]*>',
        body,
    )
    assert edit_btn is not None
    assert "lifeos.activateDraftEditor(this)" in edit_btn.group(0)


def test_base_template_includes_inline_draft_editor_handlers(client: TestClient) -> None:
    """The four vanilla-JS helpers that power the inline editor must be
    defined on `window.lifeos` so any Moment card can reach them."""
    body = client.get("/").text
    for fn in (
        "activateDraftEditor",
        "deactivateDraftEditor",
        "handleDraftEditorKey",
        "autosizeDraftEditor",
        "commitDraftEditor",
    ):
        assert f"lifeos.{fn}" in body, f"missing helper: {fn}"
    # Escape cancels + Cmd/Ctrl+Enter commits.
    assert "event.key === 'Escape'" in body
    assert "metaKey" in body and "ctrlKey" in body
    # The chain calls /edit then clicks the Accept button (which rides
    # the existing HTMX wiring).
    assert "/api/moments/" in body
    assert "'[data-action=\"accept\"]'" in body or '"[data-action=\\"accept\\"]"' in body


def test_edit_then_accept_chain_persists_new_body_on_json_path(
    client: TestClient,
    repo,
    feedback,
) -> None:
    """End-to-end verification of the edit→accept chain on the JSON path.
    Mirrors what the JS handler does: POST /edit then POST /accept. The
    accept response must reflect the edited body, proving the chain
    preserves the rewrite."""
    mom = _make_moment(insight_type=InsightType.CADENCE, draft="original draft body")
    repo.create(mom)

    new_body = "rewritten draft body for chained commit"

    edit_resp = client.post(
        f"/api/moments/{mom.id}/edit",
        json={"action_params": {"body": new_body}},
    )
    assert edit_resp.status_code == 200
    edited = edit_resp.json()
    # Edit is not a transition — state stays SUGGESTED.
    assert edited["state"] == MomentState.SUGGESTED.value
    assert edited["proposed_action"]["params"] == {"body": new_body}

    accept_resp = client.post(f"/api/moments/{mom.id}/accept")
    assert accept_resp.status_code == 200
    accepted = accept_resp.json()
    assert accepted["state"] == MomentState.ACCEPTED.value
    # The rewritten body survives the state transition — the accept
    # handler does not rehydrate from stale in-memory state.
    assert accepted["proposed_action"]["params"] == {"body": new_body}
    # History gains the ACCEPTED row appended to the SUGGESTED row.
    states = [h["to_state"] for h in accepted["state_history"]]
    assert states == [MomentState.SUGGESTED.value, MomentState.ACCEPTED.value]
    # And the feedback weight only moved on accept, not on edit.
    _, decisions = feedback.get(InsightType.CADENCE.value)
    assert decisions == 1
