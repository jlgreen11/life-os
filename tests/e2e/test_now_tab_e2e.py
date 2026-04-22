"""End-to-end browser tests for the Now-tab critical-path flows.

Five paths from DESIGN.md § "test plan" the autonomous-rewrite plan
calls out as ship-blockers:

1. **Accept moves card out** — clicking Accept on a SUGGESTED card
   removes it from ``#now-list`` (HTMX outerHTML swap).
2. **Dismiss moves card out** — clicking Dismiss removes the card.
3. **Snooze chip moves card out** — opening the snooze popover and
   picking the "1h" chip POSTs ``snooze_until`` and removes the card.
4. **WS push appends a new card** — a server-side broadcast of the
   ``ws_moment_new`` partial slides a new card into ``#now-list`` via
   the ``hx-swap-oob="afterbegin"`` mechanism.
5. **WS push removes accepted card via OOB** — broadcasting the
   ``ws_moment_done`` partial drops the card straight into the
   ``#done-today-list`` block.

Driver discovery
----------------
Performed at module import time in :mod:`tests.e2e.conftest`:

    Playwright preferred → Selenium fallback → ``pytest.skip`` + warn.

Each test is driver-agnostic by going through the :class:`_Page`
adapter below. The adapter exposes the smallest surface that covers
the five flows (goto / count / click / wait_for / wait_until_gone)
so the same test body runs unchanged whether Playwright or Selenium
is on the host.

Per-test cleanup
----------------
The browser/context is built fresh for every test (no session-level
fixture) so a flaky test cannot leak DOM, cookies, or live WebSocket
connections to the next one. The cost is ~1 s of context startup per
test which is dwarfed by the existing pytest setup overhead.

References
----------
- DESIGN.md § "test plan" for the five paths above.
- ``api/routes/now.py`` for the action endpoints under exercise.
- ``core/moment/broadcaster.py`` for the WS push surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from core.moment.types import Action, ActionKind, InsightType, Moment, MomentState
from tests.e2e.conftest import DRIVER, REF_NOW
from web.rendering import render

pytestmark = pytest.mark.skipif(
    DRIVER is None,
    reason="playwright/selenium not installed (see tests/e2e/conftest.py)",
)


# ---------------------------------------------------------------------------
# Driver-agnostic page adapter
# ---------------------------------------------------------------------------


class _Page:
    """Minimal browser page surface used by the five Now-tab E2E tests.

    Wraps a Playwright ``Page`` or a Selenium ``WebDriver``. Only the
    operations the tests actually need are exposed:

    - :meth:`goto`              — full-page navigation, waits for load.
    - :meth:`count`             — number of elements matching a CSS selector.
    - :meth:`click`             — click the first element matching a selector.
    - :meth:`wait_for`          — block until ``count(selector) >= 1``.
    - :meth:`wait_until_gone`   — block until ``count(selector) == 0``.
    - :meth:`text`              — text content of the first match (debug aid).

    Reasonable timeouts (``DEFAULT_TIMEOUT_S``) keep CI fast while still
    leaving room for HTMX swaps + WS round trips on a loaded laptop.
    """

    DEFAULT_TIMEOUT_S = 5.0
    POLL_INTERVAL_S = 0.05

    def __init__(self, kind: str, native: Any) -> None:
        self.kind = kind
        self._native = native

    def goto(self, url: str) -> None:
        if self.kind == "playwright":
            self._native.goto(url, wait_until="networkidle")
        else:
            self._native.get(url)

    def count(self, selector: str) -> int:
        if self.kind == "playwright":
            return self._native.locator(selector).count()
        from selenium.webdriver.common.by import By  # type: ignore[import-not-found]

        return len(self._native.find_elements(By.CSS_SELECTOR, selector))

    def click(self, selector: str) -> None:
        if self.kind == "playwright":
            self._native.locator(selector).first.click()
            return
        from selenium.webdriver.common.by import By  # type: ignore[import-not-found]

        self._native.find_element(By.CSS_SELECTOR, selector).click()

    def text(self, selector: str) -> str:
        if self.kind == "playwright":
            return self._native.locator(selector).first.inner_text()
        from selenium.webdriver.common.by import By  # type: ignore[import-not-found]

        return self._native.find_element(By.CSS_SELECTOR, selector).text

    def wait_for(self, selector: str, timeout: float | None = None) -> None:
        self._poll(lambda: self.count(selector) >= 1, timeout, f"selector {selector!r} did not appear")

    def wait_until_gone(self, selector: str, timeout: float | None = None) -> None:
        self._poll(lambda: self.count(selector) == 0, timeout, f"selector {selector!r} did not disappear")

    def _poll(self, predicate: Any, timeout: float | None, msg: str) -> None:
        deadline = time.time() + (timeout if timeout is not None else self.DEFAULT_TIMEOUT_S)
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(self.POLL_INTERVAL_S)
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Browser/page fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def page() -> Iterator[_Page]:
    """Build a one-shot browser/context/page for the active driver.

    Imports happen here (not at module top) so a host with neither
    driver still imports this file cleanly and the ``pytestmark``
    skipif fires before any browser code runs.
    """
    if DRIVER == "playwright":
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                pg = context.new_page()
                try:
                    yield _Page("playwright", pg)
                finally:
                    context.close()
            finally:
                browser.close()
        return

    if DRIVER == "selenium":
        from selenium import webdriver  # type: ignore[import-not-found]
        from selenium.webdriver.chrome.options import Options  # type: ignore[import-not-found]

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(10)
        try:
            yield _Page("selenium", driver)
        finally:
            driver.quit()
        return

    pytest.skip("no browser driver available")  # pragma: no cover — guarded by pytestmark.


# ---------------------------------------------------------------------------
# Moment-construction helper
# ---------------------------------------------------------------------------


def _make_moment(
    *,
    insight: str = "ping your sister",
    insight_type: InsightType = InsightType.CADENCE,
    action_kind: ActionKind = ActionKind.DRAFT_MESSAGE,
    body: str | None = "Hey — been a minute. How are you?",
    confidence: float = 0.8,
    moment_id: str | None = None,
    state: MomentState = MomentState.SUGGESTED,
) -> Moment:
    params: dict[str, str] = {}
    if body is not None:
        params["body"] = body
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=REF_NOW,
        expires_at=REF_NOW + 3 * 24 * 3600,
        insight=insight,
        evidence_hash=f"hash-{uuid.uuid4().hex[:8]}",
        proposed_action=Action(kind=action_kind, params=params),
        source_insight_type=insight_type,
        evidence=["evt-1", "evt-2"],
        state=state,
        confidence=confidence,
    )


def _broadcast(broadcaster: Any, html: str) -> None:
    """Hop onto the broadcaster's loop to deliver one HTML payload.

    The WS endpoint owns the asyncio event loop the broadcaster
    captured at first ``register``; from a sync test we use
    :func:`asyncio.run_coroutine_threadsafe` (which is exactly what
    :meth:`MomentBroadcaster.notify_sync` does) and block on the
    resulting future so the broadcast lands before we assert.
    """
    if broadcaster.loop is None:
        # No client has connected yet — wait briefly then retry once.
        time.sleep(0.2)
    if broadcaster.loop is None:
        raise AssertionError("broadcaster has no loop (no WS client ever connected)")
    fut = asyncio.run_coroutine_threadsafe(broadcaster.broadcast(html), broadcaster.loop)
    fut.result(timeout=5)


# ---------------------------------------------------------------------------
# 1. Accept moves the card out of #now-list
# ---------------------------------------------------------------------------


def test_accept_removes_card_from_now_list(page: _Page, repo: Any, server: Any) -> None:
    """DESIGN.md "test plan" #1 — accept clears the card via HTMX swap."""
    mom = _make_moment(insight="water the plants")
    repo.create(mom)

    page.goto(server.base_url + "/")
    sel = f'.moment-card[data-moment-id="{mom.id}"]'
    page.wait_for(sel)

    page.click(f'{sel} [data-action="accept"]')
    page.wait_until_gone(sel)
    # Repo confirms the underlying state machine ran end-to-end.
    assert repo.get(mom.id).state == MomentState.ACCEPTED


# ---------------------------------------------------------------------------
# 2. Dismiss moves the card out of #now-list
# ---------------------------------------------------------------------------


def test_dismiss_removes_card_from_now_list(page: _Page, repo: Any, server: Any) -> None:
    """DESIGN.md "test plan" #2 — dismiss clears the card via HTMX swap."""
    mom = _make_moment(insight="archive newsletter")
    repo.create(mom)

    page.goto(server.base_url + "/")
    sel = f'.moment-card[data-moment-id="{mom.id}"]'
    page.wait_for(sel)

    page.click(f'{sel} [data-action="dismiss"]')
    page.wait_until_gone(sel)
    assert repo.get(mom.id).state == MomentState.DISMISSED


# ---------------------------------------------------------------------------
# 3. Snooze chip sets snooze_until and removes from pending
# ---------------------------------------------------------------------------


def test_snooze_chip_removes_card_and_sets_snooze_until(page: _Page, repo: Any, server: Any) -> None:
    """DESIGN.md "test plan" #3 — Snooze "1h" chip POSTs and clears."""
    mom = _make_moment(insight="check on grandma")
    repo.create(mom)

    page.goto(server.base_url + "/")
    sel = f'.moment-card[data-moment-id="{mom.id}"]'
    page.wait_for(sel)

    # Open the snooze popover, then click the 1h preset chip.
    page.click(f"{sel} [data-snooze-popover-trigger]")
    chip_sel = f'{sel} [data-snooze-preset="1h"]'
    page.wait_for(chip_sel)
    page.click(chip_sel)

    page.wait_until_gone(sel)
    persisted = repo.get(mom.id)
    assert persisted.state == MomentState.SNOOZED
    # `snooze_until` is set to roughly now+1h (server clock); just
    # assert it landed and is in the future relative to creation.
    assert persisted.snooze_until is not None
    assert persisted.snooze_until > mom.created_at


# ---------------------------------------------------------------------------
# 4. WS push appends a new card via hx-swap-oob="afterbegin"
# ---------------------------------------------------------------------------


def test_ws_push_appends_new_card_to_now_list(page: _Page, repo: Any, server: Any, broadcaster: Any) -> None:
    """DESIGN.md "test plan" #4 — WS broadcast inserts a card live."""
    page.goto(server.base_url + "/")
    # WebSocket connect is async; give the htmx-ext-ws extension a
    # moment to dial /ws and register on the broadcaster.
    deadline = time.time() + 5.0
    while time.time() < deadline and len(broadcaster) == 0:
        time.sleep(0.05)
    assert len(broadcaster) >= 1, "browser never connected the WebSocket"

    pushed = _make_moment(insight="reply to mike")
    repo.create(pushed)
    html = render("partials/ws_moment_new.html", {"moment": pushed})
    _broadcast(broadcaster, html)

    sel = f'.moment-card[data-moment-id="{pushed.id}"]'
    page.wait_for(sel)
    assert "reply to mike" in page.text(sel)


# ---------------------------------------------------------------------------
# 5. WS push removes the accepted card by appending to #done-today-list (OOB)
# ---------------------------------------------------------------------------


def test_ws_push_drops_accepted_card_into_done_today(page: _Page, repo: Any, server: Any, broadcaster: Any) -> None:
    """DESIGN.md "test plan" #5 — terminal-state WS partial slots into DONE TODAY."""
    mom = _make_moment(insight="reply to alex")
    repo.create(mom)

    page.goto(server.base_url + "/")
    sel = f'.moment-card[data-moment-id="{mom.id}"]'
    page.wait_for(sel)

    deadline = time.time() + 5.0
    while time.time() < deadline and len(broadcaster) == 0:
        time.sleep(0.05)
    assert len(broadcaster) >= 1, "browser never connected the WebSocket"

    # Drive the state machine first so the partial reflects ACCEPTED;
    # then broadcast the DONE partial as if a sibling client had been
    # the one to accept it.
    accepted = repo.transition(mom.id, MomentState.ACCEPTED, annotation="ws e2e")
    html = render("partials/ws_moment_done.html", {"moment": accepted})
    _broadcast(broadcaster, html)

    done_sel = f'#done-today-list [data-moment-id="{mom.id}"]'
    page.wait_for(done_sel)
    # The DONE partial intentionally renders the row text in the same
    # shape as the static template so the live insert is indistinguishable
    # from a page reload — sanity-check a piece of that contract.
    assert "reply to alex" in page.text(done_sel)


# Quiet down the unused-import warning for the contextlib import — kept
# in case a future test wants to suppress noisy teardown errors from
# the broadcaster's drop-on-send branch.
_ = contextlib
