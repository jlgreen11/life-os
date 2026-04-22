# Undo toast + deferred outbox dispatch — design note

**Date:** 2026-04-22
**Status:** draft (unblocks NEXT_TASKS "Undo toast — POST endpoint" and
"Deferred outbox dispatch")
**Author:** autonomous agent, iteration 24
**Review before implementing:** flag in next human sync

---

## Context

`DESIGN.md` shows an Undo toast: every `accept / dismiss / snooze`
surfaces a bottom-right toast for 3 seconds; clicking **Undo** reverses
the action. The client-side plumbing is already in place
(`web/templates/base.html` lines 408–447). The server-side pieces are
not — the current `data-undo` button is a no-op (`toast.remove()`).

Two open questions blocked implementation:

1. The state machine in `core/moment/state.py` has **no reverse edges**.
   `DISMISSED` and `DONE` are terminal; `ACCEPTED` can only go to
   `DONE`; `SNOOZED` can go to `SUGGESTED` or `EXPIRED`. An undo from
   any of {ACCEPTED, DISMISSED, SNOOZED} back to `SUGGESTED` is
   illegal.

2. The outbox spec (engineering plan § "Outbox pattern spec" and
   `storage/schema.py` line 139) has no "not-yet-dispatchable" column —
   every pending row is immediately claim-eligible. A 3 s grace window
   before real-world side effects (send email, create calendar entry)
   needs some form of deferred dispatch.

This note resolves both and specifies boot-recovery semantics.

---

## Scope of undo

Undo applies to the **three terminal-or-terminal-ish user decisions**
made from the Now tab:

| Original transition         | Undo target |
|-----------------------------|-------------|
| `SUGGESTED → ACCEPTED`      | `SUGGESTED` |
| `SUGGESTED → DISMISSED`     | `SUGGESTED` |
| `SUGGESTED → SNOOZED`       | `SUGGESTED` |

Undo does **not** apply to:

- `ACCEPTED → DONE` (post-action completion; no grace concept)
- `SNOOZED → SUGGESTED` (automatic wake-up; user didn't initiate)
- `* → EXPIRED` (time-driven; user didn't initiate)

The grace window is **3 seconds** from the original transition's
`moment_state_history.ts`. Past 3 s, the endpoint returns `410 Gone`.

---

## Decision 1 — state-machine model: reverse edges, not in-flight states

**Chosen:** add reverse edges to `_LEGAL_TRANSITIONS`, gated by the
3-second grace window at the **route layer**.

### Rejected alternative: new in-flight states

Modeling undo as `ACCEPTED_PENDING_DISPATCH → ACCEPTED` (after 3 s
elapses) would require four new states (one per original edge), plus
a background timer to advance them. That bloats the state machine
from 6 to 10 states and introduces a new scheduled-transition
mechanism that duplicates the scheduler. Not worth it for a 3 s
window.

### Legal-transition table changes

```python
# core/moment/state.py
_LEGAL_TRANSITIONS: dict[MomentState | None, set[MomentState]] = {
    None: {MomentState.SUGGESTED},
    MomentState.SUGGESTED: {
        MomentState.ACCEPTED,
        MomentState.DISMISSED,
        MomentState.SNOOZED,
        MomentState.EXPIRED,
    },
    MomentState.ACCEPTED: {MomentState.DONE, MomentState.SUGGESTED},   # + undo
    MomentState.SNOOZED:  {MomentState.SUGGESTED, MomentState.EXPIRED},
    MomentState.DISMISSED: {MomentState.SUGGESTED},                    # + undo
    MomentState.DONE: set(),
    MomentState.EXPIRED: set(),
}
```

### Audit-log semantics

The new edges carry a mandatory `annotation` on
`moment_state_history` with the value `"undo"` so downstream
analytics can filter out bounce events from accept-rate statistics:

```
ts     from_state   to_state    annotation
---    ----------   ---------   ----------
t0     SUGGESTED    ACCEPTED    None         ← user accepted
t0+2s  ACCEPTED     SUGGESTED   "undo"       ← user hit Undo within 3s
```

The `FeedbackWeightStore.record()` call made during the original
`accept` must be **compensated** during undo: the undo endpoint
calls `feedback.record(insight_type, signal=<opposite>)` so the
EWMA doesn't drift on bounced decisions. For MVP, we compensate
by recording the inverse signal (accept→0.0, dismiss→1.0,
snooze→0.5) rather than removing the original entry.

### Grace-window enforcement (route layer)

```python
# api/routes/now.py (new)
@router.post("/api/moments/{moment_id}/undo")
def undo_moment(moment_id: str, ...) -> Response:
    moment = repo.get(moment_id)
    if moment is None:
        raise HTTPException(404)

    last = repo.last_transition(moment_id)  # newest row in state_history
    if last is None or last.to_state == MomentState.SUGGESTED:
        raise HTTPException(409, "nothing to undo")

    if last.to_state not in {ACCEPTED, DISMISSED, SNOOZED}:
        raise HTTPException(409, "state not undoable")

    if now_ts() - last.ts > 3:
        raise HTTPException(410, "grace window expired")

    repo.transition(moment_id,
                    to_state=SUGGESTED,
                    annotation="undo",
                    conn_cb=_cancel_pending_outbox)
    feedback.record(moment.source_insight_type,
                    signal=_inverse_signal[last.to_state])
    return moment_swap_html(moment)  # HTMX swap back in
```

`repo.last_transition(moment_id)` is new — one-line query against
`moment_state_history` ordered by `ts DESC LIMIT 1`. Tests will
cover the four branches (404, 409×2, 410, 200).

---

## Decision 2 — outbox extension: `not_before INTEGER NULL` column

**Chosen:** add `not_before INTEGER NULL` to `outbox`. `claim_batch`
filters rows whose `not_before > now()`. No new scheduler.

### Rejected alternative: scheduler-mediated deferred enqueue

Delaying the `enqueue` call itself (via the Moment scheduler) means
the Moment's state transition and the dispatch enqueue would span
two separate SQLite transactions — losing the atomicity that the
outbox pattern exists to provide. A crash between accept and
scheduler tick would silently drop the dispatch. Rejected.

### Schema change

```sql
ALTER TABLE outbox ADD COLUMN not_before INTEGER NULL;
```

Existing rows get `NULL`, meaning "claim-eligible immediately". The
producer path continues to pass no `not_before` and behaviour is
unchanged.

Index update: the existing `idx_outbox_state_created` still helps
filter `state='pending'`, but a covering index over
`(state, not_before, created_at)` speeds up the new claim query:

```sql
CREATE INDEX idx_outbox_state_notbefore_created
    ON outbox(state, not_before, created_at);
```

(Drop the old index at the same migration — the new one subsumes
it for every existing query.)

### Repo API changes

```python
def enqueue(
    self,
    event_id: str,
    subject: str,
    payload: dict[str, Any] | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    not_before: int | None = None,          # NEW
) -> str: ...

def claim_batch(self, limit: int = 10) -> list[OutboxEntry]:
    # New WHERE clause on the SELECT:
    #   WHERE state='pending'
    #     AND (not_before IS NULL OR not_before <= ?)
    #   ORDER BY created_at ASC, id ASC
    #   LIMIT ?
    ...

def cancel_pending(self, event_id: str, subject: str) -> bool:   # NEW
    """Delete a pending outbox row by (event_id, subject). Used by
    undo during grace window. Returns True if deleted, False if the
    row was already claimed or doesn't exist."""
    ...
```

`cancel_pending` is safe because of the `UNIQUE (event_id, subject)`
constraint — at most one row matches. It deletes **only** rows where
`state='pending'`; a row already in `in_progress` cannot be undone
(matches the "grace window expired" 410 on the state-machine side).

### Accept-flow wiring

```python
# api/routes/now.py, inside accept_moment
with repo._conn:  # same txn as the transition
    repo.transition(moment_id, to_state=ACCEPTED, conn=c, ...)
    outbox.enqueue(
        event_id=f"moment.accept:{moment_id}",
        subject=_dispatch_subject_for(moment),     # e.g. "send_message"
        payload={"moment_id": moment_id, ...},
        conn=c,
        not_before=int(now_fn()) + 3,              # 3 s grace
    )
```

Dismiss and snooze do **not** enqueue an outbox row today — nothing
needs to be dispatched. Undo for those is state-only.

### Undo interaction with outbox

Inside the undo transaction:

```python
outbox.cancel_pending(
    event_id=f"moment.accept:{moment_id}",
    subject=_dispatch_subject_for(moment),
)
# Ignore False — race with claim loop is handled by the grace check:
# if now_ts - last.ts > 3s we already returned 410, so a pending
# row at this point with not_before in the future exists.
```

Worst case: the 3 s grace has expired at the exact moment undo is
processed, the outbox row has just been claimed by the worker, and
`cancel_pending` returns False. The state undo still succeeds but
the dispatch proceeds. This is a 1-in-many-millions race and the
user's recourse is the action-specific reversal (unsend, delete
event) which is out of scope for this note.

---

## Decision 3 — boot recovery during the 3 s grace window

**Chosen:** treat in-flight grace windows as "dispatch will proceed
on next claim tick". No special handling.

### Scenario

1. User clicks Accept at `t=10.000`.
2. Server transitions `SUGGESTED → ACCEPTED`, enqueues outbox row
   with `not_before=13.000`, returns 202.
3. Process crashes at `t=10.500`.
4. Process restarts at `t=45.000`.

### Behaviour on restart

- `OutboxRepository.requeue_in_progress_on_boot()` runs; no rows are
  `in_progress` because the crash was before claim.
- Claim loop wakes at `t=45.500`. The outbox row has
  `state='pending'` and `not_before=13 <= 45`, so it is claimed
  and dispatched.
- The user's browser lost the toast (page reloads after reconnect)
  so the undo affordance is gone — consistent with "undo expired".

### Rationale

The user's intent was **"accept now; dispatch in 3 s unless I
intervene"**. If the process dies within those 3 s, dispatch
proceeds on the next boot, which matches the intent. The 3 s grace
is a UI affordance, not a durable commitment to a delivery window.

### What we explicitly do **not** do

- **Do not re-surface the Moment to the user.** The state is already
  ACCEPTED; reverting to SUGGESTED would violate the user's
  intent.
- **Do not widen the grace window on boot** (e.g. "give the user
  another 3 s after reconnect"). Hard to reason about, and the
  dispatch subject is idempotent anyway (producers key on
  `event_id` per CEO plan § "Producer idempotency").
- **Do not fail the outbox row** on a clock-skew backward jump —
  `not_before` is absolute, and `time.time()` is monotonic-ish
  enough for 3 s.

### Metric surfaced via `/health`

Add a "stale grace" counter that reports `COUNT(*)` of outbox rows
where `state='pending' AND not_before IS NOT NULL AND not_before <
now() - 60`. Non-zero means the claim loop fell behind — unrelated
to undo but free to report.

---

## Test plan

### State machine (`tests/core/moment/test_state.py`)

- Parametrized table test for the three new reverse edges:
  `(ACCEPTED, SUGGESTED)`, `(DISMISSED, SUGGESTED)` — note:
  `SNOOZED → SUGGESTED` is already legal.
- Property test: every legal pair in the table validates; every
  non-legal pair raises `IllegalTransition`.

### Undo route (`tests/api/test_routes_undo.py`, new)

- 200: accept, wait < 3 s, undo → moment back in SUGGESTED,
  feedback compensated, outbox row gone.
- 410: accept, advance clock to `+5 s`, undo → `410 Gone`,
  moment still ACCEPTED, outbox row untouched.
- 409: moment in SUGGESTED (no prior transition) → `409 Conflict`.
- 404: unknown moment id → `404 Not Found`.
- Dismiss undo (no outbox row to cancel) still succeeds and
  returns SUGGESTED.
- Idempotency: calling undo twice within the grace window — first
  returns 200, second returns 409 (already SUGGESTED).

### Outbox (`tests/storage/test_outbox_repo.py`, extend)

- `enqueue(not_before=X)` persists the column.
- `claim_batch` skips rows with `not_before > now()`; picks them up
  once `now >= not_before`.
- `cancel_pending` deletes by `(event_id, subject)`; returns False
  for already-claimed or missing rows.
- `requeue_in_progress_on_boot` does **not** touch `not_before`.
- Purge ignores `not_before` (only state + updated_at matter).

### Boot recovery (`tests/core/moment/test_scheduler.py` or new
file)

- Enqueue with `not_before = now + 3`, advance clock by 10 s (simulate
  crash + restart), call `claim_batch` — row is claimed.
- Enqueue with `not_before = now + 60`, call `claim_batch` — row is
  **not** claimed.

---

## Out of scope (for the two task bodies downstream)

- Undo UI polish: chord keys (Cmd+Z), focus return to the swapped-in
  card, multi-undo stack.
- Per-producer dispatch subjects beyond the mapping assumed here
  (`_dispatch_subject_for`); the initial mapping is two branches
  (communication producers → `"send_message"`, calendar →
  `"create_event"`) plus a generic fallback. Flesh out with
  producers in Week 11.
- Analytics rollup that filters bounce events — add after the
  `annotation="undo"` column is populated in production.
