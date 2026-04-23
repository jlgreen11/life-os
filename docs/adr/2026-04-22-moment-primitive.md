# ADR 002: Moment as first-class primitive

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

v1 fanned user-facing output across four different shapes:

- `tasks` — persisted to-dos the user accepted
- `notifications` — transient UI banners surfaced by the notification_manager
- `predictions` — predicted next actions emitted by the prediction_engine
- `briefings` — LLM-summarized daily digests

Each shape had its own table, its own service, its own acceptance hook,
its own expiry rules, and its own debug surface. A single user decision
("dismiss this suggestion") had four different codepaths depending on
which shape produced it, so:

- the acceptance-rate signal could not be aggregated cleanly (four
  denominators that didn't line up);
- the feedback loop back to producers was plumbed four times, each in a
  subtly different way (see v1 `feedback_collector` vs the per-service
  dismiss handlers);
- debugging "why did the system tell me X?" required chasing four
  different histories;
- any new insight producer had to pick one of the four shapes, and the
  choice was always a compromise.

The CEO plan for v2 (`2026-04-21-life-os-rewrite-mvp.md`, § "The Moment
primitive") proposed collapsing all four into a single entity:

> A Moment is *time + context + evidence + proposed action + state*.
> Everything the system ever wants the user to look at or do is a Moment.

## Decision

**A single `moments` table and `Moment` dataclass replace all four v1
user-facing shapes. Every insight producer in v2 emits Moments; the API,
UI, and scheduler read and mutate Moments.**

Concretely:

- `moments(id, source_insight_type, evidence_hash, state, scheduled_for,
  expires_at, payload, proposed_action, created_at, updated_at)` is the
  single table for all user-facing output.
  `UNIQUE(source_insight_type, evidence_hash)` enforces idempotent
  producers. DDL in `storage/schema.py`.
- `MomentState` is a typed enum (`suggested`, `accepted`, `dismissed`,
  `snoozed`, `done`, `expired`) with a legal-transition table in
  `core/moment/state.py::_LEGAL_TRANSITIONS`. Illegal transitions raise
  `IllegalTransition` and roll back the enclosing SQLite transaction.
- Every transition appends a row to `moment_state_history` in the same
  transaction, so the causal chain for every Moment is queryable.
- Accept / dismiss / snooze / edit all become Moment state transitions
  on the same entity. The API exposes them as four routes on a single
  resource (`POST /api/moments/{id}/{accept|dismiss|snooze|edit}`).
- Producers implement a common ABC (`core/moment/producer.py`) and are
  dispatched by `MomentEngine` uniformly — no per-shape wiring.

## Consequences

### Positive

- **One acceptance denominator.** Accept-rate math is a single SQL
  `GROUP BY source_insight_type` instead of four-table UNION ALL with
  adapters per shape.
- **Feedback loop unification.** `feedback_weights` EWMA per
  `insight_type` is wired once. Producers compete in a single
  leaderboard; the CEO-plan exit criterion ("≥3 types at ≥60% for 2
  consecutive weeks") is meaningful because the metric is uniform.
- **Debugging.** "Why did the system tell me X?" has one trail:
  `moments` row → `moment_state_history` → referenced `events` rows via
  the evidence payload. The four-way forensics from v1 are gone.
- **Producer ergonomics.** Adding a new insight type is a producer file
  in `producers/` and a registration — no new table, no new API, no new
  UI shape.
- **Idempotency is structural.** `UNIQUE(source_insight_type, evidence_hash)`
  means a producer that re-runs the same evidence cannot double-surface.
  v1 had per-service guards with drift; v2 has one constraint.

### Negative

- **Payload is denormalized.** `moments.payload` and
  `moments.proposed_action` are JSON blobs with per-type shape. The
  database cannot enforce shape; Pydantic schemas in `api/schemas.py` do
  it at the API boundary, and producer-side dataclasses do it at write
  time. A future schema-change for one insight type risks mixing shapes
  across rows; migration must consider all `source_insight_type` values.
- **State machine surface grows.** Six states × six states = 36 legal /
  illegal pairs to keep straight. We mitigate with the single locked
  `_LEGAL_TRANSITIONS` table + property-based transition tests, but a
  future state addition requires updating the table + tests + UI + API.
- **No separation of concerns between "suggestion" and "commitment".**
  v1's `tasks` table was conceptually distinct from `notifications`.
  v2 represents an accepted Moment as `state=accepted` on the same
  row. Downstream code must filter by state everywhere; a missed filter
  shows a pending suggestion where an accepted commitment was expected.

## Alternatives considered

1. **Keep the four v1 shapes, consolidate only the acceptance signal.**
   *Rejected* — the four shapes' acceptance semantics are genuinely
   different (a "dismissed notification" ≠ a "deleted task"), so any
   unified signal layer would hide the very mismatches we wanted to
   eliminate. Cheaper to change, but does not solve the root problem.

2. **Moment as an interface with per-shape concrete tables.**
   *Rejected* — defeats the unification. Polymorphism at the storage
   layer keeps per-shape surprises alive across joins and acceptance
   math.

3. **Moment as append-only event, never UPDATEd.**
   *Rejected* — the state transitions are the user's point of contact
   with the system. Modeling them as append-only would move the current
   state into a derived view, making the scheduler and UI read
   expensive. v2 keeps `moments` row UPDATEable by state alone, with
   all prior state writes preserved in `moment_state_history`.

## Follow-up

- ADR 001 (`feedback_events` table) documents the companion decision
  about where v1 feedback history lands in v2.
- The producer ABC and registry live in `core/moment/producer.py`; any
  new producer is a file in `producers/` that inherits from it.
