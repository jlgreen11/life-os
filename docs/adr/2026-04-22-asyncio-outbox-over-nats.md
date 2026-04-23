# ADR 004: Drop NATS; in-process asyncio bus + transactional outbox

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

v1 ran a NATS JetStream broker in a docker-compose sidecar. Every
connector → service and service → service interaction flowed through
it. The premise was that separating producers and consumers behind a
durable broker would let the architecture scale horizontally if user
count ever grew past one.

In practice, after 6 months of production on a single Mac Mini with
exactly one user:

- **Ops burden is real.** NATS needed restart on two occasions when
  a storage consumer lagged past its ack deadline; each restart meant
  reconciling an unknown-drift number of un-acked messages.
- **Transaction boundary is wrong.** The v1 pattern was "write to
  SQLite, publish to NATS". A crash between the two rows produced
  either lost publishes or duplicate publishes depending on order.
  The continuous-improvement agent filed 7 PRs in Q1 2026 related to
  this race; none landed a clean fix because the underlying shape is
  a distributed-transaction problem and SQLite + NATS do not share
  a transaction.
- **Scale justification never materialized.** v1 has one user. v2 is
  scoped as single-user (Phase 1) and small-family (Phase 2 ceiling).
  Horizontal scaling across processes was never going to earn the
  cost of the broker.
- **Debug and testing cost.** Every test that involves a cross-service
  event requires either a real NATS fixture or a mock that drifts from
  the real broker. The test suite paid for that complexity on every
  run.

The engineering plan (`docs/plans/2026-04-21-v2-rewrite-plan.md`
§ "Outbox pattern spec") spelled out a replacement: an in-process
asyncio message bus backed by a transactional outbox table for
side-effecting events that must escape the process (send a message,
write to calendar, etc.).

## Decision

**v2 has no external broker. In-process asyncio is the bus; a
transactional outbox table handles durable side effects.**

Concretely:

- `core/moment/broadcaster.py` is a pub/sub implemented on asyncio
  Queues. WebSocket clients subscribe; the MomentEngine publishes
  state changes. In-process only. Restart loses in-flight messages
  that had not reached the outbox — by design; they were not durable.
- `storage/repos/outbox.py` + `outbox` table (DDL in
  `storage/schema.py`) provide the durable seam. Any side-effect that
  must survive a crash (`send_message.v1`, `create_calendar_entry.v1`,
  …) is enqueued in the *same SQLite transaction* that writes the
  Moment state change. Enqueue is idempotent on `(event_id, subject)`.
- A dispatcher coroutine claims `state='pending'` rows in batches
  (BEGIN IMMEDIATE + UPDATE to `in_progress`), delivers the effect,
  and marks `done` or re-queues on failure. At `retry_count ≥ 5`
  the row transitions to `dead` and surfaces via `/api/health`.
- `requeue_in_progress_on_boot()` runs at startup and flips any
  `in_progress` rows back to `pending`, covering the "process died
  mid-delivery" case.
- No docker-compose file ships with v2. No NATS client dependency.

## Consequences

### Positive

- **Atomic state + effect.** The Moment state transition and the
  outbox enqueue happen in one SQLite transaction. Either both land
  or neither does — the v1 race disappears at the architectural
  level.
- **Single-process simplicity.** One `python -m life_os` starts
  everything. No broker, no ack deadlines, no JetStream storage
  consumer to tune.
- **Testability.** Tests hit the same SQLite they would use for any
  repo test; no fixture processes, no mock brokers. Failure modes
  (retry, dead-letter, boot recovery) are exercised with plain
  transactions.
- **Operational honesty.** The system's durability guarantees are
  SQLite's (WAL + synchronous=NORMAL). v1's story was "NATS gives
  you durability" and it did not, because the publish was outside the
  DB transaction. v2's story is shorter and actually true.

### Negative

- **Single-process ceiling.** Horizontal scaling across worker
  processes is not possible without replacing the bus. For a
  single-user system this is not a real constraint; for Phase 2
  family-share it needs re-examination (see Follow-up).
- **Outbox growth.** Rows accumulate even after delivery. The
  `purge_done_older_than` routine runs daily at 30-day retention,
  but a failure in the purge silently grows the table; `/api/health`
  does not currently alarm on outbox size.
- **No replay across processes.** A crashed process loses in-flight
  non-durable events (those that had not reached the outbox).
  WebSocket reconnect logic in the UI must handle "I missed N Moment
  updates" gracefully, which it does via full re-fetch on reconnect
  but not via gap-fill.
- **Dispatcher is a single point of failure.** Only one dispatcher
  coroutine runs per process. If it deadlocks or crashes, outbox
  delivery stops. Liveness is surfaced via scheduler heartbeat on
  `/api/health`, but a split brain (dispatcher dead, scheduler alive)
  would surface as healthy.

## Alternatives considered

1. **Keep NATS, fix the transaction race.**
   *Rejected* — the race is a distributed-transaction problem that
   SQLite + NATS cannot solve cleanly. Options are (a) two-phase
   commit (unsupported), (b) outbox pattern sending to NATS (adds
   NATS back on top of the outbox without removing it — strictly
   worse than outbox alone), (c) idempotent consumers with
   deduplication keys (possible but pushes correctness into every
   consumer). None of the three is cheaper than removing NATS.

2. **Replace NATS with Redis Streams.**
   *Rejected* — swaps one broker dependency for another, with the
   same transaction-boundary problem. The single-user scale argument
   applies identically.

3. **No broker, no outbox — synchronous side effects.**
   *Rejected* — a synchronous `send_message` call blocks the API
   response on network I/O and eats retry backoff on the request
   thread. Durability on crash is also lost. The outbox is the
   minimum viable durability layer we can ship.

## Follow-up

- Phase 2 (family share) must revisit this if the process boundary
  becomes multi-user. Options at that point include (a) per-user
  process with a shared outbox in a shared SQLite (still single-
  writer) or (b) a real broker with proper outbox relay. Neither
  needs to be decided now.
- Outbox size monitoring: surface `outbox_pending_count` and
  `outbox_dead_count` in `/api/health` and alert thresholds in
  `scripts/cutover_monitor.py` (queued task in NEXT_TASKS.md).
