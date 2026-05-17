# Life OS — Improvements Backlog

This is the **grand master plan** for Life OS improvements. It is maintained
collaboratively between:

- The **parallel improvement orchestrator** (`scripts/run-parallel-improvement.sh`),
  which runs continuously on the Mac Mini server. Each wave, the planner agent
  reads this file, selects the top items from "Backlog", moves them to
  "In Progress", and spawns workers to implement them.
- **You** (or any other Claude session) — add ideas anytime by editing this file
  directly, committing, and pushing to `master`. The orchestrator will pick them
  up on the next wave (typically within 30–60 seconds of a wave ending).

## How to use this file

- Put higher-priority items at the **top** of "Backlog". The planner picks from
  the top down.
- Each item should have a **one-line summary**, a **category**, and a short
  **rationale** (why it matters). A few sentences of implementation hints are
  helpful but optional — the planner will expand them into concrete task specs.
- "In Progress" is owned by the orchestrator. Don't hand-edit it; the planner
  rewrites it every wave.
- "Completed" is an append-only log. Structured history also lives in
  `data/improvement-runs/state.json`.
- Humans can add items anywhere under "Backlog" or "Ideas". The planner will
  not delete human-authored items without explicit instruction.

## Categories

`planned_feature` · `broken_feature` · `missing_feature` · `data_quality` ·
`integration_gap` · `test_coverage` · `code_quality` · `cleanup`

---

## Backlog

_Prioritized list of improvements the planner should pick from. The agent will
seed this from codebase analysis on its first wave after this file was added.
Feel free to hand-add items above or below whatever the agent writes._

<!-- AGENT-MANAGED: the planner adds/removes items here each wave. Human edits
     are preserved as long as they follow the item format below. -->

- **Add WAL-checkpoint retry to spatial extractor post-write verification** · `broken_feature` — Re-queued from wave 4 (no PR opened yet).
- **Add connection metrics, heartbeat, and broadcast-failure diagnostics to WebSocket manager** · `code_quality` — Re-queued from wave 4 (no PR opened yet).
- **Add per-strategy resolution counters and accuracy breakdown to behavioral accuracy tracker** · `data_quality` — Re-queued from wave 4 (no PR opened yet).
- **Add per-category extraction counters and skip-reason diagnostics to semantic fact inferrer** · `data_quality` — Re-queued from wave 4 (no PR opened yet).
- **Tighten Pydantic schema validation in web/schemas.py for command, search, and rule endpoints** · `code_quality` — Re-queued from wave 4 (no PR opened yet).
- **Update unused-capability audit to reflect features now implemented (CalDAV conflict, comm templates, episodes)** · `cleanup` — Open PR #748 from wave 4, awaiting merge.
- **Add embedding-failure tracking and per-source ingest counters to vector store** · `code_quality` — Open PR #749 from wave 4, awaiting merge.
- **Add post-write verification (sampled) to relationship signal extractor** · `broken_feature` — Open PR #747 from wave 3, awaiting merge.
- **Add prediction pipeline health section to admin UI** · `missing_feature` — Open PR #734 from wave 1, awaiting merge.
- **Add post-write verification and WAL checkpoint to store_routine and store_prediction** · `code_quality` — Open PR #737 from wave 1, awaiting merge.
- **Add post-execution verification to semantic inference and routine detection loops** · `code_quality` — Open PR #736 from wave 1, awaiting merge.
- **Add notification delivery failure diagnostics with per-domain expiry tracking** · `broken_feature` — Open PR #733 from wave 1, awaiting merge.
- **Add episode creation rate and backfill status diagnostics to data quality analyzer** · `data_quality` — Open PR #735 from wave 1, awaiting merge.
- **Add per-correlator execution timing and consecutive-zero tracking to insight engine** · `code_quality` — Open PR #738 from wave 1, awaiting merge.
- **Add prediction deduplication root-cause logging with per-type and per-trigger breakdown** · `data_quality` — Open PR #739 from wave 1, awaiting merge.
- **Expand feedback collector diagnostics with per-source breakdown and source×type cross-tab** · `code_quality` — Open PR #740 from wave 2, awaiting merge.
- **Harden onboarding input validation: cap domains, validate contact names, support multi-range quiet hours** · `code_quality` — Open PR #746 from wave 2, awaiting merge.
- **Add post-write verification and WAL checkpoint to cadence extractor profile updates** · `broken_feature` — Open PR #743 from wave 2, awaiting merge.
- **Add pre-write serialization guard and post-write verification to linguistic per-contact communication templates** · `broken_feature` — Open PR #742 from wave 2, awaiting merge.
- **Add structured skip-reason diagnostics to routine detector for zero-result cycles** · `data_quality` — Open PR #744 from wave 2, awaiting merge.
- **Add structured skip-reason diagnostics to workflow detector for sender/task/calendar candidates** · `data_quality` — Open PR #745 from wave 2, awaiting merge.
- **Add per-strategy diagnostic counters to task completion detector** · `data_quality` — Open PR #741 from wave 2, awaiting merge.
- **Fix temporal signal profile persistence failure (13,726 qualifying events, 0 profile)** · `broken_feature` — Verified largely resolved in current temporal.py (post-write verification + retry already implemented in lines 271-306); profile now shows 384,830 samples updated within the hour. Keep entry for traceability but not actionable.
- **Add update_signal_profile() return value for caller-side failure detection** · `code_quality` — Currently returns void; callers can't distinguish success from silent failure. Requires coordinated changes across all extractors. Defer until in-progress per-extractor write verifications (waves 1-3) merge to avoid sweeping conflicts.
- **Add connector error recovery hints and retry button to admin UI** · `missing_feature` — Inline OAuth re-auth flow for Google connector token-missing state. Defer until wave 1 admin UI work (PR #734) merges to avoid file conflicts.
- **Reduce notification expiry rate with 4-tier graduated delivery and delivery_attempts tracking** · `broken_feature` — Reduce 84% expiry rate via graduated channels. Defer until wave 1 notification diagnostics (PR #733) merges to avoid file conflicts.
- **Reduce prediction deduplication waste with input-state fingerprinting** · `code_quality` — Distinct from observability work in wave 1 slot 7 — this is the dedup logic fix (cap triggers, fingerprint inputs). Defer until wave 1 PR #739 merges.
- **Update unused-capability audit to remove CalDAV-conflict-detection stub claim** · `cleanup` — Promoted from Ideas. CalDAV conflict detection is fully implemented in connectors/caldav/connector.py:308-480; audit doc still lists it as a stub. Likely subsumed by wave 4 slot 3 (PR #748) once merged.
- **Add post-write verification + WAL checkpoint retry to spatial extractor (sample-mode)** · `broken_feature` — Spatial profile stuck at 209 samples, last updated 2026-04-14 (33+ days stale). Subsumed by wave 4 slot 1 re-queue at top of backlog.
- **Add data quality analyzer surfacing of conflict-detector diagnostics** · `data_quality` — Once wave-5 slot 4 lands per-strategy counters in conflict_detector.get_diagnostics(), expose them in scripts/analyze-data-quality.py output so 0-conflict runs become operator-visible.
- **Add Anthropic cloud AI engine health metrics with rolling success rate** · `code_quality` — Companion to wave-5 slot 6 (Ollama metrics). Mirror the same instrumentation on the cloud-Claude call path; both share services/ai_engine/engine.py so should land after slot 6 to avoid conflict.
- **Add per-rule-action-type counters to RulesEngine admin UI surface** · `code_quality` — Once wave-5 slot 7 lands get_diagnostics() on RulesEngine, expose those counters in /admin and /health so operators can spot the 33-day rules silence.
- **Investigate why source_weights.updated_at is null for 14 of 16 weights despite 100k+ interactions** · `data_quality` — Only email.work and email.transactional show non-null updated_at; the others (messaging.direct with 46k interactions, calendar.meetings with 1.7k, etc.) never persist. Likely a write path that records the in-memory weight but skips the SQL UPDATE.

## In Progress

_Automatically updated each wave. Do not hand-edit unless a wave is stuck._

<!-- AGENT-MANAGED -->

- **Add post-write verification + WAL-checkpoint retry to decision signal extractor** · `broken_feature` (wave 5, slot 1)
- **Add throughput counters and get_metrics() to NATS event bus** · `code_quality` (wave 5, slot 2)
- **Add regex/heuristic task extraction fallback when AI engine is unavailable** · `missing_feature` (wave 5, slot 3)
- **Add per-strategy and skip-reason diagnostics to calendar conflict detector** · `data_quality` (wave 5, slot 4)
- **Strengthen PIIShield redaction patterns (IBAN, IPv6, MAC, API keys) + edge-case tests** · `code_quality` (wave 5, slot 5)
- **Add Ollama health metrics with rolling success rate to AI engine** · `code_quality` (wave 5, slot 6)
- **Add per-action-type diagnostics and stale-rule detection to RulesEngine** · `code_quality` (wave 5, slot 7)

## Completed

_Append-only log of merged improvements. Most recent first._

<!-- AGENT-MANAGED: planner prepends completed items here. -->

- **Add rules engine empty-conditions warning and regex pattern pre-validation** · `code_quality` — PR #729 (merged 2026-05-17)
- **Add diagnostic banner to dashboard for cold-start and degraded states** · `missing_feature` — PR #723 (merged 2026-05-17)
- **Add error handling to 6 unprotected API endpoints** · `code_quality` — PR #702 (merged 2026-05-17)
- **Add signal profile freshness check to insight engine sufficiency report** · `missing_feature` — PR #701 (merged 2026-05-17)
- **Add pre-write JSON serialization guard to cadence extractor** · `broken_feature` — wave 13 (verified already implemented: cadence extractor uses plain dicts instead of defaultdict, confirmed in cadence.py:230-233)
- **Add pre-write JSON serialization guard to topic extractor** · `broken_feature` — wave 11, PR #727
- **Add pre-write JSON serialization guard to spatial extractor** · `broken_feature` — wave 11, PR #728
- **Add vector store health diagnostics and stale embedding detection** · `code_quality` — wave 11, PR #726
- **Add notification suppression telemetry and feedback logging** · `broken_feature` — wave 9 (verified already implemented: _log_automatic_feedback and dismissal suppression exist in notification_manager)
- **Set prediction persistence failure flag immediately on store exceptions** · `broken_feature` — wave 9 (verified already implemented: _persistence_failure_detected flag with full recovery in prediction_engine)
- **Add cache_age_seconds to /api/insights/summary response** · `code_quality` — wave 7, PR #720
- **Fix communication template backfill DB connection reuse and WAL checkpoint** · `broken_feature` — wave 7, PR #722
- **Fix episode backfill missing post-write verification and WAL checkpoint** · `broken_feature` — wave 7, PR #719
- **Add cold-start cycle diagnostics to behavioral accuracy tracker** · `code_quality` — wave 7, PR #721
- **Fix episode store phantom telemetry and add WAL checkpoint resilience** · `broken_feature` — wave 6, PR #714
- **Fix linguistic_inbound profile persistence with write verification and data compaction** · `broken_feature` — wave 6, PR #717
- **Add structured error reporting to dashboard calendar and insights loaders** · `code_quality` — wave 6, PR #713
- **Fix routine detection min_episodes threshold for cold-start email data** · `broken_feature` — wave 6, PR #715
- **Fix prediction loop stall — no predictions generated since March 6** · `broken_feature` — wave 6, PR #681
- **Fix mood extractor profile persistence root cause** · `broken_feature` — wave 6, PR #718
- **Add event-based fallback to semantic fact inferrer for empty episodes table** · `missing_feature` — wave 6, PR #716
- _(waves 1-5 history: PRs #677-#712 — see git log for details)_

- **Add adaptive lookback to workflow detector for stale connector data** · `broken_feature` — wave 4, PR #694
- **Fix dashboard loadMood() JSON path mismatch** · `broken_feature` — previously implemented (all UI engagement fixes plan tasks complete)
- **Fix draftReply() payload and add copy button** · `broken_feature` — previously implemented (all UI engagement fixes plan tasks complete)
- **Fix routine detector: auto-extend lookback when connector outage leaves 0 recent episodes** · `broken_feature` — wave 3, PR #685
- **Fix notification batch durability: replace in-memory _pending_batch with DB-backed status** · `broken_feature` — wave 3, PR #689
- **Reduce prediction intra-batch duplicates and persist pre-filter across cycles** · `data_quality` — wave 3, PR #688
- **Improve Google connector health_check with structured auth diagnosis** · `broken_feature` — wave 3, PR #684
- **Add periodic signal profile health check with auto-rebuild** · `missing_feature` — wave 3, PR #687
- **Fix data quality analyzer profile expectations and health score accuracy** · `data_quality` — wave 3, PR #683
- **Improve semantic fact inferrer for relationship-heavy datasets** · `data_quality` — wave 3, PR #686
- **Fix communication templates blocked by over-aggressive marketing filter** · `broken_feature` — wave 2, PR #682
- **Add email.received support to decision extractor** · `missing_feature` — wave 2, PR #680
- **Fix prediction engine time-based trigger state persistence race** · `broken_feature` — wave 2, PR #681
- **Fix routine detection consistency calculation** · `broken_feature` — wave 1, PR #678
- **Fix decision extractor crash on date-only calendar start_time** · `broken_feature` — wave 1, PR #677
- **Fix temporal extractor date-only calendar crash and profile persistence** · `broken_feature` — wave 1, PR #679

## Ideas

_Unsorted / half-baked thoughts. Humans: dump things here and the agent will
promote them to Backlog when it sees they're actionable. Agent: only promote
from Ideas → Backlog; do not delete._

- Update unused capability audit — CalDAV conflict detection is now fully implemented (connector.py:308-480) but audit still lists it as a stub.
- Linguistic outbound profile has only 11 qualifying events (email.sent/message.sent) — may need to wait for more outbound data or lower extraction thresholds.
