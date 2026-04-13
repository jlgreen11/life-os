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

- **Add connector error recovery hints and retry button to admin UI** · `missing_feature` — Returned from wave 28 (no PR). getErrorHint() JS, retry btn, error badge in admin_template.py.
- **Reduce notification expiry rate with more aggressive graduated delivery tiers** · `broken_feature` — Returned from wave 28 (no PR). 4 tiers, delivery_attempts col.
- **Add feedback collector per-source and per-type diagnostic expansion** · `code_quality` — Returned from wave 28 (no PR). by_source, latency_stats, hourly, recency_gap.
- **Add pre-write serialization guard and post-write verification to linguistic outbound profile** · `broken_feature` — Returned from wave 28 (no PR). json.dumps guard, post-write verify in linguistic.py.
- **Reduce prediction deduplication waste with input-state fingerprinting** · `code_quality` — Returned from wave 28 (no PR). Hash cursor+profiles+hour, skip unchanged.
- **Harden onboarding input validation with domain cap, name validation, and multi-range quiet hours** · `code_quality` — Returned from wave 28 (no PR). MAX_DOMAINS=10, name fallback, MAX_CONTACTS=50.
- **Add event bus throughput counters and get_metrics() method** · `code_quality` — Open PR #725 awaiting merge. Returned from wave 10.
- **Add regex-based task extraction fallback when AI engine is unavailable** · `missing_feature` — Open PR #724 awaiting merge. Returned from wave 8 (not merged).
- **Add diagnostic banner to dashboard when user model is empty** · `missing_feature` — Open PR #723 awaiting merge. Returned from wave 8 (not merged).
- **Fix temporal signal profile persistence failure (13,726 qualifying events, 0 profile)** · `broken_feature` — Open PR #700 awaiting merge. Temporal extractor writes fail silently; WAL resilience and retry needed.
- **Fix decision signal profile persistence and add fallback signal extraction** · `broken_feature` — Open PR #699 awaiting merge. Decision profile writes lost to WAL; needs write verification.
- **Add signal profile freshness check to insight engine data sufficiency report** · `missing_feature` — Open PR #701 awaiting merge. Adds freshness/staleness tracking per profile to get_data_sufficiency_report().
- **Fix missing error handling in /api/events, /api/rules, /api/contacts, /api/source-weights** · `code_quality` — Open PR #702 awaiting merge. Adds try/except guards and structured JSON 500 responses.
- **Add update_signal_profile() return value for caller-side failure detection** · `code_quality` — Currently returns void; callers can't distinguish success from silent failure. Requires coordinated changes across all extractors.

## In Progress

_Automatically updated each wave. Do not hand-edit unless a wave is stuck._

<!-- AGENT-MANAGED -->

- **Add prediction pipeline health section to admin UI** · `missing_feature` (wave 116, slot 1)
- **Add post-write verification and WAL checkpoint to store_routine and store_prediction** · `code_quality` (wave 116, slot 2)
- **Add post-execution verification to semantic inference and routine detection loops** · `code_quality` (wave 116, slot 3)
- **Add notification delivery failure diagnostics with per-domain expiry tracking** · `broken_feature` (wave 116, slot 4)
- **Add episode creation rate and backfill status diagnostics to data quality analyzer** · `data_quality` (wave 116, slot 5)
- **Add per-correlator execution timing and consecutive-zero tracking to insight engine** · `code_quality` (wave 116, slot 6)
- **Add rules engine empty-conditions warning and regex pattern pre-validation** · `code_quality` (wave 116, slot 7)

## Completed

_Append-only log of merged improvements. Most recent first._

<!-- AGENT-MANAGED: planner prepends completed items here. -->

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
