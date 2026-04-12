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

- **Fix dashboard loadMood() JSON path mismatch** · `broken_feature` — loadMood() reads data.energy but endpoint returns data.mood.energy_level. Mood bars always show defaults. From UI engagement fixes plan Task 1.
- **Fix draftReply() payload and add copy button** · `broken_feature` — draftReply() sends wrong payload format (missing incoming_message field). AI generates context-free drafts. From UI engagement fixes plan Task 2.
- **Add WAL checkpoint resilience for user_model.db writes** · `code_quality` — DB corruption events on March 5-6 wiped predictions, templates, and signal profiles. Only prediction engine checkpoints WAL; template store and signal profile writes don't. Need broader WAL checkpointing strategy.
- **Fix source weight drift cap warning** · `data_quality` — When AI drift reaches ±0.3 (MAX_DRIFT), the effective weight hits 0.0 or 1.0 and the user's explicit weight preference becomes invisible. No warning or cap-hit notification exists.

## In Progress

_Automatically updated each wave. Do not hand-edit unless a wave is stuck._

<!-- AGENT-MANAGED -->

- **Fix signal profile health check to retry persistently missing profiles** · `broken_feature` (wave 4, slot 1)
- **Broaden decision extractor signal patterns for inbound email data** · `data_quality` (wave 4, slot 2)
- **Add periodic communication template re-backfill after DB recovery** · `broken_feature` (wave 4, slot 3)
- **Reduce notification expiry rate with more aggressive auto-delivery** · `broken_feature` (wave 4, slot 4)
- **Fix prediction engine post-corruption recovery by clearing prefilter cache** · `broken_feature` (wave 4, slot 5)
- **Fix routine detector to detect temporal patterns in email-dominated data** · `broken_feature` (wave 4, slot 6)
- **Add adaptive lookback to workflow detector for stale connector data** · `broken_feature` (wave 4, slot 7)

## Completed

_Append-only log of merged improvements. Most recent first._

<!-- AGENT-MANAGED: planner prepends completed items here. -->

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

- _(none)_
