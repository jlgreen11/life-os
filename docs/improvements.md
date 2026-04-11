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

- **Fix Google connector authentication failure** · `broken_feature` — Google connector in error state since Feb 20 (50 days), no new email/calendar data flowing. Likely needs OAuth token refresh or re-authentication.
- **Reduce notification expiry rate (96%)** · `broken_feature` — 159 of 160 notifications expire without user interaction. Auto-deliver converts pending→delivered after 6h but users don't see them. Needs email/push fallback when WebSocket not connected.
- **Fix 0 routines detected despite 869 episodes** · `broken_feature` — Routine detector produces 0 routines. Wave 1 fixed consistency calculation (PR #678), but routine_detector source stopped producing events on March 6. Investigate whether the detection loop crashed or the algorithm still filters everything out.
- **Reduce prediction deduplication rate (16x)** · `data_quality` — 3882 dedup events vs 243 generation events. Even after the time_horizon dedup fix (April 10), the engine may be generating too many similar predictions. Review generation triggers and dedup window.

## In Progress

_Automatically updated each wave. Do not hand-edit unless a wave is stuck._

<!-- AGENT-MANAGED -->

- **Fix prediction engine time-based trigger state persistence race** · `broken_feature` (wave 2, slot 1)
- **Add email.received support to decision extractor** · `missing_feature` (wave 2, slot 2)
- **Fix communication templates blocked by over-aggressive marketing filter** · `broken_feature` (wave 2, slot 3)

## Completed

_Append-only log of merged improvements. Most recent first._

<!-- AGENT-MANAGED: planner prepends completed items here. -->

- **Fix routine detection consistency calculation** · `broken_feature` — wave 1, PR #678
- **Fix decision extractor crash on date-only calendar start_time** · `broken_feature` — wave 1, PR #677
- **Fix temporal extractor date-only calendar crash and profile persistence** · `broken_feature` — wave 1, PR #679

## Ideas

_Unsorted / half-baked thoughts. Humans: dump things here and the agent will
promote them to Backlog when it sees they're actionable. Agent: only promote
from Ideas → Backlog; do not delete._

- _(none)_
