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

- **Fix date-only calendar start_time crash in temporal extractor** · `broken_feature` — Same TypeError as decision extractor but in temporal.py's calendar handling path. Also needs investigation into why email.received events don't persist the temporal profile.
- **Fix prediction loop stalled since March 6** · `broken_feature` — No predictions generated for 36+ days despite loop running. 0 stored predictions in user_model.db. Investigate why generate_predictions() returns empty or store_prediction() fails silently.
- **Add email.received support to decision extractor** · `missing_feature` — Decision extractor only triggers on email.sent (11 events), calendar (207), task (0), finance (0). Adding email.received would give it 860+ events for richer decision signal extraction.
- **Fix Google connector authentication failure** · `broken_feature` — Google connector in error state since Feb 20 (50 days), no new email/calendar data flowing. Likely needs OAuth token refresh or re-authentication.
- **Wire semantic fact confirmation loop** · `missing_feature` — DB columns `is_user_corrected`, `times_confirmed`, `source_episodes` exist in facts table but are never populated. The +0.05 confidence increment on user correction doesn't work.
- **Complete linguistic profile field population** · `missing_feature` — ~60% of LinguisticProfile fields populated; hedge_ratio, assertion_ratio, emoji_density, capitalization_style remain empty despite data being available in the extraction pipeline.
- **Reduce notification expiry rate (96%)** · `broken_feature` — 159 of 160 notifications expire without user interaction. Auto-deliver converts pending→delivered after 6h but users don't see them. Needs email/push fallback when WebSocket not connected.
- **Fix communication_templates table empty** · `broken_feature` — Data quality reports 0 communication templates despite linguistic and relationship extractors having template extraction code.

## In Progress

_Automatically updated each wave. Do not hand-edit unless a wave is stuck._

<!-- AGENT-MANAGED -->

- **Fix routine detection consistency calculation averaging across interaction types** · `broken_feature` (wave 1, slot 1)
- **Fix decision extractor crash on date-only calendar start_time** · `broken_feature` (wave 1, slot 2)
- **Fix temporal extractor date-only calendar crash and profile persistence** · `broken_feature` (wave 1, slot 3)

## Completed

_Append-only log of merged improvements. Most recent first._

<!-- AGENT-MANAGED: planner prepends completed items here. -->

- _(none yet)_

## Ideas

_Unsorted / half-baked thoughts. Humans: dump things here and the agent will
promote them to Backlog when it sees they're actionable. Agent: only promote
from Ideas → Backlog; do not delete._

- _(none)_
