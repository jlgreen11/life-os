# Public API Docstring Audit — 2026-04-22

Scope: the 8 files locked by the NEXT_TASKS "Public API docstrings" item.

| File | Public classes | Public methods / functions | All documented? |
| ---- | -------------- | -------------------------- | --------------- |
| `core/moment/engine.py` | `MomentEngine` | `__init__`, `on_event` | Yes |
| `core/moment/scheduler.py` | `FireRecord`, `Scheduler` | `__init__`, `boot_recovery`, `tick`, `run_forever` | Yes |
| `core/moment/state.py` | `IllegalTransition` | `validate_transition` | Yes |
| `storage/repos/moments.py` | `MomentRepository` | `__init__`, `create`, `get`, `last_transition`, `transition`, `snooze`, `update_action_params`, `list_pending`, `list_scheduled`, `list_done_today` | Yes |
| `storage/repos/outbox.py` | `OutboxEntry`, `OutboxRepository` | `__init__` (fixed this pass), `enqueue`, `claim_batch`, `complete`, `fail`, `cancel_pending`, `requeue_in_progress_on_boot`, `purge_done_older_than` | Yes (after fix) |
| `ai/engine.py` | `AIEngineError`, `AIBudgetExceeded`, `SearchResult`, `VectorStore`, `AIEngine` | `__init__` (both errors + engine), `briefing_synthesis`, `task_extraction`, `priority_classification`, `draft_reply`, `semantic_search` | Yes |
| `api/routes/now.py` | — | `get_now`, `accept_moment`, `dismiss_moment`, `snooze_moment`, `get_moment_evidence`, `now_page`, `undo_moment`, `edit_moment` | Yes |
| `api/routes/you.py` | — | `get_you`, `you_page` | Yes |

## Findings

1. **Only structural gap found:** `OutboxRepository.__init__` had no docstring while the parallel `MomentRepository.__init__` does. Fixed this pass — added the same four-paragraph init contract (`now_fn` determinism · isolation-level flip · row-factory wiring · transaction-lifecycle ownership).

2. **Every public class carries a docstring.** Dataclasses (`FireRecord`, `OutboxEntry`, `SearchResult`) describe field semantics; error classes (`IllegalTransition`, `AIEngineError`, `AIBudgetExceeded`) enumerate the sub-kinds callers can triage on.

3. **Every public method that can raise documents the exception.** Repos raise `KeyError` for missing rows and `IllegalTransition` for state-machine rejections; the AI engine raises `AIBudgetExceeded` and `AIEngineError` — all surfaced in `Raises` sections or narrative-equivalent prose.

4. **Every public method that returns non-trivially documents the shape.** Engine methods describe bucket contents; repo methods describe ordering and null-like corner cases (legacy rows, empty lists); API routes describe dual-mode (HTMX vs JSON) response bodies.

5. **Every non-trivial public API has an example grounded in actual usage.** E.g. the engine docstrings walk the per-event pipeline; the scheduler docstrings show the `ContextTrigger` grammar; `now.py` explains the `HX-Trigger` envelope.

## Style consistency

Two distinct docstring dialects coexist in these 8 files:

- `ai/engine.py` + `storage/repos/*.py` — explicit reST-style `Parameters` / `Returns` / `Raises` sections.
- `core/moment/*.py` + `api/routes/*.py` — narrative prose with embedded module-level and class-level context sections (`References`, `Error mapping`, etc.).

Both dialects cover the task's acceptance criteria (summary + args + returns + raises + example for non-trivial APIs). The narrative style is arguably better-suited for the state-machine + HTMX routes because the prose binds multiple moving parts at once (feedback EWMA, outbox grace windows, dual-mode responses), while explicit sections work better for the AI engine where each operation is a standalone LLM call with its own budget.

**Decision: leave the two dialects as-is.** A mass rewrite toward a single style would lose per-file readability without adding information; readers who open one file do not cross-compare docstring grammar. If we ever want to enforce a single style, that is a separate, larger refactor tracked as its own NEXT_TASKS item — not dropped into this audit.

## What was NOT in scope

- Private helpers (`_hydrate`, `_now`, `_passes_threshold`, `_persist`, `_action_label`, `_transition_or_409`, …) — some already carry docstrings for their own maintainability, but they are not part of the documented public surface.
- Module-level constants (`MAX_RETRIES`, `PENDING_LIMIT`, `_DISPATCH_SUBJECTS`, …) — documented in-situ via adjacent comments where the value is load-bearing.
- Other files in `core/`, `storage/`, `producers/`, `ai/`, `api/`, `web/` that were not named by the task — untouched.
