# ADR 003: Kill soft-insight services (mood / decision / expertise / values)

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

v1 shipped four "soft-insight" services that ran LLM inference over the
user's communication history and emitted opinions about the user:

- `mood_inference` — classified current emotional state from writing
  style
- `decision_profile` — tried to characterize how the user makes
  decisions (risk-tolerant, analytical, …)
- `expertise_map` — guessed at domains the user was knowledgeable in
- `values_inference` — extracted what the user "cared about" from
  content patterns

Post-mortem on the first 6 months of v1 usage surfaced three recurring
problems:

1. **No evidence trail.** The services produced a summary the user could
   read, but no way to ask "why did you think that?". Users would
   challenge outputs and the system had nothing falsifiable to point at.
2. **Self-reinforcing drift.** `mood_at_time` stamped on outbound
   notifications and the prediction engine weighted outputs against
   those stamps; when the inference was wrong (which it was, often), the
   error propagated into suppression decisions the user could not see.
3. **Silent-failure volume.** These services emitted more than half the
   PRs the continuous-improvement agent generated in Q1 2026; they were
   the single largest source of "Moment created but never surfaced" in
   the v1 error budget.

The CEO plan (§ "Killed from v1") made the call explicit:

> Evidence-backed Moments only. Nothing fake.

A soft-insight whose only backing is "the LLM said so" is the canonical
example of "fake" in that sentence.

## Decision

**Delete the four soft-insight services from the v2 runtime. Do not port
their output tables. Archive the code under `deprecated/` for reference,
not import.**

Concretely:

- `producers/` contains only evidence-backed producers: `cadence`,
  `relationship`, `temporal`, `spatial`, `comm_template`, `routine`.
  Each emits a Moment with a payload that names the `events` rows
  it is grounded in.
- v2 schema omits every mood / decision / expertise / values table from
  v1. The migration script drops these rows rather than translating
  them; the `mood_at_time` column on v1 `feedback_log` is intentionally
  *not* carried into the v2 `feedback_events` table (see ADR 001).
- Any v2 code path (LLM prompt, briefing, draft-reply) that previously
  referenced a mood/decision/expertise/values signal is rewritten to
  either (a) use concrete observed evidence or (b) drop the
  personalization entirely and use a neutral default.
- The deprecated directory is not imported by the v2 runtime. A CI
  grep-test asserts no `from deprecated.` imports exist.

## Consequences

### Positive

- **Trust.** Every v2 Moment can answer "why?" with a specific
  `events.id` list. No more "the LLM thinks you're stressed" outputs
  the user cannot challenge.
- **Error-budget slack.** Removing the highest-volume silent-failure
  source makes the ≤1 silent-failure-per-week goal tractable.
- **Producer pipeline simplification.** LLM calls are now only used
  for (a) briefing assembly, (b) draft-reply generation — both of
  which consume evidence the user can inspect. They are no longer a
  primary *source* of insights.
- **Bias surface shrinks.** v1's "values inference" has obvious
  bias-amplification risk (training data is the user's own writing,
  outputs are summaries of that writing, then fed back as
  personalization). Deleting it removes the risk without mitigation
  theater.

### Negative

- **Lost ergonomics.** Some real value went out with the bathwater.
  The mood-aware draft-reply tone adjustment was occasionally
  excellent; v2 ships with a fixed neutral tone until a concretely
  evidenced replacement (e.g., per-contact comm-template) covers the
  same ground.
- **No quick-answer self-portrait.** The You tab in v2 is four
  plain-text sections with observed evidence per row (see DESIGN.md
  § YouTabView). v1 could generate a one-paragraph "about you"
  summary on demand; v2 refuses to.
- **Migration asymmetry.** v1 users see a subtle data loss on
  cutover: the mood / values / expertise bundles disappear. Users who
  valued these outputs will feel the regression. The cutover runbook
  flags this.
- **Deprecated code carrying cost.** `deprecated/` must stay
  unimported and uncoupled; lint tests guard against accidental
  re-import. Dead code has maintenance gravity even when archived.

## Alternatives considered

1. **Keep the services, add evidence plumbing.**
   *Rejected* — the inferences are not compositional from evidence.
   Extracting a "mood" from three emails is not a function of those
   three emails; it is a judgment call the model makes. Adding
   "citations" to a judgment call does not make the call evidenced,
   only annotated.

2. **Keep as internal-only signals, never surface to the user.**
   *Rejected* — v1 exposed `mood_at_time` internally for prediction
   suppression, not UI rendering. That already happened, and it
   already caused the self-reinforcing drift problem. "Internal-only"
   was never the safe path we hoped it would be.

3. **Replace with explicit user self-reporting.**
   *Rejected for Phase 1*, not for the future. v2 does not ask the
   user to fill in forms about themselves; if the user wants to tag
   their own state, `semantic_facts` (with confirm/deny gate) is the
   right structural home. A future self-reporting feature would build
   on that table, not on resurrected inference services.

## Follow-up

- The You tab's empty-state copy is calibrated around this decision.
  If the observed-evidence signal is thin for a new user, the section
  simply says "not enough signal yet" rather than reaching for an
  LLM to produce a paragraph.
- Mood is still tracked internally by the briefing prompt as a
  private hint to the generator, **never** exposed externally. This
  is the sole survival of the soft-insight notion and is called out
  as a non-negotiable in the engineering plan.
