# ADR 007: Web-first Phase 1; iOS native deferred to Phase 2

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

v1 shipped both a web UI (Flask + hand-rolled HTML) and a native iOS
app (`ios/LifeOS/`). Both clients talked to the same v1 API. The
iOS app existed because:

- iOS context signals (location, screen-time, focus modes) require
  a native process; there is no web path to these APIs.
- Push notifications are the most natural interaction surface for
  "right action at the right moment".
- TestFlight gives the single user a convenient install path.

After 6 months of production, the practical picture was:

- The user spent ~70% of their "Life OS UI time" on the web
  interface at their desk, not the phone.
- The native iOS app had not been updated in ~4 months because
  Apple Dev Program renewal had lapsed and re-enrollment was
  queued behind higher-priority work.
- Push notifications from the v1 system still worked because the
  last build was signed and installed; but any change required a
  re-enrollment round trip.
- The iOS context signal — the genuinely iOS-native value — flowed
  through a background context-upload service (`/api/context/event`,
  `/api/context/batch`) that did **not** need the full app UI to
  keep working. The data-collection and the UI are separable.

The CEO plan took these observations and concluded:

> Phase 1 ships a great web experience. iOS context keeps collecting
> data via the compat shim. The full iOS UI rebuild happens in
> Phase 2 once Apple Dev is re-enrolled.

The eng plan codified this by:

- Preserving the 8 iOS-compat-shim endpoints in the v2 API.
- Stubbing 4 endpoints that iOS expects but v2 does not produce
  (`/api/predictions`, `/api/notifications`, `/api/user-model`,
  `/api/insights`) with empty payloads, schema-compatible.
- Keeping the iOS scaffold compiling (Phase 1 scope) without
  wiring new features.

## Decision

**Phase 1 targets the web UI as the primary surface. iOS keeps
collecting context signals via the preserved compat-shim endpoints;
the native iOS UI is a scaffold that compiles on any Mac but is not
wired to the v2 feature set. The full iOS rebuild is Phase 2, gated
on Apple Dev Program re-enrollment.**

Concretely:

- `web/` ships a complete 4-tab experience (Now / You / People /
  Settings) per DESIGN.md, using HTMX + Tailwind + Jinja (ADR 006).
- `api/routes/context.py` preserves the 8 context-shim endpoints
  verbatim so the live iOS build (unchanged since v1) continues
  uploading signals through cutover.
- `api/routes/*_stub.py` (or inline stubs under existing routes)
  return empty, schema-compatible payloads for the 4 stubbed
  endpoints. The iOS app receives "no predictions, no
  notifications, no user model, no insights" — which is truthful
  under the ADR 003 kill.
- `ios/LifeOS/` is restructured around the 4-tab IA as pure SwiftUI
  scaffolds (RootTabView + stub NowTabView / YouTabView /
  PeopleTabView / SettingsTabView with full DESIGN.md token usage).
  Views are rendered against mock data; API wiring happens in
  Phase 2.
- Phase 1 exit criterion (≥3 insight types at ≥60% accept rate)
  is measured from the web UI's acceptance events. iOS acceptance
  is not in the exit criterion.
- The orchestrator's `scripts/run-v2-autonomous.sh` agent enforces
  "never touch `config/settings.yaml`, never commit production
  data" — but does build out the iOS scaffold layer because the
  code is agent-safe (no Apple Dev needed to compile / test
  SwiftUI scaffolds on Mac).

## Consequences

### Positive

- **Focus.** One primary UI surface means one design language to
  polish, one accessibility pass, one set of user-test sessions.
  The eng plan's 16-18 week timeline is feasible under this scope;
  it was not feasible with a native iOS rebuild in parallel.
- **Apple Dev decoupling.** Phase 1 ship does not block on Apple
  Dev re-enrollment. The user's enrollment timeline is independent
  of the rewrite cutover.
- **Context signal preservation.** The live iOS build keeps
  uploading to v2 through the preserved endpoints. Spatial and
  temporal producers consume the same signal stream they do today.
  No regression in data collection during cutover.
- **Phase 2 starts from a scaffold, not a blank page.** The iOS
  4-tab IA, design tokens, APIClient + APIClientProtocol, view
  models, and WebSocketManager are all implemented during Phase 1
  under the autonomous agent. Phase 2 wires them to real data and
  adds APNs / widgets / TestFlight.
- **Aligns with observed usage.** 70% of v1 UI time was on the web.
  Shipping the great web experience first matches where the user
  actually spends attention.

### Negative

- **"No notifications on iOS" during Phase 1.** The live iOS build
  will continue firing v1's notification logic against v2 data
  until cutover; afterward the 4 stubbed endpoints return empty
  payloads, and the iOS app goes quiet. The user sees this
  regression on their phone. They must use the web UI for
  Moment-level interactions until Phase 2 rewires iOS.
- **Two information architectures during the transition.** The web
  UI ships the new 4-tab IA; the live iOS UI still renders the v1
  Dashboard / Chat / Context / Settings layout because it has not
  been rebuilt. Until Phase 2 ships, the two surfaces do not match
  the same mental model.
- **iOS scaffold maintenance cost.** The SwiftUI scaffold under
  `ios/LifeOS/` must stay compilable as API contracts evolve.
  Every API-shape change during Phase 1 risks breaking the iOS
  compile; the scaffold tests catch this locally but the absence
  of a live integration means errors may only show up at Phase 2
  wiring time.
- **Deferred real-device validation.** The decisions baked into
  the iOS scaffold (tab order, card layout, snooze sheet
  ergonomics) are validated against SwiftUI previews and unit
  tests only. The real-device UX may surface issues the previews
  miss; these resurface in Phase 2 rather than earlier.
- **"When does Phase 2 start?" is vague.** The ADR gates Phase 2
  on Apple Dev re-enrollment. If enrollment slips, iOS slips.
  The CEO plan acknowledges this and does not set a hard Phase 2
  date.

## Alternatives considered

1. **Ship native iOS + web in parallel during Phase 1.**
   *Rejected* — requires Apple Dev enrollment on the critical
   path, doubles the surface area polished under the 16-18 week
   budget, and duplicates effort before acceptance KPIs have
   validated which insight types work. Better to validate on
   web, then rebuild iOS in Phase 2 against the insights that
   survived.

2. **Web-only, delete the iOS app.**
   *Rejected* — the iOS context signal (spatial, screen-time,
   focus modes) is the single highest-value data stream. Losing
   it to delete the UI scaffold is a bad trade. Keep the
   collection; defer the UI rebuild.

3. **iOS-only (kill the web), rebuild native.**
   *Rejected* — observed usage is 70% web; the user's desk time is
   where Life OS earns its keep for knowledge work. iOS is the
   mobile companion, not the primary surface.

## Follow-up

- Apple Dev re-enrollment: tracked as a human-only task in
  `NEXT_TASKS.md` (out of agent scope).
- Phase 2 iOS rebuild plan: to be written when Apple Dev is
  restored. Will build on the Phase 1 scaffold (RootTabView,
  4-tab IA, design tokens, APIClientProtocol, view models) which
  are already in place.
- Cutover runbook (also in NEXT_TASKS.md) must call out the
  "iOS notifications quiet" regression and instruct the operator
  to warn the user before cutover.
- Phase 1 exit criterion is measured from web-UI acceptance;
  confirm the `/api/moments/{id}/accept` route is instrumented
  with enough source-tagging to distinguish web from iOS
  acceptance if Phase 2 ships under the same criterion later.
