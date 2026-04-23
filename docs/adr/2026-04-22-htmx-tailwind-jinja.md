# ADR 006: HTMX + Tailwind + Jinja over a JS SPA

- **Status:** Accepted (backfilled)
- **Authored:** 2026-04-22
- **Decision date:** 2026-04-21
- **Author:** autonomous v2-rewrite agent (iter 16)

## Context

The v1 web UI was a thin Flask app rendering a single `web/template.py`
module that returned hand-concatenated HTML strings (no template
engine). It was minimal by design but had accumulated problems:

- `web/template.py` had grown to ~900 lines of interpolated HTML with
  no separation between layout, partials, and per-tab content. Any
  design change was a string-munging exercise.
- There was no design-token layer. Colors and spacings were hard-coded
  strings scattered across the module.
- State updates required full page reloads. The scheduler firing a
  Moment could not push to the browser; users had to refresh to see
  new suggestions.
- No accessibility tests, no axe-core pipeline, no form validation.

The CEO plan's design brief (`DESIGN.md`) calls for a calm, native-
feeling interface with specific type scale (11/13/15/17/22/28),
semantic color tokens, and a 4-tab information architecture with
live Moment-card state transitions. Implementing that on top of
interpolated HTML strings is not viable.

The common alternatives for the replacement were:

1. **React / Next SPA.** The mainstream choice in 2026.
2. **HTMX + server-rendered Jinja2 templates + Tailwind CSS.**
   Server owns the DOM; interactivity is declarative swaps and
   WebSocket targets.
3. **SvelteKit / Solid / HTMX-plus-Alpine hybrid.** Middle grounds.

## Decision

**The v2 web UI is HTMX-driven. Jinja2 renders fragments; Tailwind
applies design tokens from `DESIGN.md`; WebSocket (`/ws`) pushes
Moment state changes to the appropriate `hx-ws` target. No JavaScript
build pipeline. No SPA bundle.**

Concretely:

- `web/templates/base.html` + per-tab templates (`now.html`,
  `you.html`, `people.html`, `settings.html`) + partials directory
  for reusable components (Moment card, contact row, evidence
  sheet). FastAPI routes render templates; `hx-get` / `hx-post` on
  interactive elements swap fragments in place.
- `web/static/tokens.css` is generated from `DESIGN.md` tokens.
  Tailwind applies them via a `tailwind.config.js` mapping; design
  tokens are the single source, Tailwind is the sugar.
- WebSocket `/ws` is one connection per tab, delivering typed
  `WebSocketEvent` frames (`moment.created`, `moment.state_changed`,
  `connector.status_changed`). The client uses HTMX's `hx-ws`
  extension with `hx-swap-oob` to target specific card IDs.
- No React, no Vue, no SvelteKit, no Vite, no webpack. The only
  build step for the web UI is Tailwind's JIT pass on deploy.
- `web/template.py` is deleted; any residual call site ports to a
  Jinja template.

## Consequences

### Positive

- **No JS build pipeline.** CI does not install npm; deploy does
  not need Node.js in the runtime image. The Python-only deployment
  model matches v1's operational shape.
- **Fragment-level caching.** FastAPI can cache rendered Jinja
  fragments per Moment ID — a cheap optimization that an SPA would
  replace with client-side caching and cache-invalidation bugs.
- **Accessibility by default.** Server-rendered semantic HTML with
  ARIA roles in templates. axe-core smoke tests run against real
  rendered HTML (Week-16 task in the eng plan) instead of a dev
  build's snapshot.
- **State stays on the server.** Moment state lives in SQLite; the
  UI reads and mutates via API; no client-side state machine
  (Redux/Zustand/Jotai/Signal) to drift from the server's truth.
- **Progressive enhancement.** The app works with JS disabled as a
  sequence of ordinary form POSTs; HTMX enhances to partial swaps
  when JS is on. Graceful degradation is free, not engineered.
- **Moment-push is two lines.** The WebSocket handler dispatches
  a rendered fragment to subscribed tabs; the client swaps it in.
  The SPA equivalent is a client-side store, a reducer, and a
  render pass.

### Negative

- **Ceiling for rich client interaction.** Complex drag-and-drop,
  offline-first, long-lived optimistic state — HTMX can do them,
  but the code starts fighting the model. v2 has none of these
  requirements in Phase 1; Phase 2 iOS covers the rich-client
  story.
- **Round-trip latency on every swap.** Every Moment accept is a
  server round trip + a fragment re-render. On a Tailscale LAN
  this is fine (single-digit ms); on public internet it would not
  be. v2's deployment target is LAN-only in Phase 1, so the
  constraint is met by circumstance.
- **No client-side routing.** Tab switches are full page loads
  (with `hx-boost` preserving the scroll position and
  network activity indicator). A user watching the Now tab who
  switches to You tab incurs a real HTML request. HTMX's
  `hx-boost` softens this but does not make it instant.
- **Less hiring gravity.** React is the default modern stack; an
  HTMX codebase is a less-common skillset. For a single-maintainer
  system this is not relevant; a future team hire reads two HTMX
  tutorials and ships.
- **Design review needs real browsers.** Without a component
  library and Storybook, design iteration requires running the
  app. Mitigated by the `/plan-design-review` skill and mock
  fixtures, but it is a real cost.

## Alternatives considered

1. **React SPA served by FastAPI.**
   *Rejected* — introduces npm, Vite, bundling, hydration, a
   client-side state library, and a second mental model (server
   shape vs. client shape). None of these earn their cost for a
   four-tab single-user app. Phase 2 iOS already owns the native-
   client story; a web SPA would duplicate that effort for less
   benefit.

2. **SvelteKit.**
   *Rejected* — smaller than React but still a JS build pipeline,
   still a second mental model. The size delta vs. HTMX is
   significant and the ergonomics on "server pushes an updated
   Moment card" are not materially better.

3. **Alpine.js + Jinja, no HTMX.**
   *Rejected* — Alpine is client-side state sprinkled into
   templates. It answers the SPA ceiling question but reintroduces
   client-side state management. HTMX keeps state on the server;
   Alpine lets it drift.

## Follow-up

- axe-core accessibility smoke test wired into CI: eng plan Week-16
  task.
- WebSocket backpressure / lost-connection UX: the iOS client
  already implements exponential-backoff reconnect (ADR follow-up
  tracked under Category A tasks). The web client uses HTMX's
  built-in reconnect behavior; verify it meets the same guarantees
  before Phase 1 cutover.
- Consider removing unused Tailwind classes via a PurgeCSS pass at
  build time; the current full-file ship is ~50 KB which is
  acceptable.
