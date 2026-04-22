# Topic Stream Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current command-prompt front page with a 3-column topic stream dashboard that auto-surfaces timely info, differentiates communication types, and enables drill-down.

**Architecture:** Complete rewrite of `web/template.py` (HTML_TEMPLATE string). The file contains all HTML, CSS, and JS inline — no external files or build tools. Existing API endpoints (`/api/notifications`, `/api/tasks`, `/api/briefing`, `/api/events`, `/api/insights/summary`, `/api/user-model/mood`, `/health`) already provide all data; the frontend just needs to consume them. One new lightweight API endpoint needed for the dashboard feed.

**Tech Stack:** Vanilla JS (ES6), CSS3 custom properties, FastAPI (Python), existing WebSocket at `/ws`

**Security Note:** All user-generated content rendered in cards MUST be escaped via a dedicated `escHtml()` function before insertion into the DOM. The `innerHTML` usage throughout is safe because every dynamic value passes through HTML entity encoding first — no raw user input is ever inserted. The `escHtml()` function replaces `&`, `<`, `>`, and `"` with their HTML entity equivalents.

---

## Task 1: Add Dashboard Feed API Endpoint

The current API returns data per-type (tasks, notifications, events separately). The dashboard "Inbox" view needs a unified, priority-sorted feed across all types. Add one endpoint that aggregates them.

**Files:**
- Modify: `web/routes.py:176` (add new endpoint after command bar section)
- Modify: `web/schemas.py` (no new schema needed — query params only)

**Step 1: Add the `/api/dashboard/feed` endpoint**

Add this endpoint in `web/routes.py` after the `/api/command` route (around line 176). Insert it before the Briefing section:

```python
    # -------------------------------------------------------------------
    # Dashboard Feed (unified, priority-sorted view)
    # -------------------------------------------------------------------

    @app.get("/api/dashboard/feed")
    async def dashboard_feed(topic: Optional[str] = None, limit: int = 50):
        """Unified feed for the dashboard. Aggregates notifications, tasks,
        and recent events into a single priority-sorted list.

        Query params:
            topic: Filter by category — inbox (all), messages, email,
                   calendar, tasks, insights, system. Default: inbox.
            limit: Max items to return. Default: 50.
        """
        items = []

        # --- Notifications (all topics except 'system') ---
        if topic in (None, "inbox", "messages", "email"):
            try:
                notifications = life_os.notification_manager.get_pending(limit=limit)
                for n in notifications:
                    source_type = n.get("source", "")
                    # Classify by communication channel
                    if topic == "messages" and "message" not in source_type and "signal" not in source_type:
                        continue
                    if topic == "email" and "email" not in source_type:
                        continue
                    items.append({
                        "id": n.get("id"),
                        "kind": "notification",
                        "channel": "message" if "message" in source_type or "signal" in source_type else "email" if "email" in source_type else "system",
                        "title": n.get("title", ""),
                        "body": n.get("body", ""),
                        "priority": n.get("priority", "normal"),
                        "timestamp": n.get("created_at", n.get("timestamp", "")),
                        "source": source_type,
                        "metadata": n.get("metadata", {}),
                    })
            except Exception:
                pass

        # --- Tasks ---
        if topic in (None, "inbox", "tasks"):
            try:
                tasks = life_os.task_manager.get_tasks(status="pending", limit=limit)
                for t in tasks:
                    items.append({
                        "id": t.get("id"),
                        "kind": "task",
                        "channel": "task",
                        "title": t.get("title", ""),
                        "body": t.get("description", ""),
                        "priority": t.get("priority", "normal"),
                        "timestamp": t.get("created_at", ""),
                        "source": t.get("domain", ""),
                        "metadata": {"due_date": t.get("due_date"), "domain": t.get("domain")},
                    })
            except Exception:
                pass

        # --- Recent events (email, messages, calendar) ---
        if topic in (None, "inbox", "messages", "email", "calendar"):
            try:
                event_types = []
                if topic in (None, "inbox", "email"):
                    event_types.append("email.received")
                if topic in (None, "inbox", "messages"):
                    event_types.append("message.received")
                if topic in (None, "inbox", "calendar"):
                    event_types.extend(["calendar.event.created", "calendar.event.updated", "calendar.event.reminder"])

                for et in event_types:
                    events = life_os.event_store.get_events(event_type=et, limit=20)
                    for ev in events:
                        payload = ev.get("payload", {}) if isinstance(ev.get("payload"), dict) else {}
                        channel = "email" if "email" in et else "message" if "message" in et else "calendar"
                        items.append({
                            "id": ev.get("id"),
                            "kind": "event",
                            "channel": channel,
                            "title": payload.get("subject", payload.get("title", et)),
                            "body": payload.get("snippet", payload.get("body", payload.get("description", "")))[:200],
                            "priority": "high" if payload.get("urgency", 0) > 0.7 else "normal",
                            "timestamp": ev.get("timestamp", ""),
                            "source": ev.get("source", ""),
                            "metadata": {
                                "sender": payload.get("sender", payload.get("from", "")),
                                "sentiment": payload.get("sentiment"),
                                "action_items": payload.get("action_items", []),
                                "attendees": payload.get("attendees", []),
                                "location": payload.get("location", ""),
                                "start_time": payload.get("start_time", payload.get("start", "")),
                                "end_time": payload.get("end_time", payload.get("end", "")),
                            },
                        })
            except Exception:
                pass

        # --- Sort by priority (critical > high > normal > low), then by timestamp desc ---
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        items.sort(key=lambda x: (priority_order.get(x["priority"], 2), x.get("timestamp", "") == "", x.get("timestamp", "")), reverse=False)
        # Reverse timestamp within same priority so newest first
        items.sort(key=lambda x: (priority_order.get(x["priority"], 2),))

        return {"items": items[:limit], "count": len(items[:limit]), "topic": topic or "inbox"}
```

**Step 2: Verify the endpoint works**

Run: `curl -s http://localhost:8080/api/dashboard/feed | python3 -m json.tool | head -30`
Expected: JSON with `items` array, `count`, and `topic: "inbox"`

Run: `curl -s "http://localhost:8080/api/dashboard/feed?topic=tasks" | python3 -m json.tool | head -20`
Expected: Only task items returned

**Step 3: Commit**

```bash
git add web/routes.py
git commit -m "feat: add /api/dashboard/feed unified feed endpoint"
```

---

## Task 2: Rewrite CSS — 3-Column Layout and Card System

Replace the entire `<style>` block in `web/template.py` with the new dashboard CSS. This is the foundation everything else depends on.

**Files:**
- Modify: `web/template.py:28-93` (replace entire `<style>` block contents)

**Step 1: Replace the CSS**

Replace everything between `<style>` and `</style>` (lines 29-92) with the new 3-column layout CSS system. Key components:

- CSS custom properties (`:root` variables) for consistent theming
- `.top-bar` — horizontal bar with logo, command input, mood indicators, greeting
- `.dashboard` — flex container filling viewport below top bar
- `.topic-nav` — 160px left column with `.topic-item` entries and `.badge` counts
- `.main-feed` — flexible center column with scrollable feed
- `.card` system — context-aware cards with `.card-row`, `.card-channel`, `.card-content`, `.card-title`, `.card-meta`, `.card-body`
- `.card.expanded` + `.card-detail` — drill-down expansion
- Priority variants: `.priority-critical` (red border), `.priority-high` (orange border)
- `.sentiment-dot` — green/yellow/red sentiment indicators
- `.chip` — inline tags for action items and domains
- `.ai-sidebar` — 280px right column with `.sidebar-section`, collapsible
- `.prediction-card`, `.person-card`, `.mood-snapshot` — sidebar components
- Responsive breakpoints: 900px (sidebar hides, nav shrinks), 600px (nav becomes horizontal tabs)
- Animations: `slideIn` for new cards, `shimmer` for skeleton loading states

See design doc for full CSS specification.

**Step 2: Verify CSS renders**

Run: `curl -s http://localhost:8080/ | head -5`
Expected: `<!DOCTYPE html>` — page loads without 500 error

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "feat: replace dashboard CSS with 3-column layout system"
```

---

## Task 3: Rewrite HTML Structure — 3-Column Layout

Replace the HTML body in `web/template.py` with the new 3-column structure.

**Files:**
- Modify: `web/template.py:95-127` (replace `<body>` contents, keeping `<script>` untouched for now)

**Step 1: Replace the HTML body**

Replace the single-column layout with a 3-column dashboard:

- **Top bar:** Logo ("Life OS"), full-width command input, mood indicator bars (energy/stress), time-aware greeting, nav links (Admin, DB)
- **Mobile tabs:** Hidden by default, shown below 600px as horizontal scrollable topic tabs
- **Dashboard container:** Flex layout with 3 children:
  - **Topic nav** (left, 160px): 7 topic items (Inbox, Messages, Email, Calendar, Tasks, Insights, System) each with icon, label, badge count. Inbox active by default.
  - **Main feed** (center, flexible): "New items" banner (hidden), response area (for command output), feed header, feed content area with skeleton loading placeholders
  - **AI sidebar** (right, 280px): Collapse toggle button, 4 sections:
    - Daily Briefing with refresh button
    - Predictions
    - People Radar
    - Mood Snapshot (3 bars: energy, stress, social)
- **Status bar** (fixed bottom): Connection dot, status text, event count, connector status

All dynamic content areas have `id` attributes matching the JavaScript selectors from Task 4.

**Step 2: Verify HTML renders**

Run: `curl -s http://localhost:8080/ | grep 'class="dashboard"' | wc -l`
Expected: `1`

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "feat: replace dashboard HTML with 3-column topic stream layout"
```

---

## Task 4: Rewrite JavaScript — Feed Loading, Topic Switching, Card Rendering

Replace the entire `<script>` block with new JS that powers the dashboard.

**Files:**
- Modify: `web/template.py:129-304` (replace entire `<script>` block)

**Step 1: Replace the JavaScript**

The new JS module provides these capabilities:

### Security: HTML Escaping
- `escHtml(s)` — Escapes `&`, `<`, `>`, `"` to HTML entities. **Every** dynamic value passes through this before DOM insertion.
- `escAttr(s)` — Additionally escapes single quotes for attribute contexts.

### Core State
- `currentTopic` — Active topic filter (default: 'inbox')
- `feedItems` — Current feed data array
- `expandedCardId` — ID of currently expanded card (null if none)

### Topic Navigation
- Click handler on `#topicNav` delegates to `switchTopic(topic)`
- `switchTopic()` updates active class, changes header text, resets expansion, calls `loadFeed()`
- Mobile tabs mirror desktop nav behavior

### Feed Loading
- `loadFeed()` — Fetches `/api/dashboard/feed?topic=<current>`, stores items, calls `renderFeed()`
- `loadInsightsFeed()` — Special handler for insights topic, fetches `/api/insights/summary`
- `loadSystemFeed()` — Special handler for system topic, fetches `/health`

### Context-Aware Card Rendering
- `renderCard(item)` — Renders a single feed item as HTML based on its `channel`:
  - **email**: Shows sender, sentiment dot, time ago, body preview. Drill-down: full body, action item chips, "Draft Reply" button, "Create Task" button.
  - **message**: Shows sender, time ago, body preview. Drill-down: full body, "Create Task" button.
  - **calendar**: Shows time range, location. Drill-down: attendees, directions link.
  - **task**: Shows domain chip, due date. Drill-down: full description, "Complete" button, "Dismiss" button.
  - **other**: Generic display with source and time.

### Card Drill-Down
- `toggleCard(id)` — Toggles expansion, re-renders feed
- Escape key collapses any expanded card

### Card Actions
- `completeTask(id)` — POST to `/api/tasks/{id}/complete`
- `dismissCard(id, kind)` — POST to dismiss notification
- `draftReply(id, context)` — POST to `/api/draft`, shows draft in expanded card
- `createTaskFrom(title)` — POST to `/api/tasks`

### AI Sidebar
- `loadBriefing()` — GET `/api/briefing`, renders with skeleton loading
- `loadPredictions()` — GET `/api/insights/summary`, filters for cadence/relationship predictions
- `loadPeopleRadar()` — GET `/api/insights/summary`, filters for relationship_overdue
- `loadMood()` — GET `/api/user-model/mood`, updates energy/stress/social bars

### Badge Counts
- `loadBadges()` — Fetches notifications, tasks, insights counts, updates badge elements
- `setBadge(id, count)` — Shows/hides badge element based on count

### Command Bar
- Enter key handler sends to POST `/api/command`, renders response by type

### Real-Time
- WebSocket connects to `/ws`, triggers `loadFeed()` + `loadBadges()` on events
- Shows "New items" banner when feed scrolled down and new items arrive

### Polling
- Status: every 30s
- Badges: every 60s
- Sidebar (predictions, people, mood): every 60s

### Initial Load
All data functions called on page load: `loadFeed()`, `loadBadges()`, `loadStatus()`, `loadBriefing()`, `loadPredictions()`, `loadPeopleRadar()`, `loadMood()`

**Step 2: Verify the full page loads and is interactive**

Run: `curl -s http://localhost:8080/ | grep 'loadFeed' | wc -l`
Expected: At least `1` — confirms JS is present

Open in browser and verify:
- 3-column layout renders
- Topic clicking switches feed
- Card clicking expands/collapses
- Sidebar shows content or graceful "no data" messages

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "feat: add dashboard JS — feed loading, topic switching, card rendering, sidebar"
```

---

## Task 5: Update the Template Docstring

The module docstring at the top of `web/template.py` describes the old architecture. Update it to describe the new 3-column topic stream dashboard.

**Files:**
- Modify: `web/template.py:1-20` (replace docstring)

**Step 1: Replace the docstring**

Update to describe: 3-column layout (topic nav, main feed, AI sidebar), context-aware card rendering, topic-based filtering via `/api/dashboard/feed`, AI sidebar with briefing/predictions/people/mood, WebSocket real-time updates, responsive breakpoints.

**Step 2: Commit**

```bash
git add web/template.py
git commit -m "docs: update template.py docstring for new dashboard architecture"
```

---

## Task 6: End-to-End Verification and Polish

Smoke-test the complete dashboard and fix any rendering issues.

**Files:**
- Possibly modify: `web/template.py` (minor fixes if needed)
- Possibly modify: `web/routes.py` (endpoint fixes if needed)

**Step 1: Verify API endpoint**

Run: `curl -s http://localhost:8080/api/dashboard/feed | python3 -m json.tool | head -20`
Expected: Valid JSON with `items`, `count`, `topic`

**Step 2: Verify page loads without errors**

Run: `curl -s http://localhost:8080/ -o /dev/null -w '%{http_code}'`
Expected: `200`

**Step 3: Check for Python syntax errors**

Run: `cd /Users/jeremygreenwood/life-os && .venv/bin/python -c "from web.template import HTML_TEMPLATE; print('OK, length:', len(HTML_TEMPLATE))"`
Expected: `OK, length: <number>` — no import errors

**Step 4: Check for JS console errors**

Open `http://localhost:8080/` in browser, open DevTools console. Verify:
- No JS errors
- Network tab shows successful fetches to `/api/dashboard/feed`, `/api/briefing`, `/api/insights/summary`, `/api/user-model/mood`, `/health`
- Topic clicking works (switches feed)
- Card clicking works (expands/collapses)
- Sidebar shows content or graceful "no data" messages

**Step 5: Fix any issues found and commit**

```bash
git add -A
git commit -m "fix: dashboard polish and rendering fixes"
```

---

## Summary

| Task | What | File(s) | Steps |
|------|------|---------|-------|
| 1 | Dashboard feed API endpoint | `web/routes.py` | 3 |
| 2 | CSS rewrite (3-column + cards) | `web/template.py` | 3 |
| 3 | HTML rewrite (layout structure) | `web/template.py` | 3 |
| 4 | JS rewrite (feed + sidebar + real-time) | `web/template.py` | 3 |
| 5 | Update docstring | `web/template.py` | 2 |
| 6 | E2E verification and polish | various | 5 |

Total: 6 tasks, ~19 steps. Tasks 2-5 are sequential (they all modify `web/template.py`). Task 1 is independent and can run first or in parallel.
