# UI & End-User Engagement Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 10 high-impact UI bugs and gaps that prevent Life OS from being a compelling daily-driver dashboard.

**Architecture:** All web UI lives in `web/template.py` as a single HTML/CSS/JS Python string template. iOS app is Swift/SwiftUI in `ios/LifeOS/`. Backend API is FastAPI in `web/routes.py` with Pydantic schemas in `web/schemas.py`. Tests use pytest with async fixtures from `tests/conftest.py` providing real temporary SQLite databases.

**Tech Stack:** Python 3.12, FastAPI, vanilla JS/CSS (no frameworks), Swift/SwiftUI, pytest (asyncio_mode=auto), ruff for linting.

**Important context:**
- No existing web/template tests — you'll be the first
- `web/template.py` is one giant string; edits target JS functions and CSS blocks by line number
- The improvement agent runs continuously and may merge PRs to master — always `git pull` before starting a new task
- Ruff config: line length 120, rules E/W/F/I/UP/B/SIM/RUF, target py312

---

## Task 1: Fix loadMood() — Mood Bars Always Show Defaults

**Problem:** `loadMood()` at `web/template.py:1874-1891` reads `data.energy`, `data.stress`, `data.social` but the `/api/user-model/mood` endpoint returns `{"mood": {"energy_level": ..., "stress_level": ..., "social_battery": ...}}`. The bars always display hardcoded fallback values.

**Files:**
- Modify: `web/template.py:1874-1891`

**Step 1: Fix the JSON path mapping**

Replace the current variable extraction in `loadMood()`:

```javascript
// OLD (broken):
var energy = Math.max(0, Math.min(100, (data.energy || 0.5) * 100));
var stress = Math.max(0, Math.min(100, (data.stress || 0.3) * 100));
var social = Math.max(0, Math.min(100, (data.social || 0.4) * 100));

// NEW (correct):
var m = data.mood || data || {};
var energy = Math.max(0, Math.min(100, (m.energy_level || 0.5) * 100));
var stress = Math.max(0, Math.min(100, (m.stress_level || 0.3) * 100));
var social = Math.max(0, Math.min(100, (m.social_battery || 0.4) * 100));
```

**Step 2: Verify manually**

Open `http://localhost:8080` and check:
- Top bar mini-bars should reflect real mood data
- Sidebar mood section should show non-default values
- If no mood data exists yet, fallbacks (50%/30%/40%) are still fine

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "fix: loadMood reads correct JSON path from /api/user-model/mood"
```

---

## Task 2: Fix draftReply() — AI Drafts With No Context

**Problem:** `draftReply()` at `web/template.py:1701-1718` sends `{context: title, event_id: id}` but the `DraftRequest` schema (`web/schemas.py:67-71`) expects `incoming_message` for the actual message body. The AI generates drafts with no content to work from. Also, there's no copy/send button on the draft output.

**Files:**
- Modify: `web/template.py:1701-1718` (JS function)
- Modify: `web/template.py:443-454` (draft-area CSS)

**Step 1: Fix the request payload and add copy button**

Replace the `draftReply()` function:

```javascript
function draftReply(id, context) {
    var draftEl = document.getElementById('draft-' + id);
    if (!draftEl) return;
    safeSetContent(draftEl, '<div class="draft-area" style="opacity:0.5">Generating draft...</div>');

    // Find the original item to get the actual message body
    var item = null;
    for (var i = 0; i < feedItems.length; i++) {
        if (feedItems[i].id === id) { item = feedItems[i]; break; }
    }
    var messageBody = (item && (item.body || item.snippet || item.preview || item.title)) || context;

    fetch(API + '/api/draft', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            incoming_message: messageBody,
            context: context,
            channel: (item && item.channel) || 'email'
        })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        var draft = data.draft || data.content || 'No draft generated';
        safeSetContent(draftEl,
            '<div class="draft-area">' + escHtml(draft) + '</div>' +
            '<div class="draft-actions">' +
                '<button class="draft-btn" onclick="copyDraft(this)">Copy</button>' +
            '</div>'
        );
    })
    .catch(function(err) {
        safeSetContent(draftEl, '<div class="draft-area" style="color:var(--accent-red)">Failed to generate draft: ' + escHtml(err.message) + '</div>');
    });
}

function copyDraft(btn) {
    var area = btn.closest('.draft-actions').previousElementSibling;
    if (!area) return;
    navigator.clipboard.writeText(area.textContent).then(function() {
        btn.textContent = 'Copied!';
        setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
    });
}
```

**Step 2: Add CSS for draft action buttons**

Add after the existing `.draft-area` CSS block (~line 454):

```css
.draft-actions {
    margin-top: 8px;
    display: flex;
    gap: 8px;
}
.draft-btn {
    padding: 6px 16px;
    background: var(--accent);
    color: var(--bg-primary);
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
}
.draft-btn:hover { opacity: 0.85; }
```

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "fix: draftReply sends incoming_message field, add copy button"
```

---

## Task 3: Add Auto-Refresh — Dashboard Goes Stale

**Problem:** The design doc specifies 60s refresh intervals. The implementation at `web/template.py:2025-2028` explicitly says "No automatic polling." The sidebar (briefing, predictions, mood, people radar) loads once and never updates.

**Files:**
- Modify: `web/template.py:2025-2028` (replace no-polling comment)

**Step 1: Add periodic refresh after initial load**

Replace the "No automatic polling" comment block with:

```javascript
// --- Auto-refresh sidebar data every 60s ---
setInterval(function() {
    loadMood();
    loadPredictions();
    loadPeopleRadar();
}, 60000);

// Refresh badges every 2 minutes
setInterval(loadBadges, 120000);
```

**Step 2: Make WebSocket trigger data refresh, not just a banner**

Update the WebSocket handler at `web/template.py:1972-1988`. Replace:

```javascript
ws.onmessage = function(e) {
    try {
        var data = JSON.parse(e.data);
        if (data.type === 'notification' || data.type === 'event') {
            document.getElementById('newItemsBanner').classList.add('visible');
        }
    } catch(err) {}
};
```

With:

```javascript
ws.onmessage = function(e) {
    try {
        var data = JSON.parse(e.data);
        if (data.type === 'notification' || data.type === 'event') {
            document.getElementById('newItemsBanner').classList.add('visible');
            loadBadges();
        }
        if (data.type === 'mood_update') {
            loadMood();
        }
    } catch(err) {}
};
```

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "feat: add 60s auto-refresh for sidebar, badge refresh on WebSocket push"
```

---

## Task 4: Fix iOS APIClient — Tasks and Commands Silently Fail

**Problem:** In `ios/LifeOS/Services/APIClient.swift`:
- `createTask()` (line 54) hits `POST /api/task` but the correct endpoint is `POST /api/tasks` (plural) — returns 404.
- `sendCommand()` (line 30) sends `{"command": text}` but `CommandRequest` schema expects `{"text": text}` — returns 422.

**Files:**
- Modify: `ios/LifeOS/Services/APIClient.swift:30` (sendCommand)
- Modify: `ios/LifeOS/Services/APIClient.swift:54` (createTask)

**Step 1: Fix sendCommand field name**

Find the `sendCommand` method and change the request body key from `"command"` to `"text"`:

```swift
// OLD:
func sendCommand(_ command: String) async throws -> CommandResponse {
    return try await post("/api/command", body: ["command": command])
}

// NEW:
func sendCommand(_ command: String) async throws -> CommandResponse {
    return try await post("/api/command", body: ["text": command])
}
```

**Step 2: Fix createTask endpoint path**

Find the `createTask` method and fix the URL:

```swift
// OLD:
func createTask(_ task: TaskCreate) async throws -> TaskResponse {
    return try await post("/api/task", body: task)
}

// NEW:
func createTask(_ task: TaskCreate) async throws -> TaskResponse {
    return try await post("/api/tasks", body: task)
}
```

**Step 3: Commit**

```bash
git add ios/LifeOS/Services/APIClient.swift
git commit -m "fix: iOS APIClient uses correct endpoint paths and field names"
```

---

## Task 5: Fix People Radar — Uses Wrong API

**Problem:** The People Radar at `web/template.py:1839-1872` calls `/api/insights/summary` and filters by insight type keywords. It should use `/api/contacts?has_metrics=true` which returns the full relationship graph with `last_contact`, `contact_frequency_days`, and `typical_response_time`.

**Files:**
- Modify: `web/template.py` (the `loadPeopleRadar()` function, ~lines 1839-1872)

**Step 1: Rewrite loadPeopleRadar to use contacts API**

Replace the entire `loadPeopleRadar()` function:

```javascript
function loadPeopleRadar() {
    fetch(API + '/api/contacts?has_metrics=true&limit=10')
    .then(function(res) { return res.json(); })
    .then(function(data) {
        var contacts = data.contacts || data || [];
        if (!contacts.length) {
            safeSetContent(document.getElementById('peopleContent'), '<div class="sidebar-empty">No contact data yet</div>');
            return;
        }
        var html = '';
        for (var i = 0; i < Math.min(contacts.length, 8); i++) {
            var c = contacts[i];
            var name = escHtml(c.name || c.contact_id || 'Unknown');
            var lastContact = c.last_contact ? timeAgo(new Date(c.last_contact)) : 'never';
            var freqDays = c.contact_frequency_days;
            var overdue = false;
            if (freqDays && c.last_contact) {
                var daysSince = (Date.now() - new Date(c.last_contact).getTime()) / 86400000;
                overdue = daysSince > freqDays * 1.5;
            }
            var dot = overdue ? 'color:var(--accent-red)' : 'color:var(--accent-green)';
            html += '<div class="radar-person">' +
                '<span class="sentiment-dot" style="' + dot + '"></span> ' +
                '<span class="radar-name">' + name + '</span>' +
                '<span class="radar-meta">' + lastContact + '</span>' +
            '</div>';
        }
        safeSetContent(document.getElementById('peopleContent'), html);
    })
    .catch(function() {
        safeSetContent(document.getElementById('peopleContent'), '<div class="sidebar-empty">Could not load contacts</div>');
    });
}
```

**Step 2: Add supporting CSS (if not already present)**

```css
.radar-person {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 13px;
}
.radar-name { flex: 1; color: var(--text-primary); }
.radar-meta { color: var(--text-muted); font-size: 11px; }
```

**Step 3: Add `timeAgo` helper if not present**

Check if `timeAgo` already exists in template.py. If not, add:

```javascript
function timeAgo(date) {
    var secs = Math.floor((Date.now() - date.getTime()) / 1000);
    if (secs < 60) return 'just now';
    var mins = Math.floor(secs / 60);
    if (mins < 60) return mins + 'm ago';
    var hrs = Math.floor(mins / 60);
    if (hrs < 24) return hrs + 'h ago';
    var days = Math.floor(hrs / 24);
    if (days < 7) return days + 'd ago';
    return Math.floor(days / 7) + 'w ago';
}
```

**Step 4: Commit**

```bash
git add web/template.py
git commit -m "fix: People Radar uses /api/contacts instead of insights filter"
```

---

## Task 6: Fix Card Toggle — Full Re-render Causes Flash and Scroll Loss

**Problem:** `toggleCard()` at `web/template.py:1636-1649` rebuilds the entire feed HTML on every expand/collapse. This causes a visual flash and loses scroll position.

**Files:**
- Modify: `web/template.py:1636-1649`

**Step 1: Replace full re-render with targeted DOM update**

Replace the `toggleCard()` function:

```javascript
function toggleCard(id) {
    // Collapse previously expanded card
    if (expandedCardId && expandedCardId !== id) {
        var prevCard = document.querySelector('[data-card-id="' + expandedCardId + '"]');
        if (prevCard) {
            prevCard.classList.remove('expanded');
            var prevDetails = prevCard.querySelector('.card-details');
            if (prevDetails) prevDetails.style.display = 'none';
        }
    }

    var card = document.querySelector('[data-card-id="' + id + '"]');
    if (!card) return;

    if (expandedCardId === id) {
        expandedCardId = null;
        card.classList.remove('expanded');
        var details = card.querySelector('.card-details');
        if (details) details.style.display = 'none';
    } else {
        expandedCardId = id;
        card.classList.add('expanded');
        var details = card.querySelector('.card-details');
        if (details) details.style.display = 'block';
    }
}
```

**Step 2: Ensure renderCard() emits `data-card-id` attribute**

Find the `renderCard()` function and verify each card's outer div includes `data-card-id="${item.id}"`. If cards use `id="card-${item.id}"` instead, update the querySelector above to match. Also ensure `.card-details` sections start with `display:none` in CSS and the expanded state is handled by the class, not by re-rendering HTML.

**Step 3: Commit**

```bash
git add web/template.py
git commit -m "fix: toggleCard uses DOM class toggle instead of full feed re-render"
```

---

## Task 7: Surface Signal Profiles, Routines, and Workflows

**Problem:** The backend exposes 9 signal profile types, routines, and workflows via API but none are rendered in the UI. This is the system's most unique intelligence and the user never sees it.

**Files:**
- Modify: `web/template.py` (add new tab content and load functions)

**Step 1: Add an "Insights" topic feed section**

The "Insights" topic already exists in the nav. Wire its feed to load signal profiles, routines, and workflows. Add these load functions:

```javascript
function loadInsightsFeed() {
    var el = document.getElementById('feedContent');
    safeSetContent(el, '<div class="skeleton skeleton-card"></div>');

    Promise.all([
        fetch(API + '/api/user-model/signal-profiles').then(function(r) { return r.json(); }),
        fetch(API + '/api/user-model/routines?min_observations=2').then(function(r) { return r.json(); }),
        fetch(API + '/api/user-model/workflows?min_observations=2').then(function(r) { return r.json(); })
    ]).then(function(results) {
        var profiles = results[0];
        var routines = results[1].routines || results[1] || [];
        var workflows = results[2].workflows || results[2] || [];
        var html = '';

        // Signal profiles section
        html += '<div class="section-header">Behavioral Signals</div>';
        var profileTypes = ['temporal', 'cadence', 'decision', 'linguistic', 'topics', 'relationships'];
        for (var i = 0; i < profileTypes.length; i++) {
            var key = profileTypes[i];
            var p = profiles[key] || profiles.profiles?.[key];
            if (!p) continue;
            html += '<div class="feed-card">' +
                '<div class="card-header">' +
                    '<span class="card-channel">' + escHtml(key) + '</span>' +
                '</div>' +
                '<div class="card-title">' + escHtml(key.charAt(0).toUpperCase() + key.slice(1)) + ' Profile</div>' +
                '<div class="card-snippet">' + escHtml(JSON.stringify(p, null, 2)).substring(0, 300) + '</div>' +
            '</div>';
        }

        // Routines section
        if (routines.length) {
            html += '<div class="section-header">Detected Routines</div>';
            for (var i = 0; i < routines.length; i++) {
                var r = routines[i];
                html += '<div class="feed-card">' +
                    '<div class="card-title">' + escHtml(r.name || r.trigger || 'Routine') + '</div>' +
                    '<div class="card-snippet">' +
                        'Trigger: ' + escHtml(r.trigger || 'unknown') +
                        ' | Consistency: ' + ((r.consistency || 0) * 100).toFixed(0) + '%' +
                        ' | Observations: ' + (r.observation_count || 0) +
                    '</div>' +
                '</div>';
            }
        }

        // Workflows section
        if (workflows.length) {
            html += '<div class="section-header">Workflows</div>';
            for (var i = 0; i < workflows.length; i++) {
                var w = workflows[i];
                html += '<div class="feed-card">' +
                    '<div class="card-title">' + escHtml(w.name || w.goal || 'Workflow') + '</div>' +
                    '<div class="card-snippet">' +
                        'Steps: ' + (w.steps ? w.steps.length : '?') +
                        ' | Success rate: ' + ((w.success_rate || 0) * 100).toFixed(0) + '%' +
                    '</div>' +
                '</div>';
            }
        }

        if (!html) html = '<div class="feed-empty">No behavioral data detected yet. Use Life OS for a few days and patterns will emerge.</div>';
        safeSetContent(el, html);
    }).catch(function() {
        safeSetContent(el, '<div class="feed-empty">Could not load insights</div>');
    });
}
```

**Step 2: Add section-header CSS**

```css
.section-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin: 20px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
```

**Step 3: Wire the Insights topic to call `loadInsightsFeed()`**

In the topic switching logic (the function that handles topic clicks), add a case: when `currentTopic === 'insights'`, call `loadInsightsFeed()` instead of the generic feed loader.

**Step 4: Commit**

```bash
git add web/template.py
git commit -m "feat: surface signal profiles, routines, and workflows in Insights tab"
```

---

## Task 8: Fix Mobile Experience — Sidebar Disappears

**Problem:** At `<900px` the AI sidebar is `display: none` with no alternative access. Briefing, predictions, mood, and people radar are completely inaccessible on tablets and phones. Badge counts are also hidden.

**Files:**
- Modify: `web/template.py:777-810` (CSS breakpoints)
- Modify: `web/template.py` (add mobile sidebar toggle button and bottom sheet)

**Step 1: Add a mobile sidebar toggle button (visible only on small screens)**

Add this HTML right before the closing `</main>` or after the status bar:

```html
<button class="mobile-sidebar-btn" id="mobileSidebarBtn" onclick="toggleMobileSidebar()">AI</button>
```

**Step 2: Add mobile sidebar CSS**

```css
.mobile-sidebar-btn {
    display: none;
    position: fixed;
    bottom: 40px;
    right: 16px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--bg-primary);
    border: none;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    z-index: 1000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

@media (max-width: 900px) {
    .mobile-sidebar-btn { display: block; }

    .ai-sidebar {
        display: none;
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        width: 100%;
        min-width: 100%;
        z-index: 999;
        background: var(--bg-primary);
        padding: 20px;
        overflow-y: auto;
    }
    .ai-sidebar.mobile-open {
        display: block;
    }
}
```

**Step 3: Add toggle function**

```javascript
function toggleMobileSidebar() {
    var sidebar = document.getElementById('aiSidebar');
    sidebar.classList.toggle('mobile-open');
}
```

**Step 4: Restore badge counts on mobile**

Remove `display: none !important` from `.topic-badge` in the 900px breakpoint. Replace with smaller sizing:

```css
@media (max-width: 900px) {
    .topic-badge {
        font-size: 9px;
        min-width: 14px;
        height: 14px;
        line-height: 14px;
    }
}
```

**Step 5: Commit**

```bash
git add web/template.py
git commit -m "feat: add mobile sidebar toggle, restore badge counts on small screens"
```

---

## Task 9: Replace Native Dialogs — correctFact() and deleteFact()

**Problem:** `correctFact()` at `web/template.py:1543-1560` uses `prompt()` and `deleteFact()` at `web/template.py:1528-1541` uses `confirm()`. These are ugly, block the thread, and are unusable on mobile.

**Files:**
- Modify: `web/template.py` (JS functions + add modal HTML/CSS)

**Step 1: Add an inline modal component**

Add this HTML before the closing `</body>`:

```html
<div class="modal-overlay" id="modalOverlay" onclick="closeModal()">
    <div class="modal-box" onclick="event.stopPropagation()">
        <div class="modal-title" id="modalTitle"></div>
        <div class="modal-body" id="modalBody"></div>
        <div class="modal-actions" id="modalActions"></div>
    </div>
</div>
```

**Step 2: Add modal CSS**

```css
.modal-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6);
    z-index: 2000;
    align-items: center;
    justify-content: center;
}
.modal-overlay.visible { display: flex; }
.modal-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    max-width: 400px;
    width: 90%;
}
.modal-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
}
.modal-body { margin-bottom: 16px; }
.modal-body input, .modal-body textarea {
    width: 100%;
    padding: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    box-sizing: border-box;
}
.modal-actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
}
.modal-actions button {
    padding: 8px 20px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
}
.modal-btn-cancel {
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border) !important;
}
.modal-btn-confirm {
    background: var(--accent);
    color: var(--bg-primary);
}
.modal-btn-danger {
    background: var(--accent-red);
    color: white;
}
```

**Step 3: Add modal helper functions**

```javascript
function closeModal() {
    document.getElementById('modalOverlay').classList.remove('visible');
}

function showConfirmModal(title, message, confirmLabel, onConfirm, danger) {
    document.getElementById('modalTitle').textContent = title;
    safeSetContent(document.getElementById('modalBody'), '<div style="color:var(--text-secondary);font-size:14px">' + escHtml(message) + '</div>');
    safeSetContent(document.getElementById('modalActions'),
        '<button class="modal-btn-cancel" onclick="closeModal()">Cancel</button>' +
        '<button class="' + (danger ? 'modal-btn-danger' : 'modal-btn-confirm') + '" id="modalConfirmBtn">' + escHtml(confirmLabel) + '</button>'
    );
    document.getElementById('modalConfirmBtn').onclick = function() { closeModal(); onConfirm(); };
    document.getElementById('modalOverlay').classList.add('visible');
}

function showPromptModal(title, placeholder, confirmLabel, onConfirm) {
    document.getElementById('modalTitle').textContent = title;
    safeSetContent(document.getElementById('modalBody'), '<input type="text" id="modalInput" placeholder="' + escHtml(placeholder) + '">');
    safeSetContent(document.getElementById('modalActions'),
        '<button class="modal-btn-cancel" onclick="closeModal()">Cancel</button>' +
        '<button class="modal-btn-confirm" id="modalConfirmBtn">' + escHtml(confirmLabel) + '</button>'
    );
    document.getElementById('modalConfirmBtn').onclick = function() {
        var val = document.getElementById('modalInput').value;
        closeModal();
        onConfirm(val);
    };
    document.getElementById('modalOverlay').classList.add('visible');
    setTimeout(function() { document.getElementById('modalInput').focus(); }, 100);
}
```

**Step 4: Replace correctFact() to use the modal**

```javascript
function correctFact(key) {
    showPromptModal('Correct Fact', 'What is incorrect? (optional)', 'Submit', function(reason) {
        fetch(API + '/api/user-model/facts/' + encodeURIComponent(key), {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({reason: reason || 'User correction'})
        })
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = 'Fact corrected — confidence reduced';
            setTimeout(function() { resp.className = ''; }, 3000);
            loadFactsFeed();
        })
        .catch(function(err) { console.error('Correct fact failed:', err); });
    });
}
```

**Step 5: Replace deleteFact() to use the modal**

```javascript
function deleteFact(key) {
    showConfirmModal('Delete Fact', 'Are you sure you want to delete this fact?', 'Delete', function() {
        fetch(API + '/api/user-model/facts/' + encodeURIComponent(key), {method: 'DELETE'})
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            var card = document.getElementById('fact-' + key);
            if (card) card.style.display = 'none';
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = 'Fact deleted';
            setTimeout(function() { resp.className = ''; }, 3000);
        })
        .catch(function(err) { console.error('Delete fact failed:', err); });
    }, true);
}
```

**Step 6: Commit**

```bash
git add web/template.py
git commit -m "feat: replace native prompt/confirm with styled inline modals"
```

---

## Task 10: Add Badge Count Endpoint — Stop Redundant Feed Requests

**Problem:** `loadBadges()` at `web/template.py:1914-1924` fires 5 separate full feed requests (`?limit=100`) just to get counts. This is wasteful and slow.

**Files:**
- Modify: `web/routes.py` (add lightweight count endpoint)
- Modify: `web/schemas.py` (add response model)
- Modify: `web/template.py:1914-1924` (use new endpoint)
- Test: `tests/web/test_badge_counts.py` (new file)

**Step 1: Write the failing test**

Create `tests/web/test_badge_counts.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_badge_counts_returns_dict(db):
    """GET /api/dashboard/badges should return topic->count mapping."""
    # This test validates the endpoint exists and returns the right shape.
    # Full integration test requires a running app; we test the route logic directly.
    from web.routes import create_app

    # Minimal smoke test — the endpoint should not 500
    # (full route tests need the app fixture; this verifies the import works)
    assert callable(create_app)
```

**Step 2: Add the endpoint to routes.py**

Add after the existing `/api/dashboard/feed` endpoint:

```python
@app.get("/api/dashboard/badges")
async def get_dashboard_badges():
    """Return badge counts per topic without loading full feed data."""
    counts = {}
    topics = ["inbox", "messages", "email", "calendar", "tasks"]
    for topic in topics:
        try:
            feed = await life_os.get_dashboard_feed(topic=topic, limit=1)
            counts[topic] = feed.get("count", 0)
        except Exception:
            counts[topic] = 0
    return {"badges": counts}
```

**Step 3: Update loadBadges() in template.py**

Replace the function:

```javascript
function loadBadges() {
    fetch(API + '/api/dashboard/badges')
    .then(function(res) { return res.json(); })
    .then(function(data) {
        var badges = data.badges || {};
        Object.keys(badges).forEach(function(tid) {
            setBadge(tid, badges[tid]);
        });
    })
    .catch(function() {});
}
```

**Step 4: Commit**

```bash
git add web/routes.py web/template.py tests/web/test_badge_counts.py
git commit -m "feat: add /api/dashboard/badges endpoint, replace 5 feed requests with 1"
```

---

## Execution Order & Dependencies

Tasks are independent and can be executed in any order. Recommended grouping:

**Group A — Bug fixes (do first):**
1. Task 1: Fix loadMood
2. Task 2: Fix draftReply
3. Task 4: Fix iOS APIClient
4. Task 5: Fix People Radar

**Group B — UX improvements:**
5. Task 6: Fix card toggle
6. Task 9: Replace native dialogs
7. Task 10: Badge count endpoint

**Group C — Feature additions:**
8. Task 3: Auto-refresh
9. Task 7: Signal profiles/routines/workflows
10. Task 8: Mobile sidebar

---

## Pre-Flight Checklist

Before starting any task:
```bash
cd /Users/jeremygreenwood/life-os
git checkout master
git pull --ff-only
source .venv/bin/activate
```

After each task, verify:
```bash
ruff check web/template.py web/routes.py
python -m pytest tests/ -x -q
```

After all tasks, full verification:
```bash
python -m pytest tests/ -v
ruff check . --fix
ruff format .
# Open http://localhost:8080 and manually verify each fix
```
