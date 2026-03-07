# UI Human-Friendliness Improvements Plan

## Overview

After reviewing the full 4,039-line dashboard template, the iOS app, and the recent backend work, these are targeted improvements to make the web UI more human-friendly. Each change is scoped to `web/template.py` (CSS + JS + HTML) unless noted.

---

## 1. Toast Notification System (replace the awkward `#response` bar)

**Problem:** Feedback for actions (task created, fact confirmed, message sent, etc.) uses a `#response` div wedged between the command bar and the feed. It's easy to miss, doesn't animate, and pushes content down.

**Fix:** Add a floating toast system in the bottom-right corner. Toasts slide in, auto-dismiss after 3-4s, and stack if multiple fire. Replace all `resp.className = 'visible'; resp.textContent = '...'` patterns with `showToast('message', 'success|error|info')`.

**Changes:**
- Add `.toast-container` and `.toast` CSS (fixed bottom-right, slide-in animation, success/error/info variants)
- Add `showToast(message, type)` JS function
- Replace ~15 instances of the response-div pattern with `showToast()` calls
- Keep `#response` div only for command bar long-form AI responses

---

## 2. Empty States with Guidance

**Problem:** Empty feeds show "No items in this topic" — a dead end. New users see this for every tab and have no idea what to do.

**Fix:** Add contextual empty states per topic with an icon, explanation, and a call-to-action pointing to Admin/connectors.

**Changes:**
- In `loadFeed()`, replace the generic empty-state card with topic-specific messages:
  - Inbox: "Your inbox is empty — connect email or messaging in Admin to start seeing items here"
  - Messages: "No messages yet — configure Signal or iMessage in Admin"
  - Email: "No emails yet — connect Gmail or ProtonMail in Admin"
  - Tasks: "No tasks — type 'create task ...' in the command bar or they'll be extracted from your messages automatically"
  - etc.
- Style with a larger icon, centered layout, and a visible "Go to Admin" button

---

## 3. Card Action Feedback (micro-interactions)

**Problem:** When you click "Complete", "Dismiss", or "Confirm", the card just vanishes on next feed reload. No visual acknowledgment that the action worked.

**Fix:** Add a brief card-level transition: the card fades out and slides up when dismissed/completed. Use the toast for confirmation.

**Changes:**
- In `dismissCard()`, `completeTask()`, `actOnNotification()`: before reloading the feed, animate the card out (opacity 0, translateY -10px, 300ms), then remove it from DOM, then show a toast
- Add `.card.removing { opacity: 0; transform: translateY(-10px); transition: all 0.3s; }` CSS

---

## 4. Keyboard Shortcuts

**Problem:** Power users have no keyboard shortcuts. Everything requires clicking.

**Fix:** Add a small set of shortcuts visible via `?` help overlay:
- `r` — Refresh all
- `1-8` — Switch topics (when command bar not focused)
- `/` — Focus command bar
- `Escape` — Collapse expanded card (already exists), close modals

**Changes:**
- Add `keydown` listener on `document` (ignore when input/textarea focused)
- Add a minimal help overlay triggered by `?`
- Add `.keyboard-help` overlay CSS

---

## 5. Relative Time Improvements

**Problem:** `timeAgo()` shows "3d ago" for everything older than 24h until a week, then shows a date. No "yesterday" or day-of-week.

**Fix:** Improve the time display:
- < 1 min: "just now"
- < 1 hour: "5m ago"
- < 24h: "3h ago"
- Yesterday: "yesterday"
- < 7d: "Tuesday" (day name)
- Older: "Mar 1" (short date)

**Changes:**
- Update `timeAgo()` function

---

## 6. Sidebar Section Collapse Memory

**Problem:** The AI sidebar sections (Briefing, Predictions, People Radar, Mood, System Status) are always all open. On mobile especially, this is overwhelming.

**Fix:** Make sidebar sections collapsible with a toggle chevron. Remember collapsed state in `localStorage`.

**Changes:**
- Wrap each `.sidebar-section` title in a clickable header with toggle arrow
- Add JS to toggle section body visibility, persist to `localStorage`
- Default: Briefing and Predictions open, rest collapsed on mobile

---

## 7. Loading States for Card Actions

**Problem:** When clicking "Draft Reply", "Complete", or "Send", there's no visual indicator that the action is processing (except draft reply which shows "Generating draft..."). Buttons remain clickable.

**Fix:** Disable the clicked button and show a spinner/loading text while the request is in-flight.

**Changes:**
- Add a `withLoading(btn, promise)` helper that disables the button, changes text to "...", and re-enables on resolve/reject
- Apply to `completeTask`, `dismissCard`, `actOnNotification`, `confirmFact`, `deleteFact`

---

## Summary

| # | Improvement | Impact | Complexity |
|---|-------------|--------|------------|
| 1 | Toast notifications | High — every action becomes visible | Medium |
| 2 | Empty states with guidance | High — onboarding & discoverability | Low |
| 3 | Card action animations | Medium — feels responsive | Low |
| 4 | Keyboard shortcuts | Medium — power user efficiency | Low |
| 5 | Better relative times | Low — readability polish | Low |
| 6 | Sidebar collapse memory | Medium — less overwhelming | Low |
| 7 | Button loading states | Medium — prevents double-clicks | Low |

All changes are confined to `web/template.py` (inline CSS, HTML, JS). No backend changes, no new files, no new dependencies.
