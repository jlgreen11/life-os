# Topic Stream Dashboard — Design

## Problem

The current front page is a command prompt with flat notification/task cards. Users must type to get value. Meanwhile the backend has a briefing engine, predictions, insights, mood tracking, contact relationship analysis, calendar events, message sentiment analysis, and topic extraction — almost none of which surface automatically.

## Goal

An information-rich, topic-organized dashboard that:
- Shows timely/relevant info immediately on load
- Surfaces topical/important items prominently
- Allows drill-down into detail
- Is context-aware (differentiates email vs Signal vs calendar vs task)
- Streams AI insights via sidebar

## Layout

3-column layout within the existing dark theme:

```
┌──────────────────────────────────────────────────────────────────────┐
│  [Command Bar]                              Mood Bars   Hi Jeremy   │
├─────────┬───────────────────────────────────────┬────────────────────┤
│  TOPICS │       MAIN FEED                       │   AI SIDEBAR       │
│  (nav)  │  (context-aware cards)                │   (intelligence)   │
├─────────┴───────────────────────────────────────┴────────────────────┤
│  Status Bar: connector health, event count, last sync               │
└──────────────────────────────────────────────────────────────────────┘
```

- Topic nav: ~160px, left
- Main feed: flexible center
- AI sidebar: ~280px, right, collapsible
- Responsive: sidebar collapses to toggle, topic nav becomes horizontal tabs on narrow screens

## Topic Navigation

| Topic | Badge Source | Data |
|-------|-------------|------|
| Inbox | Unread/pending across all | Unified priority view |
| Messages | Unread messages | Signal, iMessage — threaded by contact |
| Email | Unread emails | Gmail, Proton — sender, subject, action items |
| Calendar | Today + tomorrow events | Timeline with conflict warnings |
| Tasks | Pending count | Grouped by domain, sorted by due/priority |
| Insights | New insights | Patterns, relationship alerts, cadence |
| System | Error count | Connector health, sync status |

Active topic: left border highlight (#4a9eff). Default: Inbox.

## Context-Aware Cards

### Email Cards
- Sender name, subject line
- Sentiment dot (green/yellow/red based on sentiment score)
- Action items as chips (from AI extraction)
- "Draft reply" button
- Drill-down: full body, thread, AI reply draft, action item checkboxes, "Create task" button

### Message Cards (Signal/iMessage)
- Contact name, mini-thread preview (last few messages)
- Tone indicator
- Quick-reply inline input
- Drill-down: full conversation, reply input, contact profile, last 5 interactions

### Calendar Cards
- Time block visualization, location, attendees
- Conflict warning if overlapping
- Drill-down: full details, attendee list, AI prep notes, directions link

### Task Cards
- Priority color bar (left border), title, domain tag, due date
- Drill-down: full description, related events/contacts, edit, complete button

### Insight Cards
- Pattern summary, confidence indicator, category tag
- Drill-down: explanation, underlying data, useful/dismiss feedback

## AI Sidebar

Top to bottom:

1. **Daily Briefing** — Auto-generated on first load via `/api/briefing`. 3-5 bullets. Refresh button.
2. **Predictions** — Confidence-gated (>= 0.3). Types: REMINDER, CONFLICT, OPPORTUNITY, RISK, NEED. Thumbs up/down feedback.
3. **People Radar** — Contacts overdue for outreach. Name, days since contact, preferred channel. Click to draft.
4. **Mood Snapshot** — Energy, stress, social battery as tiny horizontal bars.

Refreshes every 60s. Collapsible via toggle arrow.

## Real-Time Updates

- WebSocket pushes new events into feed without refresh
- New cards animate in (slide-down)
- Badge counts update live
- "New items" indicator when scrolled down
- Status bar shows live connection health + last sync per connector

## Card Drill-Down

Click any card to expand inline. Escape or click-away collapses. Expansion shows full detail per card type (see Context-Aware Cards above).

## Technical Constraints

- All HTML/CSS/JS embedded in Python strings (`web/template.py`)
- Vanilla JS, no frameworks, no build tools
- Existing API endpoints provide all needed data
- WebSocket already implemented for real-time push
- Dark theme: #0a0a0a background, #1a1a1a cards, #e0e0e0 text
