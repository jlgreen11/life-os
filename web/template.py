"""
Life OS -- 3-Column Topic Stream Dashboard

Single-page dashboard served at the root URL ("/").  Everything -- HTML, CSS,
and JavaScript -- is embedded in a single Python string constant so the web
module has zero static-file dependencies.

Architecture:
    Layout  -- 3-column flexbox: topic navigation (left 160px), main feed
               (center, flexible), AI sidebar (right 280px, collapsible).
               A top bar with command input, mood indicators, and nav links
               spans the full width.  A fixed status bar at the bottom shows
               connection health and event count.

    CSS     -- Dark-theme design system using CSS custom properties in :root.
               Card-based layout with context-aware rendering.  Priority
               variants add colored left borders (red for critical, orange
               for high).  Responsive breakpoints at 900px (sidebar hides,
               nav shrinks) and 600px (nav hides, mobile tabs appear).
               Animations: slideIn for new cards, shimmer for skeleton
               loading placeholders.

    HTML    -- Semantic structure: top-bar, 3-column dashboard (topic-nav,
               main-feed, ai-sidebar), status-bar, mobile-tabs.

    JS      -- Client-side logic handles:
               1. Topic navigation: switching between inbox, messages, email,
                  calendar, tasks, insights, system topics.
               2. Feed loading: fetches /api/dashboard/feed?topic=<current>,
                  renders context-aware cards based on channel type.
               3. Card expansion: click to expand with drill-down details,
                  action buttons (reply, complete, dismiss, create task).
               4. AI sidebar: briefing, predictions, people radar, mood
                  snapshot -- each with independent refresh.
               5. Command bar: sends input to POST /api/command on Enter.
               6. WebSocket: connects to /ws; shows "New items" banner on
                  push updates (user clicks to refresh -- no auto-reload).
               7. Refresh-on-open only: all data loads once when the page
                  opens.  A manual "Refresh" button reloads everything.
               8. Security: all dynamic content passes through escHtml()
                  before DOM insertion.
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life OS</title>
    <style>
        /* ================================================================
           CSS Design System -- Dark Theme
           ================================================================ */

        :root {
            --bg-primary: #0a0a0a;
            --bg-card: #1a1a1a;
            --bg-card-hover: #222;
            --bg-sidebar: #111;
            --border: #222;
            --border-hover: #444;
            --border-active: #4a9eff;
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent-blue: #4a9eff;
            --accent-green: #4aff6b;
            --accent-orange: #ff6b35;
            --accent-red: #ff3535;
            --accent-yellow: #ffc107;
            --accent-purple: #a855f7;
            --font-stack: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
        }

        /* --- Reset & Base --- */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-stack);
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow: hidden;
        }
        a { color: var(--accent-blue); text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* --- Top Bar --- */
        .top-bar {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 20px;
            background: var(--bg-sidebar);
            border-bottom: 1px solid var(--border);
            height: 56px;
            z-index: 100;
        }
        .logo {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            white-space: nowrap;
            letter-spacing: -0.5px;
        }
        .command-bar {
            flex: 1;
            max-width: 480px;
        }
        .command-bar input {
            width: 100%;
            padding: 8px 14px;
            font-size: 14px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: #fff;
            outline: none;
            transition: border-color 0.2s;
            font-family: var(--font-stack);
        }
        .command-bar input:focus { border-color: var(--border-active); }
        .command-bar input::placeholder { color: var(--text-muted); }
        .mood-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .mood-bar .mini-bar {
            width: 40px;
            height: 4px;
            background: var(--border);
            border-radius: 2px;
            overflow: hidden;
        }
        .mood-bar .mini-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }
        .greeting {
            font-size: 13px;
            color: var(--text-secondary);
            white-space: nowrap;
        }
        .nav-links {
            display: flex;
            gap: 12px;
            margin-left: auto;
            white-space: nowrap;
        }
        .nav-links a {
            font-size: 12px;
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 4px;
            transition: background 0.2s;
        }
        .nav-links a:hover {
            background: var(--bg-card);
            color: var(--text-primary);
            text-decoration: none;
        }

        /* --- Mobile Tabs (hidden by default) --- */
        .mobile-tabs {
            display: none;
            overflow-x: auto;
            white-space: nowrap;
            background: var(--bg-sidebar);
            border-bottom: 1px solid var(--border);
            padding: 0 12px;
        }
        .mobile-tab {
            display: inline-block;
            padding: 10px 14px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: color 0.2s, border-color 0.2s;
        }
        .mobile-tab.active {
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }

        /* --- 3-Column Dashboard --- */
        .dashboard {
            display: flex;
            height: calc(100vh - 73px);
        }

        /* --- Left: Topic Nav --- */
        .topic-nav {
            width: 160px;
            min-width: 160px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            padding: 12px 0;
            overflow-y: auto;
        }
        .topic-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            cursor: pointer;
            font-size: 13px;
            color: var(--text-secondary);
            border-left: 3px solid transparent;
            transition: all 0.15s;
            user-select: none;
        }
        .topic-item:hover {
            color: var(--text-primary);
            background: rgba(255,255,255,0.03);
        }
        .topic-item.active {
            color: var(--accent-blue);
            border-left-color: var(--accent-blue);
            background: rgba(74,158,255,0.05);
        }
        .topic-icon { font-size: 15px; width: 20px; text-align: center; }
        .topic-label { flex: 1; }
        .topic-badge {
            font-size: 10px;
            background: var(--accent-blue);
            color: #fff;
            padding: 1px 6px;
            border-radius: 8px;
            display: none;
            min-width: 18px;
            text-align: center;
        }
        .topic-badge.visible { display: inline-block; }

        /* --- Center: Main Feed --- */
        .main-feed {
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px 40px;
            scroll-behavior: smooth;
        }
        .feed-header {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .feed-header .refresh-btn {
            background: none;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 13px;
            cursor: pointer;
            transition: color 0.15s, border-color 0.15s;
        }
        .feed-header .refresh-btn:hover {
            color: #fff;
            border-color: var(--accent-blue);
        }
        .new-items-banner {
            display: none;
            position: sticky;
            top: 0;
            z-index: 50;
            text-align: center;
            padding: 8px;
            background: var(--accent-blue);
            color: #fff;
            border-radius: 8px;
            margin-bottom: 12px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }
        .new-items-banner.visible { display: block; }
        #response {
            background: var(--bg-sidebar);
            border-radius: 10px;
            padding: 16px 20px;
            min-height: 60px;
            white-space: pre-wrap;
            line-height: 1.6;
            display: none;
            margin-bottom: 16px;
            font-size: 14px;
        }
        #response.visible { display: block; }
        .loading-text { opacity: 0.5; }

        /* --- Cards --- */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            animation: slideIn 0.2s ease-out;
        }
        .card:hover {
            border-color: var(--border-hover);
            background: var(--bg-card-hover);
        }
        .card-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }
        .card-channel {
            font-size: 16px;
            width: 24px;
            text-align: center;
            flex-shrink: 0;
            padding-top: 2px;
        }
        .card-content { flex: 1; min-width: 0; }
        .card-title {
            font-weight: 500;
            font-size: 14px;
            margin-bottom: 3px;
            color: var(--text-primary);
        }
        .card-meta {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .card-body {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.5;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* Priority variants */
        .priority-critical { border-left: 3px solid var(--accent-red); }
        .priority-high { border-left: 3px solid var(--accent-orange); }

        /* Expanded state */
        .card.expanded {
            border-color: var(--border-active);
            background: var(--bg-card-hover);
        }
        .card.expanded .card-body {
            -webkit-line-clamp: unset;
            overflow: visible;
        }
        .card-detail {
            display: none;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
            font-size: 13px;
            line-height: 1.6;
            color: var(--text-secondary);
        }
        .card.expanded .card-detail { display: block; }
        .card-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .card-actions button {
            padding: 6px 14px;
            font-size: 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-card);
            color: var(--text-primary);
            cursor: pointer;
            font-family: var(--font-stack);
            transition: all 0.15s;
        }
        .card-actions button:hover {
            border-color: var(--accent-blue);
            background: rgba(74,158,255,0.1);
        }
        .card-actions button.btn-primary {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: #fff;
        }
        .card-actions button.btn-danger {
            border-color: var(--accent-red);
            color: var(--accent-red);
        }
        .card-actions button.btn-danger:hover {
            background: rgba(255,53,53,0.1);
        }
        .card-actions button.btn-small {
            padding: 3px 10px;
            font-size: 11px;
        }
        .card-actions button.btn-warn {
            border-color: var(--accent-yellow, #f0a020);
            color: var(--accent-yellow, #f0a020);
        }
        .card-actions button.btn-warn:hover {
            background: rgba(240,160,32,0.1);
        }

        /* Sentiment dots */
        .sentiment-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-left: 6px;
            vertical-align: middle;
        }
        .sentiment-positive { background: var(--accent-green); }
        .sentiment-neutral { background: var(--accent-yellow); }
        .sentiment-negative { background: var(--accent-red); }

        /* Chips */
        .chip {
            display: inline-block;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(74,158,255,0.12);
            color: var(--accent-blue);
            margin-right: 4px;
        }
        .chip.chip-domain {
            background: rgba(168,85,247,0.12);
            color: var(--accent-purple);
        }

        /* Draft reply area */
        .draft-area {
            margin-top: 10px;
            padding: 12px;
            background: var(--bg-primary);
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.5;
            color: var(--text-primary);
            white-space: pre-wrap;
        }
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

        /* --- Right: AI Sidebar --- */
        .ai-sidebar {
            width: 280px;
            min-width: 280px;
            background: var(--bg-sidebar);
            border-left: 1px solid var(--border);
            overflow-y: auto;
            padding: 16px;
            position: relative;
            transition: width 0.2s, min-width 0.2s, padding 0.2s;
        }
        .ai-sidebar.collapsed {
            width: 0;
            min-width: 0;
            padding: 0;
            overflow: hidden;
        }
        .sidebar-toggle {
            position: absolute;
            top: 12px;
            left: -24px;
            width: 24px;
            height: 28px;
            background: var(--bg-sidebar);
            border: 1px solid var(--border);
            border-right: none;
            border-radius: 6px 0 0 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 10px;
            color: var(--text-muted);
            z-index: 10;
        }
        .sidebar-toggle:hover { color: var(--text-primary); }
        .sidebar-section {
            margin-bottom: 20px;
        }
        .sidebar-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .sidebar-title button {
            font-size: 10px;
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 2px 4px;
            font-family: var(--font-stack);
        }
        .sidebar-title button:hover { color: var(--accent-blue); }
        .sidebar-content {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .prediction-card {
            padding: 10px 12px;
            background: var(--bg-card);
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 12px;
            line-height: 1.5;
        }
        .prediction-card .pred-label {
            color: var(--text-muted);
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .person-card {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }
        .person-card:last-child { border-bottom: none; }
        .person-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--bg-card);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: var(--text-muted);
            flex-shrink: 0;
        }
        .person-info { flex: 1; }
        .person-name { color: var(--text-primary); font-weight: 500; }
        .person-detail { color: var(--text-muted); font-size: 11px; }
        .mood-snapshot { padding: 4px 0; }
        .mood-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .mood-label {
            width: 50px;
            color: var(--text-muted);
            text-align: right;
        }
        .mood-track {
            flex: 1;
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
        }
        .mood-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s;
        }
        .mood-fill.energy { background: var(--accent-green); }
        .mood-fill.stress { background: var(--accent-orange); }
        .mood-fill.social { background: var(--accent-purple); }

        /* --- Status Bar --- */
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-sidebar);
            border-top: 1px solid var(--border);
            padding: 6px 20px;
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 28px;
            z-index: 100;
        }
        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .status-dot.ok { background: var(--accent-green); }
        .status-dot.error { background: var(--accent-red); }
        .status-items {
            display: flex;
            gap: 16px;
        }

        /* --- Skeleton Loading --- */
        .skeleton {
            background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-card-hover) 50%, var(--bg-card) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 6px;
        }
        .skeleton-card {
            height: 72px;
            margin-bottom: 8px;
            border-radius: 10px;
        }
        .skeleton-line {
            height: 12px;
            margin-bottom: 8px;
        }
        .skeleton-line.short { width: 60%; }

        /* --- Calendar Grid --- */
        .calendar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .calendar-header .cal-nav {
            background: none;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 16px;
            cursor: pointer;
            font-family: var(--font-stack);
            transition: color 0.15s, border-color 0.15s;
        }
        .calendar-header .cal-nav:hover {
            color: #fff;
            border-color: var(--accent-blue);
        }
        .calendar-header .cal-title {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
        }
        .calendar-weekdays {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            text-align: center;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 4px;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 1px;
            background: var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .calendar-day {
            background: var(--bg-card);
            min-height: 90px;
            padding: 6px;
            cursor: pointer;
            transition: background 0.15s;
            position: relative;
        }
        .calendar-day:hover {
            background: var(--bg-card-hover);
        }
        .calendar-day.calendar-today {
            box-shadow: inset 0 0 0 2px var(--accent-blue);
        }
        .calendar-day.calendar-other-month {
            opacity: 0.35;
        }
        .calendar-day.calendar-selected {
            background: var(--bg-card-hover);
            box-shadow: inset 0 0 0 2px var(--accent-blue);
        }
        .calendar-day-number {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .calendar-today .calendar-day-number {
            color: var(--accent-blue);
            font-weight: 700;
        }
        .calendar-event {
            font-size: 10px;
            padding: 2px 4px;
            border-radius: 3px;
            margin-bottom: 2px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            line-height: 1.4;
        }
        .calendar-event.all-day {
            background: rgba(74,158,255,0.2);
            color: var(--accent-blue);
            font-weight: 500;
        }
        .calendar-event.timed {
            background: rgba(168,85,247,0.15);
            color: var(--accent-purple);
        }
        .calendar-event-more {
            font-size: 10px;
            color: var(--text-muted);
            padding: 1px 4px;
        }
        .calendar-day-detail {
            margin-top: 12px;
            animation: slideIn 0.2s ease-out;
        }
        .calendar-detail-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .calendar-detail-card .cal-evt-title {
            font-weight: 500;
            font-size: 14px;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        .calendar-detail-card .cal-evt-time {
            font-size: 12px;
            color: var(--accent-blue);
            margin-bottom: 4px;
        }
        .calendar-detail-card .cal-evt-meta {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* --- Animations --- */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* --- Responsive: 900px --- */
        @media (max-width: 900px) {
            .ai-sidebar { display: none; }
            .sidebar-toggle { display: none; }
            .topic-nav {
                width: 50px;
                min-width: 50px;
            }
            .topic-label { display: none; }
            .topic-badge { display: none !important; }
            .topic-item {
                justify-content: center;
                padding: 10px 0;
            }
            .topic-icon { margin: 0; }
            .greeting { display: none; }
            .mood-bar { display: none; }
        }

        /* --- Responsive: 600px --- */
        @media (max-width: 600px) {
            .topic-nav { display: none; }
            .mobile-tabs { display: flex; }
            .main-feed { padding: 12px 14px 40px; }
            .command-bar { max-width: none; }
            .nav-links { display: none; }
        }
    </style>
</head>
<body>

    <!-- Top Bar -->
    <div class="top-bar">
        <div class="logo">Life OS</div>
        <div class="command-bar">
            <input type="text" id="commandInput"
                   placeholder="Ask anything, search, create tasks..."
                   autofocus autocomplete="off">
        </div>
        <div class="mood-bar" id="moodBar">
            <span>E</span>
            <div class="mini-bar"><div class="mini-bar-fill energy" id="miniEnergy" style="width:50%;background:var(--accent-green)"></div></div>
            <span>S</span>
            <div class="mini-bar"><div class="mini-bar-fill stress" id="miniStress" style="width:30%;background:var(--accent-orange)"></div></div>
        </div>
        <div class="greeting" id="greeting"></div>
        <div class="nav-links">
            <a href="/admin">Admin</a>
            <a href="/admin/db">DB</a>
        </div>
    </div>

    <!-- Mobile Tabs (hidden by default) -->
    <div class="mobile-tabs" id="mobileTabs"></div>

    <!-- 3-Column Dashboard -->
    <div class="dashboard">

        <!-- Left: Topic Nav -->
        <nav class="topic-nav" id="topicNav">
            <div class="topic-item active" data-topic="inbox">
                <span class="topic-icon">&#9776;</span>
                <span class="topic-label">Inbox</span>
                <span class="topic-badge" id="badge-inbox"></span>
            </div>
            <div class="topic-item" data-topic="messages">
                <span class="topic-icon">&#9993;</span>
                <span class="topic-label">Messages</span>
                <span class="topic-badge" id="badge-messages"></span>
            </div>
            <div class="topic-item" data-topic="email">
                <span class="topic-icon">&#9993;</span>
                <span class="topic-label">Email</span>
                <span class="topic-badge" id="badge-email"></span>
            </div>
            <div class="topic-item" data-topic="calendar">
                <span class="topic-icon">&#128197;</span>
                <span class="topic-label">Calendar</span>
                <span class="topic-badge" id="badge-calendar"></span>
            </div>
            <div class="topic-item" data-topic="tasks">
                <span class="topic-icon">&#9745;</span>
                <span class="topic-label">Tasks</span>
                <span class="topic-badge" id="badge-tasks"></span>
            </div>
            <div class="topic-item" data-topic="insights">
                <span class="topic-icon">&#9733;</span>
                <span class="topic-label">Insights</span>
                <span class="topic-badge" id="badge-insights"></span>
            </div>
            <div class="topic-item" data-topic="profile">
                <span class="topic-icon">&#128100;</span>
                <span class="topic-label">My Profile</span>
                <span class="topic-badge" id="badge-profile"></span>
            </div>
            <div class="topic-item" data-topic="system">
                <span class="topic-icon">&#9881;</span>
                <span class="topic-label">System</span>
                <span class="topic-badge" id="badge-system"></span>
            </div>
        </nav>

        <!-- Center: Main Feed -->
        <main class="main-feed" id="mainFeed">
            <div class="new-items-banner" id="newItemsBanner" onclick="scrollToTop()">New items available</div>
            <div id="response"></div>
            <div class="feed-header"><span id="feedHeader">Inbox</span><button class="refresh-btn" onclick="refreshAll()" title="Refresh">&#8635; Refresh</button></div>
            <div id="feedContent">
                <div class="skeleton skeleton-card"></div>
                <div class="skeleton skeleton-card"></div>
                <div class="skeleton skeleton-card"></div>
                <div class="skeleton skeleton-card"></div>
            </div>
        </main>

        <!-- Right: AI Sidebar -->
        <aside class="ai-sidebar" id="aiSidebar">
            <div class="sidebar-toggle" id="sidebarToggle" onclick="toggleSidebar()">&#9664;</div>

            <div class="sidebar-section">
                <div class="sidebar-title">
                    Daily Briefing
                    <button onclick="loadBriefing()">&#8635; refresh</button>
                </div>
                <div class="sidebar-content" id="briefingContent">
                    <div class="skeleton skeleton-line"></div>
                    <div class="skeleton skeleton-line short"></div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">Predictions</div>
                <div class="sidebar-content" id="predictionsContent">
                    <div class="skeleton skeleton-line"></div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">People Radar</div>
                <div class="sidebar-content" id="peopleContent">
                    <div class="skeleton skeleton-line"></div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">Mood</div>
                <div class="sidebar-content" id="moodSnapshot">
                    <div class="mood-snapshot">
                        <div class="mood-row">
                            <span class="mood-label">Energy</span>
                            <div class="mood-track"><div class="mood-fill energy" id="moodEnergy" style="width:50%"></div></div>
                        </div>
                        <div class="mood-row">
                            <span class="mood-label">Stress</span>
                            <div class="mood-track"><div class="mood-fill stress" id="moodStress" style="width:30%"></div></div>
                        </div>
                        <div class="mood-row">
                            <span class="mood-label">Social</span>
                            <div class="mood-track"><div class="mood-fill social" id="moodSocial" style="width:40%"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </aside>
    </div>

    <!-- Status Bar -->
    <div class="status-bar">
        <span>
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">Connecting...</span>
        </span>
        <div class="status-items">
            <span id="eventCount"></span>
            <span id="connectorStatus"></span>
        </div>
    </div>

    <script>
    /* ================================================================
       Life OS Dashboard -- Client-Side Logic

       SECURITY NOTE: All dynamic content is passed through escHtml()
       before insertion. The escHtml function encodes &, <, >, and "
       characters to prevent XSS. This is the standard mitigation for
       rendering server-provided data in HTML context.
       ================================================================ */

    // --- Security: HTML Escaping ---
    function escHtml(s) {
        if (!s) return '';
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }
    function escAttr(s) {
        return escHtml(String(s || '').replace(/'/g, '&#39;'));
    }

    // --- Configuration ---
    var API = '';

    // --- State ---
    var currentTopic = 'inbox';
    var feedItems = [];
    var expandedCardId = null;

    // --- Topics definition ---
    var topics = [
        {id: 'inbox',    icon: '\u2630', label: 'Inbox'},
        {id: 'messages', icon: '\u2709', label: 'Messages'},
        {id: 'email',    icon: '\u2709', label: 'Email'},
        {id: 'calendar', icon: '\uD83D\uDCC5', label: 'Calendar'},
        {id: 'tasks',    icon: '\u2611', label: 'Tasks'},
        {id: 'insights', icon: '\u2605', label: 'Insights'},
        {id: 'profile',  icon: '\uD83D\uDC64', label: 'My Profile'},
        {id: 'system',   icon: '\u2699', label: 'System'}
    ];

    // --- Greeting ---
    function updateGreeting() {
        var h = new Date().getHours();
        var g = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
        document.getElementById('greeting').textContent = g;
    }

    // --- Topic Navigation ---
    function switchTopic(topic) {
        currentTopic = topic;
        expandedCardId = null;
        // Update active class in desktop nav
        var items = document.querySelectorAll('.topic-item');
        for (var i = 0; i < items.length; i++) {
            items[i].classList.toggle('active', items[i].getAttribute('data-topic') === topic);
        }
        // Update active class in mobile tabs
        var tabs = document.querySelectorAll('.mobile-tab');
        for (var i = 0; i < tabs.length; i++) {
            tabs[i].classList.toggle('active', tabs[i].getAttribute('data-topic') === topic);
        }
        // Update header
        var t = topics.find(function(t) { return t.id === topic; });
        document.getElementById('feedHeader').textContent = t ? t.label : topic;
        // Load feed
        loadFeed();
    }

    // Event delegation on topic nav
    document.getElementById('topicNav').addEventListener('click', function(e) {
        var item = e.target.closest('.topic-item');
        if (item) {
            var topic = item.getAttribute('data-topic');
            if (topic) switchTopic(topic);
        }
    });

    // --- Utility Functions ---
    function channelIcon(channel) {
        switch (channel) {
            case 'email':    return '\u2709';
            case 'message':  return '\uD83D\uDCAC';
            case 'calendar': return '\uD83D\uDCC5';
            case 'task':     return '\u2611';
            case 'system':   return '\u2699';
            default:         return '\u2022';
        }
    }

    function sentimentDot(score) {
        if (score === null || score === undefined) return '';
        var n = parseFloat(score);
        if (isNaN(n)) return '';
        if (n > 0.2) return '<span class="sentiment-dot sentiment-positive" title="Positive"></span>';
        if (n < -0.2) return '<span class="sentiment-dot sentiment-negative" title="Negative"></span>';
        return '<span class="sentiment-dot sentiment-neutral" title="Neutral"></span>';
    }

    function timeAgo(ts) {
        if (!ts) return '';
        var now = Date.now();
        var t = new Date(ts).getTime();
        if (isNaN(t)) return '';
        var diff = Math.floor((now - t) / 1000);
        if (diff < 60) return 'now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
        return new Date(ts).toLocaleDateString();
    }

    function formatTime(ts) {
        if (!ts) return '';
        var d = new Date(ts);
        if (isNaN(d.getTime())) return '';
        return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    }

    // --- Safe DOM update helper ---
    // All content is escaped via escHtml before building HTML strings.
    function safeSetContent(el, htmlStr) {
        el.innerHTML = htmlStr;  // nosec: content is pre-escaped via escHtml
    }

    // --- Card Rendering ---
    function renderCard(item) {
        var id = item.id || '';
        var isExpanded = expandedCardId === id;
        var priorityClass = '';
        if (item.priority === 'critical') priorityClass = ' priority-critical';
        else if (item.priority === 'high') priorityClass = ' priority-high';
        var expandedClass = isExpanded ? ' expanded' : '';
        var meta = item.metadata || {};

        var icon = channelIcon(item.channel);
        var html = '<div class="card' + priorityClass + expandedClass + '" data-id="' + escAttr(id) + '" onclick="toggleCard(\'' + escAttr(id) + '\')">';
        html += '<div class="card-row">';
        html += '<div class="card-channel">' + icon + '</div>';
        html += '<div class="card-content">';

        // Channel-specific rendering
        if (item.channel === 'email') {
            var sender = meta.sender || item.source || '';
            html += '<div class="card-title">' + escHtml(item.title) + sentimentDot(meta.sentiment) + '</div>';
            html += '<div class="card-meta">' + escHtml(sender) + ' &middot; ' + escHtml(timeAgo(item.timestamp)) + '</div>';
            html += '<div class="card-body">' + escHtml(item.body) + '</div>';
            // Drill-down detail
            html += '<div class="card-detail">';
            html += '<div style="margin-bottom:8px">' + escHtml(item.body) + '</div>';
            if (meta.action_items && meta.action_items.length) {
                html += '<div style="margin-bottom:8px">';
                for (var i = 0; i < meta.action_items.length; i++) {
                    html += '<span class="chip">' + escHtml(meta.action_items[i]) + '</span>';
                }
                html += '</div>';
            }
            html += '<div id="draft-' + escAttr(id) + '"></div>';
            html += '<div class="card-actions">';
            html += '<button class="btn-primary" onclick="event.stopPropagation();draftReply(\'' + escAttr(id) + '\',\'' + escAttr(item.title) + '\')">Draft Reply</button>';
            html += '<button onclick="event.stopPropagation();createTaskFrom(\'' + escAttr(item.title) + '\')">Create Task</button>';
            if (item.kind === 'notification') {
                // Prediction notifications get an "Act On" button to mark them as helpful
                // (sets was_accurate=True in prediction feedback loop)
                if (item.domain === 'prediction') {
                    html += '<button class="btn-primary" onclick="event.stopPropagation();actOnNotification(\'' + escAttr(id) + '\')">Act On</button>';
                }
                html += '<button class="btn-danger" onclick="event.stopPropagation();dismissCard(\'' + escAttr(id) + '\',\'' + escAttr(item.kind) + '\')">Dismiss</button>';
            }
            html += '</div></div>';

        } else if (item.channel === 'message') {
            var sender = meta.sender || item.source || '';
            html += '<div class="card-title">' + escHtml(item.title) + '</div>';
            html += '<div class="card-meta">' + escHtml(sender) + ' &middot; ' + escHtml(timeAgo(item.timestamp)) + '</div>';
            html += '<div class="card-body">' + escHtml(item.body) + '</div>';
            html += '<div class="card-detail">';
            html += '<div style="margin-bottom:8px">' + escHtml(item.body) + '</div>';
            html += '<div class="card-actions">';
            html += '<button onclick="event.stopPropagation();createTaskFrom(\'' + escAttr(item.title) + '\')">Create Task</button>';
            if (item.kind === 'notification') {
                // Prediction notifications get an "Act On" button to mark them as helpful
                // (sets was_accurate=True in prediction feedback loop)
                if (item.domain === 'prediction') {
                    html += '<button class="btn-primary" onclick="event.stopPropagation();actOnNotification(\'' + escAttr(id) + '\')">Act On</button>';
                }
                html += '<button class="btn-danger" onclick="event.stopPropagation();dismissCard(\'' + escAttr(id) + '\',\'' + escAttr(item.kind) + '\')">Dismiss</button>';
            }
            html += '</div></div>';

        } else if (item.channel === 'calendar') {
            var start = meta.start_time ? formatTime(meta.start_time) : '';
            var end = meta.end_time ? formatTime(meta.end_time) : '';
            var timeRange = start ? (start + (end ? ' - ' + end : '')) : escHtml(timeAgo(item.timestamp));
            var location = meta.location || '';
            html += '<div class="card-title">' + escHtml(item.title) + '</div>';
            html += '<div class="card-meta">' + timeRange + (location ? ' &middot; ' + escHtml(location) : '') + '</div>';
            html += '<div class="card-detail">';
            if (meta.attendees && meta.attendees.length) {
                html += '<div style="margin-bottom:8px"><strong>Attendees:</strong> ' + escHtml(meta.attendees.join(', ')) + '</div>';
            }
            if (location) {
                html += '<div><a href="https://maps.google.com/maps?q=' + encodeURIComponent(location) + '" target="_blank">Get Directions</a></div>';
            }
            html += '</div>';

        } else if (item.channel === 'task') {
            var domain = meta.domain || item.source || '';
            var due = meta.due_date || '';
            html += '<div class="card-title">' + escHtml(item.title) + '</div>';
            html += '<div class="card-meta">';
            if (domain) html += '<span class="chip chip-domain">' + escHtml(domain) + '</span>';
            if (due) html += 'Due: ' + escHtml(due);
            else html += escHtml(timeAgo(item.timestamp));
            html += '</div>';
            html += '<div class="card-body">' + escHtml(item.body) + '</div>';
            html += '<div class="card-detail">';
            if (item.body) html += '<div style="margin-bottom:8px">' + escHtml(item.body) + '</div>';
            html += '<div class="card-actions">';
            html += '<button class="btn-primary" onclick="event.stopPropagation();completeTask(\'' + escAttr(id) + '\')">Complete</button>';
            html += '<button class="btn-danger" onclick="event.stopPropagation();dismissCard(\'' + escAttr(id) + '\',\'' + escAttr(item.kind) + '\')">Dismiss</button>';
            html += '</div></div>';

        } else {
            // Default
            html += '<div class="card-title">' + escHtml(item.title) + '</div>';
            html += '<div class="card-meta">' + escHtml(item.source || '') + ' &middot; ' + escHtml(timeAgo(item.timestamp)) + '</div>';
            html += '<div class="card-body">' + escHtml(item.body) + '</div>';
            html += '<div class="card-detail">';
            html += '<div>' + escHtml(item.body) + '</div>';
            if (item.kind === 'notification') {
                html += '<div class="card-actions">';
                // Prediction notifications get an "Act On" button to mark them as helpful
                // (sets was_accurate=True in prediction feedback loop)
                if (item.domain === 'prediction') {
                    html += '<button class="btn-primary" onclick="event.stopPropagation();actOnNotification(\'' + escAttr(id) + '\')">Act On</button>';
                }
                html += '<button class="btn-danger" onclick="event.stopPropagation();dismissCard(\'' + escAttr(id) + '\',\'' + escAttr(item.kind) + '\')">Dismiss</button>';
                html += '</div>';
            }
            html += '</div>';
        }

        html += '</div></div></div>';
        return html;
    }

    // --- Feed Loading ---
    function loadFeed() {
        if (currentTopic === 'calendar') { loadCalendarView(); return; }
        if (currentTopic === 'insights') { loadInsightsFeed(); return; }
        if (currentTopic === 'profile') { loadFactsFeed(); return; }
        if (currentTopic === 'system') { loadSystemFeed(); return; }

        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div>');

        fetch(API + '/api/dashboard/feed?topic=' + encodeURIComponent(currentTopic))
        .then(function(res) { return res.json(); })
        .then(function(data) {
            feedItems = data.items || [];
            if (feedItems.length === 0) {
                safeSetContent(el, '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No items in this topic</div></div>');
                return;
            }
            var html = '';
            for (var i = 0; i < feedItems.length; i++) {
                html += renderCard(feedItems[i]);
            }
            safeSetContent(el, html);
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load feed: ' + escHtml(err.message) + '</div></div>');
        });
    }

    // --- Calendar View ---
    var calendarMonth = new Date().getMonth();
    var calendarYear = new Date().getFullYear();
    var calendarEvents = [];
    var calendarSelectedDay = null;

    function loadCalendarView() {
        var el = document.getElementById('feedContent');
        var today = new Date();

        // Build header with nav
        var monthNames = ['January','February','March','April','May','June',
                          'July','August','September','October','November','December'];
        var html = '<div class="calendar-header">';
        html += '<button class="cal-nav" onclick="event.stopPropagation();calendarNav(-1)">&lt;</button>';
        html += '<span class="cal-title">' + escHtml(monthNames[calendarMonth]) + ' ' + calendarYear + '</span>';
        html += '<button class="cal-nav" onclick="event.stopPropagation();calendarNav(1)">&gt;</button>';
        html += '</div>';

        // Weekday headers
        html += '<div class="calendar-weekdays">';
        var days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
        for (var i = 0; i < 7; i++) html += '<div>' + days[i] + '</div>';
        html += '</div>';

        // Calculate grid dates
        var firstDay = new Date(calendarYear, calendarMonth, 1);
        var lastDay = new Date(calendarYear, calendarMonth + 1, 0);
        var startOffset = firstDay.getDay(); // 0=Sun
        var totalDays = lastDay.getDate();

        // Pad start to fill first week
        var gridStart = new Date(firstDay);
        gridStart.setDate(gridStart.getDate() - startOffset);

        // Calculate total cells (fill complete weeks)
        var totalCells = startOffset + totalDays;
        var rows = Math.ceil(totalCells / 7);
        totalCells = rows * 7;

        // Show skeleton while loading
        html += '<div class="calendar-grid" id="calendarGrid">';
        for (var c = 0; c < totalCells; c++) {
            var cellDate = new Date(gridStart);
            cellDate.setDate(cellDate.getDate() + c);
            var isOther = cellDate.getMonth() !== calendarMonth;
            var isToday = cellDate.toDateString() === today.toDateString();
            var isSelected = calendarSelectedDay && cellDate.toDateString() === calendarSelectedDay.toDateString();
            var cls = 'calendar-day';
            if (isOther) cls += ' calendar-other-month';
            if (isToday) cls += ' calendar-today';
            if (isSelected) cls += ' calendar-selected';
            var dateStr = cellDate.getFullYear() + '-' + String(cellDate.getMonth()+1).padStart(2,'0') + '-' + String(cellDate.getDate()).padStart(2,'0');
            html += '<div class="' + cls + '" data-date="' + dateStr + '" onclick="calendarDayClick(\'' + dateStr + '\')">';
            html += '<div class="calendar-day-number">' + cellDate.getDate() + '</div>';
            html += '<div class="cal-day-events" id="cal-' + dateStr + '"></div>';
            html += '</div>';
        }
        html += '</div>';

        // Detail area for selected day
        html += '<div id="calendarDayDetail" class="calendar-day-detail"></div>';

        safeSetContent(el, html);

        // Fetch events for the visible range
        var rangeStart = gridStart.getFullYear() + '-' + String(gridStart.getMonth()+1).padStart(2,'0') + '-' + String(gridStart.getDate()).padStart(2,'0');
        var rangeEndDate = new Date(gridStart);
        rangeEndDate.setDate(rangeEndDate.getDate() + totalCells);
        var rangeEnd = rangeEndDate.getFullYear() + '-' + String(rangeEndDate.getMonth()+1).padStart(2,'0') + '-' + String(rangeEndDate.getDate()).padStart(2,'0');

        fetch(API + '/api/calendar/events?start=' + rangeStart + '&end=' + rangeEnd)
        .then(function(res) { return res.json(); })
        .then(function(data) {
            calendarEvents = data.events || [];
            renderCalendarEvents();
            if (calendarSelectedDay) renderDayDetail(calendarSelectedDay);
        })
        .catch(function(err) {
            console.error('Failed to load calendar events:', err);
        });
    }

    function renderCalendarEvents() {
        for (var i = 0; i < calendarEvents.length; i++) {
            var evt = calendarEvents[i];
            var startDate = evt.start_time.substring(0, 10); // YYYY-MM-DD
            var cell = document.getElementById('cal-' + startDate);
            if (!cell) continue;

            // Limit visible events per day to 3
            var existing = cell.querySelectorAll('.calendar-event');
            if (existing.length >= 3) {
                // Check if we already have a "more" indicator
                var moreEl = cell.querySelector('.calendar-event-more');
                if (moreEl) {
                    var count = parseInt(moreEl.getAttribute('data-count')) + 1;
                    moreEl.setAttribute('data-count', count);
                    safeSetContent(moreEl, '+' + count + ' more');
                } else {
                    var more = document.createElement('div');
                    more.className = 'calendar-event-more';
                    more.setAttribute('data-count', '1');
                    more.textContent = '+1 more';
                    cell.appendChild(more);
                }
                continue;
            }

            var pill = document.createElement('div');
            pill.className = 'calendar-event ' + (evt.is_all_day ? 'all-day' : 'timed');
            var label = '';
            if (!evt.is_all_day && evt.start_time.length > 10) {
                var t = new Date(evt.start_time);
                if (!isNaN(t.getTime())) {
                    label = t.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) + ' ';
                }
            }
            pill.textContent = label + (evt.title || 'Event');
            pill.title = evt.title || '';
            cell.appendChild(pill);
        }
    }

    function calendarNav(dir) {
        calendarMonth += dir;
        if (calendarMonth > 11) { calendarMonth = 0; calendarYear++; }
        if (calendarMonth < 0) { calendarMonth = 11; calendarYear--; }
        calendarSelectedDay = null;
        loadCalendarView();
    }

    function calendarDayClick(dateStr) {
        var d = new Date(dateStr + 'T00:00:00');
        if (calendarSelectedDay && calendarSelectedDay.toDateString() === d.toDateString()) {
            calendarSelectedDay = null; // toggle off
        } else {
            calendarSelectedDay = d;
        }
        // Update selected state
        var allDays = document.querySelectorAll('.calendar-day');
        for (var i = 0; i < allDays.length; i++) {
            allDays[i].classList.toggle('calendar-selected', allDays[i].getAttribute('data-date') === dateStr);
        }
        if (calendarSelectedDay) {
            renderDayDetail(calendarSelectedDay);
        } else {
            safeSetContent(document.getElementById('calendarDayDetail'), '');
        }
    }

    function renderDayDetail(day) {
        var dateStr = day.getFullYear() + '-' + String(day.getMonth()+1).padStart(2,'0') + '-' + String(day.getDate()).padStart(2,'0');
        var dayEvents = calendarEvents.filter(function(evt) {
            return evt.start_time.substring(0, 10) === dateStr;
        });

        var detailEl = document.getElementById('calendarDayDetail');
        if (dayEvents.length === 0) {
            safeSetContent(detailEl, '<div class="calendar-detail-card"><div class="cal-evt-meta" style="text-align:center;padding:8px">No events on ' + escHtml(day.toLocaleDateString(undefined, {weekday:'long', month:'long', day:'numeric'})) + '</div></div>');
            return;
        }

        // Sort: all-day first, then by start_time
        dayEvents.sort(function(a, b) {
            if (a.is_all_day && !b.is_all_day) return -1;
            if (!a.is_all_day && b.is_all_day) return 1;
            return a.start_time < b.start_time ? -1 : 1;
        });

        var html = '<div style="font-size:13px;color:var(--text-secondary);margin-bottom:8px;font-weight:500">' +
                   escHtml(day.toLocaleDateString(undefined, {weekday:'long', month:'long', day:'numeric'})) +
                   ' — ' + dayEvents.length + ' event' + (dayEvents.length !== 1 ? 's' : '') + '</div>';

        for (var i = 0; i < dayEvents.length; i++) {
            var evt = dayEvents[i];
            html += '<div class="calendar-detail-card">';
            html += '<div class="cal-evt-title">' + escHtml(evt.title || 'Untitled Event') + '</div>';

            // Time
            if (evt.is_all_day) {
                html += '<div class="cal-evt-time">All day</div>';
            } else {
                var s = new Date(evt.start_time);
                var e = evt.end_time ? new Date(evt.end_time) : null;
                var timeStr = !isNaN(s.getTime()) ? s.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) : '';
                if (e && !isNaN(e.getTime())) timeStr += ' – ' + e.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
                if (timeStr) html += '<div class="cal-evt-time">' + escHtml(timeStr) + '</div>';
            }

            // Meta details
            var meta = [];
            if (evt.location) meta.push('\uD83D\uDCCD ' + escHtml(evt.location));
            if (evt.calendar_id) meta.push('\uD83D\uDCC5 ' + escHtml(evt.calendar_id));
            if (evt.attendees && evt.attendees.length) meta.push('\uD83D\uDC65 ' + escHtml(evt.attendees.join(', ')));
            if (evt.description) meta.push(escHtml(evt.description));
            if (meta.length) {
                html += '<div class="cal-evt-meta">' + meta.join('<br>') + '</div>';
            }
            html += '</div>';
        }

        safeSetContent(detailEl, html);
    }

    function loadInsightsFeed() {
        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div>');

        fetch(API + '/api/insights/summary')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var insights = data.insights || data.predictions || [];
            if (!Array.isArray(insights)) insights = [];
            if (insights.length === 0) {
                safeSetContent(el, '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No insights available yet</div></div>');
                return;
            }
            var html = '';
            for (var i = 0; i < insights.length; i++) {
                var ins = insights[i];
                html += '<div class="card" id="insight-' + escAttr(ins.id || i) + '">';
                html += '<div class="card-row"><div class="card-channel">\u2605</div>';
                html += '<div class="card-content">';
                html += '<div class="card-title">' + escHtml(ins.type || ins.category || 'Insight') + '</div>';
                html += '<div class="card-body">' + escHtml(ins.summary || ins.description || ins.content || ins.text || JSON.stringify(ins)) + '</div>';
                if (ins.confidence !== undefined) {
                    html += '<div class="card-meta" style="margin-top:4px">Confidence: ' + escHtml(String(Math.round(ins.confidence * 100))) + '%</div>';
                }
                if (ins.id) {
                    html += '<div class="card-actions" style="margin-top:8px;display:flex;gap:6px">';
                    html += '<button class="btn-small" onclick="event.stopPropagation();insightFeedback(\'' + escAttr(ins.id) + '\',\'useful\')">Useful</button>';
                    html += '<button class="btn-small btn-danger" onclick="event.stopPropagation();insightFeedback(\'' + escAttr(ins.id) + '\',\'dismissed\')">Dismiss</button>';
                    html += '<button class="btn-small btn-warn" onclick="event.stopPropagation();insightFeedback(\'' + escAttr(ins.id) + '\',\'not_relevant\')">Not About Me</button>';
                    html += '</div>';
                }
                html += '</div></div></div>';
            }
            safeSetContent(el, html);
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load insights: ' + escHtml(err.message) + '</div></div>');
        });
    }

    function loadFactsFeed() {
        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div>');

        fetch(API + '/api/user-model/facts')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var facts = data.facts || [];
            if (facts.length === 0) {
                safeSetContent(el, '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No facts learned yet</div></div>');
                return;
            }
            // Group facts by category
            var groups = {};
            for (var i = 0; i < facts.length; i++) {
                var f = facts[i];
                var cat = f.category || 'general';
                if (!groups[cat]) groups[cat] = [];
                groups[cat].push(f);
            }
            var html = '';
            var cats = Object.keys(groups).sort();
            for (var c = 0; c < cats.length; c++) {
                var category = cats[c];
                var items = groups[category];
                html += '<div class="card">';
                html += '<div class="card-row"><div class="card-channel">\uD83D\uDC64</div>';
                html += '<div class="card-content">';
                html += '<div class="card-title" style="text-transform:capitalize">' + escHtml(category.replace(/_/g, ' ')) + '</div>';
                for (var j = 0; j < items.length; j++) {
                    var fact = items[j];
                    html += '<div id="fact-' + escAttr(fact.key) + '" style="padding:6px 0;border-bottom:1px solid var(--border-color)">';
                    html += '<div class="card-body"><strong>' + escHtml(fact.key.replace(/_/g, ' ')) + ':</strong> ' + escHtml(String(fact.value || '')) + '</div>';
                    html += '<div class="card-meta" style="margin-top:2px">';
                    html += 'Confidence: ' + escHtml(String(Math.round((fact.confidence || 0) * 100))) + '%';
                    if (fact.times_confirmed > 0) html += ' &middot; Confirmed ' + escHtml(String(fact.times_confirmed)) + 'x';
                    if (fact.is_user_corrected) html += ' &middot; <span style="color:var(--accent-yellow,#f0a020);font-weight:600">CORRECTED</span>';
                    html += '</div>';
                    html += '<div class="card-actions" style="margin-top:4px;display:flex;gap:4px">';
                    html += '<button class="btn-small" onclick="event.stopPropagation();correctFact(\'' + escAttr(fact.key) + '\')">Correct</button>';
                    html += '<button class="btn-small btn-danger" onclick="event.stopPropagation();deleteFact(\'' + escAttr(fact.key) + '\')">Delete</button>';
                    html += '<button class="btn-small btn-warn" onclick="event.stopPropagation();notAboutMeFact(\'' + escAttr(fact.key) + '\')">Not About Me</button>';
                    html += '</div>';
                    html += '</div>';
                }
                html += '</div></div></div>';
            }
            safeSetContent(el, html);
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load facts: ' + escHtml(err.message) + '</div></div>');
        });
    }

    function deleteFact(key) {
        if (!confirm('Delete this fact?')) return;
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
    }

    function correctFact(key) {
        var reason = prompt('What is incorrect about this fact? (optional reason)');
        if (reason === null) return; // cancelled
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
            loadFactsFeed(); // Refresh to show updated state
        })
        .catch(function(err) { console.error('Correct fact failed:', err); });
    }

    function notAboutMeFact(key) {
        fetch(API + '/api/user-model/facts/' + encodeURIComponent(key), {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({reason: 'Not about me — attributed to wrong person'})
        })
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            return fetch(API + '/api/user-model/facts/' + encodeURIComponent(key), {method: 'DELETE'});
        })
        .then(function() {
            var card = document.getElementById('fact-' + key);
            if (card) card.style.display = 'none';
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = 'Marked as not about you — fact removed';
            setTimeout(function() { resp.className = ''; }, 3000);
        })
        .catch(function(err) { console.error('Not about me failed:', err); });
    }

    function loadSystemFeed() {
        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div>');

        fetch(API + '/health')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var html = '';
            // Event bus status
            html += '<div class="card"><div class="card-row"><div class="card-channel">\u2699</div>';
            html += '<div class="card-content">';
            html += '<div class="card-title">Event Bus</div>';
            html += '<div class="card-meta">' + (data.event_bus ? '<span style="color:var(--accent-green)">\u2713 Connected</span>' : '<span style="color:var(--accent-red)">\u2717 Disconnected</span>') + '</div>';
            html += '</div></div></div>';

            // Events stored
            html += '<div class="card"><div class="card-row"><div class="card-channel">\u2699</div>';
            html += '<div class="card-content">';
            html += '<div class="card-title">Events Stored</div>';
            html += '<div class="card-meta">' + escHtml(String(data.events_stored || 0)) + ' total events</div>';
            html += '</div></div></div>';

            // Vector store
            if (data.vector_store) {
                html += '<div class="card"><div class="card-row"><div class="card-channel">\u2699</div>';
                html += '<div class="card-content">';
                html += '<div class="card-title">Vector Store</div>';
                html += '<div class="card-meta">' + escHtml(String(data.vector_store.document_count || 0)) + ' documents indexed</div>';
                html += '</div></div></div>';
            }

            // Connectors
            var connectors = data.connectors || [];
            for (var i = 0; i < connectors.length; i++) {
                var c = connectors[i];
                var name = c.name || c.type || 'Connector ' + i;
                var status = c.status || c.state || 'unknown';
                var statusColor = status === 'active' || status === 'connected' ? 'var(--accent-green)' : 'var(--accent-red)';
                html += '<div class="card"><div class="card-row"><div class="card-channel">\u2699</div>';
                html += '<div class="card-content">';
                html += '<div class="card-title">' + escHtml(name) + '</div>';
                html += '<div class="card-meta"><span style="color:' + statusColor + '">' + escHtml(status) + '</span></div>';
                html += '</div></div></div>';
            }

            safeSetContent(el, html || '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No system data available</div></div>');
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load system status: ' + escHtml(err.message) + '</div></div>');
        });
    }

    // --- Card Interaction ---
    function toggleCard(id) {
        if (expandedCardId === id) {
            expandedCardId = null;
        } else {
            expandedCardId = id;
        }
        // Re-render feed
        var el = document.getElementById('feedContent');
        var html = '';
        for (var i = 0; i < feedItems.length; i++) {
            html += renderCard(feedItems[i]);
        }
        if (html) safeSetContent(el, html);
    }

    function completeTask(id) {
        fetch(API + '/api/tasks/' + encodeURIComponent(id) + '/complete', {method: 'POST'})
        .then(function() { loadFeed(); loadBadges(); })
        .catch(function(err) { console.error('Failed to complete task:', err); });
    }

    function actOnNotification(id) {
        // Mark notification as acted on (positive feedback signal).
        // For prediction notifications, this sets was_accurate=True
        // in the prediction feedback loop, allowing the system to
        // learn which predictions are helpful.
        fetch(API + '/api/notifications/' + encodeURIComponent(id) + '/act', {method: 'POST'})
        .then(function() { loadFeed(); loadBadges(); })
        .catch(function(err) { console.error('Failed to act on notification:', err); });
    }

    function dismissCard(id, kind) {
        if (kind === 'notification') {
            fetch(API + '/api/notifications/' + encodeURIComponent(id) + '/dismiss', {method: 'POST'})
            .then(function() { loadFeed(); loadBadges(); })
            .catch(function(err) { console.error('Failed to dismiss:', err); });
        } else if (kind === 'task') {
            completeTask(id);
        } else {
            // Remove from local state
            feedItems = feedItems.filter(function(item) { return item.id !== id; });
            expandedCardId = null;
            var el = document.getElementById('feedContent');
            var html = '';
            for (var i = 0; i < feedItems.length; i++) {
                html += renderCard(feedItems[i]);
            }
            safeSetContent(el, html || '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No items in this topic</div></div>');
        }
    }

    function insightFeedback(id, feedback) {
        fetch(API + '/api/insights/' + encodeURIComponent(id) + '/feedback?feedback=' + encodeURIComponent(feedback), {method: 'POST'})
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            var card = document.getElementById('insight-' + id);
            if (card) card.style.display = 'none';
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = feedback === 'useful' ? 'Marked as useful' : feedback === 'not_relevant' ? 'Marked as not about you' : 'Insight dismissed';
            setTimeout(function() { resp.className = ''; }, 3000);
        })
        .catch(function(err) { console.error('Insight feedback failed:', err); });
    }

    function draftReply(id, context) {
        var draftEl = document.getElementById('draft-' + id);
        if (!draftEl) return;
        safeSetContent(draftEl, '<div class="draft-area" style="opacity:0.5">Generating draft...</div>');

        // Look up the original feed item to get the actual message body.
        // The DraftRequest schema requires 'incoming_message' — sending only
        // 'context' (the card title) leaves the AI with no content to work from.
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

    /**
     * Copy the generated draft text to the clipboard.
     * The button sits in .draft-actions immediately after the .draft-area div.
     */
    function copyDraft(btn) {
        var area = btn.closest('.draft-actions').previousElementSibling;
        if (!area) return;
        navigator.clipboard.writeText(area.textContent).then(function() {
            btn.textContent = 'Copied!';
            setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
        });
    }

    function createTaskFrom(title) {
        fetch(API + '/api/tasks', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({title: title, priority: 'normal'})
        })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = 'Task created: ' + (data.title || title);
            setTimeout(function() { resp.className = ''; }, 3000);
            loadBadges();
        })
        .catch(function(err) { console.error('Failed to create task:', err); });
    }

    // --- Command Bar ---
    document.getElementById('commandInput').addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' || !this.value.trim()) return;
        var text = this.value.trim();
        this.value = '';
        var resp = document.getElementById('response');
        resp.className = 'visible loading-text';
        resp.textContent = 'Thinking...';

        fetch(API + '/api/command', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            resp.className = 'visible';
            if (data.type === 'search_results') {
                resp.textContent = data.results && data.results.length
                    ? data.results.map(function(r) { return '[' + Math.round(r.score * 100) + '%] ' + r.text.slice(0, 150); }).join('\n\n')
                    : 'No results found.';
            } else if (data.type === 'task_created') {
                resp.textContent = 'Task created.';
                loadFeed();
                loadBadges();
            } else {
                resp.textContent = data.content || data.briefing || JSON.stringify(data, null, 2);
            }
        })
        .catch(function(err) {
            resp.className = 'visible';
            resp.textContent = 'Error: ' + err.message;
        });
    });

    // --- Sidebar Functions ---
    function loadBriefing() {
        var el = document.getElementById('briefingContent');
        safeSetContent(el, '<div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line short"></div>');

        fetch(API + '/api/briefing')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var text = data.briefing || 'No briefing available';
            safeSetContent(el, '<div style="white-space:pre-wrap">' + escHtml(text) + '</div>');
        })
        .catch(function() {
            safeSetContent(el, '<div style="color:var(--text-muted)">Briefing unavailable</div>');
        });
    }

    function loadPredictions() {
        var el = document.getElementById('predictionsContent');

        fetch(API + '/api/predictions?limit=5')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var preds = data.predictions || [];
            if (!Array.isArray(preds)) preds = [];
            if (preds.length === 0) {
                safeSetContent(el, '<div style="color:var(--text-muted)">No predictions yet</div>');
                return;
            }
            var html = '';
            for (var i = 0; i < Math.min(preds.length, 5); i++) {
                var p = preds[i];
                html += '<div class="prediction-card" id="pred-' + escAttr(p.id) + '">';
                html += '<div class="pred-label">' + escHtml(p.prediction_type || 'Prediction') + '</div>';
                html += '<div>' + escHtml(p.description || '') + '</div>';
                if (p.confidence !== undefined) {
                    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:2px">' + escHtml(String(Math.round(p.confidence * 100))) + '% confidence</div>';
                }
                html += '<div class="card-actions" style="margin-top:6px;display:flex;gap:4px">';
                html += '<button class="btn-small" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',true,null)">Accurate</button>';
                html += '<button class="btn-small btn-danger" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',false,null)">Inaccurate</button>';
                html += '<button class="btn-small btn-warn" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',false,\'not_relevant\')">Not About Me</button>';
                html += '</div>';
                html += '</div>';
            }
            safeSetContent(el, html);
        })
        .catch(function() {
            safeSetContent(el, '<div style="color:var(--text-muted)">Predictions unavailable</div>');
        });
    }

    function predictionFeedback(id, wasAccurate, userResponse) {
        var url = API + '/api/predictions/' + encodeURIComponent(id) + '/feedback?was_accurate=' + wasAccurate;
        if (userResponse) url += '&user_response=' + encodeURIComponent(userResponse);
        fetch(url, {method: 'POST'})
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            var card = document.getElementById('pred-' + id);
            if (card) card.style.display = 'none';
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = wasAccurate ? 'Marked as accurate' : userResponse === 'not_relevant' ? 'Marked as not about you' : 'Marked as inaccurate';
            setTimeout(function() { resp.className = ''; }, 3000);
        })
        .catch(function(err) { console.error('Prediction feedback failed:', err); });
    }

    function loadPeopleRadar() {
        var el = document.getElementById('peopleContent');

        // Use the contacts API which provides last_contact, contact_frequency_days,
        // and typical_response_time — actual relationship metrics, not filtered insights.
        fetch(API + '/api/contacts?has_metrics=true&limit=10')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var contacts = data.contacts || (Array.isArray(data) ? data : []);
            if (contacts.length === 0) {
                safeSetContent(el, '<div style="color:var(--text-muted)">No contact data yet</div>');
                return;
            }
            var html = '';
            for (var i = 0; i < Math.min(contacts.length, 8); i++) {
                var c = contacts[i];
                var name = c.name || c.contact_id || 'Unknown';
                var lastContact = c.last_contact ? timeAgo(c.last_contact) : 'never';

                // Mark overdue if days since last contact exceeds 1.5× the typical frequency
                var overdue = false;
                if (c.contact_frequency_days && c.last_contact) {
                    var daysSince = (Date.now() - new Date(c.last_contact).getTime()) / 86400000;
                    overdue = daysSince > c.contact_frequency_days * 1.5;
                }
                var dotColor = overdue ? 'var(--accent-red)' : 'var(--accent-green)';

                html += '<div class="person-card">';
                html += '<div class="person-avatar" style="background:' + dotColor + '">' + escHtml(name.charAt(0).toUpperCase()) + '</div>';
                html += '<div class="person-info">';
                html += '<div class="person-name">' + escHtml(name) + '</div>';
                html += '<div class="person-detail">' + escHtml(lastContact) + '</div>';
                html += '</div></div>';
            }
            safeSetContent(el, html);
        })
        .catch(function() {
            safeSetContent(el, '<div style="color:var(--text-muted)">People radar unavailable</div>');
        });
    }

    function loadMood() {
        fetch(API + '/api/user-model/mood')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            // The endpoint returns {"mood": {"energy_level":…, "stress_level":…, "social_battery":…}}.
            // Fall back to the top-level object for any future schema changes.
            var m = data.mood || data || {};
            var energy = Math.max(0, Math.min(100, (m.energy_level || 0.5) * 100));
            var stress = Math.max(0, Math.min(100, (m.stress_level || 0.3) * 100));
            var social = Math.max(0, Math.min(100, (m.social_battery || 0.4) * 100));

            document.getElementById('moodEnergy').style.width = energy + '%';
            document.getElementById('moodStress').style.width = stress + '%';
            document.getElementById('moodSocial').style.width = social + '%';

            // Also update top bar mini bars
            document.getElementById('miniEnergy').style.width = energy + '%';
            document.getElementById('miniStress').style.width = stress + '%';
        })
        .catch(function() {
            // Silently fail -- keep defaults
        });
    }

    function toggleSidebar() {
        var sidebar = document.getElementById('aiSidebar');
        var toggle = document.getElementById('sidebarToggle');
        sidebar.classList.toggle('collapsed');
        safeSetContent(toggle, sidebar.classList.contains('collapsed') ? '&#9654;' : '&#9664;');
    }

    // --- Badge Counts ---
    function setBadge(topicId, count) {
        var badge = document.getElementById('badge-' + topicId);
        if (!badge) return;
        if (count > 0) {
            badge.textContent = count > 99 ? '99+' : String(count);
            badge.classList.add('visible');
        } else {
            badge.classList.remove('visible');
        }
    }

    function loadBadges() {
        var topicIds = ['inbox', 'messages', 'email', 'calendar', 'tasks'];
        topicIds.forEach(function(tid) {
            fetch(API + '/api/dashboard/feed?topic=' + tid + '&limit=100')
            .then(function(res) { return res.json(); })
            .then(function(data) {
                setBadge(tid, data.count || 0);
            })
            .catch(function() {});
        });
    }

    // --- Status Bar ---
    function loadStatus() {
        fetch(API + '/health')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var dot = document.getElementById('statusDot');
            var text = document.getElementById('statusText');
            var count = document.getElementById('eventCount');
            var connStatus = document.getElementById('connectorStatus');

            dot.className = 'status-dot ' + (data.status === 'ok' ? 'ok' : 'error');
            text.textContent = data.event_bus ? 'Connected' : 'Event bus disconnected';
            count.textContent = (data.events_stored || 0) + ' events';

            var connectors = data.connectors || [];
            var active = connectors.filter(function(c) { return c.status === 'active' || c.state === 'active'; }).length;
            connStatus.textContent = active + '/' + connectors.length + ' connectors';
        })
        .catch(function() {
            document.getElementById('statusDot').className = 'status-dot error';
            document.getElementById('statusText').textContent = 'Offline';
        });
    }

    // --- Manual Refresh ---
    function refreshAll() {
        loadFeed();
        loadBadges();
        loadStatus();
        loadBriefing();
        loadPredictions();
        loadPeopleRadar();
        loadMood();
    }

    // --- Scroll ---
    function scrollToTop() {
        var feed = document.getElementById('mainFeed');
        feed.scrollTop = 0;
        document.getElementById('newItemsBanner').classList.remove('visible');
        // Refresh data when user clicks the banner
        loadFeed();
        loadBadges();
    }

    // --- WebSocket ---
    function connectWS() {
        var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        var ws = new WebSocket(protocol + '//' + location.host + '/ws');
        ws.onmessage = function(e) {
            try {
                var data = JSON.parse(e.data);
                if (data.type === 'notification' || data.type === 'event') {
                    // Show the banner so the user can choose to refresh the feed.
                    document.getElementById('newItemsBanner').classList.add('visible');
                    // Also refresh badge counts immediately so the nav dot stays accurate.
                    loadBadges();
                }
                // Refresh mood bars when a mood_update event is pushed.
                if (data.type === 'mood_update') {
                    loadMood();
                }
            } catch(err) {}
        };
        ws.onclose = function() {
            setTimeout(connectWS, 5000);
        };
    }

    // --- Escape key: collapse expanded card ---
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && expandedCardId) {
            expandedCardId = null;
            var el = document.getElementById('feedContent');
            var html = '';
            for (var i = 0; i < feedItems.length; i++) {
                html += renderCard(feedItems[i]);
            }
            if (html) safeSetContent(el, html);
        }
    });

    // --- Mobile Tabs ---
    (function buildMobileTabs() {
        var container = document.getElementById('mobileTabs');
        var html = '';
        for (var i = 0; i < topics.length; i++) {
            var t = topics[i];
            var active = t.id === currentTopic ? ' active' : '';
            html += '<div class="mobile-tab' + active + '" data-topic="' + t.id + '" onclick="switchTopic(\'' + t.id + '\')">' + t.icon + ' ' + escHtml(t.label) + '</div>';
        }
        safeSetContent(container, html);
    })();

    // --- Initial Load ---
    updateGreeting();
    loadFeed();
    loadBadges();
    loadStatus();
    loadBriefing();
    loadPredictions();
    loadPeopleRadar();
    loadMood();

    // --- Auto-refresh sidebar data every 60 seconds ---
    // The main feed is not auto-refreshed to avoid surprising the user mid-scroll;
    // they use the banner / refresh button for that.  Sidebar panels (mood, predictions,
    // people radar) are cheap and benefit from staying current.
    setInterval(function() {
        loadMood();
        loadPredictions();
        loadPeopleRadar();
    }, 60000);

    // Refresh badge counts every 2 minutes so the nav stays accurate.
    setInterval(loadBadges, 120000);

    // --- WebSocket ---
    try { connectWS(); } catch(e) {}

    </script>
</body>
</html>"""
