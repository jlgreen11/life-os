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
        .stale-data-banner {
            display: none;
            background: var(--accent-yellow);
            color: #000;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 500;
            text-align: center;
            align-items: center;
            justify-content: center;
            gap: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .stale-data-banner.visible { display: flex; }
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
        .card-actions button.btn-success {
            background: var(--accent-green);
            border-color: var(--accent-green);
            color: #000;
        }
        .card-actions button.btn-success:hover {
            opacity: 0.85;
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

        /* --- Quick-reply inline compose box (message cards) --- */
        /* Shown in the drill-down of message cards so the user can type and send
           a reply without leaving the dashboard.  The textarea auto-resizes via
           JS and the send button is disabled while the request is in-flight. */
        .quick-reply-area {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }
        .quick-reply-input {
            width: 100%;
            min-height: 60px;
            padding: 8px 10px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            line-height: 1.5;
            resize: vertical;
            box-sizing: border-box;
        }
        .quick-reply-input:focus { outline: none; border-color: var(--accent); }
        .quick-reply-input:disabled { opacity: 0.5; cursor: not-allowed; }
        .quick-reply-row {
            display: flex;
            gap: 8px;
            margin-top: 6px;
            align-items: center;
        }
        .quick-reply-send {
            padding: 6px 18px;
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        }
        .quick-reply-send:hover { opacity: 0.85; }
        .quick-reply-send:disabled { opacity: 0.5; cursor: not-allowed; }
        .quick-reply-hint {
            font-size: 11px;
            color: var(--text-muted);
            flex: 1;
        }

        /* --- Insights Tab Sections --- */
        /* Section headers visually group the different data sources shown
           in the Insights tab (AI insights, signal profiles, routines, workflows). */
        .section-header {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-muted);
            margin: 20px 0 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border);
        }
        /* First section header needs no top margin */
        .section-header:first-child { margin-top: 0; }

        /* Grid layout for displaying individual signal profile metrics */
        .signal-profile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 6px;
            margin-top: 8px;
        }
        .signal-profile-item {
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 8px 10px;
        }
        .signal-profile-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 2px;
        }
        .signal-profile-value {
            font-size: 13px;
            color: var(--text-primary);
            font-weight: 600;
        }
        /* Inline metadata row for routines and workflows */
        .insight-meta-row {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin-top: 4px;
            font-size: 12px;
            color: var(--text-muted);
        }

        /* --- Inline Modal System --- */
        /* Replaces native browser confirm()/prompt() dialogs with themed UI.
           Works on mobile Safari where native dialogs are blocked. */
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
            max-width: 420px;
            width: 90%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .modal-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        .modal-body { margin-bottom: 16px; color: var(--text-secondary); font-size: 14px; }
        .modal-body input {
            width: 100%;
            padding: 10px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            box-sizing: border-box;
            margin-top: 8px;
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
            background: var(--accent-red, #e05252);
            color: white;
        }

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
        .person-card:hover {
            background: var(--bg-card-hover);
            border-radius: 6px;
        }
        .person-card .channel-badge {
            font-size: 9px;
            padding: 1px 6px;
            border-radius: 8px;
            background: var(--border);
            color: var(--text-secondary);
            text-transform: lowercase;
            white-space: nowrap;
        }
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

        /* --- System Health Dashboard Cards --- */
        .sys-card {
            padding: 10px 14px;
            margin-bottom: 6px;
        }
        .sys-card.sys-stale {
            border-left: 3px solid var(--accent-red);
        }
        .sys-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .sys-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .sys-name {
            flex: 1;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-primary);
            min-width: 100px;
        }
        .sys-status {
            font-size: 12px;
            color: var(--text-secondary);
        }
        .sys-last-seen {
            font-size: 12px;
            color: var(--text-muted);
            min-width: 60px;
            text-align: right;
        }
        .sys-rate {
            font-size: 11px;
            color: var(--text-muted);
            min-width: 70px;
            text-align: right;
        }
        .sys-error {
            font-size: 11px;
            color: var(--accent-red);
            flex-basis: 100%;
            padding-left: 18px;
            margin-top: 2px;
        }
        .sys-stale-msg {
            font-size: 11px;
            color: var(--accent-red);
            padding: 4px 0 0 18px;
            line-height: 1.4;
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

        /* --- Mobile Sidebar FAB (floating action button) ---
         * Visible only below 900px. Tapping it opens the AI sidebar as a
         * full-screen overlay. The close button inside the sidebar dismisses it. */
        .mobile-sidebar-fab {
            display: none;
        }
        .mobile-sidebar-close {
            display: none;
        }

        /* --- Responsive: 900px --- */
        @media (max-width: 900px) {
            /* Sidebar is hidden by default; .mobile-open turns it into a full-screen overlay. */
            .ai-sidebar { display: none; }
            .ai-sidebar.mobile-open {
                display: flex;
                flex-direction: column;
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                width: 100%;
                min-width: 100%;
                z-index: 998;
                background: var(--bg-primary);
                padding: 20px 20px 80px;  /* bottom pad clears the FAB */
                overflow-y: auto;
            }
            /* Show close button only inside the mobile overlay */
            .ai-sidebar.mobile-open .mobile-sidebar-close {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 8px;
            }
            .sidebar-toggle { display: none; }
            .topic-nav {
                width: 50px;
                min-width: 50px;
            }
            .topic-label { display: none; }
            /* Restore badge counts at smaller size — they're still informative on mobile. */
            .topic-badge {
                font-size: 9px;
                min-width: 14px;
                height: 14px;
                line-height: 14px;
                padding: 0 3px;
            }
            .topic-item {
                justify-content: center;
                padding: 10px 0;
            }
            .topic-icon { margin: 0; }
            .greeting { display: none; }
            .mood-bar { display: none; }
            /* Floating action button to open AI sidebar */
            .mobile-sidebar-fab {
                display: block;
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
                font-size: 13px;
                cursor: pointer;
                z-index: 999;
                box-shadow: 0 2px 12px rgba(0,0,0,0.45);
                letter-spacing: 0.02em;
            }
            .mobile-sidebar-fab:active { opacity: 0.85; }
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
            <div class="stale-data-banner" id="staleDataBanner">
                <span id="staleDataMsg"></span>
                <button onclick="dismissStaleWarning()" style="background:none;border:none;color:inherit;cursor:pointer;font-size:16px;padding:0 4px">&times;</button>
            </div>
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
            <!-- Close button shown only in mobile overlay mode (< 900px).
                 Uses .mobile-sidebar-close which is display:none by default
                 and display:flex inside .ai-sidebar.mobile-open. -->
            <div class="mobile-sidebar-close">
                <button onclick="toggleMobileSidebar()" aria-label="Close AI sidebar"
                        style="background:none;border:none;color:var(--text-secondary);font-size:22px;cursor:pointer;padding:0 4px">&#10005;</button>
            </div>
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
            // Message cards (Signal, iMessage, WhatsApp).  Inbound messages
            // from a contact show sender name + timestamp + body preview.
            // The drill-down adds:
            //   1. Full message body
            //   2. Quick-reply textarea + Send button (direct send via connector)
            //   3. Draft Reply button (AI-generated draft with Copy)
            //   4. Create Task shortcut
            var sender = meta.sender || item.source || '';
            // from_address is the reply-to address (phone / email / Signal number).
            // Prefer the explicit metadata field; fall back to the card's contact_id or source.
            var fromAddr = (meta.from_address) || item.contact_id || item.source || '';
            // channel carries the transport ("imessage", "signal", etc.) so the send
            // endpoint can route to the right connector.
            var msgChannel = (meta.channel) || item.channel || 'message';
            html += '<div class="card-title">' + escHtml(item.title) + '</div>';
            html += '<div class="card-meta">' + escHtml(sender) + ' &middot; ' + escHtml(timeAgo(item.timestamp)) + '</div>';
            html += '<div class="card-body">' + escHtml(item.body) + '</div>';
            html += '<div class="card-detail">';
            html += '<div style="margin-bottom:8px">' + escHtml(item.body) + '</div>';

            // Quick-reply compose box: lets the user type and send a reply directly
            // without leaving the dashboard.  sendQuickReply() posts to /api/messages/send.
            html += '<div class="quick-reply-area">';
            html += '<textarea id="qr-' + escAttr(id) + '" class="quick-reply-input"' +
                    ' placeholder="Type a quick reply…"' +
                    ' onclick="event.stopPropagation()"' +
                    ' onkeydown="if(event.ctrlKey&&event.key===\'Enter\'){event.stopPropagation();sendQuickReply(\'' + escAttr(id) + '\',\'' + escAttr(fromAddr) + '\',\'' + escAttr(msgChannel) + '\')}"></textarea>';
            html += '<div class="quick-reply-row">';
            html += '<button class="quick-reply-send" id="qr-btn-' + escAttr(id) + '"' +
                    ' onclick="event.stopPropagation();sendQuickReply(\'' + escAttr(id) + '\',\'' + escAttr(fromAddr) + '\',\'' + escAttr(msgChannel) + '\')">Send</button>';
            html += '<span class="quick-reply-hint">Ctrl+Enter to send</span>';
            html += '</div></div>';

            // Draft reply output area — populated asynchronously by draftReply()
            // when the user clicks the Draft Reply button.  Rendered here so
            // it sits between the quick-reply box and the action buttons.
            html += '<div id="draft-' + escAttr(id) + '"></div>';
            html += '<div class="card-actions">';
            // Draft Reply: generates an AI-drafted response using the message body
            // as context.  The sender address is passed as the contact_id so the
            // AI engine can apply per-contact communication templates.
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
        // Loads the Insights tab with five data sources in parallel:
        //   1. AI-generated behavioral insights (/api/insights/summary)
        //   2. Raw signal profiles (/api/user-model/signal-profiles) — temporal,
        //      linguistic, cadence, mood, relationships, topics, etc.
        //   3. Detected routines (/api/user-model/routines)
        //   4. Detected workflows (/api/user-model/workflows)
        //   5. Communication style templates (/api/user-model/templates) — per-contact
        //      writing style summaries learned from outbound/inbound messages.
        // Each section is rendered separately so partial failures degrade gracefully.
        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div>');

        Promise.all([
            fetch(API + '/api/insights/summary').then(function(r) { return r.json(); }).catch(function() { return {}; }),
            fetch(API + '/api/user-model/signal-profiles').then(function(r) { return r.json(); }).catch(function() { return {}; }),
            fetch(API + '/api/user-model/routines?min_observations=2').then(function(r) { return r.json(); }).catch(function() { return {}; }),
            fetch(API + '/api/user-model/workflows?min_observations=2').then(function(r) { return r.json(); }).catch(function() { return {}; }),
            fetch(API + '/api/user-model/templates?limit=20').then(function(r) { return r.json(); }).catch(function() { return {}; })
        ]).then(function(results) {
            var insightData   = results[0];
            var profileData   = results[1];
            var routineData   = results[2];
            var workflowData  = results[3];
            var templateData  = results[4];
            var html = '';
            var hasAny = false;

            // ── 1. AI-Generated Behavioral Insights ─────────────────────────
            var insights = insightData.insights || insightData.predictions || [];
            if (!Array.isArray(insights)) insights = [];
            if (insights.length > 0) {
                html += '<div class="section-header">AI Behavioral Insights</div>';
                hasAny = true;
                for (var i = 0; i < insights.length; i++) {
                    var ins = insights[i];
                    html += '<div class="card" id="insight-' + escAttr(ins.id || i) + '">';
                    html += '<div class="card-row"><div class="card-channel">\u2605</div>';
                    html += '<div class="card-content">';
                    html += '<div class="card-title">' + escHtml(ins.type || ins.category || 'Insight') + '</div>';
                    html += '<div class="card-body">' + escHtml(ins.summary || ins.description || ins.content || ins.text || '') + '</div>';
                    if (ins.confidence !== undefined) {
                        html += '<div class="card-meta" style="margin-top:4px">Confidence: ' + Math.round(ins.confidence * 100) + '%</div>';
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
            }

            // ── 2. Raw Signal Profiles ───────────────────────────────────────
            // Maps profile type names to human-readable labels shown as card titles.
            var profileLabels = {
                temporal:           'Temporal \u2014 When you work',
                linguistic:         'Linguistic \u2014 How you write',
                linguistic_inbound: 'Inbound Style \u2014 How contacts write to you',
                cadence:            'Cadence \u2014 Response patterns',
                mood_signals:       'Mood \u2014 Emotional signals',
                relationships:      'Relationships \u2014 Contact patterns',
                topics:             'Topics \u2014 Interest areas',
                spatial:            'Spatial \u2014 Location patterns',
                decision:           'Decision \u2014 Choice patterns'
            };
            var profiles = profileData.profiles || {};
            var profileTypes = profileData.types_with_data || Object.keys(profiles);
            if (profileTypes.length > 0) {
                html += '<div class="section-header">Behavioral Signal Profiles</div>';
                hasAny = true;
                for (var pi = 0; pi < profileTypes.length; pi++) {
                    var ptype = profileTypes[pi];
                    var profile = profiles[ptype];
                    if (!profile) continue;
                    var pdata = profile.data || profile;
                    var samples = profile.samples_count || 0;
                    var label = profileLabels[ptype] || ptype.replace(/_/g, ' ');
                    html += '<div class="card">';
                    html += '<div class="card-row"><div class="card-channel" style="font-size:20px">' + profileEmoji(ptype) + '</div>';
                    html += '<div class="card-content">';
                    html += '<div class="card-title">' + escHtml(label) + '</div>';
                    if (samples > 0) {
                        html += '<div class="card-meta" style="margin-bottom:6px">' + samples.toLocaleString() + ' samples</div>';
                    }
                    html += renderProfileData(pdata);
                    html += '</div></div></div>';
                }
            }

            // ── 3. Detected Routines ─────────────────────────────────────────
            var routines = routineData.routines || routineData || [];
            if (!Array.isArray(routines)) routines = [];
            if (routines.length > 0) {
                html += '<div class="section-header">Detected Routines</div>';
                hasAny = true;
                for (var ri = 0; ri < routines.length; ri++) {
                    var r = routines[ri];
                    html += '<div class="card">';
                    html += '<div class="card-row"><div class="card-channel">\uD83D\uDD04</div>';
                    html += '<div class="card-content">';
                    html += '<div class="card-title">' + escHtml(r.name || r.trigger || 'Routine') + '</div>';
                    html += '<div class="insight-meta-row">';
                    if (r.trigger) html += '<span>Trigger: ' + escHtml(r.trigger) + '</span>';
                    if (r.consistency !== undefined) html += '<span>Consistency: ' + Math.round(r.consistency * 100) + '%</span>';
                    if (r.observation_count) html += '<span>' + r.observation_count + ' observations</span>';
                    if (r.typical_duration_minutes) html += '<span>~' + Math.round(r.typical_duration_minutes) + ' min</span>';
                    html += '</div>';
                    html += '</div></div></div>';
                }
            }

            // ── 4. Detected Workflows ────────────────────────────────────────
            var workflows = workflowData.workflows || workflowData || [];
            if (!Array.isArray(workflows)) workflows = [];
            if (workflows.length > 0) {
                html += '<div class="section-header">Detected Workflows</div>';
                hasAny = true;
                for (var wi = 0; wi < workflows.length; wi++) {
                    var w = workflows[wi];
                    html += '<div class="card">';
                    html += '<div class="card-row"><div class="card-channel">\uD83D\uDCC8</div>';
                    html += '<div class="card-content">';
                    html += '<div class="card-title">' + escHtml(w.name || w.goal || 'Workflow') + '</div>';
                    html += '<div class="insight-meta-row">';
                    var stepCount = (w.steps ? w.steps.length : 0) || w.step_count || 0;
                    if (stepCount) html += '<span>' + stepCount + ' steps</span>';
                    if (w.success_rate !== undefined) html += '<span>Success: ' + Math.round(w.success_rate * 100) + '%</span>';
                    if (w.observation_count) html += '<span>' + w.observation_count + ' observations</span>';
                    html += '</div>';
                    html += '</div></div></div>';
                }
            }

            // ── 5. Communication Style Templates ─────────────────────────────
            // Per-contact, per-channel writing style summaries learned from
            // outbound and inbound messages.  Grouped by direction so the user
            // can see both how they write (user_to_contact) and how each contact
            // writes back (contact_to_user).
            var templates = (templateData.templates || []);
            if (templates.length > 0) {
                html += '<div class="section-header">Communication Style Templates</div>';
                hasAny = true;

                // Group by contact so we can show both directions side-by-side.
                var grouped = {};
                for (var ti = 0; ti < templates.length; ti++) {
                    var tmpl = templates[ti];
                    var key = (tmpl.contact_id || 'general') + '|' + (tmpl.channel || 'unknown');
                    if (!grouped[key]) grouped[key] = {contact: tmpl.contact_id, channel: tmpl.channel, items: []};
                    grouped[key].items.push(tmpl);
                }

                var groupKeys = Object.keys(grouped);
                for (var gi = 0; gi < groupKeys.length; gi++) {
                    var grp = grouped[groupKeys[gi]];
                    var contactLabel = grp.contact ? escHtml(grp.contact) : 'General';
                    var channelLabel = grp.channel ? escHtml(grp.channel) : 'unknown';

                    html += '<div class="card">';
                    html += '<div class="card-row">';
                    html += '<div class="card-channel">\uD83D\uDCDD</div>';
                    html += '<div class="card-content">';
                    html += '<div class="card-title">' + contactLabel + ' &mdash; ' + channelLabel + '</div>';

                    for (var ii = 0; ii < grp.items.length; ii++) {
                        var t = grp.items[ii];
                        var dirLabel = t.context === 'user_to_contact' ? 'You \u2192' : '\u2190 Them';
                        var formalPct = Math.round((t.formality || 0) * 100);

                        html += '<div class="insight-meta-row" style="margin-top:6px">';
                        html += '<span style="font-weight:600;color:var(--text-secondary)">' + escHtml(dirLabel) + '</span>';
                        if (t.greeting) html += '<span>Opens: <em>' + escHtml(t.greeting) + '</em></span>';
                        if (t.closing)  html += '<span>Closes: <em>' + escHtml(t.closing) + '</em></span>';
                        html += '<span>Formality: ' + formalPct + '%</span>';
                        if (t.typical_length) html += '<span>~' + Math.round(t.typical_length) + ' words</span>';
                        if (t.uses_emoji)    html += '<span>\uD83D\uDE00 Uses emoji</span>';
                        if (t.samples_analyzed) html += '<span>' + t.samples_analyzed + ' samples</span>';
                        html += '</div>';

                        // Show common phrases if present
                        if (t.common_phrases && t.common_phrases.length > 0) {
                            html += '<div class="card-meta" style="margin-top:4px">Common: ';
                            for (var pi2 = 0; pi2 < Math.min(t.common_phrases.length, 4); pi2++) {
                                html += '<span class="chip">' + escHtml(t.common_phrases[pi2]) + '</span>';
                            }
                            html += '</div>';
                        }
                        // Show tone notes if present
                        if (t.tone_notes && t.tone_notes.length > 0) {
                            html += '<div class="card-meta" style="margin-top:2px">Tone: ';
                            for (var tn = 0; tn < Math.min(t.tone_notes.length, 3); tn++) {
                                html += '<span class="chip">' + escHtml(t.tone_notes[tn]) + '</span>';
                            }
                            html += '</div>';
                        }
                    }

                    html += '</div></div></div>';
                }
            }

            if (!hasAny) {
                html = '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No insights yet. Use Life OS for a few days and patterns will emerge here.</div></div>';
            }
            safeSetContent(el, html);
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load insights: ' + escHtml(err.message) + '</div></div>');
        });
    }

    /**
     * Returns a representative emoji for each signal profile type.
     * Used as the icon in signal profile cards on the Insights tab.
     */
    function profileEmoji(type) {
        var emojis = {
            temporal:           '\u23F0',   // alarm clock
            linguistic:         '\u270D',   // writing hand
            linguistic_inbound: '\uD83D\uDCEC', // incoming envelope
            cadence:            '\u26A1',   // lightning (response speed)
            mood_signals:       '\uD83D\uDCC9', // chart (mood trends)
            relationships:      '\uD83D\uDC65', // two people
            topics:             '\uD83D\uDCDA', // books
            spatial:            '\uD83D\uDCCD', // pin
            decision:           '\u2696'    // scales
        };
        return emojis[type] || '\uD83D\uDCCA'; // bar chart fallback
    }

    /**
     * Renders a signal profile's data object as a grid of labelled metric tiles.
     *
     * Only scalar values and simple arrays of primitives are rendered — nested
     * objects are skipped to keep the UI readable. Numbers between 0 and 1
     * (exclusive) are displayed as percentages unless the key name contains
     * "count" or "total".
     *
     * @param {Object} data — Raw data dict from the signal profile.
     * @returns {string}   HTML string (signal-profile-grid or empty-state div).
     */
    function renderProfileData(data) {
        if (!data || typeof data !== 'object') return '';
        var items = [];
        var keys = Object.keys(data);
        for (var i = 0; i < keys.length; i++) {
            var k = keys[i];
            var v = data[k];
            if (v === null || v === undefined) continue;
            // Convert snake_case key to Title Case label
            var label = k.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
            var valueStr = '';
            if (typeof v === 'number') {
                // Render as percentage if in (0,1] and key doesn't suggest a raw count
                var isCount = k.indexOf('count') !== -1 || k.indexOf('total') !== -1 || k.indexOf('days') !== -1;
                valueStr = (!isCount && v > 0 && v <= 1)
                    ? Math.round(v * 100) + '%'
                    : (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2));
            } else if (typeof v === 'boolean') {
                valueStr = v ? 'Yes' : 'No';
            } else if (typeof v === 'string') {
                valueStr = v.replace(/_/g, ' ');
            } else if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'string') {
                // Compact display for string arrays (e.g., top_topics)
                valueStr = v.slice(0, 5).join(', ');
            } else if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'number') {
                // Compact display for number arrays (e.g., peak_hours: [9, 10, 14])
                valueStr = v.slice(0, 6).join(', ');
            } else {
                continue; // Skip nested objects and complex arrays
            }
            if (valueStr !== '') {
                items.push(
                    '<div class="signal-profile-item">' +
                        '<div class="signal-profile-label">' + escHtml(label) + '</div>' +
                        '<div class="signal-profile-value">' + escHtml(valueStr) + '</div>' +
                    '</div>'
                );
            }
        }
        if (!items.length) {
            return '<div style="color:var(--text-muted);font-size:12px;margin-top:4px">No data collected yet</div>';
        }
        return '<div class="signal-profile-grid">' + items.join('') + '</div>';
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

    /**
     * Delete a semantic fact from the user model.
     *
     * Uses the inline modal instead of the native confirm() dialog so the UI
     * works on mobile Safari where browser dialogs are blocked.
     *
     * @param {string} key - The fact key to delete (e.g. "preferred_language").
     */
    function deleteFact(key) {
        showConfirmModal(
            'Delete fact',
            'Are you sure you want to delete this fact? It will be removed from your user model.',
            'Delete',
            function() {
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
            },
            true  // danger = true → red confirm button
        );
    }

    /**
     * Mark a semantic fact as incorrect, reducing its confidence.
     *
     * Uses the inline modal instead of the native prompt() dialog so the UI
     * works on mobile Safari where browser dialogs are blocked.
     *
     * @param {string} key - The fact key to correct (e.g. "preferred_language").
     */
    function correctFact(key) {
        showPromptModal(
            'Correct fact',
            'What is incorrect? (optional — leave blank to just reduce confidence)',
            'Submit',
            function(reason) {
                fetch(API + '/api/user-model/facts/' + encodeURIComponent(key), {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reason: reason || 'User correction'})
                })
                .then(function(res) {
                    if (!res.ok) throw new Error('Failed');
                    var resp = document.getElementById('response');
                    resp.className = 'visible';
                    resp.textContent = 'Fact corrected \u2014 confidence reduced';
                    setTimeout(function() { resp.className = ''; }, 3000);
                    loadFactsFeed(); // Refresh to show updated confidence
                })
                .catch(function(err) { console.error('Correct fact failed:', err); });
            }
        );
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
        /* Load the System health dashboard.
         *
         * Makes two parallel requests:
         *   1. /health — event bus status, connector health checks, vector store
         *   2. /api/system/sources — per-source event stats with staleness flags
         *
         * Staleness (red warning) means a source hasn't emitted events within its
         * expected window (6h for external connectors, 24h for internal services).
         * This is the primary signal for "why isn't my data fresh?" debugging.
         */
        var el = document.getElementById('feedContent');
        safeSetContent(el, '<div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div>');

        Promise.all([
            fetch(API + '/health').then(function(r) { return r.json(); }),
            fetch(API + '/api/system/sources').then(function(r) { return r.json(); }),
            fetch(API + '/api/system/intelligence').then(function(r) { return r.json(); }).catch(function() { return {}; })
        ]).then(function(results) {
            var health = results[0];
            var sourcesData = results[1];
            var intel = results[2];
            var html = '';

            // ── Section 1: Infrastructure health ──────────────────────────
            html += '<div class="section-header">Infrastructure</div>';

            // Event bus
            var busOk = health.event_bus;
            html += '<div class="card sys-card">';
            html += '<div class="sys-row">';
            html += '<span class="sys-dot" style="background:' + (busOk ? 'var(--accent-green)' : 'var(--accent-red)') + '"></span>';
            html += '<span class="sys-name">Event Bus (NATS)</span>';
            html += '<span class="sys-status" style="color:' + (busOk ? 'var(--accent-green)' : 'var(--accent-red)') + '">' + (busOk ? '\u2713 Connected' : '\u2717 Disconnected') + '</span>';
            html += '</div></div>';

            // Vector store
            if (health.vector_store) {
                var docCount = health.vector_store.document_count || 0;
                html += '<div class="card sys-card">';
                html += '<div class="sys-row">';
                html += '<span class="sys-dot" style="background:var(--accent-green)"></span>';
                html += '<span class="sys-name">Vector Store (LanceDB)</span>';
                html += '<span class="sys-status">' + escHtml(String(docCount)) + ' docs indexed</span>';
                html += '</div></div>';
            }

            // Total events
            var total = health.events_stored || 0;
            html += '<div class="card sys-card">';
            html += '<div class="sys-row">';
            html += '<span class="sys-dot" style="background:var(--accent-blue)"></span>';
            html += '<span class="sys-name">Event Log</span>';
            html += '<span class="sys-status">' + escHtml(String(total.toLocaleString())) + ' total events</span>';
            html += '</div></div>';

            // ── Section 2: Data source sync status ─────────────────────────
            // This is the core diagnostic section — shows last-sync time and
            // flags any source that has gone silent beyond its expected window.
            var sources = sourcesData.sources || [];
            if (sources.length) {
                var staleCount = sourcesData.stale_count || 0;
                var sectionLabel = staleCount > 0
                    ? 'Data Sources \u26a0\ufe0f ' + staleCount + ' stale'
                    : 'Data Sources \u2713 all current';
                html += '<div class="section-header" style="' + (staleCount > 0 ? 'color:var(--accent-yellow)' : '') + '">' + escHtml(sectionLabel) + '</div>';

                for (var i = 0; i < sources.length; i++) {
                    var s = sources[i];
                    var isStale = s.stale;
                    var dotColor = isStale ? 'var(--accent-red)' : 'var(--accent-green)';
                    var nameColor = isStale ? 'color:var(--accent-red)' : '';

                    // Format "last seen" label
                    var lastSeenLabel = 'never';
                    if (s.hours_since !== null && s.hours_since !== undefined) {
                        var h = s.hours_since;
                        if (h < 1) lastSeenLabel = Math.round(h * 60) + 'm ago';
                        else if (h < 24) lastSeenLabel = Math.round(h) + 'h ago';
                        else lastSeenLabel = Math.round(h / 24) + 'd ago';
                    }

                    // Format event rate for the last 24h
                    var rate24h = s.events_24h > 0
                        ? escHtml(String(s.events_24h)) + ' in 24h'
                        : '<span style="color:var(--text-muted)">0 in 24h</span>';

                    html += '<div class="card sys-card' + (isStale ? ' sys-stale' : '') + '">';
                    html += '<div class="sys-row">';
                    html += '<span class="sys-dot" style="background:' + dotColor + '"></span>';
                    html += '<span class="sys-name" style="' + nameColor + '">' + escHtml(s.source) + '</span>';
                    html += '<span class="sys-last-seen" style="' + (isStale ? 'color:var(--accent-red)' : '') + '">' + escHtml(lastSeenLabel) + '</span>';
                    html += '<span class="sys-rate">' + rate24h + '</span>';
                    html += '</div>';
                    // Show stale warning inline so the user immediately understands the impact
                    if (isStale) {
                        html += '<div class="sys-stale-msg">\u26a0\ufe0f Source appears stale — check connector authentication or network</div>';
                    }
                    html += '</div>';
                }
            }

            // ── Section 3: Live connector health ──────────────────────────
            // /health polls each active connector's health_check() in real time.
            var connectors = health.connectors || [];
            if (connectors.length) {
                html += '<div class="section-header">Active Connectors</div>';
                for (var j = 0; j < connectors.length; j++) {
                    var c = connectors[j];
                    var cStatus = c.status || c.state || 'unknown';
                    var cOk = cStatus === 'ok' || cStatus === 'active' || cStatus === 'connected';
                    html += '<div class="card sys-card">';
                    html += '<div class="sys-row">';
                    html += '<span class="sys-dot" style="background:' + (cOk ? 'var(--accent-green)' : 'var(--accent-red)') + '"></span>';
                    html += '<span class="sys-name">' + escHtml(c.connector || c.name || 'Connector') + '</span>';
                    html += '<span class="sys-status" style="color:' + (cOk ? 'var(--accent-green)' : 'var(--accent-red)') + '">' + escHtml(cStatus) + '</span>';
                    if (c.details && !cOk) {
                        html += '<span class="sys-error">' + escHtml(String(c.details).slice(0, 80)) + '</span>';
                    }
                    html += '</div></div>';
                }
            }

            // ── Section 4: Database integrity ─────────────────────────────
            // Reports the output of PRAGMA quick_check per database file.
            // A "corrupted" status means certain tables or pages are unreadable,
            // which silently degrades features like signal profiles and routines.
            var dbHealth = health.db_health || {};
            var dbNames = Object.keys(dbHealth);
            if (dbNames.length) {
                var anyCorrupted = health.db_status === 'degraded';
                var dbLabel = anyCorrupted
                    ? 'Databases \u26a0\ufe0f integrity issues detected'
                    : 'Databases \u2713 all healthy';
                html += '<div class="section-header" style="' + (anyCorrupted ? 'color:var(--accent-red)' : '') + '">' + escHtml(dbLabel) + '</div>';

                for (var di = 0; di < dbNames.length; di++) {
                    var dname = dbNames[di];
                    var dinfo = dbHealth[dname];
                    var dOk = dinfo.status === 'ok';
                    var dDot = dOk ? 'var(--accent-green)' : 'var(--accent-red)';
                    var sizeMB = dinfo.size_bytes ? (dinfo.size_bytes / 1048576).toFixed(1) + ' MB' : 'not found';

                    html += '<div class="card sys-card' + (dOk ? '' : ' sys-stale') + '">';
                    html += '<div class="sys-row">';
                    html += '<span class="sys-dot" style="background:' + dDot + '"></span>';
                    html += '<span class="sys-name">' + escHtml(dname) + '.db</span>';
                    html += '<span class="sys-status" style="color:' + dDot + '">' + escHtml(dinfo.status) + '</span>';
                    html += '<span class="sys-rate">' + escHtml(sizeMB) + '</span>';
                    html += '</div>';
                    if (!dOk && dinfo.errors && dinfo.errors.length) {
                        // Show up to 2 error lines so the user has a hint without overflow
                        for (var ei = 0; ei < Math.min(dinfo.errors.length, 2); ei++) {
                            html += '<div class="sys-stale-msg" style="color:var(--accent-red)">\u26a0\ufe0f ' + escHtml(dinfo.errors[ei]) + '</div>';
                        }
                        if (dinfo.errors.length > 2) {
                            html += '<div class="sys-stale-msg" style="color:var(--text-muted)">... and ' + (dinfo.errors.length - 2) + ' more errors</div>';
                        }
                    }
                    html += '</div>';
                }
            }

            // ── Section 5: Intelligence engine diagnostics ────────────────
            // Shows prediction engine health: which prediction types are active
            // or blocked, how many predictions were generated in the last 7 days,
            // and how deep the user model is (signal profiles, routines, episodes).
            // This is the single most useful section for understanding why Life OS's
            // intelligence layer is or isn't generating useful insights.
            var overall = intel.overall || {};
            var predTypes = intel.prediction_types || {};
            var depth = intel.user_model_depth || {};
            var overallHealth = overall.health || 'unknown';
            var healthColor = overallHealth === 'healthy' ? 'var(--accent-green)'
                : overallHealth === 'degraded' ? 'var(--accent-yellow)'
                : overallHealth === 'broken' ? 'var(--accent-red)' : 'var(--text-muted)';
            var totalPreds = overall.total_predictions_7d || 0;
            var activePredTypes = overall.active_types || 0;
            var blockedPredTypes = overall.blocked_types || 0;

            if (Object.keys(predTypes).length || Object.keys(depth).length) {
                html += '<div class="section-header" style="color:' + healthColor + '">Prediction Engine — ' + escHtml(overallHealth) + '</div>';

                // Summary card: 7-day stats and model depth at a glance
                html += '<div class="card sys-card">';
                html += '<div class="sys-row"><span class="sys-dot" style="background:' + healthColor + '"></span>';
                html += '<span class="sys-name">7-day predictions</span>';
                html += '<span class="sys-status">' + escHtml(String(totalPreds)) + ' generated';
                if (activePredTypes || blockedPredTypes) {
                    html += ' \u00b7 ' + escHtml(String(activePredTypes)) + ' active, ' + escHtml(String(blockedPredTypes)) + ' blocked';
                }
                html += '</span></div>';
                // User model depth row
                if (Object.keys(depth).length) {
                    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:4px;padding:0 4px">';
                    html += escHtml(String(depth.episodes || 0)) + ' episodes \u00b7 ';
                    html += escHtml(String(depth.semantic_facts || 0)) + ' facts \u00b7 ';
                    html += escHtml(String(depth.routines || 0)) + ' routines \u00b7 ';
                    html += escHtml(String(depth.workflows || 0)) + ' workflows \u00b7 ';
                    html += escHtml(String(depth.signal_profiles || 0)) + ' signal profiles';
                    html += '</div>';
                }
                html += '</div>';

                // Per-type breakdown — only show types where we have useful data
                var typeKeys = Object.keys(predTypes);
                for (var ti = 0; ti < typeKeys.length; ti++) {
                    var tname = typeKeys[ti];
                    var tdata = predTypes[tname];
                    var tStatus = tdata.status || 'unknown';
                    var tCount = tdata.generated_last_7d || 0;
                    var tDotColor = tStatus === 'active' ? 'var(--accent-green)'
                        : tStatus === 'limited' ? 'var(--accent-yellow)' : 'var(--accent-red)';
                    var tBlockers = tdata.blockers || [];

                    html += '<div class="card sys-card' + (tStatus === 'blocked' ? ' sys-stale' : '') + '">';
                    html += '<div class="sys-row">';
                    html += '<span class="sys-dot" style="background:' + tDotColor + '"></span>';
                    html += '<span class="sys-name">' + escHtml(tname) + '</span>';
                    html += '<span class="sys-status" style="color:' + tDotColor + '">' + escHtml(tStatus) + '</span>';
                    html += '<span class="sys-rate">' + escHtml(String(tCount)) + ' in 7d</span>';
                    html += '</div>';
                    // Show up to 2 blockers if any, so the user knows what's preventing predictions
                    for (var bi = 0; bi < Math.min(tBlockers.length, 2); bi++) {
                        html += '<div class="sys-stale-msg">\u26a0\ufe0f ' + escHtml(tBlockers[bi]) + '</div>';
                    }
                    html += '</div>';
                }
            }

            if (!html) {
                html = '<div class="card"><div class="card-meta" style="text-align:center;padding:20px">No system data available</div></div>';
            }
            safeSetContent(el, html);
        })
        .catch(function(err) {
            safeSetContent(el, '<div class="card"><div class="card-meta" style="color:var(--accent-red)">Failed to load system status: ' + escHtml(err.message) + '</div></div>');
        });
    }

    // --- Card Interaction ---
    function toggleCard(id) {
        // Collapse the previously-expanded card without touching the DOM for
        // any other card. This avoids the full-feed re-render that caused a
        // visual flash and scroll-position loss on every expand/collapse.
        if (expandedCardId && expandedCardId !== id) {
            var prevCard = document.querySelector('[data-id="' + expandedCardId + '"]');
            if (prevCard) prevCard.classList.remove('expanded');
        }

        var card = document.querySelector('[data-id="' + id + '"]');
        if (!card) return;

        if (expandedCardId === id) {
            // Clicking an already-expanded card collapses it.
            expandedCardId = null;
            card.classList.remove('expanded');
        } else {
            // Expand the clicked card.  CSS rule
            // ".card.expanded .card-detail { display: block }" handles
            // revealing the detail panel — no inline style needed.
            expandedCardId = id;
            card.classList.add('expanded');
        }
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

    /**
     * Generate an AI-drafted reply for a feed item and display it in the card.
     *
     * Looks up the full feed item from feedItems[] to extract the message body
     * (the DraftRequest schema requires 'incoming_message' — passing only the
     * card title gives the AI nothing to work with).  Also passes contact_id
     * (sender address) so the AI engine can apply per-contact communication
     * templates and match the user's established style with that contact.
     *
     * The draft is rendered in the #draft-{id} placeholder inside the card
     * detail section, followed by a Copy button.
     *
     * Example (called from email and message card action buttons):
     *   onclick="draftReply('abc123', 'Re: Meeting tomorrow')"
     *
     * @param {string} id      Feed item ID (used to find the draft placeholder).
     * @param {string} context Card title / subject used as fallback message body.
     */
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

        // Extract the sender address as contact_id so the AI engine can look up
        // communication templates for this contact (formality, greeting, style).
        var contactId = (item && item.metadata && item.metadata.from_address) || null;

        fetch(API + '/api/draft', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                incoming_message: messageBody,
                context: context,
                channel: (item && item.channel) || 'email',
                contact_id: contactId
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

    /**
     * Start an outreach draft for a People Radar contact.
     *
     * Called when the user clicks a person-card in the People Radar sidebar.
     * Shows a confirmation modal, then calls POST /api/draft to generate an
     * AI outreach message, and displays it in a follow-up modal with a Copy
     * button.
     *
     * @param {string} contactId    Contact ID for the draft API.
     * @param {string} name         Display name of the contact.
     * @param {string} channel      Preferred channel (email, imessage, signal).
     * @param {string} contactEmail Contact email (for display context).
     */
    function startOutreach(contactId, name, channel, contactEmail) {
        showConfirmModal(
            'Draft a message to ' + escHtml(name) + '?',
            'Channel: ' + escHtml(channel) + (contactEmail ? ' (' + escHtml(contactEmail) + ')' : ''),
            'Draft Message',
            function() {
                // Show loading modal while draft generates
                safeSetContent(document.getElementById('modalTitle'), escHtml('Drafting message to ' + name + '...'));
                safeSetContent(document.getElementById('modalBody'),
                    '<div style="opacity:0.5;padding:12px">Generating draft...</div>');
                safeSetContent(document.getElementById('modalActions'),
                    '<button class="modal-btn-cancel" onclick="closeModal()">Cancel</button>');
                document.getElementById('modalOverlay').classList.add('visible');

                fetch(API + '/api/draft', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        incoming_message: 'Reaching out to ' + name,
                        contact_id: contactId,
                        channel: channel
                    })
                })
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    var draft = data.draft || data.content || 'No draft generated';
                    safeSetContent(document.getElementById('modalTitle'), escHtml('Outreach to ' + name));
                    safeSetContent(document.getElementById('modalBody'),
                        '<div class="draft-area" style="white-space:pre-wrap;padding:8px;background:var(--bg-card);border-radius:6px;font-size:12px;margin-bottom:8px">' + escHtml(draft) + '</div>');
                    safeSetContent(document.getElementById('modalActions'),
                        '<button class="modal-btn-cancel" onclick="closeModal()">Close</button>' +
                        '<button class="modal-btn-confirm" id="modalCopyBtn">Copy</button>');
                    document.getElementById('modalCopyBtn').onclick = function() {
                        navigator.clipboard.writeText(draft).then(function() {
                            document.getElementById('modalCopyBtn').textContent = 'Copied!';
                            setTimeout(function() {
                                var btn = document.getElementById('modalCopyBtn');
                                if (btn) btn.textContent = 'Copy';
                            }, 2000);
                        });
                    };
                })
                .catch(function(err) {
                    safeSetContent(document.getElementById('modalBody'),
                        '<div style="color:var(--accent-red)">Failed to generate draft: ' + escHtml(err.message) + '</div>');
                });
            }
        );
    }

    /**
     * Send the text in the quick-reply textarea directly via POST /api/messages/send.
     *
     * The message is routed to the right connector (iMessage or Signal) based on
     * ``channel``.  While the request is in-flight, the textarea and button are
     * disabled to prevent double-sends.  On success the textarea is cleared and a
     * toast confirmation is shown.  On failure (no connector configured, network
     * error) a descriptive toast is shown so the user knows the message was not sent.
     *
     * Ctrl+Enter in the textarea also triggers this function (wired in renderCard).
     *
     * @param {string} cardId   - The feed-item ID, used to find textarea/button.
     * @param {string} recipient - The address to send to (phone, Apple ID, Signal).
     * @param {string} channel  - Transport hint: "imessage", "signal", or "message".
     */
    function sendQuickReply(cardId, recipient, channel) {
        var textarea = document.getElementById('qr-' + cardId);
        var btn = document.getElementById('qr-btn-' + cardId);
        if (!textarea) return;

        var message = textarea.value.trim();
        if (!message) {
            textarea.focus();
            return;
        }

        // Disable controls while request is in-flight to prevent double-sends
        textarea.disabled = true;
        if (btn) btn.disabled = true;

        fetch(API + '/api/messages/send', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({recipient: recipient, message: message, channel: channel})
        })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var resp = document.getElementById('response');
            if (data.status === 'sent') {
                textarea.value = '';
                resp.className = 'visible';
                resp.textContent = 'Message sent via ' + (data.connector || channel);
            } else if (data.status === 'no_connector') {
                resp.className = 'visible';
                resp.textContent = 'No messaging connector active. Configure iMessage or Signal in Admin.';
            } else {
                resp.className = 'visible';
                resp.textContent = 'Send failed: ' + (data.details || 'Unknown error');
            }
            setTimeout(function() { resp.className = ''; }, 4000);
        })
        .catch(function(err) {
            var resp = document.getElementById('response');
            resp.className = 'visible';
            resp.textContent = 'Send failed: ' + err.message;
            setTimeout(function() { resp.className = ''; }, 4000);
        })
        .finally(function() {
            textarea.disabled = false;
            if (btn) btn.disabled = false;
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
        .then(function(res) {
            if (!res.ok) {
                return res.json().catch(function() { return {}; }).then(function(err) {
                    throw new Error(err.detail || err.error || 'Command failed (HTTP ' + res.status + ')');
                });
            }
            return res.json();
        })
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
                resp.textContent = data.content || data.briefing || data.message || 'Command processed.';
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
            safeSetContent(el, '<div style="color:var(--text-muted)">No briefing today. Connect your email or calendar in <a href="/admin" style="color:var(--accent-blue)">Settings</a> to enable daily briefings.</div>');
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
                safeSetContent(el, '<div style="color:var(--text-muted)">No predictions yet. Predictions appear as Life OS learns your communication, calendar, and spending patterns.</div>');
                return;
            }
            var html = '';
            for (var i = 0; i < Math.min(preds.length, 5); i++) {
                var p = preds[i];
                var prediction_type = p.prediction_type || '';
                var signals = p.supporting_signals || {};
                var contactEmail = signals.contact_email || '';
                var suggestedAction = p.suggested_action || '';
                html += '<div class="prediction-card" id="pred-' + escAttr(p.id) + '">';
                html += '<div class="pred-label">' + escHtml(prediction_type || 'Prediction') + '</div>';
                html += '<div>' + escHtml(p.description || '') + '</div>';
                if (p.confidence !== undefined) {
                    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:2px">' + escHtml(String(Math.round(p.confidence * 100))) + '% confidence</div>';
                }
                if (suggestedAction) {
                    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:2px">\u2192 ' + escHtml(suggestedAction) + '</div>';
                }
                html += '<div class="card-actions" style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap">';
                if (prediction_type === 'reminder' && contactEmail) {
                    html += '<button class="btn-small btn-primary" onclick="event.stopPropagation();draftPredictionReply(\'' + escAttr(p.id) + '\',\'' + escAttr(contactEmail) + '\',\'' + escAttr(p.description || '') + '\')">Draft Reply</button>';
                }
                if (prediction_type === 'conflict') {
                    html += '<button class="btn-small btn-primary" onclick="event.stopPropagation();switchTopic(\'calendar\')">View Calendar</button>';
                }
                html += '<button class="btn-small btn-success" onclick="event.stopPropagation();predictionActedOn(\'' + escAttr(p.id) + '\')">Done</button>';
                html += '<button class="btn-small" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',true,null)">Accurate</button>';
                html += '<button class="btn-small btn-danger" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',false,null)">Inaccurate</button>';
                html += '<button class="btn-small btn-warn" onclick="event.stopPropagation();predictionFeedback(\'' + escAttr(p.id) + '\',false,\'not_relevant\')">Not Me</button>';
                html += '</div>';
                if (prediction_type === 'reminder' && contactEmail) {
                    html += '<div id="pred-draft-' + escAttr(p.id) + '"></div>';
                }
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

    function predictionActedOn(id) {
        var url = API + '/api/predictions/' + encodeURIComponent(id) + '/feedback?was_accurate=true&user_response=acted_on';
        fetch(url, {method: 'POST'})
        .then(function(res) {
            if (!res.ok) throw new Error('Failed');
            var card = document.getElementById('pred-' + id);
            if (card) {
                card.style.transition = 'opacity 0.3s';
                card.style.opacity = '0';
                setTimeout(function() { card.style.display = 'none'; }, 350);
            }
        })
        .catch(function(err) { console.error('Prediction acted-on failed:', err); });
    }

    function draftPredictionReply(predId, contactEmail, context) {
        var placeholder = document.getElementById('pred-draft-' + predId);
        if (placeholder) placeholder.textContent = 'Drafting reply...';
        fetch(API + '/api/draft', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({incoming_message: context, contact_id: contactEmail, channel: 'email'})
        })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (placeholder) {
                // Content is escaped via escHtml before insertion — safe against XSS
                placeholder.innerHTML = '<div style="margin-top:6px;padding:8px;background:var(--bg-secondary);border-radius:6px;font-size:12px;white-space:pre-wrap">' + escHtml(data.draft || data.content || 'No draft generated') + '</div>';
            }
        })
        .catch(function(err) {
            if (placeholder) placeholder.textContent = 'Draft failed: ' + err.message;
        });
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
                safeSetContent(el, '<div style="color:var(--text-muted)">No contacts tracked yet. Relationship patterns appear after analyzing your emails and messages.</div>');
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

                var channel = c.preferred_channel || 'email';
                var contactEmail = (c.emails && c.emails.length > 0) ? c.emails[0] : '';
                var contactId = c.id || c.contact_id || '';

                html += '<div class="person-card" style="cursor:pointer" onclick="startOutreach(\'' + escAttr(contactId) + '\',\'' + escAttr(name) + '\',\'' + escAttr(channel) + '\',\'' + escAttr(contactEmail) + '\')">';
                html += '<div class="person-avatar" style="background:' + dotColor + '">' + escHtml(name.charAt(0).toUpperCase()) + '</div>';
                html += '<div class="person-info">';
                html += '<div class="person-name">' + escHtml(name) + ' <span class="channel-badge">' + escHtml(channel) + '</span></div>';
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

    /**
     * Toggle the AI sidebar as a full-screen overlay on small screens (< 900px).
     *
     * On wide screens this function is unreachable (the FAB is hidden by CSS),
     * so it only runs when the viewport is narrow.  Adds / removes
     * `mobile-open` class on #aiSidebar which is wired to the overlay CSS rule.
     *
     * Loads sidebar data on first open so the overlay has real content to show.
     *
     * Example (called from the FAB and the ✕ close button inside the sidebar):
     *   onclick="toggleMobileSidebar()"
     */
    function toggleMobileSidebar() {
        var sidebar = document.getElementById('aiSidebar');
        var isOpen = sidebar.classList.toggle('mobile-open');
        // Load sidebar data the first time the overlay opens so the user
        // sees real content rather than skeletons.  Subsequent opens skip
        // the fetch if data is already present (the sidebar persists in DOM).
        if (isOpen) {
            loadMood();
            loadPredictions();
            loadPeopleRadar();
        }
    }

    // --- Badge Counts ---
    // --- Inline Modal System ---
    // Replaces native browser confirm() / prompt() dialogs with themed UI that
    // works on mobile Safari (where browser dialogs are often blocked) and
    // matches the app's dark colour scheme.

    /**
     * Close the modal overlay.
     *
     * Called by the overlay backdrop click handler and every Cancel button.
     */
    function closeModal() {
        document.getElementById('modalOverlay').classList.remove('visible');
    }

    /**
     * Show a confirmation modal with optional danger styling.
     *
     * @param {string}   title        - Heading text shown in the modal.
     * @param {string}   message      - Body copy describing the action.
     * @param {string}   confirmLabel - Text on the confirm button.
     * @param {Function} onConfirm    - Callback invoked when the user confirms.
     * @param {boolean}  [danger]     - If true, styles confirm button as destructive (red).
     *
     * Example:
     *   showConfirmModal('Delete fact', 'This cannot be undone.', 'Delete',
     *       function() { deleteFactById(key); }, true);
     */
    function showConfirmModal(title, message, confirmLabel, onConfirm, danger) {
        safeSetContent(document.getElementById('modalTitle'), escHtml(title));
        safeSetContent(document.getElementById('modalBody'),
            '<div>' + escHtml(message) + '</div>');
        safeSetContent(document.getElementById('modalActions'),
            '<button class="modal-btn-cancel" onclick="closeModal()">Cancel</button>' +
            '<button class="' + (danger ? 'modal-btn-danger' : 'modal-btn-confirm') +
            '" id="modalConfirmBtn">' + escHtml(confirmLabel) + '</button>');
        document.getElementById('modalConfirmBtn').onclick = function() {
            closeModal();
            onConfirm();
        };
        document.getElementById('modalOverlay').classList.add('visible');
    }

    /**
     * Show a text-input prompt modal.
     *
     * @param {string}   title        - Heading text shown in the modal.
     * @param {string}   placeholder  - Placeholder text inside the input field.
     * @param {string}   confirmLabel - Text on the confirm button.
     * @param {Function} onConfirm    - Callback invoked with the entered string (may be empty).
     *
     * Example:
     *   showPromptModal('Add note', 'Enter note text...', 'Save',
     *       function(text) { saveNote(text); });
     */
    function showPromptModal(title, placeholder, confirmLabel, onConfirm) {
        safeSetContent(document.getElementById('modalTitle'), escHtml(title));
        safeSetContent(document.getElementById('modalBody'),
            '<input type="text" id="modalInput" placeholder="' + escAttr(placeholder) + '">');
        safeSetContent(document.getElementById('modalActions'),
            '<button class="modal-btn-cancel" onclick="closeModal()">Cancel</button>' +
            '<button class="modal-btn-confirm" id="modalConfirmBtn">' + escHtml(confirmLabel) + '</button>');
        document.getElementById('modalConfirmBtn').onclick = function() {
            var val = document.getElementById('modalInput').value;
            closeModal();
            onConfirm(val);
        };
        // Allow Enter key to confirm without clicking the button.
        document.getElementById('modalInput').onkeydown = function(e) {
            if (e.key === 'Enter') { document.getElementById('modalConfirmBtn').click(); }
        };
        document.getElementById('modalOverlay').classList.add('visible');
        // Delay focus until the next tick so the CSS transition completes first.
        setTimeout(function() {
            var inp = document.getElementById('modalInput');
            if (inp) inp.focus();
        }, 100);
    }

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
        // Single request to /api/dashboard/badges returns all counts at once.
        // This replaces the previous pattern of 5 separate full-feed requests
        // (each with limit=100) which fetched ~50 KB of items just to get counts.
        fetch(API + '/api/dashboard/badges')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var badges = data.badges || {};
            var topicIds = Object.keys(badges);
            for (var i = 0; i < topicIds.length; i++) {
                setBadge(topicIds[i], badges[topicIds[i]] || 0);
            }
        })
        .catch(function() {
            // Silently fail — badge counts are non-critical UI decoration.
        });
    }

    // --- Status Bar ---
    function loadStatus() {
        /* Refresh the bottom status bar with live health and data-freshness info.
         *
         * Fetches /health (connector health checks) and /api/system/sources
         * (per-source event stats) in parallel, then updates three status bar
         * slots:
         *   1. Connection dot — green when NATS event bus is connected
         *   2. Event count — total events logged
         *   3. Data freshness — "N sources syncing" or a stale-source warning
         *
         * The stale-source warning (⚠ N stale) links to the System topic so
         * the user can drill in to see which source stopped and why.
         */
        Promise.all([
            fetch(API + '/health').then(function(r) { return r.json(); }),
            fetch(API + '/api/system/sources').then(function(r) { return r.json(); })
        ]).then(function(results) {
            var health = results[0];
            var sourcesData = results[1];

            var dot = document.getElementById('statusDot');
            var text = document.getElementById('statusText');
            var count = document.getElementById('eventCount');
            var connStatus = document.getElementById('connectorStatus');

            dot.className = 'status-dot ' + (health.status === 'ok' ? 'ok' : 'error');
            text.textContent = health.event_bus ? 'Connected' : 'Event bus disconnected';
            count.textContent = (health.events_stored || 0).toLocaleString() + ' events';

            // Show stale-source warning if any external connector is silent —
            // this is the #1 sign the user should check their connector config.
            var staleCount = sourcesData.stale_count || 0;
            if (staleCount > 0) {
                // All values used here are numeric — no XSS risk.
                connStatus.textContent = '\u26a0\ufe0f ' + staleCount +
                    ' stale source' + (staleCount > 1 ? 's' : '');
                connStatus.style.color = 'var(--accent-yellow)';
                connStatus.style.cursor = 'pointer';
                connStatus.onclick = function() { switchTopic('system'); };
            } else {
                var activeCount = (sourcesData.sources || []).filter(function(s) { return !s.stale; }).length;
                connStatus.textContent = activeCount + ' source' + (activeCount !== 1 ? 's' : '') + ' syncing';
                connStatus.style.color = '';
                connStatus.style.cursor = '';
                connStatus.onclick = null;
            }
        })
        .catch(function() {
            document.getElementById('statusDot').className = 'status-dot error';
            document.getElementById('statusText').textContent = 'Offline';
        });
    }

    // --- Data Freshness Warning ---
    function checkDataFreshness() {
        // Don't show if user dismissed in this session
        if (sessionStorage.getItem('stale-dismissed')) return;

        fetch(API + '/api/system/sources')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            var sources = data.sources || [];
            // Filter out internal sources that are expected to be idle when no
            // external data flows in — they don't self-initiate.
            var stale = sources.filter(function(s) {
                return s.stale && s.source !== 'user_model_store' && s.source !== 'rules_engine';
            });
            var banner = document.getElementById('staleDataBanner');
            if (stale.length === 0) {
                banner.classList.remove('visible');
                return;
            }
            var msgs = stale.map(function(s) {
                var hrs = Math.round(s.hours_since || 0);
                var timeStr = hrs > 48 ? Math.round(hrs / 24) + ' days' : hrs + ' hours';
                return escHtml(s.source) + ' (' + timeStr + ' ago)';
            });
            safeSetContent(document.getElementById('staleDataMsg'),
                'Data may be outdated \u2014 ' + msgs.join(', ') + ' not syncing. <a href="/admin" style="color:inherit;text-decoration:underline">Check connector settings</a>');
            banner.classList.add('visible');
        })
        .catch(function() {});
    }

    function dismissStaleWarning() {
        document.getElementById('staleDataBanner').classList.remove('visible');
        sessionStorage.setItem('stale-dismissed', '1');
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
        checkDataFreshness();
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
            // Use targeted class removal — same pattern as toggleCard() — to
            // avoid the full feed re-render and its associated scroll-position loss.
            var card = document.querySelector('[data-id="' + expandedCardId + '"]');
            if (card) card.classList.remove('expanded');
            expandedCardId = null;
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
    checkDataFreshness();

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

    // Re-check data freshness every 5 minutes.
    setInterval(checkDataFreshness, 300000);

    // --- WebSocket ---
    try { connectWS(); } catch(e) {}

    </script>

<!-- Mobile sidebar FAB: opens the AI sidebar as a full-screen overlay on
     small screens (< 900px). Hidden via CSS on wide screens. -->
<button class="mobile-sidebar-fab" id="mobileSidebarFab"
        onclick="toggleMobileSidebar()" aria-label="Open AI sidebar">AI</button>

<!-- Inline modal overlay: replaces native confirm()/prompt() dialogs.
     Clicking the backdrop or the Cancel button closes the modal.
     The modal-box click handler stops propagation so clicking inside
     the box doesn't dismiss it. -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModal()">
    <div class="modal-box" onclick="event.stopPropagation()">
        <div class="modal-title" id="modalTitle"></div>
        <div class="modal-body" id="modalBody"></div>
        <div class="modal-actions" id="modalActions"></div>
    </div>
</div>

</body>
</html>"""
