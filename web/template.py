"""
Life OS — HTML Template for the Web UI

Single-page dashboard served at the root URL ("/").  Everything — HTML, CSS,
and JavaScript — is embedded in a single Python string constant so the web
module has zero static-file dependencies.

Architecture:
    CSS   — Dark-theme design system using CSS variables and a card-based layout.
            Built with system fonts (-apple-system stack) for native look and feel.
    HTML  — Minimal semantic structure: command bar, response area, notifications
            section, tasks section, and system status.  A fixed status bar at the
            bottom shows connection health and event count.
    JS    — Client-side logic handles:
            1. Command bar: sends input to POST /api/command on Enter, renders
               the response by type (search results, task confirmation, etc.).
            2. Polling: notifications refresh every 60 s, system status every 30 s.
            3. WebSocket: connects to /ws for real-time push updates with
               automatic reconnection on close (5-second delay).
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life OS</title>
    <style>
        /* --- CSS Reset & Base --- */
        /* Border-box sizing prevents padding from expanding element dimensions. */
        * { margin: 0; padding: 0; box-sizing: border-box; }

        /* Dark theme: near-black background (#0a0a0a) with light gray text (#e0e0e0).
           System font stack ensures native look across macOS, Windows, and Linux. */
        body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
               background: #0a0a0a; color: #e0e0e0; min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
        h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; color: #fff; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 40px; }

        /* --- Command Bar ---
           Full-width input with rounded corners and a subtle border highlight on focus.
           Acts as the primary interaction point for the dashboard. */
        .command-bar { position: relative; margin-bottom: 40px; }
        .command-bar input {
            width: 100%; padding: 16px 20px; font-size: 16px;
            background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
            color: #fff; outline: none; transition: border-color 0.2s;
        }
        .command-bar input:focus { border-color: #4a9eff; }
        .command-bar input::placeholder { color: #555; }

        /* --- Sections --- */
        .section { margin-bottom: 32px; }
        .section-title { font-size: 13px; font-weight: 600; text-transform: uppercase;
                         letter-spacing: 0.5px; color: #666; margin-bottom: 12px; }

        /* --- Card-based Layout ---
           Each notification/task/status block is a card with dark background,
           subtle border, and hover feedback.  Priority variants add a colored
           left border to draw attention (orange for high, red for critical). */
        .card { background: #1a1a1a; border: 1px solid #222; border-radius: 10px;
                padding: 16px; margin-bottom: 8px; cursor: pointer;
                transition: border-color 0.2s; }
        .card:hover { border-color: #444; }
        .card-title { font-weight: 500; margin-bottom: 4px; }
        .card-meta { font-size: 13px; color: #666; }
        .card-priority-high { border-left: 3px solid #ff6b35; }
        .card-priority-critical { border-left: 3px solid #ff3535; }

        /* --- Response Area ---
           Hidden by default (display:none); toggled visible when the command bar
           produces output.  Uses pre-wrap to preserve AI-generated formatting. */
        #response { background: #111; border-radius: 10px; padding: 20px;
                    min-height: 100px; white-space: pre-wrap; line-height: 1.6;
                    display: none; margin-bottom: 32px; font-size: 15px; }
        #response.visible { display: block; }

        /* --- Fixed Status Bar ---
           Always visible at the bottom of the viewport.  Shows a colored dot
           (green = ok, red = error) and the total event count. */
        .status-bar { position: fixed; bottom: 0; left: 0; right: 0;
                      background: #111; border-top: 1px solid #222;
                      padding: 8px 20px; font-size: 12px; color: #555;
                      display: flex; justify-content: space-between; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%;
                      display: inline-block; margin-right: 6px; }
        .status-dot.ok { background: #4aff6b; }
        .status-dot.error { background: #ff3535; }

        /* --- Loading State --- */
        .loading { opacity: 0.5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Life OS</h1>
        <p class="subtitle">Your private command center</p>
        
        <div class="command-bar">
            <input type="text" id="commandInput" 
                   placeholder="Ask anything, search, create tasks..."
                   autofocus autocomplete="off">
        </div>
        
        <div id="response"></div>
        
        <div class="section" id="notificationsSection">
            <div class="section-title">Notifications</div>
            <div id="notifications">Loading...</div>
        </div>
        
        <div class="section" id="tasksSection">
            <div class="section-title">Tasks</div>
            <div id="tasks">Loading...</div>
        </div>
        
        <div class="section">
            <div class="section-title">System</div>
            <div id="systemStatus">Loading...</div>
        </div>
    </div>
    
    <div class="status-bar">
        <span><span class="status-dot" id="statusDot"></span><span id="statusText">Connecting...</span></span>
        <span id="eventCount"></span>
    </div>

    <script>
        // --- Configuration ---
        // Empty string = same-origin requests (no CORS needed for the dashboard).
        const API = '';

        // --- Command Bar ---
        // Captures Enter keypresses, sends the input to POST /api/command,
        // and renders the response based on its ``type`` field.
        const input = document.getElementById('commandInput');
        const response = document.getElementById('response');

        input.addEventListener('keydown', async (e) => {
            // Only fire on Enter with non-empty input.
            if (e.key !== 'Enter' || !input.value.trim()) return;

            const text = input.value.trim();
            input.value = '';
            // Show the response area with a loading indicator while waiting.
            response.className = 'visible loading';
            response.textContent = 'Thinking...';

            try {
                const res = await fetch(`${API}/api/command`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text}),
                });
                const data = await res.json();
                response.className = 'visible';

                // Render response based on the command type returned by the server.
                if (data.type === 'search_results') {
                    // Format search results: show similarity score and a preview of each match.
                    response.textContent = data.results.length
                        ? data.results.map(r => `[${(r.score*100).toFixed(0)}%] ${r.text.slice(0,150)}`).join('\\n\\n')
                        : 'No results found.';
                } else if (data.type === 'task_created') {
                    response.textContent = `Task created.`;
                    // Refresh the tasks list to show the newly created task.
                    loadTasks();
                } else {
                    // Briefing, draft, AI response, or any other type — display content directly.
                    response.textContent = data.content || data.briefing || JSON.stringify(data, null, 2);
                }
            } catch (err) {
                response.className = 'visible';
                response.textContent = `Error: ${err.message}`;
            }
        });
        
        // --- Notifications Polling ---
        // Fetches pending notifications from the API and renders them as cards.
        // Called on initial load and every 60 seconds via setInterval.
        async function loadNotifications() {
            try {
                const res = await fetch(`${API}/api/notifications?limit=10`);
                const data = await res.json();
                const el = document.getElementById('notifications');
                
                if (!data.notifications.length) {
                    el.innerHTML = '<div class="card"><div class="card-meta">No notifications</div></div>';
                    return;
                }
                
                el.innerHTML = data.notifications.map(n => `
                    <div class="card card-priority-${n.priority}" onclick="dismissNotif('${n.id}')">
                        <div class="card-title">${n.title}</div>
                        <div class="card-meta">${n.body || ''}</div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('notifications').innerHTML = 
                    '<div class="card-meta">Could not load notifications</div>';
            }
        }
        
        // Dismiss a notification (called on card click) and refresh the list.
        async function dismissNotif(id) {
            await fetch(`${API}/api/notifications/${id}/dismiss`, {method: 'POST'});
            loadNotifications();
        }

        // --- Tasks Polling ---
        // Fetches pending tasks and renders them as clickable cards.
        // Clicking a task card marks it complete via the API.
        async function loadTasks() {
            try {
                const res = await fetch(`${API}/api/tasks?limit=10`);
                const data = await res.json();
                const el = document.getElementById('tasks');
                
                if (!data.tasks.length) {
                    el.innerHTML = '<div class="card"><div class="card-meta">No pending tasks</div></div>';
                    return;
                }
                
                el.innerHTML = data.tasks.map(t => `
                    <div class="card card-priority-${t.priority}" onclick="completeTask('${t.id}')">
                        <div class="card-title">${t.title}</div>
                        <div class="card-meta">${t.domain || ''} ${t.due_date ? '• Due: ' + t.due_date : ''}</div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('tasks').innerHTML = 
                    '<div class="card-meta">Could not load tasks</div>';
            }
        }
        
        // Complete a task (called on card click) and refresh the list.
        async function completeTask(id) {
            await fetch(`${API}/api/tasks/${id}/complete`, {method: 'POST'});
            loadTasks();
        }

        // --- System Status Polling ---
        // Fetches /health and updates both the status bar at the bottom of the
        // page and the "System" card with event bus, vector store, and connector info.
        // Polled every 30 seconds.
        async function loadStatus() {
            try {
                const res = await fetch(`${API}/health`);
                const data = await res.json();
                
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                const count = document.getElementById('eventCount');
                
                dot.className = `status-dot ${data.status === 'ok' ? 'ok' : 'error'}`;
                text.textContent = data.event_bus ? 'Connected' : 'Event bus disconnected';
                count.textContent = `${data.events_stored} events`;
                
                document.getElementById('systemStatus').innerHTML = `
                    <div class="card">
                        <div class="card-title">Event Bus: ${data.event_bus ? '✓ Connected' : '✗ Disconnected'}</div>
                        <div class="card-meta">
                            ${data.events_stored} events stored • 
                            Vector store: ${data.vector_store.document_count} documents •
                            ${data.connectors.length} connectors
                        </div>
                    </div>
                `;
            } catch (e) {
                document.getElementById('statusDot').className = 'status-dot error';
                document.getElementById('statusText').textContent = 'Offline';
            }
        }
        
        // --- WebSocket for Real-Time Updates ---
        // Connects to the /ws endpoint for server-pushed events.  When a
        // "notification" message arrives, the notifications list is refreshed
        // immediately (rather than waiting for the next polling interval).
        // On connection close, automatically reconnects after a 5-second delay
        // to handle transient network issues or server restarts.
        function connectWS() {
            const ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'notification') {
                    loadNotifications();
                }
            };
            // Reconnect with a 5-second backoff on any close event.
            ws.onclose = () => setTimeout(connectWS, 5000);
        }

        // --- Initial Page Load ---
        // Kick off all data fetches immediately, then set up periodic polling.
        loadNotifications();
        loadTasks();
        loadStatus();
        setInterval(loadStatus, 30000);       // Poll system health every 30 seconds.
        setInterval(loadNotifications, 60000); // Poll notifications every 60 seconds.
        // WebSocket connection is wrapped in try/catch so that environments
        // without WebSocket support (e.g. some test runners) do not throw.
        try { connectWS(); } catch(e) {}
    </script>
</body>
</html>"""
