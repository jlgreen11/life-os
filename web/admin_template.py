"""
Life OS — Admin Page Template for Connector Management
"""

ADMIN_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life OS — Admin</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
               background: #0a0a0a; color: #e0e0e0; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 40px 20px 80px; }
        h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; color: #fff; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 32px; }
        .subtitle a { color: #4a9eff; text-decoration: none; }
        .subtitle a:hover { text-decoration: underline; }

        /* Grid */
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

        /* Connector card */
        .connector-card {
            background: #1a1a1a; border: 1px solid #222; border-radius: 10px;
            padding: 16px; cursor: pointer; transition: border-color 0.2s;
        }
        .connector-card:hover { border-color: #444; }
        .connector-card.expanded { grid-column: 1 / -1; cursor: default; }

        .card-header { display: flex; justify-content: space-between; align-items: center; }
        .card-name { font-weight: 500; font-size: 15px; }
        .card-desc { font-size: 13px; color: #666; margin-top: 4px; }
        .card-meta { font-size: 12px; color: #555; margin-top: 8px; }

        /* Status badge */
        .badge {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            padding: 3px 8px; border-radius: 6px; letter-spacing: 0.3px;
        }
        .badge-active { background: #0a3d1a; color: #4aff6b; }
        .badge-inactive { background: #2a2a2a; color: #888; }
        .badge-error { background: #3d0a0a; color: #ff5555; }
        .badge-unconfigured { background: #2a2000; color: #aa8800; }

        /* Category tag */
        .category-tag {
            font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
            color: #555; margin-left: 8px;
        }

        /* Config form */
        .config-form { margin-top: 16px; border-top: 1px solid #222; padding-top: 16px; }
        .form-group { margin-bottom: 12px; }
        .form-label {
            display: block; font-size: 12px; font-weight: 500; color: #999;
            margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px;
        }
        .form-help { font-size: 11px; color: #555; margin-bottom: 4px; }
        .form-input {
            width: 100%; padding: 10px 12px; font-size: 14px;
            background: #111; border: 1px solid #333; border-radius: 8px;
            color: #e0e0e0; outline: none; transition: border-color 0.2s;
            font-family: inherit;
        }
        .form-input:focus { border-color: #4a9eff; }
        .form-input::placeholder { color: #444; }

        /* Password field with toggle */
        .password-wrap { position: relative; }
        .password-wrap .form-input { padding-right: 70px; }
        .pw-toggle {
            position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
            background: #222; border: 1px solid #333; border-radius: 4px;
            color: #999; font-size: 11px; padding: 3px 8px; cursor: pointer;
        }
        .pw-toggle:hover { background: #333; }

        /* Checkbox */
        .form-checkbox { display: flex; align-items: center; gap: 8px; }
        .form-checkbox input[type="checkbox"] {
            width: 16px; height: 16px; accent-color: #4a9eff;
        }
        .form-checkbox label { font-size: 14px; color: #ccc; cursor: pointer; }

        /* Buttons */
        .btn-row { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
        .btn {
            padding: 8px 16px; font-size: 13px; font-weight: 500;
            border: 1px solid #333; border-radius: 8px; cursor: pointer;
            transition: all 0.15s; font-family: inherit;
        }
        .btn-primary { background: #1a3a5c; border-color: #2a5a8c; color: #7ab8ff; }
        .btn-primary:hover { background: #1f4470; }
        .btn-success { background: #0a3d1a; border-color: #1a5a2a; color: #4aff6b; }
        .btn-success:hover { background: #0f4d22; }
        .btn-danger { background: #3d0a0a; border-color: #5a1a1a; color: #ff5555; }
        .btn-danger:hover { background: #4d0f0f; }
        .btn-secondary { background: #222; border-color: #333; color: #999; }
        .btn-secondary:hover { background: #2a2a2a; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Feedback toast */
        .toast {
            position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
            border-radius: 8px; font-size: 13px; font-weight: 500;
            z-index: 1000; transform: translateY(100px); opacity: 0;
            transition: all 0.3s;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast-success { background: #0a3d1a; color: #4aff6b; border: 1px solid #1a5a2a; }
        .toast-error { background: #3d0a0a; color: #ff5555; border: 1px solid #5a1a1a; }
        .toast-info { background: #1a3a5c; color: #7ab8ff; border: 1px solid #2a5a8c; }

        /* Section headers */
        .section-label {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; color: #555; margin: 24px 0 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Manage Connectors</h1>
        <p class="subtitle"><a href="/">&larr; Back to Dashboard</a> &middot; <a href="/admin/db">Database Viewer</a></p>

        <div class="section-label">API Connectors</div>
        <div class="grid" id="apiGrid"></div>

        <div class="section-label">Browser Connectors</div>
        <div class="grid" id="browserGrid"></div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
    const API = '';
    let registry = {};
    let connectors = [];

    // ---------------------------------------------------------------
    // Data loading
    // ---------------------------------------------------------------

    async function loadAll() {
        try {
            const [regRes, conRes] = await Promise.all([
                fetch(`${API}/api/admin/connectors/registry`),
                fetch(`${API}/api/admin/connectors`),
            ]);
            const regData = await regRes.json();
            const conData = await conRes.json();
            registry = regData.registry;
            connectors = conData.connectors;
            render();
        } catch (e) {
            showToast('Failed to load connectors: ' + e.message, 'error');
        }
    }

    // ---------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------

    function render() {
        const apiGrid = document.getElementById('apiGrid');
        const browserGrid = document.getElementById('browserGrid');
        apiGrid.innerHTML = '';
        browserGrid.innerHTML = '';

        for (const c of connectors) {
            const card = createCard(c);
            if (c.category === 'browser') {
                browserGrid.appendChild(card);
            } else {
                apiGrid.appendChild(card);
            }
        }
    }

    function badgeClass(status) {
        if (status === 'active') return 'badge-active';
        if (status === 'error') return 'badge-error';
        if (status === 'unconfigured') return 'badge-unconfigured';
        return 'badge-inactive';
    }

    function createCard(c) {
        const st = c.status;
        const div = document.createElement('div');
        div.className = 'connector-card';
        div.dataset.id = c.connector_id;

        let metaHtml = '';
        if (st.last_sync) metaHtml += `Last sync: ${new Date(st.last_sync).toLocaleString()}`;
        if (st.error_count > 0) metaHtml += ` &middot; ${st.error_count} errors`;
        if (st.last_error) metaHtml += `<br>Last error: ${st.last_error}`;

        div.innerHTML = `
            <div class="card-header">
                <span>
                    <span class="card-name">${c.display_name}</span>
                    <span class="category-tag">${c.category}</span>
                </span>
                <span class="badge ${badgeClass(st.status)}">${st.status}</span>
            </div>
            <div class="card-desc">${c.description}</div>
            ${metaHtml ? `<div class="card-meta">${metaHtml}</div>` : ''}
        `;

        div.addEventListener('click', () => expandCard(div, c));
        return div;
    }

    function expandCard(div, c) {
        // Collapse any other expanded cards
        document.querySelectorAll('.connector-card.expanded').forEach(el => {
            if (el !== div) collapseCard(el);
        });

        if (div.classList.contains('expanded')) return;
        div.classList.add('expanded');
        div.style.cursor = 'default';

        const reg = registry[c.connector_id];
        if (!reg) return;

        const form = document.createElement('div');
        form.className = 'config-form';

        // Build form fields
        for (const field of reg.config_fields) {
            form.appendChild(createField(field, c.config));
        }

        // Action buttons
        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const testBtn = document.createElement('button');
        testBtn.className = 'btn btn-secondary';
        testBtn.textContent = 'Test Connection';
        testBtn.onclick = (e) => { e.stopPropagation(); testConnector(div, c.connector_id); };

        const saveBtn = document.createElement('button');
        saveBtn.className = 'btn btn-primary';
        saveBtn.textContent = 'Save';
        saveBtn.onclick = (e) => { e.stopPropagation(); saveConfig(div, c.connector_id); };

        const running = c.status.running;
        const toggleBtn = document.createElement('button');
        if (running) {
            toggleBtn.className = 'btn btn-danger';
            toggleBtn.textContent = 'Disable';
            toggleBtn.onclick = (e) => { e.stopPropagation(); disableConnector(c.connector_id); };
        } else {
            toggleBtn.className = 'btn btn-success';
            toggleBtn.textContent = 'Enable';
            toggleBtn.onclick = (e) => { e.stopPropagation(); enableConnector(c.connector_id); };
        }

        const collapseBtn = document.createElement('button');
        collapseBtn.className = 'btn btn-secondary';
        collapseBtn.textContent = 'Close';
        collapseBtn.onclick = (e) => { e.stopPropagation(); collapseCard(div); };

        btnRow.append(testBtn, saveBtn, toggleBtn, collapseBtn);
        form.appendChild(btnRow);
        div.appendChild(form);
    }

    function collapseCard(div) {
        div.classList.remove('expanded');
        div.style.cursor = 'pointer';
        const form = div.querySelector('.config-form');
        if (form) form.remove();
    }

    function createField(field, currentConfig) {
        const group = document.createElement('div');
        group.className = 'form-group';

        const value = currentConfig[field.name] ?? field.default ?? '';

        if (field.field_type === 'boolean') {
            group.innerHTML = `
                <div class="form-checkbox">
                    <input type="checkbox" id="f_${field.name}" data-field="${field.name}"
                           ${value ? 'checked' : ''}>
                    <label for="f_${field.name}">${field.name}${field.required ? ' *' : ''}</label>
                </div>
            `;
            if (field.help_text) {
                group.innerHTML += `<div class="form-help">${field.help_text}</div>`;
            }
            return group;
        }

        const label = `<label class="form-label">${field.name}${field.required ? ' *' : ''}</label>`;
        const help = field.help_text ? `<div class="form-help">${field.help_text}</div>` : '';

        if (field.field_type === 'password') {
            const displayVal = typeof value === 'string' ? value : '';
            group.innerHTML = `
                ${label}${help}
                <div class="password-wrap">
                    <input class="form-input" type="password" data-field="${field.name}"
                           value="${escapeAttr(displayVal)}"
                           placeholder="${escapeAttr(field.placeholder)}">
                    <button class="pw-toggle" type="button" onclick="togglePw(this, event)">Show</button>
                </div>
            `;
        } else if (field.field_type === 'list') {
            const listVal = Array.isArray(value) ? value.join(', ') : (value || '');
            group.innerHTML = `
                ${label}${help}
                <input class="form-input" type="text" data-field="${field.name}"
                       data-type="list"
                       value="${escapeAttr(listVal)}"
                       placeholder="${escapeAttr(field.placeholder)}">
            `;
        } else if (field.field_type === 'integer') {
            group.innerHTML = `
                ${label}${help}
                <input class="form-input" type="number" data-field="${field.name}"
                       data-type="integer"
                       value="${value !== null && value !== undefined ? value : ''}"
                       placeholder="${escapeAttr(field.placeholder)}">
            `;
        } else {
            group.innerHTML = `
                ${label}${help}
                <input class="form-input" type="text" data-field="${field.name}"
                       value="${escapeAttr(String(value || ''))}"
                       placeholder="${escapeAttr(field.placeholder)}">
            `;
        }

        return group;
    }

    // ---------------------------------------------------------------
    // Form helpers
    // ---------------------------------------------------------------

    function escapeAttr(s) {
        return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function togglePw(btn, e) {
        e.stopPropagation();
        const input = btn.parentElement.querySelector('input');
        if (input.type === 'password') {
            input.type = 'text';
            btn.textContent = 'Hide';
        } else {
            input.type = 'password';
            btn.textContent = 'Show';
        }
    }

    function gatherConfig(card) {
        const config = {};
        card.querySelectorAll('[data-field]').forEach(el => {
            const name = el.dataset.field;
            if (el.type === 'checkbox') {
                config[name] = el.checked;
            } else if (el.dataset.type === 'list') {
                config[name] = el.value.split(',').map(s => s.trim()).filter(Boolean);
            } else if (el.dataset.type === 'integer') {
                config[name] = el.value ? parseInt(el.value, 10) : null;
            } else {
                config[name] = el.value;
            }
        });
        return config;
    }

    // ---------------------------------------------------------------
    // API actions
    // ---------------------------------------------------------------

    async function saveConfig(card, connectorId) {
        const config = gatherConfig(card);
        try {
            const res = await fetch(`${API}/api/admin/connectors/${connectorId}/config`, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({config}),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Save failed');
            showToast('Configuration saved', 'success');
            await loadAll();
        } catch (e) {
            showToast('Save failed: ' + e.message, 'error');
        }
    }

    async function testConnector(card, connectorId) {
        const config = gatherConfig(card);
        showToast('Testing connection...', 'info');
        try {
            const res = await fetch(`${API}/api/admin/connectors/${connectorId}/test`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({config}),
            });
            const data = await res.json();
            if (data.success) {
                showToast('Connection successful', 'success');
            } else {
                showToast('Test failed: ' + (data.detail || 'Unknown error'), 'error');
            }
        } catch (e) {
            showToast('Test failed: ' + e.message, 'error');
        }
    }

    async function enableConnector(connectorId) {
        try {
            const res = await fetch(`${API}/api/admin/connectors/${connectorId}/enable`, {
                method: 'POST',
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Enable failed');
            showToast(`${connectorId} enabled`, 'success');
            await loadAll();
        } catch (e) {
            showToast('Enable failed: ' + e.message, 'error');
        }
    }

    async function disableConnector(connectorId) {
        try {
            const res = await fetch(`${API}/api/admin/connectors/${connectorId}/disable`, {
                method: 'POST',
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Disable failed');
            showToast(`${connectorId} disabled`, 'success');
            await loadAll();
        } catch (e) {
            showToast('Disable failed: ' + e.message, 'error');
        }
    }

    // ---------------------------------------------------------------
    // Toast
    // ---------------------------------------------------------------

    let toastTimer = null;
    function showToast(msg, type) {
        const el = document.getElementById('toast');
        el.textContent = msg;
        el.className = `toast toast-${type} show`;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.classList.remove('show'); }, 4000);
    }

    // ---------------------------------------------------------------
    // Init
    // ---------------------------------------------------------------
    loadAll();
    </script>
</body>
</html>"""
