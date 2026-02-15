"""
Life OS — Database Viewer Admin Page
"""

DB_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life OS — Database</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
               background: #0a0a0a; color: #e0e0e0; min-height: 100vh; }
        .container { max-width: 100%; margin: 0 auto; padding: 40px 20px 20px; }
        h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; color: #fff; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 32px; }
        .subtitle a { color: #4a9eff; text-decoration: none; }
        .subtitle a:hover { text-decoration: underline; }

        /* Layout */
        .layout { display: flex; gap: 20px; }
        @media (max-width: 768px) { .layout { flex-direction: column; } }

        /* Sidebar */
        .sidebar { width: 220px; flex-shrink: 0; }
        .db-group { margin-bottom: 16px; }
        .db-name {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; color: #555; padding: 4px 0; margin-bottom: 4px;
        }
        .table-link {
            display: block; padding: 6px 12px; font-size: 13px; color: #888;
            border-radius: 6px; cursor: pointer; transition: all 0.15s;
            text-decoration: none;
        }
        .table-link:hover { background: #1a1a1a; color: #ccc; }
        .table-link.active { background: #1a3a5c; color: #7ab8ff; }
        .table-count {
            float: right; font-size: 11px; color: #555;
            background: #1a1a1a; padding: 1px 6px; border-radius: 4px;
        }
        .table-link.active .table-count { background: #0f2a44; color: #5a9adf; }

        /* Main content */
        .main { flex: 1; min-width: 0; overflow: hidden; }

        /* Table info bar */
        .table-info {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 12px; flex-wrap: wrap; gap: 8px;
        }
        .table-title { font-size: 16px; font-weight: 500; color: #fff; }
        .table-meta { font-size: 12px; color: #555; }

        /* Controls row */
        .controls {
            display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;
            align-items: center;
        }
        .search-input {
            flex: 1; min-width: 200px; padding: 8px 12px; font-size: 13px;
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
            color: #e0e0e0; outline: none; font-family: inherit;
        }
        .search-input:focus { border-color: #4a9eff; }
        .search-input::placeholder { color: #444; }
        .btn {
            padding: 8px 14px; font-size: 12px; font-weight: 500;
            border: 1px solid #333; border-radius: 8px; cursor: pointer;
            transition: all 0.15s; font-family: inherit; white-space: nowrap;
        }
        .btn-secondary { background: #222; border-color: #333; color: #999; }
        .btn-secondary:hover { background: #2a2a2a; color: #ccc; }
        .btn-primary { background: #1a3a5c; border-color: #2a5a8c; color: #7ab8ff; }
        .btn-primary:hover { background: #1f4470; }
        .btn-danger { background: #3d0a0a; border-color: #5a1a1a; color: #ff5555; }
        .btn-danger:hover { background: #4d0f0f; }

        /* Data table */
        .table-wrap {
            overflow: auto; background: #111; border: 1px solid #222;
            border-radius: 10px; max-height: calc(100vh - 220px);
        }
        table {
            width: max-content; min-width: 100%; border-collapse: collapse; font-size: 13px;
        }
        th {
            text-align: left; padding: 10px 12px; font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.3px; color: #666;
            background: #161616; border-bottom: 1px solid #222;
            position: sticky; top: 0; z-index: 1; cursor: pointer; user-select: none;
            white-space: nowrap;
        }
        th:hover { color: #999; }
        th .sort-arrow { font-size: 10px; margin-left: 4px; color: #444; }
        th.sorted .sort-arrow { color: #4a9eff; }
        td {
            padding: 8px 12px; border-bottom: 1px solid #1a1a1a;
            max-width: 400px; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; color: #bbb;
        }
        tr:hover td { background: #161616; }
        td.cell-null { color: #444; font-style: italic; }
        td.cell-json { color: #a0a0ff; cursor: pointer; }
        td.cell-json:hover { color: #c0c0ff; text-decoration: underline; }
        td.cell-long { cursor: pointer; }
        td.cell-long:hover { color: #fff; }

        /* Pagination */
        .pagination {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 12px; font-size: 12px; color: #555;
        }
        .page-btns { display: flex; gap: 4px; }
        .page-btn {
            padding: 4px 10px; font-size: 12px; background: #1a1a1a;
            border: 1px solid #222; border-radius: 4px; color: #888;
            cursor: pointer; font-family: inherit;
        }
        .page-btn:hover { background: #222; color: #ccc; }
        .page-btn.active { background: #1a3a5c; border-color: #2a5a8c; color: #7ab8ff; }
        .page-btn:disabled { opacity: 0.3; cursor: not-allowed; }

        /* Detail modal */
        .modal-overlay {
            position: fixed; inset: 0; background: rgba(0,0,0,0.7);
            display: none; align-items: center; justify-content: center;
            z-index: 100;
        }
        .modal-overlay.show { display: flex; }
        .modal {
            background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
            max-width: 700px; width: 90%; max-height: 80vh; display: flex;
            flex-direction: column;
        }
        .modal-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 16px 20px; border-bottom: 1px solid #222;
        }
        .modal-title { font-size: 14px; font-weight: 600; color: #fff; }
        .modal-close {
            background: none; border: none; color: #666; font-size: 18px;
            cursor: pointer; padding: 4px 8px;
        }
        .modal-close:hover { color: #fff; }
        .modal-body {
            padding: 20px; overflow-y: auto; flex: 1;
        }
        .modal-body pre {
            font-family: 'SF Mono', Menlo, monospace; font-size: 12px;
            line-height: 1.5; color: #ccc; white-space: pre-wrap;
            word-break: break-all;
        }
        .modal-body .field-row { margin-bottom: 12px; }
        .modal-body .field-name {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.3px; color: #666; margin-bottom: 2px;
        }
        .modal-body .field-value {
            font-family: 'SF Mono', Menlo, monospace; font-size: 13px;
            color: #ccc; padding: 6px 10px; background: #111;
            border-radius: 6px; word-break: break-all; white-space: pre-wrap;
        }
        .modal-body .field-value.json { color: #a0a0ff; }

        /* Empty state */
        .empty-state {
            text-align: center; padding: 60px 20px; color: #444;
        }
        .empty-state .icon { font-size: 32px; margin-bottom: 12px; }
        .empty-state .msg { font-size: 14px; }

        /* Toast */
        .toast {
            position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
            border-radius: 8px; font-size: 13px; font-weight: 500;
            z-index: 200; transform: translateY(100px); opacity: 0;
            transition: all 0.3s;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast-success { background: #0a3d1a; color: #4aff6b; border: 1px solid #1a5a2a; }
        .toast-error { background: #3d0a0a; color: #ff5555; border: 1px solid #5a1a1a; }
        .toast-info { background: #1a3a5c; color: #7ab8ff; border: 1px solid #2a5a8c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Database Viewer</h1>
        <p class="subtitle">
            <a href="/admin">&larr; Connectors</a> &middot;
            <a href="/">Dashboard</a>
        </p>

        <div class="layout">
            <nav class="sidebar" id="sidebar">Loading...</nav>
            <div class="main" id="main">
                <div class="empty-state">
                    <div class="icon">&#9776;</div>
                    <div class="msg">Select a table from the sidebar</div>
                </div>
            </div>
        </div>
    </div>

    <div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <span class="modal-title" id="modalTitle">Row Detail</span>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody"></div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
    const API = '';
    let schema = {};       // { db_name: { table_name: { columns: [...], count: N } } }
    let currentDb = null;
    let currentTable = null;
    let currentData = [];
    let currentColumns = [];
    let currentPage = 0;
    let currentSearch = '';
    let currentSort = null; // { col, dir }
    const PAGE_SIZE = 50;

    // ---------------------------------------------------------------
    // Load schema (sidebar)
    // ---------------------------------------------------------------

    async function loadSchema() {
        try {
            const res = await fetch(`${API}/api/admin/db`);
            schema = (await res.json()).databases;
            renderSidebar();
        } catch (e) {
            document.getElementById('sidebar').innerHTML =
                '<div style="color:#ff5555;font-size:13px">Failed to load schema</div>';
        }
    }

    function renderSidebar() {
        const el = document.getElementById('sidebar');
        let html = '';
        for (const [dbName, tables] of Object.entries(schema)) {
            html += `<div class="db-group"><div class="db-name">${dbName}.db</div>`;
            for (const [tblName, info] of Object.entries(tables)) {
                const active = dbName === currentDb && tblName === currentTable ? ' active' : '';
                html += `<a class="table-link${active}" onclick="selectTable('${dbName}','${tblName}')">
                    ${tblName}<span class="table-count">${info.count}</span>
                </a>`;
            }
            html += '</div>';
        }
        el.innerHTML = html;
    }

    // ---------------------------------------------------------------
    // Load table data
    // ---------------------------------------------------------------

    async function selectTable(db, table) {
        currentDb = db;
        currentTable = table;
        currentPage = 0;
        currentSearch = '';
        currentSort = null;
        renderSidebar();
        await loadTableData();
    }

    async function loadTableData() {
        const main = document.getElementById('main');
        main.innerHTML = '<div class="empty-state"><div class="msg">Loading...</div></div>';

        try {
            let url = `${API}/api/admin/db/${currentDb}/${currentTable}?limit=${PAGE_SIZE}&offset=${currentPage * PAGE_SIZE}`;
            if (currentSearch) url += `&search=${encodeURIComponent(currentSearch)}`;
            if (currentSort) url += `&sort=${currentSort.col}&dir=${currentSort.dir}`;

            const res = await fetch(url);
            const data = await res.json();
            currentData = data.rows;
            currentColumns = data.columns;
            renderTable(data);
        } catch (e) {
            main.innerHTML = `<div class="empty-state"><div class="msg" style="color:#ff5555">Error: ${e.message}</div></div>`;
        }
    }

    function renderTable(data) {
        const main = document.getElementById('main');
        const info = schema[currentDb]?.[currentTable];
        const totalRows = data.total;
        const totalPages = Math.ceil(totalRows / PAGE_SIZE);

        let html = `
            <div class="table-info">
                <span>
                    <span class="table-title">${currentDb}.${currentTable}</span>
                    <span class="table-meta">&nbsp;&middot; ${totalRows} rows &middot; ${data.columns.length} columns</span>
                </span>
                <button class="btn btn-secondary" onclick="exportCSV()">Export CSV</button>
            </div>
            <div class="controls">
                <input class="search-input" type="text" placeholder="Filter rows..."
                       value="${escapeAttr(currentSearch)}"
                       onkeydown="if(event.key==='Enter'){currentSearch=this.value;currentPage=0;loadTableData();}">
                <button class="btn btn-secondary" onclick="currentSearch=document.querySelector('.search-input').value;currentPage=0;loadTableData();">Filter</button>
                ${currentSearch ? `<button class="btn btn-secondary" onclick="currentSearch='';currentPage=0;loadTableData();">Clear</button>` : ''}
                <button class="btn btn-secondary" onclick="loadTableData();">Refresh</button>
            </div>
        `;

        if (data.rows.length === 0) {
            html += `<div class="table-wrap"><div class="empty-state"><div class="msg">${currentSearch ? 'No matching rows' : 'Table is empty'}</div></div></div>`;
        } else {
            html += '<div class="table-wrap"><table><thead><tr>';
            for (const col of data.columns) {
                const sorted = currentSort?.col === col;
                const arrow = sorted ? (currentSort.dir === 'asc' ? '&#9650;' : '&#9660;') : '&#9650;';
                html += `<th class="${sorted ? 'sorted' : ''}" onclick="sortBy('${col}')">
                    ${col}<span class="sort-arrow">${arrow}</span>
                </th>`;
            }
            html += '</tr></thead><tbody>';

            for (let i = 0; i < data.rows.length; i++) {
                const row = data.rows[i];
                html += '<tr>';
                for (const col of data.columns) {
                    const val = row[col];
                    html += renderCell(val, i, col);
                }
                html += '</tr>';
            }
            html += '</tbody></table></div>';
        }

        // Pagination
        if (totalPages > 1) {
            html += `<div class="pagination">
                <span>Page ${currentPage + 1} of ${totalPages}</span>
                <div class="page-btns">
                    <button class="page-btn" onclick="goPage(0)" ${currentPage === 0 ? 'disabled' : ''}>First</button>
                    <button class="page-btn" onclick="goPage(${currentPage - 1})" ${currentPage === 0 ? 'disabled' : ''}>Prev</button>
                    <button class="page-btn" onclick="goPage(${currentPage + 1})" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Next</button>
                    <button class="page-btn" onclick="goPage(${totalPages - 1})" ${currentPage >= totalPages - 1 ? 'disabled' : ''}>Last</button>
                </div>
            </div>`;
        }

        main.innerHTML = html;
    }

    function renderCell(val, rowIdx, col) {
        if (val === null || val === undefined) {
            return '<td class="cell-null">null</td>';
        }
        const s = String(val);
        // Detect JSON
        if (s.startsWith('{') || s.startsWith('[')) {
            try {
                JSON.parse(s);
                const preview = s.length > 60 ? s.slice(0, 60) + '...' : s;
                return `<td class="cell-json" onclick="showJson(${rowIdx},'${escapeAttr(col)}')" title="Click to expand">${escapeHtml(preview)}</td>`;
            } catch {}
        }
        // Long values
        if (s.length > 80) {
            return `<td class="cell-long" onclick="showRow(${rowIdx})" title="Click to see full row">${escapeHtml(s.slice(0, 80))}...</td>`;
        }
        return `<td>${escapeHtml(s)}</td>`;
    }

    // ---------------------------------------------------------------
    // Sort & paginate
    // ---------------------------------------------------------------

    function sortBy(col) {
        if (currentSort?.col === col) {
            currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
            currentSort = { col, dir: 'asc' };
        }
        currentPage = 0;
        loadTableData();
    }

    function goPage(page) {
        currentPage = page;
        loadTableData();
    }

    // ---------------------------------------------------------------
    // Modal
    // ---------------------------------------------------------------

    function showRow(rowIdx) {
        const row = currentData[rowIdx];
        if (!row) return;
        document.getElementById('modalTitle').textContent = `Row Detail`;
        let html = '';
        for (const col of currentColumns) {
            const val = row[col];
            const isJson = typeof val === 'string' && (val.startsWith('{') || val.startsWith('['));
            let display;
            if (val === null || val === undefined) {
                display = '<span style="color:#444;font-style:italic">null</span>';
            } else if (isJson) {
                try {
                    display = `<pre class="json">${escapeHtml(JSON.stringify(JSON.parse(val), null, 2))}</pre>`;
                } catch {
                    display = escapeHtml(String(val));
                }
            } else {
                display = escapeHtml(String(val));
            }
            html += `<div class="field-row">
                <div class="field-name">${col}</div>
                <div class="field-value${isJson ? ' json' : ''}">${display}</div>
            </div>`;
        }
        document.getElementById('modalBody').innerHTML = html;
        document.getElementById('modalOverlay').classList.add('show');
    }

    function showJson(rowIdx, col) {
        const row = currentData[rowIdx];
        if (!row) return;
        const val = row[col];
        document.getElementById('modalTitle').textContent = `${currentTable}.${col}`;
        let html;
        try {
            const parsed = JSON.parse(val);
            html = `<pre>${escapeHtml(JSON.stringify(parsed, null, 2))}</pre>`;
        } catch {
            html = `<pre>${escapeHtml(String(val))}</pre>`;
        }
        document.getElementById('modalBody').innerHTML = html;
        document.getElementById('modalOverlay').classList.add('show');
    }

    function closeModal(e) {
        if (e && e.target !== document.getElementById('modalOverlay')) return;
        document.getElementById('modalOverlay').classList.remove('show');
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // ---------------------------------------------------------------
    // Export CSV
    // ---------------------------------------------------------------

    function exportCSV() {
        if (!currentData.length) return;
        const header = currentColumns.join(',');
        const rows = currentData.map(row =>
            currentColumns.map(col => {
                const val = row[col];
                if (val === null || val === undefined) return '';
                const s = String(val).replace(/"/g, '""');
                return s.includes(',') || s.includes('"') || s.includes('\\n') ? `"${s}"` : s;
            }).join(',')
        );
        const csv = header + '\\n' + rows.join('\\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentDb}_${currentTable}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('CSV exported', 'success');
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    function escapeHtml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function escapeAttr(s) {
        return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    let toastTimer = null;
    function showToast(msg, type) {
        const el = document.getElementById('toast');
        el.textContent = msg;
        el.className = `toast toast-${type} show`;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.classList.remove('show'); }, 3000);
    }

    // ---------------------------------------------------------------
    // Init
    // ---------------------------------------------------------------
    loadSchema();
    </script>
</body>
</html>"""
