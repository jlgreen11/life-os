"""
Life OS — First-Run Setup / Onboarding Template

Apple-inspired: one idea per screen, guided flow.
Detects if onboarding is complete and redirects to dashboard.
"""

SETUP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Life OS — Setup</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
            background: #0a0a0a; color: #e0e0e0; min-height: 100vh;
            display: flex; align-items: center; justify-content: center;
        }

        .setup-container {
            max-width: 560px; width: 100%; padding: 40px 24px;
            min-height: 100vh; display: flex; flex-direction: column;
            justify-content: center;
        }

        /* Progress bar */
        .progress-bar {
            width: 100%; height: 3px; background: #222; border-radius: 2px;
            margin-bottom: 48px; overflow: hidden;
        }
        .progress-fill {
            height: 100%; background: #4a9eff; border-radius: 2px;
            transition: width 0.4s ease;
        }

        /* Step content */
        .step { display: none; animation: fadeIn 0.3s ease; }
        .step.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .step-title {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.8px; color: #555; margin-bottom: 16px;
        }
        .step-prompt {
            font-size: 22px; font-weight: 400; line-height: 1.5;
            color: #fff; margin-bottom: 36px;
        }

        /* Choice buttons */
        .choices { display: flex; flex-direction: column; gap: 10px; }
        .choice-btn {
            display: block; width: 100%; padding: 16px 20px;
            background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
            color: #ccc; font-size: 16px; font-family: inherit;
            text-align: left; cursor: pointer; transition: all 0.15s;
        }
        .choice-btn:hover { border-color: #4a9eff; color: #fff; background: #1a2a3a; }
        .choice-btn.selected {
            border-color: #4a9eff; background: #1a3a5c; color: #7ab8ff;
        }

        /* Text input */
        .text-area {
            width: 100%; min-height: 120px; padding: 16px; font-size: 16px;
            background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
            color: #e0e0e0; outline: none; font-family: inherit;
            line-height: 1.5; resize: vertical;
        }
        .text-area:focus { border-color: #4a9eff; }
        .text-area::placeholder { color: #444; }
        .input-hint {
            font-size: 13px; color: #555; margin-top: 8px;
        }

        /* Navigation */
        .nav-row {
            display: flex; justify-content: space-between; align-items: center;
            margin-top: 36px;
        }
        .nav-btn {
            padding: 12px 28px; font-size: 15px; font-weight: 500;
            border: none; border-radius: 10px; cursor: pointer;
            font-family: inherit; transition: all 0.15s;
        }
        .nav-btn-back {
            background: transparent; color: #666; border: 1px solid #333;
        }
        .nav-btn-back:hover { color: #999; border-color: #444; }
        .nav-btn-next {
            background: #4a9eff; color: #fff;
        }
        .nav-btn-next:hover { background: #5aadff; }
        .nav-btn-next:disabled { opacity: 0.3; cursor: not-allowed; }
        .nav-btn-finish {
            background: #1a5a2a; color: #4aff6b; border: 1px solid #2a7a3a;
        }
        .nav-btn-finish:hover { background: #1f6a32; }

        /* Summary screen */
        .summary-section { margin-bottom: 24px; }
        .summary-label {
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; color: #555; margin-bottom: 6px;
        }
        .summary-value {
            font-size: 15px; color: #ccc; padding: 10px 14px;
            background: #1a1a1a; border-radius: 8px; line-height: 1.5;
        }
        .summary-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
        }
        @media (max-width: 500px) { .summary-grid { grid-template-columns: 1fr; } }

        /* Loading */
        .loading-spinner {
            display: inline-block; width: 20px; height: 20px;
            border: 2px solid #333; border-top-color: #4a9eff;
            border-radius: 50%; animation: spin 0.8s linear infinite;
            margin-right: 8px; vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Skip link */
        .skip-link {
            position: fixed; bottom: 20px; right: 20px;
            font-size: 12px; color: #444; text-decoration: none;
        }
        .skip-link:hover { color: #666; }
    </style>
</head>
<body>
    <div class="setup-container">
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
        <div id="stepsContainer"></div>
    </div>
    <a href="/" class="skip-link" id="skipLink">Skip setup &rarr;</a>

    <script>
    const API = '';
    let flow = [];
    let answers = {};
    let currentStep = 0;

    // ---------------------------------------------------------------
    // Load flow
    // ---------------------------------------------------------------

    async function init() {
        // Check if already completed
        const statusRes = await fetch(`${API}/api/setup/status`);
        const status = await statusRes.json();
        if (status.completed) {
            window.location.href = '/';
            return;
        }

        const flowRes = await fetch(`${API}/api/setup/flow`);
        const flowData = await flowRes.json();
        flow = flowData.phases;
        answers = status.answers || {};
        renderStep();
    }

    // ---------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------

    function renderStep() {
        const container = document.getElementById('stepsContainer');
        const phase = flow[currentStep];
        const total = flow.length;
        const pct = ((currentStep + 1) / total) * 100;
        document.getElementById('progress').style.width = pct + '%';

        let html = `<div class="step active">`;
        html += `<div class="step-title">${phase.title}</div>`;
        html += `<div class="step-prompt">${phase.prompt}</div>`;

        if (phase.type === 'choice') {
            html += `<div class="choices">`;
            for (const opt of phase.options) {
                const selected = answers[phase.id] === opt.value ? ' selected' : '';
                const valAttr = typeof opt.value === 'boolean' ? opt.value : `'${opt.value}'`;
                html += `<button class="choice-btn${selected}" onclick="selectChoice('${phase.id}', ${valAttr}, this)">
                    ${opt.label}
                </button>`;
            }
            html += `</div>`;
        } else if (phase.type === 'free_text') {
            const existing = answers[phase.id] || '';
            html += `<textarea class="text-area" id="textInput" placeholder="${phase.hint || 'Type your answer...'}"
                       oninput="answers['${phase.id}'] = this.value">${existing}</textarea>`;
            if (phase.hint) {
                html += `<div class="input-hint">${phase.hint}</div>`;
            }
        }

        // Navigation
        html += `<div class="nav-row">`;
        if (currentStep > 0) {
            html += `<button class="nav-btn nav-btn-back" onclick="prevStep()">Back</button>`;
        } else {
            html += `<div></div>`;
        }

        if (currentStep === total - 1) {
            // Last step (close) — show finish button
            html += `<button class="nav-btn nav-btn-finish" onclick="finishSetup()">Get Started</button>`;
        } else if (phase.type === 'info') {
            html += `<button class="nav-btn nav-btn-next" onclick="nextStep()">Continue</button>`;
        } else {
            const canNext = phase.type === 'free_text' || answers[phase.id] !== undefined;
            html += `<button class="nav-btn nav-btn-next" id="nextBtn" ${canNext ? '' : 'disabled'} onclick="nextStep()">Continue</button>`;
        }
        html += `</div>`;
        html += `</div>`;

        // If this is the close step, insert a summary before the prompt
        if (phase.id === 'close') {
            html = buildSummaryStep(phase);
        }

        container.innerHTML = html;

        // Auto-focus text area
        const ta = document.getElementById('textInput');
        if (ta) setTimeout(() => ta.focus(), 100);
    }

    function buildSummaryStep(phase) {
        let html = `<div class="step active">`;
        html += `<div class="step-title">Review Your Preferences</div>`;
        html += `<div class="step-prompt" style="font-size:18px;margin-bottom:24px">Here's how I'll work for you. You can change any of this later.</div>`;

        html += `<div class="summary-grid">`;

        const labels = {
            morning_style: 'Briefing Style',
            tone: 'Tone',
            proactivity: 'Proactivity',
            autonomy: 'Autonomy',
            drafting: 'Draft Replies',
            work_life_boundary: 'Work/Life',
            vault: 'Vault',
            notifications: 'Notifications',
        };
        const valueLabels = {
            minimal: 'Just the essentials',
            balanced: 'Balanced',
            detailed: 'Full picture',
            warm: 'Warm & supportive',
            casual: 'Casual',
            professional: 'Professional',
            high: 'Very proactive',
            moderate: 'Sometimes',
            low: 'Only when asked',
            supervised: 'Ask about everything',
            batched: 'Batched digest',
            frequent: 'Real-time',
            strict_separation: 'Hard wall',
            soft_separation: 'Soft wall',
            unified: 'Blended',
        };

        for (const [id, label] of Object.entries(labels)) {
            const val = answers[id];
            if (val === undefined) continue;
            let display;
            if (typeof val === 'boolean') {
                display = val ? 'Yes' : 'No';
            } else {
                display = valueLabels[val] || val;
            }
            html += `<div class="summary-section">
                <div class="summary-label">${label}</div>
                <div class="summary-value">${display}</div>
            </div>`;
        }
        html += `</div>`;

        // Free text summaries
        if (answers.domains) {
            html += `<div class="summary-section" style="margin-top:12px">
                <div class="summary-label">Life Domains</div>
                <div class="summary-value">${escapeHtml(answers.domains)}</div>
            </div>`;
        }
        if (answers.priority_people) {
            html += `<div class="summary-section">
                <div class="summary-label">Priority People</div>
                <div class="summary-value">${escapeHtml(answers.priority_people)}</div>
            </div>`;
        }
        if (answers.quiet_hours) {
            html += `<div class="summary-section">
                <div class="summary-label">Quiet Hours</div>
                <div class="summary-value">${escapeHtml(answers.quiet_hours)}</div>
            </div>`;
        }

        html += `<div style="margin-top:24px;font-size:15px;color:#888;line-height:1.6">${phase.prompt}</div>`;

        html += `<div class="nav-row">
            <button class="nav-btn nav-btn-back" onclick="prevStep()">Back</button>
            <button class="nav-btn nav-btn-finish" id="finishBtn" onclick="finishSetup()">Get Started</button>
        </div>`;
        html += `</div>`;
        return html;
    }

    // ---------------------------------------------------------------
    // Interaction
    // ---------------------------------------------------------------

    function selectChoice(stepId, value, btn) {
        answers[stepId] = value;
        // Visual feedback
        btn.parentElement.querySelectorAll('.choice-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        // Enable next
        const nextBtn = document.getElementById('nextBtn');
        if (nextBtn) nextBtn.disabled = false;
        // Auto-advance after short delay
        setTimeout(nextStep, 300);
    }

    function nextStep() {
        const phase = flow[currentStep];
        // Submit answer to backend
        if (phase.type !== 'info' && answers[phase.id] !== undefined) {
            fetch(`${API}/api/setup/submit`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ step_id: phase.id, value: answers[phase.id] }),
            });
        }
        if (currentStep < flow.length - 1) {
            currentStep++;
            renderStep();
        }
    }

    function prevStep() {
        if (currentStep > 0) {
            currentStep--;
            renderStep();
        }
    }

    async function finishSetup() {
        const btn = document.getElementById('finishBtn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="loading-spinner"></span>Setting up...';
        }

        try {
            const res = await fetch(`${API}/api/setup/finalize`, { method: 'POST' });
            const data = await res.json();
            if (data.status === 'ok') {
                window.location.href = '/';
            } else {
                alert('Setup failed: ' + (data.detail || 'Unknown error'));
                if (btn) { btn.disabled = false; btn.textContent = 'Get Started'; }
            }
        } catch (e) {
            alert('Setup failed: ' + e.message);
            if (btn) { btn.disabled = false; btn.textContent = 'Get Started'; }
        }
    }

    function escapeHtml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
    }

    // ---------------------------------------------------------------
    // Init
    // ---------------------------------------------------------------
    init();
    </script>
</body>
</html>"""
