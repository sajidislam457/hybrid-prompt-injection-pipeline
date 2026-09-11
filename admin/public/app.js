/* SecureAI Admin frontend — research tools (localhost) */

const state = {
    unlocked: false,
    authenticated: false,
    unlocking: false,
    lastProbe: null,
    selectedCaseId: null,
    attackTypes: [],
    cornerHot: false,
    digitBuf: '',
    readyUntil: 0,
    pollTimers: {},
    trainElapsedTimer: null,
    trainStartedAt: 0,
    accElapsedTimer: null,
    accStartedAt: 0,
    accJobId: null,
    accCancelRequested: false,
    fwTimer: null,
    fwRunning: false,
    fwScenarioIdx: 0,
    reconnectTries: 0,
    quietInboxUntil: 0,
};

const $ = (id) => document.getElementById(id);

function formatApiError(data, fallback = 'Request failed') {
    if (data == null || data === '') return fallback;
    if (data instanceof Error) return data.message || fallback;
    if (typeof data === 'string') return data;
    if (typeof data === 'number' || typeof data === 'boolean') return String(data);
    const detail = data.detail !== undefined ? data.detail : data.error !== undefined ? data.error : data;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail.map((item) => {
            if (typeof item === 'string') return item;
            if (item && typeof item === 'object') {
                const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
                const msg = item.msg || item.message || JSON.stringify(item);
                return loc ? `${loc}: ${msg}` : msg;
            }
            return String(item);
        }).join('; ') || fallback;
    }
    if (detail && typeof detail === 'object') {
        if (typeof detail.message === 'string') return detail.message;
        if (typeof detail.msg === 'string') return detail.msg;
        try {
            const s = JSON.stringify(detail);
            return s && s !== '{}' ? s : fallback;
        } catch (_) {
            return fallback;
        }
    }
    return fallback;
}

async function api(path, opts = {}) {
    const res = await fetch(path, {
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
        ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const err = new Error(formatApiError(data, `HTTP ${res.status}`));
        err.status = res.status;
        err.data = data;
        throw err;
    }
    return data;
}

const FEATURE_TABS = new Set(['lab', 'accuracy', 'datasets']);
const TAB_STORAGE_KEY = 'secureai_admin_tab';

function saveActiveTab(name) {
    try {
        if (FEATURE_TABS.has(name)) sessionStorage.setItem(TAB_STORAGE_KEY, name);
        else sessionStorage.removeItem(TAB_STORAGE_KEY);
    } catch (_) { /* ignore */ }
}

function loadSavedFeatureTab() {
    try {
        const name = sessionStorage.getItem(TAB_STORAGE_KEY);
        return FEATURE_TABS.has(name) ? name : null;
    } catch (_) {
        return null;
    }
}

function showApp({ restoreTab = false } = {}) {
    state.authenticated = true;
    state.unlocked = true;
    const tab = restoreTab ? (loadSavedFeatureTab() || 'welcome') : 'welcome';
    switchTab(tab);
    refreshSystem();
    loadInbox();
    loadAccuracyReports();
    resumeActiveAccuracyJob();
}

function switchTab(tab) {
    const name = String(tab || 'welcome');
    document.querySelectorAll('#sidebar [data-tab]').forEach((x) => {
        // Welcome is not in the sidebar — clear all active nav when showing it.
        x.classList.toggle('active', name !== 'welcome' && x.dataset.tab === name);
    });
    document.querySelectorAll('.tab-content').forEach((x) => {
        x.classList.toggle('active', x.id === `${name}-tab`);
    });
    saveActiveTab(name);
    if (name === 'welcome') startFirewallFlow();
    else stopFirewallFlow();
    if (name === 'datasets') loadDatasetsPanel();
}

/* Live defense scenarios (welcome flowchart) */
const FW_NODE_ORDER = ['user', 'l1', 'l2', 'l2b', 'ret', 'l3', 'l4', 'l5', 'llm'];

const FW_SCENARIOS = [
    {
        tag: 'Scenario 1 · Safe traffic',
        title: 'Normal question reaches the LLM',
        desc: 'Benign prompt passes every layer. Firewall stays quiet and allows the request.',
        path: ['user', 'l1', 'l2', 'l2b', 'ret', 'l3', 'l4', 'l5', 'llm'],
        blockAt: null,
        outcome: 'ALLOW — safe prompt delivered to the LLM',
    },
    {
        tag: 'Scenario 2 · Pattern attack',
        title: 'Jailbreak keywords stop at Prefilter',
        desc: 'Obvious injection phrases (“ignore previous instructions…”) are caught by L1 patterns.',
        path: ['user', 'l1'],
        blockAt: 'l1',
        outcome: 'BLOCK at L1 — classic prompt-injection pattern',
    },
    {
        tag: 'Scenario 3 · ML detection',
        title: 'Disguised attack scored by classifiers',
        desc: 'Wording looks polite, but L2 classical models raise risk and the ensemble blocks.',
        path: ['user', 'l1', 'l2', 'l2b', 'ret', 'l3'],
        blockAt: 'l3',
        outcome: 'BLOCK at L3 — ensemble risk over threshold',
    },
    {
        tag: 'Scenario 4 · Semantic attack',
        title: 'Transformer catches meaning-level injection',
        desc: 'No exact pattern match. L2b DeBERTa spots the malicious intent in context.',
        path: ['user', 'l1', 'l2', 'l2b'],
        blockAt: 'l2b',
        outcome: 'BLOCK at L2b — semantic injection detected',
    },
    {
        tag: 'Scenario 5 · Known attack',
        title: 'Attack bank retrieval recognizes a repeat',
        desc: 'Similar to a trained team example in the bank — retrieval flags it early.',
        path: ['user', 'l1', 'l2', 'l2b', 'ret'],
        blockAt: 'ret',
        outcome: 'BLOCK at Retrieval — known attack fingerprint',
    },
    {
        tag: 'Scenario 6 · Ambiguous case',
        title: 'Judge decides a borderline prompt',
        desc: 'Scores conflict. L4 judge reviews the edge case and blocks the risky intent.',
        path: ['user', 'l1', 'l2', 'l2b', 'ret', 'l3', 'l4'],
        blockAt: 'l4',
        outcome: 'BLOCK at L4 — judge marks malicious',
    },
    {
        tag: 'Scenario 7 · Soft allow',
        title: 'Risky wording rewritten, then allowed',
        desc: 'Not a hard block. L5 intent gate sanitizes / clarifies, then the LLM may answer safely.',
        path: ['user', 'l1', 'l2', 'l2b', 'ret', 'l3', 'l4', 'l5', 'llm'],
        blockAt: null,
        highlight: 'l5',
        outcome: 'ALLOW via L5 — intent preserved, unsafe parts gated',
    },
];

function stopFirewallFlow() {
    if (state.fwTimer) {
        clearTimeout(state.fwTimer);
        state.fwTimer = null;
    }
    state.fwRunning = false;
}

function startFirewallFlow() {
    if (!$('fw-flow')) return;
    stopFirewallFlow();
    state.fwRunning = true;
    if (state.fwScenarioIdx == null) state.fwScenarioIdx = 0;
    runFirewallScenario();
}

function fwClearHot() {
    document.querySelectorAll('#fw-flow .is-active, #fw-flow .is-block, #fw-flow .is-pass, #fw-flow .is-hot')
        .forEach((el) => el.classList.remove('is-active', 'is-block', 'is-pass', 'is-hot'));
    const verdict = $('fw-verdict');
    if (verdict) verdict.classList.remove('is-allow', 'is-block');
}

function fwBox(id) {
    return document.querySelector(`#fw-flow [data-fw="${id}"] .fw-box`)
        || document.querySelector(`#fw-flow [data-fw="${id}"]`);
}

function fwArrowBefore(nodeId) {
    const idx = FW_NODE_ORDER.indexOf(nodeId);
    if (idx <= 0) return null;
    return document.querySelector(`#fw-flow [data-fw-arrow="${idx - 1}"]`);
}

function runFirewallScenario() {
    if (!state.fwRunning || !$('welcome-tab')?.classList.contains('active')) return;
    fwClearHot();

    const scenario = FW_SCENARIOS[state.fwScenarioIdx % FW_SCENARIOS.length];
    state.fwScenarioIdx = (state.fwScenarioIdx + 1) % FW_SCENARIOS.length;

    const tagEl = $('fw-scenario-tag');
    const titleEl = $('fw-scenario-title');
    const descEl = $('fw-scenario-desc');
    const textEl = $('fw-verdict-text');
    const verdict = $('fw-verdict');
    if (tagEl) tagEl.textContent = scenario.tag;
    if (titleEl) titleEl.textContent = scenario.title;
    if (descEl) descEl.textContent = scenario.desc;
    if (textEl) textEl.textContent = 'Tracing this scenario…';

    let i = 0;
    const stepDelay = 750;

    const tick = () => {
        if (!state.fwRunning) return;
        if (i >= scenario.path.length) {
            state.fwTimer = setTimeout(() => {
                if (!state.fwRunning) return;
                runFirewallScenario();
            }, 2200);
            return;
        }

        const nodeId = scenario.path[i];
        const box = fwBox(nodeId);
        const arrow = fwArrowBefore(nodeId);

        document.querySelectorAll('#fw-flow .fw-box.is-active').forEach((el) => {
            el.classList.remove('is-active');
            if (!el.classList.contains('is-block')) el.classList.add('is-pass');
        });

        if (arrow) arrow.classList.add('is-active');
        if (box) box.classList.add('is-active');

        const isBlock = scenario.blockAt && nodeId === scenario.blockAt;
        if (isBlock) {
            if (box) {
                box.classList.remove('is-active', 'is-pass');
                box.classList.add('is-block');
            }
            if (arrow) {
                arrow.classList.remove('is-active');
                arrow.classList.add('is-block');
            }
            if (verdict) verdict.classList.add('is-block');
            if (textEl) textEl.textContent = scenario.outcome;
            state.fwTimer = setTimeout(() => {
                if (!state.fwRunning) return;
                runFirewallScenario();
            }, 2400);
            return;
        }

        if (nodeId === 'llm') {
            if (verdict) verdict.classList.add('is-allow');
            if (textEl) textEl.textContent = scenario.outcome;
            if (scenario.highlight) {
                const h = fwBox(scenario.highlight);
                if (h) {
                    h.classList.remove('is-pass');
                    h.classList.add('is-active');
                }
            }
        } else if (textEl) {
            textEl.textContent = `Checking ${nodeId.toUpperCase()}…`;
        }

        i += 1;
        state.fwTimer = setTimeout(tick, stepDelay);
    };

    state.fwTimer = setTimeout(tick, 500);
}

async function resumeActiveAccuracyJob() {
    try {
        const active = await api('/api/accuracy/active');
        if (!active?.active || !active.job?.id) return;
        const job = active.job;
        state.accJobId = job.id;
        if ($('acc-run')) $('acc-run').disabled = true;
        setAccCancelVisible(true);
        showAccProgress(modeLabel(job.mode), job.progress?.detail || 'Evaluation in progress…');
        updateAccProgress(job);
        setAccJobStatus(job.progress?.detail || 'Evaluation still running…', 'running');
        pollJob(job.id, 'accuracy', $('acc-status'), (done) => {
            finishAccuracyJobUI(done);
        }, { progress: 'accuracy' });
    } catch (_) { /* ignore */ }
}

/* ---------- Boot (no sign-in / sign-out UI) ---------- */
showApp({ restoreTab: true });

/* ---------- Tabs ---------- */
document.querySelectorAll('#sidebar [data-tab]').forEach((li) => {
    li.addEventListener('click', () => switchTab(li.dataset.tab));
});

const logoHome = $('logo-home');
if (logoHome) {
    logoHome.addEventListener('click', () => switchTab('welcome'));
}

/* ---------- System ---------- */
async function refreshSystem() {
    try {
        const data = await api('/api/system');
        state.reconnectTries = 0;
        const evalBusy = Boolean(data.eval_busy || data.api_busy || state.accJobId);
        const ok = data.python_status === 200 && data.python_health?.pipeline_loaded;
        if (ok) {
            $('api-dot').className = 'status-dot online';
            $('api-text').textContent = 'Healthy';
        } else if (evalBusy) {
            $('api-dot').className = 'status-dot online';
            $('api-text').textContent = 'Busy (eval running)';
        } else {
            $('api-dot').className = 'status-dot offline';
            $('api-text').textContent = 'Down / degraded';
        }
        $('admin-dot').className = `status-dot ${data.admin_api_ok || evalBusy ? 'online' : 'offline'}`;
        if (data.admin_api_ok) {
            $('admin-text').textContent = 'OK';
        } else if (evalBusy) {
            $('admin-text').textContent = data.admin_fail_reason || 'Busy with held-out/ablation';
        } else if (data.admin_fail_reason) {
            $('admin-text').textContent = data.admin_fail_reason;
        } else if (data.token_configured === false) {
            $('admin-text').textContent = 'Token missing — restart Admin after .env update';
        } else {
            $('admin-text').textContent = 'Admin route fail — restart Python API';
        }

        const a = data.admin || {};
        if (a.attack_types) {
            state.attackTypes = a.attack_types;
            fillTypeSelects();
        }
    } catch (e) {
        if (state.accJobId) {
            $('api-dot').className = 'status-dot online';
            $('api-text').textContent = 'Busy (eval running)';
            $('admin-dot').className = 'status-dot online';
            $('admin-text').textContent = 'Eval in progress — no reload needed';
            return;
        }
        const msg = String(e.message || e);
        const netFail = /failed to fetch|networkerror|load failed|unreachable/i.test(msg);
        state.reconnectTries = (state.reconnectTries || 0) + 1;
        if (netFail && state.reconnectTries < 8) {
            $('api-dot').className = 'status-dot online';
            $('api-text').textContent = 'Reconnecting…';
            $('admin-dot').className = 'status-dot online';
            $('admin-text').textContent = 'Eval just finished — waiting for API';
            return;
        }
        $('api-dot').className = 'status-dot offline';
        $('api-text').textContent = 'Unreachable';
        $('admin-dot').className = 'status-dot offline';
        $('admin-text').textContent = e.message;
    }
}

/* ---------- Live: Correct / Add New ---------- */
function typeOptionsHtml(selected) {
    const types = state.attackTypes || [];
    const opts = [{ id: 'unknown', name: 'Unknown' }, ...types];
    const seen = new Set();
    return opts.filter((t) => {
        if (seen.has(t.id)) return false;
        seen.add(t.id);
        return true;
    }).map((t) =>
        `<option value="${escapeHtml(t.id)}"${t.id === selected ? ' selected' : ''}>${escapeHtml(t.name)}</option>`
    ).join('');
}

function fillTypeSelects() {
    ['add-type', 'correct-type'].forEach((id) => {
        const sel = $(id);
        if (!sel) return;
        const keep = sel.value;
        sel.innerHTML = typeOptionsHtml(keep);
    });
}

document.querySelectorAll('.live-opt').forEach((btn) => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.live-opt').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const pane = btn.dataset.live;
        $('live-correct').classList.toggle('hidden', pane !== 'correct');
        $('live-add').classList.toggle('hidden', pane !== 'add');
        if (pane === 'correct') loadInbox();
    });
});

async function checkPromptTrained() {
    const status = $('add-lookup-status');
    const prompt = ($('manual-prompt')?.value || '').trim();
    if (!status) return;
    if (!prompt) {
        status.className = 'add-lookup-status is-warn';
        status.innerHTML = '<span class="add-lookup-plain">Type a prompt first.</span>';
        return;
    }
    status.className = 'add-lookup-status';
    status.innerHTML = '<span class="add-lookup-plain">Checking…</span>';
    try {
        const data = await api('/api/lab/lookup', {
            method: 'POST',
            body: JSON.stringify({ prompt }),
        });
        if (!data.trained) {
            status.className = 'add-lookup-status is-new';
            status.innerHTML = '<span class="add-lookup-plain">Not trained yet — safe to add as new.</span>';
            return;
        }
        const typeName = data.attack_display_name || data.attack_type || 'unknown';
        const typeId = data.attack_type || 'unknown';
        const verdict = Number(data.label) === 0 ? 'Safe' : 'Malicious';
        const src = data.source || 'trained';
        const detail = `${typeName} · ${verdict} · (${src})`;
        status.className = 'add-lookup-status is-trained';
        status.innerHTML = `
            <span class="add-lookup-label">Already trained</span>
            <span class="log-type add-lookup-badge" title="${escapeHtml(detail)}">${escapeHtml(detail)}</span>
        `;
        fillTypeSelects();
        if ($('add-type') && Array.from($('add-type').options).some((o) => o.value === typeId)) {
            $('add-type').value = typeId;
        }
        if ($('add-class') && data.label != null) {
            $('add-class').value = String(data.label);
        }
    } catch (e) {
        status.className = 'add-lookup-status is-warn';
        status.innerHTML = `<span class="add-lookup-plain">${escapeHtml(e.message || 'Lookup failed')}</span>`;
    }
}

$('add-lookup-btn')?.addEventListener('click', () => checkPromptTrained());

let _lookupDebounce = null;
$('manual-prompt')?.addEventListener('input', () => {
    const status = $('add-lookup-status');
    if (status && (status.textContent || status.innerHTML)) {
        status.innerHTML = '';
        status.className = 'add-lookup-status';
    }
    clearTimeout(_lookupDebounce);
});

async function loadInbox() {
    try {
        const data = await api('/api/inbox?status=live&limit=500&offset=0');
        const nNew = (data.counts && data.counts.new) || 0;
        const badge = $('lab-badge');
        if (badge) {
            badge.textContent = nNew > 0 ? String(nNew) : 'Lab';
            badge.classList.toggle('has-new', nNew > 0);
        }
        const items = data.items || [];
        const list = $('inbox-list');
        if (!list) return;
        if (!items.length) {
            list.innerHTML = '<div class="log-row"><p class="q-prompt">No blocked prompts yet. When chat blocks anyone, it shows up here.</p></div>';
            return;
        }
        list.innerHTML = items.map((it) => {
            const type = it.team_attack_type || it.system_attack_type || 'unknown';
            return `
                <div class="log-row${state.selectedCaseId === it.id ? ' active' : ''}" data-id="${escapeHtml(it.id)}" role="button" tabindex="0">
                    <p class="q-prompt">${escapeHtml(it.prompt || it.prompt_preview || '')}</p>
                    <span class="log-type">${escapeHtml(type)}</span>
                    <span class="hidden" data-type="${escapeHtml(type)}" data-label="${it.team_label != null ? it.team_label : 1}"></span>
                </div>
            `;
        }).join('');
        list.querySelectorAll('.log-row[data-id]').forEach((el) => {
            el.addEventListener('click', () => selectLogRow(el));
        });
        if (state.selectedCaseId) {
            const keep = list.querySelector(`.log-row[data-id="${CSS.escape(state.selectedCaseId)}"]`);
            if (keep) keep.classList.add('active');
        }
    } catch (e) {
        if ($('inbox-list')) {
            $('inbox-list').innerHTML = `<div class="log-row"><p class="q-prompt">${escapeHtml(e.message)}</p></div>`;
        }
    }
}

function selectLogRow(el) {
    const id = el.dataset.id;
    if (!id) return;
    state.selectedCaseId = id;
    document.querySelectorAll('#inbox-list .log-row').forEach((r) => r.classList.remove('active'));
    el.classList.add('active');
    const prompt = el.querySelector('.q-prompt')?.textContent || '';
    const meta = el.querySelector('[data-type]');
    const type = meta?.dataset.type || 'unknown';
    const label = meta?.dataset.label || '1';
    $('correct-prompt').textContent = prompt;
    $('correct-meta').textContent = 'Set attack type and malicious / safe, then Train.';
    fillTypeSelects();
    if ($('correct-type') && Array.from($('correct-type').options).some((o) => o.value === type)) {
        $('correct-type').value = type;
    }
    $('correct-class').value = String(label);
}

async function commitLabExample() {
    const pane = document.querySelector('.live-opt.active')?.dataset.live || 'correct';
    if (pane === 'add') {
        const prompt = $('manual-prompt').value.trim();
        const label = Number($('add-class').value);
        let attackType = $('add-type').value || 'unknown';
        if (!prompt) {
            throw new Error('Type a prompt in Add New, or switch to Correct and pick a row.');
        }
        if (label === 1 && (!attackType || attackType === 'unknown')) {
            throw new Error('Pick an attack type for a malicious prompt.');
        }
        if (label === 0) attackType = attackType === 'unknown' ? 'benign' : attackType;
        await api('/api/inbox/manual', {
            method: 'POST',
            body: JSON.stringify({ prompt, attack_type: attackType, label, notes: '' }),
        });
        $('manual-prompt').value = '';
        return;
    }
    const id = state.selectedCaseId;
    if (!id) {
        throw new Error('Click a prompt in Correct, then Train.');
    }
    const label = Number($('correct-class').value);
    let attackType = $('correct-type').value || 'unknown';
    if (label === 1 && (!attackType || attackType === 'unknown')) {
        throw new Error('Pick an attack type for a malicious prompt.');
    }
    if (label === 0) attackType = attackType === 'unknown' ? 'benign' : attackType;
    await api(`/api/inbox/${id}/review`, {
        method: 'POST',
        body: JSON.stringify({
            attack_type: attackType,
            label,
            notes: '',
            discard: false,
        }),
    });
}

function formatElapsed(totalSeconds) {
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
}

function showTrainProgress(label, detail, scope = 'live') {
    const wrap = $(`${scope}-train-progress`);
    const bar = $(`${scope}-train-bar`);
    const labelEl = $(`${scope}-train-label`);
    const detailEl = $(`${scope}-train-status`);
    const elapsedEl = $(`${scope}-train-elapsed`);
    if (!wrap || !bar) return;
    wrap.classList.remove('hidden');
    requestAnimationFrame(() => wrap.classList.add('is-active'));
    bar.className = 'train-progress-bar indeterminate';
    bar.style.width = '';
    if (labelEl) labelEl.textContent = label || 'Training';
    if (detailEl) {
        detailEl.style.opacity = '0';
        detailEl.textContent = detail || '';
        requestAnimationFrame(() => { detailEl.style.opacity = ''; });
    }
    state.trainStartedAt = Date.now();
    if (elapsedEl) elapsedEl.textContent = '0:00';
    if (state.trainElapsedTimer) clearInterval(state.trainElapsedTimer);
    state.trainElapsedTimer = setInterval(() => {
        if (!elapsedEl || !state.trainStartedAt) return;
        const sec = Math.floor((Date.now() - state.trainStartedAt) / 1000);
        elapsedEl.textContent = formatElapsed(sec);
    }, 1000);
}

function finishTrainProgress(ok, label, detail, scope = 'live') {
    const wrap = $(`${scope}-train-progress`);
    const bar = $(`${scope}-train-bar`);
    const labelEl = $(`${scope}-train-label`);
    const detailEl = $(`${scope}-train-status`);
    if (state.trainElapsedTimer) {
        clearInterval(state.trainElapsedTimer);
        state.trainElapsedTimer = null;
    }
    if (bar) {
        bar.classList.remove('indeterminate');
        bar.classList.add(ok ? 'done' : 'failed');
    }
    if (labelEl) labelEl.textContent = label || (ok ? 'Done' : 'Failed');
    if (detailEl) detailEl.textContent = detail || '';
    if (wrap) wrap.classList.remove('is-active');
}

function hideTrainProgress(scope = 'live') {
    const wrap = $(`${scope}-train-progress`);
    if (wrap) {
        wrap.classList.remove('is-active');
        wrap.classList.add('hidden');
    }
    if (state.trainElapsedTimer) {
        clearInterval(state.trainElapsedTimer);
        state.trainElapsedTimer = null;
    }
    state.trainStartedAt = 0;
}

$('live-train').addEventListener('click', async () => {
    const trainBtn = $('live-train');
    trainBtn.disabled = true;
    showTrainProgress('Applying', 'Saving your correction…');
    try {
        await commitLabExample();
        showTrainProgress('Training', 'Updating live model…');
        const data = await api('/api/datasets/train', {
            method: 'POST',
            body: JSON.stringify({ include_review_queue: true, rebuild_splits: false }),
        });
        pollJob(data.job_id, 'train', $('live-train-status'), (job) => {
            trainBtn.disabled = false;
            if (job && job.status === 'ok') {
                finishTrainProgress(true, 'Done', '');
                state.selectedCaseId = null;
                $('correct-prompt').textContent = 'No prompt selected.';
                loadInbox();
            } else if (job && job.status === 'failed') {
                finishTrainProgress(false, 'Failed', job.error || 'Train failed.');
            } else {
                hideTrainProgress();
            }
            refreshSystem();
        }, { progress: true });
    } catch (e) {
        finishTrainProgress(false, 'Failed', e.message);
        trainBtn.disabled = false;
    }
});

/* ---------- Report (research eval) ---------- */

function setAccCancelVisible(visible) {
    const btn = $('acc-cancel');
    if (!btn) return;
    if (visible) {
        btn.classList.remove('hidden');
        btn.disabled = false;
        btn.setAttribute('aria-hidden', 'false');
    } else {
        btn.classList.add('hidden');
        btn.disabled = true;
        btn.setAttribute('aria-hidden', 'true');
    }
}
const ACC_MODE_HELP = {
    heldout: 'Scores the full defense on the last StratifiedGroupKFold test fold.',
    ablation: 'Re-runs the full held-out test with layers turned off.',
    cv: '5-fold StratifiedGroupKFold: refits Layer 2 each fold; groups never leak.',
};

function syncAccModeUI() {
    const mode = document.querySelector('input[name="acc-mode-radio"]:checked')?.value || 'heldout';
    if ($('acc-mode')) $('acc-mode').value = mode;
    document.querySelectorAll('.acc-mode-card').forEach((card) => {
        const input = card.querySelector('input');
        card.classList.toggle('is-selected', Boolean(input?.checked));
    });
    if ($('acc-run-hint')) {
        if (mode === 'ablation') {
            $('acc-run-hint').textContent = 'Full test set × each ablation — this can take a long time.';
        } else if (mode === 'cv') {
            $('acc-run-hint').textContent = 'Five train+eval cycles. Stay on this page; it can take a long time.';
        } else {
            $('acc-run-hint').textContent = 'Uses the full held-out test set. Stay on this page.';
        }
    }
}

document.querySelectorAll('input[name="acc-mode-radio"]').forEach((input) => {
    input.addEventListener('change', syncAccModeUI);
});

function setAccJobStatus(text, stateName = 'idle') {
    const wrap = $('acc-job-status');
    const el = $('acc-job-text');
    const msg = formatApiError(text, 'Status update');
    if (el) el.textContent = msg;
    if (wrap) {
        wrap.classList.remove('is-running', 'is-ok', 'is-fail');
        if (stateName === 'running') wrap.classList.add('is-running');
        if (stateName === 'ok') wrap.classList.add('is-ok');
        if (stateName === 'fail') wrap.classList.add('is-fail');
    }
}

function isCancelledEval(job) {
    if (state.accCancelRequested) return true;
    const err = String(job?.error || job?.progress?.detail || '');
    return /cancell?ed by user/i.test(err) || /cancell?ed/i.test(String(job?.status || ''));
}

function resetAccResults(message) {
    const el = $('acc-results');
    if (!el) return;
    el.innerHTML = `<div class="acc-empty">${escapeHtml(message || 'Run an evaluation above to see scores and tables.')}</div>`;
}

function finishAccuracyJobUI(job) {
    state.quietInboxUntil = Date.now() + 15000;
    $('acc-run').disabled = false;
    setAccCancelVisible(false);
    state.accJobId = null;
    setTimeout(() => loadAccuracyReports(), 1200);
    setTimeout(() => refreshSystem(), 1500);

    if (isCancelledEval(job)) {
        state.accCancelRequested = false;
        finishAccProgress(false, 'Cancelled', 'Stopped by you — no results to show.');
        setAccJobStatus('Cancelled. You can start a new run.', 'fail');
        resetAccResults('Cancelled — no evaluation results. Start a new run when ready.');
        if ($('acc-status')) $('acc-status').textContent = 'Cancelled by user.';
        return;
    }

    state.accCancelRequested = false;
    if (job?.status === 'ok' || (job?.report && (job.exit_code === 0 || job.report?.ablations || job.report?.metrics))) {
        finishAccProgress(true, 'Done', 'Finished — results below.');
        setAccJobStatus('Finished — results below.', 'ok');
        if (job.report) renderAccuracyReport(job.report, { mode: job.mode });
        return;
    }

    const tail = String(job?.log_tail || '').trim();
    const lastLines = tail ? tail.split(/\r?\n/).filter(Boolean).slice(-6).join(' | ') : '';
    const failMsg = formatApiError(
        job?.error || lastLines || job,
        'Evaluation failed. Check the log.'
    );
    if (job?.report && /exited unexpectedly/i.test(String(job?.error || ''))) {
        finishAccProgress(true, 'Done', 'Finished — results below.');
        setAccJobStatus('Finished — results below.', 'ok');
        renderAccuracyReport(job.report, { mode: job.mode });
        return;
    }
    finishAccProgress(false, 'Failed', failMsg);
    setAccJobStatus(failMsg, 'fail');
    // Keep Results clean on cancel/noise; only dump log for real failures.
    if (tail && $('acc-results') && !/cancell?ed/i.test(failMsg)) {
        $('acc-results').innerHTML = `<pre class="code-block">${escapeHtml(tail.slice(-4000))}</pre>`;
    } else {
        resetAccResults('Evaluation did not finish. You can start a new run.');
    }
}

function showAccProgress(label, detail) {
    const wrap = $('acc-run-progress');
    const bar = $('acc-run-bar');
    const labelEl = $('acc-run-label');
    const detailEl = $('acc-run-detail');
    const elapsedEl = $('acc-run-elapsed');
    if (!wrap || !bar) return;
    wrap.classList.remove('hidden');
    requestAnimationFrame(() => wrap.classList.add('is-active'));
    bar.className = 'train-progress-bar indeterminate';
    bar.style.width = '';
    if (labelEl) labelEl.textContent = label || 'Evaluating';
    if (detailEl) detailEl.textContent = detail || '';
    state.accStartedAt = Date.now();
    if (elapsedEl) elapsedEl.textContent = '0:00';
    if (state.accElapsedTimer) clearInterval(state.accElapsedTimer);
    state.accElapsedTimer = setInterval(() => {
        if (!elapsedEl || !state.accStartedAt) return;
        const sec = Math.floor((Date.now() - state.accStartedAt) / 1000);
        elapsedEl.textContent = formatElapsed(sec);
    }, 1000);
}

function updateAccProgress(job) {
    const bar = $('acc-run-bar');
    const labelEl = $('acc-run-label');
    const detailEl = $('acc-run-detail');
    const track = bar?.parentElement;
    const prog = job?.progress || {};
    const pct = Number(prog.pct) || 0;
    const detail = prog.detail
        || (job?.status === 'queued' ? 'Queued — waiting to start…' : `Running ${modeLabel(job?.mode)}…`);
    if (labelEl) {
        labelEl.textContent = job?.status === 'queued'
            ? 'Queued'
            : (prog.phase === 'loading' ? 'Loading' : modeLabel(job?.mode) || 'Evaluating');
    }
    const hw = prog.hw_label
        || (prog.hw_path === 'gpu' ? 'GPU path' : (prog.hw_path === 'cpu' ? 'CPU path' : ''));
    if (detailEl) detailEl.textContent = hw ? `${detail} · ${hw}` : detail;
    if (bar && track) {
        if (pct > 0) {
            bar.classList.remove('indeterminate');
            bar.style.width = `${Math.min(100, Math.max(2, pct))}%`;
            track.setAttribute('aria-valuenow', String(Math.round(pct)));
        } else {
            bar.className = 'train-progress-bar indeterminate';
            bar.style.width = '';
        }
    }
    setAccJobStatus(detail, 'running');
}

function finishAccProgress(ok, label, detail) {
    const wrap = $('acc-run-progress');
    const bar = $('acc-run-bar');
    const labelEl = $('acc-run-label');
    const detailEl = $('acc-run-detail');
    if (state.accElapsedTimer) {
        clearInterval(state.accElapsedTimer);
        state.accElapsedTimer = null;
    }
    if (bar) {
        bar.classList.remove('indeterminate');
        bar.style.width = '100%';
        bar.classList.add(ok ? 'done' : 'failed');
    }
    if (labelEl) labelEl.textContent = label || (ok ? 'Done' : 'Failed');
    if (detailEl) detailEl.textContent = detail || '';
    if (wrap) wrap.classList.remove('is-active');
    setAccCancelVisible(false);
}

function fmtPct(v) {
    if (v == null || Number.isNaN(Number(v))) return '—';
    return Number(v).toFixed(4);
}

function fmtWhen(iso) {
    if (!iso) return '';
    try {
        return new Date(iso).toLocaleString();
    } catch (_) {
        return iso;
    }
}

function modeLabel(mode) {
    if (mode === 'heldout') return 'Main test (held-out)';
    if (mode === 'ablation') return 'Ablation (layers)';
    if (mode === 'cv') return '5-fold stratified group CV';
    return mode || 'Report';
}

function renderMetricCards(m) {
    if (!m) return '';
    const cards = [
        { key: 'F1', val: m.f1, tip: 'Balance of precision & recall — main paper score' },
        { key: 'AUC-ROC', val: m.auc_roc, tip: 'Ranking quality from risk scores (threshold-free; higher is better)' },
        { key: 'Recall', val: m.recall, tip: 'Share of attacks caught (higher = fewer misses)' },
        { key: 'Precision', val: m.precision, tip: 'When we block, how often we are right' },
        { key: 'Accuracy', val: m.accuracy, tip: 'Overall correct predictions' },
        { key: 'FPR', val: m.fpr, tip: 'False alarms on safe prompts (lower is better)' },
        { key: 'Latency', val: m.latency_ms_mean != null ? `${Number(m.latency_ms_mean).toFixed(1)} ms` : null, tip: 'Average time per prompt' },
    ];
    return `<div class="acc-metric-grid">${cards.map((c) => `
        <div class="acc-metric-card" title="${escapeHtml(c.tip)}">
            <span class="acc-metric-key">${escapeHtml(c.key)}</span>
            <span class="acc-metric-val">${c.val == null ? '—' : (typeof c.val === 'number' ? fmtPct(c.val) : escapeHtml(String(c.val)))}</span>
            <span class="acc-metric-tip">${escapeHtml(c.tip)}</span>
        </div>`).join('')}</div>`;
}

function renderConfusion(cm) {
    if (!cm) return '';
    return `
        <div class="acc-confusion">
            <h4>Confusion matrix</h4>
            <p>How predictions compare to the true labels.</p>
            <div class="acc-cm-grid">
                <div class="acc-cm-cell ok"><strong>TP ${cm.tp ?? '—'}</strong><span>Attack → correctly blocked</span></div>
                <div class="acc-cm-cell ok"><strong>TN ${cm.tn ?? '—'}</strong><span>Safe → correctly allowed</span></div>
                <div class="acc-cm-cell bad"><strong>FP ${cm.fp ?? '—'}</strong><span>Safe → wrongly blocked</span></div>
                <div class="acc-cm-cell bad"><strong>FN ${cm.fn ?? '—'}</strong><span>Attack → missed</span></div>
            </div>
        </div>`;
}

function renderDecisionSources(sources) {
    if (!sources || !Object.keys(sources).length) return '';
    const rows = Object.entries(sources).sort((a, b) => Number(b[1]) - Number(a[1]));
    const total = rows.reduce((s, [, n]) => s + Number(n), 0) || 1;
    return `
        <div class="acc-table-wrap">
            <h4>Who decided?</h4>
            <p class="acc-help">Which layer made the final call on each example.</p>
            <table class="acc-table">
                <thead><tr><th>Source</th><th>Count</th><th>Share</th></tr></thead>
                <tbody>
                    ${rows.map(([k, v]) => `
                        <tr>
                            <td><code>${escapeHtml(k)}</code></td>
                            <td>${v}</td>
                            <td>
                                <div class="acc-bar-wrap">
                                    <span class="acc-bar" style="width:${Math.max(4, (100 * Number(v)) / total)}%"></span>
                                    <em>${((100 * Number(v)) / total).toFixed(1)}%</em>
                                </div>
                            </td>
                        </tr>`).join('')}
                </tbody>
            </table>
        </div>`;
}

function renderAblationTable(ablations) {
    if (!ablations) return '';
    const rows = Object.entries(ablations);
    return `
        <div class="acc-table-wrap">
            <h4>Ablation comparison</h4>
            <p class="acc-help">Same test set. Compare <code>full</code> to rows with layers removed.</p>
            <div class="acc-table-scroll">
                <table class="acc-table">
                    <thead>
                        <tr>
                            <th>Setup</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th><th>AUC</th><th>FPR</th><th>ms</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map(([name, block]) => {
                            const m = block.metrics || {};
                            const isFull = name === 'full';
                            return `<tr class="${isFull ? 'is-full' : ''}">
                                <td><strong>${escapeHtml(name)}</strong></td>
                                <td>${fmtPct(m.accuracy)}</td>
                                <td>${fmtPct(m.precision)}</td>
                                <td>${fmtPct(m.recall)}</td>
                                <td><strong>${fmtPct(m.f1)}</strong></td>
                                <td>${fmtPct(m.auc_roc)}</td>
                                <td>${fmtPct(m.fpr)}</td>
                                <td>${m.latency_ms_mean != null ? Number(m.latency_ms_mean).toFixed(1) : '—'}</td>
                            </tr>`;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </div>`;
}

function renderTypeDetection(types) {
    if (!types || !Object.keys(types).length) return '';
    const rows = Object.entries(types).sort((a, b) => (b[1].detection_rate || 0) - (a[1].detection_rate || 0));
    return `
        <div class="acc-table-wrap">
            <h4>Catch rate by attack type</h4>
            <table class="acc-table">
                <thead><tr><th>Type</th><th>Seen</th><th>Caught</th><th>Rate</th></tr></thead>
                <tbody>
                    ${rows.map(([k, info]) => `
                        <tr>
                            <td>${escapeHtml(k)}</td>
                            <td>${info.total ?? '—'}</td>
                            <td>${info.detected ?? '—'}</td>
                            <td>${fmtPct(info.detection_rate)}</td>
                        </tr>`).join('')}
                </tbody>
            </table>
        </div>`;
}

function renderAccuracyReport(report, meta = {}) {
    const el = $('acc-results');
    if (!el || !report) return;
    const mode = report.mode || meta.mode || '';
    let body = '';
    if (report.metrics) {
        body += renderMetricCards(report.metrics);
        body += renderConfusion(report.metrics.confusion_matrix);
        body += renderDecisionSources(report.metrics.decision_sources);
        body += renderTypeDetection(report.type_detection);
    }
    if (report.ablations) {
        const full = report.ablations.full?.metrics;
        if (full) {
            body += `<p class="acc-help"><strong>Highlight:</strong> scores below for <code>full</code> (all layers on).</p>`;
            body += renderMetricCards(full);
        }
        body += renderAblationTable(report.ablations);
    }
    if (report.aggregate) {
        const flat = {};
        for (const [k, s] of Object.entries(report.aggregate)) {
            flat[k] = s?.mean;
        }
        const aggNote = report.mode === 'cv'
            ? 'Mean scores across StratifiedGroupKFold validation folds (paper-primary).'
            : 'Pattern-bank averages across rounds (appendix).';
        body += `<p class="acc-help">${aggNote}</p>`;
        body += renderMetricCards(flat);
    }
    if (report.mode === 'cv' && Array.isArray(report.folds) && report.folds.length) {
        body += `<div class="acc-table-wrap"><table class="acc-table"><thead><tr>
            <th>Fold</th><th>n</th><th>Acc</th><th>F1</th><th>AUC</th><th>Groups</th><th>Overlap</th>
        </tr></thead><tbody>${report.folds.map((fold) => {
            const m = fold.metrics || {};
            return `<tr>
                <td>${fold.fold ?? ''}</td>
                <td>${m.n ?? fold.n_val ?? '—'}</td>
                <td>${fmtPct(m.accuracy)}</td>
                <td>${fmtPct(m.f1)}</td>
                <td>${fmtPct(m.auc_roc)}</td>
                <td>${fold.n_val_groups ?? '—'}</td>
                <td>${fold.group_overlap ?? 0}</td>
            </tr>`;
        }).join('')}</tbody></table></div>`;
    }
    if (!body) {
        body = `<pre class="code-block">${escapeHtml(JSON.stringify(report, null, 2))}</pre>`;
    }
    const title = String(report.title || modeLabel(mode))
        .replace(/\s*\(paper-primary\)\s*/gi, '')
        .replace(/\s*\(paper[^)]*\)\s*/gi, '')
        .trim();

    el.innerHTML = `
        <div class="acc-result-head">
            <div>
                <h4>${escapeHtml(title)}</h4>
                <p class="acc-meta-line">${escapeHtml(modeLabel(mode))} · ${escapeHtml(fmtWhen(report.generated_at || meta.mtime))}</p>
            </div>
            <span class="acc-mode-pill">${escapeHtml(modeLabel(mode))}</span>
        </div>
        ${body}
    `;
}

$('acc-run').addEventListener('click', async () => {
    syncAccModeUI();
    $('acc-run').disabled = true;
    setAccCancelVisible(true);
    showAccProgress('Starting', 'Queuing evaluation…');
    setAccJobStatus('Starting evaluation…', 'running');
    if ($('acc-status')) $('acc-status').textContent = 'Queuing…';
    resetAccResults('Evaluation running… results will appear here when finished.');
    try {
        const data = await api('/api/accuracy/run', {
            method: 'POST',
            body: JSON.stringify({
                mode: $('acc-mode').value,
                full_test: true,
            }),
        });
        state.accJobId = data.job_id;
        state.accCancelRequested = false;
        showAccProgress(modeLabel($('acc-mode').value), 'Loading pipeline…');
        pollJob(data.job_id, 'accuracy', $('acc-status'), (job) => {
            finishAccuracyJobUI(job);
        }, { progress: 'accuracy' });
    } catch (e) {
        const msg = String(e.message || e);
        // 409 = already running — attach to that job instead of showing a hard fail.
        if (e.status === 409 || /already running/i.test(msg)) {
            setAccJobStatus(msg, 'running');
            if ($('acc-status')) $('acc-status').textContent = msg;
            try {
                const active = await api('/api/accuracy/active');
                if (active?.active && active.job?.id) {
                    state.accJobId = active.job.id;
                    state.accCancelRequested = false;
                    showAccProgress(modeLabel(active.job.mode), active.job.progress?.detail || 'Resuming…');
                    updateAccProgress(active.job);
                    setAccCancelVisible(true);
                    pollJob(active.job.id, 'accuracy', $('acc-status'), (job) => {
                        finishAccuracyJobUI(job);
                    }, { progress: 'accuracy' });
                    return;
                }
            } catch (_) { /* fall through */ }
        }
        finishAccProgress(false, 'Failed', msg);
        setAccJobStatus(msg, 'fail');
        if ($('acc-status')) $('acc-status').textContent = msg;
        $('acc-run').disabled = false;
        setAccCancelVisible(false);
        if (e.status === 401) {
            setAccJobStatus('Session expired — refresh the page, then run again (eval may still be running).', 'fail');
        }
    }
});

if ($('acc-cancel')) {
    $('acc-cancel').addEventListener('click', async () => {
        setAccCancelVisible(false);
        state.accCancelRequested = true;
        const jobId = state.accJobId;
        if (jobId && state.pollTimers[jobId]) {
            clearInterval(state.pollTimers[jobId]);
            delete state.pollTimers[jobId];
        }
        resetAccResults('Cancelled — no evaluation results. Start a new run when ready.');
        finishAccProgress(false, 'Cancelled', 'Stopped by you — no results to show.');
        setAccJobStatus('Cancelled. You can start a new run.', 'fail');
        if ($('acc-status')) $('acc-status').textContent = 'Cancelled by user.';
        try {
            await api('/api/accuracy/cancel', {
                method: 'POST',
                body: JSON.stringify({ job_id: jobId || null }),
            });
        } catch (e) {
            setAccJobStatus(e.message || 'Cancel failed', 'fail');
        } finally {
            $('acc-run').disabled = false;
            state.accJobId = null;
        }
    });
}

async function loadAccuracyReports() {
    try {
        const data = await api('/api/accuracy/reports');
        const items = data.items || [];
        const list = $('acc-reports');
        if (!list) return;
        if (!items.length) {
            list.innerHTML = '<div class="acc-empty">No saved reports yet. Run an evaluation above.</div>';
            return;
        }
        list.innerHTML = items.map((r) => {
            const s = r.summary || {};
            const bits = [];
            if (s.f1 != null) bits.push(`F1 ${fmtPct(s.f1)}`);
            if (s.auc_roc != null) bits.push(`AUC ${fmtPct(s.auc_roc)}`);
            if (s.recall != null) bits.push(`Rec ${fmtPct(s.recall)}`);
            if (s.fpr != null) bits.push(`FPR ${fmtPct(s.fpr)}`);
            if (s.n != null) bits.push(`N=${s.n}`);
            return `
                <button type="button" class="acc-report-card" data-name="${escapeHtml(r.name)}">
                    <div class="acc-report-top">
                        <strong>${escapeHtml(modeLabel(r.mode) || r.title || r.name)}</strong>
                        <span>${escapeHtml(fmtWhen(r.mtime))}</span>
                </div>
                    <div class="acc-report-metrics">${bits.length ? escapeHtml(bits.join(' · ')) : escapeHtml(r.name)}</div>
                </button>`;
        }).join('');
        list.querySelectorAll('.acc-report-card').forEach((btn) => {
            btn.addEventListener('click', () => openAccuracyReport(btn.dataset.name));
        });
    } catch (_) { /* ignore */ }
}

async function openAccuracyReport(name) {
    if (!name) return;
    setAccJobStatus(`Opening ${name}…`, 'running');
    try {
        const data = await api(`/api/accuracy/reports/${encodeURIComponent(name)}`);
        renderAccuracyReport(data.report, { paper_dir: data.paper_dir, mtime: data.report?.generated_at });
        setAccJobStatus('Report loaded in Results.', 'ok');
        $('acc-results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
        setAccJobStatus(e.message, 'fail');
    }
}

syncAccModeUI();

/* ---------- Datasets ---------- */
async function loadDatasetsPanel() {
    try {
        const data = await api('/api/datasets/taxonomy');
        if ($('ds-train-count')) {
            $('ds-train-count').textContent = `Train rows: ${(data.train_rows || 0).toLocaleString()}`;
        }
    } catch (_) { /* ignore */ }
}

$('ds-upload').addEventListener('click', async () => {
    const file = $('ds-file')?.files?.[0];
    if (!file) return alert('Choose a file');
    const uploadBtn = $('ds-upload');
    const trainBtn = $('ds-train');
    uploadBtn.disabled = true;
    if (trainBtn) trainBtn.disabled = true;

    const fd = new FormData();
    fd.append('file', file);

    try {
        showTrainProgress('Uploading', `Sending ${file.name}…`, 'ds');
        const res = await fetch('/api/datasets/upload', { method: 'POST', body: fd, credentials: 'include' });
        const up = await res.json();
        if (!res.ok) throw new Error(formatApiError(up, 'Upload failed'));

        showTrainProgress('Ingesting', 'Queued normalize · map · dedupe…', 'ds');
        const queued = await api('/api/datasets/ingest', {
            method: 'POST',
            body: JSON.stringify({
                filename: up.filename,
                source: up.source_guess || undefined,
                dry_run: false,
                async_job: true,
            }),
        });
        if (!queued?.job_id) throw new Error('Ingest job was not started');

        pollJob(queued.job_id, 'train', $('ds-train-status'), (job) => {
            uploadBtn.disabled = false;
            if (trainBtn) trainBtn.disabled = false;
            if (job && job.status === 'ok') {
                const result = job.result || {};
                const added = result.appended ?? job.appended ?? 0;
                const st = result.stats || {};
                const dups = (st.duplicates_vs_train || 0) + (st.duplicates_in_file || 0);
                const trainRows = result.train_rows ?? job.train_rows;
                const detail = `Added ${Number(added).toLocaleString()} rows` +
                    (dups ? ` · skipped ${Number(dups).toLocaleString()} duplicates` : '') +
                    (trainRows != null ? ` · train now ${Number(trainRows).toLocaleString()}` : '');
                finishTrainProgress(true, 'Uploaded', detail, 'ds');
                if ($('ds-train-count') && trainRows != null) {
                    $('ds-train-count').textContent = `Train rows: ${Number(trainRows).toLocaleString()}`;
                }
                loadDatasetsPanel();
            } else {
                finishTrainProgress(
                    false,
                    'Failed',
                    formatApiError(job?.error || job?.progress?.detail || job, 'Ingest failed'),
                    'ds',
                );
            }
            refreshSystem();
        }, { progress: 'train', progressScope: 'ds', progressLabel: 'Ingesting' });
    } catch (e) {
        const msg = (e && e.message) ? e.message : formatApiError(e?.data, 'Upload failed');
        finishTrainProgress(false, 'Failed', msg, 'ds');
        uploadBtn.disabled = false;
        if (trainBtn) trainBtn.disabled = false;
    }
});

$('ds-train').addEventListener('click', async () => {
    const trainBtn = $('ds-train');
    const uploadBtn = $('ds-upload');
    trainBtn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    showTrainProgress('Training', 'Retraining Layer 2 on train.jsonl…', 'ds');
    try {
        const data = await api('/api/datasets/train', {
            method: 'POST',
            body: JSON.stringify({
                include_review_queue: true,
                rebuild_splits: false,
            }),
        });
        pollJob(data.job_id, 'train', $('ds-train-status'), (job) => {
            trainBtn.disabled = false;
            if (uploadBtn) uploadBtn.disabled = false;
            if (job && job.status === 'ok') {
                finishTrainProgress(true, 'Done', 'Layer 2 and attack bank updated.', 'ds');
                loadInbox();
                loadDatasetsPanel();
            } else {
                finishTrainProgress(false, 'Failed', formatApiError(job?.error || job, 'Train failed'), 'ds');
            }
            refreshSystem();
        }, { progress: 'train', progressScope: 'ds' });
    } catch (e) {
        finishTrainProgress(false, 'Failed', formatApiError(e, e.message), 'ds');
        trainBtn.disabled = false;
        if (uploadBtn) uploadBtn.disabled = false;
    }
});

function pollJob(jobId, kind, el, onDone, opts = {}) {
    const path = kind === 'accuracy'
        ? `/api/accuracy/jobs/${jobId}`
        : `/api/jobs/${jobId}`;
    if (state.pollTimers[jobId]) clearInterval(state.pollTimers[jobId]);
    const progressKind = opts.progress === true ? 'train' : (opts.progress || null);
    let softFails = 0;
    state.pollTimers[jobId] = setInterval(async () => {
        try {
            const job = await api(path);
            softFails = 0;
            if (kind === 'accuracy' && (job.status === 'running' || job.status === 'queued')) {
                updateAccProgress(job);
            }
            if (el && el.tagName === 'PRE') {
                if (kind === 'accuracy' && (state.accCancelRequested || isCancelledEval(job))) {
                    el.textContent = 'Cancelled by user.';
                } else {
                    el.textContent = JSON.stringify({
                        id: job.id,
                        status: job.status,
                        mode: job.mode,
                        exit_code: job.exit_code,
                        error: job.error,
                        progress: job.progress || null,
                        merged_review: job.merged_review,
                        model_reloaded: job.model_reloaded,
                        inbox_marked_trained: job.inbox_marked_trained,
                        report_metrics: job.report?.metrics || job.report?.aggregate || null,
                        log_tail: (kind === 'accuracy' && isCancelledEval(job))
                            ? null
                            : (job.log_tail ? String(job.log_tail).slice(-1200) : null),
                    }, null, 2);
                }
            } else if (progressKind === 'train' && (job.status === 'running' || job.status === 'queued')) {
                const scope = opts.progressScope || 'live';
                const labelEl = $(`${scope}-train-label`);
                const detailEl = $(`${scope}-train-status`);
                const wrap = $(`${scope}-train-progress`);
                if (wrap && !wrap.classList.contains('is-active')) wrap.classList.add('is-active');
                const defaultLabel = opts.progressLabel || 'Training';
                if (labelEl) {
                    labelEl.textContent = job.status === 'queued'
                        ? 'Queued'
                        : (job.kind === 'ingest' ? 'Ingesting' : defaultLabel);
                }
                if (detailEl) {
                    let next;
                    if (job.status === 'queued') {
                        next = 'Waiting to start…';
                    } else if (job.progress?.detail) {
                        next = job.progress.detail;
                    } else if (job.kind === 'ingest') {
                        next = 'Normalize · map · dedupe…';
                    } else if (job.merged_review != null && !job.model_reloaded) {
                        next = 'Retraining Layer 2 model…';
                    } else {
                        next = 'Updating live model…';
                    }
                    if (detailEl.textContent !== next) {
                        detailEl.style.opacity = '0.45';
                        detailEl.textContent = next;
                        requestAnimationFrame(() => { detailEl.style.opacity = ''; });
                    }
                }
            } else if (job.status === 'running' || job.status === 'queued') {
                if (el && el.tagName !== 'PRE') el.textContent = 'Working…';
            }
            if (job.status === 'ok' || job.status === 'failed') {
                clearInterval(state.pollTimers[jobId]);
                if (onDone) onDone(job);
            }
        } catch (e) {
            // Long ablation/heldout runs outlive API reloads; keep polling (disk fallback on Admin).
            softFails += 1;
            if (e.status === 401) {
                // Session died — do NOT mark the eval itself as failed.
                clearInterval(state.pollTimers[jobId]);
                if (progressKind === 'accuracy') {
                    setAccJobStatus(
                        'Session expired — unlock again to keep watching (eval still runs on the server).',
                        'running',
                    );
                    setAccCancelVisible(false);
                    // Leave Run disabled until unlock + resume; job is still server-side.
                }
                return;
            }
            if (kind === 'accuracy' && softFails < 600) {
                if (progressKind === 'accuracy') {
                    setAccJobStatus(
                        `API busy — eval still running (retry ${softFails})…`,
                        'running',
                    );
                }
                if (el && el.tagName !== 'PRE') {
                    el.textContent = `Still running (reconnect ${softFails})…`;
                }
                if (softFails % 5 === 0) {
                    try {
                        const active = await api('/api/accuracy/active');
                        if (active?.active && active.job) {
                            if (active.job.id) state.accJobId = active.job.id;
                            updateAccProgress(active.job);
                        }
                    } catch (_) { /* keep waiting */ }
                }
                return;
            }
            if (progressKind === 'train') {
                finishTrainProgress(false, 'Failed', e.message);
            } else if (progressKind === 'accuracy') {
                if (isCancelledEval({ error: e.message }) || e.status === 401) {
                    finishAccProgress(false, 'Cancelled', 'Stopped — no results to show.');
                    setAccJobStatus('Cancelled. You can start a new run.', 'fail');
                    resetAccResults('Cancelled — no evaluation results. Start a new run when ready.');
                } else {
                    finishAccProgress(false, 'Failed', e.message);
                    setAccJobStatus(e.message, 'fail');
                }
                setAccCancelVisible(false);
                if ($('acc-run')) $('acc-run').disabled = false;
                state.accJobId = null;
            } else if (el) {
                el.textContent = e.message;
            }
            clearInterval(state.pollTimers[jobId]);
            if (onDone) onDone({ status: 'failed', error: e.message });
        }
    }, 2000);
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/* Same ambient motion as public chat */
function initCursorGlow() {
    const glow = document.getElementById('cursor-glow');
    if (!glow || window.matchMedia('(pointer: coarse)').matches) return;
    let raf = null;
    let x = 0;
    let y = 0;
    document.addEventListener('mousemove', (e) => {
        x = e.clientX;
        y = e.clientY;
        document.body.classList.add('cursor-on');
        if (raf) return;
        raf = requestAnimationFrame(() => {
            glow.style.left = `${x}px`;
            glow.style.top = `${y}px`;
            raf = null;
        });
    });
    document.addEventListener('mouseleave', () => {
        document.body.classList.remove('cursor-on');
    });
}

function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const ctx = canvas.getContext('2d');
    let w = 0;
    let h = 0;
    let particles = [];
    const resize = () => {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
        const count = Math.min(55, Math.floor((w * h) / 28000));
        particles = Array.from({ length: count }, () => ({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() * 1.8 + 0.4,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            a: Math.random() * 0.45 + 0.15,
        }));
    };
    const draw = () => {
        ctx.clearRect(0, 0, w, h);
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(125, 220, 210, ${p.a})`;
            ctx.fill();
            for (let j = i + 1; j < particles.length; j++) {
                const q = particles[j];
                const dx = p.x - q.x;
                const dy = p.y - q.y;
                const d2 = dx * dx + dy * dy;
                if (d2 < 120 * 120) {
                    const alpha = (1 - Math.sqrt(d2) / 120) * 0.18;
                    ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(q.x, q.y);
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    };
    window.addEventListener('resize', resize);
    resize();
    draw();
}

initParticles();
initCursorGlow();
setInterval(() => {
    if (state.authenticated) {
        refreshSystem();
        // Skip inbox polling during long evals — reduces load while API is busy.
        if (!state.accJobId && Date.now() > (state.quietInboxUntil || 0)) loadInbox();
    }
}, 2500);
