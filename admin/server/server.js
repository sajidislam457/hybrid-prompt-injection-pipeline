/**
 * SecureAI Admin — localhost-only research console.
 * Not linked from the public chat app. Gesture unlock creates a server session.
 */

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const cookieParser = require('cookie-parser');
const rateLimit = require('express-rate-limit');
const multer = require('multer');
const FormData = require('form-data');

const ROOT = path.resolve(__dirname, '..', '..');
const ADMIN_ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ADMIN_ROOT, 'data');
const SESSIONS_FILE = path.join(DATA_DIR, 'sessions.json');
const AUDIT_FILE = path.join(ROOT, 'logs', 'admin_audit.jsonl');
const LLM_RUNTIME = path.join(ROOT, 'configs', 'llm_runtime.json');
const ACCURACY_JOBS_DIR = path.join(ROOT, 'logs', 'admin_jobs');

function loadDotEnv(filePath) {
    if (!fs.existsSync(filePath)) return;
    const buf = fs.readFileSync(filePath);
    let text;
    // Windows editors sometimes save .env as UTF-16
    if (buf.length >= 2 && buf[0] === 0xff && buf[1] === 0xfe) {
        text = buf.toString('utf16le');
    } else if (buf.length >= 2 && buf[0] === 0xfe && buf[1] === 0xff) {
        text = buf.slice(2).swap16().toString('utf16le');
    } else if (buf.length >= 4 && buf[1] === 0x00 && buf[3] === 0x00) {
        text = buf.toString('utf16le');
    } else {
        text = buf.toString('utf8');
    }
    text = text.replace(/^\uFEFF/, '').replace(/\0/g, '');
    for (const line of text.split(/\r?\n/)) {
        const t = line.trim();
        if (!t || t.startsWith('#')) continue;
        const i = t.indexOf('=');
        if (i < 0) continue;
        const key = t.slice(0, i).trim();
        let val = t.slice(i + 1).trim();
        if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
            val = val.slice(1, -1);
        }
        if (!key) continue;
        // .env wins for app config (fixes stale empty env vars)
        if (
            key.startsWith('ADMIN_') ||
            key.startsWith('OPENROUTER_') ||
            key === 'PYTHON_API' ||
            !process.env[key]
        ) {
            process.env[key] = val;
        }
    }
}

loadDotEnv(path.join(ROOT, '.env'));

const PORT = Number(process.env.ADMIN_PORT || 3002);
const HOST = process.env.ADMIN_HOST || '127.0.0.1';
const ADMIN_ENABLED = String(process.env.ADMIN_ENABLED || 'true').toLowerCase() === 'true';
const PYTHON_API = process.env.PYTHON_API || 'http://127.0.0.1:8000';
function getAdminToken() {
    return String(process.env.ADMIN_INTERNAL_TOKEN || 'localhost-admin').trim();
}
const GESTURE_CODE = String(process.env.ADMIN_GESTURE_CODE || '').trim();
const SESSION_TTL_MS = Number(process.env.ADMIN_SESSION_TTL_MIN || 30) * 60 * 1000;
const MAX_FAILS = Number(process.env.ADMIN_MAX_FAILS || 8);
const LOCK_MS = Number(process.env.ADMIN_LOCK_MIN || 15) * 60 * 1000;

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(path.join(ROOT, 'logs'), { recursive: true });

try {
    const legacy = path.join(DATA_DIR, 'password.bcrypt');
    if (fs.existsSync(legacy)) fs.unlinkSync(legacy);
} catch (_) { /* ignore */ }

const app = express();
const upload = multer({
    storage: multer.memoryStorage(),
    // Large research dumps (jayavibhav ~100MB+). Keep in sync with axios timeout below.
    limits: { fileSize: 1024 * 1024 * 1024 }, // 1 GB
});

/** @type {Map<string, {expires:number, created:number}>} */
const sessions = new Map();
let failState = { count: 0, lockedUntil: 0 };

function audit(event, extra = {}) {
    const row = {
        timestamp: new Date().toISOString(),
        event,
        ...extra,
    };
    try {
        fs.appendFileSync(AUDIT_FILE, JSON.stringify(row) + '\n', 'utf8');
    } catch (_) { /* ignore */ }
}

function loadSessionsFromDisk() {
    try {
        if (!fs.existsSync(SESSIONS_FILE)) return;
        const raw = JSON.parse(fs.readFileSync(SESSIONS_FILE, 'utf8'));
        const now = Date.now();
        for (const [id, s] of Object.entries(raw || {})) {
            if (s.expires > now) sessions.set(id, s);
        }
    } catch (_) { /* ignore */ }
}

function persistSessions() {
    const obj = {};
    const now = Date.now();
    for (const [id, s] of sessions.entries()) {
        if (s.expires > now) obj[id] = s;
    }
    fs.writeFileSync(SESSIONS_FILE, JSON.stringify(obj, null, 2), 'utf8');
}

loadSessionsFromDisk();

function setSessionCookie(res, sid) {
    res.cookie('admin_sid', sid, {
        httpOnly: true,
        sameSite: 'strict',
        secure: false,
        maxAge: SESSION_TTL_MS,
        path: '/',
    });
}

function createSession(res) {
    const now = Date.now();
    const sid = crypto.randomBytes(32).toString('hex');
    sessions.set(sid, { created: now, expires: now + SESSION_TTL_MS, lastPersist: now });
    persistSessions();
    setSessionCookie(res, sid);
    return sid;
}

/** Keep browser cookie + server expiry alive while the Lab is in use (held-out can run for hours). */
function touchSession(res, sid, s) {
    const now = Date.now();
    s.expires = now + SESSION_TTL_MS;
    setSessionCookie(res, sid);
    // Persist sliding expiry occasionally so Admin restarts don't drop long sessions.
    if (!s.lastPersist || now - s.lastPersist > 60_000) {
        s.lastPersist = now;
        persistSessions();
    }
}

function pyHeaders() {
    return {
        'X-Admin-Token': getAdminToken(),
        'Content-Type': 'application/json',
    };
}

async function pyGet(pathname, timeoutMs = 120000) {
    return axios.get(`${PYTHON_API}${pathname}`, {
        headers: pyHeaders(),
        timeout: timeoutMs,
        validateStatus: () => true,
    });
}

async function pyPost(pathname, body) {
    return axios.post(`${PYTHON_API}${pathname}`, body, {
        headers: pyHeaders(),
        timeout: 120000,
        validateStatus: () => true,
    });
}

function readJsonFileSafe(filePath) {
    try {
        if (!fs.existsSync(filePath)) return null;
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (_) {
        return null;
    }
}

function pidAlive(pid) {
    if (!pid) return false;
    try {
        process.kill(Number(pid), 0);
        return true;
    } catch (_) {
        return false;
    }
}

/** Read accuracy job from disk so UI keeps working when Python API is busy/OOM. */
function loadAccuracyJobFromDisk(jobId) {
    const short = String(jobId || '').slice(0, 8);
    if (!short) return null;
    const preferred = path.join(ACCURACY_JOBS_DIR, `accuracy_${short}.job.json`);
    let job = readJsonFileSafe(preferred);
    if (!job) {
        try {
            const files = fs.readdirSync(ACCURACY_JOBS_DIR)
                .filter((n) => n.startsWith(`accuracy_${short}`) && n.endsWith('.job.json'));
            if (files.length) {
                files.sort();
                job = readJsonFileSafe(path.join(ACCURACY_JOBS_DIR, files[files.length - 1]));
            }
        } catch (_) {
            return null;
        }
    }
    if (!job) return null;
    const out = { ...job, disk_fallback: true };
    const progressPath = out.progress_path
        || path.join(ACCURACY_JOBS_DIR, `accuracy_${short}.progress.json`);
    const prog = readJsonFileSafe(progressPath);
    if (prog) {
        out.progress = {
            phase: prog.phase || 'scoring',
            done: Number(prog.done || 0),
            total: Number(prog.total || 0),
            pct: Number(prog.pct || 0),
            detail: prog.detail || `Scoring ${prog.done || 0}/${prog.total || '?'}`,
            ablation: prog.ablation,
            errors: prog.errors,
        };
    }
    // Load report before PID checks so a finished worker isn't marked failed.
    const outJson = out.out_json;
    if (outJson && fs.existsSync(outJson)) {
        const report = readJsonFileSafe(outJson);
        if (report) out.report = report;
    }
    if (out.status === 'running' || out.status === 'queued') {
        if (out.pid && !pidAlive(out.pid)) {
            const progDone = out.progress
                && ['done', 'finished'].includes(String(out.progress.phase || '').toLowerCase())
                && Number(out.progress.pct || 0) >= 99;
            const hasReport = Boolean(out.report && (out.report.metrics || out.report.ablations || out.report.mode));
            if (progDone || hasReport || out.exit_code === 0) {
                out.status = 'ok';
                out.error = null;
                if (out.progress) {
                    out.progress.phase = out.progress.phase || 'done';
                    out.progress.detail = out.progress.detail || 'Finished';
                    out.progress.pct = Math.max(Number(out.progress.pct || 0), 100);
                }
            } else {
                out.status = 'failed';
                out.error = out.error || 'Worker process exited unexpectedly';
            }
        }
    } else if (out.status === 'ok') {
        // Clear stale false-positive error left by a PID race.
        if (out.error && /exited unexpectedly/i.test(String(out.error))) {
            out.error = null;
        }
    }
    return out;
}

function findActiveAccuracyJobFromDisk() {
    try {
        if (!fs.existsSync(ACCURACY_JOBS_DIR)) return null;
        const files = fs.readdirSync(ACCURACY_JOBS_DIR)
            .filter((n) => n.endsWith('.job.json'))
            .map((n) => ({
                n,
                m: fs.statSync(path.join(ACCURACY_JOBS_DIR, n)).mtimeMs,
            }))
            .sort((a, b) => b.m - a.m);
        for (const f of files.slice(0, 12)) {
            const job = loadAccuracyJobFromDisk(f.n.replace(/^accuracy_/, '').replace(/\.job\.json$/, ''));
            if (!job) continue;
            if (job.kind && job.kind !== 'accuracy') continue;
            if (job.status === 'running' || job.status === 'queued') {
                if (!job.pid || pidAlive(job.pid)) return job;
            }
        }
    } catch (_) { /* ignore */ }
    return null;
}

function getSession(req) {
    const sid = req.cookies && req.cookies.admin_sid;
    if (!sid) return null;
    const s = sessions.get(sid);
    if (!s) return null;
    if (Date.now() > s.expires) {
        sessions.delete(sid);
        persistSessions();
        return null;
    }
    return { id: sid, session: s };
}

function requireAuth(req, res, next) {
    if (!ADMIN_ENABLED) {
        return res.status(503).json({ error: 'Admin disabled (ADMIN_ENABLED=false)' });
    }
    let got = getSession(req);
    // Localhost research console — no sign-in UI; mint a session on first use.
    if (!got) {
        const sid = createSession(res);
        got = { id: sid, session: sessions.get(sid) };
    }
    touchSession(res, got.id, got.session);
    req.adminSession = { id: got.id, ...got.session };
    next();
}

app.set('trust proxy', false);
app.use(cors({
    origin: [`http://127.0.0.1:${PORT}`, `http://localhost:${PORT}`],
    credentials: true,
}));
app.use(express.json({ limit: '4mb' }));
app.use(cookieParser());
app.use(express.static(path.join(ADMIN_ROOT, 'public')));
// Same visual system as public chat
app.use('/theme', express.static(path.join(ROOT, 'web', 'public'), { index: false }));

const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 30,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many unlock attempts' },
});

app.get('/api/meta', (_req, res) => {
    res.json({
        name: 'SecureAI Admin',
        enabled: ADMIN_ENABLED,
        host: HOST,
        port: PORT,
    });
});

app.get('/api/session', (req, res) => {
    const got = getSession(req);
    if (!got) {
        return res.json({ authenticated: false, expires_in_ms: 0 });
    }
    touchSession(res, got.id, got.session);
    res.json({
        authenticated: true,
        expires_in_ms: Math.max(0, got.session.expires - Date.now()),
    });
});

app.post('/api/login', loginLimiter, (req, res) => {
    // Sign-in UI was removed. Sessions are minted in requireAuth.
    return res.status(410).json({ error: 'Sign-in removed; open / and use the console.' });
});

app.post('/api/logout', (req, res) => {
    const sid = req.cookies && req.cookies.admin_sid;
    if (sid) sessions.delete(sid);
    persistSessions();
    res.clearCookie('admin_sid', { path: '/' });
    audit('logout');
    res.json({ ok: true });
});

app.post('/api/logout-all', requireAuth, (_req, res) => {
    sessions.clear();
    persistSessions();
    res.clearCookie('admin_sid', { path: '/' });
    audit('logout_all');
    res.json({ ok: true });
});

app.get('/api/system', requireAuth, async (_req, res) => {
    const evalBusy = Boolean(findActiveAccuracyJobFromDisk());
    try {
        const [pyHealth, adminSys] = await Promise.all([
            axios.get(`${PYTHON_API}/health`, { timeout: 15000, validateStatus: () => true }),
            pyGet('/admin/system', 20000),
        ]);
        let adminFail = null;
        if (adminSys.status !== 200) {
            const detail = adminSys.data && (adminSys.data.detail || adminSys.data.error || adminSys.data);
            if (adminSys.status === 401) {
                adminFail = 'Bad admin token (restart API + Admin after .env change)';
            } else if (adminSys.status === 503) {
                adminFail = 'Admin routes disabled on Python (ADMIN_INTERNAL_TOKEN missing there)';
            } else if (evalBusy) {
                adminFail = 'API busy with held-out/ablation (job still running)';
            } else {
                adminFail = `HTTP ${adminSys.status}: ${typeof detail === 'string' ? detail : 'error'}`;
            }
        }
        const pyOk = pyHealth.status === 200;
        res.json({
            python_health: pyHealth.data,
            python_status: pyHealth.status,
            admin: adminSys.status === 200 ? adminSys.data : { error: adminSys.data },
            admin_api_ok: adminSys.status === 200,
            admin_fail_reason: adminFail,
            token_configured: Boolean(getAdminToken()),
            eval_busy: evalBusy,
            api_busy: evalBusy && !pyOk,
            node: { host: HOST, port: PORT, sessions: sessions.size },
        });
    } catch (e) {
        res.json({
            python_health: null,
            python_status: 0,
            admin: { error: e.message },
            admin_api_ok: false,
            admin_fail_reason: evalBusy
                ? 'API busy with held-out/ablation (job still running)'
                : (e.message || 'Python API unreachable'),
            token_configured: Boolean(getAdminToken()),
            eval_busy: evalBusy,
            api_busy: evalBusy,
            node: { host: HOST, port: PORT, sessions: sessions.size },
        });
    }
});

app.post('/api/lab/probe', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/lab/probe', { prompt: req.body.prompt });
    if (r.status === 200) {
        audit('lab_probe', { probe_id: r.data.probe_id, attack_type: r.data.final?.attack_type });
        try {
            let model = process.env.OPENROUTER_MODEL || 'openai/gpt-4o-mini';
            if (fs.existsSync(LLM_RUNTIME)) {
                const cfg = JSON.parse(fs.readFileSync(LLM_RUNTIME, 'utf8'));
                if (cfg.model) model = cfg.model;
            }
            fs.appendFileSync(
                path.join(ROOT, 'logs', 'llm_analytics.jsonl'),
                JSON.stringify({
                    timestamp: new Date().toISOString(),
                    kind: 'probe',
                    model,
                    latency_ms: r.data.elapsed_ms,
                    attack_type: r.data.final?.attack_type,
                }) + '\n'
            );
        } catch (_) { /* ignore */ }
    }
    res.status(r.status).json(r.data);
});

app.get('/api/inbox', requireAuth, async (req, res) => {
    const q = new URLSearchParams({
        status: String(req.query.status || 'live'),
        attack_type: String(req.query.attack_type || ''),
        q: String(req.query.q || ''),
        limit: String(req.query.limit || 50),
        offset: String(req.query.offset || 0),
    }).toString();
    const r = await pyGet(`/admin/inbox?${q}`);
    res.status(r.status).json(r.data);
});

app.post('/api/inbox/manual', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/inbox/manual', req.body || {});
    if (r.status === 200) audit('inbox_manual', { attack_type: req.body && req.body.attack_type });
    res.status(r.status).json(r.data);
});

app.post('/api/lab/lookup', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/lab/lookup', req.body || {});
    res.status(r.status).json(r.data);
});

app.get('/api/inbox/:id', requireAuth, async (req, res) => {
    const r = await pyGet(`/admin/inbox/${req.params.id}`);
    res.status(r.status).json(r.data);
});

app.post('/api/inbox/:id/review', requireAuth, async (req, res) => {
    const r = await pyPost(`/admin/inbox/${req.params.id}/review`, req.body || {});
    if (r.status === 200) audit('inbox_review', { case_id: req.params.id, discard: Boolean(req.body && req.body.discard) });
    res.status(r.status).json(r.data);
});

app.get('/api/labels', requireAuth, async (_req, res) => {
    const r = await pyGet('/admin/labels?limit=300');
    res.status(r.status).json(r.data);
});

app.post('/api/labels', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/labels', req.body);
    if (r.status === 200) audit('label_save', { attack_type: req.body.attack_type });
    res.status(r.status).json(r.data);
});

app.post('/api/accuracy/run', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/accuracy/run', req.body);
    if (r.status === 200) audit('accuracy_run', { job_id: r.data.job_id, mode: req.body.mode });
    res.status(r.status).json(r.data);
});

app.get('/api/accuracy/active', requireAuth, async (_req, res) => {
    try {
        const r = await pyGet('/admin/accuracy/active', 8000);
        if (r.status === 200 && r.data) return res.status(200).json(r.data);
    } catch (_) { /* fall through to disk */ }
    const job = findActiveAccuracyJobFromDisk();
    if (job) return res.json({ active: true, job });
    return res.json({ active: false, job: null });
});

app.post('/api/accuracy/cancel', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/accuracy/cancel', req.body || {});
    if (r.status === 200) audit('accuracy_cancel', { job_id: r.data?.job?.id });
    res.status(r.status).json(r.data);
});

app.get('/api/accuracy/jobs/:id', requireAuth, async (req, res) => {
    try {
        const r = await pyGet(`/admin/accuracy/jobs/${req.params.id}`, 8000);
        if (r.status === 200 && r.data) return res.status(200).json(r.data);
        // Prefer disk if Python is overloaded / timed out / 404 during reload.
        const disk = loadAccuracyJobFromDisk(req.params.id);
        if (disk) return res.status(200).json(disk);
        return res.status(r.status).json(r.data);
    } catch (_) {
        const disk = loadAccuracyJobFromDisk(req.params.id);
        if (disk) return res.status(200).json(disk);
        return res.status(503).json({ error: 'Accuracy job unavailable (API busy)' });
    }
});

app.get('/api/accuracy/reports', requireAuth, async (_req, res) => {
    const r = await pyGet('/admin/accuracy/reports');
    res.status(r.status).json(r.data);
});

app.get('/api/accuracy/reports/:name', requireAuth, async (req, res) => {
    const r = await pyGet(`/admin/accuracy/reports/${encodeURIComponent(req.params.name)}`);
    res.status(r.status).json(r.data);
});

app.post('/api/datasets/upload', requireAuth, (req, res, next) => {
    upload.single('file')(req, res, (err) => {
        if (!err) return next();
        if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
            return res.status(413).json({
                error: 'File too large (max 1 GB). Split the file or raise ADMIN upload limit.',
            });
        }
        return res.status(400).json({ error: err.message || 'Upload failed' });
    });
}, async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'file required' });
    try {
        const form = new FormData();
        form.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype || 'application/octet-stream',
        });
        const r = await axios.post(`${PYTHON_API}/admin/datasets/upload`, form, {
            headers: { ...form.getHeaders(), 'X-Admin-Token': getAdminToken() },
            maxBodyLength: Infinity,
            maxContentLength: Infinity,
            timeout: 3600000, // 60 minutes for large dataset uploads
            validateStatus: () => true,
        });
        if (r.status === 200) audit('dataset_upload', { name: req.file.originalname });
        res.status(r.status).json(r.data);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/datasets/taxonomy', requireAuth, async (_req, res) => {
    const r = await pyGet('/admin/datasets/taxonomy');
    res.status(r.status).json(r.data);
});

app.post('/api/datasets/accept', requireAuth, async (_req, res) => {
    const r = await pyPost('/admin/datasets/accept', {});
    res.status(r.status).json(r.data);
});

app.post('/api/datasets/ingest', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/datasets/ingest', req.body || {});
    if (r.status === 200) audit('dataset_ingest', {
        filename: req.body?.filename,
        source: req.body?.source,
        dry_run: Boolean(req.body?.dry_run),
        appended: r.data?.appended,
    });
    res.status(r.status).json(r.data);
});

app.post('/api/datasets/train', requireAuth, async (req, res) => {
    const r = await pyPost('/admin/datasets/train', req.body || {});
    if (r.status === 200) audit('train_start', { job_id: r.data.job_id });
    res.status(r.status).json(r.data);
});

app.get('/api/jobs/:id', requireAuth, async (req, res) => {
    const r = await pyGet(`/admin/jobs/${req.params.id}`);
    res.status(r.status).json(r.data);
});

app.get('/api/always-on', requireAuth, (_req, res) => {
    const doc = path.join(ROOT, 'docs', 'ALWAYS_ON_API.md');
    let markdown = '';
    try {
        markdown = fs.readFileSync(doc, 'utf8');
    } catch (_) {
        markdown = 'See docs/ALWAYS_ON_API.md';
    }
    res.json({
        markdown,
        script: 'scripts/install_api_service.ps1',
        tip: 'Install Python API as a Windows service so it survives reboot.',
    });
});

app.get('*', (req, res) => {
    if (req.path.startsWith('/api/')) {
        return res.status(404).json({ error: 'Not found' });
    }
    res.sendFile(path.join(ADMIN_ROOT, 'public', 'index.html'));
});

if (!ADMIN_ENABLED) {
    console.error('[admin] ADMIN_ENABLED=false — refusing to start.');
    process.exit(1);
}

app.listen(PORT, HOST, () => {
    console.log('='.repeat(60));
    console.log('SecureAI ADMIN (research) — not public');
    console.log('='.repeat(60));
    console.log(`Bound: http://${HOST}:${PORT}`);
    console.log(`Python: ${PYTHON_API}`);
    console.log('Auth: open session (no sign-in UI)');
    console.log('='.repeat(60));
});
