// ============================================================
// STATE
// ============================================================

const state = {
    messages: [],
    blockedCount: 0,
    safeCount: 0,
    isProcessing: false,
    conversationId: null,
    isAwaitingConfirmation: false,
    pendingSuggestion: null,
    securityLog: [],
    attackStats: {},
    threatLevels: {
        critical: 0,
        high: 0,
        medium: 0,
        low: 0
    }
};

// ============================================================
// ATTACK EXPLANATIONS - 500+ PATTERNS ARE USED INTERNALLY
// The 500+ patterns are in the backend model. 
// The UI only shows the explanation, not the patterns.
// ============================================================

const attackExplanations = {
    "system_extraction": {
        title: "System Extraction Attack",
        severity: "critical",
        risk: 0.9,
        description: "The prompt attempts to extract internal system information such as system prompts, configuration, or internal rules.",
        reason: "Sharing internal system prompts or configurations is a security risk. It could expose proprietary information or system vulnerabilities.",
        safe_alternative: "Ask about AI architecture in general terms instead of specific system details."
    },
    "data_extraction": {
        title: "Data Extraction Attack",
        severity: "critical",
        risk: 0.9,
        description: "The prompt tries to extract private, sensitive, or user-specific data from the system.",
        reason: "Accessing private or sensitive data violates privacy and security policies.",
        safe_alternative: "Ask about general concepts or public information instead."
    },
    "tool_injection": {
        title: "Tool Injection Attack",
        severity: "critical",
        risk: 0.9,
        description: "The prompt attempts to execute functions, commands, or API calls through the AI.",
        reason: "Executing arbitrary functions or commands could lead to system compromise or data breaches.",
        safe_alternative: "Ask about how functions or APIs work conceptually instead of executing them."
    },
    "jailbreak": {
        title: "Jailbreak Attempt",
        severity: "critical",
        risk: 0.88,
        description: "The prompt attempts to bypass safety rules and make the AI act without restrictions.",
        reason: "Bypassing safety protocols could lead to harmful or inappropriate outputs.",
        safe_alternative: "Ask about AI capabilities within safety guidelines instead."
    },
    "story_jailbreak": {
        title: "Story-Based Jailbreak",
        severity: "high",
        risk: 0.8,
        description: "The prompt uses a fictional scenario to try to bypass safety restrictions.",
        reason: "Using stories to bypass security is a known attack pattern that undermines AI safety.",
        safe_alternative: "Ask about real AI security concepts instead."
    },
    "direct_override": {
        title: "Direct Instruction Override",
        severity: "high",
        risk: 0.8,
        description: "The prompt directly tries to override the AI's safety instructions.",
        reason: "Overriding safety instructions could make the AI ignore critical safeguards.",
        safe_alternative: "Explain what you're trying to accomplish in general terms."
    },
    "context_tampering": {
        title: "Context Tampering",
        severity: "medium",
        risk: 0.65,
        description: "The prompt attempts to manipulate the AI's memory or conversation context.",
        reason: "Tampering with context could lead to inconsistent or manipulated responses.",
        safe_alternative: "Ask your question directly without trying to change context."
    },
    "multi_turn": {
        title: "Multi-Turn Attack",
        severity: "medium",
        risk: 0.6,
        description: "The prompt is structured to gradually build up to a malicious request.",
        reason: "Step-by-step attacks can bypass single-turn detection and are harder to catch.",
        safe_alternative: "Ask your main question directly."
    },
    "obfuscation": {
        title: "Obfuscation Attack",
        severity: "medium",
        risk: 0.6,
        description: "The prompt uses encoding or hidden tricks to disguise malicious intent.",
        reason: "Obfuscation hides the real meaning and can bypass simple filters.",
        safe_alternative: "Ask your question directly without encoding."
    },
    "emotional_manipulation": {
        title: "Emotional Manipulation",
        severity: "medium",
        risk: 0.55,
        description: "The prompt uses emotional appeals to pressure the AI into bypassing safety.",
        reason: "Emotional manipulation exploits the AI's helpfulness and can lead to unsafe responses.",
        safe_alternative: "Ask your question directly without emotional appeals."
    },
    "role_impersonation": {
        title: "Role Impersonation",
        severity: "medium",
        risk: 0.55,
        description: "The prompt tries to make the AI pretend to be someone or something it's not.",
        reason: "Impersonation can lead to misleading or harmful responses.",
        safe_alternative: "Ask about the topic directly without role-playing."
    },
    "indirect_injection": {
        title: "Indirect Injection",
        severity: "high",
        risk: 0.7,
        description: "The prompt references external content to try to inject malicious instructions.",
        reason: "External content can contain hidden instructions that bypass the current prompt.",
        safe_alternative: "Ask your question directly without external references."
    }
};


// ============================================================
// DOM ELEMENTS
// ============================================================

const elements = {
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    clearBtn: document.getElementById('clear-chat'),
    newBtn: document.getElementById('new-chat'),
    statusDot: document.getElementById('api-status'),
    statusText: document.getElementById('api-status-text'),
    geminiStatus: document.getElementById('gemini-status'),
    geminiStatusText: document.getElementById('gemini-status-text'),
    blockedCount: document.getElementById('blocked-count'),
    safeCount: document.getElementById('safe-count'),
    securityIndicator: document.getElementById('security-indicator'),
    securityStatus: document.getElementById('security-status'),
    securityLogBody: document.getElementById('security-log-body'),
    totalPrompts: document.getElementById('total-prompts'),
    statBlocked: document.getElementById('stat-blocked'),
    statSafe: document.getElementById('stat-safe'),
    statBlockRate: document.getElementById('stat-block-rate'),
    attackChart: document.getElementById('attack-chart'),
    threatSummary: document.getElementById('threat-summary'),
    explainableDetails: document.getElementById('explainable-details'),
    logCount: document.getElementById('log-count'),
    logTotal: document.getElementById('log-total')
};

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    loadSecurityLog();
    updateStats();
    updateAnalytics();
    renderSecurityLog();
    initVisualEffects();
    setInterval(checkHealth, 30000);
});

// ============================================================
// HEALTH CHECK
// ============================================================

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.pipeline_loaded) {
            elements.statusDot.className = 'status-dot online';
            elements.statusText.textContent = 'Online';
        } else {
            elements.statusDot.className = 'status-dot offline';
            elements.statusText.textContent = 'Offline';
        }
        
        if (data.openrouter_status === 'available' || data.gemini_status === 'available') {
            elements.geminiStatus.className = 'status-dot online';
            elements.geminiStatusText.textContent = 'Available';
        } else if (data.openrouter_status === 'configured' || data.openrouter_status === 'ready') {
            elements.geminiStatus.className = 'status-dot online';
            elements.geminiStatusText.textContent = 'Ready';
        } else {
            elements.geminiStatus.className = 'status-dot offline';
            elements.geminiStatusText.textContent = 'Unavailable';
        }
    } catch (error) {
        elements.statusDot.className = 'status-dot offline';
        elements.statusText.textContent = 'Offline';
        elements.geminiStatus.className = 'status-dot offline';
        elements.geminiStatusText.textContent = 'Unavailable';
    }
}

// ============================================================
// EVENT LISTENERS
// ============================================================

function setupEventListeners() {
    elements.sendBtn.addEventListener('click', sendMessage);
    
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    elements.chatInput.addEventListener('input', () => {
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 120) + 'px';
    });
    
    elements.clearBtn.addEventListener('click', clearChat);
    elements.newBtn.addEventListener('click', newChat);
    
    // Tab navigation
    document.querySelectorAll('.nav-section ul li').forEach(li => {
        li.addEventListener('click', () => {
            document.querySelectorAll('.nav-section ul li').forEach(l => l.classList.remove('active'));
            li.classList.add('active');
            
            const tab = li.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            const target = document.getElementById(`${tab}-tab`);
            if (target) target.classList.add('active');
            
            if (tab === 'analytics') updateAnalytics();
            if (tab === 'security') renderSecurityLog();
        });
    });
    
    // Clear log button
    document.getElementById('clear-log')?.addEventListener('click', clearSecurityLog);
    
    // Export CSV button
    document.getElementById('export-csv')?.addEventListener('click', exportSecurityLogCSV);
}

// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage() {
    const prompt = elements.chatInput.value.trim();
    if (!prompt || state.isProcessing) return;
    
    state.isProcessing = true;
    elements.sendBtn.disabled = true;
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    
    addMessage('user', prompt);
    updateSecurityIndicator('checking', 'Analyzing...');
    elements.sendBtn.classList.add('sending', 'burst');
    setTimeout(() => elements.sendBtn.classList.remove('burst'), 600);
    startDefenseScan();
    showTypingIndicator();
    
    try {
        let response;
        
        if (state.isAwaitingConfirmation && state.conversationId) {
            response = await fetch('/api/chat-conversational', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    conversation_id: state.conversationId,
                    user_message: prompt,
                    safe_suggestion: state.pendingSuggestion || null
                })
            });
        } else {
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    history: state.messages
                })
            });
        }
        
        const data = await response.json();
        console.log('📥 Response:', data);
        hideTypingIndicator();
        
        if (data.type === 'blocked') {
            const attackType = data.attack_type || 'unknown';
            const riskScore = data.risk_score || 0.8;
            const responseText = data.response || data.clarifying_question || 'I noticed your prompt was flagged. What is the actual question you want help with?';
            
            // Log to security log
            addSecurityLog({
                timestamp: new Date().toISOString(),
                prompt: prompt,
                attack_type: data.attack_display_name || attackType,
                risk_score: riskScore,
                action: 'BLOCKED'
            });
            
            // Update attack stats with canonical type key
            const typeKey = attackType || 'unknown';
            state.attackStats[typeKey] = (state.attackStats[typeKey] || 0) + 1;
            
            // Update threat levels
            const explanation = attackExplanations[attackType];
            if (explanation) {
                const severity = explanation.severity;
                if (severity === 'critical') state.threatLevels.critical++;
                else if (severity === 'high') state.threatLevels.high++;
                else if (severity === 'medium') state.threatLevels.medium++;
                else state.threatLevels.low++;
            }
            
            state.isAwaitingConfirmation = true;
            state.conversationId = data.conversation_id || null;
            const realSuggestion = (data.suggestion || '').trim();
            state.pendingSuggestion = realSuggestion || state.pendingSuggestion || null;
            
            let chatText = (responseText || '').trim();
            if (!chatText) {
                chatText = realSuggestion
                    || 'That request looks unsafe. Tell me what you meant in plain words.';
            }
            addMessage('bot', chatText, 'block');
            
            state.blockedCount++;
            updateStats();
            updateAnalytics();
            renderSecurityLog();
            saveSecurityLog();
            flashScreen('bad');
            
        } else if (data.type === 'success' || data.type === 'safe') {
            if (data.response) {
                addMessage('bot', String(data.response).trim(), 'safe');
            }
            
            state.safeCount++;
            state.isAwaitingConfirmation = false;
            state.conversationId = null;
            updateStats();
            updateAnalytics();
            flashScreen('ok');
            
        } else if (data.error) {
            addMessage('bot', `❌ ${data.error}`);
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
        addMessage('bot', `❌ Connection error: ${error.message}`);
    } finally {
        hideTypingIndicator();
        stopDefenseScan();
        elements.sendBtn.classList.remove('sending');
        state.isProcessing = false;
        elements.sendBtn.disabled = false;
        elements.chatInput.focus();
        updateSecurityIndicator('ready', 'Ready');
    }
}

// ============================================================
// SECURITY LOG FUNCTIONS
// ============================================================

function addSecurityLog(entry) {
    state.securityLog.unshift(entry);
    if (state.securityLog.length > 100) {
        state.securityLog.pop();
    }
    saveSecurityLog();
}

function loadSecurityLog() {
    const saved = localStorage.getItem('securityLog');
    if (saved) {
        try {
            state.securityLog = JSON.parse(saved);
        } catch (e) {
            state.securityLog = [];
        }
    }
}

function saveSecurityLog() {
    localStorage.setItem('securityLog', JSON.stringify(state.securityLog));
}

function clearSecurityLog() {
    if (state.securityLog.length === 0) return;
    if (!confirm('Clear all security log entries?')) return;
    state.securityLog = [];
    state.attackStats = {};
    state.threatLevels = { critical: 0, high: 0, medium: 0, low: 0 };
    renderSecurityLog();
    updateAnalytics();
    saveSecurityLog();
}

function renderSecurityLog() {
    const body = elements.securityLogBody;
    if (!body) return;
    
    if (state.securityLog.length === 0) {
        body.innerHTML = `<tr><td colspan="5" class="empty-state">No security events logged yet</td></tr>`;
        if (elements.logTotal) elements.logTotal.textContent = 'Total: 0 events';
        if (elements.logCount) elements.logCount.textContent = '0';
        return;
    }
    
    body.innerHTML = state.securityLog.slice(0, 50).map(entry => {
        const explanation = attackExplanations[entry.attack_type];
        const riskDisplay = explanation ? (explanation.risk * 100).toFixed(0) + '%' : (entry.risk_score * 100).toFixed(1) + '%';
        
        return `
            <tr>
                <td>${new Date(entry.timestamp).toLocaleString()}</td>
                <td class="prompt-cell" title="${entry.prompt}">${entry.prompt.substring(0, 50)}${entry.prompt.length > 50 ? '...' : ''}</td>
                <td><span class="attack-tag">${entry.attack_type}</span></td>
                <td>${riskDisplay}</td>
                <td><span class="action-tag ${entry.action === 'BLOCKED' ? 'blocked' : 'allowed'}">${entry.action}</span></td>
            </tr>
        `;
    }).join('');
    
    if (elements.logTotal) elements.logTotal.textContent = `Total: ${state.securityLog.length} events`;
    if (elements.logCount) elements.logCount.textContent = state.securityLog.length;
}

// ============================================================
// EXPORT TO CSV - FIXED (uses actual security log data)
// ============================================================

function exportSecurityLogCSV() {
    if (state.securityLog.length === 0) {
        alert('No security log entries to export!');
        return;
    }
    
    // Headers
    const headers = ['Time', 'Prompt', 'Attack Type', 'Risk Score', 'Action'];
    
    // Data rows - use actual security log data
    const rows = state.securityLog.map(entry => {
        const explanation = attackExplanations[entry.attack_type];
        const riskDisplay = explanation ? (explanation.risk * 100).toFixed(0) + '%' : (entry.risk_score * 100).toFixed(1) + '%';
        
        return [
            new Date(entry.timestamp).toLocaleString(),
            entry.prompt.replace(/,/g, ' ').replace(/"/g, '""'),
            entry.attack_type,
            riskDisplay,
            entry.action
        ];
    });
    
    // Build CSV
    let csv = '\uFEFF'; // UTF-8 BOM for Excel compatibility
    csv += headers.join(',') + '\n';
    rows.forEach(row => {
        csv += row.join(',') + '\n';
    });
    
    // Download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `security_log_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================================
// ANALYTICS FUNCTIONS - IMPROVED UI
// ============================================================

function updateAnalytics() {
    const total = state.blockedCount + state.safeCount;
    const blocked = state.blockedCount;
    const safe = state.safeCount;
    const blockRate = total > 0 ? (blocked / total * 100) : 0;
    
    // Update stats cards
    animateCount(elements.totalPrompts, total);
    animateCount(elements.statBlocked, blocked);
    animateCount(elements.statSafe, safe);
    if (elements.statBlockRate) {
        const start = Number(String(elements.statBlockRate.textContent).replace(/[^\d.]/g, '')) || 0;
        const end = Number(blockRate.toFixed(1));
        const duration = 450;
        const t0 = performance.now();
        const tick = (now) => {
            const p = Math.min(1, (now - t0) / duration);
            const eased = 1 - Math.pow(1 - p, 3);
            const current = start + (end - start) * eased;
            elements.statBlockRate.textContent = current.toFixed(1) + '%';
            if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }
    
    renderThreatSummary();
    renderAttackChart();
    renderExplainableDetails();
}

function renderThreatSummary() {
    const container = elements.threatSummary;
    if (!container) return;
    
    const { critical, high, medium, low } = state.threatLevels;
    const total = critical + high + medium + low;
    
    container.innerHTML = `
        <h3>Threat level breakdown</h3>
        <div class="threat-grid">
            <div class="threat-card critical">
                <div class="threat-bar"></div>
                <div class="threat-radar"></div>
                <div class="threat-info">
                    <span class="threat-label">Critical</span>
                    <span class="threat-count">${critical}</span>
                    <span class="threat-pct">${total > 0 ? (critical/total*100).toFixed(0) : 0}%</span>
                </div>
            </div>
            <div class="threat-card high">
                <div class="threat-bar"></div>
                <div class="threat-radar"></div>
                <div class="threat-info">
                    <span class="threat-label">High</span>
                    <span class="threat-count">${high}</span>
                    <span class="threat-pct">${total > 0 ? (high/total*100).toFixed(0) : 0}%</span>
                </div>
            </div>
            <div class="threat-card medium">
                <div class="threat-bar"></div>
                <div class="threat-radar"></div>
                <div class="threat-info">
                    <span class="threat-label">Medium</span>
                    <span class="threat-count">${medium}</span>
                    <span class="threat-pct">${total > 0 ? (medium/total*100).toFixed(0) : 0}%</span>
                </div>
            </div>
            <div class="threat-card low">
                <div class="threat-bar"></div>
                <div class="threat-radar"></div>
                <div class="threat-info">
                    <span class="threat-label">Low</span>
                    <span class="threat-count">${low}</span>
                    <span class="threat-pct">${total > 0 ? (low/total*100).toFixed(0) : 0}%</span>
                </div>
            </div>
        </div>
    `;
}

function renderAttackChart() {
    const chart = elements.attackChart;
    if (!chart) return;
    
    const types = Object.keys(state.attackStats);
    if (types.length === 0) {
        chart.innerHTML = '<p class="empty-state">No attack data yet</p>';
        return;
    }
    
    const maxCount = Math.max(...Object.values(state.attackStats));
    
    chart.innerHTML = types.sort((a, b) => state.attackStats[b] - state.attackStats[a]).map(type => {
        const count = state.attackStats[type];
        const percentage = maxCount > 0 ? (count / maxCount * 100) : 0;
        const label = type.replace(/_/g, ' ');
        return `
            <div class="attack-bar-row">
                <span class="chart-label">${escapeHtml(label)}</span>
                <div class="attack-bar-track">
                    <div class="attack-bar-fill" style="width: ${percentage}%;"></div>
                </div>
                <span class="chart-value">${count}</span>
            </div>
        `;
    }).join('');
}

function renderExplainableDetails() {
    const container = elements.explainableDetails;
    if (!container) return;
    
    const types = Object.keys(state.attackStats);
    if (types.length === 0) {
        container.innerHTML = '<p class="empty-state">No attack data yet</p>';
        return;
    }
    
    container.innerHTML = types.sort((a, b) => state.attackStats[b] - state.attackStats[a]).map(type => {
        const count = state.attackStats[type];
        const explanation = attackExplanations[type];
        
        if (!explanation) {
            return `
                <div class="explain-card">
                    <h4>${type}</h4>
                    <p>No explanation available for this attack type.</p>
                </div>
            `;
        }
        
        const severityColors = {
            critical: '#fb7185',
            high: '#fb923c',
            medium: '#fbbf24',
            low: '#34d399'
        };
        const color = severityColors[explanation.severity] || '#888';
        
        return `
            <div class="explain-card" style="border-left: 4px solid ${color};">
                <div class="explain-header">
                    <h4>${explanation.title}</h4>
                    <span class="severity-badge ${explanation.severity}">${explanation.severity.toUpperCase()}</span>
                    <span class="explain-count">${count} detection${count > 1 ? 's' : ''}</span>
                </div>
                <div class="explain-body">
                    <div class="explain-section">
                        <strong>Description</strong>
                        <p>${explanation.description}</p>
                    </div>
                    <div class="explain-section">
                        <strong>Reason</strong>
                        <p>${explanation.reason}</p>
                    </div>
                    <div class="explain-section">
                        <strong>Safe alternative</strong>
                        <p class="safe-alt">${explanation.safe_alternative}</p>
                    </div>
                    <div class="explain-section">
                        <strong>Risk score</strong>
                        <div class="risk-bar">
                            <div class="risk-fill" style="width: ${explanation.risk * 100}%; background: ${color};"></div>
                            <span>${(explanation.risk * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================
// INTENT-PRESERVING REWRITE CARD (Phase 1)
// ============================================================

function addIntentRewriteCard(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot enter';

    const avatar = document.createElement('div');
    avatar.className = 'avatar bot-av';
    avatar.innerHTML = '<span class="material-icons">shield</span>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content intent-rewrite-card';

    const needsClarify = !!data.needs_clarification;
    const intent = data.legitimate_intent || '';
    const suggestion = data.suggestion || '';
    const alternatives = Array.isArray(data.alternatives) ? data.alternatives.filter(Boolean) : [];
    const risks = Array.isArray(data.removed_risks) ? data.removed_risks : [];
    const fidelity = typeof data.fidelity_score === 'number' ? data.fidelity_score : null;
    const decisionSource = data.decision_source || '';
    const retrievalHit = data.retrieval && data.retrieval.hit;
    const layer2b = data.layer2b || {};
    const normSteps = (data.normalization && data.normalization.steps) || [];

    let html = '<div class="rewrite-panel">';
    html += `<div class="rewrite-title">${needsClarify ? 'Clarification needed' : 'Safer alternative (intent-preserving)'}</div>`;

    if (decisionSource) {
        html += `<div class="rewrite-row"><strong>Decision source:</strong> ${escapeHtml(decisionSource)}</div>`;
    }
    if (layer2b.backend) {
        html += `<div class="rewrite-row"><strong>Semantic detector:</strong> ${escapeHtml(layer2b.backend)}${layer2b.risk != null ? ` (risk ${(Number(layer2b.risk) * 100).toFixed(0)}%)` : ''}</div>`;
    }
    if (retrievalHit) {
        html += `<div class="rewrite-row"><strong>Attack retrieval:</strong> hit ${(Number(data.retrieval.score || 0) * 100).toFixed(0)}%</div>`;
    }
    if (normSteps.length) {
        html += `<div class="rewrite-row"><strong>Normalized:</strong> ${escapeHtml(normSteps.join(', '))}</div>`;
    }
    if (intent) {
        html += `<div class="rewrite-row"><strong>Detected intent:</strong> ${escapeHtml(intent)}</div>`;
    }
    if (risks.length) {
        html += `<div class="rewrite-row"><strong>Removed risks:</strong> ${escapeHtml(risks.join(', '))}</div>`;
    }
    if (fidelity !== null && !needsClarify) {
        html += `<div class="rewrite-row"><strong>Intent fidelity:</strong> ${(fidelity * 100).toFixed(0)}%</div>`;
    }
    if (suggestion && !needsClarify) {
        html += `<div class="rewrite-suggestion"><strong>Suggested prompt:</strong><p>${escapeHtml(suggestion)}</p></div>`;
    }
    if (needsClarify && data.clarifying_question) {
        html += `<div class="rewrite-row"><strong>Please clarify:</strong> ${escapeHtml(data.clarifying_question)}</div>`;
    }
    if (alternatives.length > 1) {
        html += '<div class="rewrite-alts"><strong>Alternatives:</strong><ul>';
        alternatives.slice(0, 3).forEach((alt) => {
            html += `<li>${escapeHtml(alt)}</li>`;
        });
        html += '</ul></div>';
    }
    if (!needsClarify && suggestion) {
        html += '<div class="rewrite-hint">Reply <em>yes</em> to use it, or describe what you meant.</div>';
    } else {
        html += '<div class="rewrite-hint">Reply with the actual task you want help with.</div>';
    }
    html += '</div>';

    contentDiv.innerHTML = html;
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function escapeHtml(text) {
    return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(role, content, flash) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} enter`;
    if (flash === 'safe') messageDiv.classList.add('flash-safe');
    if (flash === 'block') messageDiv.classList.add('flash-block');

    const avatar = document.createElement('div');
    avatar.className = `avatar ${role === 'user' ? 'user-av' : 'bot-av'}`;
    avatar.innerHTML = role === 'user'
        ? '<span class="material-icons">person</span>'
        : '<span class="av-ring"></span><span class="material-icons">smart_toy</span>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';

    // Plain text only — no HTML / <br>; CSS handles wrapping
    let cleanContent = typeof content === 'string' ? content : String(content || '');
    cleanContent = cleanContent.replace(/\*\*/g, '').replace(/\*/g, '').trim();
    cleanContent = cleanContent.replace(/\n{2,}/g, '\n').trim();
    contentDiv.textContent = cleanContent;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    state.messages.push({ role, content: cleanContent });
}

function showTypingIndicator() {
    hideTypingIndicator();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot enter typing';
    messageDiv.id = 'typing-indicator';
    messageDiv.innerHTML = `
        <div class="avatar bot-av">
            <span class="av-ring"></span>
            <span class="material-icons">smart_toy</span>
        </div>
        <div class="content">
            <span class="typing-label">Scanning</span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator')?.remove();
}

// ============================================================
// VISUAL EFFECTS
// ============================================================

function startDefenseScan() {
    document.body.classList.add('scanning');
}

function stopDefenseScan() {
    document.body.classList.remove('scanning');
}

function flashScreen(kind) {
    const cls = kind === 'bad' ? 'flash-bad' : 'flash-ok';
    document.body.classList.remove('flash-ok', 'flash-bad');
    // force reflow so animation retriggers
    void document.body.offsetWidth;
    document.body.classList.add(cls);
    setTimeout(() => document.body.classList.remove(cls), 600);
}

function animateCount(el, value, suffix = '') {
    if (!el) return;
    const end = Number(value) || 0;
    const start = Number(String(el.textContent).replace(/[^\d.]/g, '')) || 0;
    if (start === end) {
        el.textContent = suffix ? `${end}${suffix}` : String(end);
        return;
    }
    const duration = 450;
    const t0 = performance.now();
    const tick = (now) => {
        const p = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        const current = Math.round(start + (end - start) * eased);
        el.textContent = suffix ? `${current}${suffix}` : String(current);
        if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

function initVisualEffects() {
    initParticles();
    initCursorGlow();
}

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
            a: Math.random() * 0.45 + 0.15
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

// ============================================================
// OTHER FUNCTIONS
// ============================================================

function updateSecurityIndicator(status, text) {
    elements.securityIndicator.className = `security-${status}`;
    elements.securityStatus.textContent = text;
    const icon = elements.securityIndicator.querySelector('.material-icons');
    if (icon) {
        icon.textContent = status === 'checking' ? 'radar'
            : status === 'blocked' ? 'gpp_bad'
            : 'shield';
    }
}

function updateStats() {
    animateCount(elements.blockedCount, state.blockedCount);
    animateCount(elements.safeCount, state.safeCount);
}

function clearChat() {
    if (state.messages.length === 0) return;
    if (!confirm('Clear all messages?')) return;

    state.messages = [];
    state.isAwaitingConfirmation = false;
    state.pendingSuggestion = null;
    hideTypingIndicator();
    stopDefenseScan();

    elements.chatMessages.innerHTML = `
        <div class="message bot enter">
            <div class="avatar bot-av">
                <span class="av-ring"></span>
                <span class="material-icons">smart_toy</span>
            </div>
            <div class="content">Chat cleared. Start a new conversation!</div>
        </div>
    `;
}

function newChat() {
    clearChat();
}