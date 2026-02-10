const API_BASE = '/api/llm';

// State
let currentPreset = null;

// Elements
// Elements
const presetsList = document.getElementById('presets-list');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');
const modelSelect = document.getElementById('model-select');
const modeSelect = document.getElementById('mode-select');
const dopamineBar = document.getElementById('dopamine-bar');
const cortisolBar = document.getElementById('cortisol-bar');
const visionBtn = document.getElementById('vision-btn');

// Load Presets on Start
async function loadPresets() {
    try {
        const res = await fetch(`${API_BASE}/presets`);
        const data = await res.json();
        renderPresets(data.presets || []);
    } catch (e) {
        console.error("Failed to load presets", e);
        presetsList.innerHTML = '<div style="padding:1rem; color:var(--text-muted)">Failed to load presets.</div>';
    }
}

function renderPresets(presets) {
    presetsList.innerHTML = '';
    presets.forEach(preset => {
        const div = document.createElement('div');
        div.className = 'preset-item';
        div.innerHTML = `
            <div class="preset-title">${preset.title}</div>
            <div class="preset-desc">${preset.description}</div>
        `;
        div.onclick = () => loadPresetIntoInput(preset);
        presetsList.appendChild(div);
    });
}

function loadPresetIntoInput(preset) {
    currentPreset = preset;
    promptInput.value = preset.prompt;
    promptInput.focus();
    // Auto-resize textarea
    promptInput.style.height = 'auto';
    promptInput.style.height = promptInput.scrollHeight + 'px';
}

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    // Convert newlines to breaks for simple rendering, or use a markdown lib in real app
    // Simple naive markdown for code blocks
    let formattedText = typeof text === 'string' ? text
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/\n/g, '<br>') : JSON.stringify(text, null, 2);

    msgDiv.innerHTML = `
        <div class="avatar">${role === 'human' ? 'YOU' : 'AI'}</div>
        <div class="content">${formattedText}</div>
    `;

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="avatar">AI</div>
        <div class="content" style="display:flex; gap:4px; align-items:center; min-height:40px">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text) return;

    // UI Updates
    appendMessage('human', text);
    promptInput.value = '';
    promptInput.style.height = '60px';

    const typingId = showTypingIndicator();
    const mode = modeSelect.value;

    try {
        let res;
        let data;

        if (mode === 'chat') {
            res = await fetch(`${API_BASE}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    model: modelSelect.value
                })
            });
            data = await res.json();
            removeTypingIndicator(typingId);
            appendMessage('system', data.text);
        } else {
            // Consultation / G-Code Mode
            const type = mode === 'gcode' ? 'GENERATE_GCODE' : 'CONSULTATION';

            // Note: In a real app, we'd need auth headers here if enabled
            res = await fetch('/api/manufacturing/request', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer dev-token' // Mock token or implement login
                },
                body: JSON.stringify({
                    type: type,
                    payload: {
                        prompt: text,
                        machine_id: "VMC-01", // Default for now
                        material: "Aluminum 6061" // Default
                    }
                })
            });
            data = await res.json();
            removeTypingIndicator(typingId);

            // Handle different response structures
            console.log("Response:", data);

            if (data.status === 'success' || data.status === 'QUEUED') {
                const responseText = data.data ? JSON.stringify(data.data, null, 2) : (data.message || "Request processed.");
                appendMessage('system', `[${data.status}] ${responseText}`);
            } else {
                appendMessage('system', JSON.stringify(data, null, 2));
            }
        }

    } catch (e) {
        removeTypingIndicator(typingId);
        appendMessage('system', `Error: ${e.message}`);
    }
}

// Neuro-State Polling
async function updateNeuroState() {
    try {
        const res = await fetch('/api/health'); // This calls orchestrator.get_system_status()
        const data = await res.json();

        if (data && data.neuro_state) {
            // Update Dashboard
            const dopamine = data.neuro_state.dopamine || 50;
            const cortisol = data.neuro_state.cortisol || 20;

            dopamineBar.style.width = `${dopamine}%`;
            cortisolBar.style.width = `${cortisol}%`;
        }
    } catch (e) {
        // Silent fail for polling
    }
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);

promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

promptInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

if (visionBtn) {
    visionBtn.addEventListener('click', () => {
        alert("Vision Cortex Triggered: Scanning Frame...");
        // Future: POST to /api/vision/analyze
    });
}

// Helper for chips
window.setInput = (text) => {
    promptInput.value = text;
    promptInput.focus();
};

// Init
loadPresets();
setInterval(updateNeuroState, 2000); // Poll every 2s
updateNeuroState(); // Initial call
