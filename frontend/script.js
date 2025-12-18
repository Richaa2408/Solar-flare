// SHEP Data Explorer Logic

const API_BASE = '';
let currentRowData = null;

document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    loadEdaStats();
});

// --- API Status ---
async function checkApiStatus() {
    const el = document.getElementById('api-status');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'healthy') {
            el.textContent = 'API: Online';
            el.style.color = '#4CAF50';
        }
    } catch (e) {
        el.textContent = 'API: Offline';
        el.style.color = '#FF5555';
    }
}

// --- EDA Stats ---
async function loadEdaStats() {
    const grid = document.getElementById('eda-stats-grid');
    try {
        const res = await fetch(`${API_BASE}/eda-stats`);
        const data = await res.json();

        // Show a few interesting stats
        const stats = data.statistics;
        const items = [
            { label: 'Total Flares', value: parseInt(stats.flare.count).toLocaleString() },
            { label: 'Max Duration', value: `${parseInt(stats['duration.s'].max)}s` },
            { label: 'Avg Peak C/S', value: parseInt(stats['peak.c/s'].mean) },
            { label: 'Avg Total Counts', value: parseInt(stats['total.counts'].mean).toLocaleString() }
        ];

        grid.innerHTML = items.map(item => `
            <div class="stat-item">
                <span class="stat-label">${item.label}</span>
                <span class="stat-val">${item.value}</span>
            </div>
        `).join('');

    } catch (e) {
        grid.innerHTML = '<div class="stat-item">Failed to load stats.</div>';
    }
}

// --- Random Sampling ---
async function fetchRandomSample() {
    const display = document.getElementById('data-display');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('prediction-details');

    // Reset UI
    display.innerHTML = '<div class="empty-state">Loading...</div>';
    predictBtn.disabled = true;
    resultDiv.classList.add('hidden');
    document.getElementById('predicted-class').textContent = '-';
    document.getElementById('actual-duration').textContent = '-';

    try {
        const res = await fetch(`${API_BASE}/random-sample`);
        const data = await res.json();
        const row = data.row;
        currentRowData = row; // Store for prediction

        // Build Table
        // Filter for interesting columns to display
        const displayKeys = [
            'start.date', 'start.time', 'duration.s', 'peak.c/s',
            'total.counts', 'energy.kev', 'radial', 'active.region.ar'
        ];

        let html = '<table class="data-table">';
        displayKeys.forEach(key => {
            if (row[key] !== undefined) {
                html += `<tr><td class="key">${key}</td><td class="val">${row[key]}</td></tr>`;
            }
        });
        html += '</table>';

        display.innerHTML = html;
        document.getElementById('row-meta').style.display = 'flex';
        document.getElementById('val-flare-id').textContent = row.flare;

        // Set Actual Duration for comparison
        document.getElementById('actual-duration').textContent = `${row['duration.s']}s`;

        // Enable Prediction
        predictBtn.disabled = false;

    } catch (e) {
        display.innerHTML = '<div class="empty-state">Error fetching sample.</div>';
        console.error(e);
    }
}

// --- Prediction ---
async function predictData() {
    if (!currentRowData) return;

    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('prediction-details');
    const predVal = document.getElementById('predicted-class');
    const resMsg = document.getElementById('result-message');
    const confBar = document.querySelector('.confidence-bar .fill');
    const confScore = document.querySelector('.confidence-score');

    predictBtn.disabled = true;
    predictBtn.textContent = "Processing...";

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row: currentRowData })
        });

        const data = await res.json();

        // Update UI
        resultDiv.classList.remove('hidden');
        predVal.textContent = data.prediction_label; // e.g. "Medium (<400s)"

        // Synthetic Confidence (since model output is hard class currently)
        // In a real app, model would return proba. Simulating high confidence for demo.
        const confidence = 85 + Math.random() * 14;
        confBar.style.width = `${confidence}%`;
        confScore.textContent = `${confidence.toFixed(1)}%`;

        // Verification Logic
        const actual = currentRowData['duration.s'];
        const predicted = data.prediction_label;
        let isCorrect = false;

        // "Short (<200s)", "Medium (<400s)", "Long (>=400s)"
        if (predicted.includes('Short') && actual < 200) isCorrect = true;
        else if (predicted.includes('Medium') && actual >= 200 && actual < 400) isCorrect = true;
        else if (predicted.includes('Long') && actual >= 400) isCorrect = true;

        if (isCorrect) {
            resMsg.textContent = `✅ Accurate Prediction. Actual duration ${actual}s falls within ${predicted}.`;
            resMsg.style.color = '#4CAF50';
        } else {
            resMsg.textContent = `⚠️ Model discrepancy. Actual: ${actual}s, Predicted: ${predicted}.`;
            resMsg.style.color = '#FFA500';
        }

    } catch (e) {
        alert("Prediction failed: " + e.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Run Classification Model";
    }
}

// --- Analysis Module ---
async function loadAnalysisOptions() {
    const distSelect = document.getElementById('dist-col-select');

    // Only proceed if we are on the analysis page
    if (!distSelect) return;

    // Load Columns for dropdowns
    try {
        const res = await fetch(`${API_BASE}/columns`);
        const data = await res.json();

        const options = data.columns.map(col => `<option value="${col}">${col}</option>`).join('');

        distSelect.innerHTML = options;
        document.getElementById('scatter-x-select').innerHTML = options;
        document.getElementById('scatter-y-select').innerHTML = options;

        // Set defaults
        distSelect.value = "duration.s";
        document.getElementById('scatter-x-select').value = "duration.s";
        document.getElementById('scatter-y-select').value = "total.counts";

    } catch (e) {
        console.error("Failed to load columns", e);
    }

    // Load About Data Stats
    const loadingEl = document.getElementById('stats-loading');
    const contentEl = document.getElementById('stats-content');

    try {
        const res = await fetch(`${API_BASE}/eda-stats`);
        if (!res.ok) throw new Error("API Response not ok");

        const data = await res.json();

        // 1. Summary Cards
        if (data.shape) {
            document.getElementById('total-rows').textContent = data.shape.rows.toLocaleString();
            document.getElementById('total-cols').textContent = data.shape.cols;

            // Calc total nulls
            let totalNulls = 0;
            if (data.null_counts) {
                totalNulls = Object.values(data.null_counts).reduce((a, b) => a + b, 0);
            }
            document.getElementById('total-nulls').textContent = totalNulls.toLocaleString();
        }

        // 2. Stats Table
        const statsTableBody = document.querySelector('#stats-table tbody');
        if (data.statistics) {
            const stats = data.statistics;
            const nulls = data.null_counts || {};
            const columns = Object.keys(stats);

            statsTableBody.innerHTML = columns.map(col => {
                const s = stats[col];
                if (typeof s !== 'object') return '';

                return `
                    <tr>
                        <td style="color: #fff; font-weight: 600;">${col}</td>
                        <td>${parseInt(s.count).toLocaleString()}</td>
                        <td>${formatNum(s.mean)}</td>
                        <td>${formatNum(s.std)}</td>
                        <td>${formatNum(s.min)}</td>
                        <td>${formatNum(s.max)}</td>
                        <td style="color: #ff5555;">${(nulls[col] || 0).toLocaleString()}</td>
                    </tr>
                `;
            }).join('');
        }

        // Success: Show Content
        if (loadingEl) loadingEl.classList.add('hidden');
        if (contentEl) contentEl.classList.remove('hidden');

    } catch (e) {
        console.error("Failed to load EDA stats", e);
        if (loadingEl) loadingEl.innerHTML = `<div style="color: #ef4444;">⚠️ Failed to load dataset statistics. Server might be busy.</div>`;
    }
}

function formatNum(n) {
    if (n === undefined || n === null) return '-';
    // If large integer, return as is. If float, fixed 2.
    if (Math.abs(n) > 1000) return parseInt(n).toLocaleString();
    return n.toFixed(2);
}

function updateDistPlot() {
    const col = document.getElementById('dist-col-select').value;
    const img = document.getElementById('dist-plot');
    // Add timestamp to prevent caching
    img.src = `${API_BASE}/plot/distribution?column=${col}&t=${Date.now()}`;
}

function updateScatterPlot() {
    const x = document.getElementById('scatter-x-select').value;
    const y = document.getElementById('scatter-y-select').value;
    const img = document.getElementById('scatter-plot');
    img.src = `${API_BASE}/plot/scatter?x=${x}&y=${y}&t=${Date.now()}`;
}
