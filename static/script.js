let importanceChart;
let teamChart;

// --- Tab Switching ---
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const tab = item.getAttribute('data-tab');
        
        // Update Nav
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        
        // Update Content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tab).classList.add('active');

        // Contextual Actions
        if (tab === 'history') fetchHistory();
    });
});

// --- Individual Prediction ---
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        Name: document.getElementById('Name').value,
        Age: parseInt(document.getElementById('Age').value),
        Gender: document.getElementById('Gender').value,
        Department: document.getElementById('Department').value,
        Job_Title: document.getElementById('Job_Title').value,
        Experience: parseInt(document.getElementById('Experience').value),
        Promotion: document.querySelector('input[name="Promotion"]:checked').value,
        Satisfaction: parseFloat(document.getElementById('Satisfaction').value),
        Hours: parseInt(document.getElementById('Hours').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        
        if (result.error) throw new Error(result.error);

        // Show result section
        const resultSection = document.getElementById('resultSection');
        resultSection.classList.remove('hidden');
        resultSection.classList.add('animate-fadeIn');

        // Update UI Metrics
        const riskVal = (result.risk * 100).toFixed(1);
        document.getElementById('riskText').innerText = riskVal + "%";
        
        const riskBadge = document.getElementById('riskBadge');
        if (result.risk > 0.5) {
            riskBadge.innerText = "🔴 HIGH RISK";
            riskBadge.className = "status-badge mt-6 px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/30";
            document.getElementById('insightText').innerText = `Significant attrition risk detected for ${formData.Name}. Possible causes include low satisfaction or high workload. Immediate engagement is recommended.`;
        } else {
            riskBadge.innerText = "🟢 SECURE";
            riskBadge.className = "status-badge mt-6 px-4 py-2 bg-green-500/20 text-green-400 border border-green-500/30";
            document.getElementById('insightText').innerText = `${formData.Name}'s profile is highly stable with low turnover probability. Continue current engagement strategies.`;
        }

        document.getElementById('salText').innerText = "₹ " + Number(result.salary).toLocaleString();

        // Update Chart
        renderImportanceChart(result.features, result.feat_imp);

        // Scroll
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        alert("Execution Error: " + err.message);
    }
});

function renderImportanceChart(labels, data) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    if (importanceChart) importanceChart.destroy();

    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: data.map(v => v > 0.15 ? '#6366f1' : 'rgba(99, 102, 241, 0.3)'),
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, border: { display: false } },
                x: { grid: { display: false } }
            }
        }
    });
}

// --- Team Analysis ---
document.getElementById('btnBulk').addEventListener('click', async () => {
    const fileInput = document.getElementById('teamFile');
    if (!fileInput.files[0]) return alert("Please select a CSV file first.");

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        document.getElementById('btnBulk').innerText = "PROCESSING...";
        const response = await fetch('/api/bulk_predict', {
            method: 'POST',
            body: formData
        });

        const results = await response.json();
        if (results.error) throw new Error(results.error);

        document.getElementById('teamResults').classList.remove('hidden');
        
        // Calculate Summary
        const avgRisk = results.reduce((acc, curr) => acc + curr.risk, 0) / results.length;
        const highRisk = results.filter(r => r.risk > 0.6).length;
        const totalSal = results.reduce((acc, curr) => acc + curr.salary, 0);

        document.getElementById('avgStability').innerText = ((1 - avgRisk) * 100).toFixed(1) + "%";
        document.getElementById('highRiskCount').innerText = highRisk;
        document.getElementById('totalPayroll').innerText = "₹" + totalSal.toLocaleString();

        // Render Team Chart
        renderTeamChart(results);

    } catch (err) {
        alert("Batch Loading Failed: " + err.message);
    } finally {
        document.getElementById('btnBulk').innerText = "PROCESS BATCH";
    }
});

function renderTeamChart(results) {
    const ctx = document.getElementById('teamChart').getContext('2d');
    if (teamChart) teamChart.destroy();

    teamChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Employees',
                data: results.map(r => ({ x: r.salary, y: r.risk * 100 })),
                backgroundColor: 'rgba(99, 102, 241, 0.6)',
                borderColor: '#6366f1',
                pointRadius: 8,
                pointHoverRadius: 12
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Expected Salary (₹)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Leaving Risk (%)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Salary: ₹${ctx.raw.x.toLocaleString()} | Risk: ${ctx.raw.y.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

// --- History ---
async function fetchHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        const body = document.getElementById('historyBody');
        const empty = document.getElementById('historyEmpty');
        body.innerHTML = '';

        if (!data.length) {
            empty.classList.remove('hidden');
            return;
        }
        empty.classList.add('hidden');

        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.className = "hover:bg-white/5 transition-colors";
            tr.innerHTML = `
                <td class="p-4 font-bold text-white">${row.name}</td>
                <td class="p-4 text-slate-400">₹${Number(row.salary).toLocaleString()}</td>
                <td class="p-4">
                    <span class="px-2 py-1 rounded text-[10px] font-bold ${row.risk > 0.5 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}">
                        ${(row.risk * 100).toFixed(1)}%
                    </span>
                </td>
                <td class="p-4 text-slate-500 text-xs">${row.timestamp}</td>
            `;
            body.appendChild(tr);
        });
    } catch (err) {
        console.error("History fail:", err);
    }
}

document.getElementById('btnClearHistory').addEventListener('click', async () => {
    if (!confirm("Are you sure you want to wipe all prediction history?")) return;
    await fetch('/api/history', { method: 'DELETE' });
    fetchHistory();
});

// --- Simulator ---
const simInputs = ['simHrs', 'simSat', 'simBoost'];
simInputs.forEach(id => {
    document.getElementById(id).addEventListener('input', runSimulation);
});

async function runSimulation() {
    const hrs = parseInt(document.getElementById('simHrs').value);
    const sat = parseFloat(document.getElementById('simSat').value);
    const boost = parseFloat(document.getElementById('simBoost').value);

    // Update Labels
    document.getElementById('simHrsLabel').innerText = hrs;
    document.getElementById('simSatLabel').innerText = sat.toFixed(2);
    document.getElementById('simBoostLabel').innerText = "+" + boost.toFixed(2);

    // Baseline Prediction (Engineer context)
    const basePayload = {
        Age: 30, Gender: 'Male', Department: 'Engineering', Job_Title: 'Engineer',
        Experience: 5, Promotion: 'No', Satisfaction: sat, Hours: hrs, Name: 'Simulator_Base'
    };

    const optPayload = { ...basePayload, Satisfaction: Math.min(1.0, sat + boost), Name: 'Simulator_Opt' };

    const [resBase, resOpt] = await Promise.all([
        fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(basePayload) }).then(r => r.json()),
        fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(optPayload) }).then(r => r.json())
    ]);

    const r1 = resBase.risk * 100;
    const r2 = resOpt.risk * 100;
    const diff = r1 - r2;

    document.getElementById('simBefore').innerText = r1.toFixed(1) + "%";
    document.getElementById('simAfter').innerText = r2.toFixed(1) + "%";
    document.getElementById('simImpact').innerText = `-${diff.toFixed(1)}% RISK REDUCTION`;
}

// Initial Run for Simulator
runSimulation();

