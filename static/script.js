// AURORA 6.0 — authenticated SPA client
let importanceChart;
let teamChart;
let csrfToken = null;
let currentUser = null;
let historyPage = 1;
const HISTORY_PAGE_SIZE = 20;
const SIM_DEBOUNCE_MS = 300;

// ---------- API helper ----------
async function api(path, options = {}) {
    const opts = { ...options };
    opts.headers = { ...(opts.headers || {}) };
    if (opts.body && typeof opts.body === "string") {
        opts.headers["Content-Type"] = "application/json";
    }
    if (csrfToken && opts.method && opts.method !== "GET") {
        opts.headers["X-CSRF-Token"] = csrfToken;
    }
    const res = await fetch(path, opts);
    if (res.status === 401 && !path.startsWith("/api/auth/")) {
        showLogin();
        throw new Error("Session expired. Please log in again.");
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
        const msg = (data && data.error) || `Request failed (${res.status})`;
        const err = new Error(msg);
        err.status = res.status;
        err.data = data;
        throw err;
    }
    return data;
}

// ---------- Auth ----------
async function bootAuth() {
    try {
        const me = await api("/api/auth/me");
        csrfToken = me.csrf_token;
        currentUser = me.user;
        showApp();
    } catch (_) {
        showLogin();
    }
}

function showLogin() {
    document.getElementById("loginView").classList.remove("hidden");
    document.getElementById("appView").classList.add("hidden");
}

let modelSchema = null;

async function loadModelSchema() {
    try {
        modelSchema = await api("/api/model");
        populateSelect("Department", modelSchema.categories.Department);
        populateSelect("Job_Title", modelSchema.categories.Job_Title);
        populateSelect("Gender", modelSchema.categories.Gender);
        // Hide the Hours slider when the active model does not consume it
        const hoursBlock = document.getElementById("hoursBlock");
        if (hoursBlock && !modelSchema.features.includes("Hours")) {
            hoursBlock.classList.add("hidden");
        }
        document.getElementById("modelBadge").textContent =
            `MODEL: ${modelSchema.active_version.toUpperCase()}`;
        if (modelSchema.caveats && modelSchema.caveats.length) {
            document.getElementById("modelCaveat").textContent =
                modelSchema.caveats[0];
        }
    } catch (err) {
        console.error("Model schema load failed:", err);
    }
}

function populateSelect(id, values) {
    const sel = document.getElementById(id);
    if (!sel || !values || !values.length) return;
    sel.innerHTML = "";
    values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
    });
}

function showApp() {
    document.getElementById("loginView").classList.add("hidden");
    document.getElementById("appView").classList.remove("hidden");
    document.getElementById("userBadge").textContent =
        `${currentUser.username} · ${currentUser.role}`;
    document.getElementById("btnClearHistory").classList.toggle(
        "hidden", currentUser.role !== "ADMIN");
    loadModelSchema();
    api("/api/health").then(h => {
        const ok = h.status === "ok";
        document.getElementById("statusDot").className =
            "w-2 h-2 rounded-full mr-2 " + (ok ? "bg-green-500" : "bg-amber-500");
        document.getElementById("statusText").textContent =
            ok ? "ML ENGINE ONLINE" : "DEGRADED";
        document.getElementById("statusText").className =
            "text-xs font-bold " + (ok ? "text-green-500" : "text-amber-500");
    }).catch(() => {
        document.getElementById("statusDot").className = "w-2 h-2 bg-red-500 rounded-full mr-2";
        document.getElementById("statusText").textContent = "OFFLINE";
        document.getElementById("statusText").className = "text-xs font-bold text-red-500";
    });
    runSimulation();
}

document.getElementById("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("loginError");
    errEl.classList.add("hidden");
    try {
        const data = await api("/api/auth/login", {
            method: "POST",
            body: JSON.stringify({
                username: document.getElementById("loginUsername").value,
                password: document.getElementById("loginPassword").value,
            }),
        });
        csrfToken = data.csrf_token;
        currentUser = data.user;
        showApp();
    } catch (err) {
        errEl.textContent = err.message;
        errEl.classList.remove("hidden");
    }
});

document.getElementById("btnLogout").addEventListener("click", async () => {
    try { await api("/api/auth/logout", { method: "POST" }); } catch (_) { /* ignore */ }
    csrfToken = null;
    currentUser = null;
    showLogin();
});

document.getElementById("btnChangePassword").addEventListener("click", async () => {
    const current = prompt("Current password:");
    if (current === null) return;
    const next = prompt("New password (min 8 chars):");
    if (next === null) return;
    try {
        await api("/api/auth/change-password", {
            method: "POST",
            body: JSON.stringify({ current_password: current, new_password: next }),
        });
        alert("Password updated. Other sessions were revoked.");
    } catch (err) {
        alert("Password change failed: " + err.message);
    }
});

document.getElementById("dropzone").addEventListener("click", () => {
    document.getElementById("teamFile").click();
});

// ---------- Analytics tab ----------
document.getElementById("btnAnalytics").addEventListener("click", async () => {
    const fileInput = document.getElementById("analyticsFile");
    if (!fileInput.files[0]) return alert("Please select a CSV file first.");
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    try {
        document.getElementById("btnAnalytics").innerText = "ANALYZING...";
        const data = await api("/api/analytics/upload", { method: "POST", body: fd });
        const s = data.stats;
        document.getElementById("analyticsResults").classList.remove("hidden");
        document.getElementById("anRows").textContent = s.rows;
        document.getElementById("anDepts").textContent = s.by_department ? Object.keys(s.by_department).length : 0;
        document.getElementById("anCols").textContent = s.columns;
        document.getElementById("analyticsDisclaimer").textContent = data.disclaimer || "";
        document.getElementById("analyticsDisclaimer").classList.remove("hidden");

        const body = document.getElementById("anDeptBody");
        body.innerHTML = "";
        if (s.by_department) {
            Object.entries(s.by_department).forEach(([dept, e]) => {
                const tr = document.createElement("tr");
                tr.className = "border-t border-white/5";
                const fmt = v => (v === null || v === undefined) ? "—" : Number(v).toLocaleString();
                [dept, e.count, fmt(e.avg_satisfaction), fmt(e.avg_hours), fmt(e.avg_salary),
                 e.attrition_rate === null || e.attrition_rate === undefined ? "—" : (e.attrition_rate * 100).toFixed(1) + "%"
                ].forEach((txt, i) => {
                    const td = document.createElement("td");
                    td.className = "p-3 " + (i === 0 ? "font-bold text-white" : "text-slate-400");
                    td.textContent = txt;
                    tr.appendChild(td);
                });
                body.appendChild(tr);
            });
        }

        const ins = document.getElementById("anInsights");
        ins.innerHTML = "";
        (data.insights || []).forEach(i => {
            const div = document.createElement("div");
            div.className = "glass p-4 border-l-4 border-indigo-500";
            const p = document.createElement("p");
            p.className = "text-sm text-slate-300";
            p.textContent = "💡 " + i.message;
            div.appendChild(p);
            ins.appendChild(div);
        });
    } catch (err) {
        alert("Analytics failed: " + err.message);
    } finally {
        document.getElementById("btnAnalytics").innerText = "ANALYZE";
    }
});

// ---------- Tab switching ----------
document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => {
        const tab = item.getAttribute("data-tab");
        document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
        item.classList.add("active");
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        document.getElementById(tab).classList.add("active");
        if (tab === "history") { historyPage = 1; fetchHistory(); }
    });
});

// ---------- Individual prediction ----------
document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = {
        Name: document.getElementById("Name").value,
        Age: parseInt(document.getElementById("Age").value, 10),
        Gender: document.getElementById("Gender").value,
        Department: document.getElementById("Department").value,
        Job_Title: document.getElementById("Job_Title").value,
        Experience: parseInt(document.getElementById("Experience").value, 10),
        Promotion: document.querySelector('input[name="Promotion"]:checked').value,
        Satisfaction: parseFloat(document.getElementById("Satisfaction").value),
        Hours: parseInt(document.getElementById("Hours").value, 10),
        simulate: false,
    };

    try {
        const result = await api("/api/predict", {
            method: "POST",
            body: JSON.stringify(formData),
        });

        const resultSection = document.getElementById("resultSection");
        resultSection.classList.remove("hidden");
        resultSection.classList.add("animate-fadeIn");

        const riskVal = (result.risk * 100).toFixed(1);
        document.getElementById("riskText").innerText = riskVal + "%";

        const riskBadge = document.getElementById("riskBadge");
        if (result.risk > 0.5) {
            riskBadge.innerText = "HIGH RISK";
            riskBadge.className = "status-badge mt-6 px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/30";
            document.getElementById("insightText").innerText =
                `Elevated attrition risk for ${formData.Name}. Review satisfaction and workload factors.`;
        } else {
            riskBadge.innerText = "SECURE";
            riskBadge.className = "status-badge mt-6 px-4 py-2 bg-green-500/20 text-green-400 border border-green-500/30";
            document.getElementById("insightText").innerText =
                `${formData.Name}'s profile shows low predicted turnover probability.`;
        }

        document.getElementById("salText").innerText = "₹ " + Number(result.salary).toLocaleString();
        renderImportanceChart(result.features, result.feat_imp);
        resultSection.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        alert("Prediction failed: " + err.message);
    }
});

function renderImportanceChart(labels, data) {
    const ctx = document.getElementById("importanceChart").getContext("2d");
    if (importanceChart) importanceChart.destroy();
    importanceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: data.map(v => v > 0.15 ? "#6366f1" : "rgba(99, 102, 241, 0.3)"),
                borderRadius: 10,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.05)" }, border: { display: false } },
                x: { grid: { display: false } },
            },
        },
    });
}

// ---------- Team analysis (bulk CSV) ----------
document.getElementById("btnBulk").addEventListener("click", async () => {
    const fileInput = document.getElementById("teamFile");
    if (!fileInput.files[0]) return alert("Please select a CSV file first.");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const bulkErr = document.getElementById("bulkError");
    bulkErr.classList.add("hidden");

    try {
        document.getElementById("btnBulk").innerText = "PROCESSING...";
        const data = await api("/api/bulk_predict", { method: "POST", body: formData });
        const results = data.results || [];

        document.getElementById("teamResults").classList.remove("hidden");
        const avgRisk = results.length
            ? results.reduce((a, c) => a + c.risk, 0) / results.length : 0;
        const highRisk = results.filter(r => r.risk > 0.6).length;
        const totalSal = results.reduce((a, c) => a + c.salary, 0);

        document.getElementById("avgStability").innerText = ((1 - avgRisk) * 100).toFixed(1) + "%";
        document.getElementById("highRiskCount").innerText = highRisk;
        document.getElementById("totalPayroll").innerText = "₹" + totalSal.toLocaleString();
        renderTeamChart(results);

        if (data.errors && data.errors.length) {
            bulkErr.textContent =
                `${data.errors.length} of ${data.summary.total} rows failed validation and were skipped.`;
            bulkErr.classList.remove("hidden");
        }
    } catch (err) {
        bulkErr.textContent = "Batch analysis failed: " + err.message;
        bulkErr.classList.remove("hidden");
    } finally {
        document.getElementById("btnBulk").innerText = "PROCESS BATCH";
    }
});

function renderTeamChart(results) {
    const ctx = document.getElementById("teamChart").getContext("2d");
    if (teamChart) teamChart.destroy();
    teamChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                label: "Employees",
                data: results.map(r => ({ x: r.salary, y: r.risk * 100 })),
                backgroundColor: "rgba(99, 102, 241, 0.6)",
                borderColor: "#6366f1",
                pointRadius: 8,
                pointHoverRadius: 12,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: "Expected Salary", color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: { title: { display: true, text: "Leaving Risk (%)", color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Salary: ₹${ctx.raw.x.toLocaleString()} | Risk: ${ctx.raw.y.toFixed(1)}%`,
                    },
                },
            },
        },
    });
}

// ---------- History (XSS-safe rendering + pagination) ----------
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

async function fetchHistory() {
    try {
        const data = await api(`/api/history?page=${historyPage}&page_size=${HISTORY_PAGE_SIZE}`);
        const body = document.getElementById("historyBody");
        const empty = document.getElementById("historyEmpty");
        body.innerHTML = "";

        if (!data.items.length) {
            empty.classList.remove("hidden");
        } else {
            empty.classList.add("hidden");
            const canDeleteRecord = currentUser &&
                (currentUser.role === "ADMIN" || currentUser.role === "HR_MANAGER");
            data.items.forEach(row => {
                const tr = document.createElement("tr");
                tr.className = "hover:bg-white/5 transition-colors";

                const tdName = el("td", "p-4 font-bold text-white", row.name);
                tr.appendChild(tdName);
                tr.appendChild(el("td", "p-4 text-slate-400",
                    "₹" + Number(row.salary).toLocaleString()));

                const tdRisk = document.createElement("td");
                tdRisk.className = "p-4";
                const badge = el("span",
                    "px-2 py-1 rounded text-[10px] font-bold " +
                    (row.risk > 0.5 ? "bg-red-500/20 text-red-400" : "bg-green-500/20 text-green-400"),
                    ((row.risk || 0) * 100).toFixed(1) + "%");
                tdRisk.appendChild(badge);
                if (row.kind === "simulation") {
                    tdRisk.appendChild(el("span", "ml-2 text-[10px] text-slate-500 uppercase", "sim"));
                }
                tr.appendChild(tdRisk);
                tr.appendChild(el("td", "p-4 text-slate-500 text-xs", row.timestamp || ""));

                const tdAction = document.createElement("td");
                tdAction.className = "p-4";
                if (canDeleteRecord) {
                    const btn = el("button", "text-red-400 hover:text-red-300 text-xs font-bold", "Delete");
                    btn.addEventListener("click", () => deleteHistoryRecord(row.id, btn));
                    tdAction.appendChild(btn);
                }
                tr.appendChild(tdAction);
                body.appendChild(tr);
            });
        }

        const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
        document.getElementById("historyPageInfo").textContent =
            `Page ${data.page} of ${totalPages} · ${data.total} records`;
        document.getElementById("btnPrevPage").disabled = data.page <= 1;
        document.getElementById("btnNextPage").disabled = data.page >= totalPages;
    } catch (err) {
        console.error("History load failed:", err);
    }
}

async function deleteHistoryRecord(id, btn) {
    if (!confirm("Delete this record?")) return;
    try {
        btn.disabled = true;
        await api(`/api/history/${id}`, { method: "DELETE" });
        fetchHistory();
    } catch (err) {
        alert("Delete failed: " + err.message);
        btn.disabled = false;
    }
}

document.getElementById("btnPrevPage").addEventListener("click", () => {
    if (historyPage > 1) { historyPage--; fetchHistory(); }
});
document.getElementById("btnNextPage").addEventListener("click", () => {
    historyPage++; fetchHistory();
});

document.getElementById("btnClearHistory").addEventListener("click", async () => {
    if (!confirm("Permanently delete ALL prediction history?")) return;
    try {
        await api("/api/history", { method: "DELETE" });
        historyPage = 1;
        fetchHistory();
    } catch (err) {
        alert("Clear failed: " + err.message);
    }
});

// ---------- Simulator (non-persistent, debounced) ----------
const simInputs = ["simHrs", "simSat", "simBoost"];
let simTimer = null;
simInputs.forEach(id => {
    document.getElementById(id).addEventListener("input", () => {
        clearTimeout(simTimer);
        simTimer = setTimeout(runSimulation, SIM_DEBOUNCE_MS);
    });
});

async function runSimulation() {
    const hrs = parseInt(document.getElementById("simHrs").value, 10);
    const sat = parseFloat(document.getElementById("simSat").value);
    const boost = parseFloat(document.getElementById("simBoost").value);

    document.getElementById("simHrsLabel").innerText = hrs;
    document.getElementById("simSatLabel").innerText = sat.toFixed(2);
    document.getElementById("simBoostLabel").innerText = "+" + boost.toFixed(2);

    const basePayload = {
        Age: 30, Gender: "Male", Department: "Engineering", Job_Title: "Engineer",
        Experience: 5, Promotion: "No", Satisfaction: sat, Hours: hrs,
        Name: "Simulator_Base", simulate: true,
    };
    const optPayload = {
        ...basePayload,
        Satisfaction: Math.min(1.0, sat + boost),
        Name: "Simulator_Opt", simulate: true,
    };

    try {
        const [resBase, resOpt] = await Promise.all([
            api("/api/predict", { method: "POST", body: JSON.stringify(basePayload) }),
            api("/api/predict", { method: "POST", body: JSON.stringify(optPayload) }),
        ]);
        const r1 = resBase.risk * 100;
        const r2 = resOpt.risk * 100;
        const diff = r1 - r2;
        document.getElementById("simBefore").innerText = r1.toFixed(1) + "%";
        document.getElementById("simAfter").innerText = r2.toFixed(1) + "%";
        document.getElementById("simImpact").innerText = `-${diff.toFixed(1)}% RISK REDUCTION`;
    } catch (err) {
        console.error("Simulation failed:", err);
    }
}

// ---------- Boot ----------
bootAuth();
