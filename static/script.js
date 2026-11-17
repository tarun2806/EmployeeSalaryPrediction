// AURORA 6.0 — Phase 3 frontend
let importanceChart, anSalaryChart, teamChartUnused;
let csrfToken = null, currentUser = null, modelSchema = null;
let historyPage = 1, historySearchTerm = "", allHistoryItems = [];
const HISTORY_PAGE_SIZE = 20, SIM_DEBOUNCE_MS = 300;

const PAGE_META = {
    dashboard: ["Dashboard", "Workforce intelligence overview"],
    predict: ["Salary Prediction", "Salary & attrition estimates from the active model"],
    analytics: ["Analytics", "Descriptive statistics on uploaded workforce data"],
    history: ["History", "Stored model outputs — not verified outcomes"],
    simulator: ["Simulator", "Hypothetical scenario estimates"],
    settings: ["Account", "Credentials and session"],
};

async function api(path, options = {}) {
    const opts = { ...options };
    opts.headers = { ...(opts.headers || {}) };
    if (opts.body && typeof opts.body === "string") opts.headers["Content-Type"] = "application/json";
    if (csrfToken && opts.method && opts.method !== "GET") opts.headers["X-CSRF-Token"] = csrfToken;
    const res = await fetch(path, opts);
    if (res.status === 401 && !path.startsWith("/api/auth/")) { showLogin(); throw new Error("Session expired."); }
    let data = null;
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) {
        const err = new Error((data && data.error) || `Request failed (${res.status})`);
        err.status = res.status; err.data = data; throw err;
    }
    return data;
}

// ---------- Auth ----------
async function bootAuth() {
    try {
        const me = await api("/api/auth/me");
        csrfToken = me.csrf_token; currentUser = me.user;
        showApp();
    } catch (_) { showLogin(); }
}
function showLogin() {
    document.getElementById("loginView").classList.remove("hidden");
    document.getElementById("appView").classList.add("hidden");
}
function showApp() {
    document.getElementById("loginView").classList.add("hidden");
    document.getElementById("appView").classList.remove("hidden");
    document.getElementById("userBadge").textContent = `${currentUser.username} · ${currentUser.role}`;
    document.getElementById("setUser").textContent = currentUser.username;
    document.getElementById("setRole").textContent = currentUser.role;
    document.getElementById("btnClearHistory").classList.toggle("hidden", currentUser.role !== "ADMIN");
    loadModelSchema(); loadDashboard(); checkHealth();
    runSimulation();
}

document.getElementById("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("loginError");
    errEl.classList.add("hidden");
    try {
        const data = await api("/api/auth/login", { method: "POST", body: JSON.stringify({
            username: document.getElementById("loginUsername").value,
            password: document.getElementById("loginPassword").value }) });
        csrfToken = data.csrf_token; currentUser = data.user;
        showApp();
    } catch (err) { errEl.textContent = err.message; errEl.classList.remove("hidden"); }
});

document.getElementById("btnLogout").addEventListener("click", async () => {
    try { await api("/api/auth/logout", { method: "POST" }); } catch (_) {}
    csrfToken = null; currentUser = null; showLogin();
});

document.getElementById("btnChangePw").addEventListener("click", async () => {
    const msg = document.getElementById("pwMsg");
    msg.classList.add("hidden");
    try {
        await api("/api/auth/change-password", { method: "POST", body: JSON.stringify({
            current_password: document.getElementById("pwCurrent").value,
            new_password: document.getElementById("pwNew").value }) });
        msg.textContent = "Password updated. Other sessions revoked.";
        msg.style.color = "var(--green)"; msg.classList.remove("hidden");
        document.getElementById("pwCurrent").value = "";
        document.getElementById("pwNew").value = "";
    } catch (err) {
        msg.textContent = err.message; msg.style.color = "var(--red)"; msg.classList.remove("hidden");
    }
});

// ---------- Navigation ----------
document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => switchTab(item.getAttribute("data-tab")));
});
document.querySelectorAll("[data-goto]").forEach(b => {
    b.addEventListener("click", () => switchTab(b.getAttribute("data-goto")));
});
function switchTab(tab) {
    document.querySelectorAll(".nav-item").forEach(i => i.classList.toggle("active", i.getAttribute("data-tab") === tab));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    document.getElementById(tab).classList.add("active");
    const [t, s] = PAGE_META[tab] || [tab, ""];
    document.getElementById("pageTitle").textContent = t;
    document.getElementById("pageSub").textContent = s;
    if (tab === "history") { historyPage = 1; fetchHistory(); }
    if (tab === "dashboard") loadDashboard();
}

// ---------- Model schema ----------
async function loadModelSchema() {
    try {
        modelSchema = await api("/api/model");
        populateSelect("Department", modelSchema.categories.Department);
        populateSelect("Job_Title", modelSchema.categories.Job_Title);
        populateSelect("Gender", modelSchema.categories.Gender);
        if (modelSchema.features && !modelSchema.features.includes("Hours")) {
            document.getElementById("hoursBlock").classList.add("hidden");
        }
        document.getElementById("modelBadge").textContent = modelSchema.active_version.toUpperCase();
        document.getElementById("modelCaveat").textContent = (modelSchema.caveats || [""])[0];
        document.getElementById("dashModel").textContent = modelSchema.active_version;
        document.getElementById("dashDataset").textContent = modelSchema.dataset || "—";
        document.getElementById("dashCaveat").textContent = (modelSchema.caveats || []).join(" ");
        document.getElementById("setModel").textContent = modelSchema.active_version;
        document.getElementById("setDataset").textContent = modelSchema.dataset || "—";
        document.getElementById("dSalaryUnit").textContent =
            modelSchema.salary_unit === "monthly_income_dataset_currency" ? "USD / month (dataset)" : "₹ / month (est.)";
        document.getElementById("salUnit").textContent =
            modelSchema.salary_unit === "monthly_income_dataset_currency" ? "USD / month · dataset currency" : "₹ / month · estimate";
    } catch (err) { console.error("schema:", err); }
}
function populateSelect(id, values) {
    const sel = document.getElementById(id);
    if (!sel || !values || !values.length) return;
    sel.innerHTML = "";
    values.forEach(v => { const o = document.createElement("option"); o.value = o.textContent = v; sel.appendChild(o); });
}

// ---------- Health ----------
async function checkHealth() {
    try {
        const h = await api("/api/health");
        const ok = h.status === "ok";
        document.getElementById("statusDot").style.background = ok ? "var(--green)" : "var(--amber)";
        document.getElementById("statusText").textContent = ok ? "Engine online" : "Degraded";
    } catch (_) {
        document.getElementById("statusDot").style.background = "var(--red)";
        document.getElementById("statusText").textContent = "Offline";
    }
}

// ---------- Dashboard ----------
async function loadDashboard() {
    try {
        const s = await api("/api/analytics/summary");
        if (!s.total_predictions) {
            ["dTotal","dRisk","dSalary","dHigh"].forEach(id => document.getElementById(id).textContent = "0");
            document.getElementById("dashRecentBody").innerHTML = "";
            document.getElementById("dashEmpty").classList.remove("hidden");
            return;
        }
        document.getElementById("dashEmpty").classList.add("hidden");
        document.getElementById("dTotal").textContent = s.total_predictions;
        document.getElementById("dRisk").textContent = (s.avg_risk_score * 100).toFixed(1) + "%";
        document.getElementById("dRiskBar").style.width = Math.min(100, s.avg_risk_score * 100) + "%";
        document.getElementById("dRiskBar").style.background = s.avg_risk_score > 0.5 ? "var(--red)" : "var(--accent)";
        document.getElementById("dSalary").textContent = Math.round(s.avg_estimated_salary).toLocaleString();
        document.getElementById("dHigh").textContent = s.high_risk_count;

        const h = await api("/api/history?page=1&page_size=5");
        const body = document.getElementById("dashRecentBody");
        body.innerHTML = "";
        h.items.forEach(r => {
            const tr = document.createElement("tr");
            const unit = (modelSchema && modelSchema.salary_unit === "monthly_income_dataset_currency") ? "" : "₹";
            [r.name, unit + Number(r.salary).toLocaleString(undefined,{maximumFractionDigits:0}),
             ((r.risk||0)*100).toFixed(0)+"%", (r.timestamp||"").slice(0,10)].forEach((v,i)=>{
                const td=document.createElement("td");
                if(i===0) td.className="font-semibold";
                if(i===2){ const b=document.createElement("span"); b.className="badge "+(r.risk>0.5?"badge-red":"badge-green"); b.textContent=v; td.appendChild(b); return; }
                td.style.color = i===0 ? "var(--text)" : "var(--text-dim)";
                td.textContent=v; tr.appendChild(td);
            });
            body.appendChild(tr);
        });
    } catch (err) { console.error("dashboard:", err); }
}

// ---------- Predict ----------
document.getElementById("Satisfaction").addEventListener("input", e => {
    document.getElementById("satValue").textContent = parseFloat(e.target.value).toFixed(2);
});

document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("predictError");
    errEl.classList.add("hidden");
    const btn = document.getElementById("btnPredict");
    btn.disabled = true;
    document.getElementById("btnPredictLabel").textContent = "Running…";
    const payload = {
        Name: document.getElementById("Name").value || "Anonymous",
        Age: parseInt(document.getElementById("Age").value, 10),
        Gender: document.getElementById("Gender").value,
        Department: document.getElementById("Department").value,
        Job_Title: document.getElementById("Job_Title").value,
        Experience: parseInt(document.getElementById("Experience").value, 10),
        Hours: parseInt(document.getElementById("Hours").value || "160", 10),
        Promotion: document.querySelector('input[name="Promotion"]:checked').value,
        Satisfaction: parseFloat(document.getElementById("Satisfaction").value),
        simulate: false,
    };
    try {
        const result = await api("/api/predict", { method: "POST", body: JSON.stringify(payload) });
        document.getElementById("resultEmpty").classList.add("hidden");
        const card = document.getElementById("resultCard");
        card.classList.remove("hidden");
        card.classList.add("active");

        const unit = result.salary_unit === "monthly_income_dataset_currency" ? "" : "₹";
        document.getElementById("salText").textContent = unit + Number(result.salary).toLocaleString(undefined,{maximumFractionDigits:0});
        document.getElementById("riskText").textContent = (result.risk*100).toFixed(1)+"%";
        const fill = document.getElementById("riskBarFill");
        fill.style.width = (result.risk*100)+"%";
        fill.style.background = result.risk > 0.5 ? "var(--red)" : result.risk > 0.3 ? "var(--amber)" : "var(--green)";
        const badge = document.getElementById("riskBadge");
        badge.className = "badge " + (result.risk > 0.5 ? "badge-red" : result.risk > 0.3 ? "badge-amber" : "badge-green");
        badge.textContent = result.risk > 0.5 ? "HIGH RISK" : result.risk > 0.3 ? "ELEVATED" : "LOW RISK";
        document.getElementById("resultModel").textContent = "model: " + (result.model_version||"");

        // SHAP top drivers
        const shap = document.getElementById("shapList");
        shap.innerHTML = "";
        (result.explanation || []).slice(0,4).forEach(d => {
            const row = document.createElement("div");
            row.className = "flex items-center justify-between";
            const name = document.createElement("span"); name.style.color="var(--text-dim)"; name.textContent=d.feature;
            const val = document.createElement("span");
            val.className="mono font-semibold";
            val.style.color = d.impact >= 0 ? "var(--red)" : "var(--green)";
            val.textContent = (d.impact>=0?"+":"") + d.impact.toFixed(3);
            row.appendChild(name); row.appendChild(val); shap.appendChild(row);
        });
        if (!result.explanation || !result.explanation.length) {
            const p = document.createElement("p");
            p.className = "text-xs"; p.style.color = "#5a6472";
            p.textContent = "No explanation available.";
            shap.appendChild(p);
        }

        renderImportanceChart(result.features, result.feat_imp);
    } catch (err) {
        errEl.textContent = err.message; errEl.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        document.getElementById("btnPredictLabel").textContent = "Run prediction";
    }
});

function renderImportanceChart(labels, data) {
    const ctx = document.getElementById("importanceChart").getContext("2d");
    if (importanceChart) importanceChart.destroy();
    importanceChart = new Chart(ctx, {
        type: "bar",
        data: { labels, datasets: [{ data,
            backgroundColor: data.map(v => v > 0.15 ? "rgba(91,140,255,.85)" : "rgba(91,140,255,.25)"),
            borderRadius: 6, borderSkipped: false }] },
        options: { responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: "rgba(255,255,255,.05)" }, border: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } },
                x: { grid: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } } } },
    });
}

// ---------- Analytics ----------
document.getElementById("btnAnalytics").addEventListener("click", async () => {
    const fi = document.getElementById("analyticsFile");
    if (!fi.files[0]) return alert("Please select a CSV file first.");
    const fd = new FormData(); fd.append("file", fi.files[0]);
    try {
        document.getElementById("btnAnalytics").textContent = "Analyzing…";
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
        const deptNames = [], salVals = [];
        if (s.by_department) {
            Object.entries(s.by_department).forEach(([dept, e]) => {
                deptNames.push(dept); salVals.push(e.avg_salary ?? 0);
                const tr = document.createElement("tr");
                const fmt = v => (v == null ? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:0}));
                [dept, e.count, fmt(e.avg_satisfaction), fmt(e.avg_hours), fmt(e.avg_salary),
                 e.attrition_rate == null ? "—" : (e.attrition_rate*100).toFixed(1)+"%"
                ].forEach((v,i)=>{ const td=document.createElement("td");
                    td.style.color = i===0 ? "var(--text)" : "var(--text-dim)";
                    if(i===0) td.className="font-semibold"; td.textContent=v; tr.appendChild(td); });
                body.appendChild(tr);
            });
        }
        if (anSalaryChart) anSalaryChart.destroy();
        anSalaryChart = new Chart(document.getElementById("anSalaryChart"), {
            type: "bar",
            data: { labels: deptNames, datasets: [{ data: salVals, backgroundColor: "rgba(91,140,255,.7)", borderRadius: 6 }] },
            options: { responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { grid: { color: "rgba(255,255,255,.05)" }, border: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } },
                          x: { grid: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } } } },
        });

        const ins = document.getElementById("anInsights");
        ins.innerHTML = "";
        (data.insights || []).forEach(i => {
            const d = document.createElement("div");
            d.className = "card p-4"; d.style.borderLeft = "3px solid var(--accent)";
            const p = document.createElement("p"); p.className = "text-sm"; p.textContent = "💡 " + i.message;
            d.appendChild(p); ins.appendChild(d);
        });
    } catch (err) { alert("Analytics failed: " + err.message); }
    finally { document.getElementById("btnAnalytics").textContent = "Analyze"; }
});

// ---------- History ----------
async function fetchHistory() {
    try {
        const data = await api(`/api/history?page=${historyPage}&page_size=${HISTORY_PAGE_SIZE}`);
        allHistoryItems = data.items;
        renderHistory();
        const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
        document.getElementById("historyPageInfo").textContent = `Page ${data.page} of ${totalPages} · ${data.total} records`;
        document.getElementById("btnPrevPage").disabled = data.page <= 1;
        document.getElementById("btnNextPage").disabled = data.page >= totalPages;
    } catch (err) { console.error("history:", err); }
}
function renderHistory() {
    const body = document.getElementById("historyBody");
    const empty = document.getElementById("historyEmpty");
    body.innerHTML = "";
    const term = historySearchTerm.toLowerCase();
    const items = allHistoryItems.filter(r => !term || (r.name||"").toLowerCase().includes(term));
    if (!items.length) { empty.classList.remove("hidden"); return; }
    empty.classList.add("hidden");
    const canDel = currentUser && (currentUser.role === "ADMIN" || currentUser.role === "HR_MANAGER");
    items.forEach(r => {
        const tr = document.createElement("tr");
        const unit = (modelSchema && modelSchema.salary_unit === "monthly_income_dataset_currency") ? "" : "₹";
        const cells = [r.name, unit + Number(r.salary).toLocaleString(undefined,{maximumFractionDigits:0}), null, r.kind === "simulation" ? "sim" : "prediction", (r.timestamp||"").slice(0,16)];
        cells.forEach((v,i)=>{
            const td=document.createElement("td");
            if(i===0) td.className="font-semibold";
            if(i===2){ const b=document.createElement("span"); b.className="badge "+(r.risk>0.5?"badge-red":"badge-green"); b.textContent=((r.risk||0)*100).toFixed(0)+"%"; td.appendChild(b); }
            else if(i===3){ const b=document.createElement("span"); b.className="badge badge-gray"; b.textContent=v; td.appendChild(b); }
            else { td.style.color = i===0?"var(--text)":"var(--text-dim)"; td.textContent=v; }
            tr.appendChild(td);
        });
        const tdA=document.createElement("td");
        if(canDel){ const b=document.createElement("button"); b.className="btn btn-danger !text-[11px] !py-1 !px-2"; b.textContent="Delete";
            b.addEventListener("click",()=>deleteRecord(r.id,b)); tdA.appendChild(b); }
        tr.appendChild(tdA); body.appendChild(tr);
    });
}
async function deleteRecord(id, btn) {
    if(!confirm("Delete this record?")) return;
    try { btn.disabled=true; await api(`/api/history/${id}`,{method:"DELETE"}); fetchHistory(); }
    catch(e){ alert("Delete failed: "+e.message); btn.disabled=false; }
}
document.getElementById("historySearch").addEventListener("input", e => { historySearchTerm = e.target.value; renderHistory(); });
document.getElementById("btnPrevPage").addEventListener("click", () => { if(historyPage>1){historyPage--;fetchHistory();} });
document.getElementById("btnNextPage").addEventListener("click", () => { historyPage++; fetchHistory(); });
document.getElementById("btnClearHistory").addEventListener("click", async () => {
    if(!confirm("Permanently delete ALL prediction history?")) return;
    try { await api("/api/history",{method:"DELETE"}); historyPage=1; fetchHistory(); }
    catch(e){ alert("Clear failed: "+e.message); }
});

// ---------- Simulator ----------
const simInputs = ["simHrs","simSat","simBoost"];
let simTimer = null;
simInputs.forEach(id => document.getElementById(id).addEventListener("input", () => {
    clearTimeout(simTimer); simTimer = setTimeout(runSimulation, SIM_DEBOUNCE_MS);
}));
async function runSimulation() {
    const hrs = parseInt(document.getElementById("simHrs").value,10);
    const sat = parseFloat(document.getElementById("simSat").value);
    const boost = parseFloat(document.getElementById("simBoost").value);
    document.getElementById("simHrsLabel").textContent = hrs;
    document.getElementById("simSatLabel").textContent = sat.toFixed(2);
    document.getElementById("simBoostLabel").textContent = "+" + boost.toFixed(2);
    // Use active model's categories for a realistic baseline
    const cats = (modelSchema && modelSchema.categories) || {};
    const dept = (cats.Department||["Engineering"])[0];
    const title = (cats.Job_Title||["Engineer"])[0];
    const gender = (cats.Gender||["Male"])[0];
    const base = { Name:"Sim", Age:30, Gender:gender, Department:dept, Job_Title:title,
        Experience:5, Promotion:"No", Satisfaction:sat, Hours:hrs, simulate:true };
    const opt = { ...base, Satisfaction:Math.min(1,sat+boost) };
    try {
        const [rb, ro] = await Promise.all([
            api("/api/predict",{method:"POST",body:JSON.stringify(base)}),
            api("/api/predict",{method:"POST",body:JSON.stringify(opt)}) ]);
        document.getElementById("simBefore").textContent = (rb.risk*100).toFixed(1)+"%";
        document.getElementById("simAfter").textContent = (ro.risk*100).toFixed(1)+"%";
        document.getElementById("simImpact").textContent = ((rb.risk-ro.risk)*100).toFixed(1)+"% risk reduction";
    } catch(e){ console.error("sim:",e); }
}

bootAuth();
