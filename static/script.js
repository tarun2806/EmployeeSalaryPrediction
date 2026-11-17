// Employee Salary Prediction System — frontend
let importanceChart, dashDeptChart;
let modelSchema = null;
let historyPage = 1, historySearchTerm = "", allHistoryItems = [];
const HISTORY_PAGE_SIZE = 20;

// Bulk workflow state
let bulkFile = null, bulkValid = [], bulkErrors = [], bulkAllValid = null, currentJobId = null;
let bulkResultsAll = [], bulkResultsPage = 1, bulkSort = {key: "Estimated_Salary", dir: -1};
const BULK_PAGE_SIZE = 25;

const PAGE_META = {
    dashboard: ["Dashboard", "Company-wide salary planning overview"],
    predict: ["Individual Prediction", "Estimate a salary benchmark for one employee"],
    bulk: ["Bulk Salary Planning", "Upload → validate → estimate → plan increments → export"],
    history: ["Prediction History", "Stored predictions and bulk jobs"],
    model: ["Model Information", "Active model, data and honest metrics"],
};

async function api(path, options = {}) {
    const opts = { ...options };
    opts.headers = { ...(opts.headers || {}) };
    if (opts.body && typeof opts.body === "string") opts.headers["Content-Type"] = "application/json";
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (_) {}
    if (!res.ok) {
        const err = new Error((data && data.error) || `Request failed (${res.status})`);
        err.status = res.status; err.data = data; throw err;
    }
    return data;
}
const fmt = (v, cur = true) => v == null ? "—" : (cur ? Number(v).toLocaleString(undefined, {maximumFractionDigits: 0}) : v);

// ---------- Boot ----------
async function boot() {
    document.getElementById("appView").classList.remove("hidden");
    loadModelSchema(); loadDashboard(); checkHealth();
    fetchJobs();
}

// ---------- Navigation ----------
document.querySelectorAll(".nav-item").forEach(i => i.addEventListener("click", () => switchTab(i.dataset.tab)));
document.querySelectorAll("[data-goto]").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.goto)));
function switchTab(tab) {
    document.querySelectorAll(".nav-item").forEach(i => i.classList.toggle("active", i.dataset.tab === tab));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    document.getElementById(tab).classList.add("active");
    const [t, s] = PAGE_META[tab] || [tab, ""];
    document.getElementById("pageTitle").textContent = t;
    document.getElementById("pageSub").textContent = s;
    if (tab === "history") { historyPage = 1; fetchHistory(); fetchJobs(); }
    if (tab === "dashboard") loadDashboard();
}

// ---------- Model schema ----------
async function loadModelSchema() {
    try {
        modelSchema = await api("/api/model");
        populateSelect("Department", modelSchema.categories.Department);
        populateSelect("Job_Title", modelSchema.categories.Job_Title);
        populateSelect("Gender", modelSchema.categories.Gender);
        if (modelSchema.features && !modelSchema.features.includes("Hours"))
            document.getElementById("hoursBlock").classList.add("hidden");
        document.getElementById("modelBadge").textContent = modelSchema.active_version.toUpperCase();
        document.getElementById("modelCaveat").textContent = (modelSchema.caveats || [""])[0];
        document.getElementById("tplModel").textContent = modelSchema.active_version;
        const isIndia = modelSchema.kind === "india_salary";
        document.getElementById("standardFields").classList.toggle("hidden", isIndia);
        document.getElementById("indiaFields").classList.toggle("hidden", !isIndia);
        if (isIndia) {
            const cats = modelSchema.categories || {};
            populateSelect("Job_TitleI", cats.Job_Title || []);
            populateSelect("CityI", [{v:"",t:"Other / not listed"}, ...(cats.City || []).map(c => ({v:c,t:c}))]);
            populateSelect("SeniorityI", [{v:"",t:"Auto (from experience)"}, ...(cats.Seniority || []).map(c => ({v:c,t:c}))]);
            const jt = document.getElementById("Job_TitleI");
            jt.addEventListener("change", e => {
                const fam = (modelSchema.job_families || {})[e.target.value];
                document.getElementById("jobFamilyHint") && (document.getElementById("jobFamilyHint").textContent = fam ? "Family: " + fam : "");
            });
        }
        document.getElementById("miVersion").textContent = modelSchema.active_version;
        document.getElementById("miDataset").textContent = modelSchema.dataset || "—";
        document.getElementById("miUnit").textContent = isIndia ? "₹ / year (annual INR, dataset)" : (modelSchema.salary_unit === "monthly_income_dataset_currency" ? "USD / month (dataset currency)" : "₹ / month (estimate)");
        document.getElementById("miFeatures").textContent = modelSchema.features.join(", ");
        document.getElementById("miCaveats").innerHTML = "";
        (modelSchema.caveats || []).forEach(c => {
            const li = document.createElement("li"); li.textContent = "• " + c;
            document.getElementById("miCaveats").appendChild(li);
        });
        const isInd = isIndiaModel();
        const unit = isInd ? "₹" : (modelSchema.salary_unit === "monthly_income_dataset_currency" ? "" : "₹");
        document.getElementById("dUnit1").textContent = isInd ? "₹ / year (annual, dataset)" : (unit ? "₹ / month (estimate)" : "USD / month (dataset)");
        document.getElementById("salUnit").textContent = isInd ? "₹ / year · market benchmark (annual INR)" : (unit ? "₹ / month · benchmark estimate" : "USD / month · benchmark estimate");
        // categories
        const fill = (id, vals) => { const el = document.getElementById(id); el.innerHTML = "";
            (vals || []).forEach(v => { const d = document.createElement("div"); d.textContent = v; el.appendChild(d); }); };
        fill("miDepts", modelSchema.categories.Department);
        fill("miGenders", modelSchema.categories.Gender);
        const titles = document.getElementById("miTitles"); titles.innerHTML = "";
        (modelSchema.categories.Job_Title || []).forEach(t => {
            const fam = (modelSchema.job_families || {})[t] || "Other";
            const d = document.createElement("div");
            d.innerHTML = "";
            const s1 = document.createElement("span"); s1.textContent = t;
            const s2 = document.createElement("span"); s2.style.color = "var(--faint)"; s2.textContent = " · " + fam;
            d.appendChild(s1); d.appendChild(s2); titles.appendChild(d);
        });
        // metrics
        const mets = document.getElementById("miMetrics"); mets.innerHTML = "";
        const ev = modelSchema.metrics || {};
        const cards = [["Salary R²", ev.salary_test_r2], ["Salary MAE", ev.salary_test_mae],
                       ["Salary RMSE", ev.salary_test_rmse], ["Salary MedAE", ev.salary_test_medae],
                       ["Attrition ROC-AUC", ev.attrition_test_roc_auc], ["Attrition PR-AUC", ev.attrition_test_pr_auc]];
        cards.forEach(([label, val]) => {
            if (val == null) return;
            const d = document.createElement("div"); d.className = "card p-4 text-center";
            const p = document.createElement("p"); p.className = "stat-label"; p.textContent = label;
            const n = document.createElement("p"); n.className = "stat-num mt-1"; n.textContent = Number(val).toFixed(3);
            d.appendChild(p); d.appendChild(n); mets.appendChild(d);
        });
        if (!mets.children.length) { const p = document.createElement("p"); p.className = "text-xs"; p.style.color = "var(--dim)"; p.textContent = "No evaluation metrics recorded for this version."; mets.appendChild(p); }
        // template column guide
        const tc = document.getElementById("tplColumns"); tc.innerHTML = "";
        (modelSchema.template_columns || []).forEach(c => {
            const d = document.createElement("div"); d.textContent = "• " + c; tc.appendChild(d);
        });
        // dept hint for scenario overrides
        const hint = document.getElementById("deptIncHint");
        if (hint) hint.textContent = "Departments: " + (modelSchema.categories.Department || []).join(", ");
    } catch (err) { console.error("schema:", err); }
}
function populateSelect(id, values) {
    const sel = document.getElementById(id);
    if (!sel || !values || !values.length) return;
    sel.innerHTML = "";
    values.forEach(v => {
        const o = document.createElement("option");
        if (typeof v === "object" && v !== null) { o.value = v.v ?? v.value ?? ""; o.textContent = v.t ?? v.label ?? v.v; }
        else { o.value = o.textContent = v; }
        sel.appendChild(o);
    });
}
const isIndiaModel = () => modelSchema && modelSchema.kind === "india_salary";
const fmtSalary = (v) => v == null ? "—" : "₹" + Number(v).toLocaleString("en-IN", {maximumFractionDigits: 0});
document.getElementById("Job_Title").addEventListener("change", e => {
    const fam = (modelSchema && modelSchema.job_families || {})[e.target.value];
    document.getElementById("jobFamilyHint").textContent = fam ? "Family: " + fam : "";
});

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
        const [jobs, summary] = await Promise.all([api("/api/bulk/jobs"), api("/api/analytics/summary")]);
        document.getElementById("dPredictions").textContent = summary.total_predictions || 0;
        const latest = jobs.items && jobs.items[0];
        if (!latest) {
            document.getElementById("dashOnboarding").classList.remove("hidden");
            document.getElementById("dashData").classList.add("hidden");
            return;
        }
        document.getElementById("dashOnboarding").classList.add("hidden");
        document.getElementById("dashData").classList.remove("hidden");
        const s = latest.summary || {};
        document.getElementById("dEmployees").textContent = latest.valid_rows ?? s.succeeded ?? "—";
        const cur = s.avg_current_salary, bench = s.avg_estimated_salary;
        document.getElementById("dCurrent").textContent = fmt(cur);
        document.getElementById("dBenchmark").textContent = fmt(bench);
        const inc = document.getElementById("dIncrement"), pay = document.getElementById("dPayroll");
        if (latest.id) {
            // show last scenario if stored on job detail
            api(`/api/bulk/jobs/${latest.id}`).then(d => {
                if (d.scenario) {
                    inc.textContent = (d.scenario.avg_increment_pct ?? "—") + "%";
                    pay.textContent = fmt(d.scenario.total_additional_annual_cost);
                } else { inc.textContent = "—"; pay.textContent = "—"; }
            }).catch(() => {});
        }
        // dept chart from latest job results
        api(`/api/bulk/jobs/${latest.id}`).then(d => {
            const dept = {};
            (d.results || []).forEach(r => {
                if (r.Status !== "ok") return;
                const x = dept[r.Department] = dept[r.Department] || {est: [], cur: []};
                x.est.push(r.Estimated_Salary); if (r.Current_Salary != null) x.cur.push(r.Current_Salary);
            });
            const labels = Object.keys(dept);
            if (dashDeptChart) dashDeptChart.destroy();
            dashDeptChart = new Chart(document.getElementById("dashDeptChart"), {
                type: "bar",
                data: { labels,
                    datasets: [
                        { label: "Avg benchmark", data: labels.map(k => +(dept[k].est.reduce((a,b)=>a+b,0)/dept[k].est.length).toFixed(0)), backgroundColor: "rgba(91,140,255,.75)", borderRadius: 6 },
                        { label: "Avg current", data: labels.map(k => dept[k].cur.length ? +(dept[k].cur.reduce((a,b)=>a+b,0)/dept[k].cur.length).toFixed(0) : 0), backgroundColor: "rgba(124,91,255,.55)", borderRadius: 6 }] },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: "#8b95a5", font: { size: 11 } } } },
                    scales: { y: { grid: { color: "rgba(255,255,255,.05)" }, border: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } },
                              x: { grid: { display: false }, ticks: { color: "#8b95a5", font: { size: 10 } } } } },
            });
        }).catch(() => {});
        // recent jobs list
        const jb = document.getElementById("dashJobs"); jb.innerHTML = "";
        jobs.items.slice(0, 4).forEach(j => {
            const d = document.createElement("div");
            d.className = "flex justify-between items-center py-1.5 border-b"; d.style.borderColor = "var(--border)";
            const l = document.createElement("span"); l.className = "truncate"; l.style.color = "var(--text)";
            l.textContent = `#${j.id} ${j.filename}`;
            const r = document.createElement("span"); r.style.color = "var(--dim)"; r.className = "text-xs mono";
            r.textContent = `${j.valid_rows}/${j.total_rows} rows · ${(j.created_at||"").slice(5,10)}`;
            d.appendChild(l); d.appendChild(r); jb.appendChild(d);
        });
        if (!jobs.items.length) { const p = document.createElement("p"); p.className = "text-xs"; p.style.color = "var(--dim)"; p.textContent = "No jobs yet."; jb.appendChild(p); }
    } catch (err) { console.error("dashboard:", err); }
}

// ---------- Individual prediction ----------
document.getElementById("Satisfaction").addEventListener("input", e => {
    document.getElementById("satValue").textContent = parseFloat(e.target.value).toFixed(2);
});
document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = document.getElementById("predictError"); errEl.classList.add("hidden");
    const btn = document.getElementById("btnPredict"); btn.disabled = true;
    document.getElementById("btnPredictLabel").textContent = "Running…";
    const isInd = isIndiaModel();
    const payload = isInd ? {
        Name: document.getElementById("NameI").value || "Anonymous",
        Job_Title: document.getElementById("Job_TitleI").value,
        Experience: parseFloat(document.getElementById("ExperienceI").value) || 0,
        City: document.getElementById("CityI").value || null,
        Seniority: document.getElementById("SeniorityI").value || null,
        Industry: document.getElementById("IndustryI").value || null,
    } : {
        Name: document.getElementById("Name").value || "Anonymous",
        Age: parseInt(document.getElementById("Age").value, 10),
        Gender: document.getElementById("Gender").value,
        Department: document.getElementById("Department").value,
        Job_Title: document.getElementById("Job_Title").value,
        Experience: parseInt(document.getElementById("Experience").value, 10),
        Hours: parseInt(document.getElementById("Hours").value || "160", 10),
        Promotion: document.querySelector('input[name="Promotion"]:checked').value,
        Satisfaction: parseFloat(document.getElementById("Satisfaction").value),
    };
    try {
        const result = await api("/api/predict", { method: "POST", body: JSON.stringify(payload) });
        document.getElementById("resultEmpty").classList.add("hidden");
        const card = document.getElementById("resultCard"); card.classList.remove("hidden");
        document.getElementById("riskBlock").classList.toggle("hidden", isInd);
        document.getElementById("explainBlock").classList.toggle("hidden", isInd);
        document.getElementById("riskBadge").classList.add("hidden");
        if (isInd) {
            document.getElementById("salText").textContent = fmtSalary(result.salary_annual_inr);
            document.getElementById("salRangeLine").classList.remove("hidden");
            document.getElementById("salRangeLine").textContent =
                `Market range (p10–p90): ${fmtSalary(result.salary_p10_inr)} – ${fmtSalary(result.salary_p90_inr)} · ${result.market_observations ?? "?"} supporting postings`;
            document.getElementById("resultProvenance").classList.remove("hidden");
            document.getElementById("resultProvenance").textContent =
                `Title: ${result.canonical_title} · Family: ${result.job_family} · Support: ${result.support_level} · ${result.range_note || ""}`;
            document.getElementById("resultModel").textContent = "model: " + (result.model_version || "") + " · " + (result.disclaimer || "");
            if (importanceChart) { importanceChart.destroy(); importanceChart = null; }
            document.getElementById("shapList").innerHTML = "";
        } else {
            document.getElementById("riskBlock").classList.remove("hidden");
            document.getElementById("explainBlock").classList.remove("hidden");
            document.getElementById("riskBadge").classList.remove("hidden");
            document.getElementById("salRangeLine").classList.add("hidden");
            document.getElementById("resultProvenance").classList.add("hidden");
            const unit = result.salary_unit === "monthly_income_dataset_currency" ? "" : "₹";
            document.getElementById("salText").textContent = unit + fmt(result.salary);
            document.getElementById("salUnit").textContent = unit ? "₹ / month · benchmark estimate" : "USD / month · benchmark estimate";
            document.getElementById("riskText").textContent = (result.risk * 100).toFixed(1) + "%";
            const fill = document.getElementById("riskBarFill");
            fill.style.width = (result.risk * 100) + "%";
            fill.style.background = result.risk > 0.5 ? "var(--red)" : result.risk > 0.3 ? "var(--amber)" : "var(--green)";
            const badge = document.getElementById("riskBadge");
            badge.className = "badge " + (result.risk > 0.5 ? "badge-red" : result.risk > 0.3 ? "badge-amber" : "badge-green");
            badge.textContent = result.risk > 0.5 ? "HIGH RISK" : result.risk > 0.3 ? "ELEVATED" : "LOW RISK";
            document.getElementById("resultModel").textContent = "model: " + (result.model_version || "");
            const shap = document.getElementById("shapList"); shap.innerHTML = "";
            (result.explanation || []).slice(0, 4).forEach(d => {
                const row = document.createElement("div"); row.className = "flex items-center justify-between";
                const n = document.createElement("span"); n.style.color = "var(--dim)"; n.textContent = d.feature;
                const v = document.createElement("span"); v.className = "mono font-semibold";
                v.style.color = d.impact >= 0 ? "var(--red)" : "var(--green)";
                v.textContent = (d.impact >= 0 ? "+" : "") + d.impact.toFixed(3);
                row.appendChild(n); row.appendChild(v); shap.appendChild(row);
            });
            if (!result.explanation || !result.explanation.length) {
                const p = document.createElement("p"); p.className = "text-xs"; p.style.color = "var(--faint)";
                p.textContent = "No explanation available."; shap.appendChild(p);
            }
            renderImportanceChart(result.features || [], result.feat_imp || []);
        }
    } catch (err) { errEl.textContent = err.message; errEl.classList.remove("hidden"); }
    finally { btn.disabled = false; document.getElementById("btnPredictLabel").textContent = "Run prediction"; }
});
function renderImportanceChart(labels, data) {
    const ctx = document.getElementById("importanceChart").getContext("2d");
    if (importanceChart) { importanceChart.destroy(); importanceChart = null; }
    if (!labels || !labels.length) return;
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

// ---------- Bulk workflow ----------
const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("click", () => document.getElementById("bulkFile").click());
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => {
    e.preventDefault(); dropzone.classList.remove("drag");
    if (e.dataTransfer.files[0]) setBulkFile(e.dataTransfer.files[0]);
});
document.getElementById("bulkFile").addEventListener("change", e => { if (e.target.files[0]) setBulkFile(e.target.files[0]); });
function setBulkFile(f) {
    if (!/\.(csv|xlsx)$/i.test(f.name)) { alert("Only .csv and .xlsx files are supported."); return; }
    if (f.size > 5 * 1024 * 1024) { alert("File exceeds 5 MB limit."); return; }
    bulkFile = f;
    document.getElementById("bulkFileName").textContent = f.name + " · " + (f.size / 1024).toFixed(1) + " KB";
    document.getElementById("bulkFileName").classList.remove("hidden");
    validateBulkFile();
}
async function validateBulkFile() {
    const fd = new FormData(); fd.append("file", bulkFile);
    try {
        const data = await api("/api/bulk/validate", { method: "POST", body: fd });
        bulkValid = data.sample_valid || []; bulkErrors = data.errors || [];
        bulkAllValid = data; // keep counts
        document.getElementById("vTotal").textContent = data.total_rows;
        document.getElementById("vValid").textContent = data.valid_rows;
        document.getElementById("vInvalid").textContent = data.invalid_rows;
        document.getElementById("vFile").textContent = data.filename;
        // errors table
        const eb = document.getElementById("vErrorsBody"); eb.innerHTML = "";
        (data.errors || []).slice(0, 50).forEach(e => {
            const tr = document.createElement("tr");
            [e.row, e.EmpID ?? "—", e.Employee ?? "—", e.error].forEach((v, i) => {
                const td = document.createElement("td"); td.style.color = i === 3 ? "var(--red)" : "var(--dim)"; td.textContent = v; tr.appendChild(td);
            });
            eb.appendChild(tr);
        });
        document.getElementById("vErrorsWrap").classList.toggle("hidden", !(data.errors || []).length);
        const vb = document.getElementById("vValidBody"); vb.innerHTML = "";
        (data.sample_valid || []).forEach(e => {
            const tr = document.createElement("tr");
            [e.EmpID ?? "—", e.Employee ?? "—", e.Department, e.Job_Title].forEach(v => {
                const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
            });
            vb.appendChild(tr);
        });
        document.getElementById("vValidWrap").classList.toggle("hidden", !(data.sample_valid || []).length);
        document.getElementById("btnProcess").disabled = data.valid_rows === 0;
        setBulkStep(2);
    } catch (err) { alert("Validation failed: " + err.message); }
}
function setBulkStep(n) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById("bulkStep" + i);
        if (el) el.classList.toggle("hidden", i !== n);
    }
    document.querySelectorAll("#bulkStepper .step").forEach(s => {
        const sn = parseInt(s.dataset.step, 10);
        s.classList.toggle("done", sn < n);
        s.classList.toggle("active", sn === n);
    });
}
document.getElementById("btnBackUpload").addEventListener("click", () => setBulkStep(1));
document.getElementById("btnProcess").addEventListener("click", async () => {
    const btn = document.getElementById("btnProcess"); btn.disabled = true; btn.textContent = "Processing…";
    const fd = new FormData(); fd.append("file", bulkFile);
    try {
        const data = await api("/api/bulk/process", { method: "POST", body: fd });
        currentJobId = data.job_id;
        loadJobResults(data.job_id);
        setBulkStep(3);
    } catch (err) { alert("Processing failed: " + err.message); btn.disabled = false; btn.textContent = "Process salary estimates →"; }
});
async function loadJobResults(jobId) {
    const d = await api(`/api/bulk/jobs/${jobId}`);
    currentJobId = jobId;
    document.getElementById("jobIdLabel").textContent = "#" + jobId;
    document.getElementById("jobModel").textContent = d.model_version;
    bulkResultsAll = d.results || [];
    bulkErrors = d.errors || [];
    bulkResultsPage = 1;
    // dept filter options
    const df = document.getElementById("bulkDeptFilter");
    df.textContent = "";
    const allOpt = document.createElement("option"); allOpt.value = ""; allOpt.textContent = "All departments"; df.appendChild(allOpt);
    [...new Set(bulkResultsAll.map(r => r.Department))].sort().forEach(dep => {
        const o = document.createElement("option"); o.value = o.textContent = dep; df.appendChild(o);
    });
    // errors card
    const ec = document.getElementById("bulkErrorsCard");
    ec.classList.toggle("hidden", !bulkErrors.length);
    const eb = document.getElementById("bulkErrorsBody"); eb.innerHTML = "";
    bulkErrors.slice(0, 50).forEach(e => {
        const tr = document.createElement("tr");
        [e.row, e.EmpID ?? "—", e.Employee ?? "—", e.error].forEach((v, i) => {
            const td = document.createElement("td"); td.style.color = i === 3 ? "var(--red)" : "var(--dim)"; td.textContent = v; tr.appendChild(td);
        });
        eb.appendChild(tr);
    });
    renderBulkTable();
}
function renderBulkTable() {
    const term = document.getElementById("bulkSearch").value.toLowerCase();
    const dept = document.getElementById("bulkDeptFilter").value;
    let rows = bulkResultsAll.filter(r =>
        (!term || (r.Employee || "").toLowerCase().includes(term) || String(r.EmpID || "").includes(term)) &&
        (!dept || r.Department === dept));
    rows = [...rows].sort((a, b) => {
        const va = a[bulkSort.key], vb = b[bulkSort.key];
        const cmp = (va ?? -Infinity) < (vb ?? -Infinity) ? -1 : (va ?? -Infinity) > (vb ?? -Infinity) ? 1 : 0;
        return cmp * bulkSort.dir;
    });
    const total = rows.length, pages = Math.max(1, Math.ceil(total / BULK_PAGE_SIZE));
    bulkResultsPage = Math.min(bulkResultsPage, pages);
    const slice = rows.slice((bulkResultsPage - 1) * BULK_PAGE_SIZE, bulkResultsPage * BULK_PAGE_SIZE);
    const body = document.getElementById("bulkResultsBody"); body.innerHTML = "";
    slice.forEach(r => {
        const tr = document.createElement("tr");
        const delta = r.Salary_Delta;
        const cells = [r.EmpID ?? "—", r.Employee ?? "—", r.Department, r.Job_Title,
                       fmt(r.Current_Salary), fmt(r.Estimated_Salary),
                       delta == null ? "—" : (delta >= 0 ? "+" : "") + fmt(delta)];
        cells.forEach((v, i) => {
            const td = document.createElement("td");
            if (i === 6 && delta != null) td.style.color = delta >= 0 ? "var(--green)" : "var(--red)";
            td.textContent = v; tr.appendChild(td);
        });
        const tdS = document.createElement("td");
        const b = document.createElement("span");
        b.className = "badge " + (r.Status === "ok" ? "badge-green" : "badge-red");
        b.textContent = r.Status === "ok" ? "ok" : "failed";
        tdS.appendChild(b); tr.appendChild(tdS);
        body.appendChild(tr);
    });
    document.getElementById("bulkPageInfo").textContent = `Page ${bulkResultsPage} of ${pages} · ${total} rows`;
    document.getElementById("btnBulkPrev").disabled = bulkResultsPage <= 1;
    document.getElementById("btnBulkNext").disabled = bulkResultsPage >= pages;
}
document.querySelectorAll("#bulkResultsTable th[data-sort]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sort;
    if (bulkSort.key === k) bulkSort.dir *= -1; else { bulkSort = { key: k, dir: -1 }; }
    renderBulkTable();
}));
document.getElementById("bulkSearch").addEventListener("input", renderBulkTable);
document.getElementById("bulkDeptFilter").addEventListener("change", renderBulkTable);
document.getElementById("btnBulkPrev").addEventListener("click", () => { if (bulkResultsPage > 1) { bulkResultsPage--; renderBulkTable(); } });
document.getElementById("btnBulkNext").addEventListener("click", () => { bulkResultsPage++; renderBulkTable(); });
function setExportLinks() {
    if (!currentJobId) return;
    document.getElementById("dlXlsx").href = `/api/bulk/jobs/${currentJobId}/export?format=xlsx`;
    document.getElementById("dlCsv").href = `/api/bulk/jobs/${currentJobId}/export?format=csv`;
    document.getElementById("doneJobId").textContent = "#" + currentJobId;
}
document.getElementById("btnToScenario").addEventListener("click", () => { setExportLinks(); setBulkStep(4); });
document.getElementById("btnExportNow").addEventListener("click", () => {
    setExportLinks(); setBulkStep(5);
});
document.getElementById("btnBackResults").addEventListener("click", () => setBulkStep(3));
document.getElementById("btnCalcScenario").addEventListener("click", async () => {
    let deptInc = {};
    const raw = document.getElementById("deptIncrements").value.trim();
    if (raw) { try { deptInc = JSON.parse(raw); } catch { alert("Department overrides must be valid JSON, e.g. {\"Sales\": 12}"); return; } }
    const budget = document.getElementById("incBudget").value;
    const cap = document.getElementById("incCap").value;
    const body = {
        increment_pct: parseFloat(document.getElementById("incPct").value) || 0,
        dept_increments: deptInc,
        budget: budget === "" ? null : parseFloat(budget),
        cap: cap === "" ? null : parseFloat(cap),
    };
    try {
        const sc = await api(`/api/bulk/jobs/${currentJobId}/scenario`, { method: "POST", body: JSON.stringify(body) });
        document.getElementById("scenarioResults").classList.remove("hidden");
        document.getElementById("sCurM").textContent = fmt(sc.current_monthly_payroll);
        document.getElementById("sProjM").textContent = fmt(sc.projected_monthly_payroll);
        document.getElementById("sAnnual").textContent = fmt(sc.total_additional_annual_cost);
        document.getElementById("sCurA").textContent = fmt(sc.current_annual_payroll);
        document.getElementById("sProjA").textContent = fmt(sc.projected_annual_payroll);
        document.getElementById("sAvgInc").textContent = sc.avg_increment_pct + "%";
        document.getElementById("budgetWarn").classList.toggle("hidden", !sc.budget_exceeded);
        const db = document.getElementById("sDeptBody"); db.innerHTML = "";
        Object.entries(sc.department_impact || {}).forEach(([dep, v]) => {
            const tr = document.createElement("tr");
            [dep, v.employees, fmt(v.current_monthly), fmt(v.projected_monthly), fmt(v.projected_monthly - v.current_monthly)].forEach((x, i) => {
                const td = document.createElement("td"); if (i === 0) td.className = "font-semibold"; td.textContent = x; tr.appendChild(td);
            });
            db.appendChild(tr);
        });
        const rb = document.getElementById("sRowsBody"); rb.innerHTML = "";
        (sc.rows || []).slice(0, 200).forEach(r => {
            const tr = document.createElement("tr");
            [r.EmpID ?? "—", r.Employee ?? "—", r.Department, fmt(r.Current_Salary),
             r.Increment_Pct + "%", fmt(r.Revised_Salary), fmt(r.Monthly_Diff), fmt(r.Annual_Diff)
            ].forEach((x, i) => { const td = document.createElement("td"); if (i >= 6) td.style.color = "var(--dim)"; td.textContent = x; tr.appendChild(td); });
            rb.appendChild(tr);
        });
        document.getElementById("btnExportScenario").disabled = false;
    } catch (err) { alert("Scenario failed: " + err.message); }
});
document.getElementById("btnExportScenario").addEventListener("click", () => { setExportLinks(); setBulkStep(5); });

// ---------- Jobs + history ----------
async function fetchJobs() {
    try {
        const data = await api("/api/bulk/jobs");
        const body = document.getElementById("jobsBody"); body.innerHTML = "";
        document.getElementById("jobsEmpty").classList.toggle("hidden", data.items.length > 0);
        data.items.forEach(j => {
            const tr = document.createElement("tr");
            const cells = [`#${j.id}`, j.filename, (j.created_at || "").slice(0, 16),
                           j.total_rows, j.valid_rows, j.invalid_rows, j.model_version];
            cells.forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
            const tdA = document.createElement("td");
            const view = document.createElement("button"); view.className = "btn btn-ghost btn-sm"; view.textContent = "Open";
            view.addEventListener("click", () => { switchTab("bulk"); setBulkStep(1); loadJobResults(j.id); setBulkStep(3); });
            const dl = document.createElement("a"); dl.className = "btn btn-ghost btn-sm ml-1"; dl.textContent = "↓";
            dl.href = `/api/bulk/jobs/${j.id}/export?format=xlsx`; dl.download = "";
            tdA.appendChild(view); tdA.appendChild(dl); tr.appendChild(tdA);
            body.appendChild(tr);
        });
    } catch (err) { console.error("jobs:", err); }
}
async function fetchHistory() {
    try {
        const data = await api(`/api/history?page=${historyPage}&page_size=${HISTORY_PAGE_SIZE}`);
        allHistoryItems = data.items; renderHistory();
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
    const items = allHistoryItems.filter(r => !term || (r.name || "").toLowerCase().includes(term));
    if (!items.length) { empty.classList.remove("hidden"); return; }
    empty.classList.add("hidden");
    const unit = modelSchema && modelSchema.salary_unit === "monthly_income_dataset_currency" ? "" : "₹";
    items.forEach(r => {
        const tr = document.createElement("tr");
        const cells = [r.name, unit + fmt(r.salary), null, r.kind === "simulation" ? "sim" : "prediction", (r.timestamp || "").slice(0, 16)];
        cells.forEach((v, i) => {
            const td = document.createElement("td");
            if (i === 0) td.className = "font-semibold";
            if (i === 2) {
                const b = document.createElement("span"); b.className = "badge " + (r.risk > 0.5 ? "badge-red" : "badge-green");
                b.textContent = ((r.risk || 0) * 100).toFixed(0) + "%"; td.appendChild(b);
            } else if (i === 3) {
                const b = document.createElement("span"); b.className = "badge badge-gray"; b.textContent = v; td.appendChild(b);
            } else { td.style.color = i === 0 ? "var(--text)" : "var(--dim)"; td.textContent = v; }
            tr.appendChild(td);
        });
        const tdA = document.createElement("td");
        const b = document.createElement("button"); b.className = "btn btn-danger btn-sm"; b.textContent = "Delete";
        b.addEventListener("click", () => deleteRecord(r.id, b)); tdA.appendChild(b);
        tr.appendChild(tdA); body.appendChild(tr);
    });
}
async function deleteRecord(id, btn) {
    if (!confirm("Delete this record?")) return;
    try { btn.disabled = true; await api(`/api/history/${id}`, { method: "DELETE" }); fetchHistory(); }
    catch (e) { alert("Delete failed: " + e.message); btn.disabled = false; }
}
document.getElementById("historySearch").addEventListener("input", e => { historySearchTerm = e.target.value; renderHistory(); });
document.getElementById("btnPrevPage").addEventListener("click", () => { if (historyPage > 1) { historyPage--; fetchHistory(); } });
document.getElementById("btnNextPage").addEventListener("click", () => { historyPage++; fetchHistory(); });
document.getElementById("btnClearHistory").addEventListener("click", async () => {
    if (!confirm("Permanently delete ALL prediction history?")) return;
    try { await api("/api/history", { method: "DELETE" }); historyPage = 1; fetchHistory(); }
    catch (e) { alert("Clear failed: " + e.message); }
});

boot();
