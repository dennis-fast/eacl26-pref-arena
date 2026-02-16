// app.js
import { parseCSV, toCSV, downloadText, detectColumns } from "./csv.js";
import { DEFAULT_MU, DEFAULT_SIGMA, updatePair } from "./rating.js";
import { chooseNextPair } from "./selector.js";
import { saveState, loadState, clearState } from "./storage.js";

const AUTO_FILENAME = "EACL_2026_program_categorized.csv";
const BASE_K = 32;
const NEITHER_PENALTY = 10;

let RAW_HEADERS = [];
let RAW_RECORDS = [];
let COL = null;

function getDefaultState() {
  return {
    version: 2,
    ratings: {},       // id -> {mu, sigma, n, wins, losses, ties}
    history: [],       // {a,b,outcome,choice,kMult,neitherPenaltyApplied,ts}
    lastPair: [],
    mode: "active",
    topN: 60
  };
}

function normalizeHistoryEntry(h) {
  if (!h || typeof h !== "object") return null;
  if ((h.outcome !== null && h.outcome !== undefined) && (!h.a || !h.b)) return null;
  const outcome = (typeof h.outcome === "number") ? h.outcome : null;
  let choice = typeof h.choice === "string" ? h.choice : null;
  let kMult = (typeof h.kMult === "number") ? h.kMult : null;
  let neitherPenaltyApplied = Boolean(h.neitherPenaltyApplied);

  if (!choice) {
    if (outcome === 1.0) choice = "A";
    else if (outcome === 0.0) choice = "B";
    else if (outcome === 0.5) choice = "BOTH";
    else choice = "SKIP";
  }

  if (kMult === null) {
    if (outcome === 1.0 || outcome === 0.0) kMult = 1.0;
    else if (outcome === 0.5) kMult = 0.75; // legacy tie used baseK=24
    else kMult = 0.0;
  }

  if (choice === "NEITHER") neitherPenaltyApplied = true;

  return {
    a: h.a,
    b: h.b,
    outcome,
    choice,
    kMult,
    neitherPenaltyApplied,
    ts: h.ts ?? new Date().toISOString()
  };
}

function normalizeState(raw) {
  if (!raw || typeof raw !== "object" || !raw.ratings) return null;
  const base = getDefaultState();
  const history = Array.isArray(raw.history) ? raw.history : [];
  const normalizedHistory = history.map(normalizeHistoryEntry).filter(Boolean);
  return {
    ...base,
    ...raw,
    version: 2,
    history: normalizedHistory
  };
}

let state = normalizeState(loadState()) ?? getDefaultState();

let filtered = [];
let currentPair = null;

const el = (id) => document.getElementById(id);
const arena = el("arena");
const leaderboardPanel = el("leaderboardPanel");

function error(msg) {
  el("errors").innerHTML = `<div>${escapeHtml(msg)}</div>`;
}
function clearError(){ el("errors").innerHTML = ""; }

function escapeHtml(s){
  return String(s ?? "")
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
    .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

function getRating(id) {
  if (!state.ratings[id]) {
    state.ratings[id] = { mu: DEFAULT_MU, sigma: DEFAULT_SIGMA, n: 0, wins: 0, losses: 0, ties: 0 };
  }
  return state.ratings[id];
}

function hydrateItem(rec) {
  const id = String(rec[COL.id] ?? "").trim();
  const r = getRating(id);
  return {
    id,
    title: String(rec[COL.title] ?? ""),
    abstract: String(rec[COL.abstract] ?? ""),
    authors: COL.authors ? String(rec[COL.authors] ?? "") : "",
    presenter: COL.presenter ? String(rec[COL.presenter] ?? "") : "",
    session: COL.session ? String(rec[COL.session] ?? "") : "",
    location: COL.location ? String(rec[COL.location] ?? "") : "",
    date: COL.date ? String(rec[COL.date] ?? "") : "",
    time: COL.time ? String(rec[COL.time] ?? "") : "",
    cat1: COL.cat1 ? String(rec[COL.cat1] ?? "") : "",
    cat2: COL.cat2 ? String(rec[COL.cat2] ?? "") : "",
    keywords: COL.keywords ? String(rec[COL.keywords] ?? "") : "",
    mu: r.mu, sigma: r.sigma, n: r.n
  };
}

function rebuildFiltered() {
  const q = el("search").value.trim().toLowerCase();
  const cat = el("categoryFilter").value;
  const ses = el("sessionFilter").value;
  const loc = el("locationFilter").value;

  filtered = RAW_RECORDS.map(hydrateItem).filter(it => {
    if (!it.id) return false;

    if (cat && (it.cat1 !== cat && it.cat2 !== cat)) return false;
    if (ses && it.session !== ses) return false;
    if (loc && it.location !== loc) return false;

    if (!q) return true;
    const blob = [
      it.id, it.title, it.abstract, it.authors, it.presenter,
      it.keywords, it.cat1, it.cat2, it.session, it.location, it.date, it.time
    ].join(" ").toLowerCase();
    return blob.includes(q);
  });

  renderStats();
  renderLeaderboard();
  nextPair();
}

function renderStats() {
  const total = RAW_RECORDS.length;
  const shown = filtered.length;
  const nMatches = state.history.length;
  const uniqueRated = Object.values(state.ratings).filter(r => r.n > 0).length;

  el("stats").innerHTML = `
    <span class="pill">Total: <b>${total}</b></span>
    <span class="pill">Filtered: <b>${shown}</b></span>
    <span class="pill">Comparisons: <b>${nMatches}</b></span>
    <span class="pill">Rated papers: <b>${uniqueRated}</b></span>
    <span class="pill">Mode: <b>${escapeHtml(state.mode)}</b></span>
  `;
}

function renderCard(node, item, label) {
  const authorLine = formatAuthors(item.authors, item.presenter);
  node.innerHTML = `
    <div class="head" role="button" tabindex="0">
      <div><b>${label}</b> — <span class="badge">${escapeHtml(item.id)}</span></div>
      <div style="margin-top:6px;font-weight:900">${escapeHtml(item.title || "(No title)")}</div>
      <div class="k">
        μ=${item.mu.toFixed(1)} · σ=${item.sigma.toFixed(1)} · matches=${item.n}
      </div>
    </div>
    <div class="body">
      <div class="meta">
        <span class="chip">Session: ${escapeHtml(item.session || "—")}</span>
        <span class="chip">Location: ${escapeHtml(item.location || "—")}</span>
        <span class="chip">Category: ${escapeHtml(item.cat1 || "—")}${item.cat2 ? " / "+escapeHtml(item.cat2) : ""}</span>
        <span class="chip">Keywords: ${escapeHtml(item.keywords || "—")}</span>
      </div>

      <div style="margin-top:10px;color:var(--muted);font-size:13px">
        ${escapeHtml(authorLine || "")}
      </div>

      <div style="margin-top:10px;white-space:pre-wrap">
        <b>Abstract:</b>\n${escapeHtml(item.abstract || "(No abstract)")}
      </div>
    </div>
  `;

  // allow clicking header to choose
  const head = node.querySelector(".head");
  head.addEventListener("click", () => (label === "A" ? vote("A") : vote("B")));
  head.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") (label === "A" ? vote("A") : vote("B"));
  });
}

function formatAuthors(authors, presenter) {
  if (!authors) return "";
  if (!presenter) return authors;
  // best-effort bold presenter substring
  const re = new RegExp(presenter.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  if (re.test(authors)) return authors.replace(re, (m) => `**${m}**`);
  return `${authors} (presenter: ${presenter})`;
}

function nextPair() {
  if (filtered.length < 2) {
    arena.hidden = true;
    leaderboardPanel.hidden = false;
    return;
  }

  const pair = chooseNextPair(filtered, state);
  if (!pair) return;
  const [A, B] = pair;

  currentPair = { A, B };
  state.lastPair = [A, B];

  renderCard(el("cardA"), A, "A");
  renderCard(el("cardB"), B, "B");

  arena.hidden = false;
  leaderboardPanel.hidden = false;

  saveState(state);
}

function applyOutcomeCounters(rA, rB, outcome) {
  if (outcome === 1.0) { rA.wins++; rB.losses++; }
  else if (outcome === 0.0) { rB.wins++; rA.losses++; }
  else { rA.ties++; rB.ties++; }
}

function vote(choice) {
  if (!currentPair) return;

  const map = {
    A: { outcome: 1.0, kMult: 1.0, neitherPenaltyApplied: false },
    STRONG_A: { outcome: 1.0, kMult: 1.8, neitherPenaltyApplied: false },
    B: { outcome: 0.0, kMult: 1.0, neitherPenaltyApplied: false },
    STRONG_B: { outcome: 0.0, kMult: 1.8, neitherPenaltyApplied: false },
    BOTH: { outcome: 0.5, kMult: 0.8, neitherPenaltyApplied: false },
    NEITHER: { outcome: 0.5, kMult: 1.2, neitherPenaltyApplied: true },
    SKIP: { outcome: null, kMult: 0.0, neitherPenaltyApplied: false }
  };

  const cfg = map[choice];
  if (!cfg) return;

  const A = currentPair.A;
  const B = currentPair.B;

  if (cfg.outcome === null) {
    state.history.push({
      a: A.id,
      b: B.id,
      outcome: null,
      choice,
      kMult: 0.0,
      neitherPenaltyApplied: false,
      ts: new Date().toISOString()
    });
    rebuildFiltered();
    return;
  }

  const rA = getRating(A.id);
  const rB = getRating(B.id);

  updatePair(rA, rB, cfg.outcome, { baseK: BASE_K * cfg.kMult });
  if (cfg.neitherPenaltyApplied) {
    rA.mu -= NEITHER_PENALTY;
    rB.mu -= NEITHER_PENALTY;
  }

  applyOutcomeCounters(rA, rB, cfg.outcome);

  state.history.push({
    a: A.id,
    b: B.id,
    outcome: cfg.outcome,
    choice,
    kMult: cfg.kMult,
    neitherPenaltyApplied: cfg.neitherPenaltyApplied,
    ts: new Date().toISOString()
  });

  rebuildFiltered();
}

function undo() {
  // simplest robust undo: reload from scratch by replaying history except last
  if (state.history.length === 0) return;

  const history = state.history.slice(0, -1);

  // reset ratings
  state.ratings = {};
  state.history = [];
  state.lastPair = [];

  for (const h of history) {
    if (h.outcome === null || h.outcome === undefined) {
      state.history.push(h);
      continue;
    }
    const rA = getRating(h.a);
    const rB = getRating(h.b);
    const kMult = (typeof h.kMult === "number") ? h.kMult : 1.0;
    updatePair(rA, rB, h.outcome, { baseK: BASE_K * kMult });
    if (h.neitherPenaltyApplied) {
      rA.mu -= NEITHER_PENALTY;
      rB.mu -= NEITHER_PENALTY;
    }
    applyOutcomeCounters(rA, rB, h.outcome);
    state.history.push(h);
  }
  rebuildFiltered();
  saveState(state);
}

function renderLeaderboard() {
  const lb = el("leaderboard");
  const items = filtered.slice().sort((a,b) => b.mu - a.mu).slice(0, 30);
  lb.innerHTML = items.map((it, idx) => `
    <div class="row">
      <div class="top">
        <div class="title">#${idx+1} — ${escapeHtml(it.title)}</div>
        <div class="badge">μ ${it.mu.toFixed(1)} · σ ${it.sigma.toFixed(1)} · n ${it.n}</div>
      </div>
      <div class="sub">
        ${escapeHtml(it.id)} · ${escapeHtml(it.session || "—")} · ${escapeHtml(it.location || "—")} ·
        ${escapeHtml(it.cat1 || "—")}${it.cat2 ? " / "+escapeHtml(it.cat2) : ""}
      </div>
    </div>
  `).join("");
}

function populateFilters() {
  const cats = new Set();
  const sessions = new Set();
  const locations = new Set();

  for (const rec of RAW_RECORDS) {
    const it = hydrateItem(rec);
    if (it.cat1) cats.add(it.cat1);
    if (it.cat2) cats.add(it.cat2);
    if (it.session) sessions.add(it.session);
    if (it.location) locations.add(it.location);
  }

  const catSel = el("categoryFilter");
  catSel.innerHTML = `<option value="">All categories</option>` +
    [...cats].sort().map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");

  const sesSel = el("sessionFilter");
  sesSel.innerHTML = `<option value="">All sessions</option>` +
    [...sessions].sort().map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join("");

  const locSel = el("locationFilter");
  locSel.innerHTML = `<option value="">All locations</option>` +
    [...locations].sort().map(l => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`).join("");
}

async function loadCSVTextFromFetch() {
  const res = await fetch(AUTO_FILENAME, { cache: "no-store" });
  if (!res.ok) throw new Error(`Fetch failed (${res.status} ${res.statusText})`);
  return await res.text();
}

function loadCSVFromText(text) {
  const rows = parseCSV(text);
  if (rows.length < 2) throw new Error("CSV is empty or missing data.");

  RAW_HEADERS = rows[0];
  COL = detectColumns(RAW_HEADERS);

  if (!COL.id || !COL.title || !COL.abstract) {
    throw new Error(`Missing required columns. Need ID/Title/Abstract. Found headers: ${RAW_HEADERS.join(", ")}`);
  }

  RAW_RECORDS = rows.slice(1).map(r => {
    const rec = {};
    RAW_HEADERS.forEach((h, i) => rec[h] = r[i] ?? "");
    return rec;
  });

  populateFilters();
  rebuildFiltered();
}

function exportState() {
  const payload = JSON.stringify(state, null, 2);
  downloadText("eacl_pref_state.json", payload, "application/json");
}

function importState(jsonText) {
  const obj = JSON.parse(jsonText);
  if (!obj || typeof obj !== "object" || !obj.ratings) throw new Error("Invalid state JSON.");
  const normalized = normalizeState(obj);
  if (!normalized) throw new Error("Invalid state JSON.");
  state = normalized;
  saveState(state);
  rebuildFiltered();
}

function exportScoredCSV() {
  if (!RAW_HEADERS.length) return;

  const extraCols = [
    "pref_mu","pref_sigma","pref_rank","n_matches","wins","losses","ties"
  ];
  const headers = [...RAW_HEADERS];
  for (const c of extraCols) if (!headers.includes(c)) headers.push(c);

  // compute ranks across ALL papers
  const all = RAW_RECORDS.map(hydrateItem).sort((a,b) => b.mu - a.mu);
  const rankById = new Map(all.map((it, idx) => [it.id, idx+1]));

  const outRecords = RAW_RECORDS.map(rec => {
    const id = String(rec[COL.id] ?? "").trim();
    const r = state.ratings[id] ?? { mu: DEFAULT_MU, sigma: DEFAULT_SIGMA, n: 0, wins: 0, losses: 0, ties: 0 };
    const out = { ...rec };
    out.pref_mu = r.mu;
    out.pref_sigma = r.sigma;
    out.pref_rank = rankById.get(id) ?? "";
    out.n_matches = r.n;
    out.wins = r.wins;
    out.losses = r.losses;
    out.ties = r.ties;
    return out;
  });

  const csv = toCSV(headers, outRecords);
  downloadText("EACL_2026_program_scored.csv", csv, "text/csv");
}

/* --- Wire UI --- */
el("csvFile").addEventListener("change", async (e) => {
  clearError();
  const f = e.target.files?.[0];
  if (!f) return;
  const text = await f.text();
  loadCSVFromText(text);
});

el("btnAutoLoad").addEventListener("click", async () => {
  clearError();
  try {
    const text = await loadCSVTextFromFetch();
    loadCSVFromText(text);
  } catch (err) {
    error(String(err.message ?? err));
  }
});

el("btnReset").addEventListener("click", () => {
  if (!confirm("Reset all ratings and history?")) return;
  clearState();
  state = getDefaultState();
  saveState(state);
  rebuildFiltered();
});

el("btnExportState").addEventListener("click", exportState);

el("stateFile").addEventListener("change", async (e) => {
  clearError();
  const f = e.target.files?.[0];
  if (!f) return;
  try {
    importState(await f.text());
  } catch (err) {
    error(String(err.message ?? err));
  }
});

el("btnExportCSV").addEventListener("click", exportScoredCSV);

el("btnA").addEventListener("click", () => vote("A"));
el("btnStrongA").addEventListener("click", () => vote("STRONG_A"));
el("btnB").addEventListener("click", () => vote("B"));
el("btnStrongB").addEventListener("click", () => vote("STRONG_B"));
el("btnBoth").addEventListener("click", () => vote("BOTH"));
el("btnNeither").addEventListener("click", () => vote("NEITHER"));
el("btnSkip").addEventListener("click", () => vote("SKIP"));
el("btnUndo").addEventListener("click", undo);

el("search").addEventListener("input", rebuildFiltered);
el("categoryFilter").addEventListener("change", rebuildFiltered);
el("sessionFilter").addEventListener("change", rebuildFiltered);
el("locationFilter").addEventListener("change", rebuildFiltered);

el("mode").addEventListener("change", (e) => {
  state.mode = e.target.value;
  saveState(state);
  nextPair();
});

el("topN").addEventListener("input", (e) => {
  state.topN = Number(e.target.value || 60);
  saveState(state);
});

// Try auto-load on start (best effort).
(async function boot(){
  try {
    const text = await loadCSVTextFromFetch();
    loadCSVFromText(text);
  } catch {
    // ignore; user can upload
    renderStats();
  }
})();
