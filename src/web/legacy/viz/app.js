const state = {
  meta: null,
  filters: {},
  currentPoints: [],
  selectedPoints: [],
  sort: { key: 'paper_id', dir: 'asc' },
  currentClickedPaperId: null,
  lastRequestPayload: null,
  lastPlotMeta: { colorBy: 'Session', mode: 'all' },
  lastNeighbors: [],
  ratings: {},
  colorMaps: {
    Session: {},
    'Room Location': {},
  },
};

const FILTER_COLUMNS = ['Type of Presentation', 'Attendance Type', 'Room Location', 'Session'];
const VIEW_STATE_KEY = 'eacl2026_embedding_dashboard_state';
const RANKING_STORAGE_KEY = 'eacl_pref_arena_state_v1';
const DEFAULT_MU = 1500;
const FIXED_COLOR_PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd',
  '#31a354', '#756bb1', '#636363', '#e6550d',
];

function qs(id) {
  return document.getElementById(id);
}

function readRankingStateFromStorage() {
  const raw = localStorage.getItem(RANKING_STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed?.ratings || {};
  } catch {
    return {};
  }
}

function refreshRatingsFromStorage() {
  state.ratings = readRankingStateFromStorage();
}

function buildRankMap() {
  const entries = Object.entries(state.ratings || {})
    .filter(([, rating]) => rating && typeof rating.mu === 'number')
    .sort((a, b) => b[1].mu - a[1].mu);

  const rankMap = new Map();
  entries.forEach(([paperId], idx) => {
    rankMap.set(paperId, idx + 1);
  });
  return rankMap;
}

function rankForPaperId(paperId, rankMap = null) {
  const map = rankMap || buildRankMap();
  return map.get(paperId) ?? null;
}

function buildOpacityMap(points) {
  const mus = points.map((p) => Number(state.ratings?.[p.paper_id]?.mu ?? DEFAULT_MU));
  if (mus.length === 0) return new Map();
  const minMu = Math.min(...mus);
  const maxMu = Math.max(...mus);
  const span = Math.max(1e-9, maxMu - minMu);
  const opacityById = new Map();

  points.forEach((point, index) => {
    const mu = mus[index];
    const norm = (mu - minMu) / span;
    const opacity = 0.18 + (0.92 - 0.18) * norm;
    opacityById.set(point.paper_id, opacity);
  });

  return opacityById;
}

function setStatus(message, isError = false) {
  const el = qs('status-line');
  el.textContent = message;
  el.style.color = isError ? '#b42318' : '#1b2430';
}

function setLoading(loading) {
  qs('run-btn').disabled = loading;
  if (loading) {
    setStatus('Running projection...');
  }
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed: ${res.status}`);
  }
  return data;
}

function updateMethodParamVisibility() {
  const method = qs('method-select').value;
  qs('params-pca').classList.toggle('hidden', method !== 'pca');
  qs('params-tsne').classList.toggle('hidden', method !== 'tsne');
  qs('params-umap').classList.toggle('hidden', method !== 'umap');
}

function updateModeUI() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  qs('session-select').disabled = mode !== 'session';
}

function sortedUnique(values) {
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b)));
}

function createColorMap(values) {
  const sortedValues = sortedUnique(values || []);
  const map = {};
  sortedValues.forEach((value, index) => {
    map[value] = FIXED_COLOR_PALETTE[index % FIXED_COLOR_PALETTE.length];
  });
  return map;
}

function initColorMaps(meta) {
  const sessionValues = meta?.filters?.Session || [];
  const roomValues = meta?.filters?.['Room Location'] || [];
  state.colorMaps.Session = createColorMap(sessionValues);
  state.colorMaps['Room Location'] = createColorMap(roomValues);
}

function getCategoryColor(categoryValue, colorBy) {
  const key = String(categoryValue || 'Unknown');
  const colorMap = state.colorMaps[colorBy] || {};
  if (colorMap[key]) return colorMap[key];
  const fallbackIdx = Math.abs(key.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0));
  return FIXED_COLOR_PALETTE[fallbackIdx % FIXED_COLOR_PALETTE.length];
}

function buildFilterGroup(column, values) {
  const wrap = document.createElement('div');
  wrap.className = 'filter-group';

  const title = document.createElement('h3');
  title.textContent = column;
  title.style.margin = '0 0 6px';
  title.style.fontSize = '0.9rem';

  const actions = document.createElement('div');
  actions.className = 'filter-actions';

  const btnAll = document.createElement('button');
  btnAll.className = 'secondary';
  btnAll.textContent = 'Select all';
  btnAll.type = 'button';
  btnAll.addEventListener('click', () => {
    wrap.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
      cb.checked = true;
    });
  });

  const btnNone = document.createElement('button');
  btnNone.className = 'secondary';
  btnNone.textContent = 'Select none';
  btnNone.type = 'button';
  btnNone.addEventListener('click', () => {
    wrap.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
      cb.checked = false;
    });
  });

  actions.appendChild(btnAll);
  actions.appendChild(btnNone);

  const valuesWrap = document.createElement('div');
  valuesWrap.className = 'filter-values';

  values.forEach((v) => {
    const item = document.createElement('label');
    item.className = 'filter-value';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.dataset.column = column;
    cb.value = v;

    const span = document.createElement('span');
    span.textContent = v;

    item.appendChild(cb);
    item.appendChild(span);
    valuesWrap.appendChild(item);
  });

  wrap.appendChild(title);
  wrap.appendChild(actions);
  wrap.appendChild(valuesWrap);
  return wrap;
}

function renderFilters(metaFilters) {
  const container = qs('filters-container');
  container.innerHTML = '';
  FILTER_COLUMNS.forEach((col) => {
    const vals = sortedUnique(metaFilters[col] || []);
    container.appendChild(buildFilterGroup(col, vals));
  });
}

function getCurrentFilters() {
  const filters = {};
  FILTER_COLUMNS.forEach((col) => {
    const checkboxes = document.querySelectorAll(`input[type="checkbox"][data-column="${col}"]`);
    filters[col] = [...checkboxes].filter((cb) => cb.checked).map((cb) => cb.value);
  });
  return filters;
}

function getCurrentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function getProjectionParams() {
  const method = qs('method-select').value;
  if (method === 'pca') {
    return {
      whiten: qs('pca-whiten').checked,
      random_state: Number(qs('pca-random-state').value || 42),
    };
  }
  if (method === 'tsne') {
    const lrRaw = qs('tsne-learning-rate').value.trim();
    const learningRate = lrRaw.toLowerCase() === 'auto' ? 'auto' : Number(lrRaw);
    return {
      perplexity: Number(qs('tsne-perplexity').value || 30),
      learning_rate: Number.isFinite(learningRate) ? learningRate : 'auto',
      n_iter: Number(qs('tsne-n-iter').value || 1000),
      init: qs('tsne-init').value,
      metric: qs('tsne-metric').value,
      random_state: Number(qs('tsne-random-state').value || 42),
      pca_components_for_tsne_umap: Number(qs('tsne-pca-pre').value || 50),
    };
  }
  return {
    n_neighbors: Number(qs('umap-n-neighbors').value || 15),
    min_dist: Number(qs('umap-min-dist').value || 0.1),
    metric: qs('umap-metric').value,
    spread: Number(qs('umap-spread').value || 1.0),
    random_state: Number(qs('umap-random-state').value || 42),
    pca_components_for_tsne_umap: Number(qs('umap-pca-pre').value || 50),
  };
}

function buildProjectPayload() {
  const mode = getCurrentMode();
  const payload = {
    method: qs('method-select').value,
    params: getProjectionParams(),
    mode,
    session_value: mode === 'session' ? qs('session-select').value : null,
    filters: getCurrentFilters(),
    sample: {
      enabled: qs('sample-enabled').checked,
      max_points: Number(qs('sample-max-points').value || 960),
      strategy: qs('sample-strategy').value,
    },
    oral_only: qs('oral-only-toggle').checked,
    search_text: qs('search-text').value.trim(),
    search_mode: qs('search-mode').value,
  };
  state.lastRequestPayload = payload;
  return payload;
}

function pointHoverTemplate(mode) {
  if (mode === 'session') {
    return '<b>%{customdata[1]}</b><br>paper_id: %{customdata[0]}<br>rank: %{customdata[6]}<br>Room: %{customdata[3]}<extra></extra>';
  }
  return '<b>%{customdata[1]}</b><br>paper_id: %{customdata[0]}<br>rank: %{customdata[6]}<extra></extra>';
}

function toTableRowFromPoint(point, rankMap = null) {
  return {
    rank: rankForPaperId(point.paper_id, rankMap),
    paper_id: point.paper_id,
    title: point.title,
    session: point.session,
    room_location: point.room_location,
    type_presentation: point.type_presentation,
    attendance_type: point.attendance_type,
  };
}

function getHighlightedRows() {
  const hasSearch = !!qs('search-text').value.trim();
  const isHighlightMode = qs('search-mode').value === 'highlight';
  if (!hasSearch || !isHighlightMode) return [];
  const rankMap = buildRankMap();
  return state.currentPoints.filter((p) => p.matched).map((p) => toTableRowFromPoint(p, rankMap));
}

function mergeUniqueRows(rows) {
  const dedup = new Map();
  rows.forEach((row) => dedup.set(row.paper_id, row));
  return [...dedup.values()];
}

function groupByCategory(points) {
  const grouped = new Map();
  points.forEach((p) => {
    const key = p.color_value || 'Unknown';
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(p);
  });
  return grouped;
}

function buildTraces(points, mode, colorBy) {
  const grouped = groupByCategory(points);
  const traces = [];
  const opacityById = buildOpacityMap(points);
  const rankMap = buildRankMap();

  [...grouped.entries()].forEach(([category, items]) => {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: String(category),
      x: items.map((p) => p.x),
      y: items.map((p) => p.y),
      text: items.map((p) => p.title),
      customdata: items.map((p) => [
        p.paper_id,
        p.title,
        p.session,
        p.room_location,
        p.type_presentation,
        p.attendance_type,
        rankForPaperId(p.paper_id, rankMap) ?? '-',
      ]),
      hovertemplate: pointHoverTemplate(mode),
      marker: {
        size: 8,
        opacity: items.map((p) => opacityById.get(p.paper_id) ?? 0.65),
        color: getCategoryColor(category, colorBy),
      },
    });
  });

  const matched = points.filter((p) => p.matched);
  if (matched.length > 0 && qs('search-mode').value === 'highlight' && qs('search-text').value.trim()) {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: 'Search matches',
      x: matched.map((p) => p.x),
      y: matched.map((p) => p.y),
      customdata: matched.map((p) => [
        p.paper_id,
        p.title,
        p.session,
        p.room_location,
        p.type_presentation,
        p.attendance_type,
        rankForPaperId(p.paper_id, rankMap) ?? '-',
      ]),
      hovertemplate: pointHoverTemplate(mode),
      marker: {
        size: 12,
        color: 'rgba(0,0,0,0)',
        line: { width: 2, color: '#d92d20' },
      },
    });
  }

  return traces;
}

function legendAndMarginForViewport() {
  const isSmallScreen = window.matchMedia('(max-width: 900px)').matches;
  if (isSmallScreen) {
    return {
      legend: {
        orientation: 'h',
        x: 0,
        xanchor: 'left',
        y: -0.22,
        yanchor: 'top',
        traceorder: 'normal',
        font: { size: 11 },
      },
      margin: { t: 48, r: 16, b: 150, l: 48 },
    };
  }

  return {
    legend: {
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: -0.2,
      yanchor: 'top',
      traceorder: 'normal',
    },
    margin: { t: 48, r: 16, b: 90, l: 48 },
  };
}

function renderPlot(points, colorBy, mode) {
  const plot = qs('plot');
  state.lastPlotMeta = { colorBy, mode };
  const traces = buildTraces(points, mode, colorBy);
  const viewportLayout = legendAndMarginForViewport();

  Plotly.newPlot(
    plot,
    traces,
    {
      dragmode: 'lasso',
      title: `2D projection (color by ${colorBy})`,
      margin: viewportLayout.margin,
      xaxis: { title: 'x' },
      yaxis: { title: 'y' },
      legend: viewportLayout.legend,
      hovermode: 'closest',
    },
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToAdd: ['select2d', 'lasso2d'],
    }
  );

  plot.removeAllListeners('plotly_selected');
  plot.removeAllListeners('plotly_click');

  plot.on('plotly_selected', (ev) => {
    if (!ev || !ev.points) {
      state.selectedPoints = getHighlightedRows();
      renderSelectionTable();
      return;
    }
    const selected = ev.points.map((pp) => {
      const [paper_id, title, session, room_location, type_presentation, attendance_type, rank] = pp.customdata;
      return { paper_id, title, session, room_location, type_presentation, attendance_type, rank: rank === '-' ? null : Number(rank) };
    });

    state.selectedPoints = mergeUniqueRows([...selected, ...getHighlightedRows()]);
    renderSelectionTable();
  });

  plot.on('plotly_click', async (ev) => {
    if (!ev || !ev.points || ev.points.length === 0) return;
    const [paper_id, title, session, room_location, type_presentation, attendance_type] = ev.points[0].customdata;
    state.currentClickedPaperId = paper_id;
    renderDetails({ paper_id, title, session, room_location, type_presentation, attendance_type });
    await refreshNeighbors();
  });
}

function requestPlotResize() {
  const plot = qs('plot');
  if (!plot || !plot.data) return;
  const applyResize = () => {
    Plotly.Plots.resize(plot);
    const viewportLayout = legendAndMarginForViewport();
    Plotly.relayout(plot, {
      margin: viewportLayout.margin,
      legend: viewportLayout.legend,
    });
  };
  requestAnimationFrame(() => {
    applyResize();
    setTimeout(applyResize, 60);
  });
}

function renderStats(stats) {
  qs('stats-line').textContent = `n_total=${stats.n_total} | n_filtered=${stats.n_filtered} | n_returned=${stats.n_returned} | compute_ms=${stats.compute_ms}`;
}

function compareValues(a, b, dir) {
  const lhs = String(a || '');
  const rhs = String(b || '');
  const cmp = lhs.localeCompare(rhs, undefined, { numeric: true, sensitivity: 'base' });
  return dir === 'asc' ? cmp : -cmp;
}

function renderSelectionTable() {
  const tbody = qs('selection-table').querySelector('tbody');
  tbody.innerHTML = '';

  const { key, dir } = state.sort;
  const rows = [...state.selectedPoints].sort((a, b) => compareValues(a[key], b[key], dir));

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.rank ?? '-'}</td>
      <td>${row.paper_id}</td>
      <td>${row.title}</td>
      <td>${row.session}</td>
      <td>${row.room_location}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderDetails(paper) {
  qs('details-empty').classList.add('hidden');
  qs('details-content').classList.remove('hidden');
  qs('detail-title').textContent = paper.title;
  qs('detail-paper-id').textContent = paper.paper_id;
  qs('detail-session').textContent = paper.session;
  qs('detail-room').textContent = paper.room_location;
  qs('detail-type').textContent = paper.type_presentation;
  qs('detail-attendance').textContent = paper.attendance_type;
}

function renderNeighbors(rows) {
  state.lastNeighbors = rows || [];
  const tbody = qs('nn-table').querySelector('tbody');
  tbody.innerHTML = '';
  const rankMap = buildRankMap();

  rows.forEach((row) => {
    const rank = rankForPaperId(row.paper_id, rankMap);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${rank ?? '-'}</td>
      <td>${row.paper_id}</td>
      <td>${row.title}</td>
      <td>${row.cosine.toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function refreshNeighbors() {
  if (!state.currentClickedPaperId) return;
  const k = Number(qs('nn-k').value || 10);
  try {
    const res = await fetchJSON('/api/nn', {
      method: 'POST',
      body: JSON.stringify({
        paper_id: state.currentClickedPaperId,
        k,
        current_ids: state.currentPoints.map((p) => p.paper_id),
      }),
    });
    renderNeighbors(res.neighbors || []);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function downloadCSV(filename, rows) {
  const csv = rows
    .map((r) =>
      r
        .map((v) => {
          const str = String(v ?? '');
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replaceAll('"', '""')}"`;
          }
          return str;
        })
        .join(',')
    )
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportSelectedIds() {
  if (state.selectedPoints.length === 0) {
    setStatus('No selected points to export.');
    return;
  }
  const rows = [['paper_id'], ...state.selectedPoints.map((p) => [p.paper_id])];
  downloadCSV('selected_paper_ids.csv', rows);
}

function exportProjection() {
  if (state.currentPoints.length === 0) {
    setStatus('No projection data to export.');
    return;
  }

  const header = [
    'paper_id',
    'x',
    'y',
    'title',
    'session',
    'room_location',
    'type_presentation',
    'attendance_type',
  ];

  const rows = [
    header,
    ...state.currentPoints.map((p) => [
      p.paper_id,
      p.x,
      p.y,
      p.title,
      p.session,
      p.room_location,
      p.type_presentation,
      p.attendance_type,
    ]),
  ];
  downloadCSV('projection.csv', rows);
}

function collectViewState() {
  return {
    method: qs('method-select').value,
    params: getProjectionParams(),
    mode: getCurrentMode(),
    session_value: qs('session-select').value,
    filters: getCurrentFilters(),
    sample: {
      enabled: qs('sample-enabled').checked,
      max_points: Number(qs('sample-max-points').value || 960),
      strategy: qs('sample-strategy').value,
    },
    oral_only: qs('oral-only-toggle').checked,
    search_text: qs('search-text').value,
    search_mode: qs('search-mode').value,
  };
}

function applyViewState(saved) {
  if (!saved) return;

  if (saved.method) {
    qs('method-select').value = saved.method;
    updateMethodParamVisibility();
  }

  if (saved.params) {
    if (saved.method === 'pca') {
      qs('pca-whiten').checked = !!saved.params.whiten;
      if (saved.params.random_state != null) qs('pca-random-state').value = saved.params.random_state;
    }
    if (saved.method === 'tsne') {
      if (saved.params.perplexity != null) qs('tsne-perplexity').value = saved.params.perplexity;
      if (saved.params.learning_rate != null) qs('tsne-learning-rate').value = saved.params.learning_rate;
      if (saved.params.n_iter != null) qs('tsne-n-iter').value = saved.params.n_iter;
      if (saved.params.init) qs('tsne-init').value = saved.params.init;
      if (saved.params.metric) qs('tsne-metric').value = saved.params.metric;
      if (saved.params.random_state != null) qs('tsne-random-state').value = saved.params.random_state;
      if (saved.params.pca_components_for_tsne_umap != null) qs('tsne-pca-pre').value = saved.params.pca_components_for_tsne_umap;
    }
    if (saved.method === 'umap') {
      if (saved.params.n_neighbors != null) qs('umap-n-neighbors').value = saved.params.n_neighbors;
      if (saved.params.min_dist != null) qs('umap-min-dist').value = saved.params.min_dist;
      if (saved.params.metric) qs('umap-metric').value = saved.params.metric;
      if (saved.params.spread != null) qs('umap-spread').value = saved.params.spread;
      if (saved.params.random_state != null) qs('umap-random-state').value = saved.params.random_state;
      if (saved.params.pca_components_for_tsne_umap != null) qs('umap-pca-pre').value = saved.params.pca_components_for_tsne_umap;
    }
  }

  if (saved.mode) {
    const radio = document.querySelector(`input[name="mode"][value="${saved.mode}"]`);
    if (radio) radio.checked = true;
    updateModeUI();
  }

  if (saved.session_value) {
    qs('session-select').value = saved.session_value;
  }

  if (saved.sample) {
    qs('sample-enabled').checked = !!saved.sample.enabled;
    if (saved.sample.max_points != null) qs('sample-max-points').value = saved.sample.max_points;
    if (saved.sample.strategy) qs('sample-strategy').value = saved.sample.strategy;
  }

  if (saved.oral_only != null) {
    qs('oral-only-toggle').checked = !!saved.oral_only;
  }

  if (saved.search_text != null) qs('search-text').value = saved.search_text;
  if (saved.search_mode) qs('search-mode').value = saved.search_mode;

  if (saved.filters) {
    FILTER_COLUMNS.forEach((col) => {
      const allowed = new Set(saved.filters[col] || []);
      document.querySelectorAll(`input[type="checkbox"][data-column="${col}"]`).forEach((cb) => {
        cb.checked = allowed.has(cb.value);
      });
    });
  }
}

function saveViewState() {
  const view = collectViewState();
  localStorage.setItem(VIEW_STATE_KEY, JSON.stringify(view));
  const encoded = encodeURIComponent(JSON.stringify(view));
  const url = new URL(window.location.href);
  url.searchParams.set('state', encoded);
  window.history.replaceState({}, '', url);
}

function loadViewState() {
  const fromQuery = new URLSearchParams(window.location.search).get('state');
  if (fromQuery) {
    try {
      return JSON.parse(decodeURIComponent(fromQuery));
    } catch {
      return null;
    }
  }

  const raw = localStorage.getItem(VIEW_STATE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function runProjection() {
  setLoading(true);
  try {
    const payload = buildProjectPayload();
    const data = await fetchJSON('/api/project', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    state.currentPoints = data.points || [];
    state.selectedPoints = getHighlightedRows();
    renderSelectionTable();
    renderPlot(state.currentPoints, data.color_by, payload.mode);
    renderStats(data.stats);
    saveViewState();

    setStatus(`Projection complete (${payload.method.toUpperCase()}).`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setLoading(false);
  }
}

function resetFilters() {
  FILTER_COLUMNS.forEach((col) => {
    document.querySelectorAll(`input[type="checkbox"][data-column="${col}"]`).forEach((cb) => {
      cb.checked = true;
    });
  });
  setStatus('Filters reset.');
}

function wireEvents() {
  qs('method-select').addEventListener('change', updateMethodParamVisibility);
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener('change', updateModeUI);
  });

  qs('run-btn').addEventListener('click', runProjection);
  qs('reset-filters-btn').addEventListener('click', resetFilters);
  qs('export-selected-btn').addEventListener('click', exportSelectedIds);
  qs('export-projection-btn').addEventListener('click', exportProjection);
  qs('nn-refresh-btn').addEventListener('click', refreshNeighbors);

  qs('share-link-btn').addEventListener('click', async () => {
    saveViewState();
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setStatus('Share link copied to clipboard.');
    } catch {
      setStatus(`Share link: ${url}`);
    }
  });

  qs('selection-table').querySelectorAll('th[data-sort]').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (state.sort.key === key) {
        state.sort.dir = state.sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        state.sort.key = key;
        state.sort.dir = 'asc';
      }
      renderSelectionTable();
    });
  });
}

async function initMeta() {
  const meta = await fetchJSON('/api/meta');
  state.meta = meta;
  initColorMaps(meta);

  const sessions = meta.available_sessions || [];
  qs('session-select').innerHTML = sessions.map((s) => `<option value="${s}">${s}</option>`).join('');

  renderFilters(meta.filters || {});
}

async function init() {
  wireEvents();
  refreshRatingsFromStorage();
  updateMethodParamVisibility();
  updateModeUI();

  try {
    await initMeta();

    const saved = loadViewState();
    if (saved) {
      applyViewState(saved);
    }

    await runProjection();
  } catch (err) {
    setStatus(err.message, true);
  }
}

window.addEventListener('storage', (event) => {
  if (event.key !== RANKING_STORAGE_KEY) return;
  refreshRatingsFromStorage();
  renderSelectionTable();
  if (state.lastNeighbors.length > 0) renderNeighbors(state.lastNeighbors);
  if (state.currentPoints.length > 0) {
    renderPlot(state.currentPoints, state.lastPlotMeta.colorBy, state.lastPlotMeta.mode);
  }
});

window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;
  const data = event.data || {};
  if (data.type === 'viz_resize') {
    requestPlotResize();
    return;
  }
  if (data.type !== 'ranking_state_update') return;
  if (data.payload?.ratings && typeof data.payload.ratings === 'object') {
    state.ratings = data.payload.ratings;
    renderSelectionTable();
    if (state.lastNeighbors.length > 0) renderNeighbors(state.lastNeighbors);
    if (state.currentPoints.length > 0) {
      renderPlot(state.currentPoints, state.lastPlotMeta.colorBy, state.lastPlotMeta.mode);
    }
  }
});

window.addEventListener('resize', () => {
  requestPlotResize();
});

init();
