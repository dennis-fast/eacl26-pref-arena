const state = {
  points: [],
  ratings: {},
};

const RANKING_STORAGE_KEY = "eacl_pref_arena_state_v1";
const FIXED_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
  '#bcbd22', '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd',
  '#31a354', '#756bb1', '#636363', '#e6550d',
];

function el(id) {
  return document.getElementById(id);
}

function getColor(key) {
  const hash = [...String(key || '')].reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return FIXED_COLORS[Math.abs(hash) % FIXED_COLORS.length];
}

function readRatings() {
  const raw = localStorage.getItem(RANKING_STORAGE_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed?.ratings || {};
  } catch {
    return {};
  }
}

function filterPoints() {
  const search = el('search').value.trim().toLowerCase();
  const oralOnly = el('oralOnly').checked;
  return state.points.filter((p) => {
    if (oralOnly) {
      const isOral = /\boral\b/i.test(String(p.type_presentation || ''));
      const isInPerson = String(p.attendance_type || '').trim().toLowerCase() === 'in-person';
      if (!(isOral && isInPerson)) return false;
    }
    if (!search) return true;
    const blob = `${p.paper_id} ${p.title}`.toLowerCase();
    return blob.includes(search);
  });
}

function buildOpacityMap(points) {
  const mus = points.map((p) => Number(state.ratings?.[p.paper_id]?.mu ?? 1500));
  if (mus.length === 0) return new Map();
  const min = Math.min(...mus);
  const max = Math.max(...mus);
  const span = Math.max(1e-9, max - min);
  const out = new Map();
  points.forEach((p, i) => {
    const norm = (mus[i] - min) / span;
    out.set(p.paper_id, 0.18 + 0.74 * norm);
  });
  return out;
}

function render() {
  const colorBy = el('colorBy').value;
  const points = filterPoints();
  el('status').textContent = `${points.length} points · precomputed PCA projection`;

  const groups = new Map();
  points.forEach((p) => {
    const key = p[colorBy] || 'Unknown';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  });

  const opacityById = buildOpacityMap(points);
  const traces = [...groups.entries()].map(([key, items]) => ({
    type: 'scattergl',
    mode: 'markers',
    name: String(key),
    x: items.map((p) => p.x),
    y: items.map((p) => p.y),
    customdata: items.map((p) => [p.paper_id, p.title, p.session, p.room_location]),
    hovertemplate: '<b>%{customdata[1]}</b><br>%{customdata[0]}<br>Session: %{customdata[2]}<br>Room: %{customdata[3]}<extra></extra>',
    marker: {
      size: 8,
      color: getColor(key),
      opacity: items.map((p) => opacityById.get(p.paper_id) ?? 0.7),
    },
  }));

  Plotly.newPlot(
    el('plot'),
    traces,
    {
      title: `2D projection (color by ${colorBy === 'session' ? 'Session' : 'Room Location'})`,
      margin: { t: 46, r: 10, b: 90, l: 46 },
      xaxis: { title: 'x' },
      yaxis: { title: 'y' },
      legend: { orientation: 'h', x: 0, xanchor: 'left', y: -0.2, yanchor: 'top' },
      hovermode: 'closest',
    },
    { responsive: true, displaylogo: false }
  );
}

async function init() {
  const res = await fetch('../data/projection_pca.json', { cache: 'no-store' });
  const data = await res.json();
  state.points = data.points || [];
  state.ratings = readRatings();

  ['colorBy', 'search', 'oralOnly'].forEach((id) => {
    el(id).addEventListener('input', render);
    el(id).addEventListener('change', render);
  });

  window.addEventListener('storage', (event) => {
    if (event.key !== RANKING_STORAGE_KEY) return;
    state.ratings = readRatings();
    render();
  });

  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin) return;
    const dataMsg = event.data || {};
    if (dataMsg.type === 'ranking_state_update' && dataMsg.payload?.ratings) {
      state.ratings = dataMsg.payload.ratings;
      render();
      return;
    }
    if (dataMsg.type === 'viz_resize') {
      Plotly.Plots.resize(el('plot'));
    }
  });

  render();
}

init().catch((err) => {
  el('status').textContent = `Failed to load projection data: ${err.message}`;
});
