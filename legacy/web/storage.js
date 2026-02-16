// storage.js
const KEY = "eacl_pref_arena_state_v1";

export function saveState(state) {
  localStorage.setItem(KEY, JSON.stringify(state));
}

export function loadState() {
  const raw = localStorage.getItem(KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

export function clearState() {
  localStorage.removeItem(KEY);
}
