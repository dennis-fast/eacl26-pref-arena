from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from config.paths import (
    DOCS_DATA_DIR,
    GENERATED_SCORED_CSV,
    GENERATED_STATE_JSON,
    LEGACY_STATIC_DIR,
    LEGACY_WEB_DIR,
    RAW_EMBEDDINGS_NPZ,
    RAW_PROGRAM_CSV,
)

try:
    import umap
except Exception:  # pragma: no cover
    umap = None


CSV_PATH = RAW_PROGRAM_CSV
NPZ_PATH = RAW_EMBEDDINGS_NPZ
STATE_OUTPUT_PATH = GENERATED_STATE_JSON
SCORED_CSV_OUTPUT_PATH = GENERATED_SCORED_CSV

FILTER_COLUMNS = [
    "Type of Presentation",
    "Attendance Type",
    "Room Location",
    "Session",
]

RENAME_TO_CANONICAL = {
    "paper_id": ["paper_id", "Paper number", "Paper Number", "Paper ID", "id"],
    "title": ["Title", "title"],
    "type_presentation": ["Type of Presentation"],
    "attendance_type": ["Attendance Type"],
    "room_location": ["Room Location"],
    "session": ["Session"],
}


@dataclass
class DataStore:
    df: pd.DataFrame
    vectors: np.ndarray
    paper_to_index: dict[str, int]
    projection_cache: dict[str, np.ndarray]


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = set(df.columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def _to_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("Unknown").replace({"nan": "Unknown", "": "Unknown"})


def _non_virtual_mask(df: pd.DataFrame) -> pd.Series:
    attendance = df["Attendance Type"].astype(str).str.strip().str.lower()
    room = df["Room Location"].astype(str).str.strip().str.lower()
    session = df["Session"].astype(str).str.strip().str.lower()

    is_virtual = (
        attendance.eq("virtual")
        | room.eq("zoom")
        | session.str.contains(r"\bvirtual\b", na=False)
    )
    return ~is_virtual


def _normalize_learning_rate(value: Any) -> str | float:
    if isinstance(value, str):
        if value.lower() == "auto":
            return "auto"
        try:
            return float(value)
        except ValueError:
            return "auto"
    if value is None:
        return "auto"
    return float(value)


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _parse_search_terms(search_text: str) -> list[str]:
    if not search_text:
        return []
    return [term.strip() for term in search_text.split(";") if term.strip()]


def _search_regex_from_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    escaped_terms = [re.escape(term) for term in terms]
    return "|".join(escaped_terms)


def _build_subset_signature(
    method: str,
    params: dict[str, Any],
    paper_ids: list[str],
) -> str:
    key_payload = {
        "method": method,
        "params": params,
        "paper_ids": paper_ids,
    }
    blob = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _resolve_tsne_iter_param(tsne_params: dict[str, Any]) -> dict[str, Any]:
    init_code = TSNE.__init__.__code__.co_varnames
    if "max_iter" in init_code:
        tsne_params["max_iter"] = int(tsne_params.pop("n_iter"))
    else:
        tsne_params["n_iter"] = int(tsne_params["n_iter"])
    return tsne_params


def _sample_indices(
    subset_df: pd.DataFrame,
    enabled: bool,
    max_points: int,
    strategy: str,
    random_state: int,
) -> pd.Index:
    if not enabled or max_points <= 0 or len(subset_df) <= max_points:
        return subset_df.index

    rng = np.random.RandomState(random_state)

    if strategy == "stratified_by_session":
        sampled_parts: list[pd.DataFrame] = []
        grouped = subset_df.groupby("Session", sort=False)
        total = len(subset_df)

        for _, group in grouped:
            frac = len(group) / total
            n_take = max(1, int(round(max_points * frac)))
            n_take = min(n_take, len(group))
            sampled_parts.append(group.sample(n=n_take, random_state=rng.randint(0, 2**31 - 1)))

        sampled = pd.concat(sampled_parts, axis=0)

        if len(sampled) > max_points:
            sampled = sampled.sample(n=max_points, random_state=rng.randint(0, 2**31 - 1))
        elif len(sampled) < max_points:
            remaining = subset_df.loc[~subset_df.index.isin(sampled.index)]
            if len(remaining) > 0:
                n_fill = min(max_points - len(sampled), len(remaining))
                fill = remaining.sample(n=n_fill, random_state=rng.randint(0, 2**31 - 1))
                sampled = pd.concat([sampled, fill], axis=0)

        return sampled.index

    sampled = subset_df.sample(n=max_points, random_state=rng.randint(0, 2**31 - 1))
    return sampled.index


def _project_vectors(method: str, params: dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        whiten = bool(params.get("whiten", False))
        model = PCA(n_components=2, whiten=whiten, random_state=int(params.get("random_state", 42)))
        return model.fit_transform(vectors)

    pca_pre_components = int(params.get("pca_components_for_tsne_umap", 50))
    pca_pre_components = max(2, min(pca_pre_components, vectors.shape[0] - 1, vectors.shape[1]))
    vectors_for_non_linear = vectors
    if vectors.shape[1] > pca_pre_components:
        pre = PCA(n_components=pca_pre_components, random_state=int(params.get("random_state", 42)))
        vectors_for_non_linear = pre.fit_transform(vectors)

    if method == "tsne":
        learning_rate = _normalize_learning_rate(params.get("learning_rate", "auto"))
        tsne_params = {
            "n_components": 2,
            "perplexity": float(params.get("perplexity", 30.0)),
            "learning_rate": learning_rate,
            "n_iter": int(params.get("n_iter", 1000)),
            "init": str(params.get("init", "pca")),
            "metric": str(params.get("metric", "cosine")),
            "random_state": int(params.get("random_state", 42)),
        }
        tsne_params = _resolve_tsne_iter_param(tsne_params)
        try:
            model = TSNE(**tsne_params)
            return model.fit_transform(vectors_for_non_linear)
        except Exception as exc:
            if tsne_params.get("metric") == "cosine":
                tsne_params["metric"] = "euclidean"
                model = TSNE(**tsne_params)
                return model.fit_transform(vectors_for_non_linear)
            raise RuntimeError(f"t-SNE failed: {exc}") from exc

    if method == "umap":
        if umap is None:
            raise RuntimeError(
                "UMAP is unavailable. Install umap-learn (pip install umap-learn)."
            )
        model = umap.UMAP(
            n_components=2,
            n_neighbors=int(params.get("n_neighbors", 15)),
            min_dist=float(params.get("min_dist", 0.1)),
            metric=str(params.get("metric", "cosine")),
            spread=float(params.get("spread", 1.0)),
            random_state=int(params.get("random_state", 42)),
        )
        return model.fit_transform(vectors_for_non_linear)

    raise ValueError(f"Unsupported method: {method}")


def _load_data() -> DataStore:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    df = pd.read_csv(CSV_PATH)

    canonical_map: dict[str, str] = {}
    for canonical_name, candidates in RENAME_TO_CANONICAL.items():
        found = _find_first_column(df, candidates)
        if found:
            canonical_map[canonical_name] = found

    if "paper_id" not in canonical_map:
        raise ValueError(
            "No paper_id column found in CSV. Expected one of: "
            f"{RENAME_TO_CANONICAL['paper_id']}"
        )

    if "title" not in canonical_map:
        raise ValueError("No title column found in CSV. Expected 'Title' or 'title'.")

    for col in FILTER_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"

    df = df.rename(
        columns={
            canonical_map["paper_id"]: "paper_id",
            canonical_map["title"]: "Title",
        }
    )

    for col in ["paper_id", "Title", *FILTER_COLUMNS]:
        df[col] = _to_str_series(df[col])

    df = df[_non_virtual_mask(df)].copy()

    npz = np.load(NPZ_PATH, allow_pickle=True)
    if "ids" not in npz.files or "embeddings" not in npz.files:
        raise ValueError("NPZ must contain 'ids' and 'embeddings' arrays.")

    ids = npz["ids"]
    embeddings = npz["embeddings"]

    ids = np.asarray(ids).reshape(-1)
    embeddings = np.asarray(embeddings)

    if len(ids) != len(embeddings):
        raise ValueError("NPZ arrays ids and embeddings must have the same length.")

    ids = np.array([str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in ids])

    emb_df = pd.DataFrame({"paper_id": ids, "_emb_idx": np.arange(len(ids))})

    merged = df.merge(emb_df, on="paper_id", how="inner")
    merged = merged.reset_index(drop=True)

    vector_matrix = embeddings[merged["_emb_idx"].to_numpy()]
    vector_matrix = normalize(vector_matrix, norm="l2")

    merged = merged[["paper_id", "Title", *FILTER_COLUMNS, "_emb_idx"]].copy()

    paper_to_index = {pid: i for i, pid in enumerate(merged["paper_id"].tolist())}

    return DataStore(
        df=merged,
        vectors=vector_matrix,
        paper_to_index=paper_to_index,
        projection_cache={},
    )


data_store = _load_data()

app = Flask(__name__, static_folder=str(LEGACY_STATIC_DIR), static_url_path="/static")


def _build_preference_vector_from_ratings(ratings: dict[str, Any]) -> np.ndarray | None:
    if not isinstance(ratings, dict) or not ratings:
        return None

    rated_with_mu: list[tuple[str, float]] = []
    for paper_id, payload in ratings.items():
        if paper_id not in data_store.paper_to_index:
            continue
        mu = payload.get("mu") if isinstance(payload, dict) else None
        if mu is None:
            continue
        try:
            mu_value = float(mu)
        except (TypeError, ValueError):
            continue
        rated_with_mu.append((paper_id, mu_value))

    if not rated_with_mu:
        return None

    baseline_mu = 1500.0
    weighted_items: list[tuple[int, float]] = []
    for paper_id, mu_value in rated_with_mu:
        idx = data_store.paper_to_index[paper_id]
        weight = max(0.0, mu_value - baseline_mu)
        weighted_items.append((idx, weight))

    if all(weight <= 0.0 for _, weight in weighted_items):
        ranked_ids = sorted(rated_with_mu, key=lambda x: x[1], reverse=True)
        count = len(ranked_ids)
        weighted_items = [
            (data_store.paper_to_index[pid], float(count - i) / float(count))
            for i, (pid, _) in enumerate(ranked_ids)
        ]

    pref = np.zeros(data_store.vectors.shape[1], dtype=np.float64)
    total_weight = 0.0
    for idx, weight in weighted_items:
        if weight <= 0.0:
            continue
        pref += data_store.vectors[idx] * weight
        total_weight += weight

    if total_weight <= 0.0:
        return None

    norm = np.linalg.norm(pref)
    if norm <= 0:
        return None
    return (pref / norm).astype(np.float64)


@app.get("/")
def root():
    return send_from_directory(LEGACY_WEB_DIR, "index.html")


@app.get("/viz")
def viz_root():
    return send_from_directory(LEGACY_STATIC_DIR, "index.html")


@app.get("/data/<path:filename>")
def data_static(filename: str):
    return send_from_directory(DOCS_DATA_DIR, filename)


@app.get("/api/meta")
def api_meta():
    df = data_store.df
    unique_values = {
        col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in FILTER_COLUMNS
    }

    return jsonify(
        {
            "n_total": int(len(df)),
            "filters": unique_values,
            "available_sessions": unique_values.get("Session", []),
            "default_columns": {
                "title": "Title",
                "paper_id": "paper_id",
                "color_by_all": "Session",
                "color_by_session": "Room Location",
            },
        }
    )


@app.post("/api/project")
def api_project():
    started = time.perf_counter()

    payload = request.get_json(silent=True) or {}
    method = str(payload.get("method", "pca")).lower()
    params = payload.get("params", {}) or {}
    mode = str(payload.get("mode", "all")).lower()
    session_value = payload.get("session_value")
    filters = payload.get("filters", {}) or {}
    sample_cfg = payload.get("sample", {}) or {}
    oral_only = bool(payload.get("oral_only", False))
    search_text = str(payload.get("search_text", "")).strip()
    search_mode = str(payload.get("search_mode", "highlight")).lower()
    search_terms = _parse_search_terms(search_text)
    search_regex = _search_regex_from_terms(search_terms)

    df = data_store.df
    mask = pd.Series(True, index=df.index)

    for col in FILTER_COLUMNS:
        allowed_values = filters.get(col)
        if allowed_values is None:
            continue
        allowed = set(str(v) for v in allowed_values)
        mask &= df[col].isin(allowed)

    if mode == "session":
        if not session_value:
            return _json_error("session_value is required when mode='session'.")
        mask &= df["Session"] == str(session_value)

    if oral_only:
        is_oral = df["Type of Presentation"].astype(str).str.contains(r"\boral\b", case=False, na=False)
        is_in_person = df["Attendance Type"].astype(str).str.strip().str.lower().eq("in-person")
        mask &= is_oral & is_in_person

    if search_regex and search_mode == "filter":
        mask &= df["Title"].str.contains(search_regex, case=False, na=False, regex=True)

    filtered_df = df[mask].copy()
    n_filtered = int(len(filtered_df))
    if n_filtered == 0:
        compute_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(
            {
                "points": [],
                "color_by": "Session" if mode == "all" else "Room Location",
                "legend_values": [],
                "stats": {
                    "n_total": int(len(df)),
                    "n_filtered": 0,
                    "n_returned": 0,
                    "compute_ms": compute_ms,
                },
            }
        )

    random_state = int(params.get("random_state", 42))
    sample_enabled = bool(sample_cfg.get("enabled", False))
    max_points = int(sample_cfg.get("max_points", n_filtered))
    sample_strategy = str(sample_cfg.get("strategy", "random"))

    sampled_index = _sample_indices(
        filtered_df,
        enabled=sample_enabled,
        max_points=max_points,
        strategy=sample_strategy,
        random_state=random_state,
    )

    sampled_df = filtered_df.loc[sampled_index].copy()
    sampled_df = sampled_df.reset_index(drop=True)

    source_rows = filtered_df.loc[sampled_index]
    vector_positions = source_rows.index.to_numpy()
    vectors = data_store.vectors[vector_positions]

    cache_key = _build_subset_signature(
        method=method,
        params=params,
        paper_ids=sampled_df["paper_id"].tolist(),
    )

    if cache_key in data_store.projection_cache:
        projected = data_store.projection_cache[cache_key]
    else:
        try:
            projected = _project_vectors(method, params, vectors)
        except Exception as exc:
            return _json_error(str(exc), status=400)
        data_store.projection_cache[cache_key] = projected

    sampled_df["x"] = projected[:, 0].astype(float)
    sampled_df["y"] = projected[:, 1].astype(float)
    if search_regex:
        sampled_df["matched"] = sampled_df["Title"].str.contains(
            search_regex,
            case=False,
            na=False,
            regex=True,
        )
    else:
        sampled_df["matched"] = False

    color_field = "Session" if mode == "all" else "Room Location"

    points = []
    for _, row in sampled_df.iterrows():
        points.append(
            {
                "paper_id": row["paper_id"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "title": row["Title"],
                "session": row["Session"],
                "room_location": row["Room Location"],
                "type_presentation": row["Type of Presentation"],
                "attendance_type": row["Attendance Type"],
                "color_value": row[color_field],
                "matched": bool(row["matched"]),
            }
        )

    legend_values = sorted(sampled_df[color_field].dropna().astype(str).unique().tolist())

    compute_ms = round((time.perf_counter() - started) * 1000, 2)

    return jsonify(
        {
            "points": points,
            "color_by": color_field,
            "legend_values": legend_values,
            "stats": {
                "n_total": int(len(df)),
                "n_filtered": n_filtered,
                "n_returned": int(len(sampled_df)),
                "compute_ms": compute_ms,
            },
        }
    )


@app.post("/api/nn")
def api_nn():
    payload = request.get_json(silent=True) or {}
    paper_id = str(payload.get("paper_id", "")).strip()
    k = int(payload.get("k", 10))
    current_ids = payload.get("current_ids") or []

    if not paper_id:
        return _json_error("paper_id is required")

    if paper_id not in data_store.paper_to_index:
        return _json_error(f"paper_id not found: {paper_id}", status=404)

    df = data_store.df

    if current_ids:
        current_id_set = set(str(v) for v in current_ids)
        subset_idx = df.index[df["paper_id"].isin(current_id_set)].to_numpy()
    else:
        subset_idx = df.index.to_numpy()

    if len(subset_idx) == 0:
        return jsonify({"neighbors": []})

    query_idx = data_store.paper_to_index[paper_id]
    query_vec = data_store.vectors[query_idx]

    subset_vectors = data_store.vectors[subset_idx]
    sims = subset_vectors @ query_vec

    subset_df = df.loc[subset_idx].copy()
    subset_df["cosine"] = sims

    subset_df = subset_df[subset_df["paper_id"] != paper_id]
    subset_df = subset_df.sort_values("cosine", ascending=False).head(max(1, k))

    neighbors = [
        {
            "paper_id": row.paper_id,
            "title": row.Title,
            "cosine": float(row.cosine),
            "session": row.Session,
            "room_location": row["Room Location"],
            "type_presentation": row["Type of Presentation"],
            "attendance_type": row["Attendance Type"],
        }
        for _, row in subset_df.iterrows()
    ]

    return jsonify({"neighbors": neighbors})


@app.post("/api/topic_match")
def api_topic_match():
    payload = request.get_json(silent=True) or {}
    ratings = payload.get("ratings") or {}
    slots = payload.get("slots") or []
    top_k = int(payload.get("top_k", 2))
    top_k = max(1, min(top_k, 10))

    if not isinstance(slots, list):
        return _json_error("'slots' must be a list.")

    pref_vec = _build_preference_vector_from_ratings(ratings)

    results: list[dict[str, Any]] = []
    for item in slots:
        if not isinstance(item, dict):
            continue
        slot_key = str(item.get("slot_key", "")).strip()
        paper_ids_raw = item.get("paper_ids") or []
        if not isinstance(paper_ids_raw, list):
            paper_ids_raw = []

        valid_ids = [
            str(pid).strip() for pid in paper_ids_raw
            if str(pid).strip() in data_store.paper_to_index
        ]

        if pref_vec is None or not valid_ids:
            results.append(
                {
                    "slot_key": slot_key,
                    "topic_match": 0.0,
                    "n_candidates": int(len(valid_ids)),
                    "n_used": 0,
                }
            )
            continue

        vec_idx = [data_store.paper_to_index[pid] for pid in valid_ids]
        sims = data_store.vectors[vec_idx] @ pref_vec
        sims_sorted = np.sort(sims)[::-1]
        used = min(top_k, len(sims_sorted))
        score = float(np.mean(sims_sorted[:used])) if used > 0 else 0.0

        results.append(
            {
                "slot_key": slot_key,
                "topic_match": score,
                "n_candidates": int(len(valid_ids)),
                "n_used": int(used),
            }
        )

    return jsonify(
        {
            "ok": True,
            "has_preference_signal": pref_vec is not None,
            "top_k": top_k,
            "results": results,
        }
    )


@app.post("/api/save_state")
def api_save_state():
    payload = request.get_json(silent=True) or {}
    state_obj = payload.get("state")
    if state_obj is None:
        return _json_error("Missing 'state' in request body.")

    try:
        STATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATE_OUTPUT_PATH.open("w", encoding="utf-8") as fp:
            json.dump(state_obj, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
    except Exception as exc:
        return _json_error(f"Failed to save state: {exc}", status=500)

    return jsonify({"ok": True, "path": str(STATE_OUTPUT_PATH.name)})


@app.post("/api/save_scored_csv")
def api_save_scored_csv():
    payload = request.get_json(silent=True) or {}
    csv_text = payload.get("csv_text")
    if not isinstance(csv_text, str):
        return _json_error("Missing 'csv_text' string in request body.")

    try:
        SCORED_CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCORED_CSV_OUTPUT_PATH.open("w", encoding="utf-8") as fp:
            fp.write(csv_text)
            if not csv_text.endswith("\n"):
                fp.write("\n")
    except Exception as exc:
        return _json_error(f"Failed to save scored CSV: {exc}", status=500)

    return jsonify({"ok": True, "path": str(SCORED_CSV_OUTPUT_PATH.name)})


@app.get("/<path:path>")
def static_proxy(path: str):
    root_target = LEGACY_WEB_DIR / path
    if root_target.exists() and root_target.is_file():
        return send_from_directory(LEGACY_WEB_DIR, path)

    static_dir = Path(app.static_folder)
    target = static_dir / path
    if target.exists() and target.is_file():
        return send_from_directory(static_dir, path)
    return send_from_directory(LEGACY_WEB_DIR, "index.html")


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
