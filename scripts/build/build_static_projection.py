import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

try:
    import umap
except Exception:
    umap = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import (
    DOCS_PROJECTION_PCA_JSON,
    RAW_EMBEDDINGS_NPZ,
    RAW_PROGRAM_CATEGORIZED_CSV,
)

CSV_PATH = RAW_PROGRAM_CATEGORIZED_CSV
NPZ_PATH = RAW_EMBEDDINGS_NPZ
OUT_PATH = DOCS_PROJECTION_PCA_JSON


def main() -> None:
    started = time.perf_counter()
    df = pd.read_csv(CSV_PATH)
    for col in ["Paper number", "Title", "Session", "Room Location", "Type of Presentation", "Attendance Type"]:
        if col not in df.columns:
            df[col] = ""

    df = df.rename(columns={"Paper number": "paper_id", "Title": "title"})
    df["paper_id"] = df["paper_id"].astype(str)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    ids = np.asarray(npz["ids"]).reshape(-1)
    embeddings = np.asarray(npz["embeddings"])
    ids = np.array([str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in ids])

    emb_df = pd.DataFrame({"paper_id": ids, "_emb_idx": np.arange(len(ids))})
    merged = df.merge(emb_df, on="paper_id", how="inner").reset_index(drop=True)

    vectors = embeddings[merged["_emb_idx"].to_numpy()]
    vectors = normalize(vectors, norm="l2")
    pca_xy = PCA(n_components=2, random_state=42).fit_transform(vectors)

    pca_pre_components = max(2, min(50, vectors.shape[0] - 1, vectors.shape[1]))
    non_linear_vectors = vectors
    if vectors.shape[1] > pca_pre_components:
        non_linear_vectors = PCA(n_components=pca_pre_components, random_state=42).fit_transform(vectors)

    tsne_model = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=42,
    )
    tsne_init_vars = TSNE.__init__.__code__.co_varnames
    if "max_iter" in tsne_init_vars:
        tsne_model.max_iter = 1000
    else:
        tsne_model.n_iter = 1000
    tsne_xy = tsne_model.fit_transform(non_linear_vectors)

    if umap is not None:
        try:
            umap_xy = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                spread=1.0,
                random_state=42,
            ).fit_transform(non_linear_vectors)
        except Exception:
            umap_xy = pca_xy.copy()
    else:
        umap_xy = pca_xy.copy()

    points_meta = []
    for _, row in merged.iterrows():
        points_meta.append(
            {
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "session": str(row.get("Session", "")),
                "room_location": str(row.get("Room Location", "")),
                "type_presentation": str(row.get("Type of Presentation", "")),
                "attendance_type": str(row.get("Attendance Type", "")),
            }
        )

    def coords_payload(xy: np.ndarray) -> list[dict[str, float | str]]:
        result = []
        for idx, row in merged.iterrows():
            result.append(
                {
                    "paper_id": str(row["paper_id"]),
                    "x": float(xy[idx, 0]),
                    "y": float(xy[idx, 1]),
                }
            )
        return result

    methods = {
        "pca": coords_payload(pca_xy),
        "tsne": coords_payload(tsne_xy),
        "umap": coords_payload(umap_xy),
    }

    cosine = vectors @ vectors.T
    neighbors: dict[str, list[dict[str, float | str]]] = {}
    max_neighbors = min(80, max(0, len(merged) - 1))
    for i, row in merged.iterrows():
        paper_id = str(row["paper_id"])
        order = np.argsort(-cosine[i])
        rows = []
        for j in order:
            if j == i:
                continue
            j_row = merged.iloc[int(j)]
            rows.append(
                {
                    "paper_id": str(j_row["paper_id"]),
                    "title": str(j_row["title"]),
                    "cosine": float(cosine[i, j]),
                    "session": str(j_row.get("Session", "")),
                    "room_location": str(j_row.get("Room Location", "")),
                    "type_presentation": str(j_row.get("Type of Presentation", "")),
                    "attendance_type": str(j_row.get("Attendance Type", "")),
                }
            )
            if len(rows) >= max_neighbors:
                break
        neighbors[paper_id] = rows

    points = []
    for idx, row in merged.iterrows():
        points.append(
            {
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "session": str(row.get("Session", "")),
                "room_location": str(row.get("Room Location", "")),
                "type_presentation": str(row.get("Type of Presentation", "")),
                "attendance_type": str(row.get("Attendance Type", "")),
                "x": float(pca_xy[idx, 0]),
                "y": float(pca_xy[idx, 1]),
            }
        )

    payload = {
        "version": 2,
        "method": "pca",
        "n_points": len(points),
        "points": points,
        "points_meta": points_meta,
        "methods": methods,
        "neighbors": neighbors,
        "default_columns": {
            "title": "title",
            "paper_id": "paper_id",
            "color_by_all": "Session",
            "color_by_session": "Room Location",
        },
        "filters": {
            "Type of Presentation": sorted(set(str(v) for v in merged["Type of Presentation"].dropna().tolist())),
            "Attendance Type": sorted(set(str(v) for v in merged["Attendance Type"].dropna().tolist())),
            "Room Location": sorted(set(str(v) for v in merged["Room Location"].dropna().tolist())),
            "Session": sorted(set(str(v) for v in merged["Session"].dropna().tolist())),
        },
        "available_sessions": sorted(set(str(v) for v in merged["Session"].dropna().tolist())),
        "sessions": sorted(set(str(v) for v in merged["Session"].dropna().tolist())),
        "rooms": sorted(set(str(v) for v in merged["Room Location"].dropna().tolist())),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    print(f"Wrote {OUT_PATH} with {len(points)} points in {elapsed_ms} ms")


if __name__ == "__main__":
    main()
