import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

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
    df = pd.read_csv(CSV_PATH)
    for col in ["Paper number", "Title", "Session", "Room Location", "Type of Presentation", "Attendance Type"]:
        if col not in df.columns:
            df[col] = ""

    attendance = df["Attendance Type"].astype(str).str.strip().str.lower()
    room = df["Room Location"].astype(str).str.strip().str.lower()
    session = df["Session"].astype(str).str.strip().str.lower()
    is_virtual = attendance.eq("virtual") | room.eq("zoom") | session.str.contains(r"\bvirtual\b", na=False)

    df = df.loc[~is_virtual].copy()
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
    xy = PCA(n_components=2, random_state=42).fit_transform(vectors)

    merged["x"] = xy[:, 0].astype(float)
    merged["y"] = xy[:, 1].astype(float)

    points = []
    for _, row in merged.iterrows():
        points.append(
            {
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "session": str(row.get("Session", "")),
                "room_location": str(row.get("Room Location", "")),
                "type_presentation": str(row.get("Type of Presentation", "")),
                "attendance_type": str(row.get("Attendance Type", "")),
                "x": float(row["x"]),
                "y": float(row["y"]),
            }
        )

    payload = {
        "method": "pca",
        "n_points": len(points),
        "points": points,
        "sessions": sorted(set(str(v) for v in merged["Session"].dropna().tolist())),
        "rooms": sorted(set(str(v) for v in merged["Room Location"].dropna().tolist())),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(points)} points")


if __name__ == "__main__":
    main()
