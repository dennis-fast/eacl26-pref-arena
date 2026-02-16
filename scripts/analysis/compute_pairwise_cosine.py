"""
Compute a full pairwise cosine similarity matrix from embeddings.

Input:
  - NPZ with arrays: ids, embeddings

Output:
  - .npy memmap matrix (float32) of shape (N, N)
  - CSV with ids in order

Usage:
  python compute_pairwise_cosine.py --input specter2_embeddings.npz --output specter2_pairwise_cosine.npy
"""

import argparse
import sys
import numpy as np
from embedding_utils import l2_normalize, load_embeddings, write_ids_csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from config.paths import PROCESSED_PAIRWISE_COSINE_NPY, RAW_EMBEDDINGS_NPZ

DEFAULT_INPUT = RAW_EMBEDDINGS_NPZ
DEFAULT_OUTPUT = PROCESSED_PAIRWISE_COSINE_NPY


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise cosine similarity matrix.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input NPZ with ids and embeddings")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .npy matrix path")
    parser.add_argument("--chunk-size", type=int, default=256, help="Row chunk size for block compute")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    ids, emb = load_embeddings(args.input)

    emb = l2_normalize(emb)

    n = emb.shape[0]
    out = np.memmap(args.output, dtype="float32", mode="w+", shape=(n, n))

    for i in range(0, n, args.chunk_size):
        j = min(n, i + args.chunk_size)
        sims = emb[i:j] @ emb.T
        out[i:j] = sims.astype(np.float32, copy=False)
        out.flush()
        print(f"Rows {i}:{j} / {n}")

    out.flush()

    ids_csv = str(Path(args.output).with_suffix("")) + "_ids.csv"
    write_ids_csv(ids_csv, ids)

    print(f"Saved cosine matrix: {args.output}")
    print(f"Saved ids index:     {ids_csv}")


if __name__ == "__main__":
    main()
