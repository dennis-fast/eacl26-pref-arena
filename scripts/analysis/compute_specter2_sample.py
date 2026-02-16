"""
Compute SPECTER2 embeddings for a random subset of X papers from your CSV.

Input:
  - EACL_2026_program_categorized.csv (or your original program CSV)
  - Requires columns: Title, Abstract, and an ID column (Paper number / id / paper id)

Output:
  - specter2_sample_embeddings.npz  (ids + embeddings)
  - specter2_sample_embeddings.csv  (ids + title + (optional) for inspection)

Install (once):
    pip install -U "transformers>=4.48.2" "peft>=0.18.1" pandas numpy torch

Notes:
    - Loads the SPECTER2 adapter on top of the base model (PEFT).
    - Uses title + abstract concatenated as recommended (scientific-doc embeddings).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import (
    RAW_EMBEDDINGS_METADATA_CSV,
    RAW_EMBEDDINGS_NPZ,
    RAW_PROGRAM_CATEGORIZED_CSV,
)
from embedding_utils import (
    find_col,
    build_doc,
    clean_text,
    encode_texts,
    load_specter2_model,
)
import argparse

DEFAULT_CSV = RAW_PROGRAM_CATEGORIZED_CSV
DEFAULT_OUT_NPZ = RAW_EMBEDDINGS_NPZ
DEFAULT_OUT_CSV = RAW_EMBEDDINGS_METADATA_CSV

def main():
    parser = argparse.ArgumentParser(description="Compute SPECTER2 embeddings from a CSV.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Input CSV path")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of random papers (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out-npz", default=str(DEFAULT_OUT_NPZ), help="Output NPZ path")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Output metadata CSV")
    parser.add_argument("--base-model", default="allenai/specter2_base", help="Base model name")
    parser.add_argument("--adapter-model", default="allenai/specter2", help="PEFT adapter model name")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    parser.add_argument("--device", default="auto", help="Device: auto|mps|cuda|cpu")
    args = parser.parse_args()
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    headers = list(df.columns)

    col_id = find_col(headers, ["paper number", "id", "paper id", "submission id", "program id"])
    col_title = find_col(headers, ["title", "paper title"])
    col_abs = find_col(headers, ["abstract", "paper abstract"])

    missing = [name for name, col in [("id", col_id), ("title", col_title), ("abstract", col_abs)] if col is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {headers}")

    df2 = df.copy()
    df2[col_title] = df2[col_title].map(clean_text)
    df2[col_abs] = df2[col_abs].map(clean_text)
    df2 = df2[(df2[col_title].str.len() > 0) | (df2[col_abs].str.len() > 0)].reset_index(drop=True)

    if args.sample_size <= 0:
        sample = df2
    else:
        sample_size = min(args.sample_size, len(df2))
        sample = df2.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    tokenizer, model, used_adapter = load_specter2_model(args.base_model, args.adapter_model, allow_fallback=True)
    if used_adapter:
        print(f"Loaded adapter: {args.adapter_model}")
        if hasattr(model, "active_adapters"):
            print(f"Active adapters: {model.active_adapters}")
    else:
        print("Warning: adapter load failed; using base model only.")

    paper_ids = sample[col_id].astype(str).tolist()
    sep = f" {tokenizer.sep_token} " if tokenizer.sep_token else " [SEP] "
    docs = [build_doc(t, a, sep_token=sep) for t, a in zip(sample[col_title], sample[col_abs])]

    emb = encode_texts(docs, tokenizer, model, batch_size=args.batch_size, device=args.device, pooling="cls")
    emb = np.asarray(emb, dtype=np.float32)

    np.savez_compressed(args.out_npz, ids=np.array(paper_ids, dtype=object), embeddings=emb)

    sample_out = sample[[col_id, col_title]].copy()
    sample_out.rename(columns={col_id: "paper_id", col_title: "title"}, inplace=True)
    sample_out.to_csv(args.out_csv, index=False)

    print(f"Saved embeddings: {args.out_npz}  (shape={emb.shape})")
    print(f"Saved metadata:   {args.out_csv}")


if __name__ == "__main__":
    main()