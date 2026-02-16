"""
Find top-N most similar papers by cosine similarity for a given paper id.

Inputs:
  - NPZ with arrays: ids, embeddings
  - Optional CSV metadata with columns: paper_id, title

Usage:
  python find_similar_papers.py --paper-id 819-MAIN \
    --embeddings specter2_sample_embeddings.npz \
    --metadata specter2_sample_metadata.csv \
    --topk 10
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
from embedding_utils import l2_normalize, load_embeddings, load_titles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from config.paths import RAW_EMBEDDINGS_METADATA_CSV, RAW_EMBEDDINGS_NPZ

DEFAULT_EMBEDDINGS = RAW_EMBEDDINGS_NPZ
DEFAULT_METADATA = RAW_EMBEDDINGS_METADATA_CSV

KEYWORD_STOPLIST = {
  "nlp", "language", "languages", "model", "models", "learning", "deep",
  "neural", "approach", "approaches", "method", "methods", "task", "tasks",
  "study", "paper", "data", "dataset", "datasets", "analysis", "results",
  "using", "use", "based", "system", "systems",
}


def norm_text(x) -> str:
  return " ".join(str(x or "").strip().lower().split())


def parse_keywords(s, keyword_split=";", stoplist=None):
  if not s:
    return set()
  parts = [p.strip().lower() for p in str(s).split(keyword_split)]
  if stoplist:
    return {p for p in parts if p and p not in stoplist}
  return {p for p in parts if p}


def parse_secondary(s):
  if not s:
    return set()
  raw = str(s)
  parts = raw.split(",") if "," in raw else [raw]
  parts = [p.strip().lower() for p in parts]
  return {p for p in parts if p}


def is_domain_category(cat1, cat2):
  needle = "domain nlp (biomedical/clinical/legal/scientific)"
  return needle in norm_text(cat1) or needle in norm_text(cat2)


def find_header(headers, aliases):
  aliases = [norm_text(a) for a in aliases]
  for h in headers:
    hn = norm_text(h)
    for a in aliases:
      if hn == a or a in hn:
        return h
  return None


def load_extended_metadata(path):
  if not path or not os.path.exists(path):
    return None
  with open(path, "r", newline="") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []

    col_id = find_header(headers, ["paper_id", "paper number", "id", "paper id"])
    col_title = find_header(headers, ["title"]) or "title"
    col_cat1 = find_header(headers, ["category_primary", "primary category", "cat1"])
    col_cat2 = find_header(headers, ["category_secondary", "secondary category", "cat2"])
    col_kw = find_header(headers, ["keywords", "keyword"])

    if not col_id:
      return None

    meta = {}
    for row in reader:
      pid = str(row.get(col_id, "")).strip()
      if not pid:
        continue
      meta[pid] = {
        "title": str(row.get(col_title, "") or ""),
        "cat1": str(row.get(col_cat1, "") or ""),
        "cat2": str(row.get(col_cat2, "") or ""),
        "keywords": str(row.get(col_kw, "") or ""),
      }
    return meta


def rerank_score(query_pid, cand_pid, cosine, meta, weights, keyword_split, stoplist):
  q = meta.get(query_pid, {})
  c = meta.get(cand_pid, {})

  cat_q = norm_text(q.get("cat1", ""))
  cat_c = norm_text(c.get("cat1", ""))
  main_match = 1.0 if (cat_q and cat_q == cat_c) else 0.0

  sec_q = parse_secondary(q.get("cat2", ""))
  sec_c = parse_secondary(c.get("cat2", ""))
  if sec_q and sec_c:
    sec_jaccard = len(sec_q & sec_c) / len(sec_q | sec_c)
  else:
    sec_jaccard = 0.0

  kw_q = parse_keywords(q.get("keywords", ""), keyword_split, stoplist)
  kw_c = parse_keywords(c.get("keywords", ""), keyword_split, stoplist)
  if kw_q and kw_c:
    kw_jaccard = len(kw_q & kw_c) / len(kw_q | kw_c)
  else:
    kw_jaccard = 0.0

  score = (
    weights["a0"] * main_match
    + weights["a1"] * sec_jaccard
    + weights["a2"] * kw_jaccard
    + weights["a3"] * cosine
  )

  if is_domain_category(q.get("cat1", ""), q.get("cat2", "")):
    if not is_domain_category(c.get("cat1", ""), c.get("cat2", "")):
      score -= 0.35

  gate = (main_match == 0.0) and (kw_jaccard < 0.10) and (cosine < 0.90)
  return score, {
    "main_match": main_match,
    "sec_jaccard": sec_jaccard,
    "kw_jaccard": kw_jaccard,
    "gate": gate,
  }


def topk_by_cosine(ids, sims, topk):
  topk = max(1, min(topk, len(ids) - 1))
  top_idx = np.argpartition(-sims, topk)[:topk]
  top_idx = top_idx[np.argsort(-sims[top_idx])]
  return top_idx


def main():
    parser = argparse.ArgumentParser(description="Top-N similar papers by cosine similarity.")
    parser.add_argument("--paper-id", required=True, help="Paper Number / id to query")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS), help="NPZ with ids and embeddings")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="CSV with paper_id,title")
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbors to show")
    parser.add_argument("--min-cosine", type=float, default=0.80, help="Min cosine for rerank candidates")
    parser.add_argument("--candidate-multiplier", type=int, default=50, help="Candidate multiplier for rerank")
    parser.add_argument("--rerank", action="store_true", help="Enable metadata-aware reranking")
    parser.add_argument("--metadata-extended", default=None, help="CSV with categories and keywords")
    parser.add_argument("--a0", type=float, default=0.15, help="Weight for main category match")
    parser.add_argument("--a1", type=float, default=0.10, help="Weight for secondary category overlap")
    parser.add_argument("--a2", type=float, default=0.25, help="Weight for keyword overlap")
    parser.add_argument("--a3", type=float, default=1.0, help="Weight for cosine in rerank score")
    parser.add_argument("--keyword-split", default=";", help="Keyword delimiter")
    parser.add_argument("--debug-rerank", action="store_true", help="Print rerank components")
    args = parser.parse_args()

    ids, emb = load_embeddings(args.embeddings)

    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    if args.paper_id not in id_to_idx:
        raise ValueError(f"Paper id not found in embeddings: {args.paper_id}")

    emb = l2_normalize(emb)

    idx = id_to_idx[args.paper_id]
    query = emb[idx]
    sims = emb @ query

    sims[idx] = -1.0
    titles = load_titles(args.metadata)

    meta_ext = None
    if args.rerank and args.metadata_extended:
      meta_ext = load_extended_metadata(args.metadata_extended)
      if not meta_ext:
        print("Warning: extended metadata not available; falling back to cosine-only.")
        args.rerank = False

    if meta_ext:
      for pid, info in meta_ext.items():
        t = info.get("title", "")
        if t:
          titles[pid] = t

    if not args.rerank:
      top_idx = topk_by_cosine(ids, sims, args.topk)
      print(f"Query: {args.paper_id} | {titles.get(args.paper_id, '')}")
      print("Rank\tPaper ID\tCosine\tTitle")
      for rank, i in enumerate(top_idx, start=1):
        pid = ids[i]
        title = titles.get(pid, "")
        print(f"{rank}\t{pid}\t{sims[i]:.4f}\t{title}")
      return

    candidate_k = min(len(ids) - 1, args.topk * max(1, args.candidate_multiplier))
    candidate_idx = np.argpartition(-sims, candidate_k)[:candidate_k]
    candidate_idx = candidate_idx[np.argsort(-sims[candidate_idx])]

    effective_min_cosine = args.min_cosine
    if meta_ext and not is_domain_category(
      meta_ext.get(args.paper_id, {}).get("cat1", ""),
      meta_ext.get(args.paper_id, {}).get("cat2", ""),
    ):
      if effective_min_cosine < 0.88:
        effective_min_cosine = 0.88
        print("Note: non-domain query; using min_cosine=0.88 for rerank candidates.")

    filtered = [i for i in candidate_idx if sims[i] >= effective_min_cosine]
    if len(filtered) < args.topk:
      top_idx = topk_by_cosine(ids, sims, args.topk)
      print("Warning: not enough candidates above min_cosine; using cosine-only.")
      print(f"Query: {args.paper_id} | {titles.get(args.paper_id, '')}")
      print("Rank\tPaper ID\tCosine\tTitle")
      for rank, i in enumerate(top_idx, start=1):
        pid = ids[i]
        title = titles.get(pid, "")
        print(f"{rank}\t{pid}\t{sims[i]:.4f}\t{title}")
      return

    weights = {"a0": args.a0, "a1": args.a1, "a2": args.a2, "a3": args.a3}
    scored = []
    gate_hits = 0
    gate_flags = {}
    for i in filtered:
      pid = ids[i]
      score, comps = rerank_score(
        args.paper_id,
        pid,
        float(sims[i]),
        meta_ext,
        weights,
        args.keyword_split,
        KEYWORD_STOPLIST,
      )
      gate_flags[i] = comps
      if comps["gate"]:
        gate_hits += 1
      scored.append((i, score, comps))

    drop_allowed = (len(filtered) - gate_hits) >= args.topk
    reranked = []
    for i, score, comps in scored:
      if comps["gate"] and drop_allowed:
        continue
      if comps["gate"] and not drop_allowed:
        score -= 1.0
      reranked.append((i, score, comps))

    reranked.sort(key=lambda x: (-x[1], -sims[x[0]]))
    reranked = reranked[: args.topk]

    print(f"Query: {args.paper_id} | {titles.get(args.paper_id, '')}")
    print("Rank\tPaper ID\tCosine\tRerankScore\tTitle")
    for rank, (i, score, comps) in enumerate(reranked, start=1):
      pid = ids[i]
      title = titles.get(pid, "")
      print(f"{rank}\t{pid}\t{sims[i]:.4f}\t{score:.4f}\t{title}")
      if args.debug_rerank:
        print(
          f"  cat_match={comps['main_match']:.2f} sec_jaccard={comps['sec_jaccard']:.3f} "
          f"kw_jaccard={comps['kw_jaccard']:.3f}"
        )


if __name__ == "__main__":
    main()
