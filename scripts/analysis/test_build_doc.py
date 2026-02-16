import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import GENERATED_EMBEDDING_INPUT_TXT, RAW_PROGRAM_CATEGORIZED_CSV

CSV_PATH = RAW_PROGRAM_CATEGORIZED_CSV
PAPER_ID = "819-MAIN"
OUT_PATH = GENERATED_EMBEDDING_INPUT_TXT


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def find_col(headers, aliases):
    aliases = [norm(a) for a in aliases]
    for c in headers:
        cn = norm(c)
        for a in aliases:
            if cn == a or a in cn:
                return c
    return None


def clean_text(x) -> str:
    if x is None:
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_doc(title: str, abstract: str) -> str:
    t = clean_text(title)
    a = clean_text(abstract)
    if not a:
        return t
    return f"{t} [SEP] {a}"


with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []

    col_id = find_col(headers, ["paper number", "id", "paper id", "submission id", "program id"])
    col_title = find_col(headers, ["title", "paper title"])
    col_abs = find_col(headers, ["abstract", "paper abstract"])

    if not col_id or not col_title or not col_abs:
        raise ValueError(f"Missing required columns: id={col_id}, title={col_title}, abstract={col_abs}")

    row = None
    for r in reader:
        if str(r.get(col_id, "")).strip() == PAPER_ID:
            row = r
            break

    if not row:
        raise ValueError(f"Paper id not found: {PAPER_ID}")

    doc = build_doc(row.get(col_title, ""), row.get(col_abs, ""))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write(doc)

print(doc)
print(f"Wrote: {OUT_PATH}")
