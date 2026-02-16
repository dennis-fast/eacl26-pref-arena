from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPS_DIR = PROJECT_ROOT / "apps"
LEGACY_WEB_DIR = PROJECT_ROOT / "legacy" / "web"
LEGACY_STATIC_DIR = LEGACY_WEB_DIR / "static"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_GENERATED_DIR = DATA_DIR / "generated"

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

RAW_PROGRAM_CSV = DATA_RAW_DIR / "EACL_2026_program.csv"
RAW_PROGRAM_CATEGORIZED_CSV = DATA_RAW_DIR / "EACL_2026_program_categorized.csv"
RAW_EMBEDDINGS_NPZ = DATA_RAW_DIR / "specter2_sample_embeddings.npz"
RAW_EMBEDDINGS_METADATA_CSV = DATA_RAW_DIR / "specter2_sample_metadata.csv"

GENERATED_STATE_JSON = DATA_GENERATED_DIR / "eacl_pref_state.json"
GENERATED_SCORED_CSV = DATA_GENERATED_DIR / "EACL_2026_program_scored.csv"
GENERATED_EMBEDDING_INPUT_TXT = DATA_GENERATED_DIR / "embedding_input_819-main.txt"

PROCESSED_PAIRWISE_COSINE_NPY = DATA_PROCESSED_DIR / "specter2_pairwise_cosine.npy"

DOCS_PROJECTION_PCA_JSON = DOCS_DATA_DIR / "projection_pca.json"
