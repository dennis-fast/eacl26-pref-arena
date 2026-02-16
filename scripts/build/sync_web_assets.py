from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "web"


def copy_file(src_rel: str, dst_rel: str) -> None:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"synced {src_rel} -> {dst_rel}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source directory: {SRC}")

    mappings = [
        ("src/web/shared/csv.js", "docs/assets/js/csv.js"),
        ("src/web/shared/rating.js", "docs/assets/js/rating.js"),
        ("src/web/shared/selector.js", "docs/assets/js/selector.js"),
        ("src/web/shared/storage.js", "docs/assets/js/storage.js"),
        ("src/web/docs/unified_app.js", "docs/assets/js/unified_app.js"),
        ("src/web/docs/unified_styles.css", "docs/assets/css/unified_styles.css"),
        ("src/web/docs/viz/index.html", "docs/viz/index.html"),
        ("src/web/docs/viz/app.js", "docs/viz/app.js"),
        ("src/web/docs/viz/styles.css", "docs/viz/styles.css"),
        ("src/web/shared/csv.js", "legacy/web/csv.js"),
        ("src/web/shared/rating.js", "legacy/web/rating.js"),
        ("src/web/shared/selector.js", "legacy/web/selector.js"),
        ("src/web/shared/storage.js", "legacy/web/storage.js"),
        ("src/web/legacy/unified_app.js", "legacy/web/unified_app.js"),
        ("src/web/legacy/unified_styles.css", "legacy/web/unified_styles.css"),
        ("src/web/legacy/viz/index.html", "legacy/web/static/index.html"),
        ("src/web/legacy/viz/app.js", "legacy/web/static/app.js"),
        ("src/web/legacy/viz/styles.css", "legacy/web/static/styles.css"),
    ]

    for src_rel, dst_rel in mappings:
        copy_file(src_rel, dst_rel)

    print("sync complete")


if __name__ == "__main__":
    main()
