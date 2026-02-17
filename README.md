# EACL 2026 Preference Arena

Interactive browser app to rank EACL papers, inspect a 2D embedding projection, and build oral/poster schedules.

## GitHub sharing model

This project is prepared for **GitHub Pages** under:

- `https://dennis-fast.github.io/eacl26-pref-arena/`

The shareable app is served from the `docs/` folder and runs **fully client-side**:

- state is stored in browser `localStorage`
- exports are downloaded as files (no server-side file writes)

## Project structure

- `docs/` → GitHub Pages publish root
   - `docs/index.html` → unified app (overview, ranking, oral schedule, posters, embedded viz)
   - `docs/assets/js/` → client logic + shared modules
   - `docs/assets/css/` → styles
   - `docs/viz/` → static 2D projection page
   - `docs/data/` → static artifacts served in Pages
- `apps/flask/` → Flask backend app (legacy/server runtime)
- `legacy/web/` → legacy server-rendered web assets used by Flask
- `src/web/` → canonical frontend source files
- `scripts/build/sync_web_assets.py` → publishes `src/web` files into `docs/` and `legacy/web/`
- `data/raw/` → source datasets and embeddings
- `data/generated/` → runtime-generated outputs (state/scored CSV)
- `scripts/`
   - `scripts/build/` → artifact generation scripts
   - `scripts/analysis/` → analysis and utility scripts
- `config/paths.py` → centralized filesystem path constants used by server/build/analysis scripts

## Local preview (Pages-equivalent)

From repo root:

```bash
python -m http.server 8000
```

Open:

- `http://localhost:8000/docs/`

Or use:

```bash
make serve-pages
```

## Regenerate static projection artifact

When the CSV or embeddings change:

```bash
python scripts/build/build_static_projection.py
```

This rebuilds:

- `docs/data/projection_pca.json`

## Sync frontend assets

After editing files in `src/web/`, sync to deploy/runtime folders:

```bash
python scripts/build/sync_web_assets.py
```

Or use:

```bash
make sync
```

## Common developer tasks

```bash
make build-all       # sync frontend assets + rebuild projection JSON
make serve-pages     # preview GitHub Pages app
make serve-flask     # run legacy Flask app
make smoke-flask     # quick API smoke test
make test            # run structural + Flask smoke tests
```

## CI

GitHub Actions workflow is defined in [.github/workflows/ci.yml](.github/workflows/ci.yml).

On each push/PR it runs:

- `make build-all`
- `make test`

## GitHub Pages deployment

Deployment workflow is defined in [.github/workflows/pages.yml](.github/workflows/pages.yml).

It runs on pushes to `main` and:

- builds docs via `make build-all`
- deploys the `docs/` folder to GitHub Pages

Repository settings requirement:

- Settings → Pages → Build and deployment → Source: **GitHub Actions**

## Main functionality

- Pairwise ranking with uncertainty-aware Elo updates
- Overview grouped by date/session/location with rank-aware sorting
- Oral schedule recommendations (topic match + ranking signals)
- Posters grouped by date/time and sorted by rank
- Static 2D embedding projection (precomputed PCA/t-SNE/UMAP)
- Backend-free `docs/viz` interaction: filtering, search highlight/filter, sampling, nearest neighbors from static artifact

## CSV columns (minimum)

Required:

- `Paper number` (or equivalent ID)
- `Title`
- `Abstract`

Recommended for scheduling and filtering:

- `Session`, `Room Location`, `Session Date`, `Session time`
- `Type of Presentation`, `Attendance Type`
- `category_primary`, `category_secondary`, `keywords`

## Backend-free visualization notes

The visualization page (`docs/viz`) no longer requires `/api/meta`, `/api/project`, or `/api/nn`.
It loads from:

- `docs/data/projection_pca.json`

The artifact now includes:

- `points_meta`
- `methods` (`pca`, `tsne`, `umap`)
- `neighbors`
- filter/session metadata used by the UI

In Flask mode, static docs data is also served at:

- `/data/<filename>`

which enables unified-app CSV auto-load (`/data/EACL_2026_program_categorized.csv`) without needing GitHub Pages.

## Testing

`make test` uses unittest discovery and runs all `test_*.py` modules under `tests/`, including:

- smoke tests (`tests/test_smoke.py`)
- static viz contract + integration checks (`tests/test_static_viz_contract.py`)
