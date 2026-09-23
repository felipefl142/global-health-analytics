# AGENTS.md

Medallion pipeline over World Bank global-health panel data (country × year, 1990–2023) + 3 ML models + simulated A/B & DiD causal analysis. `PLAN.md` is the roadmap with per-phase acceptance criteria — read it before building new phases.

## Status (2026-09-23)

- Implemented (F0–F2): World Bank ingest → silver → gold ABT → UHC proxy features → `make train`.
- NOT implemented yet — these Makefile targets fail: `abtest` (`src/abtesting/*`), `serve` + `feast-materialize` (`src/serving/*`), `drift` (`src/monitoring/drift.py`), `dashboard` (`dashboard/app.py`). `tests/` is empty (only `__init__.py`); `.github/workflows/` is empty (CI planned, absent).
- Last full ingest FAILED: all 20 indicators errored (19× HTTP 400, 1× invalid code) per `data/bronze/ingest_log.json`. There is no parquet in bronze/silver/gold, so `silver`/`gold`/`train` currently raise FileNotFoundError. Re-run `make ingest` and check the log before debugging downstream.
- `SH_UHC_SCI` in `config/indicators.yaml` is not fetchable from the WB v2 API (snapshot-only, 2019) and always errors on ingest; the official code is `SH.UHC.NOEF` (PLAN.md). Don't "fix" the 2019 snapshot into a time series.

## Commands (Makefile, all via `.venv`)

- `make install` — creates venv and installs `requirements.txt` (requirements.txt is the dependency source of truth; pyproject has none).
- Pipeline order matters: `make ingest` → `make silver` → `make gold` → `make train`. `make abt` = silver+gold only (no ingest dependency). `make all` currently stops at the unimplemented `abtest` step.
- `make test` (pytest on `tests/`), `make lint` (ruff).
- `make clean` is destructive: removes all generated data (`data/bronze|silver|gold/*`, `data/*.duckdb`) and `models/*`.
- Run pipeline modules as `python -m src....` from the repo root. Imports like `from config import settings` rely on the repo root on sys.path (`config` has no `__init__.py`); `python src/.../file.py` will fail.
- Re-ingest a subset: `python -m src.ingestion.worldbank --codes SP.DYN.LE00.IN,SH.DYN.MORT`; add `--force` to refetch. Ingest is idempotent: existing `data/bronze/worldbank/{code}.json` is skipped, and per-indicator status is tracked in `data/bronze/ingest_log.json` (check it first).
- Redis (Feast online store) only when serving: `make redis-up` (docker compose).

## Data flow

- bronze: raw WB API JSON, one file per indicator: `data/bronze/worldbank/{code}.json`
- silver: `data/silver/worldbank_long.parquet` (long: country × year × indicator) + `countries_dim.parquet`
- gold: `data/gold/abt_country_year.parquet` — the ABT, 1 row per country × year, wide; plus `abt_quality.json` / `quality_checks.json`
- DuckDB: `src/utils/db.py` opens `data/catalog.duckdb` and exposes `silver.*` / `gold.*` views over the parquet files.
- Indicator catalog: `config/indicators.yaml` (groups: targets, uhc_inputs, uhc_official_2019, covariates). All paths/config go through `config/settings.py`, which loads `.env` (copy `.env.example`).

## Domain conventions

- Official UHC index exists only for 2019 → `src/features/uhc_index.py` builds a 0..1 proxy from weighted, percentile-normalized inputs; DiD treatment timing = first year a country crosses threshold 0.5 (`treated`, `treat_year`, `post` columns).
- Models: M1 `life_expectancy` (regression), M2 `child_mortality` (regression; under-5 is the proxy — no reliable under-1 series exists), M3 `milestone_high` (classification: uhc_index ≥ 0.8 AND life_expectancy ≥ 70).
- Splits are time-based to avoid leakage (`src/modeling/dataset.py`): train ≤2015, valid 2016–2019, test ≥2020. Do not use random splits on this panel.
- World Bank API is rate-limited/flaky: ingest uses a 0.6 s delay (`WB_API_DELAY`) + exponential backoff retries; a full run takes ~10+ min. Don't remove the delay or retries.
- All generated data and models are gitignored (`data/bronze|silver|gold/**`, `data/*.duckdb`, `models/*`, `.feast/`); keep the `.gitkeep` files.

## Style

- Code comments and docstrings are in Portuguese (pt-BR); match existing style.
- ruff: line-length 100, E501 ignored; isort first-party packages are `src` and `config`.
