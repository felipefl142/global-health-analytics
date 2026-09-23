# AGENTS.md

Medallion pipeline over World Bank + WHO GHO global-health panel data (country × year, 1990–2023, 217 countries) → EDA/hypothesis tests → 3 XGBoost models → simulated A/B + causal DiD → Feast + FastAPI + Streamlit + drift monitoring. `PLAN.md` is the roadmap (all phases F0–F12 implemented); `README.md` has the results.

## Commands (Makefile, all via `.venv`)

- `make install` — venv + `requirements.txt` (a pip freeze; the dependency source of truth — pyproject has none). `nvidia-nccl-cu13` is deliberately NOT pinned (Linux-only xgboost dep).
- Pipeline order: `make ingest` → `silver` → `gold` (runs quality checks) → `train` → `abtest` → `hypotheses` → `drift`. `make all` runs the whole chain. `make notebooks` rebuilds + executes notebooks 01–05.
- `make test` (pytest), `make lint` (ruff). Dashboard tests skip when `data/gold` / `models/` are empty.
- `make clean` is destructive: deletes `data/bronze|silver|gold/*`, `data/*.duckdb`, `models/*`, `.feast` (a full re-ingest takes ~3–5 min).
- Run modules as `python -m src....` from the repo root (`config` has no `__init__.py`; `python src/.../file.py` fails).
- Serving: `make redis-up` (docker) or `export ONLINE_STORE_TYPE=sqlite` (no Redis; used in CI and locally when docker is absent) → `make feast-materialize` → `make serve` / `make dashboard`.
- Path overrides via env: `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FEAST_REGISTRY_PATH` — use them to run the pipeline in an isolated dir (e.g. to reproduce the CI smoke job) without touching `data/` or `reports/`.

## Ingestion gotchas

- World Bank API rejects large `per_page` (50000 → HTTP 400 for every indicator); ingest paginates with `per_page: 10000` from `config/indicators.yaml`. 4xx fails fast (no retry); keep the 0.6 s delay + backoff for 5xx/429.
- The API returns errors as HTTP 200 with `[{"message": ...}]` — handled in `_api_error`.
- Indicator records have no country id/region: use `countryiso3code`; region/income/aggregates come from `/country` (`data/bronze/worldbank/_countries.json`; aggregates have region `Aggregates` and are dropped in silver).
- `SH.XPD.OOPC.TO.ZS` was removed from the API → `SH.XPD.OOPC.CH.ZS`. `SH_UHC_SCI` IS fetchable and is an annual series 2000–2023 (not a 2019 snapshot).
- Re-ingest a subset: `python -m src.ingestion.worldbank --codes A,B [--force]`; status per indicator in `data/bronze/ingest_log.json` (WHO: `ingest_log_who.json`).

## Data flow

- bronze: `data/bronze/worldbank/{code}.json`, `data/bronze/who/{code}.json` (raw payloads).
- silver: `indicators_long.parquet` (country_id ISO3, year, indicator logical name, source, value) + `countries_dim.parquet`.
- gold: `abt_country_year.parquet` (full country × year grid, wide; raw values NOT interpolated) + `abt_quality.json`, `quality_checks.json`; `feast_*.parquet` are Feast sources (written by `feast_cli export`).
- reports/: JSON results consumed by API `/experiments` and the dashboard — versioned in git.

## Domain conventions

- UHC proxy (`src/features/uhc_index.py`): within-country interpolation of structural inputs is used ONLY inside the index computation; log of spending; weighted mean over available components, NaN if < 50% of total weight (total counts components missing from the ABT). Validated vs official SCI (Pearson ≈ 0.91).
- DiD treatment: first year ≥ 2000 with a *sustained* crossing of 0.5 (before 2000 index composition changes). `always_treated` are excluded; `never_treated` requires the proxy observed in ≥ 50% of window years (unobserved microstates are not valid controls).
- The naive DiD is negative because of a pre-existing convergence trend; report the trend-adjusted estimate (`staggered_detrended`) as the headline. Don't "fix" the negative TWFE by tweaking samples.
- Models: time split train ≤ 2015 / valid 2016–19 (early stopping + conformal quantile) / test ≥ 2020. Never random splits. M3 (`milestone_high`) must not use `uhc_index` (label leakage) — enforced by `EXCLUDE_BY_TARGET` in `dataset.py`. Artefacts: `models/<name>_xgb.joblib`, `_lin.joblib`, `_eval.json`.
- Life expectancy < 25 is real (Rwanda 1994, CAR, Somalia) — quality flags it as a warning, not an error.
- Drift: PSI/KS on observed values only; missing-rate change is a separate `missing_shift` flag.

## Notebooks

- Edit `notebooks/build_notebooks.py`, never the `.ipynb` directly; `make notebooks` regenerates and executes them. Markdown narrative states numbers — re-check them against cell outputs after data or model changes.

## Style

- Code comments/docstrings in Portuguese (pt-BR, mostly without accents in code); match existing style.
- ruff: line-length 100, E501 ignored; isort first-party `src`, `config`.
- Charts: use `src/viz/style.py` (fixed categorical order, income groups as ordinal blue ramp, diverging blue↔red for signed values).
