# AGENTS.md

Medallion pipeline over World Bank global-health panel data (country × year, 1990–2023) + 3 ML models + simulated A/B & DiD causal analysis. `PLAN.md` is the roadmap with per-phase acceptance criteria — read it before building new phases.

## Status (2026-09-23)

- Branch `restart/deepseek`: recomeço do zero a partir de `56ea8b7`. **F0** (scaffolding) e **F1**
  (ingestão) concluídos; F2–F12 pendentes.
- Ingestão OK: 22 indicadores World Bank + 4 WHO GHO, sem erros (`data/bronze/ingest_log.json` e
  `data/bronze/ingest_log_who.json`).
- Códigos corrigidos: out-of-pocket = `SH.XPD.OOPC.CH.ZS` (o antigo `...TO.ZS` foi removido);
  `SH_UHC_SCI` é **série anual 2000–2023** e é buscável (o `SH.UHC.NOEF` não existe). `per_page`
  efetivo = 10000 (50000 gera HTTP 400).

## Commands (Makefile, all via `.venv`)

- `make install` — creates venv and installs `requirements.txt` (requirements.txt is the dependency source of truth; pyproject has none).
- Pipeline order matters: `make ingest` → `make silver` → `make gold` → `make train`. `make abt` = silver+gold only (no ingest dependency). `make all` currently stops at the unimplemented `abtest` step.
- `make test` (pytest on `tests/`), `make lint` (ruff).
- `make clean` is destructive: removes all generated data (`data/bronze|silver|gold/*`, `data/*.duckdb`) and `models/*`.
- Run pipeline modules as `python -m src....` from the repo root. Imports like `from config import settings` rely on the repo root on sys.path (`config` has no `__init__.py`); `python src/.../file.py` will fail.
- Re-ingest a subset: `python -m src.ingestion.worldbank --codes SP.DYN.LE00.IN,SH.DYN.MORT` (idem
  para `src.ingestion.who_gho`); add `--force` to refetch. Ingest is idempotent: existing
  `data/bronze/{worldbank,who}/{code}.json` is skipped, and per-indicator status is tracked in
  `data/bronze/ingest_log*.json` (check it first).
- Redis (Feast online store) only when serving: `make redis-up` (docker compose).

## Data flow

- bronze: raw API JSON, one file per indicator: `data/bronze/worldbank/{code}.json` (World Bank) e
  `data/bronze/who/{code}.json` (WHO GHO), com logs `ingest_log.json` / `ingest_log_who.json`.
- silver: `data/silver/worldbank_long.parquet` (long: country × year × indicator) + `countries_dim.parquet`
- gold: `data/gold/abt_country_year.parquet` — the ABT, 1 row per country × year, wide; plus `abt_quality.json` / `quality_checks.json`
- DuckDB: `src/utils/db.py` opens `data/catalog.duckdb` and exposes `silver.*` / `gold.*` views over the parquet files.
- Indicator catalog: `config/indicators.yaml` (groups: targets, uhc_inputs, uhc_official, covariates, who_gho). All paths/config go through `config/settings.py`, which loads `.env` (copy `.env.example`).

## Domain conventions

- SCI oficial cobre 2000–2023 → `src/features/uhc_index.py` builds a 0..1 proxy over 1990+ from
  weighted, percentile-normalized inputs; DiD treatment timing = first year a country crosses
  threshold 0.5 (`treated`, `treat_year`, `post` columns).
- Models: M1 `life_expectancy` (regression), M2 `child_mortality` (regression; under-5 is the proxy — no reliable under-1 series exists), M3 `milestone_high` (classification: uhc_index ≥ 0.8 AND life_expectancy ≥ 70).
- Splits are time-based to avoid leakage (`src/modeling/dataset.py`): train ≤2015, valid 2016–2019, test ≥2020. Do not use random splits on this panel.
- World Bank API is rate-limited/flaky: ingest uses a 0.6 s delay (`WB_API_DELAY`) + exponential backoff retries. Don't remove the delay or retries.
- All generated data and models are gitignored (`data/bronze|silver|gold/**`, `data/*.duckdb`, `models/*`, `.feast/`); keep the `.gitkeep` files.

## Style

- Code comments and docstrings are in Portuguese (pt-BR); match existing style.
- ruff: line-length 100, E501 ignored; isort first-party packages are `src` and `config`.
