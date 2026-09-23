# global-health-analytics

Plataforma de analytics + causal A/B sobre um painel público de saúde global
(Who/World Bank) — país × ano, 1990–2023, 200+ países.

Pipeline reprodutível no padrão **medallion** (bronze → silver → gold), com
predição (3 modelos) e a base pronta para análise causal (DiD) sobre o tema
central: **expansão de cobertura UHC → expectativa de vida**.

> Roadmap completo com critérios de aceitação: [PLAN.md](PLAN.md). Convenções
> para agentes: [AGENTS.md](AGENTS.md).

## Stack

| Camada | Tecnologia |
|---|---|
| Ingestão | `requests`/`urllib` (World Bank API v2) → JSON raw |
| Warehouse | DuckDB (catálogo com views sobre Parquet) |
| Medallion | `data/bronze` (raw) → `data/silver` (limpo/long) → `data/gold` (ABT wide) |
| Feature engineering | Proxy UHC 0..1 + timing de tratamento p/ DiD |
| Modelos | scikit-learn + LightGBM + SHAP (artefatos `.joblib` + relatórios JSON) |
| Orquestração | Makefile + docker-compose (Redis, p/ serving futuro) |
| Plano de serving | Feast (offline=DuckDB, online=Redis) + FastAPI + Streamlit + drift PSI/KS |

## Funcionalidades

- **Ingestão idempotente** do World Bank API: um JSON bruto por indicador em
  `data/bronze/worldbank/{code}.json`, retry com backoff exponencial, delay
  anti rate-limit e log por indicador em `data/bronze/ingest_log.json`.
- **Camada silver**: `worldbank_long.parquet` (long: país × ano × indicador,
  tipado e padronizado) + `countries_dim.parquet` (dimensão país).
- **Camada gold (ABT)**: `abt_country_year.parquet` — 1 linha por país × ano,
  wide, com todos os indicadores unificados + `abt_quality.json`
  (cobertura, missing, outliers).
- **Proxy UHC** (`uhc_index`, 0..1): média ponderada de insumos (médicos,
  enfermeiros, leitos, gasto por capita, saneamento/água, vacinação) com
  normalização robusta por percentis (5/95) — o índice oficial só existe em
  2019, então o proxy cobre o painel inteiro.
- **Timing de tratamento p/ DiD**: `treated`, `treat_year`, `post` — primeiro
  ano em que o país cruza o limiar do proxy UHC (default 0.5).
- **3 modelos** com split **temporal** (evita leakage em painel):
  - M1 `life_expectancy_reg` — expectativa de vida (regressão)
  - M2 `child_mortality_reg` — mortalidade infantil <5 (regressão)
  - M3 `milestone_high_clf` — marco de saúde alto: UHC ≥ 0.8 **e** LE ≥ 70 (classificação)
  - Cada modelo: baseline ridge/linear + LightGBM, métricas (RMSE/MAE/R²;
    AUC/F1), importância de features (SHAP com fallback p/ permutation) e
    artefatos em `models/*.joblib` + `models/*_eval.json`.
- **DuckDB catalogado**: `data/catalog.duckdb` com schemas `silver.*` e
  `gold.*` apontando para os parquets (`src/utils/db.py`).

## Indicadores

Catálogo em [`config/indicators.yaml`](config/indicators.yaml) (códigos
validados contra a API):

| Grupo | Conteúdo |
|---|---|
| `targets` | expectativa de vida, mortalidade <5 |
| `uhc_inputs` | gasto em saúde (capita/%GDP/público/OOP), médicos, enfermeiros, leitos, água/saneamento (básico e seguro), vacinas (sarampo, DPT) |
| `uhc_official_2019` | SCI oficial (snapshot 2019, validação transversal) |
| `covariates` | PIB/capita, urbanização, fertilidade, população |

## Estrutura do repositório

```
global-health-analytics/
├── Makefile               # orquestracao (make help)
├── config/
│   ├── indicators.yaml    # catalogo de indicadores World Bank
│   └── settings.py        # caminhos + variaveis de ambiente (.env)
├── src/
│   ├── ingestion/worldbank.py   # API -> bronze
│   ├── transform/               # bronze_to_silver, build_abt, quality
│   ├── features/uhc_index.py    # proxy UHC + timing p/ DiD
│   ├── modeling/                # dataset.py (split temporal), train.py (M1/M2/M3)
│   ├── abtesting/               # (plano: A/B simulado + DiD)
│   ├── serving/                 # (plano: Feast + FastAPI)
│   ├── monitoring/              # (plano: drift PSI/KS)
│   └── utils/                   # db.py (DuckDB), io.py
├── data/                  # bronze/silver/gold + duckdb (gitignored)
├── models/                # artefatos .joblib + eval (gitignored)
└── tests/
```

## Como usar

Pré-requisitos: Python ≥ 3.11, Make. Docker só é necessário para o Redis
(serving, ainda não implementado).

```bash
# 1. Setup
make install                # cria .venv e instala requirements.txt
cp .env.example .env        # ajuste se necessario

# 2. Pipeline (ordem importa)
make ingest                 # World Bank API -> bronze (~10+ min; rate-limited)
make silver                 # bronze -> silver (long + dim país)
make gold                   # silver -> gold (ABT + proxy UHC + labels)
make train                  # treina M1/M2/M3 -> models/

# Atalhos
make abt                    # silver + gold (sem ingest)
make all                    # ingest -> silver -> gold -> train -> abtest
```

### Ingestão

```bash
make ingest                                    # todos os indicadores
.venv/bin/python -m src.ingestion.worldbank --codes SP.DYN.LE00.IN,SH.DYN.MORT
.venv/bin/python -m src.ingestion.worldbank --force   # refaz tudo
```

- Idempotente: JSON existente em `data/bronze/worldbank/` é pulado (`--force` refaz).
- O World Bank API é instável/rate-limited: delay de 0.6 s + backoff
  exponencial (não remover). Rodada completa demora ~10+ min.
- Consulte o status por indicador em `data/bronze/ingest_log.json`.
- ⚠️ `SH_UHC_SCI` (snapshot 2019) **falha por design** na API v2 — é
  esperado 1 erro por ingestão. O código oficial de série é `SH.UHC.NOEF`.

### Consulta via DuckDB

```python
from src.utils.db import query, list_views

list_views()
query("SELECT country_name, year, uhc_index, life_expectancy \
       FROM gold.abt_country_year WHERE year >= 2020 LIMIT 5")
```

### Modelos

```bash
make train
# artefatos:
#   models/life_expectancy_reg_lgbm.joblib  (+ _lin, _eval.json)
#   models/child_mortality_reg_lgbm.joblib
#   models/milestone_high_clf_lgbm.joblib
#   models/all_eval.json  (relatório consolidado)
```

Split temporal fixo: train ≤ 2015, valid 2016–2019, test ≥ 2020
(`src/modeling/dataset.py`).

### Qualidade

```bash
make test                   # pytest
make lint                   # ruff
```

### Limpeza

```bash
make clean                  # DESTRUTIVO: remove data/bronze|silver|gold, duckdb, models/
```

## Roadmap / status

| Fase | Escopo | Status |
|---|---|---|
| F0–F1 | Setup + ingestão → bronze | ✅ |
| F2 | Silver + gold (ABT + proxy UHC) | ✅ |
| F6 | Modelos M1/M2/M3 (train/eval) | ✅ |
| F3–F5 | EDA, visualizações, testes de hipótese (notebooks) | ❌ pendente |
| F7 | A/B simulado + DiD causal (`src/abtesting/`) | ❌ pendente |
| F8 | Feature store Feast (DuckDB/Redis) | ❌ pendente |
| F9 | API FastAPI (`make serve`) | ❌ pendente |
| F10 | Dashboard Streamlit (`make dashboard`) | ❌ pendente |
| F11 | CI/CD + drift monitoring (`make drift`) | ❌ pendente |

> Alvos `abtest`, `serve`, `feast-materialize`, `drift` e `dashboard` do
> Makefile ainda não têm implementação — `make all` para no passo `abtest`.

## Configuração (`.env`)

Copie `.env.example` → `.env`. Principais variáveis:

| Variável | Default | Descrição |
|---|---|---|
| `WB_API_BASE` / `WB_API_DELAY` / `WB_RETRIES` | api.worldbank.org / 0.6 / 4 | parâmetros da ingestão |
| `DATA_DIR` / `MODELS_DIR` | `./data` / `./models` | caminhos de saída |
| `REDIS_*` / `FEAST_*` | localhost:6379 / `.feast/registry` | serving (quando implementado) |
| `DRIFT_PSI_THRESHOLD` / `DRIFT_KS_THRESHOLD` | 0.25 / 0.30 | alertas de drift (plano) |

## Notas

- Comentários/docstrings em pt-BR; ruff com line-length 100, first-party
  `src` e `config`. Rode os módulos como `python -m src....` a partir da raiz.
- O indicador infantil usa under-5 (`SH.DYN.MORT`): não existe série pública
  confiável de mortalidade <1 ano no World Bank.
