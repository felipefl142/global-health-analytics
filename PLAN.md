# Plano — Plataforma Global Health Analytics & Causal A/B (WHO/World Bank)

## Decisões

- **Dataset**: painel World Bank/WHO de saúde global (país × ano, multi-fonte), 1990–2023, ~200+ países.
- **Stack**: scripts Python + Jupyter (EDA) + Streamlit + DuckDB + Feast (offline=DuckDB, online=Redis) + FastAPI.
- **Produção**: medallion bronze/silver/gold + ABT + feature store + CI/CD + drift monitoring.
- **A/B**: simulado + causal (DiD). Estrela: expansão de UHC → expectativa de vida.
- **3 modelos separados**: (M1) expectativa de vida, (M2) mortalidade infantil, (M3) classificação
  de marco (UHC ≥ 0.8 e LE ≥ 70 anos). Modelo principal: **XGBoost** (baseline linear/ridge).

## Status

| Fase | Status |
|---|---|
| F0–F2 (setup, ingestão, silver/gold, proxy UHC, `make train`) | Código implementado — **ingestão quebrada** (ver §7) |
| F3–F12 | Pendente |

## 1. Objetivo

Construir, com ênfase em gerar conhecimento da base (EDA + viz + testes de hipótese), um pipeline
de dados reprodutível sobre um painel global de saúde, com predição, um componente A/B (simulado +
causal) e todo o serving em produção (DuckDB + medallion + Feast + FastAPI + Streamlit + CI/CD + drift).

## 2. Stack

| Camada | Tecnologia |
|---|---|
| Ingestão | `requests` (World Bank API v2), WHO GHO (API OData) → bronze |
| Warehouse | DuckDB (catálogo com views sobre Parquet) |
| Medallion | `bronze/` (raw) → `silver/` (limpo/padronizado) → `gold/` (curado) |
| ABT | `abt_country_year` (1 linha por país × ano, wide) |
| EDA/Hipóteses | Jupyter + pandas + statsmodels/linearmodels + scipy |
| Modelos | scikit-learn + **XGBoost** (baseline linear/ridge) + SHAP |
| Feature store | Feast (offline=DuckDB, online=Redis) |
| API | FastAPI + pydantic + uvicorn |
| Dashboard | Streamlit + plotly (+ folium p/ mapas) |
| A/B + causal | simulação própria + DiD escalonado + controle sintético |
| CI/CD + monitoring | GitHub Actions + PSI/KS (drift) + log de predições |
| Orquestração | Makefile + docker-compose (Redis) |

## 3. Fontes de dados e modelo de entidade

- **Fonte 1 (primária)**: World Bank API (`api.worldbank.org/v2`) — séries país × ano.
- **Fonte 2 (secundária)**: WHO Global Health Observatory (GHO, API OData) — carga de
  doença/risco (DCNT 30–70, mortalidade materna, incidência de TB, trânsito) para enriquecimento.
- **Dimensão país**: nomes/ISO padronizados (consolidação multi-fonte no silver).
- **Entidade primária**: country × year (painel). ABT = 1 linha por país × ano.
- **Entidade secundária**: country (snapshot mais recente) p/ visão transversal.

Indicadores (códigos World Bank; fonte da verdade: `config/indicators.yaml`, validados na ingestão):

- **Alvos**: `SP.DYN.LE00.IN` (expectativa de vida), `SH.DYN.MORT` (mortalidade <5 — proxy de
  mortalidade infantil; não há série <1 ano confiável).
- **Insumos UHC**:
  - gasto: `SH.XPD.CHEX.PC.CD` (per capita), `SH.XPD.CHEX.GD.ZS` (% PIB),
    `SH.XPD.GHED.GD.ZS` (gasto público % PIB), `SH.XPD.OOPC.CH.ZS` (out-of-pocket)
  - força de trabalho/estrutura: `SH.MED.PHYS.ZS` (médicos/1k), `SH.MED.NUMW.P3` (enfermeiros e
    parteiras/1k), `SH.MED.BEDS.ZS` (leitos/1k)
  - saneamento/água: `SH.STA.BASS.ZS`, `SH.H2O.BASW.ZS`, `SH.STA.SMSS.ZS`, `SH.H2O.SMDW.ZS`
  - vacinação: `SH.IMM.MEAS` (sarampo), `SH.IMM.IDPT` (DPT)
- **UHC oficial (SCI)**: `SH_UHC_SCI` — série **anual 2000–2023** na API v2 (verificado ao vivo em
  2026-09; o `SH.UHC.NOEF` do plano original não existe). Usado para validar o proxy de painel.
- **WHO GHO**: `NCDMORT3070` (DCNT 30–70), `MDG_0000000026` (materna), `MDG_0000000020` (TB),
  `RS_198` (trânsito) — fonte secundária de enriquecimento.
- **Covariados**: `NY.GDP.PCAP.CD` (PIB/capita), `SP.URB.TOTL.IN.ZS` (urbanização),
  `SP.DYN.TFRT.IN` (fertilidade), `SP.POP.TOTL` (população).

## 4. Decisão de projeto: proxy UHC

O SCI oficial cobre 2000–2023; o painel começa em 1990 e o DiD precisa de um índice em todos os
anos. Solução (implementada em `src/features/uhc_index.py`):

- Proxy `uhc_index` (0..1) por país × ano: média ponderada dos insumos UHC com normalização robusta
  por percentis 5/95.
- O SCI oficial (2000–2023) entra como validação do proxy (correlação de Pearson).
- Tratamento do DiD = primeiro ano em que o país cruza `uhc_index ≥ 0.5` (timing escalonado;
  colunas `treated`, `treat_year`, `post`).
- Marco M3 = `uhc_index ≥ 0.8` **e** expectativa de vida ≥ 70.

## 5. Estrutura do repositório

```
global-health-analytics/
├── README.md · AGENTS.md · pyproject.toml · requirements.txt · .env.example
├── Makefile · docker-compose.yml (Redis)
├── config/            # indicators.yaml, settings.py, feast/
├── src/
│   ├── ingestion/     # worldbank.py, who_gho.py            → bronze
│   ├── transform/     # bronze_to_silver.py, build_abt.py, quality.py
│   ├── features/      # uhc_index.py (proxy UHC + timing DiD)
│   ├── modeling/      # dataset.py, train.py, evaluate.py
│   ├── abtesting/     # simulate_ab.py, causal_did.py
│   ├── serving/       # app.py (FastAPI), feast_features.py
│   ├── monitoring/    # drift.py (PSI/KS)
│   └── utils/         # db.py, io.py
├── notebooks/         # 01_eda, 02_visualization, 03_hypothesis_testing, 04_modeling, 05_ab_causal
├── dashboard/         # app.py (Streamlit)
├── tests/             # ingestion, transform, models, api
├── data/              # bronze/silver/gold + duckdb (gitignored)
└── .github/workflows/ # ci.yml, monitoring.yml
```

## 6. Roadmap por fases (com critério de aceitação)

**F0 — Setup & scaffolding**
Repo, requirements, `config/`, Makefile, docker-compose (Redis), `data/` (bronze/silver/gold),
`.env.example`, `.gitignore`, ruff + pytest.
✅ `make install && make test` roda; `docker compose up` sobe Redis.

**F1 — Ingestão → Bronze**
`worldbank.py` (API → JSON raw por indicador, idempotente, retry/backoff), `who_gho.py`
(API OData → JSON raw por indicador). Validação dos códigos de indicador contra as APIs.
✅ `make ingest` produz `data/bronze/*` íntegro e `ingest_log.json` sem erros.

**F2 — Transform → Silver + Gold (ABT)**
`bronze_to_silver.py` (parse, tipagem, padronizar países, flags de qualidade, missing) → `silver/`.
`build_abt.py` (consolida wide `abt_country_year` + proxy UHC) → `gold/`.
✅ `make silver && make gold` gera ABT 1 linha/país × ano; testes de qualidade (cobertura, outliers, consistência).

**F3 — EDA (núcleo de conhecimento)**
`01_eda.ipynb`: cobertura país/ano, distribuições, matriz de missingness, correlações, tendências,
outliers, comparativo regional.
✅ Documenta a base: o que é confiável, o que falta, por quê.

**F4 — Visualizações**
`02_visualization.ipynb`: séries temporais, choropleth (folium), heatmaps, boxplots por região/renda,
evolução de indicadores.
✅ Galeria de viz reutilizada no dashboard.

**F5 — Testes de hipótese (núcleo de conhecimento)**
`03_hypothesis_testing.ipynb` — cada hipótese com H0/H1, teste, tamanho de efeito + IC + p +
significância prática. Catálogo (ajustável):
- H1: densidade de médicos ↑ ⇒ expectativa de vida ↑ (líquido de PIB/capita) — regressão de painel
- H2: saneamento > 80% ⇒ mortalidade <5 significativamente menor — t-test/Mann-Whitney + efeito
- H3: +1 DP no gasto público em saúde ⇒ Δ mensurável na expectativa de vida
- H4: urbanização associada a mortalidade <5 menor
- H5: após cruzar o limiar UHC, o crescimento da expectativa de vida acelera (DiD)
- H6: vacinação contra sarampo associada a mortalidade <5 menor

✅ Tabela consolidada de hipóteses com veredito.

**F6 — Predição (3 modelos separados)**
`04_modeling.ipynb` + `src/modeling/`:
- M1 `life_expectancy_reg` — expectativa de vida (regressão)
- M2 `child_mortality_reg` — mortalidade <5 (regressão)
- M3 `milestone_high_clf` — marco alto: `uhc_index ≥ 0.8` e LE ≥ 70 (classificação)

Split temporal fixo (evita leakage no painel): train ≤ 2015, valid 2016–2019, test ≥ 2020.
Baseline linear/ridge + **XGBoost** (`tree_method="hist"`, NaN nativo, early stopping no valid);
eval (RMSE/MAE/R²; AUC/F1/calibração) + SHAP p/ interpretação.
✅ 3 artefatos versionados (`models/<name>_xgb.joblib` + `_lin.joblib`), relatórios `*_eval.json`.

**F7 — A/B + Causal**
`05_ab_causal.ipynb` + `src/abtesting/`:
- Simulado: randomiza país × anos em treatment/control, aplica uplift no proxy UHC, prevê com o
  modelo, roda A/B completo (power, t-test, IC, Cohen's d, CUPED, correção múltipla).
- Causal (estrela): DiD escalonado UHC → expectativa de vida (linearmodels/pyfixest, FE país + ano,
  SE robustos cluster por país) + controle sintético p/ 1–2 países "herói". Diagnósticos:
  parallel trends + event study.

✅ Efeitos + ICs + plots de event study/sintético.

**F8 — Feature store (Feast)**
Offline=DuckDB (lê gold/ABT), online=Redis. Feature views `country_health_static`,
`country_health_yearly` (entity=country, ts=ano) + feature service `health_prediction_service`.
✅ `make feast-materialize` popula a online store; `get_online_features` retorna por país.

**F9 — API (FastAPI)**
`/health`, `/countries`, `POST /predict` (features da online store → 3 modelos → predições + IC +
features usadas), `GET /experiments` (resultados A/B + DiD), `POST /ab/simulate` (opcional).
Validação pydantic + testes.
✅ `make serve`; testes com TestClient passando.

**F10 — Dashboard (Streamlit)**
Páginas: EDA explorer (país/indicador/ano, séries, missingness, mapa), Hipóteses (tabela +
efeito/IC/p), Predição (input → 3 saídas + SHAP), Experimentos (A/B + event study DiD + sintético).
✅ `make dashboard` navegável.

**F11 — CI/CD + drift monitoring**
- `ci.yml`: lint (ruff) + pytest + smoke do pipeline (ingest amostra → silver → gold).
- `monitoring.yml` / `drift.py`: PSI/KS por feature (janela de treino vs. recente/serving) + log de
  predições/resíduos; alerta por threshold.

✅ CI verde no push; drift report gerado e exposto no dashboard.

**F12 — Hardening & docs**
README (arquitetura, decisões, como rodar), Makefile completo, `.env`, documentação das hipóteses e
dos modelos, git tags de versão.
✅ `make all` reproduz tudo do zero.

## 7. Riscos e pendências

- **Ingestão resolvida em F1**: 22 indicadores World Bank + 4 WHO GHO baixados sem erros
  (`data/bronze/ingest_log.json` / `ingest_log_who.json`). Causas anteriores: `per_page=50000`
  (HTTP 400) e código OOP descontinuado.
- UHC SCI cobre só 2000–2023 → proxy UHC de painel cobre 1990+ (F2/F7).
- Missingness em indicadores de saúde global → tratado no silver + flags; XGBoost lida com NaN.
- Leakage temporal em painel → split temporal + CUPED no A/B; valid usado no early stopping, então
  a métrica honesta é a de test.
- Feast + Redis dependem de docker-compose.
