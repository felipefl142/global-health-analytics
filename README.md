# Global Health Analytics — UHC, predição e causalidade

Plataforma reprodutível de analytics sobre um painel público de saúde global
(World Bank + WHO GHO), país × ano, **1990–2023, 217 países**. Cobre o ciclo
completo: ingestão em camadas (**medallion**), análise (EDA + testes de hipótese),
3 modelos (**XGBoost**), um componente **A/B simulado + causal (DiD/controle
sintético)** e serving (Feast + FastAPI + Streamlit + CI/CD + drift).

> Estrela do projeto: **expansão da cobertura UHC → expectativa de vida**.

- Roadmap com critérios de aceitação: [PLAN.md](PLAN.md)
- Convenções do repositório: [AGENTS.md](AGENTS.md)

## Arquitetura

```
World Bank API v2 ─┐
                   ├─► bronze (JSON bruto) ─► silver (long + dim) ─► gold (ABT wide + UHC)
WHO GHO (OData) ───┘                                                      │
                                                                          ├─► 3 modelos XGBoost
                                                                          ├─► A/B simulado + DiD causal
                                                                          └─► Feast ─► FastAPI / Streamlit
```

| Camada | Tecnologia |
|---|---|
| Ingestão | `requests` (World Bank API v2 + WHO GHO OData) |
| Warehouse | DuckDB (views sobre Parquet) |
| Medallion | `bronze/` → `silver/` → `gold/` |
| EDA/hipóteses | pandas, statsmodels, linearmodels, scipy, plotly, folium |
| Modelos | scikit-learn (baseline) + **XGBoost** + SHAP |
| Feature store | Feast (offline=parquet, online=sqlite/Redis) |
| API / Dashboard | FastAPI + pydantic / Streamlit + plotly + folium |
| CI/CD + monitoring | GitHub Actions + PSI/KS (drift) |

## Dados

- **22 indicadores World Bank** (alvos, insumos UHC, SCI oficial e covariados) +
  **4 indicadores WHO GHO** (DCNT 30–70, mortalidade materna, incidência de TB,
  trânsito). Fonte da verdade: `config/indicators.yaml`.
- Alvos: `SP.DYN.LE00.IN` (expectativa de vida) e `SH.DYN.MORT` (mortalidade <5 —
  proxy padrão, não há série confiável de <1 ano).
- O SCI oficial (`SH_UHC_SCI`) cobre **2000–2023**; o painel começa em 1990, então
  construímos um **proxy UHC** de painel (ver decisões).

## Como rodar

```bash
make install          # cria .venv e instala requirements.txt
make ingest           # World Bank + WHO GHO -> data/bronze
make silver           # bronze -> data/silver (long + countries_dim)
make gold             # silver -> data/gold (ABT + proxy UHC + flags de DiD)
make analise          # relatórios de EDA e de hipóteses
make train            # 3 modelos (XGBoost + baseline)
make abtest           # A/B simulado + DiD causal + controle sintético
make drift            # relatório de drift (PSI/KS)

make all              # pipeline completo acima, do zero
make test && make lint

# serviços
make feast-materialize   # materializa offline -> online (sqlite por padrão)
make serve               # FastAPI em :8000
make dashboard           # Streamlit
```

Requisitos: Python ≥ 3.11. Copie `.env.example` para `.env` para ajustar paths,
Redis e limiares de drift.

## Decisões de projeto

- **Proxy UHC (`src/features/uhc_index.py`)**: média ponderada de gasto (log),
  força de trabalho, saneamento/água e vacinação, normalizada por percentis 5/95,
  exigindo ≥ 50% do peso disponível; interpolação intra-país limitada a 5 anos.
  Correlação com o SCI oficial: **Pearson 0.90** (n=4.608).
- **Split temporal** (nunca aleatório, evita leakage): treino ≤ 2015, validação
  2016–2019, teste ≥ 2020.
- **XGBoost com NaN nativo** (`tree_method="hist"`, early stopping na validação):
  o missing não é aleatório (piora com a renda), então imputar seria enviesado.
- **Sem leakage no M3**: o rótulo `milestone_high` = `uhc_index ≥ 0.8` **e**
  `life_expectancy ≥ 70`; ambas as colunas são excluídas das features.
- **Timing do DiD**: primeiro ano ≥ 2000 em que o país cruza `uhc_index ≥ 0.5`;
  *always-treated* (já em 2000) ficam fora e controles *never-treated* exigem o
  proxy observado em ≥ 50% dos anos.

## Resultados

### Modelos (conjunto de teste ≥ 2020)

| Modelo | Tarefa | Métrica | XGBoost | Baseline linear |
|---|---|---|---|---|
| M1 `life_expectancy` | regressão | RMSE / R² | **2.54 / 0.887** | 3.25 / 0.815 |
| M2 `child_mortality` | regressão | RMSE / R² | **14.32 / 0.779** | 17.51 / 0.671 |
| M3 `milestone_high` | classificação | AUC / F1 | **0.994 / 0.912** | 0.980 / 0.847 |

As features mais importantes (|SHAP|) são mortalidade materna, fertilidade e
PIB/capita; no M3, PIB/capita, mortalidade materna e gasto em saúde.

### Testes de hipótese (`reports/hypotheses.json`)

| ID | Hipótese (resumo) | Efeito | Veredito |
|---|---|---|---|
| H1 | Médicos/1k ↑ ⇒ LE ↑ (líquido de PIB) | −1.09 | inconclusiva (sensível à especificação) |
| H2 | Saneamento > 80% ⇒ mortalidade <5 ↓ | −57.6 mortes/1k | **suportada** |
| H3 | Gasto público em saúde ↑ ⇒ LE ↑ | −0.53 | inconclusiva (sensível à especificação) |
| H4 | Urbanização ⇒ mortalidade <5 ↓ | −28.4 mortes/1k | **suportada** |
| H5 | Cruzar UHC ⇒ acelera LE (DiD) | −2.86 anos | rejeitada (convergência) |
| H6 | Vacinação (sarampo) ⇒ mortalidade <5 ↓ | −11.0 mortes/1k | **suportada** |

### A/B simulado e causal (`reports/*.json`)

- **A/B simulado** (uplift de +0.30 no UHC, unidade = país): LE **+2.04 anos**
  (p=0.074; com CUPED p=0.028) e mortalidade <5 **−18.8** (p<0.001, CUPED
  p<10⁻⁹); CUPED reduz a variância em ~80%.
- **Causal**: o event study mostra coeficientes pós-tratamento positivos, **mas
  pré-tendências positivas** (violação de tendências paralelas). O **controle
  sintético** (ex.: Moldávia, Líbia) indica **ATT negativo** — cruzar o limiar UHC
  não acelerou a LE, consistente com seleção e convergência (países mais pobres
  cresceram mais rápido).
- **Drift** (`reports/drift_report.json`): `gdp_per_capita` (PSI 0.42) e
  `fertility` (0.36) acima do limiar; a performance dos modelos no teste cresceu
  ~1.4× vs validação (regime pós-COVID), sem alerta no limiar de 1.5×.

## Estrutura

```
config/            indicators.yaml, settings.py, feast/
src/ingestion/     worldbank.py, who_gho.py
src/transform/     bronze_to_silver.py, build_abt.py, quality.py
src/features/      uhc_index.py (proxy UHC + timing DiD)
src/analysis/      eda.py, viz.py, hypotheses.py
src/modeling/      dataset.py, train.py, evaluate.py
src/abtesting/     simulate_ab.py, causal_did.py
src/serving/       app.py (FastAPI), predictor.py, feast_features.py, feast_cli.py
src/monitoring/    drift.py (PSI/KS)
src/utils/         db.py (DuckDB), io.py
notebooks/         01_eda … 05_ab_causal (executados)
dashboard/         app.py (Streamlit)
tests/             ingestion, transform, models, hypotheses, abtesting, serving, api
reports/           JSONs versionados (eda, hipóteses, A/B, causal, drift)
```

## Testes e CI

- `make test`: 31 testes (pytest), cobrindo ingestão, transform, features,
  modelagem, hipóteses, A/B, Feast e API — sem depender de rede/artefatos.
- `.github/workflows/ci.yml`: lint + testes + **smoke do pipeline** (ingest de
  amostra → silver → gold).
- `.github/workflows/monitoring.yml`: pipeline completo agendado + relatório de
  drift como artefato.
- `make lint` (ruff, line-length 100, isort first-party `src`/`config`).

## Notas de reprodutibilidade

- Dados, modelos e a online store são **gitignored**; os relatórios em `reports/`
  são versionados. `make clean` remove tudo que é gerado.
- Feast online store usa **sqlite** por padrão (`ONLINE_STORE_TYPE`); Redis via
  `docker compose up -d redis` (`make redis-up`).
