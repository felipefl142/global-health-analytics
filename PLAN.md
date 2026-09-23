Perfeito — agora tenho tudo. Resumo do que ficou definido:
- Dataset: painel WHO/World Bank global health (país × ano, multi-fonte), 1990–2023, ~200+ países
- Stack: Python scripts + Jupyter (EDA) + Streamlit + DuckDB + Feast (offline=DuckDB, online=Redis) + FastAPI
- Produção: medallion bronze/silver/gold + ABT + feature store + CI/CD + drift monitoring
- A/B: simulado + causal (DiD) — estrela: expansão de UHC → expectativa de vida
- 3 modelos separados: (M1) expectativa de vida, (M2) mortalidade infantil, (M3) classificação de marcos (UHC≥80% e LE≥70 anos)
Aqui está o plano completo:

# Plano — Plataforma Global Health Analytics & Causal A/B (WHO/World Bank)

1. Objetivo
Construir, com ênfase em gerar conhecimento da base (EDA + viz + testes de hipótese), um pipeline de dados reprodutível sobre um painel global de saúde, com predição, um componente A/B (simulado + causal) e todo o serving em produção (DuckDB + medallion + Feast + FastAPI + Streamlit + CI/CD + drift).

2. Stack final
Camada	Tecnologia
Ingestão	requests (World Bank API), CSVs WHO GHE → Parquet
Warehouse	DuckDB (catálogo sobre Parquet)
Medallion	bronze/ (raw) → silver/ (limpo/estandarizado) → gold/ (curado)
ABT	abt_country_year (1 linha por país×ano, wide)
EDA/Hipóteses	Jupyter + pandas + statsmodels/linearmodels + scipy
Modelos	scikit-learn + LightGBM (baseline linear/ridge) + SHAP
Feature store	Feast (offline=DuckDB, online=Redis)
API	FastAPI + pydantic + uvicorn
Dashboard	Streamlit + plotly (+folium p/ mapas)
A/B + causal	simulação própria + DiD escalonado + controle sintético
CI/CD + monitoring	GitHub Actions + PSI/KS (drift) + log de predições
Orquestração	Makefile + docker-compose (Redis)

3. Fontes de dados e modelo de entidade
- Fonte 1 (primária): World Bank API (api.worldbank.org/v2) — séries país×ano, estruturadas.
- Fonte 2 (secundária): WHO Global Health Estimates (GHE) — CSVs de mortalidade por causa (enriquecimento).
- Dimensão país: nomes/ISO estandarizados (consolidação multi-fonte no silver).
- Entidade primária: country × year (painel). ABT = 1 linha por país×ano.
- Entidade secundária: country (snapshot mais recente) p/ visão transversal.
Indicadores-chave (códigos World Bank — validados contra a API na ingestão):
- Alvos: SP.DYN.LE00.IN (expectativa de vida), SH.DYN.MORT (mortalidade infantil)
- UHC/insumos: SH.XPD.MEDS.SP.ZS (gasto público/capita), SH.XPD.TOTL.GD.ZS, SH.HWF.PHYS (médicos/1k), SH.HWF.NURS (enfermeiros/1k), SH.MED.BEDS.ZS, saneamento/bebida, SH.IMM.MEAS/SH.IMM.DIPT (vacinas)
- UHC oficial (snapshot 2019): SH.UHC.NOEF (SCI), SH.UHC.FRPF, SH.UHC.NOEP
- Covariados: NY.GDP.PCAP.CD (PIB/capita), SP.URB.TOTL.IN.ZS (urbanização), SP.DYN.TOTL.FE (fertilidade), SP.POP.TOTL

4. Decisão de projeto importante (UHC)
O índice oficial UHC (SCI) só existe em 2019/2021 — não vira série p/ o DiD 1990–2023. Solução:
- Construir um proxy "UHC coverage" por país×ano a partir das séries longas (gasto/capita + densidade de profissionais + saneamento + vacinação), normalizando e agregando → vira feature derivada no gold (bom exercício de feature engineering).
- O SCI oficial 2019 entra como feature de snapshot + validação transversal.
- Tratamento do DiD = ano em que o país cruza o limiar do proxy UHC (timing escalonado por país).

5. Estrutura do repositório
global-health-analytics/
├── README.md · pyproject.toml · requirements.txt · .env.example
├── Makefile · docker-compose.yml (Redis)
├── config/            # indicators.yaml, settings.py, feast/
├── src/
│   ├── ingestion/     # worldbank.py, who_ghe.py            → bronze
│   ├── transform/     # bronze_to_silver.py, build_abt.py, quality.py
│   ├── features/      # uhc_index.py (proxy UHC)
│   ├── modeling/      # dataset.py, train_le.py, train_im.py, train_milestone.py, evaluate.py
│   ├── abtesting/     # simulate_ab.py, causal_did.py
│   ├── serving/       # app.py (FastAPI), feast_features.py
│   ├── monitoring/    # drift.py (PSI/KS)
│   └── utils/         # db.py, io.py
├── notebooks/         # 01_eda, 02_visualization, 03_hypothesis_testing, 04_modeling, 05_ab_causal
├── dashboard/         # app.py (Streamlit)
├── tests/             # ingestion, transform, models, api
├── data/              # bronze/silver/gold + duckdb (gitignored)
└── .github/workflows/ # ci.yml, monitoring.yml

6. Roadmap por fases (com critério de aceitação)
F0 — Setup & scaffolding
Repo, pyproject/requirements, config/, Makefile, docker-compose (Redis), data/ (bronze/silver/gold), .env.example, .gitignore, ruff + pytest.
✅ make install && make test roda; docker-compose up sobe Redis.
F1 — Ingestão → Bronze
worldbank.py (API → Parquet raw por fonte), who_ghe.py (CSV → Parquet). Validação dos códigos de indicador contra a API.
✅ make ingest produz data/bronze/*.parquet íntegros.
F2 — Transform → Silver + Gold (ABT)
bronze_to_silver.py (parse, tipagem, padronizar países, flags de qualidade, missing) → silver/. build_abt.py (consolida wide abt_country_year + proxy UHC) → gold/.
✅ make silver && make gold gera ABT 1-linha/país×ano; testes de qualidade (cobertura, outliers, consistency).
F3 — EDA (núcleo de conhecimento)
Notebook 01_eda.ipynb: cobertura país/ano, distribuições, missingness matrix, correlações, sazonalidade/tendências, outliers, comparativo regional.
✅ Documenta a base: o que é confiável, o que falta, por quê.
F4 — Visualizações
02_visualization.ipynb: time series, choropleth (folium), heatmaps, boxplots por região/renda, evoluções de indicadores.
✅ Galeria de viz reutilizada no dashboard.
F5 — Testes de hipótese (núcleo de conhecimento)
03_hypothesis_testing.ipynb — cada hipótese com H0/H1, teste, tamanho de efeito + IC + p + significância prática. Catálogo (ajustável):
- H1: densidade de médicos ↑ ⇒ expectativa de vida ↑ (net de PIB/capita) — regressão de painel
- H2: saneamento >80% ⇒ mortalidade infantil significativamente menor — t-test/Mann-Whitney + efeito
- H3: +1 DP no gasto público/capita ⇒ Δ mensurável na expectativa de vida
- H4: urbanização associada a mortalidade infantil menor
- H5: pós-cruzar limiar UHC, crescimento da expectativa de vida acelera (DiD)
- H6: vacinação sarampo associada a mortalidade <5 menor
✅ Tabela consolidada de hipóteses com veredito.
F6 — Predição (3 modelos separados)
04_modeling.ipynb + src/modeling/:
- M1 expectativa de vida (regressão)
- M2 mortalidade infantil (regressão)
- M3 classificação de marco (UHC≥80% e LE≥70 → alta/baixa)
Split time/group-aware (evita leakage país/ano). Baseline linear/ridge + LightGBM; eval (RMSE/MAE/R²; AUC/F1/calibração) + SHAP p/ interpretação.
✅ 3 artefatos versionados (models/*.joblib), relatórios de eval.
F7 — A/B + Causal
05_ab_causal.ipynb + src/abtesting/:
- Simulado: randomiza país×anos em treatment/control, aplica uplift no proxy UHC, prevê com o modelo, roda A/B completo (power, t-test, IC, Cohen's d, CUPED, correção múltipla).
- Causal (estrela): DiD escalonado UHC→expectativa de vida (linearmodels/pyfixest, FE país+ano, SE robustas cluster-por-país) + controle sintético p/ 1–2 países "herói". Diagnósticos: parallel trends + event study.
✅ Efeitos + ICs + plots de event study/sintético.
F8 — Feature store (Feast)
Offline=DuckDB (lê gold/ABT), online=Redis. Feature views: country_health_static, country_health_yearly (entity=country, ts=ano) + feature service health_prediction_service.
✅ make feast-materialize popula a online store; get_online_features retorna por país.
F9 — API (FastAPI)
/health, /countries, POST /predict (busca features na online store, roda os 3 modelos, retorna predições + IC + features usadas), GET /experiments (resultados A/B + DiD), POST /ab/simulate (opcional). Validação pydantic + testes.
✅ make serve; testes com TestClient passando.
F10 — Dashboard (Streamlit)
Páginas: EDA explorer (país/indicador/ano, séries, missingness, mapa), Hipóteses (tabela + efeito/IC/p), Predição (input → 3 saídas + SHAP), Experimentos (A/B + DiD event study + sintético).
✅ make dashboard navegável.
F11 — CI/CD + drift monitoring
- ci.yml: lint (ruff) + pytest + smoke do pipeline (ingest amostra → silver → gold).
- monitoring.yml/drift.py: PSI/KS por feature (janela de treino vs. recente/serving) + log de predições/resíduos; alerta por threshold.
✅ CI verde no push; drift report gerado e exposto no dashboard.
F12 — Hardening & docs
README (arquitetura, decisões, como rodar), Makefile completo, .env, documentação das hipóteses e dos modelos, git tags de versão.
✅ make all reproduz tudo do zero.

7. Decisões/riscos a monitorar
- UHC SCI só 2019 → mitigado com proxy UHC de painel (F2/F7).
- Missingness em indicadores de saúde global → tratado no silver + flags; modelos robustos.
- Leakage temporal em painel → splits time/group-aware + CUPED no A/B.
- Feast + Redis precisa de docker-compose — ok.

Próximo passo
Posso começar a F0–F2 (setup + ingestão + medallion + ABT) agora. Confirma só 2 coisas:
1. Nome/pasta do repo: usar global-health-analytics/ em /home/felipefrl/coding/python/?
2. Repo git: quer que eu faça git init + commits por fase?
Me dá o ok (e os ajustes) que eu saio do plan mode e começo a construir.
