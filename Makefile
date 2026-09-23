# Orquestracao do pipeline global-health-analytics
PY      := .venv/bin/python
UV      := .venv/bin/uvicorn
ST      := .venv/bin/streamlit

.PHONY: help install ingest silver gold abt features train abtest hypotheses notebooks \
        feast-up feast-materialize serve dashboard test lint drift all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:            ## Cria venv e instala dependencias
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:             ## Ingestao World Bank API + WHO GHO -> bronze
	$(PY) -m src.ingestion.worldbank
	$(PY) -m src.ingestion.who_gho

silver:             ## bronze -> silver (limpo/estandarizado)
	$(PY) -m src.transform.bronze_to_silver

gold:               ## silver -> gold (ABT + proxy UHC) + checks de qualidade
	$(PY) -m src.transform.build_abt
	$(PY) -m src.transform.quality

abt: silver gold    ## Gera a camada gold/ABT completa

features:           ## (re)constroi o proxy UHC e feature views
	$(PY) -m src.features.uhc_index

train:              ## Treina M1/M2/M3
	$(PY) -m src.modeling.train

abtest:             ## A/B simulado + causal DiD
	$(PY) -m src.abtesting.simulate_ab
	$(PY) -m src.abtesting.causal_did

hypotheses:         ## Testes de hipotese H1-H6 -> reports/hypotheses.json
	$(PY) -m src.analysis.hypotheses

notebooks:          ## Gera e executa os notebooks 01-05 (com outputs)
	$(PY) notebooks/build_notebooks.py

redis-up:           ## Sobe Redis (online store) via docker-compose
	docker compose up -d redis

feast-materialize:  ## Materializa offline (DuckDB) -> online (Redis)
	$(PY) -m src.serving.feast_cli materialize

serve:              ## API FastAPI
	$(UV) src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:          ## Dashboard Streamlit
	$(ST) run dashboard/app.py

test:               ## Testes
	$(PY) -m pytest tests/ -v

lint:               ## Lint (ruff)
	$(PY) -m ruff check .

drift:              ## Monitoramento de drift
	$(PY) -m src.monitoring.drift

all: ingest silver gold train abtest hypotheses   ## Pipeline completo (dados -> modelos -> experimentos)

clean:              ## Remove camadas geradas e modelos
	rm -rf data/bronze/* data/silver/* data/gold/* data/*.duckdb models/* .feast
	touch data/bronze/.gitkeep data/silver/.gitkeep data/gold/.gitkeep models/.gitkeep
