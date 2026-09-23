"""API FastAPI: predicao, catalogo de paises e resultados experimentais."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from config import settings
from src.serving import predictor
from src.utils.io import read_json

app = FastAPI(title="Global Health Analytics API", version="0.1.0")


class PredictRequest(BaseModel):
    """Requisicao de predicao para um pais (com overrides opcionais de features)."""

    country_code: str = Field(..., min_length=3, max_length=3)
    sobrescrever: dict[str, float] | None = None


class ABSimRequest(BaseModel):
    """Parametros da simulacao A/B (uplift no indice UHC)."""

    delta: float = 0.30


@lru_cache
def get_preditor() -> predictor.Preditor:
    """Carrega (uma vez) os modelos de `models/`."""
    return predictor.Preditor()


@app.get("/health")
def health(preditor_: Annotated[predictor.Preditor, Depends(get_preditor)]) -> dict:
    """Status do servico e modelos carregados."""
    return {"status": "ok", "modelos": list(preditor_.modelos)}


@app.get("/countries")
def countries() -> list[dict]:
    """Lista de paises com regiao/renda."""
    abt = pd.read_parquet(
        settings.GOLD_DIR / "abt_country_year.parquet",
        columns=["country_code", "country_name", "region", "income_level"],
    )
    dados = abt.drop_duplicates("country_code").sort_values("country_code")
    return dados.to_dict(orient="records")


@app.post("/predict")
def predict(req: PredictRequest,
            preditor_: Annotated[predictor.Preditor, Depends(get_preditor)]) -> dict:
    """Prediz os 3 alvos usando features da online store (com overrides opcionais)."""
    valores = predictor.features_do_feast(req.country_code)
    if req.sobrescrever:
        valores.update(req.sobrescrever)
    return {
        "country_code": req.country_code,
        "predicoes": preditor_.prever(valores),
        "features_usadas": valores,
    }


@app.get("/experiments")
def experiments() -> dict:
    """Resultados do A/B simulado e do DiD causal."""
    saida = {}
    for nome in ("ab_simulation", "causal_did"):
        caminho = settings.REPORTS_DIR / f"{nome}.json"
        if caminho.exists():
            saida[nome] = read_json(caminho)
    return saida


@app.post("/ab/simulate")
def ab_simulate(req: ABSimRequest) -> dict:
    """Reexecuta o A/B simulado com um `delta` de UHC."""
    from src.abtesting.simulate_ab import simular

    return simular(delta=req.delta)
