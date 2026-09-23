"""API FastAPI: predicoes (M1/M2/M3), catalogo de paises e resultados de experimentos.

Endpoints:
- GET  /health        status, modelos carregados, fonte de features
- GET  /countries     lista de paises (ISO3, nome, regiao, renda)
- POST /predict       features do pais (Feast online ou ABT por ano) + overrides "e se"
                      -> 3 predicoes + intervalos + features usadas
- GET  /experiments   A/B simulado + DiD causal + hipoteses (reports/*.json)
- POST /ab/simulate   roda o A/B simulado com uplift/seed escolhidos

Uso: uvicorn src.serving.app:app --port 8000   (ou make serve)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from config import settings
from src.modeling.dataset import FEATURES
from src.serving.predictor import FeatureProvider, load_models, log_prediction, predict_all


class PredictRequest(BaseModel):
    country_id: Annotated[str, Field(min_length=3, max_length=3, examples=["BRA"])]
    year: Annotated[int | None, Field(ge=1990, le=2100, description="None = mais recente")] = None
    overrides: dict[str, float | None] = Field(
        default_factory=dict, description="Cenario 'e se': substitui features (ex.: doctors_per_1000)")

    @field_validator("country_id")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("overrides")
    @classmethod
    def known_features(cls, v: dict) -> dict:
        unknown = set(v) - set(FEATURES)
        if unknown:
            raise ValueError(f"features desconhecidas: {sorted(unknown)}")
        return v


class Prediction(BaseModel):
    country_id: str
    features_year: int | None
    features_source: str
    predictions: dict
    features: dict[str, float | None]
    overrides_applied: list[str]


class ABRequest(BaseModel):
    uplift: Annotated[float, Field(gt=0, le=2.0)] = 0.2
    seed: int = 42


def create_app(provider: FeatureProvider | None = None, models: dict | None = None,
               log_predictions: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.provider = provider or FeatureProvider()
        app.state.models = models if models is not None else load_models()
        yield

    app = FastAPI(title="Global Health Analytics API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        m = request.app.state.models
        return {"status": "ok" if len(m) == 3 else "degraded",
                "models": {k: {"name": v.name, "test": v.test_metrics} for k, v in m.items()},
                "features_source_latest": request.app.state.provider.source_latest}

    @app.get("/countries")
    def countries(request: Request) -> list[dict]:
        return request.app.state.provider.countries().to_dict(orient="records")

    @app.post("/predict", response_model=Prediction)
    def predict(req: PredictRequest, request: Request) -> Prediction:
        st = request.app.state
        if not st.models:
            raise HTTPException(503, "modelos nao carregados; rode 'make train'")
        try:
            feats, year, source = st.provider.get(req.country_id, req.year)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from e
        feats = {**feats, **req.overrides}
        preds = predict_all(st.models, feats)
        if log_predictions:
            log_prediction({"country_id": req.country_id, "features_year": year,
                            "source": source, "overrides": sorted(req.overrides),
                            "features": feats, "predictions": preds})
        return Prediction(country_id=req.country_id, features_year=year, features_source=source,
                          predictions=preds, features=feats, overrides_applied=sorted(req.overrides))

    @app.get("/experiments")
    def experiments() -> dict:
        out = {}
        for key in ("ab_simulation", "causal_did", "hypotheses"):
            path = settings.REPORTS_DIR / f"{key}.json"
            out[key] = json.loads(path.read_text()) if path.exists() else None
        if not any(out.values()):
            raise HTTPException(404, "sem relatorios; rode 'make abtest hypotheses'")
        return out

    @app.post("/ab/simulate")
    def ab_simulate(req: ABRequest) -> dict:
        from src.abtesting.simulate_ab import simulate
        return simulate(uplift=req.uplift, seed=req.seed)

    return app


app = create_app()
