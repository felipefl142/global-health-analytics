"""Testes da API FastAPI (sem modelos nem Feast reais)."""
import json

import pytest
from fastapi.testclient import TestClient

from config import settings
from src.serving import app as app_module
from src.serving import predictor


class FakePreditor:
    """Preditor falso para isolar a API dos artefatos treinados."""

    modelos = {"life_expectancy": None, "child_mortality": None, "milestone_high": None}

    def prever(self, valores):
        return {
            "life_expectancy": {"valor": 70.0, "ic90": [68.0, 72.0]},
            "child_mortality": {"valor": 20.0, "ic90": [15.0, 25.0]},
            "milestone_high": {"probabilidade": 0.9, "classe": 1},
        }


@pytest.fixture(autouse=True)
def _limpar_overrides():
    app_module.app.dependency_overrides[app_module.get_preditor] = lambda: FakePreditor()
    yield
    app_module.app.dependency_overrides.clear()


def _client() -> TestClient:
    return TestClient(app_module.app)


def test_health():
    resposta = _client().get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict(monkeypatch):
    monkeypatch.setattr(predictor, "features_do_feast",
                        lambda cc: {"gdp_per_capita": 1000.0})
    resposta = _client().post("/predict", json={"country_code": "BRA"})
    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["predicoes"]["milestone_high"]["classe"] == 1
    assert corpo["features_usadas"]["gdp_per_capita"] == 1000.0


def test_predict_valida_iso3():
    resposta = _client().post("/predict", json={"country_code": "BR"})
    assert resposta.status_code == 422


def test_experiments(tmp_path, monkeypatch):
    (tmp_path / "ab_simulation.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "causal_did.json").write_text(json.dumps({"y": 2}))
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path)

    resposta = _client().get("/experiments")
    assert resposta.status_code == 200
    assert resposta.json() == {"ab_simulation": {"x": 1}, "causal_did": {"y": 2}}
