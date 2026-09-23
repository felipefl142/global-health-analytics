"""API: contrato dos endpoints com modelos treinados no painel sintetico (sem Feast)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.modeling import dataset
from src.modeling.dataset import FEATURES
from src.modeling.train import _xgb
from src.serving.app import create_app
from src.serving.predictor import FeatureProvider, LoadedModel


@pytest.fixture
def client(abt, monkeypatch):
    monkeypatch.setattr(dataset, "load_abt", lambda: abt)
    models = {}
    for target, kind in [("life_expectancy", "regression"), ("child_mortality", "regression"),
                         ("milestone_high", "classification")]:
        md = dataset.build_dataset(target)
        m = _xgb(kind).set_params(n_estimators=20, early_stopping_rounds=None)
        m.fit(md.X_train, md.y_train)
        models[target] = LoadedModel(target, md.name, kind, m, md.features,
                                     interval_q90=1.0 if kind == "regression" else None,
                                     interval_coverage_test=0.9, test_metrics={})
    app = create_app(provider=FeatureProvider(abt=abt, use_feast=False), models=models,
                     log_predictions=False)
    with TestClient(app) as c:
        yield c


def test_health_and_countries(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["features_source_latest"] == "abt"
    cs = client.get("/countries").json()
    assert {"country_id", "country_name", "region", "income"} <= set(cs[0])


def test_predict_latest_and_by_year(client):
    r = client.post("/predict", json={"country_id": "aaa"})
    assert r.status_code == 200
    body = r.json()
    assert body["country_id"] == "AAA" and body["features_year"] == 2023
    assert set(body["predictions"]) == {"life_expectancy", "child_mortality", "milestone_high"}
    le = body["predictions"]["life_expectancy"]
    assert le["interval_90"][0] < le["prediction"] < le["interval_90"][1]
    assert 0 <= body["predictions"]["milestone_high"]["probability"] <= 1
    assert set(body["features"]) == set(FEATURES)
    assert client.post("/predict", json={"country_id": "AAA", "year": 2005}).json()["features_year"] == 2005


def test_predict_overrides_change_prediction(client):
    base = client.post("/predict", json={"country_id": "AAA"}).json()
    # AAA e o pais "pior" do painel sintetico; leva todas as features p/ o nivel do melhor
    best = client.post("/predict", json={"country_id": "FFF"}).json()["features"]
    overrides = {k: v for k, v in best.items() if v is not None}
    what_if = client.post("/predict", json={"country_id": "AAA", "overrides": overrides}).json()
    assert what_if["overrides_applied"] == sorted(overrides)
    assert what_if["features"]["fertility"] == overrides["fertility"]
    assert what_if["predictions"] != base["predictions"]


def test_child_mortality_interval_non_negative(client):
    body = client.post("/predict", json={"country_id": "FFF", "overrides": {"fertility": 1.0}}).json()
    assert body["predictions"]["child_mortality"]["interval_90"][0] >= 0


@pytest.mark.parametrize("payload,status", [
    ({"country_id": "ZZZ"}, 404),
    ({"country_id": "AAA", "year": 1985}, 422),
    ({"country_id": "AAA", "year": 1995}, 404),       # fora da grade sintetica (1998+)
    ({"country_id": "AAA", "overrides": {"nope": 1}}, 422),
    ({"country_id": "TOOLONG"}, 422),
])
def test_predict_errors(client, payload, status):
    assert client.post("/predict", json=payload).status_code == status
