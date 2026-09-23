"""Testes do feature store Feast (offline, sem materializar)."""
import pandas as pd

from config import settings
from src.serving import feast_features


def _abt_sintetica() -> pd.DataFrame:
    return pd.DataFrame({
        "country_code": ["BRA", "BRA"],
        "year": [2000, 2001],
        **{c: [1.0, 2.0] for c in feast_features.COLUNAS_ANUAIS},
        "region": ["Latin America & Caribbean"] * 2,
        "income_level": ["Upper middle income"] * 2,
    })


def test_construir_tabelas_feast(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GOLD_DIR", tmp_path)
    _abt_sintetica().to_parquet(tmp_path / "abt_country_year.parquet", index=False)

    caminhos = feast_features.construir_tabelas()
    anual = pd.read_parquet(caminhos["yearly"])
    estatico = pd.read_parquet(caminhos["static"])

    assert len(anual) == 2
    assert {"country_code", "event_timestamp", "created", *feast_features.COLUNAS_ANUAIS} <= set(
        anual.columns)
    assert len(estatico) == 1
    assert {"country_code", "event_timestamp", "region", "income_level"} <= set(
        estatico.columns)


def test_objetos_feast(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GOLD_DIR", tmp_path)
    _abt_sintetica().to_parquet(tmp_path / "abt_country_year.parquet", index=False)

    entidade, views, servico = feast_features.objetos()
    assert entidade.name == "country"
    assert {v.name for v in views} == {"country_health_yearly", "country_health_static"}
    assert servico.name == "health_prediction_service"
