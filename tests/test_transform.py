"""Testes de transform e features (silver/gold/proxy UHC) - offline."""
import json

import pandas as pd

from config import settings
from src.features.uhc_index import _normalizar, compute_timing
from src.transform import bronze_to_silver, quality


def test_normalizar_percentis():
    """Normalizacao 5/95 deve mapear p5->0, p95->1 e recortar os extremos."""
    serie = pd.Series(range(101), dtype="float")
    norm = _normalizar(serie)
    assert norm.min() == 0 and norm.max() == 1
    assert norm.iloc[5] == 0 and norm.iloc[95] == 1
    assert norm.iloc[0] == 0 and norm.iloc[100] == 1


def test_compute_timing():
    """Timing do DiD: cruzamento em 2005, always-treated (2000) e never-treated."""
    idx = pd.DataFrame({
        "country_code": ["A"] * 11 + ["B"] * 11 + ["C"] * 11,
        "year": list(range(2000, 2011)) * 3,
        "uhc_index": [0.4] * 5 + [0.6] * 6 + [0.9] * 11 + [0.2] * 11,
    })
    out = compute_timing(idx)

    a = out[out["country_code"] == "A"]
    assert int(a["treat_year"].iloc[0]) == 2005
    assert bool(a["treated"].iloc[0])
    assert a.loc[a["year"] >= 2005, "post"].all()
    assert not a.loc[a["year"] < 2005, "post"].any()

    b = out[out["country_code"] == "B"]
    assert bool(b["always_treated"].iloc[0])
    assert int(b["treat_year"].iloc[0]) == 2000

    c = out[out["country_code"] == "C"]
    assert bool(c["never_treated"].iloc[0])
    assert pd.isna(c["treat_year"].iloc[0])


def test_quality_checar_faixas():
    """Valores fora da faixa plausivel devem ser contados como violacao."""
    df = pd.DataFrame({
        "life_expectancy": [50.0, 150.0],
        "uhc_index": [0.5, -0.1],
        "year": [2000, 2001],
    })
    por_indicador = {c["indicator"]: c for c in quality.checar_faixas(df)}
    assert por_indicador["life_expectancy"]["violations"] == 1
    assert por_indicador["uhc_index"]["violations"] == 1


def test_bronze_parsing(tmp_path, monkeypatch):
    """O parse do bronze deve limpar/padronizar e filtrar agregados do WHO."""
    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path)
    (tmp_path / "worldbank").mkdir()
    (tmp_path / "who").mkdir()

    (tmp_path / "worldbank" / "_countries.json").write_text(json.dumps({"data": [{
        "id": "BRA", "name": "Brazil",
        "region": {"value": "Latin America & Caribbean "},
        "incomeLevel": {"value": "Upper middle income"},
        "lendingType": {"value": "IBRD"},
        "capitalCity": "Brasilia", "longitude": "-47", "latitude": "-15",
    }]}))
    (tmp_path / "worldbank" / "SP.DYN.LE00.IN.json").write_text(json.dumps({"data": [
        {"countryiso3code": "BRA", "country": {"id": "BRA", "value": "Brazil"},
         "date": "2000", "value": 70.0},
        {"countryiso3code": "BRA", "country": {"id": "BRA", "value": "Brazil"},
         "date": "2001", "value": None},
    ]}))
    (tmp_path / "who" / "NCDMORT3070.json").write_text(json.dumps({"data": [
        {"SpatialDimType": "COUNTRY", "SpatialDim": "BRA", "TimeDim": 2000, "NumericValue": 10.0},
        {"SpatialDimType": "REGION", "SpatialDim": "AMR", "TimeDim": 2000, "NumericValue": 12.0},
    ]}))

    cfg = settings.load_indicators()
    dim = bronze_to_silver.build_countries_dim()
    assert dim.loc[0, "country_code"] == "BRA"
    assert not bool(dim.loc[0, "is_aggregate"])

    longo = bronze_to_silver.build_worldbank_long(cfg)
    assert len(longo) == 1
    assert longo.loc[0, "indicator"] == "life_expectancy"
    assert longo.loc[0, "value"] == 70.0

    who = bronze_to_silver.build_who_long(cfg)
    assert len(who) == 1
    assert who.loc[0, "indicator"] == "ncd_mortality_30_70"
