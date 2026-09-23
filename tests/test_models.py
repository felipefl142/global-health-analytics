"""Testes de modelagem (dataset/evaluate) - offline."""
import numpy as np
import pandas as pd

from src.modeling import dataset
from src.modeling.evaluate import metricas_classificacao, metricas_regressao


def test_features_excluem_alvos_e_metadados():
    abt = pd.DataFrame(columns=[
        "country_code", "year", "life_expectancy", "child_mortality", "uhc_index",
        "uhc_sci", "gdp_per_capita", "doctors_per_1000", "treated",
    ])
    feats = dataset.features_disponiveis(abt)
    assert "gdp_per_capita" in feats and "doctors_per_1000" in feats
    for proibido in ["life_expectancy", "child_mortality", "uhc_index", "uhc_sci",
                     "year", "country_code", "treated"]:
        assert proibido not in feats


def test_rotular_marco():
    abt = pd.DataFrame({
        "uhc_index": [0.9, 0.7, 0.9, 0.85],
        "life_expectancy": [80, 80, 60, 72],
    })
    out = dataset._rotular_marco(abt)
    assert list(out["milestone_high"]) == [1, 0, 0, 1]


def test_split_temporal():
    df = pd.DataFrame({"year": [2010, 2015, 2016, 2019, 2020, 2023], "x": range(6)})
    partes = dataset.split_temporal(df)
    assert partes["train"]["year"].max() <= 2015
    assert partes["valid"]["year"].between(2016, 2019).all()
    assert partes["test"]["year"].min() >= 2020
    assert len(df) == sum(len(v) for v in partes.values())


def test_metricas_regressao():
    m = metricas_regressao(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m["rmse"] > 0 and m["mae"] > 0 and m["r2"] < 1


def test_metricas_classificacao():
    m = metricas_classificacao(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and m["f1"] == 1.0 and m["brier"] < 0.1
