"""Testes do A/B simulado e do timing causal (offline)."""
import numpy as np
import pandas as pd

from src.abtesting import simulate_ab
from src.features.uhc_index import timing_por_pais


def test_cuped_reduz_variancia():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 2 * x + rng.normal(0, 0.3, 500)
    _, reducao = simulate_ab._cuped(y, x)
    assert reducao > 0.5


def test_poder_cresce_com_o_efeito():
    fraco = simulate_ab._poder(0.2, 0.5, 100, 100)
    forte = simulate_ab._poder(2.0, 0.5, 100, 100)
    assert 0 <= fraco < forte <= 1


def test_estimar_detecta_diferenca():
    a = pd.Series(np.random.default_rng(1).normal(10, 1, 200))
    b = pd.Series(np.random.default_rng(2).normal(8, 1, 200))
    r = simulate_ab._estimar(a, b)
    assert r["efeito"] > 1 and r["p"] < 0.01 and r["se"] > 0


def test_timing_por_pais_regra_cobertura():
    anos = list(range(1990, 2024))
    tratado = pd.DataFrame({"country_code": "A", "year": anos, "uhc_index": 0.4,
                            "treat_year": 2005, "treated": True, "always_treated": False})
    nunca = pd.DataFrame({"country_code": "B", "year": anos, "uhc_index": 0.3,
                          "treat_year": pd.NA, "treated": False, "always_treated": False})
    esparso = pd.DataFrame({"country_code": "C", "year": anos[:5], "uhc_index": 0.2,
                            "treat_year": pd.NA, "treated": False, "always_treated": False})
    idx = pd.concat([tratado, nunca, esparso], ignore_index=True)

    timing = timing_por_pais(idx).set_index("country_code")
    assert bool(timing.loc["A", "treated"])
    assert not bool(timing.loc["A", "never_treated"])
    assert bool(timing.loc["B", "never_treated"])
    assert not bool(timing.loc["C", "never_treated"])  # cobertura < 50%
