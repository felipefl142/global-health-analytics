"""Drift: PSI/KS detectam deslocamento de valor; cobertura e sinal separado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import feature_drift, psi, run


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.5


def test_missingness_does_not_count_as_value_drift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 500), [np.nan] * 500]})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    d = feature_drift(ref, cur, ["x"]).iloc[0]
    assert not d["drift"]
    assert d["missing_shift"]


def test_run_flags_shifted_feature(abt):
    rep = run(abt, with_performance=False)
    by_f = {r["feature"]: r for r in rep["data_drift"]}
    # o painel sintetico tem tendencia crescente: features de nivel derivam entre <=2015 e >=2020
    assert by_f["urban_pct"]["drift"]
    assert rep["alert"]
