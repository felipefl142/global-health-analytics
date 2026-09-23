"""Analise: DiD escalonado recupera efeito conhecido, controle sintetico, A/B e hipoteses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.abtesting import causal_did as cd
from src.abtesting.simulate_ab import analyze, apply_uplift, power_analysis
from src.analysis import eda
from src.analysis.hypotheses import cliffs_delta, panel_fe


def _did_panel(effect: float, diff_trend: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Painel sintetico: 10 nunca-tratados + 3 coortes; efeito constante pos-adocao."""
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = {f"T{i}": 2006 + 3 * (i % 3) for i in range(12)}
    units = {**cohorts, **{f"N{i}": None for i in range(15)}}
    for u, g in units.items():
        base = rng.normal(60, 5)
        for y in range(2000, 2021):
            trend = 0.3 * (y - 2000) + (diff_trend * (y - 2000) if g else 0)
            val = base + trend + (effect if g and y >= g else 0) + rng.normal(0, 0.1)
            rows.append({"country_id": u, "year": y, "life_expectancy": val,
                         "treated": g is not None, "treat_year": float(g) if g else np.nan,
                         "post": int(g is not None and y >= g), "gdp_per_capita": 1000.0,
                         "population": 1e6})
    return pd.DataFrame(rows)


def test_staggered_did_recovers_known_effect():
    df = _did_panel(effect=2.0)
    r = cd.staggered_did(df, n_boot=49)
    assert r["att"] == pytest.approx(2.0, abs=0.15)
    assert r["ci95"][0] < 2.0 < r["ci95"][1]
    assert r["pretrend_ok"]


def test_detrended_did_removes_differential_trend():
    df = _did_panel(effect=0.0, diff_trend=-0.15)
    naive = cd.staggered_did(df, n_boot=29)
    adj = cd.staggered_did(df, n_boot=29, detrend=True)
    assert naive["att"] < -0.3            # tendencia vira "efeito" espurio
    assert abs(adj["att"]) < 0.15         # ajuste remove
    assert not naive["pretrend_ok"]


def test_synthetic_control_null_effect_not_significant():
    df = _did_panel(effect=0.0)
    unit = cd.pick_hero_countries(df, n=1)[0]
    r = cd.synthetic_control(df, unit)
    assert abs(r["att_post_mean"]) < 0.5
    assert r["placebo_p"] > 0.05


def test_ab_analyze_detects_effect_and_cuped_reduces_variance():
    rng = np.random.default_rng(1)
    n = 400
    x_pre = rng.normal(70, 8, n)
    treat = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    y = x_pre + 1.0 * treat + rng.normal(0, 1, n)
    r = analyze(y, treat, x_pre)
    assert r["cuped"]["variance_reduction"] > 0.9
    assert r["cuped"]["p"] < 0.001
    assert r["cuped"]["ci95"][0] < 1.0 < r["cuped"]["ci95"][1]


def test_power_analysis_mde_shrinks_with_n():
    small = power_analysis(sd=5, n_per_arm=50, effect=1)
    big = power_analysis(sd=5, n_per_arm=500, effect=1)
    assert big["mde_abs"] < small["mde_abs"]
    assert small["n_per_arm_for_80pct_power"] > 50


def test_uplift_caps_percentages():
    X = pd.DataFrame({"sanitation_basic": [90.0], "doctors_per_1000": [2.0], "gdp_per_capita": [1e4]})
    out = apply_uplift(X, 0.2)
    assert out.loc[0, "sanitation_basic"] == 100
    assert out.loc[0, "doctors_per_1000"] == pytest.approx(2.4)
    assert out.loc[0, "gdp_per_capita"] == 1e4   # covariado nao muda


def test_panel_fe_recovers_within_effect():
    rng = np.random.default_rng(3)
    rows = []
    for c in range(40):
        fe = rng.normal(0, 10)
        for y in range(2000, 2015):
            x = rng.normal(0, 1)
            rows.append({"country_id": f"C{c}", "year": y, "x": x, "log_gdp_pc": rng.normal(8, 1),
                         "y": fe + 2.0 * x + rng.normal(0, 0.5)})
    r = panel_fe(pd.DataFrame(rows), "y", "x")
    assert r["coef"] == pytest.approx(2.0 * pd.DataFrame(rows)["x"].std(), rel=0.1)


def test_cliffs_delta_extremes():
    assert cliffs_delta(np.array([1, 2]), np.array([3, 4])) == -1
    assert cliffs_delta(np.array([5]), np.array([5])) == 0


def test_eda_helpers(abt):
    cov = eda.coverage_by_indicator(abt)
    assert (cov["pct_filled"] <= 100).all()
    assert "life_expectancy" in set(cov["indicator"])
    tr = eda.trends(abt, "life_expectancy")
    assert set(tr["income"]) == {"Low income", "High income"}
    assert not eda.latest_snapshot(abt, "life_expectancy").duplicated("country_id").any()
