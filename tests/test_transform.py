"""Silver -> gold: grade da ABT, proxy UHC, timing do tratamento, label M3, qualidade."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.uhc_index import build_uhc_index, treatment_timing
from src.transform.bronze_to_silver import to_silver
from src.transform.quality import check_abt
from tests.conftest import COUNTRIES, YEARS


def test_to_silver_filters_aggregates_and_years(countries_dim):
    raw = pd.DataFrame({
        "country_id": ["AAA", "WLD", "AAA", "AAA"],
        "year": ["2000", "2000", "1980", "2001"],
        "value": [1.0, 2.0, 3.0, None],
        "indicator": "x", "indicator_code": "X", "source": "worldbank",
    })
    out = to_silver(raw, countries_dim, 1990, 2023)
    assert out[["country_id", "year"]].values.tolist() == [["AAA", 2000]]


def test_abt_is_full_grid_one_row_per_country_year(abt):
    assert len(abt) == len(COUNTRIES) * len(YEARS)
    assert not abt.duplicated(["country_id", "year"]).any()
    assert {"region", "income", "uhc_index", "milestone_high", "treat_year"} <= set(abt.columns)


def test_uhc_index_bounded_and_tracks_sci(abt):
    u = abt["uhc_index"].dropna()
    assert u.between(0, 1).all()
    assert abt["uhc_index"].corr(abt["uhc_sci"]) > 0.9


def test_uhc_index_requires_min_weight():
    wide = pd.DataFrame({"country_id": ["A"] * 3, "year": [2000, 2001, 2002],
                         "measles_imm_pct": [50.0, 60.0, 70.0]})  # so 10% do peso
    out = build_uhc_index(wide)
    assert out["uhc_index"].isna().all()


def test_uhc_interpolation_only_inside_and_short_edges():
    years = list(range(2000, 2011))
    doctors = [1.0] + [np.nan] * 3 + [5.0] + [np.nan] * 6  # pontas: 2 anos de ffill
    wide = pd.DataFrame({"country_id": "A", "year": years, "doctors_per_1000": doctors,
                         "nurses_per_1000": doctors, "health_exp_per_capita": doctors})
    out = build_uhc_index(wide)
    # 2000..2004 interpolado, 2005-2006 ffill, 2007+ NaN
    assert out["uhc_index"].notna().tolist() == [True] * 7 + [False] * 4


def test_treatment_timing_excludes_always_treated_and_needs_sustained_cross():
    abt = pd.DataFrame({
        "country_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "year": list(range(2000, 2005)) * 3,
        "uhc_index": [0.4, 0.6, 0.4, 0.55, 0.6,     # A: oscila em 2001, sustentado em 2003
                      0.7, 0.8, 0.8, 0.9, 0.9,      # B: always treated
                      0.1, 0.2, 0.2, 0.3, 0.3],     # C: never treated
    })
    out = treatment_timing(abt, threshold=0.5, start_year=2000)
    by = out.groupby("country_id").first()
    assert by.loc["A", "treat_year"] == 2003
    assert bool(by.loc["B", "always_treated"]) and not bool(by.loc["B", "treated"])
    assert not bool(by.loc["C", "treated"]) and not bool(by.loc["C", "always_treated"])
    assert out.loc[out.country_id == "A", "post"].tolist() == [0, 0, 0, 1, 1]


def test_milestone_is_nan_when_inputs_missing(abt):
    missing = abt["uhc_index"].isna() | abt["life_expectancy"].isna()
    assert abt.loc[missing, "milestone_high"].isna().all()
    known = abt.loc[~missing]
    expected = ((known["uhc_index"] >= 0.8) & (known["life_expectancy"] >= 70)).astype(int)
    assert (known["milestone_high"].astype(int) == expected).all()


def test_quality_checks_pass_on_synthetic(abt):
    rep = check_abt(abt)
    assert rep["ok"], rep["issues"]
    assert rep["duplicated_country_years"] == 0


def test_range_checks_flag_impossible_values(abt):
    bad = abt.copy()
    bad.loc[bad.index[0], "sanitation_basic"] = 130.0
    bad.loc[bad.index[1], "fertility"] = -1.0
    rep = check_abt(bad)
    assert not rep["ok"]
    by = {r["indicator"]: r for r in rep["range_checks"]}
    assert by["sanitation_basic"]["violations"] == 1
    assert by["fertility"]["violations"] == 1
    assert any("sanitation_basic" in i for i in rep["issues"])
