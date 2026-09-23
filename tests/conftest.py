"""Fixtures sinteticas (sem rede): painel pequeno no formato das camadas silver/gold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.uhc_index import DEFAULT_COMPONENTS

COUNTRIES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
YEARS = list(range(1998, 2024))


@pytest.fixture
def countries_dim() -> pd.DataFrame:
    return pd.DataFrame({
        "country_id": COUNTRIES,
        "country_name": [f"Pais {c}" for c in COUNTRIES],
        "iso2": [c[:2] for c in COUNTRIES],
        "region": ["R1", "R1", "R2", "R2", "R3", "R3"],
        "income": ["Low income", "High income"] * 3,
        "latitude": np.linspace(-30, 50, 6),
        "longitude": np.linspace(-60, 100, 6),
    })


@pytest.fixture
def silver_long() -> pd.DataFrame:
    """Silver long com tendencia crescente por pais (paises 'melhores' tem base maior)."""
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(COUNTRIES):
        for t, y in enumerate(YEARS):
            level = i / (len(COUNTRIES) - 1) * 0.6 + t / len(YEARS) * 0.4  # 0..1
            vals = {name: 10 + 90 * level + rng.normal(0, 1) for name in DEFAULT_COMPONENTS}
            vals["health_exp_per_capita"] = np.exp(3 + 5 * level)
            vals["life_expectancy"] = 50 + 30 * level + rng.normal(0, 0.5)
            vals["child_mortality"] = 150 - 140 * level + rng.normal(0, 2)
            vals["gdp_per_capita"] = np.exp(6 + 4 * level)
            vals["urban_pct"] = 20 + 60 * level
            vals["fertility"] = 6 - 4.5 * level
            vals["uhc_sci"] = 30 + 60 * level
            for name, v in vals.items():
                rows.append({"country_id": c, "year": y, "indicator": name,
                             "indicator_code": name.upper(), "source": "worldbank", "value": v})
    return pd.DataFrame(rows)


@pytest.fixture
def abt(silver_long, countries_dim) -> pd.DataFrame:
    from src.transform.build_abt import build_abt
    return build_abt(silver_long, countries_dim, YEARS[0], YEARS[-1])
