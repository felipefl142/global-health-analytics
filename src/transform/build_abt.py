"""Camada gold: constroi a Analytical Base Table (ABT) pais x ano.

A ABT e a tabela unica (1 linha por pais x ano, grade completa 1990-2023) que
unifica todos os indicadores (alvos, insumos UHC, covariados, WHO), o proxy UHC,
o timing de tratamento p/ DiD e os labels.

Saidas:
- data/gold/abt_country_year.parquet  (ABT wide)
- data/gold/abt_quality.json          (resumo de qualidade + validacao do proxy)

Uso: python -m src.transform.build_abt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import settings
from src.features.uhc_index import (
    DID_THRESHOLD,
    build_uhc_index,
    treatment_timing,
    validate_against_sci,
)
from src.utils.io import load_parquet, save_parquet

DIM_COLS = ["country_id", "country_name", "region", "income", "year"]
MILESTONE_UHC = 0.80
MILESTONE_LE = 70.0


def build_wide(long_df: pd.DataFrame, dim: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Long -> wide sobre a grade completa pais x ano (anos sem dado ficam NaN)."""
    wide = long_df.pivot_table(index=["country_id", "year"], columns="indicator",
                               values="value", aggfunc="first")
    wide.columns.name = None
    grid = pd.MultiIndex.from_product([dim["country_id"], range(start, end + 1)],
                                      names=["country_id", "year"])
    wide = wide.reindex(grid).reset_index()
    wide["year"] = wide["year"].astype(int)
    wide = wide.merge(dim[["country_id", "country_name", "region", "income"]],
                      on="country_id", how="left")
    other = sorted(c for c in wide.columns if c not in DIM_COLS)
    return wide[DIM_COLS + other]


def add_milestone(wide: pd.DataFrame, uhc_thr: float = MILESTONE_UHC,
                  le_thr: float = MILESTONE_LE) -> pd.DataFrame:
    """Label M3: marco 'alto' (proxy UHC >= 0.8 E expectativa >= 70). NaN se faltar insumo."""
    out = wide.copy()
    known = out["uhc_index"].notna() & out["life_expectancy"].notna()
    hit = (out["uhc_index"] >= uhc_thr) & (out["life_expectancy"] >= le_thr)
    out["milestone_high"] = pd.Series(np.where(known, hit.astype(int), np.nan),
                                      index=out.index).astype("Int64")
    return out


def quality_report(abt: pd.DataFrame) -> dict:
    numeric = [c for c in abt.columns if c not in DIM_COLS
               and pd.api.types.is_numeric_dtype(abt[c])]
    ms = abt["milestone_high"].dropna()
    return {
        "rows": int(len(abt)),
        "countries": int(abt["country_id"].nunique()),
        "years": [int(abt["year"].min()), int(abt["year"].max())],
        "columns": int(abt.shape[1]),
        "pct_missing": {c: round(float(abt[c].isna().mean()), 4) for c in numeric},
        "targets": {
            "life_expectancy": {"non_null": int(abt["life_expectancy"].notna().sum()),
                                "min": _safe(abt["life_expectancy"].min()),
                                "max": _safe(abt["life_expectancy"].max())},
            "child_mortality": {"non_null": int(abt["child_mortality"].notna().sum())},
            "milestone_high": {"pos": int(ms.sum()), "neg": int((ms == 0).sum()),
                               "unknown": int(abt["milestone_high"].isna().sum())},
        },
        "uhc_proxy": {
            "non_null": int(abt["uhc_index"].notna().sum()),
            "vs_official_sci": validate_against_sci(abt),
        },
        "treatment": {
            "threshold": DID_THRESHOLD,
            "treated_countries": int(abt.loc[abt["treated"], "country_id"].nunique()),
            "always_treated_countries": int(abt.loc[abt["always_treated"], "country_id"].nunique()),
        },
    }


def _safe(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def build_abt(long_df: pd.DataFrame, dim: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    wide = build_wide(long_df, dim, start, end)
    abt = build_uhc_index(wide)
    abt = treatment_timing(abt, threshold=DID_THRESHOLD)
    return add_milestone(abt)


def main() -> None:
    settings.ensure_dirs()
    cfg = settings.load_indicators()
    start, end = settings.date_range(cfg)
    print("Carregando silver -> gold (ABT) ...")
    long_df = load_parquet("silver", "indicators_long")
    dim = load_parquet("silver", "countries_dim")
    abt = build_abt(long_df, dim, start, end)
    out = save_parquet(abt, "gold", "abt_country_year")

    rep = quality_report(abt)
    qpath = settings.GOLD_DIR / "abt_quality.json"
    qpath.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"  ABT: {abt.shape[0]:,} rows x {abt.shape[1]} cols ({rep['countries']} paises)")
    print(f"  -> {out}\n  -> {qpath}")
    print(f"  proxy UHC non_null={rep['uhc_proxy']['non_null']:,} "
          f"vs SCI={rep['uhc_proxy']['vs_official_sci']}")
    print(f"  marco_alto={rep['targets']['milestone_high']} tratamento={rep['treatment']}")


if __name__ == "__main__":
    main()
