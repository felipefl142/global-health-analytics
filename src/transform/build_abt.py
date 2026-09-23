"""Camada gold: constroi a Analytical Base Table (ABT) pais x ano.

A ABT e a tabela unica (1 linha por pais x ano) que unifica todos os
indicadores (alvos, insumos UHC, covariados), o proxy UHC e os labels.

Saidas:
- data/gold/abt_country_year.parquet  (ABT wide)
- data/gold/abt_quality.json          (resumo de qualidade)

Uso: python -m src.transform.build_abt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import settings
from src.features.uhc_index import build_uhc_index, treatment_timing
from src.utils.io import load_parquet, save_parquet


def code_to_name(cfg: dict) -> dict[str, str]:
    rev = {code: name for name, code in settings.all_indicator_codes(cfg).items()}
    return rev


def build_wide(long_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rev = code_to_name(cfg)
    wide = long_df.pivot_table(
        index=["country_id", "country_name", "iso_alpha3", "region", "income", "year"],
        columns="indicator_code",
        values="value",
        aggfunc="first",
    ).reset_index()
    # renomeia colunas de codigo para nome logico
    rename = {c: rev[c] for c in wide.columns if c in rev}
    wide = wide.rename(columns=rename)
    # garante ordem: dimensoes primeiro
    dim_cols = ["country_id", "country_name", "iso_alpha3", "region", "income", "year"]
    other = [c for c in wide.columns if c not in dim_cols]
    wide = wide[dim_cols + other]
    return wide


def add_milestone(wide: pd.DataFrame, uhc_thr: float = 0.80, le_thr: float = 70.0) -> pd.DataFrame:
    """Label M3: marco de saude 'alto' (UHC proxy >= 80% E expectativa >= 70)."""
    out = wide.copy()
    has_uhc = out["uhc_index"].notna()
    cond = np.where(has_uhc, out["uhc_index"] >= uhc_thr, np.nan)
    le_ok = out["life_expectancy"] >= le_thr
    out["milestone_high"] = np.where(
        np.isnan(cond), 0,
        np.where((cond) & (le_ok), 1, 0),
    ).astype(int)
    return out


def quality_report(abt: pd.DataFrame) -> dict:
    feature_cols = [c for c in abt.columns if c not in
                    ("country_id", "country_name", "iso_alpha3", "region", "income",
                     "year", "treated", "post", "treat_year")]
    rep = {
        "rows": int(len(abt)),
        "countries": int(abt["country_id"].nunique()),
        "years": [int(abt["year"].min()), int(abt["year"].max())],
        "columns": int(abt.shape[1]),
        "pct_missing": {
            c: round(float(abt[c].isna().mean()), 4) for c in feature_cols
            if pd.api.types.is_numeric_dtype(abt[c])
        },
        "targets": {
            "life_expectancy": {"non_null": int(abt["life_expectancy"].notna().sum()),
                                "min": _safe(abt["life_expectancy"].min()),
                                "max": _safe(abt["life_expectancy"].max())},
            "child_mortality": {"non_null": int(abt["child_mortality"].notna().sum())},
            "milestone_high": {"pos": int(abt["milestone_high"].sum()),
                               "neg": int((1 - abt["milestone_high"]).sum())},
        },
        "treatment": {"treated_countries": int(abt.loc[abt["treated"], "country_id"].nunique())},
    }
    return rep


def _safe(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def main() -> None:
    settings.ensure_dirs()
    cfg = settings.load_indicators()
    print("Carregando silver -> gold (ABT) ...")
    long_df = load_parquet("silver", "worldbank_long")
    wide = build_wide(long_df, cfg)
    print(f"  wide: {wide.shape[0]:,} rows x {wide.shape[1]} cols "
          f"({wide['country_id'].nunique()} paises)")

    abt = build_uhc_index(wide)
    abt = treatment_timing(abt, threshold=0.5)
    abt = add_milestone(abt)
    out = save_parquet(abt, "gold", "abt_country_year")

    rep = quality_report(abt)
    qpath = settings.GOLD_DIR / "abt_quality.json"
    qpath.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"  -> {out}")
    print(f"  -> {qpath}")
    print(f"  paises={rep['countries']} anos={rep['years']} "
          f"marco_alto={rep['targets']['milestone_high']['pos']} "
          f"tratados={rep['treatment']['treated_countries']}")


if __name__ == "__main__":
    main()
