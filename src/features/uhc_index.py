"""Proxy de cobertura UHC (service coverage) por pais x ano.

O indice oficial UHC (SH_UHC_SCI) so existe em 2019. Para o painel 1990-2023
construimos um proxy a partir de series longas de insumos de saude:
normalizacao robusta (percentis 5/95) + media ponderada dos componentes.

Uso:
    from src.features.uhc_index import build_uhc_index
    df = build_uhc_index(wide_df)   # adiciona coluna 'uhc_index' (0..1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Componentes do proxy (colunas da ABT wide) e pesos (soma = 1.0)
DEFAULT_COMPONENTS: dict[str, float] = {
    "doctors_per_1000": 0.15,
    "nurses_per_1000": 0.15,
    "beds_per_1000": 0.10,
    "health_exp_per_capita": 0.20,
    "sanitation_safely": 0.10,
    "water_safely": 0.10,
    "measles_imm_pct": 0.10,
    "dpt_imm_pct": 0.10,
}
LO, HI = 0.05, 0.95  # percentis p/ normalizacao robusta


def _robust_minmax(s: pd.Series, lo: float = LO, hi: float = HI) -> pd.Series:
    p_lo, p_hi = np.nanpercentile(s, [lo * 100, hi * 100])
    denom = (p_hi - p_lo)
    if not np.isfinite(denom) or denom == 0:
        return pd.Series(np.nan, index=s.index)
    return ((s - p_lo) / denom).clip(0, 1)


def build_uhc_index(
    wide: pd.DataFrame,
    components: dict[str, float] | None = None,
    col: str = "uhc_index",
) -> pd.DataFrame:
    """Adiciona a coluna de proxy UHC (0..1) ao DataFrame wide."""
    comps = components or DEFAULT_COMPONENTS
    total_w = sum(comps.values())
    norm = {}
    for name, w in comps.items():
        if name not in wide.columns:
            continue
        norm[name] = _robust_minmax(pd.to_numeric(wide[name], errors="coerce")) * (w / total_w)
    if not norm:
        raise ValueError("nenhum componente do proxy UHC encontrado na ABT wide")
    idx = sum(norm.values())
    out = wide.copy()
    out[col] = idx
    return out


def treatment_timing(abt: pd.DataFrame, threshold: float = 0.5, col: str = "uhc_index") -> pd.DataFrame:
    """Ano em que cada pais cruza o limiar do proxy UHC (p/ DiD escalonado).

    Retorna a ABT com colunas:
      treated (bool) - cruzou o limiar em algum ano
      treat_year (int) - primeiro ano >= limiar (ou NaN)
      post (int) - 1 se ano >= treat_year e treated, senao 0
    """
    out = abt.copy()
    treat_year = {}
    for cid, g in out.groupby("country_id"):
        g = g.sort_values("year")
        crossed = g[g[col] >= threshold]
        treat_year[cid] = crossed["year"].min() if len(crossed) else np.nan
    out["treat_year"] = out["country_id"].map(treat_year)
    out["treated"] = out["treat_year"].notna()
    out["post"] = np.where(out["treated"] & (out["year"] >= out["treat_year"]), 1, 0)
    return out


if __name__ == "__main__":
    from src.utils.io import load_parquet
    abt = load_parquet("gold", "abt_country_year")
    res = build_uhc_index(abt)
    res = treatment_timing(res)
    print(res[["country_id", "year", "uhc_index", "treat_year", "post"]].head(10))
    print(f"\nthreshold=0.5 -> {res['treated'].sum()} paises-x-ano tratados; "
          f"{res[res['treated']]['treat_year'].nunique()} paises distintos")
