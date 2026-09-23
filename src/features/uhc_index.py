"""Proxy de cobertura UHC (0..1) por pais x ano + timing do DiD.

O SCI oficial cobre 2000-2023; para o painel 1990+ construimos um `uhc_index` a
partir dos insumos estruturais (gasto, forca de trabalho, saneamento/agua,
vacinacao). Cada componente e normalizado por percentis 5/95 (com inversao do
out-of-pocket e log do gasto per capita) e combinado por media ponderada,
exigindo ao menos 50% do peso disponivel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings

# Pesos dos componentes (somam 1.0). health_exp_gdp e public_health_exp_gdp
# entram com peso menor por sobreposicao com o gasto per capita.
PESOS: dict[str, float] = {
    "health_exp_per_capita": 0.20,
    "health_exp_gdp": 0.05,
    "public_health_exp_gdp": 0.05,
    "out_of_pocket": 0.10,
    "doctors_per_1000": 0.15,
    "nurses_per_1000": 0.10,
    "beds_per_1000": 0.05,
    "sanitation_basic": 0.10,
    "water_basic": 0.10,
    "measles_imm_pct": 0.05,
    "dpt_imm_pct": 0.05,
}
COMPONENTES_LOG = {"health_exp_per_capita"}
COMPONENTES_INVERTIDOS = {"out_of_pocket"}
COBERTURA_MINIMA = 0.5

LIMIAR_TRATAMENTO = 0.5
ANO_BASE = 2000


def _normalizar(s: pd.Series) -> pd.Series:
    """Normaliza por percentis 5/95 para o intervalo [0, 1]."""
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi <= lo:
        return pd.Series(np.nan, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def _carregar_insumos() -> pd.DataFrame:
    """Le o silver e monta a matriz pais x ano dos insumos UHC (sem agregados)."""
    long = pd.read_parquet(settings.SILVER_DIR / "worldbank_long.parquet")
    dim = pd.read_parquet(settings.SILVER_DIR / "countries_dim.parquet")
    validos = set(dim.loc[~dim["is_aggregate"], "country_code"])
    long = long[long["indicator"].isin(PESOS) & long["country_code"].isin(validos)]
    wide = long.pivot_table(
        index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean"
    ).reset_index()
    for comp in PESOS:
        if comp not in wide.columns:
            wide[comp] = np.nan
    return wide


def _interpolar(wide: pd.DataFrame) -> pd.DataFrame:
    """Interpola lacunas intra-pais (limite de 5 anos em cada direcao)."""
    wide = wide.sort_values(["country_code", "year"]).copy()
    for comp in PESOS:
        wide[comp] = wide.groupby("country_code")[comp].transform(
            lambda s: s.interpolate(limit=5, limit_direction="both")
        )
    return wide


def _calcular_indice(wide: pd.DataFrame) -> pd.Series:
    """Media ponderada dos componentes normalizados, exigindo cobertura minima."""
    norm = {}
    for comp in PESOS:
        serie = wide[comp]
        if comp in COMPONENTES_LOG:
            serie = np.log1p(serie.where(serie >= 0))
        n = _normalizar(serie)
        if comp in COMPONENTES_INVERTIDOS:
            n = 1 - n
        norm[comp] = n
    norm_df = pd.DataFrame(norm, index=wide.index)
    pesos = pd.Series(PESOS)
    disponivel = norm_df.notna().mul(pesos, axis=1).sum(axis=1)
    ponderado = norm_df.fillna(0).mul(pesos, axis=1).sum(axis=1)
    indice = ponderado / disponivel
    indice[disponivel < COBERTURA_MINIMA] = np.nan
    return indice


def compute_timing(idx: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de tratamento do DiD (treated/treat_year/post).

    Tratamento = primeiro ano >= ANO_BASE em que o pais cruza LIMIAR_TRATAMENTO.
    Paises ja tratados em ANO_BASE (`always_treated`) e nunca tratados tambem sao
    sinalizados para os filtros do DiD.
    """
    idx = idx.sort_values(["country_code", "year"]).copy()
    mask = (idx["year"] >= ANO_BASE) & (idx["uhc_index"] >= LIMIAR_TRATAMENTO)
    primeiro = idx[mask].groupby("country_code")["year"].min()
    idx["treat_year"] = idx["country_code"].map(primeiro).astype("Int64")
    idx["treated"] = idx["treat_year"].notna()
    idx["always_treated"] = idx["treat_year"].eq(ANO_BASE)
    idx["post"] = (idx["year"] >= idx["treat_year"]).fillna(False).astype(bool)
    idx["never_treated"] = idx["treat_year"].isna()
    return idx


def build_uhc_index() -> pd.DataFrame:
    """Constroi o `uhc_index` por pais x ano e o timing do DiD."""
    wide = _carregar_insumos()
    wide = _interpolar(wide)
    wide["uhc_index"] = _calcular_indice(wide)
    cols = ["country_code", "year", "uhc_index"]
    idx = wide[cols].dropna(subset=["uhc_index"]).reset_index(drop=True)
    return compute_timing(idx)


def run() -> pd.DataFrame:
    """Executa e grava `data/gold/uhc_index.parquet`."""
    idx = build_uhc_index()
    settings.ensure_dirs()
    idx.to_parquet(settings.GOLD_DIR / "uhc_index.parquet", index=False)
    tratados = int(idx.loc[idx["treated"], "country_code"].nunique())
    always = int(idx.loc[idx["always_treated"], "country_code"].nunique())
    print(f"[features] uhc_index: {len(idx)} linhas, {tratados} paises tratados "
          f"({always} always-treated)")
    return idx


if __name__ == "__main__":
    run()
