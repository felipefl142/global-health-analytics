"""Funcoes de EDA reutilizadas pelos notebooks (F3/F4) e pelo dashboard (F10).

Todas recebem a ABT (gold) e devolvem DataFrames prontos p/ tabela ou grafico.
`python -m src.analysis.eda` grava reports/eda_summary.json (resumo versionado:
cobertura, missing por renda, correlacoes com os alvos, choques, COVID, outliers).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features.uhc_index import DEFAULT_COMPONENTS

ID_COLS = ["country_id", "country_name", "region", "income", "year"]
FLAG_COLS = ["treated", "always_treated", "never_treated", "post", "treat_year",
             "uhc_weight_avail", "milestone_high"]
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]


def indicator_cols(abt: pd.DataFrame) -> list[str]:
    return [c for c in abt.columns if c not in ID_COLS + FLAG_COLS
            and pd.api.types.is_numeric_dtype(abt[c])]


def coverage_by_indicator(abt: pd.DataFrame) -> pd.DataFrame:
    """Cobertura por indicador: % de celulas pais x ano preenchidas, paises, anos."""
    rows = []
    for c in indicator_cols(abt):
        s = abt[["country_id", "year", c]].dropna()
        rows.append({"indicator": c, "pct_filled": round(len(s) / len(abt) * 100, 1),
                     "countries": s["country_id"].nunique(),
                     "first_year": int(s["year"].min()) if len(s) else None,
                     "last_year": int(s["year"].max()) if len(s) else None})
    return pd.DataFrame(rows).sort_values("pct_filled", ascending=False).reset_index(drop=True)


def missingness_by_year(abt: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Matriz indicador x ano com % de paises SEM dado (heatmap de missingness)."""
    cols = cols or indicator_cols(abt)
    return (abt.groupby("year")[cols].apply(lambda g: g.isna().mean() * 100).T.round(1))


def missingness_by_income(abt: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """% de missing por grupo de renda - o dado falta de forma aleatoria?"""
    cols = cols or indicator_cols(abt)
    out = abt.groupby("income")[cols].apply(lambda g: g.isna().mean() * 100).T.round(1)
    return out[[c for c in INCOME_ORDER if c in out.columns]]


def describe(abt: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or indicator_cols(abt)
    d = abt[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    d["skew"] = abt[cols].skew()
    return d.round(3)


def correlations(abt: pd.DataFrame, cols: list[str] | None = None, year: int | None = None,
                 method: str = "spearman") -> pd.DataFrame:
    """Correlacao (Spearman por padrao - relacoes monotonas e robusto a caudas)."""
    df = abt if year is None else abt[abt["year"] == year]
    cols = cols or indicator_cols(abt)
    return df[cols].corr(method=method).round(3)


def trends(abt: pd.DataFrame, col: str, by: str = "income", weight: str | None = "population") -> pd.DataFrame:
    """Serie temporal por grupo (media ponderada por populacao por padrao)."""
    df = abt.dropna(subset=[col, by])
    if weight and weight in df.columns:
        df = df.dropna(subset=[weight])
        g = df.groupby([by, "year"]).apply(
            lambda x: np.average(x[col], weights=x[weight]), include_groups=False)
    else:
        g = df.groupby([by, "year"])[col].mean()
    return g.rename(col).reset_index()


def regional_comparison(abt: pd.DataFrame, year: int, cols: list[str]) -> pd.DataFrame:
    """Medianas por regiao num ano."""
    return (abt[abt["year"] == year].groupby("region")[cols].median()
            .round(2).sort_values(cols[0], ascending=False))


def outliers_iqr(abt: pd.DataFrame, col: str, k: float = 3.0) -> pd.DataFrame:
    """Outliers extremos (fora de Q1 - k*IQR, Q3 + k*IQR) - p/ inspecao, nao remocao."""
    s = abt[col]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (s < q1 - k * iqr) | (s > q3 + k * iqr)
    return abt.loc[mask, ["country_id", "country_name", "year", col]].sort_values(col)


def yoy_shocks(abt: pd.DataFrame, col: str = "life_expectancy", threshold: float = -3.0) -> pd.DataFrame:
    """Quedas ano-a-ano abruptas (crises: guerras, epidemias, COVID)."""
    df = abt.sort_values(["country_id", "year"]).copy()
    df["delta"] = df.groupby("country_id")[col].diff()
    return (df.loc[df["delta"] <= threshold, ["country_name", "region", "year", col, "delta"]]
            .sort_values("delta").reset_index(drop=True))


def covid_impact(abt: pd.DataFrame, col: str = "life_expectancy") -> pd.DataFrame:
    """Variacao 2019 -> 2021 (pior ano da pandemia) e recuperacao ate 2023, por regiao."""
    w = abt.pivot_table(index=["country_id", "region"], columns="year", values=col)
    need = [y for y in (2019, 2021, 2023) if y in w.columns]
    if len(need) < 3:
        return pd.DataFrame()
    w = w[need].dropna()
    w["drop_2019_2021"] = w[2021] - w[2019]
    w["recovery_2021_2023"] = w[2023] - w[2021]
    return (w.groupby("region")[["drop_2019_2021", "recovery_2021_2023"]].median().round(2)
            .sort_values("drop_2019_2021"))


def uhc_components() -> list[str]:
    return list(DEFAULT_COMPONENTS)


def latest_snapshot(abt: pd.DataFrame, col: str) -> pd.DataFrame:
    """Valor mais recente de `col` por pais (p/ mapas)."""
    df = abt.dropna(subset=[col]).sort_values("year")
    return df.groupby("country_id").tail(1)[["country_id", "country_name", "region", "income",
                                              "year", col]]


KEY_COLS = ["life_expectancy", "child_mortality", "uhc_index", "uhc_sci", "gdp_per_capita",
            "health_exp_per_capita", "doctors_per_1000", "nurses_per_1000", "beds_per_1000",
            "sanitation_basic", "water_basic", "measles_imm_pct", "fertility", "urban_pct"]


def summary(abt: pd.DataFrame, corr_year: int = 2019) -> dict:
    """Resumo de EDA serializavel (reports/eda_summary.json)."""
    from src.features.uhc_index import validate_against_sci

    cols = [c for c in KEY_COLS if c in abt.columns]
    corr = correlations(abt, cols, year=corr_year)
    targets = [t for t in ("life_expectancy", "child_mortality") if t in corr.columns]
    extreme = {c: int(len(outliers_iqr(abt, c))) for c in cols}
    return {
        "shape": {"rows": int(len(abt)), "countries": int(abt["country_id"].nunique()),
                  "years": [int(abt["year"].min()), int(abt["year"].max())]},
        "coverage": coverage_by_indicator(abt).to_dict(orient="records"),
        "missing_by_income_pct": missingness_by_income(abt, cols).to_dict(orient="index"),
        f"spearman_with_targets_{corr_year}": corr[targets].drop(index=targets).round(3)
        .to_dict(orient="index"),
        "uhc_proxy_vs_sci": validate_against_sci(abt),
        "extreme_outliers_iqr3": {k: v for k, v in extreme.items() if v},
        "le_shocks_top10": yoy_shocks(abt).head(10).round(2).to_dict(orient="records"),
        "covid_le_by_region": covid_impact(abt).to_dict(orient="index"),
    }


def main() -> None:
    from config import settings
    from src.utils.io import load_parquet

    settings.ensure_dirs()
    rep = summary(load_parquet("gold", "abt_country_year"))
    out = settings.REPORTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    print(f"EDA: {rep['shape']} proxy vs SCI {rep['uhc_proxy_vs_sci']} -> {out}")


if __name__ == "__main__":
    main()
