"""Analise exploratoria da ABT: cobertura, missingness, correlacoes e outliers.

Os resultados alimentam o notebook `01_eda.ipynb` e o dashboard. Tambem gera
`reports/eda_summary.json` para consulta rapida (usado no README/dashboard).
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.utils.io import write_json

INDICADORES = [
    "life_expectancy", "child_mortality", "uhc_index", "uhc_sci",
    "health_exp_per_capita", "health_exp_gdp", "out_of_pocket", "gdp_per_capita",
    "doctors_per_1000", "nurses_per_1000", "beds_per_1000",
    "sanitation_basic", "sanitation_safely", "water_basic", "water_safely",
    "measles_imm_pct", "dpt_imm_pct", "urban_pct", "fertility", "population",
]
ORDEM_RENDA = ["Low income", "Lower middle income", "Upper middle income", "High income"]


def carregar_abt() -> pd.DataFrame:
    """Carrega a ABT (gold)."""
    return pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")


def resumo_cobertura(abt: pd.DataFrame) -> dict:
    """Cobertura geral e missingness por indicador."""
    cols = [c for c in INDICADORES if c in abt.columns]
    cobertura = {
        c: {
            "n": int(abt[c].notna().sum()),
            "missing_rate": round(float(abt[c].isna().mean()), 4),
        }
        for c in cols
    }
    return {
        "rows": int(len(abt)),
        "countries": int(abt["country_code"].nunique()),
        "year_min": int(abt["year"].min()),
        "year_max": int(abt["year"].max()),
        "n_regions": int(abt["region"].nunique()),
        "coverage": cobertura,
    }


def missing_por_renda(abt: pd.DataFrame) -> pd.DataFrame:
    """Taxa de missing por indicador x grupo de renda (missingness nao aleatoria)."""
    cols = [c for c in INDICADORES if c in abt.columns]
    grupos = [g for g in ORDEM_RENDA if g in set(abt["income_level"])]
    dados = {}
    for g in grupos:
        sub = abt[abt["income_level"] == g]
        dados[g] = {c: sub[c].isna().mean() for c in cols}
    return pd.DataFrame(dados).T


def correlacoes(abt: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlacao (Pearson) dos indicadores-chave."""
    cols = [c for c in INDICADORES if c in abt.columns]
    return abt[cols].corr(method="pearson")


def outliers_iqr(abt: pd.DataFrame) -> list[dict]:
    """Conta outliers por regra IQR (1.5x) para cada indicador-chave."""
    resultado = []
    for c in [x for x in INDICADORES if x in abt.columns]:
        s = abt[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        resultado.append({
            "indicator": c, "q1": round(float(q1), 3), "q3": round(float(q3), 3),
            "lower": round(float(lo), 3), "upper": round(float(hi), 3),
            "outliers": n, "outlier_rate": round(n / len(s), 4),
        })
    return resultado


def validar_uhc(abt: pd.DataFrame) -> dict:
    """Correlacao entre o proxy `uhc_index` e o SCI oficial (2000-2023)."""
    m = abt.dropna(subset=["uhc_index", "uhc_sci"])
    return {
        "n": int(len(m)),
        "pearson": round(float(m["uhc_index"].corr(m["uhc_sci"])), 4),
        "spearman": round(float(m["uhc_index"].corr(m["uhc_sci"], method="spearman")), 4),
    }


def tendencias_regionais(abt: pd.DataFrame, indicador: str) -> pd.DataFrame:
    """Media do indicador por regiao x ano (paises, sem agregados na ABT)."""
    return (abt.groupby(["region", "year"])[indicador].mean().reset_index())


def gerar_resumo() -> dict:
    """Calcula o resumo de EDA e grava `reports/eda_summary.json`."""
    abt = carregar_abt()
    resumo = {
        "cobertura": resumo_cobertura(abt),
        "missing_por_renda": missing_por_renda(abt).round(4).to_dict(),
        "correlacoes_alvo": {
            "life_expectancy": correlacoes(abt)["life_expectancy"].round(3).to_dict(),
            "child_mortality": correlacoes(abt)["child_mortality"].round(3).to_dict(),
        },
        "outliers": outliers_iqr(abt),
        "validacao_uhc": validar_uhc(abt),
    }
    write_json(settings.REPORTS_DIR / "eda_summary.json", resumo)
    return resumo


if __name__ == "__main__":
    r = gerar_resumo()
    c = r["cobertura"]
    print(f"[eda] {c['rows']} linhas, {c['countries']} paises, {c['year_min']}-{c['year_max']}; "
          f"uhc vs SCI {r['validacao_uhc']}")
