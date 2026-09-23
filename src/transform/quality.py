"""Check de qualidade de dados (silver -> gold).

Validacoes basicas de integridade e plausibilidade sobre a ABT:
- completude por pais (cobertura de anos)
- valores impossiveis em alvos (ex.: expectativa de vida fora de faixa)
- consistencia de unicao (1 linha por pais x ano)

Uso: python -m src.transform.quality
"""
from __future__ import annotations

import json

import pandas as pd

from config import settings
from src.utils.io import load_parquet


def check_abt(abt: pd.DataFrame) -> dict:
    issues: list[str] = []

    # unicidade pais x ano
    dup = abt.duplicated(subset=["country_id", "year"]).sum()
    if dup:
        issues.append(f"{dup} linhas duplicadas (pais x ano)")

    # faixa plausivel de alvos
    le = abt["life_expectancy"]
    if (le.dropna() < 20).any() or (le.dropna() > 95).any():
        issues.append("expectativa de vida fora da faixa 20-95 anos")
    cm = abt["child_mortality"]
    if (cm.dropna() < 0).any() or (cm.dropna() > 500).any():
        issues.append("mortalidade infantil (<5) fora da faixa 0-500")

    # proxy UHC em [0,1]
    u = abt["uhc_index"].dropna()
    if len(u) and (u.min() < -1e-9 or u.max() > 1 + 1e-9):
        issues.append("proxy UHC fora de [0,1]")

    # cobertura: paises com <50% dos anos
    n_years = abt["year"].nunique()
    cov = abt.groupby("country_id")["year"].nunique()
    sparse = cov[cov < 0.5 * n_years]
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "n_countries": int(abt["country_id"].nunique()),
        "n_years": int(n_years),
        "countries_below_50pct_coverage": int(len(sparse)),
        "duplicated_country_years": int(dup),
    }


def main() -> None:
    abt = load_parquet("gold", "abt_country_year")
    rep = check_abt(abt)
    out = settings.GOLD_DIR / "quality_checks.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"Quality checks -> {out}")
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    main()
