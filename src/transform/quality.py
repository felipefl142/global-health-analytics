"""Checagens de qualidade da ABT (gold) e resumo de cobertura/missingness."""
from __future__ import annotations

import pandas as pd

from config import settings
from src.utils.io import write_json

# Faixas plausiveis por indicador (min, max); None = sem limite superior.
FAIXAS: dict[str, tuple[float | None, float | None]] = {
    "life_expectancy": (10, 100),  # <20 ocorre em genocidio/guerra (RWA 1994, CAF, SSD)
    "child_mortality": (0, 500),
    "gdp_per_capita": (0, None),
    "population": (0, None),
    "fertility": (0, 15),
    "urban_pct": (0, 100),
    "health_exp_gdp": (0, 100),
    "public_health_exp_gdp": (0, 100),
    "out_of_pocket": (0, 100),
    "sanitation_basic": (0, 100),
    "water_basic": (0, 100),
    "sanitation_safely": (0, 100),
    "water_safely": (0, 100),
    "measles_imm_pct": (0, 100),
    "dpt_imm_pct": (0, 100),
    "uhc_index": (0, 1),
    "uhc_sci": (0, 100),
}

INDICADORES = [c for c in FAIXAS]


def checar_faixas(abt: pd.DataFrame) -> list[dict]:
    """Conta violacoes de faixa por indicador presente na ABT."""
    checagens = []
    for col, (lo, hi) in FAIXAS.items():
        if col not in abt.columns:
            continue
        serie = abt[col].dropna()
        fora = 0
        if lo is not None:
            fora += int((serie < lo).sum())
        if hi is not None:
            fora += int((serie > hi).sum())
        checagens.append({
            "indicator": col,
            "min": lo,
            "max": hi,
            "n": int(serie.size),
            "violations": fora,
            "status": "ok" if fora == 0 else "warn",
        })
    return checagens


def resumo(abt: pd.DataFrame) -> dict:
    """Resumo de cobertura: linhas, paises, anos e missingness por coluna."""
    cols = [c for c in INDICADORES if c in abt.columns]
    missing = {c: round(float(abt[c].isna().mean()), 4) for c in cols}
    return {
        "rows": int(len(abt)),
        "countries": int(abt["country_code"].nunique()),
        "year_min": int(abt["year"].min()),
        "year_max": int(abt["year"].max()),
        "missing_rate": dict(sorted(missing.items(), key=lambda kv: kv[1])),
    }


def run(abt: pd.DataFrame) -> dict:
    """Executa as checagens e grava `gold/abt_quality.json` + `gold/quality_checks.json`."""
    checagens = checar_faixas(abt)
    resumo_abt = resumo(abt)
    payload = {
        "summary": resumo_abt,
        "checks": checagens,
        "warnings": sum(1 for c in checagens if c["status"] != "ok"),
    }
    write_json(settings.GOLD_DIR / "abt_quality.json", payload)
    write_json(settings.GOLD_DIR / "quality_checks.json", checagens)

    faltantes = [k for k, v in resumo_abt["missing_rate"].items() if v > 0.5]
    if faltantes:
        print(f"[quality] aviso: colunas com >50% missing: {faltantes}")
    print(f"[quality] {resumo_abt['rows']} linhas, {resumo_abt['countries']} paises, "
          f"{payload['warnings']} indicadores fora de faixa")
    return payload
