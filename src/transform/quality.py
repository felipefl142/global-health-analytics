"""Check de qualidade de dados (silver -> gold).

Validacoes basicas de integridade e plausibilidade sobre a ABT:
- completude por pais (cobertura de anos)
- faixas fisicamente possiveis p/ cada indicador (RANGES): violacao = issue
- expectativa de vida de crise (< 25) = aviso listado, nao erro
- consistencia de unicao (1 linha por pais x ano)

Uso: python -m src.transform.quality   (sai com codigo 1 se houver issue)
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from config import settings
from src.utils.io import load_parquet

# Limites fisicos (min, max) - None = sem limite. Nao sao limites "tipicos": LE de periodo
# despenca em crises reais (Ruanda 1994: ~12 anos), por isso o piso e 5.
RANGES: dict[str, tuple[float | None, float | None]] = {
    "life_expectancy": (5, 95),
    "child_mortality": (0, 500),
    "maternal_mortality": (0, None),    # crises reais passam de 5000/100 mil (RWA 1994)
    "ncd_mortality_30_70": (0, 100),
    "tb_incidence": (0, None),
    "gdp_per_capita": (0, None),
    "health_exp_per_capita": (0, None),
    "population": (0, None),
    "fertility": (0, 15),
    "doctors_per_1000": (0, 20),
    "nurses_per_1000": (0, 40),
    "beds_per_1000": (0, 30),
    "uhc_index": (0, 1),
    "uhc_sci": (0, 100),
    **{c: (0, 100) for c in ("urban_pct", "health_exp_gdp", "public_health_exp_gdp",
                             "out_of_pocket", "sanitation_basic", "water_basic",
                             "sanitation_safely", "water_safely", "measles_imm_pct",
                             "dpt_imm_pct")},
}
EPS = 1e-9


def range_checks(abt: pd.DataFrame) -> list[dict]:
    """Violacoes de faixa por indicador presente na ABT."""
    out = []
    for col, (lo, hi) in RANGES.items():
        if col not in abt.columns:
            continue
        s = abt[col].dropna()
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= s < lo - EPS
        if hi is not None:
            bad |= s > hi + EPS
        out.append({"indicator": col, "min": lo, "max": hi, "n": int(len(s)),
                    "violations": int(bad.sum()),
                    "examples": abt.loc[bad[bad].index[:3], ["country_id", "year", col]]
                    .to_dict(orient="records")})
    return out


def check_abt(abt: pd.DataFrame) -> dict:
    issues: list[str] = []
    warnings: list[str] = []

    # unicidade pais x ano
    dup = abt.duplicated(subset=["country_id", "year"]).sum()
    if dup:
        issues.append(f"{dup} linhas duplicadas (pais x ano)")

    # faixas fisicas por indicador
    ranges = range_checks(abt)
    for r in ranges:
        if r["violations"]:
            issues.append(f"{r['indicator']}: {r['violations']} valores fora de "
                          f"[{r['min']}, {r['max']}] (ex.: {r['examples']})")

    # LE de crise: real, mas vale listar p/ inspecao
    if "life_expectancy" in abt.columns:
        extreme = abt.loc[abt["life_expectancy"] < 25, ["country_id", "year", "life_expectancy"]]
        for r in extreme.itertuples(index=False):
            warnings.append(f"LE extrema (crise?): {r.country_id} {r.year} = {r.life_expectancy:.1f}")

    # cobertura: paises com <50% dos anos
    n_years = abt["year"].nunique()
    cov = abt.groupby("country_id")["year"].nunique()
    sparse = cov[cov < 0.5 * n_years]
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "n_countries": int(abt["country_id"].nunique()),
        "n_years": int(n_years),
        "countries_below_50pct_coverage": int(len(sparse)),
        "duplicated_country_years": int(dup),
        "range_checks": ranges,
    }


def main() -> int:
    abt = load_parquet("gold", "abt_country_year")
    rep = check_abt(abt)
    out = settings.GOLD_DIR / "quality_checks.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"Quality checks -> {out}")
    print(json.dumps({k: v for k, v in rep.items() if k != "range_checks"}, indent=2))
    print(f"faixas: {sum(r['violations'] for r in rep['range_checks'])} violacoes em "
          f"{len(rep['range_checks'])} indicadores")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
