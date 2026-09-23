"""bronze -> silver: parse, tipagem e padronizacao das fontes World Bank e WHO GHO.

Gera em `data/silver/`:
- `worldbank_long.parquet`: country x year x indicator (World Bank, long)
- `who_long.parquet`: country x year x indicator (WHO GHO, long)
- `countries_dim.parquet`: dimensao de pais (regiao, renda, coordenadas)
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.utils.io import read_json

REGIAO_AGREGADA = "Aggregates"


def _code_to_name(cfg: dict) -> dict[str, str]:
    """Mapeia codigo do World Bank -> nome logico."""
    return {code: name for name, code in settings.all_indicator_codes(cfg).items()}


def _who_code_to_name(cfg: dict) -> dict[str, str]:
    """Mapeia codigo do WHO GHO -> nome logico."""
    return {spec["code"]: name for name, spec in settings.who_indicators(cfg).items()}


def build_countries_dim() -> pd.DataFrame:
    """Constroi a dimensao de pais a partir de `bronze/worldbank/_countries.json`."""
    rows = read_json(settings.BRONZE_DIR / "worldbank" / "_countries.json")["data"]
    registros = []
    for r in rows:
        regiao = (r.get("region") or {}).get("value", "").strip()
        renda = (r.get("incomeLevel") or {}).get("value", "").strip()
        registros.append({
            "country_code": str(r["id"]).strip().upper(),
            "country_name": (r.get("name") or "").strip(),
            "region": regiao,
            "income_level": renda,
            "lending_type": ((r.get("lendingType") or {}).get("value") or "").strip(),
            "is_aggregate": regiao == REGIAO_AGREGADA or renda == REGIAO_AGREGADA,
            "capital_city": (r.get("capitalCity") or "").strip(),
            "longitude": pd.to_numeric(r.get("longitude"), errors="coerce"),
            "latitude": pd.to_numeric(r.get("latitude"), errors="coerce"),
        })
    return pd.DataFrame(registros).drop_duplicates("country_code").reset_index(drop=True)


def build_worldbank_long(cfg: dict) -> pd.DataFrame:
    """Le os JSON brutos do World Bank e devolve a serie longa limpa."""
    nomes = _code_to_name(cfg)
    registros = []
    for code, nome in nomes.items():
        caminho = settings.BRONZE_DIR / "worldbank" / f"{code}.json"
        if not caminho.exists():
            continue
        for r in read_json(caminho)["data"]:
            valor = r.get("value")
            pais = (r.get("countryiso3code") or "").strip().upper()
            if valor is None or not pais:
                continue
            registros.append({
                "country_code": pais,
                "country_name": (r.get("country") or {}).get("value", "").strip(),
                "indicator": nome,
                "indicator_code": code,
                "year": int(r["date"]),
                "value": float(valor),
            })
    df = pd.DataFrame(registros)
    if df.empty:
        return pd.DataFrame(columns=["country_code", "country_name", "indicator",
                                     "indicator_code", "year", "value"])
    df = (df.groupby(["country_code", "country_name", "indicator", "indicator_code", "year"],
                     as_index=False)["value"].mean())
    return df.sort_values(["country_code", "indicator", "year"]).reset_index(drop=True)


def build_who_long(cfg: dict) -> pd.DataFrame:
    """Le os JSON brutos do WHO GHO e devolve a serie longa limpa (so paises)."""
    nomes = _who_code_to_name(cfg)
    registros = []
    for code, nome in nomes.items():
        caminho = settings.BRONZE_DIR / "who" / f"{code}.json"
        if not caminho.exists():
            continue
        for r in read_json(caminho)["data"]:
            if r.get("SpatialDimType") and r["SpatialDimType"] != "COUNTRY":
                continue
            valor = r.get("NumericValue")
            ano = r.get("TimeDim")
            pais = (r.get("SpatialDim") or "").strip().upper()
            if valor is None or ano is None or not pais:
                continue
            registros.append({
                "country_code": pais,
                "indicator": nome,
                "indicator_code": code,
                "year": int(ano),
                "value": float(valor),
            })
    df = pd.DataFrame(registros)
    if df.empty:
        return pd.DataFrame(columns=["country_code", "indicator", "indicator_code",
                                     "year", "value"])
    df = (df.groupby(["country_code", "indicator", "indicator_code", "year"], as_index=False)
            ["value"].mean())
    return df.sort_values(["country_code", "indicator", "year"]).reset_index(drop=True)


def run() -> dict[str, pd.DataFrame]:
    """Executa a transformacao completa e grava os parquet do silver."""
    cfg = settings.load_indicators()
    settings.ensure_dirs()

    wb_long = build_worldbank_long(cfg)
    who_long = build_who_long(cfg)
    countries = build_countries_dim()

    wb_long.to_parquet(settings.SILVER_DIR / "worldbank_long.parquet", index=False)
    who_long.to_parquet(settings.SILVER_DIR / "who_long.parquet", index=False)
    countries.to_parquet(settings.SILVER_DIR / "countries_dim.parquet", index=False)

    print(f"[silver] worldbank_long={len(wb_long)} linhas, who_long={len(who_long)} linhas, "
          f"countries={len(countries)}")
    return {"worldbank_long": wb_long, "who_long": who_long, "countries_dim": countries}


if __name__ == "__main__":
    run()
