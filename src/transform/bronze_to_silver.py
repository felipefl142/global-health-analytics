"""Camada silver: parse dos JSONs raw (bronze) -> tabelas long padronizadas.

- Chave de pais = ISO3 (World Bank `countryiso3code` / WHO `SpatialDim`).
- Agregados do World Bank (regioes, grupos de renda, "World") sao removidos: so paises.
- WHO GHO: so linhas de pais, ambos os sexos (ou sem dimensao de sexo).

Saidas:
- data/silver/indicators_long.parquet
    country_id (ISO3), year, indicator (nome logico), indicator_code, source, value
- data/silver/countries_dim.parquet
    country_id, country_name, iso2, region, income, latitude, longitude

Uso: python -m src.transform.bronze_to_silver
"""
from __future__ import annotations

import json

import pandas as pd

from config import settings
from src.ingestion.who_gho import who_indicators
from src.ingestion.worldbank import COUNTRIES_FILE
from src.utils.io import save_parquet

AGGREGATE_REGION = "Aggregates"


def parse_countries(raw: list) -> pd.DataFrame:
    """Metadados de pais do WB -> dimensao (so paises, sem agregados)."""
    rows = []
    for r in raw[1]:
        region = (r.get("region") or {}).get("value", "").strip()
        if region == AGGREGATE_REGION:
            continue
        rows.append({
            "country_id": r["id"],
            "country_name": r.get("name"),
            "iso2": r.get("iso2Code"),
            "region": region or None,
            "income": ((r.get("incomeLevel") or {}).get("value") or "").strip() or None,
            "latitude": pd.to_numeric(r.get("latitude") or None, errors="coerce"),
            "longitude": pd.to_numeric(r.get("longitude") or None, errors="coerce"),
        })
    return pd.DataFrame(rows)


def parse_worldbank(code: str, name: str, raw: list) -> pd.DataFrame:
    """Payload WB [metadata, records] -> long (inclui agregados; filtrados depois)."""
    rows = [
        {"country_id": r.get("countryiso3code") or None, "year": r.get("date"),
         "value": r.get("value")}
        for r in (raw[1] if isinstance(raw, list) and len(raw) > 1 else [])
        if "value" in r
    ]
    df = pd.DataFrame(rows, columns=["country_id", "year", "value"])
    df["indicator"] = name
    df["indicator_code"] = code
    df["source"] = "worldbank"
    return df


def parse_who(code: str, name: str, raw: dict) -> pd.DataFrame:
    """Payload OData WHO -> long (paises, ambos os sexos)."""
    rows = [
        {"country_id": r.get("SpatialDim"), "year": r.get("TimeDim"),
         "value": r.get("NumericValue")}
        for r in raw.get("value", [])
        if r.get("SpatialDimType") == "COUNTRY" and r.get("Dim1") in (None, "SEX_BTSX")
    ]
    df = pd.DataFrame(rows, columns=["country_id", "year", "value"])
    df["indicator"] = name
    df["indicator_code"] = code
    df["source"] = "who_gho"
    return df


def load_bronze_long(cfg: dict) -> pd.DataFrame:
    frames = []
    wb_dir = settings.BRONZE_DIR / "worldbank"
    for name, code in settings.all_indicator_codes(cfg).items():
        path = wb_dir / f"{code}.json"
        if path.exists():
            frames.append(parse_worldbank(code, name, json.loads(path.read_text())))
        else:
            print(f"  [aviso] bronze ausente: {path.name} (rode 'make ingest')")
    who_dir = settings.BRONZE_DIR / "who"
    for name, code in who_indicators(cfg).items():
        path = who_dir / f"{code}.json"
        if path.exists():
            frames.append(parse_who(code, name, json.loads(path.read_text())))
    if not frames:
        raise FileNotFoundError(f"nenhum dado bronze em {settings.BRONZE_DIR}; rode 'make ingest'")
    return pd.concat(frames, ignore_index=True)


def to_silver(long_df: pd.DataFrame, dim: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Tipagem, filtro de paises validos/periodo, remocao de nulos e duplicatas."""
    df = long_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["country_id"].isin(dim["country_id"])]
    df = df[df["year"].between(start, end)]
    df = df.dropna(subset=["year", "value"])
    # WHO pode ter >1 linha por pais x ano (ex.: revisoes) -> media
    df = (df.groupby(["country_id", "year", "indicator", "indicator_code", "source"],
                     as_index=False)["value"].mean())
    return df.sort_values(["indicator", "country_id", "year"]).reset_index(drop=True)


def main() -> None:
    settings.ensure_dirs()
    cfg = settings.load_indicators()
    start, end = settings.date_range(cfg)
    print("Carregando bronze -> silver ...")
    countries_path = settings.BRONZE_DIR / "worldbank" / COUNTRIES_FILE
    if not countries_path.exists():
        raise FileNotFoundError(f"{countries_path} ausente; rode 'make ingest'")
    dim = parse_countries(json.loads(countries_path.read_text()))
    long_df = to_silver(load_bronze_long(cfg), dim, start, end)

    out1 = save_parquet(long_df, "silver", "indicators_long")
    out2 = save_parquet(dim, "silver", "countries_dim")
    print(f"  rows={len(long_df):,} indicadores={long_df['indicator'].nunique()} "
          f"paises={long_df['country_id'].nunique()} anos={long_df['year'].min()}..{long_df['year'].max()}")
    print(f"  -> {out1}\n  -> {out2} ({len(dim)} paises)")

    summary = (long_df.groupby(["source", "indicator"])
               .agg(rows=("value", "size"), paises=("country_id", "nunique"),
                    ano_min=("year", "min"), ano_max=("year", "max"))
               .reset_index())
    print("\nResumo por indicador:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
