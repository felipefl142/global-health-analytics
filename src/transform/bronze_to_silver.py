"""Camada silver: parse dos JSONs raw (bronze) -> tabelas long estandardizadas.

Saidas:
- data/silver/worldbank_long.parquet
    country_id, country_name, iso_alpha3, region, income, year, indicator_code, value
- data/silver/countries_dim.parquet
    country_id, country_name, iso_alpha3, region, income, latitude, longitude

Uso: python -m src.transform.bronze_to_silver
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import settings


def _parse_records(code: str, raw: dict) -> pd.DataFrame:
    """Extrai registros de um payload [metadata, records] -> DataFrame long."""
    records = raw[1] if isinstance(raw, list) and len(raw) > 1 else []
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        if "value" not in r:  # payload de erro
            continue
        country = r.get("country") or {}
        region = r.get("region") or {}
        income = r.get("income") or {}
        date = r.get("date")
        val = r.get("value")
        try:
            year = int(date) if date is not None else None
        except (TypeError, ValueError):
            year = None
        try:
            value = float(val) if val is not None else None
        except (TypeError, ValueError):
            value = None
        rows.append({
            "country_id": r.get("id"),
            "country_name": country.get("value"),
            "iso_alpha3": r.get("iso_alpha3"),
            "region": region.get("value"),
            "income": income.get("value"),
            "year": year,
            "indicator_code": code,
            "value": value,
        })
    return pd.DataFrame(rows)


def load_bronze_long() -> pd.DataFrame:
    wb_dir = settings.BRONZE_DIR / "worldbank"
    frames = []
    for path in sorted(wb_dir.glob("*.json")):
        code = path.stem
        try:
            raw = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            print(f"  [aviso] nao consegui ler {path.name}, pulando")
            continue
        df = _parse_records(code, raw)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"nenhum dado bronze em {wb_dir}; rode 'make ingest'")
    long_df = pd.concat(frames, ignore_index=True)
    return long_df


def build_countries_dim(long_df: pd.DataFrame) -> pd.DataFrame:
    dim = (
        long_df.dropna(subset=["country_id"])
        .groupby("country_id", as_index=False)
        .agg(
            country_name=("country_name", "first"),
            iso_alpha3=("iso_alpha3", "first"),
            region=("region", "first"),
            income=("income", "first"),
        )
    )
    return dim.reset_index(drop=True)


def main() -> None:
    settings.ensure_dirs()
    print("Carregando bronze (worldbank) -> silver ...")
    long_df = load_bronze_long()
    long_df["year"] = long_df["year"].astype("Int64")
    long_df = long_df.dropna(subset=["year", "country_id", "indicator_code"])

    n_ind = long_df["indicator_code"].nunique()
    n_cty = long_df["country_id"].nunique()
    print(f"  rows={len(long_df):,} indicadores={n_ind} paises={n_cty} "
          f"anos={long_df['year'].min()}..{long_df['year'].max()}")

    from src.utils.io import save_parquet
    out1 = save_parquet(long_df, "silver", "worldbank_long")
    dim = build_countries_dim(long_df)
    out2 = save_parquet(dim, "silver", "countries_dim")
    print(f"  -> {out1}")
    print(f"  -> {out2} ({len(dim)} paises)")

    # resumo por indicador
    summary = (
        long_df.groupby("indicator_code")
        .agg(rows=("value", "size"), nonnull=("value", "count"),
             paises=("country_id", "nunique"))
        .reset_index()
    )
    print("\nResumo por indicador:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
