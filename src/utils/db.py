"""Catalogo DuckDB com views sobre os parquets das camadas silver e gold."""
from __future__ import annotations

from pathlib import Path

import duckdb

from config import settings

VIEWS: dict[str, Path] = {
    "silver.worldbank_long": settings.SILVER_DIR / "worldbank_long.parquet",
    "silver.who_long": settings.SILVER_DIR / "who_long.parquet",
    "silver.countries_dim": settings.SILVER_DIR / "countries_dim.parquet",
    "gold.abt_country_year": settings.GOLD_DIR / "abt_country_year.parquet",
    "gold.uhc_index": settings.GOLD_DIR / "uhc_index.parquet",
}


def connect(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Abre `data/catalog.duckdb` e (re)cria as views disponiveis."""
    settings.ensure_dirs()
    con = duckdb.connect(str(db_path or (settings.DATA_DIR / "catalog.duckdb")))
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    for nome, caminho in VIEWS.items():
        if Path(caminho).exists():
            uri = Path(caminho).as_posix().replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {nome} AS SELECT * FROM read_parquet('{uri}')")
    return con


if __name__ == "__main__":
    con = connect()
    tabelas = con.execute(
        "SELECT table_schema, table_name FROM information_schema.views "
        "WHERE table_schema IN ('silver', 'gold') ORDER BY 1, 2"
    ).fetchall()
    for schema, nome in tabelas:
        print(f"{schema}.{nome}")
