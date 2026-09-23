"""Helpers de DuckDB sobre as camadas de dados (bronze/silver/gold)."""
from __future__ import annotations

from pathlib import Path

import duckdb

from config import settings


def connect(data_dir: Path | str = settings.DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Conexao DuckDB com os diretorios de camadas registrados como schemas."""
    data_dir = Path(data_dir)
    con = duckdb.connect(str(data_dir / "catalog.duckdb"))
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    # Views que apontam para os parquets (se existirem)
    _attach_parquet(con, "silver", data_dir / "silver")
    _attach_parquet(con, "gold", data_dir / "gold")
    return con


def _attach_parquet(con: duckdb.DuckDBPyConnection, schema: str, folder: Path) -> None:
    if not folder.exists():
        return
    for pq in sorted(folder.glob("*.parquet")):
        view = f"{schema}.{pq.stem}"
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{pq.as_posix()}')"
        )


def query(sql: str, data_dir: Path | str = settings.DATA_DIR) -> object:
    con = connect(data_dir)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def list_views(data_dir: Path | str = settings.DATA_DIR) -> list[str]:
    con = connect(data_dir)
    try:
        rows = con.execute(
            "SELECT table_schema || '.' || table_name AS v FROM information_schema.tables "
            "WHERE table_type='VIEW' ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()
