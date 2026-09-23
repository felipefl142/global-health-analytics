"""Helpers de IO (Parquet) com convencao de camadas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings


def _layer_dir(layer: str) -> Path:
    mapping = {
        "bronze": settings.BRONZE_DIR,
        "silver": settings.SILVER_DIR,
        "gold": settings.GOLD_DIR,
    }
    if layer not in mapping:
        raise ValueError(f"camada desconhecida: {layer}")
    d = mapping[layer]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_parquet(df: pd.DataFrame, layer: str, name: str, sub: str | None = None) -> Path:
    """Salva um DataFrame como Parquet em data/<layer>/[<sub>/]<name>.parquet."""
    d = _layer_dir(layer)
    if sub:
        d = d / sub
        d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(layer: str, name: str, sub: str | None = None) -> pd.DataFrame:
    d = _layer_dir(layer)
    if sub:
        d = d / sub
    return pd.read_parquet(d / f"{name}.parquet")


def list_files(layer: str, sub: str | None = None, suffix: str = ".parquet") -> list[Path]:
    d = _layer_dir(layer)
    if sub:
        d = d / sub
    return sorted(d.glob(f"*{suffix}"))
