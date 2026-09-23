"""CLI do feature store.

Uso:
    python -m src.serving.feast_cli export        # gold/ABT -> parquets-fonte do Feast
    python -m src.serving.feast_cli apply         # registra entidades/views/service
    python -m src.serving.feast_cli materialize   # export + apply + offline -> online
    python -m src.serving.feast_cli get BRA USA   # le features online
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime

import pandas as pd

from config import settings
from src.modeling.dataset import FEATURES
from src.serving import feast_features as ff
from src.utils.io import load_parquet


def export_sources(abt: pd.DataFrame | None = None, dim: pd.DataFrame | None = None) -> dict:
    """Escreve os parquets-fonte com event_timestamp (31/12 do ano, UTC)."""
    abt = load_parquet("gold", "abt_country_year") if abt is None else abt
    dim = load_parquet("silver", "countries_dim") if dim is None else dim
    cols = FEATURES + ff.EXTRA_YEARLY
    # schema das feature views e fixo: colunas ausentes na ABT (ingestao parcial) viram NaN
    abt = abt.reindex(columns=list(dict.fromkeys([*abt.columns, *cols])))
    yearly = abt[["country_id", "year", *cols]].dropna(subset=FEATURES, how="all").copy()
    yearly[cols] = yearly[cols].astype("float64")
    yearly["event_timestamp"] = pd.to_datetime(yearly["year"].astype(str) + "-12-31", utc=True)
    yearly["year"] = yearly["year"].astype("float64")
    yearly.to_parquet(ff.YEARLY_SOURCE, index=False)

    static = dim[["country_id", "country_name", "region", "income", "latitude", "longitude"]].copy()
    static["event_timestamp"] = pd.Timestamp("1990-01-01", tz="UTC")
    static.to_parquet(ff.STATIC_SOURCE, index=False)
    return {"yearly_rows": len(yearly), "static_rows": len(static)}


def get_store():
    from feast import FeatureStore
    return FeatureStore(config=ff.repo_config())


def apply(store=None) -> None:
    store = store or get_store()
    store.apply(ff.OBJECTS)


def materialize(store=None) -> None:
    store = store or get_store()
    store.materialize(start_date=datetime(1989, 1, 1, tzinfo=UTC), end_date=datetime.now(UTC))


def get_online(country_ids: list[str], store=None) -> pd.DataFrame:
    store = store or get_store()
    fs = store.get_feature_service("health_prediction_service")
    rows = [{"country_id": c} for c in country_ids]
    return store.get_online_features(features=fs, entity_rows=rows).to_df()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "apply", "materialize", "get"])
    ap.add_argument("countries", nargs="*")
    args = ap.parse_args()
    settings.ensure_dirs()
    print(f"online store: {settings.ONLINE_STORE_TYPE}")
    if args.cmd in ("export", "materialize"):
        print("export:", export_sources())
    if args.cmd in ("apply", "materialize"):
        apply()
        print("apply: ok")
    if args.cmd == "materialize":
        materialize()
        print("materialize: ok")
    if args.cmd == "get":
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(get_online(args.countries or ["BRA"]).T)


if __name__ == "__main__":
    main()
