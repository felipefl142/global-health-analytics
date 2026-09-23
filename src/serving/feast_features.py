"""Definicao do feature repo do Feast.

- Fonte offline: parquets gerados a partir da ABT (`feast_country_*.parquet`).
- Feature views: `country_health_static` (regiao/renda) e `country_health_yearly`
  (indicadores por ano), entidade = `country`, timestamp = inicio do ano.
- Feature service: `health_prediction_service`.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

from config import settings

COLUNAS_ANUAIS = [
    "uhc_index", "life_expectancy", "child_mortality", "health_exp_per_capita",
    "doctors_per_1000", "nurses_per_1000", "sanitation_basic", "water_basic",
    "measles_imm_pct", "dpt_imm_pct", "gdp_per_capita", "fertility", "urban_pct",
    "population",
]
COLUNAS_ESTATICAS = ["region", "income_level"]
TTL = timedelta(days=365 * 60)


def construir_tabelas() -> dict[str, str]:
    """Gera os parquets consumidos pelo Feast (timestamps + created)."""
    settings.ensure_dirs()
    abt = pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")
    criado = pd.Timestamp.utcnow().tz_localize(None)

    anual = abt[["country_code", "year", *COLUNAS_ANUAIS]].copy()
    anual["event_timestamp"] = pd.to_datetime(anual["year"].astype(str) + "-01-01")
    anual["created"] = criado
    caminho_anual = settings.GOLD_DIR / "feast_country_yearly.parquet"
    anual.drop(columns=["year"]).to_parquet(caminho_anual, index=False)

    estatico = (abt.sort_values("year")
                   .groupby("country_code")[COLUNAS_ESTATICAS].first()
                   .reset_index())
    estatico["event_timestamp"] = pd.Timestamp("2024-01-01")
    estatico["created"] = criado
    caminho_estatico = settings.GOLD_DIR / "feast_country_static.parquet"
    estatico.to_parquet(caminho_estatico, index=False)

    return {"yearly": str(caminho_anual), "static": str(caminho_estatico)}


def objetos() -> tuple[Entity, list[FeatureView], FeatureService]:
    """Monta entidade, feature views e feature service para o `store.apply`."""
    caminhos = construir_tabelas()
    pais = Entity(name="country", join_keys=["country_code"], value_type=ValueType.STRING)

    fonte_anual = FileSource(name="abt_anual", path=caminhos["yearly"],
                             timestamp_field="event_timestamp",
                             created_timestamp_column="created")
    fonte_estatica = FileSource(name="abt_estatica", path=caminhos["static"],
                                timestamp_field="event_timestamp",
                                created_timestamp_column="created")

    view_anual = FeatureView(
        name="country_health_yearly", entities=[pais], ttl=TTL, source=fonte_anual,
        schema=[Field(name=c, dtype=Float32) for c in COLUNAS_ANUAIS],
    )
    view_estatica = FeatureView(
        name="country_health_static", entities=[pais], ttl=TTL, source=fonte_estatica,
        schema=[Field(name=c, dtype=String) for c in COLUNAS_ESTATICAS],
    )
    servico = FeatureService(name="health_prediction_service",
                             features=[view_anual, view_estatica])
    return pais, [view_anual, view_estatica], servico
