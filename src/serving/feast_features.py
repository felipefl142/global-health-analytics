"""Definicoes do feature store (Feast): entidade, fontes, feature views e feature service.

- Offline store: DuckDB lendo parquets da camada gold (exportados por feast_cli).
- Online store: Redis (padrao; docker compose) ou SQLite (ONLINE_STORE_TYPE=sqlite,
  p/ rodar local/CI sem Redis).
- country_health_yearly: 1 linha por pais x ano (timestamp = 31/12 do ano); a online
  store guarda o valor mais recente por pais.
- country_health_static: dimensao do pais (regiao, renda, lat/lon).
- health_prediction_service: exatamente as features que os modelos M1/M2/M3 consomem.
"""
from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.repo_config import RegistryConfig, RepoConfig
from feast.types import Float64, String

from config import settings
from src.modeling.dataset import FEATURES

YEARLY_SOURCE = settings.GOLD_DIR / "feast_country_yearly.parquet"
STATIC_SOURCE = settings.GOLD_DIR / "feast_country_static.parquet"
# alvos e SCI oficial ficam disponiveis p/ exibicao/monitoramento (nao sao features de modelo)
EXTRA_YEARLY = ["life_expectancy", "child_mortality", "uhc_sci", "population"]

country = Entity(name="country", join_keys=["country_id"], value_type=ValueType.STRING,
                 description="Pais (ISO3)")

yearly_source = FileSource(name="abt_country_yearly", path=str(YEARLY_SOURCE),
                           timestamp_field="event_timestamp")
static_source = FileSource(name="countries_static", path=str(STATIC_SOURCE),
                           timestamp_field="event_timestamp")

country_health_yearly = FeatureView(
    name="country_health_yearly",
    entities=[country],
    # indicadores de saude sao anuais e com atraso de publicacao: aceita ate 5 anos
    ttl=timedelta(days=365 * 5),
    schema=[Field(name=c, dtype=Float64) for c in FEATURES + EXTRA_YEARLY]
    + [Field(name="year", dtype=Float64)],
    source=yearly_source,
    online=True,
)

country_health_static = FeatureView(
    name="country_health_static",
    entities=[country],
    ttl=timedelta(days=365 * 100),
    schema=[Field(name="country_name", dtype=String), Field(name="region", dtype=String),
            Field(name="income", dtype=String), Field(name="latitude", dtype=Float64),
            Field(name="longitude", dtype=Float64)],
    source=static_source,
    online=True,
)

health_prediction_service = FeatureService(
    name="health_prediction_service",
    features=[country_health_yearly[FEATURES + ["year"]], country_health_static],
)

OBJECTS = [country, country_health_yearly, country_health_static, health_prediction_service]


def repo_config() -> RepoConfig:
    """RepoConfig em codigo (evita yaml duplicado; online store vem do .env)."""
    root = settings.ROOT / ".feast"
    root.mkdir(parents=True, exist_ok=True)
    if settings.ONLINE_STORE_TYPE == "redis":
        online = {"type": "redis",
                  "connection_string": f"{settings.REDIS_HOST}:{settings.REDIS_PORT},db={settings.REDIS_DB}"}
    else:
        online = {"type": "sqlite", "path": str(root / "online_store.db")}
    return RepoConfig(
        project=settings.FEAST_PROJECT,
        provider="local",
        registry=RegistryConfig(path=settings.FEAST_REGISTRY, cache_ttl_seconds=60),
        offline_store={"type": "duckdb"},
        online_store=online,
        entity_key_serialization_version=3,
        repo_path=root,
    )
