"""Configuracao central do projeto.

Carrega variaveis de ambiente (.env) e expoe caminhos e parametros
usados por ingestao, transform, modelagem e serving.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# .env antes de ler qualquer variavel (variaveis ja exportadas no shell tem precedencia)
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))

INDICATORS_FILE = CONFIG_DIR / "indicators.yaml"

# Feast
FEAST_REGISTRY = os.getenv("FEAST_REGISTRY_PATH", str(ROOT / ".feast" / "registry"))
FEAST_PROJECT = os.getenv("FEAST_PROJECT", "global_health")
ONLINE_STORE_TYPE = os.getenv("ONLINE_STORE_TYPE", "redis")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# World Bank API
WB_API_BASE = os.getenv("WB_API_BASE", "https://api.worldbank.org/v2")
WB_API_DELAY = float(os.getenv("WB_API_DELAY", "0.6"))
WB_RETRIES = int(os.getenv("WB_RETRIES", "4"))

# Drift
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25"))
DRIFT_KS_THRESHOLD = float(os.getenv("DRIFT_KS_THRESHOLD", "0.30"))



def ensure_dirs() -> None:
    """Cria a arvore de diretorios de dados."""
    for d in (DATA_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR,
              BRONZE_DIR / "worldbank", BRONZE_DIR / "who"):
        d.mkdir(parents=True, exist_ok=True)


def load_indicators() -> dict:
    """Carrega o catalogo de indicadores (config/indicators.yaml)."""
    with open(INDICATORS_FILE) as f:
        return yaml.safe_load(f)


def all_indicator_codes(cfg: dict | None = None) -> dict[str, str]:
    """Retorna {nome_logico: codigo_wb} para todos os grupos."""
    cfg = cfg or load_indicators()
    out: dict[str, str] = {}
    for group in ("targets", "uhc_inputs", "uhc_official", "covariates"):
        for name, spec in cfg.get(group, {}).items():
            out[name] = spec["code"]
    return out


def date_range(cfg: dict | None = None) -> tuple[int, int]:
    cfg = cfg or load_indicators()
    dr = cfg["date_range"]
    return int(dr["start"]), int(dr["end"])
