"""Camada de predicao usada pela API: modelos + busca de features + log de predicoes.

Busca de features (FeatureProvider):
- sem `year`: online store do Feast (valores mais recentes por pais); se o Feast nao
  estiver disponivel (sem Redis/registry), cai p/ a ABT gold (ultimo ano com dado).
- com `year`: ABT gold via DuckDB (pais x ano especifico).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import settings
from src.modeling.dataset import FEATURES

log = logging.getLogger(__name__)

MODEL_NAMES = {"life_expectancy": "life_expectancy_reg", "child_mortality": "child_mortality_reg",
               "milestone_high": "milestone_high_clf"}
PREDICTION_LOG = settings.DATA_DIR / "monitoring" / "predictions.jsonl"
# alvos nao-negativos: o intervalo conformal (absoluto, homocedastico) e truncado em 0.
# Limitacao conhecida: p/ mortalidade o erro cresce com o nivel, entao o intervalo e largo
# demais em paises de baixa mortalidade e estreito nos de alta.
NON_NEGATIVE = {"child_mortality"}


@dataclass
class LoadedModel:
    target: str
    name: str
    kind: str
    model: object
    features: list[str]
    interval_q90: float | None
    interval_coverage_test: float | None
    test_metrics: dict


def load_models(models_dir: Path = settings.MODELS_DIR) -> dict[str, LoadedModel]:
    out = {}
    for target, name in MODEL_NAMES.items():
        path = models_dir / f"{name}_xgb.joblib"
        if not path.exists():
            log.warning("modelo ausente: %s", path)
            continue
        rep = json.loads((models_dir / f"{name}_eval.json").read_text())
        model = joblib.load(path)
        out[target] = LoadedModel(
            target=target, name=name, kind=rep["kind"], model=model,
            features=list(model.feature_names_in_),
            interval_q90=rep["xgb"].get("interval_q90"),
            interval_coverage_test=rep["xgb"].get("interval_coverage_test"),
            test_metrics=rep["xgb"]["test"])
    return out


def _clean(v):
    """NaN/None -> None; numpy -> python (JSON-safe)."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


class FeatureProvider:
    """Busca features por pais (Feast online -> fallback ABT)."""

    def __init__(self, abt: pd.DataFrame | None = None, use_feast: bool = True):
        self._abt = abt
        self._store = None
        self.source_latest = "abt"
        if use_feast:
            try:
                from src.serving.feast_cli import get_store
                store = get_store()
                store.get_feature_service("health_prediction_service")  # registry ok?
                self._store = store
                self.source_latest = f"feast:{settings.ONLINE_STORE_TYPE}"
            except Exception as e:  # noqa: BLE001
                log.warning("Feast indisponivel (%s); usando ABT gold", e)

    @property
    def abt(self) -> pd.DataFrame:
        if self._abt is None:
            from src.utils.io import load_parquet
            self._abt = load_parquet("gold", "abt_country_year")
        return self._abt

    def countries(self) -> pd.DataFrame:
        cols = ["country_id", "country_name", "region", "income"]
        return self.abt[cols].drop_duplicates("country_id").sort_values("country_name")

    def get(self, country_id: str, year: int | None = None) -> tuple[dict, int | None, str]:
        """-> (features, ano das features, fonte). KeyError se pais/ano inexistente."""
        if year is None and self._store is not None:
            try:
                from src.serving.feast_cli import get_online
                row = get_online([country_id], self._store).iloc[0].to_dict()
                if any(_clean(row.get(f)) is not None for f in FEATURES):
                    yr = _clean(row.get("year"))
                    return ({f: _clean(row.get(f)) for f in FEATURES},
                            int(yr) if yr is not None else None, self.source_latest)
            except Exception as e:  # noqa: BLE001
                log.warning("falha no Feast p/ %s (%s); fallback ABT", country_id, e)
        df = self.abt[self.abt["country_id"] == country_id]
        if df.empty:
            raise KeyError(f"pais desconhecido: {country_id}")
        df = df.dropna(subset=[f for f in FEATURES if f in df.columns], how="all")
        if year is not None:
            df = df[df["year"] == year]
            if df.empty:
                raise KeyError(f"sem features p/ {country_id} em {year}")
        row = df.sort_values("year").iloc[-1]
        return {f: _clean(row.get(f)) for f in FEATURES}, int(row["year"]), "abt"


def predict_all(models: dict[str, LoadedModel], features: dict) -> dict:
    """Roda M1/M2/M3 sobre um dict de features (None -> NaN, que o XGBoost aceita)."""
    out = {}
    for target, m in models.items():
        X = pd.DataFrame([{f: features.get(f) for f in m.features}]).astype(float)
        if m.kind == "regression":
            y = float(m.model.predict(X)[0])
            q = m.interval_q90
            lo = max(y - q, 0.0) if q and target in NON_NEGATIVE else (y - q if q else None)
            out[target] = {"prediction": round(y, 3),
                           "interval_90": [round(lo, 3), round(y + q, 3)] if q else None,
                           "interval_empirical_coverage_test": m.interval_coverage_test}
        else:
            p = float(m.model.predict_proba(X)[0, 1])
            out[target] = {"probability": round(p, 4), "label": int(p >= 0.5)}
    return out


def log_prediction(record: dict, path: Path = PREDICTION_LOG) -> None:
    """Append-only JSONL (insumo do monitoramento de drift)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(UTC).isoformat(), **record}
    with path.open("a") as f:
        f.write(json.dumps(rec, default=_clean) + "\n")
