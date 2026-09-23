"""Inferencia dos 3 modelos XGBoost para a API.

Carrega os artefatos de `models/` e calcula intervalos conformais (quantil 90% dos
residuos absolutos na validacao) para os modelos de regressao.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import settings
from src.modeling import dataset
from src.utils.io import read_json

MODELOS = {
    "life_expectancy": "life_expectancy_reg",
    "child_mortality": "child_mortality_reg",
    "milestone_high": "milestone_high_clf",
}
REGRESSAO = ("life_expectancy", "child_mortality")


def features_do_feast(country_code: str) -> dict:
    """Busca as features dos modelos na online store do Feast (ultimo valor)."""
    from feast import FeatureStore

    from src.serving.feast_features import COLUNAS_ANUAIS

    st = FeatureStore(repo_path=str(settings.CONFIG_DIR / "feast"))
    resposta = st.get_online_features(
        features=[f"country_health_yearly:{c}" for c in COLUNAS_ANUAIS],
        entity_rows=[{"country_code": country_code}],
    ).to_dict()
    return {k: (v[0] if v and v[0] is not None else None)
            for k, v in resposta.items() if k != "country_code"}


class Preditor:
    """Carrega os 3 modelos e expoe `prever(dict)`."""

    def __init__(self, conformal: bool = True):
        self.modelos = {}
        self.features = {}
        self.largura: dict[str, float] = {}
        for alvo, nome in MODELOS.items():
            info = read_json(settings.MODELS_DIR / f"{nome}_eval.json")
            self.features[alvo] = info["features"]
            self.modelos[alvo] = joblib.load(settings.MODELS_DIR / f"{nome}_xgb.joblib")
        if conformal:
            self._calcular_conformal()

    def _calcular_conformal(self) -> None:
        """Largura do intervalo de 90% via residuos absolutos na validacao."""
        for alvo in REGRESSAO:
            ds = dataset.build_dataset(alvo, "regression")
            X_va, y_va = ds["valid"]
            pred = self.modelos[alvo].predict(X_va)
            self.largura[alvo] = float(np.quantile(np.abs(y_va.to_numpy() - pred), 0.9))

    def prever(self, valores: dict) -> dict:
        """Recebe {feature: valor} e devolve as predicoes dos 3 alvos."""
        def _num(v):
            try:
                return float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        resultado = {}
        for alvo, modelo in self.modelos.items():
            feats = self.features[alvo]
            X = pd.DataFrame([{f: _num(valores.get(f)) for f in feats}], columns=feats)
            X = X.astype(float)
            if alvo in REGRESSAO:
                pred = float(modelo.predict(X)[0])
                largura = self.largura.get(alvo)
                resultado[alvo] = {
                    "valor": round(pred, 3),
                    "ic90": [round(pred - largura, 3), round(pred + largura, 3)]
                    if largura else None,
                }
            else:
                prob = float(modelo.predict_proba(X)[:, 1][0])
                resultado[alvo] = {"probabilidade": round(prob, 4), "classe": int(prob >= 0.5)}
        return resultado
