"""Treino dos 3 modelos (M1/M2/M3): baseline linear + XGBoost.

Split temporal fixo (treino <= 2015, validacao 2016-2019, teste >= 2020). O XGBoost
usa `tree_method="hist"`, NaN nativo e early stopping na validacao. Salva os
artefatos em `models/` e relatorios `models/<nome>_eval.json` (metricas + importancias).
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from config import settings
from src.modeling import dataset
from src.modeling.evaluate import metricas_classificacao, metricas_regressao
from src.utils.io import write_json

SPECS = [
    {"name": "life_expectancy_reg", "target": "life_expectancy", "task": "regression"},
    {"name": "child_mortality_reg", "target": "child_mortality", "task": "regression"},
    {"name": "milestone_high_clf", "target": "milestone_high", "task": "classification"},
]
RANDOM_STATE = 42


def _baseline(task: str):
    """Baseline linear: Ridge (regressao) ou regressao logistica (classificacao)."""
    if task == "regression":
        modelo = Ridge(alpha=1.0)
    else:
        modelo = LogisticRegression(max_iter=1000)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", modelo),
    ])


def _xgb(task: str):
    """XGBoost com NaN nativo e early stopping."""
    comum = dict(
        n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", random_state=RANDOM_STATE,
        early_stopping_rounds=50, n_jobs=4,
    )
    if task == "regression":
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **comum)
    return XGBClassifier(objective="binary:logistic", eval_metric="auc", **comum)


def _pred(task: str, modelo, X):
    """Predicao: valor (regressao) ou probabilidade da classe 1 (classificacao)."""
    if task == "regression":
        return modelo.predict(X)
    return modelo.predict_proba(X)[:, 1]


def _metricas(task: str, y, pred) -> dict:
    if task == "regression":
        return metricas_regressao(y, pred)
    return metricas_classificacao(y, pred)


def _importancias(modelo, X, k: int = 10) -> list[dict]:
    """Top-k importancias via SHAP (fallback para gain nativo do XGBoost)."""
    amostra = X.sample(min(300, len(X)), random_state=RANDOM_STATE)
    try:
        import shap

        valores = shap.TreeExplainer(modelo).shap_values(amostra)
        if isinstance(valores, list):
            valores = valores[1]
        media = np.abs(np.asarray(valores)).mean(axis=0)
    except Exception:  # noqa: BLE001 - fallback robusto
        media = np.asarray(modelo.feature_importances_)
    ordem = np.argsort(media)[::-1][:k]
    return [{"feature": amostra.columns[i], "importancia": round(float(media[i]), 5)}
            for i in ordem]


def train_model(spec: dict) -> dict:
    """Treina baseline + XGBoost para um alvo e grava artefatos/relatorio."""
    ds = dataset.build_dataset(spec["target"], spec["task"])
    X_tr, y_tr = ds["train"]
    X_va, y_va = ds["valid"]
    X_te, y_te = ds["test"]
    task = spec["task"]

    relatorio = {
        "model": spec["name"], "target": spec["target"], "task": task,
        "features": ds["features"], "n_features": len(ds["features"]), "meta": ds["meta"],
        "baseline_linear": {}, "xgboost": {},
    }

    linear = _baseline(task)
    linear.fit(X_tr, y_tr)
    relatorio["baseline_linear"]["valid"] = _metricas(task, y_va, _pred(task, linear, X_va))
    relatorio["baseline_linear"]["test"] = _metricas(task, y_te, _pred(task, linear, X_te))
    joblib.dump(linear, settings.MODELS_DIR / f"{spec['name']}_lin.joblib")

    xgb = _xgb(task)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    relatorio["xgboost"]["valid"] = _metricas(task, y_va, _pred(task, xgb, X_va))
    relatorio["xgboost"]["test"] = _metricas(task, y_te, _pred(task, xgb, X_te))
    relatorio["xgboost"]["best_iteration"] = int(getattr(xgb, "best_iteration", 0) or 0)
    relatorio["xgboost"]["importancias"] = _importancias(xgb, X_te)
    joblib.dump(xgb, settings.MODELS_DIR / f"{spec['name']}_xgb.joblib")

    write_json(settings.MODELS_DIR / f"{spec['name']}_eval.json", relatorio)
    return relatorio


def run() -> list[dict]:
    """Treina os 3 modelos e grava `models/all_eval.json`."""
    settings.ensure_dirs()
    relatorios = [train_model(spec) for spec in SPECS]
    resumo = [{
        "model": r["model"], "target": r["target"], "task": r["task"],
        "valid": r["xgboost"]["valid"], "test": r["xgboost"]["test"],
        "test_linear": r["baseline_linear"]["test"],
    } for r in relatorios]
    write_json(settings.MODELS_DIR / "all_eval.json", resumo)

    for r in relatorios:
        met = r["xgboost"]["test"]
        if r["task"] == "regression":
            print(f"[train] {r['model']}: teste RMSE={met['rmse']:.2f} MAE={met['mae']:.2f} "
                  f"R2={met['r2']:.3f}")
        else:
            print(f"[train] {r['model']}: teste AUC={met['auc']:.3f} F1={met['f1']:.3f} "
                  f"Brier={met['brier']:.3f}")
    return relatorios


if __name__ == "__main__":
    run()
