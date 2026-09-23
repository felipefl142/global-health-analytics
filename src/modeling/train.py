"""Treina M1/M2/M3 e salva artefatos + relatorios de avaliacao.

- M1 (regressao): expectativa de vida
- M2 (regressao): mortalidade infantil (<5)
- M3 (classificacao): marco de saude 'alto' (UHC>=80% E LE>=70)

Para cada modelo: baseline linear/ridge + XGBoost (early stopping no split de validacao).
Saida:
- models/<name>_xgb.joblib / models/<name>_lin.joblib
- models/<name>_eval.json (metricas + importancia de features + SHAP top)

Uso: python -m src.modeling.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.modeling.dataset import TARGETS, build_dataset


def _xgb(kind: str):
    import xgboost as xgb
    params = dict(n_estimators=2000, learning_rate=0.03, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  early_stopping_rounds=50, random_state=42, n_jobs=-1)
    if kind == "regression":
        return xgb.XGBRegressor(**params)
    return xgb.XGBClassifier(eval_metric="logloss", **params)


def _linear(kind: str):
    """Baseline linear: mediana p/ NaN + padronizacao + Ridge/Logistica."""
    head = Ridge(alpha=1.0) if kind == "regression" else LogisticRegression(max_iter=2000)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), head)


def _eval_regr(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def _eval_clf(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = round(float(roc_auc_score(y_true, y_pred_proba)), 3)
    out["pos_rate_test"] = round(float(np.mean(y_true)), 3)
    return out


def _importance(model, X: pd.DataFrame, kind: str, y) -> dict:
    """Importancia de features (SHAP se possivel, senao permutation)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return {
            "method": "shap",
            "top": {X.columns[i]: round(float(mean_abs[i]), 4) for i in order[:10]},
        }
    except Exception:  # noqa: BLE001
        try:
            pi = permutation_importance(model, X, y, n_repeats=3, random_state=42, n_jobs=-1)
            order = np.argsort(pi.importances_mean)[::-1]
            return {
                "method": "permutation",
                "top": {X.columns[i]: round(float(pi.importances_mean[i]), 4) for i in order[:10]},
            }
        except Exception as e2:  # noqa: BLE001
            return {"method": "none", "error": str(e2)[:100]}


def train_one(target: str) -> dict:
    md = build_dataset(target)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"name": md.name, "kind": md.kind, "target": md.target,
                    "n_train": len(md.X_train), "n_valid": len(md.X_valid),
                    "n_test": len(md.X_test), "features": md.features}

    # Baseline linear
    lin = _linear(md.kind)
    lin.fit(md.X_train, md.y_train)
    if md.kind == "regression":
        report["linear"] = {"valid": _eval_regr(md.y_valid, lin.predict(md.X_valid)),
                            "test": _eval_regr(md.y_test, lin.predict(md.X_test))}
    else:
        report["linear"] = {"valid": _eval_clf(md.y_valid, lin.predict_proba(md.X_valid)[:, 1]),
                            "test": _eval_clf(md.y_test, lin.predict_proba(md.X_test)[:, 1])}

    # XGBoost (principal) - NaN tratado nativamente; early stopping no valid
    model = _xgb(md.kind)
    model.fit(md.X_train, md.y_train, eval_set=[(md.X_valid, md.y_valid)], verbose=False)
    if md.kind == "regression":
        p_valid = model.predict(md.X_valid)
        p_test = model.predict(md.X_test)
        report["xgb"] = {"valid": _eval_regr(md.y_valid, p_valid),
                          "test": _eval_regr(md.y_test, p_test)}
    else:
        p_valid = model.predict_proba(md.X_valid)[:, 1]
        p_test = model.predict_proba(md.X_test)[:, 1]
        report["xgb"] = {"valid": _eval_clf(md.y_valid, p_valid),
                          "test": _eval_clf(md.y_test, p_test)}
    # valid usado no early stopping -> metrica de valid levemente otimista; test e limpo
    report["xgb"]["best_iteration"] = int(model.best_iteration)
    report["importance"] = _importance(model, md.X_test, md.kind, md.y_test)

    joblib.dump(model, settings.MODELS_DIR / f"{md.name}_xgb.joblib")
    joblib.dump(lin, settings.MODELS_DIR / f"{md.name}_lin.joblib")
    eval_path = settings.MODELS_DIR / f"{md.name}_eval.json"
    eval_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  [{md.name}] xgb.test={report['xgb']['test']}  -> {eval_path.name}")
    return report


def main() -> None:
    print("Treinando modelos (split temporal 1990-2015 / 2016-2019 / 2020-2023) ...")
    reports = [train_one(t) for t in TARGETS]
    all_path = settings.MODELS_DIR / "all_eval.json"
    all_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print(f"\nRelatorio consolidado -> {all_path}")


if __name__ == "__main__":
    main()
