"""Metricas de avaliacao para os modelos de regressao e classificacao."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def metricas_regressao(y_true, y_pred) -> dict:
    """RMSE, MAE e R2."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def metricas_classificacao(y_true, y_prob, threshold: float = 0.5) -> dict:
    """AUC, F1, acuracia, precisao, recall, Brier e log-loss."""
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    y_true = np.asarray(y_true)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "prevalencia": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }
