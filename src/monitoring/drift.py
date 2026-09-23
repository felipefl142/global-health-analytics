"""Monitoramento de drift (PSI/KS) das features e da performance dos modelos.

Compara a janela de referencia (treino <= 2015) com a janela recente (>= 2020),
sinaliza features com PSI/KS acima dos limiares e verifica a queda de performance
no teste. Gera `reports/drift_report.json`, exposto no dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import settings
from src.modeling import dataset
from src.utils.io import read_json, write_json

ANO_REF_FIM = 2015
ANO_ATUAL_INI = 2020
MODELOS = ("life_expectancy_reg", "child_mortality_reg", "milestone_high_clf")


def _psi(referencia: pd.Series, atual: pd.Series, bins: int = 10) -> float | None:
    """Population Stability Index entre duas amostras."""
    ref, cur = referencia.dropna(), atual.dropna()
    if len(ref) < 10 or len(cur) < 10:
        return None
    quebras = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quebras) < 3:
        return None
    ref_pct = np.histogram(ref, bins=quebras)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=quebras)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def analisar_features(abt: pd.DataFrame) -> list[dict]:
    """PSI e KS por feature (referencia vs janela recente)."""
    referencia = abt[abt["year"] <= ANO_REF_FIM]
    atual = abt[abt["year"] >= ANO_ATUAL_INI]
    linhas = []
    for nome in dataset.features_disponiveis(abt):
        if nome not in abt.columns:
            continue
        ref_f, cur_f = referencia[nome], atual[nome]
        if ref_f.notna().sum() < 10 or cur_f.notna().sum() < 10:
            continue
        psi = _psi(ref_f, cur_f)
        ks = ks_2samp(ref_f.dropna(), cur_f.dropna())
        alerta = ((psi is not None and psi >= settings.DRIFT_PSI_THRESHOLD)
                  or ks.statistic >= settings.DRIFT_KS_THRESHOLD)
        linhas.append({
            "feature": nome,
            "psi": round(psi, 4) if psi is not None else None,
            "ks": round(float(ks.statistic), 4),
            "ks_p": float(ks.pvalue),
            "drift": bool(alerta),
        })
    return sorted(linhas, key=lambda x: (x["psi"] or 0), reverse=True)


def analisar_performance() -> list[dict]:
    """Compara a metrica de validacao (2016-2019) com a de teste (>=2020)."""
    linhas = []
    for nome in MODELOS:
        caminho = settings.MODELS_DIR / f"{nome}_eval.json"
        if not caminho.exists():
            continue
        ev = read_json(caminho)
        if ev["task"] == "regression":
            valid = ev["xgboost"]["valid"]["rmse"]
            teste = ev["xgboost"]["test"]["rmse"]
            razao = teste / valid if valid else None
            linhas.append({"model": nome, "metrica": "rmse", "valid": valid, "teste": teste,
                           "razao": round(razao, 3) if razao else None,
                           "alerta": bool(razao and razao > 1.5)})
        else:
            valid = ev["xgboost"]["valid"]["auc"]
            teste = ev["xgboost"]["test"]["auc"]
            linhas.append({"model": nome, "metrica": "auc", "valid": valid, "teste": teste,
                           "queda": round(valid - teste, 3),
                           "alerta": bool((valid - teste) > 0.1)})
    return linhas


def run() -> dict:
    """Executa o monitoramento e grava `reports/drift_report.json`."""
    abt = pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")
    features = analisar_features(abt)
    performance = analisar_performance()
    com_drift = [f["feature"] for f in features if f["drift"]]
    alertas = com_drift + [p["model"] for p in performance if p["alerta"]]
    relatorio = {
        "referencia": f"<= {ANO_REF_FIM}",
        "atual": f">= {ANO_ATUAL_INI}",
        "limiares": {"psi": settings.DRIFT_PSI_THRESHOLD, "ks": settings.DRIFT_KS_THRESHOLD},
        "features": features,
        "performance": performance,
        "alertas": alertas,
    }
    write_json(settings.REPORTS_DIR / "drift_report.json", relatorio)
    print(f"[drift] {len(com_drift)} features com drift; "
          f"{len([p for p in performance if p['alerta']])} modelos em alerta")
    return relatorio


if __name__ == "__main__":
    run()
