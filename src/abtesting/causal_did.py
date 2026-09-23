"""Causal (estrela): expansao UHC -> expectativa de vida.

- **Event study** com nao-tratados como controle (FE pais+ano, SE cluster), usando
  dummies de tempo relativo ao `treat_year` (referencia k=-1).
- **Controle sintetico** (Abadie) para 1-2 paises "heroi": pesos nao-negativos que
  somam 1 sobre o doador pool, minimizando o RMSE pre-tratamento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.optimize import minimize

from config import settings
from src.utils.io import write_json

OUTCOME = "life_expectancy"
JANELA = (-5, 10)


def event_study(abt: pd.DataFrame, outcome: str = OUTCOME, janela: tuple = JANELA) -> dict:
    """Coeficientes de event study (efeito dinamico do tratamento)."""
    d = abt[abt["treated"] | abt["never_treated"]].dropna(subset=[outcome, "uhc_index"]).copy()
    d["event_time"] = np.where(d["treated"], d["year"] - d["treat_year"].astype("float"), np.nan)
    ks = [k for k in range(janela[0], janela[1] + 1) if k != -1]
    for k in ks:
        d[f"ev_{k}"] = (d["event_time"] == k).astype(int)
    cols = [f"ev_{k}" for k in ks]

    s = d.set_index(["country_code", "year"])
    res = PanelOLS(s[outcome], s[cols], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    ci = res.conf_int()
    tabela = []
    for k in sorted([*ks, -1]):
        if k == -1:
            tabela.append({"k": k, "coef": 0.0, "ic95": [0.0, 0.0]})
            continue
        c = f"ev_{k}"
        tabela.append({"k": k, "coef": float(res.params[c]),
                       "ic95": [float(ci.loc[c].iloc[0]), float(ci.loc[c].iloc[1])]})
    return {"outcome": outcome, "event_study": tabela, "n": int(res.nobs)}


def _herois(abt: pd.DataFrame, n: int = 2) -> list[str]:
    """Paises tratados (nao always) com pre-tratamento e pos-tratamento suficientes."""
    d = abt.dropna(subset=[OUTCOME, "treat_year"]).copy()
    d = d[d["treated"] & ~d["always_treated"]]
    d["treat_year_int"] = d["treat_year"].astype(int)
    g = d.groupby("country_code").agg(tr=("treat_year_int", "first"))
    g["pre"] = (g["tr"] - 1990).clip(lower=0)
    g["post"] = (2023 - g["tr"]).clip(lower=0)
    elegiveis = g[(g["pre"] >= 12) & (g["post"] >= 8)].sort_values("post", ascending=False)
    return list(elegiveis.index[:n])


def synthetic_control(abt: pd.DataFrame, hero: str, outcome: str = OUTCOME,
                      janela: int = 10) -> dict | None:
    """Controle sintetico de Abadie para um pais tratado (janela +/- `janela` anos)."""
    linha = abt.loc[abt["country_code"] == hero, "treat_year"].dropna()
    if linha.empty:
        return None
    treat_year = int(linha.iloc[0])

    piv = abt.dropna(subset=[outcome]).pivot_table(
        index="year", columns="country_code", values=outcome, aggfunc="mean")
    if hero not in piv.columns:
        return None
    never = abt.loc[abt["never_treated"].fillna(False).astype(bool), "country_code"].unique()
    doadores = [c for c in never if c in piv.columns]

    anos = [a for a in piv.index if treat_year - janela <= a <= treat_year + janela]
    pre = [a for a in anos if a < treat_year]
    post = [a for a in anos if a >= treat_year]
    validos = [c for c in doadores
               if piv.loc[pre, c].notna().all() and piv.loc[post, c].notna().all()]
    if len(pre) < 5 or len(post) < 3 or len(validos) < 5:
        return None

    y = piv[hero]
    X = piv[validos]
    y_pre, X_pre = y.loc[pre].to_numpy(), X.loc[pre].to_numpy()

    def perda(w):
        return float(np.sum((y_pre - X_pre @ w) ** 2))

    chute = np.ones(len(validos)) / len(validos)
    res = minimize(perda, chute, method="SLSQP", bounds=[(0, 1)] * len(validos),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    pesos = res.x
    sintetico = X @ pesos
    gap = y.loc[anos] - sintetico.loc[anos]
    return {
        "hero": hero,
        "treat_year": treat_year,
        "n_doadores": len(validos),
        "pre_rmse": float(np.sqrt(np.mean((y_pre - X_pre @ pesos) ** 2))),
        "att_pos": float(gap.loc[post].mean()),
        "serie": {
            "year": [int(a) for a in anos],
            "observado": [float(v) for v in y.loc[anos]],
            "sintetico": [float(v) for v in sintetico.loc[anos]],
        },
    }


def run() -> dict:
    """Executa event study + controle sintetico e grava `reports/causal_did.json`."""
    abt = pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")
    estudo = event_study(abt)
    herois = _herois(abt)
    sinteticos = [s for h in herois if (s := synthetic_control(abt, h)) is not None]
    saida = {"event_study": estudo, "synthetic_control": sinteticos,
             "herois": herois,
             "nota": "Event study com TWFE e imune a heterogeneidade; ver Goodman-Bacon (2021)."}
    write_json(settings.REPORTS_DIR / "causal_did.json", saida)
    if sinteticos:
        print(f"[causal] herois={herois} ATT_pos="
              f"{[round(s['att_pos'], 2) for s in sinteticos]}")
    return saida


if __name__ == "__main__":
    run()
