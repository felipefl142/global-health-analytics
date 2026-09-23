"""A/B simulado: randomiza paises, aplica uplift no proxy UHC e avalia o efeito.

Modelos *dose-resposta* (desfecho ~ uhc_index + controles) traduzem o uplift de UHC
em desfecho. Roda o A/B completo: diferenca de medias, IC95, p, Cohen's d, CUPED,
poder estatistico e correcao de Bonferroni (2 desfechos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import nct
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.utils.io import write_json

RNG_SEED = 42
DELTA_UHC = 0.30
FRACAO_TRATAMENTO = 0.5
ANO_INICIO = 2020
FEATURES_DOSE = ["uhc_index", "log_gdp", "fertility", "population"]


def _com_log_gdp(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_gdp"] = np.log(d["gdp_per_capita"].where(d["gdp_per_capita"] > 0))
    return d


def _modelo_dose(abt: pd.DataFrame, desfecho: str):
    """Ajusta desfecho ~ uhc_index + controles e retorna (modelo, sigma residual)."""
    d = _com_log_gdp(abt.dropna(subset=[desfecho]))
    d = d.dropna(subset=FEATURES_DOSE)
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    modelo.fit(d[FEATURES_DOSE], d[desfecho])
    sigma = float(np.std(d[desfecho] - modelo.predict(d[FEATURES_DOSE])))
    return modelo, sigma


def _cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajuste CUPED (controle de variancia por covariavel pre-experimento)."""
    x = np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_ajustado = y - theta * (x - x.mean())
    reducao = 1 - np.var(y_ajustado, ddof=1) / np.var(y, ddof=1)
    return y_ajustado, float(max(reducao, 0.0))


def _estimar(a: pd.Series, b: pd.Series) -> dict:
    """Diferenca de medias com IC95, p-valor e Cohen's d."""
    diff = float(a.mean() - b.mean())
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    gl = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    gl = max(gl, 1.0)
    tc = stats.t.ppf(0.975, gl)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    d_cohen = diff / sp if sp > 0 else 0.0
    return {"efeito": diff, "ic95": [diff - tc * se, diff + tc * se], "p": float(p),
            "se": float(se), "cohen_d": float(d_cohen),
            "n_trat": int(len(a)), "n_ctrl": int(len(b))}


def _poder(efeito: float, se: float, n_trat: int, n_ctrl: int, alpha: float = 0.05) -> float:
    """Poder de um teste t bicaudal a partir do efeito e do erro-padrao."""
    if se <= 0:
        return 0.0
    gl = n_trat + n_ctrl - 2
    tc = stats.t.ppf(1 - alpha / 2, gl)
    nc = abs(efeito) / se
    return float(1 - nct.cdf(tc, gl, nc) + nct.cdf(-tc, gl, nc))


def simular(delta: float = DELTA_UHC, fracao: float = FRACAO_TRATAMENTO,
            seed: int = RNG_SEED) -> dict:
    """Executa a simulacao A/B (unidade = pais) para os dois desfechos."""
    abt = pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")
    rng = np.random.default_rng(seed)
    pos = abt[abt["year"] >= ANO_INICIO]

    metricas: dict[str, dict] = {}
    p_valores: list[tuple[str, float]] = []
    for desfecho in ("life_expectancy", "child_mortality"):
        modelo, _ = _modelo_dose(abt, desfecho)
        # Unidade experimental = pais: media do periodo pos por pais.
        cols = ["gdp_per_capita", "fertility", "population", "uhc_index", desfecho]
        d = pos.groupby("country_code")[cols].mean().reset_index()
        d = _com_log_gdp(d).dropna(subset=FEATURES_DOSE).copy()
        base = modelo.predict(d[FEATURES_DOSE])
        sigma = float(np.std(d[desfecho] - base))  # ruido no nivel do pais
        d_uplift = d.copy()
        d_uplift["uhc_index"] = (d_uplift["uhc_index"] + delta).clip(upper=1.0)
        uplift = modelo.predict(d_uplift[FEATURES_DOSE]) - base

        paises = sorted(d["country_code"].unique())
        n_trat = int(round(len(paises) * fracao))
        tratados = set(rng.choice(paises, size=n_trat, replace=False))
        d["tratado"] = d["country_code"].isin(tratados)

        ruido = rng.normal(0, sigma, len(d))
        d["y"] = np.where(d["tratado"], base + uplift + ruido, base + ruido)

        a = d.loc[d["tratado"], "y"]
        b = d.loc[~d["tratado"], "y"]
        est = _estimar(a, b)

        # Covariavel CUPED = basal predito pelo modelo (counterfactual sem tratamento).
        y_cuped, reducao = _cuped(d["y"].to_numpy(), base)
        cuped = _estimar(pd.Series(y_cuped[d["tratado"].to_numpy()]),
                         pd.Series(y_cuped[~d["tratado"].to_numpy()]))

        poder = _poder(est["efeito"], est["se"], est["n_trat"], est["n_ctrl"])
        cuped_poder = _poder(cuped["efeito"], cuped["se"], cuped["n_trat"], cuped["n_ctrl"])
        metricas[desfecho] = {
            **est,
            "efeito_real_uplift": float(np.mean(uplift[d["tratado"].to_numpy()])),
            "poder": float(poder),
            "cuped_efeito": cuped["efeito"],
            "cuped_ic95": cuped["ic95"],
            "cuped_p": cuped["p"],
            "cuped_poder": float(cuped_poder),
            "cuped_reducao_variancia": reducao,
        }
        p_valores.append((desfecho, est["p"]))

    n_testes = len(p_valores)
    for metrica in metricas.values():
        metrica["p_bonferroni"] = float(min(1.0, metrica["p"] * n_testes))

    return {
        "config": {"delta_uhc": delta, "fracao_tratamento": fracao, "seed": seed,
                   "periodo": f">={ANO_INICIO}", "n_testes": n_testes,
                   "unidade": "pais (media do periodo)"},
        "metricas": metricas,
        "nota": "Desfecho simulado via modelo dose-resposta; efeito verdadeiro = uplift medio.",
    }


def run() -> dict:
    """Executa o A/B simulado e grava `reports/ab_simulation.json`."""
    resultado = simular()
    write_json(settings.REPORTS_DIR / "ab_simulation.json", resultado)
    print(f"[ab] delta={resultado['config']['delta_uhc']} -> "
          f"LE {resultado['metricas']['life_expectancy']['efeito']:+.3f} anos "
          f"(p={resultado['metricas']['life_expectancy']['p']:.3g})")
    return resultado


if __name__ == "__main__":
    run()
