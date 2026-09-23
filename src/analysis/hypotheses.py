"""Testes de hipotese (F5): H1-H6 com H0/H1, teste, efeito, IC95, p e veredito.

Cada hipotese tem direcao esperada e um limiar de significancia pratica. Os
resultados consolidam em `reports/hypotheses.json`, consumido pelo notebook 03 e
pelo dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

from config import settings
from src.utils.io import write_json

LE = "life_expectancy"
CM = "child_mortality"


def carregar_abt() -> pd.DataFrame:
    """Carrega a ABT (gold)."""
    return pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")


def _z(s: pd.Series) -> pd.Series:
    """Padroniza uma serie (media 0, desvio 1) para efeitos por +1 DP."""
    return (s - s.mean()) / s.std(ddof=0)


def _panel_fe(abt: pd.DataFrame, y: str, x: list[str]):
    """PanelOLS com efeitos fixos de pais e ano e SE clusterizado por pais."""
    d = abt.dropna(subset=[y] + x).copy()
    d = d.set_index([d["country_code"], d["year"]])
    mod = PanelOLS(d[y], d[x], entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def _resultado(hid: str, hipotese: str, teste: str, efeito: float, unidade: str,
               lo: float, hi: float, p: float, n: int, esperado: str,
               limiar_pratico: float, extra: dict | None = None) -> dict:
    """Monta o registro padronizado de uma hipotese e decide o veredito."""
    direcao_ok = (efeito > 0) if esperado == "+" else (efeito < 0)
    significativo = p < 0.05
    pratico = abs(efeito) >= limiar_pratico
    efeito_alt = (extra or {}).get("efeito_entity")
    p_alt = (extra or {}).get("p_entity")
    sensivel = (
        efeito_alt is not None and p_alt is not None
        and np.sign(efeito_alt) != np.sign(efeito)
        and (p_alt < 0.05 or significativo)
    )
    if sensivel:
        veredito = "inconclusiva (sensivel a especificacao)"
    elif significativo and direcao_ok and pratico:
        veredito = "suportada"
    elif significativo and direcao_ok:
        veredito = "suportada (efeito pequeno)"
    elif significativo:
        veredito = "rejeitada (sinal oposto)"
    else:
        veredito = "nao suportada"
    registro = {
        "id": hid,
        "hipotese": hipotese,
        "H0": f"efeito <= 0 (esperado: {esperado})",
        "H1": f"efeito {'> ' if esperado == '+' else '< '}0",
        "teste": teste,
        "efeito": round(float(efeito), 4),
        "unidade": unidade,
        "ic95": [round(float(lo), 4), round(float(hi), 4)],
        "p": float(p),
        "n": int(n),
        "significativo": bool(significativo),
        "significancia_pratica": bool(pratico),
        "veredito": veredito,
    }
    if extra:
        registro.update(extra)
    return registro


def _regressao(abt: pd.DataFrame, y: str, x: str, controle: list[str] | None = None) -> dict:
    """PanelOLS com x padronizado; retorna o spec principal (FE pais+ano) e o alt (FE pais)."""
    d = abt.copy()
    d[x] = _z(d[x])
    cols = [x] + (controle or [])

    def _fit(time_effects: bool):
        s = d.dropna(subset=[y] + cols).set_index(["country_code", "year"])
        mod = PanelOLS(s[y], s[cols], entity_effects=True, time_effects=time_effects)
        return mod.fit(cov_type="clustered", cluster_entity=True)

    main, alt = _fit(True), _fit(False)
    ci = main.conf_int().loc[x]
    return {
        "efeito": float(main.params[x]),
        "lo": float(ci.iloc[0]),
        "hi": float(ci.iloc[1]),
        "p": float(main.pvalues[x]),
        "n": int(main.nobs),
        "efeito_entity": float(alt.params[x]),
        "p_entity": float(alt.pvalues[x]),
    }


def h1(abt: pd.DataFrame) -> dict:
    """H1: densidade de medicos -> expectativa de vida (liquido de PIB/capita)."""
    d = abt.copy()
    d["log_gdp"] = np.log(d["gdp_per_capita"].where(d["gdp_per_capita"] > 0))
    d["log_gdp"] = _z(d["log_gdp"])
    r = _regressao(d, LE, "doctors_per_1000", ["log_gdp"])
    return _resultado(
        "H1", "Mais medicos/1k associados a maior expectativa de vida (liquido de PIB/capita)",
        "PanelOLS FE pais+ano, SE cluster, controle log(PIB/cap)",
        r["efeito"], "anos de LE por +1 DP de medicos/1k", r["lo"], r["hi"], r["p"], r["n"],
        "+", 0.5,
        extra={"efeito_entity": r["efeito_entity"], "p_entity": r["p_entity"]},
    )


def h2(abt: pd.DataFrame) -> dict:
    """H2: saneamento basico > 80% -> mortalidade <5 menor (t-test/Mann-Whitney)."""
    d = abt.dropna(subset=["sanitation_basic", CM])
    alto = d.loc[d["sanitation_basic"] > 80, CM]
    baixo = d.loc[d["sanitation_basic"] <= 80, CM]
    t, p_t = stats.ttest_ind(alto, baixo, equal_var=False)
    _, p_u = stats.mannwhitneyu(alto, baixo, alternative="two-sided")
    diff = float(alto.mean() - baixo.mean())
    se = np.sqrt(alto.var(ddof=1) / len(alto) + baixo.var(ddof=1) / len(baixo))
    gl = (alto.var(ddof=1) / len(alto) + baixo.var(ddof=1) / len(baixo)) ** 2 / (
        (alto.var(ddof=1) / len(alto)) ** 2 / (len(alto) - 1)
        + (baixo.var(ddof=1) / len(baixo)) ** 2 / (len(baixo) - 1)
    )
    tc = stats.t.ppf(0.975, gl)
    return _resultado(
        "H2", "Paises com saneamento basico > 80% tem mortalidade <5 menor",
        "t-test de Welch + Mann-Whitney", diff, "mortes/1k (dif. de medias)",
        diff - tc * se, diff + tc * se, p_t, len(d), "-", 5.0,
        extra={"p_mannwhitney": float(p_u), "media_gt80": round(float(alto.mean()), 2),
               "media_le80": round(float(baixo.mean()), 2)},
    )


def h3(abt: pd.DataFrame) -> dict:
    """H3: +1 DP no gasto publico em saude -> variacao na expectativa de vida."""
    r = _regressao(abt, LE, "public_health_exp_gdp")
    return _resultado(
        "H3", "+1 DP no gasto publico em saude (% PIB) eleva a expectativa de vida",
        "PanelOLS FE pais+ano, SE cluster",
        r["efeito"], "anos de LE por +1 DP", r["lo"], r["hi"], r["p"], r["n"], "+", 0.5,
        extra={"efeito_entity": r["efeito_entity"], "p_entity": r["p_entity"]},
    )


def h4(abt: pd.DataFrame) -> dict:
    """H4: urbanizacao associada a menor mortalidade <5."""
    r = _regressao(abt, CM, "urban_pct")
    return _resultado(
        "H4", "Maior urbanizacao associada a menor mortalidade <5",
        "PanelOLS FE pais+ano, SE cluster",
        r["efeito"], "mortes/1k por +1 DP de urbanizacao", r["lo"], r["hi"], r["p"], r["n"],
        "-", 1.0,
        extra={"efeito_entity": r["efeito_entity"], "p_entity": r["p_entity"]},
    )


def h5(abt: pd.DataFrame, corte: int = 2010) -> dict:
    """H5: cruzar o limiar UHC acelera o ganho de expectativa de vida (DiD 2x2).

    Como os controles *never-treated* nao tem `treat_year`, usamos um corte
    calendario fixo: tratados = cruzaram o limiar ate `corte`; controles = nunca
    cruzaram; `post` = ano >= corte. Tratamento escalonado/heterogeneo fica no F7.
    """
    d = abt.copy()
    d["g"] = d["treat_year"].notna() & (d["treat_year"] <= corte)
    d = d[d["g"] | d["never_treated"]].dropna(subset=[LE, "uhc_index"]).copy()
    d["post_i"] = (d["year"] >= corte).astype(int)
    d["gd"] = d["g"].astype(int) * d["post_i"]
    res = _panel_fe(d, LE, ["gd"])
    ci = res.conf_int().loc["gd"]
    return _resultado(
        "H5", f"Apos cruzar o limiar UHC (0.5) ate {corte}, o ganho de LE acelera (DiD 2x2)",
        "DiD two-way FE (pais+ano), SE cluster", float(res.params["gd"]), "anos de LE (efeito DiD)",
        float(ci.iloc[0]), float(ci.iloc[1]), float(res.pvalues["gd"]), int(res.nobs), "+", 0.5,
        extra={"n_tratados": int(d.loc[d["g"], "country_code"].nunique()),
               "n_controles": int(d.loc[d["never_treated"], "country_code"].nunique()),
               "observacao": "Resultado negativo sugere convergencia: paises nunca "
                             "tratados (mais pobres) ganharam LE mais rapido no periodo."},
    )


def h6(abt: pd.DataFrame) -> dict:
    """H6: vacinacao contra sarampo associada a menor mortalidade <5."""
    r = _regressao(abt, CM, "measles_imm_pct")
    return _resultado(
        "H6", "Maior cobertura de sarampo associada a menor mortalidade <5",
        "PanelOLS FE pais+ano, SE cluster",
        r["efeito"], "mortes/1k por +1 DP de cobertura", r["lo"], r["hi"], r["p"], r["n"],
        "-", 1.0,
        extra={"efeito_entity": r["efeito_entity"], "p_entity": r["p_entity"]},
    )


HIPOTESES = [h1, h2, h3, h4, h5, h6]


def rodar_todas() -> list[dict]:
    """Executa H1-H6 sobre a ABT atual."""
    abt = carregar_abt()
    return [h(abt) for h in HIPOTESES]


def run() -> list[dict]:
    """Executa as hipoteses e grava `reports/hypotheses.json`."""
    resultados = rodar_todas()
    write_json(settings.REPORTS_DIR / "hypotheses.json", resultados)
    suportadas = sum(1 for r in resultados if r["veredito"].startswith("suportada"))
    print(f"[hypotheses] {suportadas}/{len(resultados)} suportadas; "
          f"detalhes em reports/hypotheses.json")
    return resultados


if __name__ == "__main__":
    run()
