"""Catalogo de testes de hipotese (F5) - cada um com H0/H1, teste, efeito, IC, p e
significancia pratica. Resultado consolidado em reports/hypotheses.json.

Convencoes:
- Regressoes de painel (H1, H3, H4, H6): FE pais + ano (variacao DENTRO do pais, liquida
  de choques globais) + controle log(PIB/capita), SE cluster por pais. A variavel de
  interesse e padronizada -> coeficiente = efeito de +1 DP.
- H2: comparacao transversal (snapshot 2019) com Mann-Whitney + Cliff's delta +
  IC bootstrap da diferenca de medianas; e OLS com log(PIB) p/ checar confundimento.
- H5: DiD escalonado com ajuste de tendencia (src.abtesting.causal_did).
- Correcao de Holm sobre os p-valores principais das 6 hipoteses.
- Significancia pratica: |efeito| >= limiar minimo relevante definido por hipotese.
- Robustez: especificacoes alternativas do MESMO estimando (variacao dentro do pais):
  FE so de pais (sem FE de ano) e FE sem Europa/Asia Central. Se alguma for significativa
  com sinal oposto ao principal, o veredito vira "inconclusiva". O pooled (entre+dentro)
  e reportado, mas mede outra coisa e nao entra no veredito.

Uso: python -m src.analysis.hypotheses
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import settings
from src.utils.io import load_parquet

SNAPSHOT_YEAR = 2019


@dataclass
class HypothesisResult:
    id: str
    title: str
    h0: str
    h1: str
    test: str
    effect: float
    effect_unit: str
    ci95: list[float]
    p: float
    n: int
    min_relevant_effect: float
    expected_sign: int  # +1 / -1: direcao prevista em H1
    extra: dict
    p_holm: float | None = None
    verdict: str | None = None


def _prep(abt: pd.DataFrame) -> pd.DataFrame:
    df = abt.copy()
    df["log_gdp_pc"] = np.log(df["gdp_per_capita"])
    return df


def panel_fe(df: pd.DataFrame, y: str, x: str, controls: tuple[str, ...] = ("log_gdp_pc",),
             entity_effects: bool = True, time_effects: bool = True) -> dict:
    """y ~ z(x) + controles (+ FE ano) (+ FE pais); SE cluster por pais. Efeito por +1 DP de x.

    entity_effects=False -> 'pooled': usa tambem a variacao ENTRE paises.
    time_effects=False -> sem controle de choques/tendencias globais comuns.
    """
    from linearmodels.panel import PanelOLS

    d = df.dropna(subset=[y, x, *controls]).copy()
    sd = d[x].std()
    d["z"] = (d[x] - d[x].mean()) / sd
    d = d.set_index(["country_id", "year"])
    res = PanelOLS(d[y], d[["z", *controls]], entity_effects=entity_effects,
                   time_effects=time_effects, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)
    lo, hi = res.conf_int().loc["z"]
    return {"coef": float(res.params["z"]), "ci95": [float(lo), float(hi)],
            "p": float(res.pvalues["z"]), "n": int(res.nobs), "sd_x": float(sd),
            "n_countries": int(d.index.get_level_values(0).nunique()),
            "r2_within": float(res.rsquared_within)}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b)."""
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff)).mean())


def _h_panel(df, hid, title, y, x, sign, min_eff, unit, h0, h1) -> HypothesisResult:
    r = panel_fe(df, y, x)
    # robustez: variacao entre paises (sem FE pais) e sem Europa/Asia Central
    # (transicao pos-sovietica dos anos 90: muitos medicos/leitos e LE em queda)
    pooled = panel_fe(df, y, x, entity_effects=False)
    no_eca = panel_fe(df[df["region"] != "Europe & Central Asia"], y, x)
    no_year = panel_fe(df, y, x, time_effects=False)
    robust = {k: {"coef": v["coef"], "ci95": v["ci95"], "p": v["p"], "n": v["n"],
                  "same_estimand": same}
              for k, v, same in (("pooled_between_within", pooled, False),
                                 ("fe_sem_europa_asia_central", no_eca, True),
                                 ("fe_pais_sem_fe_ano", no_year, True))}
    return HypothesisResult(
        id=hid, title=title, h0=h0, h1=h1,
        test="PanelOLS FE pais+ano, controle log(PIB/cap), SE cluster pais",
        effect=r["coef"], effect_unit=unit, ci95=r["ci95"], p=r["p"], n=r["n"],
        min_relevant_effect=min_eff, expected_sign=sign,
        extra={"sd_x": r["sd_x"], "n_countries": r["n_countries"], "r2_within": r["r2_within"],
               "robustness": robust})


def h2_sanitation(df: pd.DataFrame, seed: int = 42) -> HypothesisResult:
    snap = df[df["year"] == SNAPSHOT_YEAR].dropna(subset=["sanitation_basic", "child_mortality"])
    hi = snap.loc[snap["sanitation_basic"] > 80, "child_mortality"].to_numpy()
    lo = snap.loc[snap["sanitation_basic"] <= 80, "child_mortality"].to_numpy()
    u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(hi, len(hi))) - np.median(rng.choice(lo, len(lo)))
            for _ in range(2000)]
    diff = float(np.median(hi) - np.median(lo))
    # confundimento: diferenca ajustada por log(PIB) via OLS
    import statsmodels.formula.api as smf
    d = snap.dropna(subset=["log_gdp_pc"]).assign(high_san=lambda x: (x["sanitation_basic"] > 80).astype(int))
    ols = smf.ols("child_mortality ~ high_san + log_gdp_pc", data=d).fit(cov_type="HC3")
    return HypothesisResult(
        id="H2", title="Saneamento basico > 80% => mortalidade <5 menor",
        h0="Distribuicao da mortalidade <5 igual entre paises com saneamento > 80% e <= 80%",
        h1="Paises com saneamento > 80% tem mortalidade <5 menor",
        test=f"Mann-Whitney (snapshot {SNAPSHOT_YEAR}) + IC bootstrap da dif. de medianas",
        effect=diff, effect_unit="mortes/1000 nascidos (dif. de medianas)",
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        p=float(p), n=int(len(hi) + len(lo)), min_relevant_effect=5.0, expected_sign=-1,
        extra={"n_high": int(len(hi)), "n_low": int(len(lo)),
               "median_high": float(np.median(hi)), "median_low": float(np.median(lo)),
               "cliffs_delta": cliffs_delta(hi, lo),
               "adjusted_gdp": {"coef": float(ols.params["high_san"]),
                                "ci95": [float(v) for v in ols.conf_int().loc["high_san"]],
                                "p": float(ols.pvalues["high_san"])}})


def h5_did(abt: pd.DataFrame, n_boot: int = 199) -> HypothesisResult:
    from src.abtesting.causal_did import did_sample, staggered_did
    r = staggered_did(did_sample(abt), n_boot=n_boot, detrend=True)
    return HypothesisResult(
        id="H5", title="Apos cruzar o limiar UHC, a expectativa de vida acelera (DiD)",
        h0="ATT da adocao (proxy UHC >= 0.5) sobre a expectativa de vida = 0",
        h1="ATT > 0 (ganho de expectativa de vida apos cruzar o limiar)",
        test="DiD escalonado por coorte (controle nunca-tratado), ajuste de tendencia linear, "
             "bootstrap de paises",
        effect=float(r["att"]), effect_unit="anos (ATT medio pos-adocao)", ci95=r["ci95"],
        p=float(r["p"]), n=int(r["n_treated"] + r["n_never"]), min_relevant_effect=0.5,
        expected_sign=+1, extra={"n_treated": r["n_treated"], "n_never": r["n_never"],
                                 "event_study": r["event_study"]})


def sign_flips(h: HypothesisResult, alpha: float = 0.05) -> list[str]:
    """Especificacoes do mesmo estimando, significativas e com sinal oposto ao principal."""
    rb = h.extra.get("robustness", {})
    return [k for k, v in rb.items()
            if v.get("same_estimand") and v["p"] < alpha and np.sign(v["coef"]) != np.sign(h.effect)]


def verdict(h: HypothesisResult, alpha: float = 0.05) -> str:
    flips = sign_flips(h, alpha)
    if flips:
        return f"Inconclusiva - sensivel a especificacao ({', '.join(flips)} inverte o sinal)"
    sig = h.p_holm is not None and h.p_holm < alpha
    if not sig:
        return "Nao rejeita H0"
    if np.sign(h.effect) != h.expected_sign:
        return "Rejeita H0 - efeito na direcao OPOSTA a H1"
    if abs(h.effect) >= h.min_relevant_effect:
        return "Rejeita H0 - efeito relevante"
    return "Rejeita H0 - efeito pequeno (abaixo do minimo relevante)"


def run_all(abt: pd.DataFrame | None = None, n_boot: int = 199) -> list[HypothesisResult]:
    abt = load_parquet("gold", "abt_country_year") if abt is None else abt
    df = _prep(abt)
    hs = [
        _h_panel(df, "H1", "Densidade de medicos => expectativa de vida (liquido de PIB)",
                 "life_expectancy", "doctors_per_1000", +1, 0.5, "anos por +1 DP de medicos/1000",
                 "Coeficiente de medicos/1000 sobre LE = 0 (dentro do pais, dado PIB)",
                 "Mais medicos/1000 => maior LE"),
        h2_sanitation(df),
        _h_panel(df, "H3", "+1 DP no gasto publico em saude => delta na expectativa de vida",
                 "life_expectancy", "public_health_exp_gdp", +1, 0.5,
                 "anos por +1 DP de gasto publico (% PIB)",
                 "Coeficiente do gasto publico em saude (% PIB) sobre LE = 0",
                 "Mais gasto publico => maior LE"),
        _h_panel(df, "H4", "Urbanizacao associada a mortalidade <5 menor",
                 "child_mortality", "urban_pct", -1, 5.0, "mortes/1000 por +1 DP de urbanizacao",
                 "Coeficiente da urbanizacao sobre mortalidade <5 = 0",
                 "Mais urbanizacao => menor mortalidade <5"),
        h5_did(abt, n_boot=n_boot),
        _h_panel(df, "H6", "Vacinacao contra sarampo associada a mortalidade <5 menor",
                 "child_mortality", "measles_imm_pct", -1, 5.0,
                 "mortes/1000 por +1 DP de cobertura vacinal",
                 "Coeficiente da vacinacao (sarampo) sobre mortalidade <5 = 0",
                 "Mais vacinacao => menor mortalidade <5"),
    ]
    _, p_adj, _, _ = multipletests([h.p for h in hs], method="holm")
    for h, pa in zip(hs, p_adj, strict=True):
        h.p_holm = float(pa)
        h.verdict = verdict(h)
    return hs


def to_frame(hs: list[HypothesisResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": h.id, "hipotese": h.title, "efeito": round(h.effect, 3), "unidade": h.effect_unit,
        "ic95": f"[{h.ci95[0]:.2f}, {h.ci95[1]:.2f}]", "p": h.p, "p_holm": h.p_holm,
        "n": h.n, "min_relevante": h.min_relevant_effect, "veredito": h.verdict,
    } for h in hs])


def main() -> None:
    settings.ensure_dirs()
    hs = run_all()
    out = settings.REPORTS_DIR / "hypotheses.json"
    out.write_text(json.dumps([asdict(h) for h in hs], indent=2, ensure_ascii=False))
    with pd.option_context("display.max_colwidth", 60, "display.width", 250):
        print(to_frame(hs).drop(columns=["unidade"]).to_string(index=False))
    for h in hs:
        for k, v in h.extra.get("robustness", {}).items():
            print(f"  {h.id} {k:28} coef={v['coef']:+.3f} IC={np.round(v['ci95'], 2).tolist()} p={v['p']:.2g}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
