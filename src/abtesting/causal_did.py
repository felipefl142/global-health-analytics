"""Causal: expansao de UHC (proxy cruza o limiar) -> expectativa de vida.

Tres estimadores, do mais simples ao mais robusto:

1. TWFE DiD: y ~ post + FE pais + FE ano (+ log PIB), SE cluster por pais.
   Com adocao escalonada o TWFE mistura comparacoes "proibidas" (tratados antigos
   como controle de tratados novos) -> reportado so como referencia.
2. DiD escalonado por coorte (estilo Callaway & Sant'Anna, controle = nunca tratados):
   ATT(g, t) = E[y_t - y_{g-1} | coorte g] - E[y_t - y_{g-1} | nunca tratados].
   Agrega por tempo relativo e = t - g (event study; e < -1 = teste de pre-tendencia)
   e num ATT geral (media de e >= 0 ponderada pelo tamanho da coorte).
   IC por bootstrap de paises (cluster).
   Variante com ajuste de tendencia linear: ajusta uma reta aos coeficientes pre
   (e <= -1, com att(-1) = 0 por construcao) e subtrai a extrapolacao no pos -
   separa "quebra na adocao" de "tendencia diferencial que ja existia" (convergencia).
3. Controle sintetico p/ paises "herois": pesos >= 0, soma 1, sobre doadores nunca
   tratados, ajustados no pre-periodo em desvios da media pre de cada pais
   (de-meaned SCM, Ferman & Pinto) - o nivel do pais tratado fica fora do fecho convexo
   dos doadores (mais pobres); inferencia por placebo no espaco (razao RMSPE).

Amostra: 2000-2023 (composicao do proxy estavel), so tratados + nunca-tratados bem
observados (paises sem dado de insumos nao sao controle valido).

Uso: python -m src.abtesting.causal_did   -> reports/causal_did.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import settings
from src.features.uhc_index import DID_START_YEAR
from src.utils.io import load_parquet

OUTCOME = "life_expectancy"
EVENT_WINDOW = (-6, 8)
N_BOOT = 499


def did_sample(abt: pd.DataFrame, outcome: str = OUTCOME,
               start_year: int = DID_START_YEAR) -> pd.DataFrame:
    """Painel p/ DiD: >= start_year, tratados + nunca-tratados, com outcome observado."""
    keep = (abt["year"] >= start_year) & (abt["treated"] | abt["never_treated"])
    df = abt[keep].dropna(subset=[outcome])
    df = df.copy()
    df["log_gdp_pc"] = np.log(df["gdp_per_capita"])
    return df


# --------------------------------------------------------------------- 1. TWFE
def twfe(df: pd.DataFrame, outcome: str = OUTCOME, controls: bool = True) -> dict:
    """TWFE com FE pais + ano e SE cluster por pais (linearmodels.PanelOLS)."""
    from linearmodels.panel import PanelOLS

    cols = ["post"] + (["log_gdp_pc"] if controls else [])
    d = df.dropna(subset=cols + [outcome]).set_index(["country_id", "year"])
    res = PanelOLS(d[outcome], d[cols], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    lo, hi = res.conf_int().loc["post"]
    return {"estimator": "twfe", "controls": controls, "att": float(res.params["post"]),
            "se": float(res.std_errors["post"]), "ci95": [float(lo), float(hi)],
            "p": float(res.pvalues["post"]), "n_obs": int(res.nobs),
            "n_countries": int(d.index.get_level_values(0).nunique())}


# ------------------------------------------------------ 2. DiD escalonado por coorte
def _att_gt(wide: pd.DataFrame, cohorts: pd.Series, never: list[str],
            control: str = "never") -> pd.DataFrame:
    """ATT(g, t) com base g-1. wide: pais x ano (outcome).

    control="never": so nunca-tratados; "notyet": nunca-tratados + coortes ainda nao
    tratadas em max(t, base) (g' > max(t, base)).
    """
    rows = []
    years = wide.columns
    for g, members in cohorts.groupby(cohorts):
        base = g - 1
        if base not in years:
            continue
        treated = list(members.index)
        for t in years:
            e = t - g
            if not (EVENT_WINDOW[0] <= e <= EVENT_WINDOW[1]) or t == base:
                continue
            ctrl = list(never)
            if control == "notyet":
                ctrl += list(cohorts.index[cohorts > max(t, base)])
            d_tr = (wide.loc[treated, t] - wide.loc[treated, base]).dropna()
            d_nv = (wide.loc[ctrl, t] - wide.loc[ctrl, base]).dropna()
            if len(d_tr) == 0 or len(d_nv) < 2:
                continue
            rows.append({"g": g, "t": t, "e": e, "att": d_tr.mean() - d_nv.mean(),
                         "n": len(d_tr)})
    return pd.DataFrame(rows)


def _aggregate(att_gt: pd.DataFrame, detrend: bool = False) -> tuple[pd.Series, float]:
    """Event study (media ponderada por n em cada e) + ATT geral (media de e >= 0).

    detrend=True: subtrai a reta ajustada aos coeficientes pre (inclui e=-1 com att 0).
    """
    if att_gt.empty:
        return pd.Series(dtype=float), np.nan
    w = att_gt.assign(wa=att_gt["att"] * att_gt["n"])
    by_e = w.groupby("e")["wa"].sum() / w.groupby("e")["n"].sum()
    if detrend:
        pre = pd.concat([by_e[by_e.index < -1], pd.Series({-1: 0.0})])
        if len(pre) >= 3:
            slope, intercept = np.polyfit(pre.index.astype(float), pre.values, 1)
            by_e = by_e - (intercept + slope * by_e.index.astype(float))
    post = by_e[by_e.index >= 0]
    n_post = w[w["e"] >= 0].groupby("e")["n"].sum().reindex(post.index)
    overall = float((post * n_post).sum() / n_post.sum()) if len(post) else np.nan
    return by_e, overall


def staggered_did(df: pd.DataFrame, outcome: str = OUTCOME, n_boot: int = N_BOOT,
                  seed: int = 42, control: str = "never", detrend: bool = False) -> dict:
    """DiD escalonado por coorte com bootstrap de paises p/ ICs (ajuste de tendencia opcional)."""
    wide = df.pivot_table(index="country_id", columns="year", values=outcome)
    info = df.groupby("country_id")[["treated", "treat_year"]].first()
    cohorts = info.loc[info["treated"], "treat_year"].astype(int)
    cohorts = cohorts[cohorts.index.isin(wide.index)]
    never = [c for c in info.index[~info["treated"]] if c in wide.index]

    by_e, overall = _aggregate(_att_gt(wide, cohorts, never, control), detrend)

    rng = np.random.default_rng(seed)
    boot_e, boot_all = [], []
    tr_ids, nv_ids = np.array(cohorts.index), np.array(never)
    for _ in range(n_boot):
        tr_s = rng.choice(tr_ids, size=len(tr_ids), replace=True)
        nv_s = rng.choice(nv_ids, size=len(nv_ids), replace=True)
        # reindexa com sufixo p/ permitir paises repetidos na reamostragem
        tr_keys = [f"{c}#{i}" for i, c in enumerate(tr_s)]
        nv_keys = [f"{c}#n{i}" for i, c in enumerate(nv_s)]
        w_b = pd.concat([wide.loc[tr_s].set_axis(tr_keys), wide.loc[nv_s].set_axis(nv_keys)])
        coh_b = pd.Series(cohorts.loc[tr_s].values, index=tr_keys)
        be, bo = _aggregate(_att_gt(w_b, coh_b, nv_keys, control), detrend)
        boot_e.append(be)
        boot_all.append(bo)
    boot_e = pd.concat(boot_e, axis=1)
    ci_e = boot_e.quantile([0.025, 0.975], axis=1).T
    se_all = float(np.nanstd(boot_all, ddof=1))
    lo, hi = np.nanpercentile(boot_all, [2.5, 97.5])

    event = [{"e": int(e), "att": float(by_e[e]),
              "ci95": [float(ci_e.loc[e, 0.025]), float(ci_e.loc[e, 0.975])]}
             for e in by_e.index]
    pre = [r for r in event if r["e"] < -1]
    pre_violations = [r["e"] for r in pre if not (r["ci95"][0] <= 0 <= r["ci95"][1])]
    # p bilateral pela distribuicao bootstrap (consistente com o IC percentil)
    b = np.asarray(boot_all)[~np.isnan(boot_all)]
    p = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))) if len(b) else np.nan
    return {
        "estimator": "staggered_cohort_did", "control": control, "detrended": detrend,
        "att": overall, "se": se_all, "ci95": [float(lo), float(hi)], "p": p,
        "n_treated": int(len(cohorts)), "n_never": int(len(never)),
        "cohorts": {int(g): int(n) for g, n in cohorts.value_counts().sort_index().items()},
        "event_study": event,
        "pretrend_ok": len(pre_violations) == 0,
        "pretrend_violations_e": pre_violations,
    }


# ------------------------------------------------------ 3. Controle sintetico
def _fit_weights(y_pre_treated: np.ndarray, y_pre_donors: np.ndarray) -> np.ndarray:
    k = y_pre_donors.shape[1]
    obj = lambda w: np.mean((y_pre_treated - y_pre_donors @ w) ** 2)  # noqa: E731
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
                   options={"maxiter": 500})
    return res.x


def _synth_one(wide: pd.DataFrame, unit: str, donors: list[str], t0: int) -> dict:
    pre = [c for c in wide.columns if c < t0]
    post = [c for c in wide.columns if c >= t0]
    # de-meaned: cada serie menos sua media pre; o sintetico herda o nivel do tratado
    dm = wide.sub(wide[pre].mean(axis=1), axis=0)
    w = _fit_weights(dm.loc[unit, pre].values, dm.loc[donors, pre].values.T)
    synth = dm.loc[donors].T.values @ w + wide.loc[unit, pre].mean()
    gap = wide.loc[unit].values - synth
    n_pre = len(pre)
    rmspe_pre = float(np.sqrt(np.mean(gap[:n_pre] ** 2)))
    rmspe_post = float(np.sqrt(np.mean(gap[n_pre:] ** 2)))
    return {"weights": w, "synth": synth, "gap": gap, "rmspe_pre": rmspe_pre,
            "rmspe_post": rmspe_post, "ratio": rmspe_post / max(rmspe_pre, 1e-9),
            "att": float(np.mean(gap[n_pre:])), "years": list(wide.columns),
            "post_years": post}


def synthetic_control(df: pd.DataFrame, unit: str, outcome: str = OUTCOME) -> dict:
    """Controle sintetico p/ um pais tratado + placebo no espaco (p-valor por razao RMSPE)."""
    info = df.groupby("country_id")[["treated", "treat_year"]].first()
    t0 = int(info.loc[unit, "treat_year"])
    wide = df.pivot_table(index="country_id", columns="year", values=outcome)
    wide = wide.dropna(axis=0, how="any")  # doadores com serie completa
    if unit not in wide.index:
        raise ValueError(f"{unit} sem serie completa de {outcome}")
    donors = [c for c in wide.index if c in info.index and not info.loc[c, "treated"]]
    main = _synth_one(wide, unit, donors, t0)
    placebo_ratios = []
    for d in donors:
        others = [x for x in donors if x != d]
        placebo_ratios.append(_synth_one(wide, d, others, t0)["ratio"])
    p = (1 + sum(r >= main["ratio"] for r in placebo_ratios)) / (1 + len(placebo_ratios))
    top = sorted(zip(donors, main["weights"], strict=True), key=lambda x: -x[1])[:5]
    return {
        "unit": unit, "treat_year": t0, "att_post_mean": main["att"],
        "rmspe_pre": main["rmspe_pre"], "rmspe_post": main["rmspe_post"],
        "placebo_p": float(p), "n_donors": len(donors),
        "top_donors": {c: round(float(w), 3) for c, w in top if w > 0.001},
        "series": {"years": [int(y) for y in main["years"]],
                   "actual": [float(v) for v in wide.loc[unit].values],
                   "synthetic": [float(v) for v in main["synth"]]},
    }


def pick_hero_countries(df: pd.DataFrame, n: int = 2, min_pre: int = 6, min_post: int = 5,
                        outcome: str = OUTCOME) -> list[str]:
    """Tratados com pre e pos longos e serie completa; os mais populosos primeiro."""
    wide = df.pivot_table(index="country_id", columns="year", values=outcome).dropna()
    info = df.groupby("country_id").agg(treat_year=("treat_year", "first"),
                                        pop=("population", "median"))
    years = wide.columns
    ok = info[info["treat_year"].notna() & info.index.isin(wide.index)]
    ok = ok[(ok["treat_year"] - years.min() >= min_pre) & (years.max() - ok["treat_year"] + 1 >= min_post)]
    return list(ok.sort_values("pop", ascending=False).index[:n])


def run(abt: pd.DataFrame | None = None, n_boot: int = N_BOOT) -> dict:
    abt = load_parquet("gold", "abt_country_year") if abt is None else abt
    df = did_sample(abt)
    names = abt.drop_duplicates("country_id").set_index("country_id")["country_name"]
    heroes = pick_hero_countries(df)
    return {
        "outcome": OUTCOME,
        "sample": {"years": [int(df["year"].min()), int(df["year"].max())],
                   "countries": int(df["country_id"].nunique())},
        "twfe": twfe(df, controls=False),
        "twfe_controls": twfe(df, controls=True),
        "staggered": staggered_did(df, n_boot=n_boot),
        "staggered_notyet": staggered_did(df, n_boot=n_boot, control="notyet"),
        "staggered_detrended": staggered_did(df, n_boot=n_boot, detrend=True),
        "synthetic_control": [{**synthetic_control(df, u), "name": names.get(u, u)}
                              for u in heroes],
    }


def main() -> None:
    settings.ensure_dirs()
    res = run()
    out = settings.REPORTS_DIR / "causal_did.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    st = res["staggered"]
    print(f"TWFE: ATT={res['twfe']['att']:.3f} (p={res['twfe']['p']:.3g})  "
          f"c/ PIB: {res['twfe_controls']['att']:.3f} (p={res['twfe_controls']['p']:.3g})")
    print(f"Escalonado: ATT={st['att']:.3f} IC95={np.round(st['ci95'], 3).tolist()} "
          f"p={st['p']:.3g} tratados={st['n_treated']} nunca={st['n_never']} "
          f"pre-tendencia ok={st['pretrend_ok']} {st['pretrend_violations_e']}")
    ny = res["staggered_notyet"]
    print(f"Escalonado (not-yet): ATT={ny['att']:.3f} IC95={np.round(ny['ci95'], 3).tolist()} "
          f"p={ny['p']:.3g} pre-tendencia ok={ny['pretrend_ok']}")
    dt = res["staggered_detrended"]
    print(f"Escalonado (tendencia ajustada): ATT={dt['att']:.3f} "
          f"IC95={np.round(dt['ci95'], 3).tolist()} p={dt['p']:.3g}")
    for sc in res["synthetic_control"]:
        print(f"Sintetico {sc['name']} ({sc['treat_year']}): gap pos={sc['att_post_mean']:.2f} "
              f"RMSPE pre={sc['rmspe_pre']:.2f} placebo p={sc['placebo_p']:.3f} "
              f"doadores={sc['top_donors']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
