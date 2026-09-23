"""A/B simulado: "e se paises recebessem uma expansao de insumos UHC?"

Desenho:
- Unidade de randomizacao = pais (evita correlacao intra-pais de pais x ano).
- Snapshot de 2019 (pre-COVID) com features completas p/ os 3 modelos.
- Tratamento: uplift relativo nos insumos UHC (medicos, enfermeiros, leitos, gasto,
  saneamento/agua, vacinas; percentuais limitados a 100) e recalculo do proxy UHC.
- Outcome = predicao do modelo (M1/M2/M3) + ruido realista: residuo reamostrado do
  conjunto de teste do proprio modelo (sem ruido o teste seria trivial).
- Analise: poder/MDE, Welch t-test, IC da diferenca, Cohen's d, CUPED (covariavel =
  outcome observado em 2015) e correcao de Holm p/ os 3 outcomes.

Uso: python -m src.abtesting.simulate_ab [--uplift 0.2] [--seed 42]
     -> reports/ab_simulation.json
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

from config import settings
from src.features.uhc_index import DEFAULT_COMPONENTS
from src.modeling.dataset import build_dataset
from src.utils.io import load_parquet

SNAPSHOT_YEAR = 2019
PRE_YEAR = 2015
PCT_COLS = {"sanitation_basic", "water_basic", "sanitation_safely", "water_safely",
            "measles_imm_pct", "dpt_imm_pct"}
UPLIFT_COLS = set(DEFAULT_COMPONENTS) | {"sanitation_safely", "water_safely",
                                          "public_health_exp_gdp", "health_exp_gdp"}
OUTCOMES = {  # outcome -> (modelo, coluna observada p/ CUPED, "maior e melhor")
    "life_expectancy": ("life_expectancy_reg", "life_expectancy", True),
    "child_mortality": ("child_mortality_reg", "child_mortality", False),
    "milestone_prob": ("milestone_high_clf", "milestone_high", True),
}


def apply_uplift(X: pd.DataFrame, uplift: float) -> pd.DataFrame:
    """Aumenta insumos UHC em `uplift` (relativo); percentuais limitados a 100."""
    out = X.copy()
    for c in UPLIFT_COLS & set(out.columns):
        out[c] = out[c] * (1 + uplift)
        if c in PCT_COLS:
            out[c] = out[c].clip(upper=100)
    return out


def recompute_uhc(X: pd.DataFrame, abt: pd.DataFrame) -> pd.DataFrame:
    """Recalcula o proxy com a mesma normalizacao (percentis do painel original)."""
    from src.features.uhc_index import HI, LO, LOG_COMPONENTS
    out = X.copy()
    if "uhc_index" not in out.columns:
        return out
    num = pd.Series(0.0, index=out.index)
    w_av = pd.Series(0.0, index=out.index)
    for name, w in DEFAULT_COMPONENTS.items():
        if name not in out.columns:
            continue
        ref = abt[name].dropna()
        s = out[name]
        if name in LOG_COMPONENTS:
            ref, s = np.log1p(ref.clip(lower=0)), np.log1p(s.clip(lower=0))
        p_lo, p_hi = np.nanpercentile(ref, [LO * 100, HI * 100])
        norm = ((s - p_lo) / (p_hi - p_lo)).clip(0, 1)
        num += norm.fillna(0) * w
        w_av += norm.notna() * w
    out["uhc_index"] = np.where(w_av > 0, num / w_av.replace(0, np.nan), out["uhc_index"])
    return out


def _load_model(name: str):
    return joblib.load(settings.MODELS_DIR / f"{name}_xgb.joblib")


def _predict(model, X: pd.DataFrame, kind_clf: bool) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if kind_clf else model.predict(X)


def _test_residuals(target: str, model, kind_clf: bool) -> np.ndarray:
    md = build_dataset(target)
    if kind_clf:  # p/ probabilidade: residuo em escala de prob. (y - p)
        return md.y_test.values - model.predict_proba(md.X_test)[:, 1]
    return md.y_test.values - model.predict(md.X_test)


def power_analysis(sd: float, n_per_arm: int, effect: float, alpha: float = 0.05) -> dict:
    """MDE (80% de poder) com o n atual e n necessario p/ detectar o efeito observado."""
    pw = TTestIndPower()
    mde_d = pw.solve_power(nobs1=n_per_arm, alpha=alpha, power=0.8, ratio=1.0)
    d = abs(effect) / sd if sd > 0 else np.nan
    n_needed = (pw.solve_power(effect_size=d, alpha=alpha, power=0.8, ratio=1.0)
                if d and np.isfinite(d) and d > 0.01 else None)
    return {"n_per_arm": n_per_arm, "mde_cohens_d": float(mde_d), "mde_abs": float(mde_d * sd),
            "achieved_power": float(pw.power(effect_size=d, nobs1=n_per_arm, alpha=alpha))
            if np.isfinite(d) else None,
            "n_per_arm_for_80pct_power": int(np.ceil(n_needed)) if n_needed else None}


def analyze(y: np.ndarray, treat: np.ndarray, x_pre: np.ndarray | None = None) -> dict:
    """Welch t-test + IC + Cohen's d; CUPED se houver covariavel pre."""
    yt, yc = y[treat == 1], y[treat == 0]
    diff = yt.mean() - yc.mean()
    t, p = stats.ttest_ind(yt, yc, equal_var=False)
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    dof = se ** 4 / ((yt.var(ddof=1) / len(yt)) ** 2 / (len(yt) - 1)
                     + (yc.var(ddof=1) / len(yc)) ** 2 / (len(yc) - 1))
    q = stats.t.ppf(0.975, dof)
    pooled = np.sqrt(((len(yt) - 1) * yt.var(ddof=1) + (len(yc) - 1) * yc.var(ddof=1))
                     / (len(yt) + len(yc) - 2))
    res = {"n_treat": int(len(yt)), "n_control": int(len(yc)),
           "mean_treat": float(yt.mean()), "mean_control": float(yc.mean()),
           "diff": float(diff), "ci95": [float(diff - q * se), float(diff + q * se)],
           "p": float(p), "cohens_d": float(diff / pooled) if pooled > 0 else None}
    if x_pre is not None:
        ok = ~np.isnan(x_pre)
        theta = np.cov(y[ok], x_pre[ok])[0, 1] / np.var(x_pre[ok], ddof=1)
        y_c = y[ok] - theta * (x_pre[ok] - x_pre[ok].mean())
        cu = analyze(y_c, treat[ok])
        res["cuped"] = {"theta": float(theta), "diff": cu["diff"], "ci95": cu["ci95"],
                        "p": cu["p"], "n": int(ok.sum()),
                        "variance_reduction": float(1 - np.var(y_c, ddof=1) / np.var(y[ok], ddof=1))}
    return res


def simulate(uplift: float = 0.2, seed: int = 42, abt: pd.DataFrame | None = None) -> dict:
    abt = load_parquet("gold", "abt_country_year") if abt is None else abt
    rng = np.random.default_rng(seed)
    snap = abt[abt["year"] == SNAPSHOT_YEAR].dropna(subset=["life_expectancy", "uhc_index"])
    snap = snap.set_index("country_id")
    pre = abt[abt["year"] == PRE_YEAR].set_index("country_id").reindex(snap.index)
    treat = rng.permutation(np.r_[np.ones(len(snap) // 2), np.zeros(len(snap) - len(snap) // 2)])
    treat = treat.astype(int)

    results, pvals = {}, []
    for outcome, (model_name, obs_col, higher_better) in OUTCOMES.items():
        clf = model_name.endswith("_clf")
        model = _load_model(model_name)
        feats = list(model.feature_names_in_)
        X = snap[feats].astype(float)
        X_t = recompute_uhc(apply_uplift(X, uplift), abt)[feats]
        pred = np.where(treat == 1, _predict(model, X_t, clf), _predict(model, X, clf))
        resid = _test_residuals(obs_col, model, clf)
        y = pred + rng.choice(resid, size=len(pred), replace=True)
        if clf:
            y = np.clip(y, 0, 1)
        x_pre = pd.to_numeric(pre[obs_col], errors="coerce").to_numpy(dtype=float)
        true_effect = float(np.mean(_predict(model, X_t, clf) - _predict(model, X, clf)))
        r = analyze(y, treat, x_pre)
        r["true_effect_model"] = true_effect
        r["power"] = power_analysis(float(np.std(y, ddof=1)), int(min(treat.sum(), (1 - treat).sum())),
                                    true_effect)
        if "cuped" in r:  # MDE com a variancia residual apos CUPED
            vr = max(r["cuped"]["variance_reduction"], 0.0)
            r["power"]["mde_abs_cuped"] = r["power"]["mde_abs"] * float(np.sqrt(1 - vr))
        r["higher_is_better"] = higher_better
        results[outcome] = r
        pvals.append(r["cuped"]["p"] if "cuped" in r else r["p"])

    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for r, pa, rj in zip(results.values(), p_adj, reject, strict=True):
        r["p_holm"] = float(pa)
        r["significant_holm"] = bool(rj)
    return {"design": {"unit": "country", "snapshot_year": SNAPSHOT_YEAR, "pre_year": PRE_YEAR,
                       "uplift": uplift, "seed": seed, "n_countries": int(len(snap)),
                       "noise": "residuos de teste reamostrados"},
            "outcomes": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uplift", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    settings.ensure_dirs()
    res = simulate(args.uplift, args.seed)
    out = settings.REPORTS_DIR / "ab_simulation.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"A/B simulado: uplift={args.uplift:.0%} n={res['design']['n_countries']} paises")
    for k, r in res["outcomes"].items():
        cu = r.get("cuped", {})
        print(f"  {k:16} diff={r['diff']:+.3f} p={r['p']:.3g} | CUPED diff={cu.get('diff', np.nan):+.3f} "
              f"p={cu.get('p', np.nan):.3g} var-{cu.get('variance_reduction', 0):.0%} | "
              f"p_holm={r['p_holm']:.3g} | efeito modelo={r['true_effect_model']:+.3f} "
              f"MDE={r['power']['mde_abs']:.3f} (CUPED {r['power'].get('mde_abs_cuped', np.nan):.3f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
