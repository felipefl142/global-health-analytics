"""Monitoramento de drift (F11): dados (PSI/KS por feature) e performance (residuos por ano).

- Referencia = janela de treino (anos <= TRAIN_END); atual = anos >= TEST_START da ABT
  e, se existir, as features do log de predicoes da API (data/monitoring/predictions.jsonl).
- Drift de VALOR: PSI (bins nos quantis da referencia) e KS de 2 amostras, ambos so nos
  valores observados. Alerta: PSI > DRIFT_PSI_THRESHOLD ou KS > DRIFT_KS_THRESHOLD.
- Drift de COBERTURA (sinal separado): |delta da taxa de missing| > MISSING_SHIFT. Misturar os
  dois no PSI faria series que so comecam em 2000 parecerem "drift de valor".
- Performance: RMSE/MAE por ano nos anos com alvo observado vs RMSE de validacao;
  alerta se RMSE do ano > PERF_RATIO x RMSE de validacao.

Uso: python -m src.monitoring.drift [--fail-on-drift]  -> reports/drift_report.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import settings
from src.modeling.dataset import FEATURES, TEST_START, TRAIN_END
from src.utils.io import load_parquet

N_BINS = 10
EPS = 1e-4
PERF_RATIO = 1.5
MISSING_SHIFT = 0.10


def psi(ref: pd.Series, cur: pd.Series, n_bins: int = N_BINS) -> float:
    """Population Stability Index sobre os valores observados (NaN ignorado)."""
    ref_obs, cur_obs = ref.dropna(), cur.dropna()
    if ref_obs.nunique() < 2 or len(cur_obs) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref_obs, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s: pd.Series) -> np.ndarray:
        counts = np.histogram(s, bins=edges)[0].astype(float)
        return np.clip(counts / max(len(s), 1), EPS, None)

    p, q = dist(ref_obs), dist(cur_obs)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        r, c = ref[f], cur[f]
        ks = stats.ks_2samp(r.dropna(), c.dropna()) if r.notna().sum() > 1 and c.notna().sum() > 1 else None
        rows.append({
            "feature": f, "psi": psi(r, c),
            "ks": float(ks.statistic) if ks else np.nan, "ks_p": float(ks.pvalue) if ks else np.nan,
            "missing_ref": float(r.isna().mean()), "missing_cur": float(c.isna().mean()),
            "median_ref": float(r.median()), "median_cur": float(c.median()),
        })
    df = pd.DataFrame(rows)
    df["drift"] = (df["psi"] > settings.DRIFT_PSI_THRESHOLD) | (df["ks"] > settings.DRIFT_KS_THRESHOLD)
    df["missing_shift"] = (df["missing_cur"] - df["missing_ref"]).abs() > MISSING_SHIFT
    return df.sort_values("psi", ascending=False).reset_index(drop=True)


def performance_by_year(abt: pd.DataFrame) -> dict:
    """RMSE/MAE por ano dos modelos de regressao vs RMSE de validacao."""
    from src.serving.predictor import load_models
    out = {}
    for target, m in load_models().items():
        if m.kind != "regression":
            continue
        rep = json.loads((settings.MODELS_DIR / f"{m.name}_eval.json").read_text())
        valid_rmse = rep["xgb"]["valid"]["rmse"]
        d = abt[abt["year"] >= TEST_START - 4].dropna(subset=[target])
        pred = m.model.predict(d[m.features].astype(float))
        err = d[target].to_numpy() - pred
        by_year = (pd.DataFrame({"year": d["year"].to_numpy(), "err": err})
                   .groupby("year")["err"]
                   .agg(rmse=lambda e: float(np.sqrt(np.mean(e ** 2))), mae=lambda e: float(np.mean(np.abs(e))),
                        bias="mean", n="size"))
        by_year["alert"] = by_year["rmse"] > PERF_RATIO * valid_rmse
        out[target] = {"valid_rmse": valid_rmse,
                       "by_year": by_year.reset_index().round(3).to_dict(orient="records")}
    return out


def load_serving_log() -> pd.DataFrame | None:
    from src.serving.predictor import PREDICTION_LOG
    if not PREDICTION_LOG.exists():
        return None
    recs = [json.loads(line) for line in PREDICTION_LOG.read_text().splitlines() if line.strip()]
    if not recs:
        return None
    return pd.DataFrame([{**r["features"], "ts": r["ts"]} for r in recs])


def run(abt: pd.DataFrame | None = None, with_performance: bool = True) -> dict:
    abt = load_parquet("gold", "abt_country_year") if abt is None else abt
    feats = [f for f in FEATURES if f in abt.columns]
    ref = abt[abt["year"] <= TRAIN_END]
    cur = abt[abt["year"] >= TEST_START]
    data = feature_drift(ref, cur, feats)
    report = {
        "reference": f"<= {TRAIN_END}", "current": f">= {TEST_START}",
        "thresholds": {"psi": settings.DRIFT_PSI_THRESHOLD, "ks": settings.DRIFT_KS_THRESHOLD,
                       "perf_ratio": PERF_RATIO},
        "data_drift": data.round(4).replace({np.nan: None}).to_dict(orient="records"),
        "n_features_drifted": int(data["drift"].sum()),
        "n_features_missing_shift": int(data["missing_shift"].sum()),
    }
    serving = load_serving_log()
    if serving is not None and len(serving) >= 30:
        sv = feature_drift(ref, serving, [f for f in feats if f in serving.columns])
        report["serving_drift"] = sv.round(4).replace({np.nan: None}).to_dict(orient="records")
        report["n_serving_predictions"] = int(len(serving))
    if with_performance:
        report["performance"] = performance_by_year(abt)
    report["alert"] = bool(report["n_features_drifted"] > 0 or any(
        r["alert"] for p in report.get("performance", {}).values() for r in p["by_year"]))
    return report


def to_markdown(rep: dict) -> str:
    lines = [f"## Drift report — {'⚠️ ALERTA' if rep['alert'] else '✅ OK'}",
             f"Referência `{rep['reference']}` vs atual `{rep['current']}`; "
             f"{rep['n_features_drifted']} features com drift de valor, "
             f"{rep['n_features_missing_shift']} com mudança de cobertura.", "",
             "| feature | PSI | KS | missing ref → atual | drift |", "|---|---|---|---|---|"]
    for r in rep["data_drift"]:
        psi_v = "–" if r["psi"] is None else f"{r['psi']:.3f}"
        ks_v = "–" if r["ks"] is None else f"{r['ks']:.3f}"
        lines.append(f"| {r['feature']} | {psi_v} | {ks_v} | {r['missing_ref']:.0%} → {r['missing_cur']:.0%} | "
                     f"{'⚠️' if r['drift'] else ''} |")
    for t, p in rep.get("performance", {}).items():
        yrs = ", ".join(f"{r['year']}: {r['rmse']:.2f}{' ⚠️' if r['alert'] else ''}" for r in p["by_year"])
        lines += ["", f"**{t}** — RMSE por ano (validação {p['valid_rmse']}): {yrs}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 se houver alerta (CI)")
    ap.add_argument("--markdown", type=str, default="", help="escreve resumo markdown (ex.: $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()
    settings.ensure_dirs()
    rep = run()
    out = settings.REPORTS_DIR / "drift_report.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    df = pd.DataFrame(rep["data_drift"])
    print(f"Drift de dados (ref {rep['reference']} vs atual {rep['current']}):")
    print(df[["feature", "psi", "ks", "missing_ref", "missing_cur", "drift", "missing_shift"]]
          .to_string(index=False))
    for t, p in rep.get("performance", {}).items():
        yrs = ", ".join(f"{r['year']}:{r['rmse']:.2f}{'!' if r['alert'] else ''}" for r in p["by_year"])
        print(f"Performance {t} (valid RMSE {p['valid_rmse']}): {yrs}")
    print(f"ALERTA: {rep['alert']}  -> {out}")
    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write(to_markdown(rep))
    if args.fail_on_drift and rep["alert"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
