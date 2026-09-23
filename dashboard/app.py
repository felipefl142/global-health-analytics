"""Dashboard Streamlit (F10): EDA, hipoteses, predicao "e se", experimentos e monitoramento.

Le a ABT gold, os modelos e os relatorios (reports/*.json); a predicao roda in-process
com a mesma camada da API (src/serving/predictor.py).

Uso: streamlit run dashboard/app.py   (ou make dashboard)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from src.analysis import eda  # noqa: E402
from src.modeling.dataset import FEATURES  # noqa: E402
from src.viz import style  # noqa: E402

pio.templates["gh"] = style.plotly_template()
pio.templates.default = "gh"
if __name__ == "__main__":
    st.set_page_config(page_title="Global Health Analytics", layout="wide")

LABELS = {
    "life_expectancy": "Expectativa de vida (anos)", "child_mortality": "Mortalidade <5 (/1000)",
    "uhc_index": "Proxy UHC (0-1)", "uhc_sci": "SCI oficial (0-100)",
    "gdp_per_capita": "PIB per capita (US$)", "health_exp_per_capita": "Gasto em saúde per capita (US$)",
    "public_health_exp_gdp": "Gasto público em saúde (% PIB)", "out_of_pocket": "Out-of-pocket (% gasto)",
    "doctors_per_1000": "Médicos /1000", "nurses_per_1000": "Enfermeiros /1000",
    "beds_per_1000": "Leitos /1000", "sanitation_basic": "Saneamento básico (%)",
    "water_basic": "Água básica (%)", "measles_imm_pct": "Vacina sarampo (%)",
    "dpt_imm_pct": "Vacina DTP (%)", "urban_pct": "Urbanização (%)", "fertility": "Fertilidade",
    "maternal_mortality": "Mortalidade materna (/100 mil)", "ncd_mortality_30_70": "Mortalidade DCNT 30-70 (%)",
    "tb_incidence": "Incidência TB (/100 mil)",
}


def label(c: str) -> str:
    return LABELS.get(c, c)


# ------------------------------------------------------------------ dados (cache)
@st.cache_data(show_spinner=False)
def load_abt() -> pd.DataFrame:
    from src.utils.io import load_parquet
    return load_parquet("gold", "abt_country_year")


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | list | None:
    path = settings.REPORTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | list | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_serving():
    from src.serving.predictor import FeatureProvider, load_models
    return FeatureProvider(abt=load_abt(), use_feast=False), load_models()


def missing(msg: str) -> None:
    st.info(f"{msg}", icon="ℹ️")


# ------------------------------------------------------------------ paginas
def page_overview() -> None:
    abt = load_abt()
    q = load_json(str(settings.GOLD_DIR / "abt_quality.json")) or {}
    st.title("Global Health Analytics")
    st.caption("Painel país × ano (1990–2023) — World Bank + WHO GHO · medallion → ABT → "
               "hipóteses, modelos XGBoost, A/B simulado e DiD causal")
    c = st.columns(4)
    c[0].metric("Países", f"{abt.country_id.nunique()}")
    c[1].metric("Anos", f"{abt.year.min()}–{abt.year.max()}")
    sci = q.get("uhc_proxy", {}).get("vs_official_sci", {})
    c[2].metric("Proxy UHC × SCI oficial", f"r = {sci.get('pearson', float('nan')):.2f}",
                help=f"Pearson em {sci.get('n', 0):,} país-anos com os dois")
    tr = q.get("treatment", {})
    c[3].metric("Países tratados (DiD)", tr.get("treated_countries", "–"),
                help=f"cruzaram o limiar {tr.get('threshold')} após 2000; "
                     f"{tr.get('never_treated_countries')} nunca-tratados como controle")

    st.subheader("Principais achados")
    hyp = load_report("hypotheses") or []
    did = load_report("causal_did") or {}
    ab = load_report("ab_simulation") or {}
    bullets = []
    for h in hyp:
        bullets.append(f"**{h['id']}** — {h['title']}: *{h['verdict']}* "
                       f"(efeito {h['effect']:+.2f} {h['effect_unit']})")
    if did:
        dt = did["staggered_detrended"]
        bullets.append(f"**DiD causal** — ingênuo {did['staggered']['att']:+.2f} ano, mas é tendência "
                       f"pré-existente; ajustado: **{dt['att']:+.2f}** (IC95 {dt['ci95'][0]:.2f} a {dt['ci95'][1]:.2f})")
    if ab:
        le = ab["outcomes"]["life_expectancy"]
        bullets.append(f"**A/B simulado** — efeito real {le['true_effect_model']:+.2f} ano vs MDE "
                       f"{le['power']['mde_abs']:.1f} (CUPED {le['power']['mde_abs_cuped']:.1f}): subdimensionado")
    st.markdown("\n".join(f"- {b}" for b in bullets) or "Rode `make all` para gerar os relatórios.")


def page_eda() -> None:
    abt = load_abt()
    st.title("Explorador de dados")
    inds = [c for c in eda.indicator_cols(abt) if c not in ("population", "road_traffic_deaths")]
    f1, f2 = st.columns([2, 1])
    ind = f1.selectbox("Indicador", inds, index=inds.index("life_expectancy"), format_func=label)
    years = sorted(abt.loc[abt[ind].notna(), "year"].unique())
    yr = f2.select_slider("Ano", options=years, value=years[-1])

    d = abt[abt.year == yr].dropna(subset=[ind])
    reverse = ind in ("child_mortality", "maternal_mortality", "ncd_mortality_30_70", "tb_incidence",
                      "out_of_pocket", "fertility")
    scale = style.SEQUENTIAL[::-1] if reverse else style.SEQUENTIAL
    fig = px.choropleth(d, locations="country_id", color=ind, hover_name="country_name",
                        hover_data={"country_id": False, "region": True, "income": True, ind: ":.2f"},
                        color_continuous_scale=scale, labels={ind: label(ind)})
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", landcolor="rgba(128,128,128,0.15)", showland=True)
    fig.update_layout(title=f"{label(ind)} — {yr} ({len(d)} países)", height=460,
                      coloraxis_colorbar=dict(title="", thickness=12))
    if reverse:
        st.caption("Escala invertida: mais escuro = pior.")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Séries por país")
    names = abt.drop_duplicates("country_id").set_index("country_name")["country_id"]
    default = [n for n in ("Brazil", "India", "Nigeria") if n in names.index]
    sel = st.multiselect("Países (até 3 destacados)", names.index.tolist(), default=default, max_selections=3)
    ts = abt[abt.country_id.isin(names[sel])].dropna(subset=[ind])
    world = abt.groupby("year")[ind].median().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=world.year, y=world[ind], name="Mediana mundial",
                             line=dict(color=style.CONTEXT, width=2, dash="dot")))
    for i, n in enumerate(sel):
        s = ts[ts.country_name == n]
        fig.add_trace(go.Scatter(x=s.year, y=s[ind], name=n, mode="lines",
                                 line=dict(color=style.CATEGORICAL[i], width=2)))
    fig.update_layout(hovermode="x unified", yaxis_title=label(ind), height=380)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tendência por renda")
        t = eda.trends(abt, ind)
        fig = px.line(t, x="year", y=ind, color="income", color_discrete_map=style.INCOME_COLORS,
                      category_orders={"income": style.INCOME_ORDER}, labels={ind: label(ind), "year": ""})
        fig.update_layout(hovermode="x unified", height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Média ponderada por população.")
    with c2:
        st.subheader("Cobertura (missingness)")
        miss = eda.missingness_by_year(abt, inds)
        fig = px.imshow(miss, color_continuous_scale=style.SEQUENTIAL, aspect="auto", zmin=0, zmax=100,
                        labels=dict(color="% ausente", x="", y=""))
        fig.update_layout(height=360, coloraxis_colorbar=dict(thickness=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("% de países sem dado por ano; gasto/saneamento/água só existem a partir de 2000.")

    with st.expander("Tabela de cobertura por indicador"):
        st.dataframe(eda.coverage_by_indicator(abt), hide_index=True, use_container_width=True)


def page_hypotheses() -> None:
    st.title("Testes de hipótese")
    hyp = load_report("hypotheses")
    if not hyp:
        return missing("Rode `make hypotheses`.")
    st.caption("Painel: FE país + ano, controle log(PIB), SE cluster por país; efeito por +1 DP. "
               "p-valores corrigidos por Holm (6 hipóteses).")
    tbl = pd.DataFrame([{"ID": h["id"], "Hipótese": h["title"], "Efeito": h["effect"],
                         "Unidade": h["effect_unit"], "IC95": f"[{h['ci95'][0]:.2f}, {h['ci95'][1]:.2f}]",
                         "p (Holm)": h["p_holm"], "Mín. relevante": h["min_relevant_effect"],
                         "Veredito": h["verdict"]} for h in hyp])
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={"Efeito": st.column_config.NumberColumn(format="%.2f"),
                                "p (Holm)": st.column_config.NumberColumn(format="%.2g")})

    for h in hyp:
        with st.expander(f"{h['id']} — {h['title']}"):
            st.markdown(f"**H0:** {h['h0']}  \n**H1:** {h['h1']}  \n**Teste:** {h['test']}  \n"
                        f"**n:** {h['n']:,}")
            rb = h["extra"].get("robustness")
            if rb:
                rows = [{"especificação": "FE país+ano (principal)", "coef": h["effect"],
                         "ic_lo": h["ci95"][0], "ic_hi": h["ci95"][1]}]
                rows += [{"especificação": k.replace("_", " "), "coef": v["coef"], "ic_lo": v["ci95"][0],
                          "ic_hi": v["ci95"][1]} for k, v in rb.items()]
                r = pd.DataFrame(rows)
                fig = go.Figure(go.Scatter(
                    x=r.coef, y=r["especificação"], mode="markers",
                    marker=dict(size=10, color=style.HIGHLIGHT),
                    error_x=dict(type="data", symmetric=False, array=r.ic_hi - r.coef,
                                 arrayminus=r.coef - r.ic_lo, thickness=2, width=0),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
                fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
                fig.update_layout(height=220, xaxis_title=h["effect_unit"], yaxis_title="")
                st.plotly_chart(fig, use_container_width=True, key=f"rb_{h['id']}")
            if h["id"] == "H2":
                st.json({k: v for k, v in h["extra"].items()}, expanded=False)
            if h["id"] == "H5":
                st.write("Ver página **Experimentos** para o event study.")


def _shap_bar(model, X: pd.DataFrame, title: str) -> go.Figure:
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)[0]
    s = pd.Series(sv, index=X.columns)
    s = s.reindex(s.abs().sort_values().index)[-10:]
    colors = [style.DIVERGING[1] if v > 0 else style.DIVERGING[-2] for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=[label(c) for c in s.index], orientation="h",
                           marker=dict(color=colors), hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_layout(title=title, height=360, xaxis_title="contribuição SHAP", yaxis_title="")
    return fig


def page_predict() -> None:
    from src.serving.predictor import predict_all
    st.title("Predição e cenário \"e se\"")
    provider, models = load_serving()
    if len(models) < 3:
        return missing("Modelos ausentes — rode `make train`.")
    cs = provider.countries()
    names = dict(zip(cs.country_name, cs.country_id, strict=True))
    c1, c2 = st.columns([2, 1])
    name = c1.selectbox("País", list(names), index=list(names).index("Brazil") if "Brazil" in names else 0)
    abt = load_abt()
    yrs = sorted(abt.loc[(abt.country_id == names[name]) & abt[FEATURES].notna().any(axis=1), "year"])
    yr = c2.selectbox("Ano das features", yrs[::-1])
    base, fyear, _ = provider.get(names[name], int(yr))

    st.subheader("Ajuste de insumos (cenário)")
    knobs = ["doctors_per_1000", "nurses_per_1000", "health_exp_per_capita", "sanitation_basic",
             "water_basic", "measles_imm_pct", "dpt_imm_pct", "public_health_exp_gdp"]
    cols = st.columns(4)
    feats = dict(base)
    for i, k in enumerate(knobs):
        v = base.get(k)
        if v is None:
            cols[i % 4].caption(f"{label(k)}: sem dado em {fyear}")
            continue
        hi = 100.0 if ("pct" in k or k.endswith("_basic")) else float(max(abt[k].quantile(0.99), v * 2))
        feats[k] = cols[i % 4].slider(label(k), 0.0, hi, float(v), key=f"knob_{k}")
    changed = [k for k in knobs if feats.get(k) != base.get(k)]

    p0, p1 = predict_all(models, base), predict_all(models, feats)
    m = st.columns(3)
    le0, le1 = p0["life_expectancy"]["prediction"], p1["life_expectancy"]["prediction"]
    cm0, cm1 = p0["child_mortality"]["prediction"], p1["child_mortality"]["prediction"]
    ms0, ms1 = p0["milestone_high"]["probability"], p1["milestone_high"]["probability"]
    m[0].metric("Expectativa de vida", f"{le1:.1f} anos", f"{le1 - le0:+.2f}" if changed else None,
                help=f"Intervalo 90%: {p1['life_expectancy']['interval_90']} — cobertura real no teste "
                     f"{p1['life_expectancy']['interval_empirical_coverage_test']:.0%}")
    m[1].metric("Mortalidade <5", f"{cm1:.1f} /1000", f"{cm1 - cm0:+.2f}" if changed else None,
                delta_color="inverse",
                help=f"Intervalo 90%: {p1['child_mortality']['interval_90']}")
    m[2].metric("P(marco alto)", f"{ms1:.0%}", f"{(ms1 - ms0) * 100:+.1f} p.p." if changed else None)
    actual = abt[(abt.country_id == names[name]) & (abt.year == fyear)]
    if len(actual):
        a = actual.iloc[0]
        st.caption(f"Observado em {fyear}: LE {a.life_expectancy:.1f} · mortalidade <5 "
                   f"{a.child_mortality:.1f} · proxy UHC {a.uhc_index:.2f}. "
                   "Predições são associações aprendidas, não efeitos causais (ver Experimentos).")

    c1, c2 = st.columns(2)
    for col, t, title in [(c1, "life_expectancy", "Por que essa expectativa de vida?"),
                          (c2, "child_mortality", "Por que essa mortalidade <5?")]:
        mdl = models[t]
        X = pd.DataFrame([{f: feats.get(f) for f in mdl.features}]).astype(float)
        col.plotly_chart(_shap_bar(mdl.model, X, title), use_container_width=True)
    st.caption("Azul = empurra a predição para cima; vermelho = para baixo.")


def page_experiments() -> None:
    st.title("Experimentos: A/B simulado e DiD causal")
    ab, did = load_report("ab_simulation"), load_report("causal_did")
    if not (ab and did):
        return missing("Rode `make abtest`.")

    st.header("A/B simulado")
    d = ab["design"]
    st.caption(f"Randomização por país (n={d['n_countries']}), snapshot {d['snapshot_year']}, "
               f"uplift de {d['uplift']:.0%} nos insumos UHC; outcome = predição + resíduo de teste; "
               f"CUPED com {d['pre_year']}.")
    rows = []
    for k, r in ab["outcomes"].items():
        rows.append({"Outcome": label(k.replace("_prob", "")) if k != "milestone_prob" else "P(marco alto)",
                     "Efeito real (modelo)": r["true_effect_model"], "Diferença": r["diff"], "p": r["p"],
                     "Diferença CUPED": r["cuped"]["diff"], "p CUPED": r["cuped"]["p"],
                     "Redução de variância": r["cuped"]["variance_reduction"], "p Holm": r["p_holm"],
                     "MDE": r["power"]["mde_abs"], "MDE CUPED": r["power"]["mde_abs_cuped"],
                     "n/braço p/ 80%": r["power"]["n_per_arm_for_80pct_power"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ("Efeito real (modelo)", "Diferença", "Diferença CUPED", "MDE", "MDE CUPED")}
                 | {c: st.column_config.NumberColumn(format="%.2g") for c in ("p", "p CUPED", "p Holm")}
                 | {"Redução de variância": st.column_config.NumberColumn(format="percent")})
    with st.expander("Rodar nova simulação"):
        c1, c2 = st.columns(2)
        up = c1.slider("Uplift nos insumos", 0.05, 1.0, 0.2, 0.05)
        seed = c2.number_input("Seed", value=42, step=1)
        if st.button("Simular"):
            from src.abtesting.simulate_ab import simulate
            with st.spinner("Simulando..."):
                res = simulate(uplift=up, seed=int(seed), abt=load_abt())
            st.dataframe(pd.DataFrame({k: {"efeito_real": v["true_effect_model"], "diff": v["diff"],
                                           "p_cuped": v["cuped"]["p"], "p_holm": v["p_holm"]}
                                       for k, v in res["outcomes"].items()}).T)

    st.header("DiD: cruzar o limiar UHC acelera a expectativa de vida?")
    est = [("TWFE", did["twfe"]), ("TWFE + log PIB", did["twfe_controls"]),
           ("Escalonado (nunca-tratados)", did["staggered"]),
           ("Escalonado (ainda-não-tratados)", did["staggered_notyet"]),
           ("Escalonado + ajuste de tendência", did["staggered_detrended"])]
    fig = go.Figure(go.Scatter(
        x=[e["att"] for _, e in est], y=[n for n, _ in est], mode="markers",
        marker=dict(size=11, color=[style.CONTEXT] * 4 + [style.HIGHLIGHT]),
        error_x=dict(type="data", symmetric=False, thickness=2, width=0,
                     array=[e["ci95"][1] - e["att"] for _, e in est],
                     arrayminus=[e["att"] - e["ci95"][0] for _, e in est]),
        hovertemplate="%{y}: ATT %{x:.2f} anos<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
    fig.update_layout(height=300, xaxis_title="ATT (anos de expectativa de vida), IC95",
                      yaxis=dict(autorange="reversed"), title="Da estimativa ingênua à robusta")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "staggered", "Event study sem ajuste"),
                            (c2, "staggered_detrended", "Event study com ajuste de tendência")]:
        ev = pd.DataFrame(did[key]["event_study"])
        f = go.Figure()
        f.add_trace(go.Scatter(x=list(ev.e) + list(ev.e[::-1]),
                               y=[c[1] for c in ev.ci95] + [c[0] for c in ev.ci95][::-1],
                               fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                               hoverinfo="skip", showlegend=False))
        f.add_trace(go.Scatter(x=ev.e, y=ev.att, mode="lines+markers", line=dict(color=style.HIGHLIGHT, width=2),
                               marker=dict(size=8), name="ATT", showlegend=False,
                               hovertemplate="e=%{x}: %{y:.2f} anos<extra></extra>"))
        f.add_hline(y=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
        f.add_vline(x=-0.5, line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"))
        f.update_layout(title=title, height=340, xaxis_title="anos relativos à adoção", yaxis_title="ATT (anos)")
        col.plotly_chart(f, use_container_width=True)
    st.caption("Sem ajuste, os coeficientes pré-adoção já descem em linha reta: os nunca-tratados (mais pobres) "
               "convergem mais rápido. Removida essa tendência, não há quebra na adoção.")

    st.subheader("Controle sintético (de-meaned)")
    cols = st.columns(max(len(did["synthetic_control"]), 1))
    for col, sc in zip(cols, did["synthetic_control"], strict=False):
        s = sc["series"]
        f = go.Figure()
        f.add_trace(go.Scatter(x=s["years"], y=s["actual"], name="real", line=dict(color=style.HIGHLIGHT, width=2)))
        f.add_trace(go.Scatter(x=s["years"], y=s["synthetic"], name="sintético",
                               line=dict(color=style.CONTEXT, width=2, dash="dash")))
        f.add_vline(x=sc["treat_year"] - 0.5, line=dict(color="rgba(128,128,128,0.6)", width=1))
        f.update_layout(title=f"{sc['name']} — adoção {sc['treat_year']}, placebo p={sc['placebo_p']:.2f}",
                        height=320, hovermode="x unified", yaxis_title="anos")
        col.plotly_chart(f, use_container_width=True)


def page_monitoring() -> None:
    st.title("Monitoramento de drift")
    rep = load_report("drift_report")
    if not rep:
        return missing("Rode `make drift`.")
    alert = rep["alert"]
    st.markdown(f"{'⚠️ **Alerta ativo**' if alert else '✅ **Sem alerta**'} — referência {rep['reference']} "
                f"vs atual {rep['current']} · limiares PSI {rep['thresholds']['psi']}, KS {rep['thresholds']['ks']}")
    dd = pd.DataFrame(rep["data_drift"])
    st.subheader("Drift de valor e de cobertura por feature")
    fig = go.Figure(go.Bar(
        x=dd.psi, y=[label(f) for f in dd.feature], orientation="h",
        marker=dict(color=[style.HIGHLIGHT if d else style.CONTEXT for d in dd.drift]),
        customdata=np.stack([dd.ks, dd.missing_ref, dd.missing_cur], axis=1),
        hovertemplate="%{y}<br>PSI %{x:.3f} · KS %{customdata[0]:.3f}<br>"
                      "missing %{customdata[1]:.0%} → %{customdata[2]:.0%}<extra></extra>"))
    fig.add_vline(x=rep["thresholds"]["psi"], line=dict(color="rgba(128,128,128,0.7)", dash="dash"))
    fig.update_layout(height=480, xaxis_title="PSI (valores observados)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Azul = drift de valor acima do limiar. Mudança de cobertura (séries que só começam em 2000) "
               "é sinal separado na tabela abaixo.")
    st.dataframe(dd, hide_index=True, use_container_width=True)

    perf = rep.get("performance", {})
    if perf:
        st.subheader("Performance por ano (RMSE vs validação)")
        cols = st.columns(len(perf))
        for col, (t, p) in zip(cols, perf.items(), strict=True):
            by = pd.DataFrame(p["by_year"])
            f = go.Figure(go.Bar(x=by.year, y=by.rmse,
                                 marker=dict(color=[style.DIVERGING[-2] if a else style.HIGHLIGHT for a in by.alert]),
                                 hovertemplate="%{x}: RMSE %{y:.2f}<extra></extra>"))
            f.add_hline(y=p["valid_rmse"], line=dict(color="rgba(128,128,128,0.7)", dash="dash"),
                        annotation_text="RMSE validação", annotation_position="top left")
            f.update_layout(title=label(t), height=320, yaxis_title="RMSE", xaxis=dict(dtick=1))
            col.plotly_chart(f, use_container_width=True)
        st.caption("Vermelho = RMSE acima de 1.5× o da validação (alerta).")
    if "serving_drift" in rep:
        st.subheader(f"Drift no serving ({rep['n_serving_predictions']} predições logadas)")
        st.dataframe(pd.DataFrame(rep["serving_drift"]), hide_index=True, use_container_width=True)


PAGES = [
    st.Page(page_overview, title="Visão geral", icon="🌍", default=True),
    st.Page(page_eda, title="Explorador", icon="🔎", url_path="eda"),
    st.Page(page_hypotheses, title="Hipóteses", icon="🧪", url_path="hipoteses"),
    st.Page(page_predict, title="Predição", icon="📈", url_path="predicao"),
    st.Page(page_experiments, title="Experimentos", icon="⚖️", url_path="experimentos"),
    st.Page(page_monitoring, title="Monitoramento", icon="📡", url_path="monitoramento"),
]

if __name__ == "__main__":  # streamlit run executa como __main__; import (testes) nao navega
    st.navigation(PAGES).run()
