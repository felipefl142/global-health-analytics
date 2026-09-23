"""Dashboard Streamlit: EDA, hipóteses, predição e experimentos.

Rode com `make dashboard` (ou `streamlit run dashboard/app.py`).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from config import settings  # noqa: E402
from src.analysis import eda, viz  # noqa: E402
from src.serving import predictor as pred_mod  # noqa: E402
from src.utils.io import read_json  # noqa: E402

st.set_page_config(page_title="Global Health Analytics", layout="wide")


@st.cache_data
def carregar_abt() -> pd.DataFrame:
    """ABT em cache (gold)."""
    return eda.carregar_abt()


@st.cache_resource
def carregar_preditor() -> pred_mod.Preditor:
    """Carrega os modelos apenas uma vez."""
    return pred_mod.Preditor()


def pagina_eda(abt: pd.DataFrame) -> None:
    """Explorador de séries, missingness e mapa."""
    st.header("EDA — explorador da base")
    indicadores = [c for c in eda.INDICADORES if c in abt.columns]
    col1, col2 = st.columns([1, 2])
    with col1:
        indicador = st.selectbox("Indicador", indicadores)
        ano = st.slider("Ano", int(abt["year"].min()), int(abt["year"].max()), 2023)
    with col2:
        paises = st.multiselect("Países (série)", sorted(abt["country_code"].unique()),
                                default=["BRA", "USA", "IND", "NGA"])
        if paises:
            st.plotly_chart(viz.fig_serie_temporal(abt, indicador, paises=paises),
                            use_container_width=True)

    st.subheader("Mapa")
    mapa = viz.fig_mapa(abt, indicador, ano)
    components.html(mapa._repr_html_(), height=480)

    st.subheader("Missing por grupo de renda")
    st.dataframe(eda.missing_por_renda(abt).round(3))


def pagina_hipoteses() -> None:
    """Tabela consolidada de hipóteses."""
    st.header("Hipóteses — vereditos")
    caminho = settings.REPORTS_DIR / "hypotheses.json"
    if not caminho.exists():
        st.warning("Rode `python -m src.analysis.hypotheses` para gerar o relatório.")
        return
    dados = pd.DataFrame(read_json(caminho))
    st.dataframe(
        dados[["id", "hipotese", "efeito", "ic95", "p", "n", "veredito"]],
        use_container_width=True,
    )


def pagina_predicao(abt: pd.DataFrame) -> None:
    """Predição dos 3 modelos para um país."""
    st.header("Predição — 3 modelos")
    pais = st.selectbox("País", sorted(abt["country_code"].unique()), index=0)
    if st.button("Prever", type="primary"):
        try:
            valores = pred_mod.features_do_feast(pais)
            resultado = carregar_preditor().prever(valores)
            st.json(resultado)
            with st.expander("Features usadas (online store)"):
                st.json(valores)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao prever: {exc}. Rode `make feast-materialize`.")


def pagina_experimentos() -> None:
    """A/B simulado e causal (event study + controle sintético)."""
    st.header("Experimentos — A/B e causal")
    ab_path = settings.REPORTS_DIR / "ab_simulation.json"
    if ab_path.exists():
        ab = read_json(ab_path)
        st.subheader("A/B simulado")
        st.dataframe(pd.DataFrame(ab["metricas"]).T, use_container_width=True)

    did_path = settings.REPORTS_DIR / "causal_did.json"
    if did_path.exists():
        causal = read_json(did_path)
        st.subheader("Event study (UHC → expectativa de vida)")
        st.dataframe(pd.DataFrame(causal["event_study"]["event_study"]),
                     use_container_width=True)
        st.subheader("Controle sintético")
        for s in causal["synthetic_control"]:
            st.markdown(f"**{s['hero']}** — pre-RMSE={s['pre_rmse']:.2f}, ATT={s['att_pos']:+.2f}")
            st.line_chart(pd.DataFrame(s["serie"]).set_index("year"))


def main() -> None:
    """Roteia as páginas do dashboard."""
    st.sidebar.title("Global Health Analytics")
    abt = carregar_abt()
    pagina = st.sidebar.radio("Página", ["EDA", "Hipóteses", "Predição", "Experimentos"])
    if pagina == "EDA":
        pagina_eda(abt)
    elif pagina == "Hipóteses":
        pagina_hipoteses()
    elif pagina == "Predição":
        pagina_predicao(abt)
    else:
        pagina_experimentos()


main()
