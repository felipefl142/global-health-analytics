"""Estilo visual compartilhado (notebooks = matplotlib, dashboard = plotly).

Regras (paleta de referencia validada p/ daltonismo):
- Categorico: ordem FIXA de matizes, nunca ciclada; cor segue a entidade (regiao X tem
  sempre a mesma cor). Scatter/mapa por categoria: no maximo 3 categorias coloridas.
- Grupos de renda sao ORDINAIS -> rampa azul clara->escura (nao categorico).
- Magnitude (mapas, heatmaps de missing): azul sequencial.
- Polaridade (correlacao, efeitos +/-): divergente azul <-> vermelho com meio cinza.
- Marcas finas (linhas 2px), grade/eixos recessivos, texto em tinta neutra.
"""
from __future__ import annotations

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f5c1c0", "#ec8a89",
             "#d03b3b", "#8f1f1f"]
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
INCOME_COLORS = dict(zip(INCOME_ORDER, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], strict=True))
REGION_ORDER = ["East Asia & Pacific", "Europe & Central Asia", "Latin America & Caribbean",
                "Middle East, North Africa, Afghanistan & Pakistan", "North America",
                "South Asia", "Sub-Saharan Africa"]
REGION_COLORS = dict(zip(REGION_ORDER, CATEGORICAL, strict=False))
HIGHLIGHT, CONTEXT = "#2a78d6", "#c3c2b7"   # um destaque vs contexto cinza

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def use_matplotlib() -> None:
    """Aplica o estilo nos graficos matplotlib (notebooks)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "figure.dpi": 110, "savefig.dpi": 110, "figure.figsize": (9, 4.5),
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": INK["grid"],
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "text.color": INK["primary"], "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 5,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "image.cmap": "viridis",
    })


def mpl_sequential():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)


def mpl_diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)


def plotly_template():
    """Template plotly equivalente (dashboard)."""
    import plotly.graph_objects as go

    axis = dict(gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False,
                tickfont=dict(color=INK["secondary"]), title_font=dict(color=INK["secondary"]))
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK["primary"], size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL, xaxis={**axis, "showgrid": False}, yaxis=axis,
        hoverlabel=dict(font_family=FONT), legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=10),
    ))
