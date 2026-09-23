"""Construtores reutilizaveis de visualizacoes (notebook 02 e dashboard F10)."""
from __future__ import annotations

import folium
import pandas as pd
import plotly.express as px

GEOJSON_PAISES = (
    "https://raw.githubusercontent.com/python-visualization/folium/main/"
    "examples/data/world-countries.json"
)
TILES = "cartodbpositron"


def fig_serie_temporal(abt: pd.DataFrame, indicador: str, paises: list[str] | None = None,
                       regiao: str | None = None):
    """Linha temporal por pais (se `paises`) ou agregada por regiao."""
    df = abt
    if paises:
        df = df[df["country_code"].isin(paises)]
    if regiao:
        df = df[df["region"] == regiao]
    cor = "country_code" if paises else "region"
    return px.line(df, x="year", y=indicador, color=cor,
                   labels={"year": "ano", "country_code": "pais", "region": "regiao"},
                   title=f"{indicador} ao longo do tempo")


def fig_heatmap_regiao(abt: pd.DataFrame, indicador: str):
    """Heatmap regiao x ano (media do indicador)."""
    piv = abt.pivot_table(index="region", columns="year", values=indicador, aggfunc="mean")
    return px.imshow(piv, aspect="auto", color_continuous_scale="Viridis",
                     labels={"x": "ano", "y": "regiao", "color": indicador},
                     title=f"{indicador} — media por regiao e ano")


def fig_box_renda(abt: pd.DataFrame, indicador: str, ano: int | None = None):
    """Boxplot do indicador por grupo de renda (ano opcional)."""
    df = abt if ano is None else abt[abt["year"] == ano]
    titulo = f"{indicador} por renda" + (f" ({ano})" if ano else "")
    return px.box(df, x="income_level", y=indicador, color="income_level",
                  labels={"income_level": "grupo de renda"}, title=titulo)


def fig_mapa(abt: pd.DataFrame, indicador: str, ano: int, coluna: str = "country_code"):
    """Mapa coropletico (folium) do indicador em um ano.

    Usa um GeoJSON remoto por ISO-3; requer internet para renderizar os tiles.
    """
    df = abt.loc[abt["year"] == ano, [coluna, indicador]].dropna()
    mapa = folium.Map(location=[10, 0], zoom_start=2, tiles=TILES)
    folium.Choropleth(
        geo_data=GEOJSON_PAISES,
        name=indicador,
        data=df,
        columns=[coluna, indicador],
        key_on="feature.id",
        fill_color="YlGnBu",
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name=f"{indicador} ({ano})",
    ).add_to(mapa)
    folium.LayerControl().add_to(mapa)
    return mapa
