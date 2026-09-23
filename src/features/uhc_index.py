"""Proxy de cobertura UHC (service coverage) por pais x ano.

O SCI oficial (SH_UHC_SCI) so cobre 2000-2023. Para o painel 1990-2023 construimos
um proxy a partir de insumos de saude:

1. Insumos estruturais (medicos, enfermeiros, leitos, saneamento, agua, gasto) mudam
   devagar e sao reportados de forma esparsa -> interpolacao linear dentro do pais,
   so entre observacoes (sem extrapolar alem de MAX_EDGE_FILL anos nas pontas).
   A interpolacao so e usada no calculo do proxy; as colunas da ABT ficam cruas.
2. Gasto per capita (US$ correntes) e muito assimetrico -> log antes de normalizar.
3. Normalizacao robusta por percentis 5/95 do painel inteiro -> [0, 1].
4. Media ponderada sobre os componentes disponiveis; exige >= MIN_WEIGHT do peso total,
   senao NaN.

Uso:
    from src.features.uhc_index import build_uhc_index
    df = build_uhc_index(wide_df)   # adiciona 'uhc_index' (0..1) e 'uhc_weight_avail'
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Componentes do proxy (colunas da ABT wide) e pesos (soma = 1.0)
DEFAULT_COMPONENTS: dict[str, float] = {
    "doctors_per_1000": 0.15,
    "nurses_per_1000": 0.15,
    "beds_per_1000": 0.10,
    "health_exp_per_capita": 0.20,
    "sanitation_basic": 0.10,
    "water_basic": 0.10,
    "measles_imm_pct": 0.10,
    "dpt_imm_pct": 0.10,
}
LOG_COMPONENTS = {"health_exp_per_capita"}
# Vacinacao varia ano a ano (campanhas, crises) -> nao interpola
INTERPOLATE = {"doctors_per_1000", "nurses_per_1000", "beds_per_1000",
               "health_exp_per_capita", "sanitation_basic", "water_basic"}
MAX_EDGE_FILL = 2   # anos de ffill/bfill permitidos alem da ultima/primeira observacao
MIN_WEIGHT = 0.5    # fracao minima do peso total disponivel p/ calcular o indice
LO, HI = 0.05, 0.95  # percentis p/ normalizacao robusta
DID_THRESHOLD = 0.5
# Gasto/saneamento/agua so existem a partir de 2000: antes disso a composicao do indice
# muda e gera "cruzamentos" artificiais. O timing do tratamento so considera >= 2000.
DID_START_YEAR = 2000


def _robust_minmax(s: pd.Series, lo: float = LO, hi: float = HI) -> pd.Series:
    p_lo, p_hi = np.nanpercentile(s, [lo * 100, hi * 100])
    denom = p_hi - p_lo
    if not np.isfinite(denom) or denom == 0:
        return pd.Series(np.nan, index=s.index)
    return ((s - p_lo) / denom).clip(0, 1)


def _interpolate_within_country(wide: pd.DataFrame, col: str) -> pd.Series:
    """Interpolacao linear por pais (ordenado por ano) + ffill/bfill curto nas pontas."""
    def f(s: pd.Series) -> pd.Series:
        s = s.interpolate(method="linear", limit_area="inside")
        return s.ffill(limit=MAX_EDGE_FILL).bfill(limit=MAX_EDGE_FILL)
    ordered = wide.sort_values(["country_id", "year"])
    out = ordered.groupby("country_id")[col].transform(f)
    return out.reindex(wide.index)


def build_uhc_index(
    wide: pd.DataFrame,
    components: dict[str, float] | None = None,
    col: str = "uhc_index",
) -> pd.DataFrame:
    """Adiciona o proxy UHC (0..1) e a fracao de peso disponivel ao DataFrame wide."""
    requested = components or DEFAULT_COMPONENTS
    comps = {k: v for k, v in requested.items() if k in wide.columns}
    if not comps:
        raise ValueError("nenhum componente do proxy UHC encontrado na ABT wide")
    total_w = sum(requested.values())  # peso total inclui componentes ausentes da ABT
    num = pd.Series(0.0, index=wide.index)
    w_avail = pd.Series(0.0, index=wide.index)
    for name, w in comps.items():
        s = pd.to_numeric(wide[name], errors="coerce")
        if name in INTERPOLATE:
            s = _interpolate_within_country(wide.assign(**{name: s}), name)
        if name in LOG_COMPONENTS:
            s = np.log1p(s.clip(lower=0))
        norm = _robust_minmax(s)
        has = norm.notna()
        num = num + norm.fillna(0) * w
        w_avail = w_avail + has * w
    frac = w_avail / total_w
    out = wide.copy()
    out[col] = np.where(frac >= MIN_WEIGHT, num / w_avail.replace(0, np.nan), np.nan)
    out["uhc_weight_avail"] = frac.round(3)
    return out


def treatment_timing(abt: pd.DataFrame, threshold: float = DID_THRESHOLD,
                     col: str = "uhc_index", start_year: int = DID_START_YEAR) -> pd.DataFrame:
    """Ano em que cada pais cruza o limiar do proxy UHC (p/ DiD escalonado).

    - So anos >= start_year (composicao do indice estavel).
    - Cruzamento sustentado: acima do limiar no ano t E no proximo ano observado
      (evita tratar oscilacao de vacinacao como tratamento).
    - So conta como tratado quem foi observado ABAIXO do limiar antes de cruzar
      (tem periodo pre); quem ja comeca acima e 'always_treated' e fica fora do
      grupo de tratamento (nao identifica o efeito no DiD).

    Colunas adicionadas:
      treat_year (float) - primeiro ano >= limiar apos periodo pre (ou NaN)
      treated (bool)     - tem treat_year
      always_treated (bool)
      post (int)         - 1 se treated e ano >= treat_year
    """
    out = abt.copy()
    treat_year: dict = {}
    always: dict = {}
    window = out[out["year"] >= start_year].dropna(subset=[col])
    for cid, g in window.groupby("country_id"):
        g = g.sort_values("year")
        above = g[col] >= threshold
        sustained = above & above.shift(-1, fill_value=True)
        if above.iloc[0]:
            always[cid] = True
        elif sustained.any():
            treat_year[cid] = g.loc[sustained, "year"].min()
    out["treat_year"] = out["country_id"].map(treat_year).astype(float)
    out["treated"] = out["treat_year"].notna()
    out["always_treated"] = out["country_id"].isin(always)
    out["post"] = np.where(out["treated"] & (out["year"] >= out["treat_year"]), 1, 0)
    return out


def validate_against_sci(abt: pd.DataFrame, col: str = "uhc_index") -> dict:
    """Correlacao do proxy com o SCI oficial (0-100) nas linhas com ambos."""
    both = abt.dropna(subset=[col, "uhc_sci"])
    if len(both) < 10:
        return {"n": int(len(both))}
    return {
        "n": int(len(both)),
        "pearson": round(float(both[col].corr(both["uhc_sci"])), 3),
        "spearman": round(float(both[col].corr(both["uhc_sci"], method="spearman")), 3),
    }


if __name__ == "__main__":
    from src.utils.io import load_parquet
    abt = load_parquet("gold", "abt_country_year")
    print(abt[["country_id", "year", "uhc_index", "uhc_weight_avail", "treat_year", "post"]]
          .dropna(subset=["uhc_index"]).head(10))
    print("\nvalidacao vs SCI oficial:", validate_against_sci(abt))
    print(f"tratados={abt.loc[abt['treated'], 'country_id'].nunique()} "
          f"always_treated={abt.loc[abt['always_treated'], 'country_id'].nunique()}")
