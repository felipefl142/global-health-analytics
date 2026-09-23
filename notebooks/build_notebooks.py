"""Gera e executa os notebooks de analise (F3-F7) a partir de celulas versionadas aqui.

A logica fica em src/ (testada); os notebooks so orquestram, plotam e narram.
Manter o conteudo aqui evita diffs gigantes de JSON e garante que os notebooks rodam
de ponta a ponta sobre a ABT atual.

Uso: python notebooks/build_notebooks.py [--no-exec] [--only 01_eda]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from config import settings
from src.utils.io import load_parquet
from src.analysis import eda
from src.viz import style

style.use_matplotlib()
pd.set_option("display.max_columns", 40, "display.width", 160, "display.precision", 2)
abt = load_parquet("gold", "abt_country_year")
print(f"ABT: {abt.shape[0]:,} linhas x {abt.shape[1]} colunas | "
      f"{abt.country_id.nunique()} paises | {abt.year.min()}-{abt.year.max()}")"""


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


# ------------------------------------------------------------------ 01 EDA
NB01 = [
    md("""
# 01 — EDA: o que a base sabe (e o que não sabe)

Painel **país × ano** (1990–2023) do World Bank + WHO GHO, consolidado na ABT
`gold/abt_country_year`. Objetivo: documentar **cobertura, confiabilidade e padrões de
ausência** antes de qualquer modelo ou teste.
"""),
    code(SETUP),
    md("## 1. Cobertura por indicador\n% de células país × ano preenchidas, nº de países e janela de anos."),
    code("cov = eda.coverage_by_indicator(abt)\ncov"),
    md("""
**Leitura:** demografia (LE, fertilidade, urbanização, população) é completa. Os insumos de
saúde se dividem em dois grupos: **séries de gasto, saneamento e água começam em 2000**
(nada antes), e a força de trabalho (médicos, enfermeiros, leitos) cobre só ~50% das
células — os países não reportam todo ano. `road_traffic_deaths` é um snapshot de 2021 e
não serve como série.
"""),
    md("## 2. Missingness ao longo do tempo"),
    code("""\
cols = cov.loc[cov.pct_filled.between(3, 99.9), "indicator"].tolist()
miss = eda.missingness_by_year(abt, cols)
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(miss.values, aspect="auto", cmap=style.mpl_sequential(), vmin=0, vmax=100)
ax.set_yticks(range(len(miss.index)), miss.index, fontsize=8)
xt = list(range(0, len(miss.columns), 3))
ax.set_xticks(xt, [miss.columns[i] for i in xt], fontsize=8)
ax.grid(False)
ax.set_title("% de países sem dado, por indicador e ano (mais escuro = mais ausente)")
fig.colorbar(im, ax=ax, shrink=0.7, label="% ausente")
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** a borda de 2000 é nítida — é por isso que o **proxy UHC** muda de composição
nessa data e o DiD usa só 2000–2023. Médicos/leitos têm "buracos" espalhados (reporte
esporádico), tratados com interpolação intra-país **só** dentro do cálculo do proxy.
"""),
    md("## 3. O dado falta ao acaso? Missingness por grupo de renda"),
    code("eda.missingness_by_income(abt, cols)"),
    md("""
**Leitura:** não é aleatório. Países de **baixa renda** faltam muito mais em médicos e leitos;
já o SCI oficial falta mais em **alta renda** (microestados e territórios sem estimativa WHO).
Consequência: análises "complete-case" enviesam a amostra — por isso os modelos usam XGBoost
(NaN nativo) e os testes reportam o *n* efetivo.
"""),
    md("## 4. Distribuições"),
    code("eda.describe(abt, ['life_expectancy','child_mortality','gdp_per_capita','health_exp_per_capita','doctors_per_1000','uhc_index','uhc_sci'])"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (c, log) in zip(axes, [("life_expectancy", False), ("child_mortality", True), ("gdp_per_capita", True)]):
    s = abt[c].dropna()
    ax.hist(np.log10(s) if log else s, bins=40, color=style.HIGHLIGHT, edgecolor=style.INK["surface"], linewidth=0.5)
    ax.set_title(c + (" (log10)" if log else ""), fontsize=10)
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** PIB, gasto e mortalidade são fortemente assimétricos (skew alto) → usar **log**
em regressões e testes não-paramétricos em comparações de grupos. A expectativa de vida tem
cauda esquerda de crises (ver §7).
"""),
    md("## 5. Correlações (Spearman, 2019)"),
    code("""\
cc = ['life_expectancy','child_mortality','uhc_index','uhc_sci','gdp_per_capita','health_exp_per_capita',
      'public_health_exp_gdp','out_of_pocket','doctors_per_1000','nurses_per_1000','sanitation_basic',
      'measles_imm_pct','urban_pct','fertility']
corr = eda.correlations(abt, cc, year=2019)
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.values, cmap=style.mpl_diverging(), vmin=-1, vmax=1)
ax.set_xticks(range(len(cc)), cc, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(cc)), cc, fontsize=8); ax.grid(False)
for i in range(len(cc)):
    for j in range(len(cc)):
        ax.text(j, i, f"{corr.values[i,j]:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(corr.values[i,j]) > 0.6 else style.INK["secondary"])
ax.set_title("Correlação de Spearman entre indicadores (2019)")
fig.colorbar(im, ax=ax, shrink=0.7); plt.tight_layout(); plt.show()"""),
    code("""\
print("Proxy UHC vs SCI oficial:", json.loads((settings.GOLD_DIR / "abt_quality.json").read_text())["uhc_proxy"]["vs_official_sci"])"""),
    md("""
**Leitura:** quase tudo se correlaciona com quase tudo em corte transversal — o eixo comum é
**desenvolvimento** (PIB, fertilidade, urbanização). Por isso testes de hipótese precisam de
controle (log PIB) e, de preferência, variação **dentro** do país (efeitos fixos). O proxy UHC
acompanha o SCI oficial de perto (Pearson ≈ 0.91), o que valida a construção.
"""),
    md("## 6. Tendências por renda (média ponderada por população)"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, c in zip(axes, ["life_expectancy", "child_mortality"]):
    t = eda.trends(abt, c)
    for inc in style.INCOME_ORDER:
        s = t[t.income == inc]
        ax.plot(s.year, s[c], color=style.INCOME_COLORS[inc], label=inc)
        ax.annotate(inc.replace(" income", ""), (s.year.iloc[-1], s[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=style.INK["secondary"], va="center")
    ax.axvspan(2019.5, 2021.5, color=style.CONTEXT, alpha=0.25, lw=0)
    ax.set_title(c); ax.set_xlim(1990, 2029)
axes[0].legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** convergência — os países de baixa renda ganharam **~15 anos** de expectativa de
vida desde 1990 (50 → 65), contra ~6 anos dos de alta renda. A mortalidade <5 caiu em
todos os grupos. A faixa cinza marca **2020–2021 (COVID)**: a única queda simultânea em
todos os grupos de renda no período.
"""),
    md("## 7. Choques e outliers: quedas abruptas de expectativa de vida"),
    code("eda.yoy_shocks(abt).head(12)"),
    code("eda.covid_impact(abt)"),
    md("""
**Leitura:** as maiores quedas são **crises reais**, não erro de dado (genocídio de Ruanda 1994,
conflitos na República Centro-Africana, Somália, Bósnia, terremoto no Haiti 2010). Ficam na base,
sinalizadas como aviso em `quality_checks.json`. O COVID tirou ~1.8 ano (mediana) de América
Latina e Oriente Médio/Norte da África entre 2019 e 2021, e a recuperação até 2023 foi completa
na maioria das regiões. Isso importa para os modelos: o **período de teste (2020–2023) é um choque**
fora da distribuição de treino.
"""),
    md("## 8. Comparativo regional (medianas, 2019)"),
    code("eda.regional_comparison(abt, 2019, ['life_expectancy','child_mortality','uhc_index','doctors_per_1000','health_exp_per_capita','sanitation_basic'])"),
    md("""
## Conclusões da EDA

- **Confiável:** demografia e alvos (LE, mortalidade <5) — cobertura quase completa, 217 países.
- **Parcial:** insumos de saúde (só a partir de 2000 p/ gasto/saneamento; força de trabalho esporádica).
- **Missing não aleatório:** depende da renda → usar modelos que lidam com NaN e reportar *n*.
- **Confundimento por desenvolvimento** domina as correlações → testes com controles e efeitos fixos (notebook 03).
- **Choques** (guerras, COVID) são reais e afetam o período de teste dos modelos (notebook 04).
"""),
]

# ------------------------------------------------------------------ 02 VIZ
NB02 = [
    md("""
# 02 — Visualizações

Galeria de gráficos reutilizada no dashboard (versões interativas em plotly lá).
Convenções: renda é **ordinal** (rampa azul clara→escura), regiões usam cores categóricas
fixas com rótulo direto, magnitude usa azul sequencial.
"""),
    code(SETUP),
    md("## 1. Mapa: expectativa de vida mais recente por país"),
    code("""\
snap = eda.latest_snapshot(abt, "life_expectancy").merge(
    load_parquet("silver", "countries_dim")[["country_id", "latitude", "longitude"]], on="country_id")
snap = snap.merge(abt[abt.year == 2023][["country_id", "population"]], on="country_id", how="left")
fig, ax = plt.subplots(figsize=(11, 5.2))
sc = ax.scatter(snap.longitude, snap.latitude, c=snap.life_expectancy, cmap=style.mpl_sequential(),
                s=np.clip(np.sqrt(snap.population.fillna(1e5)) / 60, 8, 400),
                edgecolor=style.INK["surface"], linewidth=0.6)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.grid(False)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(sc, ax=ax, shrink=0.7, label="anos")
ax.set_title("Expectativa de vida (último ano disponível) — tamanho = população")
plt.tight_layout(); plt.show()"""),
    md("O dashboard traz o coroplético interativo (plotly, por ISO3) com seletor de indicador e ano."),
    md("## 2. Evolução por região (mediana)"),
    code("""\
fig, ax = plt.subplots(figsize=(11, 4.8))
med = abt.groupby(["region", "year"]).life_expectancy.median().reset_index()
for reg in style.REGION_ORDER:
    s = med[med.region == reg]
    ax.plot(s.year, s.life_expectancy, color=style.REGION_COLORS[reg])
    ax.annotate(reg.split(",")[0], (s.year.iloc[-1], s.life_expectancy.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=style.INK["secondary"], va="center")
ax.set_xlim(1990, 2031); ax.set_title("Expectativa de vida — mediana por região")
plt.tight_layout(); plt.show()"""),
    md("## 3. Boxplots por grupo de renda (2019)"),
    code("""\
s19 = abt[abt.year == 2019]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, c in zip(axes, ["life_expectancy", "child_mortality", "uhc_index"]):
    data = [s19.loc[s19.income == g, c].dropna() for g in style.INCOME_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=style.INK["primary"]))
    for patch, g in zip(bp["boxes"], style.INCOME_ORDER):
        patch.set_facecolor(style.INCOME_COLORS[g]); patch.set_edgecolor(style.INK["surface"])
    ax.set_xticks(range(1, 5), ["Baixa", "Média-baixa", "Média-alta", "Alta"], fontsize=8)
    ax.set_title(c, fontsize=10)
plt.tight_layout(); plt.show()"""),
    md("## 4. Preston curve: PIB per capita × expectativa de vida"),
    code("""\
fig, ax = plt.subplots(figsize=(9, 5))
for yr, color in [(2000, style.CONTEXT), (2019, style.HIGHLIGHT)]:
    d = abt[abt.year == yr].dropna(subset=["gdp_per_capita", "life_expectancy"])
    ax.scatter(d.gdp_per_capita, d.life_expectancy, s=14, color=color, alpha=0.8, label=str(yr),
               edgecolor=style.INK["surface"], linewidth=0.4)
ax.set_xscale("log"); ax.legend()
ax.set_xlabel("PIB per capita (US$ correntes, log)"); ax.set_ylabel("Expectativa de vida")
ax.set_title("Curva de Preston: a mesma renda compra mais anos de vida em 2019 do que em 2000")
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** a relação é côncava no log do PIB (ganhos grandes na base da distribuição) e a
**curva se desloca para cima** entre 2000 e 2019: a mesma renda compra mais anos de vida
(tecnologia, vacinas, antirretrovirais). É o motivo de os testes terem efeitos fixos de ano.
"""),
    md("## 5. Proxy UHC vs SCI oficial"),
    code("""\
d = abt.dropna(subset=["uhc_index", "uhc_sci"])
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(d.uhc_sci, d.uhc_index, s=6, alpha=0.35, color=style.HIGHLIGHT, lw=0)
ax.axhline(0.5, color=style.INK["muted"], lw=1, ls="--")
ax.annotate("limiar DiD (0.5)", (22, 0.52), fontsize=8, color=style.INK["secondary"])
ax.set_xlabel("SCI oficial (0-100)"); ax.set_ylabel("proxy UHC (0-1)")
ax.set_title(f"Proxy vs SCI oficial (Pearson {d.uhc_index.corr(d.uhc_sci):.2f}, n={len(d):,})")
plt.tight_layout(); plt.show()"""),
    md("## 6. Heatmap: expectativa de vida por região × ano"),
    code("""\
piv = abt.groupby(["region", "year"]).life_expectancy.median().unstack()
fig, ax = plt.subplots(figsize=(12, 3.8))
im = ax.imshow(piv.values, aspect="auto", cmap=style.mpl_sequential())
ax.set_yticks(range(len(piv.index)), [r.split(",")[0] for r in piv.index], fontsize=8)
xt = list(range(0, len(piv.columns), 3)); ax.set_xticks(xt, [piv.columns[i] for i in xt], fontsize=8)
ax.grid(False); fig.colorbar(im, ax=ax, label="anos"); ax.set_title("Mediana da expectativa de vida")
plt.tight_layout(); plt.show()"""),
]

# ------------------------------------------------------------------ 03 HIPOTESES
NB03 = [
    md("""
# 03 — Testes de hipótese

Cada hipótese: H0/H1, teste, **tamanho de efeito + IC95 + p**, correção de **Holm** para
as 6 hipóteses e **significância prática** (efeito mínimo relevante definido antes).
A lógica está em `src/analysis/hypotheses.py` (testada).

Desenho dos testes de painel (H1, H3, H4, H6): efeitos fixos de **país** (só variação dentro
do país) e de **ano** (choques globais), controle log(PIB/capita), SE cluster por país; a
variável de interesse é padronizada (efeito por +1 DP).
"""),
    code(SETUP),
    code("""\
from src.analysis.hypotheses import run_all, to_frame
hs = run_all(abt, n_boot=199)
tbl = to_frame(hs)
tbl"""),
    md("## Robustez: variação entre países vs dentro do país"),
    code("""\
rows = []
for h in hs:
    rb = h.extra.get("robustness")
    if not rb: continue
    rows.append({"id": h.id, "FE país+ano (principal)": h.effect,
                 "pooled (entre+dentro)": rb["pooled_between_within"]["coef"],
                 "FE sem Europa/Ásia Central": rb["fe_sem_europa_asia_central"]["coef"]})
pd.DataFrame(rows).set_index("id").round(2)"""),
    code("""\
fig, ax = plt.subplots(figsize=(9, 3.8))
labels, eff, lo, hi = [], [], [], []
for h in hs:
    if h.id in ("H1", "H3"):
        labels.append(f"{h.id} (anos de LE)")
    elif h.id in ("H4", "H6"):
        labels.append(f"{h.id} (mortes <5 /1000)")
    else:
        continue
    eff.append(h.effect); lo.append(h.ci95[0]); hi.append(h.ci95[1])
y = np.arange(len(labels))
ax.errorbar(eff, y, xerr=[np.array(eff) - lo, np.array(hi) - eff], fmt="o", color=style.HIGHLIGHT,
            ecolor=style.HIGHLIGHT, elinewidth=2, capsize=0, markersize=7)
ax.axvline(0, color=style.INK["muted"], lw=1)
ax.set_yticks(y, labels); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_title("Efeito de +1 DP (FE país+ano) com IC95 — escalas diferentes por hipótese")
plt.tight_layout(); plt.show()"""),
    md("## H2 em detalhe e H5 (event study)"),
    code("""\
h2 = next(h for h in hs if h.id == "H2")
print(json.dumps({k: v for k, v in h2.extra.items()}, indent=1, default=float))"""),
    code("""\
h5 = next(h for h in hs if h.id == "H5")
ev = pd.DataFrame(h5.extra["event_study"])
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(ev.e, [c[0] for c in ev.ci95], [c[1] for c in ev.ci95], color=style.HIGHLIGHT, alpha=0.15, lw=0)
ax.plot(ev.e, ev.att, "o-", color=style.HIGHLIGHT)
ax.axhline(0, color=style.INK["muted"], lw=1); ax.axvline(-0.5, color=style.INK["muted"], lw=1, ls="--")
ax.set_xlabel("anos relativos ao cruzamento do limiar UHC"); ax.set_ylabel("ATT (anos de LE)")
ax.set_title("H5: event study com ajuste de tendência — sem quebra na adoção")
plt.tight_layout(); plt.show()"""),
    code("""\
for h in hs:
    print(f"{h.id}: {h.verdict}  (efeito {h.effect:+.2f} {h.effect_unit}; p_holm={h.p_holm:.2g})")"""),
    md("""
## Leitura consolidada

- **H2, H4, H6 — confirmadas e relevantes.** Saneamento > 80% vem com mortalidade <5 **~43/1000
  menor** (diferença de medianas; Cliff's δ ≈ −0.92). Ajustando por log PIB o efeito cai para
  ~25/1000, mas persiste (ver `adjusted_gdp`) — parte da diferença é desenvolvimento. Urbanização
  e vacinação contra sarampo têm associação **dentro do país** com menor mortalidade <5, robusta à
  exclusão de Europa/Ásia Central.
- **H1 — direção oposta dentro do país.** Entre países, mais médicos ↔ maior LE (pooled positivo).
  Mas, dentro de um mesmo país e descontados choques globais e o PIB, aumentos de densidade médica
  **não** vêm com ganhos de LE (coeficiente negativo, mesmo sem a transição pós-soviética). A
  associação clássica é **entre países** (desenvolvimento), não um efeito marginal de mais médicos.
- **H3 — não rejeitada** após Holm: o gasto público (% PIB) dentro do país não tem efeito detectável
  sobre LE.
- **H5 — não rejeitada.** O DiD ingênuo dá efeito *negativo*, mas o event study mostra uma
  tendência diferencial **pré-existente** (convergência dos países mais pobres). Com ajuste de
  tendência, o ATT é ~0 (IC inclui 0): cruzar o limiar do proxy não acelera a LE de forma
  detectável. Detalhes no notebook 05.
"""),
]

# ------------------------------------------------------------------ 04 MODELAGEM
NB04 = [
    md("""
# 04 — Modelagem (M1, M2, M3)

- **M1** expectativa de vida (regressão), **M2** mortalidade <5 (regressão),
  **M3** marco alto: proxy UHC ≥ 0.8 **e** LE ≥ 70 (classificação).
- Split **temporal** (sem vazamento): treino ≤ 2015, validação 2016–2019 (early stopping),
  teste 2020–2023. Baseline linear (imputação + padronização + Ridge/Logística) vs **XGBoost**.
- M3 não usa `uhc_index` como feature (o label é definido por ele).
"""),
    code(SETUP),
    code("""\
import joblib
from src.modeling.dataset import build_dataset, TARGETS
reps = json.loads((settings.MODELS_DIR / "all_eval.json").read_text())
rows = []
for r in reps:
    for m in ("linear", "xgb"):
        rows.append({"modelo": r["name"], "algoritmo": m, **{f"test_{k}": v for k, v in r[m]["test"].items()}})
pd.DataFrame(rows)"""),
    md("""
**Leitura:** XGBoost supera o baseline linear nos três modelos (M1 R² ≈ 0.85 vs 0.74; M2 ≈ 0.77
vs 0.54). O M3 tem AUC alto também no baseline: o marco é quase uma função determinística dos
insumos (é um problema "fácil" por construção).
"""),
    md("## Resíduos no teste: o choque da COVID"),
    code("""\
md_ = build_dataset("life_expectancy")
m1 = joblib.load(settings.MODELS_DIR / "life_expectancy_reg_xgb.joblib")
te = abt.loc[md_.X_test.index, ["country_id", "year"]].assign(y=md_.y_test, pred=m1.predict(md_.X_test))
te["resid"] = te.y - te.pred
by_year = te.groupby("year").resid.agg(["mean", "median", lambda s: np.sqrt((s**2).mean())]).rename(columns={"<lambda_0>": "rmse"})
by_year.round(2)"""),
    code("""\
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(by_year.index, by_year["mean"], color=style.HIGHLIGHT, width=0.6)
ax.axhline(0, color=style.INK["muted"], lw=1)
ax.set_title("M1: resíduo médio (real − previsto) no teste, por ano"); ax.set_ylabel("anos")
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** só **2021** (pico da pandemia) é superestimado (resíduo negativo) — o modelo não
"sabe" da COVID. Em 2022–2023 acontece o contrário: ele **subestima** cada vez mais (+0.8 e
+1.5 ano), porque a recuperação pós-pandemia e a tendência secular de ganho de LE não estão nas
features (o ano não é feature). O erro cresce com o horizonte de previsão — é o drift que o
monitoramento (F11) precisa capturar, e um argumento para retreinar periodicamente.
"""),
    md("## Interpretação: SHAP"),
    code("""\
for r in reps:
    top = r["importance"]["top"]
    print(f"{r['name']:22} ({r['importance']['method']}): " + ", ".join(f"{k}={v:.2f}" for k, v in list(top.items())[:6]))"""),
    code("""\
import shap
X = md_.X_test
sv = shap.TreeExplainer(m1).shap_values(X)
imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index, imp.values, color=style.HIGHLIGHT, height=0.6)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_title("M1 — |SHAP| médio no teste (anos de LE)"); plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** fertilidade e PIB dominam — os modelos capturam sobretudo o **estágio de
desenvolvimento**. Insumos de saúde (saneamento, vacinas, proxy UHC) entram depois. Como na
EDA, isso é associação preditiva, **não** causal: para efeito causal ver notebook 05.
"""),
]

# ------------------------------------------------------------------ 05 A/B + CAUSAL
NB05 = [
    md("""
# 05 — A/B simulado + inferência causal

Duas perguntas diferentes:

1. **A/B simulado** — *se* randomizássemos uma expansão de insumos UHC entre países, conseguiríamos
   detectar o efeito? (poder, CUPED, correção múltipla)
2. **Causal observacional** — países que cruzaram o limiar do proxy UHC tiveram ganho de LE?
   (DiD escalonado, event study, controle sintético)
"""),
    code(SETUP),
    md("## 1. A/B simulado (unidade = país, snapshot 2019, uplift de 20% nos insumos)"),
    code("""\
from src.abtesting.simulate_ab import simulate
ab = simulate(uplift=0.2, seed=42, abt=abt)
rows = []
for k, r in ab["outcomes"].items():
    rows.append({"outcome": k, "efeito_real_modelo": r["true_effect_model"], "diff": r["diff"], "p": r["p"],
                 "diff_cuped": r["cuped"]["diff"], "p_cuped": r["cuped"]["p"],
                 "reducao_var_cuped": r["cuped"]["variance_reduction"], "p_holm": r["p_holm"],
                 "MDE": r["power"]["mde_abs"], "MDE_cuped": r["power"]["mde_abs_cuped"],
                 "n_por_braco_80pct": r["power"]["n_per_arm_for_80pct_power"]})
pd.DataFrame(rows).round(3)"""),
    code("""\
# poder empirico: repete a randomizacao com varias seeds (efeito fixo, ruido novo)
from src.abtesting.simulate_ab import simulate
sig = {"raw": 0, "cuped": 0}; n_sims = 40
for s in range(n_sims):
    r = simulate(uplift=0.2, seed=1000 + s, abt=abt)["outcomes"]["life_expectancy"]
    sig["raw"] += r["p"] < 0.05; sig["cuped"] += r["cuped"]["p"] < 0.05
print({k: f"{v / n_sims:.0%}" for k, v in sig.items()}, "de poder empirico (LE, alpha=5%)")"""),
    md("""
**Leitura:** com ~190 países (≈96 por braço) o experimento é **subdimensionado**: o MDE para LE
é de ~3.4 anos sem CUPED e ~1.6 com CUPED, enquanto o efeito "verdadeiro" (segundo o M1) de
+20% nos insumos é ~1.2 ano. CUPED (covariável = LE observada em 2015) reduz ~80% da variância
e quadruplica o poder empírico (~15% → ~60% nas simulações) — mas nem assim chega a 80%;
seriam necessários ~790 países por braço sem CUPED. Randomizar no nível país é
estatisticamente caro: precisaria de mais unidades (subnacionais) ou horizonte maior.
"""),
    md("## 2. DiD: da estimativa ingênua à robusta"),
    code("""\
from src.abtesting.causal_did import run
cd = run(abt, n_boot=199)
tab = pd.DataFrame([
    {"estimador": "TWFE (FE país+ano)", **{k: cd["twfe"][k] for k in ("att", "p")}, "ic95": cd["twfe"]["ci95"]},
    {"estimador": "TWFE + log PIB", **{k: cd["twfe_controls"][k] for k in ("att", "p")}, "ic95": cd["twfe_controls"]["ci95"]},
    {"estimador": "Escalonado (nunca-tratados)", **{k: cd["staggered"][k] for k in ("att", "p")}, "ic95": cd["staggered"]["ci95"]},
    {"estimador": "Escalonado (ainda-não-tratados)", **{k: cd["staggered_notyet"][k] for k in ("att", "p")}, "ic95": cd["staggered_notyet"]["ci95"]},
    {"estimador": "Escalonado + ajuste de tendência", **{k: cd["staggered_detrended"][k] for k in ("att", "p")}, "ic95": cd["staggered_detrended"]["ci95"]},
])
tab["ic95"] = tab.ic95.map(lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]")
print("coortes (ano de adocao: n paises):", cd["staggered"]["cohorts"])
tab.round(3)"""),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, key, title in [(axes[0], "staggered", "Sem ajuste: tendência pré-existente"),
                       (axes[1], "staggered_detrended", "Com ajuste de tendência linear")]:
    ev = pd.DataFrame(cd[key]["event_study"])
    ax.fill_between(ev.e, [c[0] for c in ev.ci95], [c[1] for c in ev.ci95], color=style.HIGHLIGHT, alpha=0.15, lw=0)
    ax.plot(ev.e, ev.att, "o-", color=style.HIGHLIGHT)
    ax.axhline(0, color=style.INK["muted"], lw=1); ax.axvline(-0.5, color=style.INK["muted"], lw=1, ls="--")
    ax.set_title(title); ax.set_xlabel("anos relativos à adoção")
axes[0].set_ylabel("ATT (anos de LE)")
plt.tight_layout(); plt.show()"""),
    md("""
**Leitura:** o painel da esquerda é o diagnóstico-chave. Os coeficientes **pré-adoção já descem
em linha reta** — os tratados (renda média) crescem mais devagar que os nunca-tratados (os mais
pobres, em convergência acelerada) *antes* do tratamento. Não há quebra em e = 0: o "efeito
negativo" do TWFE é essa tendência. Removida a tendência (direita), o ATT fica em torno de zero.

Limites: o tratamento é o cruzamento de um índice que sobe continuamente (não um choque de
política discreto), e o grupo nunca-tratado é estruturalmente diferente. É evidência de
**ausência de efeito detectável**, não prova de efeito nulo.
"""),
    md("## 3. Controle sintético (de-meaned) para países \"herói\""),
    code("""\
fig, axes = plt.subplots(1, len(cd["synthetic_control"]), figsize=(12, 4))
for ax, sc in zip(np.atleast_1d(axes), cd["synthetic_control"]):
    s = sc["series"]
    ax.plot(s["years"], s["actual"], color=style.HIGHLIGHT, label="real")
    ax.plot(s["years"], s["synthetic"], color=style.INK["muted"], ls="--", label="sintético")
    ax.axvline(sc["treat_year"] - 0.5, color=style.INK["muted"], lw=1)
    ax.set_title(f"{sc['name']} (adoção {sc['treat_year']}) — placebo p={sc['placebo_p']:.2f}")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
for sc in cd["synthetic_control"]:
    print(sc["name"], "| gap pós médio:", round(sc["att_post_mean"], 2), "| RMSPE pré:", round(sc["rmspe_pre"], 2),
          "| principais doadores:", sc["top_donors"])"""),
    md("""
**Leitura:** para os dois países tratados mais populosos com pré e pós-período longos, o gap
pós-adoção é pequeno e **negativo** (China −0.7, Vietnã −0.3 ano) e não se distingue dos placebos
(p de permutação 0.18 e 0.62): **nenhum efeito positivo detectável**. Consistente com o DiD
ajustado. Ressalva: com 9 anos de pré-período e ~60 doadores, o ajuste da China é quase perfeito
(RMSPE pré ≈ 0) — sinal de sobreajuste, o que torna o gap pós menos informativo.

## Conclusão

A base não sustenta a narrativa "cruzar um limiar de cobertura UHC acelera a expectativa de vida"
— as diferenças que aparecem são explicadas por trajetórias pré-existentes de convergência.
O experimento simulado mostra por quê isso é difícil de medir: mesmo *com* randomização,
~190 países dão pouco poder para efeitos de ~1 ano.
"""),
]

NOTEBOOKS = {
    "01_eda": NB01, "02_visualization": NB02, "03_hypothesis_testing": NB03,
    "04_modeling": NB04, "05_ab_causal": NB05,
}


def build(name: str, cells: list, execute: bool = True) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if execute:
        NotebookClient(nb, timeout=1200, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    path = NB_DIR / f"{name}.ipynb"
    nbf.write(nb, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for name, cells in NOTEBOOKS.items():
        if args.only and args.only != name:
            continue
        print(f"  {name} ...", flush=True)
        print(f"    -> {build(name, cells, execute=not args.no_exec)}")


if __name__ == "__main__":
    main()
