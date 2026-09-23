"""Montagem do dataset de modelagem com split temporal (evita leakage no painel).

Split fixo: treino <= 2015, validacao 2016-2019, teste >= 2020. As features sao os
insumos do sistema de saude + covariados + WHO GHO; alvos, colunas de DiD e
metadados ficam de fora.
"""
from __future__ import annotations

import pandas as pd

from config import settings

TRAIN_END = 2015
VALID_END = 2019

# Colunas que nunca entram como feature (alvos, metadados, DiD, identificadores).
EXCLUDE: set[str] = {
    "country_code", "country_name", "region", "income_level", "lending_type",
    "longitude", "latitude", "year",
    "life_expectancy", "child_mortality", "uhc_index", "uhc_sci", "milestone_high",
    "treated", "treat_year", "post", "always_treated", "never_treated",
}


def features_disponiveis(abt: pd.DataFrame) -> list[str]:
    """Colunas numericas elegiveis como feature."""
    return [c for c in abt.columns if c not in EXCLUDE]


def _rotular_marco(abt: pd.DataFrame) -> pd.DataFrame:
    """Cria o alvo do M3: UHC >= 0.8 e expectativa de vida >= 70."""
    abt = abt.copy()
    abt["milestone_high"] = (
        (abt["uhc_index"] >= 0.8) & (abt["life_expectancy"] >= 70)
    ).astype(int)
    return abt


def split_temporal(dados: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split temporal fixo para evitar leakage: <=2015 / 2016-2019 / >=2020."""
    return {
        "train": dados[dados["year"] <= TRAIN_END],
        "valid": dados[(dados["year"] > TRAIN_END) & (dados["year"] <= VALID_END)],
        "test": dados[dados["year"] > VALID_END],
    }


def build_dataset(target: str, task: str = "regression") -> dict:
    """Devolve os conjuntos train/valid/test para um alvo.

    O alvo `milestone_high` e derivado (classificacao). As features nao incluem
    `uhc_index` nem `life_expectancy` (componentes do rotulo) para evitar leakage.
    """
    abt = pd.read_parquet(settings.GOLD_DIR / "abt_country_year.parquet")
    if target == "milestone_high":
        abt = _rotular_marco(abt)

    candidatas = features_disponiveis(abt)
    dados = abt.dropna(subset=[target])
    dados = dados[dados[candidatas].notna().any(axis=1)]
    partes = split_temporal(dados)
    treino, valid, teste = partes["train"], partes["valid"], partes["test"]

    # Mantem apenas features com algum valor observado no treino (evita imputer vazio).
    feats = [c for c in candidatas if treino[c].notna().any()]

    def _par(df):
        return df[feats].astype(float), df[target].astype(float)

    X_tr, y_tr = _par(treino)
    X_va, y_va = _par(valid)
    X_te, y_te = _par(teste)
    return {
        "target": target,
        "task": task,
        "features": feats,
        "train": (X_tr, y_tr),
        "valid": (X_va, y_va),
        "test": (X_te, y_te),
        "meta": {
            "n_train": int(len(treino)),
            "n_valid": int(len(valid)),
            "n_test": int(len(teste)),
            "train_years": [int(treino["year"].min()), int(treino["year"].max())],
            "valid_years": [int(valid["year"].min()), int(valid["year"].max())],
            "test_years": [int(teste["year"].min()), int(teste["year"].max())],
        },
    }


def resumo_datasets() -> pd.DataFrame:
    """Tabela-resumo dos 3 alvos (tamanhos por split)."""
    linhas = []
    for target, task in (("life_expectancy", "regression"),
                         ("child_mortality", "regression"),
                         ("milestone_high", "classification")):
        ds = build_dataset(target, task)
        linhas.append({"alvo": target, "tarefa": task, "features": len(ds["features"]),
                       **{k: v for k, v in ds["meta"].items() if k.startswith("n_")}})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print(resumo_datasets().to_string(index=False))
