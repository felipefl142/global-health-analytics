"""Construcao de datasets de treinamento a partir da ABT (gold).

Split temporal (painel): treina em anos passados, testa em anos futuros,
para nao vazar informacao (forecasting de series temporais de painel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from src.utils.io import load_parquet

# Colunas de feature (insumos UHC + covariados). 'population' e escala, nao prediz taxas.
FEATURES: list[str] = [
    "health_exp_per_capita", "health_exp_gdp", "public_health_exp_gdp", "out_of_pocket",
    "doctors_per_1000", "nurses_per_1000", "beds_per_1000",
    "sanitation_basic", "water_basic", "sanitation_safely", "water_safely",
    "measles_imm_pct", "dpt_imm_pct",
    "gdp_per_capita", "urban_pct", "fertility",
    "uhc_index",
]

TARGETS = {
    "life_expectancy": "M1",
    "child_mortality": "M2",
    "milestone_high": "M3",
}

# Split temporal padrao
TRAIN_END = 2015
VALID_START = 2016
VALID_END = 2019
TEST_START = 2020


@dataclass
class ModelData:
    name: str
    kind: str  # "regression" | "classification"
    target: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    features: list[str]


def load_abt() -> pd.DataFrame:
    return load_parquet("gold", "abt_country_year")


def _clean(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    cols = ["country_id", "year"] + features + [target]
    out = df[cols].dropna(subset=[target]).copy()
    return out


def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["year"] <= TRAIN_END],
        "valid": df[(df["year"] >= VALID_START) & (df["year"] <= VALID_END)],
        "test": df[df["year"] >= TEST_START],
    }


def build_dataset(target: str, features: list[str] | None = None) -> ModelData:
    feats = features or FEATURES
    abt = load_abt()
    clean = _clean(abt, target, feats)
    kind = "classification" if target == "milestone_high" else "regression"
    splits = time_split(clean)
    name = f"{target}_{'clf' if kind == 'classification' else 'reg'}"
    return ModelData(
        name=name, kind=kind, target=target,
        X_train=splits["train"][feats], y_train=splits["train"][target],
        X_valid=splits["valid"][feats], y_valid=splits["valid"][target],
        X_test=splits["test"][feats], y_test=splits["test"][target],
        features=feats,
    )


if __name__ == "__main__":
    for t in TARGETS:
        md = build_dataset(t)
        print(f"{md.name:24} kind={md.kind:14} "
              f"train={len(md.X_train):,} valid={len(md.X_valid):,} test={len(md.X_test):,} "
              f"features={len(md.features)}")
