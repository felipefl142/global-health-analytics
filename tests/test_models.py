"""Modelagem: split temporal sem vazamento, exclusao de features por alvo, treino XGBoost."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import dataset
from src.modeling.train import _importance, _linear, _xgb


def test_time_split_is_disjoint_and_ordered():
    df = pd.DataFrame({"year": range(1990, 2024)})
    s = dataset.time_split(df)
    assert s["train"]["year"].max() < s["valid"]["year"].min()
    assert s["valid"]["year"].max() < s["test"]["year"].min()
    assert sum(len(v) for v in s.values()) == len(df)


def test_milestone_does_not_use_uhc_index(monkeypatch, abt):
    monkeypatch.setattr(dataset, "load_abt", lambda: abt)
    md = dataset.build_dataset("milestone_high")
    assert "uhc_index" not in md.features
    assert md.kind == "classification"
    md_le = dataset.build_dataset("life_expectancy")
    assert "uhc_index" in md_le.features


def test_models_handle_nan_and_shap():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    X.iloc[::5, 1] = np.nan
    y = X["a"] * 2 + rng.normal(size=300)
    for kind, target in [("regression", y), ("classification", (y > 0).astype(int))]:
        m = _xgb(kind)
        m.fit(X[:200], target[:200], eval_set=[(X[200:250], target[200:250])], verbose=False)
        imp = _importance(m, X[250:], kind, target[250:])
        assert imp["method"] == "shap"
        assert next(iter(imp["top"])) == "a"
        lin = _linear(kind).fit(X[:200], target[:200])  # baseline aceita NaN (imputer)
        assert len(lin.predict(X[250:])) == 50
