"""Dashboard: cada pagina renderiza sem excecao sobre os artefatos reais (pula sem dados)."""
from __future__ import annotations

import pytest

from config import settings

pytestmark = pytest.mark.skipif(
    not (settings.GOLD_DIR / "abt_country_year.parquet").exists()
    or not (settings.MODELS_DIR / "all_eval.json").exists(),
    reason="requer ABT gold e modelos (make gold train)")

PAGES = ["page_overview", "page_eda", "page_hypotheses", "page_predict", "page_experiments",
         "page_monitoring"]


def _render(page: str) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    root = Path.cwd()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("dash_app", root / "dashboard" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    getattr(mod, page)()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_render, args=(page,), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title
