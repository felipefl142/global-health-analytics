"""Catalogo de indicadores e configuracao (F0)."""
from __future__ import annotations

from config import settings
from src.ingestion.who_gho import who_specs


def test_catalogo_tem_grupos_e_codigos_chave():
    cfg = settings.load_indicators()
    for group in ("targets", "uhc_inputs", "uhc_official", "covariates", "who_gho"):
        assert cfg.get(group), f"grupo vazio/ausente: {group}"
    codes = settings.all_indicator_codes(cfg)
    assert codes["life_expectancy"] == "SP.DYN.LE00.IN"
    assert codes["child_mortality"] == "SH.DYN.MORT"
    assert codes["out_of_pocket"] == "SH.XPD.OOPC.CH.ZS"   # .TO.ZS foi removido da API
    assert len(set(codes.values())) == len(codes), "codigo WB duplicado no catalogo"
    assert who_specs(cfg)["ncd_mortality_30_70"]["sex"] == "SEX_BTSX"


def test_paginacao_segura_e_janela():
    cfg = settings.load_indicators()
    assert cfg["per_page"] <= 20000        # 50000 -> HTTP 400 na API
    assert settings.date_range(cfg) == (1990, 2023)
