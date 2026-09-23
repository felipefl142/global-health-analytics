"""Smoke test de F0: garante que o scaffolding e a configuracao carregam."""
from config import settings


def test_carregamento_dos_indicadores():
    """O catalogo de indicadores deve ter os grupos esperados e codigos validos."""
    cfg = settings.load_indicators()
    for grupo in ("targets", "uhc_inputs", "uhc_official_2019", "covariates"):
        assert grupo in cfg
        assert cfg[grupo], f"grupo vazio: {grupo}"

    codigos = settings.all_indicator_codes(cfg)
    assert codigos["life_expectancy"] == "SP.DYN.LE00.IN"
    assert codigos["child_mortality"] == "SH.DYN.MORT"


def test_intervalo_de_datas():
    """O painel deve cobrir 1990-2023."""
    assert settings.date_range() == (1990, 2023)


def test_criacao_de_diretorios(tmp_path, monkeypatch):
    """ensure_dirs deve criar bronze/silver/gold e a subpasta worldbank."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(settings, "SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr(settings, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")

    settings.ensure_dirs()

    assert (tmp_path / "bronze" / "worldbank").is_dir()
    assert (tmp_path / "silver").is_dir()
    assert (tmp_path / "gold").is_dir()
