"""Testes da ingestao (World Bank e WHO GHO) - sem acesso a rede."""
from config import settings
from src.ingestion import who_gho, worldbank


def _patch_dirs(tmp_path, monkeypatch):
    """Redireciona os diretorios de dados para um tmp_path isolado."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(settings, "SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr(settings, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "WB_API_DELAY", 0)


def test_deteccao_de_erro_da_api():
    """A API v2 devolve HTTP 200 com corpo de erro; isso deve ser detectado."""
    erro = [{"message": [{"id": "175", "value": "Invalid format"}]}]
    ok = [{"page": 1}, [{"date": "2000"}]]
    assert worldbank._is_api_error(erro)
    assert not worldbank._is_api_error(ok)
    assert "Invalid format" in worldbank._error_message(erro)


def test_fetch_indicator_pagina(monkeypatch):
    """fetch_indicator deve concatenar todas as paginas."""
    monkeypatch.setattr(settings, "WB_API_DELAY", 0)

    def fake_page(code, start, end, page, per_page, retries):
        if page == 1:
            return [{"page": 1, "pages": 2, "total": 2}, [{"date": "2000"}]]
        return [{"page": 2, "pages": 2, "total": 2}, [{"date": "2001"}]]

    monkeypatch.setattr(worldbank, "_request_page", fake_page)
    meta, registros = worldbank.fetch_indicator("SP.DYN.LE00.IN", settings.load_indicators())

    assert meta["pages"] == 2
    assert [r["date"] for r in registros] == ["2000", "2001"]


def test_ingest_worldbank_idempotente(tmp_path, monkeypatch):
    """A segunda passada deve ser toda 'skipped' e o log nao ter erros."""
    _patch_dirs(tmp_path, monkeypatch)

    def fake_fetch(code, cfg):
        return {"page": 1, "pages": 1, "total": 1}, [{"country": {"id": "BRA"}, "value": 1.0}]

    monkeypatch.setattr(worldbank, "fetch_indicator", fake_fetch)

    log, erros = worldbank.ingest_worldbank(with_countries=False)
    assert not erros
    assert all(x["status"] == "ok" for x in log)
    assert (tmp_path / "bronze" / "worldbank" / "SP.DYN.LE00.IN.json").exists()
    assert (tmp_path / "bronze" / "ingest_log.json").exists()

    log2, erros2 = worldbank.ingest_worldbank(with_countries=False)
    assert not erros2
    assert all(x["status"] == "skipped" for x in log2)


def test_who_build_filter():
    """O filtro OData deve incluir anos e, quando houver, a dimensao de sexo."""
    com_sexo = who_gho._build_filter({"sex": "SEX_BTSX"}, 1990, 2023)
    assert "TimeDim ge 1990" in com_sexo
    assert "TimeDim le 2023" in com_sexo
    assert "Dim1 eq 'SEX_BTSX'" in com_sexo

    sem_sexo = who_gho._build_filter({}, 2000, 2023)
    assert "Dim1" not in sem_sexo
