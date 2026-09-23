"""Ingestao: paginacao, erros da API e parse dos payloads brutos (sem rede)."""
from __future__ import annotations

import pytest

from src.ingestion import worldbank as wb
from src.transform.bronze_to_silver import parse_countries, parse_who, parse_worldbank


def test_fetch_paged_concatenates_pages(monkeypatch):
    pages = {
        1: [{"page": 1, "pages": 2, "total": 3}, [{"value": 1}, {"value": 2}]],
        2: [{"page": 2, "pages": 2, "total": 3}, [{"value": 3}]],
    }
    calls = []

    def fake_get(url, retries, delay):
        page = int(url.split("&page=")[1].split("&")[0])
        calls.append(page)
        return pages[page]

    monkeypatch.setattr(wb, "_get_json", fake_get)
    monkeypatch.setattr(wb.time, "sleep", lambda s: None)
    meta, recs = wb.fetch_paged("country/all/indicator/X", "", per_page=2)
    assert calls == [1, 2]
    assert [r["value"] for r in recs] == [1, 2, 3]
    assert meta["total"] == 3


def test_fetch_indicator_reports_api_error_message(monkeypatch):
    err = [{"message": [{"id": "175", "value": "The indicator was not found"}]}]
    monkeypatch.setattr(wb, "_get_json", lambda url, retries, delay: err)
    res = wb.fetch_indicator("NOPE", 2000, 2001, per_page=10)
    assert res["status"] == "error"
    assert "not found" in res["detail"]


def test_get_json_does_not_retry_on_4xx(monkeypatch):
    import urllib.error

    attempts = []

    def boom(req, timeout):
        attempts.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(wb.urllib.request, "urlopen", boom)
    with pytest.raises(wb.PermanentError):
        wb._get_json("http://x", retries=4, delay=0)
    assert len(attempts) == 1


def test_parse_countries_drops_aggregates():
    raw = [{}, [
        {"id": "BRA", "iso2Code": "BR", "name": "Brazil",
         "region": {"value": "Latin America & Caribbean "}, "incomeLevel": {"value": "Upper middle income"},
         "latitude": "-15.78", "longitude": "-47.93"},
        {"id": "WLD", "iso2Code": "1W", "name": "World", "region": {"value": "Aggregates"},
         "incomeLevel": {"value": "Aggregates"}, "latitude": "", "longitude": ""},
    ]]
    dim = parse_countries(raw)
    assert dim["country_id"].tolist() == ["BRA"]
    assert dim.loc[0, "region"] == "Latin America & Caribbean"  # strip
    assert dim.loc[0, "latitude"] == pytest.approx(-15.78)


def test_parse_worldbank_uses_iso3():
    raw = [{}, [{"countryiso3code": "BRA", "country": {"id": "BR"}, "date": "2020", "value": 75.1}]]
    df = parse_worldbank("SP.DYN.LE00.IN", "life_expectancy", raw)
    assert df.loc[0, "country_id"] == "BRA"
    assert df.loc[0, "indicator"] == "life_expectancy"


def test_parse_who_keeps_countries_both_sexes():
    raw = {"value": [
        {"SpatialDimType": "COUNTRY", "SpatialDim": "BRA", "TimeDim": 2019, "Dim1": "SEX_BTSX", "NumericValue": 20.0},
        {"SpatialDimType": "COUNTRY", "SpatialDim": "BRA", "TimeDim": 2019, "Dim1": "SEX_MLE", "NumericValue": 25.0},
        {"SpatialDimType": "REGION", "SpatialDim": "AMR", "TimeDim": 2019, "Dim1": "SEX_BTSX", "NumericValue": 1.0},
        {"SpatialDimType": "COUNTRY", "SpatialDim": "ARG", "TimeDim": 2019, "Dim1": None, "NumericValue": 30.0},
    ]}
    df = parse_who("NCDMORT3070", "ncd", raw)
    assert sorted(df["country_id"]) == ["ARG", "BRA"]
    assert df.loc[df.country_id == "BRA", "value"].item() == 20.0


def test_who_segue_nextlink_e_filtra_no_servidor(monkeypatch):
    from src.ingestion import who_gho

    calls = []
    pages = {
        "first": {"value": [{"v": 1}, {"v": 2}], "@odata.nextLink": "http://api/NEXT"},
        "http://api/NEXT": {"value": [{"v": 3}]},
    }

    def fake_get(url, retries, delay):
        calls.append(url)
        return pages["first"] if len(calls) == 1 else pages[url]

    monkeypatch.setattr(who_gho, "_get_json", fake_get)
    out = who_gho.fetch_indicator("http://api", {"code": "X", "sex": "SEX_BTSX"}, 1990, 2023)
    assert [r["v"] for r in out["value"]] == [1, 2, 3]
    assert "TimeDim%20ge%201990" in calls[0] and "SEX_BTSX" in calls[0]
    assert calls[1] == "http://api/NEXT"


def test_ingest_sai_com_erro_se_indicador_falha(monkeypatch, tmp_path):
    from config import settings

    monkeypatch.setattr(settings, "BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr(settings, "ensure_dirs", lambda: None)
    monkeypatch.setattr(wb, "fetch_countries", lambda out_dir, force=False: 0)
    monkeypatch.setattr(wb, "fetch_indicator",
                        lambda code, s, e, pp: {"status": "error", "detail": "boom", "data": None})
    monkeypatch.setattr(wb.time, "sleep", lambda s: None)
    (tmp_path / "bronze" / "worldbank").mkdir(parents=True)
    assert wb.main(["--codes", "life_expectancy"]) == 1   # aceita nome logico
