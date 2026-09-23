"""Ingestao do World Bank API -> camada bronze (raw).

Principios:
- Idempotente e resumivel: se data/bronze/worldbank/{code}.json ja existe e e valido,
  o indicador e pulado (a menos de --force).
- Paginacao explicita: a API rejeita per_page muito grande (HTTP 400 com 50000).
- Retry com backoff exponencial so p/ erros transitorios (rede/5xx/429); 4xx falha rapido.
- Salva a resposta BRUTA (metadata + registros de todas as paginas) p/ preservar proveniencia.
- Metadados de pais (regiao, renda, lat/lon; separa agregados) em _countries.json.
- Loga o resultado de cada indicador em data/bronze/ingest_log.json.

Uso:
    python -m src.ingestion.worldbank                 # todos os indicadores
    python -m src.ingestion.worldbank --force         # refazer tudo
    python -m src.ingestion.worldbank --codes SP.DYN.LE00.IN,SH.DYN.MORT
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from config import settings

COUNTRIES_FILE = "_countries.json"


class PermanentError(Exception):
    """Erro nao-transitorio (4xx, indicador inexistente) - nao vale retry."""


def _get_json(url: str, retries: int, delay: float) -> object:
    """GET com retry/backoff p/ erros transitorios; 4xx (exceto 429) falha na hora."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "global-health-projeto/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise PermanentError(f"HTTP {e.code}: {e.reason}") from e
            last_err = e
        except Exception as e:  # noqa: BLE001  (rede, timeout, JSON truncado)
            last_err = e
        time.sleep(min(2 ** attempt * 2, 30) + delay)
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def _api_error(payload: object) -> str | None:
    """A API responde erros como [{"message": [...]}] com HTTP 200."""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) \
            and "message" in payload[0]:
        return str(payload[0]["message"])[:200]
    return None


def fetch_paged(path: str, params: str, per_page: int,
                retries: int | None = None, delay: float | None = None) -> list:
    """Busca todas as paginas de um endpoint -> [metadata_pag1, registros_concatenados]."""
    retries = settings.WB_RETRIES if retries is None else retries
    delay = settings.WB_API_DELAY if delay is None else delay
    records: list = []
    meta: dict = {}
    page, pages = 1, 1
    while page <= pages:
        url = f"{settings.WB_API_BASE}/{path}?format=json&per_page={per_page}&page={page}{params}"
        data = _get_json(url, retries, delay)
        err = _api_error(data)
        if err:
            raise PermanentError(err)
        if not (isinstance(data, list) and len(data) == 2):
            raise PermanentError(f"shape inesperada: {str(data)[:120]}")
        if page == 1:
            meta = data[0]
            pages = int(meta.get("pages") or 1)
        records.extend(data[1] or [])
        page += 1
        if page <= pages:
            time.sleep(delay)
    return [meta, records]


def fetch_indicator(code: str, start: int, end: int, per_page: int) -> dict:
    """Busca um indicador (todas as paginas). Retorna dict de status (+ payload em 'data')."""
    try:
        data = fetch_paged(f"country/all/indicator/{code}", f"&date={start}:{end}", per_page)
    except (PermanentError, RuntimeError) as e:
        return {"status": "error", "detail": str(e)[:200], "data": None}
    recs = data[1]
    if not recs:
        return {"status": "empty", "rows": 0, "nonnull": 0, "data": data}
    nonnull = sum(1 for x in recs if x.get("value") is not None)
    return {"status": "ok", "rows": len(recs), "nonnull": nonnull, "data": data}


def fetch_countries(out_dir, force: bool = False) -> int:
    """Metadados de paises (regiao/renda/lat/lon). Agregados tem region == 'Aggregates'."""
    target = out_dir / COUNTRIES_FILE
    if target.exists() and not force:
        return len(json.loads(target.read_text())[1])
    data = fetch_paged("country", "", per_page=500)
    target.write_text(json.dumps(data))
    return len(data[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refazer indicadores ja baixados")
    ap.add_argument("--codes", type=str, default="", help="subconjunto de codigos (virgulas)")
    args = ap.parse_args()

    cfg = settings.load_indicators()
    start, end = settings.date_range(cfg)
    per_page = cfg.get("per_page", 10000)
    codes = settings.all_indicator_codes(cfg)

    if args.codes:
        wanted = {c.strip() for c in args.codes.split(",") if c.strip()}
        codes = {k: v for k, v in codes.items() if v in wanted}

    settings.ensure_dirs()
    out_dir = settings.BRONZE_DIR / "worldbank"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.BRONZE_DIR / "ingest_log.json"
    log = {}
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except Exception:  # noqa: BLE001
            log = {}

    n_cty = fetch_countries(out_dir, force=args.force)
    print(f"Metadados de paises: {n_cty} entradas -> {out_dir / COUNTRIES_FILE}")

    print(f"Ingestando {len(codes)} indicadores ({start}-{end}) -> {out_dir}")
    for name, code in codes.items():
        target = out_dir / f"{code}.json"
        if target.exists() and not args.force:
            try:
                existing = json.loads(target.read_text())
                recs = existing[1] if isinstance(existing, list) and len(existing) > 1 else []
                log[code] = {"name": name, "status": "cached", "rows": len(recs),
                             "updated_at": log.get(code, {}).get("updated_at")}
                print(f"  [cached] {name:24} {code:18} rows={len(recs)}")
                continue
            except Exception:  # noqa: BLE001
                pass  # arquivo corrompido, refaz
        t0 = time.time()
        res = fetch_indicator(code, start, end, per_page)
        raw = res.pop("data", None)  # payload bruto (nao entra no log)
        res["name"] = name
        res["seconds"] = round(time.time() - t0, 1)
        res["updated_at"] = datetime.now(UTC).isoformat()
        log[code] = res
        if res["status"] == "ok" and raw is not None:
            target.write_text(json.dumps(raw))  # camada bronze = dado bruto
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        print(f"  [{res['status']:6}] {name:24} {code:18} {res.get('rows', 0):5} rows  "
              f"{res['seconds']}s {res.get('detail', '')}")
        time.sleep(settings.WB_API_DELAY)

    ok = sum(1 for c in codes.values() if log.get(c, {}).get("status") in ("ok", "cached"))
    print(f"\nConcluido: {ok}/{len(codes)} ok. Log: {log_path}")


if __name__ == "__main__":
    main()
