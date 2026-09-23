"""Ingestao do World Bank API -> camada bronze (raw).

Principios:
- Idempotente e resumivel: se data/bronze/worldbank/{code}.json ja existe e e valido,
  o indicador e pulado (a menos de --force).
- Retry com backoff exponencial (API publica e instavel).
- Salva a resposta BRUTA (com metadata) para preservar proveniencia.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from src.utils.io import save_parquet  # noqa: F401  (mantem convencao)


def _wb_url(code: str, start: int, end: int, per_page: int) -> str:
    return (
        f"{settings.WB_API_BASE}/country/all/indicator/{code}"
        f"?format=json&date={start}:{end}&per_page={per_page}"
    )


def fetch_indicator(
    code: str, start: int, end: int, per_page: int = 50000,
    retries: int | None = None, delay: float | None = None,
) -> dict:
    """Busca um indicador com retry/backoff. Retorna dict de status."""
    retries = settings.WB_RETRIES if retries is None else retries
    delay = settings.WB_API_DELAY if delay is None else delay
    url = _wb_url(code, start, end, per_page)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "global-health-projeto/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            # resposta valida: [metadata, [records...]]
            if isinstance(data, list) and len(data) == 2 and isinstance(data[1], list):
                recs = data[1]
                if recs and "value" in recs[0]:
                    nonnull = sum(1 for x in recs if x.get("value") is not None)
                    return {"status": "ok", "rows": len(recs), "nonnull": nonnull, "data": data}
                if not recs:
                    return {"status": "empty", "rows": 0, "nonnull": 0, "data": data}
                # lista mas primeiro item e erro
                return {"status": "error", "detail": str(recs[0])[:120], "data": data}
            return {"status": "error", "detail": f"shape inesperada: {str(data)[:120]}", "data": data}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt * 2, 30) + delay)
    return {"status": "error", "detail": last_err, "data": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refazer indicadores ja baixados")
    ap.add_argument("--codes", type=str, default="", help="subconjunto de codigos (virgulas)")
    args = ap.parse_args()

    cfg = settings.load_indicators()
    start, end = settings.date_range(cfg)
    per_page = cfg.get("per_page", 50000)
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
        res["updated_at"] = datetime.now(timezone.utc).isoformat()
        log[code] = res
        if res["status"] == "ok" and raw is not None:
            target.write_text(json.dumps(raw))  # camada bronze = dado bruto
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        print(f"  [{res['status']:6}] {name:24} {code:18} {res.get('rows', 0):5} rows  {res['seconds']}s")
        time.sleep(settings.WB_API_DELAY)

    ok = sum(1 for v in log.values() if v.get("status") in ("ok", "cached"))
    print(f"\nConcluido: {ok}/{len(codes)} ok. Log: {log_path}")


if __name__ == "__main__":
    main()
