"""Ingestao do WHO Global Health Observatory (OData API) -> camada bronze (raw).

Fonte secundaria de enriquecimento (mortalidade por causa, materna, TB, transito).
Mesmos principios do worldbank.py: idempotente, resposta bruta, log por indicador.

Saida: data/bronze/who/{code}.json  (payload OData bruto: {"value": [...]})

Uso:
    python -m src.ingestion.who_gho            # todos os indicadores WHO do catalogo
    python -m src.ingestion.who_gho --force
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

from config import settings
from src.ingestion.worldbank import PermanentError, _get_json


def who_indicators(cfg: dict | None = None) -> dict[str, str]:
    """{nome_logico: codigo_gho} do bloco who_gho do catalogo."""
    cfg = cfg or settings.load_indicators()
    return {name: spec["code"] for name, spec in cfg.get("who_gho", {}).get("indicators", {}).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = settings.load_indicators()
    base = cfg["who_gho"]["api_base"]
    settings.ensure_dirs()
    out_dir = settings.BRONZE_DIR / "who"
    log_path = settings.BRONZE_DIR / "ingest_log_who.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    codes = who_indicators(cfg)
    print(f"Ingestando {len(codes)} indicadores WHO GHO -> {out_dir}")
    for name, code in codes.items():
        target = out_dir / f"{code}.json"
        if target.exists() and not args.force:
            print(f"  [cached] {name:24} {code}")
            log[code] = {**log.get(code, {}), "name": name, "status": "cached"}
            continue
        t0 = time.time()
        try:
            data = _get_json(f"{base}/{code}", settings.WB_RETRIES, settings.WB_API_DELAY)
            rows = data.get("value", []) if isinstance(data, dict) else []
            status = "ok" if rows else "empty"
            if rows:
                target.write_text(json.dumps(data))
            res = {"status": status, "rows": len(rows)}
        except (PermanentError, RuntimeError) as e:
            res = {"status": "error", "detail": str(e)[:200]}
        res.update(name=name, seconds=round(time.time() - t0, 1),
                   updated_at=datetime.now(UTC).isoformat())
        log[code] = res
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        print(f"  [{res['status']:6}] {name:24} {code:16} {res.get('rows', 0):6} rows")

    ok = sum(1 for c in codes.values() if log.get(c, {}).get("status") in ("ok", "cached"))
    print(f"\nConcluido: {ok}/{len(codes)} ok. Log: {log_path}")


if __name__ == "__main__":
    main()
