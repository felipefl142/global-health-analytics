"""Ingestao do WHO Global Health Observatory (API OData) -> camada bronze.

Fonte secundaria de enriquecimento (DCNT 30-70, mortalidade materna, incidencia
de TB, mortalidade por transito). Salva o JSON bruto por indicador em
`data/bronze/who/<codigo>.json`, de forma idempotente, e registra o status em
`data/bronze/ingest_log_who.json`.
"""
from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

import requests

from config import settings
from src.utils.io import write_json


def _now() -> str:
    """Timestamp UTC em ISO-8601 (segundos)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _build_filter(spec: dict, start: int, end: int) -> str:
    """Monta o `$filter` OData (intervalo de anos + filtro de sexo, se houver)."""
    parts = [f"TimeDim ge {start}", f"TimeDim le {end}"]
    if spec.get("sex"):
        parts.append(f"Dim1 eq '{spec['sex']}'")
    return " and ".join(parts)


def _request(url: str, params: dict | None, retries: int, delay: float) -> dict:
    """GET com retry/backoff exponencial; retorna o corpo JSON."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
    raise RuntimeError(f"falha ao baixar {url}: {last_error}")


def fetch_indicator(code: str, spec: dict, cfg: dict) -> list:
    """Baixa todas as linhas de um indicador do GHO (segue `@odata.nextLink`)."""
    start, end = settings.date_range(cfg)
    url = f"{settings.WHO_API_BASE}/{code}"
    params: dict | None = {"$filter": _build_filter(spec, start, end), "$format": "json"}
    records: list = []
    while url:
        payload = _request(url, params, settings.WB_RETRIES, settings.WB_API_DELAY)
        records.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")
        params = None  # o nextLink ja carrega os parametros
    return records


def ingest_who(codes: list[str] | None = None, force: bool = False) -> tuple[list, list]:
    """Executa a ingestao do GHO e devolve (log, erros)."""
    cfg = settings.load_indicators()
    settings.ensure_dirs()
    catalog = settings.who_indicators(cfg)
    if codes:
        wanted = set(codes)
        catalog = {k: v for k, v in catalog.items() if k in wanted or v["code"] in wanted}

    out_dir = settings.BRONZE_DIR / "who"
    log: list[dict] = []
    for idx, (name, spec) in enumerate(catalog.items()):
        if idx:
            time.sleep(settings.WB_API_DELAY)
        code = spec["code"]
        target = out_dir / f"{code}.json"
        if target.exists() and not force:
            log.append({"name": name, "code": code, "status": "skipped", "file": str(target)})
            print(f"[skip] {code} ({name})")
            continue

        started = time.perf_counter()
        try:
            records = fetch_indicator(code, spec, cfg)
            write_json(target, {
                "code": code,
                "name": name,
                "date_range": list(settings.date_range(cfg)),
                "fetched_at": _now(),
                "rows": len(records),
                "data": records,
            })
            log.append({
                "name": name,
                "code": code,
                "status": "ok",
                "rows": len(records),
                "seconds": round(time.perf_counter() - started, 2),
                "file": str(target),
            })
            print(f"[ok]   {code} ({name}): {len(records)} linhas")
        except Exception as exc:  # noqa: BLE001 - registra e segue para o proximo
            log.append({"name": name, "code": code, "status": "error", "error": str(exc)})
            print(f"[erro] {code} ({name}): {exc}")

    write_json(
        settings.BRONZE_DIR / "ingest_log_who.json",
        {"generated_at": _now(), "source": "who_gho", "indicators": log},
    )
    errors = [item for item in log if item["status"] == "error"]
    return log, errors


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m src.ingestion.who_gho [--codes A,B] [--force]`."""
    parser = argparse.ArgumentParser(description="Ingestao WHO GHO -> bronze")
    parser.add_argument("--codes", help="lista separada por virgula (codigo GHO ou nome logico)")
    parser.add_argument("--force", action="store_true", help="refaz mesmo se o arquivo existir")
    args = parser.parse_args(argv)

    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None
    log, errors = ingest_who(codes=codes, force=args.force)
    ok = sum(1 for x in log if x["status"] == "ok")
    skipped = sum(1 for x in log if x["status"] == "skipped")
    print(f"\n[who] ok={ok} skipped={skipped} erros={len(errors)} "
          f"-> {settings.BRONZE_DIR / 'ingest_log_who.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
