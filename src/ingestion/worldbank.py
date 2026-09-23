"""Ingestao do World Bank API v2 -> camada bronze.

Baixa cada indicador do catalogo (`config/indicators.yaml`) no intervalo de anos
configurado, percorre as paginas da API e salva o JSON bruto em
`data/bronze/worldbank/<codigo>.json`. A execucao e idempotente: arquivos
existentes sao pulados, a menos que `--force` seja usado. O status por
indicador e registrado em `data/bronze/ingest_log.json`.
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


def _is_api_error(payload) -> bool:
    """A API v2 devolve HTTP 200 com `[{"message": [...]}]` em codigos invalidos."""
    return (
        isinstance(payload, list)
        and len(payload) >= 1
        and isinstance(payload[0], dict)
        and "message" in payload[0]
    )


def _error_message(payload) -> str:
    """Extrai a mensagem legivel de um payload de erro da API."""
    msgs = payload[0].get("message", [])
    if isinstance(msgs, list):
        return "; ".join(str(m.get("value", m)) for m in msgs)
    return str(msgs)


def _request_page(code: str, start: int, end: int, page: int, per_page: int,
                  retries: int) -> list:
    """Baixa uma pagina de um indicador com retry/backoff exponencial."""
    url = f"{settings.WB_API_BASE}/country/all/indicator/{code}"
    params = {
        "format": "json",
        "per_page": per_page,
        "date": f"{start}:{end}",
        "page": page,
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if _is_api_error(payload):
                raise ValueError(_error_message(payload))
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(settings.WB_API_DELAY * (2 ** attempt))
    raise RuntimeError(f"falha ao baixar {code} (pagina {page}): {last_error}")


def fetch_indicator(code: str, cfg: dict) -> tuple[dict, list]:
    """Baixa todas as paginas de um indicador e retorna (meta, registros brutos)."""
    start, end = settings.date_range(cfg)
    per_page = int(cfg.get("per_page", 10000))
    records: list = []
    meta: dict = {}
    page, pages = 1, 1
    while page <= pages:
        if page > 1:
            time.sleep(settings.WB_API_DELAY)
        payload = _request_page(code, start, end, page, per_page, settings.WB_RETRIES)
        page_meta, rows = payload[0], payload[1] or []
        if page == 1:
            meta = page_meta
        pages = int(page_meta.get("pages", 1))
        records.extend(rows)
        page += 1
    return meta, records


def ingest_worldbank(codes: list[str] | None = None, force: bool = False) -> tuple[list, list]:
    """Executa a ingestao e devolve (log, erros).

    `codes` filtra por codigo WB ou nome logico (ambos aceitos).
    """
    cfg = settings.load_indicators()
    settings.ensure_dirs()
    catalog = settings.all_indicator_codes(cfg)
    if codes:
        wanted = set(codes)
        catalog = {k: v for k, v in catalog.items() if k in wanted or v in wanted}

    out_dir = settings.BRONZE_DIR / "worldbank"
    log: list[dict] = []
    for idx, (name, code) in enumerate(catalog.items()):
        if idx:
            time.sleep(settings.WB_API_DELAY)
        target = out_dir / f"{code}.json"
        if target.exists() and not force:
            log.append({"name": name, "code": code, "status": "skipped", "file": str(target)})
            print(f"[skip] {code} ({name})")
            continue

        started = time.perf_counter()
        try:
            meta, records = fetch_indicator(code, cfg)
            payload = {
                "code": code,
                "name": name,
                "date_range": list(settings.date_range(cfg)),
                "fetched_at": _now(),
                "meta": meta,
                "data": records,
            }
            write_json(target, payload)
            log.append({
                "name": name,
                "code": code,
                "status": "ok",
                "rows": len(records),
                "pages": int((meta or {}).get("pages", 1)),
                "seconds": round(time.perf_counter() - started, 2),
                "file": str(target),
            })
            print(f"[ok]   {code} ({name}): {len(records)} linhas")
        except Exception as exc:  # noqa: BLE001 - registra e segue para o proximo
            log.append({"name": name, "code": code, "status": "error", "error": str(exc)})
            print(f"[erro] {code} ({name}): {exc}")

    write_json(
        settings.BRONZE_DIR / "ingest_log.json",
        {"generated_at": _now(), "source": "worldbank", "indicators": log},
    )
    errors = [item for item in log if item["status"] == "error"]
    return log, errors


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m src.ingestion.worldbank [--codes A,B] [--force]`."""
    parser = argparse.ArgumentParser(description="Ingestao World Bank API v2 -> bronze")
    parser.add_argument("--codes", help="lista separada por virgula (codigo WB ou nome logico)")
    parser.add_argument("--force", action="store_true", help="refaz mesmo se o arquivo existir")
    args = parser.parse_args(argv)

    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None
    log, errors = ingest_worldbank(codes=codes, force=args.force)
    ok = sum(1 for x in log if x["status"] == "ok")
    skipped = sum(1 for x in log if x["status"] == "skipped")
    print(f"\n[ingest] ok={ok} skipped={skipped} erros={len(errors)} "
          f"-> {settings.BRONZE_DIR / 'ingest_log.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
