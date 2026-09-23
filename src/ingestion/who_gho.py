"""Ingestao do WHO Global Health Observatory (OData API) -> camada bronze (raw).

Fonte secundaria de enriquecimento (mortalidade por causa, materna, TB, transito).
Mesmos principios do worldbank.py: idempotente, resposta bruta, log por indicador.
- `$filter` OData no servidor: janela de anos do catalogo + sexo (se o indicador tiver `sex`).
- Segue `@odata.nextLink` (a API pode paginar; sem isso a serie seria truncada em silencio).

Saida: data/bronze/who/{code}.json  (payload OData: {"value": [...todas as paginas]})

Uso:
    python -m src.ingestion.who_gho            # todos os indicadores WHO do catalogo
    python -m src.ingestion.who_gho --force --codes NCDMORT3070,tb_incidence

Sai com codigo 1 se algum indicador pedido falhar.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from datetime import UTC, datetime

from config import settings
from src.ingestion.worldbank import PermanentError, _get_json


def who_indicators(cfg: dict | None = None) -> dict[str, str]:
    """{nome_logico: codigo_gho} do bloco who_gho do catalogo."""
    return {name: spec["code"] for name, spec in who_specs(cfg).items()}


def who_specs(cfg: dict | None = None) -> dict[str, dict]:
    """{nome_logico: spec} (code, name, sex opcional) do bloco who_gho."""
    cfg = cfg or settings.load_indicators()
    return cfg.get("who_gho", {}).get("indicators", {})


def odata_filter(spec: dict, start: int, end: int) -> str:
    """Filtro OData: janela de anos + sexo (quando o indicador tem dimensao de sexo)."""
    parts = [f"TimeDim ge {start}", f"TimeDim le {end}"]
    if spec.get("sex"):
        parts.append(f"Dim1 eq '{spec['sex']}'")
    return " and ".join(parts)


def fetch_indicator(base: str, spec: dict, start: int, end: int) -> dict:
    """Baixa todas as paginas de um indicador (segue @odata.nextLink)."""
    query = urllib.parse.urlencode({"$filter": odata_filter(spec, start, end)},
                                   quote_via=urllib.parse.quote)
    url: str | None = f"{base}/{spec['code']}?{query}"
    rows: list = []
    while url:
        data = _get_json(url, settings.WB_RETRIES, settings.WB_API_DELAY)
        if not isinstance(data, dict):
            raise PermanentError(f"shape inesperada: {str(data)[:120]}")
        rows.extend(data.get("value", []))
        url = data.get("@odata.nextLink")  # ja traz os parametros da consulta
    return {"value": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--codes", type=str, default="",
                    help="subconjunto (virgulas): codigo GHO ou nome logico")
    args = ap.parse_args(argv)

    cfg = settings.load_indicators()
    base = cfg["who_gho"]["api_base"]
    start, end = settings.date_range(cfg)
    settings.ensure_dirs()
    out_dir = settings.BRONZE_DIR / "who"
    log_path = settings.BRONZE_DIR / "ingest_log_who.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    specs = who_specs(cfg)
    if args.codes:
        wanted = {c.strip() for c in args.codes.split(",") if c.strip()}
        specs = {k: v for k, v in specs.items() if k in wanted or v["code"] in wanted}
    codes = {name: spec["code"] for name, spec in specs.items()}
    print(f"Ingestando {len(codes)} indicadores WHO GHO -> {out_dir}")
    for name, code in codes.items():
        target = out_dir / f"{code}.json"
        if target.exists() and not args.force:
            print(f"  [cached] {name:24} {code}")
            log[code] = {**log.get(code, {}), "name": name, "status": "cached"}
            continue
        t0 = time.time()
        try:
            data = fetch_indicator(base, specs[name], start, end)
            rows = data["value"]
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
    return 0 if ok == len(codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
