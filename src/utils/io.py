"""Utilitarios de leitura/escrita (JSON)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Serializa `obj` em JSON UTF-8, criando os diretorios pais."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """Le um arquivo JSON UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
