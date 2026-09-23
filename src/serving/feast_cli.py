"""CLI do Feast: aplica definicoes, materializa (offline -> online) e consulta."""
from __future__ import annotations

import argparse
from datetime import datetime

from feast import FeatureStore

from config import settings
from src.serving import feast_features


def store() -> FeatureStore:
    """Abre o feature repo em `config/feast`."""
    return FeatureStore(repo_path=str(settings.CONFIG_DIR / "feast"))


def aplicar() -> FeatureStore:
    """Registra entidade, feature views e feature service."""
    st = store()
    entidade, views, servico = feast_features.objetos()
    st.apply([entidade, *views, servico])
    return st


def materializar(inicio: datetime | None = None, fim: datetime | None = None) -> FeatureStore:
    """Copia as features do offline (parquet) para a online store."""
    st = aplicar()
    inicio = inicio or datetime(1990, 1, 1)
    fim = fim or datetime.now()
    st.materialize(start_date=inicio, end_date=fim)
    print(f"[feast] materializado {inicio:%Y-%m-%d} -> {fim:%Y-%m-%d}")
    return st


def online(country_code: str) -> dict:
    """Retorna features online de um pais."""
    st = store()
    return st.get_online_features(
        features=[
            "country_health_yearly:uhc_index",
            "country_health_yearly:life_expectancy",
            "country_health_static:region",
            "country_health_static:income_level",
        ],
        entity_rows=[{"country_code": country_code}],
    ).to_dict()


def main(argv: list[str] | None = None) -> int:
    """CLI: `apply` | `materialize` | `online --country BRA`."""
    parser = argparse.ArgumentParser(description="Feast CLI (projeto global_health)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("apply")
    sub.add_parser("materialize")
    online_cmd = sub.add_parser("online")
    online_cmd.add_argument("--country", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "apply":
        aplicar()
        print("[feast] definicoes aplicadas")
    elif args.cmd == "materialize":
        materializar()
    else:
        print(online(args.country))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
