"""silver -> gold: monta a ABT (1 linha por pais x ano) com proxy UHC e WHO GHO."""
from __future__ import annotations

import pandas as pd

from config import settings
from src.features.uhc_index import build_uhc_index
from src.transform import quality

META = ["country_code", "country_name", "region", "income_level", "year"]
UHC = ["uhc_index", "treated", "treat_year", "post", "always_treated", "never_treated"]


def _pivot(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long -> wide por pais x ano."""
    return long.pivot_table(
        index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean"
    ).reset_index()


def build_abt() -> pd.DataFrame:
    """Constroi a ABT combinando World Bank, WHO GHO e o proxy UHC."""
    wb = pd.read_parquet(settings.SILVER_DIR / "worldbank_long.parquet")
    who = pd.read_parquet(settings.SILVER_DIR / "who_long.parquet")
    dim = pd.read_parquet(settings.SILVER_DIR / "countries_dim.parquet")
    uhc = build_uhc_index()

    abt = _pivot(wb).merge(_pivot(who), on=["country_code", "year"], how="left")
    abt = abt.merge(uhc, on=["country_code", "year"], how="left")
    abt = abt.merge(
        dim[["country_code", "country_name", "region", "income_level", "lending_type",
             "longitude", "latitude", "is_aggregate"]],
        on="country_code", how="left",
    )
    abt = abt[~abt["is_aggregate"].fillna(False)].drop(columns=["is_aggregate"])
    abt = abt.sort_values(["country_code", "year"]).reset_index(drop=True)

    indicadores = [c for c in abt.columns if c not in META + UHC]
    return abt[META + sorted(indicadores) + UHC]


def run() -> pd.DataFrame:
    """Executa e grava `data/gold/abt_country_year.parquet` + relatorios de qualidade."""
    settings.ensure_dirs()
    abt = build_abt()
    abt.to_parquet(settings.GOLD_DIR / "abt_country_year.parquet", index=False)
    quality.run(abt)
    print(f"[gold] ABT: {len(abt)} linhas, {abt['country_code'].nunique()} paises, "
          f"{abt['year'].min()}-{abt['year'].max()}")
    return abt


if __name__ == "__main__":
    run()
