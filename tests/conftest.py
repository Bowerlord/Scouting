"""
conftest.py — Fixtures pytest partagées pour les tests d'intégration

Contrairement aux tests unitaires (test_features.py, test_models.py) qui
travaillent sur des données synthétiques inline, ces fixtures font tourner
le VRAI pipeline (cleaner → feature engineering) sur la fixture committée
tests/fixtures/oracle_sample.csv (générée par generate_fixture.py).

Scope "session" : le pipeline ne tourne qu'une fois pour toute la suite.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.cleaner import run_cleaning_pipeline
from src.data.feature_engineering import (
    add_zscores,
    aggregate_player_stats,
    calculate_kill_participation,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_YEARS = [2024, 2025]


@pytest.fixture(scope="session")
def raw_fixture_df() -> pd.DataFrame:
    """La fixture brute, telle que la lirait load_raw_data."""
    return pd.read_csv(FIXTURES_DIR / "oracle_sample.csv")


@pytest.fixture(scope="session")
def pipeline_tmp_dir(tmp_path_factory, raw_fixture_df) -> Path:
    """Répertoire temporaire contenant la fixture découpée en CSV par année,
    avec le nommage attendu par load_raw_data."""
    tmp = tmp_path_factory.mktemp("raw_data")
    years = pd.to_datetime(raw_fixture_df["date"]).dt.year
    for year in FIXTURE_YEARS:
        raw_fixture_df[years == year].to_csv(
            tmp / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv",
            index=False,
        )
    return tmp


@pytest.fixture(scope="session")
def cleaned_df(pipeline_tmp_dir) -> pd.DataFrame:
    """Sortie du VRAI run_cleaning_pipeline sur la fixture."""
    return run_cleaning_pipeline(
        years=FIXTURE_YEARS,
        raw_dir=pipeline_tmp_dir,
        output_path=pipeline_tmp_dir / "cleaned_matches.csv",
        metadata_path=pipeline_tmp_dir / "refresh_metadata.json",
    )


@pytest.fixture(scope="session")
def features_df(cleaned_df) -> pd.DataFrame:
    """Features joueur/split via les vraies fonctions de feature engineering."""
    df = calculate_kill_participation(cleaned_df.copy())
    df = aggregate_player_stats(df)
    return add_zscores(df)
