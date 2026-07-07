"""
test_integration_pipeline.py — Tests d'intégration du vrai pipeline

Là où test_features.py / test_models.py valident la LOGIQUE sur des données
synthétiques inline, cette suite exerce les VRAIES fonctions de src/ de bout
en bout sur la fixture tests/fixtures/oracle_sample.csv :

  schéma → cleaner (target datée incluse) → feature engineering →
  talent scorer (split OOT, LR, calibration, scoring) → clusterer → métadonnées

Les cas de target sont scriptés dans la fixture (voir generate_fixture.py) :
  futurestar   → positif 2024 (débute en LEC en janvier 2025)
  risingstar   → positif 2025 (débute en LEC en décembre 2025)
  lateprospect → négatif (début LEC > 18 mois après ses matchs)
  washedup     → négatif (ex-LEC relégué en ERL)
"""

import json

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.data.schema import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SchemaValidationError,
    validate_raw_schema,
)
from src.models.clusterer import CLUSTER_FEATURES, cluster_position
from src.models.talent_scorer import (
    FEATURE_COLS,
    build_logistic_regression,
    evaluate_model,
    make_out_of_time_split,
    score_all_players,
)
from src.utils.metadata import write_refresh_metadata

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Validation de schéma (src/data/schema.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    def test_fixture_passes_validation(self, raw_fixture_df):
        """La fixture (sans killparticipation, comme les exports réels) passe."""
        validate_raw_schema(raw_fixture_df, source="oracle_sample.csv")

    def test_optional_column_absent_is_tolerated(self, raw_fixture_df):
        """killparticipation est optionnelle : absente de la fixture, pas d'erreur."""
        assert "killparticipation" in OPTIONAL_COLUMNS
        assert "killparticipation" not in raw_fixture_df.columns

    def test_missing_required_column_raises(self, raw_fixture_df):
        with pytest.raises(SchemaValidationError, match="golddiffat15"):
            validate_raw_schema(
                raw_fixture_df.drop(columns=["golddiffat15"]), source="test"
            )

    def test_all_missing_columns_listed_in_one_error(self, raw_fixture_df):
        """L'erreur liste TOUTES les colonnes manquantes, pas juste la première."""
        broken = raw_fixture_df.drop(columns=["dpm", "cspm"])
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_raw_schema(broken, source="test")
        assert "dpm" in str(exc_info.value)
        assert "cspm" in str(exc_info.value)

    def test_non_numeric_column_raises(self, raw_fixture_df):
        broken = raw_fixture_df.copy()
        broken["dpm"] = "pas-un-nombre"
        with pytest.raises(SchemaValidationError, match="dpm"):
            validate_raw_schema(broken, source="test")

    def test_required_columns_derived_from_config(self):
        """REQUIRED_COLUMNS = KEY_COLUMNS moins les optionnelles."""
        from src.config import KEY_COLUMNS

        assert set(REQUIRED_COLUMNS) == set(KEY_COLUMNS) - OPTIONAL_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pipeline de nettoyage complet (src/data/cleaner.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleaningPipelineIntegration:
    def test_output_files_written(self, cleaned_df, pipeline_tmp_dir):
        assert (pipeline_tmp_dir / "cleaned_matches.csv").exists()
        assert (pipeline_tmp_dir / "refresh_metadata.json").exists()

    def test_team_rows_removed(self, cleaned_df):
        assert "team" not in set(cleaned_df["position"].unique())

    def test_names_normalized(self, cleaned_df):
        """'  MessyName ' (fixture) → 'messyname', original conservé."""
        names = set(cleaned_df["playername"].unique())
        assert "messyname" in names
        assert "  MessyName " not in names
        originals = set(cleaned_df["playername_original"].unique())
        assert "  MessyName " in originals or "MessyName" in originals

    def test_no_nan_in_numeric_stats(self, cleaned_df):
        """L'imputation par médiane ligue × année ne laisse aucun NaN."""
        assert cleaned_df["vspm"].isna().sum() == 0

    def test_dated_target_positive_within_horizon(self, cleaned_df):
        """futurestar (LFL 2024, LEC en 2025-01) : positif sur ses matchs ERL 2024."""
        rows = cleaned_df[
            (cleaned_df["playername"] == "futurestar") & (cleaned_df["league"] == "LFL")
        ]
        assert len(rows) > 0
        assert rows["promoted_to_lec"].all()

    def test_dated_target_positive_in_test_year(self, cleaned_df):
        """risingstar (LFL2 2025, LEC en 2025-12) : positif sur ses matchs ERL 2025."""
        rows = cleaned_df[
            (cleaned_df["playername"] == "risingstar")
            & (cleaned_df["league"] == "LFL2")
        ]
        assert len(rows) > 0
        assert rows["promoted_to_lec"].all()

    def test_dated_target_negative_beyond_horizon(self, cleaned_df):
        """lateprospect : début LEC > 18 mois après ses matchs → négatif."""
        rows = cleaned_df[
            (cleaned_df["playername"] == "lateprospect")
            & (cleaned_df["league"] == "LFL")
        ]
        assert len(rows) > 0
        assert not rows["promoted_to_lec"].any()

    def test_dated_target_negative_for_relegated_ex_lec(self, cleaned_df):
        """washedup (LEC 2024 → LFL2 2025) : ex-LEC relégué, jamais positif."""
        rows = cleaned_df[
            (cleaned_df["playername"] == "washedup")
            & (cleaned_df["league"] == "LFL2")
        ]
        assert len(rows) > 0
        assert not rows["promoted_to_lec"].any()

    def test_dated_target_negative_for_never_promoted(self, cleaned_df):
        rows = cleaned_df[cleaned_df["playername"] == "lfl1top"]
        assert len(rows) > 0
        assert not rows["promoted_to_lec"].any()

    def test_refresh_metadata_content(self, pipeline_tmp_dir, cleaned_df):
        with open(pipeline_tmp_dir / "refresh_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["data_years"] == [2024, 2025]
        assert meta["n_rows"] == len(cleaned_df)
        assert meta["data_max_date"].startswith("2025-12")
        assert meta["generated_at"]  # date d'exécution présente


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature engineering (src/data/feature_engineering.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureEngineeringIntegration:
    def test_kill_participation_recomputed_and_bounded(self, features_df):
        assert "killparticipation" in features_df.columns
        assert features_df["killparticipation"].between(0, 1).all()

    def test_kill_participation_formula_on_one_game(self, cleaned_df):
        """Vérification manuelle : KP = (kills + assists) / kills de l'équipe."""
        from src.data.feature_engineering import calculate_kill_participation

        df = calculate_kill_participation(cleaned_df.copy())
        game = df[df["gameid"] == df["gameid"].iloc[0]]
        team = game[game["teamname"] == game["teamname"].iloc[0]]
        team_kills = team["kills"].sum()
        player = team.iloc[0]
        expected = (
            min((player["kills"] + player["assists"]) / team_kills, 1.0)
            if team_kills > 0
            else 0.0
        )
        assert player["killparticipation"] == pytest.approx(expected)

    def test_one_row_per_player_split(self, features_df):
        group_cols = ["playername", "league", "_source_year", "split", "position", "teamname"]
        assert not features_df.duplicated(subset=group_cols).any()

    def test_min_games_filter_applied(self, features_df):
        assert (features_df["games_played"] >= 5).all()

    def test_zscores_centered_within_groups(self, features_df):
        """Moyenne des z-scores ≈ 0 dans chaque groupe ligue/année/split/position."""
        group_means = features_df.groupby(
            ["league", "_source_year", "split", "position"]
        )["dpm_zscore"].mean()
        assert group_means.abs().max() < 1e-6

    def test_promoted_flag_max_aggregated(self, features_df):
        """futurestar : ses lignes joueur/split 2024 restent positives après agrégation."""
        rows = features_df[
            (features_df["playername"] == "futurestar")
            & (features_df["_source_year"] == 2024)
        ]
        assert len(rows) == 2  # Spring + Summer
        assert rows["promoted_to_lec"].all()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Talent scorer (src/models/talent_scorer.py)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def trained_calibrated_model(features_df):
    """Split OOT + LR calibrée entraînés une fois pour les tests de scoring."""
    X_train, y_train, _, _, _, _ = make_out_of_time_split(features_df)
    calibrated = CalibratedClassifierCV(
        clone(build_logistic_regression()), method="sigmoid", cv=2
    )
    calibrated.fit(X_train, y_train)
    return calibrated


class TestTalentScorerIntegration:
    def test_out_of_time_split_isolation(self, features_df):
        _, _, _, _, df_train, df_test = make_out_of_time_split(features_df)
        assert set(df_train["_source_year"].unique()) == {2024}
        assert set(df_test["_source_year"].unique()) == {2025}
        assert set(df_train["league"]).isdisjoint({"LEC"})
        assert set(df_test["league"]).isdisjoint({"LEC"})

    def test_split_has_positives_in_train_and_test(self, features_df):
        _, y_train, _, y_test, _, _ = make_out_of_time_split(features_df)
        assert y_train.sum() >= 2  # futurestar (Spring + Summer 2024)
        assert y_test.sum() >= 2  # risingstar (Spring + Summer 2025)

    def test_evaluate_model_metrics(self, trained_calibrated_model, features_df):
        _, _, X_test, y_test, _, _ = make_out_of_time_split(features_df)
        metrics = evaluate_model("LR calibrée", trained_calibrated_model, X_test, y_test)
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["k"] == int(y_test.sum())

    def test_score_all_players_output(self, trained_calibrated_model, features_df):
        scored = score_all_players(trained_calibrated_model, features_df, FEATURE_COLS)
        assert scored["talent_score"].between(0, 100).all()
        assert scored["score_percentile"].between(0, 100).all()
        # Le meilleur joueur de chaque poste est au percentile 100
        assert (scored.groupby("position")["score_percentile"].max() == 100).all()
        # Seules les lignes ERL sont scorées
        assert "LEC" not in set(scored["league"].unique())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Clusterer (src/models/clusterer.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestClustererIntegration:
    def test_cluster_position_end_to_end(self, features_df):
        df_mid = features_df[
            (features_df["position"] == "mid")
            & (features_df["league"] != "LEC")
        ].copy()
        available = [f for f in CLUSTER_FEATURES if f in df_mid.columns]
        result = cluster_position(df_mid, available, "mid", k_range=range(2, 4))

        assert result["best_k"] in (2, 3)
        assert len(result["labels"]) == len(df_mid)
        assert set(result["labels"]) == set(range(result["best_k"]))
        assert result["embedding"].shape == (len(df_mid), 2)
        assert len(result["profiles"]) == result["best_k"]
        assert all("archetype" in p for p in result["profiles"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Métadonnées de fraîcheur (src/utils/metadata.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefreshMetadata:
    def test_write_refresh_metadata(self, tmp_path):
        df = pd.DataFrame(
            {
                "date": ["2025-06-01 17:00:00", "2025-07-03 18:00:00"],
                "playername": ["a", "b"],
                "_source_year": [2025, 2025],
            }
        )
        out = tmp_path / "refresh_metadata.json"
        meta = write_refresh_metadata(df, output_path=out)
        assert out.exists()
        assert meta["data_max_date"] == "2025-07-03"
        assert meta["data_years"] == [2025]
        assert meta["n_rows"] == 2
        assert meta["n_players"] == 2
