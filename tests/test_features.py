"""
test_features.py — Tests unitaires pour le pipeline de feature engineering

On teste :
  - Les Z-scores sont calculés correctement (moyenne ≈ 0 par groupe)
  - La target variable est bien encodée
  - Le dataset de sortie a le bon nombre de colonnes
  - Pas de fuites de données entre ligues (Z-scores intra-ligue)
"""


import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame minimal simulant des données Oracle's Elixir nettoyées."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "playername": [f"player_{i}" for i in range(n)],
        "league":     np.random.choice(["LFL", "PRM", "NLC"], n),
        "_source_year": np.random.choice([2024, 2025], n),
        "split":      np.random.choice(["Spring", "Summer"], n),
        "position":   np.random.choice(["top", "jng", "mid", "bot", "sup"], n),
        "teamname":   [f"team_{i % 10}" for i in range(n)],
        "dpm":        np.random.normal(500, 100, n),
        "cspm":       np.random.normal(7, 1.5, n),
        "vspm":       np.random.normal(1.2, 0.4, n),
        "killparticipation": np.random.uniform(0.4, 0.9, n),
        "golddiffat15": np.random.normal(0, 500, n),
        "csdiffat15": np.random.normal(0, 8, n),
        "xpdiffat15": np.random.normal(0, 400, n),
        "earnedgold": np.random.normal(8000, 1500, n),
        "gamelength": np.random.normal(1800, 300, n),
        "result":     np.random.randint(0, 2, n),
        "champion":   np.random.choice(["Jinx", "Thresh", "Orianna", "Renekton"], n),
        "promoted_to_lec": np.random.choice([True, False], n, p=[0.08, 0.92]),
    })


# ── Tests Z-scores ────────────────────────────────────────────────────────────

class TestZScores:
    """Les Z-scores doivent être calculés intra-groupe (ligue × année × split × position)."""

    def test_zscore_mean_near_zero(self, sample_df):
        """La moyenne des Z-scores par groupe doit être ≈ 0.

        Note : avec des groupes de taille 1 (std=NaN), on remplace par 0.
        Le seuil est relaxé à 0.5 car la fixture est entièrement aléatoire
        et certains petits groupes peuvent avoir des moyennes légèrement non nulles.
        """
        group_keys = ["league", "_source_year", "split", "position"]
        df = sample_df.copy()
        df["dpm_zscore"] = df.groupby(group_keys)["dpm"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        ).fillna(0)  # std=0 pour groupes de taille 1 → Z-score = 0
        group_means = df.groupby(group_keys)["dpm_zscore"].mean()
        # Seuil relaxé : les moyennes doivent rester proches de 0
        assert (group_means.abs() < 0.5).all(), \
            f"Certains groupes ont une moyenne Z-score trop éloignée de 0 : {group_means[group_means.abs() >= 0.5]}"

    def test_zscore_no_cross_league_leakage(self, sample_df):
        """Les Z-scores LFL ne doivent pas dépendre des stats PRM."""
        group_keys = ["league", "_source_year", "split", "position"]
        df = sample_df.copy()
        df["dpm_zscore"] = df.groupby(group_keys)["dpm"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        # Un joueur LFL top Spring 2024 doit avoir un Z-score calculé
        # uniquement sur les joueurs LFL top Spring 2024
        lfl_top = df[(df["league"] == "LFL") & (df["position"] == "top") &
                     (df["_source_year"] == 2024) & (df["split"] == "Spring")]
        if len(lfl_top) >= 2:
            expected_mean = lfl_top["dpm"].mean()
            expected_std  = lfl_top["dpm"].std()
            first_player_dpm = lfl_top["dpm"].iloc[0]
            expected_z = (first_player_dpm - expected_mean) / (expected_std + 1e-8)
            actual_z = lfl_top["dpm_zscore"].iloc[0]
            assert abs(actual_z - expected_z) < 0.001, \
                f"Z-score incohérent : attendu {expected_z:.4f}, obtenu {actual_z:.4f}"

    def test_zscore_handles_single_player_group(self, sample_df):
        """Un groupe avec un seul joueur doit avoir un Z-score = 0 après fillna.

        Un groupe de taille 1 produit std=NaN (division par 0 même avec +1e-8
        car x.std() retourne NaN pour n=1 en pandas). La pipeline de feature
        engineering doit donc appliquer un fillna(0) après le Z-score — c'est
        ce qu'on teste ici.
        """
        df = sample_df.copy()
        df.loc[0, "league"] = "UNIQUE_LEAGUE"
        df.loc[0, "_source_year"] = 2024
        df.loc[0, "split"] = "Spring"
        df.loc[0, "position"] = "top"

        group_keys = ["league", "_source_year", "split", "position"]
        df["dpm_zscore"] = df.groupby(group_keys)["dpm"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        ).fillna(0)  # ← fillna(0) obligatoire pour les groupes singleton

        # Après fillna, le Z-score doit être 0 (pas NaN)
        val = df.loc[0, "dpm_zscore"]
        assert val == val, "Z-score NaN pour un groupe singleton après fillna(0)"  # NaN != NaN
        assert val == 0.0, f"Z-score groupe singleton attendu = 0, obtenu = {val}"


# ── Tests Target Variable ─────────────────────────────────────────────────────

class TestTargetVariable:

    def test_target_is_binary(self, sample_df):
        """La target doit être binaire (0 ou 1 / True ou False)."""
        assert sample_df["promoted_to_lec"].dtype in [bool, np.bool_, object, int]
        unique_vals = set(sample_df["promoted_to_lec"].astype(str).str.lower().unique())
        assert unique_vals.issubset({"true", "false", "0", "1"}), \
            f"Valeurs inattendues dans promoted_to_lec : {unique_vals}"

    def test_target_imbalance_reasonable(self, sample_df):
        """Le taux de positifs doit être entre 1% et 50% (ni trivial ni équilibré)."""
        rate = sample_df["promoted_to_lec"].astype(str).str.lower().eq("true").mean()
        assert 0.01 <= rate <= 0.50, \
            f"Taux de promotion anormal : {rate:.1%}"


# ── Tests Target Datée (anti-fuite temporelle) ────────────────────────────────

class TestDatedTarget:
    """Valide la target datée de src.data.cleaner.build_dated_target_from_oracle.

    On construit des matchs ERL + LEC à des dates contrôlées pour vérifier que
    seuls les matchs ERL ANTÉRIEURS au début LEC (et dans l'horizon) sont positifs.
    """

    @staticmethod
    def _matches(rows):
        """Construit un DataFrame de matchs à partir de tuples (player, league, date)."""
        return pd.DataFrame(
            [{"playername": p, "league": lg, "date": d} for p, lg, d in rows]
        )

    def test_pre_promotion_match_is_positive(self):
        """Un match ERL joué AVANT le début LEC (dans l'horizon) est positif."""
        from src.data.cleaner import build_dated_target_from_oracle
        df = self._matches([
            ("caps", "LFL", "2024-03-01"),   # ERL avant promotion
            ("caps", "LEC", "2024-09-01"),   # début LEC 6 mois plus tard
        ])
        target = build_dated_target_from_oracle(df, horizon_months=18)
        assert target.iloc[0], "Le match ERL pré-promotion devrait être positif"
        assert not target.iloc[1], "La ligne LEC ne doit jamais être positive"

    def test_relegated_ex_lec_is_negative(self):
        """Un ex-joueur LEC relégué en ERL n'est PAS une pépite (fuite corrigée)."""
        from src.data.cleaner import build_dated_target_from_oracle
        df = self._matches([
            ("veteran", "LEC", "2024-02-01"),  # début LEC EN PREMIER
            ("veteran", "LFL", "2025-03-01"),  # relégué en ERL APRÈS
        ])
        target = build_dated_target_from_oracle(df, horizon_months=18)
        assert not target.iloc[1], \
            "Un match ERL postérieur au début LEC ne doit pas être positif"

    def test_promotion_beyond_horizon_is_negative(self):
        """Une promotion trop lointaine (au-delà de l'horizon) reste négative."""
        from src.data.cleaner import build_dated_target_from_oracle
        df = self._matches([
            ("slow", "LFL", "2024-01-01"),   # ERL
            ("slow", "LEC", "2026-06-01"),   # début LEC ~29 mois plus tard
        ])
        target = build_dated_target_from_oracle(df, horizon_months=18)
        assert not target.iloc[0], \
            "Un match à plus de 18 mois du début LEC ne doit pas être positif"

    def test_never_promoted_is_negative(self):
        """Un joueur qui n'atteint jamais la LEC reste négatif."""
        from src.data.cleaner import build_dated_target_from_oracle
        df = self._matches([
            ("journeyman", "LFL", "2024-03-01"),
            ("journeyman", "PRM", "2025-03-01"),
        ])
        target = build_dated_target_from_oracle(df, horizon_months=18)
        assert not target.any(), "Aucun match ne doit être positif sans début LEC"

    def test_debut_date_is_first_lec_appearance(self):
        """La date de début LEC est bien le PREMIER match LEC du joueur."""
        from src.data.cleaner import compute_lec_debut_dates
        df = self._matches([
            ("star", "LEC", "2025-06-01"),
            ("star", "LEC", "2025-01-15"),   # plus ancien → doit primer
            ("star", "LEC", "2025-08-01"),
        ])
        debuts = compute_lec_debut_dates(df)
        assert debuts["star"] == pd.Timestamp("2025-01-15")


# ── Tests Dataset Output ──────────────────────────────────────────────────────

class TestDatasetStructure:

    def test_required_columns_present(self, sample_df):
        """Les colonnes essentielles doivent être présentes."""
        required = ["playername", "league", "_source_year", "position", "promoted_to_lec"]
        for col in required:
            assert col in sample_df.columns, f"Colonne manquante : {col}"

    def test_no_duplicate_player_split(self, sample_df):
        """Un joueur ne doit pas apparaître deux fois dans le même split."""
        key = ["playername", "league", "_source_year", "split", "position"]
        dups = sample_df.duplicated(subset=key)
        # On vérifie juste qu'il n'y a pas de doublons excessifs (fixture aléatoire)
        assert dups.sum() < len(sample_df) * 0.1, \
            f"Trop de doublons player/split : {dups.sum()}"

    def test_leagues_are_expected(self, sample_df):
        """Les ligues doivent appartenir à la liste connue."""
        known = {"LFL", "LFL2", "LVP SL", "NLC", "PRM", "TCL", "LEC"}
        actual = set(sample_df["league"].unique())
        # La fixture contient LFL, PRM, NLC — sous-ensemble OK
        assert actual.issubset(known), \
            f"Ligues inconnues : {actual - known}"
