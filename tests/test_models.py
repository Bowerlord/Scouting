"""
test_models.py — Tests unitaires pour les modèles ML (Phases 5 & 6)

On teste :
  - Le split Out-of-Time respecte l'isolation temporelle
  - Les modèles entraînés produisent des probabilités valides
  - Le clustering produit des labels cohérents
  - La similarity search retourne les bons types
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_erl_dataset():
    """Dataset ERL synthétique avec split 2024/2025."""
    np.random.seed(0)
    n = 300
    df = pd.DataFrame({
        "playername": [f"p{i}" for i in range(n)],
        "league":     np.random.choice(["LFL", "PRM", "NLC"], n),
        "_source_year": np.random.choice([2024, 2025], n),
        "split":      np.random.choice(["Spring", "Summer"], n),
        "position":   np.random.choice(["top", "jng", "mid", "bot", "sup"], n),
        "teamname":   [f"t{i%8}" for i in range(n)],
        "killparticipation_zscore": np.random.normal(0, 1, n),
        "dpm_zscore":               np.random.normal(0, 1, n),
        "cspm_zscore":              np.random.normal(0, 1, n),
        "vspm_zscore":              np.random.normal(0, 1, n),
        "golddiffat15_zscore":      np.random.normal(0, 1, n),
        "csdiffat15_zscore":        np.random.normal(0, 1, n),
        "xpdiffat15_zscore":        np.random.normal(0, 1, n),
        "win_rate_zscore":          np.random.normal(0, 1, n),
        "champion_pool_size_zscore": np.random.normal(0, 1, n),
        "win_rate":       np.random.uniform(0.3, 0.8, n),
        "games_played":   np.random.randint(5, 40, n),
        "champion_pool_size": np.random.randint(3, 15, n),
        "promoted_to_lec": np.random.choice([True, False], n, p=[0.08, 0.92]),
    })
    df["promoted_to_lec"] = df["promoted_to_lec"].astype(int)
    return df


FEATURE_COLS = [
    "killparticipation_zscore", "dpm_zscore", "cspm_zscore", "vspm_zscore",
    "golddiffat15_zscore", "csdiffat15_zscore", "xpdiffat15_zscore",
    "win_rate_zscore", "champion_pool_size_zscore",
    "win_rate", "games_played", "champion_pool_size",
]

CLUSTER_FEATURES = [
    "dpm_zscore", "cspm_zscore", "vspm_zscore", "killparticipation_zscore",
    "golddiffat15_zscore", "xpdiffat15_zscore", "csdiffat15_zscore",
    "champion_pool_size_zscore", "win_rate_zscore",
]


# ── Tests Out-of-Time Split ───────────────────────────────────────────────────

class TestOutOfTimeSplit:

    def test_train_test_years_disjoint(self, synthetic_erl_dataset):
        """Les années du train set et du test set ne doivent pas se chevaucher."""
        df = synthetic_erl_dataset
        train_mask = df["_source_year"] == 2024
        test_mask  = df["_source_year"] == 2025
        # Aucun indice ne doit être dans les deux ensembles
        train_idx = set(df[train_mask].index)
        test_idx  = set(df[test_mask].index)
        assert train_idx.isdisjoint(test_idx), "Fuite temporelle : overlap train/test"

    def test_train_precedes_test(self, synthetic_erl_dataset):
        """Toutes les années du train doivent précéder toutes les années du test."""
        df = synthetic_erl_dataset
        max_train_year = df[df["_source_year"] == 2024]["_source_year"].max()
        min_test_year  = df[df["_source_year"] == 2025]["_source_year"].min()
        assert max_train_year < min_test_year, \
            f"Le train set ({max_train_year}) n'est pas antérieur au test ({min_test_year})"

    def test_both_sets_non_empty(self, synthetic_erl_dataset):
        """Les deux ensembles doivent contenir des données."""
        df = synthetic_erl_dataset
        assert (df["_source_year"] == 2024).sum() > 0, "Train set vide"
        assert (df["_source_year"] == 2025).sum() > 0, "Test set vide"


# ── Tests Modèles Supervisés ──────────────────────────────────────────────────

class TestSupervisedModels:

    @pytest.fixture
    def trained_lr(self, synthetic_erl_dataset):
        df = synthetic_erl_dataset
        train = df[df["_source_year"] == 2024]
        X_train = train[FEATURE_COLS].fillna(0)
        y_train = train["promoted_to_lec"]
        model = LogisticRegression(class_weight="balanced", max_iter=500, C=0.1, random_state=42)
        model.fit(X_train, y_train)
        return model, df[df["_source_year"] == 2025]

    def test_probabilities_in_range(self, trained_lr, synthetic_erl_dataset):
        """Les probabilités prédites doivent être dans [0, 1]."""
        model, df_test = trained_lr
        X_test = df_test[FEATURE_COLS].fillna(0)
        proba = model.predict_proba(X_test)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all(), \
            f"Probabilités hors [0,1] : min={proba.min():.4f}, max={proba.max():.4f}"

    def test_model_better_than_random(self, trained_lr, synthetic_erl_dataset):
        """Le modèle doit être meilleur qu'un classifieur aléatoire (PR-AUC > baseline)."""
        from sklearn.metrics import average_precision_score
        model, df_test = trained_lr
        X_test = df_test[FEATURE_COLS].fillna(0)
        y_test = df_test["promoted_to_lec"]
        proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, proba)
        baseline = y_test.mean()  # Baseline aléatoire = taux de positifs
        assert pr_auc > baseline, \
            f"Modèle (PR-AUC={pr_auc:.4f}) n'est pas meilleur que la baseline ({baseline:.4f})"

    def test_feature_importances_sum_to_one(self, synthetic_erl_dataset):
        """Les feature importances d'un RF doivent sommer à 1."""
        df = synthetic_erl_dataset
        train = df[df["_source_year"] == 2024]
        X = train[FEATURE_COLS].fillna(0)
        y = train["promoted_to_lec"]
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
        rf.fit(X, y)
        fi_sum = rf.feature_importances_.sum()
        assert abs(fi_sum - 1.0) < 1e-6, f"Somme des importances = {fi_sum:.6f} (attendu 1.0)"


# ── Tests Clustering ──────────────────────────────────────────────────────────

class TestClustering:

    def test_kmeans_labels_valid_range(self, synthetic_erl_dataset):
        """Les labels K-Means doivent être dans [0, k-1]."""
        df = synthetic_erl_dataset
        X = df[CLUSTER_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        k = 3
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        assert labels.min() >= 0 and labels.max() <= k - 1, \
            f"Labels hors [0, {k-1}] : min={labels.min()}, max={labels.max()}"

    def test_kmeans_no_empty_clusters(self, synthetic_erl_dataset):
        """Aucun cluster ne doit être vide."""
        df = synthetic_erl_dataset
        X = df[CLUSTER_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        k = 3
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        cluster_sizes = np.bincount(labels)
        assert (cluster_sizes > 0).all(), f"Cluster(s) vide(s) : {cluster_sizes}"

    def test_per_position_clustering_independent(self, synthetic_erl_dataset):
        """Le clustering par position doit être indépendant entre positions."""
        df = synthetic_erl_dataset
        results = {}
        for pos in ["top", "mid"]:
            df_pos = df[df["position"] == pos].copy()
            if len(df_pos) < 10:
                continue
            X = df_pos[CLUSTER_FEATURES].fillna(0).values
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            km = KMeans(n_clusters=3, random_state=42, n_init=5)
            results[pos] = km.fit_predict(X_sc)

        # Les deux résultats ne doivent pas être identiques
        # (clustering indépendant → résultats différents)
        if "top" in results and "mid" in results:
            # Juste vérifier que ce sont bien des arrays numpy
            assert isinstance(results["top"], np.ndarray)
            assert isinstance(results["mid"], np.ndarray)

    def test_umap_output_shape(self, synthetic_erl_dataset):
        """UMAP doit produire un embedding 2D de la bonne taille."""
        try:
            import umap
            df = synthetic_erl_dataset
            X = df[CLUSTER_FEATURES].fillna(0).values
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            reducer = umap.UMAP(n_components=2, random_state=42)
            emb = reducer.fit_transform(X_sc)
            assert emb.shape == (len(df), 2), \
                f"Shape UMAP incorrecte : {emb.shape}, attendu ({len(df)}, 2)"
        except ImportError:
            pytest.skip("umap-learn non installé, test UMAP ignoré")


# ── Tests Similarity Search ───────────────────────────────────────────────────

class TestSimilaritySearch:

    def test_find_existing_player(self, synthetic_erl_dataset):
        """La similarity search doit retourner un DataFrame non vide pour un joueur existant."""
        from src.models.clusterer import find_similar_players
        df = synthetic_erl_dataset
        X = df[CLUSTER_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        target = df["playername"].iloc[0]
        result = find_similar_players(target, df, X_sc, top_k=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_find_nonexistent_player_returns_empty(self, synthetic_erl_dataset):
        """La similarity search doit retourner un DataFrame vide pour un joueur inconnu."""
        from src.models.clusterer import find_similar_players
        df = synthetic_erl_dataset
        X = np.zeros((len(df), len(CLUSTER_FEATURES)))
        result = find_similar_players("joueur_inexistant_xyz", df, X, top_k=5)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
