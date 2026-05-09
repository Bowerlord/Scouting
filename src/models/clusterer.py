"""
clusterer.py — Phase 6 : Playstyle Clustering

Approche finale : Clustering par position (5 modèles K-Means indépendants).

Pourquoi par position ?
  Itération 1 (clustering global) : silhouette=0.145, archetypes vagues.
  Les ADC et supports ne sont pas comparables sur les mêmes métriques
  (DPM élevé n'a pas le même sens pour un ADC vs. un support).

  → K-Means par position donne des silhouette scores plus élevés et
    des archetypes qui ont du sens esport.

Pourquoi pas DBSCAN ?
  Testé, abandonné : la "malédiction de la dimensionnalité" (9 features
  après StandardScaler) rend les distances euclidiennes uniformes.
  - eps petit → 100% bruit
  - eps grand → 1 méga-cluster
  K-Means est plus robuste sur des données tabulaires normalisées.

Usage :
  python -m src.models.clusterer
  python -m src.models.clusterer --position mid   # clustering d'une position
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import CLUSTER_PARAMS, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.utils.logger import logger

warnings.filterwarnings("ignore")

# ── Features ──────────────────────────────────────────────────────────────────
CLUSTER_FEATURES = [
    "dpm_zscore",               # Agressivité offensive
    "cspm_zscore",              # Farming / économie
    "vspm_zscore",              # Contrôle de vision
    "killparticipation_zscore", # Implication dans les combats
    "golddiffat15_zscore",      # Dominance en early game (gold)
    "xpdiffat15_zscore",        # Dominance en early game (xp)
    "csdiffat15_zscore",        # Pression de lane early
    "champion_pool_size_zscore",# Diversité / polyvalence
    "win_rate_zscore",          # Performance relative
]

ERL_LEAGUES = ["LFL", "LFL2", "LVP SL", "NLC", "PRM", "TCL"]
POSITIONS   = ["top", "jng", "mid", "bot", "sup"]

# Noms narratifs d'archetypes par position, indexés sur les features dominantes.
# Ces libellés sont générés automatiquement via `_archetype_label()`.
ARCHETYPE_RULES = [
    # (condition_fn, label)
    (lambda r: r.get("dpm_zscore", 0) > 0.3 and r.get("killparticipation_zscore", 0) > 0.3, "Carry agressif"),
    (lambda r: r.get("dpm_zscore", 0) > 0.5, "High DPS"),
    (lambda r: r.get("cspm_zscore", 0) > 0.3 and r.get("golddiffat15_zscore", 0) > 0.2, "Farmer dominant"),
    (lambda r: r.get("csdiffat15_zscore", 0) > 0.4, "Lane bully"),
    (lambda r: r.get("vspm_zscore", 0) > 0.5, "Vision controller"),
    (lambda r: r.get("xpdiffat15_zscore", 0) > 0.3 and r.get("golddiffat15_zscore", 0) > 0.3, "Early dominant"),
    (lambda r: r.get("champion_pool_size_zscore", 0) > 0.5, "Versatile"),
    (lambda r: r.get("win_rate_zscore", 0) > 0.5, "High performer"),
    (lambda r: all(abs(r.get(f, 0)) < 0.15 for f in CLUSTER_FEATURES[:5]), "Profil équilibré"),
]


def _archetype_label(row: pd.Series) -> str:
    traits = [label for cond, label in ARCHETYPE_RULES if cond(row)]
    return " | ".join(traits) if traits else "Profil neutre"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chargement
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features_players.csv"
    if not path.exists():
        raise FileNotFoundError("features_players.csv introuvable. Lancez `make features`.")
    df = pd.read_csv(path)
    if "promoted_to_lec" in df.columns:
        df["promoted_to_lec"] = df["promoted_to_lec"].astype(str).str.lower() == "true"
    df = df[df["league"].isin(ERL_LEAGUES)].copy()
    logger.info(f"Dataset ERL : {len(df):,} lignes")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. K optimal par silhouette
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_k(X: np.ndarray, k_range: range, position: str) -> tuple[int, list, list]:
    """Teste chaque k, retourne le k avec le meilleur silhouette score."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else 0.0
        silhouettes.append(sil)
        logger.info(f"  [{position}] k={k:2d} | Inertie={km.inertia_:,.0f} | Silhouette={sil:.4f}")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    logger.info(f"  [{position}] => k optimal = {best_k} (silhouette={max(silhouettes):.4f})")
    return best_k, inertias, silhouettes


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Clustering d'une position
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_position(
    df_pos: pd.DataFrame,
    features: list[str],
    position: str,
    k_range: range,
) -> dict:
    """
    Entraîne K-Means + UMAP pour une position donnée.
    Retourne un dict avec le modèle, les labels, le scaler, l'embedding et les métriques.
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"POSITION : {position.upper()} ({len(df_pos)} joueurs)")
    logger.info(f"{'='*55}")

    X = df_pos[features].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # ── K-Means ───────────────────────────────────────────────────────────────
    best_k, inertias, silhouettes = find_optimal_k(X_sc, k_range, position)
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_sc)
    final_sil = silhouette_score(X_sc, labels)

    sizes = dict(zip(*np.unique(labels, return_counts=True)))
    logger.info(f"  Tailles clusters : {sizes}")

    # ── UMAP (ou PCA fallback) ────────────────────────────────────────────────
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=RANDOM_STATE)
        emb = reducer.fit_transform(X_sc)
        reducer_type = "umap"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)
        emb = reducer.fit_transform(X_sc)
        reducer_type = "pca"
    logger.info(f"  Réduction 2D ({reducer_type}) : OK")

    # ── Profiling des clusters ────────────────────────────────────────────────
    df_tmp = df_pos.copy()
    df_tmp["cluster"] = labels
    has_promo = "promoted_to_lec" in df_tmp.columns

    cluster_profiles = []
    for c in sorted(df_tmp["cluster"].unique()):
        mask = df_tmp["cluster"] == c
        sub = df_tmp[mask]
        means = sub[features].mean().to_dict()
        profile = {
            "cluster": int(c),
            "n_players": int(mask.sum()),
            "n_promoted": int(sub["promoted_to_lec"].sum()) if has_promo else 0,
            "promo_rate": float(sub["promoted_to_lec"].mean()) if has_promo else 0.0,
            "archetype": _archetype_label(pd.Series(means)),
            **{f: round(v, 4) for f, v in means.items()},
        }
        cluster_profiles.append(profile)
        promo_info = f" | promos={profile['n_promoted']} ({profile['promo_rate']:.1%})" if has_promo else ""
        logger.info(f"  Cluster {c} [{profile['archetype']}] : {profile['n_players']} joueurs{promo_info}")

    return {
        "position":   position,
        "model":      km,
        "scaler":     scaler,
        "reducer":    reducer,
        "reducer_type": reducer_type,
        "labels":     labels,
        "embedding":  emb,
        "features":   features,
        "best_k":     best_k,
        "silhouette": final_sil,
        "inertias":   inertias,
        "silhouettes": silhouettes,
        "k_range":    list(k_range),
        "profiles":   cluster_profiles,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Similarity Search
# ═══════════════════════════════════════════════════════════════════════════════

def find_similar_players(
    target_name: str,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Distance euclidienne dans l'espace normalisé pour trouver les joueurs
    au style le plus proche du joueur cible.

    Usage : "Quel joueur ERL ressemble le plus à [star LEC] ?"
    """
    mask = df["playername"].str.lower() == target_name.lower()
    if not mask.any():
        logger.warning(f"Joueur '{target_name}' non trouvé.")
        return pd.DataFrame()

    idx = df[mask].index[0]
    pos_in_array = df.index.get_loc(idx)
    target_vec = X_scaled[pos_in_array]

    distances = np.linalg.norm(X_scaled - target_vec, axis=1)
    df_res = df.copy()
    df_res["similarity_distance"] = distances
    result = (
        df_res[~mask]
        .sort_values("similarity_distance")
        .head(top_k)[["playername", "league", "position", "similarity_distance", "_source_year"]]
        .reset_index(drop=True)
    )
    logger.info(f"\nJoueurs similaires à '{target_name}' :")
    for _, row in result.iterrows():
        logger.info(f"  {row['playername']:<20} {row['league']:<10} {row['position']:<5} dist={row['similarity_distance']:.3f}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Orchestrateur
# ═══════════════════════════════════════════════════════════════════════════════

def run_clustering_pipeline():
    """
    Pipeline Phase 6 complet :
      1. Chargement des features
      2. Pour chaque position : K-Means (k optimal), UMAP, profiling
      3. Assemblage du dataset enrichi
      4. Sauvegarde des modèles, métriques et CSV
    """
    logger.info("=" * 60)
    logger.info("KCORP SCOUTING — PHASE 6 : PLAYSTYLE CLUSTERING PAR POSITION")
    logger.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    k_range = CLUSTER_PARAMS["k_range"]

    # ── Clustering par position ───────────────────────────────────────────────
    results_by_position = {}
    summary_rows = []

    # Préparer les colonnes dans df
    df["cluster_position"] = -1
    df["umap_x"] = np.nan
    df["umap_y"] = np.nan

    for pos in POSITIONS:
        mask = df["position"] == pos
        df_pos = df[mask].copy()
        if len(df_pos) < 20:
            logger.warning(f"Position '{pos}' : trop peu de données ({len(df_pos)}), skip.")
            continue

        res = cluster_position(df_pos, available_features, pos, k_range)
        results_by_position[pos] = res

        # Remplir les colonnes dans df principal
        df.loc[mask, "cluster_position"] = res["labels"]
        df.loc[mask, "umap_x"] = res["embedding"][:, 0]
        df.loc[mask, "umap_y"] = res["embedding"][:, 1]

        # Ligne de résumé
        best_cluster = max(res["profiles"], key=lambda p: p["promo_rate"])
        summary_rows.append({
            "position": pos,
            "n_players": len(df_pos),
            "best_k": res["best_k"],
            "silhouette": round(res["silhouette"], 4),
            "best_archetype": best_cluster["archetype"],
            "best_archetype_promo_rate": round(best_cluster["promo_rate"], 4),
        })

        # Sauvegarde des modèles de la position
        safe = pos.replace("/", "_")
        joblib.dump(res["model"],   MODELS_DIR / f"clusterer_kmeans_{safe}.pkl")
        joblib.dump(res["scaler"],  MODELS_DIR / f"clusterer_scaler_{safe}.pkl")
        joblib.dump(res["reducer"], MODELS_DIR / f"clusterer_reducer_{safe}.pkl")

    # ── Tableau de synthèse ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    logger.info("\n" + "=" * 60)
    logger.info("RESUME PAR POSITION")
    logger.info("=" * 60)
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['position'].upper():<5} k={row['best_k']} "
            f"sil={row['silhouette']:.4f} "
            f"best_archetype=[{row['best_archetype']}] "
            f"promo={row['best_archetype_promo_rate']:.1%}"
        )

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    # Dataset enrichi
    report_cols = [
        "playername", "league", "_source_year", "split", "position", "teamname",
        "cluster_position", "umap_x", "umap_y", "promoted_to_lec",
        "win_rate", "games_played", "dpm_zscore", "cspm_zscore",
        "golddiffat15_zscore", "win_rate_zscore",
    ]
    report_cols = [c for c in report_cols if c in df.columns]
    df[report_cols].to_csv(METRICS_DIR / "clustering_results.csv", index=False)

    # Profils de clusters par position (JSON)
    all_profiles = {pos: res["profiles"] for pos, res in results_by_position.items()}
    with open(METRICS_DIR / "cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False, default=str)

    # Scores k par position (pour la visualisation du coude)
    k_scores = {
        pos: {
            "k_range": res["k_range"],
            "inertias": res["inertias"],
            "silhouettes": res["silhouettes"],
            "best_k": res["best_k"],
        }
        for pos, res in results_by_position.items()
    }
    with open(METRICS_DIR / "kmeans_k_scores.json", "w", encoding="utf-8") as f:
        json.dump(k_scores, f, indent=2, ensure_ascii=False)

    # Résumé CSV
    summary_df.to_csv(METRICS_DIR / "clustering_summary.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6 TERMINEE")
    logger.info(f"  Positions traitées : {list(results_by_position.keys())}")
    logger.info(f"  Joueurs clustérés  : {len(df):,}")
    logger.info(f"  Fichiers générés   : clustering_results.csv, cluster_profiles.json")
    logger.info("=" * 60)

    return df, results_by_position, summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=str, default=None,
                        help="Filtre sur une position (top/jng/mid/bot/sup)")
    args = parser.parse_args()

    if args.position:
        df_full = load_data()
        avail = [f for f in CLUSTER_FEATURES if f in df_full.columns]
        mask  = df_full["position"] == args.position
        cluster_position(df_full[mask].copy(), avail, args.position,
                         CLUSTER_PARAMS["k_range"])
    else:
        run_clustering_pipeline()
