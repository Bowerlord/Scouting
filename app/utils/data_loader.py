"""
data_loader.py — Chargement des données pré-calculées pour l'interface Streamlit

Ce module centralise l'accès aux fichiers de résultats du pipeline ML :
  - Scores de talent des joueurs (talent_scores_players.csv)
  - Résultats de clustering par joueur (clustering_results.csv)
  - Profils des clusters par archétype (cluster_profiles.json)

Pourquoi cette approche ?
  → L'app Streamlit ne doit JAMAIS relancer le pipeline ML ni charger les modèles .pkl.
    Ces fichiers CSV/JSON sont des snapshots figés des résultats, optimisés pour la lecture.
  → Toutes les fonctions de chargement utilisent @st.cache_data pour éviter de relire
    le disque à chaque interaction utilisateur — Streamlit re-exécute le script entier
    à chaque widget, donc sans cache chaque clic rechargerait 2005 lignes de CSV.

Intégration dans le projet :
  app/utils/data_loader.py
    → lit reports/metrics/ (3 niveaux au-dessus via Path(__file__).parent.parent.parent)
    → appelé par app/app.py et app/pages/*.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Chemins du projet
# ══════════════════════════════════════════════════════════════════════════════

# app/utils/data_loader.py → app/utils/ → app/ → racine du projet
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_METRICS_DIR = _PROJECT_ROOT / "reports" / "metrics"
_FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions de chargement avec cache Streamlit
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_model_metrics() -> dict:
    """
    Charge les métriques du pipeline d'entraînement (talent_score_results.json).

    Utilisé par la page d'accueil pour afficher des chiffres à jour (meilleur
    modèle, PR-AUC, années train/test) au lieu de valeurs codées en dur qui
    divergeraient à chaque ré-entraînement.

    Returns:
        dict: Contenu du JSON (comparison, best_model, brier scores, ...).
              Dict vide si le fichier n'existe pas — l'appelant doit gérer
              l'absence de clés.
    """
    path = _METRICS_DIR / "talent_score_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_refresh_metadata() -> dict:
    """
    Charge les métadonnées de fraîcheur des snapshots (refresh_metadata.json).

    Écrit par src/utils/metadata.py à chaque exécution du pipeline de
    nettoyage. Permet d'afficher « Données à jour du X » dans la sidebar.

    Returns:
        dict: {generated_at, data_max_date, data_years, n_rows, n_players}.
              Dict vide si le fichier n'existe pas (snapshots antérieurs à
              son introduction) — l'appelant doit gérer l'absence de clés.
    """
    path = _METRICS_DIR / "refresh_metadata.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_talent_scores() -> pd.DataFrame:
    """
    Charge le fichier des scores de talent par joueur.

    Colonnes attendues (produites par src/models/talent_scorer.py) :
      playername, league, _source_year, split, position, teamname,
      talent_score, promoted_to_lec, win_rate, games_played,
      champion_pool_size, dpm_zscore, cspm_zscore, golddiffat15_zscore

    Pourquoi @st.cache_data et pas @st.cache_resource ?
      cache_data sérialise le DataFrame → chaque widget reçoit une copie immutable.
      cache_resource partagerait la référence (dangereux si une page modifie le df).

    Args:
        Aucun — chemin résolu automatiquement depuis _METRICS_DIR.

    Returns:
        pd.DataFrame: 2005 joueurs avec scores de talent et métriques de performance.

    Raises:
        FileNotFoundError: Si le CSV n'a pas encore été généré par le pipeline ML.
                           Lancez `make run-pipeline` pour produire ce fichier.
    """
    path = _METRICS_DIR / "talent_scores_players.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n\n"
            "Le pipeline ML doit être exécuté pour générer ce fichier.\n"
            "Commande : make run-pipeline  (ou python -m src.models.talent_scorer)"
        )

    df = pd.read_csv(path)

    # Normaliser la colonne cible en booléen — le CSV peut stocker 0/1 ou True/False
    if "promoted_to_lec" in df.columns:
        df["promoted_to_lec"] = df["promoted_to_lec"].astype(bool)

    return df


@st.cache_data
def load_clustering_results() -> pd.DataFrame:
    """
    Charge les résultats de clustering par joueur.

    Associe chaque joueur à son cluster K-Means (par position) et à l'archétype
    correspondant. Colonnes attendues : playername, position, cluster, archetype.
    Des colonnes supplémentaires (league, _source_year) peuvent être présentes
    selon la version du pipeline.

    Args:
        Aucun — chemin résolu automatiquement depuis _METRICS_DIR.

    Returns:
        pd.DataFrame: Assignation cluster par joueur. DataFrame vide si le fichier
                      n'existe pas (la page gérera l'absence de clustering).

    Raises:
        FileNotFoundError: Si le CSV n'a pas encore été généré par le pipeline ML.
    """
    path = _METRICS_DIR / "clustering_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n\n"
            "Le pipeline de clustering doit être exécuté.\n"
            "Commande : make run-pipeline  (ou python -m src.models.clusterer)"
        )
    return pd.read_csv(path)


@st.cache_data
def load_cluster_profiles() -> dict:
    """
    Charge les profils des clusters par position (archetypes, z-scores moyens).

    Structure JSON retournée :
      {
        "top": [
          {"cluster": 0, "archetype": "Profil neutre", "n_players": 66,
           "promo_rate": 0.0, "dpm_zscore": -1.04, ...},
          ...
        ],
        "jng": [...], "mid": [...], "bot": [...], "sup": [...]
      }

    Ce dictionnaire est utilisé :
      - Dans Scout Mode pour construire le filtre par archétype
      - Dans Profil Joueur pour afficher l'archétype d'un cluster donné

    Args:
        Aucun — chemin résolu automatiquement depuis _METRICS_DIR.

    Returns:
        dict: Profils des clusters indexés par position (clé = position lowercase).

    Raises:
        FileNotFoundError: Si cluster_profiles.json est absent.
    """
    path = _METRICS_DIR / "cluster_profiles.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n\n"
            "Lancez le pipeline de clustering pour générer ce fichier."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires (non cachées — légères, pas de I/O)
# ══════════════════════════════════════════════════════════════════════════════


def get_archetype(cluster_profiles: dict, position: str, cluster_id: int) -> str:
    """
    Retourne le libellé d'archétype pour un cluster donné d'une position.

    Recherche dans le dictionnaire cluster_profiles le profil correspondant
    à (position, cluster_id) et retourne son archétype.

    Args:
        cluster_profiles: Dictionnaire retourné par load_cluster_profiles().
        position: Position du joueur ('top', 'jng', 'mid', 'bot', 'sup').
                  Insensible à la casse (normalisé en lowercase).
        cluster_id: Numéro entier du cluster (0, 1, 2...).

    Returns:
        str: Libellé de l'archétype (ex: "Carry agressif | High DPS | ..."),
             ou "Inconnu" si la combinaison position/cluster n'est pas trouvée.
    """
    profiles = cluster_profiles.get(position.lower(), [])
    for profile in profiles:
        if profile.get("cluster") == cluster_id:
            return profile.get("archetype", "Inconnu")
    return "Inconnu"


def list_archetypes(cluster_profiles: dict) -> list[str]:
    """
    Retourne la liste dédupliquée de tous les archetypes disponibles.

    Utilisée dans Scout Mode pour construire le menu de filtrage par archétype.
    Les archetypes "Profil neutre" sont inclus car ils représentent une catégorie valide.

    Args:
        cluster_profiles: Dictionnaire retourné par load_cluster_profiles().

    Returns:
        list[str]: Archetypes triés alphabétiquement.
    """
    archetypes = set()
    for pos_profiles in cluster_profiles.values():
        for profile in pos_profiles:
            arch = profile.get("archetype", "")
            if arch:
                archetypes.add(arch)
    return sorted(archetypes)
