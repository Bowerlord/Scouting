"""
config.py — Configuration centralisée du projet KCorp Scouting Tool

Ce fichier regroupe toutes les constantes et paramètres du projet :
  - Chemins vers les dossiers de données
  - Liste des ligues ciblées (avec leurs noms dans Oracle's Elixir)
  - Années de données à utiliser
  - Hyperparamètres par défaut
  - Configuration du split temporel

Pourquoi un fichier de config centralisé ?
  → Éviter les "magic numbers" et les chemins codés en dur dans le code.
  → Modifier un paramètre à un seul endroit au lieu de chercher dans 10 fichiers.
  → Faciliter la reproductibilité : on sait exactement quels paramètres ont été utilisés.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Chemins du projet
# ══════════════════════════════════════════════════════════════════════════════

# Racine du projet (remonte de 1 niveau depuis src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Dossiers de sortie
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# ══════════════════════════════════════════════════════════════════════════════
# Données Oracle's Elixir
# ══════════════════════════════════════════════════════════════════════════════

# Années de données à télécharger
# Choix : 2024-2026 uniquement.
# La scène pro LoL évolue très vite (changements de méta, reworks de champions,
# restructuration des ligues). Des données avant 2024 introduiraient du bruit
# plutôt que du signal.
DATA_YEARS = [2024, 2025, 2026]
# Note : 2026 est partiel (Spring Split en cours au moment du projet)

# IDs Google Drive des fichiers CSV Oracle's Elixir
# Pattern : {année}_LoL_esports_match_data_from_OraclesElixir.csv
GOOGLE_DRIVE_IDS = {
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",  # Partiel — Spring Split 2026
}

# ══════════════════════════════════════════════════════════════════════════════
# Ligues ciblées
# ══════════════════════════════════════════════════════════════════════════════

# Ligues ERL Division 1 (niveau "LFL-like" dans chaque pays)
# Les clés doivent correspondre EXACTEMENT aux valeurs de la colonne `league`
# des CSV Oracle's Elixir (mêmes libellés que les pages byTournament du site).
# Une clé absente des données déclenche un warning de filter_target_leagues
# (+ tentative de match partiel) — vérifier les logs du workflow Data Refresh
# après tout ajout. EMEA Masters ("EM") est volontairement exclu : c'est le
# tournoi inter-ERL, ses lignes doublonneraient les joueurs des ligues membres.
ERL_DIV1_LEAGUES = {
    "LFL": "🇫🇷 France — La Ligue Française de LoL",
    "PRM": "🇩🇪 Allemagne — Prime League Pro Division",
    "LVP SL": "🇪🇸 Espagne — Liga de Videojuegos Profesional SuperLiga",
    "NLC": "🇬🇧 UK/Nordics — Northern League of Legends Championship",
    "TCL": "🇹🇷 Turquie — Turkish Championship League",
    # ── ERLs mineures (libellés vérifiés sur oracleselixir.com) ──
    "LPLOL": "🇵🇹 Portugal — Liga Portuguesa de League of Legends",
    "Hitpoint Masters": "🇨🇿🇸🇰 Tchéquie/Slovaquie — Hitpoint Masters",
    "Ultraliga": "🇵🇱 Pologne — Ultraliga",
    "PG Nationals": "🇮🇹 Italie — PG Nationals",
    # ── ERLs mineures (codes OE historiques, à confirmer au prochain refresh) ──
    "EBL": "🇷🇸 Balkans — Esports Balkan League",
    "GLL": "🇬🇷 Grèce — Greek Legends League",
    "AL": "🌍 MENA — Arabian League",
}

# Ligues ERL Division 2 (niveau inférieur)
# Validé en Phase 2 via df['league'].unique() sur les CSV Oracle's Elixir :
#   - Seule la LFL2 est présente comme division 2 séparée
#   - Les autres pays n'ont pas de Div 2 dans les données Oracle's Elixir
#   - Les noms initiaux ("PRM 2nd Division", etc.) n'existaient pas dans les CSV
ERL_DIV2_LEAGUES = {
    "LFL2": "🇫🇷 France — LFL Division 2",
}

# Ligue top (LEC) — utilisée pour la target variable
TOP_LEAGUE = "LEC"

# Toutes les ERLs (Div 1 + Div 2), SANS la LEC — c'est le périmètre des joueurs
# scorés/clusterisés. Source unique de vérité : ne pas redéfinir cette liste
# dans les modules (talent_scorer, clusterer, cleaner l'importent d'ici).
ERL_LEAGUES = list(ERL_DIV1_LEAGUES.keys()) + list(ERL_DIV2_LEAGUES.keys())

# Toutes les ligues combinées (pour le filtrage)
ALL_TARGET_LEAGUES = ERL_LEAGUES + [TOP_LEAGUE]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration du Machine Learning
# ══════════════════════════════════════════════════════════════════════════════

# Split temporel
# Pourquoi pas un random split ? Parce qu'on a des données temporelles.
# Un random split créerait du "data leakage" : le modèle verrait des matchs
# futurs pendant l'entraînement, ce qui gonflerait artificiellement les scores.
TRAIN_YEARS = [2024]
TEST_YEARS = [2025]

# Horizon de la target datée (en mois).
# Un match ERL est étiqueté "promoted_to_lec=True" uniquement si le joueur
# débute en LEC APRÈS ce match, et dans cet horizon. Cela évite deux fuites :
#   1. Un ex-joueur LEC relégué en ERL n'est PAS une pépite (sa promotion est
#      passée) → sa promotion étant antérieure à ses matchs ERL, il reste négatif.
#   2. On ne "prédit" pas une promotion déjà survenue au moment du match observé.
# 18 mois capture les promotions "de la saison suivante" (ex: scouté en Summer,
# promu au Spring d'après) sans étiqueter positivement des splits trop lointains.
PROMOTION_HORIZON_MONTHS = 18

# Colonnes clés d'Oracle's Elixir pour les features
# (liste non exhaustive, sera affinée en Phase 4)
KEY_COLUMNS = [
    "gameid",
    "league",
    "split",
    "date",
    "position",
    "playername",
    "teamname",
    "champion",
    "result",
    # --- Stats de laning ---
    "kills",
    "deaths",
    "assists",
    "killsat15",
    "deathsat15",
    "assistsat15",
    "csat15",
    "csdiffat15",
    "golddiffat15",
    "xpdiffat15",
    # --- Stats globales ---
    "totalgold",
    "earnedgold",
    "minionkills",
    "monsterkills",
    "wardskilled",
    "wardsplaced",
    "visionscore",
    "damagetochampions",
    "damageshare",
    "killparticipation",
    "dpm",            # Damage per minute
    "cspm",           # CS per minute
    "vspm",           # Vision score per minute
    "earnedgoldshare",
    "gamelength",
]

# Hyperparamètres par défaut
RANDOM_STATE = 42  # Pour la reproductibilité

# Random Forest
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "class_weight": "balanced",  # Pour gérer le déséquilibre de classes
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 10,  # Ratio approximatif (sera calculé dynamiquement)
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
}

# Clustering
# NB : les paramètres DBSCAN et les configs PyTorch (autoencoder, feedforward)
# d'itérations abandonnées ont été supprimés — K-Means par position est
# l'approche retenue (voir src/models/clusterer.py).
CLUSTER_PARAMS = {
    "k_range": range(3, 12),  # Range de k à tester pour K-Means
}

# ══════════════════════════════════════════════════════════════════════════════
# Configuration du téléchargement
# ══════════════════════════════════════════════════════════════════════════════

GDRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes
