"""Configuration de l'API : chemins et paramètres, surchargeables par variables d'environnement."""

import os
import tempfile
from pathlib import Path

# api/config.py -> api/ -> racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Le répertoire des résultats du pipeline. Surchargeable pour les tests et pour
# le conteneur, où les données peuvent être montées ailleurs que dans le dépôt.
METRICS_DIR = Path(os.getenv("SCOUTING_METRICS_DIR", PROJECT_ROOT / "reports" / "metrics"))

# Où atterrissent les snapshots récupérés au démarrage. Volontairement sous
# tempfile.gettempdir() : le système de fichiers d'un conteneur Cloud Run est en
# lecture seule partout ailleurs, et écrire dans METRICS_DIR échouerait donc en
# production tout en marchant en local, ce qui est le pire des deux mondes.
REFRESHED_METRICS_DIR = Path(os.getenv("SCOUTING_REFRESH_DIR", Path(tempfile.gettempdir()) / "scouting-metrics"))

# Pagination : une valeur haute par défaut casse les clients, une valeur basse
# multiplie les allers-retours. 50 tient dans un écran, 500 est le plafond dur.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# Nombre de joueurs similaires renvoyés par défaut.
DEFAULT_SIMILAR_COUNT = 5
MAX_SIMILAR_COUNT = 50

API_TITLE = "Scouting LoL API"
API_DESCRIPTION = (
    "Accès programmatique aux résultats du pipeline de scouting esport : "
    "scores de talent, archétypes de jeu et joueurs similaires sur les ligues "
    "régionales européennes de League of Legends."
)
