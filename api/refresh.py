"""
refresh.py — Récupération des snapshots publiés, au démarrage de l'API.

Pourquoi ce module existe
-------------------------
L'image Docker embarque les snapshots présents à sa construction
(`COPY reports/metrics/` dans le Dockerfile). C'est ce qui rend le conteneur
autonome, mais cela fige aussi les données au jour du déploiement : le
2026-09-07, l'API servait encore les résultats du 20 juillet alors que le dépôt
venait d'être rafraîchi. Redéployer à la main après chaque refresh hebdomadaire
n'est pas tenable, et personne ne s'aperçoit d'un oubli avant longtemps.

Ce module renverse la charge : au premier chargement, l'API va chercher les
snapshots publiés sur la branche par défaut du dépôt. Les fichiers de l'image
restent le socle, et servent de repli quand le réseau ou GitHub fait défaut.

Trois garde-fous, parce qu'une source distante peut échouer de trois façons
différentes et qu'aucune ne doit rendre l'API indisponible :
  1. tout ou rien — les fichiers atterrissent dans un répertoire temporaire et
     ne sont adoptés que si les deux fichiers requis sont arrivés entiers ;
  2. taille minimale — une page d'erreur HTML de 2 Ko ne doit jamais passer
     pour un CSV, c'est exactement le piège qui a coûté sept semaines côté
     acquisition ;
  3. échec silencieux mais tracé — en cas de problème on garde les données
     embarquées et on le dit dans les logs, sans faire échouer le démarrage.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import requests

# Le logger standard, et non celui de `src` : l'image de l'API ne contient que
# `api/` et `reports/metrics/` (voir le Dockerfile). Importer `src` ici passerait
# les tests en local et casserait le conteneur au démarrage.
logger = logging.getLogger(__name__)

# Les deux fichiers sans lesquels `load()` lève DataNotAvailable. S'ils ne sont
# pas récupérés tous les deux, le rafraîchissement entier est abandonné.
FICHIERS_REQUIS = (
    "talent_scores_players.csv",
    "clustering_results.csv",
)

# Utiles mais non bloquants : leur absence dégrade l'API sans l'empêcher de
# répondre (archétypes, métriques de modèle, date de fraîcheur).
FICHIERS_OPTIONNELS = (
    "cluster_profiles.json",
    "talent_score_results.json",
    "refresh_metadata.json",
)

# Un CSV de résultats fait plusieurs dizaines de Ko. En dessous, c'est une page
# d'erreur ou un fichier tronqué, jamais un snapshot exploitable.
TAILLE_MINIMALE_CSV = 2_000

# Les JSON, eux, sont légitimement petits : refresh_metadata.json fait 159
# octets et talent_score_results.json 1,5 Ko. Un seuil de taille les rejetait
# tous les deux, ce qui privait l'API de sa date de fraîcheur — précisément
# l'information qui manquait au départ. Constaté au premier essai réel le
# 2026-09-07. Pour eux le contrôle juste n'est pas la taille mais la syntaxe :
# un JSON qui parse est un JSON, une page d'erreur n'en est jamais un.
TAILLE_MINIMALE_JSON = 20

URL_PAR_DEFAUT = "https://raw.githubusercontent.com/Bowerlord/Scouting/main/reports/metrics"


def snapshots_url() -> str:
    """Base d'URL des snapshots publiés, surchargeable pour un fork ou un test."""
    return os.getenv("SCOUTING_SNAPSHOTS_URL", URL_PAR_DEFAUT).rstrip("/")


def rafraichissement_actif() -> bool:
    """Le rafraîchissement distant est actif sauf désactivation explicite.

    Mis à "0" dans les tests et partout où le réseau ne doit pas être touché.
    """
    return os.getenv("SCOUTING_REFRESH_SNAPSHOTS", "1").strip() not in {"0", "false", ""}


def _contenu_plausible(nom: str, contenu: bytes) -> bool:
    """Vérifie qu'un contenu téléchargé est le fichier attendu, pas une erreur.

    Le contrôle diffère par type parce que la bonne question diffère :
      - un JSON doit parser, quelle que soit sa taille ;
      - un CSV doit peser son poids et commencer par un en-tête.
    Une réponse « 404: Not Found » de GitHub échoue aux deux.
    """
    if nom.endswith(".json"):
        if len(contenu) < TAILLE_MINIMALE_JSON:
            logger.warning(f"Snapshot {nom} rejeté : {len(contenu)} octets, trop court.")
            return False
        try:
            json.loads(contenu.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as erreur:
            logger.warning(f"Snapshot {nom} rejeté : JSON illisible ({erreur}).")
            return False
        return True

    if len(contenu) < TAILLE_MINIMALE_CSV:
        logger.warning(
            f"Snapshot {nom} rejeté : {len(contenu)} octets, minimum attendu "
            f"{TAILLE_MINIMALE_CSV}. Probablement une page d'erreur plutôt "
            f"qu'un fichier de résultats."
        )
        return False

    premiere_ligne = contenu.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    if "playername" not in premiere_ligne:
        logger.warning(
            f"Snapshot {nom} rejeté : l'en-tête ne ressemble pas à un export du "
            f"pipeline (colonne `playername` absente). Reçu : {premiere_ligne[:120]!r}"
        )
        return False

    return True


def _telecharger(nom: str, destination: Path, timeout: int) -> bool:
    """Télécharge un fichier de snapshot. Renvoie False sur le moindre doute."""
    url = f"{snapshots_url()}/{nom}"
    try:
        reponse = requests.get(url, timeout=timeout)
    except requests.RequestException as erreur:
        logger.warning(f"Snapshot {nom} injoignable : {erreur}")
        return False

    if reponse.status_code != 200:
        logger.warning(f"Snapshot {nom} : HTTP {reponse.status_code}")
        return False

    if not _contenu_plausible(nom, reponse.content):
        return False

    destination.write_bytes(reponse.content)
    return True


def recuperer_snapshots(cible: Path, timeout: int = 20) -> Path | None:
    """Télécharge les snapshots publiés dans `cible`.

    Args:
        cible: répertoire où écrire les fichiers. Créé au besoin.
        timeout: délai réseau par fichier, en secondes.

    Returns:
        Le répertoire si les fichiers requis sont tous arrivés, sinon None.
        None n'est pas une erreur : l'appelant garde les données embarquées.
    """
    if not rafraichissement_actif():
        logger.info("Rafraîchissement des snapshots désactivé (SCOUTING_REFRESH_SNAPSHOTS).")
        return None

    # Un répertoire temporaire, adopté seulement s'il est complet : une reprise
    # partielle sur le répertoire actif laisserait des fichiers de deux dates
    # différentes, et la fusion joueurs/clusters échouerait sur des clés
    # dépareillées, ce qui est bien pire que des données un peu anciennes.
    with tempfile.TemporaryDirectory(prefix="scouting-snapshots-") as brouillon:
        dossier = Path(brouillon)

        for nom in FICHIERS_REQUIS:
            if not _telecharger(nom, dossier / nom, timeout):
                logger.warning(
                    f"Rafraîchissement abandonné : {nom} n'a pas pu être récupéré. "
                    f"L'API garde les snapshots embarqués dans l'image."
                )
                return None

        for nom in FICHIERS_OPTIONNELS:
            _telecharger(nom, dossier / nom, timeout)

        cible.mkdir(parents=True, exist_ok=True)
        for fichier in dossier.iterdir():
            shutil.copy2(fichier, cible / fichier.name)

    logger.info(f"Snapshots rafraîchis depuis {snapshots_url()} vers {cible}")
    return cible
