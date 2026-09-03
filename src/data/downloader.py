"""
downloader.py — Téléchargement des CSV Oracle's Elixir

Ce module gère le téléchargement automatique des données de match depuis
Oracle's Elixir (hébergé sur Google Drive).

Fonctionnalités :
  - Téléchargement des CSV pour 2024-2026 (configurable dans config.py)
  - Cache local intelligent (pas de re-téléchargement si le fichier existe)
  - Retry avec backoff exponentiel en cas d'erreur réseau
  - Gestion des fichiers volumineux Google Drive (confirmation anti-virus)
  - Logging structuré à chaque étape

Usage :
  # Depuis le Makefile
  make data

  # En Python
  from src.data.downloader import download_all
  files = download_all()

  # En CLI
  python -m src.data.downloader

Pourquoi Google Drive et pas une API ?
  Oracle's Elixir fournit ses données en CSV sur Google Drive. C'est gratuit,
  complet, et couvre toutes les ligues (ERLs, LEC, etc.). Pas besoin d'API key.

Pourquoi un cache local ?
  Les fichiers CSV font ~150-200 Mo chacun. On ne veut pas les re-télécharger
  à chaque exécution du pipeline. Le cache vérifie si le fichier existe déjà
  localement et sa taille pour s'assurer qu'il n'est pas corrompu.
"""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import requests

from src.config import (
    DATA_YEARS,
    GDRIVE_DOWNLOAD_URL,
    GOOGLE_DRIVE_IDS,
    MAX_RETRIES,
    MIN_VALID_CSV_BYTES,
    RAW_DATA_DIR,
    RETRY_DELAY,
)
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# Résultats de téléchargement
# ═══════════════════════════════════════════════════════════════════════════════

# Code de sortie POSIX EX_TEMPFAIL : « échec temporaire, réessayer plus tard ».
# Il sépare deux situations que le pipeline confondait, et qui n'appellent pas
# du tout la même réaction :
#   - la source est indisponible (quota Google Drive) → rien à corriger ici,
#     le workflow s'arrête proprement et retentera au prochain passage ;
#   - le pipeline est cassé (ID mort, fichier corrompu, schéma) → il faut agir,
#     le workflow doit échouer bruyamment.
EXIT_SOURCE_UNAVAILABLE = 75


class DownloadOutcome(Enum):
    """Issue d'une tentative de téléchargement."""

    OK = "ok"
    # Google Drive a répondu une page « Quota exceeded ». Cause externe et
    # temporaire : le compte Drive d'Oracle's Elixir a dépassé son quota de
    # partage, ce qui bloque TOUS ses fichiers, pas seulement les récents.
    SOURCE_UNAVAILABLE = "source_unavailable"
    # Le contenu reçu n'est pas exploitable pour une autre raison : ID expiré,
    # partage modifié, fichier tronqué, en-tête inattendu. Là, il faut agir.
    INVALID_CONTENT = "invalid_content"


@dataclass
class DownloadReport:
    """Bilan d'un appel à download_all."""

    files: dict[int, Path] = field(default_factory=dict)
    outcomes: dict[int, DownloadOutcome] = field(default_factory=dict)

    @property
    def missing(self) -> list[int]:
        return [y for y, o in self.outcomes.items() if o is not DownloadOutcome.OK]

    @property
    def source_unavailable(self) -> bool:
        """True si des fichiers manquent ET que tous les échecs sont externes.

        C'est le cas qui ne doit PAS faire échouer le workflow : il n'y a
        rien à corriger dans le dépôt.
        """
        failures = [o for o in self.outcomes.values() if o is not DownloadOutcome.OK]
        return bool(failures) and all(
            o is DownloadOutcome.SOURCE_UNAVAILABLE for o in failures
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════════════


def _get_filename(year: int) -> str:
    """
    Génère le nom de fichier CSV pour une année donnée.

    Convention Oracle's Elixir :
      {année}_LoL_esports_match_data_from_OraclesElixir.csv
    """
    return f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _get_confirm_token(response: requests.Response) -> str | None:
    """
    Extrait le token de confirmation pour les fichiers volumineux Google Drive.

    Explication technique :
      Quand un fichier dépasse ~100 Mo sur Google Drive, Google affiche une
      page de confirmation anti-virus ("Ce fichier est trop volumineux pour
      être analysé"). Pour télécharger quand même, il faut récupérer un token
      dans les cookies de la réponse et le renvoyer dans une 2e requête.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _download_from_gdrive(file_id: str, destination: Path) -> bool:
    """
    Télécharge un fichier depuis Google Drive avec gestion des gros fichiers.

    Processus :
      1. Envoie une première requête GET
      2. Si Google demande une confirmation → récupère le token dans les cookies
      3. Renvoie la requête avec le token de confirmation
      4. Écrit le fichier en streaming (pour ne pas saturer la RAM)
      5. Vérifie que le fichier n'est pas une page HTML d'erreur

    Args:
        file_id: L'identifiant Google Drive du fichier
        destination: Le chemin local de destination

    Returns:
        DownloadOutcome.OK si le fichier est valide,
        SOURCE_UNAVAILABLE si Google bloque temporairement (quota),
        INVALID_CONTENT dans tous les autres cas de contenu inexploitable.
    """
    session = requests.Session()

    # Première requête : peut retourner le fichier directement OU une page
    # de confirmation si le fichier est trop volumineux
    logger.debug(f"Requête initiale vers Google Drive (file_id={file_id})")
    response = session.get(
        GDRIVE_DOWNLOAD_URL,
        params={"id": file_id},
        stream=True,
        timeout=30,
    )

    # Vérifier si Google demande une confirmation (fichier volumineux)
    token = _get_confirm_token(response)
    if token:
        logger.info("Fichier volumineux détecté — confirmation anti-virus en cours...")
        response = session.get(
            GDRIVE_DOWNLOAD_URL,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=30,
        )

    response.raise_for_status()

    # ── Garde-fou n°1 : le Content-Type ──────────────────────────────────
    # Google répond en text/html quand il sert une page d'erreur (quota
    # dépassé, fichier retiré, permissions changées) plutôt que le fichier.
    # C'est le signal le plus fiable, et il arrive AVANT toute écriture.
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        preview = response.text[:2000]
        if "quota" in preview.lower():
            logger.error(
                "Google Drive a répondu « Quota exceeded » : le fichier est "
                "temporairement indisponible parce que le compte qui l'héberge "
                "a dépassé son quota de partage. Vérifié le 2026-09-03 : le "
                "blocage touche TOUS les fichiers d'Oracle's Elixir, y compris "
                "ceux de 2016 que personne ne télécharge. Ce n'est donc ni un "
                "problème de schéma, ni un ID expiré, et il n'y a rien à "
                "corriger dans ce dépôt."
            )
            return DownloadOutcome.SOURCE_UNAVAILABLE

        logger.error(
            "Google Drive a renvoyé une page HTML au lieu du CSV, sans "
            "mentionner de quota. L'ID est probablement expiré ou le partage "
            "a changé (voir GOOGLE_DRIVE_IDS dans src/config.py)."
        )
        return DownloadOutcome.INVALID_CONTENT

    # Écriture en streaming : on lit le fichier par chunks de 32 Ko
    # au lieu de tout charger en mémoire (les CSV font ~150+ Mo)
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    # ── Garde-fou n°2 : la taille ────────────────────────────────────────
    # Les CSV Oracle's Elixir font 24 à 77 Mo. Le seuil précédent était de
    # 1 Ko, ce qui laissait passer les pages d'erreur de Google (2 Ko) :
    # le fichier était alors annoncé « téléchargé (0.0 Mo) », et l'échec
    # ne se manifestait que 3 étapes plus loin, sous la forme trompeuse
    # d'une dérive de schéma. Le seuil est aligné sur celui du cache.
    file_size = destination.stat().st_size
    if file_size < MIN_VALID_CSV_BYTES:
        head = destination.read_text(encoding="utf-8", errors="ignore")[:500]
        logger.error(
            f"Fichier rejeté : {destination.name} ne fait que {file_size} "
            f"octets (minimum attendu : {MIN_VALID_CSV_BYTES}). "
            f"Contenu reçu : {head[:200]!r}"
        )
        destination.unlink()
        return DownloadOutcome.INVALID_CONTENT

    # ── Garde-fou n°3 : l'en-tête CSV ────────────────────────────────────
    # Un fichier de la bonne taille peut quand même ne pas être le bon CSV.
    # La colonne `gameid` est présente dans tous les exports Oracle's Elixir.
    with open(destination, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    if "gameid" not in header.lower():
        logger.error(
            f"Fichier rejeté : l'en-tête de {destination.name} ne ressemble "
            f"pas à un export Oracle's Elixir (colonne `gameid` absente). "
            f"En-tête reçu : {header[:200]!r}"
        )
        destination.unlink()
        return DownloadOutcome.INVALID_CONTENT

    size_mb = file_size / (1024 * 1024)
    logger.success(f"Téléchargé : {destination.name} ({size_mb:.1f} Mo)")
    return DownloadOutcome.OK


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions principales
# ═══════════════════════════════════════════════════════════════════════════════


def download_csv(
    year: int, force: bool = False
) -> tuple[Path | None, DownloadOutcome]:
    """
    Télécharge le CSV Oracle's Elixir pour une année donnée.

    Le fichier est sauvegardé dans data/raw/. Si le fichier existe déjà
    et que force=False, le téléchargement est ignoré (cache).

    Args:
        year: L'année du dataset (2024 ou 2025)
        force: Si True, re-télécharge même si le fichier existe

    Returns:
        Un couple (chemin du CSV ou None, issue du téléchargement).
        L'issue permet à l'appelant de distinguer une panne externe
        (quota Drive) d'un vrai problème de contenu.

    Raises:
        ValueError: Si l'année n'est pas dans GOOGLE_DRIVE_IDS
    """
    if year not in GOOGLE_DRIVE_IDS:
        raise ValueError(
            f"Année {year} non disponible. "
            f"Années valides : {list(GOOGLE_DRIVE_IDS.keys())}"
        )

    # Créer le dossier data/raw/ s'il n'existe pas
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = _get_filename(year)
    filepath = RAW_DATA_DIR / filename

    # ── Cache local ──────────────────────────────────────────────────────
    # On vérifie si le fichier existe ET s'il fait plus de 1 Mo
    # (un fichier de moins de 1 Mo est probablement corrompu)
    if filepath.exists() and not force:
        file_size = filepath.stat().st_size
        if file_size > MIN_VALID_CSV_BYTES:  # > 1 Mo = probablement valide
            size_mb = file_size / (1024 * 1024)
            logger.info(
                f"Cache hit : {filename} ({size_mb:.1f} Mo) — "
                f"téléchargement ignoré. Utilisez force=True pour re-télécharger."
            )
            return filepath, DownloadOutcome.OK
        else:
            logger.warning(
                f"Fichier {filename} trouvé mais trop petit ({file_size} octets). "
                f"Re-téléchargement..."
            )

    # ── Téléchargement avec retry ────────────────────────────────────────
    file_id = GOOGLE_DRIVE_IDS[year]
    logger.info(f"📥 Téléchargement de {filename} depuis Google Drive...")

    # Dernière issue observée : sert à répondre « source indisponible » plutôt
    # que « contenu invalide » quand les trois tentatives ont buté sur le quota.
    last_outcome = DownloadOutcome.INVALID_CONTENT

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            last_outcome = _download_from_gdrive(file_id, filepath)
            if last_outcome is DownloadOutcome.OK:
                return filepath, DownloadOutcome.OK
            else:
                logger.warning(
                    f"Tentative {attempt}/{MAX_RETRIES} échouée pour {filename}"
                )
                # Inutile d'insister sur un quota : il ne se libère pas en
                # quelques secondes, et les trois tentatives sont du bruit.
                if last_outcome is DownloadOutcome.SOURCE_UNAVAILABLE:
                    break
        except requests.exceptions.Timeout:
            logger.warning(
                f"Tentative {attempt}/{MAX_RETRIES} — Timeout (le serveur met trop "
                f"de temps à répondre)"
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Tentative {attempt}/{MAX_RETRIES} — Erreur de connexion "
                f"(vérifiez votre connexion internet)"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Tentative {attempt}/{MAX_RETRIES} — Erreur réseau : {e}"
            )

        if attempt < MAX_RETRIES:
            wait_time = RETRY_DELAY * attempt  # Backoff exponentiel simple
            logger.info(f"⏳ Nouvelle tentative dans {wait_time} secondes...")
            time.sleep(wait_time)

    if last_outcome is DownloadOutcome.SOURCE_UNAVAILABLE:
        logger.error(
            f"⏸️  {filename} indisponible à la source (quota Google Drive). "
            f"Rien à corriger dans le dépôt, la prochaine exécution retentera."
        )
    else:
        logger.error(
            f"❌ Échec du téléchargement de {filename} après {MAX_RETRIES} "
            f"tentatives. Vérifiez votre connexion et les IDs Google Drive "
            f"dans config.py."
        )
    return None, last_outcome


def download_all(
    years: list[int] | None = None,
    force: bool = False,
) -> DownloadReport:
    """
    Télécharge les CSV pour toutes les années configurées.

    Args:
        years: Liste des années à télécharger. Par défaut : DATA_YEARS de config.py
        force: Si True, force le re-téléchargement de tous les fichiers

    Returns:
        Un DownloadReport : les fichiers récupérés, l'issue par année, et
        la propriété `source_unavailable` qui dit si l'échec est entièrement
        imputable à la source plutôt qu'au dépôt.
    """
    if years is None:
        years = DATA_YEARS

    report = DownloadReport()
    logger.info(f"{'='*60}")
    logger.info("📥 ORACLE'S ELIXIR — Téléchargement des données")
    logger.info(f"   Années : {years}")
    logger.info(f"   Destination : {RAW_DATA_DIR}")
    logger.info(f"{'='*60}")

    for year in years:
        filepath, outcome = download_csv(year, force=force)
        report.outcomes[year] = outcome
        if filepath is not None:
            report.files[year] = filepath

    # Résumé
    success = len(report.files)
    total = len(years)
    logger.info(f"{'='*60}")
    if success == total:
        logger.success(f"✅ Téléchargement terminé : {success}/{total} fichiers OK")
    elif report.source_unavailable:
        logger.warning(
            f"⏸️  Source indisponible : {success}/{total} fichiers récupérés. "
            f"Google Drive bloque le partage d'Oracle's Elixir (quota). "
            f"Années manquantes : {report.missing}"
        )
    else:
        logger.warning(
            f"⚠️  Téléchargement partiel : {success}/{total} fichiers récupérés. "
            f"Années manquantes : {report.missing}"
        )
    logger.info(f"{'='*60}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI : python -m src.data.downloader
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Sortie en code 1 si un seul fichier manque, pour que le pipeline
    # (Makefile, CI, workflow Data Refresh) s'arrête ICI plutôt que de
    # continuer sur des fichiers absents ou tronqués. Sans ça, l'échec
    # réapparaissait 3 étapes plus loin sous la forme d'une fausse
    # « dérive de schéma Oracle's Elixir », qui envoyait chercher le
    # problème au mauvais endroit.
    #
    # Deux codes de sortie distincts, parce que les deux situations n'appellent
    # pas la même réaction côté CI :
    #   75 (EX_TEMPFAIL) → la source est indisponible, le workflow s'arrête
    #                      proprement sans alerter : il n'y a rien à corriger.
    #    1               → le pipeline est réellement cassé, il faut alerter.
    _report = download_all()

    if _report.source_unavailable:
        logger.warning(
            f"Arrêt propre : les données ne sont pas récupérables pour "
            f"l'instant (années {_report.missing}). La cause est extérieure "
            f"au dépôt, aucune action n'est requise. Code de sortie "
            f"{EXIT_SOURCE_UNAVAILABLE}."
        )
        sys.exit(EXIT_SOURCE_UNAVAILABLE)

    if _report.missing:
        logger.error(
            f"Arrêt du pipeline : téléchargement incomplet, années "
            f"manquantes {_report.missing}. Les étapes suivantes ne sont pas "
            f"lancées car elles échoueraient avec un message trompeur."
        )
        sys.exit(1)
