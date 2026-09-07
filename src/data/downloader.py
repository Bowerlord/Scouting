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

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path

import requests

from src.config import (
    DATA_YEARS,
    GDRIVE_DOWNLOAD_URL,
    GOOGLE_DRIVE_IDS,
    MAX_RETRIES,
    METRICS_DIR,
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
    # Google Drive a répondu une page « Quota exceeded ». Cause externe : le
    # compte Drive d'Oracle's Elixir a dépassé son quota de partage, ce qui
    # bloque TOUS ses fichiers, pas seulement les récents.
    # Nuance vérifiée le 2026-09-03 : le quota ne s'applique qu'aux accès
    # anonymes. Un navigateur connecté télécharge le même fichier sans
    # difficulté, donc une acquisition authentifiée reste possible.
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
        return bool(failures) and all(o is DownloadOutcome.SOURCE_UNAVAILABLE for o in failures)


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


def _drive_api_credentials() -> tuple[str, str] | None:
    """
    Récupère le jeton d'accès Drive et le projet de quota, s'ils existent.

    Les deux viennent de l'environnement, jamais du dépôt :
      - GOOGLE_DRIVE_ACCESS_TOKEN : jeton OAuth portant le scope drive.readonly.
        En CI il est produit par `google-github-actions/auth`, qui s'authentifie
        par fédération d'identité OIDC. Aucune clé n'est stockée nulle part, ce
        qui tombe bien : la politique du projet Google Cloud interdit de toute
        façon la création de clés de compte de service.
      - GOOGLE_CLOUD_PROJECT : sans lui, l'API Drive répond 403 avec un message
        sur les « Application Default Credentials » qui envoie chercher au
        mauvais endroit. Constaté en test le 2026-09-07.

    En local, aucune des deux n'est définie : on garde la voie anonyme, donc
    rien ne change pour qui clone le dépôt.
    """
    token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN", "").strip()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if token and project:
        return token, project
    return None


def _looks_like_oracles_elixir_csv(destination: Path) -> bool:
    """
    Vérifie qu'un fichier téléchargé est bien un export Oracle's Elixir.

    Deux garde-fous, factorisés pour que les deux chemins d'acquisition aient
    exactement la même exigence : la taille minimale, puis la présence de la
    colonne `gameid`. Un fichier rejeté est supprimé, pour ne pas empoisonner
    le cache local.
    """
    file_size = destination.stat().st_size
    if file_size < MIN_VALID_CSV_BYTES:
        head = destination.read_text(encoding="utf-8", errors="ignore")[:200]
        logger.error(
            f"Fichier rejeté : {destination.name} ne fait que {file_size} "
            f"octets (minimum attendu : {MIN_VALID_CSV_BYTES}). "
            f"Contenu reçu : {head!r}"
        )
        destination.unlink()
        return False

    with open(destination, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    if "gameid" not in header.lower():
        logger.error(
            f"Fichier rejeté : l'en-tête de {destination.name} ne ressemble "
            f"pas à un export Oracle's Elixir (colonne `gameid` absente). "
            f"En-tête reçu : {header[:200]!r}"
        )
        destination.unlink()
        return False

    return True


def _download_via_drive_api(file_id: str, destination: Path, token: str, project: str) -> DownloadOutcome:
    """
    Télécharge un fichier par l'API Drive, en authentifié.

    Pourquoi cette voie existe, et pourquoi elle passe en premier :
      Le compte qui héberge les exports d'Oracle's Elixir dépasse en permanence
      son quota de partage. Google répond alors une page « Quota exceeded » à
      tout téléchargement ANONYME, ce qui a figé les données publiées pendant
      sept semaines, du 20 juillet au 7 septembre 2026, sans que rien n'échoue.
      Vérifié en réel le 2026-09-07 : le même fichier, demandé à l'API Drive
      avec un jeton OAuth valide, revient en HTTP 200 et 67 Mo. Le quota ne
      frappe donc que les accès non authentifiés.

    Returns:
        OK si le fichier est valide, SOURCE_UNAVAILABLE si l'API refuse pour une
        raison temporaire, INVALID_CONTENT si le contenu est inexploitable.
    """
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Goog-User-Project": project,
    }

    response = requests.get(
        url,
        headers=headers,
        params={"alt": "media"},
        stream=True,
        timeout=60,
    )

    # 401 et 403 ne disent pas la même chose et n'appellent pas la même
    # réaction : le premier est un jeton expiré ou mal scopé, donc un problème
    # de configuration du dépôt ; le second peut être un quota d'API, donc une
    # cause externe qui se règle d'elle-même.
    if response.status_code == 401:
        logger.error(
            "L'API Drive a refusé le jeton (401). Il est expiré, ou il ne porte "
            "pas le scope https://www.googleapis.com/auth/drive.readonly. "
            "Vérifier l'étape d'authentification du workflow."
        )
        return DownloadOutcome.INVALID_CONTENT

    if response.status_code == 403:
        logger.warning(f"L'API Drive a répondu 403, repli sur la voie anonyme. Détail : {response.text[:300]!r}")
        return DownloadOutcome.SOURCE_UNAVAILABLE

    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    if not _looks_like_oracles_elixir_csv(destination):
        return DownloadOutcome.INVALID_CONTENT

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.success(f"Téléchargé par l'API Drive (authentifié) : {destination.name} ({size_mb:.1f} Mo)")
    return DownloadOutcome.OK


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

    # ── Voie authentifiée, essayée en premier ────────────────────────────
    # Elle n'est disponible qu'en CI, où le workflow fournit un jeton. En cas
    # d'échec temporaire de l'API, on retombe sur la voie anonyme plutôt que
    # d'abandonner : elle marche encore les jours où le quota se libère.
    credentials = _drive_api_credentials()
    if credentials is not None:
        token, project = credentials
        logger.info("Acquisition authentifiée par l'API Drive...")
        outcome = _download_via_drive_api(file_id, destination, token, project)
        if outcome is DownloadOutcome.OK:
            return outcome
        logger.warning("L'API Drive n'a pas abouti, nouvelle tentative par la voie anonyme.")

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
                "Google Drive a répondu « Quota exceeded » : le compte qui "
                "héberge le fichier a dépassé son quota de partage. Vérifié le "
                "2026-09-03 : le blocage touche TOUS les fichiers d'Oracle's "
                "Elixir, y compris ceux de 2016 que personne ne télécharge. Ce "
                "n'est donc ni un problème de schéma, ni un ID expiré.\n"
                "Précision importante, vérifiée le même jour : ce quota ne "
                "frappe que les téléchargements ANONYMES, ceux que fait ce "
                "script. Le même fichier se télécharge sans problème depuis un "
                "navigateur connecté à un compte Google. Une acquisition "
                "authentifiée (API Drive + compte de service, clé en secret de "
                "dépôt) est donc une piste ouverte et non testée, plutôt qu'une "
                "impasse."
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

    # ── Garde-fous n°2 et n°3 : la taille, puis l'en-tête CSV ────────────
    # Les mêmes que sur la voie authentifiée, et volontairement au même
    # endroit du code : une seule définition, donc aucun risque que les deux
    # chemins d'acquisition finissent par accepter des choses différentes.
    # Rappel de ce qu'ils attrapent : une page d'erreur Google fait 2 Ko et
    # passait autrefois pour un fichier « téléchargé (0.0 Mo) », l'échec ne
    # se manifestant que trois étapes plus loin sous la forme trompeuse
    # d'une dérive de schéma.
    if not _looks_like_oracles_elixir_csv(destination):
        return DownloadOutcome.INVALID_CONTENT

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.success(f"Téléchargé : {destination.name} ({size_mb:.1f} Mo)")
    return DownloadOutcome.OK


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions principales
# ═══════════════════════════════════════════════════════════════════════════════


def download_csv(year: int, force: bool = False) -> tuple[Path | None, DownloadOutcome]:
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
        raise ValueError(f"Année {year} non disponible. Années valides : {list(GOOGLE_DRIVE_IDS.keys())}")

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
            logger.warning(f"Fichier {filename} trouvé mais trop petit ({file_size} octets). Re-téléchargement...")

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
                logger.warning(f"Tentative {attempt}/{MAX_RETRIES} échouée pour {filename}")
                # Inutile d'insister sur un quota : il ne se libère pas en
                # quelques secondes, et les trois tentatives sont du bruit.
                if last_outcome is DownloadOutcome.SOURCE_UNAVAILABLE:
                    break
        except requests.exceptions.Timeout:
            logger.warning(f"Tentative {attempt}/{MAX_RETRIES} — Timeout (le serveur met trop de temps à répondre)")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Tentative {attempt}/{MAX_RETRIES} — Erreur de connexion (vérifiez votre connexion internet)"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentative {attempt}/{MAX_RETRIES} — Erreur réseau : {e}")

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
    logger.info(f"{'=' * 60}")
    logger.info("📥 ORACLE'S ELIXIR — Téléchargement des données")
    logger.info(f"   Années : {years}")
    logger.info(f"   Destination : {RAW_DATA_DIR}")
    logger.info(f"{'=' * 60}")

    for year in years:
        filepath, outcome = download_csv(year, force=force)
        report.outcomes[year] = outcome
        if filepath is not None:
            report.files[year] = filepath

    # Résumé
    success = len(report.files)
    total = len(years)
    logger.info(f"{'=' * 60}")
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
            f"⚠️  Téléchargement partiel : {success}/{total} fichiers récupérés. Années manquantes : {report.missing}"
        )
    logger.info(f"{'=' * 60}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Garde-fou de fraîcheur
# ═══════════════════════════════════════════════════════════════════════════════

# Au-delà de ce délai, une source indisponible cesse d'être un incident passager.
# Le chiffre vient d'un cas réel : entre le 20 juillet et le 7 septembre 2026, le
# workflow est sorti en vert chaque semaine pendant que les données publiées
# vieillissaient de sept semaines. Personne n'a rien vu, précisément parce que
# rien n'échouait. Trois semaines laissent passer une panne ordinaire et
# rattrapent celle qui s'installe.
MAX_JOURS_SANS_REFRESH = 21


def jours_depuis_dernier_refresh() -> int | None:
    """
    Âge, en jours, des snapshots actuellement publiés.

    La source de vérité est `generated_at` dans refresh_metadata.json, écrit à
    chaque refresh réussi et versionné. Pas d'état à maintenir à côté, et
    surtout : on mesure la fraîcheur réelle de ce que voient les utilisateurs,
    pas le nombre d'exécutions ratées.

    Returns:
        Le nombre de jours, ou None si le fichier est absent ou illisible.
    """
    metadata = METRICS_DIR / "refresh_metadata.json"
    if not metadata.exists():
        return None
    try:
        contenu = json.loads(metadata.read_text(encoding="utf-8"))
        genere_le = date.fromisoformat(contenu["generated_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    return (date.today() - genere_le).days


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
        _age = jours_depuis_dernier_refresh()

        # Une source indisponible ne justifie une sortie silencieuse que tant
        # que les données publiées restent fraîches. Passé le délai, le silence
        # devient le problème : le workflow doit échouer bruyamment, même si la
        # cause reste extérieure au dépôt.
        if _age is not None and _age > MAX_JOURS_SANS_REFRESH:
            logger.error(
                f"Les snapshots publiés datent de {_age} jours, au-delà du "
                f"seuil de {MAX_JOURS_SANS_REFRESH}. La source est toujours "
                f"indisponible, mais un échec silencieux qui dure n'est plus un "
                f"incident passager : le dashboard sert des données périmées. "
                f"Vérifier l'authentification Drive du workflow avant de "
                f"conclure à une panne externe."
            )
            sys.exit(1)

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
