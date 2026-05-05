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

import time
import requests
from pathlib import Path

from src.config import (
    DATA_YEARS,
    GOOGLE_DRIVE_IDS,
    GDRIVE_DOWNLOAD_URL,
    RAW_DATA_DIR,
    MAX_RETRIES,
    RETRY_DELAY,
)
from src.utils.logger import logger


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
        True si le téléchargement a réussi, False sinon
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

    # Écriture en streaming : on lit le fichier par chunks de 32 Ko
    # au lieu de tout charger en mémoire (les CSV font ~150+ Mo)
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    # Vérification de sécurité : si le fichier fait moins de 1 Ko,
    # c'est probablement une page d'erreur HTML et non un CSV
    file_size = destination.stat().st_size
    if file_size < 1000:
        with open(destination, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(500)
            if "<html" in content.lower():
                logger.error(
                    "Le fichier téléchargé est une page HTML d'erreur, "
                    "pas un CSV. Le lien Google Drive est peut-être expiré."
                )
                destination.unlink()
                return False

    size_mb = file_size / (1024 * 1024)
    logger.success(f"Téléchargé : {destination.name} ({size_mb:.1f} Mo)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions principales
# ═══════════════════════════════════════════════════════════════════════════════


def download_csv(year: int, force: bool = False) -> Path | None:
    """
    Télécharge le CSV Oracle's Elixir pour une année donnée.

    Le fichier est sauvegardé dans data/raw/. Si le fichier existe déjà
    et que force=False, le téléchargement est ignoré (cache).

    Args:
        year: L'année du dataset (2024 ou 2025)
        force: Si True, re-télécharge même si le fichier existe

    Returns:
        Le chemin vers le fichier CSV, ou None en cas d'échec

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
        if file_size > 1_000_000:  # > 1 Mo = probablement valide
            size_mb = file_size / (1024 * 1024)
            logger.info(
                f"Cache hit : {filename} ({size_mb:.1f} Mo) — "
                f"téléchargement ignoré. Utilisez force=True pour re-télécharger."
            )
            return filepath
        else:
            logger.warning(
                f"Fichier {filename} trouvé mais trop petit ({file_size} octets). "
                f"Re-téléchargement..."
            )

    # ── Téléchargement avec retry ────────────────────────────────────────
    file_id = GOOGLE_DRIVE_IDS[year]
    logger.info(f"📥 Téléchargement de {filename} depuis Google Drive...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            success = _download_from_gdrive(file_id, filepath)
            if success:
                return filepath
            else:
                logger.warning(
                    f"Tentative {attempt}/{MAX_RETRIES} échouée pour {filename}"
                )
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

    logger.error(
        f"❌ Échec du téléchargement de {filename} après {MAX_RETRIES} tentatives. "
        f"Vérifiez votre connexion et les IDs Google Drive dans config.py."
    )
    return None


def download_all(
    years: list[int] | None = None,
    force: bool = False,
) -> dict[int, Path]:
    """
    Télécharge les CSV pour toutes les années configurées.

    Args:
        years: Liste des années à télécharger. Par défaut : DATA_YEARS de config.py
        force: Si True, force le re-téléchargement de tous les fichiers

    Returns:
        Dictionnaire {année: chemin_fichier} pour les téléchargements réussis
    """
    if years is None:
        years = DATA_YEARS

    results = {}
    logger.info(f"{'='*60}")
    logger.info(f"📥 ORACLE'S ELIXIR — Téléchargement des données")
    logger.info(f"   Années : {years}")
    logger.info(f"   Destination : {RAW_DATA_DIR}")
    logger.info(f"{'='*60}")

    for year in years:
        filepath = download_csv(year, force=force)
        if filepath is not None:
            results[year] = filepath

    # Résumé
    success = len(results)
    total = len(years)
    logger.info(f"{'='*60}")
    if success == total:
        logger.success(f"✅ Téléchargement terminé : {success}/{total} fichiers OK")
    else:
        logger.warning(
            f"⚠️  Téléchargement partiel : {success}/{total} fichiers récupérés. "
            f"Années manquantes : {[y for y in years if y not in results]}"
        )
    logger.info(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI : python -m src.data.downloader
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    download_all()
