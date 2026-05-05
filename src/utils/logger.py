"""
logger.py — Configuration du logging structuré avec Loguru

Pourquoi Loguru plutôt que le module logging standard de Python ?
  → Syntaxe plus simple et plus lisible (pas besoin de configurer des handlers)
  → Couleurs automatiques dans le terminal
  → Rotation de fichiers intégrée
  → Format structuré avec contexte (fichier, ligne, fonction)

Ce module configure un logger global utilisé dans tout le projet.
Usage dans n'importe quel autre fichier :
    from src.utils.logger import logger
    logger.info("Mon message")
"""

import sys
import os
from pathlib import Path
from loguru import logger

# ── Suppression du logger par défaut ─────────────────────────────────────────
# Loguru ajoute un logger stderr par défaut. On le supprime pour le reconfigurer.
logger.remove()

# ── Configuration du niveau de log ───────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Dossier de logs ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Format des messages ──────────────────────────────────────────────────────
# Format lisible pour le terminal (avec couleurs)
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Format pour les fichiers de log (sans couleurs, avec plus de détails)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# ── Logger console ───────────────────────────────────────────────────────────
# Affiche les logs dans le terminal avec des couleurs
logger.add(
    sys.stderr,
    format=CONSOLE_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
)

# ── Logger fichier ───────────────────────────────────────────────────────────
# Sauvegarde les logs dans un fichier avec rotation automatique
# Rotation : nouveau fichier tous les jours ou quand le fichier dépasse 10 Mo
# Rétention : on garde les logs des 7 derniers jours
logger.add(
    LOG_DIR / "kcorp_scouting_{time:YYYY-MM-DD}.log",
    format=FILE_FORMAT,
    level="DEBUG",  # On log tout dans le fichier, même si la console filtre
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# ── Export ────────────────────────────────────────────────────────────────────
# Re-export le logger configuré pour qu'il soit importable directement
__all__ = ["logger"]
