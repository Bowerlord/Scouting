"""
metadata.py — Métadonnées de fraîcheur des snapshots

Écrit reports/metrics/refresh_metadata.json à chaque exécution du pipeline
de nettoyage, pour que le dashboard puisse afficher « Données à jour du X »
au lieu de laisser l'utilisateur deviner l'âge des snapshots.

Les dates sont tronquées au jour (pas d'heure) pour limiter le bruit de diff
dans les commits de rafraîchissement automatique.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import METRICS_DIR
from src.utils.logger import logger

REFRESH_METADATA_FILENAME = "refresh_metadata.json"


def write_refresh_metadata(df: pd.DataFrame, output_path: Path | None = None) -> dict:
    """
    Écrit les métadonnées de fraîcheur à partir du dataset nettoyé.

    Args:
        df: DataFrame nettoyé (doit contenir `date`, `playername`,
            `_source_year`).
        output_path: Chemin de sortie. Par défaut :
            reports/metrics/refresh_metadata.json

    Returns:
        dict: Les métadonnées écrites.
    """
    if output_path is None:
        output_path = METRICS_DIR / REFRESH_METADATA_FILENAME

    data_max_date = None
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            data_max_date = dates.max().date().isoformat()

    metadata = {
        "generated_at": date.today().isoformat(),
        "data_max_date": data_max_date,
        "data_years": sorted(int(y) for y in df["_source_year"].unique())
        if "_source_year" in df.columns
        else [],
        "n_rows": int(len(df)),
        "n_players": int(df["playername"].nunique())
        if "playername" in df.columns
        else 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    logger.info(f"📅 Métadonnées de fraîcheur écrites → {output_path}")
    return metadata
