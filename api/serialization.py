"""Conversion des lignes pandas en objets JSON.

pandas représente les valeurs manquantes par NaN, que JSON ne sait pas
encoder : sérialiser directement produit soit une erreur, soit le littéral
`NaN` que la plupart des clients refusent. Toute donnée qui sort de l'API
passe donc par ici.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def clean_value(value: Any) -> Any:
    """Rend une valeur sérialisable en JSON, ou None si elle est manquante."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [clean_value(item) for item in value.tolist()]
    if pd.isna(value):
        return None
    return value


def row_to_dict(row: pd.Series, rename: dict[str, str] | None = None) -> dict:
    """Convertit une ligne en dictionnaire propre, avec renommage optionnel."""
    mapping = rename or {}
    return {mapping.get(key, key): clean_value(value) for key, value in row.items()}


def frame_to_dicts(frame: pd.DataFrame, rename: dict[str, str] | None = None) -> list[dict]:
    """Convertit un DataFrame en liste de dictionnaires propres."""
    return [row_to_dict(row, rename) for _, row in frame.iterrows()]
