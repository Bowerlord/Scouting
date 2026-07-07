"""
schema.py — Validation du schéma des CSV bruts Oracle's Elixir

Oracle's Elixir a déjà fait dériver son schéma (la colonne `killparticipation`
a disparu des exports récents). Sans garde-fou, une colonne manquante traverse
silencieusement le pipeline et produit des snapshots faux. Ce module valide
le schéma juste après le chargement et échoue bruyamment avec la liste
complète des colonnes en cause.

Usage :
  from src.data.schema import validate_raw_schema, SchemaValidationError
  validate_raw_schema(df, source="2024_LoL_esports_....csv")
"""

import pandas as pd

from src.config import KEY_COLUMNS
from src.utils.logger import logger

# Colonnes de KEY_COLUMNS connues pour manquer des exports récents : leur
# absence déclenche un warning (elles sont recalculées en aval), pas une erreur.
OPTIONAL_COLUMNS = {"killparticipation"}

REQUIRED_COLUMNS = [c for c in KEY_COLUMNS if c not in OPTIONAL_COLUMNS]

# Sous-ensemble de colonnes qui doivent être coercibles en numérique.
NUMERIC_COLUMNS = [
    "result",
    "kills",
    "deaths",
    "assists",
    "csdiffat15",
    "golddiffat15",
    "xpdiffat15",
    "dpm",
    "cspm",
    "vspm",
    "damageshare",
    "earnedgoldshare",
    "gamelength",
]

# Taille d'échantillon pour le contrôle de type (les CSV font ~150k lignes).
_DTYPE_SAMPLE_SIZE = 1000


class SchemaValidationError(ValueError):
    """Le CSV brut ne respecte pas le schéma Oracle's Elixir attendu."""


def validate_raw_schema(df: pd.DataFrame, source: str = "<inconnu>") -> None:
    """
    Valide qu'un DataFrame brut Oracle's Elixir contient les colonnes requises
    et que les colonnes numériques sont coercibles.

    Args:
        df: DataFrame issu de pd.read_csv sur un export Oracle's Elixir.
        source: Nom du fichier (ou autre identifiant) pour le message d'erreur.

    Raises:
        SchemaValidationError: liste TOUTES les colonnes requises manquantes
            et les colonnes numériques non coercibles, en une seule erreur.
    """
    problems: list[str] = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        problems.append(
            f"colonnes requises manquantes : {sorted(missing_required)}"
        )

    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_optional:
        logger.warning(
            f"⚠️  {source} : colonnes optionnelles absentes "
            f"{sorted(missing_optional)} (recalculées en feature engineering)"
        )

    non_numeric = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue  # déjà signalée dans missing_required
        sample = df[col].dropna().head(_DTYPE_SAMPLE_SIZE)
        if sample.empty:
            continue
        coerced = pd.to_numeric(sample, errors="coerce")
        if coerced.isna().any():
            non_numeric.append(col)
    if non_numeric:
        problems.append(
            f"colonnes numériques non coercibles : {sorted(non_numeric)}"
        )

    if problems:
        raise SchemaValidationError(
            f"Schéma invalide pour {source} — dérive probable du format "
            f"Oracle's Elixir : " + " ; ".join(problems)
        )

    logger.info(f"✅ Schéma validé pour {source}")
