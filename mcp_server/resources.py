"""Ressources MCP : le dictionnaire de données, lu depuis les modèles dbt.

C'est ce qui sépare un serveur MCP utile d'un simple emballage d'API. Un agent
qui reçoit une colonne `gold_diff_at_15_zscore` sans explication invente son
interprétation ; avec la description, il sait que c'est un écart-type et non
une quantité d'or.

La documentation n'est pas recopiée ici : elle est extraite des fichiers YAML
de dbt, qui sont la source de vérité. Une description modifiée dans dbt se
propage donc au serveur MCP sans intervention, et il n'existe pas deux versions
de la même explication qui divergent avec le temps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# mcp_server/resources.py -> mcp_server/ -> racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBT_MODELS_DIR = PROJECT_ROOT / "dbt" / "models"


def _load_yaml_files() -> list[dict[str, Any]]:
    """Charge tous les fichiers de propriétés dbt du projet."""
    documents: list[dict[str, Any]] = []
    if not DBT_MODELS_DIR.exists():
        return documents

    for path in sorted(DBT_MODELS_DIR.rglob("*.yml")):
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(content, dict):
            documents.append(content)
    return documents


def build_data_dictionary() -> dict[str, Any]:
    """Construit le dictionnaire de données à partir des modèles dbt."""
    models: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for document in _load_yaml_files():
        for model in document.get("models", []) or []:
            columns = []
            for column in model.get("columns", []) or []:
                tests = column.get("tests", []) or []
                constraints = []
                for test in tests:
                    if isinstance(test, str):
                        constraints.append(test)
                    elif isinstance(test, dict):
                        for test_name, config in test.items():
                            arguments = (config or {}).get("arguments", config) or {}
                            values = arguments.get("values")
                            constraints.append(f"{test_name}: {values}" if values else test_name)

                columns.append(
                    {
                        "name": column.get("name"),
                        "description": (column.get("description") or "").strip(),
                        "constraints": constraints,
                    }
                )

            models.append(
                {
                    "name": model.get("name"),
                    "description": (model.get("description") or "").strip(),
                    "columns": columns,
                }
            )

        for source in document.get("sources", []) or []:
            sources.append(
                {
                    "name": source.get("name"),
                    "description": (source.get("description") or "").strip(),
                    "tables": [table.get("name") for table in source.get("tables", []) or []],
                }
            )

    return {
        "how_to_read": {
            "talent_score": (
                "Score du modèle, de 0 à 100 environ. Distribution très asymétrique, médiane "
                "autour de 2,6 : comparer par le percentile, jamais par l'écart brut."
            ),
            "z_scores": (
                "Toute colonne suffixée _zscore est exprimée en écarts-types par rapport aux "
                "autres joueurs du même poste dans la même ligue. 0 est la moyenne."
            ),
            "has_reliable_sample": (
                "Vrai à partir de 10 matchs. En dessous, le score n'est pas interprétable et ne "
                "doit pas être présenté comme un signal."
            ),
            "grain": (
                "Une ligne par joueur, saison, split et équipe. Un joueur qui change d'équipe en "
                "cours de split a donc deux lignes pour la même saison."
            ),
        },
        "models": models,
        "sources": sources,
    }


def register(server) -> None:
    """Déclare les ressources sur le serveur MCP."""

    @server.resource(
        "scouting://data-dictionary",
        name="Dictionnaire de données",
        description=(
            "Description de chaque modèle et de chaque colonne, extraite des modèles dbt, "
            "avec les règles de lecture des scores. À consulter avant d'interpréter des chiffres."
        ),
        mime_type="application/json",
    )
    def data_dictionary() -> str:
        return json.dumps(build_data_dictionary(), ensure_ascii=False, indent=2)

    @server.resource(
        "scouting://methodology",
        name="Méthodologie et limites",
        description="Comment les scores et les archétypes sont produits, et ce qu'ils ne disent pas.",
        mime_type="text/markdown",
    )
    def methodology() -> str:
        readme = PROJECT_ROOT / "README.md"
        if not readme.exists():
            return "# Méthodologie\n\nREADME introuvable."

        text = readme.read_text(encoding="utf-8")
        # On ne renvoie que la section des limites : c'est la partie qu'un agent
        # doit connaître pour ne pas surinterpréter, et envoyer 540 lignes de
        # README à chaque lecture serait du gaspillage de contexte.
        marker = "## ⚠️ Limites & Biais"
        if marker in text:
            section = text[text.index(marker) :]
            end = section.find("\n## ", 1)
            return section[:end] if end > 0 else section
        return text[:4000]
