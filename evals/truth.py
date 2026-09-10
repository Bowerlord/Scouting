"""La vérité terrain, calculée en SQL sur les fichiers du pipeline.

Le point important est l'indépendance : l'agent lit les données par l'API, ce
module les lit par DuckDB, directement sur les CSV produits par le pipeline.
Aucune ligne de code n'est partagée entre les deux chemins. Une réponse jugée
juste l'est donc par un calcul qui ne dépend pas de ce qu'on évalue.

Corollaire assumé : si le pipeline change de forme, ce sont ces requêtes qui
cassent en premier, et c'est voulu. Un banc qui continue de passer au vert quand
les données ont changé sous lui ne mesure plus rien.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import yaml

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
ROOT = Path(__file__).resolve().parents[1]


class TruthError(RuntimeError):
    """La vérité terrain n'a pas pu être calculée."""


@dataclass
class Question:
    """Une question du jeu de référence, et ce qu'on attend d'elle."""

    id: str
    family: str
    question: str
    expects: str
    sql: str | None = None
    tolerance: float = 0.0
    why: str | None = None
    expected: Any = None

    @property
    def is_trap(self) -> bool:
        return self.expects == "refusal"


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    return yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))


def _connection() -> duckdb.DuckDBPyConnection:
    meta = _config()["meta"]
    players = ROOT / meta["players_csv"]
    clusters = ROOT / meta["clusters_csv"]

    for path in (players, clusters):
        if not path.exists():
            raise TruthError(
                f"Fichier de résultats introuvable : {path}. "
                "Lancer le pipeline ou récupérer les snapshots avant d'évaluer."
            )

    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW players AS SELECT * FROM read_csv_auto('{players.as_posix()}')")
    connection.execute(f"CREATE VIEW clusters AS SELECT * FROM read_csv_auto('{clusters.as_posix()}')")
    return connection


def _resolve(connection: duckdb.DuckDBPyConnection, question: Question) -> Any:
    if question.is_trap:
        return None

    if not question.sql:
        raise TruthError(f"{question.id} : question non piégée sans requête de vérité.")

    rows = connection.execute(question.sql).fetchall()
    if not rows:
        raise TruthError(f"{question.id} : la requête de vérité ne renvoie aucune ligne.")

    values = [row[0] for row in rows]

    if question.expects == "names":
        return [str(value) for value in values]
    if question.expects == "number":
        return float(values[0])
    return str(values[0])


@lru_cache(maxsize=1)
def load_questions() -> tuple[Question, ...]:
    """Charge les questions et calcule leur réponse attendue.

    Le calcul est fait une fois au chargement, pas à la notation : une vérité
    qui se recalcule à chaque comparaison finirait par mesurer la vitesse de
    DuckDB plutôt que celle de l'agent.
    """
    config = _config()
    connection = _connection()

    questions: list[Question] = []
    try:
        for raw in config["questions"]:
            question = Question(
                id=raw["id"],
                family=raw["family"],
                question=raw["question"],
                expects=raw["expects"],
                sql=raw.get("sql"),
                tolerance=float(raw.get("tolerance", 0.0)),
                why=raw.get("why"),
            )
            question.expected = _resolve(connection, question)
            questions.append(question)
    finally:
        connection.close()

    _check_unique_ids(questions)
    return tuple(questions)


def _check_unique_ids(questions: list[Question]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise TruthError(f"Identifiant de question en double : {question.id}")
        seen.add(question.id)


def families() -> list[str]:
    """Les familles présentes, dans l'ordre où elles apparaissent."""
    ordered: list[str] = []
    for question in load_questions():
        if question.family not in ordered:
            ordered.append(question.family)
    return ordered


def main() -> None:  # pragma: no cover — inspection manuelle
    """`python -m evals.truth` — affiche les vérités calculées, pour les relire."""
    for question in load_questions():
        expected = "refus attendu" if question.is_trap else question.expected
        print(f"{question.id}  [{question.family:<11}]  {expected}")
        print(f"          {question.question}")


if __name__ == "__main__":  # pragma: no cover
    main()
