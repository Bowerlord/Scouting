"""Le journal des passes : mesurer la variance sur plusieurs nuits.

Pourquoi ce module existe. Cinq passes du banc coûtent environ 800 000 jetons,
et le palier gratuit de Groq en accorde 200 000 par jour et par modèle, partagés
avec la démo. Relevé le 2026-09-14 : lancées d'un coup, les passes 3 à 5 n'ont
contenu que des refus du fournisseur. La variance se mesure donc en plusieurs
fois, une demi-passe par nuit, et ce journal garde chaque réponse d'une nuit à
l'autre pour que l'agrégat soit calculé comme si tout avait tourné d'un trait.

Deux règles :

- **une panne du fournisseur ne compte pas comme une passe.** Elle est gardée
  dans le journal, comptée à part dans le rapport, et la question sera reposée
  une autre nuit. Sinon, une nuit de quota épuisé ferait passer l'agent pour
  instable ;
- **chaque nuit prend la tranche la moins avancée**, et non une alternance fixe :
  une nuit ratée est rattrapée d'elle-même.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.agent import Answer, ToolInvocation
from agent.providers import Usage
from evals.scoring import Grade, Verdict
from evals.truth import Question


def tranche(questions: list[Question], numero: int, total: int) -> list[Question]:
    """Les questions de la tranche `numero` sur `total`, par entrelacement.

    L'entrelacement garde les trois familles dans chaque tranche.
    """
    if not 1 <= numero <= total:
        raise ValueError(f"Tranche {numero}/{total} invalide")
    return [q for i, q in enumerate(questions) if i % total == numero - 1]


def ajouter(chemin: Path, outcomes: list[Any], fournisseur: str, modele: str) -> int:
    """Ajoute au journal une ligne par réponse. Renvoie le nombre de lignes écrites."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lignes = 0
    with chemin.open("a", encoding="utf-8") as fichier:
        for outcome in outcomes:
            for answer, note in zip(outcome.answers, outcome.grades):
                ligne = {
                    "horodatage": horodatage,
                    "fournisseur": fournisseur,
                    "modele": modele,
                    "question_id": outcome.question.id,
                    "reponse": answer.to_dict(),
                    "verdict": {
                        "verdict": note.verdict,
                        "expected": note.expected,
                        "got": note.got,
                        "detail": note.detail,
                    },
                }
                fichier.write(json.dumps(ligne, ensure_ascii=False, default=str) + "\n")
                lignes += 1
    return lignes


def _reponse(donnees: dict[str, Any]) -> Answer:
    return Answer(
        question=donnees.get("question", ""),
        text=donnees.get("text", ""),
        trace=[ToolInvocation(**appel) for appel in donnees.get("trace", [])],
        usage=Usage(**donnees.get("usage", {})),
        latency_ms=donnees.get("latency_ms", 0.0),
        model=donnees.get("model", ""),
        steps=donnees.get("steps", 0),
        truncated=donnees.get("truncated", False),
        provider_error=donnees.get("provider_error"),
    )


def lire(chemin: Path, questions: list[Question]) -> tuple[list[Any], dict[str, Any]]:
    """Reconstruit les résultats par question depuis le journal.

    Renvoie les résultats (pannes écartées) et un résumé du journal : nombre de
    nuits, pannes écartées, modèle et fournisseur.
    """
    from evals.run import QuestionOutcome

    par_id = {q.id: QuestionOutcome(question=q) for q in questions}
    pannes = 0
    nuits: set[str] = set()
    modeles: Counter[str] = Counter()
    fournisseurs: Counter[str] = Counter()

    if chemin.exists():
        for brute in chemin.read_text(encoding="utf-8").splitlines():
            if not brute.strip():
                continue
            ligne = json.loads(brute)
            outcome = par_id.get(ligne["question_id"])
            if outcome is None:
                continue  # question retirée du jeu depuis
            nuits.add(ligne["horodatage"][:10])
            modeles[ligne["modele"]] += 1
            fournisseurs[ligne["fournisseur"]] += 1
            v = ligne["verdict"]
            if v["verdict"] == Verdict.ERREUR_FOURNISSEUR:
                pannes += 1
                continue
            outcome.answers.append(_reponse(ligne["reponse"]))
            outcome.grades.append(
                Grade(
                    question_id=ligne["question_id"],
                    family=outcome.question.family,
                    verdict=v["verdict"],
                    expected=v["expected"],
                    got=v["got"],
                    detail=v.get("detail", ""),
                )
            )

    meta = {
        "nuits": len(nuits),
        "pannes_ecartees": pannes,
        "modele": modeles.most_common(1)[0][0] if modeles else "inconnu",
        "fournisseur": fournisseurs.most_common(1)[0][0] if fournisseurs else "inconnu",
    }
    return list(par_id.values()), meta


def passes_par_question(outcomes: list[Any]) -> dict[str, int]:
    return {o.question.id: len(o.grades) for o in outcomes}


def prochaine_tranche(outcomes: list[Any], total: int) -> int:
    """La tranche dont la question la moins avancée a le moins de passes valides."""
    comptes = [len(o.grades) for o in outcomes]
    retards = [min(comptes[i::total]) if comptes[i::total] else 0 for i in range(total)]
    return retards.index(min(retards)) + 1
