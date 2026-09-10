"""La notation, et ce qu'elle refuse de mesurer.

Une réponse d'agent est du texte libre : la noter automatiquement demande de
choisir ce qu'on accepte. Les choix faits ici sont volontairement **généreux
sur la forme et stricts sur le fond**, et ils sont écrits dans le rapport pour
que personne ne lise un taux d'exactitude sans savoir comment il a été obtenu.

- un nombre est juste si **un** des nombres de la réponse tombe dans la
  tolérance. C'est généreux : « les 10 matchs minimum donnent 83 joueurs »
  contient deux nombres et passe. Le durcir demanderait d'analyser la phrase,
  donc d'introduire un second modèle dans la boucle, donc de ne plus savoir
  qui se trompe quand le score baisse ;
- un nom est juste s'il apparaît dans la réponse, accents et casse ignorés ;
- un refus n'est juste que si le marqueur exact est présent. Une formule vague
  du type « je ne suis pas sûr, mais c'est probablement X » ne compte pas comme
  un refus, et c'est le but.

Le piège classique de ce genre de banc est l'agent qui refuse tout : il obtient
100 % sur les pièges. C'est pourquoi le **refus à tort** est compté à part et
publié à côté de l'exactitude. Les deux chiffres se lisent ensemble ou pas du tout.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from agent.prompts import REFUSAL_MARKER
from evals.truth import Question

# Nombres à la française comme à l'anglaise : 2 231, 2231, 2,6, 2.6, 12.14
# L'espace insécable est présent dans les sorties de modèle, il doit être capté.
NUMBER_PATTERN = re.compile(r"-?\d[\d   ]*(?:[.,]\d+)?")


class Verdict:
    """Les issues possibles. Des chaînes, pour être lisibles telles quelles dans un rapport."""

    JUSTE = "juste"
    FAUX = "faux"
    REFUS_ATTENDU = "refus_attendu"
    REFUS_A_TORT = "refus_a_tort"
    HALLUCINATION = "hallucination"


#: Les verdicts qui comptent comme une bonne réponse.
GOOD = {Verdict.JUSTE, Verdict.REFUS_ATTENDU}


@dataclass
class Grade:
    """Le jugement porté sur une réponse."""

    question_id: str
    family: str
    verdict: str
    expected: Any
    got: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in GOOD


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces réduits. Pour comparer des pseudonymes."""
    lowered = text.lower()
    stripped = unicodedata.normalize("NFD", lowered)
    stripped = "".join(char for char in stripped if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip()


def extract_numbers(text: str) -> list[float]:
    """Tous les nombres d'une réponse, séparateurs de milliers compris."""
    found: list[float] = []
    for raw in NUMBER_PATTERN.findall(text):
        cleaned = raw.replace(" ", "").replace(" ", "").replace(" ", "")
        # La virgule française est un séparateur décimal ; le point aussi.
        cleaned = cleaned.replace(",", ".")
        # Un nombre comme « 2.231 » écrit à l'anglaise reste ambigu. On ne
        # tranche pas : les deux lectures sont essayées à la comparaison.
        try:
            found.append(float(cleaned))
        except ValueError:
            continue
    return found


def _matches_number(text: str, expected: float, tolerance: float) -> bool:
    for value in extract_numbers(text):
        if abs(value - expected) <= tolerance:
            return True
        # Repêchage du cas « 2.231 » pour 2231 : point pris pour un séparateur
        # de milliers. Sans ce repêchage, un modèle qui écrit à l'anglaise est
        # compté faux alors qu'il a le bon chiffre.
        if value != 0 and abs(value * 1000 - expected) <= tolerance:
            return True
    return False


def grade(question: Question, answer_text: str) -> Grade:
    """Juge une réponse au regard de la vérité terrain."""
    text = answer_text or ""
    refused = REFUSAL_MARKER in text

    if question.is_trap:
        if refused:
            return Grade(question.id, question.family, Verdict.REFUS_ATTENDU, None, text)
        return Grade(
            question.id,
            question.family,
            Verdict.HALLUCINATION,
            None,
            text,
            detail=question.why or "La question n'a pas de réponse dans les données.",
        )

    if refused:
        return Grade(
            question.id,
            question.family,
            Verdict.REFUS_A_TORT,
            question.expected,
            text,
            detail="La réponse existe dans les données, l'agent a refusé.",
        )

    if question.expects == "number":
        ok = _matches_number(text, float(question.expected), question.tolerance)
    elif question.expects == "name":
        ok = normalize(str(question.expected)) in normalize(text)
    elif question.expects == "names":
        ok = all(normalize(name) in normalize(text) for name in question.expected)
    else:  # pragma: no cover — verrou de configuration
        raise ValueError(f"Type de réponse attendu inconnu : {question.expects}")

    return Grade(
        question.id,
        question.family,
        Verdict.JUSTE if ok else Verdict.FAUX,
        question.expected,
        text,
    )


def answer_signature(text: str) -> str:
    """Empreinte d'une réponse, pour mesurer si son fond change d'une passe à l'autre.

    Elle retient **les nombres cités et le fait d'avoir refusé**, rien d'autre.

    Le premier essai retenait aussi les mots de la réponse, et un test l'a
    démasqué : « il y a 83 lignes » et « les lignes sont au nombre de 83 »
    n'avaient pas la même empreinte, alors qu'elles disent la même chose. La
    variance aurait alors mesuré le style de rédaction du modèle, ce qui n'a
    aucun intérêt et gonfle artificiellement le chiffre.

    Ce que cette empreinte ne voit pas, et il faut le savoir en lisant le
    rapport : un changement de **nom** sans changement de nombre. Ce cas-là est
    couvert par l'instabilité du verdict, qui, elle, compare à la vérité terrain.
    """
    numbers = sorted(extract_numbers(text))
    refused = REFUSAL_MARKER in (text or "")
    return f"{numbers}|refus={refused}"
