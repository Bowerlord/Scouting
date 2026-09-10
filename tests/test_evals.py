"""Tests du banc lui-même.

Un banc d'évaluation non testé est le pire des deux mondes : il produit des
chiffres, donc on les croit, et rien ne garantit qu'ils veulent dire quelque
chose. Ces tests vérifient surtout que la notation ne se laisse pas tromper.
"""

from __future__ import annotations

import pytest

from agent.prompts import REFUSAL_MARKER
from evals.run import percentile, render_report, summarize
from evals.scoring import Verdict, answer_signature, extract_numbers, grade, normalize
from evals.truth import Question, load_questions

# ── Jeu de référence ──────────────────────────────────────────────────────────


def test_le_jeu_de_reference_a_ses_trois_familles_et_ses_quarante_questions():
    questions = load_questions()
    assert len(questions) == 40

    familles = {q.family for q in questions}
    assert familles == {"factuelle", "comparative", "piege"}


def test_chaque_question_non_piegee_a_une_verite_calculee():
    for question in load_questions():
        if question.is_trap:
            assert question.expected is None
            assert question.why, f"{question.id} : un piège doit dire pourquoi il en est un"
        else:
            assert question.expected is not None, f"{question.id} : vérité terrain absente"
            assert question.sql


def test_les_identifiants_sont_uniques():
    identifiants = [q.id for q in load_questions()]
    assert len(identifiants) == len(set(identifiants))


# ── Notation ──────────────────────────────────────────────────────────────────


def _question(**kwargs) -> Question:
    base = {
        "id": "T01",
        "family": "factuelle",
        "question": "q",
        "expects": "number",
        "tolerance": 0.0,
        "expected": 83.0,
    }
    base.update(kwargs)
    return Question(**base)


def test_un_nombre_juste_est_reconnu_meme_avec_un_separateur_de_milliers():
    assert grade(_question(expected=2231.0), "Il y a 2 231 lignes joueur.").verdict == Verdict.JUSTE
    assert grade(_question(expected=2231.0), "Il y a 2231 lignes.").verdict == Verdict.JUSTE
    assert grade(_question(expected=2231.0), "Il y en a 2.231.").verdict == Verdict.JUSTE


def test_un_nombre_faux_est_compte_faux():
    assert grade(_question(expected=83.0), "Il y a 91 joueurs.").verdict == Verdict.FAUX


def test_un_nom_est_reconnu_sans_tenir_compte_des_accents_ni_de_la_casse():
    question = _question(expects="name", expected="zoelys")
    assert grade(question, "C'est ZOÉLYS qui domine.").verdict == Verdict.JUSTE


def test_une_liste_de_noms_exige_tous_les_noms():
    question = _question(expects="names", expected=["a1", "b2", "c3"])
    assert grade(question, "Dans l'ordre : a1, b2, c3.").verdict == Verdict.JUSTE
    assert grade(question, "Dans l'ordre : a1, b2.").verdict == Verdict.FAUX


def test_un_refus_sur_un_piege_est_la_bonne_reponse():
    piege = _question(expects="refusal", family="piege", expected=None, why="hors périmètre")
    assert grade(piege, f"{REFUSAL_MARKER} la LEC n'est pas couverte").verdict == Verdict.REFUS_ATTENDU


def test_repondre_a_un_piege_est_une_hallucination():
    piege = _question(expects="refusal", family="piege", expected=None, why="hors périmètre")
    verdict = grade(piege, "Le meilleur mid de LEC est Caps.").verdict
    assert verdict == Verdict.HALLUCINATION


def test_refuser_une_question_repondable_est_compte_a_part():
    """C'est la mesure qui empêche de tricher en refusant tout."""
    verdict = grade(_question(expected=83.0), f"{REFUSAL_MARKER} je ne sais pas").verdict
    assert verdict == Verdict.REFUS_A_TORT


def test_une_esquive_sans_le_marqueur_nest_pas_un_refus():
    piege = _question(expects="refusal", family="piege", expected=None, why="hors périmètre")
    grade_obtenu = grade(piege, "Je ne suis pas certain, mais c'est probablement Caps.")
    assert grade_obtenu.verdict == Verdict.HALLUCINATION


def test_un_type_attendu_inconnu_est_une_erreur_de_configuration():
    with pytest.raises(ValueError):
        grade(_question(expects="hologramme"), "peu importe")


# ── Outils de mesure ──────────────────────────────────────────────────────────


def test_extraction_des_nombres():
    assert extract_numbers("83 joueurs, 2 231 lignes, score 2,6") == [83.0, 2231.0, 2.6]


def test_normalisation():
    assert normalize("  ZOÉLYS   est   là ") == "zoelys est la"


def test_la_signature_ignore_la_reformulation_mais_pas_le_chiffre():
    a = answer_signature("Il y a 83 lignes joueur.")
    b = answer_signature("Les lignes joueur sont au nombre de 83.")
    c = answer_signature("Il y a 84 lignes joueur.")
    assert a == b, "une reformulation ne doit pas compter comme une instabilité"
    assert a != c, "un chiffre différent doit compter comme une instabilité"


def test_la_signature_distingue_un_refus_dune_reponse():
    assert answer_signature("DONNEES_INSUFFISANTES rien à dire") != answer_signature("rien à dire")


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([], 0.5) == 0.0


# ── Rapport ───────────────────────────────────────────────────────────────────


def _summary_minimal() -> dict:
    from agent.agent import Answer
    from evals.run import QuestionOutcome
    from evals.scoring import Grade

    question = _question(expects="number", expected=83.0)
    outcome = QuestionOutcome(question=question)
    outcome.answers.append(Answer(question="q", text="83", model="claude-sonnet-5"))
    outcome.grades.append(Grade("T01", "factuelle", Verdict.JUSTE, 83.0, "83"))

    return summarize([outcome], runs=1, model="m", provider_name="fake", wall_seconds=1.0, missing_replays=0)


def test_le_rapport_contient_les_mesures_qui_comptent():
    rapport = render_report(_summary_minimal(), previous=None)
    for attendu in (
        "Exactitude globale",
        "Refus à tort",
        "Hallucinations",
        "Instabilité du verdict",
        "Coût par question",
    ):
        assert attendu in rapport


def test_le_rapport_affiche_lecart_avec_le_precedent():
    precedent = _summary_minimal()
    precedent["exactitude"] = 0.5
    rapport = render_report(_summary_minimal(), previous=precedent)
    assert "pt" in rapport, "un écart chiffré doit apparaître, sinon la non-régression n'est pas lisible"


def test_le_regime_de_rejeu_est_annonce_dans_le_rapport():
    resume = _summary_minimal()
    resume["fournisseur"] = "cassette"
    assert "rejeu" in render_report(resume, previous=None)
