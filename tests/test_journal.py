"""Tests du journal des passes de nuit et de son agrégation."""

from __future__ import annotations

from agent.agent import Answer
from evals import journal
from evals.aggregate import rapport
from evals.run import QuestionOutcome
from evals.scoring import Grade, Verdict
from evals.truth import load_questions


def _outcome(question, verdicts: list[str]) -> QuestionOutcome:
    outcome = QuestionOutcome(question=question)
    for verdict in verdicts:
        outcome.answers.append(Answer(question=question.question, text=f"réponse {verdict}", model="m"))
        outcome.grades.append(
            Grade(question_id=question.id, family=question.family, verdict=verdict, expected=None, got="")
        )
    return outcome


def test_les_tranches_couvrent_tout_le_jeu_sans_doublon() -> None:
    questions = load_questions()
    t1, t2 = journal.tranche(questions, 1, 2), journal.tranche(questions, 2, 2)
    assert len(t1) + len(t2) == len(questions)
    assert {q.id for q in t1}.isdisjoint({q.id for q in t2})


def test_chaque_tranche_garde_les_trois_familles() -> None:
    questions = load_questions()
    for numero in (1, 2):
        assert {q.family for q in journal.tranche(questions, numero, 2)} == {q.family for q in questions}


def test_aller_retour_par_le_journal(tmp_path) -> None:
    questions = load_questions()[:3]
    chemin = tmp_path / "journal.jsonl"
    outcomes = [_outcome(q, [Verdict.JUSTE]) for q in questions]
    assert journal.ajouter(chemin, outcomes, fournisseur="groq", modele="m") == 3

    relus, meta = journal.lire(chemin, questions)
    assert [len(o.grades) for o in relus] == [1, 1, 1]
    assert meta["modele"] == "m" and meta["fournisseur"] == "groq"


def test_une_panne_ne_compte_pas_comme_une_passe(tmp_path) -> None:
    question = load_questions()[0]
    chemin = tmp_path / "journal.jsonl"
    journal.ajouter(chemin, [_outcome(question, [Verdict.ERREUR_FOURNISSEUR])], "groq", "m")
    journal.ajouter(chemin, [_outcome(question, [Verdict.JUSTE])], "groq", "m")

    relus, meta = journal.lire(chemin, [question])
    assert len(relus[0].grades) == 1
    assert meta["pannes_ecartees"] == 1


def test_la_prochaine_tranche_rattrape_la_plus_en_retard() -> None:
    questions = load_questions()[:4]
    outcomes = [_outcome(q, [Verdict.JUSTE] * n) for q, n in zip(questions, [2, 1, 2, 2])]
    # Tranche 2 = questions d'indice impair : la deuxième n'a qu'une passe.
    assert journal.prochaine_tranche(outcomes, 2) == 2


def test_le_rapport_ne_se_dit_complet_qu_a_la_cible() -> None:
    questions = load_questions()[:2]
    outcomes = [_outcome(questions[0], [Verdict.JUSTE] * 5), _outcome(questions[1], [Verdict.JUSTE] * 4)]
    meta = {"nuits": 9, "pannes_ecartees": 0, "modele": "m", "fournisseur": "groq"}
    summary, texte = rapport(outcomes, meta, cible=5)
    assert not summary["complet"]
    assert "Mesure en cours" in texte


def test_l_instabilite_se_lit_entre_les_nuits() -> None:
    questions = load_questions()[:2]
    outcomes = [
        _outcome(questions[0], [Verdict.JUSTE, Verdict.FAUX]),
        _outcome(questions[1], [Verdict.JUSTE, Verdict.JUSTE]),
    ]
    meta = {"nuits": 2, "pannes_ecartees": 0, "modele": "m", "fournisseur": "groq"}
    summary, _ = rapport(outcomes, meta, cible=2)
    assert summary["instabilite_verdict"] == 0.5
    assert summary["complet"]


def test_un_journal_absent_se_lit_comme_en_cours(tmp_path) -> None:
    outcomes, meta = journal.lire(tmp_path / "absent.jsonl", load_questions())
    summary, _ = rapport(outcomes, meta, cible=5)
    assert not summary["complet"]
    assert journal.prochaine_tranche(outcomes, 2) == 1
