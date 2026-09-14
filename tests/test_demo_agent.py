"""Tests de la logique de la page « Agent » du dashboard.

La page est publique : n'importe qui peut y taper n'importe quoi, et un modèle
y répond. Ces tests vérifient ce qui protège la page, pas son apparence.
"""

from __future__ import annotations

import json
from datetime import date

from app.utils import demo_agent as demo
from evals.scoring import Verdict
from evals.truth import load_questions


def test_les_questions_proposees_existent_dans_le_banc_et_dans_la_bonne_famille():
    questions = {q.id: q for q in load_questions()}
    for famille, identifiants in demo.QUESTIONS_PROPOSEES.items():
        for identifiant in identifiants:
            assert identifiant in questions, identifiant
            assert questions[identifiant].family == famille


def test_chaque_verdict_du_banc_a_un_libelle():
    verdicts = {valeur for nom, valeur in vars(Verdict).items() if not nom.startswith("_")}
    assert verdicts <= set(demo.VERDICTS)


def test_le_quota_journalier_bloque_puis_se_rouvre_le_lendemain():
    quota = demo.QuotaJournalier(par_jour=2)
    lundi, mardi = date(2026, 9, 14), date(2026, 9, 15)
    assert quota.consommer(lundi)
    assert quota.consommer(lundi)
    assert not quota.consommer(lundi)
    assert quota.consommer(mardi)


def test_une_question_vide_ou_trop_longue_est_refusee():
    assert demo.nettoyer_question("   ")[0] is None
    assert demo.nettoyer_question(None)[0] is None
    assert demo.nettoyer_question("x" * (demo.LONGUEUR_MAX_QUESTION + 1))[0] is None
    assert demo.nettoyer_question("  meilleur   mid ?  ") == ("meilleur mid ?", None)


def test_aucun_texte_affiche_ne_peut_injecter_du_html():
    hostile = '<script>alert("x")</script>'
    assert "<script>" not in demo.texte(hostile, "es-reponse")
    assert "<script>" not in demo.ligne_trace(1, hostile, {"search": hostile}, 12.0, False)
    assert "<script>" not in demo.bloc_verdict(hostile, 1, 1000.0, None)


def test_la_verite_se_lit_comme_un_lecteur_l_attend():
    assert demo.format_verite("number", 2231.0) == "2 231"
    assert demo.format_verite("number", 2.5) == "2,50"
    assert demo.format_verite("names", ["a", "b"]) == "a, b"
    assert demo.format_verite("refusal", None) == "Aucune réponse dans les données"


def test_un_cout_inconnu_n_est_jamais_affiche_comme_nul():
    assert demo.format_cout(None) == "coût non chiffré"
    assert demo.format_cout(0.00066) == "0,0007 €"


def _rapport(dossier, nom, **champs):
    (dossier / f"{nom}.json").write_text(json.dumps(champs))


def test_le_resume_ne_retient_qu_un_rapport_multi_passes_sans_pannes(tmp_path):
    _rapport(tmp_path, "2026-09-10-1449", fournisseur="groq", passes=5, erreurs_fournisseur=0.0, exactitude=0.8)
    _rapport(tmp_path, "2026-09-14-0840", fournisseur="groq", passes=1, erreurs_fournisseur=0.0, exactitude=0.85)
    _rapport(tmp_path, "2026-09-14-0900", fournisseur="heuristique", passes=5, exactitude=0.7)
    _rapport(tmp_path, "2026-09-14-1228", fournisseur="groq", passes=5, erreurs_fournisseur=0.7, exactitude=0.275)
    (tmp_path / "2026-09-14-1300.json").write_text("{pas du json")
    assert demo.resume_banc(tmp_path)["exactitude"] == 0.8
    assert demo.resume_banc(tmp_path / "vide") is None


def test_un_quota_depasse_est_reconnu():
    assert demo.quota_fournisseur_atteint("RateLimitError: Error code: 429 - tokens per day")
    assert not demo.quota_fournisseur_atteint("APIConnectionError: timeout")
    assert not demo.quota_fournisseur_atteint(None)
