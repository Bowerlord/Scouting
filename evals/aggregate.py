"""Agrège le journal des passes de nuit en un rapport de variance.

    python -m evals.aggregate evals/journal/gpt-oss-120b.jsonl --out evals/reports/variance
    python -m evals.aggregate evals/journal/gpt-oss-120b.jsonl --status   # complet ou en_cours
    python -m evals.aggregate evals/journal/gpt-oss-120b.jsonl --next-tranche 2

Le rapport a la même forme que celui d'une exécution à plusieurs passes. Il ne
se déclare complet que quand **chaque** question a le nombre de passes valides
visé : une moyenne sur des questions inégalement mesurées serait un chiffre de
plus, pas une mesure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals import journal
from evals.run import _forcer_utf8, render_report, summarize
from evals.truth import load_questions


def rapport(outcomes: list, meta: dict, cible: int) -> tuple[dict, str]:
    mesurees = [o for o in outcomes if o.grades]
    passes = min((len(o.grades) for o in outcomes), default=0)
    if not mesurees:
        return {"passes": 0, "complet": False, **meta}, "> Journal vide : aucune passe de nuit encore.\n"
    summary = summarize(
        outcomes=mesurees,
        runs=max(passes, 1) if passes else 1,
        model=meta["modele"],
        provider_name=meta["fournisseur"],
        wall_seconds=0.0,
        missing_replays=0,
    )
    # summarize ne sait pas que les passes viennent de nuits différentes.
    summary["passes"] = passes
    summary["instabilite_verdict"] = sum(1 for o in mesurees if o.verdict_unstable) / max(len(mesurees), 1)
    summary["instabilite_reponse"] = sum(1 for o in mesurees if o.answer_unstable) / max(len(mesurees), 1)
    summary["nuits"] = meta["nuits"]
    summary["pannes_ecartees"] = meta["pannes_ecartees"]
    summary["complet"] = passes >= cible

    etat = (
        f"**Mesure complète** : {passes} passes valides par question."
        if summary["complet"]
        else f"**Mesure en cours** : {passes} passe(s) valide(s) sur {cible} pour la question la moins avancée. "
        "Ces chiffres ne sont pas encore publiables."
    )
    entete = (
        f"> Agrégat du journal des passes de nuit : {meta['nuits']} nuit(s), "
        f"{meta['pannes_ecartees']} réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).\n>\n"
        f"> {etat}\n\n"
    )
    return summary, entete + render_report(summary, None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agrège le journal des passes de nuit.")
    parser.add_argument("journal", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Répertoire du rapport de variance")
    parser.add_argument("--target", type=int, default=5, help="Passes valides visées par question")
    parser.add_argument("--status", action="store_true", help="Affiche seulement « complet » ou « en_cours »")
    parser.add_argument(
        "--next-tranche", type=int, default=None, metavar="N", help="Affiche la tranche à passer, sur N"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outcomes, meta = journal.lire(args.journal, load_questions())

    if args.next_tranche:
        print(journal.prochaine_tranche(outcomes, args.next_tranche))
        return 0

    summary, texte = rapport(outcomes, meta, args.target)
    if args.status:
        print("complet" if summary["complet"] else "en_cours")
        return 0

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "variance.md").write_text(texte, encoding="utf-8")
        (args.out / "variance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(texte)
    return 0


if __name__ == "__main__":
    _forcer_utf8()
    sys.exit(main())
