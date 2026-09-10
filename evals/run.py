"""Le banc : exécute le jeu de référence N fois et publie un rapport chiffré.

    python -m evals.run                      # rejeu, gratuit, déterministe
    python -m evals.run --runs 5             # cinq passes, pour mesurer la variance
    python -m evals.run --provider anthropic --record   # exécution réelle, et enregistrement

Ce que le rapport dit, et que la plupart des démos taisent :

- l'exactitude **par famille**, parce qu'une moyenne globale cache qu'un agent
  peut être excellent sur les questions factuelles et catastrophique sur les pièges ;
- le **refus à tort**, sans lequel un agent qui refuse tout obtiendrait un score parfait ;
- la **variance** entre passes, c'est-à-dire la part de questions dont la réponse
  change d'une exécution à l'autre. C'est la mesure qui manque partout, et c'est
  celle qui décide si un système non déterministe est utilisable en production ;
- la **latence** et le **coût en euros par question**, parce qu'un agent juste
  mais à 40 centimes la question ne sera jamais déployé.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.agent import Answer, ScoutAgent
from agent.providers import CassetteProvider, ProviderError, get_provider
from evals.scoring import Grade, Verdict, answer_signature, grade
from evals.truth import Question, families, load_questions

REPORTS_DIR = Path(__file__).with_name("reports")


@dataclass
class QuestionOutcome:
    """Le résultat des N passes sur une même question."""

    question: Question
    grades: list[Grade] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(1 for g in self.grades if g.ok) / len(self.grades)

    @property
    def verdict_unstable(self) -> bool:
        """Le verdict change-t-il d'une passe à l'autre ?"""
        return len({g.verdict for g in self.grades}) > 1

    @property
    def answer_unstable(self) -> bool:
        """Le fond de la réponse change-t-il, à reformulation près ?"""
        return len({answer_signature(a.text) for a in self.answers}) > 1

    @property
    def dominant_verdict(self) -> str:
        return Counter(g.verdict for g in self.grades).most_common(1)[0][0]


def percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(share * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


# ── Exécution ─────────────────────────────────────────────────────────────────


def run_bench(
    runs: int,
    provider_name: str | None,
    model: str | None,
    record: bool,
    cassette: Path | None,
    only: str | None,
    verbose: bool,
) -> dict[str, Any]:
    questions = [q for q in load_questions() if not only or only.lower() in q.id.lower()]
    if not questions:
        raise SystemExit(f"Aucune question ne correspond au filtre : {only}")

    provider = get_provider(provider_name, model=model, record=record, cassette=cassette)
    agent = ScoutAgent(provider=provider)

    outcomes = [QuestionOutcome(question=q) for q in questions]
    started = time.perf_counter()
    missing_replays = 0

    for run_index in range(1, runs + 1):
        for outcome in outcomes:
            try:
                answer = agent.ask(outcome.question.question)
            except ProviderError as error:
                missing_replays += 1
                answer = Answer(
                    question=outcome.question.question,
                    text=f"[banc] {error}",
                    model=getattr(provider, "model", "inconnu"),
                )
            outcome.answers.append(answer)
            outcome.grades.append(grade(outcome.question, answer.text))

        if verbose:
            done = sum(1 for o in outcomes if o.grades[-1].ok)
            print(f"  passe {run_index}/{runs} : {done}/{len(outcomes)} bonnes réponses", file=sys.stderr)

    if isinstance(provider, CassetteProvider):
        provider.save()

    return summarize(
        outcomes=outcomes,
        runs=runs,
        model=getattr(provider, "model", "inconnu"),
        provider_name=getattr(provider, "name", "inconnu"),
        wall_seconds=time.perf_counter() - started,
        missing_replays=missing_replays,
    )


def summarize(
    outcomes: list[QuestionOutcome],
    runs: int,
    model: str,
    provider_name: str,
    wall_seconds: float,
    missing_replays: int,
) -> dict[str, Any]:
    all_grades = [g for o in outcomes for g in o.grades]
    all_answers = [a for o in outcomes for a in o.answers]

    latencies = [a.latency_ms for a in all_answers]
    # Un modèle dont le tarif n'est pas renseigné n'a pas un coût nul, il a un
    # coût inconnu. Additionner en traitant None comme zéro publierait un total
    # faux, ce qu'un banc de mesure ne doit jamais faire.
    couts = [a.cost_eur() for a in all_answers]
    tarif_connu = all(cout is not None for cout in couts)
    total_cost = sum(cout for cout in couts if cout is not None) if tarif_connu else None
    tool_calls = [len(a.trace) for a in all_answers]
    tool_names = Counter(name for a in all_answers for name in a.tool_names)
    tool_errors = sum(1 for a in all_answers for call in a.trace if call.is_error)

    by_family: dict[str, dict[str, Any]] = {}
    for family in families():
        family_outcomes = [o for o in outcomes if o.question.family == family]
        if not family_outcomes:
            continue
        grades = [g for o in family_outcomes for g in o.grades]
        by_family[family] = {
            "questions": len(family_outcomes),
            "exactitude": sum(1 for g in grades if g.ok) / len(grades),
            "verdicts": dict(Counter(g.verdict for g in grades)),
        }

    non_traps = [o for o in outcomes if not o.question.is_trap]
    traps = [o for o in outcomes if o.question.is_trap]

    return {
        "horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fournisseur": provider_name,
        "modele": model,
        "passes": runs,
        "questions": len(outcomes),
        "reponses": len(all_answers),
        "rejeux_manquants": missing_replays,
        "duree_s": round(wall_seconds, 1),
        "exactitude": sum(1 for g in all_grades if g.ok) / len(all_grades),
        "par_famille": by_family,
        "refus_a_tort": _rate(all_grades, Verdict.REFUS_A_TORT),
        "hallucinations": _rate(all_grades, Verdict.HALLUCINATION),
        "refus_correct": (
            sum(1 for o in traps for g in o.grades if g.verdict == Verdict.REFUS_ATTENDU)
            / max(sum(len(o.grades) for o in traps), 1)
        ),
        "instabilite_verdict": (
            sum(1 for o in outcomes if o.verdict_unstable) / len(outcomes) if runs > 1 else 0.0
        ),
        "instabilite_reponse": (
            sum(1 for o in outcomes if o.answer_unstable) / len(outcomes) if runs > 1 else 0.0
        ),
        "latence_p50_ms": round(percentile(latencies, 0.50), 1),
        "latence_p95_ms": round(percentile(latencies, 0.95), 1),
        "cout_total_eur": None if total_cost is None else round(total_cost, 4),
        "cout_par_question_eur": (
            None if total_cost is None else round(total_cost / max(len(all_answers), 1), 6)
        ),
        "jetons": {
            "entree": sum(a.usage.input_tokens for a in all_answers),
            "sortie": sum(a.usage.output_tokens for a in all_answers),
        },
        "outils_par_question": round(statistics.mean(tool_calls), 2) if tool_calls else 0.0,
        "outils_utilises": dict(tool_names.most_common()),
        "erreurs_outil": tool_errors,
        "echecs": [
            {
                "id": o.question.id,
                "famille": o.question.family,
                "question": o.question.question,
                "attendu": "refus" if o.question.is_trap else o.question.expected,
                "verdict": o.dominant_verdict,
                "taux_reussite": round(o.success_rate, 2),
                "reponse": o.answers[-1].text[:400],
                "outils": o.answers[-1].tool_names,
            }
            for o in outcomes
            if o.success_rate < 1.0
        ],
        "instables": [
            {"id": o.question.id, "famille": o.question.family, "taux_reussite": round(o.success_rate, 2)}
            for o in outcomes
            if o.verdict_unstable
        ],
        "_reponses_attendues": len(non_traps),
    }


def _rate(grades: list[Grade], verdict: str) -> float:
    return sum(1 for g in grades if g.verdict == verdict) / len(grades)


# ── Rapport ───────────────────────────────────────────────────────────────────


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %"


def _delta(current: float, previous: float | None, higher_is_better: bool = True) -> str:
    if previous is None:
        return "—"
    difference = (current - previous) * 100
    if abs(difference) < 0.05:
        return "="
    signe = "+" if difference > 0 else ""
    bon = difference > 0 if higher_is_better else difference < 0
    return f"{signe}{difference:.1f} pt {'✅' if bon else '⚠️'}"


def render_report(summary: dict[str, Any], previous: dict[str, Any] | None) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Banc d'évaluation de l'agent ERL Scout")
    add("")
    add(f"*Exécuté le {summary['horodatage']} — {summary['questions']} questions × {summary['passes']} passes*")
    add("")
    add(f"Modèle : `{summary['modele']}` · fournisseur : `{summary['fournisseur']}` · durée : {summary['duree_s']} s")
    if summary["fournisseur"] == "cassette":
        add("")
        add(
            "> Régime **rejeu** : les réponses du modèle viennent d'enregistrements. "
            "Le banc mesure ici la non-régression du code et du prompt, à modèle figé. "
            "La variabilité propre au modèle ne se mesure qu'en exécution réelle."
        )
    if summary["rejeux_manquants"]:
        add("")
        add(
            f"> ⚠️ **{summary['rejeux_manquants']} rejeux manquants.** "
            "Refaire les enregistrements : `python -m evals.run --provider anthropic --record`"
        )
    add("")

    add("## Les chiffres")
    add("")
    add("| Mesure | Valeur | Écart au rapport précédent |")
    add("|---|---|---|")
    ecart_exactitude = _delta(summary["exactitude"], _get(previous, "exactitude"))
    add(f"| **Exactitude globale** | {_pct(summary['exactitude'])} | {ecart_exactitude} |")
    for family, stats in summary["par_famille"].items():
        previous_family = (previous or {}).get("par_famille", {}).get(family, {}).get("exactitude")
        ecart = _delta(stats["exactitude"], previous_family)
        add(
            f"| Exactitude — {family} ({stats['questions']} questions) "
            f"| {_pct(stats['exactitude'])} | {ecart} |"
        )
    ecart = _delta(summary["refus_correct"], _get(previous, "refus_correct"))
    add(f"| Refus correct sur les pièges | {_pct(summary['refus_correct'])} | {ecart} |")
    ecart = _delta(summary["refus_a_tort"], _get(previous, "refus_a_tort"), higher_is_better=False)
    add(f"| **Refus à tort** | {_pct(summary['refus_a_tort'])} | {ecart} |")
    ecart = _delta(summary["hallucinations"], _get(previous, "hallucinations"), higher_is_better=False)
    add(f"| **Hallucinations** | {_pct(summary['hallucinations'])} | {ecart} |")
    ecart = _delta(summary["instabilite_verdict"], _get(previous, "instabilite_verdict"), higher_is_better=False)
    add(f"| Instabilité du verdict | {_pct(summary['instabilite_verdict'])} | {ecart} |")
    ecart = _delta(summary["instabilite_reponse"], _get(previous, "instabilite_reponse"), higher_is_better=False)
    add(f"| Instabilité des chiffres cités | {_pct(summary['instabilite_reponse'])} | {ecart} |")
    add(f"| Latence p50 | {summary['latence_p50_ms']:.0f} ms | — |")
    add(f"| Latence p95 | {summary['latence_p95_ms']:.0f} ms | — |")
    cout_question = summary["cout_par_question_eur"]
    cout_total = summary["cout_total_eur"]
    if cout_question is None:
        add(f"| Coût par question | non chiffré — tarif de `{summary['modele']}` non relevé | — |")
        add("| Coût total de l'exécution | non chiffré | — |")
    else:
        add(f"| Coût par question | {cout_question:.5f} € | — |")
        add(f"| Coût total de l'exécution | {cout_total:.4f} € | — |")
    add(f"| Appels d'outils par question | {summary['outils_par_question']} | — |")
    add(f"| Appels d'outils en erreur | {summary['erreurs_outil']} | — |")
    add("")

    add("### Comment lire ces chiffres")
    add("")
    add(
        "**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait "
        "« données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : "
        "c'est le taux de refus à tort qui le démasque."
    )
    add("")
    add(
        "**Une hallucination est une réponse affirmative à une question sans réponse.** "
        "C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : "
        "la réponse est bien formée, plausible, et fausse."
    )
    add("")

    if summary["outils_utilises"]:
        add("### Outils appelés")
        add("")
        add("| Outil | Appels |")
        add("|---|---|")
        for name, count in summary["outils_utilises"].items():
            add(f"| `{name}` | {count} |")
        add("")

    if summary["echecs"]:
        add(f"## Les {len(summary['echecs'])} questions qui échouent")
        add("")
        add(
            "Cette section est la raison d'être du banc. Savoir *lesquelles* échouent "
            "vaut plus que le taux global."
        )
        add("")
        for failure in summary["echecs"]:
            add(f"**{failure['id']}** · {failure['famille']} · réussie {_pct(failure['taux_reussite'])} des passes")
            add("")
            add(f"> {failure['question']}")
            add("")
            add(f"- Attendu : `{failure['attendu']}`")
            add(f"- Verdict dominant : `{failure['verdict']}`")
            add(f"- Outils appelés : {', '.join(f'`{t}`' for t in failure['outils']) or 'aucun'}")
            add(f"- Dernière réponse : {failure['reponse'] or '(vide)'}")
            add("")
    else:
        add("## Échecs")
        add("")
        add("Aucun. Les 40 questions passent sur toutes les passes.")
        add("")

    if summary["instables"]:
        add("## Questions instables d'une passe à l'autre")
        add("")
        add("| Question | Famille | Taux de réussite |")
        add("|---|---|---|")
        for item in summary["instables"]:
            add(f"| {item['id']} | {item['famille']} | {_pct(item['taux_reussite'])} |")
        add("")

    add("---")
    add("")
    add(
        "*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, "
        "par un chemin indépendant de celui qu'emprunte l'agent. "
        "Détail des questions : `evals/questions.yaml`.*"
    )
    add("")
    return "\n".join(lines)


def _get(summary: dict[str, Any] | None, key: str) -> float | None:
    return None if summary is None else summary.get(key)


def previous_summary(directory: Path) -> dict[str, Any] | None:
    """Le dernier rapport en date, pour calculer les écarts."""
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


# ── Entrée ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Banc d'évaluation de l'agent ERL Scout.")
    parser.add_argument("--runs", type=int, default=1, help="Nombre de passes sur le jeu (5 pour la variance)")
    parser.add_argument("--provider", default=None, help="cassette (défaut), anthropic ou openai")
    parser.add_argument("--model", default=None, help="Modèle, si fournisseur réel")
    parser.add_argument("--record", action="store_true", help="Enregistre les réponses pour le rejeu")
    parser.add_argument("--cassette", type=Path, default=None, help="Fichier d'enregistrements")
    parser.add_argument("--only", default=None, help="Ne garder que les questions dont l'identifiant contient ceci")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="Répertoire des rapports")
    parser.add_argument(
        "--fail-under", type=float, default=None, help="Sortie en échec sous ce taux d'exactitude (0 à 1)"
    )
    parser.add_argument("--quiet", action="store_true", help="Ne pas afficher la progression")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("Banc d'évaluation ERL Scout", file=sys.stderr)

    summary = run_bench(
        runs=args.runs,
        provider_name=args.provider,
        model=args.model,
        record=args.record,
        cassette=args.cassette,
        only=args.only,
        verbose=not args.quiet,
    )

    previous = previous_summary(args.out)
    report = render_report(summary, previous)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    (args.out / f"{stamp}.md").write_text(report, encoding="utf-8")
    (args.out / f"{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "latest.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nRapport écrit dans {args.out / f'{stamp}.md'}", file=sys.stderr)

    if args.fail_under is not None and summary["exactitude"] < args.fail_under:
        print(
            f"ÉCHEC : exactitude {summary['exactitude']:.1%} sous le seuil {args.fail_under:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
