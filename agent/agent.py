"""La boucle d'appel d'outils.

Volontairement courte. Tout ce qui rend l'agent mesurable est ici : la trace
des outils appelés, les jetons consommés, le coût et la latence. Un agent dont
on ne sait pas ce qu'il a fait pour répondre ne peut pas être débogué quand il
se trompe, et c'est pourtant à ce moment-là qu'on en a besoin.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent import tools as tool_registry
from agent.prompts import REFUSAL_MARKER, build_system_prompt
from agent.providers import LLMProvider, Usage, get_provider

logger = logging.getLogger(__name__)

# Au-delà, l'agent tourne en rond. Observé : les boucles infinies viennent
# presque toujours d'un outil qui renvoie une erreur que le modèle réessaie à
# l'identique. On coupe, et la trace montre la répétition.
MAX_STEPS = 6


@dataclass
class ToolInvocation:
    """Un appel d'outil effectivement exécuté."""

    name: str
    arguments: dict[str, Any]
    duration_ms: float
    result_chars: int
    is_error: bool


@dataclass
class Answer:
    """Ce que l'agent renvoie, et de quoi juger la réponse."""

    question: str
    text: str
    trace: list[ToolInvocation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    steps: int = 0
    truncated: bool = False
    #: Renseigné quand le fournisseur lui-même refuse de répondre : schéma
    #: d'outil rejeté, quota dépassé, coupure réseau. La question est alors
    #: perdue, mais elle seule. Une exécution de 200 questions ne doit pas
    #: mourir sur la 41e, sinon le banc ne mesure rien le jour où ça compte.
    provider_error: str | None = None

    @property
    def refused(self) -> bool:
        """Vrai si l'agent a explicitement déclaré la donnée insuffisante."""
        return REFUSAL_MARKER in self.text

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.trace]

    def cost_eur(self) -> float | None:
        """Le coût, ou None si le tarif du modèle n'est pas renseigné."""
        return self.usage.cost_eur(self.model)

    def to_dict(self) -> dict[str, Any]:
        cost = self.cost_eur()
        return {
            "question": self.question,
            "text": self.text,
            "trace": [call.__dict__ for call in self.trace],
            "usage": self.usage.__dict__,
            "latency_ms": round(self.latency_ms, 1),
            "cost_eur": None if cost is None else round(cost, 6),
            "model": self.model,
            "steps": self.steps,
            "truncated": self.truncated,
            "refused": self.refused,
            "provider_error": self.provider_error,
        }


class ScoutAgent:
    """Répond à une question sur les données de scouting."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        system_extra: str | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.provider = provider or get_provider()
        self.system = build_system_prompt(system_extra)
        self.max_steps = max_steps

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        answer = Answer(question=question, text="", model=getattr(self.provider, "model", "inconnu"))

        for step in range(1, self.max_steps + 1):
            try:
                completion = self.provider.complete(self.system, messages, tool_registry.TOOLS)
            except Exception as error:  # noqa: BLE001 — on veut vraiment tout attraper ici
                # Le fournisseur a refusé : schéma rejeté, quota, réseau. On perd
                # cette question, pas l'exécution entière. Le message est conservé
                # tel quel, parce que c'est lui qui permet le diagnostic ensuite.
                logger.warning("Le fournisseur a refusé de répondre : %s", error)
                answer.provider_error = f"{type(error).__name__}: {error}"
                # Surtout pas le marqueur de refus ici : un refus est une réponse
                # de l'agent, et il vaut un point sur les questions pièges. Une
                # panne du fournisseur ne doit rien rapporter à personne.
                answer.text = f"[banc] Le fournisseur n'a pas répondu : {type(error).__name__}."
                answer.steps = step
                answer.latency_ms = (time.perf_counter() - started) * 1000
                return answer

            answer.usage = answer.usage + completion.usage
            answer.steps = step

            if not completion.tool_calls:
                answer.text = completion.text
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text,
                    "tool_calls": [call.__dict__ for call in completion.tool_calls],
                }
            )

            for call in completion.tool_calls:
                result = self._run_tool(call.name, call.arguments, answer)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        else:
            # La boucle s'est épuisée sans réponse finale : on le dit plutôt que
            # de renvoyer le dernier texte intermédiaire, qui n'est pas une réponse.
            answer.truncated = True
            answer.text = (
                f"{REFUSAL_MARKER} L'agent n'a pas convergé en {self.max_steps} étapes "
                "sans produire de réponse finale."
            )

        answer.latency_ms = (time.perf_counter() - started) * 1000
        return answer

    def _run_tool(self, name: str, arguments: dict[str, Any], answer: Answer) -> str:
        started = time.perf_counter()
        result = tool_registry.execute(name, arguments)
        duration = (time.perf_counter() - started) * 1000

        is_error = '"error"' in result[:200]
        answer.trace.append(
            ToolInvocation(
                name=name,
                arguments=arguments,
                duration_ms=round(duration, 1),
                result_chars=len(result),
                is_error=is_error,
            )
        )
        if is_error:
            logger.debug("Outil %s en erreur : %s", name, result[:200])
        return result


def ask(question: str, **kwargs: Any) -> Answer:
    """Raccourci pour une question isolée."""
    return ScoutAgent(**kwargs).ask(question)


def main() -> None:  # pragma: no cover — point d'entrée interactif
    """`python -m agent "ta question"` — utile pour essayer une question à la main."""
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if len(sys.argv) < 2:
        print('Usage : python -m agent "quels sont les meilleurs mid de LFL en 2026 ?"')
        raise SystemExit(2)

    result = ask(" ".join(sys.argv[1:]))
    cost = result.cost_eur()
    print(result.text)
    print()
    print(
        f"— {result.steps} étapes, outils : {', '.join(result.tool_names) or 'aucun'}, "
        f"{result.usage.input_tokens + result.usage.output_tokens} jetons, "
        f"{'tarif inconnu' if cost is None else f'{cost:.4f} €'}, {result.latency_ms:.0f} ms"
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
