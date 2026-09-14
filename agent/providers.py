"""Les fournisseurs de modèle, et le mode rejeu qui rend le banc gratuit.

Trois fournisseurs, une seule interface :

- `anthropic` et `openai` appellent un vrai modèle. Ils demandent une clé ;
- `cassette` rejoue des réponses enregistrées sur disque. Aucune clé, aucun
  coût, aucune variance parasite.

Pourquoi le mode rejeu existe, et ce n'est pas un détail d'implémentation : un
banc d'évaluation qui ne tourne qu'avec une clé payante ne tourne pas en
intégration continue, donc il ne tourne jamais, donc il ne sert à rien. Ici la
CI rejoue les enregistrements à chaque commit et échoue si l'exactitude baisse.
Les enregistrements se refont quand on veut avec `--record`.

Une conséquence à assumer : le rejeu mesure la non-régression du **code** et du
**prompt**, pas la variabilité du modèle. Celle-là ne se mesure qu'en exécution
réelle, et le rapport indique toujours lequel des deux régimes l'a produit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

CASSETTE_PATH = Path(os.getenv("SCOUTING_CASSETTE", "evals/cassettes/default.json"))

# ── Tarifs ────────────────────────────────────────────────────────────────────
#
# Prix en dollars par million de jetons, et taux de conversion retenu.
#
# UNE RÈGLE, ET ELLE EST STRICTE : un modèle absent de cette table n'a pas un
# coût estimé, il a un coût **inconnu**. `cost_eur` renvoie alors None et le
# rapport écrit « non chiffré » au lieu d'un nombre.
#
# La version précédente appliquait le tarif de Sonnet à tout modèle inconnu.
# Un modèle bon marché à 0,25 $ le million se serait vu attribuer un coût douze
# fois trop élevé, et la comparaison entre modèles, qui est tout l'intérêt de la
# manœuvre, aurait été fausse sans que rien ne le signale.
#
# Les tarifs bougent. Avant de publier un chiffre de coût, revérifier sur la
# page du fournisseur et mettre à jour la date ci-dessous.
USD_PER_EUR = 1.08
TARIFS_RELEVES_LE = "2026-09-10"

PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # modèle: (entrée, sortie)
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    # La référence heuristique n'appelle aucun modèle : son coût est nul, et
    # l'écrire ici évite qu'un rapport publie un coût inventé.
    "heuristique-v1": (0.0, 0.0),
    # Groq, relevé le 2026-09-11 sur la page de tarifs du fournisseur.
    "openai/gpt-oss-120b": (0.15, 0.6),
    # Modèles à bas coût : à renseigner après relevé sur la page du fournisseur.
    # Laisser une entrée absente est volontaire tant que le tarif n'a pas été
    # vérifié en direct — mieux vaut « non chiffré » qu'un chiffre inventé.
}


class ProviderError(RuntimeError):
    """Le fournisseur ne peut pas répondre : clé absente, SDK manquant, rejeu introuvable."""


@dataclass
class ToolCall:
    """Un appel d'outil demandé par le modèle."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    """Ce que l'appel a consommé."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def cost_eur(self, model: str) -> float | None:
        """Le coût en euros, ou None si le tarif du modèle n'est pas connu.

        None n'est pas un détail : c'est ce qui empêche un rapport d'annoncer
        un coût faux pour un modèle dont personne n'a relevé le prix.
        """
        tarif = PRICING_USD_PER_MTOK.get(model)
        if tarif is None:
            return None
        entree, sortie = tarif
        usd = (self.input_tokens * entree + self.output_tokens * sortie) / 1_000_000
        return usd / USD_PER_EUR


@dataclass
class Completion:
    """Une réponse du modèle : du texte, des appels d'outils, ou les deux."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"


class LLMProvider(Protocol):
    """Ce que l'agent attend d'un fournisseur."""

    name: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Completion: ...


# ── Clé de rejeu ──────────────────────────────────────────────────────────────


def conversation_key(model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Empreinte stable d'un état de conversation.

    Elle inclut les outils : changer une description d'outil change ce que le
    modèle répond, donc un enregistrement fait avant ce changement ne le
    représente plus. Mieux vaut un rejeu manquant, qui se voit, qu'un rejeu
    périmé, qui ment.
    """
    payload = json.dumps(
        {"model": model, "system": system, "messages": messages, "tools": tools},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ── Rejeu ─────────────────────────────────────────────────────────────────────


class CassetteProvider:
    """Rejoue des réponses enregistrées. Aucune clé, aucun coût, déterministe.

    Peut aussi envelopper un vrai fournisseur pour enregistrer : c'est le mode
    `--record` du banc.
    """

    name = "cassette"

    def __init__(
        self,
        path: Path = CASSETTE_PATH,
        inner: LLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.inner = inner
        self.model = model or (inner.model if inner else "cassette")
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.model = raw.get("model", self.model)
        self._entries = raw.get("entries", {})

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"model": self.model, "entries": self._entries},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._entries)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Completion:
        key = conversation_key(self.model, system, messages, tools)
        entry = self._entries.get(key)

        if entry is not None:
            return Completion(
                text=entry.get("text", ""),
                tool_calls=[ToolCall(**call) for call in entry.get("tool_calls", [])],
                usage=Usage(**entry.get("usage", {})),
                stop_reason=entry.get("stop_reason", "end_turn"),
            )

        if self.inner is None:
            raise ProviderError(
                "Aucun enregistrement pour cet état de conversation. "
                "Refaire les enregistrements avec : python -m evals.run --record "
                "(une clé d'API est alors nécessaire)."
            )

        completion = self.inner.complete(system, messages, tools)
        self._entries[key] = {
            "text": completion.text,
            "tool_calls": [call.__dict__ for call in completion.tool_calls],
            "usage": completion.usage.__dict__,
            "stop_reason": completion.stop_reason,
        }
        self._dirty = True
        return completion


# ── Anthropic ─────────────────────────────────────────────────────────────────


class AnthropicProvider:
    """Appelle l'API Anthropic. Nécessite `anthropic` et ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover — dépend de l'environnement
            raise ProviderError("Le paquet `anthropic` n'est pas installé : pip install anthropic") from error

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY n'est pas définie.")

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools
        ]

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]

            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message["tool_call_id"],
                                "content": message["content"],
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call["arguments"],
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({"role": role, "content": message["content"]})
        return converted

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Completion:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=self._messages(messages),
            tools=self._tools(tools),
        )

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        return Completion(
            text="".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason or "end_turn",
        )


# ── OpenAI ────────────────────────────────────────────────────────────────────


#: Points d'entrée compatibles avec le dialecte OpenAI.
#:
#: Ils parlent tous le même protocole d'appel de fonctions, donc une seule
#: implémentation les couvre : seuls changent l'URL de base, la variable
#: d'environnement qui porte la clé, et le modèle par défaut.
#:
#: Deux réserves à garder en tête, et elles comptent plus que le prix :
#:
#: 1. **La qualité de l'appel de fonctions varie énormément.** Un modèle bon
#:    marché peut très bien converser et très mal choisir ses outils, ce qui est
#:    précisément ce qu'on lui demande ici. C'est au banc de trancher, pas à
#:    l'intuition ni à la fiche technique.
#: 2. **La donnée sort de l'UE** chez plusieurs de ces fournisseurs. Sans
#:    conséquence ici (statistiques d'esport publiques), rédhibitoire ailleurs.
ENDPOINTS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "GLM_API_KEY",
        "default_model": "glm-4-flash",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-small-latest",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # llama-3.3-70b-versatile a été déprécié par Groq le 2026-08-16.
        # Remplaçant recommandé par le fournisseur, et il sait appeler des outils.
        "default_model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "deepseek/deepseek-chat",
    },
    # Modèle local : aucune clé, aucune donnée qui sort de la machine.
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
        "default_model": "qwen2.5:7b",
    },
}


class OpenAIProvider:
    """Tout fournisseur parlant le dialecte OpenAI : OpenAI, DeepSeek, GLM, Groq, Ollama…

    Une seule classe pour tous, parce qu'ils partagent le protocole. Ajouter un
    fournisseur ne demande donc pas d'écrire du code, mais une entrée dans
    `ENDPOINTS` — et c'est bien ce qu'on veut : chaque implémentation en double
    est une occasion de diverger en silence.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        endpoint: str = "openai",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover — dépend de l'environnement
            raise ProviderError("Le paquet `openai` n'est pas installé : pip install openai") from error

        config = ENDPOINTS.get(endpoint)
        if config is None:
            connus = ", ".join(sorted(ENDPOINTS))
            raise ProviderError(f"Point d'entrée inconnu : {endpoint}. Connus : {connus}.")

        variable = config["api_key_env"]
        cle = os.getenv(variable) if variable else "local"
        if not cle:
            raise ProviderError(f"{variable} n'est pas définie (fournisseur {endpoint}).")

        self.name = endpoint
        self.model = model or config["default_model"]
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=cle, base_url=config["base_url"] or None)

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            if message["role"] == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                )
            elif message["role"] == "assistant" and message.get("tool_calls"):
                converted.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                                },
                            }
                            for call in message["tool_calls"]
                        ],
                    }
                )
            else:
                converted.append({"role": message["role"], "content": message["content"]})
        return converted

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Completion:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._messages(system, messages),
            tools=self._tools(tools),
        )
        choice = response.choices[0].message

        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (choice.tool_calls or [])
        ]

        return Completion(
            text=(choice.content or "").strip(),
            tool_calls=calls,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
            stop_reason=response.choices[0].finish_reason or "stop",
        )


# ── Fabrique ──────────────────────────────────────────────────────────────────


def get_provider(
    name: str | None = None,
    model: str | None = None,
    record: bool = False,
    cassette: Path | None = None,
) -> LLMProvider:
    """Construit le fournisseur demandé.

    Sans argument ni variable d'environnement, c'est le rejeu qui est choisi :
    le comportement par défaut doit être celui qui marche sans clé, sinon la
    première commande que quelqu'un tape échoue et il n'essaie pas la seconde.
    """
    name = (name or os.getenv("SCOUTING_LLM_PROVIDER") or "heuristique").lower()
    path = cassette or CASSETTE_PATH

    if name == "heuristique":
        from agent.baseline import BaselineProvider

        if record:
            raise ProviderError("--record demande un vrai fournisseur : --provider anthropic ou openai.")
        return BaselineProvider()

    if name == "cassette":
        if record:
            raise ProviderError("--record demande un vrai fournisseur : --provider anthropic ou openai.")
        return CassetteProvider(path=path)

    if name == "anthropic":
        inner: LLMProvider = AnthropicProvider(model=model or "claude-sonnet-5")
    elif name in ENDPOINTS:
        inner = OpenAIProvider(model=model, endpoint=name)
    else:
        connus = ", ".join(["heuristique", "cassette", "anthropic", *sorted(ENDPOINTS)])
        raise ProviderError(f"Fournisseur inconnu : {name}. Connus : {connus}.")

    return CassetteProvider(path=path, inner=inner, model=inner.model) if record else inner
