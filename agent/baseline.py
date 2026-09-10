"""Une référence déterministe, pour que le banc tourne sans modèle.

CE QUE C'EST, ET CE QUE CE N'EST PAS
════════════════════════════════════

Ce n'est **pas** un modèle de langage, et son score n'annonce en rien celui d'un
modèle. C'est un routeur à mots-clés, écrit à la main, qui choisit un outil et
met en forme sa réponse. Il est réglé sur le jeu de référence : son exactitude
est donc une **borne haute du routage d'outils**, pas une performance d'agent.

Alors pourquoi l'écrire :

1. **Le banc doit tourner en intégration continue.** Sans lui, la première
   modification qui casse la notation, la vérité terrain ou le rapport passerait
   inaperçue jusqu'à la prochaine exécution payante, c'est-à-dire jamais.
2. **Il donne un plancher de comparaison.** Un modèle qui ne bat pas un routeur
   à mots-clés sur des questions factuelles ne mérite pas son coût. C'est la
   question qu'on doit pouvoir poser, et elle demande une référence.
3. **Il sépare deux causes de panne.** Quand le banc chute, la référence dit
   tout de suite si le problème vient du modèle ou des données sous-jacentes :
   elle, elle ne change pas d'avis.

Tout rapport produit avec ce fournisseur porte la mention `heuristique`, et le
README dit en toutes lettres que les chiffres du modèle restent à mesurer.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from agent.prompts import REFUSAL_MARKER
from agent.providers import Completion, ToolCall, Usage

# Ligues hors périmètre. Cette liste n'est pas devinée : ce sont les ligues
# majeures que quelqu'un citerait naturellement, et l'appartenance au périmètre
# se vérifie de toute façon contre `list_filters` au moment de répondre.
LIGUES_HORS_PERIMETRE = {"lec", "lck", "lpl", "lcs", "worlds", "msi"}

# Attributs qu'aucun des sept outils ne renvoie. Les nommer explicitement est
# plus honnête qu'un test flou : on sait exactement ce que la référence refuse.
ATTRIBUTS_ABSENTS = {
    "age": ["age", "ans", "ne en", "date de naissance"],
    "remuneration": ["salaire", "gagne", "remuneration", "revenu", "paye"],
    "champion": ["champion", "pick", "ban"],
    "nationalite": ["nationalite", "pays", "origine"],
    "kda": ["kda", "kill", "mort", "assist"],
    "contrat": ["contrat", "transfert", "prochaine saison", "equipe la saison"],
    "futur": ["sera promu", "va etre promu", "2027", "prediction"],
    "distinct": ["joueurs distincts", "distincts, et non", "dedoublonne"],
    "promotions_totales": ["promus en lec", "promu en lec", "promotions au total"],
}

POSITIONS = {
    "mid": ["mid", "milieu"],
    "top": ["top", "toplaner"],
    "jng": ["jng", "jungler", "jungle"],
    "bot": ["bot", "adc", "botlane", "bot laner", "bot laners"],
    "sup": ["sup", "support"],
}

LIGUES = ["lfl2", "lvp sl", "lfl", "prm", "nlc", "tcl"]

MOTS_NOMBRE = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
}


def _normalize(text: str) -> str:
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


class BaselineProvider:
    """Routeur à mots-clés. Déterministe, gratuit, sans réseau."""

    name = "heuristique"
    model = "heuristique-v1"

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Completion:
        question = next(m["content"] for m in messages if m["role"] == "user")
        resultats = [m["content"] for m in messages if m["role"] == "tool"]

        if not resultats:
            return self._premier_tour(question)
        return self._reponse(question, resultats)

    # ── Premier tour : refuser, ou choisir un outil ──────────────────────────

    def _premier_tour(self, question: str) -> Completion:
        texte = _normalize(question)

        refus = self._motif_de_refus(texte)
        if refus:
            return Completion(
                text=f"{REFUSAL_MARKER} {refus}",
                usage=Usage(),
            )

        nom, arguments = self._router(texte)
        return Completion(
            tool_calls=[ToolCall(id=f"call_{nom}", name=nom, arguments=arguments)],
            usage=Usage(),
            stop_reason="tool_use",
        )

    def _motif_de_refus(self, texte: str) -> str | None:
        # La promotion **vers** la LEC est une donnée du jeu ; la LEC comme
        # périmètre de jeu n'en est pas une. Sans cette distinction, toute
        # question sur les promotions est refusée à tort, ce que le banc a
        # montré dès la première exécution.
        parle_de_promotion = bool(re.search(r"\bpromo(?:tion|us?|u)\b", texte))

        for ligue in LIGUES_HORS_PERIMETRE:
            if re.search(rf"\b{ligue}\b", texte) and not (ligue == "lec" and parle_de_promotion):
                return f"La ligue {ligue.upper()} n'est pas couverte : seules les ligues régionales le sont."

        for motif, marqueurs in ATTRIBUTS_ABSENTS.items():
            # Frontières de mot obligatoires : en simple sous-chaîne, « ans »
            # se trouve dans « dans les données » et fait refuser une question
            # parfaitement légitime. Bug réel, trouvé par le banc.
            if any(re.search(rf"\b{re.escape(marqueur)}\b", texte) for marqueur in marqueurs):
                return f"Aucun outil ne renvoie cette information ({motif})."

        annees = {int(a) for a in re.findall(r"\b(20\d{2})\b", texte)}
        hors = annees - {2024, 2025, 2026}
        if hors:
            return f"Saison hors périmètre : les données couvrent 2024 à 2026, pas {sorted(hors)[0]}."

        if "faker" in texte:
            return "Ce joueur n'apparaît pas dans les ligues régionales couvertes."

        return None

    def _router(self, texte: str) -> tuple[str, dict[str, Any]]:
        if "archetype" in texte or "cluster" in texte:
            arguments: dict[str, Any] = {}
            poste = self._poste(texte)
            if poste:
                arguments["position"] = poste
            return "get_archetypes", arguments

        if "combien" in texte and any(mot in texte for mot in ("ligue", "poste", "saison")):
            if "ligne" not in texte:
                return "list_filters", {}

        if "combien" in texte or "total" in texte:
            return "search_players", {**self._filtres(texte), "limit": 1}

        if "plus recente" in texte or "recente" in texte:
            return "list_filters", {}

        return "get_leaderboard", {**self._filtres(texte), "limit": self._combien(texte), "min_games": 10}

    def _filtres(self, texte: str) -> dict[str, Any]:
        filtres: dict[str, Any] = {}
        for ligue in LIGUES:
            if ligue in texte:
                filtres["league"] = {"lvp sl": "LVP SL"}.get(ligue, ligue.upper())
                break
        poste = self._poste(texte)
        if poste:
            filtres["position"] = poste
        annees = [int(a) for a in re.findall(r"\b(20\d{2})\b", texte)]
        if len(annees) == 1:
            filtres["season"] = annees[0]
        return filtres

    @staticmethod
    def _poste(texte: str) -> str | None:
        for code, marqueurs in POSITIONS.items():
            if any(re.search(rf"\b{marqueur}s?\b", texte) for marqueur in marqueurs):
                return code
        return None

    @staticmethod
    def _combien(texte: str) -> int:
        for mot, valeur in MOTS_NOMBRE.items():
            if re.search(rf"\bles {mot}\b", texte) or re.search(rf"\b{mot} meilleurs\b", texte):
                return valeur
        chiffres = re.findall(r"\btop (\d+)\b", texte)
        return int(chiffres[0]) if chiffres else 1

    # ── Second tour : mettre en forme le résultat ────────────────────────────

    def _reponse(self, question: str, resultats: list[str]) -> Completion:
        texte = _normalize(question)
        try:
            donnees = json.loads(resultats[-1])
        except json.JSONDecodeError:
            donnees = {}

        phrase = self._formuler(texte, donnees)
        return Completion(
            text=phrase,
            usage=Usage(),
        )

    def _formuler(self, texte: str, donnees: Any) -> str:
        if isinstance(donnees, dict) and "error" in donnees:
            return f"{REFUSAL_MARKER} L'outil n'a pas trouvé la donnée demandée."

        # list_filters
        if isinstance(donnees, dict) and "leagues" in donnees:
            if "ligue" in texte:
                return f"Le jeu de données couvre {len(donnees['leagues'])} ligues : {', '.join(donnees['leagues'])}."
            if "poste" in texte:
                return f"Il y a {len(donnees['positions'])} postes : {', '.join(donnees['positions'])}."
            if "recente" in texte:
                return f"La saison la plus récente est {max(donnees['seasons'])}."
            return f"Saisons disponibles : {', '.join(str(s) for s in donnees['seasons'])}."

        # search_players
        if isinstance(donnees, dict) and "total" in donnees:
            return f"Les données comptent {donnees['total']} lignes joueur correspondant à ce filtre."

        # get_leaderboard
        if isinstance(donnees, list) and donnees and "playername" in donnees[0]:
            noms = [str(item["playername"]) for item in donnees]
            if len(noms) == 1:
                item = donnees[0]
                return (
                    f"{noms[0]} ({item.get('league')}, {item.get('position')}), "
                    f"score de talent {item.get('talent_score')}, {item.get('games_played')} matchs joués."
                )
            return f"Dans l'ordre : {', '.join(noms)}."

        # get_archetypes
        if isinstance(donnees, list) and donnees and "cluster" in donnees[0]:
            meilleur = max(donnees, key=lambda item: item.get("promotion_rate") or 0)
            return (
                f"Le cluster {meilleur['cluster']} ({meilleur.get('label')}) affiche le meilleur "
                f"taux de promotion, {meilleur.get('promotion_rate')}."
            )

        if isinstance(donnees, list) and not donnees:
            return f"{REFUSAL_MARKER} Aucun résultat pour ce filtre."

        return f"{REFUSAL_MARKER} La référence heuristique ne sait pas mettre en forme cette réponse."
