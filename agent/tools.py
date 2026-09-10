"""Les outils que l'agent a le droit d'appeler.

Choix de conception, et c'est le principal : **pas de text-to-SQL libre**. Ce
que l'agent peut faire est borné aux sept routes déjà exposées par l'API et le
serveur MCP. Trois conséquences, toutes voulues :

- une réponse fausse vient soit d'un mauvais choix d'outil, soit d'une mauvaise
  lecture du résultat. Jamais d'une requête inventée. Le diagnostic est donc
  possible, ce qui est exactement ce que le banc d'évaluation cherche à mesurer ;
- la surface d'attaque est nulle : aucun SQL ne transite ;
- l'agent, le dashboard et le serveur MCP lisent les mêmes chiffres, puisqu'ils
  passent tous par `Backend`.

Les descriptions sont volontairement longues. Une description qui dit seulement
ce que fait l'outil laisse l'agent choisir au hasard entre deux outils voisins ;
celles-ci disent **quand** l'utiliser et **comment lire** ce qui revient.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_server.backend import Backend

# Une seule instance : `Backend` décide de son mode (API distante ou lecture
# locale) au premier appel puis le fige, et on veut que ce choix vaille pour
# toute la session plutôt que d'être rejoué à chaque question.
_backend = Backend()


def backend() -> Backend:
    """Le point d'accès aux données. Exposé pour que les tests le remplacent."""
    return _backend


def set_backend(replacement: Backend) -> None:
    """Remplace la source de données. Réservé aux tests."""
    global _backend
    _backend = replacement


# ── Déclaration des outils ────────────────────────────────────────────────────
#
# Format neutre, traduit ensuite au dialecte de chaque fournisseur dans
# `providers.py`. Écrire deux fois la même liste, une par fournisseur, serait
# le meilleur moyen de les laisser diverger en silence.

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_players",
        "description": (
            "Recherche des joueurs avec des filtres. À utiliser pour toute question commençant "
            "par « quels joueurs », « combien de joueurs », ou portant sur une ligue ou un poste "
            "entier. Le champ `total` de la réponse donne le nombre de joueurs correspondant au "
            "filtre, toutes pages confondues : c'est lui qu'il faut lire pour une question de "
            "dénombrement, jamais la longueur de la liste `items` qui n'est qu'une page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "league": {
                    "type": "string",
                    "description": "Ligue : LFL, LFL2, PRM, NLC, LVP SL ou TCL",
                },
                "position": {
                    "type": "string",
                    "description": "Poste : top, jng, mid, bot ou sup",
                },
                "season": {"type": "integer", "description": "Saison : 2024, 2025 ou 2026"},
                "min_games": {
                    "type": "integer",
                    "description": "Matchs minimum. Mettre 10 pour ne garder que des scores fiables",
                },
                "search": {"type": "string", "description": "Recherche partielle sur le pseudonyme"},
                "sort_by": {
                    "type": "string",
                    "description": "Colonne de tri, par défaut talent_score",
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre de résultats renvoyés, 1 à 100",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_player",
        "description": (
            "Fiche complète d'un joueur : toutes ses saisons connues, ses métriques de "
            "performance en z-scores et son archétype de jeu. Renvoie un objet contenant "
            "`error` si le joueur est inconnu ; dans ce cas ne pas inventer de chiffre, "
            "réessayer avec search_players ou répondre que le joueur n'est pas dans les données."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "playername": {
                    "type": "string",
                    "description": "Pseudonyme du joueur, insensible à la casse",
                }
            },
            "required": ["playername"],
        },
    },
    {
        "name": "find_similar_players",
        "description": (
            "Joueurs au profil de jeu le plus proche d'un joueur donné. La proximité se mesure "
            "sur les z-scores de performance, pas sur le score de talent : deux joueurs au même "
            "score global peuvent jouer de façons opposées. La comparaison reste dans le même "
            "poste. Une distance de 0 signifie un profil identique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "playername": {"type": "string", "description": "Joueur de référence"},
                "count": {"type": "integer", "description": "Nombre de voisins, 1 à 20"},
            },
            "required": ["playername"],
        },
    },
    {
        "name": "get_leaderboard",
        "description": (
            "Classement des meilleurs talents par score. Le filtre sur le nombre de matchs vaut "
            "10 par défaut, et il faut le laisser : sans lui la tête du classement est occupée "
            "par des joueurs à trois matchs dont le score n'a aucune valeur prédictive. "
            "À utiliser pour « qui est le meilleur », « top N », « meilleur joueur de telle ligue »."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Restreindre à un poste"},
                "league": {"type": "string", "description": "Restreindre à une ligue"},
                "season": {"type": "integer", "description": "Restreindre à une saison"},
                "min_games": {"type": "integer", "description": "Matchs minimum, 10 par défaut"},
                "limit": {"type": "integer", "description": "Taille du classement, 1 à 50"},
            },
            "required": [],
        },
    },
    {
        "name": "get_archetypes",
        "description": (
            "Les archétypes de jeu issus du clustering, avec leur taux de promotion en LEC. "
            "C'est l'outil des questions du type « quel style de jeu mène réellement à monter » : "
            "il donne `promotion_rate`, la part des joueurs du groupe effectivement promus, et "
            "pas seulement la description du profil."
        ),
        "parameters": {
            "type": "object",
            "properties": {"position": {"type": "string", "description": "Restreindre à un poste"}},
            "required": [],
        },
    },
    {
        "name": "list_filters",
        "description": (
            "Les valeurs réellement présentes dans les données : ligues, postes, saisons, splits "
            "et colonnes triables. À appeler en premier en cas de doute sur un nom de ligue, "
            "plutôt que de deviner et d'obtenir un résultat vide qu'on prendrait pour une absence."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_status",
        "description": (
            "État du service et fraîcheur des données : nombre de joueurs chargés, date du "
            "dernier rafraîchissement du pipeline, ligues et postes couverts. À utiliser pour "
            "toute question sur le périmètre ou l'actualité des données."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_NAMES = {tool["name"] for tool in TOOLS}


# ── Exécution ─────────────────────────────────────────────────────────────────


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def execute(name: str, arguments: dict[str, Any] | None = None) -> str:
    """Exécute un outil et renvoie sa réponse en JSON.

    Ne lève jamais : une erreur est renvoyée à l'agent sous forme de JSON avec
    un champ `error`. Un agent qui reçoit une erreur lisible peut se rattraper ;
    une exception coupe la conversation et fait perdre la question, ce qui
    fausserait la mesure d'exactitude en la confondant avec une panne.
    """
    arguments = {key: value for key, value in (arguments or {}).items() if value is not None}

    if name not in TOOL_NAMES:
        return _dump({"error": f"Outil inconnu : {name}", "outils_disponibles": sorted(TOOL_NAMES)})

    try:
        return _dump(_dispatch(name, arguments))
    except LookupError as error:
        return _dump(
            {
                "error": str(error),
                "hint": "Vérifier l'orthographe avec search_players, ou list_filters pour les valeurs valides.",
            }
        )
    except Exception as error:  # noqa: BLE001 — toute panne doit revenir à l'agent, pas remonter
        return _dump({"error": f"{type(error).__name__}: {error}"})


def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
    api = backend()

    if name == "search_players":
        return api.get("/players", arguments)

    if name == "get_player":
        return api.get(f"/players/{arguments['playername']}")

    if name == "find_similar_players":
        playername = arguments.pop("playername")
        return api.get(f"/players/{playername}/similar", arguments)

    if name == "get_leaderboard":
        return api.get("/leaderboard", arguments)

    if name == "get_archetypes":
        return api.get("/archetypes", arguments)

    if name == "list_filters":
        return api.get("/filters")

    if name == "get_status":
        payload = api.get("/health")
        payload["backend_mode"] = api.mode
        return payload

    raise AssertionError(f"Outil déclaré mais non implémenté : {name}")
