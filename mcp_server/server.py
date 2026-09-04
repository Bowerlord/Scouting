"""Serveur MCP : les données de scouting, utilisables par un agent.

Lancer :        python -m mcp_server
Dans Claude Code, déclarer le serveur avec la commande ci-dessus et, si l'API
est déployée, la variable SCOUTING_API_URL pointant vers elle.

Ce que le serveur apporte par rapport à un simple accès à l'API : les
descriptions. Un agent qui reçoit `talent_score` sans explication compare des
scores bruts et se trompe, parce que la distribution est très asymétrique.
Chaque outil dit donc non seulement ce qu'il fait, mais comment lire ce qu'il
renvoie.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from mcp_server import __version__
from mcp_server.backend import Backend

# httpx journalise chaque requête au niveau INFO. En transport stdio, tout
# bruit inutile encombre les logs du client sans rien apprendre à personne.
logging.getLogger("httpx").setLevel(logging.WARNING)

backend = Backend()

server = MCPServer(
    name="scouting-lol",
    version=__version__,
    title="Scouting esport League of Legends",
    instructions=(
        "Ce serveur donne accès aux résultats d'un modèle de détection de talents sur les "
        "ligues régionales européennes de League of Legends (LFL, LFL2, PRM, NLC, LVP SL, TCL), "
        "saisons 2024 à 2026.\n\n"
        "Deux règles de lecture à respecter dans toute réponse :\n"
        "1. Le score de talent va de 0 à 100 environ mais sa distribution est très asymétrique "
        "(médiane autour de 2,6). Comparer deux joueurs par leur percentile, jamais par l'écart "
        "brut de leurs scores.\n"
        "2. En dessous de 10 matchs joués, un score n'est pas interprétable. Le champ "
        "`has_reliable_sample` le signale ; ne jamais présenter un joueur sous ce seuil comme "
        "un talent sans le dire.\n\n"
        "Les z-scores sont exprimés en écarts-types par rapport aux autres joueurs du même poste "
        "dans la même ligue : 0 est la moyenne, +1 est un écart-type au-dessus."
    ),
)


def _dump(payload: Any) -> str:
    """Réponse compacte et lisible. L'agent lit du JSON, pas un tableau formaté."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ── Outils ────────────────────────────────────────────────────────────────────


@server.tool(
    description=(
        "Recherche des joueurs avec des filtres. Utiliser cet outil pour toute question "
        "commençant par « quels joueurs », « combien de joueurs » ou portant sur une ligue "
        "ou un poste entier. Renvoie le total correspondant au filtre en plus de la page."
    )
)
def search_players(
    league: Annotated[str | None, Field(description="Ligue : LFL, LFL2, PRM, NLC, LVP SL ou TCL")] = None,
    position: Annotated[str | None, Field(description="Poste : top, jng, mid, bot ou sup")] = None,
    season: Annotated[int | None, Field(description="Saison : 2024, 2025 ou 2026")] = None,
    min_games: Annotated[int | None, Field(description="Matchs minimum. Mettre 10 pour des scores fiables")] = None,
    search: Annotated[str | None, Field(description="Recherche partielle sur le pseudonyme")] = None,
    sort_by: Annotated[str, Field(description="Colonne de tri, par défaut le score de talent")] = "talent_score",
    limit: Annotated[int, Field(description="Nombre de résultats, 1 à 100", ge=1, le=100)] = 20,
) -> str:
    payload = backend.get(
        "/players",
        {
            "league": league,
            "position": position,
            "season": season,
            "min_games": min_games,
            "search": search,
            "sort_by": sort_by,
            "limit": limit,
        },
    )
    return _dump(payload)


@server.tool(
    description=(
        "Fiche complète d'un joueur : toutes ses saisons connues, ses métriques de performance "
        "en z-scores et son archétype de jeu. Renvoie une erreur explicite si le joueur est inconnu."
    )
)
def get_player(
    playername: Annotated[str, Field(description="Pseudonyme du joueur, insensible à la casse")],
) -> str:
    try:
        return _dump(backend.get(f"/players/{playername}"))
    except LookupError as error:
        return _dump(
            {
                "error": str(error),
                "hint": "Essayer search_players avec le paramètre search pour trouver l'orthographe exacte.",
            }
        )


@server.tool(
    description=(
        "Trouve les joueurs au profil de jeu le plus proche d'un joueur donné. "
        "La proximité se mesure sur les z-scores de performance et non sur le score de talent : "
        "deux joueurs au même score global peuvent jouer de façons opposées. La comparaison reste "
        "dans la même position, parce qu'un support et un mid n'ont pas des métriques comparables. "
        "Une distance de 0 signifie un profil identique."
    )
)
def find_similar_players(
    playername: Annotated[str, Field(description="Joueur de référence")],
    count: Annotated[int, Field(description="Nombre de voisins, 1 à 20", ge=1, le=20)] = 5,
) -> str:
    try:
        return _dump(backend.get(f"/players/{playername}/similar", {"count": count}))
    except LookupError as error:
        return _dump({"error": str(error)})


@server.tool(
    description=(
        "Classement des meilleurs talents. Le filtre sur le nombre de matchs est actif par "
        "défaut : sans lui, la tête du classement est occupée par des joueurs à trois matchs "
        "dont le score n'a aucune valeur prédictive."
    )
)
def get_leaderboard(
    position: Annotated[str | None, Field(description="Restreindre à un poste")] = None,
    league: Annotated[str | None, Field(description="Restreindre à une ligue")] = None,
    season: Annotated[int | None, Field(description="Restreindre à une saison")] = None,
    min_games: Annotated[int, Field(description="Matchs minimum, 10 par défaut", ge=0)] = 10,
    limit: Annotated[int, Field(description="Taille du classement, 1 à 50", ge=1, le=50)] = 10,
) -> str:
    payload = backend.get(
        "/leaderboard",
        {"position": position, "league": league, "season": season, "min_games": min_games, "limit": limit},
    )
    return _dump(payload)


@server.tool(
    description=(
        "Les archétypes de jeu issus du clustering, avec leur taux de promotion en LEC. "
        "C'est l'outil à utiliser pour toute question du type « quel style de jeu mène "
        "réellement à monter » : il donne la part des joueurs de chaque profil qui ont été promus, "
        "et pas seulement la description du profil."
    )
)
def get_archetypes(
    position: Annotated[str | None, Field(description="Restreindre à un poste")] = None,
) -> str:
    return _dump(backend.get("/archetypes", {"position": position}))


@server.tool(
    description=(
        "Les valeurs de filtres réellement présentes dans les données : ligues, postes, saisons, "
        "splits et colonnes triables. À appeler en premier en cas de doute sur une valeur, plutôt "
        "que de deviner un nom de ligue et d'obtenir un résultat vide."
    )
)
def list_filters() -> str:
    return _dump(backend.get("/filters"))


@server.tool(
    description=(
        "État du service et fraîcheur des données : nombre de joueurs chargés, date du dernier "
        "rafraîchissement du pipeline, et si le serveur lit l'API distante ou les fichiers locaux."
    )
)
def get_status() -> str:
    payload = backend.get("/health")
    payload["backend_mode"] = backend.mode
    payload["api_url"] = backend.api_url if backend.mode == "api" else None
    return _dump(payload)
