"""Routes de référence : classement, archétypes, valeurs de filtres."""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.data import get_store
from api.schemas import Archetype, PlayerSummary
from api.serialization import clean_value, frame_to_dicts

router = APIRouter(tags=["référence"])

RENAME = {"_source_year": "season"}


@router.get("/leaderboard", response_model=list[PlayerSummary], summary="Meilleurs talents")
def leaderboard(
    position: str | None = Query(default=None, description="Restreindre à un poste"),
    league: str | None = Query(default=None, description="Restreindre à une ligue"),
    season: int | None = Query(default=None, description="Restreindre à une saison"),
    min_games: int = Query(default=10, ge=0, description="Nombre minimum de matchs, pour écarter le bruit"),
    limit: int = Query(default=20, ge=1, le=200),
) -> list[PlayerSummary]:
    """Classement par score de talent.

    Le filtre sur le nombre de matchs est actif par défaut : sans lui, la tête
    du classement est occupée par des joueurs à trois matchs dont le score
    n'a aucune valeur prédictive.
    """
    frame = get_store().players
    if position:
        frame = frame[frame["position"].str.lower() == position.lower()]
    if league:
        frame = frame[frame["league"].str.lower() == league.lower()]
    if season:
        frame = frame[frame["_source_year"] == season]
    frame = frame[frame["games_played"] >= min_games]

    top = frame.nlargest(limit, "talent_score")
    return [PlayerSummary(**item) for item in frame_to_dicts(top, RENAME)]


@router.get("/archetypes", response_model=list[Archetype], summary="Archétypes de jeu")
def archetypes(
    position: str | None = Query(default=None, description="Restreindre à un poste"),
) -> list[Archetype]:
    """Les groupes issus du clustering, avec leur taux de promotion en LEC.

    Le taux de promotion est le chiffre qui compte : il dit lesquels de ces
    profils de jeu ont réellement débouché sur une montée, et transforme une
    description statistique en information de recrutement.

    Les effectifs sont recomptés depuis les résultats plutôt que lus tels
    quels, pour qu'un fichier de profils antérieur au dernier rafraîchissement
    ne fasse pas afficher un effectif faux.
    """
    store = get_store()
    observed = store.clusters.groupby(["position", "cluster"]).size().to_dict()

    # Ces champs décrivent le groupe, pas une métrique de jeu : ils sont
    # remontés en propriétés et retirés des traits, sinon ils polluent la
    # comparaison des profils.
    meta_fields = {"cluster", "n_players", "n_promoted", "promo_rate", "archetype"}

    result: list[Archetype] = []
    for pos, profiles in store.archetypes.items():
        if position and pos.lower() != position.lower():
            continue
        if not isinstance(profiles, list):
            continue

        for profile in profiles:
            cluster_id = profile.get("cluster")
            if cluster_id is None:
                continue
            cluster_int = int(cluster_id)

            traits = {
                key: round(float(value), 4)
                for key, value in profile.items()
                if key not in meta_fields and isinstance(value, (int, float)) and not isinstance(value, bool)
            }

            result.append(
                Archetype(
                    position=pos,
                    cluster=cluster_int,
                    label=str(profile.get("archetype") or f"Cluster {cluster_int}"),
                    player_count=observed.get((pos, cluster_int), profile.get("n_players")),
                    promoted_count=profile.get("n_promoted"),
                    promotion_rate=profile.get("promo_rate"),
                    traits=traits,
                )
            )

    return sorted(result, key=lambda item: (item.position, item.cluster))


@router.get("/filters", summary="Valeurs disponibles pour les filtres")
def filters() -> dict:
    """Les valeurs réellement présentes dans les données.

    Sert à un client (dashboard, agent, serveur MCP) qui doit proposer des
    filtres sans les coder en dur, donc sans se désynchroniser des données.
    """
    store = get_store()
    return {
        "leagues": store.leagues,
        "positions": store.positions,
        "seasons": sorted(int(year) for year in store.players["_source_year"].dropna().unique()),
        "splits": sorted(str(split) for split in store.players["split"].dropna().unique()),
        "sortable_columns": [
            column for column in store.players.columns if not column.startswith("_") and column != "playername_original"
        ],
        "player_count": clean_value(store.player_count),
    }
