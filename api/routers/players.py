"""Routes joueurs : recherche, fiche, joueurs similaires, classement."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.config import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_SIMILAR_COUNT,
    MAX_PAGE_SIZE,
    MAX_SIMILAR_COUNT,
)
from api.data import find_similar, get_store
from api.schemas import PlayerDetail, PlayerPage, PlayerSummary, SimilarPlayer
from api.serialization import frame_to_dicts

router = APIRouter(prefix="/players", tags=["joueurs"])

# Le pipeline nomme l'année `_source_year`. Le préfixe souligné est une
# convention interne qui n'a rien à faire dans un contrat public.
RENAME = {"_source_year": "season"}


def _filtered(league, position, split, season, min_games, search):
    """Applique les filtres communs aux routes de liste."""
    frame = get_store().players
    if league:
        frame = frame[frame["league"].str.lower() == league.lower()]
    if position:
        frame = frame[frame["position"].str.lower() == position.lower()]
    if split:
        frame = frame[frame["split"].astype(str).str.lower() == split.lower()]
    if season:
        frame = frame[frame["_source_year"] == season]
    if min_games:
        frame = frame[frame["games_played"] >= min_games]
    if search:
        frame = frame[frame["playername"].str.contains(search, case=False, na=False)]
    return frame


@router.get("", response_model=PlayerPage, summary="Rechercher des joueurs")
def list_players(
    league: str | None = Query(default=None, description="Filtrer sur une ligue (LFL, Prime League, ...)"),
    position: str | None = Query(default=None, description="Filtrer sur un poste (top, jng, mid, bot, sup)"),
    split: str | None = Query(default=None, description="Filtrer sur un split (Spring, Summer)"),
    season: int | None = Query(default=None, description="Filtrer sur une saison"),
    min_games: int | None = Query(default=None, ge=0, description="Nombre minimum de matchs joués"),
    search: str | None = Query(default=None, description="Recherche partielle sur le pseudonyme"),
    sort_by: str = Query(default="talent_score", description="Colonne de tri"),
    descending: bool = Query(default=True, description="Tri décroissant"),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> PlayerPage:
    frame = _filtered(league, position, split, season, min_games, search)

    if sort_by not in frame.columns:
        raise HTTPException(status_code=422, detail=f"Colonne de tri inconnue : {sort_by}")

    # na_position="last" garde les joueurs sans score en fin de liste plutôt
    # que de les faire remonter en tête d'un tri décroissant.
    frame = frame.sort_values(sort_by, ascending=not descending, na_position="last")
    page = frame.iloc[offset : offset + limit]

    return PlayerPage(
        total=int(len(frame)),
        limit=limit,
        offset=offset,
        items=[PlayerSummary(**item) for item in frame_to_dicts(page, RENAME)],
    )


@router.get("/{playername}", response_model=list[PlayerDetail], summary="Fiche d'un joueur")
def get_player(playername: str) -> list[PlayerDetail]:
    """Renvoie toutes les saisons connues du joueur, de la plus récente à la plus ancienne.

    La réponse est une liste et non un objet unique : un joueur peut avoir
    plusieurs lignes, une par saison et par split, et n'en montrer qu'une
    obligerait le client à deviner laquelle.
    """
    frame = get_store().players
    matches = frame[frame["playername"].str.lower() == playername.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Joueur inconnu : {playername}")

    matches = matches.sort_values("_source_year", ascending=False)
    return [PlayerDetail(**item) for item in frame_to_dicts(matches, RENAME)]


@router.get("/{playername}/similar", response_model=list[SimilarPlayer], summary="Joueurs au profil proche")
def get_similar_players(
    playername: str,
    count: int = Query(default=DEFAULT_SIMILAR_COUNT, ge=1, le=MAX_SIMILAR_COUNT),
) -> list[SimilarPlayer]:
    store = get_store()
    matches = store.players[store.players["playername"].str.lower() == playername.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Joueur inconnu : {playername}")

    reference = matches.sort_values("_source_year", ascending=False).iloc[0]
    similar = find_similar(store, reference, count)
    if similar.empty:
        return []

    return [SimilarPlayer(**item) for item in frame_to_dicts(similar, RENAME)]
