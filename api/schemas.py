"""Schémas de réponse de l'API.

Ils servent deux choses : valider ce qui sort, et documenter l'API dans le
schéma OpenAPI généré automatiquement. Les champs optionnels le sont parce
que le pipeline ne calcule pas toutes les métriques pour tous les joueurs
(un joueur avec trop peu de matchs n'a pas de z-scores fiables).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlayerSummary(BaseModel):
    """Un joueur tel qu'il apparaît dans une liste ou un classement."""

    playername: str = Field(description="Pseudonyme normalisé, identifiant du joueur")
    league: str = Field(description="Ligue régionale (LFL, Prime League, Superliga, ...)")
    season: int | None = Field(default=None, description="Année de la saison")
    split: str | None = Field(default=None, description="Split de la saison (Spring, Summer)")
    position: str = Field(description="Poste (top, jng, mid, bot, sup)")
    teamname: str | None = None
    talent_score: float | None = Field(
        default=None,
        description=(
            "Score de talent produit par le modèle, de 0 à 100 environ. La médiane observée "
            "est autour de 2,6 : la distribution est très asymétrique, se comparer au "
            "percentile plutôt qu'au score brut."
        ),
    )
    score_percentile: float | None = Field(default=None, description="Rang percentile du score, 0 à 100")
    games_played: int | None = None
    win_rate: float | None = None
    archetype: str | None = Field(default=None, description="Archétype de jeu issu du clustering")


class PlayerDetail(PlayerSummary):
    """Fiche complète d'un joueur, avec ses métriques de performance."""

    champion_pool_size: int | None = None
    promoted_to_lec: bool | None = None
    dpm_zscore: float | None = Field(default=None, description="Dégâts par minute, en écarts-types de sa ligue")
    cspm_zscore: float | None = Field(default=None, description="Sbires par minute, en écarts-types")
    golddiffat15_zscore: float | None = Field(default=None, description="Différentiel d'or à 15 min, en écarts-types")
    cluster: int | None = None


class SimilarPlayer(PlayerSummary):
    """Un joueur proche, avec sa distance au joueur de référence."""

    distance: float = Field(description="Distance euclidienne sur les z-scores de performance. 0 = profil identique")


class PlayerPage(BaseModel):
    """Une page de résultats."""

    total: int = Field(description="Nombre total de joueurs correspondant au filtre, toutes pages confondues")
    limit: int
    offset: int
    items: list[PlayerSummary]


class Archetype(BaseModel):
    """Un archétype de jeu, tel que produit par le clustering."""

    position: str = Field(description="Poste concerné : les archétypes sont calculés poste par poste")
    cluster: int
    label: str = Field(description="Description lisible de l archétype, composée des traits dominants")
    player_count: int | None = Field(default=None, description="Nombre de joueurs-saisons dans ce groupe")
    promoted_count: int | None = Field(default=None, description="Combien ont été promus en LEC")
    promotion_rate: float | None = Field(
        default=None,
        description=(
            "Part des joueurs du groupe promus en LEC. C'est le chiffre qui rend un "
            "archétype intéressant à recruter."
        ),
    )
    traits: dict[str, float] = Field(
        default_factory=dict,
        description="Moyennes des métriques du groupe, en écarts-types par rapport à la ligue",
    )


class HealthResponse(BaseModel):
    """État de l'API et fraîcheur des données servies."""

    status: str = Field(description="ok si les données sont chargées et lisibles")
    version: str
    players_loaded: int
    data_loaded_at: str
    data_refreshed_at: str | None = Field(default=None, description="Date du dernier rafraîchissement du pipeline")
    leagues: list[str]
    positions: list[str]


class RouteStats(BaseModel):
    """Compteurs d'exploitation d'une route."""

    requests: int
    errors: int
    p50_ms: float
    p95_ms: float
    max_ms: float


class MetricsResponse(BaseModel):
    """Métriques d'exploitation de l'API, pour le monitoring."""

    uptime_seconds: float
    total_requests: int
    total_errors: int
    routes: dict[str, RouteStats]
