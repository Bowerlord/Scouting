"""Routes d'exploitation : santé et métriques.

Séparées des routes métier parce qu'elles répondent à un autre public : un
orchestrateur de conteneurs, une sonde de supervision, ou Alexandre en train
de vérifier que le service tient.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import __version__
from api.data import DataNotAvailable, get_store
from api.observability import registry
from api.schemas import HealthResponse, MetricsResponse

router = APIRouter(tags=["exploitation"])


@router.get("/health", response_model=HealthResponse, summary="État du service")
def health() -> HealthResponse:
    """Vérifie que les données sont chargées et lisibles.

    Renvoie 503 si elles ne le sont pas : un conteneur qui répond 200 sans
    données est pire qu'un conteneur mort, parce que l'orchestrateur le laisse
    dans le pool et que les erreurs arrivent chez l'utilisateur.
    """
    try:
        store = get_store()
    except DataNotAvailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    refreshed_at = store.refresh_metadata.get("refreshed_at") or store.refresh_metadata.get("last_refresh")

    return HealthResponse(
        status="ok",
        version=__version__,
        players_loaded=store.player_count,
        data_loaded_at=store.loaded_at.isoformat(),
        data_refreshed_at=str(refreshed_at) if refreshed_at else None,
        leagues=store.leagues,
        positions=store.positions,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Métriques d'exploitation")
def metrics() -> MetricsResponse:
    """Volume, taux d'erreur et latences par route, depuis le démarrage."""
    return MetricsResponse(**registry.snapshot())
