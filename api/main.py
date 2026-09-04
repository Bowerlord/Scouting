"""Point d'entrée de l'API.

Lancer en développement :   uvicorn api.main:app --reload
Documentation interactive :  http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api import __version__
from api.config import API_DESCRIPTION, API_TITLE
from api.data import DataNotAvailable, get_store
from api.observability import MetricsMiddleware
from api.routers import players, reference, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les données au démarrage plutôt qu'à la première requête.

    Le coût est payé une fois, au bon moment : si les fichiers manquent, le
    problème apparaît au déploiement et non chez le premier utilisateur.
    On ne lève pas pour autant, sinon le conteneur redémarre en boucle sans
    rien dire ; `/health` répondra 503 et l'exploitant saura pourquoi.
    """
    try:
        get_store()
    except DataNotAvailable as error:  # pragma: no cover - dépend du disque
        print(f"[api] Démarrage sans données : {error}")
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(MetricsMiddleware)

# Le dashboard Streamlit et, plus tard, l'agent en langage naturel appellent
# l'API depuis une autre origine. Ouvert en lecture seule : l'API n'expose
# aucune écriture, il n'y a donc rien à protéger derrière une origine stricte.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(reference.router)
app.include_router(players.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
