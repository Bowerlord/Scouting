"""Accès aux données, par l'API si possible, par le disque sinon.

Pourquoi ce double chemin : le serveur MCP doit rester utilisable quand l'API
n'est pas déployée, sinon il ne sert à rien tant que l'hébergement n'est pas
fait. Mais quand l'API est là, c'est elle qui fait foi, pour que l'agent et le
dashboard voient exactement les mêmes chiffres.

Le mode retenu est décidé une fois au démarrage et annoncé dans les logs :
un serveur qui bascule silencieusement d'une source à l'autre rend tout
diagnostic impossible.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_URL = os.getenv("SCOUTING_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("SCOUTING_API_TIMEOUT", "10"))


class Backend:
    """Source de données du serveur MCP."""

    def __init__(self, api_url: str = DEFAULT_API_URL) -> None:
        self.api_url = api_url.rstrip("/")
        self._client: httpx.Client | None = None
        self._mode: str | None = None

    # ── Choix du mode ────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """'api' ou 'local'. Déterminé au premier appel, puis figé."""
        if self._mode is None:
            self._mode = "api" if self._api_reachable() else "local"
        return self._mode

    def _api_reachable(self) -> bool:
        try:
            response = httpx.get(f"{self.api_url}/health", timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.api_url, timeout=REQUEST_TIMEOUT)
        return self._client

    # ── Appels ───────────────────────────────────────────────────────────────

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Interroge l'API, ou reproduit la réponse en local si elle est absente."""
        cleaned = {key: value for key, value in (params or {}).items() if value is not None}

        if self.mode == "api":
            response = self.client.get(path, params=cleaned)
            if response.status_code == 404:
                raise LookupError(response.json().get("detail", "Ressource introuvable"))
            response.raise_for_status()
            return response.json()

        return self._local(path, cleaned)

    def _local(self, path: str, params: dict[str, Any]) -> Any:
        """Repli hors ligne : on appelle les mêmes fonctions que l'API, en direct.

        Importées ici et non en tête de module pour que le serveur démarre même
        si les dépendances de l'API ne sont pas installées, tant que l'API
        distante répond.
        """
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as client:
            response = client.get(path, params=params)
            if response.status_code == 404:
                raise LookupError(response.json().get("detail", "Ressource introuvable"))
            response.raise_for_status()
            return response.json()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
