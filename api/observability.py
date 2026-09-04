"""Mesure d'exploitation de l'API : latence, volume, erreurs.

Le choix fait ici : garder les mesures en mémoire, dans un registre borné,
plutôt que d'ajouter Prometheus. Le projet n'a ni cluster ni collecteur, et
un registre de quelques kilo-octets répond à la seule question qui compte
pour l'instant, « est-ce que ça répond vite et est-ce que ça casse ».
Le format de sortie reste assez proche de Prometheus pour qu'un exporteur
puisse être branché plus tard sans toucher aux routes.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Fenêtre glissante de latences par route. 500 points suffisent pour un p95
# stable et bornent la mémoire, quel que soit le trafic.
_WINDOW = 500


class MetricsRegistry:
    """Compteurs et latences par route, protégés par un verrou."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
        self._requests: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self.started_at = time.monotonic()

    def record(self, route: str, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self._latencies[route].append(duration_ms)
            self._requests[route] += 1
            if status_code >= 500:
                self._errors[route] += 1

    @staticmethod
    def _percentile(values: list[float], share: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        # Index par rang le plus proche : suffisant ici, et sans dépendance.
        index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
        return round(ordered[index], 2)

    def snapshot(self) -> dict:
        with self._lock:
            routes = {}
            for route, latencies in self._latencies.items():
                values = list(latencies)
                routes[route] = {
                    "requests": self._requests[route],
                    "errors": self._errors[route],
                    "p50_ms": self._percentile(values, 0.50),
                    "p95_ms": self._percentile(values, 0.95),
                    "max_ms": round(max(values), 2) if values else 0.0,
                }
            return {
                "uptime_seconds": round(time.monotonic() - self.started_at, 1),
                "total_requests": sum(self._requests.values()),
                "total_errors": sum(self._errors.values()),
                "routes": routes,
            }

    def reset(self) -> None:
        """Remet les compteurs à zéro. Réservé aux tests."""
        with self._lock:
            self._latencies.clear()
            self._requests.clear()
            self._errors.clear()
            self.started_at = time.monotonic()


registry = MetricsRegistry()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Chronomètre chaque requête et l'enregistre sous le motif de sa route.

    On enregistre sous le motif (`/players/{playername}`) et non sous le chemin
    concret : sinon chaque joueur créerait sa propre série et le registre
    exploserait au premier crawl.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        route = request.scope.get("route")
        label = getattr(route, "path", None) or request.url.path
        registry.record(f"{request.method} {label}", duration_ms, response.status_code)

        # Utile en développement et lisible dans les logs d'un proxy.
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"
        return response
