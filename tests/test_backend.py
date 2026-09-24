"""Tests du choix de mode du backend : API distante ou repli local."""

from __future__ import annotations

from mcp_server import backend
from mcp_server.backend import Backend


def test_api_reachable_fixes_api_mode(monkeypatch) -> None:
    monkeypatch.setattr(Backend, "_api_reachable", lambda self: True)
    b = Backend("http://api.test")
    assert b.mode == "api"
    assert b._mode == "api"


def test_unreachable_api_falls_back_to_local_when_possible(monkeypatch) -> None:
    monkeypatch.setattr(Backend, "_api_reachable", lambda self: False)
    monkeypatch.setattr(backend, "_local_available", lambda: True)
    b = Backend("http://api.test")
    assert b.mode == "local"


def test_no_frozen_local_mode_without_fastapi(monkeypatch) -> None:
    """API en veille et pas de fastapi : on retente l'API au lieu de figer un mode cassé."""
    monkeypatch.setattr(backend, "_local_available", lambda: False)
    reponses = iter([False, True])
    monkeypatch.setattr(Backend, "_api_reachable", lambda self: next(reponses))
    b = Backend("http://api.test")
    assert b.mode == "api"
    assert b._mode is None
    assert b.mode == "api"
    assert b._mode == "api"


def test_probe_timeout_covers_a_cold_start() -> None:
    assert backend.PROBE_TIMEOUT >= 10
