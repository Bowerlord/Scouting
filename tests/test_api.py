"""Tests de l'API REST.

Ils tournent sur les vrais fichiers de résultats du dépôt plutôt que sur des
fixtures inventées : le but est justement d'attraper une rupture de contrat
entre le pipeline et l'API, par exemple une colonne renommée en amont. Les
tests sont ignorés proprement si les résultats ne sont pas là.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi non installé")
from fastapi.testclient import TestClient  # noqa: E402

from api.config import METRICS_DIR  # noqa: E402
from api.main import app  # noqa: E402
from api.observability import MetricsRegistry  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (METRICS_DIR / "talent_scores_players.csv").exists(),
    reason="résultats du pipeline absents : lancer make pipeline",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def known_player(client: TestClient) -> str:
    """Un joueur qui existe réellement, pour ne pas coder un pseudo en dur."""
    response = client.get("/players", params={"limit": 1})
    return response.json()["items"][0]["playername"]


# ── Santé et référence ────────────────────────────────────────────────────────


def test_health_reports_loaded_data(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["players_loaded"] > 0
    assert body["positions"], "les postes doivent être renseignés"


def test_filters_expose_real_values(client: TestClient) -> None:
    body = client.get("/filters").json()
    assert body["leagues"], "au moins une ligue"
    assert body["player_count"] > 0
    # Une colonne interne ne doit pas fuiter dans un contrat public.
    assert not any(column.startswith("_") for column in body["sortable_columns"])


def test_archetypes_carry_promotion_rate(client: TestClient) -> None:
    archetypes = client.get("/archetypes").json()
    assert archetypes, "le clustering doit produire des archétypes"

    for archetype in archetypes:
        assert archetype["position"]
        if archetype["promotion_rate"] is not None:
            assert 0.0 <= archetype["promotion_rate"] <= 1.0
        # Les champs descriptifs du groupe ne doivent pas se retrouver
        # mélangés aux métriques de jeu.
        assert "n_players" not in archetype["traits"]


# ── Joueurs ───────────────────────────────────────────────────────────────────


def test_list_players_paginates(client: TestClient) -> None:
    first = client.get("/players", params={"limit": 5}).json()
    second = client.get("/players", params={"limit": 5, "offset": 5}).json()

    assert len(first["items"]) == 5
    assert first["total"] == second["total"]
    assert first["items"][0]["playername"] != second["items"][0]["playername"]


def test_list_players_sorted_descending_by_default(client: TestClient) -> None:
    items = client.get("/players", params={"limit": 10}).json()["items"]
    scores = [item["talent_score"] for item in items if item["talent_score"] is not None]
    assert scores == sorted(scores, reverse=True)


def test_filter_by_league_is_applied(client: TestClient) -> None:
    league = client.get("/filters").json()["leagues"][0]
    items = client.get("/players", params={"league": league, "limit": 20}).json()["items"]
    assert items
    assert {item["league"] for item in items} == {league}


def test_unknown_sort_column_is_rejected(client: TestClient) -> None:
    assert client.get("/players", params={"sort_by": "colonne_absente"}).status_code == 422


def test_player_detail_returns_all_seasons(client: TestClient, known_player: str) -> None:
    seasons = client.get(f"/players/{known_player}").json()
    assert seasons
    years = [entry["season"] for entry in seasons if entry["season"] is not None]
    assert years == sorted(years, reverse=True), "les saisons doivent être ordonnées de la plus récente"


def test_player_lookup_is_case_insensitive(client: TestClient, known_player: str) -> None:
    assert client.get(f"/players/{known_player.upper()}").status_code == 200


def test_unknown_player_returns_404(client: TestClient) -> None:
    response = client.get("/players/ce_joueur_n_existe_pas")
    assert response.status_code == 404
    assert "inconnu" in response.json()["detail"].lower()


def test_similar_players_share_the_position_and_exclude_self(client: TestClient, known_player: str) -> None:
    reference = client.get(f"/players/{known_player}").json()[0]
    similar = client.get(f"/players/{known_player}/similar", params={"count": 5}).json()

    assert similar, "un joueur doit avoir des voisins dans sa position"
    assert all(entry["position"] == reference["position"] for entry in similar)
    assert all(entry["playername"] != known_player for entry in similar)
    # La distance croît : le plus proche vient en premier.
    distances = [entry["distance"] for entry in similar]
    assert distances == sorted(distances)


def test_leaderboard_filters_out_small_sample_sizes(client: TestClient) -> None:
    """Sans seuil de matchs, la tête du classement est prise par du bruit."""
    strict = client.get("/leaderboard", params={"min_games": 20, "limit": 10}).json()
    assert strict
    assert all(entry["games_played"] >= 20 for entry in strict)


# ── Exploitation ──────────────────────────────────────────────────────────────


def test_metrics_group_by_route_pattern(client: TestClient, known_player: str) -> None:
    """Les métriques doivent être agrégées par motif, pas par chemin concret.

    Sinon un simple parcours du catalogue de joueurs crée une série par joueur
    et le registre grossit sans limite.
    """
    client.get(f"/players/{known_player}")
    client.get("/players/ce_joueur_n_existe_pas")

    routes = client.get("/metrics").json()["routes"]
    assert "GET /players/{playername}" in routes
    assert not any(known_player in route for route in routes)


def test_response_time_header_is_present(client: TestClient) -> None:
    response = client.get("/health")
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_percentiles_are_ordered() -> None:
    registry = MetricsRegistry()
    for value in range(1, 101):
        registry.record("GET /x", float(value), 200)

    stats = registry.snapshot()["routes"]["GET /x"]
    assert stats["requests"] == 100
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def test_server_errors_are_counted_separately() -> None:
    registry = MetricsRegistry()
    registry.record("GET /x", 1.0, 200)
    registry.record("GET /x", 1.0, 404)
    registry.record("GET /x", 1.0, 500)

    stats = registry.snapshot()["routes"]["GET /x"]
    assert stats["requests"] == 3
    # Un 404 est une réponse correcte de l'API, pas une panne : seul le 500 compte.
    assert stats["errors"] == 1
