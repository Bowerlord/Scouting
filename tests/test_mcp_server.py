"""Tests du serveur MCP.

Ils vérifient trois choses différentes :
  - que les outils sont bien déclarés avec des descriptions utilisables ;
  - que le dictionnaire de données reste synchronisé avec les modèles dbt ;
  - que les outils renvoient réellement des données, en appelant le backend.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp", reason="SDK MCP non installé")
pytest.importorskip("fastapi", reason="fastapi non installé")

from api.config import METRICS_DIR  # noqa: E402
from mcp_server import resources  # noqa: E402
from mcp_server.server import server  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (METRICS_DIR / "talent_scores_players.csv").exists(),
    reason="résultats du pipeline absents",
)

EXPECTED_TOOLS = {
    "search_players",
    "get_player",
    "find_similar_players",
    "get_leaderboard",
    "get_archetypes",
    "list_filters",
    "get_status",
}


@pytest.fixture(scope="module")
def tools() -> dict:
    import anyio

    listed = anyio.run(server.list_tools)
    return {tool.name: tool for tool in listed}


def test_all_tools_are_declared(tools: dict) -> None:
    assert EXPECTED_TOOLS <= set(tools)


def test_every_tool_has_a_usable_description(tools: dict) -> None:
    """Un outil sans description est un outil que l'agent utilisera de travers."""
    for name, tool in tools.items():
        assert tool.description, f"{name} n'a pas de description"
        assert len(tool.description) > 60, f"la description de {name} est trop courte pour guider un agent"


def test_server_instructions_carry_the_reading_rules() -> None:
    """Les deux pièges d'interprétation doivent être annoncés au niveau du serveur."""
    instructions = server.instructions or ""
    assert "percentile" in instructions, "la règle de comparaison par percentile doit être annoncée"
    assert "10 matchs" in instructions, "le seuil d'échantillon fiable doit être annoncé"


# ── Ressources ────────────────────────────────────────────────────────────────


def test_data_dictionary_is_built_from_dbt_models() -> None:
    dictionary = resources.build_data_dictionary()
    names = {model["name"] for model in dictionary["models"]}

    assert "fct_player_season" in names, "la table de faits doit être documentée"
    assert "mart_archetype_performance" in names
    assert dictionary["how_to_read"]["talent_score"]


def test_data_dictionary_carries_dbt_constraints() -> None:
    """Les tests dbt renseignent l'agent sur les valeurs admises."""
    dictionary = resources.build_data_dictionary()
    facts = next(model for model in dictionary["models"] if model["name"] == "fct_player_season")
    position = next(column for column in facts["columns"] if column["name"] == "position")

    assert any("accepted_values" in constraint for constraint in position["constraints"])


# ── Comportement réel des outils ──────────────────────────────────────────────


def _call(tool_name: str, **arguments):
    """Appelle un outil et renvoie sa charge utile décodée.

    Le SDK a changé de forme de retour entre ses versions : un tuple en 1.x,
    un CallToolResult en 2.x. On accepte les deux plutôt que d'épingler une
    version, pour que ces tests survivent à une montée de version du SDK.
    """
    import anyio

    result = anyio.run(lambda: server.call_tool(tool_name, arguments))

    content = getattr(result, "content", None)
    if content is None:
        content = result[0] if isinstance(result, tuple) else result

    if isinstance(content, list) and content and hasattr(content[0], "text"):
        return json.loads(content[0].text)
    return content


def test_leaderboard_tool_returns_reliable_players() -> None:
    entries = _call("get_leaderboard", limit=5)
    assert len(entries) == 5
    assert all(entry["games_played"] >= 10 for entry in entries)


def test_archetypes_tool_exposes_promotion_rate() -> None:
    archetypes = _call("get_archetypes", position="top")
    assert archetypes
    assert all("promotion_rate" in archetype for archetype in archetypes)


def test_unknown_player_returns_a_helpful_error() -> None:
    """L'agent doit recevoir de quoi se corriger, pas seulement un échec."""
    payload = _call("get_player", playername="ce_joueur_n_existe_pas")
    assert "error" in payload
    assert "search_players" in payload["hint"]
