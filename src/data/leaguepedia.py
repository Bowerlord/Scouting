"""
leaguepedia.py — Requêtes vers l'API Cargo de Leaguepedia (lol.fandom.com)

Ce module récupère les données de carrière des joueurs à partir de Leaguepedia
pour construire la target variable du modèle supervisé.

Rôle dans le pipeline :
  On a besoin de savoir quels joueurs d'ERL ont été "promus" en LEC (ou dans
  une ligue de niveau supérieur). Oracle's Elixir fournit les stats de match,
  mais pas l'historique de carrière. Leaguepedia comble ce manque.

Stratégie de construction de la target variable :
  1. Méthode principale (Oracle's Elixir) : Si un joueur apparaît dans une
     ERL en année N et dans le LEC en année N ou N+1 → promoted = 1
  2. Méthode complémentaire (Leaguepedia) : Enrichir avec les données de
     transferts pour capturer les joueurs promus entre les splits

L'API Cargo de Leaguepedia est un wrapper SQL sur MediaWiki. On peut requêter
des tables comme :
  - ScoreboardPlayers : joueurs par match
  - TournamentRosters : rosters par équipe par tournoi
  - PlayerRedirects : alias des joueurs

Usage :
  from src.data.leaguepedia import fetch_player_careers, build_promotion_labels
  careers = fetch_player_careers()
  labels = build_promotion_labels(careers)

Limites connues :
  - L'API Leaguepedia peut être lente et parfois indisponible
  - Rate limiting : on espace les requêtes de 1 seconde
  - Les noms de joueurs ne sont pas toujours cohérents entre sources
  - Ce module est "best effort" : le pipeline fonctionne même sans Leaguepedia
"""

import time
import json
import requests
from pathlib import Path

from src.config import (
    ERL_DIV1_LEAGUES,
    ERL_DIV2_LEAGUES,
    TOP_LEAGUE,
    EXTERNAL_DATA_DIR,
    DATA_YEARS,
)
from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration de l'API Leaguepedia
# ═══════════════════════════════════════════════════════════════════════════════

LEAGUEPEDIA_API_URL = "https://lol.fandom.com/api.php"

# Délai entre chaque requête API (en secondes) pour respecter le rate limiting
API_DELAY = 1.5

# Nombre maximum de résultats par requête Cargo
CARGO_LIMIT = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions bas niveau : requêtes Cargo
# ═══════════════════════════════════════════════════════════════════════════════


def _cargo_query(
    tables: str,
    fields: str,
    where: str = "",
    order_by: str = "",
    limit: int = CARGO_LIMIT,
    offset: int = 0,
) -> list[dict]:
    """
    Exécute une requête Cargo sur l'API Leaguepedia.

    L'API Cargo est un wrapper SQL qui permet de requêter les données
    structurées de Leaguepedia. C'est plus fiable que le scraping.

    Args:
        tables: Nom(s) de la/des table(s) Cargo (ex: "ScoreboardPlayers")
        fields: Colonnes à récupérer (ex: "Name, Team, DateTime_UTC")
        where: Clause WHERE (ex: "Tournament LIKE '%LFL%'")
        order_by: Clause ORDER BY
        limit: Nombre maximum de résultats
        offset: Offset pour la pagination

    Returns:
        Liste de dictionnaires avec les résultats
    """
    params = {
        "action": "cargoquery",
        "format": "json",
        "tables": tables,
        "fields": fields,
        "limit": limit,
        "offset": offset,
    }

    if where:
        params["where"] = where
    if order_by:
        params["order_by"] = order_by

    try:
        response = requests.get(
            LEAGUEPEDIA_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "KCorpScoutingTool/1.0 (educational project)"},
        )
        response.raise_for_status()
        data = response.json()

        if "cargoquery" not in data:
            logger.warning(f"Réponse Cargo inattendue : {list(data.keys())}")
            return []

        # Les résultats Cargo sont wrappés dans un objet {"title": {...}}
        results = [item["title"] for item in data["cargoquery"]]
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur API Leaguepedia : {e}")
        return []
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Erreur de parsing de la réponse Leaguepedia : {e}")
        return []


def _cargo_query_all(
    tables: str,
    fields: str,
    where: str = "",
    order_by: str = "",
) -> list[dict]:
    """
    Exécute une requête Cargo avec pagination automatique.

    L'API Cargo est limitée à 500 résultats par requête. Cette fonction
    pagine automatiquement pour récupérer tous les résultats.

    Returns:
        Liste complète de tous les résultats
    """
    all_results = []
    offset = 0

    while True:
        results = _cargo_query(
            tables=tables,
            fields=fields,
            where=where,
            order_by=order_by,
            limit=CARGO_LIMIT,
            offset=offset,
        )

        if not results:
            break

        all_results.extend(results)
        logger.debug(
            f"Cargo pagination : {len(all_results)} résultats récupérés "
            f"(offset={offset})"
        )

        # Si on a reçu moins de résultats que la limite, on a tout récupéré
        if len(results) < CARGO_LIMIT:
            break

        offset += CARGO_LIMIT
        time.sleep(API_DELAY)  # Respecter le rate limiting

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions principales
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_tournament_rosters(leagues: list[str], years: list[int]) -> list[dict]:
    """
    Récupère les rosters (joueurs par équipe) pour les tournois ciblés.

    Cette fonction requête la table TournamentRosters de Leaguepedia pour
    obtenir la liste des joueurs par équipe par tournoi dans les ligues
    et années spécifiées.

    Args:
        leagues: Liste des codes de ligue (ex: ["LFL", "PRM", "LEC"])
        years: Liste des années (ex: [2024, 2025])

    Returns:
        Liste de dicts avec les champs : Player, Team, Tournament, League, Year
    """
    # Construction de la clause WHERE pour filtrer les ligues et années
    # On utilise LIKE pour capturer les variantes de noms de tournois
    league_conditions = []
    for league in leagues:
        for year in years:
            league_conditions.append(
                f"(T.League = '{league}' AND T.Year = {year})"
            )

    if not league_conditions:
        return []

    where_clause = " OR ".join(league_conditions)

    logger.info(
        f"📋 Leaguepedia : récupération des rosters pour {len(leagues)} "
        f"ligues × {len(years)} années..."
    )

    results = _cargo_query_all(
        tables="TournamentRosters=TR, Tournaments=T",
        fields=(
            "TR.Player, TR.Team, TR.Tournament, "
            "T.League, T.Year, T.DateStart, T.Split"
        ),
        where=where_clause,
        order_by="T.Year, T.League, TR.Team",
    )

    logger.info(f"   → {len(results)} entrées de roster récupérées")
    return results


def fetch_player_appearances(leagues: list[str], years: list[int]) -> list[dict]:
    """
    Récupère les apparitions de joueurs en match dans les ligues ciblées.

    Utilise la table ScoreboardPlayers pour savoir dans quelle ligue
    chaque joueur a joué, à quelle date.

    C'est une alternative aux TournamentRosters, plus fiable car basée
    sur les matchs réellement joués (pas juste les rosters déclarés).

    Args:
        leagues: Codes de ligue
        years: Années

    Returns:
        Liste de dicts avec : Name, Team, Tournament, DateTime_UTC
    """
    league_conditions = []
    for league in leagues:
        for year in years:
            league_conditions.append(
                f"(T.League = '{league}' AND T.Year = {year})"
            )

    if not league_conditions:
        return []

    where_clause = " OR ".join(league_conditions)

    logger.info(
        f"🎮 Leaguepedia : récupération des apparitions en match pour "
        f"{len(leagues)} ligues × {len(years)} années..."
    )

    results = _cargo_query_all(
        tables="ScoreboardPlayers=SP, ScoreboardGames=SG, Tournaments=T",
        fields=(
            "SP.Name, SP.Team, SP.Champion, SP.Role, "
            "SG.Tournament, T.League, T.Year, SG.DateTime_UTC"
        ),
        where=where_clause + " AND SG.Tournament = T.Name AND SP.GameId = SG.GameId",
        order_by="T.Year, SP.Name",
    )

    logger.info(f"   → {len(results)} apparitions en match récupérées")
    return results


def build_player_career_map(
    appearances: list[dict],
) -> dict[str, dict[str, set[str]]]:
    """
    Construit une map {joueur → {année → set(ligues)}}.

    À partir des apparitions en match, on reconstruit l'historique de carrière
    de chaque joueur : dans quelle ligue a-t-il joué chaque année ?

    Exemple de résultat :
        {
            "Caps": {
                "2024": {"LEC"},
                "2025": {"LEC"},
            },
            "SomeRookie": {
                "2024": {"LFL"},
                "2025": {"LEC"},  # → Promu !
            }
        }

    Args:
        appearances: Liste de dicts issus de fetch_player_appearances

    Returns:
        Map joueur → année → set de ligues
    """
    career_map = {}

    for entry in appearances:
        player = entry.get("Name", "").strip()
        league = entry.get("League", "").strip()
        year = entry.get("Year", "").strip()

        if not player or not league or not year:
            continue

        if player not in career_map:
            career_map[player] = {}
        if year not in career_map[player]:
            career_map[player][year] = set()

        career_map[player][year].add(league)

    logger.info(
        f"📊 Career map construite : {len(career_map)} joueurs uniques"
    )
    return career_map


def build_promotion_labels(
    career_map: dict[str, dict[str, set[str]]],
    top_league: str = TOP_LEAGUE,
    erl_leagues: list[str] | None = None,
) -> dict[str, bool]:
    """
    Détermine quels joueurs ont été "promus" vers la LEC.

    Logique :
      Un joueur est considéré "promu" s'il remplit ces critères :
      1. Il a joué dans au moins une ERL (Div 1 ou Div 2)
      2. Il a ensuite joué en LEC (même année ou année suivante)

    Cette logique est volontairement simple. Des cas edge existent :
      - Joueurs qui font des allers-retours (promu puis redescendu)
      - Joueurs importés d'autres régions (pas vraiment "promus")
      - Joueurs qui sautent directement du Solo Queue à la LEC

    Args:
        career_map: Map {joueur → {année → set(ligues)}}
        top_league: La ligue "top" (par défaut LEC)
        erl_leagues: Liste des ERLs. Par défaut : toutes les ERLs de config.py

    Returns:
        Dict {nom_joueur: True/False} indiquant si le joueur a été promu
    """
    if erl_leagues is None:
        erl_leagues = list(ERL_DIV1_LEAGUES.keys()) + list(ERL_DIV2_LEAGUES.keys())

    promotions = {}
    promoted_count = 0

    for player, years_data in career_map.items():
        sorted_years = sorted(years_data.keys())
        was_in_erl = False
        was_promoted = False

        for i, year in enumerate(sorted_years):
            leagues_this_year = years_data[year]

            # Vérifier si le joueur est en ERL cette année
            if any(league in erl_leagues for league in leagues_this_year):
                was_in_erl = True

            # Vérifier si le joueur est en LEC cette année
            if top_league in leagues_this_year and was_in_erl:
                was_promoted = True
                break

        promotions[player] = was_promoted
        if was_promoted:
            promoted_count += 1

    logger.info(
        f"🏆 Labels de promotion construits : {promoted_count} joueurs promus "
        f"sur {len(promotions)} total ({promoted_count / max(len(promotions), 1) * 100:.1f}%)"
    )
    return promotions


def save_career_data(
    career_map: dict,
    promotions: dict,
    output_dir: Path = EXTERNAL_DATA_DIR,
) -> Path:
    """
    Sauvegarde les données de carrière et les labels de promotion en JSON.

    On sauvegarde les résultats Leaguepedia pour ne pas avoir à re-requêter
    l'API à chaque exécution du pipeline.

    Args:
        career_map: Map joueur → années → ligues
        promotions: Dict joueur → promoted (True/False)
        output_dir: Dossier de destination

    Returns:
        Chemin du fichier sauvegardé
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convertir les sets en listes pour la sérialisation JSON
    serializable_map = {}
    for player, years_data in career_map.items():
        serializable_map[player] = {
            year: list(leagues) for year, leagues in years_data.items()
        }

    data = {
        "career_map": serializable_map,
        "promotions": promotions,
        "metadata": {
            "source": "Leaguepedia Cargo API",
            "leagues_queried": list(ERL_DIV1_LEAGUES.keys())
            + list(ERL_DIV2_LEAGUES.keys())
            + [TOP_LEAGUE],
            "years_queried": DATA_YEARS,
        },
    }

    output_path = output_dir / "leaguepedia_careers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.success(f"💾 Données Leaguepedia sauvegardées → {output_path}")
    return output_path


def load_career_data(
    input_dir: Path = EXTERNAL_DATA_DIR,
) -> tuple[dict, dict] | None:
    """
    Charge les données de carrière depuis le cache JSON.

    Returns:
        Tuple (career_map, promotions) ou None si le fichier n'existe pas
    """
    filepath = input_dir / "leaguepedia_careers.json"
    if not filepath.exists():
        logger.info("Pas de cache Leaguepedia trouvé — requête API nécessaire")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    career_map = {
        player: {year: set(leagues) for year, leagues in years.items()}
        for player, years in data["career_map"].items()
    }
    promotions = data["promotions"]

    logger.info(
        f"Cache Leaguepedia chargé : {len(career_map)} joueurs, "
        f"{sum(1 for v in promotions.values() if v)} promus"
    )
    return career_map, promotions


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline complet
# ═══════════════════════════════════════════════════════════════════════════════


def run_leaguepedia_pipeline(force: bool = False) -> tuple[dict, dict]:
    """
    Exécute le pipeline complet de récupération des données Leaguepedia.

    Étapes :
      1. Vérifier si un cache existe (pour éviter de re-requêter)
      2. Récupérer les apparitions en match via l'API Cargo
      3. Construire la career map (joueur → années → ligues)
      4. Construire les labels de promotion
      5. Sauvegarder en JSON

    Args:
        force: Si True, ignore le cache et re-requête l'API

    Returns:
        Tuple (career_map, promotions)
    """
    logger.info(f"{'='*60}")
    logger.info(f"📚 LEAGUEPEDIA — Récupération des données de carrière")
    logger.info(f"{'='*60}")

    # Vérifier le cache
    if not force:
        cached = load_career_data()
        if cached is not None:
            return cached

    # Récupérer les apparitions en match
    all_leagues = (
        list(ERL_DIV1_LEAGUES.keys())
        + list(ERL_DIV2_LEAGUES.keys())
        + [TOP_LEAGUE]
    )

    appearances = fetch_player_appearances(
        leagues=all_leagues,
        years=DATA_YEARS,
    )

    if not appearances:
        logger.warning(
            "⚠️  Aucune donnée récupérée depuis Leaguepedia. "
            "L'API est peut-être indisponible. "
            "Le pipeline continuera sans ces données — la target variable "
            "sera construite uniquement depuis Oracle's Elixir."
        )
        return {}, {}

    # Construire la career map et les labels
    career_map = build_player_career_map(appearances)
    promotions = build_promotion_labels(career_map)

    # Sauvegarder
    save_career_data(career_map, promotions)

    logger.info(f"{'='*60}")
    logger.success("✅ Pipeline Leaguepedia terminé")
    logger.info(f"{'='*60}")

    return career_map, promotions


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI : python -m src.data.leaguepedia
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    career_map, promotions = run_leaguepedia_pipeline()

    # Afficher un résumé
    promoted_players = [p for p, v in promotions.items() if v]
    if promoted_players:
        logger.info(f"\n🏆 Joueurs promus identifiés ({len(promoted_players)}) :")
        for player in sorted(promoted_players)[:20]:
            logger.info(f"   → {player}")
        if len(promoted_players) > 20:
            logger.info(f"   ... et {len(promoted_players) - 20} autres")
