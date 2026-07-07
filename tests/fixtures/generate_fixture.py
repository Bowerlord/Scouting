"""
generate_fixture.py — Génère la fixture oracle_sample.csv (données synthétiques)

CSV synthétique reproduisant le schéma brut Oracle's Elixir, assez riche pour
exercer le VRAI pipeline (cleaner → feature engineering → talent scorer →
clusterer) dans les tests d'intégration :

- colonnes = KEY_COLUMNS SANS `killparticipation` (reproduit la dérive réelle
  du format Oracle's Elixir) + quelques colonnes parasites,
- années 2024 (TRAIN_YEARS) et 2025 (TEST_YEARS), ligues LFL / LFL2 / LEC,
- 12 lignes par match : 10 joueurs (5 positions × 2 équipes) + 2 lignes "team",
- 6 matchs par équipe et par split (survit au filtre min_games=5),
- cas de target scriptés (voir PLAYER CASES ci-dessous),
- NaN épars dans `vspm` (teste l'imputation) et un pseudo mal formaté.

Cas de target datée (horizon 18 mois, LEC = promotion) :
- futurestar   : LFL 2024 → débute en LEC le 2025-01-15  → positif en 2024
- risingstar   : LFL2 2025 → débute en LEC le 2025-12-15 → positif en 2025
- lateprospect : LFL Spring 2024 uniquement, débute en LEC le 2025-12-15,
                 soit > 18 mois après ses matchs de févr. 2024 → négatif
- washedup     : LEC début 2024 puis relégué en LFL2 2025  → négatif (ex-LEC)
- tous les autres joueurs : jamais promus → négatifs

Usage :
  python tests/fixtures/generate_fixture.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUTPUT = Path(__file__).parent / "oracle_sample.csv"

POSITIONS = ["top", "jng", "mid", "bot", "sup"]
CHAMPIONS = [
    "Aatrox", "Ahri", "Akali", "Ashe", "Azir", "Corki", "Draven", "Ezreal",
    "Gnar", "Jinx", "KSante", "LeeSin", "Leona", "Lulu", "Nautilus", "Orianna",
    "Rell", "Sejuani", "Thresh", "Viego", "Xayah", "Yone", "Zeri", "Zoe",
]

# Moyennes de stats par position (dpm, cspm, vspm) pour des données plausibles
POSITION_PROFILES = {
    "top": (450, 8.0, 1.0),
    "jng": (350, 5.5, 1.4),
    "mid": (550, 8.5, 1.1),
    "bot": (600, 9.0, 1.0),
    "sup": (180, 1.2, 2.6),
}

# Joueurs avec un cas de target scripté (voir docstring)
FUTURESTAR = "futurestar"      # LFL T1 mid 2024, LEC dès 2025-01-15
RISINGSTAR = "risingstar"      # LFL2 T1 jng 2025, LEC dès 2025-12-15
LATEPROSPECT = "lateprospect"  # LFL T2 top Spring 2024, LEC 2025-12-15 (hors horizon)
WASHEDUP = "washedup"          # LEC T1 sup 2024, LFL2 T2 sup 2025 (ex-LEC relégué)
MESSY_NAME = "  MessyName "    # teste la normalisation strip().lower()


def roster(league: str, team: str, year: int) -> dict[str, str]:
    """Retourne {position: playername} pour une équipe/année, avec les cas scriptés."""
    players = {pos: f"{league.lower()}{team[-1]}{pos}" for pos in POSITIONS}

    if league == "LFL" and team.endswith("1"):
        players["mid"] = FUTURESTAR if year == 2024 else "lflsubmid"
        players["bot"] = MESSY_NAME  # pseudo mal formaté, toutes années
    if league == "LFL" and team.endswith("2") and year == 2024:
        # lateprospect ne joue que le Spring — remplacé au Summer (géré plus bas)
        players["top"] = LATEPROSPECT
    if league == "LFL2" and team.endswith("1"):
        players["jng"] = RISINGSTAR if year == 2025 else "oldjungler"
    if league == "LFL2" and team.endswith("2"):
        players["sup"] = WASHEDUP if year == 2025 else "oldsupp"
    if league == "LEC" and team.endswith("1"):
        players["sup"] = WASHEDUP if year == 2024 else "lecnewsup"
        players["mid"] = FUTURESTAR if year == 2025 else "lecmid1"
    return players


def split_dates(year: int, split: str, n_rounds: int) -> list[str]:
    """Dates espacées d'une semaine : Spring dès mi-janvier, Summer dès mi-juin."""
    start = pd.Timestamp(f"{year}-01-15") if split == "Spring" else pd.Timestamp(f"{year}-06-15")
    return [(start + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n_rounds)]


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    def add_game(gameid: str, league: str, split: str, date: str, year: int,
                 team_a: str, team_b: str, rosters: dict[str, dict[str, str]]) -> None:
        winner = rng.choice([team_a, team_b])
        gamelength = int(rng.integers(1600, 2600))
        for team in (team_a, team_b):
            result = int(team == winner)
            team_rows = []
            for pos in POSITIONS:
                player = rosters[team][pos]
                dpm_mu, cspm_mu, vspm_mu = POSITION_PROFILES[pos]
                # Les futurs promus sur-performent : donne un signal au modèle
                boost = 1.25 if player in (FUTURESTAR, RISINGSTAR) else 1.0
                kills = int(rng.poisson(3 * boost))
                deaths = int(rng.poisson(2))
                assists = int(rng.poisson(6 * boost))
                row = {
                    "gameid": gameid,
                    "league": league,
                    "split": split,
                    "date": date,
                    "position": pos,
                    "playername": player,
                    "teamname": team,
                    "champion": CHAMPIONS[int(rng.integers(0, len(CHAMPIONS)))],
                    "result": result,
                    "kills": kills,
                    "deaths": deaths,
                    "assists": assists,
                    "killsat15": int(rng.poisson(1)),
                    "deathsat15": int(rng.poisson(1)),
                    "assistsat15": int(rng.poisson(2)),
                    "csat15": round(float(rng.normal(130 if pos != "sup" else 20, 15)), 1),
                    "csdiffat15": round(float(rng.normal(5 * (boost - 1) * 10, 12)), 1),
                    "golddiffat15": round(float(rng.normal(150 * (boost - 1) * 10, 400)), 1),
                    "xpdiffat15": round(float(rng.normal(100 * (boost - 1) * 10, 300)), 1),
                    "totalgold": int(rng.normal(12000, 1500)),
                    "earnedgold": int(rng.normal(8000, 1200)),
                    "minionkills": int(rng.normal(220 if pos != "sup" else 35, 30)),
                    "monsterkills": int(rng.normal(120 if pos == "jng" else 10, 8)),
                    "wardskilled": int(rng.poisson(6)),
                    "wardsplaced": int(rng.poisson(25 if pos == "sup" else 10)),
                    "visionscore": round(float(rng.normal(80 if pos == "sup" else 40, 10)), 1),
                    "damagetochampions": int(rng.normal(dpm_mu * gamelength / 60, 2000)),
                    "damageshare": round(float(np.clip(rng.normal(0.2, 0.04), 0.05, 0.45)), 4),
                    "dpm": round(float(rng.normal(dpm_mu * boost, 60)), 2),
                    "cspm": round(float(rng.normal(cspm_mu * boost, 0.6)), 2),
                    # NaN épars (~5%) pour tester l'imputation par médiane
                    "vspm": round(float(rng.normal(vspm_mu, 0.3)), 2)
                    if rng.random() > 0.05
                    else np.nan,
                    "earnedgoldshare": round(float(np.clip(rng.normal(0.2, 0.03), 0.05, 0.4)), 4),
                    "gamelength": gamelength,
                    # Colonnes parasites : les CSV réels ont ~166 colonnes
                    "patch": f"{14 if year == 2024 else 15}.{int(rng.integers(1, 15))}",
                    "datacompleteness": "complete",
                    "url": f"https://example.invalid/{gameid}",
                }
                rows.append(row)
                team_rows.append(row)
            # 2 lignes "team" par match (agrégats) — supprimées par le cleaner
            rows.append({
                **{k: team_rows[0][k] for k in ("gameid", "league", "split", "date",
                                                "teamname", "result", "gamelength",
                                                "patch", "datacompleteness", "url")},
                "position": "team",
                "playername": np.nan,
                "champion": np.nan,
                "kills": sum(r["kills"] for r in team_rows),
                "deaths": sum(r["deaths"] for r in team_rows),
                "assists": sum(r["assists"] for r in team_rows),
            })

    # ── Ligues ERL : 2 équipes, 6 matchs par split, Spring + Summer, 2024 + 2025
    for league in ("LFL", "LFL2"):
        for year in (2024, 2025):
            rosters = {f"{league} T{i}": roster(league, f"T{i}", year) for i in (1, 2)}
            for split in ("Spring", "Summer"):
                # lateprospect ne joue que le Spring 2024 (remplacé ensuite)
                if league == "LFL" and year == 2024 and split == "Summer":
                    rosters["LFL T2"] = dict(rosters["LFL T2"], top="lflsubtop")
                for i, date in enumerate(split_dates(year, split, n_rounds=6)):
                    gameid = f"{league}_{year}_{split}_{i}"
                    add_game(gameid, league, split, date, year,
                             f"{league} T1", f"{league} T2", rosters)

    # ── LEC : définit les dates de début (promotions) des cas scriptés
    for year, split, n_rounds in ((2024, "Spring", 4), (2025, "Spring", 4)):
        rosters = {f"LEC T{i}": roster("LEC", f"T{i}", year) for i in (1, 2)}
        for i, date in enumerate(split_dates(year, split, n_rounds)):
            gameid = f"LEC_{year}_{split}_{i}"
            add_game(gameid, "LEC", split, date, year, "LEC T1", "LEC T2", rosters)

    # Matchs LEC de décembre 2025 : débuts de risingstar et lateprospect
    december_rosters = {f"LEC T{i}": roster("LEC", f"T{i}", 2025) for i in (1, 2)}
    december_rosters["LEC T1"] = dict(december_rosters["LEC T1"], jng=RISINGSTAR)
    december_rosters["LEC T2"] = dict(december_rosters["LEC T2"], top=LATEPROSPECT)
    for i in range(2):
        date = (pd.Timestamp("2025-12-15") + pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        add_game(f"LEC_2025_Winter_{i}", "LEC", "Winter", date, 2025,
                 "LEC T1", "LEC T2", december_rosters)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUTPUT, index=False)
    print(f"Fixture générée : {OUTPUT} ({len(df)} lignes × {len(df.columns)} colonnes)")
