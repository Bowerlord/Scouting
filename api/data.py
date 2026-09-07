"""Chargement et accès aux résultats du pipeline.

Le chargement est fait une fois au démarrage puis gardé en mémoire : les
fichiers pèsent moins d'un mégaoctet et ne changent qu'au rafraîchissement
hebdomadaire, donc relire le disque à chaque requête coûterait sans rien
apporter. `reload()` existe pour les tests et pour un rechargement à chaud.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from api.config import METRICS_DIR, REFRESHED_METRICS_DIR
from api.refresh import recuperer_snapshots

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STORE: "DataStore | None" = None

# Colonnes utilisées pour mesurer la distance entre deux joueurs. Ce sont des
# z-scores, donc déjà centrées-réduites : aucune normalisation supplémentaire
# n'est nécessaire, et une distance euclidienne y est directement comparable.
SIMILARITY_FEATURES = ["dpm_zscore", "cspm_zscore", "golddiffat15_zscore", "win_rate_zscore"]


class DataNotAvailable(RuntimeError):
    """Les fichiers de résultats sont absents ou illisibles."""


@dataclass
class DataStore:
    """Les résultats du pipeline, chargés en mémoire."""

    players: pd.DataFrame
    clusters: pd.DataFrame
    archetypes: dict
    model_metrics: dict
    refresh_metadata: dict
    loaded_at: datetime
    source_dir: Path

    @property
    def player_count(self) -> int:
        return int(len(self.players))

    @property
    def leagues(self) -> list[str]:
        return sorted(self.players["league"].dropna().unique().tolist())

    @property
    def positions(self) -> list[str]:
        return sorted(self.players["position"].dropna().unique().tolist())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataNotAvailable(f"Fichier de résultats introuvable : {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load(metrics_dir: Path | None = None) -> DataStore:
    """Charge les résultats depuis le disque. Lève DataNotAvailable si absents."""
    directory = Path(metrics_dir) if metrics_dir else METRICS_DIR

    players = _read_csv(directory / "talent_scores_players.csv")
    clusters = _read_csv(directory / "clustering_results.csv")

    # Le clustering porte les archétypes et les coordonnées UMAP, le scoring
    # porte le talent_score. Les deux décrivent les mêmes joueurs-saisons : on
    # les fusionne une fois ici pour que les routes n'aient jamais à le refaire.
    # teamname fait partie de la cle : six joueurs ont change d equipe en cours
    # de split en 2025 et ont donc deux lignes distinctes pour la meme saison.
    # Sans lui la fusion echoue, et la dedupliquer ferait disparaitre une moitie
    # de saison bien reelle.
    key = ["playername", "league", "_source_year", "split", "position", "teamname"]
    cluster_cols = key + [c for c in ("cluster", "archetype", "umap_x", "umap_y", "win_rate_zscore") if c in clusters]
    merged = players.merge(clusters[cluster_cols], on=key, how="left", validate="one_to_one")

    return DataStore(
        players=merged,
        clusters=clusters,
        archetypes=_read_json(directory / "cluster_profiles.json"),
        model_metrics=_read_json(directory / "talent_score_results.json"),
        refresh_metadata=_read_json(directory / "refresh_metadata.json"),
        loaded_at=datetime.now(timezone.utc),
        source_dir=directory,
    )


def _load_avec_rafraichissement() -> DataStore:
    """Charge les snapshots publiés, avec repli sur ceux embarqués dans l'image.

    L'image fige les données au jour de sa construction. Sans ce rafraîchissement,
    l'API continue de servir les résultats de son dernier déploiement même quand
    le dépôt a été mis à jour : constaté le 2026-09-07, sept semaines d'écart
    entre le dashboard et l'API. Les fichiers de l'image restent le socle, et
    toute défaillance de la source distante y ramène sans bruit.
    """
    frais = recuperer_snapshots(REFRESHED_METRICS_DIR)
    if frais is not None:
        try:
            return load(frais)
        except Exception as erreur:  # noqa: BLE001 - tout échec doit ramener au socle
            logger.warning(
                f"Snapshots récupérés mais inexploitables ({erreur}). Repli sur ceux embarqués dans l'image."
            )
    return load()


def get_store() -> DataStore:
    """Renvoie le magasin de données, en le chargeant au premier appel."""
    global _STORE
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                _STORE = _load_avec_rafraichissement()
    return _STORE


def reload(metrics_dir: Path | None = None) -> DataStore:
    """Recharge depuis le disque. Utilisé par les tests et un éventuel endpoint d'admin."""
    global _STORE
    with _LOCK:
        _STORE = load(metrics_dir)
    return _STORE


def find_similar(store: DataStore, row: pd.Series, count: int) -> pd.DataFrame:
    """Renvoie les joueurs les plus proches sur le profil de jeu.

    La proximité est mesurée sur les z-scores de performance, pas sur le score
    de talent : deux joueurs peuvent avoir le même score global en jouant de
    façons opposées, et c'est précisément ce qu'un recruteur veut distinguer.
    On reste dans la même position, parce qu'un support et un mid n'ont pas
    des métriques comparables.
    """
    features = [c for c in SIMILARITY_FEATURES if c in store.players.columns]
    pool = store.players[store.players["position"] == row["position"]].copy()
    pool = pool[pool["playername"] != row["playername"]].dropna(subset=features)
    if pool.empty:
        return pool

    deltas = pool[features].to_numpy(dtype=float) - np.asarray(row[features], dtype=float)
    pool["distance"] = np.linalg.norm(deltas, axis=1)
    return pool.nsmallest(count, "distance")
