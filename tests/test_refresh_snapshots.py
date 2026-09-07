"""
test_refresh_snapshots.py — Récupération des snapshots publiés par l'API

Contexte : l'image Docker embarque les résultats présents à sa construction.
Le 2026-09-07, le dépôt venait d'être rafraîchi et l'API servait encore les
données du 20 juillet, sept semaines plus tôt, sans que rien ne le signale.
Le module `api.refresh` va chercher les snapshots publiés au premier
chargement ; ces tests portent sur ce qu'il fait quand la source déraille,
parce que c'est là que se joue la disponibilité de l'API.
"""

from pathlib import Path

import pytest

from api import refresh

# ── Doublure de réponse HTTP ──────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def _contenu_valide(nom: str) -> bytes:
    """Un contenu qui passe les contrôles : JSON valide, ou CSV du pipeline."""
    if nom.endswith(".json"):
        return b'{"generated_at": "2026-09-07", "n_players": 995}'
    entete = b"playername,league,talent_score\n"
    return entete + b"unjoueur,LFL,0.5\n" * 200


@pytest.fixture
def reseau(monkeypatch):
    """Remplace requests.get et journalise les URL demandées."""
    monkeypatch.setenv("SCOUTING_REFRESH_SNAPSHOTS", "1")
    demandes: list[str] = []

    def _installer(reponse_pour):
        def _get(url, timeout=None):
            demandes.append(url)
            return reponse_pour(url)

        monkeypatch.setattr(refresh.requests, "get", _get)
        return demandes

    return _installer


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_desactivation_ne_touche_pas_au_reseau(tmp_path: Path, monkeypatch):
    """Sans le réseau, un clone du dépôt et la suite de tests doivent tourner."""
    monkeypatch.setenv("SCOUTING_REFRESH_SNAPSHOTS", "0")

    def _interdit(*args, **kwargs):
        raise AssertionError("le réseau ne doit pas être appelé quand c'est désactivé")

    monkeypatch.setattr(refresh.requests, "get", _interdit)

    assert refresh.recuperer_snapshots(tmp_path) is None


def test_snapshots_complets_sont_adoptes(tmp_path: Path, reseau):
    reseau(lambda url: FakeResponse(_contenu_valide(url)))

    resultat = refresh.recuperer_snapshots(tmp_path)

    assert resultat == tmp_path
    for nom in refresh.FICHIERS_REQUIS:
        assert (tmp_path / nom).exists()


def test_fichier_requis_manquant_annule_tout(tmp_path: Path, reseau):
    """Tout ou rien : mélanger deux dates casserait la fusion joueurs/clusters."""

    def _reponse(url):
        if url.endswith("clustering_results.csv"):
            return FakeResponse(b"", status_code=404)
        return FakeResponse(_contenu_valide(url))

    reseau(_reponse)

    assert refresh.recuperer_snapshots(tmp_path) is None
    # Le répertoire cible ne doit contenir aucun fichier à moitié récupéré.
    assert not list(tmp_path.glob("*.csv"))


def test_page_erreur_rejetee(tmp_path: Path, reseau):
    """Le piège qui a coûté sept semaines côté acquisition, dans l'autre sens.

    Une page d'erreur HTML répond 200 et ressemble à un fichier. Ici deux
    contrôles l'arrêtent avant qu'elle ne remplace un vrai snapshot : la taille,
    puis l'absence de la colonne `playername` dans l'en-tête.
    """
    page = b"<!DOCTYPE html><html><body>Not found</body></html>"
    reseau(lambda url: FakeResponse(page))

    assert refresh.recuperer_snapshots(tmp_path) is None


def test_panne_reseau_ramene_au_socle(tmp_path: Path, reseau, monkeypatch):
    """Une source injoignable n'est pas une erreur : on garde l'image."""

    def _get(url, timeout=None):
        raise refresh.requests.RequestException("réseau coupé")

    monkeypatch.setenv("SCOUTING_REFRESH_SNAPSHOTS", "1")
    monkeypatch.setattr(refresh.requests, "get", _get)

    assert refresh.recuperer_snapshots(tmp_path) is None


def test_url_surchargeable(tmp_path: Path, reseau, monkeypatch):
    """Un fork doit pouvoir pointer ses propres snapshots sans toucher au code."""
    monkeypatch.setenv("SCOUTING_SNAPSHOTS_URL", "https://exemple.test/metrics/")
    demandes = reseau(lambda url: FakeResponse(_contenu_valide(url)))

    refresh.recuperer_snapshots(tmp_path)

    assert demandes
    assert all(u.startswith("https://exemple.test/metrics/") for u in demandes)


def test_fichiers_optionnels_absents_ne_bloquent_pas(tmp_path: Path, reseau):
    """Sans les archétypes, l'API répond quand même, en mode dégradé."""

    def _reponse(url):
        if url.endswith(".json"):
            return FakeResponse(b"", status_code=404)
        return FakeResponse(_contenu_valide(url))

    reseau(_reponse)

    assert refresh.recuperer_snapshots(tmp_path) == tmp_path
    assert (tmp_path / "talent_scores_players.csv").exists()
    assert not (tmp_path / "cluster_profiles.json").exists()


def test_json_court_accepte(tmp_path: Path, reseau):
    """refresh_metadata.json fait 159 octets et porte la date de fraîcheur.

    Un seuil de taille uniforme le rejetait, ce qui laissait `data_refreshed_at`
    nul dans /health : l'API paraissait n'avoir jamais été rafraîchie alors que
    ses données l'étaient. Trouvé au premier essai réel contre GitHub.
    """
    court = b'{"generated_at": "2026-09-07"}'
    reseau(lambda url: FakeResponse(court if url.endswith(".json") else _contenu_valide(url)))

    assert refresh.recuperer_snapshots(tmp_path) == tmp_path
    assert (tmp_path / "refresh_metadata.json").read_bytes() == court


def test_json_invalide_rejete(tmp_path: Path, reseau):
    """Une page d'erreur renvoyée en 200 ne doit pas passer pour un JSON."""
    page = b"<html>404: Not Found</html>"
    reseau(lambda url: FakeResponse(page if url.endswith(".json") else _contenu_valide(url)))

    refresh.recuperer_snapshots(tmp_path)

    assert not (tmp_path / "refresh_metadata.json").exists()


def test_csv_sans_entete_attendu_rejete(tmp_path: Path, reseau):
    """Un CSV assez gros mais d'une autre source n'est pas un snapshot valide."""
    intrus = b"colonne_a,colonne_b\n" + b"1,2\n" * 500
    reseau(lambda url: FakeResponse(intrus if url.endswith(".csv") else _contenu_valide(url)))

    assert refresh.recuperer_snapshots(tmp_path) is None
