"""
test_downloader.py — Tests unitaires des garde-fous de téléchargement

Contexte : le workflow Data Refresh a échoué 8 semaines d'affilée en
annonçant une « dérive de schéma Oracle's Elixir ». Le vrai problème était
ailleurs : Google Drive renvoyait une page HTML « Quota exceeded » de 2 Ko,
que le contrôle d'alors laissait passer (il ne testait le contenu qu'en
dessous de 1 Ko). Le fichier était écrit tel quel, annoncé « téléchargé
(0.0 Mo) », et l'échec ne remontait que trois étapes plus loin.

On teste donc les trois garde-fous, dans l'ordre où ils s'appliquent :
  - Content-Type HTML → rejet immédiat, avant écriture
  - fichier trop petit → rejet et suppression
  - en-tête sans colonne `gameid` → rejet et suppression
et le cas nominal, qui doit toujours passer.
"""

from pathlib import Path

import pytest

from src.config import MIN_VALID_CSV_BYTES
from src.data.downloader import (
    DownloadOutcome,
    DownloadReport,
    _download_from_gdrive,
)

# ── Doublure de réponse HTTP ──────────────────────────────────────────────────


class FakeResponse:
    """Réponse requests minimale : juste ce que _download_from_gdrive lit."""

    def __init__(self, content: bytes, content_type: str = "application/octet-stream"):
        self._content = content
        self.headers = {"Content-Type": content_type}
        self.cookies = {}

    @property
    def text(self) -> str:
        return self._content.decode("utf-8", errors="ignore")

    def iter_content(self, chunk_size: int = 32768):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, response: FakeResponse):
        self._response = response

    def get(self, *args, **kwargs):
        return self._response


@pytest.fixture
def patch_session(monkeypatch):
    """Remplace requests.Session par une doublure servant `response`."""

    def _patch(response: FakeResponse):
        monkeypatch.setattr(
            "src.data.downloader.requests.Session", lambda: FakeSession(response)
        )

    return _patch


# ── Contenus de test ──────────────────────────────────────────────────────────

QUOTA_PAGE = (
    b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>"
    b"</head><body>Too many users have viewed or downloaded this file "
    b"recently.</body></html>"
)

VALID_HEADER = b"gameid,datacompleteness,league,playername,position,kills\n"


def _valid_csv() -> bytes:
    """Un CSV plausible, au-dessus du seuil de taille minimal."""
    row = b"ESPORTSTMNT01_1,complete,LFL,someplayer,mid,3\n"
    repeats = (MIN_VALID_CSV_BYTES // len(row)) + 10
    return VALID_HEADER + row * repeats


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_page_quota_google_classee_source_indisponible(tmp_path: Path, patch_session):
    """Le cas qui cassait le refresh : HTML de quota, pas de CSV.

    L'issue doit être SOURCE_UNAVAILABLE et non INVALID_CONTENT : c'est ce
    qui permet au workflow de s'arrêter sans alerter, puisqu'il n'y a rien
    à corriger dans le dépôt.
    """
    patch_session(FakeResponse(QUOTA_PAGE, content_type="text/html; charset=utf-8"))
    destination = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"

    assert (
        _download_from_gdrive("fake_id", destination)
        is DownloadOutcome.SOURCE_UNAVAILABLE
    )
    assert not destination.exists(), "aucun fichier ne doit être écrit"


def test_page_html_sans_quota_classee_contenu_invalide(tmp_path: Path, patch_session):
    """Une page HTML qui ne parle pas de quota = ID mort, donc action requise."""
    page = b"<!DOCTYPE html><html><head><title>Page not found</title></head></html>"
    patch_session(FakeResponse(page, content_type="text/html; charset=utf-8"))
    destination = tmp_path / "introuvable.csv"

    assert (
        _download_from_gdrive("fake_id", destination) is DownloadOutcome.INVALID_CONTENT
    )


def test_fichier_trop_petit_rejete_et_supprime(tmp_path: Path, patch_session):
    """Un CSV tronqué ne doit pas être accepté comme valide."""
    patch_session(FakeResponse(VALID_HEADER + b"ESPORTSTMNT01_1,complete,LFL,x,mid,3\n"))
    destination = tmp_path / "tronque.csv"

    assert (
        _download_from_gdrive("fake_id", destination) is DownloadOutcome.INVALID_CONTENT
    )
    assert not destination.exists()


def test_entete_non_oracle_rejetee(tmp_path: Path, patch_session):
    """Bonne taille mais mauvais fichier : l'en-tête doit contenir `gameid`."""
    contenu = b"colonne_a,colonne_b\n" + b"1,2\n" * MIN_VALID_CSV_BYTES
    patch_session(FakeResponse(contenu))
    destination = tmp_path / "mauvais_schema.csv"

    assert (
        _download_from_gdrive("fake_id", destination) is DownloadOutcome.INVALID_CONTENT
    )
    assert not destination.exists()


def test_csv_valide_accepte(tmp_path: Path, patch_session):
    """Cas nominal : un vrai CSV passe les trois garde-fous."""
    patch_session(FakeResponse(_valid_csv()))
    destination = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"

    assert _download_from_gdrive("fake_id", destination) is DownloadOutcome.OK
    assert destination.exists()
    assert destination.stat().st_size > MIN_VALID_CSV_BYTES


# ── Arbitrage panne externe / pipeline cassé ──────────────────────────────────


def test_rapport_quota_seul_est_une_panne_externe():
    """Trois quotas : le workflow doit s'arrêter proprement, sans alerter."""
    rapport = DownloadReport(
        outcomes=dict.fromkeys([2024, 2025, 2026], DownloadOutcome.SOURCE_UNAVAILABLE)
    )
    assert rapport.source_unavailable is True
    assert rapport.missing == [2024, 2025, 2026]


def test_rapport_un_seul_contenu_invalide_suffit_a_alerter():
    """Un ID mort au milieu de deux quotas reste un vrai problème."""
    rapport = DownloadReport(
        outcomes={
            2024: DownloadOutcome.SOURCE_UNAVAILABLE,
            2025: DownloadOutcome.INVALID_CONTENT,
            2026: DownloadOutcome.SOURCE_UNAVAILABLE,
        }
    )
    assert rapport.source_unavailable is False


def test_rapport_tout_ok_nest_pas_une_panne():
    rapport = DownloadReport(
        outcomes=dict.fromkeys([2024, 2025, 2026], DownloadOutcome.OK)
    )
    assert rapport.source_unavailable is False
    assert rapport.missing == []
