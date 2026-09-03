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
from src.data.downloader import _download_from_gdrive

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


def test_page_quota_google_rejetee_sans_ecriture(tmp_path: Path, patch_session):
    """Le cas qui cassait le refresh : HTML de quota, pas de CSV."""
    patch_session(FakeResponse(QUOTA_PAGE, content_type="text/html; charset=utf-8"))
    destination = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"

    assert _download_from_gdrive("fake_id", destination) is False
    assert not destination.exists(), "aucun fichier ne doit être écrit"


def test_fichier_trop_petit_rejete_et_supprime(tmp_path: Path, patch_session):
    """Un CSV tronqué ne doit pas être accepté comme valide."""
    patch_session(FakeResponse(VALID_HEADER + b"ESPORTSTMNT01_1,complete,LFL,x,mid,3\n"))
    destination = tmp_path / "tronque.csv"

    assert _download_from_gdrive("fake_id", destination) is False
    assert not destination.exists()


def test_entete_non_oracle_rejetee(tmp_path: Path, patch_session):
    """Bonne taille mais mauvais fichier : l'en-tête doit contenir `gameid`."""
    contenu = b"colonne_a,colonne_b\n" + b"1,2\n" * MIN_VALID_CSV_BYTES
    patch_session(FakeResponse(contenu))
    destination = tmp_path / "mauvais_schema.csv"

    assert _download_from_gdrive("fake_id", destination) is False
    assert not destination.exists()


def test_csv_valide_accepte(tmp_path: Path, patch_session):
    """Cas nominal : un vrai CSV passe les trois garde-fous."""
    patch_session(FakeResponse(_valid_csv()))
    destination = tmp_path / "2024_LoL_esports_match_data_from_OraclesElixir.csv"

    assert _download_from_gdrive("fake_id", destination) is True
    assert destination.exists()
    assert destination.stat().st_size > MIN_VALID_CSV_BYTES
