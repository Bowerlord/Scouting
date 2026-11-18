"""
app.py — Coquille Streamlit de l'application web ERL Scout.

Refondue le 2026-11-18. Le dashboard Streamlit (liste de surveillance,
leaderboard, profil, scout mode) est remplacé par une application React, dans
web/, qui fait ce que Streamlit ne sait pas faire : glisser-déposer, fiche en
tiroir, recherche Ctrl+K, état gardé dans le navigateur.

L'adresse publique reste scouting-lol-erl.streamlit.app. Streamlit ne sert donc
plus que de coquille : il lit les résultats du pipeline, les injecte dans la
page, et affiche l'application construite (app/web/index.html) en plein écran.

Pourquoi injecter les données plutôt que laisser l'app appeler l'API :
  - aucune dépendance à la version déployée de l'API ni à son réveil à froid ;
  - les données sont celles du dépôt, rafraîchies le lundi et le jeudi.
L'API reste utilisée pour les profils proches et par l'agent.

Pour reconstruire l'application après une modification de web/ :
  cd web && npm run build:streamlit

Lancement :
  streamlit run app/app.py        (depuis la racine du projet)
"""

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from api.data import load  # noqa: E402  (le chemin du dépôt doit précéder l'import)
from api.serialization import frame_to_dicts  # noqa: E402

st.set_page_config(page_title="ERL Scout", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

METRIQUES = RACINE / "reports" / "metrics"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

COLONNES = [
    "playername", "playername_original", "league", "_source_year", "split", "position", "teamname",
    "talent_score", "games_played", "win_rate", "archetype", "champion_pool_size", "promoted_to_lec",
    "dpm_zscore", "cspm_zscore", "golddiffat15_zscore",
]


@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees() -> str:
    """Les joueurs au format de l'API, sérialisés une fois par heure au plus."""
    joueurs = load(METRIQUES).players
    lignes = frame_to_dicts(joueurs[[c for c in COLONNES if c in joueurs.columns]], {"_source_year": "season"})
    meta = METRIQUES / "refresh_metadata.json"
    mis_a_jour = json.loads(meta.read_text(encoding="utf-8")).get("generated_at") if meta.exists() else None
    # « </ » fermerait la balise script si un pseudo le contenait.
    return json.dumps({"rows": lignes, "refreshed": mis_a_jour}, ensure_ascii=False).replace("</", "<\\/")


# Tout le chrome de Streamlit disparaît : la page n'est plus que l'application.
st.markdown(
    """
    <style>
    header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
    .block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow: hidden; }
    iframe { display: block; width: 100% !important; height: 100vh !important; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not PAGE.exists():
    st.error("L'application n'est pas construite : lancer « npm run build:streamlit » dans web/.")
    st.stop()

page = PAGE.read_text(encoding="utf-8").replace(
    "<head>", f"<head><script>window.__ERL__={charger_donnees()};</script>", 1
)
components.html(page, height=900, scrolling=False)
