"""
app.py — Point d'entrée principal du KCorp Scouting Tool

Cette page d'accueil présente :
  - Un titre et une description de l'outil
  - Les métriques globales du dataset (joueurs analysés, ligues, promus en LEC)
  - Un guide de navigation vers les trois pages principales

Pourquoi Streamlit ?
  → Interface web interactive sans JavaScript, idéale pour des dashboards ML internes.
    Le re-render automatique à chaque widget simplifie le code au prix d'une
    re-exécution complète du script à chaque interaction — d'où l'importance
    du @st.cache_data dans utils/data_loader.py.

Lancement :
  streamlit run app/app.py        (depuis la racine du projet)

Intégration :
  app.py → utils/data_loader.py → reports/metrics/talent_scores_players.csv
"""

import sys
from pathlib import Path

# Ajouter app/ au sys.path pour que 'from utils.data_loader import ...' fonctionne
# quelle que soit la façon dont Streamlit est lancé
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.data_loader import load_talent_scores

# ══════════════════════════════════════════════════════════════════════════════
# Configuration de la page
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KCorp Scouting Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# En-tête
# ══════════════════════════════════════════════════════════════════════════════

st.title("🎯 KCorp Scouting Tool")
st.markdown(
    """
    Outil de scouting esport alimenté par le **Machine Learning** pour League of Legends.

    Identifiez les talents émergents des ligues ERL (LFL, PRM, LVP SL, NLC, TCL)
    avant leur promotion en **LEC** — grâce à un modèle entraîné sur les données
    Oracle's Elixir 2024-2025.
    """
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Métriques globales du dataset
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_talent_scores()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Joueurs analysés",
            value=f"{len(df):,}",
        )

    with col2:
        st.metric(
            label="Ligues couvertes",
            value=df["league"].nunique(),
        )

    with col3:
        n_promoted = int(df["promoted_to_lec"].sum())
        st.metric(
            label="Promus en LEC",
            value=n_promoted,
            help="Joueurs ayant effectivement atteint la LEC dans les données.",
        )

    with col4:
        years = sorted(df["_source_year"].dropna().unique().astype(int).tolist())
        years_label = " · ".join(str(y) for y in years)
        st.metric(
            label="Années",
            value=years_label,
        )

except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "L'application est prête. Lancez le pipeline ML pour générer les données, "
        "puis relancez Streamlit."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Guide de navigation
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Explorer l'outil")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        ### 📊 Leaderboard
        Classement de tous les joueurs par **talent score**.

        - Filtres : position, ligue, année, score minimum
        - Tableau interactif avec indicateur de promotion LEC
        - Bar chart top 20 coloré par ligue
        """
    )

with col_b:
    st.markdown(
        """
        ### 👤 Profil Joueur
        Vue détaillée d'un joueur individuel.

        - Radar chart multi-dimensionnel des z-scores
        - Métriques : talent score, win rate, games played, champion pool
        - Archétype de jeu issu du clustering K-Means
        """
    )

with col_c:
    st.markdown(
        """
        ### 🔍 Scout Mode
        Scouting avancé par critères et par similarité.

        - Shortlist filtrée par position, archétype, ligue, score
        - Recherche de joueurs similaires par clustering ML
        - Scatter plot comparatif des z-scores dans le cluster
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — informations sur le modèle
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### À propos du modèle")
    st.markdown(
        """
        **Pipeline ML** entraîné sur Oracle's Elixir.

        - **Meilleur modèle** : Logistic Regression (PR-AUC : 0.256)
        - **Clustering** : K-Means par position
        - **Features** : z-scores DPM, CSPM, Gold@15, XP@15...
        - **Split temporel** : entraîné 2024 → testé 2025
        """
    )
    st.markdown("---")
    st.caption("KCorp Scouting Tool — 2025")
