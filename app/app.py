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
from utils.data_loader import load_model_metrics, load_refresh_metadata, load_talent_scores

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
    Oracle's Elixir.
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
        # len(df) compte des lignes joueur/split ; on affiche les deux niveaux
        st.metric(
            label="Joueurs analysés",
            value=f"{df['playername'].nunique():,}",
            help=f"{len(df):,} lignes joueur × split au total.",
        )

    with col2:
        st.metric(
            label="Ligues couvertes",
            value=df["league"].nunique(),
        )

    with col3:
        n_promoted = df.loc[df["promoted_to_lec"], "playername"].nunique()
        st.metric(
            label="Promus en LEC",
            value=int(n_promoted),
            help="Joueurs uniques dont la promotion en LEC est observée dans les données.",
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

    # Chiffres lus depuis talent_score_results.json (généré par le pipeline)
    # pour rester exacts après chaque ré-entraînement.
    metrics = load_model_metrics()
    best_model = metrics.get("best_model", "N/A")
    comparison = metrics.get("comparison", [])
    best_pr_auc = next(
        (m.get("pr_auc") for m in comparison if m.get("model_name") == best_model),
        None,
    )
    pr_auc_label = f"(PR-AUC : {best_pr_auc:.3f})" if best_pr_auc is not None else ""
    train_years = metrics.get("train_years", [])
    test_years = metrics.get("test_years", [])
    split_label = (
        f"entraîné {'/'.join(map(str, train_years))} → testé {'/'.join(map(str, test_years))}"
        if train_years and test_years
        else "split temporel Out-of-Time"
    )

    st.markdown(
        f"""
        **Pipeline ML** entraîné sur Oracle's Elixir.

        - **Meilleur modèle** : {best_model} {pr_auc_label}
        - **Calibration** : probabilités calibrées (Platt)
        - **Clustering** : K-Means par position
        - **Features** : z-scores DPM, CSPM, Gold@15, XP@15...
        - **Split temporel** : {split_label}
        """
    )
    st.markdown("---")

    # Indicateur de fraîcheur des snapshots (absent → omis silencieusement)
    refresh_meta = load_refresh_metadata()
    if refresh_meta.get("data_max_date"):
        generated_label = (
            f" (pipeline exécuté le {refresh_meta['generated_at']})"
            if refresh_meta.get("generated_at")
            else ""
        )
        st.caption(
            f"📅 Données à jour du {refresh_meta['data_max_date']}{generated_label}"
        )

    st.caption("KCorp Scouting Tool")
