"""
1_Leaderboard.py — Classement interactif des joueurs par talent score

Cette page affiche :
  - Des filtres interactifs dans la sidebar (position, ligue, année, score minimum)
  - Un tableau trié par talent_score décroissant, avec colonne Promu LEC (✅/❌)
    et une ProgressColumn visuelle pour le score
  - Un bar chart horizontal Plotly des 20 meilleurs joueurs, coloré par ligue

Pourquoi un bar chart horizontal pour le top 20 ?
  → Les noms de joueurs sont longs — en horizontal, ils restent lisibles sans rotation.
    La couleur par ligue permet d'identifier visuellement la diversité géographique
    des meilleurs talents sans avoir à lire chaque ligne du tableau.

Intégration :
  pages/1_Leaderboard.py → utils/data_loader.py → reports/metrics/talent_scores_players.csv
"""

import sys
from pathlib import Path

# Remonter de pages/ → app/ pour que 'utils' soit importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.express as px
import streamlit as st
from utils.data_loader import load_talent_scores

# ══════════════════════════════════════════════════════════════════════════════
# Configuration de la page
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Leaderboard — KCorp Scouting",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Leaderboard")
st.markdown("Classement de tous les joueurs ERL par **talent score** ML.")

# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_talent_scores()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Filtres — sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Filtres")

    positions_available = sorted(df["position"].dropna().unique().tolist())
    selected_positions = st.multiselect(
        "Position",
        options=positions_available,
        default=positions_available,
    )

    leagues_available = sorted(df["league"].dropna().unique().tolist())
    selected_leagues = st.multiselect(
        "Ligue",
        options=leagues_available,
        default=leagues_available,
    )

    years_available = sorted(df["_source_year"].dropna().unique().astype(int).tolist())
    selected_year = st.selectbox(
        "Année",
        options=["Toutes"] + [str(y) for y in years_available],
        index=0,
    )

    # talent_score est sur une échelle 0-100 (probabilité calibrée × 100)
    min_score = st.slider(
        "Score minimum",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Application des filtres
# ══════════════════════════════════════════════════════════════════════════════

filtered = df.copy()

if selected_positions:
    filtered = filtered[filtered["position"].isin(selected_positions)]

if selected_leagues:
    filtered = filtered[filtered["league"].isin(selected_leagues)]

if selected_year != "Toutes":
    filtered = filtered[filtered["_source_year"] == int(selected_year)]

filtered = filtered[filtered["talent_score"] >= min_score]
filtered = filtered.sort_values("talent_score", ascending=False).reset_index(drop=True)
filtered.index += 1  # Classement lisible à partir de 1

# Affichage : pseudo dans sa casse d'origine quand le CSV la fournit
# (playername reste en minuscules — c'est la clé de jointure du pipeline)
if "playername_original" in filtered.columns:
    filtered["playername"] = filtered["playername_original"].fillna(filtered["playername"])

st.caption(f"{len(filtered)} joueur(s) affiché(s) après filtrage")

# ══════════════════════════════════════════════════════════════════════════════
# Tableau principal
# ══════════════════════════════════════════════════════════════════════════════

# Colonnes préférées (on garde seulement celles présentes dans le CSV)
display_cols = [
    "playername", "position", "league", "teamname",
    "_source_year", "split", "talent_score", "score_percentile",
    "win_rate", "games_played", "champion_pool_size",
    "promoted_to_lec",
]
display_cols = [c for c in display_cols if c in filtered.columns]

display_df = filtered[display_cols].copy()

# Formatage de la colonne cible en emoji
display_df["promoted_to_lec"] = display_df["promoted_to_lec"].apply(
    lambda x: "✅" if x else "❌"
)

if "talent_score" in display_df.columns:
    display_df["talent_score"] = display_df["talent_score"].round(1)

if "score_percentile" in display_df.columns:
    display_df["score_percentile"] = display_df["score_percentile"].round(1)

if "win_rate" in display_df.columns:
    display_df["win_rate"] = display_df["win_rate"].round(3)

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "playername": st.column_config.TextColumn("Joueur"),
        "position": st.column_config.TextColumn("Poste"),
        "league": st.column_config.TextColumn("Ligue"),
        "teamname": st.column_config.TextColumn("Équipe"),
        "_source_year": st.column_config.NumberColumn("Année", format="%d"),
        "split": st.column_config.TextColumn("Split"),
        "talent_score": st.column_config.ProgressColumn(
            "Talent Score",
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
        ),
        "score_percentile": st.column_config.NumberColumn(
            "Percentile (pos.)",
            format="%.1f",
            help="Rang percentile du joueur au sein de sa position (100 = meilleur).",
        ),
        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.3f"),
        "games_played": st.column_config.NumberColumn("Games"),
        "champion_pool_size": st.column_config.NumberColumn("Champ Pool"),
        "promoted_to_lec": st.column_config.TextColumn("Promu LEC"),
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# Bar chart horizontal — Top 20
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Top 20 — Talent Score")

top20 = filtered.head(20).copy()

if top20.empty:
    st.info("Aucun joueur ne correspond aux filtres sélectionnés.")
else:
    # Étiquette : nom + ligue pour identifier rapidement chaque barre
    top20["label"] = top20["playername"] + "  (" + top20["league"] + ")"

    # Trier par score croissant : Plotly affiche de bas en haut,
    # donc le meilleur joueur apparaîtra en haut du graphique
    top20_sorted = top20.sort_values("talent_score", ascending=True)

    fig = px.bar(
        top20_sorted,
        x="talent_score",
        y="label",
        color="league",
        orientation="h",
        title=f"Top {len(top20)} joueurs — Talent Score",
        labels={
            "talent_score": "Talent Score",
            "label": "",
            "league": "Ligue",
        },
        color_discrete_sequence=px.colors.qualitative.Plotly,
        text="talent_score",
    )

    fig.update_traces(
        texttemplate="%{x:.3f}",
        textposition="outside",
        textfont_color="#ffffff",
    )

    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#ffffff", size=12),
        title_font=dict(color="#0BFCE4", size=16),
        showlegend=True,
        legend=dict(
            bgcolor="#1a1a2e",
            bordercolor="#333355",
            borderwidth=1,
        ),
        height=max(400, len(top20) * 30),
        margin=dict(l=20, r=120, t=60, b=20),
    )

    fig.update_xaxes(
        range=[0, 1.1],
        gridcolor="#333355",
        zerolinecolor="#555577",
        tickfont=dict(color="#aaaacc"),
        title_font=dict(color="#aaaacc"),
    )
    fig.update_yaxes(
        gridcolor="#333355",
        tickfont=dict(color="#ffffff"),
    )

    st.plotly_chart(fig, use_container_width=True)
