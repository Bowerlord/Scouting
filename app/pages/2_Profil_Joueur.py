"""
2_Profil_Joueur.py — Vue détaillée d'un joueur individuel

Cette page affiche pour un joueur sélectionné :
  - Métriques clés : talent score, promotion LEC, win rate, games played, champion pool
  - Archétype de jeu issu du clustering K-Means (via cluster_profiles.json)
  - Radar chart Plotly sur les z-scores disponibles dans le CSV

Pourquoi un radar chart pour les z-scores ?
  → Visualisation multi-dimensionnelle intuitive des forces et faiblesses d'un joueur
    par rapport à la moyenne de la ligue (z-score = 0 = dans la moyenne).
    Un joueur elite aura des z-scores positifs dans toutes les dimensions clés.
    Le radar chart révèle le style de jeu : un top très positif en DPM mais négatif
    en vision est clairement un "carry agressif" plutôt qu'un "playmaker".

Intégration :
  pages/2_Profil_Joueur.py → utils/data_loader.py → reports/metrics/
    → talent_scores_players.csv  (métriques joueur)
    → clustering_results.csv     (cluster du joueur)
    → cluster_profiles.json      (archétype du cluster)
"""

import sys
from pathlib import Path

# Remonter de pages/ → app/ pour que 'utils' soit importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import (
    get_archetype,
    load_cluster_profiles,
    load_clustering_results,
    load_talent_scores,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration de la page
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Profil Joueur — KCorp Scouting",
    page_icon="👤",
    layout="wide",
)

st.title("👤 Profil Joueur")
st.markdown("Vue détaillée des métriques et de l'archétype ML d'un joueur.")

# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

try:
    df_talent = load_talent_scores()
    cluster_profiles = load_cluster_profiles()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Clustering optionnel — l'app reste fonctionnelle sans clustering_results.csv
try:
    df_clusters = load_clustering_results()
    clustering_available = True
except FileNotFoundError:
    df_clusters = pd.DataFrame()
    clustering_available = False

# ══════════════════════════════════════════════════════════════════════════════
# Sélection du joueur
# ══════════════════════════════════════════════════════════════════════════════

player_names = sorted(df_talent["playername"].dropna().unique().tolist())
selected_player = st.selectbox(
    "Rechercher un joueur",
    options=player_names,
    help="Tapez pour filtrer la liste.",
)

if not selected_player:
    st.info("Sélectionnez un joueur pour afficher son profil.")
    st.stop()

# Si un joueur apparaît sur plusieurs splits/années, on prend la ligne
# avec le meilleur talent_score (la "saison peak" du joueur)
player_rows = df_talent[df_talent["playername"] == selected_player]
player = player_rows.sort_values("talent_score", ascending=False).iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# En-tête du profil
# ══════════════════════════════════════════════════════════════════════════════

position_str = str(player.get("position", "")).upper()
team_str = str(player.get("teamname", "N/A"))
league_str = str(player.get("league", "N/A"))
split_str = str(player.get("split", "N/A"))
year_str = str(int(player.get("_source_year", 0))) if player.get("_source_year") else "N/A"

st.subheader(selected_player)
st.caption(f"{position_str} · {team_str} · {league_str} · Split {split_str} {year_str}")

# ══════════════════════════════════════════════════════════════════════════════
# Métriques clés
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Talent Score",
        value=f"{player.get('talent_score', 0):.1f}/100",
        help="Probabilité calibrée de promotion en LEC (×100).",
    )
    percentile = player.get("score_percentile")
    if percentile is not None and not pd.isna(percentile):
        # percentile=98 → meilleur que 98% des joueurs → "Top 2%" (plancher à 1%)
        top_pct = max(100 - float(percentile), 1)
        st.caption(f"Top {top_pct:.0f}% des {position_str} ERL")

with col2:
    promoted = bool(player.get("promoted_to_lec", False))
    st.metric(
        label="Promotion LEC",
        value="✅ Promu" if promoted else "❌ Non promu",
    )

with col3:
    win_rate = player.get("win_rate", 0)
    st.metric(
        label="Win Rate",
        value=f"{win_rate:.1%}",
    )

with col4:
    st.metric(
        label="Games joués",
        value=int(player.get("games_played", 0)),
    )

with col5:
    st.metric(
        label="Champion pool",
        value=int(player.get("champion_pool_size", 0)),
        help="Nombre de champions différents joués sur la saison.",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Archétype via clustering
# ══════════════════════════════════════════════════════════════════════════════

archetype_label = "Non disponible (clustering_results.csv manquant)"
cluster_id = None

if clustering_available and not df_clusters.empty and "playername" in df_clusters.columns:
    cluster_rows = df_clusters[df_clusters["playername"] == selected_player]
    if not cluster_rows.empty:
        cluster_row = cluster_rows.iloc[0]
        cluster_id = cluster_row.get("cluster", None)
        position = str(player.get("position", "")).lower()

        if cluster_id is not None and not pd.isna(cluster_id):
            # Priorité à l'archétype stocké directement dans clustering_results
            if "archetype" in cluster_row.index and cluster_row["archetype"]:
                archetype_label = str(cluster_row["archetype"])
            else:
                archetype_label = get_archetype(cluster_profiles, position, int(cluster_id))

col_arch, col_cluster = st.columns([3, 1])
with col_arch:
    st.markdown(f"**Archétype ML :** {archetype_label}")
with col_cluster:
    if cluster_id is not None and not pd.isna(cluster_id):
        st.markdown(f"**Cluster :** #{int(cluster_id)} — {position_str}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Radar chart des z-scores
# ══════════════════════════════════════════════════════════════════════════════

# Candidats z-score dans l'ordre de pertinence visuelle.
# On affiche seulement ceux présents dans le CSV pour éviter les KeyError.
zscore_candidates = [
    ("dpm_zscore", "DPM"),
    ("cspm_zscore", "CSPM"),
    ("golddiffat15_zscore", "Gold@15"),
    ("vspm_zscore", "Vision"),
    ("killparticipation_zscore", "Kill Part."),
    ("xpdiffat15_zscore", "XP@15"),
    ("csdiffat15_zscore", "CS@15"),
    ("win_rate_zscore", "Win Rate"),
    ("champion_pool_size_zscore", "Champ Pool"),
]

available = [
    (col, label)
    for col, label in zscore_candidates
    if col in player.index and pd.notna(player[col])
]

if len(available) < 3:
    st.warning(
        "Pas assez de métriques z-score disponibles pour afficher le radar chart "
        f"({len(available)}/3 minimum). Vérifiez les colonnes du CSV."
    )
else:
    cols_z, labels_z = zip(*available)
    values = [float(player.get(c, 0)) for c in cols_z]

    # Fermer la boucle du radar (premier point = dernier point)
    labels_closed = list(labels_z) + [labels_z[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # Trace principale — joueur sélectionné
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(11, 252, 228, 0.15)",
            line=dict(color="#0BFCE4", width=2.5),
            name=selected_player,
        )
    )

    # Ligne de référence à z=0 (= moyenne de la ligue par position et split)
    zeros_closed = [0.0] * len(labels_closed)
    fig.add_trace(
        go.Scatterpolar(
            r=zeros_closed,
            theta=labels_closed,
            line=dict(color="#888899", width=1, dash="dash"),
            name="Moyenne ligue (z=0)",
            showlegend=True,
        )
    )

    # Range dynamique : au moins ±1.5 pour que le graphique soit lisible
    max_abs = max((abs(v) for v in values), default=1.0)
    axis_range = max(max_abs * 1.25, 1.5)

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(
                visible=True,
                range=[-axis_range, axis_range],
                gridcolor="#333355",
                tickfont=dict(color="#aaaacc", size=10),
                tickmode="linear",
                dtick=0.5,
            ),
            angularaxis=dict(
                gridcolor="#333355",
                tickfont=dict(color="#ffffff", size=12),
                direction="clockwise",
            ),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        title=dict(
            text=f"Profil de performance — {selected_player}",
            font=dict(color="#0BFCE4", size=16),
        ),
        legend=dict(
            bgcolor="#1a1a2e",
            bordercolor="#333355",
            borderwidth=1,
        ),
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Les **z-scores** mesurent l'écart à la moyenne par position et par split. "
        "z = +1 signifie 1 écart-type au-dessus de la moyenne, z = −1 en dessous. "
        "La ligne pointillée représente un joueur dans la moyenne (z = 0)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Toutes les saisons du joueur
# ══════════════════════════════════════════════════════════════════════════════

if len(player_rows) > 1:
    st.divider()
    st.subheader("Toutes les saisons")

    all_seasons_cols = [c for c in [
        "_source_year", "split", "league", "teamname",
        "talent_score", "win_rate", "games_played", "champion_pool_size",
    ] if c in player_rows.columns]

    seasons_display = player_rows[all_seasons_cols].sort_values(
        ["_source_year", "split"], ascending=[False, True]
    ).copy()

    if "talent_score" in seasons_display.columns:
        seasons_display["talent_score"] = seasons_display["talent_score"].round(1)
    if "win_rate" in seasons_display.columns:
        seasons_display["win_rate"] = seasons_display["win_rate"].round(3)

    st.dataframe(
        seasons_display,
        use_container_width=True,
        column_config={
            "_source_year": st.column_config.NumberColumn("Année", format="%d"),
            "talent_score": st.column_config.ProgressColumn(
                "Talent Score", min_value=0.0, max_value=100.0, format="%.1f"
            ),
        },
        hide_index=True,
    )
