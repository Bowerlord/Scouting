"""
3_Scout_Mode.py — Mode scouting avancé avec shortlist et recherche par similarité ML

Cette page propose deux outils de scouting complémentaires :

1. Shortlist par critères
   - Filtres : position, archétype (issu du clustering), ligue, score minimum
   - Résultat : tableau trié par talent_score des joueurs correspondants

2. Recherche par similarité ML
   - Saisir un joueur de référence
   - Retourne tous les joueurs du même cluster K-Means et de la même position
   - Scatter plot comparatif des z-scores dans le cluster
   - Permet de trouver des alternatives ou des talents moins connus

Pourquoi la similarité par cluster K-Means ?
  → K-Means par position regroupe les joueurs ayant des profils de z-scores similaires.
    "Même cluster" signifie "même style de jeu" selon le modèle ML.
    C'est une approche plus robuste que la simple distance sur les z-scores bruts
    car le clustering a déjà identifié les groupes naturels dans les données.

Intégration :
  pages/3_Scout_Mode.py → utils/data_loader.py → reports/metrics/
    → talent_scores_players.csv  (scores + métriques)
    → clustering_results.csv     (assignation cluster par joueur)
    → cluster_profiles.json      (archetypes par position/cluster)
"""

import sys
from pathlib import Path

# Remonter de pages/ → app/ pour que 'utils' soit importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import (
    get_archetype,
    list_archetypes,
    load_cluster_profiles,
    load_clustering_results,
    load_talent_scores,
)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration de la page
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Scout Mode — KCorp Scouting",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Scout Mode")
st.markdown(
    "Générez une **shortlist** ciblée ou trouvez des joueurs similaires "
    "grâce au clustering ML."
)

# ══════════════════════════════════════════════════════════════════════════════
# Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

try:
    df_talent = load_talent_scores()
    cluster_profiles = load_cluster_profiles()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Clustering optionnel — la shortlist fonctionne sans, seule la similarité est désactivée
try:
    df_clusters = load_clustering_results()
    clustering_available = True
except FileNotFoundError:
    df_clusters = pd.DataFrame()
    clustering_available = False

# ══════════════════════════════════════════════════════════════════════════════
# Merge talent scores + clustering
# ══════════════════════════════════════════════════════════════════════════════

# On joint sur playername + position + année + split (quand disponibles) pour
# associer chaque ligne joueur/saison à SON cluster, plutôt que sur playername
# seul : un joueur a plusieurs lignes (une par split/année), et un merge sur
# playername seul créerait un produit cartésien entre ses lignes de score et
# ses lignes de clustering (doublons + mauvaises associations cluster/saison).
if clustering_available and not df_clusters.empty and "playername" in df_clusters.columns:
    merge_keys = [k for k in ["playername", "position", "_source_year", "split"]
                  if k in df_talent.columns and k in df_clusters.columns]
    cluster_cols = merge_keys + [c for c in ["cluster", "archetype"]
                                  if c in df_clusters.columns]
    df = df_talent.merge(
        df_clusters[cluster_cols],
        on=merge_keys,
        how="left",
    )
else:
    df = df_talent.copy()
    df["cluster"] = pd.NA
    df["archetype"] = pd.NA

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Shortlist par critères
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Shortlist par critères")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    positions_avail = sorted(df["position"].dropna().unique().tolist())
    sel_positions = st.multiselect(
        "Position",
        options=positions_avail,
        default=positions_avail,
        key="sl_pos",
    )

with col_f2:
    archetypes_all = list_archetypes(cluster_profiles)
    sel_archetype = st.selectbox(
        "Archétype",
        options=["Tous"] + archetypes_all,
        index=0,
        key="sl_arch",
        help="Archetypes définis par le clustering K-Means.",
    )

with col_f3:
    leagues_avail = sorted(df["league"].dropna().unique().tolist())
    sel_leagues = st.multiselect(
        "Ligue",
        options=leagues_avail,
        default=leagues_avail,
        key="sl_league",
    )

with col_f4:
    sel_min_score = st.slider(
        "Score minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        format="%.2f",
        key="sl_score",
    )

# --- Application des filtres ---

shortlist = df.copy()

if sel_positions:
    shortlist = shortlist[shortlist["position"].isin(sel_positions)]

if sel_leagues:
    shortlist = shortlist[shortlist["league"].isin(sel_leagues)]

shortlist = shortlist[shortlist["talent_score"] >= sel_min_score]

if sel_archetype != "Tous":
    if "archetype" in shortlist.columns and shortlist["archetype"].notna().any():
        shortlist = shortlist[shortlist["archetype"] == sel_archetype]
    elif "cluster" in shortlist.columns:
        # Reconstituer l'archétype depuis cluster_profiles quand il n'est pas
        # directement dans clustering_results.csv
        def _matches_archetype(row: pd.Series) -> bool:
            cl = row.get("cluster")
            if pd.isna(cl):
                return False
            pos = str(row.get("position", "")).lower()
            return get_archetype(cluster_profiles, pos, int(cl)) == sel_archetype

        shortlist = shortlist[shortlist.apply(_matches_archetype, axis=1)]

shortlist = shortlist.sort_values("talent_score", ascending=False).reset_index(drop=True)
shortlist.index += 1

st.caption(f"{len(shortlist)} joueur(s) dans la shortlist")

# --- Tableau de la shortlist ---

sl_cols = [c for c in [
    "playername", "position", "league", "teamname",
    "_source_year", "split", "talent_score",
    "win_rate", "games_played", "champion_pool_size",
    "promoted_to_lec", "cluster",
] if c in shortlist.columns]

sl_display = shortlist[sl_cols].copy()

if "promoted_to_lec" in sl_display.columns:
    sl_display["promoted_to_lec"] = sl_display["promoted_to_lec"].apply(
        lambda x: "✅" if x else "❌"
    )
if "talent_score" in sl_display.columns:
    sl_display["talent_score"] = sl_display["talent_score"].round(4)
if "win_rate" in sl_display.columns:
    sl_display["win_rate"] = sl_display["win_rate"].round(3)

st.dataframe(
    sl_display,
    use_container_width=True,
    column_config={
        "playername": st.column_config.TextColumn("Joueur"),
        "position": st.column_config.TextColumn("Poste"),
        "league": st.column_config.TextColumn("Ligue"),
        "teamname": st.column_config.TextColumn("Équipe"),
        "_source_year": st.column_config.NumberColumn("Année", format="%d"),
        "split": st.column_config.TextColumn("Split"),
        "talent_score": st.column_config.ProgressColumn(
            "Talent Score", min_value=0.0, max_value=1.0, format="%.4f"
        ),
        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.3f"),
        "games_played": st.column_config.NumberColumn("Games"),
        "champion_pool_size": st.column_config.NumberColumn("Champ Pool"),
        "promoted_to_lec": st.column_config.TextColumn("Promu LEC"),
        "cluster": st.column_config.NumberColumn("Cluster", format="%d"),
    },
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Recherche par similarité (même cluster K-Means)
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Joueurs similaires (clustering ML)")

if not clustering_available or df["cluster"].isna().all():
    st.warning(
        "Les données de clustering ne sont pas disponibles. "
        "Assurez-vous que `clustering_results.csv` a été généré par le pipeline ML."
    )
else:
    player_names = sorted(df["playername"].dropna().unique().tolist())

    ref_player = st.selectbox(
        "Joueur de référence",
        options=player_names,
        key="sim_player",
        help="Trouvez des joueurs ayant le même profil de jeu (même cluster K-Means).",
    )

    if ref_player:
        ref_rows = df[df["playername"] == ref_player]

        if ref_rows.empty:
            st.error(f"Joueur '{ref_player}' introuvable dans le dataset.")
        else:
            # Ligne du joueur avec le meilleur talent_score (peak saison)
            ref = ref_rows.sort_values("talent_score", ascending=False).iloc[0]
            ref_cluster = ref.get("cluster")
            ref_position = str(ref.get("position", ""))

            st.info(
                f"**{ref_player}** · {ref_position.upper()} · "
                f"Cluster #{ref_cluster if pd.notna(ref_cluster) else 'N/A'} · "
                f"Talent Score : {ref.get('talent_score', 0):.4f}"
            )

            if pd.isna(ref_cluster):
                st.warning(
                    "Ce joueur n'a pas de cluster assigné dans `clustering_results.csv`."
                )
            else:
                # Joueurs du même cluster et même position, excluant le joueur de référence
                similar = df[
                    (df["cluster"] == ref_cluster)
                    & (df["position"] == ref_position)
                    & (df["playername"] != ref_player)
                ].sort_values("talent_score", ascending=False).reset_index(drop=True)
                similar.index += 1

                st.caption(
                    f"{len(similar)} joueur(s) similaires "
                    f"(cluster #{int(ref_cluster)}, position {ref_position.upper()})"
                )

                sim_cols = [c for c in [
                    "playername", "league", "teamname", "_source_year", "split",
                    "talent_score", "win_rate", "games_played", "promoted_to_lec",
                ] if c in similar.columns]

                sim_display = similar[sim_cols].copy()

                if "promoted_to_lec" in sim_display.columns:
                    sim_display["promoted_to_lec"] = sim_display["promoted_to_lec"].apply(
                        lambda x: "✅" if x else "❌"
                    )
                if "talent_score" in sim_display.columns:
                    sim_display["talent_score"] = sim_display["talent_score"].round(4)
                if "win_rate" in sim_display.columns:
                    sim_display["win_rate"] = sim_display["win_rate"].round(3)

                st.dataframe(
                    sim_display,
                    use_container_width=True,
                    column_config={
                        "playername": st.column_config.TextColumn("Joueur"),
                        "_source_year": st.column_config.NumberColumn("Année", format="%d"),
                        "talent_score": st.column_config.ProgressColumn(
                            "Talent Score", min_value=0.0, max_value=1.0, format="%.4f"
                        ),
                        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.3f"),
                        "promoted_to_lec": st.column_config.TextColumn("Promu LEC"),
                    },
                )

                # ── Scatter plot comparatif des z-scores dans le cluster ──────────────

                # On utilise les deux premiers z-scores disponibles pour les axes X/Y
                zscore_cols = [c for c in [
                    "dpm_zscore", "cspm_zscore", "golddiffat15_zscore",
                    "vspm_zscore", "killparticipation_zscore",
                ] if c in df.columns]

                if len(zscore_cols) >= 2 and len(similar) > 0:
                    st.subheader(f"Comparaison des z-scores — cluster #{int(ref_cluster)}")

                    # Inclure le joueur de référence pour le positionner dans le scatter
                    ref_row_df = ref_rows.sort_values("talent_score", ascending=False).head(1).copy()
                    combined = pd.concat([ref_row_df, similar], ignore_index=True)

                    # Marquer le joueur de référence pour le distinguer visuellement
                    combined["_marker"] = combined["playername"].apply(
                        lambda n: "★ " + n if n == ref_player else n
                    )

                    axis_x = zscore_cols[0]
                    axis_y = zscore_cols[1]
                    label_x = axis_x.replace("_zscore", " (z-score)")
                    label_y = axis_y.replace("_zscore", " (z-score)")

                    fig = px.scatter(
                        combined,
                        x=axis_x,
                        y=axis_y,
                        color="league",
                        size="talent_score",
                        size_max=25,
                        hover_name="_marker",
                        hover_data={"talent_score": ":.4f", "_marker": False},
                        title=(
                            f"Joueurs similaires à {ref_player} "
                            f"— cluster #{int(ref_cluster)} / {ref_position.upper()}"
                        ),
                        labels={axis_x: label_x, axis_y: label_y, "league": "Ligue"},
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )

                    fig.update_layout(
                        plot_bgcolor="#1a1a2e",
                        paper_bgcolor="#1a1a2e",
                        font=dict(color="#ffffff"),
                        title_font=dict(color="#0BFCE4", size=15),
                        legend=dict(bgcolor="#1a1a2e", bordercolor="#333355", borderwidth=1),
                        height=480,
                    )
                    fig.update_xaxes(
                        gridcolor="#333355",
                        zeroline=True,
                        zerolinecolor="#555577",
                        zerolinewidth=1.5,
                        tickfont=dict(color="#aaaacc"),
                    )
                    fig.update_yaxes(
                        gridcolor="#333355",
                        zeroline=True,
                        zerolinecolor="#555577",
                        zerolinewidth=1.5,
                        tickfont=dict(color="#aaaacc"),
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Axe X : {label_x}  ·  Axe Y : {label_y}  · "
                        "Taille des bulles proportionnelle au talent score."
                    )
