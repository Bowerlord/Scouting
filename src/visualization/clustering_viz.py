"""
clustering_viz.py — Visualisations Phase 6 : Playstyle Clustering

Figures générées :
  07_umap_by_position.png  — Scatter UMAP (5 sous-graphiques, un par position)
                             coloré par cluster, avec les promus mis en avant
  08_cluster_profiles.png  — Radar/heatmap des features moyennes par cluster
  09_elbow_silhouette.png  — Courbes inertie + silhouette pour chaque position
"""

import io
import json
import sys
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#1e293b",
    "figure.facecolor": "#0f172a",
    "text.color": "#f1f5f9",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "axes.edgecolor": "#334155",
    "grid.color": "#1e293b",
})

POSITION_COLORS_CLUSTERS = {
    0: "#f97316",  # orange
    1: "#60a5fa",  # bleu
    2: "#34d399",  # vert
    3: "#c084fc",  # violet
    4: "#fb7185",  # rose
}
POSITION_ORDER = ["top", "jng", "mid", "bot", "sup"]
POSITION_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot (ADC)", "sup": "Support"}


def load_data():
    df = pd.read_csv(METRICS_DIR / "clustering_results.csv")
    if "promoted_to_lec" in df.columns:
        df["promoted_to_lec"] = df["promoted_to_lec"].astype(str).str.lower() == "true"
    with open(METRICS_DIR / "cluster_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    with open(METRICS_DIR / "kmeans_k_scores.json", encoding="utf-8") as f:
        k_scores = json.load(f)
    return df, profiles, k_scores


# ── Figure 7 : UMAP scatter par position ──────────────────────────────────────
def plot_umap_by_position(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Playstyle Clustering — Projection UMAP par Position\n"
                 "(chaque point = un joueur/split | etoile = promu LEC)",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    for ax_idx, pos in enumerate(POSITION_ORDER):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("#1e293b")

        df_pos = df[df["position"] == pos].copy()
        if df_pos.empty or df_pos["umap_x"].isna().all():
            ax.set_title(POSITION_LABELS[pos], fontsize=12)
            ax.axis("off")
            continue

        clusters = sorted(df_pos["cluster_position"].unique())

        # Scatter par cluster
        for c in clusters:
            mask_c = df_pos["cluster_position"] == c
            color = POSITION_COLORS_CLUSTERS.get(int(c), "#94a3b8")

            # Non-promus
            sub_np = df_pos[mask_c & ~df_pos["promoted_to_lec"]]
            ax.scatter(sub_np["umap_x"], sub_np["umap_y"],
                       c=color, alpha=0.45, s=18, linewidths=0)

            # Promus (mis en avant)
            sub_p = df_pos[mask_c & df_pos["promoted_to_lec"]]
            ax.scatter(sub_p["umap_x"], sub_p["umap_y"],
                       c=color, alpha=1.0, s=70, marker="*",
                       edgecolors="white", linewidths=0.5, zorder=5)

        # Légende clusters
        legend_patches = [
            mpatches.Patch(color=POSITION_COLORS_CLUSTERS.get(int(c), "#94a3b8"),
                           label=f"Cluster {c}")
            for c in clusters
        ]
        ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
                  facecolor="#0f172a", edgecolor="#334155")

        n_promus = df_pos["promoted_to_lec"].sum()
        ax.set_title(f"{POSITION_LABELS[pos]}  ({len(df_pos)} joueurs, {n_promus} promus LEC *)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Masquer le 6e sous-graphique (vide)
    axes_flat[-1].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "07_umap_by_position.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Sauvegarde : {out}")


# ── Figure 8 : Heatmap des profils de clusters ────────────────────────────────
def plot_cluster_profiles(profiles):
    feature_short = {
        "dpm_zscore":                "DPM",
        "cspm_zscore":               "CSPM",
        "vspm_zscore":               "VSPM",
        "killparticipation_zscore":  "KP",
        "golddiffat15_zscore":       "Gold@15",
        "xpdiffat15_zscore":         "XP@15",
        "csdiffat15_zscore":         "CS@15",
        "champion_pool_size_zscore": "Pool",
        "win_rate_zscore":           "WR",
    }
    feat_keys = list(feature_short.keys())

    fig, axes = plt.subplots(1, len(POSITION_ORDER), figsize=(20, 5))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Profils moyens des clusters par position (Z-scores)",
                 fontsize=13, fontweight="bold", y=1.03)

    vmax = 0.8

    for ax, pos in zip(axes, POSITION_ORDER):
        ax.set_facecolor("#1e293b")
        pos_profiles = profiles.get(pos, [])
        if not pos_profiles:
            ax.axis("off")
            continue

        data = []
        row_labels = []
        for p in sorted(pos_profiles, key=lambda x: x["cluster"]):
            row = [p.get(f, 0.0) for f in feat_keys]
            data.append(row)
            promo_pct = p.get("promo_rate", 0) * 100
            row_labels.append(f"C{p['cluster']}  {promo_pct:.0f}%↑")

        mat = np.array(data)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(feat_keys)))
        ax.set_xticklabels([feature_short[f] for f in feat_keys], rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        # Valeurs dans les cellules
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(val) > 0.4 else "#94a3b8")

        ax.set_title(POSITION_LABELS[pos], fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()
    out = FIGURES_DIR / "08_cluster_profiles_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Sauvegarde : {out}")


# ── Figure 9 : Coude + Silhouette par position ────────────────────────────────
def plot_elbow_silhouette(k_scores):
    fig, axes = plt.subplots(2, len(POSITION_ORDER), figsize=(20, 7))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Methode du coude et Silhouette score par position",
                 fontsize=13, fontweight="bold", y=1.02)

    for col, pos in enumerate(POSITION_ORDER):
        if pos not in k_scores:
            continue
        ks = k_scores[pos]
        k_range = ks["k_range"]
        inertias = ks["inertias"]
        silhouettes = ks["silhouettes"]
        best_k = ks["best_k"]
        best_idx = k_range.index(best_k)

        # Inertie (coude)
        ax_top = axes[0][col]
        ax_top.set_facecolor("#1e293b")
        ax_top.plot(k_range, inertias, "o-", color="#60a5fa", lw=2)
        ax_top.axvline(best_k, color="#facc15", ls="--", lw=1.2, alpha=0.8)
        ax_top.set_title(POSITION_LABELS[pos], fontsize=10, fontweight="bold")
        ax_top.set_ylabel("Inertie" if col == 0 else "", fontsize=8)
        ax_top.tick_params(labelsize=7)

        # Silhouette
        ax_bot = axes[1][col]
        ax_bot.set_facecolor("#1e293b")
        ax_bot.plot(k_range, silhouettes, "s-", color="#34d399", lw=2)
        ax_bot.scatter([best_k], [silhouettes[best_idx]],
                       color="#facc15", s=80, zorder=5)
        ax_bot.set_xlabel("k", fontsize=8)
        ax_bot.set_ylabel("Silhouette" if col == 0 else "", fontsize=8)
        ax_bot.tick_params(labelsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "09_elbow_silhouette.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Sauvegarde : {out}")


if __name__ == "__main__":
    print("Generation des visualisations Phase 6...")
    df, profiles, k_scores = load_data()
    plot_umap_by_position(df)
    plot_cluster_profiles(profiles)
    plot_elbow_silhouette(k_scores)
    print("\n3 graphiques generes dans reports/figures/")
