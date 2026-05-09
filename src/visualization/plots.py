"""
plots.py — Fonctions de visualisation réutilisables

Ce module centralise les fonctions de plot communes aux notebooks et scripts
de visualisation, évitant la duplication de code.

Usage :
    from src.visualization.plots import (
        set_dark_style, bar_horizontal, scatter_umap, heatmap_clusters
    )
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path


# ── Style global ──────────────────────────────────────────────────────────────

DARK_BG = "#0f172a"
PANEL_BG = "#1e293b"
TEXT_COLOR = "#f1f5f9"
MUTED_COLOR = "#64748b"
ACCENT_COLORS = ["#60a5fa", "#34d399", "#f97316", "#c084fc", "#fb7185", "#facc15"]

LEAGUE_COLORS = {
    "LFL":    "#3b82f6",
    "LFL2":   "#93c5fd",
    "PRM":    "#f59e0b",
    "LVP SL": "#ef4444",
    "NLC":    "#8b5cf6",
    "TCL":    "#10b981",
}


def set_dark_style() -> None:
    """Applique le style sombre uniforme à tous les graphiques du projet."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "text.color":     TEXT_COLOR,
        "axes.labelcolor": MUTED_COLOR,
        "xtick.color":    MUTED_COLOR,
        "ytick.color":    MUTED_COLOR,
        "axes.edgecolor": "#334155",
        "grid.color":     PANEL_BG,
        "axes.grid":      False,
    })


# ── Fonctions de plot ─────────────────────────────────────────────────────────

def bar_horizontal(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    color: str = "#60a5fa",
    highlight_indices: list[int] | None = None,
    highlight_color: str = "#facc15",
    value_format: str = "{:.1f}",
    max_val: float = 100.0,
) -> None:
    """
    Graphique horizontal de type leaderboard.

    Args:
        ax               : Axe matplotlib cible
        labels           : Labels des barres (de haut en bas)
        values           : Valeurs correspondantes
        color            : Couleur de base des barres
        highlight_indices: Indices des barres à mettre en évidence (ex: promus)
        highlight_color  : Couleur de surbrillance
        value_format     : Format d'affichage des valeurs
        max_val          : Valeur maximale de l'axe x
    """
    ax.set_facecolor(PANEL_BG)
    y_pos = range(len(labels) - 1, -1, -1)

    for i, (label, val) in enumerate(zip(labels, values)):
        y = len(labels) - 1 - i
        bar_color = highlight_color if (highlight_indices and i in highlight_indices) else color
        ax.barh(y, max_val, color=PANEL_BG, height=0.65, zorder=1)
        ax.barh(y, val, color=bar_color, height=0.65, alpha=0.85, zorder=2)
        ax.text(1, y, f"  {label}", va="center", fontsize=8.5, color=TEXT_COLOR, zorder=3)
        ax.text(val - 0.5, y, value_format.format(val),
                va="center", ha="right", fontsize=8, fontweight="bold",
                color="white", zorder=4)

    ax.set_xlim(0, max_val + 3)
    ax.set_yticks([])
    ax.set_xlabel("Score", fontsize=9, color=MUTED_COLOR)


def scatter_umap(
    ax: plt.Axes,
    df: pd.DataFrame,
    cluster_col: str = "cluster_position",
    promoted_col: str = "promoted_to_lec",
    colors: list[str] | None = None,
    title: str = "",
) -> None:
    """
    Scatter UMAP avec clusters colorés et promus mis en évidence (étoile).

    Args:
        ax           : Axe matplotlib cible
        df           : DataFrame avec colonnes umap_x, umap_y, cluster_col, promoted_col
        cluster_col  : Nom de la colonne de labels K-Means
        promoted_col : Nom de la colonne booléenne de promotion
        colors       : Palette de couleurs (une par cluster)
        title        : Titre du sous-graphique
    """
    if colors is None:
        colors = ACCENT_COLORS

    ax.set_facecolor(PANEL_BG)
    clusters = sorted(df[cluster_col].dropna().unique())

    legend_patches = []
    for i, c in enumerate(clusters):
        mask_c = df[cluster_col] == c
        color = colors[i % len(colors)]

        non_promoted = df[mask_c & ~df[promoted_col]]
        promoted = df[mask_c & df[promoted_col]]

        ax.scatter(non_promoted["umap_x"], non_promoted["umap_y"],
                   c=color, alpha=0.45, s=18, linewidths=0)
        ax.scatter(promoted["umap_x"], promoted["umap_y"],
                   c=color, alpha=1.0, s=70, marker="*",
                   edgecolors="white", linewidths=0.5, zorder=5)
        legend_patches.append(mpatches.Patch(color=color, label=f"Cluster {int(c)}"))

    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              facecolor=DARK_BG, edgecolor="#334155")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=8, color=MUTED_COLOR)
    ax.set_ylabel("UMAP 2", fontsize=8, color=MUTED_COLOR)
    ax.tick_params(labelsize=7)


def heatmap_clusters(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    vmax: float = 0.8,
) -> None:
    """
    Heatmap RdYlGn pour les profils de clusters (Z-scores moyens).

    Args:
        ax         : Axe matplotlib cible
        matrix     : Matrice (n_clusters × n_features) de Z-scores moyens
        row_labels : Labels des lignes (ex: "C0  20%↑")
        col_labels : Labels des colonnes (noms courts des features)
        title      : Titre du sous-graphique
        vmax       : Valeur max de l'échelle couleur (symétrique)
    """
    ax.set_facecolor(PANEL_BG)
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if abs(val) > 0.4 else MUTED_COLOR)


def pr_curve(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    """
    Trace une courbe Précision-Rappel sur un axe donné.

    Args:
        ax        : Axe matplotlib cible
        y_true    : Labels vrais (0/1)
        y_proba   : Probabilités prédites
        label     : Libellé dans la légende
        color     : Couleur de la courbe
        linestyle : Style de ligne
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_score = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, color=color, lw=2, linestyle=linestyle,
            label=f"{label}  (PR-AUC = {auc_score:.3f})")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Sauvegarde une figure avec les paramètres standards du projet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Sauvegarde : {path}")
