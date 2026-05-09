"""
talent_score_viz.py — Visualisations Phase 5 : Talent Score

Génère 5 graphiques sauvegardés dans reports/figures/ :
  1. Courbes Précision-Rappel (3 modèles)
  2. Comparaison des métriques (bar chart)
  3. Feature Importances (Random Forest)
  4. Distribution des scores (promus vs non-promus)
  5. Leaderboard Top 30 — scatter plot
"""

import io
import json
import sys
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.config import (
    MODELS_DIR, METRICS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TRAIN_YEARS, TEST_YEARS
)
from src.models.talent_scorer import (
    FEATURE_COLS, TARGET_COL, load_features, make_out_of_time_split
)

warnings.filterwarnings("ignore")

# ── Style global ─────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = {
    "lr":     "#60a5fa",
    "rf":     "#34d399",
    "rf_tuned": "#86efac",  # vert clair
    "xgb":    "#f97316",
    "promoted":     "#facc15",
    "not_promoted":  "#64748b",
}
FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "text.color": "#f1f5f9",
})

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Chargement des données et modèles ────────────────────────────────────────
def load_all():
    df = load_features()
    X_train, y_train, X_test, y_test, df_train, df_test = make_out_of_time_split(df)
    available = [c for c in FEATURE_COLS if c in X_train.columns]
    X_test = X_test[available].fillna(0)

    models = {}
    for key, fname in [
        ("lr",       "talent_scorer_logistic_regression_baseline_.pkl"),
        ("rf",       "talent_scorer_random_forest.pkl"),
        ("rf_tuned", "talent_scorer_random_forest_tuned_.pkl"),
        ("xgb",      "talent_scorer_xgboost.pkl"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = joblib.load(p)

    scores_df = pd.read_csv(METRICS_DIR / "talent_scores_players.csv")
    with open(METRICS_DIR / "talent_score_results.json", encoding="utf-8") as f:
        metrics = json.load(f)

    return df, X_test, y_test, models, scores_df, metrics, available


# ── Figure 1 : Courbes Précision-Rappel ──────────────────────────────────────
def plot_pr_curves(X_test, y_test, models):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f172a")

    labels = {
        "lr":       "Logistic Regression",
        "rf":       "Random Forest",
        "rf_tuned": "Random Forest (tuned)",
        "xgb":      "XGBoost",
    }
    linestyles = {"lr": "-", "rf": "--", "rf_tuned": "-.", "xgb": ":"}

    for key, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        auc = average_precision_score(y_test, y_proba)
        ax.plot(
            rec, prec,
            color=PALETTE[key],
            lw=2.5,
            ls=linestyles[key],
            label=f"{labels[key]}  (PR-AUC = {auc:.3f})",
        )

    # Baseline aléatoire
    baseline = y_test.mean()
    ax.axhline(baseline, color="#475569", lw=1.5, ls=":", label=f"Baseline aléatoire ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Courbes Précision-Rappel — Comparaison des modèles\n"
                 f"(Out-of-Time : Train {TRAIN_YEARS} → Test {TEST_YEARS})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.2, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "01_pr_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Sauvegardé : {out}")


# ── Figure 2 : Comparaison métriques ─────────────────────────────────────────
def plot_metrics_comparison(metrics):
    comparison = pd.DataFrame(metrics["comparison"])
    model_labels = {
        "Logistic Regression (baseline)": "Logistic\nRegression",
        "Random Forest":                  "Random\nForest",
        "Random Forest (tuned)":          "RF\n(tuned)",
        "XGBoost":                        "XGBoost",
    }
    comparison["label"] = comparison["model_name"].map(model_labels)
    comparison = comparison.sort_values("pr_auc", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Comparaison des metriques - Phase 5", fontsize=15, fontweight="bold", y=1.02)

    colors = [PALETTE["lr"], PALETTE["rf"], PALETTE["rf_tuned"], PALETTE["xgb"]]

    for ax, col, title, color_order in [
        (axes[0], "pr_auc", "PR-AUC  (↑ meilleur)", colors),
        (axes[1], "roc_auc", "ROC-AUC  (↑ meilleur)", colors),
    ]:
        bars = ax.bar(comparison["label"], comparison[col], color=color_order, width=0.55, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(1.0, comparison[col].max() * 1.25))
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, comparison[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = FIGURES_DIR / "02_metrics_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Sauvegardé : {out}")


# ── Figure 3 : Feature Importances (RF) ──────────────────────────────────────
def plot_feature_importances():
    fi_path = METRICS_DIR / "feature_importance_random_forest.csv"
    if not fi_path.exists():
        print("⚠️  feature_importance_random_forest.csv introuvable, skip.")
        return

    fi = pd.read_csv(fi_path).head(12)

    # Labels lisibles
    rename = {
        "games_played": "Matchs joués",
        "win_rate_zscore": "Win Rate (z-score)",
        "win_rate": "Win Rate (brut)",
        "dpm_zscore": "DPM (z-score)",
        "champion_pool_size": "Champion Pool (brut)",
        "champion_pool_size_zscore": "Champion Pool (z-score)",
        "golddiffat15_zscore": "Gold Diff @15 (z-score)",
        "killparticipation_zscore": "Kill Participation (z-score)",
        "cspm_zscore": "CSPM (z-score)",
        "xpdiffat15_zscore": "XP Diff @15 (z-score)",
        "csdiffat15_zscore": "CS Diff @15 (z-score)",
        "vspm_zscore": "VSPM (z-score)",
    }
    fi["label"] = fi["feature"].map(rename).fillna(fi["feature"])
    fi = fi.sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")

    norm = plt.Normalize(fi["importance"].min(), fi["importance"].max())
    colors = plt.cm.YlGn(norm(fi["importance"].values))

    bars = ax.barh(fi["label"], fi["importance"], color=colors, height=0.65, zorder=3)
    ax.xaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, fi["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    ax.set_title("Feature Importances — Random Forest\n(gain moyen, normalisé à 100%)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Importance relative", fontsize=12)
    ax.set_xlim(0, fi["importance"].max() * 1.18)

    fig.tight_layout()
    out = FIGURES_DIR / "03_feature_importances.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Sauvegardé : {out}")


# ── Figure 4 : Distribution des Talent Scores ────────────────────────────────
def plot_score_distribution(scores_df):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f172a")

    promoted = scores_df[scores_df["promoted_to_lec"] == True]["talent_score"]
    not_promoted = scores_df[scores_df["promoted_to_lec"] == False]["talent_score"]

    ax.hist(not_promoted, bins=40, color=PALETTE["not_promoted"], alpha=0.75,
            label=f"Non-promus (n={len(not_promoted):,})", zorder=2)
    ax.hist(promoted, bins=20, color=PALETTE["promoted"], alpha=0.9,
            label=f"Promus en LEC (n={len(promoted):,})", zorder=3)

    ax.axvline(promoted.median(), color=PALETTE["promoted"], lw=2, ls="--",
               label=f"Médiane promus : {promoted.median():.0f}")
    ax.axvline(not_promoted.median(), color=PALETTE["not_promoted"], lw=2, ls="--",
               label=f"Médiane non-promus : {not_promoted.median():.0f}")

    ax.set_xlabel("Talent Score (0–100)", fontsize=13)
    ax.set_ylabel("Nombre de joueurs", fontsize=13)
    ax.set_title("Distribution des Talent Scores\nPromus vs Non-promus",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIGURES_DIR / "04_score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Sauvegardé : {out}")


# ── Figure 5 : Leaderboard Top 30 ────────────────────────────────────────────
def plot_leaderboard(scores_df):
    # Prendre le meilleur score par joueur (max sur tous ses splits)
    best = (scores_df
            .sort_values("talent_score", ascending=False)
            .drop_duplicates(subset=["playername"])
            .head(30)
            .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor("#0f172a")

    league_colors = {
        "LFL": "#3b82f6", "LFL2": "#93c5fd",
        "PRM": "#f59e0b", "LVP SL": "#ef4444",
        "NLC": "#8b5cf6", "TCL": "#10b981",
    }

    y_pos = range(len(best) - 1, -1, -1)

    for i, (_, row) in enumerate(best.iterrows()):
        y = len(best) - 1 - i
        color = league_colors.get(row["league"], "#94a3b8")
        promoted = row.get("promoted_to_lec", False)

        # Barre de fond grise
        ax.barh(y, 100, color="#1e293b", height=0.7, zorder=1)
        # Barre de score
        ax.barh(y, row["talent_score"], color=color, height=0.7, alpha=0.85, zorder=2)

        # Étoile si promu
        star = " ⭐" if promoted else ""
        label = f"  #{i+1}  {row['playername'].upper()}{star}  [{row['league']} · {row['position'].upper()}]"
        ax.text(1, y, label, va="center", fontsize=9.5, fontweight="bold" if promoted else "normal",
                color="#f8fafc", zorder=3)
        ax.text(row["talent_score"] - 1, y, f"{row['talent_score']:.1f}",
                va="center", ha="right", fontsize=9, fontweight="bold", color="white", zorder=4)

    # Légende ligues
    patches = [mpatches.Patch(color=c, label=l) for l, c in league_colors.items()]
    patches.append(mpatches.Patch(color="white", label="⭐ = Promu LEC confirmé"))
    ax.legend(handles=patches, fontsize=9, loc="lower right", framealpha=0.2, ncol=2)

    ax.set_xlim(0, 110)
    ax.set_yticks([])
    ax.set_xlabel("Talent Score", fontsize=12)
    ax.set_title("🏆 Leaderboard — Top 30 Talents ERL\n(meilleur score par joueur, tous splits confondus)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axvline(50, color="#475569", lw=1, ls=":")
    ax.grid(False)

    fig.tight_layout()
    out = FIGURES_DIR / "05_leaderboard_top30.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Sauvegardé : {out}")


# -- Figure 6 : Leaderboard par ligue ----------------------------------------
def plot_league_leaderboards(scores_df):
    """
    Pour chaque ligue, affiche le Top 5 des joueurs par Talent Score.
    Permet de voir les pépites dans chaque contexte, sans que la LFL écrase tout.
    """
    erl_leagues = ["LFL", "PRM", "LVP SL", "NLC", "TCL", "LFL2"]
    league_colors = {
        "LFL": "#3b82f6", "LFL2": "#93c5fd",
        "PRM": "#f59e0b", "LVP SL": "#ef4444",
        "NLC": "#8b5cf6", "TCL": "#10b981",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Top 5 Talents par Ligue\n(meilleur score par joueur)",
                 fontsize=15, fontweight="bold", y=1.01)

    axes_flat = axes.flatten()

    for ax, league in zip(axes_flat, erl_leagues):
        ax.set_facecolor("#1e293b")
        df_league = (
            scores_df[scores_df["league"] == league]
            .sort_values("talent_score", ascending=False)
            .drop_duplicates(subset=["playername"])
            .head(5)
            .reset_index(drop=True)
        )

        if df_league.empty:
            ax.set_title(league, fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        color = league_colors.get(league, "#94a3b8")
        y_pos = range(len(df_league) - 1, -1, -1)

        for i, (_, row) in enumerate(df_league.iterrows()):
            y = len(df_league) - 1 - i
            promoted = row.get("promoted_to_lec", False)
            ax.barh(y, 100, color="#1e293b", height=0.65, zorder=1)
            ax.barh(y, row["talent_score"], color=color, height=0.65, alpha=0.85, zorder=2)
            star = " *" if promoted else ""
            label = f"  #{i+1} {row['playername'].upper()}{star} [{row['position'].upper()}]"
            ax.text(1, y, label, va="center", fontsize=8.5,
                    fontweight="bold" if promoted else "normal",
                    color="#f8fafc", zorder=3)
            ax.text(row["talent_score"] - 1, y, f"{row['talent_score']:.1f}",
                    va="center", ha="right", fontsize=8, fontweight="bold",
                    color="white", zorder=4)

        ax.set_title(f"{league}  (Top 5)", fontsize=12, fontweight="bold",
                     color=color, pad=8)
        ax.set_xlim(0, 105)
        ax.set_yticks([])
        ax.set_xlabel("Talent Score", fontsize=9, color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax.text(98, -0.65, "* = Promu LEC", ha="right", fontsize=7.5,
                color="#94a3b8", style="italic")

    fig.tight_layout()
    out = FIGURES_DIR / "06_leaderboard_by_league.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Sauvegarde : {out}")


# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("📊 Génération des visualisations Phase 5...")
    df, X_test, y_test, models, scores_df, metrics, available = load_all()

    plot_pr_curves(X_test, y_test, models)
    plot_metrics_comparison(metrics)
    plot_feature_importances()
    plot_score_distribution(scores_df)
    plot_leaderboard(scores_df)
    plot_league_leaderboards(scores_df)

    print(f"\n6 graphiques generes dans : {FIGURES_DIR}")
