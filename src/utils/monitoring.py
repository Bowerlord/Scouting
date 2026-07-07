"""
monitoring.py — Historisation et comparaison des métriques entre réentraînements

Avant ce module, talent_score_results.json était écrasé à chaque run : aucune
trace des runs précédents, aucune alerte en cas de dégradation après un refresh
hebdomadaire. Ici :

- chaque réentraînement APPEND une ligne dans reports/metrics/metrics_history.jsonl
  (une ligne JSON par run — diffs git minimaux, historique complet),
- le run courant est comparé au précédent (delta PR-AUC / Brier),
- un rapport markdown est écrit dans reports/metrics/monitoring_report.md —
  utilisé comme corps de la PR du workflow Data Refresh, pour que le reviewer
  voie le delta sans ouvrir les fichiers.

Dégradation : PR-AUC courant < (1 − DEGRADATION_RELATIVE_DROP) × PR-AUC précédent.
"""

import json
from datetime import date
from pathlib import Path

from src.config import METRICS_DIR
from src.utils.logger import logger

HISTORY_FILENAME = "metrics_history.jsonl"
REPORT_FILENAME = "monitoring_report.md"

# Chute relative de PR-AUC au-delà de laquelle le run est signalé comme dégradé
DEGRADATION_RELATIVE_DROP = 0.10


def _default_history_path() -> Path:
    return METRICS_DIR / HISTORY_FILENAME


def append_metrics_history(entry: dict, history_path: Path | None = None) -> dict:
    """
    Ajoute une entrée (une ligne JSON) à l'historique des réentraînements.

    Args:
        entry: Métriques du run. Clés attendues : best_model, pr_auc, roc_auc,
            brier_calibrated, train_size, test_size, n_train_positives,
            n_players_scored. `run_date` est ajouté si absent.
        history_path: Fichier jsonl. Par défaut :
            reports/metrics/metrics_history.jsonl

    Returns:
        dict: L'entrée écrite (avec run_date).
    """
    if history_path is None:
        history_path = _default_history_path()

    entry = dict(entry)
    entry.setdefault("run_date", date.today().isoformat())

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

    logger.info(f"📈 Historique des métriques mis à jour → {history_path}")
    return entry


def read_metrics_history(history_path: Path | None = None) -> list[dict]:
    """Lit tout l'historique (liste vide si le fichier n'existe pas)."""
    if history_path is None:
        history_path = _default_history_path()
    if not history_path.exists():
        return []
    with open(history_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_with_previous(history_path: Path | None = None) -> dict:
    """
    Compare le dernier run de l'historique avec le précédent.

    Returns:
        dict: {current, previous, delta_pr_auc, delta_brier, degraded}.
            `previous` est None (et degraded False) au premier run.
    """
    history = read_metrics_history(history_path)
    if not history:
        raise ValueError(
            "Historique vide — appeler append_metrics_history avant compare_with_previous"
        )

    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None

    if previous is None:
        return {
            "current": current,
            "previous": None,
            "delta_pr_auc": None,
            "delta_brier": None,
            "degraded": False,
        }

    delta_pr_auc = round(current["pr_auc"] - previous["pr_auc"], 4)
    delta_brier = (
        round(current["brier_calibrated"] - previous["brier_calibrated"], 4)
        if current.get("brier_calibrated") is not None
        and previous.get("brier_calibrated") is not None
        else None
    )
    degraded = current["pr_auc"] < previous["pr_auc"] * (1 - DEGRADATION_RELATIVE_DROP)

    return {
        "current": current,
        "previous": previous,
        "delta_pr_auc": delta_pr_auc,
        "delta_brier": delta_brier,
        "degraded": degraded,
    }


def _fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def render_monitoring_report(comparison: dict, report_path: Path | None = None) -> str:
    """
    Écrit le rapport markdown du run courant (corps de la PR de refresh).

    Args:
        comparison: Sortie de compare_with_previous().
        report_path: Par défaut : reports/metrics/monitoring_report.md

    Returns:
        str: Le contenu markdown écrit.
    """
    if report_path is None:
        report_path = METRICS_DIR / REPORT_FILENAME

    current = comparison["current"]
    previous = comparison["previous"]

    lines = [
        "## 📈 Monitoring du réentraînement",
        "",
        f"Run du **{current.get('run_date', '—')}** — meilleur modèle : "
        f"**{current.get('best_model', '—')}**",
        "",
    ]

    if comparison["degraded"]:
        drop_pct = DEGRADATION_RELATIVE_DROP * 100
        lines += [
            f"> ⚠️ **Dégradation détectée** : le PR-AUC a chuté de plus de {drop_pct:.0f} % "
            f"par rapport au run précédent ({_fmt(previous['pr_auc'])} → "
            f"{_fmt(current['pr_auc'])}). Vérifier les données avant de merger.",
            "",
        ]

    if previous is not None:
        delta_pr = f"{comparison['delta_pr_auc']:+.4f}"
        delta_brier = (
            f"{comparison['delta_brier']:+.4f}"
            if comparison["delta_brier"] is not None
            else "—"
        )
        lines += [
            "| Métrique | Run précédent | Run courant | Δ |",
            "|---|---|---|---|",
            f"| PR-AUC | {_fmt(previous.get('pr_auc'))} "
            f"| {_fmt(current.get('pr_auc'))} | {delta_pr} |",
            f"| Brier (calibré) | {_fmt(previous.get('brier_calibrated'))} "
            f"| {_fmt(current.get('brier_calibrated'))} | {delta_brier} |",
            f"| Lignes train (positifs) | {previous.get('train_size', '—')} "
            f"({previous.get('n_train_positives', '—')}) | {current.get('train_size', '—')} "
            f"({current.get('n_train_positives', '—')}) | |",
            f"| Joueurs scorés | {previous.get('n_players_scored', '—')} "
            f"| {current.get('n_players_scored', '—')} | |",
        ]
    else:
        lines += [
            "Premier run historisé — pas de comparaison disponible.",
            "",
            f"- PR-AUC : {_fmt(current.get('pr_auc'))}",
            f"- Brier (calibré) : {_fmt(current.get('brier_calibrated'))}",
            f"- Lignes train : {current.get('train_size', '—')} "
            f"({current.get('n_train_positives', '—')} positifs)",
        ]

    lines += [
        "",
        "### Avant de merger",
        "- [ ] PR-AUC cohérent avec les runs précédents (`reports/metrics/metrics_history.jsonl`)",
        "- [ ] `data_max_date` plausible dans `reports/metrics/refresh_metadata.json`",
        "",
    ]

    content = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    logger.info(f"📝 Rapport de monitoring écrit → {report_path}")
    return content
