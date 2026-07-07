"""
test_monitoring.py — Tests de l'historisation des métriques (src/utils/monitoring.py)

Vérifie le cycle complet : append (jsonl append-only) → comparaison avec le
run précédent (delta, détection de dégradation) → rapport markdown utilisé
comme corps de la PR du workflow Data Refresh.
"""

import json

import pytest

from src.utils.monitoring import (
    DEGRADATION_RELATIVE_DROP,
    append_metrics_history,
    compare_with_previous,
    read_metrics_history,
    render_monitoring_report,
)


def make_entry(pr_auc: float, **overrides) -> dict:
    entry = {
        "best_model": "Logistic Regression (baseline)",
        "pr_auc": pr_auc,
        "roc_auc": 0.71,
        "brier_calibrated": 0.062,
        "train_size": 739,
        "test_size": 977,
        "n_train_positives": 49,
        "n_players_scored": 2005,
    }
    entry.update(overrides)
    return entry


class TestAppendMetricsHistory:
    def test_creates_then_appends(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=path)
        append_metrics_history(make_entry(0.14), history_path=path)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        # Chaque ligne est un JSON autonome (format jsonl)
        assert json.loads(lines[0])["pr_auc"] == 0.13
        assert json.loads(lines[1])["pr_auc"] == 0.14

    def test_run_date_added_if_absent(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        written = append_metrics_history(make_entry(0.13), history_path=path)
        assert "run_date" in written

    def test_explicit_run_date_preserved(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        written = append_metrics_history(
            make_entry(0.13, run_date="2026-01-01"), history_path=path
        )
        assert written["run_date"] == "2026-01-01"

    def test_read_returns_empty_when_missing(self, tmp_path):
        assert read_metrics_history(tmp_path / "absent.jsonl") == []


class TestCompareWithPrevious:
    def test_first_run_has_no_previous(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=path)
        comparison = compare_with_previous(history_path=path)
        assert comparison["previous"] is None
        assert comparison["delta_pr_auc"] is None
        assert comparison["degraded"] is False

    def test_empty_history_raises(self, tmp_path):
        with pytest.raises(ValueError):
            compare_with_previous(history_path=tmp_path / "absent.jsonl")

    def test_improvement_not_degraded(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=path)
        append_metrics_history(make_entry(0.15), history_path=path)
        comparison = compare_with_previous(history_path=path)
        assert comparison["delta_pr_auc"] == pytest.approx(0.02)
        assert comparison["degraded"] is False

    def test_small_drop_not_degraded(self, tmp_path):
        """Une baisse sous le seuil relatif (10 %) n'est pas une dégradation."""
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=path)
        append_metrics_history(make_entry(0.125), history_path=path)  # −3.8 %
        assert compare_with_previous(history_path=path)["degraded"] is False

    def test_large_drop_degraded(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=path)
        drop = 0.13 * (1 - DEGRADATION_RELATIVE_DROP - 0.05)  # −15 %
        append_metrics_history(make_entry(round(drop, 4)), history_path=path)
        assert compare_with_previous(history_path=path)["degraded"] is True

    def test_compares_last_two_runs_only(self, tmp_path):
        path = tmp_path / "metrics_history.jsonl"
        for pr_auc in (0.10, 0.13, 0.14):
            append_metrics_history(make_entry(pr_auc), history_path=path)
        comparison = compare_with_previous(history_path=path)
        assert comparison["previous"]["pr_auc"] == 0.13
        assert comparison["current"]["pr_auc"] == 0.14


class TestRenderMonitoringReport:
    def test_first_run_report(self, tmp_path):
        history = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=history)
        report_path = tmp_path / "monitoring_report.md"
        content = render_monitoring_report(
            compare_with_previous(history_path=history), report_path=report_path
        )
        assert report_path.exists()
        assert "Premier run historisé" in content
        assert "⚠️" not in content

    def test_report_shows_delta_table(self, tmp_path):
        history = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=history)
        append_metrics_history(make_entry(0.14), history_path=history)
        content = render_monitoring_report(
            compare_with_previous(history_path=history),
            report_path=tmp_path / "monitoring_report.md",
        )
        assert "+0.0100" in content
        assert "⚠️" not in content

    def test_degraded_report_contains_warning(self, tmp_path):
        history = tmp_path / "metrics_history.jsonl"
        append_metrics_history(make_entry(0.13), history_path=history)
        append_metrics_history(make_entry(0.08), history_path=history)  # −38 %
        content = render_monitoring_report(
            compare_with_previous(history_path=history),
            report_path=tmp_path / "monitoring_report.md",
        )
        assert "⚠️" in content
        assert "Dégradation détectée" in content
