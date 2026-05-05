# ══════════════════════════════════════════════════════════════════════════════
# KCorp Scouting Tool — Makefile
# ══════════════════════════════════════════════════════════════════════════════
# Ce Makefile automatise les tâches courantes du pipeline ML.
# Usage : make <commande>
#
# Exemples :
#   make data       → Télécharge les données brutes
#   make clean      → Nettoie et filtre les données
#   make features   → Construit les features
#   make train      → Entraîne le modèle de Talent Score
#   make cluster    → Exécute le Playstyle Clustering
#   make all        → Pipeline complet (data → features → train → cluster)
#   make test       → Lance les tests unitaires

.PHONY: help data clean features train cluster all test lint format

# ── Commande par défaut ──────────────────────────────────────────────────────
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║        🏆 KCorp Scouting Tool — Commandes              ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make data      │ Télécharge les CSV Oracle's Elixir   ║"
	@echo "║  make clean     │ Nettoie et filtre les données        ║"
	@echo "║  make features  │ Construit les features ML            ║"
	@echo "║  make train     │ Entraîne le Talent Score             ║"
	@echo "║  make cluster   │ Exécute le Playstyle Clustering      ║"
	@echo "║  make all       │ Pipeline complet                     ║"
	@echo "║  make test      │ Lance les tests unitaires            ║"
	@echo "║  make lint      │ Vérifie le code (ruff)               ║"
	@echo "║  make format    │ Formate le code (black)              ║"
	@echo "╚══════════════════════════════════════════════════════════╝"

# ── Pipeline de données ──────────────────────────────────────────────────────
data:
	@echo "📥 Téléchargement des données Oracle's Elixir (2024-2026)..."
	python -m src.data.downloader

clean: data
	@echo "🧹 Nettoyage et filtrage des données..."
	python -m src.data.cleaner

features: clean
	@echo "⚙️  Construction des features..."
	python -m src.data.feature_engineering

# ── Modélisation ─────────────────────────────────────────────────────────────
train: features
	@echo "🎯 Entraînement du Talent Score..."
	python -m src.models.talent_scorer

cluster: features
	@echo "🔬 Exécution du Playstyle Clustering..."
	python -m src.models.clusterer

# ── Pipeline complet ─────────────────────────────────────────────────────────
all: data clean features train cluster
	@echo "✅ Pipeline complet terminé !"

# ── Qualité du code ──────────────────────────────────────────────────────────
test:
	@echo "🧪 Lancement des tests..."
	python -m pytest tests/ -v

lint:
	@echo "🔍 Vérification du code..."
	python -m ruff check src/ tests/

format:
	@echo "✨ Formatage du code..."
	python -m black src/ tests/
