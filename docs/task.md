# 🏆 KCorp Scouting Tool — Suivi des Tâches

## Phase 1 — Setup & Structure du Projet ✅
- [x] Tous les fichiers de setup créés et validés

## Phase 2 — Acquisition & Nettoyage des Données ✅
- [x] downloader.py, leaguepedia.py, cleaner.py
- [x] 3 CSV (2024-2026), 39,950 lignes, 916 joueurs, 65 promus

## Phase 3 — EDA ✅
- [x] `notebooks/01_data_exploration.ipynb`

## Phase 4 — Feature Engineering ✅
- [x] `src/data/feature_engineering.py`
- [x] `notebooks/02_feature_engineering.ipynb`

## Phase 5 — Talent Score ML ✅
- [x] `src/models/talent_scorer.py` — LR baseline, RF, XGBoost, tuning RandomizedSearchCV
- [x] `src/visualization/talent_score_viz.py` — 6 graphiques (PR curves, métriques, feature importances, distribution, leaderboards)
- [ ] `src/models/neural_scorer.py` — Feedforward PyTorch (prévu)
- [ ] `src/models/evaluate.py` — SHAP values (prévu)
- [x] `notebooks/03_talent_scoring.ipynb`
- **Résultat** : LR PR-AUC=0.256, RF PR-AUC=0.190 | Meilleur modèle : Logistic Regression

## Phase 6 — Clustering ✅
- [x] `src/models/clusterer.py` — K-Means par position (5 modèles), UMAP, profiling, similarity search
- [x] `src/visualization/clustering_viz.py` — 3 figures (UMAP scatter, heatmap profils, coude+silhouette)
- [ ] `src/models/autoencoder.py` — Autoencoder PyTorch (optionnel / phase suivante)
- [x] `notebooks/04_clustering_analysis.ipynb`
- **Résultat** : k=3 par position | Cluster "Early dominant" = 15-23% promus selon la position

## Phase 7 — Synthèse & Documentation ✅
- [x] `notebooks/05_results_synthesis.ipynb`
- [x] `src/visualization/plots.py`
- [x] Tests unitaires
- [x] README final complet
