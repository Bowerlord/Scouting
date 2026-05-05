# 🏆 KCorp Scouting Tool — Plan d'Implémentation (v2)

## Objectif

Construire un outil ML de scouting de bout en bout pour identifier les "pépites" (jeunes talents prometteurs) dans les ligues mineures européennes de League of Legends. Le projet combine un **Talent Score** (classification supervisée) et un **Playstyle Clustering** (non-supervisé) dans un pipeline professionnel.

---

## Périmètre confirmé

| Décision | Choix |
|---|---|
| **Approche ML** | A (Talent Score) + B (Clustering) |
| **Jeu** | League of Legends |
| **Ligues ciblées** | LFL, LFL Div 2, Prime League, Prime League 2nd Div, LVP SuperLiga, LVP SL 2nd Div, NLC, NLC 2nd Div, TCL, TCL Academy + **LEC** (pour la target "promu") |
| **Données** | **2024-2026 uniquement** (la scène pro LoL évolue trop vite, les données avant 2024 ne sont plus représentatives) |
| **Repo** | Nouveau repo `kcorp-scouting` from scratch |
| **Dashboard** | Streamlit prévu pour plus tard (architecture prête) |
| **Deep Learning** | Light (feedforward comparatif) + Medium (autoencoder pour embeddings) |
| **Style README** | Rédigé comme un rapport d'étudiant en fin de cursus — raisonnement détaillé, chaque phase expliquée avec le "pourquoi", problèmes rencontrés documentés |

---

## Architecture de conception — Les 7 phases

Le README expliquera ces phases en détail, **comme un journal de bord** : à chaque fin de phase, on documente ce qu'on a fait, pourquoi, et les difficultés rencontrées.

### Phase 1 — Setup & Structure du Projet
> *Fondation : on ne construit rien de solide sur du sable*

- Initialiser le repo avec la structure Cookiecutter Data Science
- Configurer `pyproject.toml`, `requirements.txt`, `.gitignore`, `Makefile`
- Mettre en place le logging structuré (loguru)
- Créer le fichier de configuration centralisée (`config.py`)
- Documenter le README avec la vision du projet

### Phase 2 — Acquisition des Données (Data Ingestion)
> *Garbage in, garbage out — tout commence par des données propres*

- **Oracle's Elixir** : Télécharger les CSV **2024-2026** (pas avant — la scène évolue trop)
- **Leaguepedia** : Requêter l'API Cargo pour les historiques de transferts joueurs (qui a été promu en LEC/LFL Div 1 ?)
- Stocker les données brutes dans `data/raw/`
- Documenter chaque source dans un `DATA_SOURCES.md`

### Phase 3 — Nettoyage & Transformation (Data Cleaning)
> *80% du temps d'un Data Scientist*

- Filtrer les ligues cibles (LFL, PRM, LVP, NLC, TCL + divisions inférieures + LEC)
- Normaliser les noms de joueurs/équipes (problème fréquent entre sources)
- Gérer les valeurs manquantes (imputation ou suppression documentée)
- Construire une target variable binaire : `promoted_to_top_league` (le joueur a-t-il été promu LEC/top ERL dans les 2 ans ?)
- Sauvegarder le dataset nettoyé dans `data/interim/`

### Phase 4 — Feature Engineering
> *L'étape qui sépare un bon modèle d'un modèle moyen*

- **Stats brutes agrégées** : Moyennes et écarts-types par joueur par split
- **Stats relatives** : Z-scores par rapport à la moyenne de sa ligue (un joueur dominant dans une ligue faible vs. moyen dans une ligue forte)
- **Métriques de progression** : Évolution des stats entre splits consécutifs
- **Champion pool** : Diversité, champions signature, taux de victoire par champion
- **Métriques d'efficacité** : Damage per gold, kill participation, vision per minute
- **Consistency score** : Écart-type normalisé (un joueur constant vs. inconstant)
- Sauvegarder dans `data/processed/`

### Phase 5 — Modélisation (ML)
> *Le cœur du projet — mais pas la majorité du travail*

**5.A — Talent Score (Supervisé)**
- Split temporel (train: 2024, test: 2025) — **jamais de random split pour des données temporelles**
- Baseline : Logistic Regression
- Modèle principal : Random Forest + Gradient Boosting (XGBoost ou LightGBM)
- Comparatif Deep Learning : Feedforward PyTorch (même features)
- Métriques : AUC-ROC, Precision-Recall, F1 (classes déséquilibrées → peu de joueurs promus)
- Explicabilité : SHAP values pour chaque prédiction

**5.B — Playstyle Clustering (Non-supervisé)**
- Standardisation des features
- Réduction dimensionnelle : PCA puis UMAP/t-SNE pour visualisation
- K-Means avec méthode du coude + silhouette score
- DBSCAN comme alternative
- Autoencoder PyTorch : apprendre des embeddings latents, puis K-Means dessus
- Profiling des clusters : quelles caractéristiques définissent chaque style de jeu ?
- "Trouve le prochain X" : distance cosine entre un joueur amateur et un joueur star LEC

### Phase 6 — Évaluation & Résultats
> *Un modèle sans évaluation rigoureuse ne vaut rien*

- Rapport de classification complet
- Confusion matrix, courbes ROC/PR
- SHAP summary plots et force plots
- Visualisations clustering (scatter 2D/3D avec labels)
- Top 10 des "pépites" identifiées vs. réalité (validation sur 2025)

### Phase 7 — Documentation & Présentation
> *Un projet invisible n'existe pas*

- README détaillé avec les 7 phases expliquées (style étudiant, raisonnement clair)
- Notebooks numérotés et commentés
- `reports/figures/` avec les visualisations clés
- Tags de version Git

---

## Arborescence complète du repo

```
kcorp-scouting/
│
├── README.md                           # README détaillé (style rapport d'étudiant)
├── LICENSE                             # MIT
├── .gitignore                          # Python, data/, models/, .env
├── .env.example                        # RIOT_API_KEY= (optionnel)
├── pyproject.toml                      # Métadonnées projet + config tools
├── requirements.txt                    # Dépendances pinées
├── Makefile                            # make data | make features | make train | make cluster
├── DATA_SOURCES.md                     # Documentation des sources de données
│
├── data/
│   ├── raw/                            # CSV Oracle's Elixir bruts (.gitignore)
│   ├── interim/                        # Données nettoyées et filtrées
│   ├── processed/                      # Feature matrices prêtes pour le ML
│   └── external/                       # Données Leaguepedia (transferts, rosters)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA : distribution des stats, ligues, splits
│   ├── 02_feature_engineering.ipynb    # Construction et validation des features
│   ├── 03_talent_scoring.ipynb         # Modèle supervisé (Sklearn + PyTorch)
│   ├── 04_clustering_analysis.ipynb    # Clustering + visualisations
│   └── 05_results_synthesis.ipynb      # Synthèse des résultats, top pépites
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, hyperparams, noms de ligues
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py               # Téléchargement Oracle's Elixir
│   │   ├── leaguepedia.py              # Requêtes API Cargo (transferts joueurs)
│   │   ├── cleaner.py                  # Nettoyage et filtrage multi-ligues
│   │   └── feature_engineering.py      # Construction des features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── talent_scorer.py            # Pipeline Sklearn (RF, XGBoost)
│   │   ├── neural_scorer.py            # Feedforward PyTorch
│   │   ├── autoencoder.py              # Autoencoder PyTorch (embeddings)
│   │   ├── clusterer.py                # K-Means, DBSCAN
│   │   └── evaluate.py                 # Métriques, rapports, SHAP
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py                    # Fonctions de plot réutilisables
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py                   # Configuration loguru
│
├── models/                             # Modèles sauvegardés (.pkl, .pt) (.gitignore)
│   └── .gitkeep
│
├── reports/
│   ├── figures/                        # Visualisations pour le README
│   └── metrics/                        # Résultats d'évaluation (JSON)
│
└── tests/
    ├── __init__.py
    ├── test_cleaner.py
    ├── test_features.py
    └── test_models.py
```

---

## Style du README — "Rapport Étudiant"

Le README suivra cette structure narrative :

```
Pour chaque phase :
  1. 🎯 Objectif — Ce qu'on cherche à accomplir
  2. 🧠 Raisonnement — Pourquoi cette approche et pas une autre
  3. 🔧 Ce qu'on a fait — Les choix techniques concrets
  4. ⚠️ Problèmes rencontrés — Les difficultés et comment on les a résolues
  5. ✅ Résultat — Ce qu'on obtient à la fin de cette phase
```

Exemple pour la Phase 2 :
> **Phase 2 — Acquisition des Données**
> 
> 🎯 *Objectif* : Récupérer les données de match des ligues ERL européennes.
>
> 🧠 *Raisonnement* : Nous avons choisi Oracle's Elixir comme source principale car c'est la seule source gratuite qui regroupe les stats détaillées de toutes les ERLs dans un format CSV structuré. Nous limitons les données à 2024-2026 car la scène LoL évolue rapidement (changements de méta, reworks de champions) et des données trop anciennes introduiraient du bruit.
>
> ⚠️ *Problème rencontré* : Les fichiers CSV de Google Drive nécessitent une gestion spéciale pour les téléchargements volumineux (confirmation anti-virus). Nous avons implémenté un mécanisme de retry avec backoff exponentiel.
>
> ✅ *Résultat* : 3 fichiers CSV (~500 Mo total) couvrant 2024-2026 stockés dans `data/raw/`.

---

## Ligues ciblées — Mapping Oracle's Elixir

| Pays | Division 1 | Division 2 (estimé) |
|---|---|---|
| 🇫🇷 France | `LFL` | `LFL Division 2` / `LFL2` |
| 🇩🇪 Allemagne | `PRM` (Prime League) | `PRM Division 2` / `PRM 2nd Division` |
| 🇪🇸 Espagne | `LVP SL` (SuperLiga) | `LVP SL 2nd Division` |
| 🇬🇧 UK/Nordics | `NLC` | `NLC 2nd Division` |
| 🇹🇷 Turquie | `TCL` | `TCL Academy` |
| 🇪🇺 Europe (top) | `LEC` | — (target variable) |

> [!IMPORTANT]
> Les noms exacts dans les CSV Oracle's Elixir seront vérifiés programmatiquement en Phase 2 via `df['league'].unique()`. Certaines divisions 2 peuvent ne pas être couvertes — ce n'est pas bloquant.

---

## Ordre d'exécution proposé

| Étape | Description | Fichiers |
|---|---|---|
| **1** | Setup du repo, config, deps | `pyproject.toml`, `requirements.txt`, `Makefile`, `config.py`, `logger.py`, `.gitignore` |
| **2** | Data pipeline | `downloader.py`, `leaguepedia.py`, `cleaner.py` |
| **3** | EDA | `notebooks/01_data_exploration.ipynb` |
| **4** | Feature engineering | `feature_engineering.py`, `notebooks/02_feature_engineering.ipynb` |
| **5** | Talent Score ML | `talent_scorer.py`, `neural_scorer.py`, `evaluate.py`, `notebooks/03_talent_scoring.ipynb` |
| **6** | Clustering | `autoencoder.py`, `clusterer.py`, `notebooks/04_clustering_analysis.ipynb` |
| **7** | Synthèse | `notebooks/05_results_synthesis.ipynb`, `plots.py` |
| **8** | Tests | `test_cleaner.py`, `test_features.py`, `test_models.py` |
| **9** | README final | `README.md`, `DATA_SOURCES.md` |

---

## Changements par rapport au plan v1

| Modification | Avant (v1) | Après (v2) |
|---|---|---|
| **Années de données** | 2022-2026 | **2024-2026** (la scène bouge trop, données anciennes = bruit) |
| **Split temporel** | train: 2022-2024, test: 2025 | **train: 2024, test: 2025** (+ 2026 si données suffisantes) |
| **Style README** | Standard technique | **Rapport d'étudiant** — raisonnement étape par étape, problèmes rencontrés |

---

## Verification Plan

### Automated Tests
- `pytest tests/` après chaque module
- Vérification du téléchargement des données : `make data` doit produire des CSV dans `data/raw/`
- Vérification du pipeline complet : `make all` doit tourner sans erreur
- Vérification des shapes des DataFrames à chaque étape

### Manual Verification
- Inspecter visuellement `df['league'].unique()` pour confirmer les noms de ligues
- Vérifier que la target variable `promoted_to_top_league` est cohérente avec la réalité connue
- Valider les top pépites prédites vs. les transferts réels de 2025
- Review visuel des clusters (font-ils sens du point de vue esport ?)
