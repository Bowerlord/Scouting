# 🏆 KCorp Scouting Tool

> **Un outil de Machine Learning pour identifier les pépites esport dans les ligues mineures européennes de League of Legends.**

Ce projet simule un outil de scouting data-driven pour la structure esport [Karmine Corp](https://www.karminecorp.fr/). L'objectif : analyser les performances des joueurs amateurs dans les ERLs (European Regional Leagues) et prédire lesquels ont le potentiel pour évoluer au plus haut niveau (LEC).

---

## 🎯 Contexte & Objectif

### Le problème métier

Dans l'esport professionnel, le recrutement de talents se fait encore largement "à l'œil" : les scouts regardent des matchs, échangent avec des coachs, et se fient à leur intuition. Pourtant, une quantité massive de données statistiques est disponible publiquement pour chaque match professionnel.

**Et si on pouvait systématiser cette analyse avec du Machine Learning ?**

Ce projet explore cette question en construisant deux outils complémentaires :

1. **🎯 Talent Score** — Un modèle supervisé qui attribue un score de potentiel (0-100) à chaque joueur d'ERL, basé sur ses statistiques de match. Le modèle apprend à partir des joueurs qui ont *réellement* été promus en LEC par le passé.

2. **🔬 Playstyle Clustering** — Un modèle non-supervisé qui regroupe les joueurs par style de jeu. Cela permet de répondre à des questions comme : *"Trouve-moi un joueur amateur dont le profil statistique ressemble à Caps"*.

### Pourquoi ce projet est intéressant

- **Données réelles** : Pas de datasets de Kaggle — on travaille avec des données brutes de matchs professionnels
- **Pipeline complet** : De l'acquisition des données au modèle final, en passant par le nettoyage et le feature engineering
- **Problème concret** : Le scouting dans l'esport est un vrai besoin industriel
- **Multi-approche ML** : Supervisé + Non-supervisé + Deep Learning comparatif

---

## 🏗️ Architecture du Projet — Les 7 Phases de Conception

Ce projet a été conçu en 7 phases itératives. Chaque phase est documentée ci-dessous avec :
- 🎯 **Objectif** — Ce qu'on cherche à accomplir
- 🧠 **Raisonnement** — Pourquoi cette approche et pas une autre
- 🔧 **Ce qu'on a fait** — Les choix techniques concrets
- ⚠️ **Problèmes rencontrés** — Les difficultés et comment on les a résolues
- ✅ **Résultat** — Ce qu'on obtient à la fin de cette phase

### Phase 1 — Setup & Structure du Projet ✅

🎯 **Objectif** : Poser les fondations d'un projet professionnel, reproductible et maintenable.

🧠 **Raisonnement** : Avant d'écrire la moindre ligne de code ML, il faut structurer le projet correctement. Un bon data scientist ne commence pas par entraîner un modèle — il commence par organiser son code. Nous avons suivi la convention [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/), qui est un standard reconnu dans l'industrie.

🔧 **Ce qu'on a fait** :
- Structure de dossiers avec séparation claire : `data/` (raw → interim → processed), `src/` (code modulaire), `notebooks/` (exploration), `tests/`
- Configuration centralisée dans `src/config.py` pour éviter les "magic numbers"
- Logging structuré avec [Loguru](https://github.com/Delgan/loguru) (plus lisible que le logging standard Python)
- `Makefile` pour automatiser le pipeline (`make data`, `make train`, etc.)
- `.gitignore` qui exclut les données volumineuses et les modèles entraînés

✅ **Résultat** : Un squelette de projet prêt à accueillir le code ML, avec toutes les bonnes pratiques d'ingénierie en place.

### Phase 2 — Acquisition des Données ✅

🎯 **Objectif** : Récupérer les données de match des ligues ERL européennes et les données de carrière des joueurs.

🧠 **Raisonnement** : Nous avons choisi [Oracle's Elixir](https://oracleselixir.com/tools/downloads) comme source principale car c'est la seule source gratuite qui regroupe les stats détaillées (~100+ colonnes) de **toutes les ERLs** dans un format CSV structuré. Nous limitons les données à **2024-2026** car la scène LoL évolue rapidement (changements de méta, reworks de champions, restructuration des ligues) — des données plus anciennes introduiraient du bruit plutôt que du signal. Note : le fichier 2026 est partiel (Spring Split en cours), ce qui est normal et suffisant pour enrichir notre dataset. En complément, nous avons préparé un module [Leaguepedia](https://lol.fandom.com) pour enrichir les historiques de carrière des joueurs.

🔧 **Ce qu'on a fait** :
- Module `src/data/downloader.py` : téléchargement depuis Google Drive avec cache local, retry (backoff exponentiel), gestion des fichiers volumineux (confirmation anti-virus Google Drive)
- Module `src/data/leaguepedia.py` : requêtes API Cargo vers Leaguepedia avec pagination automatique et rate limiting (1.5s entre requêtes). Conçu comme "best effort" — le pipeline fonctionne même si l'API est indisponible
- Module `src/data/cleaner.py` : pipeline de nettoyage en 8 étapes (filtrage ligues, filtrage joueurs, sélection colonnes, normalisation noms, imputation NaN, target variable, validation)

⚠️ **Problèmes rencontrés** :
1. **Divisions 2 manquantes** : Nous espérions trouver les divisions 2 (PRM 2nd Division, LVP SL 2nd Division, etc.) dans Oracle's Elixir. En vérifiant avec `df['league'].unique()`, seule la **LFL2** existe en tant que ligue séparée. Les autres divisions 2 ne sont pas couvertes. Ce n'est pas bloquant — les 5 ERLs Div 1 + LFL2 + LEC fournissent suffisamment de données.
2. **Colonne `killparticipation` absente** : Cette colonne attendue n'est pas dans les CSV récents d'Oracle's Elixir. On la recalculera manuellement en Phase 4 (Feature Engineering).
3. **Noms de joueurs** : Des incohérences entre les sources (majuscules, espaces) ont nécessité une normalisation systématique en lowercase.


✅ **Résultat** :
- 3 CSV téléchargés : 2024 (75.5 Mo) + 2025 (75.6 Mo) + 2026 (24.3 Mo, année partielle)
- Après nettoyage : **39,950 lignes** × 36 colonnes
- **916 joueurs uniques** dans 7 ligues : LEC, LFL, LFL2, LVP SL, NLC, PRM, TCL
- **65 joueurs promus** identifiés (7.5% des joueurs ERL → target variable)
- Dataset nettoyé sauvegardé dans `data/interim/cleaned_matches.csv` (8.5 Mo)

### Phase 3 — Nettoyage & Transformation ✅

> *Note : la Phase 3 a été fusionnée avec la Phase 2. Le nettoyage est intégré directement dans `cleaner.py` car les deux étapes sont indissociables — on ne peut pas nettoyer des données qu'on n'a pas encore chargées.*

Le nettoyage suit un pipeline de 8 étapes séquentielles, chacune documentée dans le code source :

| Étape | Opération | Résultat |
|---|---|---|
| 1 | Charger les CSV bruts (2024-2026) | 281,748 lignes × 166 colonnes |
| 2 | Découvrir les ligues | 50+ ligues identifiées dans les données |
| 3 | Filtrer les ligues ciblées | → 47,940 lignes (17.0% du total) |
| 4 | Exclure les lignes "team" | → 39,950 lignes joueur |
| 5 | Sélectionner les colonnes utiles | → 34 colonnes (de 166) |
| 6 | Normaliser les noms | 916 joueurs uniques |
| 7 | Imputer les valeurs manquantes | 4,340 NaN → 0 (médiane par ligue) |
| 8 | Construire la target variable | 65 joueurs promus en LEC |

### Exploration des Données (EDA) ✅

Une exploration approfondie du dataset a été menée dans le notebook `01_data_exploration.ipynb`. Voici les insights majeurs qui guideront les prochaines étapes :

1. **Déséquilibre de la Target** : Seulement ~7.5% des joueurs ERL (65 sur 872) sont promus en LEC. Le dataset est donc fortement déséquilibré (~12:1). Il faudra adapter les métriques (utiliser PR-AUC plutôt que Accuracy) et possiblement gérer ce déséquilibre (class_weights, SMOTE).
2. **Hétérogénéité Inter-Ligues** : Une comparaison des distributions montre que les statistiques brutes varient selon le niveau de la ligue (un joueur dominant en LFL2 n'équivaut pas à un joueur moyen en LEC). Il est primordial de standardiser les performances par rapport à la moyenne de la ligue de chaque joueur (z-scores).
3. **Spécificités par Position** : Les distributions (ex: Kills, DPM, CS) varient naturellement entre les rôles (ex: un Support obtiendra des scores drastiquement différents d'un ADC). Une normalisation *par position* sera incontournable.
4. **Features Prometteuses** : L'Early Game (diffs à 15 mins) et les indicateurs d'efficacité (DPM, CSPM) discriminent nettement les talents promus des non-promus. Les promus ont aussi tendance à afficher un *Champion Pool* plus large.
5. **Corrélations Fortes** : Présence de redondances (ex: kills ↔ assists, total gold ↔ earned gold), ce qui influencera notre sélection de features ou nous poussera à utiliser des réductions de dimension.

### Phase 4 — Feature Engineering ✅

À partir des constats de l'EDA, nous avons développé le script `src/data/feature_engineering.py` qui transforme les données de matchs en un jeu de données "joueurs" prêt pour le ML.

1. **Recalcul de la Kill Participation** : Récupération de `killparticipation = (kills + assists) / team_kills` manquante dans la source.
2. **Agrégation par Joueur & Split** : Moyenne des statistiques par joueur (DPM, CSPM, VSPM, etc.) sur chaque split. Les joueurs avec moins de 5 matchs joués sur un split (remplaçants) ont été exclus pour éviter le bruit.
3. **Calcul des Tiers de Niveau (Z-Scores)** : Conversion des métriques clés en **écarts-types (z-scores) relatifs** calculés strictement par recoupement `[ligue, année, split, position]`. Une feature `dpm_zscore = 1.5` indique donc que le joueur inflige beaucoup plus de dégâts que ses concurrents directs *au même poste et dans la même ligue*.
4. **Validation (Notebook)** : Le notebook `02_feature_engineering.ipynb` confirme mathématiquement et visuellement que le Z-Score a un pouvoir de séparation élevé : les futures pépites promues (cibles) génèrent des Z-Scores positifs structurellement (notamment sur DPM, Gold@15 et Win Rate).

Le fichier généré est `data/processed/features_players.csv` (2 405 lignes, 31 colonnes).

### Phase 5 — Modélisation 🔜
### Phase 6 — Évaluation & Résultats 🔜
### Phase 7 — Documentation & Présentation 🔜

---

## 📊 Données

Les données proviennent de deux sources principales :
- **[Oracle's Elixir](https://oracleselixir.com/tools/downloads)** — Stats match-par-match de toutes les ligues pro (CSV, ~100+ colonnes)
- **[Leaguepedia](https://lol.fandom.com)** — Historique de transferts des joueurs (pour la target variable)

📄 Voir [DATA_SOURCES.md](DATA_SOURCES.md) pour la documentation complète des sources.

### Ligues couvertes

| Pays | Ligue | Statut dans les données |
|---|---|---|
| 🇫🇷 France | LFL | ✅ Présente |
| 🇫🇷 France | LFL2 (Division 2) | ✅ Présente |
| 🇩🇪 Allemagne | Prime League (PRM) | ✅ Présente |
| 🇪🇸 Espagne | LVP SuperLiga | ✅ Présente |
| 🇬🇧 UK/Nordics | NLC | ✅ Présente |
| 🇹🇷 Turquie | TCL | ✅ Présente |
| 🇪🇺 Europe | **LEC** *(target variable)* | ✅ Présente |

---

## 🔧 Stack Technique

| Catégorie | Technologies |
|---|---|
| **Data** | Python, Pandas, NumPy |
| **ML classique** | Scikit-Learn, XGBoost |
| **Deep Learning** | PyTorch |
| **Explicabilité** | SHAP |
| **Réduction dim.** | UMAP, PCA |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Logging** | Loguru |

---

## 📁 Structure du Repository

```
kcorp-scouting/
│
├── README.md                        # Ce fichier
├── DATA_SOURCES.md                  # Documentation des sources de données
├── LICENSE                          # MIT
├── Makefile                         # Commandes automatisées
├── pyproject.toml                   # Config projet Python
├── requirements.txt                 # Dépendances
│
├── data/                            # Données (non versionnées)
│   ├── raw/                         # CSV bruts Oracle's Elixir
│   ├── interim/                     # Données nettoyées
│   ├── processed/                   # Features prêtes pour le ML
│   └── external/                    # Données Leaguepedia
│
├── notebooks/                       # Jupyter notebooks (exploration)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_talent_scoring.ipynb
│   ├── 04_clustering_analysis.ipynb
│   └── 05_results_synthesis.ipynb
│
├── src/                             # Code source (production)
│   ├── config.py
│   ├── data/                        # Pipeline de données
│   ├── models/                      # Modèles ML
│   ├── visualization/               # Plots
│   └── utils/                       # Logger, helpers
│
├── models/                          # Modèles sauvegardés (.pkl, .pt)
├── reports/                         # Figures et métriques
└── tests/                           # Tests unitaires
```

---

## 🚀 Quick Start

```bash
# 1. Cloner le repo
git clone https://github.com/bower/kcorp-scouting.git
cd kcorp-scouting

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline complet
make all
```

---

## 📓 Notebooks

| Notebook | Contenu |
|---|---|
| `01_data_exploration` | EDA : distributions des stats par ligue et position |
| `02_feature_engineering` | Construction et validation des features |
| `03_talent_scoring` | Modèle supervisé : baseline → RF → XGBoost → PyTorch |
| `04_clustering_analysis` | K-Means, DBSCAN, Autoencoder + UMAP |
| `05_results_synthesis` | Top 10 pépites, validation vs. réalité |

---

## ⚠️ Limites & Biais

- **Biais de sélection** : Seuls les joueurs qui ont déjà joué en ERL sont dans les données. Les talents non détectés (joueurs Solo Queue) sont invisibles.
- **Target variable imparfaite** : "Promu en LEC" ≠ "bon joueur". Certains bons joueurs n'ont jamais eu l'opportunité.
- **Ce que le modèle ne capture pas** : Mental (tilt, pression), communication, capacité d'adaptation, motivation.
- **Données limitées à 2024-2026** : La méta de LoL change entre les saisons — un modèle entraîné sur 2024 peut être moins pertinent pour 2026.

---

## 📈 Améliorations Futures

- [ ] 📱 Dashboard Streamlit interactif
- [ ] 🎮 Intégration des données Solo Queue (Riot API)
- [ ] 📈 Modèle temporel (LSTM) pour capturer la progression
- [ ] 🌍 Extension aux ERLs mineures (Benelux, Italie, etc.)

---

## 📝 Licence

MIT — Voir [LICENSE](LICENSE)
