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

### Phase 5 — Modélisation ML (Talent Score) ✅

🎯 **Objectif** : Entraîner un modèle capable de prédire `promoted_to_lec` — identifier, parmi les joueurs ERL, ceux qui ont le potentiel d'évoluer en LEC.

🧠 **Raisonnement** :

1. **Out-of-Time split (Train 2024 → Test 2025)** : Un split aléatoire créerait du *data leakage* — le modèle verrait des données du futur pendant l'entraînement. Le split temporel simule un vrai déploiement : "Je forme le modèle sur 2024, et je l'utilise pour scorer les joueurs 2025."

2. **PR-AUC comme métrique principale** : Avec ~8% de joueurs promus, l'accuracy est trompeuse (un modèle naïf "tout-0" ferait 92%). La PR-AUC se concentre sur la qualité du ranking des positifs.

3. **Trois modèles comparés** : LR (baseline) → RF → XGBoost. Le RF a été optimisé avec `RandomizedSearchCV` (60 itérations, 5-fold stratifié).

🔧 **Ce qu'on a fait** :
- Split : **739 lignes train** (2024) | **977 lignes test** (2025), déséquilibre géré par `class_weight='balanced'`
- Features : 9 Z-scores relatifs + 3 valeurs brutes (`games_played`, `win_rate`, `champion_pool_size`)
- Tuning RF trouvé : `max_depth=6`, `min_samples_leaf=4`, `max_features=0.5`
- Visualisations : 6 figures générées dans `reports/figures/`

⚠️ **Problèmes rencontrés** :
1. **RF tuned < RF défaut** : Avec seulement 739 exemples, la relation est trop linéaire pour que le RF surpasse la LR. La complexité se retourne contre lui.
2. **Dominance de la LFL** : La LFL représente ~18/30 dans le leaderboard global. Un leaderboard par ligue a été ajouté.
3. **`games_played` très influent (24-32%)** : Conservé délibérément — c'est un signal réel (titulaire = essentiel à l'équipe).

✅ **Résultat** :

| Modèle | PR-AUC | ROC-AUC | Precision@104 |
|---|---|---|---|
| **Logistic Regression** 🏆 | **0.2557** | 0.729 | 24.0% |
| Random Forest | 0.1903 | 0.695 | 21.2% |
| RF (tuned) | 0.1820 | 0.689 | 19.2% |
| XGBoost | 0.1652 | 0.661 | 13.5% |

- **Modèle retenu : Logistic Regression** — interprétable et le plus performant
- **2 005 joueurs ERL scorés** dans `reports/metrics/talent_scores_players.csv`
- Médiane promus : **58/100** vs médiane non-promus : **21/100** — bon pouvoir de séparation

### Phase 6 — Playstyle Clustering ✅

🎯 **Objectif** : Regrouper les joueurs ERL par *style de jeu* plutôt que par niveau. Permettre des requêtes du type : *"Trouve-moi un joueur ERL dont le profil ressemble à Caps"* ou *"Quels midlaners ont un style carry agressif ?"*

🧠 **Raisonnement** :

Le Talent Score (Phase 5) répond à **"Qui va progresser ?"**. Le clustering répond à **"Comment joue-t-il ?"**. Ces deux dimensions sont complémentaires pour un scout :
- Un club peut chercher un joueur avec un style précis (ex: un support playmaker plutôt qu'un enchanteresse)
- Identifier des styles "LEC-ready" aide à comprendre *pourquoi* certains joueurs sont promus

**Algorithmes testés** : K-Means, DBSCAN, puis clustering par position.

🔧 **Ce qu'on a fait — Itération 1 : K-Means global (toutes positions)**

On démarre avec un K-Means global sur les 2 005 joueurs ERL, en utilisant les 9 Z-scores comme features. La méthode du coude + silhouette score donnent **k=3** comme optimal (silhouette=0.145).

| Cluster | Archétype généré | Joueurs | Taux promotion LEC |
|---|---|---|---|
| 0 | Profil équilibré | 890 | 6.3% |
| 1 | Profil neutre | 444 | 1.8% |
| **2** | **High DPS + Farmer + Early dominant** | 671 | **20.1%** |

Le Cluster 2 a un taux de promotion **3× supérieur** aux autres — signal fort que le profil "dominant en DPS, farming et early game" est le profil LEC-ready par excellence.

⚠️ **Problème rencontré — Échec de DBSCAN (documenté pour transparence)**

DBSCAN (Density-Based Spatial Clustering) a été testé en complément de K-Means. Le résultat est sans appel : **DBSCAN ne fonctionne pas sur ce dataset**, et comprendre pourquoi est une leçon importante.

**Cause : la malédiction de la dimensionnalité (*curse of dimensionality*)**

En espace de haute dimension (9 features après StandardScaler), les distances euclidiennes entre points tendent à se concentrer autour d'une même valeur — tous les points deviennent "également éloignés" les uns des autres. DBSCAN repose sur la notion de *densité locale* (voisins dans un rayon eps), qui perd tout son sens dans cet espace.

Résultats observés :
- `eps` petit (0.3–1.0) → 100% de bruit, 0 cluster formé
- `eps` moyen (1.5–2.5) → 1 méga-cluster absorbe tout, avec beaucoup de bruit
- `eps` grand (3.0–4.0) → 1 seul cluster, tous les points fusionnés

**Conclusion documentée** : DBSCAN est efficace en 2D ou 3D, sur des données avec des densités très variables. Sur des données tabulaires à 9 dimensions normalisées, K-Means est plus adapté.

⚠️ **Problème rencontré — Clustering global = archetypes trop génériques**

Avec k=3 et toutes les positions mélangées, les clusters manquent de granularité. Un support qui a un bon DPM n'est pas comparable à un ADC qui a un bon DPM — ce sont des métriques qui n'ont pas la même signification selon le rôle.

**Solution : Clustering par position**

On réexécute K-Means séparément pour chacune des 5 positions (top, jng, mid, bot, sup). Les avantages :
- La variance intra-cluster est naturellement réduite (les ADC sont comparés entre eux)
- Les archetypes ont du sens esport ("ADC carry vs. ADC farmer vs. ADC utility")
- Les silhouette scores augmentent significativement

🔧 **Ce qu'on a fait — Itération 2 : Clustering par position**

5 modèles K-Means indépendants, un par position. Le k optimal est recherché indépendamment pour chaque rôle via silhouette score. UMAP est appliqué par position pour la visualisation 2D.

✅ **Résultat** :

| Position | k optimal | Silhouette | Archétype le plus promu | Taux promo |
|---|---|---|---|---|
| Top | 3 | 0.169 | Carry agressif + Early dominant | **20.8%** |
| Jungle | 3 | 0.165 | High DPS + Early dominant | **23.0%** |
| Mid | 3 | 0.150 | High DPS + Farmer + Early dominant | **17.7%** |
| Bot (ADC) | 3 | 0.161 | High DPS + Farmer + Early dominant | **17.7%** |
| Support | 3 | 0.148 | High DPS + Vision controller + Versatile | **15.4%** |

**Signal transversal** : à chaque position, le cluster avec le plus haut taux de promotion LEC est le cluster "actif/dominant" — agressif en early game, farming efficace, haut DPS. Cela valide la cohérence du Talent Score avec le Clustering.

- 10 modèles sauvegardés (`clusterer_kmeans_{pos}.pkl` + `clusterer_scaler_{pos}.pkl`)
- Dataset enrichi : `reports/metrics/clustering_results.csv`
- Profils complets : `reports/metrics/cluster_profiles.json`

### Phase 7 — Synthèse & Documentation ✅

🎯 **Objectif** : Transformer les résultats bruts en outils d'aide à la décision concrets.

🧠 **Raisonnement** : Un modèle ML n'a de valeur que s'il répond à un besoin métier. La synthèse doit fusionner le "Qui" (Talent Score) et le "Comment" (Clustering) pour offrir une vue à 360° sur chaque joueur.

🔧 **Ce qu'on a fait** :
- Fusion des datasets (Scores + Clusters + Archetypes)
- Création du `05_results_synthesis.ipynb`
- Implémentation d'une suite de tests unitaires avec `pytest` pour garantir la robustesse du pipeline (tests sur les Z-scores, le data leakage, les probabilités, etc.)
- Centralisation des plots dans `src/visualization/plots.py`
- Rédaction finale de ce README orienté portfolio technique

✅ **Résultat** : Un pipeline End-to-End prêt pour le scouting, entièrement documenté et testé. Le projet démontre comment des concepts complexes (curse of dimensionality, class imbalance, time leakage) ont été gérés en conditions réelles.

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
| `01_data_exploration` | EDA : distributions des stats par ligue et position, déséquilibre des classes |
| `02_feature_engineering` | Construction et validation des features (Z-scores intra-ligue) |
| `03_talent_scoring` | Modèle supervisé : baselines, tuning Random Forest, Logistic Regression (retenue) |
| `04_clustering_analysis` | Clustering K-Means par position, analyse de l'échec de DBSCAN, UMAP |
| `05_results_synthesis` | Top 10 pépites, requêtes de scouting métier, fusion Score/Style |

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
