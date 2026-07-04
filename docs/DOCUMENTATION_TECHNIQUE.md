# Documentation Technique — KCorp Scouting Tool

> Public visé : développeur ou data scientist reprenant le projet.
> Pour une présentation accessible à tous, voir [DOCUMENTATION_FONCTIONNELLE.md](DOCUMENTATION_FONCTIONNELLE.md).

---

## 1. Vue d'ensemble

Le projet prédit, pour chaque joueur des ligues régionales européennes de League of Legends (ERLs), sa probabilité de débuter en **LEC** (la ligue européenne majeure) dans les 18 mois — le **Talent Score** — et regroupe les joueurs par **style de jeu** (clustering K-Means par position). Les résultats alimentent un dashboard Streamlit.

### Architecture du pipeline

```
                    ┌─────────────────────────────────────────────────┐
                    │              SOURCES DE DONNÉES                 │
                    │  Oracle's Elixir (CSV Google Drive, 2024-2026)  │
                    │  Leaguepedia (API Cargo — cross-check optionnel)│
                    └───────────────────────┬─────────────────────────┘
                                            │  make data
                                            ▼
  src/data/downloader.py     ─────►  data/raw/{année}_LoL_esports_....csv
                                            │  make clean
                                            ▼
  src/data/cleaner.py        ─────►  data/interim/cleaned_matches.csv
    (filtrage ligues/joueurs,               │  make features
     target datée 18 mois)                  ▼
  src/data/feature_engineering.py ─►  data/processed/features_players.csv
                                            │
                          ┌─────────────────┴─────────────────┐
                          │ make train                        │ make cluster
                          ▼                                   ▼
  src/models/talent_scorer.py               src/models/clusterer.py
    (LR/RF/XGB, calibration,                  (K-Means par position,
     percentiles)                              archétypes)
          │                                          │
          ▼                                          ▼
  reports/metrics/talent_scores_players.csv  reports/metrics/clustering_results.csv
  reports/metrics/talent_score_results.json  reports/metrics/cluster_profiles.json
  models/talent_scorer_best.pkl              models/clusterer_kmeans_{pos}.pkl
          │                                          │
          └─────────────────┬────────────────────────┘
                            │  (snapshots committés dans git)
                            ▼
              app/ (Streamlit) — ne relance JAMAIS le pipeline ML
              ├── app.py                    (accueil, métriques globales)
              ├── utils/data_loader.py      (lecture CSV + @st.cache_data)
              └── pages/1_Leaderboard, 2_Profil_Joueur, 3_Scout_Mode
```

Principe clé : **le dashboard lit des snapshots pré-calculés**, jamais les modèles `.pkl` ni les données brutes. `make run-pipeline` (= `train` + `cluster`) régénère les deux CSV de `reports/metrics/`, qui sont whitelistés dans `.gitignore` et committés pour le déploiement Streamlit Cloud.

---

## 2. Configuration — `src/config.py`

| Paramètre | Valeur | Effet |
|---|---|---|
| `DATA_YEARS` | `[2024, 2025, 2026]` | Années téléchargées et chargées (2026 partiel). |
| `GOOGLE_DRIVE_IDS` | dict {année: id} | IDs des CSV Oracle's Elixir sur Google Drive. |
| `ERL_DIV1_LEAGUES` | LFL, PRM, LVP SL, NLC, TCL | ERLs division 1 ciblées. |
| `ERL_DIV2_LEAGUES` | LFL2 | Seule Div 2 présente dans les données. |
| `TOP_LEAGUE` | `"LEC"` | Ligue cible de la target variable. |
| `TRAIN_YEARS` / `TEST_YEARS` | `[2024]` / `[2025]` | Split temporel Out-of-Time. ⚠️ 2026 n'est ni en train ni en test : ses lignes sont scorées mais jamais évaluées. |
| `PROMOTION_HORIZON_MONTHS` | `18` | Fenêtre de la target datée (voir §3.3). |
| `KEY_COLUMNS` | ~34 colonnes | Colonnes conservées à l'étape de nettoyage. `killparticipation` y figure mais est absente des CSV récents → recalculée en Phase 4. |
| `RANDOM_STATE` | `42` | Reproductibilité (splits, modèles, K-Means). |
| `RF_PARAMS`, `XGB_PARAMS` | dicts | Hyperparamètres de base (le RF est ensuite tuné). |
| `CLUSTER_PARAMS["k_range"]` | `range(3, 12)` | Valeurs de k testées par silhouette. |

Paramètres **morts** (aucun code ne les consomme) : `AUTOENCODER_PARAMS`, `FEEDFORWARD_PARAMS`, `CLUSTER_PARAMS["dbscan_*"]` — vestiges d'itérations abandonnées.

---

## 3. Pipeline de données

### 3.1 Téléchargement — `src/data/downloader.py`

- Télécharge les CSV Oracle's Elixir (~75 Mo/année) depuis Google Drive, avec gestion du token anti-virus des gros fichiers, retry avec backoff (3 tentatives), et **cache local** : un fichier existant > 1 Mo n'est pas re-téléchargé (`force=True` pour forcer).
- Garde-fou : un fichier < 1 Ko contenant du HTML est détecté comme page d'erreur et supprimé.

### 3.2 Nettoyage — `src/data/cleaner.py` (`run_cleaning_pipeline`)

8 étapes journalisées :
1. **Chargement** des CSV bruts (292 k lignes × 166 colonnes pour 2024-26), colonne `_source_year` ajoutée.
2. **Découverte des ligues** (log informatif).
3. **Filtrage ligues** : exact match sur `ALL_TARGET_LEAGUES` (7 ligues), avec repli en match partiel case-insensitive si une ligue manque. → ~49 k lignes.
4. **Filtrage joueurs** : suppression des lignes `position == "team"` (2 par match). → ~41 k lignes.
5. **Sélection de colonnes** : 166 → ~34 (`KEY_COLUMNS`).
6. **Normalisation des noms** : `strip().lower()` ; l'original est gardé dans `playername_original` (non utilisé en aval — le dashboard affiche donc les pseudos en minuscules).
7. **Valeurs manquantes** : drop si NaN d'identité ; imputation des stats par **médiane de ligue** (puis médiane globale en repli) ; catégorielles → `"unknown"`.
8. **Target datée** (voir ci-dessous), puis validation (assertions loggées) et sauvegarde → `data/interim/cleaned_matches.csv`.

### 3.3 Target datée — cœur méthodologique

Fonctions : `compute_lec_debut_dates()` et `build_dated_target_from_oracle()`.

```
promoted_to_lec(match ERL) = True  ssi
    date_début_LEC(joueur) existe
    ET date_début_LEC > date(match)                      ← pas d'ex-LEC relégué
    ET date_début_LEC ≤ date(match) + 18 mois            ← horizon borné
```

- `date_début_LEC` = date du **premier** match LEC du joueur dans les données (MIN sur les lignes LEC).
- Corrige deux fuites de l'ancienne logique (« a joué en LEC un jour ») : un ex-LEC relégué en ERL n'est plus positif, et on ne « prédit » plus une promotion déjà passée.
- **Censure à droite** : une promotion postérieure aux données (ex. LEC 2027) est inobservable → étiquetée négative. Les scores des splits récents sont donc mécaniquement sous-estimés.
- Chiffres (pipeline de juillet 2026) : 109 joueurs avec date de début LEC, 2 275 lignes-matchs positives, 43 joueurs « pré-promotion » sur 872 joueurs ERL.
- Leaguepedia (`src/data/leaguepedia.py`) n'est plus qu'un **cross-check informatif** : l'API Cargo ne fournit pas de dates de transfert exploitables, donc l'utiliser comme label ré-introduirait la fuite. Le module complet (career map, labels N/N+1) reste utilisable en CLI mais n'alimente pas la target.

### 3.4 Features — `src/data/feature_engineering.py`

1. **Kill Participation recalculée** : `(kills + assists) / kills_équipe_du_match`, bornée [0, 1] (absente des exports récents d'Oracle's Elixir).
2. **Agrégation** par `[playername, league, _source_year, split, position, teamname]` :
   - moyennes des stats de jeu (dpm, cspm, vspm, diffs@15, damageshare…),
   - `games_played` (count), `win_rate` (moyenne de `result`), `champion_pool_size` (nunique),
   - `promoted_to_lec` = **max** sur le split (un seul match pré-promotion suffit à marquer le split).
   - Filtre de représentativité : `games_played ≥ 5`, sinon la ligne est exclue (311 lignes exclues → 2 438 lignes joueur/split).
3. **Z-scores** par groupe `[league, _source_year, split, position]` : `(x − μ_groupe) / σ_groupe`, 0 si σ = 0 ou groupe singleton. C'est ce qui rend les joueurs comparables entre ligues de niveaux différents.
4. Sortie → `data/processed/features_players.csv` (2 438 × 31).

⚠️ Ce module utilise des chemins **relatifs** (`data/interim/...`) et non les constantes de `config.py` : il doit être lancé depuis la racine du projet.

---

## 4. Modélisation

### 4.1 Talent Score — `src/models/talent_scorer.py`

- **Features** (12) : 9 z-scores + 3 brutes (`win_rate`, `games_played`, `champion_pool_size`). Cible : `promoted_to_lec`.
- **Split Out-of-Time** : train = 2024 (739 lignes, 49 positifs, 6,6 %), test = 2025 (977 lignes, 61 positifs). Seules les lignes ERL entrent dans train/test (la LEC ne sert qu'à la target).
- **3 modèles** : LogisticRegression (pipeline StandardScaler, C=0.1, class_weight=balanced) ; RandomForest (`RF_PARAMS`) ; XGBoost (`scale_pos_weight` calculé dynamiquement, import **optionnel** — le pipeline le saute si non installé).
- **Tuning RF** : `RandomizedSearchCV` 60 itérations, scoring `"average_precision"`, CV = **`StratifiedGroupKFold(5)` avec `groups=playername`** — un joueur multi-saisons n'est jamais à la fois en train et en validation d'un fold (anti-fuite).
- **Sélection** : meilleur PR-AUC test → LogisticRegression (PR-AUC 0.1325 vs baseline aléatoire 0.062).
- **Calibration** : le meilleur modèle est enveloppé dans `CalibratedClassifierCV(method="sigmoid", cv=5)` (Platt ; isotonic écarté : ~49 positifs en train). Transformation monotone → classement inchangé. **Brier score test : 0.161 → 0.062.** C'est le modèle calibré qui est sérialisé (`models/talent_scorer_best.pkl`) et qui score les joueurs.
- **Scoring final** (`score_all_players`) : toutes les lignes ERL (train + test + 2026) reçoivent `talent_score = P(promotion) × 100` et `score_percentile` = rang percentile **par position** (`rank(pct=True) × 100` ; 100 = meilleur du poste).

Métriques de référence (run juillet 2026) :

| Modèle | PR-AUC | ROC-AUC | Precision@61 |
|---|---|---|---|
| Logistic Regression 🏆 | 0.1325 | 0.709 | 13,1 % |
| Random Forest | 0.1282 | 0.711 | 11,5 % |
| RF (tuned) | 0.1181 | 0.674 | 13,1 % |
| XGBoost | 0.0899 | 0.610 | 9,8 % |

### 4.2 Playstyle Clustering — `src/models/clusterer.py`

- **5 modèles K-Means indépendants**, un par position (un DPM élevé n'a pas le même sens pour un support et un ADC). Entrée : les 9 z-scores, re-standardisés (`StandardScaler`) sur les 2 005 lignes ERL.
- **k choisi par silhouette** sur `k_range` (3-11) → k=3 partout (silhouettes ~0,15-0,17 : clusters peu séparés, à interpréter comme des tendances).
- **Réduction 2D** pour visualisation : UMAP si installé, sinon PCA (`umap_x`, `umap_y`).
- **Archétypes** : libellés générés par règles à seuils sur les z-scores moyens du cluster (`ARCHETYPE_RULES` : « Carry agressif » si dpm_z > 0.3 et kp_z > 0.3, « Vision controller » si vspm_z > 0.5, etc. ; « Profil neutre » sinon).
- Sorties : `clustering_results.csv` (avec `cluster_position`, son alias `cluster` attendu par le dashboard, et `archetype` par ligne), `cluster_profiles.json` (profils moyens par position/cluster), `kmeans_k_scores.json` (courbes coude/silhouette), modèles `.pkl` par position.
- `find_similar_players()` (distance euclidienne dans l'espace standardisé) existe mais n'est utilisée que par les tests/notebooks — le dashboard fait sa similarité par « même cluster + même position ».

---

## 5. Schémas des fichiers générés

### `reports/metrics/talent_scores_players.csv` (2 005 lignes, 1 par joueur/split ERL)

| Colonne | Type | Description |
|---|---|---|
| `playername` | str | Pseudo normalisé (minuscules). |
| `league`, `_source_year`, `split`, `position`, `teamname` | str/int | Contexte de la ligne. |
| `talent_score` | float | Probabilité calibrée de promotion × 100 (plage réelle ≈ 0,1-70). |
| `score_percentile` | float | Rang percentile intra-position, (0, 100]. |
| `promoted_to_lec` | bool | Target datée (vérité terrain). |
| `win_rate`, `games_played`, `champion_pool_size` | num | Stats brutes du split. |
| `dpm_zscore`, `cspm_zscore`, `golddiffat15_zscore` | float | Sous-ensemble de z-scores (pour l'affichage). |

### `reports/metrics/clustering_results.csv` (2 005 lignes)

Identifiants identiques + `cluster_position` (int), `cluster` (alias), `archetype` (str), `umap_x/y` (float), `promoted_to_lec`, `win_rate`, `games_played`, `dpm_zscore`, `cspm_zscore`, `golddiffat15_zscore`, `win_rate_zscore`.

### JSON

- `talent_score_results.json` : tableau `comparison` (métriques par modèle), `best_model`, tailles train/test, hyperparamètres RF tunés, `rf_tuned_cv_pr_auc`, `calibration_method`, `brier_score_raw`, `brier_score_calibrated`.
- `cluster_profiles.json` : `{position: [{cluster, n_players, n_promoted, promo_rate, archetype, <z-scores moyens>}]}`.

---

## 6. Dashboard Streamlit — `app/`

- **`utils/data_loader.py`** : 3 loaders (`load_talent_scores`, `load_clustering_results`, `load_cluster_profiles`) décorés `@st.cache_data` (Streamlit ré-exécute tout le script à chaque interaction). Chemins résolus relativement au fichier (`reports/metrics/` trois niveaux au-dessus). Erreur explicite si un CSV manque (avec la commande pour le régénérer). Helpers non cachés : `get_archetype`, `list_archetypes`.
- **`app.py`** : accueil, métriques globales, guide de navigation. La sidebar contient des chiffres codés en dur (voir revue de code — à synchroniser avec `talent_score_results.json`).
- **`pages/1_Leaderboard.py`** : tableau trié par `talent_score` avec filtres (position, ligue, année, score minimum 0-100) + bar chart top 20 Plotly.
- **`pages/2_Profil_Joueur.py`** : sélection d'un joueur → la ligne « peak » (meilleur talent_score) est affichée : métriques, percentile (« Top X % des {POS} ERL »), archétype (via `clustering_results.csv`, repli sur `cluster_profiles.json`), radar chart Plotly des z-scores, historique des saisons.
- **`pages/3_Scout_Mode.py`** : (1) shortlist multi-critères (position/archétype/ligue/score), (2) « joueurs similaires » = même cluster K-Means + même position que la saison peak du joueur de référence, avec scatter comparatif. La jointure scores↔clusters se fait sur `playername+position+_source_year+split` pour associer chaque saison à SON cluster.
- **Échelle** : toutes les `ProgressColumn` et sliders sont en 0-100, alignés sur le CSV.
- **Déploiement** : Streamlit Community Cloud clone le repo et lit les CSV committés ; `.streamlit/config.toml` fixe le thème sombre. Procédure pas-à-pas dans le README.

---

## 7. Qualité : tests, CI, outillage

- **`tests/test_features.py`** (13 tests) : z-scores intra-groupe (moyenne ≈ 0, pas de fuite inter-ligue, groupes singletons), format de la target, **5 tests de la target datée** (pré-promotion positif, ex-LEC relégué négatif, hors horizon négatif, jamais promu négatif, MIN des dates LEC), structure du dataset.
- **`tests/test_models.py`** (15 tests) : isolation du split OOT, probabilités valides, PR-AUC > baseline, importances RF sommant à 1, K-Means (labels valides, pas de cluster vide), UMAP (skippé si non installé), similarity search, **percentiles** (plage (0,100], max=100 par position), **échelle 0-100 du talent_score**, **non-fuite des groupes** en `StratifiedGroupKFold`.
- **CI GitHub Actions** (`.github/workflows/ci.yml`) : matrice Python 3.10/3.11, `ruff check src/ tests/` + `pytest tests/`. Dépendances volontairement minimales (pas de torch/xgboost/umap — `xgboost` est un import optionnel dans `talent_scorer`, le test UMAP se skippe). Déclencheurs : push sur `main`, PR, manuel.
- **Makefile** : cibles chaînées `data → clean → features → train/cluster`, `run-pipeline` (artefacts dashboard), `app`, `test`, `lint`, `format`. Utilise `python` (sous Windows sans alias, préférer `py -m …`).
- **Lint** : ruff (`E, F, I, W`, ligne 120). Formatage : black (120).

---

## 8. Limites techniques connues

1. **Censure à droite** de la target (cf. §3.3) : les splits récents sont sous-étiquetés.
2. **Matching par pseudo** : la target et les jointures reposent sur `playername` en minuscules — deux joueurs homonymes seraient fusionnés ; un changement de pseudo casse l'historique. Pas de fuzzy matching Oracle's ↔ Leaguepedia.
3. **Imputation pré-split** : les médianes de ligue (étape 7 du nettoyage) sont calculées sur toutes les années confondues — vecteur de fuite mineur mais réel.
4. **2026 non évalué** : `TEST_YEARS=[2025]` ; les lignes 2026 sont scorées sans jamais contribuer aux métriques.
5. **Silhouettes faibles** (~0,15) : les archétypes sont des tendances, pas des catégories nettes.
6. **Petit dataset** (739 lignes de train, 49 positifs) : variance élevée des métriques ; la LR bat les modèles complexes.
7. Les points ouverts détectés lors de la revue de code de juillet 2026 (bugs dans `src/visualization/`, chiffres codés en dur dans `app.py`, dépendances inutilisées…) sont suivis hors documentation — voir le rapport de revue.
