"""
talent_scorer.py — Phase 5 : Modélisation du Talent Score

Ce module implémente le pipeline complet de modélisation supervisée pour prédire
la variable `promoted_to_lec` : identifier les joueurs ERL qui ont le potentiel
d'évoluer en LEC.

Approche :
  - Out-of-Time split : Train sur 2024, Test sur 2025/2026
    → Évite le data leakage temporel (le modèle ne "voit pas le futur")
  - Métrique principale : PR-AUC (Precision-Recall AUC)
    → Adaptée aux datasets déséquilibrés (~7.5% de positifs)
    → L'accuracy classique serait trompeuse ici (un modèle "tout-0" ferait 92.5%)
  - 3 modèles comparés :
    1. Logistic Regression (baseline interprétable)
    2. Random Forest avec class_weight='balanced'
    3. XGBoost avec scale_pos_weight ajusté

Usage :
  make train
  # ou
  python -m src.models.talent_scorer
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    XGB_PARAMS,
    TRAIN_YEARS,
    TEST_YEARS,
)
from src.utils.logger import logger

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et Split Temporel
# ═══════════════════════════════════════════════════════════════════════════════

# Features utilisées pour l'entraînement.
# On utilise les Z-scores (relatifs intra-ligue/position) plutôt que les valeurs
# brutes pour éviter que le modèle apprenne les différences de niveau entre ligues
# plutôt que le talent réel du joueur.
FEATURE_COLS = [
    "killparticipation_zscore",
    "dpm_zscore",
    "cspm_zscore",
    "vspm_zscore",
    "golddiffat15_zscore",
    "csdiffat15_zscore",
    "xpdiffat15_zscore",
    "win_rate_zscore",
    "champion_pool_size_zscore",
    # Valeurs brutes complémentaires (contexte)
    "win_rate",
    "games_played",
    "champion_pool_size",
]

TARGET_COL = "promoted_to_lec"


def load_features() -> pd.DataFrame:
    """Charge le dataset de features généré par la Phase 4."""
    features_path = PROCESSED_DATA_DIR / "features_players.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"❌ {features_path} introuvable. Lancez d'abord `make features`."
        )
    df = pd.read_csv(features_path)

    # Correction du type booléen (pandas lit les booleans CSV comme strings)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower() == "true"

    logger.info(f"📂 Features chargées : {len(df):,} lignes × {len(df.columns)} colonnes")
    return df


def make_out_of_time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Crée un split temporel (Out-of-Time) :
      - Train : années TRAIN_YEARS (config.py → [2024])
      - Test  : années TEST_YEARS  (config.py → [2025, 2026])

    Pourquoi pas un split aléatoire ?
    → Avec un random split, des matchs du futur pourraient se retrouver dans
      le train set. Le modèle apprendrait à "prédire" des promotions qu'il a
      déjà "vues" dans les données d'entraînement. C'est du data leakage.
    → L'Out-of-Time split simule un vrai scénario de déploiement :
      "Je forme le modèle sur 2024, et je l'utilise pour scorer les joueurs 2025."

    Remarque : Pour les joueurs promus, leur `promoted_to_lec` est marqué à True
    sur TOUTES leurs lignes (tous les splits). On filtre les ligues ERL uniquement
    pour le train/test afin de ne pas inclure les performances LEC dans les features.
    """
    # Ne garder que les ligues ERL (pas la LEC elle-même — la LEC est la target)
    erl_leagues = ["LFL", "LFL2", "LVP SL", "NLC", "PRM", "TCL"]
    df_erl = df[df["league"].isin(erl_leagues)].copy()

    # Vérification des features disponibles
    available_features = [c for c in FEATURE_COLS if c in df_erl.columns]
    missing = [c for c in FEATURE_COLS if c not in df_erl.columns]
    if missing:
        logger.warning(f"⚠️  Features manquantes ignorées : {missing}")

    # Split temporel
    train_mask = df_erl["_source_year"].isin(TRAIN_YEARS)
    test_mask = df_erl["_source_year"].isin(TEST_YEARS)

    df_train = df_erl[train_mask]
    df_test = df_erl[test_mask]

    X_train = df_train[available_features].fillna(0)
    y_train = df_train[TARGET_COL].astype(int)
    X_test = df_test[available_features].fillna(0)
    y_test = df_test[TARGET_COL].astype(int)

    logger.info(f"📊 Split temporel Out-of-Time :")
    logger.info(f"   Train ({TRAIN_YEARS}) : {len(X_train):,} lignes | {y_train.sum()} promus ({y_train.mean():.1%})")
    logger.info(f"   Test  ({TEST_YEARS})  : {len(X_test):,} lignes  | {y_test.sum()} promus ({y_test.mean():.1%})")

    return X_train, y_train, X_test, y_test, df_train, df_test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Définition des Modèles
# ═══════════════════════════════════════════════════════════════════════════════

def build_logistic_regression() -> Pipeline:
    """
    Baseline : Régression Logistique.

    Avantages :
      - Très interprétable : les coefficients montrent directement l'impact de
        chaque feature sur la probabilité de promotion
      - Rapide à entraîner
      - Bonne référence pour juger si les modèles plus complexes apportent
        vraiment quelque chose

    Paramètres :
      - class_weight='balanced' : ajuste automatiquement les poids des classes
        inversement proportionnellement à leur fréquence (compense le déséquilibre)
      - max_iter=1000 : augmenté car les données peuvent nécessiter plus d'itérations
        pour converger avec StandardScaler

    Note : La LR nécessite une normalisation (StandardScaler) car elle est sensible
    à l'échelle des features. Les Z-scores sont déjà normalisés, mais games_played
    et champion_pool_size ne le sont pas.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=0.1,  # Régularisation L2 (évite le surapprentissage sur petit dataset)
        )),
    ])


def build_random_forest() -> RandomForestClassifier:
    """
    Modèle principal : Random Forest.

    Avantages par rapport à la LR :
      - Capture les non-linéarités (ex: un très bon DPM combiné à un bon Win Rate
        peut être plus prédictif que chaque feature seule)
      - Robuste aux outliers et aux features corrélées
      - Fournit une importance des features (feature importance)
      - Pas besoin de normalisation

    Les hyperparamètres viennent de config.py (RF_PARAMS) :
      - class_weight='balanced' : comme pour la LR
      - max_depth=10 : limite la profondeur pour éviter le surapprentissage
      - n_jobs=-1 : utilise tous les cœurs CPU disponibles
    """
    return RandomForestClassifier(**RF_PARAMS)


def build_xgboost() -> xgb.XGBClassifier:
    """
    Modèle avancé : XGBoost (Gradient Boosting).

    Avantages par rapport au Random Forest :
      - Souvent plus performant sur des datasets tabulaires de taille moyenne
      - scale_pos_weight gère le déséquilibre différemment (pondération de la loss)
      - Régularisation intégrée (L1/L2)

    scale_pos_weight sera calculé dynamiquement dans train_models() pour être
    exact (= ratio négatifs/positifs dans le train set).
    """
    params = XGB_PARAMS.copy()
    # Le scale_pos_weight sera mis à jour dynamiquement avant l'entraînement
    params["use_label_encoder"] = False
    params["verbosity"] = 0
    return xgb.XGBClassifier(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Entraînement et Évaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Évalue un modèle entraîné sur le test set.

    Métriques calculées :
      - PR-AUC  : Aire sous la courbe Précision-Rappel (métrique principale)
        → Insensible au déséquilibre, mesure la qualité du ranking
      - ROC-AUC : Aire sous la courbe ROC (métrique secondaire, pour référence)
      - Precision@K : Parmi les K joueurs avec le plus haut score, combien sont
        réellement des promus ? (K = nombre de promus dans le test set)
      - Recall@K : Parmi tous les promus réels, combien sont dans le Top-K ?

    Pourquoi PR-AUC et pas ROC-AUC ou Accuracy ?
      - Accuracy : "92% correct" si on prédit toujours 0 → inutile
      - ROC-AUC : moins informatif quand les classes sont très déséquilibrées
      - PR-AUC : se concentre sur les positifs (les promus), ce qui est notre
        vrai objectif : trouver les pépites dans la masse
    """
    # Probabilités de promotion (score continu 0-1)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Precision & Recall @ K (K = nb de promus réels dans le test set)
    k = int(y_test.sum())
    top_k_indices = np.argsort(y_proba)[::-1][:k]
    top_k_true = y_test.iloc[top_k_indices].values
    precision_at_k = top_k_true.sum() / k
    recall_at_k = top_k_true.sum() / y_test.sum()

    logger.info(f"\n{'─'*50}")
    logger.info(f"📊 [{name}]")
    logger.info(f"   PR-AUC       : {pr_auc:.4f}  ← métrique principale")
    logger.info(f"   ROC-AUC      : {roc_auc:.4f}")
    logger.info(f"   Precision@{k} : {precision_at_k:.2%}  (Parmi le Top-{k} scoré, X% sont vrais promus)")
    logger.info(f"   Recall@{k}    : {recall_at_k:.2%}  (On capture X% des vrais promus dans le Top-{k})")
    logger.info(f"{'─'*50}")

    return {
        "model_name": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        f"precision_at_{k}": round(precision_at_k, 4),
        f"recall_at_{k}": round(recall_at_k, 4),
        "k": k,
    }


def get_feature_importances(
    model_name: str,
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extrait et trie les importances de features selon le type de modèle.

    - Pour la LR (Pipeline) : coefficients absolus (impact directionnel)
    - Pour le RF / XGBoost  : feature_importances_ (gain moyen)

    Les importances sont normalisées pour sommer à 1 (%).
    """
    if isinstance(model, Pipeline):
        # Logistic Regression encapsulée dans un Pipeline
        coefs = np.abs(model.named_steps["model"].coef_[0])
        importances = coefs / coefs.sum()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"\n🔍 Top 5 features [{model_name}] :")
    for _, row in df_imp.head(5).iterrows():
        logger.info(f"   {row['feature']:<35} {row['importance']:.2%}")

    return df_imp


def score_all_players(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Applique le modèle sur tous les joueurs ERL (train + test) pour générer
    le Talent Score final (probabilité 0-100).

    C'est la sortie principale du projet : un classement des joueurs ERL
    par potentiel de promotion en LEC.
    """
    erl_leagues = ["LFL", "LFL2", "LVP SL", "NLC", "PRM", "TCL"]
    df_erl = df[df["league"].isin(erl_leagues)].copy()

    available_features = [c for c in feature_cols if c in df_erl.columns]
    X = df_erl[available_features].fillna(0)

    df_erl["talent_score"] = model.predict_proba(X)[:, 1] * 100

    # Sélection des colonnes pour le rapport final
    report_cols = [
        "playername", "league", "_source_year", "split", "position", "teamname",
        "talent_score", "promoted_to_lec",
        "win_rate", "games_played", "champion_pool_size",
        "dpm_zscore", "cspm_zscore", "golddiffat15_zscore",
    ]
    report_cols = [c for c in report_cols if c in df_erl.columns]

    return df_erl[report_cols].sort_values("talent_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Tuning du Random Forest
# ═══════════════════════════════════════════════════════════════════════════════

def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 60,
    cv_folds: int = 5,
) -> RandomForestClassifier:
    """
    Recherche les meilleurs hyperparamètres du Random Forest via RandomizedSearchCV.

    Stratégie :
      - RandomizedSearch plutôt que GridSearch : plus efficace sur un grand espace
        de paramètres (on teste n_iter combinaisons aléatoires au lieu de toutes)
      - StratifiedKFold : préserve le ratio de classes dans chaque fold
        (important car ~8% de positifs — sans stratification, certains folds
        pourraient avoir 0 promus)
      - Métrique : PR-AUC (average_precision_score)
        → Cohérent avec l'évaluation finale

    Espace de recherche :
      - n_estimators     : nombre d'arbres (plus = moins de variance, mais plus lent)
      - max_depth        : profondeur max (None = arbres complets → risque overfitting)
      - min_samples_split: split minimum (plus grand = arbres moins profonds = moins d'overfitting)
      - min_samples_leaf : feuilles minimum (régularisation)
      - max_features     : nb de features testées à chaque split (sqrt = recommandé)
      - class_weight     : 'balanced' ou 'balanced_subsample' (le 2e applique le
        rééquilibrage sur chaque bootstrap sample, parfois plus robuste)

    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        n_iter: Nombre de combinaisons aléatoires à tester
        cv_folds: Nombre de folds pour la cross-validation

    Returns:
        Le meilleur RandomForestClassifier entraîné sur le train set complet
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RF TUNING — RandomizedSearchCV ({n_iter} iterations, {cv_folds}-fold CV)")
    logger.info(f"{'='*60}")

    param_distributions = {
        "n_estimators":      [100, 200, 300, 500, 700],
        "max_depth":         [4, 6, 8, 10, 12, 15, None],
        "min_samples_split": [2, 5, 10, 20, 30],
        "min_samples_leaf":  [1, 2, 4, 8],
        "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
        "class_weight":      ["balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # StratifiedKFold pour respecter le ratio de classes à chaque fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # Scorer basé sur PR-AUC (average_precision_score)
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=pr_auc_scorer,
        cv=cv,
        refit=True,       # Re-entraîne le meilleur modèle sur tout le train set
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    search.fit(X_train, y_train)

    logger.info(f"\nMeilleure PR-AUC (CV) : {search.best_score_:.4f}")
    logger.info(f"Meilleurs hyperparamètres :")
    for param, value in search.best_params_.items():
        logger.info(f"   {param:<25} = {value}")

    return search.best_estimator_, search.best_params_, search.best_score_


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Orchestrateur Principal
# ═══════════════════════════════════════════════════════════════════════════════

def run_talent_scoring_pipeline():
    """
    Point d'entrée du module : entraîne et évalue les 3 modèles,
    sauvegarde le meilleur modèle et les résultats.
    """
    logger.info("=" * 60)
    logger.info("🎯 KCORP SCOUTING PIPELINE — PHASE 5 : TALENT SCORE")
    logger.info("=" * 60)

    # ── Chargement et split ──────────────────────────────────────────────────
    df = load_features()
    X_train, y_train, X_test, y_test, df_train, df_test = make_out_of_time_split(df)
    available_features = [c for c in FEATURE_COLS if c in X_train.columns]

    # ── Calcul dynamique de scale_pos_weight pour XGBoost ───────────────────
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(f"⚖️  Déséquilibre train : {neg_count} négatifs / {pos_count} positifs → scale_pos_weight={scale_pos_weight:.1f}")

    # ── Construction des modèles ─────────────────────────────────────────────
    models = {
        "Logistic Regression (baseline)": build_logistic_regression(),
        "Random Forest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }
    # Mise à jour du scale_pos_weight XGBoost avec la valeur calculée
    models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

    # ── Entraînement des modèles de base ────────────────────────────────────
    logger.info("\n--- Modeles de base ---")
    results = []
    trained_models = {}
    feature_importances = {}

    for name, model in models.items():
        logger.info(f"\nEntrainement : {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        fi = get_feature_importances(name, model, available_features)
        if not fi.empty:
            feature_importances[name] = fi

    # ── Tuning du Random Forest ──────────────────────────────────────────────
    logger.info("\n--- Tuning Random Forest (RandomizedSearchCV) ---")
    rf_tuned, best_rf_params, cv_score = tune_random_forest(
        X_train, y_train, n_iter=60, cv_folds=5
    )
    trained_models["Random Forest (tuned)"] = rf_tuned
    rf_tuned_metrics = evaluate_model("Random Forest (tuned)", rf_tuned, X_test, y_test)
    results.append(rf_tuned_metrics)

    rf_tuned_fi = get_feature_importances("Random Forest (tuned)", rf_tuned, available_features)
    if not rf_tuned_fi.empty:
        feature_importances["Random Forest (tuned)"] = rf_tuned_fi
        rf_tuned_fi.to_csv(METRICS_DIR / "feature_importance_random_forest_tuned.csv", index=False)

    # ── Sélection du meilleur modèle (selon PR-AUC) ──────────────────────────
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    best_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_name]

    logger.info(f"\n🏆 Meilleur modèle : {best_name}")
    logger.info(f"   PR-AUC : {results_df.iloc[0]['pr_auc']:.4f}")

    # ── Scoring de tous les joueurs ERL ─────────────────────────────────────
    logger.info("\n🎯 Scoring de tous les joueurs ERL...")
    all_scores = score_all_players(best_model, df, FEATURE_COLS)

    logger.info(f"\n🏅 Top 10 Talents ERL :")
    logger.info(f"{'Rang':<5} {'Joueur':<20} {'Ligue':<10} {'Pos':<5} {'Score':<8} {'Promu?'}")
    for i, (_, row) in enumerate(all_scores.head(10).iterrows(), 1):
        promoted = "✅" if row.get("promoted_to_lec", False) else "❌"
        logger.info(
            f"  {i:<4} {str(row['playername']):<20} {str(row['league']):<10} "
            f"{str(row['position']):<5} {row['talent_score']:.1f}/100   {promoted}"
        )

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR.parent / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    # Meilleur modèle sérialisé
    model_path = MODELS_DIR / "talent_scorer_best.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"\n💾 Modèle sauvegardé : {model_path}")

    # Tous les modèles (pour comparaison dans le notebook)
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODELS_DIR / f"talent_scorer_{safe_name}.pkl")

    # Métriques JSON (pour le notebook et le rapport)
    metrics_path = METRICS_DIR / "talent_score_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "comparison": results_df.to_dict(orient="records"),
                "best_model": best_name,
                "train_years": TRAIN_YEARS,
                "test_years": TEST_YEARS,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(available_features),
                "features_used": available_features,
                "rf_tuned_params": best_rf_params,
                "rf_tuned_cv_pr_auc": round(cv_score, 4),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Metriques sauvegardees : {metrics_path}")

    # Scores joueurs CSV
    scores_path = METRICS_DIR / "talent_scores_players.csv"
    all_scores.to_csv(scores_path, index=False)
    logger.info(f"📊 Scores joueurs sauvegardés : {scores_path}")

    # Feature importances CSV par modèle
    for name, fi_df in feature_importances.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fi_df.to_csv(METRICS_DIR / f"feature_importance_{safe_name}.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 5 TERMINÉE")
    logger.info(f"   Modèles comparés  : {len(models)}")
    logger.info(f"   Meilleur modèle   : {best_name}")
    logger.info(f"   PR-AUC (test)     : {results_df.iloc[0]['pr_auc']:.4f}")
    logger.info(f"   Joueurs scorés    : {len(all_scores):,}")
    logger.info("=" * 60)

    return trained_models, results_df, all_scores


if __name__ == "__main__":
    run_talent_scoring_pipeline()
