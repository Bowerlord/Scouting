"""
cleaner.py — Nettoyage et filtrage des données multi-ligues

Ce module est le cœur du pipeline de données. Il prend les CSV bruts
d'Oracle's Elixir et produit un dataset nettoyé, filtré par ligue,
avec la target variable "promoted_to_top_league" ajoutée.

Étapes du nettoyage (8 étapes loggées, + validation et sauvegarde) :
  1. Charger les CSV bruts (DATA_YEARS, config.py)
  2. Découvrir les ligues disponibles (log informatif)
  3. Filtrer les ligues ciblées (LFL, PRM, LVP SL, NLC, TCL + Div 2 + LEC)
  4. Filtrer les lignes "joueur" (exclure les lignes "team" agrégées)
  5. Sélectionner les colonnes pertinentes
  6. Normaliser les noms de joueurs (lowercase + strip ; casse d'origine
     conservée dans playername_original pour l'affichage)
  7. Gérer les valeurs manquantes (médiane par ligue × année)
  8. Construire la target variable DATÉE "promoted_to_lec" depuis les dates
     Oracle's Elixir (Leaguepedia = cross-check informatif seulement)
  Puis : validation du dataset et sauvegarde dans data/interim/

Pourquoi 80% du temps est passé ici ?
  Les données réelles sont TOUJOURS sales. Oracle's Elixir est bien structuré,
  mais on rencontre quand même des problèmes :
  - Noms de ligues incohérents (ex: "LFL" vs "La Ligue Française")
  - Noms de joueurs avec des espaces en trop, des majuscules incohérentes
  - Valeurs manquantes dans certaines colonnes (surtout les stats early game)
  - Lignes "team" mélangées avec les lignes "joueur" (12 lignes par match :
    10 joueurs + 2 lignes team)

Usage :
  make clean
  # ou
  python -m src.data.cleaner
"""


import numpy as np
import pandas as pd

from src.config import (
    ALL_TARGET_LEAGUES,
    DATA_YEARS,
    ERL_DIV1_LEAGUES,
    ERL_DIV2_LEAGUES,
    ERL_LEAGUES,
    INTERIM_DATA_DIR,
    KEY_COLUMNS,
    PROMOTION_HORIZON_MONTHS,
    RAW_DATA_DIR,
    TOP_LEAGUE,
)
from src.data.leaguepedia import load_career_data
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des données brutes
# ═══════════════════════════════════════════════════════════════════════════════


def load_raw_data(years: list[int] | None = None) -> pd.DataFrame:
    """
    Charge et concatène les CSV Oracle's Elixir pour les années spécifiées.

    Les CSV Oracle's Elixir contiennent 12 lignes par match :
      - 10 lignes "joueur" (5 par équipe, une par position)
      - 2 lignes "team" (agrégation des stats de l'équipe)

    On les identifie grâce à la colonne "position" :
      - joueur : "top", "jng", "mid", "bot", "sup"
      - team : "team"

    Args:
        years: Années à charger. Par défaut : DATA_YEARS de config.py

    Returns:
        DataFrame concaténé avec toutes les années
    """
    if years is None:
        years = DATA_YEARS

    dfs = []
    for year in years:
        filename = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        filepath = RAW_DATA_DIR / filename

        if not filepath.exists():
            logger.warning(
                f"⚠️  Fichier {filename} introuvable dans {RAW_DATA_DIR}. "
                f"Avez-vous exécuté 'make data' ? Année {year} ignorée."
            )
            continue

        logger.info(f"📂 Chargement de {filename}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["_source_year"] = year  # Traçabilité : d'où vient chaque ligne
            dfs.append(df)
            logger.info(f"   → {len(df):,} lignes, {len(df.columns)} colonnes")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de {filename} : {e}")
            continue

    if not dfs:
        logger.error(
            "❌ Aucun fichier CSV chargé. Vérifiez que les données sont "
            "téléchargées dans data/raw/"
        )
        return pd.DataFrame()

    # Concaténation de toutes les années
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"📊 Total chargé : {len(df):,} lignes × {len(df.columns)} colonnes")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Découverte des ligues disponibles
# ═══════════════════════════════════════════════════════════════════════════════


def discover_leagues(df: pd.DataFrame) -> dict[str, int]:
    """
    Affiche toutes les ligues présentes dans les données.

    C'est une étape exploratoire cruciale : les noms de ligues dans les CSV
    Oracle's Elixir ne sont pas toujours ce qu'on attend. Par exemple :
      - La LFL peut être "LFL" ou "La Ligue Française"
      - La TCL peut être "TCL" ou "TUR"

    Cette fonction permet de vérifier et corriger les noms dans config.py.

    Args:
        df: DataFrame brut

    Returns:
        Dict {nom_ligue: nombre_de_lignes}
    """
    league_counts = df["league"].value_counts().to_dict()

    logger.info(f"\n{'='*60}")
    logger.info("🌍 LIGUES DISPONIBLES DANS LES DONNÉES")
    logger.info(f"{'='*60}")
    for league, count in sorted(league_counts.items(), key=lambda x: -x[1]):
        # Marquer les ligues qu'on cible
        marker = ""
        if league in ERL_DIV1_LEAGUES:
            marker = " ← ERL Div 1 ✅"
        elif league in ERL_DIV2_LEAGUES:
            marker = " ← ERL Div 2 ✅"
        elif league == TOP_LEAGUE:
            marker = " ← TOP (target) 🎯"
        logger.info(f"   {league:30s} : {count:>8,} lignes{marker}")
    logger.info(f"{'='*60}")

    return league_counts


# ═══════════════════════════════════════════════════════════════════════════════
# Filtrage et nettoyage
# ═══════════════════════════════════════════════════════════════════════════════


def filter_target_leagues(
    df: pd.DataFrame,
    target_leagues: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filtre le DataFrame pour ne garder que les ligues ciblées.

    Deux cas de figure :
      1. Le nom de la ligue est exact → match direct
      2. Le nom est légèrement différent → on fait un match partiel (case-insensitive)

    Args:
        df: DataFrame brut
        target_leagues: Ligues à garder. Par défaut : ALL_TARGET_LEAGUES

    Returns:
        DataFrame filtré
    """
    if target_leagues is None:
        target_leagues = ALL_TARGET_LEAGUES

    # Étape 1 : essayer un match exact
    mask = df["league"].isin(target_leagues)
    matched_leagues = df.loc[mask, "league"].unique().tolist()
    unmatched_targets = [lg for lg in target_leagues if lg not in matched_leagues]

    if unmatched_targets:
        logger.warning(
            f"⚠️  Ligues non trouvées en match exact : {unmatched_targets}. "
            f"Tentative de match partiel (case-insensitive)..."
        )

        # Étape 2 : match partiel case-insensitive pour les ligues non trouvées
        all_leagues = df["league"].unique()
        for target in unmatched_targets:
            for league in all_leagues:
                if target.lower() in league.lower() or league.lower() in target.lower():
                    logger.info(f"   Match partiel trouvé : '{target}' → '{league}'")
                    mask = mask | (df["league"] == league)
                    break

    df_filtered = df[mask].copy()

    logger.info(
        f"🔍 Filtrage ligues : {len(df):,} → {len(df_filtered):,} lignes "
        f"({len(df_filtered) / len(df) * 100:.1f}%)"
    )
    logger.info(f"   Ligues retenues : {sorted(df_filtered['league'].unique())}")

    return df_filtered


def filter_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre pour ne garder que les lignes "joueur" (exclure les lignes "team").

    Oracle's Elixir contient 12 lignes par match :
      - 10 lignes joueur (position = top, jng, mid, bot, sup)
      - 2 lignes team (position = team)

    On ne garde que les lignes joueur car on fait du scouting individuel.

    Args:
        df: DataFrame

    Returns:
        DataFrame sans les lignes "team"
    """
    valid_positions = ["top", "jng", "mid", "bot", "sup"]

    if "position" not in df.columns:
        logger.warning("Colonne 'position' absente — impossible de filtrer les lignes team")
        return df

    df_players = df[df["position"].str.lower().isin(valid_positions)].copy()

    removed = len(df) - len(df_players)
    logger.info(
        f"👤 Filtrage joueurs : {len(df):,} → {len(df_players):,} lignes "
        f"({removed:,} lignes 'team' supprimées)"
    )

    return df_players


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les colonnes pertinentes pour le pipeline ML.

    On ne garde que les colonnes définies dans KEY_COLUMNS (config.py)
    plus la colonne de traçabilité _source_year.

    Pourquoi sélectionner maintenant ?
      - Réduire la taille du DataFrame (de ~100 colonnes à ~35)
      - Éviter de traîner des colonnes inutiles dans tout le pipeline
      - Documenter explicitement quelles données on utilise

    Args:
        df: DataFrame

    Returns:
        DataFrame avec les colonnes sélectionnées
    """
    # Colonnes présentes dans les données ET dans notre liste
    available_cols = [col for col in KEY_COLUMNS if col in df.columns]
    missing_cols = [col for col in KEY_COLUMNS if col not in df.columns]

    if missing_cols:
        logger.warning(
            f"⚠️  {len(missing_cols)} colonnes attendues absentes des données : "
            f"{missing_cols}"
        )

    # Ajouter la colonne de traçabilité si elle existe
    extra_cols = ["_source_year"]
    available_cols += [c for c in extra_cols if c in df.columns]

    df_selected = df[available_cols].copy()
    logger.info(
        f"📋 Sélection colonnes : {len(df.columns)} → {len(available_cols)} colonnes"
    )

    return df_selected


def normalize_player_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de joueurs pour la cohérence.

    Problèmes fréquents dans les données :
      - Espaces en trop : "  Caps  " → "Caps"
      - Casse incohérente : "caps" vs "Caps" vs "CAPS"
      - Caractères spéciaux ou Unicode

    On normalise en lowercase + strip pour faciliter les jointures
    avec Leaguepedia.

    Note : on garde le nom original dans une colonne séparée pour
    l'affichage final.

    Args:
        df: DataFrame

    Returns:
        DataFrame avec noms normalisés
    """
    if "playername" not in df.columns:
        return df

    # Sauvegarder le nom original pour l'affichage
    df["playername_original"] = df["playername"]

    # Normalisation : strip + lowercase
    df["playername"] = (
        df["playername"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    n_unique = df["playername"].nunique()
    logger.info(f"📝 Noms normalisés : {n_unique:,} joueurs uniques")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le dataset.

    Stratégie documentée :
      - Colonnes d'identité (playername, teamname, league) : drop si NaN
        (une ligne sans joueur n'a pas de sens)
      - Colonnes numériques (stats) : remplir avec la médiane de la
        ligue × année (pas la moyenne — plus robuste aux outliers)
      - Colonnes catégorielles : remplir avec "unknown"

    Pourquoi la médiane par ligue ET par année ?
      - Par ligue : le niveau de jeu varie entre ligues (un CS/min moyen en
        LFL n'est pas le même qu'en LFL2).
      - Par année : le split train/test est temporel (2024 vs 2025+). Une
        médiane calculée toutes années confondues ferait fuir de
        l'information du test set vers les lignes de train via les valeurs
        imputées. En groupant par (league, _source_year), chaque année
        n'utilise que ses propres statistiques.

    Args:
        df: DataFrame

    Returns:
        DataFrame sans NaN critiques
    """
    initial_count = len(df)
    initial_nan = df.isna().sum().sum()

    # 1. Drop les lignes sans identité (joueur, équipe, ligue)
    critical_cols = ["playername", "teamname", "league"]
    existing_critical = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=existing_critical)

    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(
            f"⚠️  {dropped} lignes supprimées (NaN dans colonnes critiques : "
            f"{existing_critical})"
        )

    # 2. Remplir les stats numériques avec la médiane par ligue
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclure les colonnes non-stats
    exclude = ["_source_year", "result"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # Grouper par année en plus de la ligue quand la traçabilité existe
    # (anti-fuite : voir docstring)
    impute_keys = ["league", "_source_year"] if "_source_year" in df.columns else ["league"]

    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            # Imputation par médiane de la ligue × année
            df[col] = df.groupby(impute_keys)[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Si encore des NaN (groupe entier sans données), médiane globale
            remaining_nan = df[col].isna().sum()
            if remaining_nan > 0:
                df[col] = df[col].fillna(df[col].median())

    # 3. Colonnes catégorielles : remplir avec "unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    final_nan = df.isna().sum().sum()
    logger.info(
        f"🩹 Valeurs manquantes : {initial_nan:,} → {final_nan:,} NaN restants"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Target Variable
# ═══════════════════════════════════════════════════════════════════════════════


def compute_lec_debut_dates(df: pd.DataFrame) -> dict[str, "pd.Timestamp"]:
    """
    Calcule la date du PREMIER match LEC de chaque joueur (= sa "promotion").

    C'est le pivot temporel de la target datée : on considère qu'un joueur est
    "promu" à la date de son premier match LEC dans les données. Tout match ERL
    antérieur à cette date (et dans l'horizon) devient un exemple positif ;
    tout match ERL postérieur (ex-LEC relégué) reste négatif.

    Args:
        df: DataFrame complet (incluant les lignes LEC), avec une colonne `date`.

    Returns:
        Dict {playername: Timestamp du premier match LEC}. Vide si aucune donnée
        LEC datable n'est disponible.
    """
    if "date" not in df.columns:
        logger.warning("⚠️  Colonne 'date' absente — impossible de dater les promotions")
        return {}

    lec = df[df["league"] == TOP_LEAGUE].copy()
    if lec.empty:
        return {}

    lec["_date_parsed"] = pd.to_datetime(lec["date"], errors="coerce")
    lec = lec.dropna(subset=["_date_parsed"])
    if lec.empty:
        logger.warning("⚠️  Aucune date LEC exploitable après parsing")
        return {}

    debut_dates = lec.groupby("playername")["_date_parsed"].min().to_dict()
    logger.info(f"📅 Dates de début LEC calculées pour {len(debut_dates)} joueurs")
    return debut_dates


def build_dated_target_from_oracle(
    df: pd.DataFrame, horizon_months: int = PROMOTION_HORIZON_MONTHS
) -> pd.Series:
    """
    Construit une target *datée* "promoted_to_lec" depuis Oracle's Elixir.

    Un match ERL est positif si, et seulement si, le joueur débute en LEC
    STRICTEMENT APRÈS ce match, et dans un horizon de `horizon_months` mois.

    Pourquoi dater la target ? L'ancienne logique ("a joué en ERL et en LEC
    un jour → promu") introduisait deux fuites temporelles :
      1. Un ex-joueur LEC relégué en ERL était étiqueté "pépite" alors que sa
         promotion est ANTÉRIEURE aux stats observées.
      2. Sur le test set, on "prédisait" des promotions déjà survenues.
    En exigeant que le début LEC soit dans le futur du match observé, ces deux
    cas sont corrigés : le signal appris est bien "ce joueur VA être promu".

    Args:
        df: DataFrame complet (lignes ERL + LEC), avec une colonne `date`.
        horizon_months: Fenêtre max (en mois) entre le match et le début LEC.

    Returns:
        pd.Series booléenne alignée sur df.index (True = match ERL pré-promotion).
    """
    erl_mask = df["league"].isin(ERL_LEAGUES)

    debut_dates = compute_lec_debut_dates(df)
    if not debut_dates:
        logger.warning("⚠️  Target datée impossible (pas de dates LEC) → tout à False")
        return pd.Series(False, index=df.index)

    row_dates = pd.to_datetime(df["date"], errors="coerce")
    debut_for_row = df["playername"].map(debut_dates)  # NaT si jamais promu
    horizon_end = row_dates + pd.DateOffset(months=horizon_months)

    # Positif ssi : ligne ERL ET début LEC connu ET début dans (match, match+horizon]
    is_future_promotion = (
        erl_mask
        & debut_for_row.notna()
        & row_dates.notna()
        & (debut_for_row > row_dates)
        & (debut_for_row <= horizon_end)
    )
    target = is_future_promotion.fillna(False).astype(bool)

    n_pos_rows = int(target.sum())
    n_pos_players = df.loc[target, "playername"].nunique()
    n_erl_players = df.loc[erl_mask, "playername"].nunique()
    logger.info(
        f"🎯 Target datée (horizon {horizon_months} mois) : {n_pos_rows:,} matchs positifs, "
        f"{n_pos_players} joueurs pré-promotion sur {n_erl_players} joueurs ERL"
    )
    return target


def add_target_variable(
    df: pd.DataFrame, horizon_months: int = PROMOTION_HORIZON_MONTHS
) -> pd.DataFrame:
    """
    Ajoute la colonne target *datée* "promoted_to_lec" au DataFrame.

    La target est construite depuis les dates d'Oracle's Elixir (toujours
    disponibles) : un match ERL est positif uniquement si le joueur débute en
    LEC dans les `horizon_months` mois QUI SUIVENT le match. Voir
    `build_dated_target_from_oracle` pour le détail et la justification.

    Leaguepedia n'est utilisé qu'en cross-check informatif : n'exposant pas de
    dates de transfert exploitables ici, il ne peut pas dater les promotions.
    L'intégrer comme label ré-introduirait la fuite temporelle qu'on corrige,
    on le limite donc à un rapport de couverture (best effort, souvent absent).

    Args:
        df: DataFrame nettoyé (lignes ERL + LEC), avec une colonne `date`.
        horizon_months: Fenêtre max (en mois) entre le match et le début LEC.

    Returns:
        DataFrame avec la colonne booléenne "promoted_to_lec".
    """
    # Target datée depuis Oracle's Elixir
    df["promoted_to_lec"] = build_dated_target_from_oracle(df, horizon_months)

    # Cross-check Leaguepedia (optionnel, informatif seulement)
    leaguepedia_data = load_career_data()
    if leaguepedia_data is not None:
        _, leaguepedia_promotions = leaguepedia_data
        lp_players = {
            p.lower().strip() for p, v in leaguepedia_promotions.items() if v
        }
        if lp_players:
            dated_players = set(df.loc[df["promoted_to_lec"], "playername"].unique())
            confirmed = lp_players & dated_players
            undatable = lp_players - dated_players
            logger.info(
                f"📚 Cross-check Leaguepedia : {len(lp_players)} promus déclarés, "
                f"{len(confirmed)} confirmés par la target datée, "
                f"{len(undatable)} non datables (hors fenêtre ou hors données Oracle)"
            )

    # Stats
    erl_mask = df["league"].isin(ERL_LEAGUES)
    promoted_rows = int(df.loc[erl_mask, "promoted_to_lec"].sum())
    total_rows = int(erl_mask.sum())
    logger.info(
        f"📊 Target datée ajoutée : {promoted_rows:,} lignes 'promoted' "
        f"sur {total_rows:,} lignes ERL"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Vérifie la qualité du dataset nettoyé avec des assertions.

    Ces vérifications attrapent les bugs tôt dans le pipeline plutôt qu'au
    moment de l'entraînement du modèle (où c'est beaucoup plus dur à debugger).

    Args:
        df: DataFrame nettoyé

    Returns:
        True si toutes les validations passent
    """
    logger.info(f"\n{'='*60}")
    logger.info("🔍 VALIDATION DU DATASET")
    logger.info(f"{'='*60}")

    all_ok = True

    # 1. Pas de lignes vides
    if len(df) == 0:
        logger.error("❌ Le DataFrame est vide !")
        return False
    logger.info(f"   ✅ {len(df):,} lignes présentes")

    # 2. Joueurs uniques
    n_players = df["playername"].nunique()
    if n_players < 10:
        logger.warning(f"   ⚠️  Seulement {n_players} joueurs uniques — très peu")
        all_ok = False
    else:
        logger.info(f"   ✅ {n_players:,} joueurs uniques")

    # 3. Ligues attendues
    leagues = set(df["league"].unique())
    erl_present = [lg for lg in ERL_DIV1_LEAGUES if lg in leagues]
    logger.info(f"   ✅ ERLs Div 1 présentes : {erl_present}")

    if TOP_LEAGUE in leagues:
        logger.info(f"   ✅ {TOP_LEAGUE} présent (pour la target variable)")
    else:
        logger.warning(f"   ⚠️  {TOP_LEAGUE} absent — target variable impactée")

    # 4. Positions
    if "position" in df.columns:
        positions = df["position"].unique()
        logger.info(f"   ✅ Positions : {sorted(positions)}")

    # 5. Target variable
    if "promoted_to_lec" in df.columns:
        n_promoted = df["promoted_to_lec"].sum()
        logger.info(
            f"   ✅ Target : {int(n_promoted):,} lignes promoted "
            f"({n_promoted / len(df) * 100:.2f}% du dataset)"
        )
    else:
        logger.warning("   ⚠️  Colonne 'promoted_to_lec' absente")
        all_ok = False

    # 6. Pas de NaN dans les colonnes critiques
    critical = ["playername", "teamname", "league", "position"]
    for col in critical:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"   ⚠️  {nan_count} NaN dans '{col}'")
                all_ok = False

    logger.info(f"{'='*60}")
    if all_ok:
        logger.success("✅ Toutes les validations passent !")
    else:
        logger.warning("⚠️  Certaines validations ont échoué — vérifiez les warnings ci-dessus")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline complet
# ═══════════════════════════════════════════════════════════════════════════════


def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Exécute le pipeline complet de nettoyage.

    Étapes (dans l'ordre — voir la docstring du module) :
      1-8. Chargement → découverte des ligues → filtrages → sélection de
           colonnes → normalisation des noms → valeurs manquantes → target datée
      Puis : validation du dataset et sauvegarde dans data/interim/

    Returns:
        DataFrame nettoyé
    """
    logger.info(f"\n{'='*60}")
    logger.info("🧹 PIPELINE DE NETTOYAGE — Démarrage")
    logger.info(f"{'='*60}\n")

    # 1. Charger les données brutes
    logger.info("📂 ÉTAPE 1/8 — Chargement des données brutes")
    df = load_raw_data()
    if df.empty:
        return df

    # 2. Découvrir les ligues disponibles
    logger.info("\n🌍 ÉTAPE 2/8 — Découverte des ligues disponibles")
    discover_leagues(df)

    # 3. Filtrer les ligues ciblées
    logger.info("\n🔍 ÉTAPE 3/8 — Filtrage des ligues ciblées")
    df = filter_target_leagues(df)

    # 4. Filtrer les lignes joueur
    logger.info("\n👤 ÉTAPE 4/8 — Filtrage des lignes joueur")
    df = filter_player_rows(df)

    # 5. Sélectionner les colonnes
    logger.info("\n📋 ÉTAPE 5/8 — Sélection des colonnes")
    df = select_columns(df)

    # 6. Normaliser les noms
    logger.info("\n📝 ÉTAPE 6/8 — Normalisation des noms de joueurs")
    df = normalize_player_names(df)

    # 7. Gérer les valeurs manquantes
    logger.info("\n🩹 ÉTAPE 7/8 — Gestion des valeurs manquantes")
    df = handle_missing_values(df)

    # 8. Ajouter la target variable
    logger.info("\n🎯 ÉTAPE 8/8 — Construction de la target variable")
    df = add_target_variable(df)

    # Validation
    validate_dataset(df)

    # Sauvegarder
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / "cleaned_matches.csv"
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.success(f"\n💾 Dataset nettoyé sauvegardé → {output_path} ({size_mb:.1f} Mo)")

    # Résumé final
    logger.info(f"\n{'='*60}")
    logger.info("📊 RÉSUMÉ DU DATASET NETTOYÉ")
    logger.info(f"{'='*60}")
    logger.info(f"   Lignes     : {len(df):,}")
    logger.info(f"   Colonnes   : {len(df.columns)}")
    logger.info(f"   Joueurs    : {df['playername'].nunique():,}")
    logger.info(f"   Ligues     : {sorted(df['league'].unique())}")
    logger.info(f"   Années     : {sorted(df['_source_year'].unique()) if '_source_year' in df.columns else 'N/A'}")
    logger.info(f"{'='*60}\n")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI : python -m src.data.cleaner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_cleaning_pipeline()
