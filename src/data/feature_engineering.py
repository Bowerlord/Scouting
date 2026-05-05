"""
feature_engineering.py — Transformation des données pour le Machine Learning

Ce module correspond à la Phase 4 du projet KCorp Scouting Tool. 
Son objectif est de transformer les statistiques de matchs brutes (une ligne par match) 
en un ensemble de "features" (caractéristiques) par joueur et par segment (split), 
prêtes à être ingérées par un algorithme d'apprentissage automatique.

Pourquoi faisons-nous cela ?
  1. Le scouting s'évalue sur le long terme : On n'évalue pas un joueur sur 1 match, 
     mais sur ses performances lissées au cours d'une saison/split.
  2. Le problème des ligues hétérogènes : Faire 500 DPM en LFL2 n'a pas la même 
     valeur qu'en LEC. Il faut normaliser les statistiques.
  3. Le problème des postes : Un support aura toujours moins de CS qu'un ADC. 
     Il faut comparer les joueurs uniquement à leurs pairs au même poste.

Étapes du pipeline :
  1. Chargement des données nettoyées (issues de cleaner.py).
  2. Recalcul de la Kill Participation (KP), absente des données brutes récentes.
  3. Agrégation : Regroupement par [Joueur, Ligue, Année, Split, Position] pour 
     obtenir les moyennes (DPM, CSPM, etc.) et des métadonnées (Champion Pool).
  4. Création des Z-scores : Calcul de la performance relative (écart-type à la moyenne)
     en isolant chaque sous-groupe (ex: Toplaners de LFL au Spring 2024).
  5. Sauvegarde du fichier `features_players.csv`.

Usage :
  make features
  # ou
  python -m src.data.feature_engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et Préparation Initiale
# ═══════════════════════════════════════════════════════════════════════════════

def load_cleaned_data() -> pd.DataFrame:
    """
    Charge le dataset nettoyé issu de la Phase 2/3.
    Gère également la conversion des types booléens qui pourraient avoir été 
    sauvegardés comme chaînes de caractères ('True'/'False') dans le CSV.
    """
    data_path = Path("data/interim/cleaned_matches.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ {data_path} n'existe pas. Veuillez exécuter `make clean` d'abord."
        )
    
    df = pd.read_csv(data_path)
    
    # Correction des types : pandas lit souvent les booléens de CSV comme des objets/strings
    if 'promoted_to_lec' in df.columns:
        df['promoted_to_lec'] = df['promoted_to_lec'].astype(str).str.lower() == 'true'
        
    return df


def calculate_kill_participation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule la Kill Participation (KP), une métrique fondamentale en scouting,
    qui n'est plus fournie dans les exports récents d'Oracle's Elixir.
    
    Formule : KP = (Kills du joueur + Assists du joueur) / Total des Kills de l'équipe
    """
    logger.info("🔧 Calcul de la Kill Participation...")
    
    # Étape 1 : Calculer le total des kills pour chaque équipe dans chaque match
    team_kills = df.groupby(['gameid', 'teamname'])['kills'].sum().reset_index()
    team_kills = team_kills.rename(columns={'kills': 'team_kills'})
    
    # Étape 2 : Réintégrer ce total sur chaque ligne joueur
    df = pd.merge(df, team_kills, on=['gameid', 'teamname'], how='left')
    
    # Étape 3 : Calculer le ratio, en évitant les divisions par zéro
    # (Un match parfait où l'équipe fait 0 kill donnerait une division par 0)
    df['killparticipation'] = np.where(
        df['team_kills'] > 0,
        (df['kills'] + df['assists']) / df['team_kills'],
        0.0
    )
    
    # Sécurité : borner la valeur entre 0 et 1 (100%)
    df['killparticipation'] = df['killparticipation'].clip(0, 1.0) 
    
    # Nettoyage : retirer la colonne temporaire
    df = df.drop(columns=['team_kills'])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Agrégation par Segment (Split)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Condense les données de matchs en données "par joueur et par split".
    C'est ici qu'on crée l'unité d'analyse pour notre modèle ML. 
    Une ligne = Les performances d'un joueur X lors du split Y de l'année Z.
    """
    # L'index unique de chaque performance :
    group_cols = ['playername', 'league', '_source_year', 'split', 'position', 'teamname']
    
    # Les statistiques brutes que nous voulons moyenner sur tout le split
    mean_cols = [
        'kills', 'deaths', 'assists', 'killparticipation', 
        'dpm', 'cspm', 'vspm', 
        'golddiffat15', 'csdiffat15', 'xpdiffat15', 
        'damageshare', 'earnedgoldshare'
    ]
    
    # Sécurité : ne garder que les colonnes qui existent réellement
    mean_cols = [c for c in mean_cols if c in df.columns]
    
    # Dictionnaire d'agrégation
    agg_dict = {
        'gameid': 'count',                 # Compte le nombre de matchs joués
        'result': 'mean',                  # La moyenne des résultats (0 ou 1) = le Win Rate
        'champion': 'nunique',             # Nombre de champions uniques = Taille du Champion Pool
        'promoted_to_lec': 'max',          # Le joueur a-t-il été promu suite à ces performances ?
    }
    
    # Ajouter la moyenne pour toutes les statistiques de jeu
    for col in mean_cols:
        agg_dict[col] = 'mean'
        
    logger.info(f"📊 Agrégation par joueur/split (base: {len(df):,} lignes de matchs)...")
    player_stats = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Renommer pour la clarté
    player_stats = player_stats.rename(columns={
        'gameid': 'games_played',
        'result': 'win_rate',
        'champion': 'champion_pool_size'
    })
    
    # ── Filtrage Anti-Bruit ──
    # Les joueurs ayant joué très peu de matchs (ex: remplaçants d'un soir) ont 
    # des statistiques peu représentatives. On fixe un seuil de 5 matchs minimum
    # (environ l'équivalent d'une semaine complète de compétition).
    min_games = 5
    initial_players = len(player_stats)
    player_stats = player_stats[player_stats['games_played'] >= min_games]
    filtered_out = initial_players - len(player_stats)
    
    logger.info(
        f"🧹 Filtre de représentativité (< {min_games} matchs) : "
        f"{filtered_out} entrées exclues (Reste: {len(player_stats)})."
    )
    
    return player_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Création des Features Relatives (Z-Scores)
# ═══════════════════════════════════════════════════════════════════════════════

def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les Z-scores relatifs.
    C'est le secret d'un bon algorithme de scouting esport.
    
    Le Z-score indique à quel point une valeur s'éloigne de la moyenne de son groupe,
    mesuré en "écarts-types".
      Z = (Valeur - Moyenne_du_groupe) / Ecart_type_du_groupe
      
    Un DPM_zscore de +2.0 signifie "Ce joueur fait des dégâts exceptionnels par 
    rapport aux autres joueurs évoluant EXACTEMENT dans la même ligue, la même 
    année, au même split, et à la même position".
    """
    logger.info("📐 Calcul des Z-scores (intra-ligue/position/split)...")
    
    # Sélection des features intéressantes à standardiser
    base_features = [
        'killparticipation', 'dpm', 'cspm', 'vspm', 
        'golddiffat15', 'csdiffat15', 'xpdiffat15',
        'win_rate', 'champion_pool_size'
    ]
    base_features = [c for c in base_features if c in df.columns]
    
    # Définition de "l'environnement immédiat" du joueur (son groupe de comparaison)
    grouping = df.groupby(['league', '_source_year', 'split', 'position'])
    
    for feature in base_features:
        z_col = f"{feature}_zscore"
        
        # transform() applique la fonction à chaque sous-groupe indépendamment.
        # Nous ajoutons une protection contre la division par zéro dans le cas rare 
        # où l'écart type d'un groupe serait nul (ex: tous les joueurs ont exactement 
        # la même stat, ou le groupe est trop petit).
        df[z_col] = grouping[feature].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Remplacer les valeurs NaN (si la variance est NaN) par 0 (qui représente la moyenne)
        df[z_col] = df[z_col].fillna(0)
        
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Orchestrateur Principal
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering_pipeline():
    """Point d'entrée du module, exécute toutes les étapes dans l'ordre."""
    logger.info("============================================================")
    logger.info("⚙️  KCORP SCOUTING PIPELINE — PHASE 4 : FEATURE ENGINEERING")
    logger.info("============================================================")
    
    # Étape 1
    df = load_cleaned_data()
    
    # Étape 2
    df = calculate_kill_participation(df)
    
    # Étape 3
    player_df = aggregate_player_stats(df)
    
    # Étape 4
    featured_df = add_zscores(player_df)
    
    # Sauvegarde
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "features_players.csv"
    
    featured_df.to_csv(out_file, index=False)
    
    logger.info("============================================================")
    logger.info("✅ FEATURE ENGINEERING TERMINÉ")
    logger.info(f"   Lignes (Joueurs/Split) : {len(featured_df):,}")
    logger.info(f"   Colonnes générées      : {len(featured_df.columns)}")
    logger.info(f"   Fichier sauvegardé     : {out_file}")
    logger.info("============================================================")

if __name__ == "__main__":
    run_feature_engineering_pipeline()
