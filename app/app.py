"""
app.py — La liste de surveillance d'ERL Scout.

Refondue le 2026-09-07. La version précédente présentait l'outil : un titre,
quatre métriques sur le jeu de données et trois colonnes de texte décrivant les
pages. Un scout qui ouvre l'application veut voir des joueurs, pas lire ce que
fait l'application.

La structure est donc celle du métier, pas celle du logiciel : cinq blocs, un
par poste. Personne ne cherche « un joueur », on cherche un jungler.

Lancement :
  streamlit run app/app.py        (depuis la racine du projet)
"""

import sys
from pathlib import Path

# Ajouter app/ au sys.path pour que 'from utils.data_loader import ...' fonctionne
# quelle que soit la façon dont Streamlit est lancé
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.data_loader import load_refresh_metadata, load_talent_scores

from utils import theme

st.set_page_config(
    page_title="ERL Scout",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.appliquer()

# ══════════════════════════════════════════════════════════════════════════════
# Données
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_talent_scores()
except FileNotFoundError as erreur:
    st.error(str(erreur))
    st.info("Lancez le pipeline pour générer les résultats, puis relancez l'application.")
    st.stop()

meta = load_refresh_metadata()
saisons = sorted(df["_source_year"].dropna().unique().astype(int).tolist())

theme.entete(
    f"Ligues régionales européennes · {saisons[0]}\u2013{saisons[-1]}",
    "Liste de surveillance",
    "Les joueurs dont le profil se rapproche le plus de ceux qui sont réellement "
    "montés en LEC. Le score est une position relative, pas une note : un joueur "
    "à 90 n'est pas deux fois meilleur qu'un joueur à 45.",
)

# ══════════════════════════════════════════════════════════════════════════════
# Filtres
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="es-kicker">Filtres</div>', unsafe_allow_html=True)

    saison = st.selectbox(
        "Saison",
        options=["Toutes"] + [str(a) for a in reversed(saisons)],
        index=1 if len(saisons) > 1 else 0,
    )

    ligues = sorted(df["league"].dropna().unique().tolist())
    ligue = st.multiselect("Ligue", options=ligues, default=ligues)

    # Le seuil par defaut n'est pas cosmetique : en dessous de dix matchs, la
    # tete de liste est occupee par des joueurs a trois parties dont le score
    # n'a aucune valeur predictive.
    min_matchs = st.slider("Matchs minimum", min_value=0, max_value=40, value=10, step=1)

    par_poste = st.slider("Joueurs par poste", min_value=3, max_value=15, value=6, step=1)

filtre = df.copy()
if saison != "Toutes":
    filtre = filtre[filtre["_source_year"] == int(saison)]
if ligue:
    filtre = filtre[filtre["league"].isin(ligue)]
filtre = filtre[filtre["games_played"] >= min_matchs]

# ══════════════════════════════════════════════════════════════════════════════
# La liste, poste par poste
# ══════════════════════════════════════════════════════════════════════════════

if filtre.empty:
    st.warning("Aucun joueur ne correspond à ces filtres. Élargissez la saison ou baissez le seuil de matchs.")
    st.stop()

st.write("")
st.markdown(
    '<div class="es-legende">Écart au niveau moyen de la ligue · '
    "différentiel d'or à 15 minutes</div>",
    unsafe_allow_html=True,
)

for code, libelle in theme.POSTES:
    bloc = filtre[filtre["position"] == code]
    if bloc.empty:
        continue

    meilleurs = bloc.nlargest(par_poste, "talent_score")
    lignes = [theme.entete_poste(libelle, len(bloc))]

    for rang, (_, joueur) in enumerate(meilleurs.iterrows(), start=1):
        z = joueur.get("golddiffat15_zscore")
        lignes.append(
            theme.ligne_joueur(
                rang=rang,
                nom=str(joueur.get("playername_original") or joueur["playername"]),
                club=f"{joueur['teamname']} · {joueur['league']}",
                score=float(joueur["talent_score"]),
                z=None if z is None or z != z else float(z),
                promu=bool(joueur.get("promoted_to_lec", False)),
            )
        )

    st.markdown("".join(lignes), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Pied de page
# ══════════════════════════════════════════════════════════════════════════════

st.write("")# `data_max_date` est la date du dernier match du jeu de donnees, et non celle
# de generation du fichier : c'est elle qui dit jusqu'ou vont les resultats,
# donc la seule qui interesse quelqu'un qui lit un classement.
derniere = meta.get("data_max_date") if meta else None
pied = f"Oracle's Elixir · {len(filtre):,} joueurs-saisons affiches"
if derniere:
    pied += f" · donnees jusqu'au {derniere}"

st.markdown(f'<div class="es-legende">{pied}</div>', unsafe_allow_html=True)
