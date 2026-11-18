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

import os

import streamlit as st
from utils.data_loader import load_refresh_metadata, load_talent_scores

from utils import demo_agent, theme

st.set_page_config(
    page_title="ERL Scout",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.appliquer()

# L'API Cloud Run s'endort sans trafic : on la réveille dès l'arrivée sur
# l'accueil, pour qu'elle soit prête si le visiteur passe sur la page Agent.
if "api_reveillee" not in st.session_state:
    st.session_state.api_reveillee = True
    demo_agent.reveiller_api(os.getenv("SCOUTING_API_URL", demo_agent.API_PUBLIQUE))

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
    "Ton cinq de départ",
    "Le meilleur choix à chaque poste, et ses remplaçants. Le potentiel LEC dit à "
    "quel point les stats d'un joueur ressemblent à celles des joueurs qui ont été "
    "recrutés en LEC. C'est un classement, pas une note : 90 n'est pas deux fois mieux que 45.",
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

    par_poste = st.slider("Joueurs par poste", min_value=1, max_value=8, value=4, step=1)

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

def _z(joueur, colonne: str) -> float | None:
    v = joueur.get(colonne)
    return None if v is None or v != v else float(v)


def _fiche(joueur) -> dict:
    return {
        "nom": str(joueur.get("playername_original") or joueur["playername"]),
        "club": f"{joueur['teamname']}, {joueur['league']}, {int(joueur['games_played'])} matchs",
        "score": float(joueur["talent_score"]),
        "z_or": _z(joueur, "golddiffat15_zscore"),
        "z_degats": _z(joueur, "dpm_zscore"),
        "z_farm": _z(joueur, "cspm_zscore"),
        "promu": bool(joueur.get("promoted_to_lec", False)),
    }


# Un joueur a une ligne par split : on garde sa meilleure, sinon il occupe
# lui-même les places de remplaçant.
uniques = filtre.sort_values("talent_score", ascending=False).drop_duplicates(["playername", "position"])

cartes = []
for code, libelle in theme.POSTES:
    bloc = uniques[uniques["position"] == code]
    if bloc.empty:
        continue
    meilleurs = [_fiche(j) for _, j in bloc.head(par_poste).iterrows()]
    cartes.append(theme.carte_poste(libelle, meilleurs[0], meilleurs[1:]))

st.markdown(
    f'<div class="dr-row">{"".join(cartes)}</div>'
    '<div class="dr-key"><span>Trait vertical des jauges : la moyenne des joueurs '
    "du même poste dans sa ligue</span><span><b>Bleu à droite</b> : mieux que la "
    "moyenne</span><span>Au-dessus de 95, l'écart entre deux joueurs est faible : "
    "le score suit aussi les résultats de l'équipe</span></div>",
    unsafe_allow_html=True,
)

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
