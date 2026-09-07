"""Système visuel d'ERL Scout — direction « liste de surveillance ».

Refondu le 2026-09-07. Le jet précédent changeait la palette et la police sans
toucher à la structure : un gabarit repeint reste un gabarit.

Ce qui gouverne les choix ici :

  - **La couleur encode, elle ne décore pas.** Un seul accent, réservé au
    marqueur de promotion en LEC, qui est le fait le plus important du jeu de
    données. Tout le reste se hiérarchise par la valeur et la position.
  - **La hiérarchie passe par des bordures d'un pixel, pas par des ombres.**
    C'est le point commun des deux références étudiées, Linear et Seline
    Analytics, et c'est ce qui donne un rendu d'instrument plutôt que de
    maquette.
  - **Plafond de graisse à 600.** Aucun 700. Une interface qui crie n'a pas
    de hiérarchie, elle a du gras.
  - **Les écarts-types se lisent de part et d'autre d'une médiane.** Les
    dessiner comme des barres partant de zéro est une erreur de lecture :
    ce sont des écarts, pas des quantités.
"""

import streamlit as st

COULEURS = {
    "fond": "#14161A",
    "surface": "#1B1E23",
    "surface_haute": "#22262C",
    "hairline": "#2E333A",
    "hairline_forte": "#3C424B",
    "texte": "#DDE1E6",
    "texte_faible": "#8B939E",
    "texte_tres_faible": "#5E656F",
    "accent": "#E4674A",
    "positif": "#7FA88F",
    "negatif": "#A8776F",
}

# Les cinq postes, dans l'ordre de la carte : de la voie du haut au support.
# Cet ordre est celui que tout joueur de LoL a en tête ; le trier autrement
# obligerait le lecteur à chercher.
POSTES = [
    ("top", "Top"),
    ("jng", "Jungle"),
    ("mid", "Mid"),
    ("bot", "Bot"),
    ("sup", "Support"),
]

SEQUENCE_GRAPHIQUE = ["#E4674A", "#7FA88F", "#8B939E", "#B08C64", "#7E8AA3", "#9B7F9E"]


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;500;600&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Typographie ──────────────────────────────────────────────────────────── */
/* Barlow Condensed pour les noms et les titres : elle a la compacite des
   plaques et des feuilles de match, ce qui colle au sujet. Barlow pour le
   corps. JetBrains Mono pour tout chiffre. IBM Plex a ete ecartee : elle
   servait deja ailleurs, et reprendre la meme paire d'un projet a l'autre
   produit exactement le rendu interchangeable qu'on cherche a eviter. */

html, body, [class*="css"], .stMarkdown, p, li, label {
    font-family: 'Barlow', -apple-system, sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Barlow Condensed', 'Barlow', sans-serif !important;
    letter-spacing: 0.01em !important;
}

h1 { font-size: 2.6rem !important; font-weight: 600 !important; line-height: 1.05 !important; }
h2 { font-size: 1.4rem !important; font-weight: 500 !important; margin-top: 2rem !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; }

/* Aucun 700 nulle part : une interface qui crie n'a pas de hierarchie. */
strong, b { font-weight: 600 !important; }

[data-testid="stMetricValue"], .stDataFrame, code, pre, .es-num {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

/* ── Chrome de Streamlit ──────────────────────────────────────────────────── */

[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1240px;
}

/* ── Barre laterale ───────────────────────────────────────────────────────── */

[data-testid="stSidebar"] {
    background: __SURFACE__;
    border-right: 1px solid __HAIRLINE__;
}

[data-testid="stSidebarNav"] li:first-child a p {
    visibility: hidden;
    position: relative;
    min-width: 8rem;
}

[data-testid="stSidebarNav"] li:first-child a p::after {
    content: "Watchlist";
    visibility: visible;
    position: absolute;
    left: 0;
    top: 0;
    white-space: nowrap;
}

/* ── En-tete de page ──────────────────────────────────────────────────────── */

.es-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    /* Le gris le plus faible etait illisible sur le fond. Un contexte qu'on ne
       peut pas lire ne sert a rien. */
    color: __TEXTE_FAIBLE__;
}

.es-lede {
    color: __TEXTE_FAIBLE__;
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 60ch;
}

/* ── Bloc de poste ────────────────────────────────────────────────────────── */
/* La structure primaire de l'outil. Un scout cherche un jungler, pas un
   joueur : le poste est donc le premier niveau de lecture, avant le score. */

.es-poste {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    border-bottom: 1px solid __HAIRLINE_FORTE__;
    padding-bottom: 0.35rem;
    margin: 1.6rem 0 0.2rem 0;
}

.es-poste-nom {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: __TEXTE__;
}

.es-poste-compte {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: __TEXTE_TRES_FAIBLE__;
}

/* ── Ligne de joueur ──────────────────────────────────────────────────────── */

.es-ligne {
    display: grid;
    /* Six colonnes pour six elements : rang, nom, club, ecart, score, marqueur.
       Le premier jet en declarait cinq, si bien que le marqueur passait a la
       ligne suivante et doublait la hauteur de chaque ligne. Invisible dans le
       code, evident au rendu. */
    grid-template-columns: 1.6rem minmax(0, 1fr) 9rem 6.5rem 3.2rem 2.6rem;
    align-items: center;
    column-gap: 0.7rem;
    row-gap: 0;
    padding: 0.3rem 0.55rem;
    border-bottom: 1px solid __HAIRLINE__;
    line-height: 1.25;
}

.es-ligne:hover { background: __SURFACE__; }

.es-rang {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: __TEXTE_TRES_FAIBLE__;
    text-align: right;
}

.es-nom {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: __TEXTE__;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.es-club {
    font-size: 0.76rem;
    color: __TEXTE_FAIBLE__;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.es-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    font-weight: 500;
    color: __TEXTE__;
    text-align: right;
}

/* Le marqueur de promotion est le seul endroit ou l'accent apparait. C'est le
   fait le plus important du jeu de donnees : ce joueur est reellement monte. */
.es-ligne > div { line-height: 1.25; }

.es-lec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    color: __ACCENT__;
    border: 1px solid __ACCENT__;
    border-radius: 2px;
    padding: 0 0.25rem;
    text-align: center;
}

.es-lec-vide { visibility: hidden; }

/* ── Ecart-type ───────────────────────────────────────────────────────────── */
/* L'element signature. Un z-score est un ecart a la moyenne de sa ligue, pas
   une quantite : le dessiner en barre partant de zero, comme le faisait la
   version precedente, le fait lire a l'envers. Ici la mediane est materialisee
   et la barre s'en ecarte vers la droite ou vers la gauche. */

.es-ecart {
    position: relative;
    height: 12px;
}

.es-ecart::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: __HAIRLINE_FORTE__;
}

.es-ecart-barre {
    position: absolute;
    top: 3px;
    height: 6px;
    background: __TEXTE__;
    opacity: 0.75;
}

.es-ecart-barre.pos { left: 50%; }
.es-ecart-barre.neg { right: 50%; background: __TEXTE_TRES_FAIBLE__; }

.es-legende {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    color: __TEXTE_FAIBLE__;
    text-transform: uppercase;
}

/* ── Widgets ──────────────────────────────────────────────────────────────── */
/* Streamlit peint tous ses widgets avec primaryColor : les etiquettes de
   filtre et les curseurs sortaient donc en orange. Or l'accent est reserve au
   marqueur de promotion en LEC. Un accent qui apparait partout ne signale plus
   rien, et c'est exactement la discipline que la refonte devait apporter. */

[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: __SURFACE_HAUTE__ !important;
    color: __TEXTE__ !important;
    border: 1px solid __HAIRLINE_FORTE__ !important;
    border-radius: 3px !important;
    font-size: 0.72rem !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] svg { fill: __TEXTE_FAIBLE__ !important; }

/* Le rail du curseur est peint en degrade inline par Streamlit a partir de
   primaryColor : le surcharger en CSS est fragile. La configuration met donc
   un gris neutre dans primaryColor, et l'accent du projet reste reserve au
   marqueur de promotion. */

[data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    color: __TEXTE_FAIBLE__ !important;
}

[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    color: __TEXTE_FAIBLE__ !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: __SURFACE_HAUTE__;
    border-color: __HAIRLINE_FORTE__;
    font-size: 0.82rem;
}
</style>
"""


def appliquer() -> None:
    """Injecte la feuille de style. À appeler en tête de chaque page."""
    css = _CSS
    for cle, valeur in {
        "__SURFACE__": COULEURS["surface"],
        "__SURFACE_HAUTE__": COULEURS["surface_haute"],
        "__HAIRLINE__": COULEURS["hairline"],
        "__HAIRLINE_FORTE__": COULEURS["hairline_forte"],
        "__TEXTE__": COULEURS["texte"],
        "__TEXTE_FAIBLE__": COULEURS["texte_faible"],
        "__TEXTE_TRES_FAIBLE__": COULEURS["texte_tres_faible"],
        "__ACCENT__": COULEURS["accent"],
    }.items():
        css = css.replace(cle, valeur)
    st.markdown(css, unsafe_allow_html=True)


def entete(kicker: str, titre: str, lede: str = "") -> None:
    """En-tête de page : une ligne de contexte, le titre, un chapô optionnel."""
    st.markdown(f'<div class="es-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"# {titre}")
    if lede:
        st.markdown(f'<div class="es-lede">{lede}</div>', unsafe_allow_html=True)


def _barre_ecart(z: float | None, etendue: float = 2.5) -> str:
    """Dessine un z-score comme un écart de part et d'autre de la médiane.

    `etendue` borne l'affichage : au-delà de 2,5 écarts-types la barre sature,
    parce qu'un joueur à +6 écraserait visuellement tous les autres sans rien
    apprendre de plus. La valeur exacte reste dans l'infobulle du tableau.
    """
    if z is None:
        return '<div class="es-ecart"></div>'

    part = min(abs(z) / etendue, 1.0) * 50
    sens = "pos" if z >= 0 else "neg"
    return (
        '<div class="es-ecart">'
        f'<div class="es-ecart-barre {sens}" style="width:{part:.1f}%"></div>'
        "</div>"
    )


def ligne_joueur(rang: int, nom: str, club: str, score: float, z: float | None, promu: bool) -> str:
    """Une ligne de la liste de surveillance."""
    marqueur = '<div class="es-lec">LEC</div>' if promu else '<div class="es-lec es-lec-vide">LEC</div>'
    return (
        '<div class="es-ligne">'
        f'<div class="es-rang">{rang}</div>'
        f'<div class="es-nom">{nom}</div>'
        f'<div class="es-club">{club}</div>'
        f"{_barre_ecart(z)}"
        f'<div class="es-score">{score:.1f}</div>'
        f"{marqueur}"
        "</div>"
    )


def entete_poste(libelle: str, compte: int) -> str:
    """Le séparateur qui ouvre un bloc de poste."""
    return (
        '<div class="es-poste">'
        f'<div class="es-poste-nom">{libelle}</div>'
        f'<div class="es-poste-compte">{compte} joueurs suivis</div>'
        "</div>"
    )
