"""Thème visuel du dashboard.

Streamlit expose peu de leviers de style : `config.toml` couvre quatre couleurs
et une famille de police, ce qui suffit à changer l'ambiance mais pas à sortir
du rendu par défaut. Le reste passe par cette feuille de style.

Ce qui est corrigé ici, et pourquoi :

  - **La typographie.** La police par défaut de Streamlit est reconnaissable
    entre toutes. IBM Plex Sans a du caractère et évoque l'outil technique,
    IBM Plex Mono sert aux chiffres pour qu'ils s'alignent.
  - **Le chrome de Streamlit.** Le bouton « Deploy », le menu et le pied de page
    signalent un prototype. Un outil qu'on montre à un recruteur n'affiche pas
    les boutons de l'outil qui l'a construit.
  - **Les cartes de métriques.** Par défaut elles flottent sans limite visible.
    Une bordure et un fond les posent, et le chiffre passe en monospace.
  - **La densité.** Streamlit espace généreusement, ce qui donne un rendu mou.
    Les marges sont resserrées et la hiérarchie des titres accentuée.
"""

import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Palette
# ══════════════════════════════════════════════════════════════════════════════

COULEURS = {
    "fond": "#0D0D0F",
    "surface": "#17171A",
    "surface_haute": "#1F1F24",
    "bordure": "#2A2A31",
    "texte": "#E8E6E3",
    "texte_faible": "#8A8A93",
    "accent": "#E0A82E",
    "positif": "#4A9E6F",
    "negatif": "#C2544D",
}

# Échelle pour les graphiques, dérivée de la palette plutôt que reprise d'une
# bibliothèque : sans cela, les figures ne ressemblent pas au reste de la page.
SEQUENCE_GRAPHIQUE = ["#E0A82E", "#7C8CA1", "#4A9E6F", "#C2544D", "#9B7BB8", "#5E9BB5"]


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Typographie ──────────────────────────────────────────────────────────── */

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

h1 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    font-size: 2.1rem !important;
    padding-top: 0 !important;
}

h2 {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    font-size: 1.35rem !important;
    margin-top: 2.2rem !important;
    padding-bottom: 0.4rem !important;
}

h3 {
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: __TEXTE__ !important;
}

/* Les chiffres sont tabulaires partout : sans cela, une valeur qui change de
   largeur fait sautiller la ligne entiere. */
[data-testid="stMetricValue"],
.stDataFrame, code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

/* ── Chrome de Streamlit ──────────────────────────────────────────────────── */

/* Le bouton Deploy, le menu et le pied de page signalent un prototype. */
[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] {
    display: none !important;
}

/* ── Densite ──────────────────────────────────────────────────────────────── */

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1180px;
}

/* ── Cartes de metriques ──────────────────────────────────────────────────── */

[data-testid="stMetric"] {
    background: __SURFACE__;
    border: 1px solid __BORDURE__;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: __TEXTE_FAIBLE__ !important;
}

/* Streamlit tronque les libelles trop longs par des points de suspension, ce
   qui donnait « JOUEURS ANALYS... ». On laisse le texte passer a la ligne :
   un libelle coupe est pire qu'un libelle sur deux lignes. */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.3 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    /* Sans cela, une valeur longue comme une plage d'annees est tronquee par
       des points de suspension. Constate au rendu le 2026-09-07. */
    white-space: normal !important;
    overflow: visible !important;
}

/* ── Barre laterale ───────────────────────────────────────────────────────── */

[data-testid="stSidebar"] {
    background: __SURFACE__;
    border-right: 1px solid __BORDURE__;
}

[data-testid="stSidebarNav"] a {
    border-radius: 4px;
}

/* La navigation nomme la page d'entree d'apres son fichier, donc « app ».
   Renommer app.py casserait les commandes documentees dans le README et la
   configuration de deploiement ; on corrige donc a l'affichage. */
[data-testid="stSidebarNav"] li:first-child a p {
    visibility: hidden;
    position: relative;
    /* Le conteneur prend la largeur du texte masque, donc celle de « app ».
       Sans cette largeur minimale, le remplacement s'affiche tronque. */
    min-width: 7rem;
}

[data-testid="stSidebarNav"] li:first-child a p::after {
    content: "Accueil";
    visibility: visible;
    position: absolute;
    left: 0;
    top: 0;
    white-space: nowrap;
}

/* ── Tableaux ─────────────────────────────────────────────────────────────── */

.stDataFrame {
    border: 1px solid __BORDURE__;
    border-radius: 6px;
}

/* ── Separateurs et champs ────────────────────────────────────────────────── */

hr {
    border-color: __BORDURE__ !important;
    margin: 1.6rem 0 !important;
}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: __SURFACE__;
    border-color: __BORDURE__;
}

/* ── Elements de marque ───────────────────────────────────────────────────── */

.es-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: __ACCENT__;
    margin-bottom: 0.35rem;
}

.es-chapo {
    color: __TEXTE_FAIBLE__;
    font-size: 1rem;
    line-height: 1.6;
    max-width: 62ch;
}
</style>
"""


def appliquer() -> None:
    """Injecte la feuille de style. À appeler en tête de chaque page."""
    css = (
        _CSS.replace("__SURFACE__", COULEURS["surface"])
        .replace("__BORDURE__", COULEURS["bordure"])
        .replace("__TEXTE__", COULEURS["texte"])
        .replace("__TEXTE_FAIBLE__", COULEURS["texte_faible"])
        .replace("__ACCENT__", COULEURS["accent"])
    )
    st.markdown(css, unsafe_allow_html=True)


def entete(surtitre: str, titre: str, chapo: str = "") -> None:
    """En-tête de page : un surtitre discret, le titre, un chapô optionnel.

    Le surtitre remplace les emoji qui ouvraient chaque page. Un emoji dans un
    titre est le marqueur le plus immédiat d'un tableau de bord non conçu.
    """
    st.markdown(f'<div class="es-eyebrow">{surtitre}</div>', unsafe_allow_html=True)
    st.markdown(f"# {titre}")
    if chapo:
        st.markdown(f'<div class="es-chapo">{chapo}</div>', unsafe_allow_html=True)
    st.write("")
