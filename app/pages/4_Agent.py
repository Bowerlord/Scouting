"""
4_Agent.py — L'agent en direct, jugé contre la vérité terrain.

Créée le 2026-09-14. Une démo d'agent montre qu'il répond. Cette page montre
**s'il a raison** : pour chacune des questions du banc, la réponse du modèle est
posée contre la réponse calculée en SQL par un chemin indépendant, avec le
verdict, les outils appelés et ce que la réponse a coûté.

Une question libre est possible, mais elle n'a pas de vérité terrain, et la page
le dit plutôt que d'afficher un verdict qu'elle ne peut pas établir.

Configuration (Streamlit Cloud, onglet Secrets) :
  GROQ_API_KEY = "..."                      # sans elle, la référence déterministe répond
  SCOUTING_API_URL = "https://..."          # facultatif, l'API Cloud Run par défaut
"""

import os
import sys
from datetime import date
from pathlib import Path

# pages/ → app/ pour `utils`, et la racine du dépôt pour `agent` et `evals`.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from utils import demo_agent as demo
from utils import theme

API_PUBLIQUE = "https://scouting-api-7750158787.europe-west1.run.app"
RAPPORTS = Path(__file__).resolve().parents[2] / "evals" / "reports"


def _secret(nom: str) -> str | None:
    """Lit un secret Streamlit sans planter quand aucun fichier de secrets n'existe."""
    try:
        valeur = st.secrets.get(nom)
    except Exception:  # noqa: BLE001 — absence de secrets.toml en local
        valeur = None
    return valeur or os.getenv(nom)


# Avant tout import de `agent` : le backend lit l'URL de l'API à l'import.
os.environ.setdefault("SCOUTING_API_URL", _secret("SCOUTING_API_URL") or API_PUBLIQUE)
if _secret("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = _secret("GROQ_API_KEY")

from agent.agent import ScoutAgent  # noqa: E402
from agent.providers import get_provider  # noqa: E402
from evals.scoring import grade  # noqa: E402
from evals.truth import load_questions  # noqa: E402

st.set_page_config(page_title="Agent — ERL Scout", page_icon="◆", layout="wide")
theme.appliquer()


@st.cache_resource
def _agent() -> ScoutAgent:
    fournisseur = "groq" if os.getenv("GROQ_API_KEY") else "heuristique"
    return ScoutAgent(provider=get_provider(fournisseur))


@st.cache_resource
def _quota() -> demo.QuotaJournalier:
    return demo.QuotaJournalier()


@st.cache_resource
def _questions() -> dict:
    return {q.id: q for q in load_questions()}


agent = _agent()
questions = _questions()
reel = agent.provider.name not in demo.FOURNISSEURS_SIMULES if hasattr(agent.provider, "name") else False

# ══════════════════════════════════════════════════════════════════════════════
# En-tête
# ══════════════════════════════════════════════════════════════════════════════

banc = demo.resume_banc(RAPPORTS)
if banc:
    passes = f"{banc['passes']} passe{'s' if banc['passes'] > 1 else ''}"
    kicker = (
        f"Agent · mesuré sur {banc['questions']} questions × {passes} · "
        f"{banc['exactitude'] * 100:.1f} % juste · {banc['hallucinations'] * 100:.1f} % d'hallucinations"
    ).replace(".", ",")
else:
    kicker = "Agent · banc d'évaluation"

theme.entete(
    kicker,
    "Pose une question, vérifie la réponse",
    "L'agent répond en langage naturel en appelant l'API d'ERL Scout. Sur les questions du banc, "
    "sa réponse est comparée à la vérité calculée en SQL, sans rien partager avec lui. "
    "Les pièges n'ont pas de réponse dans les données : la seule bonne réponse est un refus.",
)

if not reel:
    st.markdown(
        '<div class="es-verite-note">Aucune clé de modèle configurée : c\'est la référence '
        "déterministe du banc qui répond, un routeur à mots-clés sans modèle de langage.</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Choix de la question
# ══════════════════════════════════════════════════════════════════════════════

st.write("")
choix = None
colonnes = st.columns(len(demo.QUESTIONS_PROPOSEES))
for colonne, (famille, identifiants) in zip(colonnes, demo.QUESTIONS_PROPOSEES.items()):
    with colonne:
        st.markdown(f'<div class="es-famille">{demo.LIBELLES_FAMILLE[famille]}</div>', unsafe_allow_html=True)
        for identifiant in identifiants:
            if st.button(questions[identifiant].question, key=f"q-{identifiant}", use_container_width=True):
                choix = (identifiant, questions[identifiant].question)

with st.form("question-libre", clear_on_submit=False, border=False):
    libre = st.text_input(
        "Ou pose ta propre question",
        max_chars=demo.LONGUEUR_MAX_QUESTION,
        placeholder="Quels sont les trois meilleurs supports de LFL en 2026, avec au moins 10 matchs ?",
    )
    if st.form_submit_button("Poser la question"):
        propre, erreur = demo.nettoyer_question(libre)
        if erreur:
            st.warning(erreur)
        else:
            choix = (None, propre)

# ══════════════════════════════════════════════════════════════════════════════
# Exécution, sous quota
# ══════════════════════════════════════════════════════════════════════════════

if choix:
    posees = st.session_state.get("posees", 0)
    if posees >= demo.QUESTIONS_PAR_SESSION:
        st.warning(
            f"Limite de {demo.QUESTIONS_PAR_SESSION} questions par visite atteinte. "
            "Le rapport complet du banc est dans le dépôt, dossier evals/reports."
        )
    elif not _quota().consommer(date.today()):
        st.warning("Limite quotidienne de la démo atteinte. Elle se rouvre demain.")
    else:
        st.session_state["posees"] = posees + 1
        with st.spinner("L'agent interroge l'API…"):
            reponse = agent.ask(choix[1])
        st.session_state["dernier"] = (choix[0], reponse)

# ══════════════════════════════════════════════════════════════════════════════
# Réponse face à la vérité
# ══════════════════════════════════════════════════════════════════════════════

if "dernier" in st.session_state:
    identifiant, reponse = st.session_state["dernier"]
    question = questions.get(identifiant) if identifiant else None

    verdict = None
    if question is not None:
        verdict = grade(
            question, reponse.text, provider_error=reponse.provider_error, truncated=reponse.truncated
        ).verdict

    st.markdown(demo.texte(reponse.question, "es-question"), unsafe_allow_html=True)

    if question is None:
        verite = (
            '<div class="es-kicker">Vérité terrain</div>'
            '<div class="es-verite-note">Question libre : aucune vérité calculée. Seules les 40 '
            "questions du banc sont vérifiées, le verdict ne s'affiche que pour elles.</div>"
        )
    elif question.is_trap:
        verite = (
            '<div class="es-kicker">Vérité terrain</div>'
            + demo.texte(demo.format_verite(question.expects, question.expected), "es-verite")
            + demo.texte(question.why or "", "es-verite-note")
        )
    else:
        verite = (
            '<div class="es-kicker">Vérité terrain · SQL</div>'
            + demo.texte(demo.format_verite(question.expects, question.expected), "es-verite")
        )

    st.markdown(
        '<div class="es-face">'
        f'<div><div class="es-kicker">Réponse de l\'agent</div>{demo.texte(reponse.text, "es-reponse")}</div>'
        f"<div>{verite}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        demo.bloc_verdict(verdict, len(reponse.trace), reponse.latency_ms, reponse.cost_eur()),
        unsafe_allow_html=True,
    )

    if reponse.trace:
        st.write("")
        st.markdown('<div class="es-legende">Ce que l\'agent a appelé, dans l\'ordre</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                demo.ligne_trace(rang, appel.name, appel.arguments, appel.duration_ms, appel.is_error)
                for rang, appel in enumerate(reponse.trace, start=1)
            ),
            unsafe_allow_html=True,
        )

    # En dernier : la preuve se consulte après le verdict, elle ne s'intercale
    # pas entre la réponse et le jugement.
    if question is not None and question.sql:
        st.write("")
        with st.expander("La requête SQL qui établit la vérité"):
            st.code(question.sql.strip(), language="sql")

# ══════════════════════════════════════════════════════════════════════════════
# Pied de page
# ══════════════════════════════════════════════════════════════════════════════

st.write("")
modele = getattr(agent.provider, "model", "référence déterministe")
restantes = max(demo.QUESTIONS_PAR_SESSION - st.session_state.get("posees", 0), 0)
st.markdown(
    f'<div class="es-legende">Modèle {modele} · {restantes} question{"s" if restantes > 1 else ""} '
    "restante" + ("s" if restantes > 1 else "") + " pour cette visite · sept outils en lecture seule, aucun SQL "
    "écrit par le modèle</div>",
    unsafe_allow_html=True,
)
