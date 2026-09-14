"""La logique de la page « Agent », séparée de Streamlit pour être testée.

Ce module n'importe pas Streamlit. Tout ce qui décide quelque chose (quota,
nettoyage de la question, lecture de la vérité terrain, rendu HTML échappé)
vit ici ; la page ne fait que brancher ces fonctions sur des widgets.

Deux garde-fous portent la page publique, et ils sont volontairement simples :

- **un quota par session et un quota par jour**, parce que la page est ouverte à
  n'importe qui et que le fournisseur a des limites. Le palier gratuit de Groq
  plafonne déjà, mais une page qui répond « limite atteinte » vaut mieux qu'une
  page qui renvoie une erreur de fournisseur ;
- **tout texte affiché est échappé.** La question vient d'un inconnu et la
  réponse d'un modèle : ni l'une ni l'autre n'a le droit d'injecter du HTML.
"""

from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

#: Deux questions par famille, choisies pour montrer ce que le banc mesure :
#: une réponse directe, une comparaison qui demande plusieurs appels, et un
#: piège dont la seule bonne réponse est un refus.
QUESTIONS_PROPOSEES: dict[str, tuple[str, ...]] = {
    "factuelle": ("F02", "F09"),
    "comparative": ("C01", "C05"),
    "piege": ("P01", "P06"),
}

LIBELLES_FAMILLE = {
    "factuelle": "Factuelle",
    "comparative": "Comparative",
    "piege": "Piège",
}

LONGUEUR_MAX_QUESTION = 300
QUESTIONS_PAR_SESSION = 5
#: Calé sur le quota gratuit de Groq : 200 000 jetons par jour et par modèle, et
#: environ 6 500 jetons par question relevés le 2026-09-14 sur qwen3.8-27b.
#: Au-delà de ~30 questions, c'est le fournisseur qui refuserait ; mieux vaut que
#: la page dise « limite de la démo » que d'afficher une panne.
QUESTIONS_PAR_JOUR = 25

#: Chaque verdict du banc, son libellé et sa tonalité. Une non-réponse n'est
#: ni bonne ni mauvaise : elle est affichée en neutre, comme dans le rapport.
VERDICTS: dict[str, tuple[str, str]] = {
    "juste": ("Juste", "bon"),
    "refus_attendu": ("Refus attendu", "bon"),
    "faux": ("Faux", "mauvais"),
    "refus_a_tort": ("Refus à tort", "mauvais"),
    "hallucination": ("Hallucination", "mauvais"),
    "non_convergence": ("Sans réponse", "neutre"),
    "erreur_fournisseur": ("Fournisseur indisponible", "neutre"),
}

FOURNISSEURS_SIMULES = {"heuristique", "cassette", "inconnu"}


# ── Quota ─────────────────────────────────────────────────────────────────────


@dataclass
class QuotaJournalier:
    """Compteur partagé entre toutes les sessions, remis à zéro chaque jour."""

    par_jour: int = QUESTIONS_PAR_JOUR
    jour: date | None = None
    utilisees: int = 0
    _verrou: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def consommer(self, aujourd_hui: date) -> bool:
        """Réserve une question. Faux si la limite du jour est atteinte."""
        with self._verrou:
            if self.jour != aujourd_hui:
                self.jour, self.utilisees = aujourd_hui, 0
            if self.utilisees >= self.par_jour:
                return False
            self.utilisees += 1
            return True


def nettoyer_question(texte: str | None) -> tuple[str | None, str | None]:
    """Renvoie (question, erreur). Une seule des deux est renseignée."""
    propre = " ".join((texte or "").split())
    if not propre:
        return None, "Écris une question, ou choisis-en une parmi celles du banc."
    if len(propre) > LONGUEUR_MAX_QUESTION:
        return None, f"Question trop longue : {LONGUEUR_MAX_QUESTION} caractères au plus."
    return propre, None


# ── Lecture ───────────────────────────────────────────────────────────────────


def _nombre(valeur: float) -> str:
    if float(valeur).is_integer():
        return f"{int(valeur):,}".replace(",", " ")
    return f"{valeur:.2f}".replace(".", ",")


def format_verite(expects: str, expected: Any) -> str:
    """La réponse attendue, telle qu'un lecteur la comprend."""
    if expects == "refusal":
        return "Aucune réponse dans les données"
    if expects == "names":
        return ", ".join(str(nom) for nom in expected)
    if expects == "number":
        return _nombre(float(expected))
    return str(expected)


def format_arguments(arguments: dict[str, Any]) -> str:
    return " ".join(f"{cle}={valeur}" for cle, valeur in arguments.items()) or "sans argument"


def format_cout(cout_eur: float | None) -> str:
    if cout_eur is None:
        return "coût non chiffré"
    return f"{cout_eur:.4f} €".replace(".", ",")


#: Au-delà, un rapport mesure le quota du fournisseur, pas l'agent. Relevé le
#: 2026-09-14 : le palier gratuit de Groq a lâché pendant la passe 2 sur 5, et
#: le rapport publiait 27,5 % d'exactitude pour un agent à 90 %.
PANNES_MAX = 0.10


def resume_banc(repertoire: Path) -> dict[str, Any] | None:
    """Le dernier rapport exploitable mesuré sur un vrai modèle, ou None.

    Trois filtres, et chacun a une raison :
    - la référence déterministe est écartée : la page affiche ce que vaut
      l'agent réel, pas le plancher de la CI ;
    - une seule passe ne dit rien de la stabilité, or c'est la question que
      pose un système non déterministe ;
    - un rapport noyé de pannes du fournisseur ne mesure pas l'agent.
    Sans rapport qui passe les trois, la page n'affiche aucun chiffre plutôt
    qu'un chiffre trompeur.
    """
    for chemin in sorted(repertoire.glob("*.json"), reverse=True):
        try:
            rapport = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rapport.get("fournisseur") in FOURNISSEURS_SIMULES:
            continue
        if rapport.get("passes", 1) < 2 or rapport.get("erreurs_fournisseur", 0) > PANNES_MAX:
            continue
        return rapport
    return None


def quota_fournisseur_atteint(erreur: str | None) -> bool:
    """Vrai si le fournisseur a refusé pour cause de limite de débit ou de quota."""
    return bool(erreur) and ("429" in erreur or "RateLimit" in erreur or "rate_limit" in erreur)


# ── Rendu HTML ────────────────────────────────────────────────────────────────


def bloc_verdict(verdict: str | None, outils: int, latence_ms: float, cout_eur: float | None) -> str:
    """La ligne qui tranche : verdict, puis ce que la réponse a coûté."""
    if verdict is None:
        libelle, ton = "Question libre, non vérifiée", "neutre"
    else:
        libelle, ton = VERDICTS.get(verdict, (verdict, "neutre"))
    secondes = f"{latence_ms / 1000:.1f}".replace(".", ",")
    mesures = f"{outils} outil{'s' if outils > 1 else ''} · {secondes} s · {format_cout(cout_eur)}"
    return (
        f'<div class="es-verdict {ton}">'
        f'<span class="es-verdict-libelle">{html.escape(libelle)}</span>'
        f'<span class="es-verdict-mesures">{html.escape(mesures)}</span>'
        "</div>"
    )


def ligne_trace(rang: int, nom: str, arguments: dict[str, Any], duree_ms: float, erreur: bool) -> str:
    """Un appel d'outil, dans l'ordre où l'agent l'a fait."""
    etat = '<span class="es-trace-erreur">erreur</span>' if erreur else ""
    return (
        '<div class="es-trace">'
        f'<div class="es-rang">{rang}</div>'
        f'<div class="es-trace-outil">{html.escape(nom)}</div>'
        f'<div class="es-trace-args">{html.escape(format_arguments(arguments))}{etat}</div>'
        f'<div class="es-trace-duree">{duree_ms:.0f} ms</div>'
        "</div>"
    )


def texte(contenu: str, classe: str) -> str:
    """Un paragraphe échappé. Les sauts de ligne du modèle sont conservés."""
    return f'<div class="{classe}">{html.escape(contenu).replace(chr(10), "<br>")}</div>'
