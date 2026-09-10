"""Tests de l'agent : la boucle d'outils, la traçabilité, le coût.

Aucun de ces tests n'appelle un modèle. Un test qui dépend du réseau et d'une
clé payante finit désactivé au premier échec, et une suite désactivée ne
protège rien.
"""

from __future__ import annotations

import json

import pytest

from agent import tools as tool_registry
from agent.agent import ScoutAgent
from agent.baseline import BaselineProvider
from agent.prompts import REFUSAL_MARKER, build_system_prompt
from agent.providers import (
    CassetteProvider,
    Completion,
    ProviderError,
    ToolCall,
    Usage,
    conversation_key,
    get_provider,
)


class FakeProvider:
    """Fournisseur scripté : rend les complétions qu'on lui donne, dans l'ordre."""

    name = "fake"
    model = "claude-sonnet-5"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = list(completions)
        self.calls: list[dict] = []

    def complete(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.completions:
            return Completion(text="fin")
        return self.completions.pop(0)


# ── Outils ────────────────────────────────────────────────────────────────────


def test_les_outils_declarent_un_schema_exploitable():
    for tool in tool_registry.TOOLS:
        assert tool["name"]
        assert len(tool["description"]) > 80, f"{tool['name']} : description trop courte pour guider un choix"
        assert tool["parameters"]["type"] == "object"


def test_un_outil_inconnu_renvoie_une_erreur_au_lieu_de_lever():
    payload = json.loads(tool_registry.execute("outil_qui_nexiste_pas"))
    assert "error" in payload
    assert "outils_disponibles" in payload


def test_un_joueur_inconnu_renvoie_une_erreur_lisible():
    payload = json.loads(tool_registry.execute("get_player", {"playername": "joueur-inexistant-xyz"}))
    assert "error" in payload
    assert "hint" in payload


def test_les_filtres_reels_sont_servis():
    payload = json.loads(tool_registry.execute("list_filters"))
    assert "LFL" in payload["leagues"]
    assert set(payload["positions"]) >= {"top", "jng", "mid", "bot", "sup"}


# ── Boucle de l'agent ─────────────────────────────────────────────────────────


def test_lagent_execute_loutil_puis_repond():
    provider = FakeProvider(
        [
            Completion(
                tool_calls=[ToolCall(id="c1", name="list_filters", arguments={})],
                usage=Usage(10, 5),
            ),
            Completion(text="Six ligues.", usage=Usage(20, 8)),
        ]
    )
    answer = ScoutAgent(provider=provider).ask("Combien de ligues ?")

    assert answer.text == "Six ligues."
    assert answer.tool_names == ["list_filters"]
    assert answer.usage.input_tokens == 30
    assert answer.usage.output_tokens == 13
    assert answer.steps == 2
    assert not answer.truncated


def test_le_resultat_de_loutil_est_bien_renvoye_au_modele():
    provider = FakeProvider(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="list_filters", arguments={})]),
            Completion(text="ok"),
        ]
    )
    ScoutAgent(provider=provider).ask("Combien de ligues ?")

    dernier = provider.calls[-1]["messages"]
    message_outil = [m for m in dernier if m["role"] == "tool"]
    assert message_outil, "le résultat de l'outil doit revenir au modèle"
    assert "LFL" in message_outil[0]["content"]


def test_une_boucle_sans_fin_est_coupee_et_signalee():
    provider = FakeProvider(
        [Completion(tool_calls=[ToolCall(id=f"c{i}", name="list_filters", arguments={})]) for i in range(20)]
    )
    answer = ScoutAgent(provider=provider, max_steps=3).ask("question insoluble")

    assert answer.truncated
    assert answer.refused, "une non-convergence doit être déclarée, pas maquillée en réponse"
    assert answer.steps == 3


def test_une_panne_doutil_ne_casse_pas_la_conversation():
    provider = FakeProvider(
        [
            Completion(tool_calls=[ToolCall(id="c1", name="get_player", arguments={"playername": "inconnu-xyz"})]),
            Completion(text="Ce joueur est absent des données."),
        ]
    )
    answer = ScoutAgent(provider=provider).ask("Score de inconnu-xyz ?")

    assert answer.trace[0].is_error
    assert answer.text == "Ce joueur est absent des données."


def test_le_cout_suit_le_modele():
    answer_usage = Usage(input_tokens=1_000_000, output_tokens=0)
    # 3 $ le million en entrée sur sonnet, converti à 1,08 $ pour 1 €.
    assert answer_usage.cost_eur("claude-sonnet-5") == pytest.approx(3.0 / 1.08, rel=1e-6)
    assert answer_usage.cost_eur("gpt-4o-mini") == pytest.approx(0.15 / 1.08, rel=1e-6)


def test_le_prompt_porte_les_deux_regles_de_lecture():
    prompt = build_system_prompt()
    assert "percentile" in prompt
    assert "10 matchs" in prompt
    assert REFUSAL_MARKER in prompt


def test_une_consigne_supplementaire_est_ajoutee_sans_ecraser_le_prompt():
    prompt = build_system_prompt("Réponds en une seule phrase.")
    assert "Réponds en une seule phrase." in prompt
    assert "percentile" in prompt


# ── Rejeu ─────────────────────────────────────────────────────────────────────


def test_la_cle_de_rejeu_change_quand_les_outils_changent():
    base = conversation_key("m", "sys", [{"role": "user", "content": "q"}], [{"name": "a"}])
    autre = conversation_key("m", "sys", [{"role": "user", "content": "q"}], [{"name": "b"}])
    assert base != autre, "un enregistrement fait avec d'autres outils ne doit pas être rejoué"


def test_un_rejeu_manquant_echoue_bruyamment(tmp_path):
    provider = CassetteProvider(path=tmp_path / "vide.json")
    with pytest.raises(ProviderError, match="Aucun enregistrement"):
        provider.complete("sys", [{"role": "user", "content": "q"}], [])


def test_enregistrer_puis_rejouer_rend_la_meme_reponse(tmp_path):
    chemin = tmp_path / "cassette.json"
    interne = FakeProvider([Completion(text="réponse enregistrée", usage=Usage(7, 3))])

    enregistreur = CassetteProvider(path=chemin, inner=interne, model=interne.model)
    premiere = enregistreur.complete("sys", [{"role": "user", "content": "q"}], [])
    enregistreur.save()

    rejoueur = CassetteProvider(path=chemin)
    seconde = rejoueur.complete("sys", [{"role": "user", "content": "q"}], [])

    assert premiere.text == seconde.text == "réponse enregistrée"
    assert seconde.usage.input_tokens == 7


def test_le_fournisseur_par_defaut_marche_sans_cle():
    provider = get_provider()
    assert provider.name == "heuristique"


def test_enregistrer_sans_vrai_fournisseur_est_refuse():
    with pytest.raises(ProviderError, match="--record"):
        get_provider("heuristique", record=True)


# ── Référence heuristique ─────────────────────────────────────────────────────


def test_la_reference_refuse_une_ligue_hors_perimetre():
    answer = ScoutAgent(provider=BaselineProvider()).ask("Qui est le meilleur mid de LEC en 2026 ?")
    assert answer.refused


def test_la_reference_ne_confond_pas_la_lec_perimetre_et_la_promotion_en_lec():
    """Bug réel trouvé par le banc : toute question sur les promotions était refusée."""
    answer = ScoutAgent(provider=BaselineProvider()).ask(
        "Parmi les archétypes de jeu du poste mid, quel numéro de cluster "
        "affiche le meilleur taux de promotion en LEC ?"
    )
    assert not answer.refused


def test_la_reference_ne_prend_plus_ans_pour_dans():
    """Bug réel : « dans les données » contenait « ans » et déclenchait un refus."""
    answer = ScoutAgent(provider=BaselineProvider()).ask("Combien de postes différents existent dans les données ?")
    assert not answer.refused
    assert "5" in answer.text


def test_la_reference_est_deterministe():
    agent = ScoutAgent(provider=BaselineProvider())
    question = "Quel joueur a le score de talent le plus élevé, en ne gardant que ceux qui ont au moins 10 matchs ?"
    assert agent.ask(question).text == agent.ask(question).text


# ── Fournisseurs compatibles OpenAI ───────────────────────────────────────────


def test_les_points_dentree_declarent_ce_quil_faut():
    from agent.providers import ENDPOINTS

    for nom, config in ENDPOINTS.items():
        assert "base_url" in config, nom
        assert "api_key_env" in config, nom
        assert config["default_model"], nom


def test_deepseek_et_glm_sont_selectionnables():
    from agent.providers import ENDPOINTS

    assert {"deepseek", "glm", "groq", "openrouter", "ollama"} <= set(ENDPOINTS)


def test_un_fournisseur_inconnu_liste_les_fournisseurs_connus():
    with pytest.raises(ProviderError, match="deepseek"):
        get_provider("fournisseur-imaginaire")


def test_une_cle_absente_nomme_la_variable_attendue(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        get_provider("deepseek")


def test_un_tarif_non_releve_donne_un_cout_inconnu_et_non_un_cout_faux():
    """La régression à ne jamais réintroduire : appliquer le tarif de Sonnet à tout modèle."""
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_eur("deepseek-chat") is None
    assert usage.cost_eur("claude-sonnet-5") is not None
