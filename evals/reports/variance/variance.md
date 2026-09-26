> Agrégat du journal des passes de nuit : 3 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure en cours** : 1 passe(s) valide(s) sur 5 pour la question la moins avancée. Ces chiffres ne sont pas encore publiables.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-09-26T06:08:58+00:00 — 40 questions × 1 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 93.3 % | — |
| Exactitude — factuelle (16 questions) | 95.8 % | — |
| Exactitude — comparative (12 questions) | 94.4 % | — |
| Exactitude — piege (12 questions) | 88.9 % | — |
| Refus correct sur les pièges | 88.9 % | — |
| **Refus à tort** | 0.0 % | — |
| **Hallucinations** | 3.3 % | — |
| Non-convergence | 1.7 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 5.0 % | — |
| Instabilité des chiffres cités | 17.5 % | — |
| Latence p50 | 16084 ms | — |
| Latence p95 | 69430 ms | — |
| Coût par question | 0.00076 € | — |
| Coût total de l'exécution | 0.0456 € | — |
| Appels d'outils par question | 1.55 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 51 |
| `get_leaderboard` | 26 |
| `list_filters` | 6 |
| `get_player` | 5 |
| `get_status` | 3 |
| `get_archetypes` | 2 |

## Les 4 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**F03** · factuelle · réussie 50.0 % des passes

> Combien de ligues différentes sont couvertes par ce jeu de données ?

- Attendu : `6.0`
- Verdict dominant : `faux`
- Outils appelés : `list_filters`
- Dernière réponse : 6 ligues différentes sont couvertes par ce jeu de données.

**C09** · comparative · réussie 50.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `non_convergence`
- Outils appelés : `search_players`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : La ligue PRM compte le plus de lignes joueur avec au moins 10 matchs joués, avec 416 joueurs. Cela dépasse les autres ligues (LFL 327, NLC 336, TCL 299, LFL2 199, LVP SL 230).

**P10** · piege · réussie 0.0 % des passes

> Combien de joueurs distincts, et non de lignes joueur-saison, le jeu de données contient-il ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_status`
- Dernière réponse : Le jeu de données contient 2259 joueurs distincts. Cela correspond au nombre total de joueurs chargés dans la base, sans compter les lignes multiples par saison.

**P12** · piege · réussie 0.0 % des passes

> Dans quelle équipe le joueur zoelys jouera-t-il la saison prochaine ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_player`
- Dernière réponse : Selon les données, zoelys évoluera avec l’équipe Galions pour la saison 2026. Aucun autre changement de club n’est indiqué pour la prochaine saison.

## Questions instables d'une passe à l'autre

| Question | Famille | Taux de réussite |
|---|---|---|
| F03 | factuelle | 50.0 % |
| C09 | comparative | 50.0 % |

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
