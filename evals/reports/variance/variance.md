> Agrégat du journal des passes de nuit : 10 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure complète** : 5 passes valides par question.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-10-03T06:27:19+00:00 — 40 questions × 5 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 92.5 % | — |
| Exactitude — factuelle (16 questions) | 96.2 % | — |
| Exactitude — comparative (12 questions) | 91.7 % | — |
| Exactitude — piege (12 questions) | 88.3 % | — |
| Refus correct sur les pièges | 88.3 % | — |
| **Refus à tort** | 0.5 % | — |
| **Hallucinations** | 3.5 % | — |
| Non-convergence | 1.5 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 12.5 % | — |
| Instabilité des chiffres cités | 62.5 % | — |
| Latence p50 | 16084 ms | — |
| Latence p95 | 80639 ms | — |
| Coût par question | 0.00076 € | — |
| Coût total de l'exécution | 0.1529 € | — |
| Appels d'outils par question | 1.57 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 164 |
| `get_leaderboard` | 96 |
| `list_filters` | 25 |
| `get_player` | 14 |
| `get_status` | 10 |
| `get_archetypes` | 5 |

## Les 6 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**F03** · factuelle · réussie 60.0 % des passes

> Combien de ligues différentes sont couvertes par ce jeu de données ?

- Attendu : `6.0`
- Verdict dominant : `juste`
- Outils appelés : `list_filters`
- Dernière réponse : Il y a 6 ligues différentes couvertes par ce jeu de données. (LFL, LFL2, LVP SL, NLC, PRM et TCL).

**F16** · factuelle · réussie 80.0 % des passes

> Quel est le meilleur joueur de PRM toutes saisons et tous postes, avec au moins 10 matchs ?

- Attendu : `irrelevant`
- Verdict dominant : `juste`
- Outils appelés : `get_leaderboard`
- Dernière réponse : Le meilleur joueur du PRM (toutes saisons, tous postes, minimum 10 matchs) est « irrelevant » avec un talent_score de 89.0674105511206. Son score le place au 99.33184855233851ᵉ percentile. Aucun autre joueur de la ligue ne dépasse ce niveau.

**C04** · comparative · réussie 60.0 % des passes

> Quel poste compte le plus de lignes joueur dans les données ?

- Attendu : `jng`
- Verdict dominant : `juste`
- Outils appelés : `search_players`, `search_players`, `search_players`, `list_filters`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : [banc] L'agent n'a pas convergé en 8 étapes sans produire de réponse finale.

**C09** · comparative · réussie 40.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `juste`
- Outils appelés : `search_players`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : La ligue PRM compte le plus de lignes joueur avec au moins 10 matchs joués, avec 416 joueurs. Aucun autre ligue (LFL 327, LFL2 199, NLC 336, LVP SL 230, TCL 299) n’atteint ce total.

**P10** · piege · réussie 0.0 % des passes

> Combien de joueurs distincts, et non de lignes joueur-saison, le jeu de données contient-il ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_status`
- Dernière réponse : Le jeu de données contient 2259 joueurs distincts. Ceci correspond au nombre indiqué par le service d’état des données.

**P12** · piege · réussie 60.0 % des passes

> Dans quelle équipe le joueur zoelys jouera-t-il la saison prochaine ?

- Attendu : `refus`
- Verdict dominant : `refus_attendu`
- Outils appelés : aucun
- Dernière réponse : DONNEES_INSUFFISANTES – les données ne contiennent que les équipes des saisons 2024 à 2026 et ne prévoient pas la saison prochaine.

## Questions instables d'une passe à l'autre

| Question | Famille | Taux de réussite |
|---|---|---|
| F03 | factuelle | 60.0 % |
| F16 | factuelle | 80.0 % |
| C04 | comparative | 60.0 % |
| C09 | comparative | 40.0 % |
| P12 | piege | 60.0 % |

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
