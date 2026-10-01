> Agrégat du journal des passes de nuit : 8 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure en cours** : 4 passe(s) valide(s) sur 5 pour la question la moins avancée. Ces chiffres ne sont pas encore publiables.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-10-01T07:16:47+00:00 — 40 questions × 4 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 91.9 % | — |
| Exactitude — factuelle (16 questions) | 95.3 % | — |
| Exactitude — comparative (12 questions) | 91.7 % | — |
| Exactitude — piege (12 questions) | 87.5 % | — |
| Refus correct sur les pièges | 87.5 % | — |
| **Refus à tort** | 0.6 % | — |
| **Hallucinations** | 3.8 % | — |
| Non-convergence | 1.2 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 12.5 % | — |
| Instabilité des chiffres cités | 60.0 % | — |
| Latence p50 | 16084 ms | — |
| Latence p95 | 78218 ms | — |
| Coût par question | 0.00077 € | — |
| Coût total de l'exécution | 0.1231 € | — |
| Appels d'outils par question | 1.57 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 131 |
| `get_leaderboard` | 77 |
| `list_filters` | 20 |
| `get_player` | 12 |
| `get_status` | 8 |
| `get_archetypes` | 4 |

## Les 6 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**F03** · factuelle · réussie 50.0 % des passes

> Combien de ligues différentes sont couvertes par ce jeu de données ?

- Attendu : `6.0`
- Verdict dominant : `faux`
- Outils appelés : `list_filters`
- Dernière réponse : Le jeu de données couvre six ligues différentes. Les ligues listées sont : LFL, LFL2, LVP SL, NLC, PRM et TCL.

**F16** · factuelle · réussie 75.0 % des passes

> Quel est le meilleur joueur de PRM toutes saisons et tous postes, avec au moins 10 matchs ?

- Attendu : `irrelevant`
- Verdict dominant : `juste`
- Outils appelés : `get_leaderboard`
- Dernière réponse : Le meilleur joueur du PRM (tous postes, toutes saisons, au moins 10 matchs) est « irrelevant ». Son talent_score est 89.06741055112042, ce qui le place au 99.33184855233851ᵉ percentile.

**C04** · comparative · réussie 75.0 % des passes

> Quel poste compte le plus de lignes joueur dans les données ?

- Attendu : `jng`
- Verdict dominant : `juste`
- Outils appelés : `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : Le poste « jng » compte le plus de lignes joueur dans les données, avec un total de 457 entrées.

**C09** · comparative · réussie 25.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `faux`
- Outils appelés : `list_filters`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : La ligue LFL compte le plus de lignes joueur avec au moins 10 matchs joués, avec 327 joueurs.

**P10** · piege · réussie 0.0 % des passes

> Combien de joueurs distincts, et non de lignes joueur-saison, le jeu de données contient-il ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_status`
- Dernière réponse : Le jeu de données contient 2259 joueurs distincts. Cela correspond au nombre de profils uniques, indépendamment du nombre de saisons ou de lignes associées.

**P12** · piege · réussie 50.0 % des passes

> Dans quelle équipe le joueur zoelys jouera-t-il la saison prochaine ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : aucun
- Dernière réponse : DONNEES_INSUFFISANTES – les données couvrent seulement les saisons 2024 à 2026, aucune information n’est disponible pour la saison prochaine.

## Questions instables d'une passe à l'autre

| Question | Famille | Taux de réussite |
|---|---|---|
| F03 | factuelle | 50.0 % |
| F16 | factuelle | 75.0 % |
| C04 | comparative | 75.0 % |
| C09 | comparative | 25.0 % |
| P12 | piege | 50.0 % |

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
