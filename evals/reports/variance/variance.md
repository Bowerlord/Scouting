> Agrégat du journal des passes de nuit : 6 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure en cours** : 3 passe(s) valide(s) sur 5 pour la question la moins avancée. Ces chiffres ne sont pas encore publiables.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-09-29T06:53:56+00:00 — 40 questions × 3 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 91.7 % | — |
| Exactitude — factuelle (16 questions) | 95.8 % | — |
| Exactitude — comparative (12 questions) | 91.7 % | — |
| Exactitude — piege (12 questions) | 86.1 % | — |
| Refus correct sur les pièges | 86.1 % | — |
| **Refus à tort** | 0.8 % | — |
| **Hallucinations** | 4.2 % | — |
| Non-convergence | 1.7 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 12.5 % | — |
| Instabilité des chiffres cités | 55.0 % | — |
| Latence p50 | 16084 ms | — |
| Latence p95 | 78218 ms | — |
| Coût par question | 0.00078 € | — |
| Coût total de l'exécution | 0.0934 € | — |
| Appels d'outils par question | 1.59 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 100 |
| `get_leaderboard` | 58 |
| `list_filters` | 15 |
| `get_player` | 9 |
| `get_status` | 6 |
| `get_archetypes` | 3 |

## Les 6 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**F03** · factuelle · réussie 67.0 % des passes

> Combien de ligues différentes sont couvertes par ce jeu de données ?

- Attendu : `6.0`
- Verdict dominant : `juste`
- Outils appelés : `list_filters`
- Dernière réponse : Il y a 6 ligues différentes couvertes par le jeu de données. (LFL, LFL2, LVP SL, NLC, PRM et TCL).

**F16** · factuelle · réussie 67.0 % des passes

> Quel est le meilleur joueur de PRM toutes saisons et tous postes, avec au moins 10 matchs ?

- Attendu : `irrelevant`
- Verdict dominant : `juste`
- Outils appelés : `get_leaderboard`
- Dernière réponse : Le meilleur joueur de PRM toutes saisons et tous postes, avec au moins 10 matchs, est : irrelevant. Son talent_score est de 89.06741055112042 (score_percentile = 99.33184855233851) et il a joué 46 matchs.

**C04** · comparative · réussie 67.0 % des passes

> Quel poste compte le plus de lignes joueur dans les données ?

- Attendu : `jng`
- Verdict dominant : `juste`
- Outils appelés : `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : Le poste qui compte le plus de lignes joueur dans les données est le jungle, avec un total de 457 joueurs. Aucun autre poste n’atteint ce nombre.

**C09** · comparative · réussie 33.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `non_convergence`
- Outils appelés : `list_filters`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : La LFL compte le plus de lignes joueur avec au moins 10 matchs joués, avec 327 joueurs.

**P10** · piege · réussie 0.0 % des passes

> Combien de joueurs distincts, et non de lignes joueur-saison, le jeu de données contient-il ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_status`
- Dernière réponse : Le jeu de données contient 2259 joueurs distincts. Cette valeur représente le nombre total de profils uniques, sans compter les lignes séparées par saison.

**P12** · piege · réussie 33.0 % des passes

> Dans quelle équipe le joueur zoelys jouera-t-il la saison prochaine ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_player`
- Dernière réponse : Le joueur zoelys sera dans l’équipe des Galions pour la saison prochaine.

## Questions instables d'une passe à l'autre

| Question | Famille | Taux de réussite |
|---|---|---|
| F03 | factuelle | 67.0 % |
| F16 | factuelle | 67.0 % |
| C04 | comparative | 67.0 % |
| C09 | comparative | 33.0 % |
| P12 | piege | 33.0 % |

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
