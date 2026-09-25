> Agrégat du journal des passes de nuit : 2 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure en cours** : 1 passe(s) valide(s) sur 5 pour la question la moins avancée. Ces chiffres ne sont pas encore publiables.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-09-25T06:11:15+00:00 — 40 questions × 1 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 90.0 % | — |
| Exactitude — factuelle (16 questions) | 93.8 % | — |
| Exactitude — comparative (12 questions) | 91.7 % | — |
| Exactitude — piege (12 questions) | 83.3 % | — |
| Refus correct sur les pièges | 83.3 % | — |
| **Refus à tort** | 0.0 % | — |
| **Hallucinations** | 5.0 % | — |
| Non-convergence | 2.5 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 0.0 % | — |
| Instabilité des chiffres cités | 0.0 % | — |
| Latence p50 | 17435 ms | — |
| Latence p95 | 69430 ms | — |
| Coût par question | 0.00078 € | — |
| Coût total de l'exécution | 0.0314 € | — |
| Appels d'outils par question | 1.62 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 34 |
| `get_leaderboard` | 20 |
| `list_filters` | 5 |
| `get_player` | 3 |
| `get_status` | 2 |
| `get_archetypes` | 1 |

## Les 4 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**F03** · factuelle · réussie 0.0 % des passes

> Combien de ligues différentes sont couvertes par ce jeu de données ?

- Attendu : `6.0`
- Verdict dominant : `faux`
- Outils appelés : `list_filters`
- Dernière réponse : Le jeu de données couvre six ligues différentes. Les ligues répertoriées sont : LFL, LFL2, LVP SL, NLC, PRM et TCL.

**C09** · comparative · réussie 0.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `non_convergence`
- Outils appelés : `search_players`, `list_filters`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`, `search_players`
- Dernière réponse : [banc] L'agent n'a pas convergé en 8 étapes sans produire de réponse finale.

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

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
