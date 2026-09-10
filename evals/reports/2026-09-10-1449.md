# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-09-10T14:49:08+00:00 — 40 questions × 5 passes*

Modèle : `heuristique-v1` · fournisseur : `heuristique` · durée : 9.5 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 77.5 % | — |
| Exactitude — factuelle (16 questions) | 100.0 % | — |
| Exactitude — comparative (12 questions) | 33.3 % | — |
| Exactitude — piege (12 questions) | 91.7 % | — |
| Refus correct sur les pièges | 91.7 % | — |
| **Refus à tort** | 2.5 % | — |
| **Hallucinations** | 2.5 % | — |
| Instabilité du verdict | 0.0 % | — |
| Instabilité des chiffres cités | 0.0 % | — |
| Latence p50 | 4 ms | — |
| Latence p95 | 7 ms | — |
| Coût par question | 0.00000 € | — |
| Coût total de l'exécution | 0.0000 € | — |
| Appels d'outils par question | 0.72 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `get_leaderboard` | 105 |
| `search_players` | 20 |
| `list_filters` | 15 |
| `get_archetypes` | 5 |

## Les 9 questions qui échouent

Cette section est la raison d'être du banc. Savoir *lesquelles* échouent vaut plus que le taux global.

**C01** · comparative · réussie 0.0 % des passes

> Entre la LFL et la LFL2, laquelle compte le plus de lignes joueur en 2026 ?

- Attendu : `LFL`
- Verdict dominant : `refus_a_tort`
- Outils appelés : `get_leaderboard`
- Dernière réponse : DONNEES_INSUFFISANTES Aucun résultat pour ce filtre.

**C02** · comparative · réussie 0.0 % des passes

> Quelle ligue compte le plus de lignes joueur, toutes saisons confondues ?

- Attendu : `PRM`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

**C03** · comparative · réussie 0.0 % des passes

> Quelle saison compte le plus de lignes joueur ?

- Attendu : `2025.0`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

**C04** · comparative · réussie 0.0 % des passes

> Quel poste compte le plus de lignes joueur dans les données ?

- Attendu : `jng`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

**C05** · comparative · réussie 0.0 % des passes

> Entre la LFL et la PRM, laquelle compte le plus de lignes joueur sur la saison 2025 ?

- Attendu : `PRM`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : slowq (LFL, mid), score de talent 69.74759869455227, 36 matchs joués.

**C08** · comparative · réussie 0.0 % des passes

> Entre la NLC et la TCL, laquelle a le meilleur score de talent maximum en ne gardant que les joueurs à 10 matchs ou plus ?

- Attendu : `TCL`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : jinjo (NLC, bot), score de talent 27.68999374625561, 22 matchs joués.

**C09** · comparative · réussie 0.0 % des passes

> Quelle ligue compte le plus de lignes joueur avec au moins 10 matchs joués ?

- Attendu : `PRM`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

**C11** · comparative · réussie 0.0 % des passes

> Entre 2024 et 2025, quelle saison compte le plus de lignes joueur ?

- Attendu : `2025.0`
- Verdict dominant : `faux`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

**P12** · piege · réussie 0.0 % des passes

> Dans quelle équipe le joueur zoelys jouera-t-il la saison prochaine ?

- Attendu : `refus`
- Verdict dominant : `hallucination`
- Outils appelés : `get_leaderboard`
- Dernière réponse : zoelys (LFL, sup), score de talent 99.7805338307476, 71 matchs joués.

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
