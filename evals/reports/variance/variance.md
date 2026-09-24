> Agrégat du journal des passes de nuit : 1 nuit(s), 0 réponse(s) perdue(s) sur panne du fournisseur, écartée(s) et reposée(s).
>
> **Mesure en cours** : 0 passe(s) valide(s) sur 5 pour la question la moins avancée. Ces chiffres ne sont pas encore publiables.

# Banc d'évaluation de l'agent ERL Scout

*Exécuté le 2026-09-24T17:41:58+00:00 — 20 questions × 0 passes*

Modèle : `openai/gpt-oss-120b` · fournisseur : `groq` · durée : 0.0 s

## Les chiffres

| Mesure | Valeur | Écart au rapport précédent |
|---|---|---|
| **Exactitude globale** | 90.0 % | — |
| Exactitude — factuelle (8 questions) | 87.5 % | — |
| Exactitude — comparative (6 questions) | 83.3 % | — |
| Exactitude — piege (6 questions) | 100.0 % | — |
| Refus correct sur les pièges | 100.0 % | — |
| **Refus à tort** | 0.0 % | — |
| **Hallucinations** | 0.0 % | — |
| Non-convergence | 5.0 % | — |
| Pannes du fournisseur | 0.0 % | — |
| Instabilité du verdict | 0.0 % | — |
| Instabilité des chiffres cités | 0.0 % | — |
| Latence p50 | 16408 ms | — |
| Latence p95 | 50815 ms | — |
| Coût par question | 0.00076 € | — |
| Coût total de l'exécution | 0.0153 € | — |
| Appels d'outils par question | 1.55 | — |
| Appels d'outils en erreur | 0 | — |

### Comment lire ces chiffres

**L'exactitude et le refus à tort se lisent ensemble.** Un agent qui répondrait « données insuffisantes » à tout obtiendrait 100 % sur les pièges et 0 % ailleurs : c'est le taux de refus à tort qui le démasque.

**Une hallucination est une réponse affirmative à une question sans réponse.** C'est le défaut le plus coûteux d'un système de ce genre, parce qu'il ne se voit pas : la réponse est bien formée, plausible, et fausse.

### Outils appelés

| Outil | Appels |
|---|---|
| `search_players` | 19 |
| `get_leaderboard` | 6 |
| `list_filters` | 2 |
| `get_player` | 2 |
| `get_status` | 1 |
| `get_archetypes` | 1 |

## Les 2 questions qui échouent

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

---

*Vérité terrain calculée en SQL par DuckDB sur les fichiers du pipeline, par un chemin indépendant de celui qu'emprunte l'agent. Détail des questions : `evals/questions.yaml`.*
