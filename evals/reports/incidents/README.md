# Rapports non publiés comme mesure

Ces rapports sont gardés pour la traçabilité, pas pour être lus comme des résultats.
Ils sont hors du dossier `reports/` principal, qui sert de référence aux écarts
calculés par `evals/run.py` et au bandeau de la page Agent.

## 2026-09-14-1228 : quota journalier du fournisseur épuisé

- 40 questions × 5 passes prévues, `openai/gpt-oss-120b` via Groq, palier gratuit
- Limite atteinte au milieu de la passe 2 : 200 000 jetons par jour, par modèle, dont environ
  160 000 déjà consommés par l'exécution du matin
- 70 % des réponses sont des refus du fournisseur (HTTP 429). Exactitude agrégée : 27,5 %, sans
  signification
- Reconstitution publiée dans le README : passe 1 complète à 36/40, passe 2 à 19/20 sur les
  questions servies (F01 à C04), F10 seule question instable. Elle se vérifie sur les compteurs
  `par_famille.verdicts` et `echecs[].taux_reussite` du JSON
