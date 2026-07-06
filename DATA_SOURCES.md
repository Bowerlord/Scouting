# 📊 Sources de Données — KCorp Scouting Tool

Ce document décrit les sources de données utilisées dans le projet, leur format, leur couverture, et les choix techniques associés.

---

## 1. Oracle's Elixir (Source Principale)

| | |
|---|---|
| **URL** | [oracleselixir.com/tools/downloads](https://oracleselixir.com/tools/downloads) |
| **Format** | CSV téléchargeables depuis Google Drive |
| **Contenu** | Stats match-par-match de toutes les ligues professionnelles et ERLs |
| **Granularité** | ~100+ colonnes par ligne (1 ligne = 1 joueur dans 1 match) |
| **Couverture temporelle** | Depuis 2014. **Nous utilisons 2024-2026 uniquement.** |
| **Licence** | Gratuit pour usage personnel/éducatif |
| **Taille approximative** | ~150-200 Mo par année |

### Colonnes utilisées

Les colonnes clés sont documentées dans `src/config.py` (variable `KEY_COLUMNS`). Les principales :

- **Identifiants** : `gameid`, `league`, `split`, `date`, `playername`, `teamname`
- **Résultat** : `result` (1 = victoire, 0 = défaite)
- **Stats de laning (early game)** : `csdiffat15`, `golddiffat15`, `xpdiffat15`
- **Stats globales** : `kills`, `deaths`, `assists`, `dpm`, `cspm`, `vspm`
- **Stats d'efficacité** : `damageshare`, `killparticipation`, `earnedgoldshare`
- **Durée** : `gamelength` (en secondes)

### Téléchargement

Les CSV sont hébergés sur Google Drive. Le module `src/data/downloader.py` gère :
- Le téléchargement automatique avec les IDs Google Drive
- Le cache local (pas de re-téléchargement si le fichier existe)
- La gestion des fichiers volumineux (confirmation anti-virus Google Drive)
- Le retry avec backoff exponentiel

---

## 2. Leaguepedia / LoL Fandom Wiki (Source Complémentaire)

| | |
|---|---|
| **URL** | [lol.fandom.com](https://lol.fandom.com) |
| **Accès** | API MediaWiki via `cargoquery` (requêtes HTTP `requests`, sans bibliothèque dédiée) |
| **Contenu** | Rosters, apparitions en match, résultats de tournois |
| **Rate limiting** | Respecter les limites de l'API (~1 requête/seconde) |

### Utilisation dans le projet

⚠️ **Rôle réduit depuis l'introduction de la target datée.** La target
`promoted_to_lec` est construite **uniquement depuis les dates de matchs
Oracle's Elixir** : un match ERL est positif si le joueur débute en LEC dans
les 18 mois qui suivent (voir `src/data/cleaner.py`,
`PROMOTION_HORIZON_MONTHS`). L'API Cargo ne fournissant pas de dates de
transfert exploitables, utiliser Leaguepedia comme label ré-introduirait la
fuite temporelle corrigée.

Leaguepedia sert donc de **cross-check informatif** (best effort) : si un cache
`data/external/leaguepedia_careers.json` existe, le nettoyage logge combien de
promus déclarés par Leaguepedia sont confirmés par la target datée. Le module
`src/data/leaguepedia.py` reste exécutable en CLI pour régénérer ce cache.

### Difficultés connues

- **Normalisation des noms** : Un même joueur peut avoir des noms différents entre Oracle's Elixir et Leaguepedia (ex: "Caps" vs "caps" vs "G2 Caps") — le matching se fait en minuscules, sans fuzzy matching.
- **Couverture** : Tous les transferts ne sont pas documentés sur Leaguepedia

---

## 3. Riot Games API (Source Optionnelle — Non implémentée dans le MVP)

| | |
|---|---|
| **URL** | [developer.riotgames.com](https://developer.riotgames.com) |
| **Contenu** | Stats Solo Queue, Ranked, Champion Mastery |
| **Limites** | Rate limit (20 req/s avec clé dev), nécessite les summoner names exacts |
| **Statut** | 🔜 Prévu pour une version future |

---

## Schéma du flux de données

```
Oracle's Elixir (CSV)                  Leaguepedia (API)
    │                                       │
    │  2024.csv, 2025.csv                   │  career_history, transfers
    │                                       │
    ▼                                       ▼
┌─────────────────────────────────────────────────┐
│              data/raw/                           │
│  (données brutes, non modifiées)                 │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼  cleaner.py
┌─────────────────────────────────────────────────┐
│              data/interim/                       │
│  (données nettoyées, filtrées par ligue,         │
│   noms normalisés, target variable ajoutée)      │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼  feature_engineering.py
┌─────────────────────────────────────────────────┐
│              data/processed/                     │
│  (feature matrices prêtes pour le ML,            │
│   1 ligne = 1 joueur × 1 split)                  │
└─────────────────────────────────────────────────┘
```
