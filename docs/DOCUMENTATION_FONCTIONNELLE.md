# KCorp Scouting Tool — Documentation Fonctionnelle

> Ce document explique le projet **sans prérequis technique**. Si vous savez ce
> qu'est League of Legends, vous avez tout ce qu'il faut. Les termes marqués
> d'une étoile* sont définis dans le [glossaire](#8-glossaire) en fin de document.
> Pour le détail des algorithmes et du code, voir [DOCUMENTATION_TECHNIQUE.md](DOCUMENTATION_TECHNIQUE.md).

---

## 1. Le projet en une page

**Le problème.** Chaque année, les équipes de la LEC* (le championnat d'Europe de
League of Legends) cherchent leurs futurs joueurs dans les ligues régionales —
la LFL en France, la Prime League en Allemagne, etc. Repérer « la pépite » parmi
des centaines de joueurs est un travail long, subjectif, et les scouts ne
peuvent pas regarder tous les matchs de toutes les ligues.

**La solution.** Cet outil analyse automatiquement les statistiques de tous les
matchs professionnels des ligues régionales européennes (plus de 40 000 lignes de
données sur 2024-2026) et produit pour chaque joueur :

1. **Un Talent Score** — une note qui estime ses chances de monter en LEC dans
   les 18 prochains mois, calculée par un modèle d'apprentissage automatique*
   entraîné sur les promotions passées ;
2. **Un style de jeu (archétype)** — « carry agressif », « contrôleur de
   vision »… déterminé en regroupant les joueurs aux profils statistiques
   similaires ;
3. **Un dashboard** — un site web interactif pour explorer ces résultats :
   classements, fiches joueur, recherche de profils similaires.

**Pour qui ?** Un directeur sportif qui prépare son mercato, un scout qui veut
prioriser ses visionnages, un analyste ou un fan curieux.

**Ce que l'outil n'est pas.** Une boule de cristal. Il dit « ce joueur a le
profil statistique des joueurs qui ont été promus par le passé » — pas « ce
joueur sera promu ». La section [6](#6-ce-que-loutil-ne-sait-pas-faire) détaille
honnêtement ses limites.

---

## 2. Le contexte esport en deux minutes

Le League of Legends professionnel européen est organisé en pyramide :

```
                    ┌─────────────┐
                    │     LEC     │   ← L'élite européenne (10 équipes).
                    └──────▲──────┘      C'est ici que tout le monde veut jouer.
                           │  promotion (recrutement)
        ┌──────────────────┴──────────────────┐
        │            ERLs (Div 1)             │   ← Les ligues régionales :
        │   LFL 🇫🇷  PRM 🇩🇪  LVP SL 🇪🇸        │      un championnat par pays.
        │   NLC 🇬🇧  TCL 🇹🇷                    │      Le vivier de talents.
        └──────────────────▲──────────────────┘
                           │
                    ┌──────┴──────┐
                    │  LFL2 (Div 2)│  ← Division inférieure française.
                    └─────────────┘
```

Il n'y a pas de montée/descente sportive automatique entre ERL et LEC : un
joueur « monte » quand une équipe LEC le **recrute**. C'est précisément cet
événement — le recrutement d'un joueur d'ERL par la LEC — que le projet cherche
à anticiper. Sur la période étudiée, **environ 6 joueurs d'ERL sur 100** font le
saut : c'est un événement rare, ce qui rend la prédiction difficile (et
intéressante).

Chaque saison est découpée en **splits** (Spring, Summer…). L'outil évalue les
joueurs **par split** : la même personne peut donc avoir plusieurs « fiches »,
une par demi-saison, ce qui permet de suivre sa progression.

---

## 3. Le Talent Score, expliqué simplement

### D'où vient la note ?

Le modèle a « appris » en observant les données 2024 : pour chaque joueur d'ERL,
il connaissait ses statistiques (dégâts par minute, contrôle de la vision,
avance à 15 minutes de jeu, variété de champions joués, taux de victoire,
nombre de matchs joués…) **et** s'il a été recruté en LEC dans les 18 mois qui
ont suivi. Le modèle a cherché quelles combinaisons de statistiques distinguent
les futurs promus des autres. On a ensuite vérifié qu'il fonctionne sur des
données qu'il n'avait jamais vues (2025) — comme un examen avec des questions
nouvelles.

Détail important : les statistiques sont toujours comparées **entre joueurs du
même poste, de la même ligue et du même split**. Faire 500 dégâts par minute en
LFL2 n'a pas la même valeur qu'en LFL, et un support aura toujours moins de CS*
qu'un ADC. L'outil mesure donc « à quel point ce joueur domine SON
environnement », pas des chiffres bruts.

### Comment lire la note ?

- **Le Talent Score est une probabilité sur 100.** Un score de 40 signifie :
  « parmi les joueurs qui ont ce profil statistique, environ 40 % ont été
  promus en LEC dans les 18 mois ».
- **Les notes paraissent basses — c'est normal.** Comme seuls ~6 % des joueurs
  montent, même les tout meilleurs profils plafonnent vers 60-70/100. Un score
  de 30 est déjà **cinq fois** le taux de base : c'est un signal fort.
- **Le percentile est là pour ça.** À côté du score, l'outil affiche le rang du
  joueur au sein de son poste : « Top 3 % des midlaners ERL » est immédiatement
  parlant, quel que soit le niveau absolu des probabilités.

### Que veut dire la coche « Promu LEC » ?

Dans les tableaux, ✅ indique que le joueur a **réellement** débuté en LEC dans
les 18 mois suivant ce split (c'est la « vérité terrain » historique). Elle sert
à juger l'outil : les ✅ doivent se concentrer en haut des classements. Un ❌ en
haut de classement n'est pas forcément une erreur — c'est peut-être une pépite
pas encore recrutée, ou dont la promotion est trop récente pour figurer dans
les données.

---

## 4. Les styles de jeu (archétypes)

Deux joueurs peuvent avoir le même niveau avec des manières de jouer opposées.
Pour capturer cela, l'outil regroupe les joueurs de chaque poste en **familles
de profils statistiques** (3 familles par poste), sans a priori humain : c'est
la machine qui trouve les regroupements naturels dans les données
(technique dite de *clustering*\*).

Chaque famille reçoit ensuite un libellé lisible selon ses points forts moyens :

| Libellé | Ce que ça veut dire concrètement |
|---|---|
| Carry agressif | Beaucoup de dégâts ET très impliqué dans les kills de l'équipe. |
| High DPS | Dégâts par minute nettement au-dessus de la moyenne du poste. |
| Farmer dominant | Excellent pour accumuler CS* et or. |
| Lane bully | Prend l'avantage en CS sur son adversaire direct en début de partie. |
| Early dominant | Avance en or ET en expérience à 15 minutes. |
| Vision controller | Score de vision (poser/détruire des balises) très élevé. |
| Versatile | Joue un très grand nombre de champions différents. |
| High performer | Taux de victoire nettement supérieur à la moyenne. |
| Profil neutre | Aucune statistique ne se détache — profil dans la moyenne. |

Un même groupe peut cumuler plusieurs libellés (« High DPS | Lane bully | Early
dominant »). Fait notable : dans toutes les positions, **la famille au profil
agressif/dominant en début de partie concentre 2 à 4 fois plus de promotions
LEC** que les profils neutres — la LEC recrute des joueurs qui dominent leur
lane.

À savoir : les frontières entre familles sont **statistiquement floues** (les
joueurs forment un continuum plus que des îlots). Traitez l'archétype comme une
tendance descriptive, pas comme une étiquette rigide.

---

## 5. Guide du dashboard, page par page

Le dashboard s'ouvre dans un navigateur (en local : `make app`, ou via le lien
Streamlit Cloud du projet). Trois pages, accessibles dans le menu de gauche.

### 📊 Leaderboard — « Qui sont les meilleurs prospects ? »

Le classement général de tous les joueurs ERL par Talent Score.
- **Filtres** (barre latérale) : poste, ligue, année, score minimum.
- Chaque ligne = un joueur sur un split, avec score, percentile, taux de
  victoire, matchs joués, taille du champion pool, et la coche « Promu LEC ».
- En bas, un graphique du top 20 coloré par ligue.

*Cas d'usage : « Donne-moi les 10 meilleurs junglers hors LFL en 2025. »*

### 👤 Profil Joueur — « Que vaut ce joueur précisément ? »

Tapez un pseudo pour ouvrir sa fiche (sa meilleure saison est affichée) :
- Ses métriques clés : Talent Score, percentile (« Top X % des MID ERL »),
  statut de promotion, win rate, matchs joués, champion pool.
- Son **archétype** de style de jeu.
- Un **radar** de ses forces/faiblesses relatives : chaque axe est une
  statistique comparée à la moyenne de son poste (le pointillé = joueur moyen ;
  plus la surface dépasse le pointillé, plus le joueur domine).
- L'historique de toutes ses saisons, pour voir sa progression.

*Cas d'usage : « Le coach a entendu parler de X — sortez-moi sa fiche pour la
réunion de 14 h. »*

### 🔍 Scout Mode — « Trouve-moi un profil précis »

Deux outils pour un besoin de recrutement ciblé :
1. **Shortlist par critères** : croisez poste + archétype + ligue + score
   minimum → la liste des joueurs qui cochent toutes les cases.
   *« Je cherche un support orienté vision en LFL avec un score ≥ 30. »*
2. **Joueurs similaires** : choisissez un joueur de référence → l'outil liste
   tous les joueurs du même poste ayant le même style de jeu (même famille
   statistique), classés par Talent Score, avec un graphique comparatif.
   *« On ne peut pas se payer le joueur A — qui joue comme lui, en moins
   cher/moins connu ? »*

---

## 6. Ce que l'outil ne sait PAS faire

À garder en tête avant toute décision basée sur les chiffres :

1. **Il ne voit que les statistiques de matchs officiels.** Ni la solo queue*,
   ni le comportement en équipe, ni la personnalité, ni la maîtrise de
   l'anglais, ni l'âge ou le statut contractuel — autant de facteurs décisifs
   dans un vrai recrutement.
2. **Les saisons récentes sont défavorisées.** Un joueur brillant du dernier
   split n'a « pas encore eu le temps » d'être promu : les données ne peuvent
   pas encore contenir sa promotion, donc le modèle a appris sur des exemples
   où ces cas comptent comme « non promu ». Ses vrais chiffres sont
   probablement meilleurs que son score.
3. **Corrélation n'est pas garantie.** Le modèle reproduit les recrutements
   passés de la LEC, avec leurs éventuels biais (préférence pour la LFL, pour
   les profils agressifs…). Si la LEC recrutait mal, l'outil apprend à
   « prédire des recrutements », pas à « détecter le talent absolu ».
4. **La précision reste modeste.** Sur les données test, environ 1 joueur sur 7
   du haut du classement a réellement été promu — nettement mieux que le hasard
   (1 sur 16), mais loin d'une certitude. L'outil **priorise le travail des
   scouts**, il ne le remplace pas.
5. **Les joueurs sont identifiés par leur pseudo.** Un changement de pseudo, ou
   deux joueurs portant le même, peuvent brouiller l'historique.
6. **Peu de matchs = prudence.** Les joueurs avec moins de 5 matchs dans un
   split sont exclus, mais un score calculé sur 6 matchs reste plus fragile
   qu'un score sur 30.

---

## 7. D'où viennent les données ?

- **Oracle's Elixir** (oracleselixir.com) : la référence communautaire des
  statistiques esport LoL. Le projet télécharge ses exports de matchs
  2024-2026 pour toutes les ligues (seules les 7 ligues cibles sont
  conservées : LEC, LFL, LFL2, PRM, LVP SL, NLC, TCL).
- Les données sont mises à jour **manuellement** : quelqu'un doit relancer le
  pipeline (`make run-pipeline`) puis publier les résultats pour que le
  dashboard reflète les derniers splits.
- Le dashboard public ne recalcule rien : il lit des résultats pré-calculés,
  ce qui le rend rapide et gratuit à héberger.

---

## 8. Glossaire

| Terme | Définition |
|---|---|
| **LEC** | League of Legends EMEA Championship — le championnat d'élite européen. |
| **ERL** | European Regional League — ligue régionale (LFL, PRM…), l'antichambre de la LEC. |
| **Split** | Demi-saison de compétition (Spring, Summer…). |
| **Promotion** | Recrutement d'un joueur d'ERL par une équipe LEC. |
| **Apprentissage automatique (ML)** | Programme qui déduit des règles à partir d'exemples passés au lieu d'être programmé règle par règle. |
| **Talent Score** | Note sur 100 = probabilité estimée qu'un joueur soit promu en LEC dans les 18 mois suivant le split observé. |
| **Percentile** | Rang relatif : « percentile 97 » = meilleur que 97 % des joueurs de son poste. |
| **Archétype / Clustering** | Famille de joueurs au profil statistique similaire, découverte automatiquement par regroupement des données. |
| **CS** | Creep Score — nombre de sbires/monstres tués, mesure de la capacité à accumuler des ressources. |
| **DPM / VSPM** | Dégâts par minute / Score de vision par minute. |
| **Diff@15** | Avance (or, CS ou expérience) sur l'adversaire direct à la 15ᵉ minute. |
| **Champion pool** | Nombre de champions différents joués — mesure la polyvalence. |
| **Solo queue** | Parties classées jouées individuellement, hors compétition officielle (non couvertes par l'outil). |
| **Win rate** | Pourcentage de matchs gagnés. |

---

## 9. FAQ

**Le score de mon joueur préféré est bas alors qu'il est très bon. Pourquoi ?**
Trois explications possibles : (1) ses statistiques *relatives à son poste et sa
ligue* sont moins dominantes qu'il n'y paraît à l'œil ; (2) son split est récent
et souffre de l'effet « pas encore eu le temps d'être promu » ; (3) le modèle
valorise beaucoup le volume de matchs joués — un remplaçant est pénalisé.

**Pourquoi le meilleur joueur n'a-t-il « que » 70/100 ?**
Parce que le score est une probabilité honnête : même un profil parfait n'est
jamais certain d'être recruté (places limitées, facteurs hors statistiques).
Utilisez le percentile pour comparer les joueurs entre eux.

**Un joueur apparaît plusieurs fois dans le classement. C'est un bug ?**
Non : chaque ligne est un couple joueur × split. Cela permet de voir l'évolution
d'un joueur d'une demi-saison à l'autre.

**L'outil couvre-t-il d'autres régions (Corée, Amérique) ou la solo queue ?**
Pas actuellement. Le périmètre est : ERLs européennes → LEC, matchs officiels
uniquement. Ce sont des extensions envisagées dans la feuille de route.

**À quelle fréquence les données sont-elles à jour ?**
À chaque exécution manuelle du pipeline. La date des données correspond au
dernier split présent dans les exports Oracle's Elixir téléchargés.
