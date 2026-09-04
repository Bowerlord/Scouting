-- Un percentile en dehors de 0-100 signifie que le calcul amont est cassé.
-- Ce test est écrit à la main plutôt qu'avec un test générique parce qu'il
-- porte sur une propriété métier, pas sur une contrainte de colonne.

select
    player_id,
    season,
    score_percentile
from {{ ref('fct_player_season') }}
where score_percentile < 0
   or score_percentile > 100
