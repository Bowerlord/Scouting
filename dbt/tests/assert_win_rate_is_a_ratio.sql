-- Le taux de victoire doit être un ratio entre 0 et 1.
-- Une valeur supérieure à 1 trahirait un pourcentage mal converti en amont,
-- l'erreur la plus courante et la plus silencieuse sur ce genre de colonne.

select
    player_id,
    season,
    win_rate
from {{ ref('fct_player_season') }}
where win_rate < 0
   or win_rate > 1
