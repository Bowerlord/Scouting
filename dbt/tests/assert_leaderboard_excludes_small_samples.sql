-- Le classement ne doit contenir que des saisons à l'échantillon suffisant.
-- Ce test protège une décision produit, pas une contrainte technique : si
-- quelqu'un retire le filtre du modèle un jour, le build le dira.

select
    player_id,
    season,
    games_played
from {{ ref('mart_leaderboard') }}
where games_played < 10
