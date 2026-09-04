-- Le taux de promotion en LEC par archétype de jeu.
--
-- C'est le modèle qui transforme une description statistique en information
-- de recrutement : il ne dit pas seulement à quoi ressemble un groupe de
-- joueurs, il dit lequel de ces profils débouche réellement sur une montée.

with seasons as (

    select * from {{ ref('fct_player_season') }}
    where cluster_id is not null

),

by_archetype as (

    select
        position,
        cluster_id,
        max(archetype)                                      as archetype,

        count(*)                                            as player_season_count,
        count(distinct player_id)                           as distinct_player_count,
        sum(case when promoted_to_lec then 1 else 0 end)    as promoted_count,

        round(avg(talent_score), 3)                         as avg_talent_score,
        round(avg(win_rate), 4)                             as avg_win_rate,
        round(avg(dpm_zscore), 4)                           as avg_dpm_zscore,
        round(avg(cspm_zscore), 4)                          as avg_cspm_zscore,
        round(avg(gold_diff_at_15_zscore), 4)               as avg_gold_diff_at_15_zscore

    from seasons
    group by position, cluster_id

)

select
    *,
    round(promoted_count * 1.0 / nullif(player_season_count, 0), 4) as promotion_rate
from by_archetype
