-- Scores de talent, nettoyés et renommés.
--
-- Deux corrections faites ici plutôt qu'en aval :
--   1. `_source_year` devient `season`. Le préfixe souligné est une convention
--      interne du pipeline et n'a rien à faire dans un modèle exposé.
--   2. `split` vaut la chaîne 'unknown' quand l'information manque, ce qui
--      fausse tout dénombrement par split. On la ramène à NULL, qui est la
--      façon correcte de dire « non renseigné » en SQL.

with source as (

    select * from {{ source('pipeline', 'talent_scores') }}

),

renamed as (

    select
        lower(trim(playername))                             as player_id,
        playername_original                                 as player_display_name,
        league,
        cast(_source_year as integer)                       as season,
        nullif(split, 'unknown')                            as split,
        position,
        teamname                                            as team_name,

        cast(talent_score as double)                        as talent_score,
        cast(score_percentile as double)                    as score_percentile,
        cast(promoted_to_lec as boolean)                    as promoted_to_lec,

        cast(win_rate as double)                            as win_rate,
        cast(games_played as integer)                       as games_played,
        cast(champion_pool_size as integer)                 as champion_pool_size,

        cast(dpm_zscore as double)                          as dpm_zscore,
        cast(cspm_zscore as double)                         as cspm_zscore,
        cast(golddiffat15_zscore as double)                 as gold_diff_at_15_zscore

    from source

)

select * from renamed
