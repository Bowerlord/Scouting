-- Une ligne par joueur, agrégée sur toutes ses saisons.
--
-- Répond à la question qu'un recruteur pose en premier et à laquelle la table
-- de faits ne répond pas directement : « qui est ce joueur, sur la durée ».

with seasons as (

    select * from {{ ref('fct_player_season') }}

),

aggregated as (

    select
        player_id,
        max(player_display_name)                            as player_display_name,

        count(*)                                            as season_count,
        min(season)                                         as first_season,
        max(season)                                         as last_season,

        -- Un joueur peut changer de poste ou de ligue. On retient la valeur la
        -- plus récente, celle qui décrit ce qu'il est aujourd'hui.
        arg_max(position, season)                           as current_position,
        arg_max(league, season)                             as current_league,
        arg_max(team_name, season)                          as current_team,
        arg_max(archetype, season)                          as current_archetype,

        sum(games_played)                                   as total_games_played,
        max(talent_score)                                   as best_talent_score,
        arg_max(talent_score, season)                       as latest_talent_score,
        bool_or(promoted_to_lec)                            as ever_promoted_to_lec,
        bool_or(has_reliable_sample)                        as has_any_reliable_season

    from seasons
    group by player_id

)

select * from aggregated
