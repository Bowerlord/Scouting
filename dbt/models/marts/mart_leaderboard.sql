-- Le classement prêt à servir, celui que l'API et le dashboard consomment.
--
-- Il n'inclut que les saisons à l'échantillon suffisant. Ce choix est fait ici,
-- une fois, plutôt que dans chaque client : un classement dont la tête est
-- occupée par des joueurs à trois matchs n'est pas un classement, et laisser
-- chaque consommateur poser son propre filtre garantit qu'ils afficheront des
-- chiffres différents pour la même question.

with reliable as (

    select * from {{ ref('fct_player_season') }}
    where has_reliable_sample

),

ranked as (

    select
        player_id,
        player_display_name,
        league,
        season,
        split,
        position,
        team_name,
        archetype,
        talent_score,
        score_percentile,
        games_played,
        win_rate,
        promoted_to_lec,

        row_number() over (order by talent_score desc)                        as overall_rank,
        row_number() over (partition by position order by talent_score desc)  as position_rank,
        row_number() over (partition by league   order by talent_score desc)  as league_rank

    from reliable

)

select * from ranked
