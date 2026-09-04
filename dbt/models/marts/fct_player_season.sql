-- La table de faits centrale : une ligne par joueur, saison, split et équipe.
--
-- Le grain est important et il n'est pas évident : six joueurs ont changé
-- d'équipe en cours de split en 2025, et ont donc deux lignes pour la même
-- saison. Retirer `team_name` de la clé ferait disparaître une demi-saison
-- bien réelle, et le test d'unicité en aval est là pour empêcher qu'on le
-- fasse par mégarde.

with players as (

    select * from {{ ref('stg_players') }}

),

clusters as (

    select * from {{ ref('stg_clusters') }}

),

joined as (

    select
        players.player_id,
        players.player_display_name,
        players.league,
        players.season,
        players.split,
        players.position,
        players.team_name,

        players.talent_score,
        players.score_percentile,
        players.promoted_to_lec,

        players.games_played,
        players.win_rate,
        players.champion_pool_size,

        players.dpm_zscore,
        players.cspm_zscore,
        players.gold_diff_at_15_zscore,
        clusters.win_rate_zscore,

        clusters.cluster_id,
        clusters.archetype,
        clusters.umap_x,
        clusters.umap_y,

        -- Un échantillon trop court rend le score ininterprétable. Le seuil de
        -- 10 matchs n'est pas arbitraire : en dessous, le classement brut est
        -- dominé par des joueurs à trois matchs. Le marquer ici évite que
        -- chaque consommateur réinvente son propre seuil dans son coin.
        players.games_played >= 10                          as has_reliable_sample

    from players
    left join clusters
        on  players.player_id = clusters.player_id
        and players.league    = clusters.league
        and players.season    = clusters.season
        and players.position  = clusters.position
        and players.team_name = clusters.team_name
        and coalesce(players.split, 'n/a') = coalesce(clusters.split, 'n/a')

)

select * from joined
