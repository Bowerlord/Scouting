-- Résultats du clustering, nettoyés et alignés sur les mêmes clés que
-- stg_players pour que la jointure en aval soit exacte.

with source as (

    select * from {{ source('pipeline', 'clustering') }}

),

renamed as (

    select
        lower(trim(playername))                             as player_id,
        league,
        cast(_source_year as integer)                       as season,
        nullif(split, 'unknown')                            as split,
        position,
        teamname                                            as team_name,

        cast(cluster as integer)                            as cluster_id,
        archetype                                           as archetype,
        cast(umap_x as double)                              as umap_x,
        cast(umap_y as double)                              as umap_y,
        cast(win_rate_zscore as double)                     as win_rate_zscore

    from source

)

select * from renamed
