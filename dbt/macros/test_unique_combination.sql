{#
  Test générique d'unicité sur une combinaison de colonnes.

  dbt-utils fournit ce test, mais l'installer ajouterait un gestionnaire de
  paquets et un dossier dbt_packages à un projet qui n'a besoin que de ce
  seul contrôle. Vingt lignes de SQL coûtent moins cher qu'une dépendance.

  Le test échoue s'il existe au moins une combinaison présente plus d'une fois.
#}

{% test dbt_utils_unique_combination(model, combination_of_columns) %}

with validation as (

    select
        {{ combination_of_columns | join(", ") }},
        count(*) as occurrences
    from {{ model }}
    group by {{ combination_of_columns | join(", ") }}
    having count(*) > 1

)

select * from validation

{% endtest %}
