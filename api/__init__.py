"""API REST du Scouting Tool.

Expose en HTTP les résultats du pipeline ML (scores de talent, clustering,
archétypes) qui n'étaient jusqu'ici lisibles que par le dashboard Streamlit.

Pourquoi une API : le dashboard et le pipeline partageaient le même processus,
ce qui rendait les résultats inutilisables par autre chose que Streamlit.
L'API sépare la lecture des données de leur présentation, et devient le point
d'entrée unique pour le dashboard, l'agent en langage naturel et le serveur MCP.
"""

__version__ = "1.0.0"
