"""Serveur MCP du Scouting Tool.

Expose les données de scouting comme outils utilisables par n'importe quel
agent compatible MCP (Claude Code, Claude Desktop, et les autres clients du
protocole). L'agent interroge les joueurs, les archétypes et le classement en
langage naturel, sans écrire une ligne de SQL ni connaître le schéma.

Le serveur ne réimplémente rien : il appelle l'API REST du projet quand elle
est joignable, et retombe sur la lecture directe des résultats sinon.
"""

__version__ = "1.0.0"
