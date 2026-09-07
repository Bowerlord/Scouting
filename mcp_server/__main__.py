"""Point d'entrée du serveur MCP.

    python -m mcp_server

Le transport est stdio : c'est celui qu'attendent Claude Code et Claude Desktop
pour un serveur local. Rien n'est écrit sur la sortie standard en dehors du
protocole lui-même, sinon le client rejette la connexion ; les messages de
diagnostic partent donc sur l'erreur standard.
"""

from __future__ import annotations

import sys

from mcp_server import resources
from mcp_server.server import backend, server


def main() -> None:
    resources.register(server)

    print(
        f"[scouting-mcp] source des données : {backend.mode}"
        + (f" ({backend.api_url})" if backend.mode == "api" else " (fichiers du dépôt)"),
        file=sys.stderr,
    )

    try:
        server.run(transport="stdio")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
