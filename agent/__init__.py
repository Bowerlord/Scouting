"""Agent en langage naturel sur les données de scouting.

Ce paquet n'est pas un chatbot. Il existe pour être mesuré : tout ce qu'il
produit est traçable (quels outils appelés, avec quels arguments) et chiffré
(jetons, coût, latence), parce que c'est le banc d'évaluation de `evals/` qui
donne sa valeur à l'ensemble.
"""

from agent.agent import Answer, ScoutAgent
from agent.providers import ProviderError, get_provider

__all__ = ["Answer", "ScoutAgent", "ProviderError", "get_provider"]
__version__ = "0.1.0"
