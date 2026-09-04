# ══════════════════════════════════════════════════════════════════════════════
# Image de l'API de scouting
#
# Construction en deux étages : le premier compile les dépendances, le second
# n'embarque que le résultat. L'étage final ne contient donc ni compilateur ni
# cache pip, ce qui réduit la taille et la surface d'attaque.
#
# Build :  docker build -t scouting-api .
# Run   :  docker run -p 8000:8000 scouting-api
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS builder

WORKDIR /build

# Les dépendances sont copiées seules et installées avant le code : tant que ce
# fichier ne change pas, Docker réutilise le cache de cette couche même quand
# le code change, ce qui divise le temps de build.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


FROM python:3.11-slim AS runtime

# Les octets Python compilés n'apportent rien dans un conteneur jetable, et la
# sortie non tamponnée est nécessaire pour que les logs sortent en temps réel.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Un utilisateur non privilégié : si l'application est compromise, l'attaquant
# n'est pas root dans le conteneur.
RUN useradd --create-home --uid 1000 scouting

WORKDIR /app

COPY --from=builder /install /usr/local
COPY api/ ./api/
COPY reports/metrics/ ./reports/metrics/

USER scouting

EXPOSE 8000

# La sonde interroge la vraie route de santé, celle qui vérifie que les données
# sont chargées. Un conteneur qui répond mais n'a pas ses données est marqué
# malsain et sorti du pool, au lieu de servir des erreurs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# La forme shell est nécessaire ici pour que ${PORT} soit interprété : les
# hébergeurs de conteneurs (Cloud Run, Railway, Fly) imposent le port par
# variable d'environnement.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
