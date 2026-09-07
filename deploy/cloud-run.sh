#!/usr/bin/env bash
#
# Déploiement de l'API de scouting sur Google Cloud Run.
#
# Prérequis, à faire une fois par Alexandre (ils demandent un compte, donc ils
# ne peuvent pas être scriptés) :
#   1. Un projet Google Cloud avec la facturation activée
#   2. Le SDK gcloud installé : https://cloud.google.com/sdk/docs/install
#   3. gcloud auth login
#
# Ensuite :
#   ./deploy/cloud-run.sh <ID_DU_PROJET>
#
# Coût attendu : zéro. Le palier gratuit de Cloud Run couvre 2 millions de
# requêtes par mois et l'API n'en servira que quelques centaines. Le service
# est configuré pour descendre à zéro instance quand personne ne l'appelle,
# donc il ne facture rien au repos.

set -euo pipefail

PROJET="${1:-}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-scouting-api}"
DEPOT="${DEPOT:-scouting}"

if [[ -z "${PROJET}" ]]; then
  echo "Usage : $0 <ID_DU_PROJET>" >&2
  echo "Exemple : $0 scouting-lol-472103" >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJET}/${DEPOT}/${SERVICE}"

echo "==> Projet   : ${PROJET}"
echo "==> Région   : ${REGION}  (europe-west1 : Belgique, la plus proche de Paris)"
echo "==> Image    : ${IMAGE}"
echo

echo "==> Activation des API nécessaires"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJET}"

echo "==> Création du dépôt d'images (ignorée s'il existe déjà)"
gcloud artifacts repositories create "${DEPOT}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Images de l'API de scouting" \
  --project "${PROJET}" 2>/dev/null || echo "    dépôt déjà présent"

echo "==> Construction de l'image par Cloud Build"
# Construite à distance plutôt qu'en local : Docker n'est pas installé sur le
# poste, et Cloud Build produit une image amd64 quelle que soit la machine.
gcloud builds submit --tag "${IMAGE}:latest" --project "${PROJET}" .

echo "==> Déploiement sur Cloud Run"
# --allow-unauthenticated : l'API est en lecture seule sur des données
# publiques, elle n'expose aucune écriture et ne contient aucun secret.
# --min-instances=0 : le service s'éteint au repos, donc il ne coûte rien.
# --memory=512Mi : les résultats du pipeline pèsent moins d'un mégaoctet,
# l'essentiel de la mémoire va à pandas.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=3 \
  --memory=512Mi \
  --cpu=1 \
  --timeout=30s \
  --port=8000 \
  --project "${PROJET}"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJET}" --format='value(status.url)')"

echo
echo "==> Déployé sur ${URL}"
echo "==> Vérification de la santé du service"

# Un déploiement qui réussit ne prouve pas que le service sert les données.
# On interroge la route qui vérifie qu'elles sont chargées.
# Extraction sans dependance : `python` n'existe pas sur toutes les machines
# (sur celle-ci l'interpreteur s'appelle `py`), et exiger jq ajouterait une
# dependance pour lire un seul entier.
SANTE="$(curl --silent --fail "${URL}/health")"
JOUEURS="$(printf '%s' "${SANTE}" | tr ',' '
' | grep -o '"players_loaded":[0-9]*' | grep -o '[0-9]*')"
echo "    ${JOUEURS} joueurs chargés"

if [[ "${JOUEURS}" -le 0 ]]; then
  echo "    ERREUR : le service répond mais ne sert aucune donnée" >&2
  exit 1
fi

echo
echo "Documentation interactive : ${URL}/docs"
echo "Métriques d'exploitation  : ${URL}/metrics"
echo
echo "Pensez à ajouter cette URL en tête du README, à côté du lien Streamlit."
