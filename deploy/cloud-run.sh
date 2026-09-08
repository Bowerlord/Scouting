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
  echo "L'ID se lit sur https://console.cloud.google.com/billing, onglet" >&2
  echo "« Vos projets », colonne Identifiant. Ce n'est pas le nom du projet." >&2
  exit 1
fi

# Un ID de projet inexistant fait echouer le deploiement sur
# UREQ_PROJECT_BILLING_NOT_FOUND, qui accuse a tort la facturation. Constate en
# reel le 2026-09-08 : une soiree perdue a chercher une carte bancaire alors que
# la commande visait un projet qui n'existait pas. On verifie donc avant.
if ! gcloud projects describe "${PROJET}" >/dev/null 2>&1; then
  echo "Le projet « ${PROJET} » est introuvable, ou le compte connecte n'y a pas acces." >&2
  echo "Verifiez l'ID exact, puis « gcloud auth list » pour le compte actif." >&2
  echo "Sans cette verification, Google repondrait UREQ_PROJECT_BILLING_NOT_FOUND," >&2
  echo "ce qui laisse croire a tort a un probleme de facturation." >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJET}/${DEPOT}/${SERVICE}"

echo "==> Projet   : ${PROJET}"
echo "==> Région   : ${REGION}  (europe-west1 : Belgique, la plus proche de Paris)"
echo "==> Image    : ${IMAGE}"
echo

echo "==> Activation des API nécessaires"
# compute.googleapis.com n'est pas evidente et son absence coute cher :
# Cloud Build s'execute sous le compte de service Compute Engine du projet,
# et ce compte n'existe qu'une fois cette API activee. Sans elle, le build
# echoue sur un PERMISSION_DENIED qui accuse a tort le compte utilisateur.
# Constate en reel le 2026-09-07, sur un compte pourtant proprietaire.
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  --project "${PROJET}"

# L'activation rend la main avant que les comptes de service et leurs droits
# soient propages. Sans cette pause, le premier essai echoue et le second
# reussit, ce qui fait passer le script pour instable alors que c'est une course.
echo "==> Attente de la propagation des permissions (60 s)"
sleep 60

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
