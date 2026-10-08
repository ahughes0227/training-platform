#!/usr/bin/env bash
# One-time GCP setup for running goals on Vertex AI, then the demo goal.
#
# Run from the repository root in Cloud Shell (or anywhere gcloud is logged in
# as a project owner):
#
#   gcloud config set project YOUR_PROJECT
#   bash scripts/gcp_goal_setup.sh            # set up, then run the demo
#   SKIP_DEMO=1 bash scripts/gcp_goal_setup.sh  # set up only
#
# It is safe to re-run: every step checks before it creates. It creates
#   - the Vertex AI, Artifact Registry, Cloud Build and Storage APIs (enabled)
#   - a bucket gs://PROJECT-defect-goals for datasets, requests and run outputs
#   - an Artifact Registry repository `defect-platform`
#   - a service account `defect-trainer` that Vertex jobs run as
#   - a trainer image built by Cloud Build from this checkout
#   - vertex.yaml, the config `defect goal ... --vertex vertex.yaml` reads
#
# The image is a development build: it is not GPU-certified, so models it
# trains are for evaluation and cannot be staged for release. The demo runs on
# a CPU machine (n1-standard-4), so no GPU quota is needed.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
if [ -z "$PROJECT" ]; then
  echo "Set a project first: gcloud config set project YOUR_PROJECT" >&2
  exit 1
fi
BUCKET="gs://${PROJECT}-defect-goals"
REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT}/defect-platform"
SERVICE_ACCOUNT="defect-trainer@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REPOSITORY}/defect-trainer:$(git rev-parse --short=12 HEAD)"
BASE_IMAGE="${TRAINING_BASE_IMAGE:-python:3.12-slim}"

step() { printf '\n== %s\n' "$*"; }

step "Enabling APIs in ${PROJECT}"
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com storage.googleapis.com iam.googleapis.com --project "$PROJECT"

step "Bucket ${BUCKET}"
if ! gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
fi

step "Artifact Registry repository ${REPOSITORY}"
if ! gcloud artifacts repositories describe defect-platform --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  gcloud artifacts repositories create defect-platform --repository-format docker \
    --location "$REGION" --project "$PROJECT"
fi

step "Service account ${SERVICE_ACCOUNT}"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create defect-trainer --display-name "Defect trainer (Vertex jobs)" \
    --project "$PROJECT"
fi
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" --role roles/storage.objectAdmin >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" --condition None \
  --member "serviceAccount:${SERVICE_ACCOUNT}" --role roles/logging.logWriter >/dev/null

step "Pinning the base image ${BASE_IMAGE}"
case "$BASE_IMAGE" in
  *@sha256:*) PINNED_BASE="$BASE_IMAGE" ;;
  *)
    docker pull -q "$BASE_IMAGE" >/dev/null
    PINNED_BASE="$(docker inspect --format '{{index .RepoDigests 0}}' "$BASE_IMAGE")"
    ;;
esac
echo "$PINNED_BASE"

step "Building the trainer image with Cloud Build (several minutes)"
if ! gcloud artifacts docker images describe "$IMAGE" >/dev/null 2>&1; then
  gcloud builds submit . --project "$PROJECT" --region "$REGION" \
    --config infra/cloudbuild/trainer-candidate.yaml \
    --substitutions "_TRAINING_BASE_IMAGE=${PINNED_BASE},_IMAGE_URI=${IMAGE}"
fi
DIGEST="$(gcloud artifacts docker images describe "$IMAGE" --format 'value(image_summary.digest)')"
IMAGE_DIGEST="${REPOSITORY}/defect-trainer@${DIGEST}"
echo "$IMAGE_DIGEST"

step "Writing vertex.yaml"
cat > vertex.yaml <<EOF
project: ${PROJECT}
region: ${REGION}
service_account: ${SERVICE_ACCOUNT}
staging_uri: ${BUCKET}/goals
image_digest: ${IMAGE_DIGEST}
machine_type: n1-standard-4
max_run_hours: 1
poll_seconds: 20
allow_uncertified_image: true
EOF
cat vertex.yaml

if [ -n "${SKIP_DEMO:-}" ]; then
  echo
  echo "Setup done. Run a goal with: uv run defect goal start goal.yaml --vertex vertex.yaml"
  exit 0
fi

step "Running the demo goal on Vertex AI"
# The demo builds its tiny dataset and backbone here, so it needs the train
# extra locally. Keep the environment and cache off a small home disk.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/defect-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HUB_OFFLINE=1
gcloud auth application-default print-access-token >/dev/null 2>&1 || gcloud auth application-default login
uv sync --extra train --extra data --extra cloud
uv run defect goal demo "${DEMO_DIR:-$HOME/goal-demo-vertex}" --vertex vertex.yaml
