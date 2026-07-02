#!/bin/bash
set -e

PROJECT_ID="imconvo-501215"
REGION="australia-southeast1"
BUCKET_NAME="imconvo-model-weights-${PROJECT_ID}"
REPO_NAME="imconvo-repo"
SERVICE_NAME="imconvo-backend"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "=============================================="
echo "🚀 Deploying ImConvo Backend to Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Bucket:     gs://${BUCKET_NAME}"
echo "=============================================="

# 0. Enable required Google Cloud APIs
echo "🔌 0/5 Enabling required GCP APIs (Artifact Registry, Cloud Build, Cloud Run)..."
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com --quiet

# 1. Create the Storage Bucket in Sydney (ignore error if it already exists)
echo "📦 1/5 Checking storage bucket..."
if ! gcloud --quiet storage buckets describe gs://${BUCKET_NAME} &>/dev/null; then
    echo "Creating bucket gs://${BUCKET_NAME}..."
    gcloud --quiet storage buckets create gs://${BUCKET_NAME} --location=${REGION}
else
    echo "Bucket gs://${BUCKET_NAME} already exists."
fi

# 2. Sync checkpoints structure (only copies new/changed files)
echo "📤 2/5 Syncing checkpoints weights to bucket..."
gcloud --quiet storage rsync checkpoints gs://${BUCKET_NAME} --recursive

# 3. Create Artifact Registry Docker repository in Sydney (ignore error if exists)
echo "📦 3/5 Checking Artifact Registry repository..."
if ! gcloud --quiet artifacts repositories describe ${REPO_NAME} --location=${REGION} &>/dev/null; then
    echo "Creating Artifact Registry repository '${REPO_NAME}'..."
    gcloud --quiet artifacts repositories create ${REPO_NAME} \
      --repository-format=docker \
      --location=${REGION} \
      --description="Docker repository for ImConvo Backend"
else
    echo "Repository '${REPO_NAME}' already exists."
fi

# 4. Build and push image via Google Cloud Build
echo "🏗️  4/5 Submitting build to Google Cloud Build..."
gcloud --quiet builds submit --tag ${IMAGE_TAG} .

# 5. Deploy to Cloud Run with GCS FUSE mount
echo "🚀 5/5 Deploying backend service to Google Cloud Run (4GB RAM, GCS FUSE mount)..."
gcloud --quiet run deploy ${SERVICE_NAME} \
  --image=${IMAGE_TAG} \
  --region=${REGION} \
  --memory=4Gi \
  --add-volume=name=weights-volume,type=cloud-storage,bucket=${BUCKET_NAME} \
  --add-volume-mount=volume=weights-volume,mount-path=/code/checkpoints \
  --allow-unauthenticated

echo "=============================================="
echo "✅ Deployed successfully!"
echo "Find your service URL in the terminal output above."
echo "=============================================="
