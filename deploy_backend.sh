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

# 1. Create the Storage Bucket in Sydney (ignore error if it already exists)
echo "📦 1/5 Checking storage bucket..."
if ! gcloud storage buckets describe gs://${BUCKET_NAME} &>/dev/null; then
    echo "Creating bucket gs://${BUCKET_NAME}..."
    gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}
else
    echo "Bucket gs://${BUCKET_NAME} already exists."
fi

# 2. Upload checkpoints structure
echo "📤 2/5 Uploading checkpoints weights to bucket..."
gcloud storage cp -r checkpoints/* gs://${BUCKET_NAME}/

# 3. Create Artifact Registry Docker repository in Sydney (ignore error if exists)
echo "📦 3/5 Checking Artifact Registry repository..."
if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} &>/dev/null; then
    echo "Creating Artifact Registry repository '${REPO_NAME}'..."
    gcloud artifacts repositories create ${REPO_NAME} \
      --repository-format=docker \
      --location=${REGION} \
      --description="Docker repository for ImConvo Backend"
else
    echo "Repository '${REPO_NAME}' already exists."
fi

# 4. Build and push image via Google Cloud Build
echo "🏗️  4/5 Submitting build to Google Cloud Build..."
gcloud builds submit --tag ${IMAGE_TAG} .

# 5. Deploy to Cloud Run with GCS FUSE mount
echo "🚀 5/5 Deploying backend service to Google Cloud Run (4GB RAM, GCS FUSE mount)..."
gcloud run deploy ${SERVICE_NAME} \
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
