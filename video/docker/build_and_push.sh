#!/bin/bash
# Build and push Docker image to Google Artifact Registry for Vertex AI

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-europe-west1}"
REPOSITORY="${ARTIFACT_REGISTRY_REPO:-vertex-ai-models}"
IMAGE_NAME="video-llava-context-analyzer"
IMAGE_TAG="latest"

# Derived variables
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building and Pushing Docker Image"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY}"
echo "Image URI: ${IMAGE_URI}"
echo "=========================================="

# Verify we're in the right directory
if [ ! -f "video_llava_analyzer.py" ]; then
    echo "Error: video_llava_analyzer.py not found in current directory"
    echo "Please run this script from the video_context_understanding directory"
    exit 1
fi

# Copy the video file from parent directory
echo "Copying chaplin_speech.mp4..."
if [ ! -f "chaplin_speech.mp4" ]; then
    cp ../chaplin_speech.mp4 .
    echo "Video file copied successfully"
else
    echo "Video file already exists in current directory"
fi

# Authenticate with gcloud
echo "Authenticating with gcloud..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Create Artifact Registry repository if it doesn't exist
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe ${REPOSITORY} \
    --location=${REGION} \
    --project=${PROJECT_ID} 2>/dev/null || \
gcloud artifacts repositories create ${REPOSITORY} \
    --repository-format=docker \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --description="Docker repository for Vertex AI models"

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_URI} .

# Push the Docker image
echo "Pushing Docker image to Artifact Registry..."
docker push ${IMAGE_URI}

echo "=========================================="
echo "Build and push completed successfully!"
echo "Image URI: ${IMAGE_URI}"
echo "=========================================="
echo ""
echo "Next step: Run ./submit_vertex_job.sh to submit the job to Vertex AI"
