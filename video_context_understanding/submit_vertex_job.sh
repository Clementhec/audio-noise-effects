#!/bin/bash
# Submit Video-LLaVA inference job to Vertex AI Custom Jobs

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY="${ARTIFACT_REGISTRY_REPO:-vertex-ai-models}"
IMAGE_NAME="video-llava-context-analyzer"
IMAGE_TAG="latest"
JOB_NAME="video-llava-inference-$(date +%Y%m%d-%H%M%S)"
STAGING_BUCKET="${GCS_BUCKET:-gs://your-bucket-name}"

# GPU Configuration
# Options (from cheapest to most expensive):
# - NVIDIA_TESLA_K80 (oldest, cheapest, requires n1-standard-8+)
# - NVIDIA_TESLA_P4 (inference-optimized, good for models)
# - NVIDIA_TESLA_T4 (balanced)
# - NVIDIA_L4 (newest, efficient)
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_P100"
ACCELERATOR_COUNT="1"

# Derived variables
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Submitting Vertex AI Custom Job"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "Machine Type: ${MACHINE_TYPE}"
echo "Accelerator: ${ACCELERATOR_TYPE} x ${ACCELERATOR_COUNT}"
echo "Staging Bucket: ${STAGING_BUCKET}"
echo "=========================================="

# Create the custom job configuration
cat > /tmp/vertex_job_config.json <<EOF
{
  "workerPoolSpecs": [
    {
      "machineSpec": {
        "machineType": "${MACHINE_TYPE}",
        "acceleratorType": "${ACCELERATOR_TYPE}",
        "acceleratorCount": ${ACCELERATOR_COUNT}
      },
      "replicaCount": 1,
      "containerSpec": {
        "imageUri": "${IMAGE_URI}",
        "args": [
          "chaplin_speech.mp4",
          "--output-dir",
          "/workspace/output",
          "--frames",
          "8"
        ],
        "env": [
          {
            "name": "GCS_OUTPUT_BUCKET",
            "value": "${STAGING_BUCKET}"
          }
        ]
      }
    }
  ]
}
EOF

echo "Job configuration created at /tmp/vertex_job_config.json"
echo ""

# Submit the custom job
echo "Submitting job to Vertex AI..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --display-name="${JOB_NAME}" \
  --config=/tmp/vertex_job_config.json 
echo ""
echo "=========================================="
echo "Job submitted successfully!"
echo "=========================================="
echo ""
echo "To monitor the job, run:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "To view job details:"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "To stream logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "Note: The output will be inside the container at /workspace/output"
echo "To access it, you'll need to modify the Dockerfile to copy output to GCS"
echo "=========================================="
