# Vertex AI Setup Guide for Video-LLaVA Analyzer

This guide will walk you through running the Video-LLaVA analyzer on Google Cloud Vertex AI with GPU support.

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and configured
3. **Docker** installed locally
4. **Required GCP APIs** enabled:
   - Vertex AI API
   - Artifact Registry API
   - Cloud Storage API

## Step 1: Initial Setup

### 1.1 Set Environment Variables

Edit the environement variables in `.env`

### 1.2 Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project ${GCP_PROJECT_ID}
```

### 1.3 Enable Required APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage-component.googleapis.com
```

### 1.4 Create a GCS Bucket for Output

```bash
gcloud storage buckets create ${GCS_BUCKET} --uniform-bucket-level-access
```

## Step 2: Build and Push Docker Image

### 2.3 Build and Push

```bash
./build_and_push.sh
```

This script will:
- Copy the video file from parent directory
- Configure Docker authentication for Artifact Registry
- Create the Artifact Registry repository if needed
- Build the Docker image with all dependencies
- Push the image to Google Artifact Registry

**Note:** The build process may take 10-15 minutes as it downloads CUDA base image and installs all dependencies.

## Step 3: Submit the Vertex AI Job

### 3.1 Update Job Configuration

Edit `submit_vertex_job.sh` to customize:
- GPU type (default: NVIDIA_TESLA_T4)
- Machine type (default: n1-standard-4)
- Number of frames to sample (default: 8)

### 3.2 Submit the Job

```bash
./submit_vertex_job.sh
```

This will create and submit a Vertex AI Custom Job with GPU support.

## Step 4: Monitor the Job

### 4.1 List All Jobs

```bash
gcloud ai custom-jobs list \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

### 4.2 View Job Details

```bash
# Replace JOB_NAME with the actual job name from the output
gcloud ai custom-jobs describe JOB_NAME \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

### 4.3 Stream Logs in Real-Time

```bash
# Replace JOB_NAME with the actual job name
gcloud ai custom-jobs stream-logs JOB_NAME \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

## Optimization Tips

### Using Preemptible VMs (Lower Cost)

To reduce costs, you can use preemptible VMs by modifying the job config:

```json
"machineSpec": {
  "machineType": "n1-standard-4",
  "acceleratorType": "NVIDIA_TESLA_T4",
  "acceleratorCount": 1
},
"scheduling": {
  "preemptible": true
}
```
