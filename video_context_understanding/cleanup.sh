#!/bin/bash

# Delete the Artifact Registry repository
gcloud artifacts repositories delete ${ARTIFACT_REGISTRY_REPO} \
  --location=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}

# Delete the GCS bucket
gsutil rm -r gs://your-bucket-name/

# Cancel running jobs (if any)
gcloud ai custom-jobs cancel JOB_NAME \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
