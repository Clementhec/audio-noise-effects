#!/bin/bash
# Setup script for GCP deployment

set -e

echo "Setting up GCP environment for Vertex AI deployment..."

# Install Google Cloud SDK if not already installed
if ! command -v gcloud &> /dev/null; then
    echo "Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi

# Install required Python packages
pip install google-cloud-aiplatform google-cloud-storage

# Authenticate with GCP
echo "Authenticating with GCP..."
gcloud auth login
gcloud auth application-default login

# Set project and configure Docker for GCP
read -p "Enter your GCP Project ID: " PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required GCP APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Configure Docker authentication
gcloud auth configure-docker

echo "✓ GCP setup complete!"
echo "Project ID: $PROJECT_ID"
