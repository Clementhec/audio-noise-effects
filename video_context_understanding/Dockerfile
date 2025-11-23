# Dockerfile for Video-LLaVA Analyzer on Vertex AI
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_HOME=/workspace/model_cache

# Install system dependencies including gcloud SDK
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the analyzer script and wrapper
COPY video_llava_analyzer.py .
COPY run_with_gcs_upload.sh .
RUN chmod +x run_with_gcs_upload.sh

# Copy the video file
COPY chaplin_speech.mp4 .

# Create output directory
RUN mkdir -p /workspace/output

# Set the entrypoint to the wrapper script
ENTRYPOINT ["/workspace/run_with_gcs_upload.sh"]

# Default command (can be overridden)
CMD ["chaplin_speech.mp4", "--output-dir", "/workspace/output"]
