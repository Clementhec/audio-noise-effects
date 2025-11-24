# Dockerfile for Video-LLaVA Analyzer on Vertex AI
# Multi-stage build for optimal caching and smaller image size

# Stage 1: Build dependencies and install gcloud SDK
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies and gcloud SDK in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && curl -sSL https://sdk.cloud.google.com | bash \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime image
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_HOME=/workspace/model_cache
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Install only runtime dependencies (lighter, no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy gcloud SDK from builder stage
COPY --from=builder /root/google-cloud-sdk /root/google-cloud-sdk

# Set working directory
WORKDIR /workspace

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies (cached unless requirements.txt changes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files (changes more frequently, so put after deps)
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
