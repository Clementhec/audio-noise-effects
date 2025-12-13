#!/bin/bash
# Wrapper script to run video analysis and upload results to GCS

set -e

# GCS bucket for output (passed as environment variable)
GCS_OUTPUT_BUCKET="${GCS_OUTPUT_BUCKET:-}"

echo "=========================================="
echo "Running Video-LLaVA Analyzer"
echo "=========================================="

# Run the analysis
python3 video_llava_analyzer.py "$@"

# Upload results to GCS if bucket is specified
if [ -n "${GCS_OUTPUT_BUCKET}" ]; then
    echo ""
    echo "=========================================="
    echo "Uploading results to GCS"
    echo "=========================================="
    echo "Bucket: ${GCS_OUTPUT_BUCKET}"

    # Copy all output files to GCS
    if [ -d "/workspace/output" ]; then
        gsutil -m cp -r /workspace/output/* ${GCS_OUTPUT_BUCKET}/
        echo "Results uploaded successfully to ${GCS_OUTPUT_BUCKET}"
    else
        echo "Warning: Output directory not found"
    fi
fi

echo "=========================================="
echo "Job completed!"
echo "=========================================="
