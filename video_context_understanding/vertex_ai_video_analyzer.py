#!/usr/bin/env python3
"""
Video Context Understanding using Google Vertex AI API.

This script analyzes video content using Vertex AI's Gemini models.

Usage:
    python vertex_ai_video_analyzer.py <path_to_video.mp4> --prompt "Your prompt" --project YOUR_PROJECT_ID

Example:
    python vertex_ai_video_analyzer.py video.mp4 --prompt "Describe what happens in this video" --project my-gcp-project
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
except ImportError:
    print("Error: vertexai package not found.")
    print("Install it with: pip install google-cloud-aiplatform")
    sys.exit(1)


def analyze_video_vertexai(
    video_path: str,
    prompt: str,
    project_id: str,
    location: str = "us-central1",
    model_name: str = "gemini-1.5-pro"
) -> dict:
    """
    Analyze video using Vertex AI Gemini model.

    Args:
        video_path: Path to the video file (.mp4)
        prompt: Text prompt for video analysis
        project_id: Google Cloud Project ID
        location: GCP region (default: us-central1)
        model_name: Gemini model to use

    Returns:
        dict: Analysis results
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Initializing Vertex AI...")
    print(f"Project: {project_id}")
    print(f"Location: {location}")
    print(f"Model: {model_name}")

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load video file
    print(f"\nLoading video: {video_path}")
    with open(video_path, "rb") as f:
        video_data = f.read()

    # Create video part
    video_part = Part.from_data(
        data=video_data,
        mime_type="video/mp4"
    )

    # Create model
    model = GenerativeModel(model_name)

    print(f"\nAnalyzing video with prompt:")
    print(f"'{prompt}'")
    print("-" * 80)

    # Generate analysis
    response = model.generate_content(
        [video_part, prompt],
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 8192,
        }
    )

    print("\nAnalysis complete!")

    return {
        "video_path": video_path,
        "prompt": prompt,
        "model": model_name,
        "project_id": project_id,
        "location": location,
        "response": response.text,
        "timestamp": datetime.now().isoformat()
    }


def save_to_json(results: dict, output_path: Optional[str] = None) -> str:
    """
    Save analysis results to JSON file.

    Args:
        results: Analysis results dictionary
        output_path: Optional output file path. If not provided, auto-generates name.

    Returns:
        str: Path to saved file
    """
    if output_path is None:
        video_name = Path(results["video_path"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{video_name}_vertexai_analysis_{timestamp}.json"

    # Ensure directory exists
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return str(output_path)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze video using Google Vertex AI Gemini models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vertex_ai_video_analyzer.py video.mp4 \\
      --prompt "Describe the main events in this video" \\
      --project my-gcp-project

  python vertex_ai_video_analyzer.py video.mp4 \\
      --prompt "What emotions are expressed?" \\
      --project my-project \\
      --location europe-west1 \\
      --model gemini-1.5-flash

  python vertex_ai_video_analyzer.py video.mp4 \\
      --prompt "Provide a timeline of actions" \\
      --project my-project \\
      --output results/analysis.json
        """
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (.mp4)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video analysis"
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Google Cloud Project ID"
    )

    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-pro",
        choices=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-002", "gemini-1.5-flash-002"],
        help="Gemini model to use (default: gemini-1.5-pro)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not specified)"
    )

    args = parser.parse_args()

    try:
        # Analyze video
        results = analyze_video_vertexai(
            video_path=args.video_path,
            prompt=args.prompt,
            project_id=args.project,
            location=args.location,
            model_name=args.model
        )

        # Print results
        print("\n" + "=" * 80)
        print("VIDEO ANALYSIS RESULT")
        print("=" * 80)
        print(results["response"])
        print("=" * 80 + "\n")

        # Save results
        save_to_json(results, args.output)

        print("Analysis complete!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
