#!/usr/bin/env python3
"""
Video Context Understanding using Google Gemini API.

This script analyzes video content with temporal awareness using Gemini's
native video understanding capabilities.

Usage:
    python gemini_video_analyzer.py <path_to_video.mp4> [--api-key YOUR_KEY]

Example:
    python gemini_video_analyzer.py chaplin_speech.mp4
    python gemini_video_analyzer.py chaplin_speech.mp4 --api-key "your-api-key"
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not found.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)


def configure_gemini(api_key: Optional[str] = None):
    """
    Configure Gemini API with credentials.

    Args:
        api_key: Optional API key. If not provided, will try GOOGLE_API_KEY env var.
    """
    if api_key:
        genai.configure(api_key=api_key)
    elif os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        print("Error: No API key provided.")
        print("Set GOOGLE_API_KEY environment variable or use --api-key flag")
        sys.exit(1)


def analyze_video_with_timeline(
    video_path: str,
    model_name: str = "gemini-1.5-pro",
    custom_prompt: Optional[str] = None
) -> dict:
    """
    Analyze video and extract timeline of actions.

    Args:
        video_path: Path to the video file
        model_name: Gemini model to use
        custom_prompt: Optional custom prompt

    Returns:
        dict: Analysis results with timeline
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Uploading video: {video_path}")
    print("This may take a moment depending on file size...")

    # Upload video to Gemini
    video_file = genai.upload_file(path=video_path)
    print(f"Video uploaded successfully: {video_file.name}")

    # Wait for video to be processed
    import time
    while video_file.state.name == "PROCESSING":
        print("Processing video...")
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state}")

    print("Video ready for analysis")

    # Create model
    model = genai.GenerativeModel(model_name)

    # Default prompt for timeline extraction
    if custom_prompt is None:
        prompt = """Analyze this video and provide a detailed timeline of all actions and events.

For each significant action or event, provide:
1. Approximate timestamp (in seconds or minutes:seconds format)
2. Description of what's happening
3. Key visual elements or context

Format your response as a structured timeline with clear timestamps."""
    else:
        prompt = custom_prompt

    print(f"\nAnalyzing video with {model_name}...")
    print("-" * 80)

    # Generate analysis
    response = model.generate_content(
        [video_file, prompt],
        request_options={"timeout": 600}  # 10 minute timeout for long videos
    )

    # Clean up uploaded file
    genai.delete_file(video_file.name)
    print(f"Cleaned up uploaded file: {video_file.name}")

    return {
        "video_path": video_path,
        "model": model_name,
        "prompt": prompt,
        "analysis": response.text,
        "timestamp": datetime.now().isoformat()
    }


def save_results(results: dict, output_dir: str = "output") -> str:
    """
    Save analysis results to JSON file.

    Args:
        results: Analysis results dictionary
        output_dir: Output directory path

    Returns:
        str: Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    video_name = Path(results["video_path"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{video_name}_gemini_timeline_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return str(output_file)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze video content and extract timeline using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variable GOOGLE_API_KEY
  export GOOGLE_API_KEY="your-api-key"
  python gemini_video_analyzer.py chaplin_speech.mp4

  # Using command-line API key
  python gemini_video_analyzer.py chaplin_speech.mp4 --api-key "your-key"

  # Custom prompt
  python gemini_video_analyzer.py video.mp4 --prompt "What emotions are expressed?"

  # Use Gemini Flash (faster, cheaper)
  python gemini_video_analyzer.py video.mp4 --model gemini-1.5-flash
        """
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (MP4, MOV, etc.)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (or set GOOGLE_API_KEY environment variable)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-pro",
        choices=["gemini-1.5-pro", "gemini-1.5-flash"],
        help="Gemini model to use (default: gemini-1.5-pro)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for video analysis"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: ./output)"
    )

    args = parser.parse_args()

    try:
        # Configure API
        configure_gemini(args.api_key)

        # Analyze video
        results = analyze_video_with_timeline(
            args.video_path,
            model_name=args.model,
            custom_prompt=args.prompt
        )

        # Print results
        print("\n" + "=" * 80)
        print("VIDEO TIMELINE ANALYSIS")
        print("=" * 80)
        print(results["analysis"])
        print("=" * 80 + "\n")

        # Save results
        save_results(results, args.output_dir)

        print("✓ Analysis complete!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
