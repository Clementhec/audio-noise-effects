#!/usr/bin/env python3
"""
Video Context Understanding using TwelveLabs Marengo API.

This script analyzes video content using TwelveLabs' Marengo model.

Usage:
    python twelvelabs_video_analyzer.py <path_to_video.mp4> --prompt "Your prompt" --api-key YOUR_API_KEY

Example:
    python twelvelabs_video_analyzer.py video.mp4 --prompt "Describe what happens in this video" --api-key "tlk_xxx"
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from twelvelabs import TwelveLabs
except ImportError:
    print("Error: twelvelabs package not found.")
    print("Install it with: pip install twelvelabs")
    sys.exit(1)


def analyze_video_twelvelabs(
    video_path: str,
    prompt: str,
    api_key: str,
    index_name: Optional[str] = None,
    model_name: str = "marengo-retrieval-2.6"
) -> dict:
    """
    Analyze video using TwelveLabs Marengo model.

    Args:
        video_path: Path to the video file (.mp4)
        prompt: Text prompt for video analysis
        api_key: TwelveLabs API key
        index_name: Optional index name. If not provided, will create temporary index.
        model_name: TwelveLabs model to use (default: marengo-retrieval-2.6)

    Returns:
        dict: Analysis results
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Initializing TwelveLabs client...")
    print(f"Model: {model_name}")

    # Initialize client
    client = TwelveLabs(api_key=api_key)

    # Create or use existing index
    if index_name is None:
        # Create temporary index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_name = f"temp_video_analysis_{timestamp}"
        create_index = True
    else:
        # Check if index exists
        try:
            existing_indexes = client.index.list()
            create_index = not any(idx.name == index_name for idx in existing_indexes)
        except:
            create_index = True

    index_id = None
    if create_index:
        print(f"\nCreating index: {index_name}")
        index = client.index.create(
            name=index_name,
            engines=[
                {
                    "name": "marengo2.6",
                    "options": ["visual", "conversation", "text_in_video"],
                }
            ]
        )
        index_id = index.id
        print(f"Index created with ID: {index_id}")
    else:
        # Get existing index
        indexes = client.index.list()
        for idx in indexes:
            if idx.name == index_name:
                index_id = idx.id
                break
        print(f"Using existing index: {index_name} (ID: {index_id})")

    # Upload video
    print(f"\nUploading video: {video_path}")
    task = client.task.create(
        index_id=index_id,
        file=video_path,
        language="en"
    )

    print(f"Upload task created with ID: {task.id}")
    print("Waiting for video processing...")

    # Wait for video to be indexed
    def on_task_update(task):
        print(f"  Status: {task.status}")

    task.wait_for_done(
        sleep_interval=5,
        callback=on_task_update
    )

    if task.status != "ready":
        raise ValueError(f"Video processing failed with status: {task.status}")

    print("Video indexed successfully!")
    video_id = task.video_id

    # Generate analysis using conversation/search
    print(f"\nAnalyzing video with prompt:")
    print(f"'{prompt}'")
    print("-" * 80)

    # Use generate API for text generation
    try:
        response = client.generate.text(
            video_id=video_id,
            prompt=prompt
        )
        analysis_text = response.data
    except AttributeError:
        # Fallback to search if generate doesn't work
        search_results = client.search.query(
            index_id=index_id,
            query_text=prompt,
            options=["visual", "conversation", "text_in_video"]
        )

        # Extract and format search results
        results_data = []
        for result in search_results.data[:5]:  # Top 5 results
            results_data.append({
                "score": result.score,
                "start": result.start,
                "end": result.end,
                "confidence": result.confidence,
                "metadata": result.metadata if hasattr(result, 'metadata') else None
            })

        analysis_text = {
            "search_results": results_data,
            "prompt": prompt
        }

    print("\nAnalysis complete!")

    result = {
        "video_path": video_path,
        "video_id": video_id,
        "prompt": prompt,
        "model": model_name,
        "index_id": index_id,
        "index_name": index_name,
        "response": analysis_text,
        "timestamp": datetime.now().isoformat()
    }

    return result


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
        output_path = f"{video_name}_twelvelabs_analysis_{timestamp}.json"

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
        description="Analyze video using TwelveLabs Marengo model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using API key from command line
  python twelvelabs_video_analyzer.py video.mp4 \\
      --prompt "Describe the main events in this video" \\
      --api-key "tlk_xxxxx"

  # Using environment variable
  export TWELVE_LABS_API_KEY="tlk_xxxxx"
  python twelvelabs_video_analyzer.py video.mp4 \\
      --prompt "What emotions are expressed?"

  # Using existing index
  python twelvelabs_video_analyzer.py video.mp4 \\
      --prompt "Provide a timeline of actions" \\
      --index my-video-index \\
      --api-key "tlk_xxxxx"

  # Specify output file
  python twelvelabs_video_analyzer.py video.mp4 \\
      --prompt "Analyze this video" \\
      --api-key "tlk_xxxxx" \\
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
        "--api-key",
        type=str,
        default=None,
        help="TwelveLabs API key (or set TWELVE_LABS_API_KEY environment variable)"
    )

    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Index name to use (creates temporary index if not specified)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="marengo-retrieval-2.6",
        help="TwelveLabs model to use (default: marengo-retrieval-2.6)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not specified)"
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv("TWELVE_LABS_API_KEY")
    if not api_key:
        print("Error: No API key provided.")
        print("Set TWELVE_LABS_API_KEY environment variable or use --api-key flag")
        sys.exit(1)

    try:
        # Analyze video
        results = analyze_video_twelvelabs(
            video_path=args.video_path,
            prompt=args.prompt,
            api_key=api_key,
            index_name=args.index,
            model_name=args.model
        )

        # Print results
        print("\n" + "=" * 80)
        print("VIDEO ANALYSIS RESULT")
        print("=" * 80)

        response = results["response"]
        if isinstance(response, dict):
            print(json.dumps(response, indent=2))
        else:
            print(response)

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
