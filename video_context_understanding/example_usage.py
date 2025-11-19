#!/usr/bin/env python3
"""
Example usage of the video_llava_analyzer module programmatically.

This demonstrates how to use the video analysis functions in your own code.
"""

from video_llava_analyzer import load_model, process_video, save_output
from argparse import ArgumentParser

def example_custom_prompt(video_path:str, prompt:str):
    """Video analysis with custom prompt."""
    print("Example 2: Custom Prompt Analysis")
    print("-" * 50)

    # Load model
    model, processor = load_model()

    # Process with custom prompt
    description = process_video(
        video_path=video_path,
        model=model,
        processor=processor,
        prompt=prompt,
        max_new_tokens=300
    )

    print(f"Custom Analysis: {description}")

    # Save with custom prompt info
    save_output(video_path, description, prompt=custom_prompt)
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video-path", type=str)
    parser.add_argument("--prompt", type=str, default="")
    
    args = parser.parse_args()
    video_path = args.video_path
    prompt = args.prompt

    print("Video-LLaVA Analyzer - Programmatic Usage Examples")
    print("=" * 50)
    print()

    example_custom_prompt(video_path, prompt)

    print("\nNote: Update the video_path variables with actual video files to run these examples.")
