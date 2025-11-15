#!/usr/bin/env python3
"""
Example usage of the video_llava_analyzer module programmatically.

This demonstrates how to use the video analysis functions in your own code.
"""

from video_llava_analyzer import load_model, process_video, save_output


def example_basic_analysis():
    """Basic video analysis example."""
    print("Example 1: Basic Video Analysis")
    print("-" * 50)

    # Path to your video
    video_path = "path/to/your/video.mp4"

    # Load model once
    model, processor = load_model()

    # Process video
    description = process_video(
        video_path=video_path,
        model=model,
        processor=processor
    )

    print(f"Description: {description}")

    # Save output
    save_output(video_path, description)
    print()


def example_custom_prompt():
    """Video analysis with custom prompt."""
    print("Example 2: Custom Prompt Analysis")
    print("-" * 50)

    video_path = "path/to/your/video.mp4"

    # Load model
    model, processor = load_model()

    # Custom prompt
    custom_prompt = "What are the main objects and actions in this video?"

    # Process with custom prompt
    description = process_video(
        video_path=video_path,
        model=model,
        processor=processor,
        prompt=custom_prompt,
        max_new_tokens=300
    )

    print(f"Custom Analysis: {description}")

    # Save with custom prompt info
    save_output(video_path, description, prompt=custom_prompt)
    print()


def example_batch_processing():
    """Process multiple videos with the same model."""
    print("Example 3: Batch Processing")
    print("-" * 50)

    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        "path/to/video3.mp4"
    ]

    # Load model once for all videos (efficient!)
    model, processor = load_model()

    for video_path in video_paths:
        print(f"\nProcessing: {video_path}")

        description = process_video(
            video_path=video_path,
            model=model,
            processor=processor
        )

        print(f"Result: {description[:100]}...")
        save_output(video_path, description, output_dir="batch_results")

    print("\nBatch processing complete!")
    print()


def example_detailed_analysis():
    """Detailed analysis with more frames."""
    print("Example 4: Detailed Analysis (More Frames)")
    print("-" * 50)

    video_path = "path/to/your/video.mp4"

    # Load model
    model, processor = load_model()

    # Process with more frames for better understanding
    description = process_video(
        video_path=video_path,
        model=model,
        processor=processor,
        num_frames=16,  # More frames = better detail
        max_new_tokens=400  # Longer description
    )

    print(f"Detailed Description: {description}")
    save_output(video_path, description, output_dir="detailed_analysis")
    print()


if __name__ == "__main__":
    print("Video-LLaVA Analyzer - Programmatic Usage Examples")
    print("=" * 50)
    print()

    # Uncomment the examples you want to run:

    # example_basic_analysis()
    # example_custom_prompt()
    # example_batch_processing()
    # example_detailed_analysis()

    print("\nNote: Update the video_path variables with actual video files to run these examples.")
